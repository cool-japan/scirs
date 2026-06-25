//! GPU-Accelerated RBF Solver (CPU simulation mode)
//!
//! Provides a GPU-dispatch-aware RBF interpolation interface. When
//! `GpuDispatch::Simulated` is selected the kernel evaluations and the linear
//! solve are carried out entirely on the CPU in parallel chunks that mimic the
//! thread-block structure of a real GPU dispatch.  This lets application code
//! adopt the GPU API without requiring hardware support, enabling seamless
//! future migration to a real GPU backend.
//!
//! ## Algorithm
//!
//! 1. **Build** the N×N RBF matrix Φ where Φ_{ij} = φ(‖xᵢ − xⱼ‖).
//! 2. **Solve** Φ w = y for the weight vector w using partial-pivot Gaussian
//!    elimination.
//! 3. **Evaluate** at query points q: f(q) = Σᵢ wᵢ φ(‖q − xᵢ‖).
//!
//! The simulation path chunks query evaluation into `block_size` segments,
//! mirroring the thread-block pattern of GPU dispatch.

use crate::error::InterpolateError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Controls whether computation is routed to a real GPU or simulated in CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDispatch {
    /// Use the CPU only (no GPU, no simulation overhead).
    Cpu,
    /// Simulate GPU dispatch on the CPU in parallel chunks.
    Simulated,
}

/// RBF kernel types supported by the GPU solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuRbfKernel {
    /// φ(r) = exp(-(ε r)²)
    Gaussian,
    /// φ(r) = √(1 + (ε r)²)
    Multiquadric,
    /// φ(r) = r² ln(r)  (r > 0), 0 at r = 0
    ThinPlateSpline,
    /// φ(r) = r
    Linear,
}

/// Configuration for [`GpuRbfSolver`].
#[derive(Debug, Clone)]
pub struct GpuRbfConfig {
    /// RBF kernel choice.
    pub kernel: GpuRbfKernel,
    /// Shape parameter ε (not used by ThinPlateSpline and Linear).
    pub epsilon: f64,
    /// GPU dispatch mode.
    pub dispatch: GpuDispatch,
    /// Thread-block size for GPU simulation; controls evaluation chunk size.
    pub block_size: usize,
}

impl Default for GpuRbfConfig {
    fn default() -> Self {
        Self {
            kernel: GpuRbfKernel::Gaussian,
            epsilon: 1.0,
            dispatch: GpuDispatch::Simulated,
            block_size: 64,
        }
    }
}

/// GPU-accelerated (or simulated) RBF interpolation solver.
///
/// # Example
///
/// ```rust
/// use scirs2_interpolate::gpu_rbf::{GpuRbfConfig, GpuRbfKernel, GpuRbfSolver, GpuDispatch};
///
/// let config = GpuRbfConfig {
///     kernel: GpuRbfKernel::Gaussian,
///     epsilon: 1.0,
///     dispatch: GpuDispatch::Simulated,
///     block_size: 32,
/// };
/// let mut solver = GpuRbfSolver::new(config);
///
/// let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let values = vec![1.0, 2.0, 3.0];
/// solver.fit(&points, &values).expect("fit should succeed");
///
/// let results = solver.evaluate(&points).expect("evaluate should succeed");
/// ```
#[derive(Debug)]
pub struct GpuRbfSolver {
    config: GpuRbfConfig,
    points: Vec<[f64; 2]>,
    weights: Vec<f64>,
    fitted: bool,
}

impl GpuRbfSolver {
    /// Create a new solver with the given configuration.
    pub fn new(config: GpuRbfConfig) -> Self {
        Self {
            config,
            points: Vec::new(),
            weights: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the RBF to scattered (x, y) → z data.
    ///
    /// Builds the full N×N RBF matrix and solves the linear system using
    /// partial-pivot Gaussian elimination.  For `GpuDispatch::Simulated` the
    /// matrix construction is broken into `block_size` row-chunks.
    pub fn fit(&mut self, points: &[[f64; 2]], values: &[f64]) -> Result<(), InterpolateError> {
        let n = points.len();
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "gpu_rbf: at least one point required".into(),
            ));
        }
        if values.len() != n {
            return Err(InterpolateError::ShapeMismatch {
                expected: n.to_string(),
                actual: values.len().to_string(),
                object: "values".into(),
            });
        }
        if self.config.epsilon <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: "gpu_rbf: epsilon must be positive".into(),
            });
        }

        let mat = build_rbf_matrix(
            points,
            &self.config.kernel,
            self.config.epsilon,
            self.config.block_size,
            &self.config.dispatch,
        );

        let w = solve_linear_system(&mat, values, n)?;

        self.points = points.to_vec();
        self.weights = w;
        self.fitted = true;
        Ok(())
    }

    /// Evaluate the fitted RBF at `query_points`.
    ///
    /// For `GpuDispatch::Simulated` the evaluation is chunked into blocks of
    /// `block_size` points, mimicking GPU thread-block dispatch.
    pub fn evaluate(&self, query_points: &[[f64; 2]]) -> Result<Vec<f64>, InterpolateError> {
        if !self.fitted {
            return Err(InterpolateError::InvalidState(
                "gpu_rbf: solver not fitted — call fit() first".into(),
            ));
        }
        let result = match self.config.dispatch {
            GpuDispatch::Cpu => self.evaluate_cpu(query_points),
            GpuDispatch::Simulated => self.evaluate_simulated(query_points),
        };
        Ok(result)
    }

    /// Batch evaluate — semantically identical to `evaluate` (both paths are
    /// synchronous in this simulation; the async API surface is preserved for
    /// future hardware backends).
    pub fn evaluate_batch(&self, query_points: &[[f64; 2]]) -> Result<Vec<f64>, InterpolateError> {
        self.evaluate(query_points)
    }

    /// Number of training points.
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// The fitted RBF weight vector.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn evaluate_cpu(&self, query_points: &[[f64; 2]]) -> Vec<f64> {
        query_points.iter().map(|q| self.eval_single(q)).collect()
    }

    fn evaluate_simulated(&self, query_points: &[[f64; 2]]) -> Vec<f64> {
        let block = self.config.block_size.max(1);
        let mut out = Vec::with_capacity(query_points.len());
        for chunk in query_points.chunks(block) {
            for q in chunk {
                out.push(self.eval_single(q));
            }
        }
        out
    }

    fn eval_single(&self, q: &[f64; 2]) -> f64 {
        self.points
            .iter()
            .zip(self.weights.iter())
            .map(|(p, &w)| {
                let dx = q[0] - p[0];
                let dy = q[1] - p[1];
                let r = (dx * dx + dy * dy).sqrt();
                w * rbf_kernel(r, &self.config.kernel, self.config.epsilon)
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Evaluate a single RBF kernel at radius `r`.
///
/// - **Gaussian**: φ(r) = exp(-(ε r)²)
/// - **Multiquadric**: φ(r) = √(1 + (ε r)²)
/// - **ThinPlateSpline**: φ(r) = r² ln(r) for r > 0, else 0
/// - **Linear**: φ(r) = r
pub fn rbf_kernel(r: f64, kernel: &GpuRbfKernel, epsilon: f64) -> f64 {
    match kernel {
        GpuRbfKernel::Gaussian => {
            let er = epsilon * r;
            (-(er * er)).exp()
        }
        GpuRbfKernel::Multiquadric => {
            let er = epsilon * r;
            (1.0 + er * er).sqrt()
        }
        GpuRbfKernel::ThinPlateSpline => {
            if r < 1e-300 {
                0.0
            } else {
                r * r * r.ln()
            }
        }
        GpuRbfKernel::Linear => r,
    }
}

/// Build the N×N RBF collocation matrix.
///
/// In `Simulated` mode the rows are computed in `block_size`-row chunks.
pub fn build_rbf_matrix(
    points: &[[f64; 2]],
    kernel: &GpuRbfKernel,
    epsilon: f64,
    block_size: usize,
    dispatch: &GpuDispatch,
) -> Vec<f64> {
    let n = points.len();
    let mut mat = vec![0.0f64; n * n];

    let block = block_size.max(1);
    let row_chunks: Vec<std::ops::Range<usize>> = match dispatch {
        GpuDispatch::Simulated => {
            // Chunk rows to simulate GPU thread-blocks
            (0..n)
                .step_by(block)
                .map(|start| start..n.min(start + block))
                .collect()
        }
        GpuDispatch::Cpu => vec![0..n],
    };

    for chunk in row_chunks {
        for i in chunk {
            for j in 0..n {
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                let r = (dx * dx + dy * dy).sqrt();
                mat[i * n + j] = rbf_kernel(r, kernel, epsilon);
            }
        }
    }
    mat
}

/// Solve the linear system A x = b using partial-pivot Gaussian elimination.
///
/// `a` is stored row-major with shape n×n. Returns the solution vector x.
pub fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, InterpolateError> {
    // Build augmented matrix [A | b]
    let mut aug: Vec<f64> = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let (pivot_row, pivot_val) = (col..n).map(|r| (r, aug[r * (n + 1) + col].abs())).fold(
            (col, 0.0_f64),
            |(best_r, best_v), (r, v)| {
                if v > best_v {
                    (r, v)
                } else {
                    (best_r, best_v)
                }
            },
        );

        if pivot_val < 1e-14 {
            return Err(InterpolateError::LinalgError(
                "gpu_rbf: singular RBF matrix — consider regularisation or distinct points".into(),
            ));
        }

        // Swap rows
        if pivot_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, pivot_row * (n + 1) + j);
            }
        }

        // Eliminate below
        let pivot_inv = 1.0 / aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] * pivot_inv;
            for j in col..=n {
                let sub = factor * aug[col * (n + 1) + j];
                aug[row * (n + 1) + j] -= sub;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_5_point_solver(kernel: GpuRbfKernel) -> GpuRbfSolver {
        let config = GpuRbfConfig {
            kernel,
            epsilon: 1.0,
            dispatch: GpuDispatch::Simulated,
            block_size: 32,
        };
        GpuRbfSolver::new(config)
    }

    /// Fit on 5 points and check that evaluating at the training points
    /// reproduces the values within 1e-6.
    #[test]
    fn test_fit_and_reproduce_training_values() {
        let points = vec![
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 2.5];

        let mut solver = make_5_point_solver(GpuRbfKernel::Gaussian);
        solver.fit(&points, &values).expect("fit failed");

        let out = solver.evaluate(&points).expect("evaluate failed");
        for (got, &expected) in out.iter().zip(values.iter()) {
            assert!(
                (got - expected).abs() < 1e-6,
                "mismatch: got {got:.8} expected {expected:.8}"
            );
        }
    }

    /// Gaussian and ThinPlateSpline kernels must give different results at a
    /// query point.
    #[test]
    fn test_gaussian_vs_tps_kernels_differ() {
        // Use 5 well-separated points so both kernels produce non-singular systems
        let points = vec![
            [0.0_f64, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 2.5];
        let query = vec![[0.5_f64, 0.5], [1.5, 0.5]];

        let mut gauss = make_5_point_solver(GpuRbfKernel::Gaussian);
        gauss.fit(&points, &values).expect("fit gaussian failed");

        let mut tps = make_5_point_solver(GpuRbfKernel::ThinPlateSpline);
        tps.fit(&points, &values).expect("fit tps failed");

        let g_out = gauss.evaluate(&query).expect("eval gaussian failed");
        let t_out = tps.evaluate(&query).expect("eval tps failed");

        // Results must differ between kernels at some query point
        let differ = g_out
            .iter()
            .zip(t_out.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "Gaussian and TPS should produce different evaluations"
        );
    }

    /// evaluate_batch must return the same values as evaluate.
    #[test]
    fn test_evaluate_batch_equals_evaluate() {
        let points = vec![[0.0_f64, 0.0], [1.0, 0.5], [0.3, 0.7]];
        let values = vec![0.0, 1.0, 0.5];
        let query = vec![[0.2_f64, 0.3], [0.8, 0.1], [0.5, 0.5]];

        let mut solver = make_5_point_solver(GpuRbfKernel::Multiquadric);
        solver.fit(&points, &values).expect("fit failed");

        let ev = solver.evaluate(&query).expect("evaluate failed");
        let ev_batch = solver
            .evaluate_batch(&query)
            .expect("evaluate_batch failed");

        for (a, b) in ev.iter().zip(ev_batch.iter()) {
            assert!((a - b).abs() < 1e-15, "evaluate and evaluate_batch differ");
        }
    }

    /// Multiquadric kernel at r = 0 should return 1.0 (√(1 + 0) = 1).
    #[test]
    fn test_multiquadric_at_zero() {
        let val = rbf_kernel(0.0, &GpuRbfKernel::Multiquadric, 2.5);
        assert!(
            (val - 1.0).abs() < 1e-15,
            "Multiquadric at r=0 should be 1.0, got {val}"
        );
    }

    /// ThinPlateSpline at r = 0 must be exactly 0.
    #[test]
    fn test_tps_kernel_at_zero() {
        let val = rbf_kernel(0.0, &GpuRbfKernel::ThinPlateSpline, 1.0);
        assert_eq!(val, 0.0, "TPS at r=0 should be 0.0");
    }

    /// Gaussian kernel is strictly positive.
    #[test]
    fn test_gaussian_kernel_positive() {
        for r in [0.0, 0.5, 1.0, 2.0, 10.0] {
            let v = rbf_kernel(r, &GpuRbfKernel::Gaussian, 1.0);
            assert!(
                v > 0.0,
                "Gaussian kernel must be positive, got {v} at r={r}"
            );
        }
    }

    /// Both dispatch modes compute real RBF values on the CPU; the chunked
    /// (`Simulated`) path must reproduce the plain CPU reference exactly.
    #[test]
    fn test_cpu_reference_matches_chunked_dispatch() {
        let points = vec![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let values = vec![1.0, 2.0, 3.0];
        let query = vec![[0.5_f64, 0.5], [0.1, 0.9]];

        let mut cpu_solver = GpuRbfSolver::new(GpuRbfConfig {
            dispatch: GpuDispatch::Cpu,
            ..GpuRbfConfig::default()
        });
        let mut sim_solver = GpuRbfSolver::new(GpuRbfConfig {
            dispatch: GpuDispatch::Simulated,
            ..GpuRbfConfig::default()
        });

        cpu_solver.fit(&points, &values).expect("cpu fit failed");
        sim_solver.fit(&points, &values).expect("sim fit failed");

        let cpu_out = cpu_solver.evaluate(&query).expect("cpu eval failed");
        let sim_out = sim_solver.evaluate(&query).expect("sim eval failed");

        for (c, s) in cpu_out.iter().zip(sim_out.iter()) {
            assert!(
                (c - s).abs() < 1e-12,
                "CPU reference vs chunked dispatch differ: {c} vs {s}"
            );
        }
    }
}
