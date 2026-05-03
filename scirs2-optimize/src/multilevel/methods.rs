//! Multi-level and multi-fidelity optimization methods
//!
//! Implements:
//! - MultilevelOptimizer (coarse-to-fine)
//! - VariableFidelity model manager
//! - MFRBF surrogate with additive correction
//! - TrustHierarchy (hierarchical trust regions)
//! - MultigridOptimizer (multigrid-inspired V/W cycles)

use crate::error::{OptimizeError, OptimizeResult};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a multi-level / multi-fidelity optimization
#[derive(Debug, Clone)]
pub struct MultilevelResult {
    /// Optimal solution
    pub x: Vec<f64>,
    /// Function value at optimal solution (high-fidelity)
    pub fun: f64,
    /// Number of high-fidelity function evaluations
    pub n_hifi_evals: usize,
    /// Number of low-fidelity function evaluations
    pub n_lofi_evals: usize,
    /// Total number of iterations across all levels
    pub n_iter: usize,
    /// Whether the algorithm converged
    pub success: bool,
    /// Termination message
    pub message: String,
}

// ---------------------------------------------------------------------------
// FidelityLevel
// ---------------------------------------------------------------------------

/// A single fidelity level with an evaluation cost multiplier
pub struct FidelityLevel<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// Function for this fidelity level
    pub func: F,
    /// Relative cost (higher = more expensive)
    pub cost: f64,
}

impl<F: Fn(&[f64]) -> f64> FidelityLevel<F> {
    /// Create a new fidelity level
    pub fn new(func: F, cost: f64) -> Self {
        FidelityLevel { func, cost }
    }

    /// Evaluate the function
    pub fn eval(&self, x: &[f64]) -> f64 {
        (self.func)(x)
    }
}

// ---------------------------------------------------------------------------
// MultilevelOptions / MultilevelOptimizer
// ---------------------------------------------------------------------------

/// Options for the multi-level optimizer
#[derive(Debug, Clone)]
pub struct MultilevelOptions {
    /// Maximum iterations per fidelity level
    pub max_iter_per_level: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Step size for gradient estimation
    pub fd_step: f64,
    /// Initial step size for gradient descent
    pub step_size: f64,
    /// Step shrinkage factor in line search
    pub step_shrink: f64,
    /// Number of coarse-level pre-smoothing steps before switching to fine
    pub pre_smooth: usize,
    /// Minimum improvement before promoting to next level
    pub level_promotion_tol: f64,
}

impl Default for MultilevelOptions {
    fn default() -> Self {
        MultilevelOptions {
            max_iter_per_level: 100,
            tol: 1e-6,
            fd_step: 1e-7,
            step_size: 0.1,
            step_shrink: 0.5,
            pre_smooth: 5,
            level_promotion_tol: 1e-3,
        }
    }
}

/// Multi-level (coarse-to-fine) optimizer
///
/// Starts at the coarsest (lowest-fidelity) level, optimizes until convergence
/// or plateau, then promotes to the next finer level using the current solution
/// as warm start.
pub struct MultilevelOptimizer<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// Fidelity levels ordered from coarse to fine
    levels: Vec<FidelityLevel<F>>,
    /// Initial point
    x0: Vec<f64>,
    /// Algorithm options
    options: MultilevelOptions,
}

impl<F: Fn(&[f64]) -> f64> MultilevelOptimizer<F> {
    /// Create a new multi-level optimizer
    ///
    /// Levels should be ordered from coarsest (cheapest) to finest (most expensive).
    pub fn new(levels: Vec<FidelityLevel<F>>, x0: Vec<f64>, options: MultilevelOptions) -> Self {
        MultilevelOptimizer {
            levels,
            x0,
            options,
        }
    }

    /// Run the multi-level optimization
    pub fn minimize(&mut self) -> OptimizeResult<MultilevelResult> {
        if self.levels.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "At least one fidelity level required".to_string(),
            ));
        }
        let n = self.x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        let mut x = self.x0.clone();
        let mut n_hifi = 0usize;
        let mut n_lofi = 0usize;
        let mut total_iter = 0usize;
        let n_levels = self.levels.len();
        let h = self.options.fd_step;

        for (level_idx, level) in self.levels.iter().enumerate() {
            let is_finest = level_idx == n_levels - 1;
            let max_iter = if is_finest {
                self.options.max_iter_per_level * 3
            } else {
                self.options.max_iter_per_level
            };

            let mut f_prev = level.eval(&x);
            if is_finest {
                n_hifi += 1;
            } else {
                n_lofi += 1;
            }

            for _iter in 0..max_iter {
                total_iter += 1;

                // Finite-difference gradient
                let mut grad = vec![0.0f64; n];
                for i in 0..n {
                    let mut xf = x.clone();
                    xf[i] += h;
                    let f_fwd = level.eval(&xf);
                    grad[i] = (f_fwd - f_prev) / h;
                    if is_finest {
                        n_hifi += 1;
                    } else {
                        n_lofi += 1;
                    }
                }

                let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if gnorm < self.options.tol {
                    break;
                }

                // Armijo line search
                let mut step = self.options.step_size;
                let c1 = 1e-4;
                let mut x_new = x.clone();
                for _ in 0..30 {
                    for i in 0..n {
                        x_new[i] = x[i] - step * grad[i];
                    }
                    let f_new = level.eval(&x_new);
                    if is_finest {
                        n_hifi += 1;
                    } else {
                        n_lofi += 1;
                    }
                    if f_new <= f_prev - c1 * step * gnorm * gnorm {
                        break;
                    }
                    step *= self.options.step_shrink;
                }

                let f_new = level.eval(&x_new);
                if is_finest {
                    n_hifi += 1;
                } else {
                    n_lofi += 1;
                }

                let delta = (f_new - f_prev).abs();
                x = x_new;
                f_prev = f_new;

                if delta < self.options.tol * (1.0 + f_prev.abs()) {
                    break;
                }
            }

            // Check if further refinement is needed
            if !is_finest {
                // Only promote if improvement was significant
                let _current_f = f_prev;
            }
        }

        // Final evaluation at finest level
        let final_f = self
            .levels
            .last()
            .map(|l| l.eval(&x))
            .unwrap_or(f64::INFINITY);
        n_hifi += 1;

        Ok(MultilevelResult {
            x,
            fun: final_f,
            n_hifi_evals: n_hifi,
            n_lofi_evals: n_lofi,
            n_iter: total_iter,
            success: true,
            message: "Multi-level optimization completed".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// VariableFidelity
// ---------------------------------------------------------------------------

/// Options for variable fidelity optimization
#[derive(Debug, Clone)]
pub struct VariableFidelityOptions {
    /// Budget: total weighted evaluation cost allowed
    pub total_budget: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Finite difference step
    pub fd_step: f64,
    /// Trust region radius for switching to high fidelity
    pub hifi_trust_radius: f64,
}

impl Default for VariableFidelityOptions {
    fn default() -> Self {
        VariableFidelityOptions {
            total_budget: 1000.0,
            max_iter: 500,
            tol: 1e-6,
            fd_step: 1e-7,
            hifi_trust_radius: 0.1,
        }
    }
}

/// Variable fidelity model manager
///
/// Manages switching between low- and high-fidelity models based on
/// a budget constraint. Uses low-fidelity exploration and high-fidelity
/// validation/correction near candidate optima.
pub struct VariableFidelity<FL, FH>
where
    FL: Fn(&[f64]) -> f64,
    FH: Fn(&[f64]) -> f64,
{
    /// Low-fidelity (cheap) model
    pub low_fi: FL,
    /// High-fidelity (expensive) model
    pub high_fi: FH,
    /// Cost of one low-fidelity evaluation (in budget units)
    pub cost_low: f64,
    /// Cost of one high-fidelity evaluation (in budget units)
    pub cost_high: f64,
    /// Algorithm options
    pub options: VariableFidelityOptions,
}

impl<FL, FH> VariableFidelity<FL, FH>
where
    FL: Fn(&[f64]) -> f64,
    FH: Fn(&[f64]) -> f64,
{
    /// Create a new variable fidelity optimizer
    pub fn new(
        low_fi: FL,
        high_fi: FH,
        cost_low: f64,
        cost_high: f64,
        options: VariableFidelityOptions,
    ) -> Self {
        VariableFidelity {
            low_fi,
            high_fi,
            cost_low,
            cost_high,
            options,
        }
    }

    /// Evaluate low-fidelity model
    pub fn eval_low(&self, x: &[f64]) -> f64 {
        (self.low_fi)(x)
    }

    /// Evaluate high-fidelity model
    pub fn eval_high(&self, x: &[f64]) -> f64 {
        (self.high_fi)(x)
    }

    /// Run variable fidelity optimization
    pub fn minimize(&self, x0: &[f64]) -> OptimizeResult<MultilevelResult> {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        let mut x = x0.to_vec();
        let mut budget_used = 0.0f64;
        let h = self.options.fd_step;
        let mut n_hifi = 0usize;
        let mut n_lofi = 0usize;
        let mut n_iter = 0usize;

        // Phase 1: Low-fidelity exploration
        let mut f_low = self.eval_low(&x);
        budget_used += self.cost_low;
        n_lofi += 1;

        while budget_used < self.options.total_budget * 0.7 && n_iter < self.options.max_iter {
            n_iter += 1;

            // FD gradient using low-fi
            let mut grad = vec![0.0f64; n];
            for i in 0..n {
                let mut xf = x.clone();
                xf[i] += h;
                grad[i] = (self.eval_low(&xf) - f_low) / h;
                budget_used += self.cost_low;
                n_lofi += 1;
            }

            let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if gnorm < self.options.tol {
                break;
            }

            // Gradient step
            let step = 0.05f64;
            let mut x_new = vec![0.0f64; n];
            for i in 0..n {
                x_new[i] = x[i] - step * grad[i];
            }
            let f_new = self.eval_low(&x_new);
            budget_used += self.cost_low;
            n_lofi += 1;

            if f_new < f_low {
                x = x_new;
                f_low = f_new;
            } else {
                break;
            }
        }

        // Phase 2: High-fidelity refinement near the low-fi optimum
        let mut f_high = self.eval_high(&x);
        budget_used += self.cost_high;
        n_hifi += 1;

        // Additive correction: δ = f_H(x) - f_L(x)
        let delta_correction = f_high - f_low;

        while budget_used < self.options.total_budget && n_iter < self.options.max_iter * 2 {
            n_iter += 1;

            // Use corrected surrogate: f_L(x) + δ for gradient
            let mut grad = vec![0.0f64; n];
            for i in 0..n {
                let mut xf = x.clone();
                xf[i] += h;
                // Corrected model gradient
                grad[i] = (self.eval_low(&xf) + delta_correction - f_high) / h;
                budget_used += self.cost_low;
                n_lofi += 1;
            }

            let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if gnorm < self.options.tol {
                break;
            }

            // Try step
            let step = 0.02f64;
            let mut x_new = vec![0.0f64; n];
            for i in 0..n {
                x_new[i] = x[i] - step * grad[i];
            }

            // Validate with high-fidelity if within trust radius
            let dist: f64 = x_new
                .iter()
                .zip(x.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if dist < self.options.hifi_trust_radius {
                let f_new_high = self.eval_high(&x_new);
                budget_used += self.cost_high;
                n_hifi += 1;
                if f_new_high < f_high {
                    x = x_new;
                    f_high = f_new_high;
                } else {
                    break;
                }
            } else {
                // Use low-fi acceptance
                let f_new_low = self.eval_low(&x_new);
                budget_used += self.cost_low;
                n_lofi += 1;
                if f_new_low + delta_correction < f_high {
                    x = x_new;
                }
                break;
            }
        }

        Ok(MultilevelResult {
            x,
            fun: f_high,
            n_hifi_evals: n_hifi,
            n_lofi_evals: n_lofi,
            n_iter,
            success: true,
            message: format!(
                "Variable fidelity optimization completed (budget used: {:.2})",
                budget_used
            ),
        })
    }
}

// ---------------------------------------------------------------------------
// Linear algebra helper
// ---------------------------------------------------------------------------

/// Solve the linear system A x = b using Gaussian elimination with partial
/// pivoting.  Returns `None` if A is singular (or nearly so).
fn gaussian_elimination(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    debug_assert_eq!(a.len(), n);

    // Build augmented matrix [A | b]
    let mut mat: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &rhs)| {
            let mut r = row.clone();
            r.push(rhs);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivot: find row with largest absolute value in this column
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                mat[r1][col]
                    .abs()
                    .partial_cmp(&mat[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        mat.swap(col, pivot_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-15 {
            return None; // singular
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for j in col..=n {
                let val = mat[col][j] * factor;
                mat[row][j] -= val;
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = mat[i][n];
        for j in (i + 1)..n {
            sum -= mat[i][j] * x[j];
        }
        x[i] = sum / mat[i][i];
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// MFRBF: Multi-Fidelity RBF Surrogate
// ---------------------------------------------------------------------------

/// Options for the MFRBF surrogate
#[derive(Debug, Clone)]
pub struct MfRbfOptions {
    /// RBF kernel width parameter
    pub length_scale: f64,
    /// Maximum surrogate iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Number of initial sample points
    pub n_initial_samples: usize,
    /// Budget for high-fidelity evaluations during optimization
    pub hifi_budget: usize,
}

impl Default for MfRbfOptions {
    fn default() -> Self {
        MfRbfOptions {
            length_scale: 1.0,
            max_iter: 50,
            tol: 1e-5,
            n_initial_samples: 5,
            hifi_budget: 20,
        }
    }
}

/// Sample point for the RBF surrogate
#[derive(Debug, Clone)]
struct SamplePoint {
    x: Vec<f64>,
    f_low: f64,
    f_high: f64,
}

impl SamplePoint {
    fn correction(&self) -> f64 {
        self.f_high - self.f_low
    }
}

/// Multi-fidelity RBF surrogate with additive correction.
///
/// Builds a correction surface δ(x) = f_H(x) - f_L(x) using RBF interpolation
/// on a small number of high-fidelity samples, then optimizes f_L(x) + δ̂(x).
pub struct MfRbf<FL, FH>
where
    FL: Fn(&[f64]) -> f64,
    FH: Fn(&[f64]) -> f64,
{
    /// Low-fidelity model
    pub low_fi: FL,
    /// High-fidelity model
    pub high_fi: FH,
    /// Algorithm options
    pub options: MfRbfOptions,
    /// Sample points collected so far
    samples: Vec<SamplePoint>,
}

impl<FL, FH> MfRbf<FL, FH>
where
    FL: Fn(&[f64]) -> f64,
    FH: Fn(&[f64]) -> f64,
{
    /// Create a new MfRbf optimizer
    pub fn new(low_fi: FL, high_fi: FH, options: MfRbfOptions) -> Self {
        MfRbf {
            low_fi,
            high_fi,
            options,
            samples: Vec::new(),
        }
    }

    fn eval_low(&self, x: &[f64]) -> f64 {
        (self.low_fi)(x)
    }

    fn eval_high(&self, x: &[f64]) -> f64 {
        (self.high_fi)(x)
    }

    /// Gaussian RBF kernel
    fn rbf_kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        let dist_sq: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        (-dist_sq / (2.0 * self.options.length_scale.powi(2))).exp()
    }

    /// Evaluate the correction surrogate at x using RBF interpolation
    fn eval_correction_surrogate(&self, x: &[f64]) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }

        // Solve the RBF system on the fly: δ̂(x) = Σ_i w_i φ(||x - x_i||)
        // Weights w = Φ^{-1} δ  where Φ_ij = φ(||x_i - x_j||)
        let ns = self.samples.len();
        let mut phi = vec![vec![0.0f64; ns]; ns];
        for i in 0..ns {
            for j in 0..ns {
                phi[i][j] = self.rbf_kernel(&self.samples[i].x, &self.samples[j].x);
            }
            // Nugget regularization for numerical stability
            phi[i][i] += 1e-6;
        }

        let corrections: Vec<f64> = self.samples.iter().map(|s| s.correction()).collect();

        // Solve Φ w = δ using Gaussian elimination with partial pivoting.
        // Jacobi iteration diverges when Φ is not diagonally dominant (common for
        // clustered RBF centres), so we need a direct solver here.
        let w = match gaussian_elimination(&phi, &corrections) {
            Some(sol) => sol,
            None => return 0.0,
        };

        // Evaluate surrogate: δ̂(x) = Σ w_i φ(||x - x_i||)
        w.iter()
            .zip(self.samples.iter())
            .map(|(wi, si)| wi * self.rbf_kernel(x, &si.x))
            .sum()
    }

    /// Add a sample (evaluate both fidelities)
    pub fn add_sample(&mut self, x: Vec<f64>) {
        let f_low = self.eval_low(&x);
        let f_high = self.eval_high(&x);
        self.samples.push(SamplePoint { x, f_low, f_high });
    }

    /// Evaluate the corrected surrogate: f_L(x) + δ̂(x)
    pub fn eval_corrected(&self, x: &[f64]) -> f64 {
        self.eval_low(x) + self.eval_correction_surrogate(x)
    }

    /// Run the MFRBF optimization
    pub fn minimize(&mut self, x0: &[f64]) -> OptimizeResult<MultilevelResult> {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        let mut x = x0.to_vec();
        let h = 1e-7f64;
        let mut n_hifi = 0usize;
        let mut n_lofi = 0usize;
        let mut n_iter = 0usize;

        // Initial sampling around x0
        self.add_sample(x.clone());
        n_hifi += 1;
        n_lofi += 1;

        for k in 1..self.options.n_initial_samples {
            // Perturb x0 to generate diverse samples
            let mut xs = x.clone();
            let offset = 0.5 * (k as f64);
            xs[k % n] += offset;
            self.add_sample(xs);
            n_hifi += 1;
            n_lofi += 1;
        }

        // Optimize corrected surrogate using gradient descent with Armijo line-search.
        // Track the best point seen to avoid returning a worse solution when the RBF
        // correction is perturbed by newly added high-fidelity samples.
        let mut f_prev = self.eval_corrected(&x);
        let mut x_best = x.clone();
        let mut f_best = f_prev;

        for _iter in 0..self.options.max_iter {
            n_iter += 1;

            let mut grad = vec![0.0f64; n];
            for i in 0..n {
                let mut xf = x.clone();
                xf[i] += h;
                grad[i] = (self.eval_corrected(&xf) - f_prev) / h;
                n_lofi += 1; // surrogate uses low-fi
            }

            let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if gnorm < self.options.tol {
                break;
            }

            // Armijo backtracking line-search: start with step=0.1 and halve until
            // sufficient decrease is achieved (or a minimum step is reached).
            let armijo_c = 1e-4;
            let mut step = 0.1f64;
            let mut x_new = vec![0.0f64; n];
            let mut f_new = f_prev;
            let mut descent_found = false;
            for _ls in 0..20 {
                for i in 0..n {
                    x_new[i] = x[i] - step * grad[i];
                }
                f_new = self.eval_corrected(&x_new);
                n_lofi += 1;
                if f_new <= f_prev - armijo_c * step * gnorm * gnorm {
                    descent_found = true;
                    break;
                }
                step *= 0.5;
            }

            if descent_found || f_new < f_prev {
                x = x_new.clone();
                f_prev = f_new;

                // Track the best point over all iterations so that RBF perturbations
                // from new high-fidelity samples cannot cause us to return a worse x.
                if f_new < f_best {
                    x_best = x.clone();
                    f_best = f_new;
                }

                // Refine with high-fidelity if budget allows
                if n_hifi < self.options.hifi_budget {
                    self.add_sample(x.clone());
                    n_hifi += 1;
                    n_lofi += 1;
                    // Recompute corrected value with updated RBF
                    f_prev = self.eval_corrected(&x);
                    // Retain best in case corrected value jumped up after new sample
                    if f_prev < f_best {
                        f_best = f_prev;
                        x_best = x.clone();
                    }
                }
            }
            // No descent: gradient norm already checked above; continue to let
            // subsequent iterations explore further rather than terminating early.
        }

        // Final high-fidelity evaluation at the best surrogate point found
        let final_hifi = self.eval_high(&x_best);
        n_hifi += 1;

        Ok(MultilevelResult {
            x: x_best,
            fun: final_hifi,
            n_hifi_evals: n_hifi,
            n_lofi_evals: n_lofi,
            n_iter,
            success: true,
            message: "MFRBF optimization completed".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// TrustHierarchy
// ---------------------------------------------------------------------------

/// Options for the hierarchical trust region
#[derive(Debug, Clone)]
pub struct TrustHierarchyOptions {
    /// Initial trust region radii per level
    pub initial_radii: Vec<f64>,
    /// Minimum trust region radius
    pub min_radius: f64,
    /// Maximum trust region radius
    pub max_radius: f64,
    /// Acceptance threshold (rho_min)
    pub eta1: f64,
    /// Good step threshold (rho_good)
    pub eta2: f64,
    /// Expansion factor
    pub gamma_inc: f64,
    /// Contraction factor
    pub gamma_dec: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl Default for TrustHierarchyOptions {
    fn default() -> Self {
        TrustHierarchyOptions {
            initial_radii: vec![1.0, 0.1],
            min_radius: 1e-8,
            max_radius: 10.0,
            eta1: 0.1,
            eta2: 0.75,
            gamma_inc: 2.0,
            gamma_dec: 0.5,
            max_iter: 200,
            tol: 1e-6,
        }
    }
}

/// Hierarchical trust region management across fidelity levels.
///
/// Uses a hierarchy of trust regions, where the radius at finer levels is
/// bounded by the radius at coarser levels. The algorithm cycles through
/// levels, performing trust-region steps and updating radii.
pub struct TrustHierarchy<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// Models at each level (ordered coarse to fine)
    pub levels: Vec<F>,
    /// Algorithm options
    pub options: TrustHierarchyOptions,
}

impl<F: Fn(&[f64]) -> f64> TrustHierarchy<F> {
    /// Create a new hierarchical trust region solver
    pub fn new(levels: Vec<F>, options: TrustHierarchyOptions) -> Self {
        TrustHierarchy { levels, options }
    }

    /// Solve the optimization problem
    pub fn minimize(&self, x0: &[f64]) -> OptimizeResult<MultilevelResult> {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }
        if self.levels.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "At least one level required".to_string(),
            ));
        }

        let n_levels = self.levels.len();
        let mut radii: Vec<f64> = if self.options.initial_radii.len() >= n_levels {
            self.options.initial_radii[..n_levels].to_vec()
        } else {
            let mut r = self.options.initial_radii.clone();
            while r.len() < n_levels {
                r.push(*r.last().unwrap_or(&1.0) * 0.5);
            }
            r
        };

        let mut x = x0.to_vec();
        let h = 1e-7f64;
        let mut n_evals = 0usize;
        let mut n_iter = 0usize;

        let finest_idx = n_levels - 1;
        let mut f_cur = (self.levels[finest_idx])(&x);
        n_evals += 1;

        for _outer in 0..self.options.max_iter {
            n_iter += 1;

            // Compute gradient using finest level
            let mut grad = vec![0.0f64; n];
            for i in 0..n {
                let mut xf = x.clone();
                xf[i] += h;
                grad[i] = ((self.levels[finest_idx])(&xf) - f_cur) / h;
                n_evals += 1;
            }

            let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if gnorm < self.options.tol {
                break;
            }

            let radius = radii[finest_idx];

            // Cauchy point step (steepest descent clipped to trust region)
            let cauchy_step = (radius / gnorm).min(radius);
            let mut x_trial = vec![0.0f64; n];
            for i in 0..n {
                x_trial[i] = x[i] - cauchy_step * grad[i] / gnorm;
            }

            // Compute actual and predicted reduction
            let f_trial = (self.levels[finest_idx])(&x_trial);
            n_evals += 1;
            let actual_red = f_cur - f_trial;
            let predicted_red = cauchy_step * gnorm; // gradient model prediction

            if predicted_red.abs() < 1e-14 {
                break;
            }

            let rho = actual_red / predicted_red;

            // Update trust region radius
            if rho < self.options.eta1 {
                radii[finest_idx] = (radius * self.options.gamma_dec).max(self.options.min_radius);
            } else {
                // Accept step
                x = x_trial;
                f_cur = f_trial;

                if rho > self.options.eta2 {
                    radii[finest_idx] =
                        (radius * self.options.gamma_inc).min(self.options.max_radius);
                }

                // Propagate radius changes to coarser levels
                for l in (0..finest_idx).rev() {
                    radii[l] = radii[l + 1] * 2.0;
                }
            }

            if radii[finest_idx] < self.options.min_radius {
                break;
            }
        }

        Ok(MultilevelResult {
            x,
            fun: f_cur,
            n_hifi_evals: n_evals,
            n_lofi_evals: 0,
            n_iter,
            success: radii[finest_idx] >= self.options.min_radius,
            message: "Trust hierarchy optimization completed".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// MultigridOptimizer (V/W cycle)
// ---------------------------------------------------------------------------

/// Options for the multigrid optimizer
#[derive(Debug, Clone)]
pub struct MultigridOptions {
    /// Number of V-cycles to perform
    pub n_cycles: usize,
    /// Number of smoothing steps per level visit
    pub n_smooth: usize,
    /// Type of cycle: 'V' or 'W'
    pub cycle_type: CycleType,
    /// Convergence tolerance
    pub tol: f64,
    /// Gradient step size for smoothing
    pub smooth_step: f64,
    /// Restriction/interpolation factor between levels
    pub coarsening_factor: f64,
}

/// Multigrid cycle type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleType {
    /// V-cycle (one visit per level)
    V,
    /// W-cycle (two visits at coarser levels)
    W,
}

impl Default for MultigridOptions {
    fn default() -> Self {
        MultigridOptions {
            n_cycles: 20,
            n_smooth: 3,
            cycle_type: CycleType::V,
            tol: 1e-6,
            smooth_step: 0.1,
            coarsening_factor: 2.0,
        }
    }
}

/// Multigrid-inspired optimizer using a hierarchy of approximation levels.
///
/// The "grid levels" here represent different approximation granularities.
/// Restriction maps a fine iterate to a coarser space; prolongation
/// (interpolation) maps the coarse correction back to the fine space.
///
/// On each smoothing step, gradient descent is applied. The multigrid
/// philosophy avoids slow convergence of fine-level iterations by first
/// making progress on coarser levels.
pub struct MultigridOptimizer<F>
where
    F: Fn(&[f64]) -> f64,
{
    /// Objective function (same function evaluated at different scales)
    pub func: F,
    /// Dimensionality at finest level
    pub n_fine: usize,
    /// Number of multigrid levels
    pub n_levels: usize,
    /// Algorithm options
    pub options: MultigridOptions,
}

impl<F: Fn(&[f64]) -> f64> MultigridOptimizer<F> {
    /// Create a new multigrid optimizer
    pub fn new(func: F, n_fine: usize, n_levels: usize, options: MultigridOptions) -> Self {
        MultigridOptimizer {
            func,
            n_fine,
            n_levels,
            options,
        }
    }

    /// Evaluate function
    fn eval(&self, x: &[f64]) -> f64 {
        (self.func)(x)
    }

    /// Restrict fine vector to coarser level (simple averaging / decimation)
    fn restrict(&self, x_fine: &[f64], coarse_n: usize) -> Vec<f64> {
        let fine_n = x_fine.len();
        if coarse_n == 0 || fine_n == 0 {
            return vec![];
        }
        let mut x_coarse = vec![0.0f64; coarse_n];
        for i in 0..coarse_n {
            let ratio = fine_n as f64 / coarse_n as f64;
            let start = (i as f64 * ratio) as usize;
            let end = ((i + 1) as f64 * ratio) as usize;
            let end = end.min(fine_n);
            if start >= end {
                x_coarse[i] = x_fine[start.min(fine_n - 1)];
            } else {
                let sum: f64 = x_fine[start..end].iter().sum();
                x_coarse[i] = sum / (end - start) as f64;
            }
        }
        x_coarse
    }

    /// Prolongate (interpolate) coarse correction to fine level
    fn prolongate(&self, correction_coarse: &[f64], fine_n: usize) -> Vec<f64> {
        let coarse_n = correction_coarse.len();
        if coarse_n == 0 || fine_n == 0 {
            return vec![0.0f64; fine_n];
        }
        let mut fine = vec![0.0f64; fine_n];
        for i in 0..fine_n {
            let t = i as f64 / (fine_n - 1).max(1) as f64;
            let coarse_pos = t * (coarse_n - 1).max(0) as f64;
            let coarse_i = coarse_pos as usize;
            let frac = coarse_pos - coarse_i as f64;
            let v0 = correction_coarse[coarse_i.min(coarse_n - 1)];
            let v1 = if coarse_i + 1 < coarse_n {
                correction_coarse[coarse_i + 1]
            } else {
                v0
            };
            fine[i] = v0 * (1.0 - frac) + v1 * frac;
        }
        fine
    }

    /// Perform `n_smooth` gradient descent steps at current level
    fn smooth(&self, x: &[f64], n_smooth: usize, step: f64, nfev: &mut usize) -> Vec<f64> {
        let n = x.len();
        let h = 1e-7f64;
        let mut xc = x.to_vec();

        for _ in 0..n_smooth {
            let f0 = self.eval(&xc);
            *nfev += 1;
            let mut grad = vec![0.0f64; n];
            for i in 0..n {
                let mut xf = xc.clone();
                xf[i] += h;
                grad[i] = (self.eval(&xf) - f0) / h;
                *nfev += 1;
            }
            let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if gnorm < 1e-10 {
                break;
            }
            for i in 0..n {
                xc[i] -= step * grad[i];
            }
        }
        xc
    }

    /// Perform one V-cycle starting at `level` (0 = finest)
    fn v_cycle(&self, x: Vec<f64>, level: usize, nfev: &mut usize) -> Vec<f64> {
        let n = x.len();

        // Pre-smoothing
        let mut x = self.smooth(&x, self.options.n_smooth, self.options.smooth_step, nfev);

        if level < self.n_levels - 1 && n > 1 {
            // Restrict to coarser level
            let coarse_n = (n as f64 / self.options.coarsening_factor).ceil() as usize;
            let coarse_n = coarse_n.max(1);
            let x_coarse = self.restrict(&x, coarse_n);

            // Recurse at coarser level
            let x_coarse_smooth = self.v_cycle(x_coarse.clone(), level + 1, nfev);

            // Compute correction
            let correction: Vec<f64> = x_coarse_smooth
                .iter()
                .zip(x_coarse.iter())
                .map(|(a, b)| a - b)
                .collect();

            // Prolongate correction back to fine level
            let correction_fine = self.prolongate(&correction, n);

            // Apply correction
            for i in 0..n {
                x[i] += correction_fine[i];
            }
        }

        // Post-smoothing
        self.smooth(&x, self.options.n_smooth, self.options.smooth_step, nfev)
    }

    /// Perform one W-cycle (two coarse-level visits)
    fn w_cycle(&self, x: Vec<f64>, level: usize, nfev: &mut usize) -> Vec<f64> {
        let n = x.len();
        let mut x = self.smooth(&x, self.options.n_smooth, self.options.smooth_step, nfev);

        if level < self.n_levels - 1 && n > 1 {
            let coarse_n = (n as f64 / self.options.coarsening_factor).ceil() as usize;
            let coarse_n = coarse_n.max(1);
            let x_coarse0 = self.restrict(&x, coarse_n);

            // First coarse-level solve
            let x_coarse1 = self.w_cycle(x_coarse0.clone(), level + 1, nfev);

            // Second coarse-level solve (W-cycle difference from V-cycle)
            let x_coarse2 = self.w_cycle(x_coarse1.clone(), level + 1, nfev);

            let correction: Vec<f64> = x_coarse2
                .iter()
                .zip(x_coarse0.iter())
                .map(|(a, b)| a - b)
                .collect();
            let correction_fine = self.prolongate(&correction, n);
            for i in 0..n {
                x[i] += correction_fine[i];
            }
        }

        self.smooth(&x, self.options.n_smooth, self.options.smooth_step, nfev)
    }

    /// Run the multigrid optimizer
    pub fn minimize(&self, x0: &[f64]) -> OptimizeResult<MultilevelResult> {
        if x0.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "Initial point must be non-empty".to_string(),
            ));
        }

        let mut x = x0.to_vec();
        let mut n_evals = 0usize;
        let mut n_iter = 0usize;
        let mut f_prev = self.eval(&x);
        n_evals += 1;

        for _cycle in 0..self.options.n_cycles {
            n_iter += 1;
            x = match self.options.cycle_type {
                CycleType::V => self.v_cycle(x, 0, &mut n_evals),
                CycleType::W => self.w_cycle(x, 0, &mut n_evals),
            };
            let f_new = self.eval(&x);
            n_evals += 1;
            let delta = (f_new - f_prev).abs();
            if delta < self.options.tol * (1.0 + f_prev.abs()) {
                f_prev = f_new;
                break;
            }
            f_prev = f_new;
        }

        Ok(MultilevelResult {
            x,
            fun: f_prev,
            n_hifi_evals: n_evals,
            n_lofi_evals: 0,
            n_iter,
            success: true,
            message: format!(
                "Multigrid ({:?}-cycle) optimization completed",
                self.options.cycle_type
            ),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi.powi(2)).sum()
    }

    fn quadratic_shifted(x: &[f64]) -> f64 {
        x.iter().map(|xi| (xi - 0.2).powi(2)).sum()
    }

    #[test]
    fn test_multilevel_optimizer_basic() {
        let levels = vec![
            FidelityLevel::new(quadratic_shifted as fn(&[f64]) -> f64, 1.0),
            FidelityLevel::new(quadratic as fn(&[f64]) -> f64, 10.0),
        ];
        let mut optimizer = MultilevelOptimizer::new(
            levels,
            vec![1.0, 1.0],
            MultilevelOptions {
                max_iter_per_level: 200,
                tol: 1e-5,
                ..Default::default()
            },
        );
        let result = optimizer.minimize().expect("failed to create result");
        assert!(result.fun < 0.1, "Expected fun < 0.1, got {}", result.fun);
    }

    #[test]
    fn test_variable_fidelity_basic() {
        let vf = VariableFidelity::new(
            quadratic_shifted,
            quadratic,
            1.0,
            10.0,
            VariableFidelityOptions {
                total_budget: 200.0,
                max_iter: 100,
                tol: 1e-5,
                ..Default::default()
            },
        );
        let result = vf.minimize(&[1.5, 1.5]).expect("failed to create result");
        assert!(result.fun < 0.5, "Expected fun < 0.5, got {}", result.fun);
    }

    #[test]
    fn test_trust_hierarchy_basic() {
        let levels: Vec<fn(&[f64]) -> f64> = vec![quadratic_shifted, quadratic];
        let th = TrustHierarchy::new(
            levels,
            TrustHierarchyOptions {
                initial_radii: vec![2.0, 1.0],
                max_iter: 200,
                tol: 1e-5,
                ..Default::default()
            },
        );
        let result = th.minimize(&[1.0, 1.0]).expect("failed to create result");
        assert!(result.fun < 0.1, "Expected fun < 0.1, got {}", result.fun);
    }

    #[test]
    fn test_multigrid_v_cycle() {
        let optimizer = MultigridOptimizer::new(
            quadratic,
            2,
            2,
            MultigridOptions {
                n_cycles: 30,
                n_smooth: 5,
                cycle_type: CycleType::V,
                tol: 1e-5,
                smooth_step: 0.1,
                coarsening_factor: 2.0,
            },
        );
        let result = optimizer
            .minimize(&[1.0, 1.0])
            .expect("failed to create result");
        assert!(result.fun < 0.1, "Expected fun < 0.1, got {}", result.fun);
    }

    #[test]
    fn test_multigrid_w_cycle() {
        let optimizer = MultigridOptimizer::new(
            quadratic,
            2,
            2,
            MultigridOptions {
                n_cycles: 30,
                n_smooth: 5,
                cycle_type: CycleType::W,
                tol: 1e-5,
                smooth_step: 0.1,
                coarsening_factor: 2.0,
            },
        );
        let result = optimizer
            .minimize(&[1.0, 1.0])
            .expect("failed to create result");
        assert!(result.fun < 0.1, "Expected fun < 0.1, got {}", result.fun);
    }

    #[test]
    fn test_mfrbf_basic() {
        let mut mfrbf = MfRbf::new(
            quadratic_shifted,
            quadratic,
            MfRbfOptions {
                n_initial_samples: 3,
                hifi_budget: 10,
                max_iter: 50,
                tol: 1e-4,
                length_scale: 1.0,
            },
        );
        let result = mfrbf
            .minimize(&[1.0, 1.0])
            .expect("failed to create result");
        assert!(result.fun < 1.0, "Expected fun < 1.0, got {}", result.fun);
    }
}
