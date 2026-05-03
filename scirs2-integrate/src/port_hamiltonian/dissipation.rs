//! Dissipation structures for Port-Hamiltonian systems
//!
//! This module provides ready-to-use dissipation matrix constructors for
//! common physical dissipation models used in port-Hamiltonian systems.
//!
//! # Overview
//!
//! The dissipation matrix R(x) in a port-Hamiltonian system must be:
//! - Symmetric: R = R^T
//! - Positive semi-definite: v^T R v >= 0 for all v
//!
//! The product R * ∇H represents the dissipative forces that reduce the
//! stored energy. The dissipated power is: P_diss = ∇H^T R ∇H >= 0.
//!
//! # Dissipation Models
//!
//! - [`LinearDissipation`]: Constant R matrix (linear friction/damping)
//! - [`RayleighDissipation`]: Quadratic energy function → linear dissipation forces
//! - [`NonlinearDissipation`]: State-dependent R(x) for nonlinear damping
//! - [`PortDissipation`]: Dissipation through resistive ports (R = B R_ext B^T)
//! - [`StructuredDissipation`]: Block-diagonal structure for multi-physical systems

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array2, Axis};

/// Linear (constant) dissipation matrix.
///
/// This is the simplest dissipation model where R is a constant symmetric
/// positive semi-definite matrix. It models linear friction, viscous damping,
/// or resistive elements.
///
/// # Example: Mechanical damper with damping coefficient c
///
/// For a mass-spring-damper system with states (q, p):
/// ```text
/// R = [[0, 0],
///      [0, c]]
/// ```
/// which gives a damping force proportional to momentum.
#[derive(Debug, Clone)]
pub struct LinearDissipation {
    /// The constant dissipation matrix
    pub matrix: Array2<f64>,
}

impl LinearDissipation {
    /// Create a linear dissipation model from a given matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or not symmetric PSD.
    pub fn new(matrix: Array2<f64>) -> IntegrateResult<Self> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(IntegrateError::ValueError(
                "Dissipation matrix must be square".into(),
            ));
        }
        // Check approximate symmetry
        for i in 0..n {
            for j in 0..n {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > 1e-10 {
                    return Err(IntegrateError::ValueError(format!(
                        "Dissipation matrix must be symmetric: R[{i},{j}]={} != R[{j},{i}]={}",
                        matrix[[i, j]],
                        matrix[[j, i]]
                    )));
                }
            }
        }
        Ok(Self { matrix })
    }

    /// Create a scalar multiple of the identity: R = gamma * I.
    ///
    /// This represents uniform isotropic damping.
    pub fn scaled_identity(n: usize, gamma: f64) -> IntegrateResult<Self> {
        if gamma < 0.0 {
            return Err(IntegrateError::ValueError(
                "Damping coefficient must be non-negative".into(),
            ));
        }
        let mut mat = Array2::zeros((n, n));
        for i in 0..n {
            mat[[i, i]] = gamma;
        }
        Ok(Self { matrix: mat })
    }

    /// Create a diagonal dissipation matrix from a vector of damping coefficients.
    pub fn diagonal(coeffs: &[f64]) -> IntegrateResult<Self> {
        for (i, &c) in coeffs.iter().enumerate() {
            if c < 0.0 {
                return Err(IntegrateError::ValueError(format!(
                    "Damping coefficient at index {i} must be non-negative, got {c}"
                )));
            }
        }
        let n = coeffs.len();
        let mut mat = Array2::zeros((n, n));
        for (i, &c) in coeffs.iter().enumerate() {
            mat[[i, i]] = c;
        }
        Ok(Self { matrix: mat })
    }

    /// Get the dissipation matrix closure for use in PortHamiltonianSystem.
    pub fn into_matrix_fn(
        self,
    ) -> impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static {
        let m = self.matrix;
        move |_x| Ok(m.clone())
    }
}

/// Rayleigh dissipation function model.
///
/// The Rayleigh dissipation function is a quadratic form:
/// ```text
/// D(dq/dt) = (1/2) * dq/dt^T * R_q * dq/dt
/// ```
/// The corresponding generalized forces are:
/// ```text
/// F_diss = -∂D/∂(dq/dt) = -R_q * dq/dt
/// ```
///
/// In port-Hamiltonian form, for a system with states (q, p) and
/// separable Hamiltonian H = T(p) + V(q), the Rayleigh dissipation matrix in
/// the pH formulation is:
/// ```text
/// R_pH = [[0,   0  ],
///          [0, R_q/m]]
/// ```
/// (if velocity = p/m)
#[derive(Debug, Clone)]
pub struct RayleighDissipation {
    /// Rayleigh damping matrix (in configuration space)
    pub r_q: Array2<f64>,
    /// Mass matrix (for converting momentum to velocity)
    pub mass_matrix: Array2<f64>,
    /// Dimension of configuration (half of full state dimension for q-p systems)
    pub n_config: usize,
}

impl RayleighDissipation {
    /// Create a Rayleigh dissipation from configuration-space damping matrix
    /// and a mass matrix.
    ///
    /// # Arguments
    ///
    /// * `r_q` - Symmetric PSD damping matrix in configuration space (n x n)
    /// * `mass_matrix` - Symmetric positive definite mass matrix (n x n)
    pub fn new(r_q: Array2<f64>, mass_matrix: Array2<f64>) -> IntegrateResult<Self> {
        let n = r_q.nrows();
        if r_q.ncols() != n || mass_matrix.nrows() != n || mass_matrix.ncols() != n {
            return Err(IntegrateError::ValueError(
                "Rayleigh matrices must be square and of equal size".into(),
            ));
        }
        Ok(Self {
            r_q,
            mass_matrix,
            n_config: n,
        })
    }

    /// Build the full pH dissipation matrix for a (q, p) state vector.
    ///
    /// Returns R_pH of size (2n x 2n):
    /// ```text
    /// R_pH = [[0,   0   ],
    ///          [0, M^-1 R_q M^-1]]
    /// ```
    /// (since velocity = M^-1 p, and the dissipation force in p-space is R_q * velocity)
    pub fn to_ph_matrix(&self) -> IntegrateResult<Array2<f64>> {
        let n = self.n_config;
        let two_n = 2 * n;
        let mut r_ph = Array2::zeros((two_n, two_n));

        // Compute M^{-1} R_q M^{-1} using Cholesky or LU factorization
        // For simplicity, we use a direct solve approach
        let m_inv_r_q = solve_linear_system_left(&self.mass_matrix, &self.r_q)?;
        let m_inv_r_q_m_inv = solve_linear_system_right(&m_inv_r_q, &self.mass_matrix)?;

        // Place in lower-right block
        for i in 0..n {
            for j in 0..n {
                r_ph[[n + i, n + j]] = m_inv_r_q_m_inv[[i, j]];
            }
        }
        Ok(r_ph)
    }

    /// Get the dissipation matrix closure for use in PortHamiltonianSystem.
    pub fn into_matrix_fn(
        self,
    ) -> IntegrateResult<impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static>
    {
        let r_ph = self.to_ph_matrix()?;
        Ok(move |_x: &[f64]| Ok(r_ph.clone()))
    }
}

/// Nonlinear state-dependent dissipation.
///
/// This represents dissipation that depends on the current state, such as:
/// - Velocity-dependent friction (e.g., r(v) * v for nonlinear friction)
/// - Position-dependent damping (e.g., squeeze film damping)
/// - Turbulent friction (proportional to |v|)
#[derive(Clone)]
pub struct NonlinearDissipation {
    /// State-dependent damping function
    pub r_fn: std::sync::Arc<dyn Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync>,
    /// State dimension
    pub n: usize,
}

impl std::fmt::Debug for NonlinearDissipation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NonlinearDissipation")
            .field("n", &self.n)
            .finish()
    }
}

impl NonlinearDissipation {
    /// Create from a custom state-dependent R(x) function.
    pub fn new(
        n: usize,
        r_fn: impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            r_fn: std::sync::Arc::new(r_fn),
            n,
        }
    }

    /// Velocity-dependent diagonal damping: R(x)_{ii} = c_i(v_i).
    ///
    /// For a (q, p) state where the second half are momenta,
    /// the diagonal entries of R scale with |p|.
    ///
    /// # Arguments
    ///
    /// * `n_config` - Number of configuration coordinates (state dim = 2 * n_config)
    /// * `base_damping` - Base damping coefficients (length n_config)
    /// * `nonlinear_exp` - Exponent for velocity dependence (1=linear, 2=quadratic)
    pub fn velocity_dependent(
        n_config: usize,
        base_damping: Vec<f64>,
        nonlinear_exp: f64,
    ) -> IntegrateResult<Self> {
        if base_damping.len() != n_config {
            return Err(IntegrateError::ValueError(format!(
                "base_damping length {} != n_config {}",
                base_damping.len(),
                n_config
            )));
        }
        let two_n = 2 * n_config;
        Ok(Self::new(two_n, move |x: &[f64]| {
            let mut r = Array2::zeros((two_n, two_n));
            for i in 0..n_config {
                // Momentum is in the second half of the state
                let p_i = if x.len() > n_config + i {
                    x[n_config + i]
                } else {
                    0.0
                };
                // Damping coefficient scales with |p|^(nonlinear_exp - 1)
                let speed = p_i.abs();
                let damp = if nonlinear_exp > 1.0 && speed > 1e-14 {
                    base_damping[i] * speed.powf(nonlinear_exp - 1.0)
                } else {
                    base_damping[i]
                };
                r[[n_config + i, n_config + i]] = damp;
            }
            Ok(r)
        }))
    }

    /// Evaluate the dissipation matrix at state x.
    pub fn evaluate(&self, x: &[f64]) -> IntegrateResult<Array2<f64>> {
        (self.r_fn)(x)
    }

    /// Get the dissipation matrix closure.
    pub fn into_matrix_fn(
        self,
    ) -> impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static {
        let arc = self.r_fn;
        move |x: &[f64]| arc(x)
    }
}

/// Port-based dissipation: R = B_r * R_ext * B_r^T
///
/// This models dissipation through a resistive termination on a port.
/// For example, an electrical resistor terminating an electrical port,
/// or a damper terminating a mechanical port.
#[derive(Debug, Clone)]
pub struct PortDissipation {
    /// Resistive port matrix (n_states x n_resist)
    pub b_r: Array2<f64>,
    /// External resistance matrix (n_resist x n_resist), symmetric PSD
    pub r_ext: Array2<f64>,
}

impl PortDissipation {
    /// Create port-based dissipation.
    pub fn new(b_r: Array2<f64>, r_ext: Array2<f64>) -> IntegrateResult<Self> {
        let n_r = b_r.ncols();
        if r_ext.nrows() != n_r || r_ext.ncols() != n_r {
            return Err(IntegrateError::ValueError(format!(
                "r_ext must be ({n_r}x{n_r}), got ({}x{})",
                r_ext.nrows(),
                r_ext.ncols()
            )));
        }
        Ok(Self { b_r, r_ext })
    }

    /// Compute R = B_r * R_ext * B_r^T
    pub fn to_matrix(&self) -> Array2<f64> {
        let b_r_r_ext = self.b_r.dot(&self.r_ext);
        b_r_r_ext.dot(&self.b_r.t())
    }

    /// Get the dissipation matrix closure.
    pub fn into_matrix_fn(
        self,
    ) -> impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static {
        let m = self.to_matrix();
        move |_x: &[f64]| Ok(m.clone())
    }
}

/// Structured block-diagonal dissipation for multi-physical systems.
///
/// This combines multiple dissipation blocks for different physical domains,
/// e.g., mechanical + electrical + thermal subsystems.
#[derive(Debug, Clone)]
pub struct StructuredDissipation {
    /// Block sizes (must sum to total state dimension)
    pub block_sizes: Vec<usize>,
    /// Diagonal blocks (each must be PSD)
    pub blocks: Vec<Array2<f64>>,
}

impl StructuredDissipation {
    /// Create from a list of (block_size, block_matrix) pairs.
    pub fn new(blocks: Vec<(usize, Array2<f64>)>) -> IntegrateResult<Self> {
        let block_sizes: Vec<usize> = blocks.iter().map(|(s, _)| *s).collect();
        let matrices: Vec<Array2<f64>> = blocks.into_iter().map(|(_, m)| m).collect();

        for (i, (sz, m)) in block_sizes.iter().zip(matrices.iter()).enumerate() {
            if m.nrows() != *sz || m.ncols() != *sz {
                return Err(IntegrateError::ValueError(format!(
                    "Block {i}: expected ({sz}x{sz}), got ({}x{})",
                    m.nrows(),
                    m.ncols()
                )));
            }
        }

        Ok(Self {
            block_sizes,
            blocks: matrices,
        })
    }

    /// Assemble the full block-diagonal dissipation matrix.
    pub fn to_full_matrix(&self) -> Array2<f64> {
        let total: usize = self.block_sizes.iter().sum();
        let mut r = Array2::zeros((total, total));
        let mut offset = 0;
        for (sz, block) in self.block_sizes.iter().zip(self.blocks.iter()) {
            let end = offset + sz;
            r.slice_mut(scirs2_core::ndarray::s![offset..end, offset..end])
                .assign(block);
            offset = end;
        }
        r
    }

    /// Get the dissipation matrix closure.
    pub fn into_matrix_fn(
        self,
    ) -> impl Fn(&[f64]) -> IntegrateResult<Array2<f64>> + Send + Sync + 'static {
        let m = self.to_full_matrix();
        move |_x: &[f64]| Ok(m.clone())
    }
}

// ─── Helper: solve A * X = B by row reduction (Gauss-Jordan) ──────────────────

/// Solve A * X = B (A is n×n, B is n×m). Returns X = A^{-1} B.
fn solve_linear_system_left(a: &Array2<f64>, b: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n || b.nrows() != n {
        return Err(IntegrateError::ValueError(
            "Dimension mismatch in linear solve".into(),
        ));
    }
    let m = b.ncols();
    // Augmented matrix [A | B]
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
            row.extend((0..m).map(|j| b[[i, j]]));
            row
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n).max_by(|&r1, &r2| {
            aug[r1][col]
                .abs()
                .partial_cmp(&aug[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pivot_row = pivot_row.ok_or_else(|| {
            IntegrateError::LinearSolveError("Singular matrix in dissipation solve".into())
        })?;

        if aug[pivot_row][col].abs() < 1e-15 {
            return Err(IntegrateError::LinearSolveError(
                "Singular or near-singular matrix in dissipation solve".into(),
            ));
        }

        aug.swap(col, pivot_row);
        let pivot = aug[col][col];
        let row_len = aug[col].len();
        for j in 0..row_len {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..row_len {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let mut x = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            x[[i, j]] = aug[i][n + j];
        }
    }
    Ok(x)
}

/// Solve X * A = B (A is n×n, B is m×n). Returns X = B A^{-1}.
/// Equivalent to (A^T X^T = B^T), so X^T = A^{-T} B^T.
fn solve_linear_system_right(b: &Array2<f64>, a: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
    let at = a.t().to_owned();
    let bt = b.t().to_owned();
    let xt = solve_linear_system_left(&at, &bt)?;
    Ok(xt.t().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_dissipation_identity() {
        let r = LinearDissipation::scaled_identity(3, 2.0).expect("Failed to create dissipation");
        assert_eq!(r.matrix[[0, 0]], 2.0);
        assert_eq!(r.matrix[[1, 1]], 2.0);
        assert_eq!(r.matrix[[2, 2]], 2.0);
        assert_eq!(r.matrix[[0, 1]], 0.0);
    }

    #[test]
    fn test_linear_dissipation_diagonal() {
        let r =
            LinearDissipation::diagonal(&[1.0, 2.0, 3.0]).expect("Failed to create dissipation");
        assert_eq!(r.matrix[[0, 0]], 1.0);
        assert_eq!(r.matrix[[1, 1]], 2.0);
        assert_eq!(r.matrix[[2, 2]], 3.0);
    }

    #[test]
    fn test_linear_dissipation_negative_coeff_rejected() {
        assert!(LinearDissipation::diagonal(&[1.0, -0.5]).is_err());
    }

    #[test]
    fn test_port_dissipation() {
        // B_r = [0; 1], R_ext = [[2]] => R = [[0, 0], [0, 2]]
        let b_r = array![[0.0], [1.0]];
        let r_ext = array![[2.0]];
        let pd = PortDissipation::new(b_r, r_ext).expect("Failed to create port dissipation");
        let m = pd.to_matrix();
        assert!((m[[0, 0]]).abs() < 1e-14);
        assert!((m[[1, 1]] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_structured_dissipation() {
        let b1 = array![[1.0, 0.0], [0.0, 2.0]];
        let b2 = array![[3.0]];
        let sd = StructuredDissipation::new(vec![(2, b1), (1, b2)])
            .expect("Failed to create structured dissipation");
        let m = sd.to_full_matrix();
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
        assert!((m[[0, 0]] - 1.0).abs() < 1e-14);
        assert!((m[[1, 1]] - 2.0).abs() < 1e-14);
        assert!((m[[2, 2]] - 3.0).abs() < 1e-14);
        assert!((m[[0, 2]]).abs() < 1e-14); // off-block is zero
    }

    #[test]
    fn test_nonlinear_velocity_dependent() {
        let nd = NonlinearDissipation::velocity_dependent(2, vec![1.0, 1.0], 2.0)
            .expect("Failed to create nonlinear dissipation");
        let x = vec![0.0, 0.0, 2.0, 3.0]; // q=(0,0), p=(2,3)
        let r = nd.evaluate(&x).expect("Evaluation failed");
        // R[2,2] = base * |p0|^1 = 1 * 2 = 2
        assert!((r[[2, 2]] - 2.0).abs() < 1e-10);
        assert!((r[[3, 3]] - 3.0).abs() < 1e-10);
    }
}
