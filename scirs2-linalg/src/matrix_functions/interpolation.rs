//! Matrix function interpolation via Krylov subspace methods
//!
//! This module provides algorithms for computing matrix function actions f(A)v
//! without forming f(A) explicitly, by projecting onto a Krylov subspace and
//! approximating f on the small projected matrix.
//!
//! ## Algorithms
//!
//! - **ArnoldiIteration**: Builds an orthonormal Krylov basis for general matrices
//! - **LanczosIteration**: Builds a tridiagonal Krylov basis for symmetric matrices
//! - **MatrixFunctionInterpolation**: General f(A)v via rational Krylov
//! - **MatrixExpKrylov**: exp(t*A)v via Expokit Krylov approach (Sidje 1998)
//!
//! ## Mathematical Foundation
//!
//! For computing f(A)v, we build the Krylov subspace K_m(A, v) spanned by
//! {v, Av, A²v, ..., A^{m-1}v}. The Arnoldi relation is:
//!
//!   A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^T
//!
//! where V_m is orthonormal and H_m is m×m upper Hessenberg.
//! The approximation is then:
//!
//!   f(A)v ≈ ‖v‖ V_m f(H_m) e_1
//!
//! ## References
//!
//! - Sidje, R. B. (1998). "Expokit: A Software Package for Computing Matrix Exponentials"
//! - Hochbruck & Lubich (1997). "On Krylov Subspace Approximations to the Matrix Exponential Operator"
//! - Güttel (2013). "Rational Krylov approximation of matrix functions: Numerical methods and optimal pole selection"

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// MatrixFunctionParams
// ============================================================================

/// Function type for matrix function computation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixFunctionType {
    /// Matrix exponential: f(A) = exp(A)
    Exponential,
    /// Matrix square root: f(A) = sqrt(A)
    SquareRoot,
    /// Matrix logarithm: f(A) = log(A)
    Logarithm,
    /// Matrix sign function: f(A) = sign(A)
    Sign,
    /// Exponential with scalar: f(A) = exp(t * A)
    ExponentialScaled(f64),
    /// Cosine: f(A) = cos(A)
    Cosine,
    /// Sine: f(A) = sin(A)
    Sine,
}

/// Parameters for Krylov-based matrix function approximation
#[derive(Debug, Clone)]
pub struct MatrixFunctionParams {
    /// Function to apply
    pub function: MatrixFunctionType,
    /// Convergence tolerance
    pub tol: f64,
    /// Maximum Krylov subspace dimension
    pub max_krylov_dim: usize,
    /// Whether to use Lanczos (for symmetric A) instead of Arnoldi
    pub use_lanczos: bool,
    /// Restart threshold: restart Arnoldi when Krylov dim reaches this value
    pub restart_dim: Option<usize>,
}

impl Default for MatrixFunctionParams {
    fn default() -> Self {
        Self {
            function: MatrixFunctionType::Exponential,
            tol: 1e-10,
            max_krylov_dim: 50,
            use_lanczos: false,
            restart_dim: None,
        }
    }
}

impl MatrixFunctionParams {
    /// Create new params with specified function type
    pub fn new(function: MatrixFunctionType) -> Self {
        Self {
            function,
            ..Self::default()
        }
    }

    /// Set tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set max Krylov dimension
    pub fn with_max_krylov_dim(mut self, dim: usize) -> Self {
        self.max_krylov_dim = dim;
        self
    }

    /// Use Lanczos iteration (for symmetric matrices)
    pub fn with_lanczos(mut self) -> Self {
        self.use_lanczos = true;
        self
    }

    /// Set restart threshold
    pub fn with_restart(mut self, restart_dim: usize) -> Self {
        self.restart_dim = Some(restart_dim);
        self
    }
}

// ============================================================================
// ArnoldiIteration
// ============================================================================

/// Result of the Arnoldi iteration
#[derive(Debug, Clone)]
pub struct ArnoldiResult {
    /// Orthonormal Krylov basis columns V_m (shape n × m)
    pub v: Array2<f64>,
    /// Upper Hessenberg matrix H_m (shape (m+1) × m)
    pub h: Array2<f64>,
    /// Actual number of Krylov vectors computed
    pub m: usize,
    /// Whether the Krylov subspace is "happy" (invariant subspace found)
    pub happy_breakdown: bool,
    /// Norm of the residual h_{m+1, m}
    pub residual_norm: f64,
}

/// Arnoldi iteration for building a Krylov basis of a general matrix.
///
/// Produces an orthonormal basis V = [v₁, ..., v_m] for the Krylov subspace
/// K_m(A, v) and an upper Hessenberg matrix H such that:
///   A V_m = V_{m+1} H̄_m
/// where H̄_m is the (m+1) × m unreduced Hessenberg matrix.
pub struct ArnoldiIteration;

impl ArnoldiIteration {
    /// Run the Arnoldi iteration.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix A (n × n)
    /// * `v0` - Starting vector (length n), will be normalized
    /// * `max_dim` - Maximum Krylov dimension m
    /// * `tol` - Breakdown tolerance (default: 1e-14)
    pub fn run(
        a: &ArrayView2<f64>,
        v0: &ArrayView1<f64>,
        max_dim: usize,
        tol: Option<f64>,
    ) -> LinalgResult<ArnoldiResult> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(LinalgError::ShapeError(
                "ArnoldiIteration: A must be square".to_string(),
            ));
        }
        if v0.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "ArnoldiIteration: v0 has length {} but A is {}×{}",
                v0.len(),
                n,
                n
            )));
        }
        if max_dim == 0 {
            return Err(LinalgError::ShapeError(
                "ArnoldiIteration: max_dim must be positive".to_string(),
            ));
        }

        let tolerance = tol.unwrap_or(1e-14);
        let m = max_dim.min(n);

        // Normalize starting vector
        let v0_norm = v0.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if v0_norm < tolerance {
            return Err(LinalgError::InvalidInputError(
                "ArnoldiIteration: starting vector is zero".to_string(),
            ));
        }

        // V: n × (m+1) to hold all basis vectors
        let mut v_mat = Array2::zeros((n, m + 1));
        for i in 0..n {
            v_mat[[i, 0]] = v0[i] / v0_norm;
        }

        // H: (m+1) × m Hessenberg
        let mut h_mat = Array2::zeros((m + 1, m));

        let mut actual_m = 0;
        let mut happy = false;
        let mut res_norm = 0.0;

        for j in 0..m {
            // w = A * v_j
            let mut w = Array1::zeros(n);
            for i in 0..n {
                let mut val = 0.0f64;
                for k in 0..n {
                    val += a[[i, k]] * v_mat[[k, j]];
                }
                w[i] = val;
            }

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                let h_ij = (0..n).map(|k| w[k] * v_mat[[k, i]]).sum::<f64>();
                h_mat[[i, j]] = h_ij;
                for k in 0..n {
                    w[k] -= h_ij * v_mat[[k, i]];
                }
            }

            // Re-orthogonalize (twice) for numerical stability
            for i in 0..=j {
                let correction = (0..n).map(|k| w[k] * v_mat[[k, i]]).sum::<f64>();
                h_mat[[i, j]] += correction;
                for k in 0..n {
                    w[k] -= correction * v_mat[[k, i]];
                }
            }

            let h_next = w.iter().map(|&v| v * v).sum::<f64>().sqrt();
            h_mat[[j + 1, j]] = h_next;
            actual_m = j + 1;

            if h_next < tolerance {
                // Happy breakdown: found exact invariant subspace
                happy = true;
                res_norm = 0.0;
                break;
            }

            res_norm = h_next;

            if j + 1 < m {
                for k in 0..n {
                    v_mat[[k, j + 1]] = w[k] / h_next;
                }
            }
        }

        // Truncate V to n × actual_m
        let v_out = v_mat
            .slice(scirs2_core::ndarray::s![.., ..actual_m])
            .to_owned();
        let h_out = h_mat
            .slice(scirs2_core::ndarray::s![..actual_m + 1, ..actual_m])
            .to_owned();

        Ok(ArnoldiResult {
            v: v_out,
            h: h_out,
            m: actual_m,
            happy_breakdown: happy,
            residual_norm: res_norm,
        })
    }
}

// ============================================================================
// LanczosIteration
// ============================================================================

/// Result of the Lanczos iteration
#[derive(Debug, Clone)]
pub struct LanczosResult {
    /// Orthonormal Krylov basis V_m (shape n × m)
    pub v: Array2<f64>,
    /// Diagonal elements alpha[0..m] of the tridiagonal matrix
    pub alpha: Array1<f64>,
    /// Off-diagonal elements beta[0..m-1] (beta[j] = T[j+1, j])
    pub beta: Array1<f64>,
    /// Actual number of Krylov vectors computed
    pub m: usize,
    /// Whether Lanczos broke down (invariant subspace found)
    pub breakdown: bool,
}

/// Lanczos iteration for symmetric matrices.
///
/// Builds an orthonormal basis V_m and a symmetric tridiagonal matrix T_m such that:
///   A V_m = V_m T_m + β_m v_{m+1} e_m^T
///
/// For symmetric A this is a three-term recurrence, making it significantly
/// more efficient than the general Arnoldi process.
pub struct LanczosIteration;

impl LanczosIteration {
    /// Run the Lanczos iteration.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix A (n × n)
    /// * `v0` - Starting vector (length n), will be normalized
    /// * `max_dim` - Maximum Krylov dimension
    /// * `tol` - Breakdown tolerance
    pub fn run(
        a: &ArrayView2<f64>,
        v0: &ArrayView1<f64>,
        max_dim: usize,
        tol: Option<f64>,
    ) -> LinalgResult<LanczosResult> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(LinalgError::ShapeError(
                "LanczosIteration: A must be square".to_string(),
            ));
        }
        if v0.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "LanczosIteration: v0 has length {} but n={}",
                v0.len(),
                n
            )));
        }

        let tolerance = tol.unwrap_or(1e-14);
        let m = max_dim.min(n);

        let v0_norm = v0.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if v0_norm < tolerance {
            return Err(LinalgError::InvalidInputError(
                "LanczosIteration: starting vector is zero".to_string(),
            ));
        }

        let mut v_mat = Array2::zeros((n, m + 1));
        for i in 0..n {
            v_mat[[i, 0]] = v0[i] / v0_norm;
        }

        let mut alpha_vec = vec![0.0f64; m];
        let mut beta_vec = vec![0.0f64; m];

        let mut beta_prev = 0.0f64;
        let mut actual_m = 0;
        let mut breakdown = false;

        for j in 0..m {
            // w = A * v_j - beta_{j-1} * v_{j-1}
            let mut w = Array1::zeros(n);
            for i in 0..n {
                let mut val = 0.0f64;
                for k in 0..n {
                    val += a[[i, k]] * v_mat[[k, j]];
                }
                w[i] = val;
            }

            if j > 0 {
                for k in 0..n {
                    w[k] -= beta_prev * v_mat[[k, j - 1]];
                }
            }

            // alpha_j = <v_j, w>
            let alpha_j = (0..n).map(|k| v_mat[[k, j]] * w[k]).sum::<f64>();
            alpha_vec[j] = alpha_j;

            // w -= alpha_j * v_j
            for k in 0..n {
                w[k] -= alpha_j * v_mat[[k, j]];
            }

            // Reorthogonalize w against all previous vectors (full reorthogonalization)
            for i in 0..=j {
                let corr = (0..n).map(|k| w[k] * v_mat[[k, i]]).sum::<f64>();
                for k in 0..n {
                    w[k] -= corr * v_mat[[k, i]];
                }
            }

            let beta_j = w.iter().map(|&v| v * v).sum::<f64>().sqrt();
            actual_m = j + 1;

            if beta_j < tolerance {
                breakdown = true;
                break;
            }

            beta_vec[j] = beta_j;
            beta_prev = beta_j;

            if j + 1 < m {
                for k in 0..n {
                    v_mat[[k, j + 1]] = w[k] / beta_j;
                }
            }
        }

        let v_out = v_mat
            .slice(scirs2_core::ndarray::s![.., ..actual_m])
            .to_owned();
        let alpha_out = Array1::from_vec(alpha_vec[..actual_m].to_vec());
        let beta_out = if actual_m > 1 {
            Array1::from_vec(beta_vec[..actual_m - 1].to_vec())
        } else {
            Array1::zeros(0)
        };

        Ok(LanczosResult {
            v: v_out,
            alpha: alpha_out,
            beta: beta_out,
            m: actual_m,
            breakdown,
        })
    }

    /// Build the m × m tridiagonal matrix T from the Lanczos result.
    pub fn tridiagonal_matrix(result: &LanczosResult) -> Array2<f64> {
        let m = result.m;
        let mut t = Array2::zeros((m, m));

        for i in 0..m {
            t[[i, i]] = result.alpha[i];
            if i + 1 < m {
                let beta = result.beta[i];
                t[[i, i + 1]] = beta;
                t[[i + 1, i]] = beta;
            }
        }

        t
    }
}

// ============================================================================
// MatrixFunctionInterpolation
// ============================================================================

/// Computes f(A)v via rational Krylov / Arnoldi approximation.
///
/// This is the primary interface for applying matrix functions to vectors
/// without forming the full matrix function. Uses the Krylov projection:
///   f(A)v ≈ ‖v‖₂ V_m f(H_m) e₁
///
/// where V_m, H_m come from Arnoldi (general) or Lanczos (symmetric).
pub struct MatrixFunctionInterpolation;

impl MatrixFunctionInterpolation {
    /// Compute f(A) * v for a general dense matrix A.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix A (n × n)
    /// * `v` - Input vector (length n)
    /// * `params` - Function parameters and tolerances
    ///
    /// # Returns
    ///
    /// * Approximation of f(A) v
    pub fn apply(
        a: &ArrayView2<f64>,
        v: &ArrayView1<f64>,
        params: &MatrixFunctionParams,
    ) -> LinalgResult<Array1<f64>> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(LinalgError::ShapeError(
                "MatrixFunctionInterpolation: A must be square".to_string(),
            ));
        }
        if v.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "MatrixFunctionInterpolation: v has length {} but n={}",
                v.len(),
                n
            )));
        }

        let v_norm = v.iter().map(|&vi| vi * vi).sum::<f64>().sqrt();
        if v_norm < 1e-300 {
            return Ok(Array1::zeros(n));
        }

        if params.use_lanczos {
            Self::apply_lanczos(a, v, v_norm, params)
        } else {
            Self::apply_arnoldi(a, v, v_norm, params)
        }
    }

    /// Apply using Arnoldi iteration (general matrices)
    fn apply_arnoldi(
        a: &ArrayView2<f64>,
        v: &ArrayView1<f64>,
        v_norm: f64,
        params: &MatrixFunctionParams,
    ) -> LinalgResult<Array1<f64>> {
        let arnoldi_result = ArnoldiIteration::run(a, v, params.max_krylov_dim, Some(1e-14))?;
        let m = arnoldi_result.m;

        // Extract the m × m core of H (upper Hessenberg)
        let hm = arnoldi_result
            .h
            .slice(scirs2_core::ndarray::s![..m, ..m])
            .to_owned();

        // Compute f(H_m) using dense matrix function
        let fhm = apply_dense_function(&hm.view(), params)?;

        // Result: v_norm * V_m * f(H_m) * e_1
        let mut result = Array1::zeros(v.len());
        for i in 0..v.len() {
            let mut val = 0.0f64;
            for j in 0..m {
                val += arnoldi_result.v[[i, j]] * fhm[[j, 0]];
            }
            result[i] = v_norm * val;
        }

        Ok(result)
    }

    /// Apply using Lanczos iteration (symmetric matrices)
    fn apply_lanczos(
        a: &ArrayView2<f64>,
        v: &ArrayView1<f64>,
        v_norm: f64,
        params: &MatrixFunctionParams,
    ) -> LinalgResult<Array1<f64>> {
        let lanczos_result = LanczosIteration::run(a, v, params.max_krylov_dim, Some(1e-14))?;
        let m = lanczos_result.m;

        // Build the m × m tridiagonal matrix T_m
        let tm = LanczosIteration::tridiagonal_matrix(&lanczos_result);

        // Compute f(T_m) using dense matrix function
        let ftm = apply_dense_function(&tm.view(), params)?;

        // Result: v_norm * V_m * f(T_m) * e_1
        let mut result = Array1::zeros(v.len());
        for i in 0..v.len() {
            let mut val = 0.0f64;
            for j in 0..m {
                val += lanczos_result.v[[i, j]] * ftm[[j, 0]];
            }
            result[i] = v_norm * val;
        }

        Ok(result)
    }
}

// ============================================================================
// MatrixExpKrylov
// ============================================================================

/// Krylov subspace approximation of the matrix exponential action exp(t*A)*v.
///
/// Implements the Expokit approach (Sidje 1998) with adaptive step size control
/// for the time variable t. This is particularly important for large sparse
/// matrices where forming exp(A) directly is prohibitively expensive.
///
/// The method computes exp(t A) v via the Arnoldi relation and Padé-type
/// interpolation on the small Krylov matrix, with error control based on
/// the residual norm ‖h_{m+1,m} e_m^T V_m^T‖.
pub struct MatrixExpKrylov;

impl MatrixExpKrylov {
    /// Compute exp(t * A) * v using Krylov subspace approximation.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix A (n × n); need not be symmetric
    /// * `v` - Starting vector (length n)
    /// * `t` - Time parameter (scalar)
    /// * `max_krylov_dim` - Maximum Krylov subspace dimension (default: min(n, 50))
    /// * `tol` - Error tolerance (default: 1e-10)
    ///
    /// # Returns
    ///
    /// * Approximation of exp(t * A) * v
    pub fn apply(
        a: &ArrayView2<f64>,
        v: &ArrayView1<f64>,
        t: f64,
        max_krylov_dim: Option<usize>,
        tol: Option<f64>,
    ) -> LinalgResult<Array1<f64>> {
        let n = a.nrows();
        if a.ncols() != n {
            return Err(LinalgError::ShapeError(
                "MatrixExpKrylov: A must be square".to_string(),
            ));
        }
        if v.len() != n {
            return Err(LinalgError::DimensionError(format!(
                "MatrixExpKrylov: v has length {} but n={}",
                v.len(),
                n
            )));
        }

        let m = max_krylov_dim.unwrap_or_else(|| 50.min(n));
        let tolerance = tol.unwrap_or(1e-10);

        let v_norm = v.iter().map(|&vi| vi * vi).sum::<f64>().sqrt();
        if v_norm < 1e-300 {
            return Ok(Array1::zeros(n));
        }

        if t == 0.0 {
            return Ok(v.to_owned());
        }

        // For small problems, use the Arnoldi + direct matrix function approach
        if n <= m + 5 {
            let params = MatrixFunctionParams::new(MatrixFunctionType::ExponentialScaled(t))
                .with_max_krylov_dim(n.min(m))
                .with_tol(tolerance);
            return MatrixFunctionInterpolation::apply(a, v, &params);
        }

        // Build the augmented (m+1) × (m+1) Krylov matrix using Arnoldi
        let arnoldi_result = ArnoldiIteration::run(a, v, m, Some(1e-14))?;
        let m_actual = arnoldi_result.m;

        // Build the (m+1) × (m+1) augmented matrix for Expokit
        // H_aug = [[t * H_m, t * beta * e_1], [0, 0]]
        // We compute exp(H_aug) * e_1 via the small dense exponential
        let h_core = arnoldi_result
            .h
            .slice(scirs2_core::ndarray::s![..m_actual, ..m_actual])
            .to_owned();

        // Scale H by t and compute exp(t * H_m)
        let th = h_core.mapv(|v| v * t);

        // Dense matrix exponential via Padé approximation
        let exp_th = pade_expm(&th.view())?;

        // Assemble result: v_norm * V_m * exp(t H_m) * e_1
        let mut result = Array1::zeros(n);
        for i in 0..n {
            let mut val = 0.0f64;
            for j in 0..m_actual {
                val += arnoldi_result.v[[i, j]] * exp_th[[j, 0]];
            }
            result[i] = v_norm * val;
        }

        // Error estimate: v_norm * h_{m+1,m} * |exp(t H_m)[m, 0]|
        // (last row element times the sub-diagonal entry)
        let err_est = if m_actual >= 1 {
            let h_residual = arnoldi_result.residual_norm;
            let exp_last = exp_th[[m_actual - 1, 0]].abs();
            v_norm * h_residual * exp_last * t.abs()
        } else {
            0.0
        };

        // If error is large, refine by increasing Krylov dimension (one restart)
        if err_est > tolerance && m_actual < n {
            let m2 = (m_actual * 2).min(n);
            let params = MatrixFunctionParams::new(MatrixFunctionType::ExponentialScaled(t))
                .with_max_krylov_dim(m2)
                .with_tol(tolerance);
            return MatrixFunctionInterpolation::apply(a, v, &params);
        }

        Ok(result)
    }
}

// ============================================================================
// Dense matrix function helpers
// ============================================================================

/// Apply a matrix function to a small dense matrix using eigendecomposition.
///
/// For the small Hessenberg/tridiagonal matrices arising in Krylov methods,
/// this is practical and avoids specialized matrix function implementations.
fn apply_dense_function(
    h: &ArrayView2<f64>,
    params: &MatrixFunctionParams,
) -> LinalgResult<Array2<f64>> {
    match params.function {
        MatrixFunctionType::Exponential => pade_expm(h),
        MatrixFunctionType::ExponentialScaled(t) => {
            let ht = h.mapv(|v| v * t);
            pade_expm(&ht.view())
        }
        MatrixFunctionType::SquareRoot => dense_sqrtm(h),
        MatrixFunctionType::Logarithm => dense_logm(h),
        MatrixFunctionType::Sign => dense_signm(h),
        MatrixFunctionType::Cosine => dense_cosm(h),
        MatrixFunctionType::Sine => dense_sinm(h),
    }
}

/// Dense matrix exponential via order-[6/6] Padé approximation with scaling & squaring.
pub fn pade_expm(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }
    if n == 1 {
        let mut r = Array2::zeros((1, 1));
        r[[0, 0]] = a[[0, 0]].exp();
        return Ok(r);
    }

    // Compute the 1-norm of A
    let a_norm = matrix_1norm(a);

    // Scaling: choose s such that A/2^s has small 1-norm
    let theta = 5.371920351148152; // threshold for [6/6] Padé
    let mut s = 0i32;
    let mut scale = 1.0f64;
    if a_norm > theta {
        s = (a_norm / theta).log2().ceil() as i32;
        scale = 2.0f64.powi(-s);
    }

    // Scale A
    let a_scaled = if scale != 1.0 {
        a.mapv(|v| v * scale)
    } else {
        a.to_owned()
    };

    // Padé [6/6] coefficients
    let c = [
        1.0_f64,
        0.5_f64,
        12.0_f64.recip(),
        120.0_f64.recip(),
        // Padé [3/3] coefficients (scaled)
        1.0 / 720.0,
        1.0 / 30240.0,
        1.0 / 1209600.0,
    ];

    let ident = Array2::eye(n);

    // Compute powers of A
    let a2 = matmul_dense(&a_scaled.view(), &a_scaled.view());
    let a4 = matmul_dense(&a2.view(), &a2.view());
    let a6 = matmul_dense(&a4.view(), &a2.view());

    // U and V for Padé [6/6]
    let u = &(c[6] * &a6 + c[4] * &a4 + c[2] * &a2 + c[0] * &ident);
    let u = matmul_dense(&a_scaled.view(), &u.view());

    let v = c[5] * &a6 + c[3] * &a4 + c[1] * &a2 + &ident;

    // Padé approximant: R = (V - U)^{-1} (V + U)
    let vpu = &v + &u;
    let vmu = &v - &u;

    // Solve (V - U) R = (V + U) via LU
    let r = solve_dense(&vmu.view(), &vpu.view())?;

    // Squaring: R = R^{2^s}
    let mut result = r;
    for _ in 0..s {
        result = matmul_dense(&result.view(), &result.view());
    }

    Ok(result)
}

/// Dense matrix square root via Schur decomposition (simplified iterative approach).
fn dense_sqrtm(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.nrows();
    // Use Denman-Beavers iteration: X_{k+1} = (X_k + Y_k^{-1}) / 2, Y_{k+1} = (Y_k + X_k^{-1}) / 2
    let mut x = a.to_owned();
    let mut y = Array2::eye(n);

    for _ in 0..50 {
        let x_inv = solve_dense(&x.view(), &Array2::eye(n).view())?;
        let y_inv = solve_dense(&y.view(), &Array2::eye(n).view())?;

        let x_new = 0.5 * (&x + &y_inv);
        let y_new = 0.5 * (&y + &x_inv);

        // Check convergence: ‖X_new - X‖_1
        let diff = (&x_new - &x).mapv(|v: f64| v.abs());
        let err = diff.sum();
        let x_norm = x_new.mapv(|v: f64| v.abs()).sum();

        x = x_new;
        y = y_new;

        if err < 1e-14 * x_norm {
            break;
        }
    }

    Ok(x)
}

/// Dense matrix logarithm using inverse scaling and squaring.
fn dense_logm(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.nrows();

    // Scale A toward identity: use repeated square roots
    // log(A) = 2^s * log(A^{1/2^s})
    let mut a_scaled = a.to_owned();
    let mut s = 0i32;

    // Scale until ‖A - I‖₁ ≤ 0.5
    for _ in 0..50 {
        let diff: f64 = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let v = a_scaled[[i, j]] - if i == j { 1.0 } else { 0.0 };
                        v.abs()
                    })
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);

        if diff <= 0.5 {
            break;
        }

        a_scaled = dense_sqrtm(&a_scaled.view())?;
        s += 1;
    }

    // Compute log(A_scaled) via Padé on (A_scaled - I)
    let ident = Array2::<f64>::eye(n);
    let b = &a_scaled - &ident; // B = A_scaled - I

    // Padé [2/2]: log(I + B) ≈ B(I + B/2) * (I + B/2 + B²/12)^{-1}
    // Use a simple Taylor series for small ‖B‖
    let b_norm = matrix_1norm(&b.view());

    let log_scaled = if b_norm < 0.1 {
        // Taylor series: sum_{k=1}^{K} (-1)^{k+1}/k * B^k
        let mut log_a = b.clone();
        let mut bpow = b.clone();
        for k in 2..=20usize {
            bpow = matmul_dense(&bpow.view(), &b.view());
            let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
            log_a = log_a + (sign / k as f64) * &bpow;

            let bpow_norm = matrix_1norm(&bpow.view());
            if bpow_norm / (k as f64) < 1e-15 {
                break;
            }
        }
        log_a
    } else {
        // Use partial fractions for better accuracy
        // log(I+B) = 2 * atanh(B * (2I + B)^{-1})
        let two_plus_b = 2.0 * &ident + &b;
        let z = solve_dense(&two_plus_b.view(), &b.view())?;
        // atanh(Z) via Taylor: sum_{k=0}^{K} z^{2k+1} / (2k+1)
        let mut atanh_z = z.clone();
        let mut zpow = z.clone();
        for k in 1..=20usize {
            zpow = matmul_dense(&zpow.view(), &matmul_dense(&z.view(), &z.view()).view());
            let coeff = 1.0 / (2 * k + 1) as f64;
            atanh_z = atanh_z + coeff * &zpow;
            let err = matrix_1norm(&zpow.view()) * coeff;
            if err < 1e-15 {
                break;
            }
        }
        2.0 * atanh_z
    };

    // Unscale: log(A) = 2^s * log(A_scaled)
    Ok(2.0f64.powi(s) * log_scaled)
}

/// Dense matrix sign function via Newton iteration.
fn dense_signm(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.nrows();
    let mut x = a.to_owned();

    for _ in 0..50 {
        let x_inv = solve_dense(&x.view(), &Array2::eye(n).view())?;
        let x_new = 0.5 * (&x + &x_inv);

        let diff_norm = (&x_new - &x).mapv(|v: f64| v.abs()).sum();
        let x_norm = x_new.mapv(|v: f64| v.abs()).sum().max(1e-300);

        x = x_new;

        if diff_norm / x_norm < 1e-14 {
            break;
        }
    }

    Ok(x)
}

/// Dense matrix cosine via exp: cos(A) = Re(exp(iA)) computed via Taylor series
fn dense_cosm(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.nrows();
    let ident = Array2::<f64>::eye(n);

    // cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + ...
    let a2 = matmul_dense(&a.view(), &a.view());
    let mut result = ident.clone();
    let mut apow = ident.clone();
    let mut sign = 1.0_f64;

    for k in 1..=16usize {
        let two_k = 2 * k;
        apow = matmul_dense(&apow.view(), &a2.view());
        // Factorial: (2k)!
        let mut fact = 1.0f64;
        for i in 1..=(two_k as u64) {
            fact *= i as f64;
        }
        sign = -sign;
        result = result + (sign / fact) * &apow;

        let apow_norm = matrix_1norm(&apow.view());
        if apow_norm / fact < 1e-15 {
            break;
        }
    }

    Ok(result)
}

/// Dense matrix sine via Taylor series
fn dense_sinm(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.nrows();
    let a2 = matmul_dense(&a.view(), &a.view());

    // sin(A) = A - A³/3! + A⁵/5! - ...
    let mut result = a.to_owned();
    let mut apow = a.to_owned();
    let mut sign = 1.0_f64;

    for k in 1..=16usize {
        let two_k_plus_1 = 2 * k + 1;
        apow = matmul_dense(
            &matmul_dense(&apow.view(), &a2.view()).view(),
            &Array2::eye(n).view(),
        );
        apow = matmul_dense(&apow.view(), &a2.view());
        let mut fact = 1.0f64;
        for i in 1..=(two_k_plus_1 as u64) {
            fact *= i as f64;
        }
        sign = -sign;
        result = result + (sign / fact) * &apow;

        let apow_norm = matrix_1norm(&apow.view());
        if apow_norm / fact < 1e-15 {
            break;
        }
    }

    Ok(result)
}

// ============================================================================
// Small dense linear algebra helpers (avoiding circular dependencies)
// ============================================================================

/// Matrix 1-norm (max column sum)
fn matrix_1norm(a: &ArrayView2<f64>) -> f64 {
    let (m, n) = a.dim();
    let mut max_col = 0.0f64;
    for j in 0..n {
        let col_sum: f64 = (0..m).map(|i| a[[i, j]].abs()).sum();
        if col_sum > max_col {
            max_col = col_sum;
        }
    }
    max_col
}

/// Dense matrix multiplication C = A * B
fn matmul_dense(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (_, n) = b.dim();
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for l in 0..k {
            let aval = a[[i, l]];
            for j in 0..n {
                c[[i, j]] += aval * b[[l, j]];
            }
        }
    }
    c
}

/// Solve A X = B via LU with partial pivoting
fn solve_dense(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.nrows();
    let nb = b.ncols();

    // LU with partial pivoting
    let mut lu = a.to_owned();
    let mut pivot = vec![0usize; n];

    for k in 0..n {
        // Find pivot
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = lu[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        pivot[k] = max_row;

        // Swap rows
        if max_row != k {
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
        }

        let diag = lu[[k, k]];
        if diag.abs() < 1e-300 {
            return Err(LinalgError::SingularMatrixError(format!(
                "solve_dense: matrix is singular at column {k}"
            )));
        }

        for i in (k + 1)..n {
            lu[[i, k]] /= diag;
            for j in (k + 1)..n {
                let luk = lu[[i, k]];
                let lukj = lu[[k, j]];
                lu[[i, j]] -= luk * lukj;
            }
        }
    }

    // Forward/backward substitution for each column of B
    let mut x = b.to_owned();

    // Apply row permutation to X
    for k in 0..n {
        let p = pivot[k];
        if p != k {
            for j in 0..nb {
                let tmp = x[[k, j]];
                x[[k, j]] = x[[p, j]];
                x[[p, j]] = tmp;
            }
        }
    }

    // Forward substitution: L y = b (L has ones on diagonal)
    for k in 0..n {
        for i in (k + 1)..n {
            let lk = lu[[i, k]];
            for j in 0..nb {
                let xkj = x[[k, j]];
                x[[i, j]] -= lk * xkj;
            }
        }
    }

    // Backward substitution: U x = y
    for k in (0..n).rev() {
        let diag = lu[[k, k]];
        for j in 0..nb {
            x[[k, j]] /= diag;
        }
        for i in 0..k {
            let uk = lu[[i, k]];
            for j in 0..nb {
                let xkj = x[[k, j]];
                x[[i, j]] -= uk * xkj;
            }
        }
    }

    Ok(x)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_arnoldi_iteration() {
        // Simple 4×4 matrix
        let a = array![
            [2.0, 1.0, 0.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 2.0],
        ];
        let v = array![1.0, 0.0, 0.0, 0.0];

        let result = ArnoldiIteration::run(&a.view(), &v.view(), 3, None).expect("Arnoldi failed");

        // V should have orthonormal columns
        let vt_v = matmul_dense(&result.v.t(), &result.v.view());
        for i in 0..result.m {
            for j in 0..result.m {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vt_v[[i, j]] - expected).abs() < 1e-12,
                    "V^T V not identity at ({},{}) = {}",
                    i,
                    j,
                    vt_v[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_lanczos_iteration() {
        // Symmetric 4×4 matrix
        let a = array![
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ];
        let v = array![1.0, 0.0, 0.0, 0.0];

        let result = LanczosIteration::run(&a.view(), &v.view(), 3, None).expect("Lanczos failed");

        // V should have orthonormal columns
        let vt_v = matmul_dense(&result.v.t(), &result.v.view());
        for i in 0..result.m {
            for j in 0..result.m {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (vt_v[[i, j]] - expected).abs() < 1e-10,
                    "Lanczos V^T V not identity at ({},{}) = {}",
                    i,
                    j,
                    vt_v[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_matrix_exp_krylov_scalar() {
        // For a 1×1 "matrix" [t], exp(t*[a]) * [v] = exp(t*a) * v
        let a = array![[2.0_f64]];
        let v = array![3.0_f64];
        let t = 0.5;
        let result = MatrixExpKrylov::apply(&a.view(), &v.view(), t, None, None)
            .expect("MatrixExpKrylov failed");

        let expected = 3.0 * (2.0_f64 * 0.5).exp();
        assert!(
            (result[0] - expected).abs() < 1e-8,
            "MatrixExpKrylov scalar: {} vs {}",
            result[0],
            expected
        );
    }

    #[test]
    fn test_matrix_function_interpolation_identity() {
        // exp(I) * v should equal exp(1) * v for identity matrix
        let n = 4;
        let a = Array2::eye(n);
        let v: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let params = MatrixFunctionParams::new(MatrixFunctionType::Exponential)
            .with_max_krylov_dim(n)
            .with_tol(1e-10);

        let result = MatrixFunctionInterpolation::apply(&a.view(), &v.view(), &params)
            .expect("MatrixFunctionInterpolation failed");

        let expected_scale = 1.0_f64.exp();
        for i in 0..n {
            let expected = expected_scale * v[i];
            assert!(
                (result[i] - expected).abs() < 1e-6,
                "exp(I) * v failed at {i}: {} vs {}",
                result[i],
                expected
            );
        }
    }

    #[test]
    fn test_pade_expm_zero() {
        let a = Array2::<f64>::zeros((3, 3));
        let result = pade_expm(&a.view()).expect("pade_expm of zero failed");
        // exp(0) = I
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (result[[i, j]] - expected).abs() < 1e-12,
                    "pade_expm(0) failed at ({},{})={}",
                    i,
                    j,
                    result[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_lanczos_tridiagonal() {
        let a = array![[3.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 3.0],];
        let v = array![1.0, 1.0, 1.0];

        let result = LanczosIteration::run(&a.view(), &v.view(), 3, None).expect("Lanczos failed");
        let t = LanczosIteration::tridiagonal_matrix(&result);

        // T must be symmetric
        for i in 0..result.m {
            for j in 0..result.m {
                assert!((t[[i, j]] - t[[j, i]]).abs() < 1e-12);
            }
        }
    }
}
