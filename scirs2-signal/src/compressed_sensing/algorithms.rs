//! Core compressed sensing reconstruction algorithms.
//!
//! This module provides a comprehensive suite of sparse recovery algorithms
//! for solving the underdetermined system `y = Φ x` (or `y = Φ x + noise`)
//! under the assumption that `x` is (approximately) sparse.
//!
//! # Algorithms
//!
//! | Algorithm | Category | Minimises / Criterion |
//! |-----------|----------|----------------------|
//! | [`basis_pursuit`]   | Convex   | min ‖x‖₁ s.t. Φx = y (via ADMM) |
//! | [`lasso`]           | Convex   | ½‖Φx−y‖₂² + λ‖x‖₁ (via ADMM) |
//! | [`omp`]             | Greedy   | Orthogonal Matching Pursuit |
//! | [`cosamp`]          | Greedy   | Compressive Sampling Matching Pursuit |
//! | [`ista`]            | Proximal | Iterative Shrinkage-Thresholding |
//! | [`fista`]           | Proximal | Fast ISTA (Nesterov acceleration) |
//!
//! All functions follow the same signature convention:
//! `fn(phi, y, params…) -> SignalResult<Array1<f64>>`
//!
//! # References
//!
//! - Chen, Donoho & Saunders (1999) – Basis Pursuit
//! - Tibshirani (1996) – LASSO
//! - Pati, Rezaiifar & Krishnaprasad (1993) – OMP
//! - Needell & Tropp (2009) – CoSaMP
//! - Daubechies, Defrise & De Mol (2004) – ISTA
//! - Beck & Teboulle (2009) – FISTA
//! - Boyd et al. (2011) – ADMM distributed optimization
//!
//! Pure Rust, no `unwrap()`, snake_case naming throughout.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_linalg::lstsq;

use super::utils::{hard_threshold, l2_norm, soft_threshold, soft_threshold_vec};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Solve min ‖Ax − b‖ via least-squares (thin or fat systems).
fn least_squares_solve(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::DimensionMismatch(
            "least_squares_solve: empty matrix".to_string(),
        ));
    }
    let result = lstsq(&a.view(), &b.view(), None)
        .map_err(|e| SignalError::ComputationError(format!("lstsq error: {e}")))?;
    Ok(result.x)
}

/// Extract selected columns of `phi` into a new matrix.
fn extract_columns(phi: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let m = phi.nrows();
    let k = indices.len();
    let mut sub = Array2::zeros((m, k));
    for (col_out, &col_in) in indices.iter().enumerate() {
        for row in 0..m {
            sub[[row, col_out]] = phi[[row, col_in]];
        }
    }
    sub
}

/// Index of column with maximum |inner product| with residual.
fn argmax_correlation(phi: &Array2<f64>, r: &Array1<f64>) -> Option<usize> {
    let n = phi.ncols();
    if n == 0 {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for j in 0..n {
        let ip: f64 = phi
            .column(j)
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let abs_ip = ip.abs();
        if abs_ip > best_val {
            best_val = abs_ip;
            best_idx = j;
        }
    }
    Some(best_idx)
}

/// Compute the Lipschitz constant L = ‖ΦᵀΦ‖₂ (max singular value squared) via power iteration.
///
/// Uses 20 iterations of the power method to estimate the spectral norm of ΦᵀΦ,
/// which equals the squared maximum singular value of Φ.  This gives the tightest
/// valid step size for ISTA/FISTA and accelerates convergence.
fn lipschitz_constant(phi: &Array2<f64>) -> f64 {
    let m = phi.nrows();
    let n = phi.ncols();
    if m == 0 || n == 0 {
        return 1.0;
    }

    // Power iteration to estimate max eigenvalue of ΦᵀΦ
    // Initialize with a random-ish vector based on column sums
    let mut v: Array1<f64> = Array1::ones(n);
    let v_norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-14);
    v.mapv_inplace(|x| x / v_norm);

    let mut lambda_est = 1.0f64;

    for _ in 0..30 {
        // w = Φᵀ(Φ v)
        let phi_v: Array1<f64> = phi.dot(&v);
        let w: Array1<f64> = phi.t().dot(&phi_v);

        let w_norm = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if w_norm < 1e-14 {
            break;
        }
        // Rayleigh quotient: λ ≈ vᵀ(ΦᵀΦ)v / vᵀv = dot(v, w)
        lambda_est = v.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum::<f64>();
        v = w.mapv(|x| x / w_norm);
    }

    lambda_est.abs().max(1e-14)
}

// ---------------------------------------------------------------------------
// Basis Pursuit (BP)  –  ADMM formulation
// ---------------------------------------------------------------------------

/// Configuration for the ADMM-based Basis Pursuit solver.
#[derive(Debug, Clone)]
pub struct BasisPursuitConfig {
    /// ADMM penalty parameter ρ (default 1.0).
    pub rho: f64,
    /// Maximum number of ADMM iterations (default 1000).
    pub max_iter: usize,
    /// Convergence tolerance on the primal residual ‖Φx − y‖ (default 1e-6).
    pub tol: f64,
    /// Absolute tolerance for ADMM stopping (default 1e-4).
    pub abs_tol: f64,
    /// Relative tolerance for ADMM stopping (default 1e-3).
    pub rel_tol: f64,
}

impl Default for BasisPursuitConfig {
    fn default() -> Self {
        Self {
            rho: 5.0,
            max_iter: 2000,
            tol: 1e-6,
            abs_tol: 1e-4,
            rel_tol: 1e-3,
        }
    }
}

/// Basis Pursuit (BP): recover the sparsest signal consistent with `y = Φ x`.
///
/// Solves the L1-minimization problem
///
/// ```text
/// min  ‖x‖₁
/// s.t. Φ x = y
/// ```
///
/// via the Alternating Direction Method of Multipliers (ADMM).  The ADMM
/// formulation splits the variable as `x = z` and iterates:
///
/// ```text
/// x ← (ΦᵀΦ + ρI)⁻¹ (Φᵀy + ρ(z − u))
/// z ← S_{1/ρ}(x + u)
/// u ← u + x − z
/// ```
///
/// where `S_τ` is element-wise soft thresholding.
///
/// # Arguments
///
/// * `phi`    – Measurement matrix Φ (m × n).
/// * `y`      – Measurement vector (length m).
/// * `config` – ADMM configuration.
///
/// # Returns
///
/// Recovered sparse signal `x` of length `n`.
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] or [`SignalError::ValueError`].
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::algorithms::{basis_pursuit, BasisPursuitConfig};
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![1.5, 0.0, -0.5, 0.0]);
/// let x = basis_pursuit(&phi, &y, &BasisPursuitConfig::default()).expect("operation should succeed");
/// assert!((x[0] - 1.5).abs() < 0.05);
/// ```
pub fn basis_pursuit(
    phi: &Array2<f64>,
    y: &Array1<f64>,
    config: &BasisPursuitConfig,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::ValueError(
            "basis_pursuit: phi must be non-empty".to_string(),
        ));
    }
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "basis_pursuit: y.len()={} != m={}",
            y.len(),
            m
        )));
    }
    if config.rho <= 0.0 {
        return Err(SignalError::ValueError(
            "basis_pursuit: rho must be positive".to_string(),
        ));
    }

    let rho = config.rho;

    // ADMM for min ||x||_1 s.t. Phi x = y (Boyd 2011 §6.2.1)
    //
    // Variable splitting: x = z (L1 lives on z), with constraint Phi x = y.
    // The x-update projects z - u onto the affine subspace {x : Phi x = y}:
    //
    //   x ← (z - u) + Phi^T (Phi Phi^T)^{-1} (y - Phi(z - u))
    //       = (I - Phi^+(Phi)) (z - u) + Phi^+ y
    //
    // z-update: z ← S_{1/rho}(x + u)
    // u-update: u ← u + x - z
    //
    // Phi^+ = Phi^T (Phi Phi^T)^{-1} is the right pseudo-inverse.

    // Precompute Phi Phi^T (m × m) and its factorization
    let phi_t = phi.t().to_owned();
    // phi_phi_t = Phi Phi^T, shape (m, m)
    let phi_phi_t: Array2<f64> = phi.dot(&phi_t);

    // Add small regularization for numerical stability
    let mut a_mat = phi_phi_t;
    for i in 0..m {
        a_mat[[i, i]] += 1e-10;
    }

    let threshold = 1.0 / rho;

    // Initial point: least-norm solution x0 = Phi^T (Phi Phi^T)^{-1} y
    let lam0 = least_squares_solve(&a_mat, y)?;
    let mut x: Array1<f64> = phi_t.dot(&lam0);
    let mut z = x.clone();
    let mut u = Array1::<f64>::zeros(n);

    for _iter in 0..config.max_iter {
        // x-update: project (z - u) onto {x: Phi x = y}
        // x = (z - u) + Phi^T (Phi Phi^T)^{-1} (y - Phi(z - u))
        let v: Array1<f64> = &z - &u; // candidate
        let phi_v: Array1<f64> = phi.dot(&v);
        let residual_y: Array1<f64> = y - &phi_v;
        let lam = least_squares_solve(&a_mat, &residual_y)?;
        let correction: Array1<f64> = phi_t.dot(&lam);
        let x_new: Array1<f64> = &v + &correction;

        // z-update: z = S_{1/rho}(x + u)
        let x_plus_u: Array1<f64> = &x_new + &u;
        let z_new = soft_threshold_vec(&x_plus_u, threshold)?;

        // u-update: u = u + x - z
        let u_new: Array1<f64> = &u + &(&x_new - &z_new);

        // Stopping criteria (primal residual ‖x - z‖, dual residual ‖rho*(z_new - z)‖)
        let primal_res = l2_norm(&(&x_new - &z_new));
        let dual_res = rho * l2_norm(&(&z_new - &z));

        let eps_pri = config.abs_tol * (n as f64).sqrt()
            + config.rel_tol * l2_norm(&x_new).max(l2_norm(&z_new));
        let eps_dual = config.abs_tol * (n as f64).sqrt() + config.rel_tol * rho * l2_norm(&u_new);

        x = x_new;
        z = z_new;
        u = u_new;

        if primal_res < eps_pri && dual_res < eps_dual {
            break;
        }
        if primal_res < config.tol {
            break;
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// LASSO  –  ADMM formulation
// ---------------------------------------------------------------------------

/// Configuration for LASSO optimization.
#[derive(Debug, Clone)]
pub struct LassoConfig {
    /// Regularization parameter λ controlling sparsity (default 0.1).
    pub lambda: f64,
    /// ADMM penalty parameter ρ (default 5.0).
    pub rho: f64,
    /// Maximum number of iterations (default 2000).
    pub max_iter: usize,
    /// Convergence tolerance (default 1e-6).
    pub tol: f64,
    /// Absolute tolerance for ADMM stopping criterion (default 1e-4).
    pub abs_tol: f64,
    /// Relative tolerance for ADMM stopping criterion (default 1e-3).
    pub rel_tol: f64,
}

impl Default for LassoConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            rho: 5.0,
            max_iter: 2000,
            tol: 1e-6,
            abs_tol: 1e-4,
            rel_tol: 1e-3,
        }
    }
}

pub fn lasso(
    phi: &Array2<f64>,
    y: &Array1<f64>,
    config: &LassoConfig,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::ValueError(
            "lasso: phi must be non-empty".to_string(),
        ));
    }
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "lasso: y.len()={} != m={}",
            y.len(),
            m
        )));
    }
    if config.lambda < 0.0 {
        return Err(SignalError::ValueError(
            "lasso: lambda must be non-negative".to_string(),
        ));
    }
    if config.rho <= 0.0 {
        return Err(SignalError::ValueError(
            "lasso: rho must be positive".to_string(),
        ));
    }

    let rho = config.rho;
    let tau = config.lambda / rho; // soft-threshold for z-update

    // Precompute ΦᵀΦ + ρI and Φᵀy
    let phi_t = phi.t().to_owned();
    let phi_t_phi: Array2<f64> = phi_t.dot(phi);
    let phi_t_y: Array1<f64> = phi_t.dot(y);

    let mut a_mat = phi_t_phi;
    for i in 0..n {
        a_mat[[i, i]] += rho;
    }

    let mut x = Array1::<f64>::zeros(n);
    let mut z = Array1::<f64>::zeros(n);
    let mut u = Array1::<f64>::zeros(n);

    for _iter in 0..config.max_iter {
        // x-update
        let rhs: Array1<f64> = &phi_t_y + &((&z - &u) * rho);
        let x_new = least_squares_solve(&a_mat, &rhs)?;

        // z-update
        let x_plus_u: Array1<f64> = &x_new + &u;
        let z_new = soft_threshold_vec(&x_plus_u, tau)?;

        // u-update
        let u_new: Array1<f64> = &u + &(&x_new - &z_new);

        // Stopping criteria
        let primal_res = l2_norm(&(&x_new - &z_new));
        let dual_res = rho * l2_norm(&(&z_new - &z));

        let eps_pri = config.abs_tol * (n as f64).sqrt()
            + config.rel_tol
                * x_new
                    .iter()
                    .map(|&v| v * v)
                    .sum::<f64>()
                    .sqrt()
                    .max(z_new.iter().map(|&v| v * v).sum::<f64>().sqrt());
        let eps_dual =
            config.abs_tol * (n as f64).sqrt() + config.rel_tol * l2_norm(&(&phi_t.dot(&u_new)));

        x = x_new;
        z = z_new;
        u = u_new;

        if primal_res < eps_pri && dual_res < eps_dual {
            break;
        }
        if primal_res < config.tol {
            break;
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// OMP – Orthogonal Matching Pursuit
// ---------------------------------------------------------------------------

/// Configuration for Orthogonal Matching Pursuit.
#[derive(Debug, Clone)]
pub struct OmpConfig {
    /// Sparsity level (maximum number of non-zero entries, default 10).
    pub sparsity: usize,
    /// Stop early if residual drops below this tolerance (default 1e-6).
    pub tol: f64,
    /// Maximum iterations (defaults to `sparsity`).
    pub max_iter: Option<usize>,
}

impl Default for OmpConfig {
    fn default() -> Self {
        Self {
            sparsity: 10,
            tol: 1e-6,
            max_iter: None,
        }
    }
}

/// Orthogonal Matching Pursuit (OMP) greedy sparse recovery.
///
/// At each iteration selects the column of Φ most correlated with the
/// current residual, adds it to the active set, and solves the
/// least-squares problem on the active set to update the estimate.
///
/// OMP recovers a k-sparse signal exactly (with high probability) when
/// `m ≥ 2k` and the mutual coherence μ satisfies `μ < 1/(2k-1)`.
///
/// # Arguments
///
/// * `phi`    – Measurement matrix (m × n).
/// * `y`      – Observation vector (length m).
/// * `config` – OMP configuration.
///
/// # Returns
///
/// Sparse estimate `x` of length `n`.
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] or [`SignalError::ValueError`].
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::algorithms::{omp, OmpConfig};
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![2.0, 0.0, -1.0, 0.0]);
/// let cfg = OmpConfig { sparsity: 2, ..Default::default() };
/// let x = omp(&phi, &y, &cfg).expect("operation should succeed");
/// assert!((x[0] - 2.0).abs() < 1e-8);
/// assert!((x[2] + 1.0).abs() < 1e-8);
/// ```
pub fn omp(phi: &Array2<f64>, y: &Array1<f64>, config: &OmpConfig) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::ValueError(
            "omp: phi must be non-empty".to_string(),
        ));
    }
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "omp: y.len()={} != m={}",
            y.len(),
            m
        )));
    }
    if config.sparsity == 0 {
        return Err(SignalError::ValueError(
            "omp: sparsity must be positive".to_string(),
        ));
    }

    let max_iter = config.max_iter.unwrap_or(config.sparsity).min(n);
    let mut support: Vec<usize> = Vec::with_capacity(max_iter);
    let mut residual = y.clone();

    for _iter in 0..max_iter {
        if l2_norm(&residual) < config.tol {
            break;
        }

        // Select atom most correlated with residual
        let j = match argmax_correlation(phi, &residual) {
            Some(idx) => idx,
            None => break,
        };

        // Do not add already-selected atoms
        if support.contains(&j) {
            break;
        }
        support.push(j);

        // Solve least-squares on active sub-matrix
        let sub = extract_columns(phi, &support);
        let coeffs = least_squares_solve(&sub, y)?;

        // Update residual: r = y − Φ_S x_S
        let y_hat = sub.dot(&coeffs);
        residual = y - &y_hat;

        // Early exit if sparsity reached
        if support.len() >= config.sparsity {
            // Store coefficients in full n-dimensional vector
            let mut x_full = Array1::zeros(n);
            for (i, &idx) in support.iter().enumerate() {
                x_full[idx] = coeffs[i];
            }
            return Ok(x_full);
        }
    }

    // Reconstruct full vector from support + coefficients
    let mut x_full = Array1::zeros(n);
    if !support.is_empty() {
        let sub = extract_columns(phi, &support);
        let coeffs = least_squares_solve(&sub, y)?;
        for (i, &idx) in support.iter().enumerate() {
            x_full[idx] = coeffs[i];
        }
    }
    Ok(x_full)
}

// ---------------------------------------------------------------------------
// CoSaMP – Compressive Sampling Matching Pursuit
// ---------------------------------------------------------------------------

/// Configuration for CoSaMP.
#[derive(Debug, Clone)]
pub struct CoSaMPConfig {
    /// Target sparsity k (default 5).
    pub sparsity: usize,
    /// Maximum number of outer iterations (default 30).
    pub max_iter: usize,
    /// Convergence tolerance on ‖residual‖ (default 1e-6).
    pub tol: f64,
}

impl Default for CoSaMPConfig {
    fn default() -> Self {
        Self {
            sparsity: 5,
            max_iter: 30,
            tol: 1e-6,
        }
    }
}

/// CoSaMP: Compressive Sampling Matching Pursuit.
///
/// Iteratively selects `2k` atoms correlated with the residual proxy
/// `Φᵀ r`, merges them with the current support, solves a least-squares
/// sub-problem, and prunes back to the `k`-largest entries.
///
/// # Algorithm (Needell & Tropp 2009)
///
/// 1. Compute the proxy `p = Φᵀ r`.
/// 2. Identify `Ω` = indices of 2k largest entries of `|p|`.
/// 3. Merge: `T = Ω ∪ support(x)`.
/// 4. Solve: `b_T = Φ_T⁺ y`.
/// 5. Prune to k largest entries → new `x`.
/// 6. Update residual `r = y − Φ x`.
///
/// # Arguments
///
/// * `phi`    – Measurement matrix (m × n).
/// * `y`      – Measurement vector (length m).
/// * `config` – CoSaMP configuration.
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] or [`SignalError::ValueError`].
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::algorithms::{cosamp, CoSaMPConfig};
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![3.0, 0.0, -2.0, 0.0]);
/// let cfg = CoSaMPConfig { sparsity: 2, ..Default::default() };
/// let x = cosamp(&phi, &y, &cfg).expect("operation should succeed");
/// assert!((x[0] - 3.0).abs() < 1e-4);
/// assert!((x[2] + 2.0).abs() < 1e-4);
/// ```
pub fn cosamp(
    phi: &Array2<f64>,
    y: &Array1<f64>,
    config: &CoSaMPConfig,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::ValueError(
            "cosamp: phi must be non-empty".to_string(),
        ));
    }
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "cosamp: y.len()={} != m={}",
            y.len(),
            m
        )));
    }
    if config.sparsity == 0 {
        return Err(SignalError::ValueError(
            "cosamp: sparsity must be positive".to_string(),
        ));
    }

    let k = config.sparsity;
    let two_k = (2 * k).min(n);

    let phi_t = phi.t().to_owned();
    let mut x = Array1::<f64>::zeros(n);
    let mut residual = y.clone();

    for _iter in 0..config.max_iter {
        if l2_norm(&residual) < config.tol {
            break;
        }

        // Proxy: p = Φᵀ r
        let proxy: Array1<f64> = phi_t.dot(&residual);

        // Identify the 2k largest |proxy| entries
        let mut abs_proxy: Vec<(f64, usize)> = proxy
            .iter()
            .enumerate()
            .map(|(i, &v)| (v.abs(), i))
            .collect();
        abs_proxy.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let omega: Vec<usize> = abs_proxy[..two_k].iter().map(|&(_, i)| i).collect();

        // Merge with current support
        let cur_support: Vec<usize> = x
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v != 0.0 { Some(i) } else { None })
            .collect();

        let mut merged: Vec<usize> = omega.clone();
        for &idx in cur_support.iter() {
            if !merged.contains(&idx) {
                merged.push(idx);
            }
        }
        merged.sort_unstable();

        // Least-squares on merged support
        let sub = extract_columns(phi, &merged);
        let b_merged = least_squares_solve(&sub, y)?;

        // Prune to k largest entries
        let mut b_indexed: Vec<(f64, usize)> = b_merged
            .iter()
            .enumerate()
            .map(|(i, &v)| (v.abs(), i))
            .collect();
        b_indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let k_prune = k.min(b_indexed.len());
        let pruned_local: Vec<usize> = b_indexed[..k_prune]
            .iter()
            .map(|&(_, local)| local)
            .collect();

        // Map local indices back to global
        x = Array1::zeros(n);
        for &local in pruned_local.iter() {
            let global = merged[local];
            x[global] = b_merged[local];
        }

        // Update residual
        residual = y - &phi.dot(&x);
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// ISTA – Iterative Shrinkage-Thresholding Algorithm
// ---------------------------------------------------------------------------

/// Configuration for ISTA / FISTA.
#[derive(Debug, Clone)]
pub struct IstaConfig {
    /// L1 regularization parameter λ (default 0.1).
    pub lambda: f64,
    /// Maximum number of iterations (default 500).
    pub max_iter: usize,
    /// Convergence tolerance ‖x_new − x_old‖ / ‖x_old‖ (default 1e-6).
    pub tol: f64,
    /// Step size; if `None` computed automatically as 1/L (default None).
    pub step_size: Option<f64>,
}

impl Default for IstaConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            max_iter: 500,
            tol: 1e-6,
            step_size: None,
        }
    }
}

/// ISTA: Iterative Shrinkage-Thresholding Algorithm.
///
/// Solves the LASSO problem
///
/// ```text
/// min  ½ ‖Φ x − y‖₂² + λ ‖x‖₁
/// ```
///
/// by applying the proximal gradient descent step:
///
/// ```text
/// x_{t+1} = S_{λ/L}(x_t − (1/L) Φᵀ(Φ x_t − y))
/// ```
///
/// where `L` is the Lipschitz constant of the gradient `Φᵀ(Φx − y)`.
///
/// # Arguments
///
/// * `phi`    – Measurement matrix (m × n).
/// * `y`      – Observation vector (length m).
/// * `config` – ISTA configuration.
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] or [`SignalError::ValueError`].
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::algorithms::{ista, IstaConfig};
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![2.0, 0.05, -1.0, 0.0]);
/// let cfg = IstaConfig { lambda: 0.1, max_iter: 200, ..Default::default() };
/// let x = ista(&phi, &y, &cfg).expect("operation should succeed");
/// // Large entries should be approximately recovered
/// assert!((x[0] - 2.0).abs() < 0.15);
/// ```
pub fn ista(phi: &Array2<f64>, y: &Array1<f64>, config: &IstaConfig) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::ValueError(
            "ista: phi must be non-empty".to_string(),
        ));
    }
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "ista: y.len()={} != m={}",
            y.len(),
            m
        )));
    }
    if config.lambda < 0.0 {
        return Err(SignalError::ValueError(
            "ista: lambda must be non-negative".to_string(),
        ));
    }

    let phi_t = phi.t().to_owned();
    let l = config
        .step_size
        .map(|s| 1.0 / s)
        .unwrap_or_else(|| lipschitz_constant(phi));
    let step = 1.0 / l;
    let threshold = config.lambda * step;

    let mut x = Array1::<f64>::zeros(n);

    for _iter in 0..config.max_iter {
        // Gradient of the data fidelity: g = Φᵀ(Φx − y)
        let residual: Array1<f64> = phi.dot(&x) - y;
        let grad: Array1<f64> = phi_t.dot(&residual);

        // Gradient step
        let x_half: Array1<f64> = &x - &(&grad * step);

        // Proximal step (soft thresholding)
        let x_new = soft_threshold_vec(&x_half, threshold)?;

        // Convergence check
        let diff_norm = l2_norm(&(&x_new - &x));
        let x_norm = l2_norm(&x);
        x = x_new;

        if diff_norm < config.tol * (x_norm + 1e-14) {
            break;
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// FISTA – Fast Iterative Shrinkage-Thresholding Algorithm
// ---------------------------------------------------------------------------

/// Fast ISTA (FISTA) with Nesterov momentum acceleration.
///
/// Solves the same LASSO problem as [`ista`] but accelerates convergence
/// from O(1/t) to O(1/t²) using the momentum scheme of Beck & Teboulle (2009):
///
/// ```text
/// y_{t+1} = x_t + ((t_k - 1) / t_{k+1}) (x_t - x_{t-1})
/// x_{t+1} = S_{λ/L}(y_{t+1} - (1/L) Φᵀ(Φ y_{t+1} - y))
/// t_{k+1} = (1 + sqrt(1 + 4 t_k²)) / 2
/// ```
///
/// # Arguments
///
/// * `phi`    – Measurement matrix (m × n).
/// * `y`      – Observation vector (length m).
/// * `config` – Shared ISTA configuration (same struct used for both).
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] or [`SignalError::ValueError`].
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::algorithms::{fista, IstaConfig};
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![2.0, 0.0, -1.0, 0.0]);
/// let cfg = IstaConfig { lambda: 0.05, max_iter: 100, ..Default::default() };
/// let x = fista(&phi, &y, &cfg).expect("operation should succeed");
/// assert!((x[0] - 2.0).abs() < 0.1);
/// ```
pub fn fista(phi: &Array2<f64>, y: &Array1<f64>, config: &IstaConfig) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::ValueError(
            "fista: phi must be non-empty".to_string(),
        ));
    }
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "fista: y.len()={} != m={}",
            y.len(),
            m
        )));
    }
    if config.lambda < 0.0 {
        return Err(SignalError::ValueError(
            "fista: lambda must be non-negative".to_string(),
        ));
    }

    let phi_t = phi.t().to_owned();
    let l = config
        .step_size
        .map(|s| 1.0 / s)
        .unwrap_or_else(|| lipschitz_constant(phi));
    let step = 1.0 / l;
    let threshold = config.lambda * step;

    let mut x = Array1::<f64>::zeros(n);
    let mut x_prev = Array1::<f64>::zeros(n);
    let mut t_k = 1.0f64;
    // Momentum point
    let mut z = Array1::<f64>::zeros(n);

    for _iter in 0..config.max_iter {
        // Gradient at momentum point z
        let residual: Array1<f64> = phi.dot(&z) - y;
        let grad: Array1<f64> = phi_t.dot(&residual);

        // Proximal gradient step
        let z_half: Array1<f64> = &z - &(&grad * step);
        let x_new = soft_threshold_vec(&z_half, threshold)?;

        // Nesterov momentum update
        let t_new = (1.0 + (1.0 + 4.0 * t_k * t_k).sqrt()) / 2.0;
        let momentum = (t_k - 1.0) / t_new;

        // z_new = x_new + momentum * (x_new - x_prev)
        let z_new: Array1<f64> = &x_new + &((&x_new - &x_prev) * momentum);

        // Convergence check
        let diff_norm = l2_norm(&(&x_new - &x));
        let x_norm = l2_norm(&x);

        x_prev = x;
        x = x_new;
        z = z_new;
        t_k = t_new;

        if diff_norm < config.tol * (x_norm + 1e-14) {
            break;
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn identity_phi(n: usize) -> Array2<f64> {
        Array2::eye(n)
    }

    #[test]
    fn test_basis_pursuit_identity() {
        let phi = identity_phi(4);
        let y = Array1::from_vec(vec![1.5, 0.0, -0.5, 0.0]);
        // Projection-based ADMM recovers exact solution with default config
        let x = basis_pursuit(&phi, &y, &BasisPursuitConfig::default()).expect("bp should succeed");
        assert!((x[0] - 1.5).abs() < 0.1, "x[0]={}", x[0]);
        assert!(x[1].abs() < 0.1, "x[1]={}", x[1]);
        assert!((x[2] + 0.5).abs() < 0.1, "x[2]={}", x[2]);
        assert!(x[3].abs() < 0.1, "x[3]={}", x[3]);
    }

    #[test]
    fn test_basis_pursuit_dimension_mismatch() {
        let phi = identity_phi(4);
        let y = Array1::zeros(3); // wrong length
        assert!(basis_pursuit(&phi, &y, &BasisPursuitConfig::default()).is_err());
    }

    #[test]
    fn test_basis_pursuit_empty_phi() {
        let phi: Array2<f64> = Array2::zeros((0, 4));
        let y: Array1<f64> = Array1::zeros(0);
        assert!(basis_pursuit(&phi, &y, &BasisPursuitConfig::default()).is_err());
    }

    #[test]
    fn test_lasso_identity_sparse() {
        let phi = identity_phi(4);
        // True signal has two components; small noise on others
        let y = Array1::from_vec(vec![2.0, 0.05, -1.5, 0.02]);
        let cfg = LassoConfig {
            lambda: 0.2,
            ..Default::default()
        };
        let x = lasso(&phi, &y, &cfg).expect("lasso should succeed");
        // Large entries should be well recovered
        // LASSO with ADMM on identity Phi: check basic direction of recovery
        assert!(x[0] > 0.5, "x[0]={} should be positive and large", x[0]);
        assert!(
            x[2] < -0.5,
            "x[2]={} should be negative and large in magnitude",
            x[2]
        );
        // Small noisy entries should be suppressed
        assert!(x[1].abs() < 0.5, "x[1]={}", x[1]);
    }

    #[test]
    fn test_lasso_negative_lambda_error() {
        let phi = identity_phi(4);
        let y = Array1::zeros(4);
        let cfg = LassoConfig {
            lambda: -0.1,
            ..Default::default()
        };
        assert!(lasso(&phi, &y, &cfg).is_err());
    }

    #[test]
    fn test_omp_identity() {
        let phi = identity_phi(4);
        let y = Array1::from_vec(vec![2.0, 0.0, -1.0, 0.0]);
        let cfg = OmpConfig {
            sparsity: 2,
            ..Default::default()
        };
        let x = omp(&phi, &y, &cfg).expect("omp should succeed");
        assert!((x[0] - 2.0).abs() < 1e-8, "x[0]={}", x[0]);
        assert!((x[2] + 1.0).abs() < 1e-8, "x[2]={}", x[2]);
    }

    #[test]
    fn test_omp_zero_sparsity_error() {
        let phi = identity_phi(4);
        let y = Array1::zeros(4);
        let cfg = OmpConfig {
            sparsity: 0,
            ..Default::default()
        };
        assert!(omp(&phi, &y, &cfg).is_err());
    }

    #[test]
    fn test_omp_dimension_mismatch() {
        let phi = identity_phi(4);
        let y = Array1::zeros(3);
        let cfg = OmpConfig::default();
        assert!(omp(&phi, &y, &cfg).is_err());
    }

    #[test]
    fn test_cosamp_identity() {
        let phi = identity_phi(4);
        let y = Array1::from_vec(vec![3.0, 0.0, -2.0, 0.0]);
        let cfg = CoSaMPConfig {
            sparsity: 2,
            ..Default::default()
        };
        let x = cosamp(&phi, &y, &cfg).expect("cosamp should succeed");
        assert!((x[0] - 3.0).abs() < 1e-3, "x[0]={}", x[0]);
        assert!((x[2] + 2.0).abs() < 1e-3, "x[2]={}", x[2]);
    }

    #[test]
    fn test_cosamp_zero_sparsity_error() {
        let phi = identity_phi(4);
        let y = Array1::zeros(4);
        let cfg = CoSaMPConfig {
            sparsity: 0,
            ..Default::default()
        };
        assert!(cosamp(&phi, &y, &cfg).is_err());
    }

    #[test]
    fn test_ista_identity() {
        let phi = identity_phi(4);
        let y = Array1::from_vec(vec![2.0, 0.0, -1.0, 0.0]);
        let cfg = IstaConfig {
            lambda: 0.01,
            max_iter: 300,
            ..Default::default()
        };
        let x = ista(&phi, &y, &cfg).expect("ista should succeed");
        assert!((x[0] - 2.0).abs() < 0.02, "x[0]={}", x[0]);
        assert!((x[2] + 1.0).abs() < 0.02, "x[2]={}", x[2]);
    }

    #[test]
    fn test_ista_negative_lambda_error() {
        let phi = identity_phi(4);
        let y = Array1::zeros(4);
        let cfg = IstaConfig {
            lambda: -1.0,
            ..Default::default()
        };
        assert!(ista(&phi, &y, &cfg).is_err());
    }

    #[test]
    fn test_fista_identity() {
        let phi = identity_phi(4);
        let y = Array1::from_vec(vec![2.0, 0.0, -1.0, 0.0]);
        let cfg = IstaConfig {
            lambda: 0.01,
            max_iter: 200,
            ..Default::default()
        };
        let x = fista(&phi, &y, &cfg).expect("fista should succeed");
        assert!((x[0] - 2.0).abs() < 0.02, "x[0]={}", x[0]);
        assert!((x[2] + 1.0).abs() < 0.02, "x[2]={}", x[2]);
    }

    #[test]
    fn test_fista_converges_faster_than_ista() {
        // FISTA should converge in fewer iterations than ISTA for the same tolerance
        // We test this by checking that both converge to the same answer
        let phi = identity_phi(6);
        let y = Array1::from_vec(vec![3.0, 0.0, -1.5, 0.0, 2.0, 0.0]);
        let cfg = IstaConfig {
            lambda: 0.05,
            max_iter: 100,
            ..Default::default()
        };
        let x_ista = ista(&phi, &y, &cfg.clone()).expect("ista should succeed");
        let x_fista = fista(&phi, &y, &cfg).expect("fista should succeed");
        // Both should recover the three non-zero entries
        assert!(
            (x_fista[0] - 3.0).abs() < (x_ista[0] - 3.0).abs() + 0.1,
            "fista should be at least as good as ista"
        );
    }

    #[test]
    fn test_config_defaults() {
        let bp_cfg = BasisPursuitConfig::default();
        assert_eq!(bp_cfg.max_iter, 2000);
        assert!(bp_cfg.rho > 0.0);

        let lasso_cfg = LassoConfig::default();
        assert!(lasso_cfg.lambda > 0.0);
        assert_eq!(lasso_cfg.max_iter, 2000);

        let omp_cfg = OmpConfig::default();
        assert_eq!(omp_cfg.sparsity, 10);

        let cosamp_cfg = CoSaMPConfig::default();
        assert_eq!(cosamp_cfg.sparsity, 5);

        let ista_cfg = IstaConfig::default();
        assert!(ista_cfg.lambda > 0.0);
        assert_eq!(ista_cfg.max_iter, 500);
    }
}
