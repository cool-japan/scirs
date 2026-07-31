//! Sparse signal recovery algorithms for compressed sensing.
//!
//! This module provides a collection of algorithms to recover a sparse signal
//! `x` from underdetermined linear measurements `y = Φx + noise`, where
//! `Φ ∈ ℝ^{m×n}` with `m ≪ n`.
//!
//! # Algorithms
//!
//! | Function / Struct      | Type     | Description                                  |
//! |------------------------|----------|----------------------------------------------|
//! | [`mp`]                 | Greedy   | Basic Matching Pursuit                        |
//! | [`omp`]                | Greedy   | Orthogonal Matching Pursuit                   |
//! | [`CoSaMP`]             | Greedy   | Compressive Sampling Matching Pursuit          |
//! | [`subspace_pursuit`]   | Greedy   | Subspace Pursuit (Dai & Milenkovic)            |
//! | [`basis_pursuit`]      | Convex   | L1 minimization via ADMM                      |
//! | [`irls`]               | Iterative| Iteratively Reweighted Least Squares (Lp)     |
//!
//! # References
//!
//! - Mallat & Zhang (1993) – Matching Pursuit
//! - Pati et al. (1993) – Orthogonal Matching Pursuit
//! - Needell & Tropp (2009) – CoSaMP
//! - Dai & Milenkovic (2009) – Subspace Pursuit
//! - Boyd et al. (2011) – ADMM for L1 minimization
//! - Daubechies et al. (2010) – IRLS for sparse recovery
//!
//! Pure Rust, no unwrap(), snake_case naming throughout.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use scirs2_linalg::{lstsq, solve};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Solve the least-squares problem min ‖Ax - b‖ for a (possibly tall) `A`.
/// Falls back to the pseudo-inverse via the normal equations when `m > n`.
fn least_squares(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Err(SignalError::DimensionMismatch(
            "least_squares: empty matrix".to_string(),
        ));
    }
    // Use lstsq from scirs2-linalg
    let result = lstsq(&a.view(), &b.view(), None)
        .map_err(|e| SignalError::ComputationError(format!("lstsq failed: {e}")))?;
    Ok(result.x)
}

/// L2 norm of a vector.
#[inline]
fn l2_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Absolute-value argmax: returns index of the column most correlated with `r`.
fn abs_argmax_inner_products(phi: &Array2<f64>, r: &Array1<f64>) -> Option<usize> {
    let (_m, n) = phi.dim();
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

/// Soft-threshold (proximal operator for L1).
#[inline]
fn soft_threshold(val: f64, thresh: f64) -> f64 {
    if val > thresh {
        val - thresh
    } else if val < -thresh {
        val + thresh
    } else {
        0.0
    }
}

/// Build a sub-matrix from selected column indices.
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

/// Scatter a sub-vector back to a full vector.
fn scatter(full_n: usize, indices: &[usize], vals: &Array1<f64>) -> Array1<f64> {
    let mut out = Array1::zeros(full_n);
    for (&idx, &v) in indices.iter().zip(vals.iter()) {
        out[idx] = v;
    }
    out
}

// ---------------------------------------------------------------------------
// Matching Pursuit (MP)
// ---------------------------------------------------------------------------

/// Basic Matching Pursuit (MP): greedy sparse recovery.
///
/// At each iteration the dictionary column most correlated with the current
/// residual is selected.  The coefficient is updated by a scalar projection
/// step (no re-orthogonalisation).  Repeat for `sparsity` iterations.
///
/// # Arguments
///
/// * `phi`      – Dictionary / measurement matrix (m × n).
/// * `y`        – Measurement vector of length m.
/// * `sparsity` – Maximum number of non-zero coefficients k.
///
/// # Returns
///
/// Coefficient vector of length n.
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] for incompatible dimensions.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::recovery::mp;
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![1.0, 0.0, 0.5, 0.0]);
/// let x = mp(&phi, &y, 2).expect("operation should succeed");
/// assert_eq!(x.len(), 4);
/// ```
pub fn mp(phi: &Array2<f64>, y: &Array1<f64>, sparsity: usize) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "mp: y has {} elements but phi has {m} rows",
            y.len()
        )));
    }
    if n == 0 || sparsity == 0 {
        return Ok(Array1::zeros(n));
    }

    let mut coeffs = Array1::<f64>::zeros(n);
    let mut residual = y.clone();

    // Precompute column norms squared for normalised projection
    let col_norms_sq: Vec<f64> = (0..n)
        .map(|j| phi.column(j).iter().map(|&v| v * v).sum::<f64>())
        .collect();

    for _ in 0..sparsity {
        // Find the column with maximum absolute inner product with residual
        let mut best_idx = 0usize;
        let mut best_abs = f64::NEG_INFINITY;
        for j in 0..n {
            let ip: f64 = phi
                .column(j)
                .iter()
                .zip(residual.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let abs_ip = ip.abs();
            if abs_ip > best_abs {
                best_abs = abs_ip;
                best_idx = j;
            }
        }

        let col_norm_sq = col_norms_sq[best_idx];
        if col_norm_sq < 1e-14 {
            break;
        }

        // Compute projection coefficient
        let ip: f64 = phi
            .column(best_idx)
            .iter()
            .zip(residual.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let alpha = ip / col_norm_sq;

        // Update coefficients and residual
        coeffs[best_idx] += alpha;
        for row in 0..m {
            residual[row] -= alpha * phi[[row, best_idx]];
        }

        if l2_norm(&residual) < 1e-12 {
            break;
        }
    }

    Ok(coeffs)
}

// ---------------------------------------------------------------------------
// Orthogonal Matching Pursuit (OMP)
// ---------------------------------------------------------------------------

/// Orthogonal Matching Pursuit (OMP).
///
/// Extends MP by orthogonally projecting the measurements onto the span of
/// selected atoms at every step, eliminating inter-atom interference.
/// Under RIP conditions, OMP exactly recovers k-sparse signals in k steps.
///
/// # Arguments
///
/// * `phi`      – Measurement matrix (m × n).
/// * `y`        – Measurement vector of length m.
/// * `sparsity` – Maximum number of non-zero coefficients k.
/// * `tol`      – Residual L2 tolerance for early stopping.
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] for incompatible dimensions.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::recovery::omp;
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![2.0, 0.0, -1.0, 0.0]);
/// let x = omp(&phi, &y, 2, 1e-8).expect("operation should succeed");
/// assert!((x[0] - 2.0).abs() < 1e-6);
/// assert!((x[2] - (-1.0)).abs() < 1e-6);
/// ```
pub fn omp(
    phi: &Array2<f64>,
    y: &Array1<f64>,
    sparsity: usize,
    tol: f64,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "omp: y has {} elements but phi has {m} rows",
            y.len()
        )));
    }
    if n == 0 || sparsity == 0 {
        return Ok(Array1::zeros(n));
    }

    let mut support: Vec<usize> = Vec::with_capacity(sparsity);
    let mut residual = y.clone();

    for _ in 0..sparsity {
        if l2_norm(&residual) < tol {
            break;
        }

        // Select the column with maximum absolute correlation
        let best_idx = abs_argmax_inner_products(phi, &residual)
            .ok_or_else(|| SignalError::ComputationError("omp: empty matrix".to_string()))?;

        if support.contains(&best_idx) {
            break;
        }
        support.push(best_idx);

        // Build sub-matrix for the current support
        let sub_phi = extract_columns(phi, &support);

        // Solve the least-squares problem: min ‖sub_phi c - y‖
        let c = least_squares(&sub_phi, y)?;

        // Recompute residual
        let approx = sub_phi.dot(&c);
        residual = y - &approx;
    }

    if support.is_empty() {
        return Ok(Array1::zeros(n));
    }

    // Final least-squares on the identified support
    let sub_phi = extract_columns(phi, &support);
    let c = least_squares(&sub_phi, y)?;
    Ok(scatter(n, &support, &c))
}

// ---------------------------------------------------------------------------
// CoSaMP
// ---------------------------------------------------------------------------

/// Compressive Sampling Matching Pursuit (CoSaMP).
///
/// CoSaMP is a greedy algorithm with provable reconstruction guarantees.  In
/// each iteration it: (1) identifies the 2k largest components of the proxy
/// vector `Φ^T r`; (2) merges this with the current support; (3) performs an
/// orthogonal projection; (4) prunes to the k largest coefficients.
///
/// # References
///
/// Needell & Tropp (2009) – "CoSaMP: Iterative signal recovery from incomplete
/// and inaccurate measurements"
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::recovery::CoSaMP;
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![3.0, 0.0, -2.0, 0.0]);
/// let cosamp = CoSaMP::new(2, 20, 1e-6);
/// let x = cosamp.recover(&phi, &y).expect("operation should succeed");
/// assert!((x[0] - 3.0).abs() < 1e-4);
/// ```
pub struct CoSaMP {
    /// Sparsity level k.
    pub sparsity: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the residual.
    pub tol: f64,
}

impl CoSaMP {
    /// Create a new CoSaMP solver.
    pub fn new(sparsity: usize, max_iter: usize, tol: f64) -> Self {
        Self {
            sparsity,
            max_iter,
            tol,
        }
    }

    /// Recover the sparse signal from measurements `y = Φ x`.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::DimensionMismatch`] for incompatible dimensions.
    pub fn recover(&self, phi: &Array2<f64>, y: &Array1<f64>) -> SignalResult<Array1<f64>> {
        let (m, n) = phi.dim();
        if y.len() != m {
            return Err(SignalError::DimensionMismatch(format!(
                "CoSaMP: y length {} != phi rows {m}",
                y.len()
            )));
        }
        let k = self.sparsity;
        if k == 0 || n == 0 {
            return Ok(Array1::zeros(n));
        }

        let mut x_est = Array1::<f64>::zeros(n);
        let mut residual = y.clone();

        for _iter in 0..self.max_iter {
            if l2_norm(&residual) < self.tol {
                break;
            }

            // Step 1: Proxy vector and select 2k largest entries
            let proxy = phi.t().dot(&residual);
            let mut idx_vals: Vec<(usize, f64)> = proxy
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v.abs()))
                .collect();
            idx_vals.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let proxy_support: Vec<usize> =
                idx_vals[..k.min(n) * 2].iter().map(|&(i, _)| i).collect();

            // Step 2: Merge with current support
            let mut merged: Vec<usize> = proxy_support.clone();
            for i in 0..n {
                if x_est[i].abs() > 1e-14 && !merged.contains(&i) {
                    merged.push(i);
                }
            }
            merged.sort_unstable();
            merged.dedup();

            // Step 3: Least-squares on merged support
            let sub_phi = extract_columns(phi, &merged);
            let c = least_squares(&sub_phi, y)?;
            let b = scatter(n, &merged, &c);

            // Step 4: Keep only k largest coefficients
            let mut idx_b: Vec<(usize, f64)> =
                b.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
            idx_b.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            x_est = Array1::zeros(n);
            for &(i, _) in idx_b[..k.min(n)].iter() {
                x_est[i] = b[i];
            }

            // Update residual
            let approx = phi.dot(&x_est);
            residual = y - &approx;
        }

        Ok(x_est)
    }
}

// ---------------------------------------------------------------------------
// Subspace Pursuit
// ---------------------------------------------------------------------------

/// Subspace Pursuit (SP) algorithm for sparse recovery.
///
/// SP iteratively refines a k-dimensional support estimate by:
/// 1. Computing the proxy `Φ^T r` and selecting the k largest entries.
/// 2. Merging with the current support and solving LS on the union.
/// 3. Pruning back to k entries with largest magnitudes.
///
/// # References
///
/// Dai & Milenkovic (2009) – "Subspace Pursuit for Compressive Sensing Signal
/// Reconstruction"
///
/// # Arguments
///
/// * `phi`      – Measurement matrix (m × n).
/// * `y`        – Measurement vector of length m.
/// * `sparsity` – Target sparsity k.
/// * `max_iter` – Maximum iteration count.
/// * `tol`      – Residual L2 convergence tolerance.
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] for incompatible dimensions.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::recovery::subspace_pursuit;
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(5);
/// let y = Array1::from_vec(vec![1.0, 0.0, 2.0, 0.0, 0.0]);
/// let x = subspace_pursuit(&phi, &y, 2, 30, 1e-6).expect("operation should succeed");
/// assert!((x[0] - 1.0).abs() < 1e-4);
/// assert!((x[2] - 2.0).abs() < 1e-4);
/// ```
pub fn subspace_pursuit(
    phi: &Array2<f64>,
    y: &Array1<f64>,
    sparsity: usize,
    max_iter: usize,
    tol: f64,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "subspace_pursuit: y length {} != phi rows {m}",
            y.len()
        )));
    }
    let k = sparsity;
    if k == 0 || n == 0 {
        return Ok(Array1::zeros(n));
    }

    // Initialise support using k largest entries of Φ^T y
    let proxy_init = phi.t().dot(y);
    let mut idx_init: Vec<(usize, f64)> = proxy_init
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    idx_init.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut support: Vec<usize> = idx_init[..k.min(n)].iter().map(|&(i, _)| i).collect();
    support.sort_unstable();

    let mut x_est = Array1::<f64>::zeros(n);

    // Initial LS estimate
    {
        let sub_phi = extract_columns(phi, &support);
        let c = least_squares(&sub_phi, y)?;
        for (&idx, &val) in support.iter().zip(c.iter()) {
            x_est[idx] = val;
        }
    }

    let mut prev_residual_norm = f64::INFINITY;

    for _ in 0..max_iter {
        let approx = phi.dot(&x_est);
        let residual = y - &approx;
        let res_norm = l2_norm(&residual);

        if res_norm < tol {
            break;
        }
        if (prev_residual_norm - res_norm).abs() < tol * 1e-3 {
            break;
        }
        prev_residual_norm = res_norm;

        // Compute proxy and select k new candidates
        let proxy = phi.t().dot(&residual);
        let mut idx_proxy: Vec<(usize, f64)> = proxy
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        idx_proxy
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let new_candidates: Vec<usize> = idx_proxy[..k.min(n)].iter().map(|&(i, _)| i).collect();

        // Merge support and new candidates
        let mut merged = support.clone();
        for &c_idx in &new_candidates {
            if !merged.contains(&c_idx) {
                merged.push(c_idx);
            }
        }
        merged.sort_unstable();

        // LS on merged support
        let sub_phi = extract_columns(phi, &merged);
        let c = least_squares(&sub_phi, y)?;
        let b = scatter(n, &merged, &c);

        // Prune to k largest
        let mut idx_b: Vec<(usize, f64)> =
            b.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
        idx_b.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        x_est = Array1::zeros(n);
        support.clear();
        for &(i, _) in idx_b[..k.min(n)].iter() {
            x_est[i] = b[i];
            support.push(i);
        }
        support.sort_unstable();
    }

    Ok(x_est)
}

// ---------------------------------------------------------------------------
// Basis Pursuit via ADMM
// ---------------------------------------------------------------------------

/// Solve Basis Pursuit: min ‖x‖_1 subject to y = Φ x.
///
/// Uses the Alternating Direction Method of Multipliers (ADMM) applied to the
/// equality-constrained L1 problem (Boyd et al. 2011, §6.2.1). The
/// variable-splitting formulation introduces `z = x`, with the x-update
/// projecting onto the affine feasible set `{x : Φx = y}` at every iteration:
///
/// - x-update: `x ← (z − u) + Φᵀ(ΦΦᵀ)⁻¹ (y − Φ(z − u))`
/// - z-update: `z ← soft_threshold(x + u, 1/ρ)`
/// - u-update: `u ← u + x − z`
///
/// Because the x-update always re-projects onto the exact constraint
/// `Φx = y`, the iterates converge to a feasible, sparsity-promoting
/// solution regardless of `ρ` (which only affects the convergence rate).
///
/// # Arguments
///
/// * `phi`      – Measurement matrix (m × n).
/// * `y`        – Measurement vector of length m.
/// * `rho`      – ADMM penalty parameter (typical: 1.0).
/// * `max_iter` – Maximum number of ADMM iterations.
/// * `tol`      – Primal / dual residual convergence tolerance.
///
/// # Returns
///
/// Recovered sparse coefficient vector of length n.
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] for incompatible dimensions.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::recovery::basis_pursuit;
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![1.5, 0.0, -0.5, 0.0]);
/// let x = basis_pursuit(&phi, &y, 1.0, 500, 1e-6).expect("operation should succeed");
/// assert!((x[0] - 1.5).abs() < 1e-3);
/// ```
pub fn basis_pursuit(
    phi: &Array2<f64>,
    y: &Array1<f64>,
    rho: f64,
    max_iter: usize,
    tol: f64,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "basis_pursuit: y length {} != phi rows {m}",
            y.len()
        )));
    }
    if n == 0 {
        return Ok(Array1::zeros(n));
    }
    if rho <= 0.0 {
        return Err(SignalError::ValueError(
            "basis_pursuit: rho must be positive".to_string(),
        ));
    }

    // Gram matrix Φ Φ^T (m × m), regularized for numerical stability. The
    // x-update solves against this matrix to project onto {x : Phi x = y}.
    let phi_t = phi.t().to_owned();
    let mut gram = phi.dot(&phi_t);
    for i in 0..m {
        gram[[i, i]] += 1e-10;
    }

    let threshold = 1.0 / rho;

    // Initialize at the least-norm feasible point x0 = Φ^T (Φ Φ^T)^{-1} y.
    let lam0 = solve_symmetric_positive_definite(&gram, y)?;
    let mut x: Array1<f64> = phi_t.dot(&lam0);
    let mut z = x.clone();
    let mut u = Array1::<f64>::zeros(n); // scaled dual variable

    for _ in 0..max_iter {
        let z_old = z.clone();

        // x-update: project (z - u) onto the affine set {x : Phi x = y}
        let v: Array1<f64> = &z - &u;
        let residual_y: Array1<f64> = y - &phi.dot(&v);
        let lam = solve_symmetric_positive_definite(&gram, &residual_y)?;
        x = &v + &phi_t.dot(&lam);

        // z-update: soft-threshold
        let x_plus_u = &x + &u;
        z = x_plus_u.mapv(|val| soft_threshold(val, threshold));

        // u-update
        u = u + &x - &z;

        // Check convergence: primal residual ‖x - z‖ and dual ‖ρ(z - z_old)‖
        let primal_res = l2_norm(&(&x - &z));
        let dual_res = rho * l2_norm(&(&z - &z_old));

        if primal_res < tol && dual_res < tol {
            break;
        }
    }

    Ok(z)
}

/// Solve a symmetric positive definite system A x = b using Cholesky
/// decomposition via Gaussian elimination (pure Rust, no LAPACK).
fn solve_symmetric_positive_definite(
    a: &Array2<f64>,
    b: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
            row.push(b[i]);
            row
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "solve_spd: matrix is singular or nearly singular".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for k in col..=n {
                let val = aug[col][k] * factor;
                aug[row][k] -= val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut rhs = aug[i][n];
        for j in (i + 1)..n {
            rhs -= aug[i][j] * x[j];
        }
        let denom = aug[i][i];
        if denom.abs() < 1e-14 {
            return Err(SignalError::ComputationError(
                "solve_spd: zero diagonal during back-substitution".to_string(),
            ));
        }
        x[i] = rhs / denom;
    }

    Array1::from_vec(x)
        .into_shape_with_order((n,))
        .map_err(|e| SignalError::ComputationError(format!("shape error: {e}")))
}

// ---------------------------------------------------------------------------
// IRLS (Iteratively Reweighted Least Squares)
// ---------------------------------------------------------------------------

/// Configuration for the IRLS solver.
#[derive(Clone, Debug)]
pub struct IrlsConfig {
    /// Exponent p in the Lp norm (0 < p ≤ 2; typical values: 0.5, 1.0).
    pub p: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on successive iterate change.
    pub tol: f64,
    /// Small regularization ε to avoid division by zero in weight computation.
    pub epsilon: f64,
    /// Optional LASSO regularization λ (use 0.0 for pure Lp recovery).
    pub lambda: f64,
}

impl Default for IrlsConfig {
    fn default() -> Self {
        Self {
            p: 1.0,
            max_iter: 200,
            tol: 1e-8,
            epsilon: 1e-6,
            lambda: 0.0,
        }
    }
}

/// Iteratively Reweighted Least Squares (IRLS) for Lp minimization.
///
/// Minimizes `‖x‖_p^p` subject to `Φx = y` by reformulating the Lp objective
/// as a sequence of weighted L2 problems.  At each step, the weight of entry
/// `x_i` is set to `w_i = (|x_i|^{p-2} + ε)^{-1}` (or its regularized
/// variant), and the weighted LS problem is solved.
///
/// This converges to the Lp minimizer for `0 < p ≤ 2`.  Setting `p = 1` gives
/// sparse (L1) solutions.  For `p < 1`, even sparser solutions are obtained.
///
/// # References
///
/// Daubechies, DeVore, Fornasier & Güntürk (2010) – "Iteratively reweighted
/// least squares minimization for sparse recovery"
///
/// # Arguments
///
/// * `phi`    – Measurement matrix (m × n).
/// * `y`      – Measurement vector of length m.
/// * `config` – IRLS configuration (p, iterations, tolerances).
///
/// # Errors
///
/// Returns [`SignalError::DimensionMismatch`] for incompatible dimensions.
///
/// # Example
///
/// ```
/// use scirs2_signal::compressed_sensing::recovery::{irls, IrlsConfig};
/// use scirs2_core::ndarray::{Array1, Array2};
/// let phi = Array2::eye(4);
/// let y = Array1::from_vec(vec![2.0, 0.0, -1.0, 0.0]);
/// let config = IrlsConfig { p: 1.0, max_iter: 50, ..Default::default() };
/// let x = irls(&phi, &y, &config).expect("operation should succeed");
/// assert!((x[0] - 2.0).abs() < 1e-4);
/// ```
pub fn irls(phi: &Array2<f64>, y: &Array1<f64>, config: &IrlsConfig) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();
    if y.len() != m {
        return Err(SignalError::DimensionMismatch(format!(
            "irls: y length {} != phi rows {m}",
            y.len()
        )));
    }
    if n == 0 {
        return Ok(Array1::zeros(n));
    }
    if config.p <= 0.0 || config.p > 2.0 {
        return Err(SignalError::ValueError(format!(
            "irls: p must be in (0, 2], got {}",
            config.p
        )));
    }

    let eps = config.epsilon.max(1e-14);

    // Initialize with the minimum-norm LS solution: x = Φ^T (Φ Φ^T)^{-1} y
    let phi_t = phi.t().to_owned();
    let gram = phi.dot(&phi_t); // m × m

    // For small m, use direct solve; otherwise fall back to identity weights
    let x0 = {
        let gram_reg = {
            let mut g = gram.clone();
            for i in 0..m {
                g[[i, i]] += 1e-8;
            }
            g
        };
        match solve_symmetric_positive_definite(&gram_reg, y) {
            Ok(alpha) => phi_t.dot(&alpha),
            Err(_) => {
                // Fall back to the pseudo-inverse via normal equations
                let normal = phi_t.dot(phi);
                let mut n_reg = normal.clone();
                for i in 0..n {
                    n_reg[[i, i]] += 1e-8;
                }
                match solve_symmetric_positive_definite(&n_reg, &phi_t.dot(y)) {
                    Ok(x) => x,
                    Err(_) => Array1::zeros(n),
                }
            }
        }
    };

    let mut x = x0;

    for _ in 0..config.max_iter {
        // Compute IRLS weights: w_i = 1 / (|x_i|^{2-p} + ε)
        let weights: Array1<f64> = x.mapv(|xi| {
            let abs_xi = xi.abs();
            1.0 / (abs_xi.powf(2.0 - config.p) + eps)
        });

        // Optional LASSO term: add λ * I to the weight matrix
        let lam = config.lambda;

        // Build the weighted normal equation: (Φ^T W Φ + λI) x = Φ^T W^{1/2} y
        // where W = diag(weights).  We use the re-weighting approach:
        // scale each column of Φ by sqrt(w_j) and solve via normal equations.
        //
        // Actually the IRLS normal equation for min sum w_i x_i^2 s.t. Φx=y is:
        //   min_x  (x^T W x + λ ‖x‖^2)  s.t. Φx=y
        //   → x = W^{-1} Φ^T (Φ W^{-1} Φ^T)^{-1} y
        //
        // Compute: D = W^{-1} = diag(1/weights), then Φ D Φ^T (m×m) + λ I

        let d_inv: Array1<f64> = weights.mapv(|w| 1.0 / (w + eps));

        // B = Φ * D (scale each column of Φ by d_inv[j])
        let mut b_mat = phi.to_owned();
        for j in 0..n {
            let d_j = d_inv[j];
            for i in 0..m {
                b_mat[[i, j]] *= d_j;
            }
        }

        // G = B Φ^T  (m × m)
        let mut g_mat = b_mat.dot(&phi_t);
        for i in 0..m {
            g_mat[[i, i]] += lam.max(1e-12);
        }

        let alpha = match solve_symmetric_positive_definite(&g_mat, y) {
            Ok(a) => a,
            Err(_) => {
                // Fallback: add more regularization
                for i in 0..m {
                    g_mat[[i, i]] += 1e-6;
                }
                solve_symmetric_positive_definite(&g_mat, y).unwrap_or_else(|_| Array1::zeros(m))
            }
        };

        // x = D Φ^T alpha
        let x_new: Array1<f64> = (0..n)
            .map(|j| {
                d_inv[j]
                    * phi_t
                        .column(j)
                        .iter()
                        .zip(alpha.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f64>()
            })
            .collect::<Vec<f64>>()
            .into();

        let delta = l2_norm(&(&x_new - &x));
        x = x_new;

        if delta < config.tol {
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
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_mp_identity() {
        // With identity dictionary, MP should recover exactly
        let phi = Array2::eye(4);
        let y = Array1::from_vec(vec![1.0, 0.0, -0.5, 0.0]);
        let x = mp(&phi, &y, 2).expect("mp should succeed");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[2] + 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_omp_identity() {
        let phi = Array2::eye(4);
        let y = Array1::from_vec(vec![2.0, 0.0, -1.0, 0.0]);
        let x = omp(&phi, &y, 2, 1e-8).expect("omp should succeed");
        assert!((x[0] - 2.0).abs() < 1e-8);
        assert!((x[2] + 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_cosamp_identity() {
        let phi = Array2::eye(4);
        let y = Array1::from_vec(vec![3.0, 0.0, -2.0, 0.0]);
        let cosamp = CoSaMP::new(2, 30, 1e-8);
        let x = cosamp.recover(&phi, &y).expect("cosamp should succeed");
        assert!((x[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_subspace_pursuit_identity() {
        let phi = Array2::eye(5);
        let y = Array1::from_vec(vec![1.0, 0.0, 2.0, 0.0, 0.0]);
        let x = subspace_pursuit(&phi, &y, 2, 50, 1e-8).expect("sp should succeed");
        assert!((x[0] - 1.0).abs() < 1e-4);
        assert!((x[2] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_basis_pursuit_identity() {
        let phi = Array2::eye(4);
        let y = Array1::from_vec(vec![1.5, 0.0, -0.5, 0.0]);
        // rho=5 gives threshold=0.2, allowing recovery of 1.5 and -0.5
        let x = basis_pursuit(&phi, &y, 5.0, 1000, 1e-6).expect("bp should succeed");
        // ADMM BP recovers signal with some bias from the ADMM constraint
        assert!((x[0] - 1.5).abs() < 0.15, "x[0]={}", x[0]);
    }

    #[test]
    fn test_irls_l1() {
        let phi = Array2::eye(4);
        let y = Array1::from_vec(vec![2.0, 0.0, -1.0, 0.0]);
        let config = IrlsConfig {
            p: 1.0,
            max_iter: 100,
            ..Default::default()
        };
        let x = irls(&phi, &y, &config).expect("irls should succeed");
        assert!((x[0] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_irls_config_default() {
        let cfg = IrlsConfig::default();
        assert_eq!(cfg.p, 1.0);
        assert_eq!(cfg.max_iter, 200);
    }
}
