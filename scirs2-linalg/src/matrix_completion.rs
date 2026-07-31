//! Matrix Completion Algorithms
//!
//! This module provides algorithms for recovering a low-rank matrix from
//! partial observations (missing entries). Applications include:
//! - Recommendation systems (collaborative filtering)
//! - Sensor network data recovery
//! - Image inpainting
//!
//! # Algorithms
//!
//! - **Nuclear Norm Minimization**: Convex relaxation via proximal gradient
//! - **Alternating Least Squares (ALS)**: Fast iterative low-rank factorization
//! - **Singular Value Thresholding (SVT)**: Proximal operator for nuclear norm
//! - **Soft-Impute**: Iterative SVD-based imputation
//! - **Missing Value Pattern Handling**: Utilities for masked/observed data
//!
//! # References
//!
//! - Candes & Recht (2009). "Exact matrix completion via convex optimization."
//! - Mazumder, Hastie, Tibshirani (2010). "Spectral regularization algorithms
//!   for learning large incomplete matrices." (Soft-Impute)
//! - Cai, Candes, Shen (2010). "A singular value thresholding algorithm for
//!   matrix rank minimization." (SVT)

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Normal};
use std::fmt::Debug;
use std::iter::Sum;

use crate::decomposition::svd;
use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Observation Mask
// ============================================================================

/// Represents the pattern of observed entries in a matrix.
#[derive(Debug, Clone)]
pub struct ObservationMask {
    /// (row, col) pairs of observed entries
    pub observed: Vec<(usize, usize)>,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
}

impl ObservationMask {
    /// Create a mask from a boolean matrix (true = observed).
    pub fn from_bool_matrix(mask: &ArrayView2<bool>) -> Self {
        let (nrows, ncols) = mask.dim();
        let mut observed = Vec::new();
        for i in 0..nrows {
            for j in 0..ncols {
                if mask[[i, j]] {
                    observed.push((i, j));
                }
            }
        }
        ObservationMask {
            observed,
            nrows,
            ncols,
        }
    }

    /// Create a mask from observed (row, col) pairs.
    pub fn from_indices(observed: Vec<(usize, usize)>, nrows: usize, ncols: usize) -> Self {
        ObservationMask {
            observed,
            nrows,
            ncols,
        }
    }

    /// Create a mask where non-NaN entries are observed.
    pub fn from_nan_matrix<F: Float>(matrix: &ArrayView2<F>) -> Self {
        let (nrows, ncols) = matrix.dim();
        let mut observed = Vec::new();
        for i in 0..nrows {
            for j in 0..ncols {
                if !matrix[[i, j]].is_nan() {
                    observed.push((i, j));
                }
            }
        }
        ObservationMask {
            observed,
            nrows,
            ncols,
        }
    }

    /// Return the fraction of observed entries.
    pub fn observation_ratio(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        self.observed.len() as f64 / total as f64
    }

    /// Check if a particular entry is observed.
    pub fn is_observed(&self, row: usize, col: usize) -> bool {
        self.observed.contains(&(row, col))
    }
}

/// Result of matrix completion
#[derive(Debug, Clone)]
pub struct CompletionResult<F> {
    /// Completed matrix
    pub matrix: Array2<F>,
    /// Number of iterations used
    pub iterations: usize,
    /// Final objective value / residual
    pub residual: F,
    /// Whether the algorithm converged
    pub converged: bool,
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for matrix completion algorithms
#[derive(Debug, Clone)]
pub struct CompletionConfig<F> {
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: F,
    /// Target rank (for ALS and Soft-Impute)
    pub rank: Option<usize>,
    /// Regularization parameter lambda
    pub lambda: F,
    /// Step size / learning rate (for proximal gradient)
    pub step_size: Option<F>,
}

impl<F: Float> CompletionConfig<F> {
    /// Create default configuration
    pub fn new(lambda: F) -> Self {
        Self {
            max_iter: 200,
            tolerance: F::from(1e-6).unwrap_or(F::epsilon()),
            rank: None,
            lambda,
            step_size: None,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: F) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set target rank
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.rank = Some(rank);
        self
    }

    /// Set step size
    pub fn with_step_size(mut self, step_size: F) -> Self {
        self.step_size = Some(step_size);
        self
    }
}

// ============================================================================
// Singular Value Thresholding (SVT)
// ============================================================================

/// Singular Value Thresholding operator.
///
/// Applies soft-thresholding to the singular values: D_tau(X) = U * S_tau * V^T
/// where S_tau = diag(max(sigma_i - tau, 0)).
///
/// This is the proximal operator for the nuclear norm.
///
/// # Arguments
///
/// * `x` - Input matrix
/// * `tau` - Threshold value
///
/// # Returns
///
/// * Thresholded matrix
pub fn singular_value_threshold<F>(x: &ArrayView2<F>, tau: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    let (u, s, vt) = svd(x, false, None)?;

    let k = s.len();

    // Soft-threshold singular values
    let mut s_thresh = Array1::zeros(k);
    let mut effective_rank = 0;
    for i in 0..k {
        let val = s[i] - tau;
        if val > F::zero() {
            s_thresh[i] = val;
            effective_rank += 1;
        }
    }

    if effective_rank == 0 {
        return Ok(Array2::zeros(x.dim()));
    }

    // Reconstruct: U[:, :r] * diag(s_thresh[:r]) * Vt[:r, :]
    let r = effective_rank;
    let u_r = u.slice(s![.., ..r]).to_owned();
    let vt_r = vt.slice(s![..r, ..]).to_owned();

    let (m, n) = x.dim();
    let mut result = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut val = F::zero();
            for kk in 0..r {
                val += u_r[[i, kk]] * s_thresh[kk] * vt_r[[kk, j]];
            }
            result[[i, j]] = val;
        }
    }

    Ok(result)
}

// ============================================================================
// SVT Algorithm for Matrix Completion
// ============================================================================

/// Matrix completion via Singular Value Thresholding (SVT).
///
/// Solves: min ||X||_* subject to X_ij = M_ij for (i,j) in Omega
/// where ||X||_* is the nuclear norm and Omega is the set of observed entries.
///
/// # Arguments
///
/// * `observed_values` - Matrix with observed values (unobserved can be 0 or NaN)
/// * `mask` - Observation mask
/// * `config` - Algorithm configuration (lambda = threshold parameter)
///
/// # Returns
///
/// * `CompletionResult` with completed matrix
///
/// # References
///
/// Cai, Candes, Shen (2010). "A singular value thresholding algorithm."
pub fn svt_completion<F>(
    observed_values: &ArrayView2<F>,
    mask: &ObservationMask,
    config: &CompletionConfig<F>,
) -> LinalgResult<CompletionResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = observed_values.dim();
    if mask.nrows != m || mask.ncols != n {
        return Err(LinalgError::DimensionError(
            "Mask dimensions do not match matrix dimensions".to_string(),
        ));
    }

    let tau = config.lambda;
    let delta = config.step_size.unwrap_or_else(|| {
        F::from(1.2 * (m * n) as f64 / mask.observed.len().max(1) as f64).unwrap_or(F::one())
    });

    // Initialize Y = delta * P_Omega(M) (sparse initialization)
    let mut y = Array2::zeros((m, n));
    for &(i, j) in &mask.observed {
        y[[i, j]] = delta * observed_values[[i, j]];
    }

    let mut x = Array2::zeros((m, n));
    let mut converged = false;
    let mut last_residual = F::infinity();
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // X = D_tau(Y) = SVT(Y, tau)
        x = singular_value_threshold(&y.view(), tau)?;

        // Compute residual on observed entries
        let mut residual = F::zero();
        let mut obs_count = F::zero();
        for &(i, j) in &mask.observed {
            let diff = observed_values[[i, j]] - x[[i, j]];
            residual += diff * diff;
            obs_count += F::one();
        }
        residual = if obs_count > F::zero() {
            (residual / obs_count).sqrt()
        } else {
            F::zero()
        };

        // Check convergence
        let rel_change = if last_residual > F::epsilon() {
            (last_residual - residual).abs() / last_residual
        } else {
            F::zero()
        };

        if rel_change < config.tolerance && iter > 0 {
            converged = true;
            last_residual = residual;
            break;
        }
        last_residual = residual;

        // Update Y = Y + delta * P_Omega(M - X)
        for &(i, j) in &mask.observed {
            y[[i, j]] += delta * (observed_values[[i, j]] - x[[i, j]]);
        }
    }

    Ok(CompletionResult {
        matrix: x,
        iterations,
        residual: last_residual,
        converged,
    })
}

// ============================================================================
// Nuclear Norm Minimization (Proximal Gradient)
// ============================================================================

/// Matrix completion via nuclear norm minimization using proximal gradient descent.
///
/// Solves: min (1/2) ||P_Omega(X - M)||_F^2 + lambda * ||X||_*
///
/// Uses the Iterative Soft-Thresholding Algorithm (ISTA).
///
/// # Arguments
///
/// * `observed_values` - Matrix with observed values
/// * `mask` - Observation mask
/// * `config` - Algorithm configuration
///
/// # Returns
///
/// * `CompletionResult` with completed matrix
pub fn nuclear_norm_completion<F>(
    observed_values: &ArrayView2<F>,
    mask: &ObservationMask,
    config: &CompletionConfig<F>,
) -> LinalgResult<CompletionResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = observed_values.dim();
    if mask.nrows != m || mask.ncols != n {
        return Err(LinalgError::DimensionError(
            "Mask dimensions do not match matrix dimensions".to_string(),
        ));
    }

    let lambda = config.lambda;
    let step = config.step_size.unwrap_or(F::one());

    // Initialize X = 0
    let mut x = Array2::zeros((m, n));

    // Set observed entries
    for &(i, j) in &mask.observed {
        x[[i, j]] = observed_values[[i, j]];
    }

    let mut converged = false;
    let mut last_residual = F::infinity();
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Gradient step: G = X - step * P_Omega(X - M)
        let mut g = x.clone();
        for &(i, j) in &mask.observed {
            g[[i, j]] -= step * (x[[i, j]] - observed_values[[i, j]]);
        }

        // Proximal step: X_new = SVT(G, step * lambda)
        let x_new = singular_value_threshold(&g.view(), step * lambda)?;

        // Compute change
        let mut change = F::zero();
        let mut norm_x = F::zero();
        for i in 0..m {
            for j in 0..n {
                let diff = x_new[[i, j]] - x[[i, j]];
                change += diff * diff;
                norm_x += x_new[[i, j]] * x_new[[i, j]];
            }
        }
        let rel_change = if norm_x > F::epsilon() {
            change.sqrt() / norm_x.sqrt()
        } else {
            change.sqrt()
        };

        x = x_new;

        // Compute residual on observed entries
        let mut residual = F::zero();
        let mut obs_count = F::zero();
        for &(i, j) in &mask.observed {
            let diff = observed_values[[i, j]] - x[[i, j]];
            residual += diff * diff;
            obs_count += F::one();
        }
        residual = if obs_count > F::zero() {
            (residual / obs_count).sqrt()
        } else {
            F::zero()
        };

        if rel_change < config.tolerance {
            converged = true;
            last_residual = residual;
            break;
        }
        last_residual = residual;
    }

    Ok(CompletionResult {
        matrix: x,
        iterations,
        residual: last_residual,
        converged,
    })
}

// ============================================================================
// Alternating Least Squares (ALS)
// ============================================================================

/// Matrix completion via Alternating Least Squares (ALS).
///
/// Factorizes X = U * V^T where U is m x k and V is n x k,
/// and alternately optimizes U and V to minimize
/// sum_{(i,j) in Omega} (M_ij - (UV^T)_ij)^2 + lambda * (||U||_F^2 + ||V||_F^2)
///
/// # Arguments
///
/// * `observed_values` - Matrix with observed values
/// * `mask` - Observation mask
/// * `config` - Algorithm configuration (rank required)
///
/// # Returns
///
/// * `CompletionResult` with completed matrix
pub fn als_completion<F>(
    observed_values: &ArrayView2<F>,
    mask: &ObservationMask,
    config: &CompletionConfig<F>,
) -> LinalgResult<CompletionResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = observed_values.dim();
    if mask.nrows != m || mask.ncols != n {
        return Err(LinalgError::DimensionError(
            "Mask dimensions do not match matrix dimensions".to_string(),
        ));
    }

    let rank = config
        .rank
        .ok_or_else(|| LinalgError::InvalidInput("ALS requires a target rank".to_string()))?;

    if rank == 0 || rank > m.min(n) {
        return Err(LinalgError::InvalidInput(format!(
            "Rank ({rank}) must be in [1, {}]",
            m.min(n)
        )));
    }

    let lambda = config.lambda;

    // Initialize U and V with small random values
    let mut rng = scirs2_core::random::rng();
    let normal = Normal::new(0.0, 0.1).map_err(|e| {
        LinalgError::ComputationError(format!("Failed to create distribution: {e}"))
    })?;

    let mut u_factor = Array2::zeros((m, rank));
    let mut v_factor = Array2::zeros((n, rank));

    for i in 0..m {
        for j in 0..rank {
            u_factor[[i, j]] = F::from(normal.sample(&mut rng)).unwrap_or(F::zero());
        }
    }
    for i in 0..n {
        for j in 0..rank {
            v_factor[[i, j]] = F::from(normal.sample(&mut rng)).unwrap_or(F::zero());
        }
    }

    // Build row-indexed and col-indexed observation maps for fast lookup
    let mut row_obs: Vec<Vec<(usize, F)>> = vec![Vec::new(); m]; // row -> [(col, val)]
    let mut col_obs: Vec<Vec<(usize, F)>> = vec![Vec::new(); n]; // col -> [(row, val)]
    for &(i, j) in &mask.observed {
        row_obs[i].push((j, observed_values[[i, j]]));
        col_obs[j].push((i, observed_values[[i, j]]));
    }

    let mut converged = false;
    let mut last_residual = F::infinity();
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Update U: for each row i, solve least squares
        for i in 0..m {
            if row_obs[i].is_empty() {
                continue;
            }
            let n_obs = row_obs[i].len();
            // Build V_Omega_i (n_obs x rank) and b_i (n_obs)
            let mut v_sub = Array2::zeros((n_obs, rank));
            let mut b_vec = Array1::zeros(n_obs);
            for (idx, &(j, val)) in row_obs[i].iter().enumerate() {
                for kk in 0..rank {
                    v_sub[[idx, kk]] = v_factor[[j, kk]];
                }
                b_vec[idx] = val;
            }

            // Solve (V_sub^T * V_sub + lambda * I) * u_i = V_sub^T * b
            let vt_v = v_sub.t().dot(&v_sub);
            let vt_b = v_sub.t().dot(&b_vec);

            let mut gram = vt_v;
            for kk in 0..rank {
                gram[[kk, kk]] += lambda;
            }

            // Solve via Cholesky-like approach (small system)
            if let Ok(sol) = solve_small_system(&gram.view(), &vt_b) {
                for kk in 0..rank {
                    u_factor[[i, kk]] = sol[kk];
                }
            }
        }

        // Update V: for each column j, solve least squares
        for j in 0..n {
            if col_obs[j].is_empty() {
                continue;
            }
            let n_obs = col_obs[j].len();
            let mut u_sub = Array2::zeros((n_obs, rank));
            let mut b_vec = Array1::zeros(n_obs);
            for (idx, &(i, val)) in col_obs[j].iter().enumerate() {
                for kk in 0..rank {
                    u_sub[[idx, kk]] = u_factor[[i, kk]];
                }
                b_vec[idx] = val;
            }

            let ut_u = u_sub.t().dot(&u_sub);
            let ut_b = u_sub.t().dot(&b_vec);

            let mut gram = ut_u;
            for kk in 0..rank {
                gram[[kk, kk]] += lambda;
            }

            if let Ok(sol) = solve_small_system(&gram.view(), &ut_b) {
                for kk in 0..rank {
                    v_factor[[j, kk]] = sol[kk];
                }
            }
        }

        // Compute residual
        let mut residual = F::zero();
        let mut count = F::zero();
        for &(i, j) in &mask.observed {
            let mut pred = F::zero();
            for kk in 0..rank {
                pred += u_factor[[i, kk]] * v_factor[[j, kk]];
            }
            let diff = observed_values[[i, j]] - pred;
            residual += diff * diff;
            count += F::one();
        }
        residual = if count > F::zero() {
            (residual / count).sqrt()
        } else {
            F::zero()
        };

        let rel_change = if last_residual > F::epsilon() {
            (last_residual - residual).abs() / last_residual
        } else {
            F::zero()
        };

        if rel_change < config.tolerance && iter > 0 {
            converged = true;
            last_residual = residual;
            break;
        }
        last_residual = residual;
    }

    // Construct completed matrix X = U * V^T
    let matrix = u_factor.dot(&v_factor.t());

    Ok(CompletionResult {
        matrix,
        iterations,
        residual: last_residual,
        converged,
    })
}

/// Solve a small dense linear system Ax = b via LU-like elimination.
fn solve_small_system<F>(a: &ArrayView2<F>, b: &Array1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + scirs2_core::ndarray::ScalarOperand + Send + Sync + 'static,
{
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(LinalgError::DimensionError(
            "System dimensions mismatch".to_string(),
        ));
    }

    // Gaussian elimination with partial pivoting
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        // Find pivot
        let mut max_val = F::zero();
        let mut max_row = col;
        for row in col..n {
            let abs_val = aug[[row, col]].abs();
            if abs_val > max_val {
                max_val = abs_val;
                max_row = row;
            }
        }

        if max_val < F::epsilon() * F::from(100.0).unwrap_or(F::one()) {
            // Nearly singular: use regularization
            aug[[col, col]] += F::epsilon() * F::from(1000.0).unwrap_or(F::one());
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        if pivot.abs() < F::epsilon() {
            continue;
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let val = aug[[col, j]];
                aug[[row, j]] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        let diag = aug[[i, i]];
        x[i] = if diag.abs() > F::epsilon() {
            sum / diag
        } else {
            F::zero()
        };
    }

    Ok(x)
}

// ============================================================================
// Soft-Impute Algorithm
// ============================================================================

/// Matrix completion via Soft-Impute algorithm.
///
/// Iteratively replaces missing values with SVD-based predictions,
/// applying nuclear norm regularization via soft-thresholding.
///
/// At each step:
/// 1. Fill in missing entries with current predictions
/// 2. Compute SVT of the filled matrix
/// 3. Repeat until convergence
///
/// # Arguments
///
/// * `observed_values` - Matrix with observed values
/// * `mask` - Observation mask
/// * `config` - Algorithm configuration
///
/// # Returns
///
/// * `CompletionResult` with completed matrix
///
/// # References
///
/// Mazumder, Hastie, Tibshirani (2010). "Spectral regularization algorithms
/// for learning large incomplete matrices."
pub fn soft_impute<F>(
    observed_values: &ArrayView2<F>,
    mask: &ObservationMask,
    config: &CompletionConfig<F>,
) -> LinalgResult<CompletionResult<F>>
where
    F: Float
        + NumAssign
        + Sum
        + Debug
        + scirs2_core::ndarray::ScalarOperand
        + Send
        + Sync
        + 'static,
{
    let (m, n) = observed_values.dim();
    if mask.nrows != m || mask.ncols != n {
        return Err(LinalgError::DimensionError(
            "Mask dimensions do not match matrix dimensions".to_string(),
        ));
    }

    let lambda = config.lambda;

    // Initialize X = 0
    let mut x = Array2::zeros((m, n));

    let mut converged = false;
    let mut last_residual = F::infinity();
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Fill: Z = P_Omega(M) + P_Omega_perp(X_old)
        // i.e., use observed values where available, current X elsewhere
        let mut z = x.clone();
        for &(i, j) in &mask.observed {
            z[[i, j]] = observed_values[[i, j]];
        }

        // SVT: X_new = D_lambda(Z)
        let x_new = singular_value_threshold(&z.view(), lambda)?;

        // Compute convergence criterion
        let mut change = F::zero();
        let mut norm_x = F::zero();
        for i in 0..m {
            for j in 0..n {
                let diff = x_new[[i, j]] - x[[i, j]];
                change += diff * diff;
                norm_x += x_new[[i, j]] * x_new[[i, j]];
            }
        }
        let rel_change = if norm_x > F::epsilon() {
            change.sqrt() / norm_x.sqrt()
        } else {
            change.sqrt()
        };

        x = x_new;

        // Compute residual on observed entries
        let mut residual = F::zero();
        let mut count = F::zero();
        for &(i, j) in &mask.observed {
            let diff = observed_values[[i, j]] - x[[i, j]];
            residual += diff * diff;
            count += F::one();
        }
        residual = if count > F::zero() {
            (residual / count).sqrt()
        } else {
            F::zero()
        };

        if rel_change < config.tolerance && iter > 0 {
            converged = true;
            last_residual = residual;
            break;
        }
        last_residual = residual;
    }

    Ok(CompletionResult {
        matrix: x,
        iterations,
        residual: last_residual,
        converged,
    })
}

// ============================================================================
// Utility: Fill missing values
// ============================================================================

/// Fill missing (NaN) values with a specified strategy.
///
/// # Arguments
///
/// * `matrix` - Matrix potentially containing NaN values
/// * `strategy` - Fill strategy: "zero", "mean", "median", "row_mean", "col_mean"
///
/// # Returns
///
/// * Matrix with NaN values replaced
pub fn fill_missing<F>(matrix: &ArrayView2<F>, strategy: &str) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Debug + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = matrix.dim();
    let mut result = matrix.to_owned();

    match strategy {
        "zero" => {
            for i in 0..m {
                for j in 0..n {
                    if result[[i, j]].is_nan() {
                        result[[i, j]] = F::zero();
                    }
                }
            }
        }
        "mean" => {
            // Global mean of non-NaN entries
            let mut sum = F::zero();
            let mut count = F::zero();
            for &val in matrix.iter() {
                if !val.is_nan() {
                    sum += val;
                    count += F::one();
                }
            }
            let mean = if count > F::zero() {
                sum / count
            } else {
                F::zero()
            };
            for i in 0..m {
                for j in 0..n {
                    if result[[i, j]].is_nan() {
                        result[[i, j]] = mean;
                    }
                }
            }
        }
        "row_mean" => {
            for i in 0..m {
                let mut sum = F::zero();
                let mut count = F::zero();
                for j in 0..n {
                    if !matrix[[i, j]].is_nan() {
                        sum += matrix[[i, j]];
                        count += F::one();
                    }
                }
                let row_mean = if count > F::zero() {
                    sum / count
                } else {
                    F::zero()
                };
                for j in 0..n {
                    if result[[i, j]].is_nan() {
                        result[[i, j]] = row_mean;
                    }
                }
            }
        }
        "col_mean" => {
            for j in 0..n {
                let mut sum = F::zero();
                let mut count = F::zero();
                for i in 0..m {
                    if !matrix[[i, j]].is_nan() {
                        sum += matrix[[i, j]];
                        count += F::one();
                    }
                }
                let col_mean = if count > F::zero() {
                    sum / count
                } else {
                    F::zero()
                };
                for i in 0..m {
                    if result[[i, j]].is_nan() {
                        result[[i, j]] = col_mean;
                    }
                }
            }
        }
        _ => {
            return Err(LinalgError::InvalidInput(format!(
                "Unknown fill strategy: '{strategy}'. Use 'zero', 'mean', 'row_mean', or 'col_mean'"
            )));
        }
    }

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_low_rank_observed(
        m: usize,
        n: usize,
        rank: usize,
        obs_fraction: f64,
    ) -> (Array2<f64>, ObservationMask) {
        let mut rng = scirs2_core::random::rng();
        let normal =
            Normal::new(0.0, 1.0).unwrap_or_else(|_| panic!("Failed to create distribution"));

        // Generate low-rank matrix
        let mut u_gen = Array2::zeros((m, rank));
        let mut v_gen = Array2::zeros((n, rank));
        for i in 0..m {
            for j in 0..rank {
                u_gen[[i, j]] = normal.sample(&mut rng);
            }
        }
        for i in 0..n {
            for j in 0..rank {
                v_gen[[i, j]] = normal.sample(&mut rng);
            }
        }
        let full_matrix = u_gen.dot(&v_gen.t());

        // Random observation mask
        let mut observed = Vec::new();
        for i in 0..m {
            for j in 0..n {
                let r: f64 = rng.random();
                if r < obs_fraction {
                    observed.push((i, j));
                }
            }
        }

        // Ensure at least some observations per row and column
        for i in 0..m {
            let has_obs = observed.iter().any(|&(r, _)| r == i);
            if !has_obs {
                let j: usize = rng.random_range(0..n);
                observed.push((i, j));
            }
        }
        for j in 0..n {
            let has_obs = observed.iter().any(|&(_, c)| c == j);
            if !has_obs {
                let i: usize = rng.random_range(0..m);
                observed.push((i, j));
            }
        }

        let mask = ObservationMask::from_indices(observed, m, n);
        (full_matrix, mask)
    }

    #[test]
    fn test_observation_mask_from_bool() {
        let mask_arr = array![[true, false, true], [false, true, false]];
        let mask = ObservationMask::from_bool_matrix(&mask_arr.view());
        assert_eq!(mask.nrows, 2);
        assert_eq!(mask.ncols, 3);
        assert_eq!(mask.observed.len(), 3);
        assert!(mask.is_observed(0, 0));
        assert!(!mask.is_observed(0, 1));
    }

    #[test]
    fn test_observation_mask_from_nan() {
        let mat = array![[1.0, f64::NAN, 3.0], [f64::NAN, 5.0, f64::NAN]];
        let mask = ObservationMask::from_nan_matrix(&mat.view());
        assert_eq!(mask.observed.len(), 3);
        assert!(mask.is_observed(0, 0));
        assert!(!mask.is_observed(0, 1));
    }

    #[test]
    fn test_observation_ratio() {
        let mask = ObservationMask::from_indices(vec![(0, 0), (1, 1)], 3, 3);
        let ratio = mask.observation_ratio();
        assert!((ratio - 2.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_svt_basic() {
        let a = array![[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]];
        let result = singular_value_threshold(&a.view(), 1.0);
        assert!(result.is_ok());
        let thresholded = result.expect("SVT failed");

        // SVD of a: singular values are 3 and 2
        // After thresholding by 1: 2 and 1
        // So Frobenius norm should be sqrt(4 + 1) = sqrt(5)
        let frob_sq: f64 = thresholded.iter().map(|&x| x * x).sum();
        assert!(
            (frob_sq - 5.0).abs() < 0.5,
            "Frobenius norm squared should be ~5, got {frob_sq}"
        );
    }

    #[test]
    fn test_svt_full_threshold() {
        let a = array![[1.0, 0.0], [0.0, 0.5]];
        // Threshold larger than all singular values
        let result = singular_value_threshold(&a.view(), 2.0);
        assert!(result.is_ok());
        let thresholded = result.expect("SVT failed");

        // Should be all zeros
        for &val in thresholded.iter() {
            assert!(val.abs() < 1e-10, "Should be zero after full threshold");
        }
    }

    #[test]
    fn test_svt_completion_simple() {
        let (full_mat, mask) = make_low_rank_observed(8, 6, 2, 0.8);

        let config = CompletionConfig::new(0.1)
            .with_max_iter(100)
            .with_tolerance(1e-4);

        let result = svt_completion(&full_mat.view(), &mask, &config);
        assert!(result.is_ok());
        let comp = result.expect("SVT completion failed");

        assert_eq!(comp.matrix.nrows(), 8);
        assert_eq!(comp.matrix.ncols(), 6);
        assert!(comp.iterations > 0);
    }

    #[test]
    fn test_nuclear_norm_completion() {
        let (full_mat, mask) = make_low_rank_observed(8, 6, 2, 0.8);

        let config = CompletionConfig::new(0.01)
            .with_max_iter(50)
            .with_tolerance(1e-4);

        let result = nuclear_norm_completion(&full_mat.view(), &mask, &config);
        assert!(result.is_ok());
        let comp = result.expect("Nuclear norm completion failed");

        assert_eq!(comp.matrix.nrows(), 8);
        assert_eq!(comp.matrix.ncols(), 6);
    }

    #[test]
    fn test_als_completion_basic() {
        let (full_mat, mask) = make_low_rank_observed(10, 8, 2, 0.7);

        let config = CompletionConfig::new(0.01)
            .with_max_iter(100)
            .with_tolerance(1e-4)
            .with_rank(2);

        let result = als_completion(&full_mat.view(), &mask, &config);
        assert!(result.is_ok());
        let comp = result.expect("ALS completion failed");

        assert_eq!(comp.matrix.nrows(), 10);
        assert_eq!(comp.matrix.ncols(), 8);
    }

    #[test]
    fn test_als_requires_rank() {
        let (full_mat, mask) = make_low_rank_observed(5, 5, 2, 0.8);
        let config = CompletionConfig::new(0.01); // No rank set
        assert!(als_completion(&full_mat.view(), &mask, &config).is_err());
    }

    #[test]
    fn test_als_invalid_rank() {
        let (full_mat, mask) = make_low_rank_observed(5, 5, 2, 0.8);
        let config = CompletionConfig::new(0.01).with_rank(0);
        assert!(als_completion(&full_mat.view(), &mask, &config).is_err());

        let config2 = CompletionConfig::new(0.01).with_rank(100);
        assert!(als_completion(&full_mat.view(), &mask, &config2).is_err());
    }

    #[test]
    fn test_soft_impute_basic() {
        let (full_mat, mask) = make_low_rank_observed(8, 6, 2, 0.8);

        let config = CompletionConfig::new(0.05)
            .with_max_iter(50)
            .with_tolerance(1e-4);

        let result = soft_impute(&full_mat.view(), &mask, &config);
        assert!(result.is_ok());
        let comp = result.expect("Soft-Impute failed");

        assert_eq!(comp.matrix.nrows(), 8);
        assert_eq!(comp.matrix.ncols(), 6);
    }

    #[test]
    fn test_soft_impute_observed_entries_fit() {
        let (full_mat, mask) = make_low_rank_observed(6, 5, 1, 0.9);

        let config = CompletionConfig::new(0.001)
            .with_max_iter(200)
            .with_tolerance(1e-6);

        let comp = soft_impute(&full_mat.view(), &mask, &config).expect("Soft-Impute failed");

        // Check that observed entries are close to original
        let mut max_obs_err = 0.0_f64;
        for &(i, j) in &mask.observed {
            let err = (full_mat[[i, j]] - comp.matrix[[i, j]]).abs();
            if err > max_obs_err {
                max_obs_err = err;
            }
        }
        // With low lambda, observed entries should be well-approximated
        // (not exact due to regularization)
        assert!(
            max_obs_err < 5.0,
            "Max observed error too large: {max_obs_err}"
        );
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let mat = array![[1.0, 2.0], [3.0, 4.0]];
        let bad_mask = ObservationMask::from_indices(vec![(0, 0)], 3, 3); // Wrong dims

        let config = CompletionConfig::new(0.1);
        assert!(svt_completion(&mat.view(), &bad_mask, &config).is_err());
        assert!(nuclear_norm_completion(&mat.view(), &bad_mask, &config).is_err());
        assert!(soft_impute(&mat.view(), &bad_mask, &config).is_err());
        assert!(als_completion(&mat.view(), &bad_mask, &config.clone().with_rank(1)).is_err());
    }

    #[test]
    fn test_fill_missing_zero() {
        let mat = array![[1.0, f64::NAN], [f64::NAN, 4.0]];
        let filled = fill_missing(&mat.view(), "zero").expect("fill zero failed");
        assert_eq!(filled[[0, 0]], 1.0);
        assert_eq!(filled[[0, 1]], 0.0);
        assert_eq!(filled[[1, 0]], 0.0);
        assert_eq!(filled[[1, 1]], 4.0);
    }

    #[test]
    fn test_fill_missing_mean() {
        let mat = array![[1.0, f64::NAN], [f64::NAN, 3.0]];
        let filled = fill_missing(&mat.view(), "mean").expect("fill mean failed");
        // Mean of observed = (1 + 3) / 2 = 2.0
        assert!((filled[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((filled[[1, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_fill_missing_row_mean() {
        let mat = array![[1.0, f64::NAN, 3.0], [4.0, 5.0, f64::NAN]];
        let filled = fill_missing(&mat.view(), "row_mean").expect("fill row_mean failed");
        // Row 0 mean = (1 + 3) / 2 = 2.0
        assert!((filled[[0, 1]] - 2.0).abs() < 1e-10);
        // Row 1 mean = (4 + 5) / 2 = 4.5
        assert!((filled[[1, 2]] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_fill_missing_col_mean() {
        let mat = array![[1.0, f64::NAN], [3.0, 4.0], [f64::NAN, 6.0]];
        let filled = fill_missing(&mat.view(), "col_mean").expect("fill col_mean failed");
        // Col 0 mean = (1 + 3) / 2 = 2.0
        assert!((filled[[2, 0]] - 2.0).abs() < 1e-10);
        // Col 1 mean = (4 + 6) / 2 = 5.0
        assert!((filled[[0, 1]] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fill_missing_invalid_strategy() {
        let mat = array![[1.0, f64::NAN]];
        assert!(fill_missing(&mat.view(), "invalid").is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = CompletionConfig::new(0.5_f64)
            .with_max_iter(500)
            .with_tolerance(1e-8)
            .with_rank(3)
            .with_step_size(0.1);

        assert_eq!(config.max_iter, 500);
        assert!((config.tolerance - 1e-8).abs() < 1e-15);
        assert_eq!(config.rank, Some(3));
        assert!((config.step_size.expect("step") - 0.1).abs() < 1e-15);
    }

    #[test]
    fn test_solve_small_system() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![5.0, 4.0];
        let x = solve_small_system(&a.view(), &b).expect("solve failed");

        // Verify: Ax should be close to b
        let ax = a.dot(&x);
        assert!((ax[0] - 5.0).abs() < 1e-6);
        assert!((ax[1] - 4.0).abs() < 1e-6);
    }
}
