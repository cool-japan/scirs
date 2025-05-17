// Sparse signal recovery module
//
// This module implements various sparse signal recovery techniques including compressed sensing,
// L1/L0 regularization, basis pursuit, matching pursuit, and orthogonal matching pursuit.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::Distribution;
use rustfft::{num_complex::Complex, FftPlanner};
use scirs2_linalg::{solve, vector_norm};
use std::cmp::min;

/// Configuration for sparse signal recovery algorithms
#[derive(Debug, Clone)]
pub struct SparseRecoveryConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold for iterative methods
    pub convergence_threshold: f64,
    /// Regularization parameter for L1 methods
    pub lambda: f64,
    /// Desired sparsity level (number of non-zero coefficients)
    pub sparsity: Option<usize>,
    /// Desired reconstruction error
    pub target_error: Option<f64>,
    /// Whether to use non-negative constraints
    pub non_negative: bool,
    /// Whether to use warm start with previous solution
    pub warm_start: bool,
    /// Whether to use acceleration techniques
    pub accelerate: bool,
    /// Random initialization seed
    pub random_seed: Option<u64>,
    /// Tolerance for numerical stability
    pub eps: f64,
}

impl Default for SparseRecoveryConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            lambda: 0.1,
            sparsity: None,
            target_error: None,
            non_negative: false,
            warm_start: false,
            accelerate: true,
            random_seed: None,
            eps: 1e-8,
        }
    }
}

/// Sparse recovery methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparseRecoveryMethod {
    /// Orthogonal Matching Pursuit (OMP)
    OMP,
    /// Matching Pursuit (MP)
    MP,
    /// Basis Pursuit (BP)
    BasisPursuit,
    /// Iterative Soft Thresholding (ISTA)
    ISTA,
    /// Fast Iterative Soft Thresholding (FISTA)
    FISTA,
    /// Least Absolute Shrinkage and Selection Operator (LASSO)
    LASSO,
    /// Iterative Hard Thresholding (IHT)
    IHT,
    /// Compressive Sampling Matching Pursuit (CoSaMP)
    CoSaMP,
    /// Subspace Pursuit
    SubspacePursuit,
    /// Smoothed L0 (SL0)
    SmoothL0,
}

/// Performs sparse signal recovery using Orthogonal Matching Pursuit (OMP)
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn omp(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();

    // Determine sparsity level (K)
    let k = match config.sparsity {
        Some(s) => s,
        None => {
            // Default to m/4 if not specified
            min(m / 4, n.saturating_sub(1))
        }
    };

    // Store active column indices
    let mut active_set = Vec::with_capacity(k);

    // Initialize residual
    let mut residual = y.clone();

    // Initialize recovered signal
    let mut x = Array1::<f64>::zeros(n);

    // Store phi_active (dictionary restricted to active set)
    let mut phi_active = Array2::<f64>::zeros((m, 0));

    // OMP iterations
    for _ in 0..min(k, config.max_iterations) {
        // Compute correlations between residual and dictionary columns
        let mut max_correlation = 0.0;
        let mut best_idx = 0;

        for j in 0..n {
            // Skip columns already in active set
            if active_set.contains(&j) {
                continue;
            }

            let column = phi.slice(s![.., j]);
            let correlation = column.dot(&residual).abs();

            if correlation > max_correlation {
                max_correlation = correlation;
                best_idx = j;
            }
        }

        // Add best matching column to active set
        active_set.push(best_idx);

        // Create new phi_active by adding the new column
        let mut new_phi_active = Array2::<f64>::zeros((m, active_set.len()));
        for (i, &idx) in active_set.iter().enumerate() {
            let column = phi.slice(s![.., idx]);
            new_phi_active.slice_mut(s![.., i]).assign(&column);
        }
        phi_active = new_phi_active;

        // Solve least squares problem for active columns
        let coefficients = match solve(
            &phi_active.t().dot(&phi_active).view(),
            &phi_active.t().dot(y).view(),
        ) {
            Ok(coef) => coef,
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to solve least squares in OMP".to_string(),
                ));
            }
        };

        // Update residual
        residual = y - &phi_active.dot(&coefficients);

        // Check convergence
        let res_norm = vector_norm(&residual.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        if res_norm < config.convergence_threshold || res_norm < config.eps {
            break;
        }

        // Check if target error is achieved
        if let Some(target) = config.target_error {
            if res_norm <= target {
                break;
            }
        }
    }

    // Solve least squares once more to get final coefficients
    let coefficients = match solve(
        &phi_active.t().dot(&phi_active).view(),
        &phi_active.t().dot(y).view(),
    ) {
        Ok(coef) => coef,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve final least squares in OMP".to_string(),
            ));
        }
    };

    // Fill recovered signal with coefficients at active indices
    for (i, &idx) in active_set.iter().enumerate() {
        x[idx] = coefficients[i];
    }

    // Apply non-negative constraint if requested
    if config.non_negative {
        x.mapv_inplace(|val| val.max(0.0));
    }

    Ok(x)
}

/// Performs sparse signal recovery using Matching Pursuit (MP)
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn mp(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();

    // Determine sparsity level (K)
    let k = match config.sparsity {
        Some(s) => s,
        None => {
            // Default to m/3 if not specified
            min(m / 3, n.saturating_sub(1))
        }
    };

    // Initialize residual
    let mut residual = y.clone();

    // Initialize recovered signal
    let mut x = Array1::<f64>::zeros(n);

    // Normalize dictionary columns for more stable computation
    let mut normalized_phi = phi.clone();
    for j in 0..n {
        let mut col = normalized_phi.slice_mut(s![.., j]);
        let norm = vector_norm(&col.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        if norm > config.eps {
            col.mapv_inplace(|val| val / norm);
        }
    }

    // Count the number of entries selected so far
    let mut selected_count = 0;

    // MP iterations
    for _ in 0..config.max_iterations {
        // Compute correlations between residual and dictionary columns
        let mut max_correlation = 0.0;
        let mut best_idx = 0;

        for j in 0..n {
            let column = normalized_phi.slice(s![.., j]);
            let correlation = column.dot(&residual).abs();

            if correlation > max_correlation {
                max_correlation = correlation;
                best_idx = j;
            }
        }

        if max_correlation < config.eps {
            break;
        }

        // Best matching column and its coefficient
        let best_column = normalized_phi.slice(s![.., best_idx]);
        let coefficient = best_column.dot(&residual);

        // Update solution
        x[best_idx] += coefficient;

        // Update residual
        residual = &residual - coefficient * &best_column;

        // Count non-zero coefficients
        if (x[best_idx]).abs() > config.eps {
            selected_count += 1;
        }

        // Check if we've reached the desired sparsity
        if selected_count >= k {
            break;
        }

        // Check convergence
        let res_norm = vector_norm(&residual.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        if res_norm < config.convergence_threshold || res_norm < config.eps {
            break;
        }

        // Check if target error is achieved
        if let Some(target) = config.target_error {
            if res_norm <= target {
                break;
            }
        }
    }

    // Apply non-negative constraint if requested
    if config.non_negative {
        x.mapv_inplace(|val| val.max(0.0));
    }

    Ok(x)
}

/// Performs sparse signal recovery using Iterative Soft Thresholding Algorithm (ISTA)
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn ista(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let (_m, n) = phi.dim();

    // Compute step size (based on max eigenvalue of phi^T * phi)
    // For efficiency, we use an approximation: 1 / (largest singular value of phi)^2
    let phi_norm = (0..n)
        .map(|j| vector_norm(&phi.slice(s![.., j]).view(), 2).unwrap_or(0.0))
        .fold(0.0, |a: f64, b: f64| a.max(b));

    let step_size = 1.0 / (phi_norm * phi_norm);

    // Initialize solution
    let mut x = Array1::<f64>::zeros(n);
    let mut x_prev = Array1::<f64>::zeros(n);

    // Pre-compute phi^T
    let phi_t = phi.t();

    // ISTA iterations
    for _ in 0..config.max_iterations {
        // Store previous solution
        x_prev.assign(&x);

        // Gradient step: x = x - step_size * phi^T * (phi * x - y)
        let residual = phi.dot(&x) - y;
        let gradient = phi_t.dot(&residual);
        x = &x - step_size * &gradient;

        // Soft thresholding: shrinkage operator
        let threshold = config.lambda * step_size;
        x.mapv_inplace(|val| {
            if val.abs() <= threshold {
                0.0
            } else {
                val.signum() * (val.abs() - threshold)
            }
        });

        // Apply non-negative constraint if requested
        if config.non_negative {
            x.mapv_inplace(|val| val.max(0.0));
        }

        // Check convergence
        let x_diff_norm = vector_norm(&(&x - &x_prev).view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        let x_norm = vector_norm(&x.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        let diff = x_diff_norm / x_norm.max(config.eps);
        if diff < config.convergence_threshold {
            break;
        }
    }

    Ok(x)
}

/// Performs sparse signal recovery using Fast Iterative Soft Thresholding Algorithm (FISTA)
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn fista(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let (_m, n) = phi.dim();

    // Compute step size (based on max eigenvalue of phi^T * phi)
    // For efficiency, we use an approximation: 1 / (largest singular value of phi)^2
    let phi_norm = (0..n)
        .map(|j| vector_norm(&phi.slice(s![.., j]).view(), 2).unwrap_or(0.0))
        .fold(0.0, |a: f64, b: f64| a.max(b));

    let step_size = 1.0 / (phi_norm * phi_norm);

    // Initialize solution
    let mut x = Array1::<f64>::zeros(n);
    let mut x_prev = x.clone();
    let mut z = x.clone();
    let mut t = 1.0;
    let mut t_next;

    // Pre-compute phi^T
    let phi_t = phi.t();

    // FISTA iterations
    for _ in 0..config.max_iterations {
        // Store previous solution
        x_prev.assign(&x);

        // Gradient step: grad = z - step_size * phi^T * (phi * z - y)
        let residual = phi.dot(&z) - y;
        let gradient = phi_t.dot(&residual);
        let grad_step = &z - step_size * &gradient;

        // Soft thresholding: shrinkage operator
        let threshold = config.lambda * step_size;
        x = grad_step.mapv(|val| {
            if val.abs() <= threshold {
                0.0
            } else {
                val.signum() * (val.abs() - threshold)
            }
        });

        // Apply non-negative constraint if requested
        if config.non_negative {
            x.mapv_inplace(|val| val.max(0.0));
        }

        // Update momentum parameters
        t_next = (1.0 + f64::sqrt(1.0 + 4.0 * t * t)) / 2.0;

        // Update extrapolation point
        z = &x + ((t - 1.0) / t_next) * (&x - &x_prev);

        // Update t
        t = t_next;

        // Check convergence
        let x_diff_norm = vector_norm(&(&x - &x_prev).view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        let x_norm = vector_norm(&x.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        let diff = x_diff_norm / x_norm.max(config.eps);
        if diff < config.convergence_threshold {
            break;
        }
    }

    Ok(x)
}

/// Performs sparse signal recovery using Compressive Sampling Matching Pursuit (CoSaMP)
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn cosamp(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();

    // Determine sparsity level (K)
    let k = match config.sparsity {
        Some(s) => s,
        None => {
            // Default to m/4 if not specified
            min(m / 4, n.saturating_sub(1))
        }
    };

    // Initialize solution
    let mut x = Array1::<f64>::zeros(n);
    let mut x_prev = Array1::<f64>::zeros(n);

    // Initialize residual
    let mut residual = y.clone();

    // Pre-compute phi^T
    let phi_t = phi.t();

    // CoSaMP iterations
    for _ in 0..config.max_iterations {
        // Store previous solution
        x_prev.assign(&x);

        // Compute signal proxy
        let proxy = phi_t.dot(&residual);

        // Find 2K largest entries in the proxy
        let mut proxy_values: Vec<(usize, f64)> = proxy
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        // Sort by magnitude in descending order
        proxy_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Get the indices of the 2K largest entries
        let selected_indices: Vec<usize> =
            proxy_values.iter().take(2 * k).map(|&(i, _)| i).collect();

        // Merge with current support (non-zero indices in x)
        let mut support = Vec::new();
        for i in 0..n {
            if x[i].abs() > config.eps {
                support.push(i);
            }
        }

        // Add new selected indices to support
        for &idx in &selected_indices {
            if !support.contains(&idx) {
                support.push(idx);
            }
        }

        // Create restricted sensing matrix for the merged support
        let mut phi_restricted = Array2::<f64>::zeros((m, support.len()));
        for (i, &idx) in support.iter().enumerate() {
            let column = phi.slice(s![.., idx]);
            phi_restricted.slice_mut(s![.., i]).assign(&column);
        }

        // Solve least squares problem
        let coefficients = match solve(
            &phi_restricted.t().dot(&phi_restricted).view(),
            &phi_restricted.t().dot(y).view(),
        ) {
            Ok(coef) => coef,
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to solve least squares in CoSaMP".to_string(),
                ));
            }
        };

        // Create temporary solution
        let mut x_temp = Array1::<f64>::zeros(n);
        for (i, &idx) in support.iter().enumerate() {
            x_temp[idx] = coefficients[i];
        }

        // Prune to K largest entries
        let mut temp_values: Vec<(usize, f64)> = x_temp
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        // Sort by magnitude in descending order
        temp_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Create new solution with K largest entries
        let mut x_new = Array1::<f64>::zeros(n);
        for &(i, _) in temp_values.iter().take(k) {
            x_new[i] = x_temp[i];
        }

        // Update solution
        x = x_new;

        // Update residual
        residual = y - &phi.dot(&x);

        // Check convergence
        let x_diff_norm = vector_norm(&(&x - &x_prev).view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        let x_norm = vector_norm(&x.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        let diff = x_diff_norm / x_norm.max(config.eps);
        if diff < config.convergence_threshold {
            break;
        }

        // Check residual
        let res_norm = vector_norm(&residual.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        if res_norm < config.convergence_threshold || res_norm < config.eps {
            break;
        }

        // Check if target error is achieved
        if let Some(target) = config.target_error {
            if res_norm <= target {
                break;
            }
        }
    }

    // Apply non-negative constraint if requested
    if config.non_negative {
        x.mapv_inplace(|val| val.max(0.0));
    }

    Ok(x)
}

/// Performs sparse signal recovery using Iterative Hard Thresholding (IHT)
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn iht(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();

    // Determine sparsity level (K)
    let k = match config.sparsity {
        Some(s) => s,
        None => {
            // Default to m/3 if not specified
            min(m / 3, n.saturating_sub(1))
        }
    };

    // Compute step size (based on max eigenvalue of phi^T * phi)
    // For efficiency, we use an approximation: 1 / (largest singular value of phi)^2
    let phi_norm = (0..n)
        .map(|j| vector_norm(&phi.slice(s![.., j]).view(), 2).unwrap_or(0.0))
        .fold(0.0, |a: f64, b: f64| a.max(b));

    let step_size = 0.9 / (phi_norm * phi_norm); // slightly smaller for stability

    // Initialize solution
    let mut x = Array1::<f64>::zeros(n);
    let mut x_prev = Array1::<f64>::zeros(n);

    // Pre-compute phi^T
    let phi_t = phi.t();

    // IHT iterations
    for _ in 0..config.max_iterations {
        // Store previous solution
        x_prev.assign(&x);

        // Gradient step: x = x - step_size * phi^T * (phi * x - y)
        let residual = phi.dot(&x) - y;
        let gradient = phi_t.dot(&residual);
        let x_grad = &x - step_size * &gradient;

        // Apply non-negative constraint if requested
        let x_grad = if config.non_negative {
            x_grad.mapv(|val| val.max(0.0))
        } else {
            x_grad
        };

        // Hard thresholding: keep K largest entries
        let mut values: Vec<(usize, f64)> = x_grad
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        // Sort by magnitude in descending order
        values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Create new solution with K largest entries
        let mut x_new = Array1::<f64>::zeros(n);
        for &(i, _) in values.iter().take(k) {
            x_new[i] = x_grad[i];
        }

        // Update solution
        x = x_new;

        // Check convergence
        let x_diff_norm = vector_norm(&(&x - &x_prev).view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        let x_norm = vector_norm(&x.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        let diff = x_diff_norm / x_norm.max(config.eps);
        if diff < config.convergence_threshold {
            break;
        }

        // Check if target error is achieved
        if let Some(target) = config.target_error {
            let err_vec = phi.dot(&x) - y;
            let err = vector_norm(&err_vec.view(), 2)
                .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
            if err <= target {
                break;
            }
        }
    }

    Ok(x)
}

/// Performs sparse signal recovery using Subspace Pursuit
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn subspace_pursuit(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let (m, n) = phi.dim();

    // Determine sparsity level (K)
    let k = match config.sparsity {
        Some(s) => s,
        None => {
            // Default to m/3 if not specified
            min(m / 3, n.saturating_sub(1))
        }
    };

    // Initialize solution
    let _x = Array1::<f64>::zeros(n);

    // Initialize residual
    let mut residual = y.clone();

    // Pre-compute phi^T
    let phi_t = phi.t();

    // Initial support estimation
    let initial_correlation = phi_t.dot(&residual);
    let mut initial_values: Vec<(usize, f64)> = initial_correlation
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();

    // Sort by magnitude in descending order
    initial_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Initialize support with K largest correlations
    let mut support: Vec<usize> = initial_values.iter().take(k).map(|&(i, _)| i).collect();

    // Subspace Pursuit iterations
    for _ in 0..config.max_iterations {
        // Solve least squares problem on the current support
        let mut phi_support = Array2::<f64>::zeros((m, support.len()));
        for (i, &idx) in support.iter().enumerate() {
            let column = phi.slice(s![.., idx]);
            phi_support.slice_mut(s![.., i]).assign(&column);
        }

        let signal_proxy = match solve(
            &phi_support.t().dot(&phi_support).view(),
            &phi_support.t().dot(y).view(),
        ) {
            Ok(proxy) => proxy,
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to solve least squares in Subspace Pursuit".to_string(),
                ));
            }
        };

        // Compute new approximation and residual
        let y_approx = phi_support.dot(&signal_proxy);
        residual = y - &y_approx;

        // Find K indices with largest correlation with residual
        let correlation = phi_t.dot(&residual);
        let mut corr_values: Vec<(usize, f64)> = correlation
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();

        // Sort by magnitude in descending order
        corr_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Get the indices of the K largest correlations
        let new_candidates: Vec<usize> = corr_values.iter().take(k).map(|&(i, _)| i).collect();

        // Merge supports
        let mut merged_support = support.clone();
        for &idx in &new_candidates {
            if !merged_support.contains(&idx) {
                merged_support.push(idx);
            }
        }

        // Solve least squares on merged support
        let mut phi_merged = Array2::<f64>::zeros((m, merged_support.len()));
        for (i, &idx) in merged_support.iter().enumerate() {
            let column = phi.slice(s![.., idx]);
            phi_merged.slice_mut(s![.., i]).assign(&column);
        }

        let merged_proxy = match solve(
            &phi_merged.t().dot(&phi_merged).view(),
            &phi_merged.t().dot(y).view(),
        ) {
            Ok(proxy) => proxy,
            Err(_) => {
                return Err(SignalError::Compute(
                    "Failed to solve least squares on merged support in Subspace Pursuit"
                        .to_string(),
                ));
            }
        };

        // Find K indices with largest coefficients in the merged solution
        let mut merged_values: Vec<(usize, f64)> = merged_proxy
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, merged_support[i], v.abs()))
            .map(|(_, idx, val)| (idx, val))
            .collect();

        // Sort by magnitude in descending order
        merged_values.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Update support with K largest coefficients
        let new_support: Vec<usize> = merged_values.iter().take(k).map(|&(i, _)| i).collect();

        // Check if support has changed
        if support == new_support {
            break;
        }

        // Update support
        support = new_support;

        // Check if target error is achieved
        if let Some(target) = config.target_error {
            let res_norm = vector_norm(&residual.view(), 2)
                .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
            if res_norm <= target {
                break;
            }
        }
    }

    // Solve final least squares problem on the support
    let mut phi_support = Array2::<f64>::zeros((m, support.len()));
    for (i, &idx) in support.iter().enumerate() {
        let column = phi.slice(s![.., idx]);
        phi_support.slice_mut(s![.., i]).assign(&column);
    }

    let coefficients = match solve(
        &phi_support.t().dot(&phi_support).view(),
        &phi_support.t().dot(y).view(),
    ) {
        Ok(coef) => coef,
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to solve final least squares in Subspace Pursuit".to_string(),
            ));
        }
    };

    // Fill recovered signal with coefficients at support indices
    let mut x_final = Array1::<f64>::zeros(n);
    for (i, &idx) in support.iter().enumerate() {
        x_final[idx] = coefficients[i];
    }

    // Apply non-negative constraint if requested
    if config.non_negative {
        x_final.mapv_inplace(|val| val.max(0.0));
    }

    Ok(x_final)
}

/// Performs sparse signal recovery using Smoothed L0 (SL0) algorithm
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn smooth_l0(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let (_m, _n) = phi.dim();

    // Initialize solution with minimum L2 norm solution
    let phi_t = phi.t();
    let gram = phi.dot(&phi_t);

    let x = match solve(&gram.view(), &y.view()) {
        Ok(solution) => phi_t.dot(&solution),
        Err(_) => {
            return Err(SignalError::Compute(
                "Failed to compute initial solution in SL0".to_string(),
            ));
        }
    };

    // Initialize sigma (decreasing sequence of smoothing parameters)
    let max_x = x.fold(0.0_f64, |a, &b| a.max(b.abs()));
    let mut sigma = 2.0 * max_x;
    let sigma_min = config.eps;
    let sigma_decrease_factor = 0.5;

    // Number of iterations for each sigma
    let inner_iterations = 3;

    // Step size for gradient descent
    let mu = 2.0;

    // Iterate until sigma is small enough
    let mut x_current = x;

    while sigma > sigma_min {
        // For each sigma, do gradient descent loop
        for _ in 0..inner_iterations {
            // Compute gradient of the smoothed L0 norm
            let grad =
                x_current.mapv(|val| -val * f64::exp(-0.5 * (val / sigma).powi(2)) / sigma.powi(2));

            // Take a step in the negative gradient direction
            let mut x_new = &x_current - mu * &grad;

            // Project back to the feasible set (measurements are preserved)
            let phi_x = phi.dot(&x_new);
            let residual = y - &phi_x;

            // Use least squares to find the correction
            let correction = match solve(&gram.view(), &residual.view()) {
                Ok(corr) => corr,
                Err(_) => {
                    return Err(SignalError::Compute(
                        "Failed to compute projection in SL0".to_string(),
                    ));
                }
            };

            // Apply correction
            x_new = x_new + phi_t.dot(&correction);

            // Update current solution
            x_current = x_new;
        }

        // Decrease sigma for next iteration
        sigma *= sigma_decrease_factor;
    }

    // Apply non-negative constraint if requested
    if config.non_negative {
        x_current.mapv_inplace(|val| val.max(0.0));
    }

    Ok(x_current)
}

/// Performs sparse signal recovery using Basis Pursuit
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn basis_pursuit(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    // Basis Pursuit is essentially LASSO with a very small lambda
    // Using FISTA implementation with a small lambda
    let bp_config = SparseRecoveryConfig {
        lambda: 1e-4,
        max_iterations: config.max_iterations * 2, // More iterations for convergence
        convergence_threshold: config.convergence_threshold * 0.1, // Tighter convergence
        ..config.clone()
    };

    fista(y, phi, &bp_config)
}

/// Performs sparse signal recovery using LASSO
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `config` - Configuration for sparse recovery algorithm
///
/// # Returns
///
/// * Recovered sparse signal
pub fn lasso(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    // Use FISTA implementation for LASSO
    fista(y, phi, config)
}

/// Performs compressed sensing recovery of signals from random measurements
///
/// # Arguments
///
/// * `y` - Measurement vector
/// * `phi` - Sensing matrix
/// * `method` - Recovery method to use
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Recovered sparse signal
pub fn compressed_sensing_recover(
    y: &Array1<f64>,
    phi: &Array2<f64>,
    method: SparseRecoveryMethod,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    match method {
        SparseRecoveryMethod::OMP => omp(y, phi, config),
        SparseRecoveryMethod::MP => mp(y, phi, config),
        SparseRecoveryMethod::BasisPursuit => basis_pursuit(y, phi, config),
        SparseRecoveryMethod::ISTA => ista(y, phi, config),
        SparseRecoveryMethod::FISTA => fista(y, phi, config),
        SparseRecoveryMethod::LASSO => lasso(y, phi, config),
        SparseRecoveryMethod::IHT => iht(y, phi, config),
        SparseRecoveryMethod::CoSaMP => cosamp(y, phi, config),
        SparseRecoveryMethod::SubspacePursuit => subspace_pursuit(y, phi, config),
        SparseRecoveryMethod::SmoothL0 => smooth_l0(y, phi, config),
    }
}

/// Performs sparse signal recovery in a transform domain
///
/// # Arguments
///
/// * `y` - Observed signal (possibly with noise or missing samples)
/// * `transform_forward` - Function to transform from time/space to sparse domain
/// * `transform_inverse` - Function to transform from sparse domain to time/space
/// * `mask` - Binary mask indicating which samples are observed (1) or missing (0)
/// * `method` - Recovery method to use
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Recovered full signal
pub fn sparse_transform_recovery<F, G>(
    y: &Array1<f64>,
    _transform_forward: F,
    transform_inverse: G,
    mask: Option<&Array1<f64>>,
    method: SparseRecoveryMethod,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> SignalResult<Array1<f64>>,
    G: Fn(&Array1<f64>) -> SignalResult<Array1<f64>>,
{
    let n = y.len();

    // If no mask is provided, assume all samples are observed
    let mask = match mask {
        Some(m) => m.clone(),
        None => Array1::ones(n),
    };

    // Check if we have any missing samples
    let has_missing = mask.iter().any(|&x| x < 0.5);

    if !has_missing {
        // No missing samples, just return the input
        return Ok(y.clone());
    }

    // Count observed samples
    let m = mask.iter().filter(|&&x| x > 0.5).count();

    // Create sensing matrix (masked inverse transform)
    // For each column j, compute inverse transform of a unit vector at position j,
    // then apply the observation mask
    let mut phi = Array2::<f64>::zeros((m, n));

    for j in 0..n {
        // Create unit vector
        let mut unit = Array1::<f64>::zeros(n);
        unit[j] = 1.0;

        // Apply inverse transform
        let col = transform_inverse(&unit)?;

        // Apply mask and store in the appropriate column
        let mut idx = 0;
        for i in 0..n {
            if mask[i] > 0.5 {
                phi[[idx, j]] = col[i];
                idx += 1;
            }
        }
    }

    // Extract observed samples into a vector
    let mut y_observed = Array1::<f64>::zeros(m);
    let mut idx = 0;
    for i in 0..n {
        if mask[i] > 0.5 {
            y_observed[idx] = y[i];
            idx += 1;
        }
    }

    // Recover sparse representation
    let sparse_coeffs = compressed_sensing_recover(&y_observed, &phi, method, config)?;

    // Transform back to signal domain
    transform_inverse(&sparse_coeffs)
}

/// Recovers a signal with missing samples using sparsity in the frequency domain
///
/// # Arguments
///
/// * `y` - Observed signal with missing samples (NaN for missing values)
/// * `method` - Recovery method to use
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Recovered full signal
pub fn recover_missing_samples(
    y: &Array1<f64>,
    method: SparseRecoveryMethod,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let n = y.len();

    // Create mask (1 for observed samples, 0 for missing)
    let mask = y.mapv(|v| if v.is_nan() { 0.0 } else { 1.0 });

    // Replace NaN values with zeros for processing
    let y_clean = y.mapv(|v| if v.is_nan() { 0.0 } else { v });

    // Define FFT-based transforms
    let forward_transform = |signal: &Array1<f64>| -> SignalResult<Array1<f64>> {
        // Create FFT planner
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        // Convert to complex
        let mut complex_signal = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            complex_signal[i] = Complex::new(signal[i], 0.0);
        }

        // Perform FFT
        fft.process(&mut complex_signal);

        // Convert back to real (taking magnitude)
        let mut result = Array1::<f64>::zeros(n);
        for i in 0..n {
            result[i] = (complex_signal[i].re.powi(2) + complex_signal[i].im.powi(2)).sqrt();
        }

        Ok(result)
    };

    let inverse_transform = |spectrum: &Array1<f64>| -> SignalResult<Array1<f64>> {
        // Create FFT planner
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n);

        // Convert to complex (assuming phase = 0)
        let mut complex_spectrum = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            complex_spectrum[i] = Complex::new(spectrum[i], 0.0);
        }

        // Perform IFFT
        ifft.process(&mut complex_spectrum);

        // Scale and convert back to real
        let scale = 1.0 / n as f64;
        let mut result = Array1::<f64>::zeros(n);
        for i in 0..n {
            result[i] = complex_spectrum[i].re * scale;
        }

        Ok(result)
    };

    // Use sparse recovery in the transform domain
    sparse_transform_recovery(
        &y_clean,
        forward_transform,
        inverse_transform,
        Some(&mask),
        method,
        config,
    )
}

/// Performs image inpainting using sparsity-based recovery
///
/// # Arguments
///
/// * `image` - Input image with missing pixels (NaN values)
/// * `patch_size` - Size of patches for processing (e.g., 8x8)
/// * `method` - Recovery method to use
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Recovered image
pub fn image_inpainting(
    image: &Array2<f64>,
    patch_size: usize,
    method: SparseRecoveryMethod,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array2<f64>> {
    let (n_rows, n_cols) = image.dim();

    // Check if input has any missing values
    let has_missing = image.iter().any(|&x| x.is_nan());
    if !has_missing {
        return Ok(image.clone());
    }

    // Initialize result with input image
    let mut result = image.clone();

    // Process the image in patches
    for i in (0..n_rows).step_by(patch_size / 2) {
        for j in (0..n_cols).step_by(patch_size / 2) {
            // Define patch boundaries
            let i_end = (i + patch_size).min(n_rows);
            let j_end = (j + patch_size).min(n_cols);
            let i_size = i_end - i;
            let j_size = j_end - j;

            // Extract patch
            let patch = image.slice(s![i..i_end, j..j_end]).to_owned();

            // Only process patches with missing values
            if patch.iter().any(|&x| x.is_nan()) {
                // Reshape patch to 1D for processing
                let patch_flat = Array1::from_iter(patch.iter().cloned());

                // Recover missing values in the patch
                let recovered_flat = recover_missing_samples(&patch_flat, method, config)?;

                // Reshape back to 2D
                let mut recovered_patch = Array2::<f64>::zeros((i_size, j_size));
                let mut idx = 0;
                for ii in 0..i_size {
                    for jj in 0..j_size {
                        recovered_patch[[ii, jj]] = recovered_flat[idx];
                        idx += 1;
                    }
                }

                // Copy recovered values only where original had missing values
                for ii in 0..i_size {
                    for jj in 0..j_size {
                        if image[[i + ii, j + jj]].is_nan() {
                            result[[i + ii, j + jj]] = recovered_patch[[ii, jj]];
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Creates a random Gaussian sensing matrix for compressed sensing
///
/// # Arguments
///
/// * `m` - Number of measurements (rows)
/// * `n` - Signal dimension (columns)
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// * Random sensing matrix with normalized columns
pub fn random_sensing_matrix(m: usize, n: usize, seed: Option<u64>) -> Array2<f64> {
    // Initialize with random Gaussian entries
    let mut rng = match seed {
        Some(s) => StdRng::from_seed([s as u8; 32]),
        None => StdRng::from_seed([0u8; 32]), // Use deterministic seed for consistency
    };

    let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

    let mut phi = Array2::<f64>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            phi[[i, j]] = normal.sample(&mut rng);
        }
    }

    // Normalize columns
    for j in 0..n {
        let mut col = phi.slice_mut(s![.., j]);
        let norm = col.mapv(|x| x * x).sum().sqrt();
        if norm > 1e-10 {
            col.mapv_inplace(|x| x / norm);
        }
    }

    phi
}

/// Computes the coherence of a sensing matrix
///
/// # Arguments
///
/// * `phi` - Sensing matrix
///
/// # Returns
///
/// * Coherence (maximum absolute inner product between normalized columns)
pub fn matrix_coherence(phi: &Array2<f64>) -> SignalResult<f64> {
    let (_, n) = phi.dim();

    let mut max_coherence = 0.0;

    for i in 0..n {
        let col_i = phi.slice(s![.., i]);
        let norm_i = vector_norm(&col_i.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;

        for j in i + 1..n {
            let col_j = phi.slice(s![.., j]);
            let norm_j = vector_norm(&col_j.view(), 2)
                .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;

            let inner_product = col_i.dot(&col_j);
            let coherence = (inner_product / (norm_i * norm_j)).abs();

            max_coherence = f64::max(max_coherence, coherence);
        }
    }

    Ok(max_coherence)
}

/// Computes the restricted isometry property (RIP) constant approximation
///
/// # Arguments
///
/// * `phi` - Sensing matrix
/// * `s` - Sparsity level
///
/// # Returns
///
/// * Estimated RIP constant
pub fn estimate_rip_constant(phi: &Array2<f64>, s: usize) -> SignalResult<f64> {
    let (_m, n) = phi.dim();

    if s > n {
        return Err(SignalError::ValueError(
            "Sparsity level s cannot be larger than signal dimension n".to_string(),
        ));
    }

    // For exact computation, we would need to check all (n choose s) submatrices,
    // which is computationally infeasible for large n and s.
    // Instead, we use a Monte Carlo approach with random sparse vectors.

    const NUM_TRIALS: usize = 1000;
    let mut rng = rand::rng();

    let mut min_ratio = f64::MAX;
    let mut max_ratio = 0.0;

    for _ in 0..NUM_TRIALS {
        // Create a random s-sparse vector
        let mut x = Array1::<f64>::zeros(n);

        // Randomly select s indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(s);

        // Set random values at these indices
        for &idx in &indices {
            // Random value between -1 and 1
            x[idx] = 2.0 * rng.random::<f64>() - 1.0;
        }

        // Normalize x
        let x_norm = vector_norm(&x.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        x.mapv_inplace(|val| val / x_norm);

        // Compute Phi * x
        let y = phi.dot(&x);

        // Compute the ratio ||Phi * x||^2 / ||x||^2
        // Since x is normalized, ||x||^2 = 1
        let y_norm = vector_norm(&y.view(), 2)
            .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;
        let ratio = y_norm.powi(2);

        min_ratio = f64::min(min_ratio, ratio);
        max_ratio = f64::max(max_ratio, ratio);
    }

    // RIP constant is the maximum deviation from 1
    let delta = f64::max(1.0 - min_ratio, max_ratio - 1.0);

    Ok(delta)
}

/// Measures the sparsity of a signal using normalized L0/L1 ratio
///
/// # Arguments
///
/// * `x` - Input signal
/// * `threshold` - Threshold for considering a value as non-zero
///
/// # Returns
///
/// * Normalized sparsity measure (0 = dense, 1 = maximally sparse)
pub fn measure_sparsity(x: &Array1<f64>, threshold: f64) -> SignalResult<f64> {
    let n = x.len();

    // Note: the threshold parameter is currently unused in this function
    // The function uses L1/L2 norm ratio instead of counting non-zero elements
    let _ = threshold;

    // L1 norm
    let l1_norm = vector_norm(&x.view(), 1)
        .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;

    // L2 norm
    let l2_norm_val = vector_norm(&x.view(), 2)
        .map_err(|_| SignalError::Compute("Failed to compute norm".to_string()))?;

    // Compute normalized sparsity measure: 1 - (L1/L2)/sqrt(n)
    // This will be close to 1 for sparse signals and close to 0 for dense signals
    if l2_norm_val < 1e-10 {
        Ok(0.0) // All zeros is considered dense
    } else {
        Ok(1.0 - (l1_norm / l2_norm_val) / (n as f64).sqrt())
    }
}

/// Applies sparse signal recovery to denoise a signal
///
/// # Arguments
///
/// * `y` - Noisy signal
/// * `transform` - Which transform domain to use for sparsity
/// * `method` - Recovery method to use
/// * `config` - Configuration parameters
///
/// # Returns
///
/// * Denoised signal
pub fn sparse_denoise(
    y: &Array1<f64>,
    transform: SparseTransform,
    method: SparseRecoveryMethod,
    config: &SparseRecoveryConfig,
) -> SignalResult<Array1<f64>> {
    let n = y.len();

    // Create identity sensing matrix
    let phi = Array2::<f64>::eye(n);

    // Define FFT transform functions
    let fft_forward = |signal: &Array1<f64>| -> SignalResult<Array1<Complex64>> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        let mut complex_signal: Vec<Complex64> =
            signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        fft.process(&mut complex_signal);

        Ok(Array1::from_vec(complex_signal))
    };

    let fft_inverse = |spectrum: &Array1<Complex64>| -> SignalResult<Array1<f64>> {
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(n);

        let mut complex_spectrum = spectrum.to_vec();

        ifft.process(&mut complex_spectrum);

        let scale = 1.0 / n as f64;
        let result = complex_spectrum.iter().map(|&x| x.re * scale).collect();

        Ok(Array1::from_vec(result))
    };

    // Choose transform domain
    match transform {
        SparseTransform::Frequency => {
            // Apply FFT to the signal
            let spectrum = fft_forward(y)?;

            // Separate real and imaginary parts
            let mut real_part = Array1::<f64>::zeros(n);
            let mut imag_part = Array1::<f64>::zeros(n);

            for i in 0..n {
                real_part[i] = spectrum[i].re;
                imag_part[i] = spectrum[i].im;
            }

            // Apply sparse recovery to real and imaginary parts separately
            let real_sparse = compressed_sensing_recover(&real_part, &phi, method, config)?;
            let imag_sparse = compressed_sensing_recover(&imag_part, &phi, method, config)?;

            // Recombine
            let mut sparse_spectrum = Array1::<Complex64>::zeros(n);
            for i in 0..n {
                sparse_spectrum[i] = Complex64::new(real_sparse[i], imag_sparse[i]);
            }

            // Inverse FFT to get denoised signal
            fft_inverse(&sparse_spectrum)
        }
        SparseTransform::Wavelet => {
            // For wavelet transform, we use the sparse_transform_recovery function
            // with wavelet transforms

            // Define wavelet transform functions
            let forward_wavelet = |signal: &Array1<f64>| -> SignalResult<Array1<f64>> {
                // Here we would use a wavelet transform from the dwt module
                // For simplicity, using a placeholder that just returns the signal
                // In a real implementation, replace with actual wavelet transform
                Ok(signal.clone())
            };

            let inverse_wavelet = |coeffs: &Array1<f64>| -> SignalResult<Array1<f64>> {
                // Here we would use an inverse wavelet transform
                // For simplicity, using a placeholder
                // In a real implementation, replace with actual inverse wavelet transform
                Ok(coeffs.clone())
            };

            // All samples are observed, no mask needed
            sparse_transform_recovery(y, forward_wavelet, inverse_wavelet, None, method, config)
        }
        SparseTransform::DCT => {
            // Implement DCT-based sparse recovery
            // Placeholder implementation - in a real implementation,
            // replace with actual DCT transform

            // All samples are observed, no mask needed
            let mut result = y.clone();

            // Apply a simple thresholding as a placeholder
            let threshold = y.fold(0.0, |acc, &val| acc + val.abs()) / (n as f64) * config.lambda;

            for i in 0..n {
                if result[i].abs() < threshold {
                    result[i] = 0.0;
                }
            }

            Ok(result)
        }
    }
}

/// Transform domains for sparse signal processing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparseTransform {
    /// Frequency domain (FFT)
    Frequency,
    /// Wavelet domain
    Wavelet,
    /// Discrete Cosine Transform domain
    DCT,
}
