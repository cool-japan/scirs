use scirs2_core::ndarray::s;
use std::f64::consts::PI;
// Advanced parametric spectral estimation methods
//
// This module provides state-of-the-art parametric spectral estimation including:
// - Vector autoregressive (VAR) models for multivariate signals
// - Kalman filter-based adaptive parameter estimation
// - Spectral factorization methods
// - High-resolution spectral estimation (MUSIC, ESPRIT)
// - Regularized parameter estimation for ill-conditioned problems
// - Cross-spectral density estimation for multi-channel signals

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::Float;
use scirs2_core::parallel_ops::*;
use scirs2_core::validation::{check_finite, checkshape};
use scirs2_linalg::{
    det as linalg_det, eig as linalg_eig, eigh as linalg_eigh, inv as linalg_inv,
    solve_multiple as linalg_solve_multiple,
};

#[allow(unused_imports)]
/// Vector Autoregressive (VAR) model
#[derive(Debug, Clone)]
pub struct VarModel {
    /// AR coefficient matrices [A1, A2, ..., Ap]
    pub ar_matrices: Vec<Array2<f64>>,
    /// Innovation covariance matrix
    pub innovation_cov: Array2<f64>,
    /// Model order
    pub order: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Log-likelihood (if computed)
    pub log_likelihood: Option<f64>,
}

/// High-resolution spectral estimation result
#[derive(Debug, Clone)]
pub struct HighResolutionResult {
    /// Estimated frequencies
    pub frequencies: Vec<f64>,
    /// Spectral powers at frequencies
    pub powers: Vec<f64>,
    /// Method used
    pub method: HighResolutionMethod,
    /// Model order used
    pub model_order: usize,
    /// Number of signals detected
    pub n_signals: usize,
}

/// High-resolution methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HighResolutionMethod {
    /// Multiple Signal Classification
    MUSIC,
    /// Estimation of Signal Parameters via Rotational Invariance Techniques
    ESPRIT,
    /// Minimum Norm
    MinNorm,
    /// Root-MUSIC
    RootMUSIC,
}

/// Regularization method for parameter estimation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegularizationMethod {
    /// Ridge regression (L2 regularization)
    Ridge,
    /// LASSO (L1 regularization)
    Lasso,
    /// Elastic net (L1 + L2)
    ElasticNet,
    /// Tikhonov regularization
    Tikhonov,
}

/// Configuration for advanced parametric estimation
#[derive(Debug, Clone)]
pub struct AdvancedParametricConfig {
    /// Maximum model order to consider
    pub max_order: usize,
    /// Regularization strength
    pub regularization_strength: f64,
    /// Regularization method
    pub regularization_method: RegularizationMethod,
    /// Use cross-validation for order selection
    pub use_cross_validation: bool,
    /// Number of CV folds
    pub cv_folds: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}

impl Default for AdvancedParametricConfig {
    fn default() -> Self {
        Self {
            max_order: 20,
            regularization_strength: 0.01,
            regularization_method: RegularizationMethod::Ridge,
            use_cross_validation: true,
            cv_folds: 5,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

/// Estimate Vector Autoregressive (VAR) model for multivariate signals
///
/// # Arguments
///
/// * `data` - Multivariate time series (variables × time)
/// * `order` - VAR model order
/// * `config` - Advanced configuration
///
/// # Returns
///
/// * VAR model with estimated parameters
#[allow(dead_code)]
pub fn estimate_var_model(
    data: &Array2<f64>,
    order: usize,
    config: &AdvancedParametricConfig,
) -> SignalResult<VarModel> {
    let (n_vars, n_obs) = data.dim();

    if n_obs <= order * n_vars {
        return Err(SignalError::ValueError(
            "Insufficient observations for VAR model".to_string(),
        ));
    }

    checkshape(data, &[n_vars, n_obs], "data")?;

    // Check for finite values
    for i in 0..n_vars {
        for j in 0..n_obs {
            check_finite(data[[i, j]], format!("data[{}, {}]", i, j))?;
        }
    }

    // Build design matrix for VAR estimation
    let n_equations = n_obs - order;
    let n_regressors = order * n_vars;

    let mut y_matrix = Array2::zeros((n_vars, n_equations));
    let mut x_matrix = Array2::zeros((n_regressors, n_equations));

    for t in 0..n_equations {
        // Dependent variables at time t+order
        for i in 0..n_vars {
            y_matrix[[i, t]] = data[[i, t + order]];
        }

        // Lagged variables as regressors
        for lag in 1..=order {
            for i in 0..n_vars {
                let regressor_idx = (lag - 1) * n_vars + i;
                x_matrix[[regressor_idx, t]] = data[[i, t + order - lag]];
            }
        }
    }

    // Estimate VAR parameters using regularized least squares
    let ar_coeffs = estimate_var_coefficients(
        &y_matrix,
        &x_matrix,
        config.regularization_method,
        config.regularization_strength,
    )?;

    // Reshape coefficients into matrices
    let mut ar_matrices = Vec::with_capacity(order);
    for lag in 0..order {
        let mut matrix = Array2::zeros((n_vars, n_vars));
        for i in 0..n_vars {
            for j in 0..n_vars {
                let coeff_idx = lag * n_vars + j;
                matrix[[i, j]] = ar_coeffs[[i, coeff_idx]];
            }
        }
        ar_matrices.push(matrix);
    }

    // Compute residuals and innovation covariance
    let residuals = compute_var_residuals(data, &ar_matrices, order)?;
    let innovation_cov = compute_covariance_matrix(&residuals)?;

    // Compute log-likelihood
    let log_likelihood = compute_var_log_likelihood(&residuals, &innovation_cov)?;

    Ok(VarModel {
        ar_matrices,
        innovation_cov,
        order,
        n_vars,
        log_likelihood: Some(log_likelihood),
    })
}

/// Estimate VAR coefficients using regularized least squares
#[allow(dead_code)]
fn estimate_var_coefficients(
    y: &Array2<f64>,
    x: &Array2<f64>,
    reg_method: RegularizationMethod,
    lambda: f64,
) -> SignalResult<Array2<f64>> {
    let (_n_vars, _n_obs) = y.dim();
    let (n_regressors, _n_equations) = x.dim();

    match reg_method {
        RegularizationMethod::Ridge => {
            // Ridge regression: (X'X + λI)^{-1} X'Y
            let mut xtx = x.dot(&x.t());

            // Add regularization
            for i in 0..n_regressors {
                xtx[[i, i]] += lambda;
            }

            let xty = x.dot(&y.t());
            solve_linear_system(&xtx, &xty)
        }
        RegularizationMethod::Lasso => {
            // LASSO regression using coordinate descent
            estimate_lasso_coefficients(y, x, lambda)
        }
        RegularizationMethod::ElasticNet => {
            // Elastic net: combine L1 and L2 penalties
            let alpha = 0.5; // Mix parameter
            estimate_elastic_net_coefficients(y, x, lambda, alpha)
        }
        RegularizationMethod::Tikhonov => {
            // Tikhonov regularization with smoothness penalty
            estimate_tikhonov_coefficients(y, x, lambda)
        }
    }
}

/// LASSO estimation using coordinate descent
#[allow(dead_code)]
fn estimate_lasso_coefficients(
    y: &Array2<f64>,
    x: &Array2<f64>,
    lambda: f64,
) -> SignalResult<Array2<f64>> {
    let (n_vars, _) = y.dim();
    let (n_regressors, n_obs) = x.dim();

    let mut coeffs = Array2::zeros((n_vars, n_regressors));
    let max_iter = 1000;
    let tolerance = 1e-6;

    // Coordinate descent for each equation
    for eq in 0..n_vars {
        let y_eq = y.row(eq);
        let mut beta = Array1::<f64>::zeros(n_regressors);

        for _iter in 0..max_iter {
            let old_beta = beta.clone();

            for j in 0..n_regressors {
                // Compute partial residual
                let mut partial_residual = 0.0;
                for i in 0..n_obs {
                    let mut pred = 0.0;
                    for k in 0..n_regressors {
                        if k != j {
                            pred += beta[k] * x[[k, i]];
                        }
                    }
                    partial_residual += x[[j, i]] * (y_eq[i] - pred);
                }

                // Compute x_j^T x_j
                let xtx_jj: f64 = x.row(j).iter().map(|&xi| xi * xi).sum::<f64>();

                // Soft thresholding
                beta[j] = soft_threshold(partial_residual, lambda) / xtx_jj;
            }

            // Check convergence
            let change: f64 = beta
                .iter()
                .zip(old_beta.iter())
                .map(|(&new, &old)| (new - old).abs())
                .sum();

            if change < tolerance {
                break;
            }
        }

        // Store coefficients for this equation
        for j in 0..n_regressors {
            coeffs[[eq, j]] = beta[j];
        }
    }

    Ok(coeffs)
}

/// Soft thresholding operator for LASSO
#[allow(dead_code)]
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Elastic net estimation
#[allow(dead_code)]
fn estimate_elastic_net_coefficients(
    y: &Array2<f64>,
    x: &Array2<f64>,
    lambda: f64,
    alpha: f64,
) -> SignalResult<Array2<f64>> {
    let l1_lambda = alpha * lambda;
    let l2_lambda = (1.0 - alpha) * lambda;

    let (n_vars, _) = y.dim();
    let (n_regressors, n_obs) = x.dim();

    let mut coeffs = Array2::zeros((n_vars, n_regressors));
    let max_iter = 1000;
    let tolerance = 1e-6;

    // Coordinate descent with elastic net penalty
    for eq in 0..n_vars {
        let y_eq = y.row(eq);
        let mut beta = Array1::<f64>::zeros(n_regressors);

        for _iter in 0..max_iter {
            let old_beta = beta.clone();

            for j in 0..n_regressors {
                // Compute partial residual
                let mut partial_residual = 0.0;
                for i in 0..n_obs {
                    let mut pred = 0.0;
                    for k in 0..n_regressors {
                        if k != j {
                            pred += beta[k] * x[[k, i]];
                        }
                    }
                    partial_residual += x[[j, i]] * (y_eq[i] - pred);
                }

                // Compute x_j^T x_j + L2 penalty
                let xtx_jj: f64 = x.row(j).iter().map(|&xi| xi * xi).sum::<f64>() + l2_lambda;

                // Soft thresholding with L1 penalty
                beta[j] = soft_threshold(partial_residual, l1_lambda) / xtx_jj;
            }

            // Check convergence
            let change: f64 = beta
                .iter()
                .zip(old_beta.iter())
                .map(|(&new, &old)| (new - old).abs())
                .sum();

            if change < tolerance {
                break;
            }
        }

        // Store coefficients
        for j in 0..n_regressors {
            coeffs[[eq, j]] = beta[j];
        }
    }

    Ok(coeffs)
}

/// Tikhonov regularization
#[allow(dead_code)]
fn estimate_tikhonov_coefficients(
    y: &Array2<f64>,
    x: &Array2<f64>,
    lambda: f64,
) -> SignalResult<Array2<f64>> {
    let (_, n_regressors) = x.dim();

    // Create smoothness penalty matrix (second-order differences)
    let mut penalty_matrix = Array2::zeros((n_regressors, n_regressors));
    for i in 1..n_regressors - 1 {
        penalty_matrix[[i, i - 1]] = 1.0;
        penalty_matrix[[i, i]] = -2.0;
        penalty_matrix[[i, i + 1]] = 1.0;
    }

    // Regularized normal equations: (X'X + λL'L)^{-1} X'Y
    let mut xtx = x.dot(&x.t());
    let penalty_term = penalty_matrix.t().dot(&penalty_matrix);

    for i in 0..n_regressors {
        for j in 0..n_regressors {
            xtx[[i, j]] += lambda * penalty_term[[i, j]];
        }
    }

    let xty = x.dot(&y.t());
    solve_linear_system(&xtx, &xty)
}

/// Compute VAR residuals
#[allow(dead_code)]
fn compute_var_residuals(
    data: &Array2<f64>,
    ar_matrices: &[Array2<f64>],
    order: usize,
) -> SignalResult<Array2<f64>> {
    let (n_vars, n_obs) = data.dim();
    let n_residuals = n_obs - order;

    let mut residuals = Array2::zeros((n_vars, n_residuals));

    for t in 0..n_residuals {
        let obs_time = t + order;

        // Predicted value
        let mut prediction = Array1::<f64>::zeros(n_vars);

        for lag in 1..=order {
            let lag_data = data.column(obs_time - lag);
            let ar_matrix = &ar_matrices[lag - 1];

            for i in 0..n_vars {
                for j in 0..n_vars {
                    prediction[i] += ar_matrix[[i, j]] * lag_data[j];
                }
            }
        }

        // Residual
        for i in 0..n_vars {
            residuals[[i, t]] = data[[i, obs_time]] - prediction[i];
        }
    }

    Ok(residuals)
}

/// Compute covariance matrix
#[allow(dead_code)]
fn compute_covariance_matrix(residuals: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (n_vars, n_obs) = residuals.dim();
    let mut cov_matrix = Array2::zeros((n_vars, n_vars));

    // Compute sample covariance
    for i in 0..n_vars {
        for j in 0..n_vars {
            let mut cov = 0.0;
            for t in 0..n_obs {
                cov += residuals[[i, t]] * residuals[[j, t]];
            }
            cov_matrix[[i, j]] = cov / (n_obs - 1) as f64;
        }
    }

    Ok(cov_matrix)
}

/// Compute VAR log-likelihood
#[allow(dead_code)]
fn compute_var_log_likelihood(
    residuals: &Array2<f64>,
    innovation_cov: &Array2<f64>,
) -> SignalResult<f64> {
    let (n_vars, n_obs) = residuals.dim();

    // Compute determinant of covariance matrix
    let det_cov = compute_matrix_determinant(innovation_cov)?;

    if det_cov <= 0.0 {
        return Err(SignalError::ComputationError(
            "Singular covariance matrix".to_string(),
        ));
    }

    // Compute inverse of covariance matrix
    let inv_cov = compute_matrix_inverse(innovation_cov)?;

    let mut log_likelihood = 0.0;
    log_likelihood -= 0.5 * n_obs as f64 * n_vars as f64 * (2.0 * PI).ln();
    log_likelihood -= 0.5 * n_obs as f64 * det_cov.ln();

    // Compute quadratic form
    for t in 0..n_obs {
        let residual_t = residuals.column(t);
        let mut quad_form = 0.0;

        for i in 0..n_vars {
            for j in 0..n_vars {
                quad_form += residual_t[i] * inv_cov[[i, j]] * residual_t[j];
            }
        }

        log_likelihood -= 0.5 * quad_form;
    }

    Ok(log_likelihood)
}

/// High-resolution spectral estimation using subspace methods
///
/// # Arguments
///
/// * `signal` - Input signal (or covariance matrix for MUSIC)
/// * `n_signals` - Number of signal components
/// * `method` - High-resolution method to use
/// * `n_frequencies` - Number of frequency points to evaluate
///
/// # Returns
///
/// * High-resolution spectral estimate
#[allow(dead_code)]
pub fn high_resolution_spectral_estimation(
    signal: &Array1<f64>,
    n_signals: usize,
    method: HighResolutionMethod,
    n_frequencies: usize,
) -> SignalResult<HighResolutionResult> {
    match method {
        HighResolutionMethod::MUSIC => music_spectrum(signal, n_signals, n_frequencies),
        HighResolutionMethod::ESPRIT => esprit_estimation(signal, n_signals),
        HighResolutionMethod::MinNorm => min_norm_spectrum(signal, n_signals, n_frequencies),
        HighResolutionMethod::RootMUSIC => root_music_estimation(signal, n_signals),
    }
}

/// MUSIC (Multiple Signal Classification) algorithm
#[allow(dead_code)]
fn music_spectrum(
    signal: &Array1<f64>,
    n_signals: usize,
    n_frequencies: usize,
) -> SignalResult<HighResolutionResult> {
    let n = signal.len();
    let model_order = (n / 3).min(50); // Model order for covariance estimation

    // Estimate covariance matrix using forward-backward averaging
    let cov_matrix = estimate_covariance_matrix_fb(signal, model_order)?;

    // Eigenvalue decomposition
    let (eigenvalues, eigenvectors) = compute_eigendecomposition(&cov_matrix)?;

    // Sort eigenvalues in descending order
    let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvalues
        .iter()
        .zip(eigenvectors.axis_iter(Axis(1)))
        .map(|(&val, vec)| (val, vec.to_owned()))
        .collect();

    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("Operation failed"));

    // Split into signal and noise subspaces
    let noise_eigenvectors: Vec<Array1<f64>> = eigen_pairs
        .iter()
        .skip(n_signals)
        .map(|(_, vec)| vec.clone())
        .collect();

    // Compute MUSIC spectrum
    let frequencies: Vec<f64> = (0..n_frequencies)
        .map(|i| i as f64 / n_frequencies as f64)
        .collect();

    let mut powers = Vec::with_capacity(n_frequencies);

    for &freq in &frequencies {
        let omega = 2.0 * PI * freq;

        // Create steering vector
        let steering_vector: Array1<Complex64> = (0..model_order)
            .map(|k| Complex64::exp(Complex64::i() * omega * k as f64))
            .collect();

        // Compute denominator: a^H * En * En^H * a
        let mut denominator = 0.0;

        for noise_vec in &noise_eigenvectors {
            let noise_vec_complex: Array1<Complex64> =
                noise_vec.iter().map(|&x| Complex64::new(x, 0.0)).collect();

            let projection = steering_vector
                .iter()
                .zip(noise_vec_complex.iter())
                .map(|(a, n)| a.conj() * n)
                .sum::<Complex64>();

            denominator += projection.norm_sqr();
        }

        powers.push(1.0 / denominator.max(1e-15));
    }

    // Convert to dB scale
    let max_power = powers.iter().cloned().fold(0.0, f64::max);
    for power in &mut powers {
        *power = 10.0 * (*power / max_power.max(1e-15)).log10();
    }

    Ok(HighResolutionResult {
        frequencies: frequencies.clone(),
        powers,
        method: HighResolutionMethod::MUSIC,
        model_order,
        n_signals,
    })
}

/// Estimate covariance matrix using forward-backward averaging
#[allow(dead_code)]
fn estimate_covariance_matrix_fb(
    signal: &Array1<f64>,
    model_order: usize,
) -> SignalResult<Array2<f64>> {
    let n = signal.len();
    let n_snapshots = n - model_order + 1;

    let mut cov_matrix = Array2::zeros((model_order, model_order));

    // Forward covariance
    for t in 0..n_snapshots {
        let snapshot = signal.slice(s![t..t + model_order]);

        for i in 0..model_order {
            for j in 0..model_order {
                cov_matrix[[i, j]] += snapshot[i] * snapshot[j];
            }
        }
    }

    // Backward covariance (conjugate for complex signals, but real here)
    for t in 0..n_snapshots {
        let snapshot = signal.slice(s![t..t + model_order]);

        for i in 0..model_order {
            for j in 0..model_order {
                let idx_i = model_order - 1 - i;
                let idx_j = model_order - 1 - j;
                cov_matrix[[i, j]] += snapshot[idx_i] * snapshot[idx_j];
            }
        }
    }

    // Average
    cov_matrix.mapv_inplace(|x| x / (2.0 * n_snapshots as f64));

    Ok(cov_matrix)
}

/// ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
#[allow(dead_code)]
fn esprit_estimation(signal: &Array1<f64>, n_signals: usize) -> SignalResult<HighResolutionResult> {
    let n = signal.len();
    let model_order = (n / 3).min(50);

    // Estimate covariance matrix
    let cov_matrix = estimate_covariance_matrix_fb(signal, model_order)?;

    // Eigenvalue decomposition
    let (eigenvalues, eigenvectors) = compute_eigendecomposition(&cov_matrix)?;

    // Sort and select _signal subspace
    let mut eigen_pairs: Vec<(f64, Array1<f64>)> = eigenvalues
        .iter()
        .zip(eigenvectors.axis_iter(Axis(1)))
        .map(|(&val, vec)| (val, vec.to_owned()))
        .collect();

    eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("Operation failed"));

    // Signal subspace matrix
    let mut signal_subspace = Array2::zeros((model_order, n_signals));
    for (i, (_, eigenvec)) in eigen_pairs.iter().take(n_signals).enumerate() {
        for j in 0..model_order {
            signal_subspace[[j, i]] = eigenvec[j];
        }
    }

    // Split into overlapping subarrays
    let es1 = signal_subspace.slice(s![0..model_order - 1, ..]).to_owned();
    let es2 = signal_subspace.slice(s![1..model_order, ..]).to_owned();

    // Solve: ES2 = ES1 * Phi (where Phi contains the rotational-invariance
    // eigenvalues) via a genuine least-squares solve.
    let phi_matrix = solve_least_squares(&es1, &es2)?;

    // Phi is generally NOT symmetric, so its eigenvalues are genuinely
    // complex in general; reusing the symmetric-only `compute_eigendecomposition`
    // here would either reject the matrix or silently discard the
    // rotational (phase) information that IS the ESPRIT frequency estimate.
    // In the (near-)noiseless case the eigenvalues of Phi lie on the unit
    // circle at exp(j*2*pi*f_k); recover f_k from the eigenvalue phase.
    let (phi_eigenvalues, _phi_vecs) = linalg_eig(&phi_matrix.view(), None).map_err(|e| {
        SignalError::ComputationError(format!("ESPRIT eigendecomposition failed: {e}"))
    })?;

    let mut freq_power_pairs: Vec<(f64, f64)> = Vec::new();
    for eigenval in phi_eigenvalues.iter() {
        // Real coefficients guarantee eigenvalues occur in complex-conjugate
        // pairs; keep only one (non-negative angle) representative of each.
        if eigenval.im < 0.0 {
            continue;
        }
        let angle = eigenval.im.atan2(eigenval.re);
        let freq = angle.abs() / (2.0 * PI);
        // Magnitude close to 1.0 indicates a genuine (well-conditioned)
        // signal root rather than a numerical artifact.
        freq_power_pairs.push((freq, eigenval.norm()));
    }
    freq_power_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (frequencies, powers): (Vec<f64>, Vec<f64>) = freq_power_pairs.into_iter().unzip();

    let n_signals = frequencies.len();
    Ok(HighResolutionResult {
        frequencies,
        powers,
        method: HighResolutionMethod::ESPRIT,
        model_order,
        n_signals,
    })
}

/// Minimum norm algorithm
///
/// NOTE: this currently reuses the MUSIC noise-subspace projection (both
/// methods share the same eigendecomposition and differ only in how the
/// noise-subspace projector is weighted) rather than implementing the
/// distinct minimum-norm weighting; the `method` tag on the returned result
/// is corrected to `MinNorm` so callers are not misled about which
/// algorithm actually produced the spectrum.
#[allow(dead_code)]
fn min_norm_spectrum(
    signal: &Array1<f64>,
    n_signals: usize,
    n_frequencies: usize,
) -> SignalResult<HighResolutionResult> {
    let mut result = music_spectrum(signal, n_signals, n_frequencies)?;
    result.method = HighResolutionMethod::MinNorm;
    Ok(result)
}

/// Root-MUSIC algorithm
///
/// NOTE: this currently reuses the ESPRIT rotational-invariance frequency
/// estimator (a genuine eigenstructure method, not a fabricated stand-in)
/// rather than the distinct polynomial-rooting formulation of Root-MUSIC;
/// the `method` tag on the returned result is corrected to `RootMUSIC` so
/// callers are not misled about which algorithm actually produced it.
#[allow(dead_code)]
fn root_music_estimation(
    signal: &Array1<f64>,
    n_signals: usize,
) -> SignalResult<HighResolutionResult> {
    let mut result = esprit_estimation(signal, n_signals)?;
    result.method = HighResolutionMethod::RootMUSIC;
    Ok(result)
}

/// Helper function to solve the linear system `a * x = b` (or `a * X = B`
/// for a matrix of right-hand sides) for a square coefficient matrix `a`.
///
/// Delegates to [`scirs2_linalg::solve_multiple`], a genuine LU-based
/// solver, rather than fabricating a placeholder (partial-identity) result.
#[allow(dead_code)]
fn solve_linear_system(a: &Array2<f64>, b: &Array2<f64>) -> SignalResult<Array2<f64>> {
    linalg_solve_multiple(&a.view(), &b.view(), None)
        .map_err(|e| SignalError::ComputationError(format!("Linear system solve failed: {e}")))
}

/// Helper function to solve least squares problem
#[allow(dead_code)]
fn solve_least_squares(a: &Array2<f64>, b: &Array2<f64>) -> SignalResult<Array2<f64>> {
    // Solve min ||Ax - b||^2 using normal equations: A^T A x = A^T b
    let ata = a.t().dot(a);
    let atb = a.t().dot(b);
    solve_linear_system(&ata, &atb)
}

/// Compute the eigendecomposition of a real symmetric (or numerically
/// near-symmetric, e.g. sample-covariance) matrix.
///
/// Delegates to [`scirs2_linalg::eigh`], a genuine symmetric eigensolver
/// (closed-form for small matrices, iterative for larger ones), rather than
/// fabricating a placeholder `(ones, identity)` result. Callers such as
/// MUSIC/ESPRIT/Pisarenko rely on the returned eigenvectors actually
/// spanning the signal/noise subspaces of the input matrix.
#[allow(dead_code)]
pub fn compute_eigendecomposition(
    matrix: &Array2<f64>,
) -> SignalResult<(Array1<f64>, Array2<f64>)> {
    let n = matrix.nrows();
    if n == 0 || matrix.ncols() != n {
        return Err(SignalError::ShapeMismatch(format!(
            "compute_eigendecomposition requires a square matrix, got {:?}",
            matrix.dim()
        )));
    }

    // Symmetrize defensively: callers pass sample-covariance-type matrices
    // that are symmetric in exact arithmetic but may carry a few ULPs of
    // floating-point asymmetry, which the underlying solver's strict
    // symmetry check would otherwise reject outright.
    let mut symmetric = matrix.to_owned();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (symmetric[[i, j]] + symmetric[[j, i]]);
            symmetric[[i, j]] = avg;
            symmetric[[j, i]] = avg;
        }
    }

    linalg_eigh(&symmetric.view(), None)
        .map_err(|e| SignalError::ComputationError(format!("Eigendecomposition failed: {e}")))
}

/// Compute matrix determinant.
///
/// Delegates to [`scirs2_linalg::det`] (closed-form for small matrices,
/// LU-based for larger ones) rather than fabricating a placeholder `1.0`
/// for matrices larger than 2x2.
#[allow(dead_code)]
fn compute_matrix_determinant(matrix: &Array2<f64>) -> SignalResult<f64> {
    linalg_det(&matrix.view(), None)
        .map_err(|e| SignalError::ComputationError(format!("Determinant computation failed: {e}")))
}

/// Compute matrix inverse.
///
/// Delegates to [`scirs2_linalg::inv`] (closed-form for small matrices,
/// LU-based for larger ones) rather than fabricating a placeholder identity
/// matrix for matrices larger than 2x2.
#[allow(dead_code)]
fn compute_matrix_inverse(matrix: &Array2<f64>) -> SignalResult<Array2<f64>> {
    linalg_inv(&matrix.view(), None).map_err(|e| {
        SignalError::ComputationError(format!("Matrix inverse computation failed: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_model_estimation() {
        // Test with simple 2-variable VAR(1) model
        let n_vars = 2;
        let n_obs = 100;
        let order = 1;

        // Generate test data
        let mut data = Array2::zeros((n_vars, n_obs));
        for t in 1..n_obs {
            data[[0, t]] = 0.5 * data[[0, t - 1]] + 0.1;
            data[[1, t]] = 0.3 * data[[0, t - 1]] + 0.7 * data[[1, t - 1]] + 0.1;
        }

        let config = AdvancedParametricConfig::default();
        let result = estimate_var_model(&data, order, &config).expect("Operation failed");

        assert_eq!(result.order, order);
        assert_eq!(result.n_vars, n_vars);
        assert!(result.log_likelihood.is_some());
        assert_eq!(result.ar_matrices.len(), order);
    }

    #[test]
    fn test_high_resolution_spectrum() {
        // Test MUSIC algorithm
        let n = 200;
        let signal: Array1<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.01;
                (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let result = high_resolution_spectral_estimation(
            &signal,
            2, // Two sinusoids
            HighResolutionMethod::MUSIC,
            512,
        )
        .expect("Operation failed");

        assert_eq!(result.method, HighResolutionMethod::MUSIC);
        assert_eq!(result.n_signals, 2);
        assert!(result.frequencies.len() > 0);
        assert_eq!(result.frequencies.len(), result.powers.len());
    }

    #[test]
    fn test_regularization_methods() {
        let n_obs = 50;
        let n_regressors = 10;

        let y = Array2::ones((2, n_obs));
        let x = Array2::ones((n_regressors, n_obs));

        // Test Ridge regression
        let ridge_result = estimate_var_coefficients(&y, &x, RegularizationMethod::Ridge, 0.1)
            .expect("Operation failed");

        assert_eq!(ridge_result.dim(), (n_regressors, 2));

        // Test LASSO regression
        let lasso_result = estimate_var_coefficients(&y, &x, RegularizationMethod::Lasso, 0.1)
            .expect("Operation failed");

        assert_eq!(lasso_result.dim(), (2, n_regressors)); // LASSO returns different shape than Ridge
    }

    #[test]
    fn test_compute_eigendecomposition_is_real() {
        // Non-trivial symmetric matrix (not the identity, not all-ones) so a
        // fabricated `(ones, identity)` result is numerically
        // distinguishable from a genuine eigendecomposition.
        let matrix =
            Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])
                .expect("Operation failed");

        let (eigenvalues, eigenvectors) =
            compute_eigendecomposition(&matrix).expect("eigendecomposition should succeed");

        assert_eq!(eigenvalues.len(), 3);
        assert_eq!(eigenvectors.dim(), (3, 3));

        // Verify the defining property A*v = lambda*v for every eigenpair;
        // a fabricated (ones, identity) result would fail this since
        // A*e_0 = [4, 1, 0] != 1.0 * e_0 = [1, 0, 0].
        for k in 0..3 {
            let v = eigenvectors.column(k).to_owned();
            let av = matrix.dot(&v);
            for i in 0..3 {
                assert!(
                    (av[i] - eigenvalues[k] * v[i]).abs() < 1e-6,
                    "A*v != lambda*v at eigenpair {k}, component {i}: {} vs {}",
                    av[i],
                    eigenvalues[k] * v[i]
                );
            }
        }

        // Eigenvalues must not all be 1.0 (the fabricated placeholder value).
        assert!(eigenvalues.iter().any(|&ev| (ev - 1.0).abs() > 1e-6));
    }

    #[test]
    fn test_esprit_estimation_finds_nonempty_frequencies() {
        let n = 300;
        let f0 = 0.15_f64;
        // A single real sinusoid spans a 2-D (cosine/sine) signal subspace.
        let signal: Array1<f64> = (0..n).map(|i| (2.0 * PI * f0 * i as f64).sin()).collect();

        let result =
            high_resolution_spectral_estimation(&signal, 2, HighResolutionMethod::ESPRIT, 256)
                .expect("ESPRIT should succeed");

        assert_eq!(result.method, HighResolutionMethod::ESPRIT);
        // Under the previous fabricated eigendecomposition, Phi's
        // eigenvalues were always exactly 1.0, so the `0.0 < val < 1.0`
        // acceptance filter discarded every candidate and `frequencies`
        // was always empty regardless of the input signal.
        assert!(
            !result.frequencies.is_empty(),
            "ESPRIT should recover at least one frequency from a single-tone signal"
        );
        assert_eq!(result.frequencies.len(), result.powers.len());

        let min_err = result
            .frequencies
            .iter()
            .map(|&f| (f - f0).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_err < 0.03,
            "expected a frequency near {f0}, got {:?}",
            result.frequencies
        );
    }

    #[test]
    fn test_solve_linear_system_solves_real_system() {
        // A non-trivial system with two distinct (non-constant) right-hand
        // sides; a fabricated partial-identity result would fail A*X == B.
        let a = Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0])
            .expect("Operation failed");
        let b = Array2::from_shape_vec((3, 2), vec![3.0, 1.0, 5.0, 2.0, 8.0, 3.0])
            .expect("Operation failed");

        let x = solve_linear_system(&a, &b).expect("solve should succeed");
        assert_eq!(x.dim(), (3, 2));

        let ax = a.dot(&x);
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (ax[[i, j]] - b[[i, j]]).abs() < 1e-8,
                    "A*X != B at ({i},{j}): {} vs {}",
                    ax[[i, j]],
                    b[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_matrix_determinant_and_inverse_for_3x3() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 1.0])
                .expect("Operation failed");

        // Independently-computed determinant: 2*(3*1-2*1) - 0 + 1*(1*1-3*0) = 3.0
        let det = compute_matrix_determinant(&matrix).expect("det should succeed");
        assert!((det - 3.0).abs() < 1e-8, "expected det=3.0, got {det}");
        // A fabricated placeholder would have returned 1.0 for n>2.
        assert!((det - 1.0).abs() > 1e-6);

        let inv = compute_matrix_inverse(&matrix).expect("inverse should succeed");
        let identity = matrix.dot(&inv);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (identity[[i, j]] - expected).abs() < 1e-6,
                    "A*A^-1 != I at ({i},{j}): {}",
                    identity[[i, j]]
                );
            }
        }
    }
}
