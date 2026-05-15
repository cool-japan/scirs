// Robust system identification algorithms

use super::utils::{
    calculate_robust_fit, detect_outliers, estimate_robust_scale, solve_least_squares,
    solve_weighted_least_squares, update_huber_weights,
};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2};

/// Configuration for robust estimation methods
#[derive(Debug, Clone)]
pub struct RobustEstimationConfig {
    /// Outlier detection threshold (in standard deviations)
    pub outlier_threshold: f64,
    /// Huber loss threshold parameter
    pub huber_threshold: f64,
    /// Fraction of data to trim in LTS (0.0 to 0.5)
    pub lts_fraction: f64,
    /// Enable fault detection and isolation
    pub enable_fault_detection: bool,
    /// Maximum fraction of outliers to handle (0.0 to 0.5)
    pub max_outlier_fraction: f64,
    /// Breakdown point for robust estimators (0.0 to 0.5)
    pub breakdown_point: f64,
    /// Use iterative reweighting for M-estimators
    pub use_iterative_reweighting: bool,
}

impl Default for RobustEstimationConfig {
    fn default() -> Self {
        Self {
            outlier_threshold: 3.0,
            huber_threshold: 1.345,
            lts_fraction: 0.75,
            enable_fault_detection: true,
            max_outlier_fraction: 0.3,
            breakdown_point: 0.25,
            use_iterative_reweighting: true,
        }
    }
}

/// Robust system identification result
#[derive(Debug, Clone)]
pub struct RobustSysIdResult {
    /// Estimated model parameters
    pub parameters: Array1<f64>,
    /// Model fit percentage
    pub fit_percentage: f64,
    /// Robust residuals
    pub robust_residuals: Array1<f64>,
    /// Outlier indices detected
    pub outlier_indices: Vec<usize>,
    /// Robust scale estimate
    pub robust_scale: f64,
    /// Breakdown point achieved
    pub breakdown_point: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
}

/// Robust least squares estimation using M-estimators
///
/// This function performs robust parameter estimation using M-estimators with
/// Huber loss function to handle outliers in the data.
///
/// # Arguments
///
/// * `input` - Input signal data
/// * `output` - Output signal data
/// * `order` - Model order (number of parameters to estimate)
/// * `config` - Robust estimation configuration
///
/// # Returns
///
/// * Robust system identification result
#[allow(dead_code)]
pub fn robust_least_squares(
    input: &Array1<f64>,
    output: &Array1<f64>,
    order: usize,
    config: &RobustEstimationConfig,
) -> SignalResult<RobustSysIdResult> {
    if input.len() != output.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Input and output length mismatch: expected {}, got {}",
            input.len(),
            output.len()
        )));
    }

    if input.len() < order {
        return Err(SignalError::InvalidArgument(format!(
            "Order {} cannot exceed data length {}",
            order,
            input.len()
        )));
    }

    let n = input.len() - order;

    // Build regression matrix
    let mut regressor = Array2::zeros((n, order));
    for i in 0..n {
        for j in 0..order {
            regressor[[i, j]] = input[i + order - 1 - j];
        }
    }

    let target = output.slice(s![order..]).to_owned();

    // Initialize with standard least squares
    let mut parameters = solve_least_squares(&regressor, &target)?;
    let mut weights = Array1::ones(n);
    let mut robust_scale = estimate_robust_scale(&target, &regressor, &parameters)?;

    let mut converged = false;
    let mut iteration = 0;

    // Iteratively reweighted least squares
    while iteration < 50 && !converged {
        let residuals = &target - &regressor.dot(&parameters);

        // Update weights using Huber function
        update_huber_weights(
            &residuals,
            robust_scale,
            config.huber_threshold,
            &mut weights,
        );

        // Weighted least squares
        let new_parameters = solve_weighted_least_squares(&regressor, &target, &weights)?;

        // Check convergence
        let diff = &new_parameters - &parameters;
        let diff_norm = (diff.mapv(|x| x * x).sum()).sqrt();
        let params_norm = (parameters.mapv(|x| x * x).sum()).sqrt();
        let parameter_change = diff_norm / params_norm;
        converged = parameter_change < 1e-6;

        parameters = new_parameters;
        robust_scale = estimate_robust_scale(&target, &regressor, &parameters)?;
        iteration += 1;
    }

    // Calculate final residuals and detect outliers
    let residuals = &target - &regressor.dot(&parameters);
    let outlier_indices = detect_outliers(&residuals, robust_scale, config.outlier_threshold);
    let fit_percentage = calculate_robust_fit(&target, &regressor, &parameters);

    Ok(RobustSysIdResult {
        parameters,
        fit_percentage,
        robust_residuals: residuals,
        outlier_indices,
        robust_scale,
        breakdown_point: config.breakdown_point,
        iterations: iteration,
        converged,
    })
}

/// Fault-tolerant system identification with automatic outlier rejection
///
/// This function performs system identification while automatically detecting
/// and rejecting outliers, faults, and measurement errors.
#[allow(dead_code)]
pub fn fault_tolerant_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    order: usize,
    config: &RobustEstimationConfig,
) -> SignalResult<RobustSysIdResult> {
    if !config.enable_fault_detection {
        return robust_least_squares(input, output, order, config);
    }

    let n = input.len() - order;
    let mut regressor = Array2::zeros((n, order));
    for i in 0..n {
        for j in 0..order {
            regressor[[i, j]] = input[i + order - 1 - j];
        }
    }

    let target = output.slice(s![order..]).to_owned();

    // Phase 1: Initial robust estimation
    let initial_result = robust_least_squares(input, output, order, config)?;

    // Phase 2: Fault detection using residual analysis
    let mut outlier_mask = vec![false; n];
    for &idx in &initial_result.outlier_indices {
        if idx < n {
            outlier_mask[idx] = true;
        }
    }

    // Phase 3: Re-estimation without outliers
    let clean_indices: Vec<usize> = (0..n).filter(|&i| !outlier_mask[i]).collect();

    if clean_indices.len() < order {
        return Ok(initial_result); // Not enough clean data, return initial result
    }

    // Build clean regression matrix
    let mut clean_regressor = Array2::zeros((clean_indices.len(), order));
    let mut clean_target = Array1::zeros(clean_indices.len());

    for (i, &clean_idx) in clean_indices.iter().enumerate() {
        clean_target[i] = target[clean_idx];
        for j in 0..order {
            clean_regressor[[i, j]] = regressor[[clean_idx, j]];
        }
    }

    // Final robust estimation on clean data
    let final_parameters = solve_least_squares(&clean_regressor, &clean_target)?;
    let final_residuals = &target - &regressor.dot(&final_parameters);
    let final_scale = estimate_robust_scale(&target, &regressor, &final_parameters)?;
    let final_fit = calculate_robust_fit(&target, &regressor, &final_parameters);

    Ok(RobustSysIdResult {
        parameters: final_parameters,
        fit_percentage: final_fit,
        robust_residuals: final_residuals,
        outlier_indices: initial_result.outlier_indices,
        robust_scale: final_scale,
        breakdown_point: config.breakdown_point,
        iterations: initial_result.iterations + 1,
        converged: true,
    })
}

/// Adaptive robust system identification that automatically selects the best method
#[allow(dead_code)]
pub fn adaptive_robust_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    order: usize,
) -> SignalResult<RobustSysIdResult> {
    // Try different robust methods and select the best one
    let mut best_result: Option<RobustSysIdResult> = None;
    let mut best_fit = 0.0;

    let methods = vec![
        RobustEstimationConfig {
            huber_threshold: 1.345,
            ..Default::default()
        },
        RobustEstimationConfig {
            huber_threshold: 2.0,
            lts_fraction: 0.75,
            ..Default::default()
        },
        RobustEstimationConfig {
            huber_threshold: 1.0,
            lts_fraction: 0.8,
            enable_fault_detection: true,
            ..Default::default()
        },
    ];

    for config in methods {
        if let Ok(result) = fault_tolerant_identification(input, output, order, &config) {
            if result.fit_percentage > best_fit {
                best_fit = result.fit_percentage;
                best_result = Some(result);
            }
        }
    }

    best_result
        .ok_or_else(|| SignalError::ComputationError("All robust methods failed".to_string()))
}
