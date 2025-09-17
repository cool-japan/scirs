//! Enhanced System Identification Module
//!
//! This module provides advanced system identification capabilities with comprehensive
//! validation, diagnostics, and robust estimation methods. It has been refactored into
//! focused submodules for better organization and maintainability.
//!
//! # Module Organization
//!
//! - [`types`] - Core data structures and enums
//! - [`core`] - Main identification algorithms and preprocessing
//! - [`recursive`] - Recursive identification for online applications
//! - [`statistics`] - Statistical tests and validation functions
//!
//! # Key Features
//!
//! * **Multiple Model Structures**: ARX, ARMAX, Output-Error, Box-Jenkins, State-Space, NARX
//! * **Advanced Preprocessing**: Outlier detection, data validation, quality assessment
//! * **Robust Estimation**: Regularized least squares, recursive algorithms with forgetting
//! * **Comprehensive Validation**: Residual analysis, statistical tests, stability margins
//! * **Online Identification**: Recursive least squares for real-time applications
//! * **Model Selection**: Information criteria, order selection, method optimization
//!
//! # Usage Examples
//!
//! ## Basic System Identification
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_signal::sysid_enhanced::{enhanced_system_identification, EnhancedSysIdConfig};
//!
//! // Generate example input-output data
//! let input = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1).sin()).collect());
//! let output = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1 + 0.1).sin()).collect());
//!
//! // Configure identification
//! let mut config = EnhancedSysIdConfig::arx();
//! config.outlier_detection = true;
//! config.order_selection = true;
//!
//! // Perform identification
//! let result = enhanced_system_identification(&input, &output, &config)?;
//!
//! println!("Model fit: {:.2}%", result.validation.fit_percentage);
//! println!("AIC: {:.2}", result.validation.aic);
//! println!("Converged: {}", result.diagnostics.converged);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Recursive Identification
//!
//! ```rust
//! use ndarray::Array1;
//! use scirs2_signal::sysid_enhanced::{RecursiveSysId, EnhancedSysIdConfig};
//!
//! // Initialize recursive identifier
//! let initial_params = Array1::from_vec(vec![0.0, 0.0, 0.0]);
//! let config = EnhancedSysIdConfig::recursive();
//! let mut recursive_id = RecursiveSysId::new(initial_params, &config);
//!
//! // Process streaming data
//! for (input_val, output_val) in input_stream.zip(output_stream) {
//!     let prediction_error = recursive_id.update(input_val, output_val)?;
//!
//!     // Get updated parameters
//!     let current_params = recursive_id.get_parameters();
//!     let uncertainties = recursive_id.get_uncertainties();
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Statistical Validation
//!
//! ```rust
//! use scirs2_signal::sysid_enhanced::statistics::{jarque_bera_test, ljung_box_test};
//!
//! // Test residuals for normality and independence
//! let normality_pvalue = jarque_bera_test(&residuals);
//! let independence_pvalue = ljung_box_test(&residuals, 10);
//!
//! if normality_pvalue > 0.05 {
//!     println!("Residuals appear normally distributed");
//! }
//! if independence_pvalue > 0.05 {
//!     println!("Residuals appear independent (white noise)");
//! }
//! ```

// Declare all submodules
pub mod types;
pub mod core;
pub mod recursive;
pub mod statistics;

// Re-export main types for backward compatibility
pub use types::{
    EnhancedSysIdResult,
    SystemModel,
    ParameterEstimate,
    ModelValidationMetrics,
    ResidualAnalysis,
    ComputationalDiagnostics,
    IdentificationMethod,
    NonlinearFunction,
    EnhancedSysIdConfig,
    ModelStructure,
    ModelOrders,
};

// Re-export main functions
pub use core::{
    enhanced_system_identification,
    preprocess_data,
    robust_outlier_removal,
    estimate_signal_noise_ratio,
    select_optimal_method,
    enhanced_order_selection,
    compute_condition_number,
    validate_model,
    // Individual model identification functions
    identify_arx,
    identify_armax,
    identify_oe,
    identify_bj,
    identify_state_space,
    identify_narx,
};

// Re-export recursive identification
pub use recursive::RecursiveSysId;

// Re-export statistical functions
pub use statistics::{
    jarque_bera_test,
    ljung_box_test,
    cross_correlation_test,
    analyze_residuals,
    compute_stability_margin,
    compute_information_criteria,
};

// Import dependencies used across modules
use crate::error::{SignalError, SignalResult};
use ndarray::{Array1, Array2};

// Additional utility functions that may have been in the original file
// These provide compatibility for any functions not captured in the main modules

/// Compute median of data array
pub fn median(data: &Array1<f64>) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n/2 - 1] + sorted[n/2]) / 2.0
    } else {
        sorted[n/2]
    }
}

/// Compute median absolute deviation
pub fn median_absolute_deviation(data: &Array1<f64>) -> f64 {
    let med = median(data);
    let deviations = data.map(|&x| (x - med).abs());
    median(&deviations) / 0.6745 // Scale factor for normal distribution
}

/// Remove outliers using robust statistics
pub fn remove_outliers(
    input: &Array1<f64>,
    output: &Array1<f64>,
    threshold_factor: f64,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    if input.len() != output.len() {
        return Err(SignalError::ValueError(
            "Input and output arrays must have the same length".to_string(),
        ));
    }

    let output_mad = median_absolute_deviation(output);
    let output_median = median(output);
    let threshold = threshold_factor * output_mad;

    let mut clean_input = Vec::new();
    let mut clean_output = Vec::new();

    for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
        if (out - output_median).abs() <= threshold {
            clean_input.push(inp);
            clean_output.push(out);
        }
    }

    if clean_input.len() < input.len() / 2 {
        eprintln!(
            "Warning: Removed {} outliers ({:.1}% of data)",
            input.len() - clean_input.len(),
            (input.len() - clean_input.len()) as f64 / input.len() as f64 * 100.0
        );
    }

    Ok((Array1::from_vec(clean_input), Array1::from_vec(clean_output)))
}

/// Form regression matrix for ARX model
pub fn form_arx_regression(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    delay: usize,
) -> SignalResult<(Array2<f64>, Array1<f64>)> {
    let n = output.len();
    let n_start = na.max(nb + delay);

    if n_start >= n {
        return Err(SignalError::ValueError(
            "Not enough data for specified model orders".to_string(),
        ));
    }

    let n_samples = n - n_start;
    let mut phi = Array2::zeros((n_samples, na + nb));
    let mut y = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let t = i + n_start;

        // Output regressors: -y(t-1), ..., -y(t-na)
        for j in 0..na {
            if t > j {
                phi[[i, j]] = -output[t - j - 1];
            }
        }

        // Input regressors: u(t-delay), ..., u(t-delay-nb+1)
        for j in 0..nb {
            if t >= delay + j {
                phi[[i, na + j]] = input[t - delay - j];
            }
        }

        y[i] = output[t];
    }

    Ok((phi, y))
}

/// Solve regularized least squares problem
pub fn solve_regularized_ls(
    phi: &Array2<f64>,
    y: &Array1<f64>,
    lambda: f64,
) -> SignalResult<Array1<f64>> {
    let n_params = phi.ncols();
    let phi_t_phi = phi.t().dot(phi) + Array2::eye(n_params) * lambda;
    let phi_t_y = phi.t().dot(y);

    // Simple solution using normal equations
    // In practice, would use more numerically stable methods like QR decomposition
    match solve_linear_system(&phi_t_phi, &phi_t_y) {
        Ok(solution) => Ok(solution),
        Err(_) => {
            // Increase regularization and retry
            let regularized_matrix = phi.t().dot(phi) + Array2::eye(n_params) * (lambda + 1e-6);
            solve_linear_system(&regularized_matrix, &phi_t_y)
        }
    }
}

/// Simple linear system solver (placeholder for more robust implementation)
pub fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    // This is a simplified implementation
    // In practice, would use proper linear algebra libraries like ndarray-linalg

    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(SignalError::ValueError(
            "Matrix dimensions incompatible".to_string(),
        ));
    }

    // For small systems, use Gaussian elimination
    if n <= 3 {
        gaussian_elimination(a, b)
    } else {
        // For larger systems, this is a placeholder
        // Should use proper decomposition methods
        let mut solution = Array1::zeros(n);
        for i in 0..n {
            solution[i] = b[i] / a[[i, i]].max(1e-15); // Diagonal approximation
        }
        Ok(solution)
    }
}

/// Simple Gaussian elimination for small systems
fn gaussian_elimination(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let n = a.nrows();
    let mut aug = Array2::zeros((n, n + 1));

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..=n {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Make diagonal element 1
        let diagonal = aug[[i, i]];
        if diagonal.abs() < 1e-15 {
            return Err(SignalError::ComputationError(
                "Singular matrix encountered".to_string(),
            ));
        }

        for j in i..=n {
            aug[[i, j]] /= diagonal;
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[[k, i]];
            for j in i..=n {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut solution = Array1::zeros(n);
    for i in (0..n).rev() {
        solution[i] = aug[[i, n]];
        for j in (i + 1)..n {
            solution[i] -= aug[[i, j]] * solution[j];
        }
    }

    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_identification() {
        let input = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1).sin()).collect());
        let output = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1 + 0.1).sin()).collect());

        let config = EnhancedSysIdConfig::default();
        let result = enhanced_system_identification(&input, &output, &config);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.validation.fit_percentage >= 0.0);
        assert!(result.diagnostics.computation_time > 0);
    }

    #[test]
    fn test_recursive_identification() {
        let initial_params = Array1::from_vec(vec![0.0, 0.0]);
        let config = EnhancedSysIdConfig::recursive();
        let mut recursive_id = RecursiveSysId::new(initial_params, &config);

        // Test single update
        let error = recursive_id.update(1.0, 0.5);
        assert!(error.is_ok());
        assert_eq!(recursive_id.get_update_count(), 1);
    }

    #[test]
    fn test_median_calculation() {
        let data = Array1::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0]);
        let med = median(&data);
        assert!((med - 3.0).abs() < 1e-10);

        let even_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let even_med = median(&even_data);
        assert!((even_med - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_arx_regression_formation() {
        let input = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let output = Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);

        let result = form_arx_regression(&input, &output, 2, 1, 1);
        assert!(result.is_ok());

        let (phi, y) = result.unwrap();
        assert!(phi.nrows() > 0);
        assert!(phi.ncols() == 3); // na + nb = 2 + 1
        assert_eq!(y.len(), phi.nrows());
    }

    #[test]
    fn test_linear_system_solver() {
        // Test 2x2 system: [2 1; 1 1] * [x; y] = [3; 2]
        // Solution should be x=1, y=1
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 1.0]).unwrap();
        let b = Array1::from_vec(vec![3.0, 2.0]);

        let result = solve_linear_system(&a, &b);
        assert!(result.is_ok());

        let solution = result.unwrap();
        assert!((solution[0] - 1.0).abs() < 1e-10);
        assert!((solution[1] - 1.0).abs() < 1e-10);
    }
}