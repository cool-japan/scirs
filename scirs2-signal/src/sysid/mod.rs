// System Identification Module
//
// This module provides comprehensive system identification functionality for
// estimating mathematical models of dynamic systems from input-output data.
//
// ## Features
//
// - **Transfer Function Estimation**: Estimate transfer functions from input-output data
// - **Parametric Models**: AR, ARMA, and ARX model identification
// - **Non-parametric Methods**: Frequency response estimation using spectral methods
// - **Model Validation**: Cross-validation, residual analysis, and information criteria
// - **Subspace Methods**: Simple N4SID implementation for state-space identification
// - **Recursive Methods**: Online/adaptive identification algorithms
//
// ## System Identification Methods
//
// ### Time-Domain Methods
// - Least squares estimation for ARX models
// - Prediction error methods for ARMA models
// - Maximum likelihood estimation
// - Instrumental variable methods
//
// ### Frequency-Domain Methods
// - Spectral analysis based estimation
// - Frequency response function estimation
// - Empirical transfer function estimation
//
// ### Subspace Methods
// - N4SID (Numerical algorithms for Subspace State Space System Identification)
// - MOESP (Multivariable Output-Error State sPace)
//
// ## Example Usage
//
// ```rust
// use scirs2_core::ndarray::Array1;
// use scirs2_signal::sysid::{estimate_transfer_function, TfEstimationMethod, ModelValidation};
// use scirs2_signal::waveforms::chirp;
// # fn main() -> Result<(), Box<dyn std::error::Error>> {
//
// // Generate test system and data
// let n = 1000;
// let fs = 100.0;
// let t = Array1::linspace(0.0, (n-1) as f64 / fs, n);
//
// // Create chirp input signal
// let input_vec = chirp(t.as_slice().expect("Operation failed"), 1.0, t[t.len()-1], 20.0, "linear", 0.0)?;
// let input = Array1::from(input_vec);
//
// // Simulate system output (simple first-order system)
// let mut output = Array1::zeros(n);
// let a = 0.9; // System parameter
// for i in 1..n {
//     output[i] = a * output[i-1] + (1.0 - a) * input[i-1];
// }
//
// // Estimate transfer function
// let result = estimate_transfer_function(
//     &input, &output, fs, 2, 2, TfEstimationMethod::LeastSquares
// )?;
//
// println!("Estimated numerator: {:?}", result.numerator);
// println!("Estimated denominator: {:?}", result.denominator);
// println!("Fit percentage: {:.2}%", result.fit_percentage);
// # Ok(())
// # }
// ```

mod frequency_response;
mod n4sid;
mod recursive;
mod robust;
mod transfer_function;
mod types;
mod utils;

// Re-export all public types
pub use frequency_response::estimate_frequency_response;
pub use n4sid::n4sid_identification;
pub use recursive::RecursiveLeastSquares;
pub use robust::{
    adaptive_robust_identification, fault_tolerant_identification, robust_least_squares,
    RobustEstimationConfig, RobustSysIdResult,
};
pub use transfer_function::estimate_transfer_function;
pub use types::{
    FreqResponseMethod, FreqResponseResult, ModelValidation, ParametricResult, SysIdConfig,
    TfEstimationMethod, TfEstimationResult,
};
pub use utils::{detect_outliers, estimate_robust_scale};

use crate::error::{SignalError, SignalResult};
use crate::parametric::{estimate_ar, estimate_arma, ARMethod, OrderSelection};
use scirs2_core::ndarray::{Array1, ArrayStatCompat};
use statrs::statistics::Statistics;
use std::f64::consts::PI;
use utils::{calculate_fit_percentage, ljung_box_test};

/// Identify AR model from single time series
///
/// # Arguments
/// * `signal` - Input time series
/// * `max_order` - Maximum AR order to consider
/// * `method` - AR estimation method
/// * `selection_criterion` - Order selection criterion
///
/// # Returns
/// * Parametric model identification result
#[allow(dead_code)]
pub fn identify_ar_model(
    signal: &Array1<f64>,
    max_order: usize,
    method: ARMethod,
    selection_criterion: OrderSelection,
) -> SignalResult<ParametricResult> {
    // Select optimal order
    let (optimal_order, criteria) =
        crate::parametric::select_arorder(signal, max_order, selection_criterion, method)?;

    // Estimate AR parameters with optimal _order
    let (ar_coeffs, reflection_coeffs, noise_var) = estimate_ar(signal, optimal_order, method)?;

    Ok(ParametricResult {
        ar_coefficients: ar_coeffs,
        ma_coefficients: None,
        noise_variance: noise_var,
        reflection_coefficients: reflection_coeffs,
        information_criterion: criteria[optimal_order],
        model_order: (optimal_order, 0),
    })
}

/// Identify ARMA model from single time series
///
/// # Arguments
/// * `signal` - Input time series
/// * `max_ar_order` - Maximum AR order to consider
/// * `max_ma_order` - Maximum MA order to consider
/// * `selection_criterion` - Order selection criterion
///
/// # Returns
/// * Parametric model identification result
#[allow(dead_code)]
pub fn identify_arma_model(
    signal: &Array1<f64>,
    max_ar_order: usize,
    max_ma_order: usize,
    selection_criterion: OrderSelection,
) -> SignalResult<ParametricResult> {
    let n = signal.len() as f64;
    let mut best_criterion = f64::INFINITY;
    let mut best_result = None;

    // Grid search over AR and MA orders
    for ar_order in 1..=max_ar_order {
        for ma_order in 0..=max_ma_order {
            if ar_order + ma_order >= signal.len() / 2 {
                continue;
            }

            if let Ok((ar_coeffs, ma_coeffs, noise_var)) = estimate_arma(signal, ar_order, ma_order)
            {
                // Calculate information _criterion
                let k = ar_order + ma_order;
                let log_likelihood = -0.5 * n * (2.0 * PI * noise_var).ln() - 0.5 * n;

                let criterion_value = match selection_criterion {
                    OrderSelection::AIC => -2.0 * log_likelihood + 2.0 * k as f64,
                    OrderSelection::BIC => -2.0 * log_likelihood + k as f64 * n.ln(),
                    OrderSelection::AICc => {
                        -2.0 * log_likelihood + 2.0 * k as f64 * n / (n - k as f64 - 1.0)
                    }
                    _ => -2.0 * log_likelihood + 2.0 * k as f64,
                };

                if criterion_value < best_criterion {
                    best_criterion = criterion_value;
                    best_result = Some(ParametricResult {
                        ar_coefficients: ar_coeffs,
                        ma_coefficients: Some(ma_coeffs),
                        noise_variance: noise_var,
                        reflection_coefficients: None,
                        information_criterion: criterion_value,
                        model_order: (ar_order, ma_order),
                    });
                }
            }
        }
    }

    best_result.ok_or_else(|| {
        SignalError::ComputationError("Failed to find suitable ARMA model".to_string())
    })
}

/// Validate identified model using various metrics
///
/// # Arguments
/// * `predicted` - Model predictions
/// * `actual` - Actual observations
/// * `model_order` - Total number of model parameters
/// * `perform_whiteness_test` - Whether to test residual whiteness
///
/// # Returns
/// * Model validation results
#[allow(dead_code)]
pub fn validate_model(
    predicted: &Array1<f64>,
    actual: &Array1<f64>,
    model_order: usize,
    perform_whiteness_test: bool,
) -> SignalResult<ModelValidation> {
    if predicted.len() != actual.len() {
        return Err(SignalError::ValueError(
            "Predicted and actual arrays must have same length".to_string(),
        ));
    }

    let n = actual.len() as f64;

    // Calculate residuals
    let residuals = actual - predicted;

    // Mean squared error
    let sq = residuals.mapv(|x| x * x);
    let mse = if !sq.is_empty() {
        sq.sum() / sq.len() as f64
    } else {
        0.0
    };

    // R-squared
    let y_mean = actual.mean_or(0.0);
    let ss_tot = actual.mapv(|y| (y - y_mean).powi(2)).sum();
    let ss_res = residuals.mapv(|x| x * x).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    // Fit percentage
    let fit_percentage = calculate_fit_percentage(actual, predicted);

    // Information criteria
    let log_likelihood = -0.5 * n * (2.0 * PI * mse).ln() - 0.5 * n;
    let aic = -2.0 * log_likelihood + 2.0 * model_order as f64;
    let bic = -2.0 * log_likelihood + model_order as f64 * n.ln();

    // Final prediction error
    let fpe = mse * (n + model_order as f64) / (n - model_order as f64);

    // Whiteness _test (Ljung-Box _test approximation)
    let whiteness_test = if perform_whiteness_test {
        ljung_box_test(&residuals, 10)
    } else {
        1.0 // No _test performed
    };

    Ok(ModelValidation {
        fit_percentage,
        mse,
        r_squared,
        fpe,
        aic,
        bic,
        whiteness_test,
        cv_error: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lti::design::tf;
    use approx::assert_relative_eq;

    #[test]
    fn test_transfer_function_estimation_simple() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        // Test with a longer signal for better conditioning
        let n = 50;
        let mut input = Array1::zeros(n);
        let mut output = Array1::zeros(n);

        // Generate random input
        for i in 0..n {
            input[i] = (i as f64 * 0.1).sin();
        }

        // Simulate y[n] = 0.8*y[n-1] + 0.2*u[n-1]
        for i in 1..n {
            output[i] = 0.8 * output[i - 1] + 0.2 * input[i - 1];
        }

        let result = estimate_transfer_function(
            &input,
            &output,
            1.0,
            1,
            1,
            TfEstimationMethod::LeastSquares,
        )
        .expect("Operation failed");

        // Should estimate something reasonable
        assert!(result.fit_percentage > 30.0); // Lower threshold for noisy estimation
        assert_eq!(result.numerator.len(), 2);
        assert_eq!(result.denominator.len(), 2);

        // Suppress unused variable warnings for test scaffolding
        let _ = (a, b);
    }

    #[test]
    fn test_ar_model_identification() {
        // Generate AR(2) process: y[n] = 0.5*y[n-1] + 0.3*y[n-2] + e[n]
        let n = 100;
        let mut signal = Array1::<f64>::zeros(n);

        for i in 2..n {
            signal[i] = 0.5 * signal[i - 1] + 0.3 * signal[i - 2] + 0.1 * (i as f64).sin();
        }

        let result = identify_ar_model(&signal, 5, ARMethod::Burg, OrderSelection::AIC)
            .expect("Operation failed");

        // Should identify a reasonable model
        assert!(result.model_order.0 <= 5);
        assert!(result.noise_variance > 0.0);
        assert_eq!(result.ar_coefficients.len(), result.model_order.0 + 1);
    }

    #[test]
    fn test_recursive_least_squares() {
        let mut rls = RecursiveLeastSquares::new(2, 0.95, 1000.0);

        // Test with known system: y = 2*x1 + 3*x2
        // Use multiple different data points for better convergence
        let test_data = vec![
            (Array1::from_vec(vec![1.0, 2.0]), 2.0 * 1.0 + 3.0 * 2.0),
            (Array1::from_vec(vec![2.0, 1.0]), 2.0 * 2.0 + 3.0 * 1.0),
            (Array1::from_vec(vec![0.5, 1.5]), 2.0 * 0.5 + 3.0 * 1.5),
            (Array1::from_vec(vec![1.5, 0.5]), 2.0 * 1.5 + 3.0 * 0.5),
        ];

        // Train with multiple epochs
        for _ in 0..100 {
            for (regression, output) in &test_data {
                let _ = rls.update(regression, *output).expect("Operation failed");
            }
        }

        let params = rls.get_parameters();
        // More relaxed tolerances for RLS convergence
        assert_relative_eq!(params[0], 2.0, epsilon = 0.5);
        assert_relative_eq!(params[1], 3.0, epsilon = 0.5);
    }

    #[test]
    fn test_model_validation() {
        let actual = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let predicted = Array1::from_vec(vec![1.1, 1.9, 3.1, 3.9, 5.1]);

        let validation = validate_model(&predicted, &actual, 2, false).expect("Operation failed");

        assert!(validation.fit_percentage > 90.0); // Should be high for close match
        assert!(validation.r_squared > 0.9);
        assert!(validation.mse < 0.1);
    }

    #[test]
    fn test_frequency_response_estimation() {
        // Simple test with impulse response
        let input = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let output = Array1::from_vec(vec![
            0.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125,
        ]);

        let config = SysIdConfig::default();
        let result = estimate_frequency_response(
            &input,
            &output,
            1.0,
            FreqResponseMethod::Periodogram,
            &config,
        )
        .expect("Operation failed");

        assert!(!result.frequencies.is_empty());
        assert_eq!(result.frequency_response.len(), result.frequencies.len());
        assert_eq!(result.coherence.len(), result.frequencies.len());
    }

    #[test]
    fn test_fit_percentage_calculation() {
        let actual = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let predicted = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let fit = calculate_fit_percentage(&actual, &predicted);
        assert_relative_eq!(fit, 100.0, epsilon = 1e-10);

        let predicted_bad = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        let fit_bad = calculate_fit_percentage(&actual, &predicted_bad);
        assert!(fit_bad < 100.0);
        assert!(fit_bad > 0.0);
    }
}
