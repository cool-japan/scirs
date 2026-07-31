//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::prelude::*;

use scirs2_linalg::inv;

use super::functions::enhanced_system_identification;
use super::functions_5::{mad, median_helper};
use super::types::{
    AdaptiveIdentifier, EnhancedSysIdConfig, EnhancedSysIdResult, IdentificationMethod,
    ModelStructure, ParameterEstimate, RecursiveSysId, SystemModel,
};

#[cfg(test)]
mod tests {
    use super::super::functions::enhanced_system_identification;
    use super::super::functions_3::{
        advanced_model_selection, mimo_system_identification, robust_system_identification,
        simd_optimized_identification,
    };
    use super::super::types::{
        AdaptiveIdentifier, EnhancedSysIdConfig, ModelStructure, RecursiveSysId, SystemModel,
    };
    use super::{
        advanced_model_selection_enhanced, mimo_system_identification_advanced,
        robust_outlier_removal,
    };
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_recursive_sysid() {
        let config = EnhancedSysIdConfig::default();
        let mut sysid = RecursiveSysId::new(Array1::zeros(2), &config);
        for i in 0..10 {
            let input = (i as f64).sin();
            let output = 0.5 * input;
            let error = sysid.update(input, output).expect("Operation failed");
            assert!(error.is_finite());
        }
        let params = sysid.get_parameters();
        assert_eq!(params.len(), 2);
    }
    #[test]
    fn test_arx_identification() {
        // Use sinusoidal input so the system has zero DC steady-state and zero ICs are valid
        let n = 200;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin());
        let mut output = Array1::zeros(n);
        for i in 1..n {
            output[i] = 0.9 * output[i - 1] + 0.1 * input[i - 1];
        }
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            max_order: 2,
            order_selection: false,
            ..Default::default()
        };
        let result =
            enhanced_system_identification(&input, &output, &config).expect("Operation failed");
        assert!(matches!(result.model, SystemModel::ARX { .. }));
        assert!(result.validation.fit_percentage > 90.0);
    }
    #[test]
    fn test_cross_validation() {
        let n = 200;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin());
        let mut output = Array1::zeros(n);
        for i in 2..n {
            output[i] = 0.7 * output[i - 1] - 0.2 * output[i - 2] + 0.5 * input[i - 1];
        }
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            cv_folds: Some(5),
            max_order: 4,
            ..Default::default()
        };
        let result =
            enhanced_system_identification(&input, &output, &config).expect("Operation failed");
        assert!(result.validation.cv_fit.is_some());
        assert!(result.validation.cv_fit.expect("Operation failed") > 80.0);
    }
    #[test]
    fn test_robust_identification() {
        let n = 100;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin());
        let mut output = Array1::zeros(n);
        for i in 1..n {
            output[i] = 0.8 * output[i - 1] + 0.3 * input[i - 1];
        }
        output[20] += 5.0;
        output[50] -= 4.0;
        output[80] += 3.0;
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            outlier_detection: true,
            max_order: 2,
            ..Default::default()
        };
        let result =
            robust_system_identification(&input, &output, &config).expect("Operation failed");
        assert!(matches!(result.model, SystemModel::ARX { .. }));
        assert!(result.validation.fit_percentage > 85.0);
    }
    #[test]
    fn test_simd_optimization() {
        let n = 1000;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.01).sin());
        let mut output = Array1::zeros(n);
        for i in 1..n {
            output[i] = 0.9 * output[i - 1] + 0.2 * input[i - 1];
        }
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            parallel: true,
            max_order: 2,
            ..Default::default()
        };
        let result =
            simd_optimized_identification(&input, &output, &config).expect("Operation failed");
        assert!(matches!(result.model, SystemModel::ARX { .. }));
        assert!(result.validation.fit_percentage > 85.0);
        assert!(result.diagnostics.computation_time > 0);
    }
    #[test]
    fn test_mimo_identification() {
        let n = 100;
        let inputs = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                (i as f64 * 0.1).sin()
            } else {
                (i as f64 * 0.1).cos()
            }
        });
        let outputs = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                if i > 0 {
                    0.8 * i as f64 + 0.2 * inputs[[i - 1, 0]]
                } else {
                    0.0
                }
            } else {
                if i > 0 {
                    0.7 * i as f64 + 0.3 * inputs[[i - 1, 1]]
                } else {
                    0.0
                }
            }
        });
        let config = EnhancedSysIdConfig {
            model_structure: ModelStructure::ARX,
            max_order: 2,
            ..Default::default()
        };
        let results =
            mimo_system_identification(&inputs, &outputs, &config).expect("Operation failed");
        assert_eq!(results.len(), 2);
        for result in results {
            assert!(matches!(result.model, SystemModel::ARX { .. }));
        }
    }
    #[test]
    fn test_adaptive_identifier() {
        let mut identifier = AdaptiveIdentifier::new(EnhancedSysIdConfig::default());
        let n = 100;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.1).sin());
        let mut output = Array1::zeros(n);
        for i in 1..n {
            let coeff = if i < 50 { 0.8 } else { 0.6 };
            output[i] = coeff * output[i - 1] + 0.2 * input[i - 1];
        }
        let input1 = input.slice(scirs2_core::ndarray::s![..50]).to_owned();
        let output1 = output.slice(scirs2_core::ndarray::s![..50]).to_owned();
        let adapted1 = identifier
            .update_model(&input1, &output1)
            .expect("Operation failed");
        assert!(adapted1);
        let input2 = input.slice(scirs2_core::ndarray::s![50..]).to_owned();
        let output2 = output.slice(scirs2_core::ndarray::s![50..]).to_owned();
        let _adapted2 = identifier
            .update_model(&input2, &output2)
            .expect("Operation failed");
        assert!(identifier.get_current_model().is_some());
    }
    #[test]
    fn test_model_selection() {
        let n = 200;
        let input = Array1::from_shape_fn(n, |i| (i as f64 * 0.05).sin());
        let mut output = Array1::zeros(n);
        for i in 2..n {
            output[i] = 0.7 * output[i - 1] - 0.1 * output[i - 2] + 0.5 * input[i - 1];
        }
        let candidates = vec![
            ModelStructure::ARX,
            ModelStructure::ARMAX,
            ModelStructure::OE,
        ];
        let (best_structure, best_result) =
            advanced_model_selection(&input, &output, &candidates).expect("Operation failed");
        assert!(matches!(
            best_structure,
            ModelStructure::ARX | ModelStructure::ARMAX | ModelStructure::OE
        ));
        assert!(best_result.validation.fit_percentage > 70.0);
    }
}
/// ARX model identification implementation
#[allow(dead_code)]
fn identify_arx_complete(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    delay: usize,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    let n = input.len();
    let total_params = na + nb;
    if n <= total_params + delay {
        return Err(SignalError::ValueError(
            "Insufficient data for ARX identification".to_string(),
        ));
    }
    let start_idx = na.max(nb + delay);
    let data_len = n - start_idx;
    let mut phi = Array2::zeros((data_len, total_params));
    let mut y = Array1::zeros(data_len);
    for i in 0..data_len {
        let t = start_idx + i;
        y[i] = output[t];
        for j in 0..na {
            phi[[i, j]] = -output[t - j - 1];
        }
        for j in 0..nb {
            if t >= delay + j {
                phi[[i, na + j]] = input[t - delay - j];
            }
        }
    }
    let phi_t = phi.t();
    let phi_t_phi = phi_t.dot(&phi);
    let _phi_t_y = phi_t.dot(&y);
    let condition_number = estimate_condition_number(&phi_t_phi);
    if condition_number > 1e12 {
        eprintln!(
            "Warning: Ill-conditioned regression matrix (cond = {:.2e})",
            condition_number
        );
    }
    let theta = solve_least_squares(&phi, &y)?;
    let a_params = if na > 0 {
        theta.slice(s![..na]).to_owned()
    } else {
        Array1::zeros(0)
    };
    let b_params = if nb > 0 {
        theta.slice(s![na..]).to_owned()
    } else {
        Array1::zeros(0)
    };
    let y_pred = phi.dot(&theta);
    let residuals = &y - &y_pred;
    let cost = residuals.iter().map(|&r| r * r).sum::<f64>();
    let sigma2 = cost / (data_len as f64 - total_params as f64).max(1.0);
    let covariance = if condition_number < 1e10 {
        match invert_matrix(&phi_t_phi) {
            Ok(inv) => inv * sigma2,
            Err(_) => Array2::eye(total_params) * sigma2,
        }
    } else {
        Array2::eye(total_params) * sigma2
    };
    let std_errors = covariance.diag().mapv(|x| x.sqrt());
    let confidence_intervals: Vec<(f64, f64)> = theta
        .iter()
        .zip(std_errors.iter())
        .map(|(&param, &std_err)| {
            let margin = 1.96 * std_err;
            (param - margin, param + margin)
        })
        .collect();
    let model = SystemModel::ARX {
        a: a_params,
        b: b_params,
        delay,
    };
    let parameters = ParameterEstimate {
        values: theta,
        covariance,
        std_errors,
        confidence_intervals,
    };
    Ok((model, parameters, 1, true, cost))
}
/// MIMO system identification function
#[allow(dead_code)]
pub fn mimo_system_identification_advanced(
    inputs: &Array2<f64>,
    outputs: &Array2<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<Vec<EnhancedSysIdResult>> {
    let (n_samples, n_inputs) = inputs.dim();
    let (n_samples_out, n_outputs) = outputs.dim();
    if n_samples != n_samples_out {
        return Err(SignalError::ValueError(
            "Input and output sample counts must match".to_string(),
        ));
    }
    let mut results = Vec::with_capacity(n_outputs);
    for output_idx in 0..n_outputs {
        let output_vec = outputs.column(output_idx).to_owned();
        let input_vec = if n_inputs == 1 {
            inputs.column(0).to_owned()
        } else {
            inputs.column(0).to_owned()
        };
        let result = enhanced_system_identification(&input_vec, &output_vec, config)?;
        results.push(result);
    }
    Ok(results)
}
/// Advanced model selection function
#[allow(dead_code)]
pub fn advanced_model_selection_enhanced(
    input: &Array1<f64>,
    output: &Array1<f64>,
    candidates: &[ModelStructure],
) -> SignalResult<(ModelStructure, EnhancedSysIdResult)> {
    let mut best_aic = f64::INFINITY;
    let mut best_structure = candidates[0];
    let mut best_result = None;
    for &structure in candidates {
        let config = EnhancedSysIdConfig {
            model_structure: structure,
            cv_folds: Some(3),
            ..Default::default()
        };
        match enhanced_system_identification(input, output, &config) {
            Ok(result) => {
                if result.validation.aic < best_aic {
                    best_aic = result.validation.aic;
                    best_structure = structure;
                    best_result = Some(result);
                }
            }
            Err(_) => {
                continue;
            }
        }
    }
    match best_result {
        Some(result) => Ok((best_structure, result)),
        None => Err(SignalError::ComputationError(
            "No valid model could be identified".to_string(),
        )),
    }
}
/// Solve least squares problem using SVD for numerical stability
#[allow(dead_code)]
pub(super) fn solve_least_squares(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let at = a.t();
    let ata = at.dot(a);
    let atb = at.dot(b);
    match invert_matrix(&ata) {
        Ok(inv) => Ok(inv.dot(&atb)),
        Err(_) => {
            let pinv = pseudo_inverse(a)?;
            Ok(pinv.dot(b))
        }
    }
}
/// Estimate condition number of a matrix
#[allow(dead_code)]
fn estimate_condition_number(matrix: &Array2<f64>) -> f64 {
    let trace = matrix.diag().sum();
    let det_approx = matrix.diag().iter().product::<f64>().abs();
    if det_approx < 1e-15 {
        1e16
    } else {
        (trace / det_approx).abs()
    }
}
/// Matrix inversion using scirs2_linalg
#[allow(dead_code)]
pub(super) fn invert_matrix(matrix: &Array2<f64>) -> Result<Array2<f64>, SignalError> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(SignalError::ValueError("Matrix must be square".to_string()));
    }
    inv(&matrix.view(), None)
        .map_err(|e| SignalError::ComputationError(format!("Matrix inversion failed: {e}")))
}
/// Compute pseudo-inverse of a matrix
#[allow(dead_code)]
fn pseudo_inverse(matrix: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (m, n) = matrix.dim();
    if m >= n {
        let at = matrix.t();
        let ata = at.dot(matrix);
        let ata_inv = invert_matrix(&ata)?;
        Ok(ata_inv.dot(&at))
    } else {
        let at = matrix.t();
        let aat = matrix.dot(&at);
        let aat_inv = invert_matrix(&aat)?;
        Ok(at.dot(&aat_inv))
    }
}
/// Create companion form matrix for polynomial coefficients
#[allow(dead_code)]
fn companion_form_matrix(coeffs: &Array1<f64>) -> Array2<f64> {
    let n = coeffs.len();
    if n == 0 {
        return Array2::zeros((1, 1));
    }
    let mut companion = Array2::zeros((n, n));
    for i in 0..n {
        companion[[0, i]] = -coeffs[i];
    }
    for i in 1..n {
        companion[[i, i - 1]] = 1.0;
    }
    companion
}
/// Enhanced robust outlier detection and removal
#[allow(dead_code)]
pub(super) fn robust_outlier_removal(
    input: &Array1<f64>,
    output: &Array1<f64>,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    let n = input.len();
    let mut valid_indices = Vec::new();
    let input_median = median_helper(&input.to_vec());
    let output_median = median_helper(&output.to_vec());
    let input_mad = mad(&input.to_vec(), input_median);
    let output_mad = mad(&output.to_vec(), output_median);
    let input_threshold = 3.5 * input_mad;
    let output_threshold = 3.5 * output_mad;
    for i in 0..n {
        let input_dev = (input[i] - input_median).abs();
        let output_dev = (output[i] - output_median).abs();
        if input_dev <= input_threshold && output_dev <= output_threshold {
            valid_indices.push(i);
        }
    }
    if valid_indices.len() < n / 2 {
        return Err(SignalError::ValueError(
            "Too many outliers detected. Data may be corrupted.".to_string(),
        ));
    }
    let clean_input = Array1::from_iter(valid_indices.iter().map(|&i| input[i]));
    let clean_output = Array1::from_iter(valid_indices.iter().map(|&i| output[i]));
    Ok((clean_input, clean_output))
}
/// Compute variance of an Array1
fn array_variance(arr: &Array1<f64>) -> f64 {
    let n = arr.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = arr.sum() / n;
    arr.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
}

/// Estimate signal-to-noise ratio
#[allow(dead_code)]
pub(super) fn estimate_signal_noise_ratio(
    input: &Array1<f64>,
    output: &Array1<f64>,
) -> SignalResult<f64> {
    let n = input.len() as f64;
    let sum_x = input.sum();
    let sum_y = output.sum();
    let sum_xx = input.dot(input);
    let sum_xy = input.dot(output);
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    let mut residuals = Vec::with_capacity(input.len());
    for i in 0..input.len() {
        let predicted = slope * input[i] + intercept;
        residuals.push(output[i] - predicted);
    }
    let signal_power = array_variance(output);
    let noise_power = array_variance(&Array1::from_vec(residuals));
    if noise_power < 1e-15 {
        Ok(100.0)
    } else {
        Ok(10.0 * (signal_power / noise_power).log10())
    }
}
/// Select optimal identification method based on data characteristics
#[allow(dead_code)]
pub(super) fn select_optimal_method(
    input: &Array1<f64>,
    output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<IdentificationMethod> {
    let n = input.len();
    if n < 100 {
        return Ok(IdentificationMethod::RecursiveLeastSquares);
    }
    let snr = estimate_signal_noise_ratio(input, output)?;
    if snr < 10.0 {
        return Ok(IdentificationMethod::InstrumentalVariable);
    }
    if n > 1000 && snr > 20.0 {
        return Ok(IdentificationMethod::Subspace);
    }
    Ok(IdentificationMethod::PEM)
}
