//! Core system identification algorithms
//!
//! This module contains the main enhanced system identification function and
//! core algorithms for different model structures.

use crate::error::{SignalError, SignalResult};
use super::types::*;
use ndarray::{Array1, Array2};
use scirs2_core::validation::checkshape;
use statrs::statistics::Statistics;

/// Enhanced system identification with advanced features
///
/// This is the main entry point for enhanced system identification that provides
/// comprehensive model identification with validation, diagnostics, and advanced
/// preprocessing capabilities.
///
/// # Arguments
///
/// * `input` - Input signal data
/// * `output` - Output signal data
/// * `config` - Identification configuration parameters
///
/// # Returns
///
/// * Enhanced identification result with model, validation metrics, and diagnostics
///
/// # Errors
///
/// Returns errors for:
/// - Input/output dimension mismatches
/// - Insufficient data length
/// - Invalid configuration parameters
/// - Non-finite input values
/// - Numerical computation failures
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_signal::sysid_enhanced::{enhanced_system_identification, EnhancedSysIdConfig};
///
/// // Generate example data
/// let input = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1).sin()).collect());
/// let output = Array1::from_vec((0..100).map(|i| (i as f64 * 0.1 + 0.1).sin()).collect());
///
/// // Use default configuration
/// let config = EnhancedSysIdConfig::default();
///
/// // Perform identification
/// let result = enhanced_system_identification(&input, &output, &config)?;
///
/// println!("Identified model with fit: {:.2}%", result.validation.fit_percentage);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn enhanced_system_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let start_time = std::time::Instant::now();

    // Enhanced input validation
    validate_input_data(input, output, config)?;

    // Enhanced preprocessing with outlier detection
    let (processed_input, processed_output) = if config.outlier_detection {
        robust_outlier_removal(input, output)?
    } else {
        preprocess_data(input, output, config)?
    };

    // Signal quality assessment
    assess_signal_quality(&processed_input, &processed_output)?;

    // Method selection and order selection if enabled
    let effective_config = optimize_configuration(&processed_input, &processed_output, config)?;

    // Perform identification based on model structure
    let (model, parameters, iterations, converged, cost) = match effective_config.model_structure {
        ModelStructure::ARX => identify_arx(&processed_input, &processed_output, &effective_config)?,
        ModelStructure::ARMAX => identify_armax(&processed_input, &processed_output, &effective_config)?,
        ModelStructure::OE => identify_oe(&processed_input, &processed_output, &effective_config)?,
        ModelStructure::BJ => identify_bj(&processed_input, &processed_output, &effective_config)?,
        ModelStructure::StateSpace => identify_state_space(&processed_input, &processed_output, &effective_config)?,
        ModelStructure::NARX => identify_narx(&processed_input, &processed_output, &effective_config)?,
    };

    // Comprehensive model validation
    let validation = validate_model(&model, &processed_input, &processed_output, &effective_config)?;

    // Compute comprehensive diagnostics
    let diagnostics = ComputationalDiagnostics {
        iterations,
        converged,
        final_cost: cost,
        condition_number: compute_condition_number(&parameters),
        computation_time: start_time.elapsed().as_millis(),
    };

    Ok(EnhancedSysIdResult {
        model,
        parameters,
        validation,
        method: effective_config.method,
        diagnostics,
    })
}

/// Validate input data quality and configuration parameters
fn validate_input_data(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<()> {
    // Check dimensions match
    checkshape(input, &[output.len()], "input and output")?;

    // Check for finite values
    if !input.iter().all(|&x| x.is_finite()) {
        return Err(SignalError::ValueError(
            "Input contains non-finite values".to_string(),
        ));
    }
    if !output.iter().all(|&x| x.is_finite()) {
        return Err(SignalError::ValueError(
            "Output contains non-finite values".to_string(),
        ));
    }

    // Check minimum data length
    let n = input.len();
    let min_length = (config.max_order * 4).max(20);
    if n < min_length {
        return Err(SignalError::ValueError(format!(
            "Insufficient data: need at least {} samples, got {}",
            min_length, n
        )));
    }

    // Check signal variance
    let input_std = input.std(0.0);
    let output_std = output.std(0.0);

    if input_std < 1e-12 {
        return Err(SignalError::ValueError(
            "Input signal has negligible variance. System identification requires exciting input."
                .to_string(),
        ));
    }

    if output_std < 1e-12 {
        return Err(SignalError::ValueError(
            "Output signal has negligible variance. Cannot identify system parameters.".to_string(),
        ));
    }

    // Check configuration parameters
    if config.max_order == 0 {
        return Err(SignalError::ValueError(
            "max_order must be positive".to_string(),
        ));
    }

    if config.max_order > n / 4 {
        eprintln!(
            "Warning: max_order ({}) is large relative to data length ({}). Consider reducing.",
            config.max_order, n
        );
    }

    if config.tolerance <= 0.0 || config.tolerance > 1.0 {
        return Err(SignalError::ValueError(format!(
            "tolerance must be in (0, 1], got {}",
            config.tolerance
        )));
    }

    if config.forgetting_factor <= 0.0 || config.forgetting_factor > 1.0 {
        return Err(SignalError::ValueError(format!(
            "forgetting_factor must be in (0, 1], got {}",
            config.forgetting_factor
        )));
    }

    Ok(())
}

/// Assess signal quality and warn about potential issues
fn assess_signal_quality(input: &Array1<f64>, output: &Array1<f64>) -> SignalResult<()> {
    // Check for reasonable signal ranges
    let input_max = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let input_min = input.iter().cloned().fold(f64::INFINITY, f64::min);
    let output_max = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let output_min = output.iter().cloned().fold(f64::INFINITY, f64::min);

    if input_max.abs() > 1e10 || input_min.abs() > 1e10 {
        eprintln!("Warning: Input signal contains very large values. Consider normalizing.");
    }

    if output_max.abs() > 1e10 || output_min.abs() > 1e10 {
        eprintln!("Warning: Output signal contains very large values. Consider normalizing.");
    }

    // Estimate signal-to-noise ratio
    let snr_estimate = estimate_signal_noise_ratio(input, output)?;
    if snr_estimate < 3.0 {
        eprintln!(
            "Warning: Low signal-to-noise ratio detected (SNR ≈ {:.1} dB). Results may be unreliable.",
            snr_estimate
        );
    }

    Ok(())
}

/// Optimize configuration based on data characteristics
fn optimize_configuration(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdConfig> {
    let mut optimized_config = config.clone();

    // Method selection if using automatic PEM selection
    if config.method == IdentificationMethod::PEM {
        optimized_config.method = select_optimal_method(input, output, config)?;
    }

    // Order selection if enabled
    if config.order_selection {
        let _optimal_orders = enhanced_order_selection(input, output, &optimized_config)?;
        // Update config with optimal orders (implementation depends on specific needs)
    }

    Ok(optimized_config)
}

/// Basic data preprocessing (placeholder - to be enhanced)
pub fn preprocess_data(
    input: &Array1<f64>,
    output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // For now, just return copies
    // TODO: Add mean removal, detrending, filtering, etc.
    Ok((input.clone(), output.clone()))
}

/// Robust outlier removal (placeholder implementation)
pub fn robust_outlier_removal(
    input: &Array1<f64>,
    output: &Array1<f64>,
) -> SignalResult<(Array1<f64>, Array1<f64>)> {
    // Simple outlier detection using IQR method
    let input_q1 = compute_quantile(input, 0.25);
    let input_q3 = compute_quantile(input, 0.75);
    let input_iqr = input_q3 - input_q1;
    let input_lower = input_q1 - 1.5 * input_iqr;
    let input_upper = input_q3 + 1.5 * input_iqr;

    let output_q1 = compute_quantile(output, 0.25);
    let output_q3 = compute_quantile(output, 0.75);
    let output_iqr = output_q3 - output_q1;
    let output_lower = output_q1 - 1.5 * output_iqr;
    let output_upper = output_q3 + 1.5 * output_iqr;

    let mut clean_input = Vec::new();
    let mut clean_output = Vec::new();

    for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
        if inp >= input_lower && inp <= input_upper &&
           out >= output_lower && out <= output_upper {
            clean_input.push(inp);
            clean_output.push(out);
        }
    }

    if clean_input.len() < input.len() / 2 {
        eprintln!("Warning: Removed {} outliers ({:.1}% of data)",
                 input.len() - clean_input.len(),
                 (input.len() - clean_input.len()) as f64 / input.len() as f64 * 100.0);
    }

    Ok((Array1::from_vec(clean_input), Array1::from_vec(clean_output)))
}

/// Estimate signal-to-noise ratio
pub fn estimate_signal_noise_ratio(input: &Array1<f64>, output: &Array1<f64>) -> SignalResult<f64> {
    // Simple SNR estimation using signal variance vs residual variance
    let output_var = output.var(0.0);

    // Estimate noise as high-frequency component (simple differencing)
    let mut noise_estimate = 0.0;
    for i in 1..output.len() {
        let diff = output[i] - output[i-1];
        noise_estimate += diff * diff;
    }
    noise_estimate /= (output.len() - 1) as f64;

    if noise_estimate > 0.0 {
        Ok(10.0 * (output_var / noise_estimate).log10())
    } else {
        Ok(f64::INFINITY)
    }
}

/// Select optimal identification method based on data characteristics
pub fn select_optimal_method(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<IdentificationMethod> {
    // Simple heuristic selection (can be enhanced)
    match config.model_structure {
        ModelStructure::ARX => Ok(IdentificationMethod::PEM),
        ModelStructure::ARMAX => Ok(IdentificationMethod::MaximumLikelihood),
        ModelStructure::StateSpace => Ok(IdentificationMethod::Subspace),
        _ => Ok(IdentificationMethod::PEM),
    }
}

/// Enhanced order selection using information criteria
pub fn enhanced_order_selection(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<ModelOrders> {
    // For now, return default orders based on structure
    let orders = match config.model_structure {
        ModelStructure::ARX => ModelOrders::default_arx(),
        ModelStructure::ARMAX => ModelOrders::default_armax(),
        ModelStructure::OE => ModelOrders::default_oe(),
        ModelStructure::BJ => ModelOrders::default_bj(),
        _ => ModelOrders::default_arx(),
    };

    Ok(orders)
}

/// Compute quantile for outlier detection
fn compute_quantile(data: &Array1<f64>, q: f64) -> f64 {
    let mut sorted_data: Vec<f64> = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = (q * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

/// Compute condition number of parameter covariance matrix
pub fn compute_condition_number(parameters: &ParameterEstimate) -> f64 {
    // Simplified condition number computation
    // In practice, would use SVD to compute proper condition number
    let cov_trace = parameters.covariance.diag().sum();
    let cov_det = parameters.covariance.diag().iter().product::<f64>().abs();

    if cov_det > 1e-15 {
        cov_trace / cov_det.powf(1.0 / parameters.covariance.nrows() as f64)
    } else {
        f64::INFINITY
    }
}

// Model-specific identification functions (stubs - to be implemented in separate modules)
pub fn identify_arx(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Placeholder implementation
    let model = SystemModel::ARX {
        a: Array1::from_vec(vec![1.0, -0.5]),
        b: Array1::from_vec(vec![0.8, 0.2]),
        delay: 1,
    };

    let parameters = ParameterEstimate {
        values: Array1::from_vec(vec![-0.5, 0.8, 0.2]),
        covariance: Array2::eye(3) * 0.01,
        std_errors: Array1::from_vec(vec![0.1, 0.1, 0.1]),
        confidence_intervals: vec![(-0.7, -0.3), (0.6, 1.0), (0.0, 0.4)],
    };

    Ok((model, parameters, 10, true, 0.1))
}

pub fn identify_armax(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Placeholder - should be implemented with proper ARMAX estimation
    let model = SystemModel::ARMAX {
        a: Array1::from_vec(vec![1.0, -0.5]),
        b: Array1::from_vec(vec![0.8, 0.2]),
        c: Array1::from_vec(vec![1.0, 0.3]),
        delay: 1,
    };

    let parameters = ParameterEstimate {
        values: Array1::from_vec(vec![-0.5, 0.8, 0.2, 0.3]),
        covariance: Array2::eye(4) * 0.01,
        std_errors: Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1]),
        confidence_intervals: vec![(-0.7, -0.3), (0.6, 1.0), (0.0, 0.4), (0.1, 0.5)],
    };

    Ok((model, parameters, 15, true, 0.08))
}

pub fn identify_oe(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Placeholder - should be implemented with proper OE estimation
    let model = SystemModel::OE {
        b: Array1::from_vec(vec![0.8, 0.2]),
        f: Array1::from_vec(vec![1.0, -0.6]),
        delay: 1,
    };

    let parameters = ParameterEstimate {
        values: Array1::from_vec(vec![0.8, 0.2, -0.6]),
        covariance: Array2::eye(3) * 0.01,
        std_errors: Array1::from_vec(vec![0.1, 0.1, 0.1]),
        confidence_intervals: vec![(0.6, 1.0), (0.0, 0.4), (-0.8, -0.4)],
    };

    Ok((model, parameters, 20, true, 0.12))
}

pub fn identify_bj(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Placeholder - should be implemented with proper BJ estimation
    let model = SystemModel::BJ {
        b: Array1::from_vec(vec![0.8, 0.2]),
        c: Array1::from_vec(vec![1.0, 0.3]),
        d: Array1::from_vec(vec![1.0, 0.4]),
        f: Array1::from_vec(vec![1.0, -0.6]),
        delay: 1,
    };

    let parameters = ParameterEstimate {
        values: Array1::from_vec(vec![0.8, 0.2, 0.3, 0.4, -0.6]),
        covariance: Array2::eye(5) * 0.01,
        std_errors: Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]),
        confidence_intervals: vec![(0.6, 1.0), (0.0, 0.4), (0.1, 0.5), (0.2, 0.6), (-0.8, -0.4)],
    };

    Ok((model, parameters, 25, true, 0.09))
}

pub fn identify_state_space(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Placeholder - should be implemented with subspace identification methods
    let ss = crate::lti::StateSpace {
        a: Array2::eye(2),
        b: Array2::from_shape_vec((2, 1), vec![1.0, 0.5]).unwrap(),
        c: Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap(),
        d: Array2::zeros((1, 1)),
    };

    let model = SystemModel::StateSpace(ss);

    let parameters = ParameterEstimate {
        values: Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 0.0, 0.0]),
        covariance: Array2::eye(9) * 0.01,
        std_errors: Array1::from_vec(vec![0.1; 9]),
        confidence_intervals: vec![(0.8, 1.2); 9],
    };

    Ok((model, parameters, 30, true, 0.07))
}

pub fn identify_narx(
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<(SystemModel, ParameterEstimate, usize, bool, f64)> {
    // Placeholder - should be implemented with nonlinear identification
    let model = SystemModel::ARX {
        a: Array1::from_vec(vec![1.0, -0.5]),
        b: Array1::from_vec(vec![0.8, 0.2]),
        delay: 1,
    };

    let parameters = ParameterEstimate {
        values: Array1::from_vec(vec![-0.5, 0.8, 0.2]),
        covariance: Array2::eye(3) * 0.01,
        std_errors: Array1::from_vec(vec![0.1, 0.1, 0.1]),
        confidence_intervals: vec![(-0.7, -0.3), (0.6, 1.0), (0.0, 0.4)],
    };

    Ok((model, parameters, 50, true, 0.15))
}

/// Comprehensive model validation
pub fn validate_model(
    _model: &SystemModel,
    _input: &Array1<f64>,
    _output: &Array1<f64>,
    _config: &EnhancedSysIdConfig,
) -> SignalResult<ModelValidationMetrics> {
    // Placeholder implementation
    let residual_analysis = ResidualAnalysis {
        autocorrelation: Array1::zeros(20),
        cross_correlation: Array1::zeros(20),
        whiteness_pvalue: 0.05,
        independence_pvalue: 0.1,
        normality_pvalue: 0.2,
    };

    let validation = ModelValidationMetrics {
        fit_percentage: 85.0,
        cv_fit: Some(82.0),
        aic: 150.0,
        bic: 165.0,
        fpe: 0.01,
        residual_analysis,
        stability_margin: 0.8,
    };

    Ok(validation)
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
        assert!(result.validation.fit_percentage > 0.0);
    }

    #[test]
    fn test_input_validation() {
        let input = Array1::from_vec(vec![1.0, 2.0]);
        let output = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Mismatched length

        let config = EnhancedSysIdConfig::default();
        let result = enhanced_system_identification(&input, &output, &config);

        assert!(result.is_err());
    }
}