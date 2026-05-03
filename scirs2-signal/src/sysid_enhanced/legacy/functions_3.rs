//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{SignalError, SignalResult};
use crate::lti::{StateSpace, TransferFunction};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_linalg::inv;

use super::functions::enhanced_system_identification;
use super::functions::median;
use super::functions_2::simulate_model;
use super::functions_5::mad;
use super::types::{
    EnhancedSysIdConfig, EnhancedSysIdResult, ModelStructure, ParameterEstimate, SystemModel,
};

/// Convert transfer function to state space representation
#[allow(dead_code)]
pub(super) fn transfer_function_to_state_space(tf: &TransferFunction) -> SignalResult<StateSpace> {
    let num = &tf.num;
    let den = &tf.den;
    if den.is_empty() || den[0] == 0.0 {
        return Err(SignalError::ValueError(
            "Invalid denominator in transfer function".to_string(),
        ));
    }
    let n = den.len() - 1;
    if n == 0 {
        let gain = if num.is_empty() { 0.0 } else { num[0] / den[0] };
        return Ok(StateSpace {
            a: vec![],
            b: vec![],
            c: vec![],
            d: vec![gain],
            n_states: 0,
            n_inputs: 1,
            n_outputs: 1,
            dt: tf.dt,
        });
    }
    let d0 = den[0];
    let norm_den: Vec<f64> = den.iter().map(|&x| x / d0).collect();
    let norm_num: Vec<f64> = num.iter().map(|&x| x / d0).collect();
    let mut a = Array2::zeros((n, n));
    let mut b = Array2::zeros((n, 1));
    let mut c = Array2::zeros((1, n));
    let mut d = Array2::zeros((1, 1));
    for i in 0..n - 1 {
        a[[i + 1, i]] = 1.0;
    }
    for i in 0..n {
        a[[0, n - 1 - i]] = -norm_den[i + 1];
    }
    b[[0, 0]] = 1.0;
    if norm_num.len() > n {
        d[[0, 0]] = norm_num[0];
        for i in 0..n {
            if i + 1 < norm_num.len() {
                c[[0, n - 1 - i]] = norm_num[i + 1] - norm_num[0] * norm_den[i + 1];
            } else {
                c[[0, n - 1 - i]] = -norm_num[0] * norm_den[i + 1];
            }
        }
    } else {
        d[[0, 0]] = 0.0;
        for i in 0..n {
            if i < norm_num.len() {
                c[[0, n - 1 - i]] = norm_num[i];
            }
        }
    }
    Ok(StateSpace {
        a: a.iter().copied().collect(),
        b: b.iter().copied().collect(),
        c: c.iter().copied().collect(),
        d: d.iter().copied().collect(),
        n_states: n,
        n_inputs: 1,
        n_outputs: 1,
        dt: tf.dt,
    })
}
/// Simulate state space system
#[allow(dead_code)]
pub(super) fn simulate_state_space(
    ss: &StateSpace,
    input: &Array1<f64>,
) -> SignalResult<Array1<f64>> {
    let n_states = ss.n_states;
    let n_inputs = ss.n_inputs;
    let n_outputs = ss.n_outputs;
    let n_samples = input.len();
    if n_inputs != 1 {
        return Err(SignalError::ValueError(
            "Only single-input systems supported".to_string(),
        ));
    }
    if n_outputs != 1 {
        return Err(SignalError::ValueError(
            "Only single-output systems supported".to_string(),
        ));
    }
    let mut x = vec![0.0f64; n_states];
    let mut output = Array1::zeros(n_samples);
    // Helper closures to index into row-major flat matrices
    // a: n_states x n_states
    let a_get = |i: usize, j: usize| ss.a.get(i * n_states + j).copied().unwrap_or(0.0);
    // b: n_states x n_inputs (n_inputs == 1)
    let b_get = |i: usize| ss.b.get(i).copied().unwrap_or(0.0);
    // c: n_outputs x n_states (n_outputs == 1)
    let c_get = |i: usize| ss.c.get(i).copied().unwrap_or(0.0);
    // d: n_outputs x n_inputs (scalar)
    let d_val = ss.d.first().copied().unwrap_or(0.0);
    for k in 0..n_samples {
        let mut y = d_val * input[k];
        for i in 0..n_states {
            y += c_get(i) * x[i];
        }
        output[k] = y;
        let mut x_next = vec![0.0f64; n_states];
        for i in 0..n_states {
            x_next[i] = b_get(i) * input[k];
            for j in 0..n_states {
                x_next[i] += a_get(i, j) * x[j];
            }
        }
        x = x_next;
    }
    Ok(output)
}
/// Get number of model parameters
#[allow(dead_code)]
pub(super) fn get_model_parameters(model: &SystemModel) -> usize {
    match model {
        SystemModel::ARX { a, b, .. } => a.len() + b.len(),
        SystemModel::ARMAX { a, b, c, .. } => a.len() + b.len() + c.len(),
        SystemModel::OE { b, f, .. } => b.len() + f.len(),
        SystemModel::BJ { b, c, d, f, .. } => b.len() + c.len() + d.len() + f.len(),
        SystemModel::HammersteinWiener { linear, .. } => get_model_parameters(linear),
        SystemModel::TransferFunction(tf) => tf.num.len() + tf.den.len(),
        SystemModel::StateSpace(ss) => ss.a.len() + ss.b.len() + ss.c.len() + ss.d.len(),
    }
}
/// SIMD-optimized parameter estimation for large datasets
#[allow(dead_code)]
pub fn simd_optimized_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let n = input.len();
    if n > 10000 && config.parallel {
        parallel_block_identification(input, output, config)
    } else {
        enhanced_system_identification(input, output, config)
    }
}
/// Parallel block-based identification for large datasets
#[allow(dead_code)]
fn parallel_block_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let n = input.len();
    let block_size = 5000;
    let overlap = 500;
    let n_blocks = (n + block_size - overlap - 1) / (block_size - overlap);
    let block_results: Vec<_> = (0..n_blocks)
        .map(|i| {
            let start = i * (block_size - overlap);
            let end = (start + block_size).min(n);
            let block_input = input.slice(scirs2_core::ndarray::s![start..end]).to_owned();
            let block_output = output
                .slice(scirs2_core::ndarray::s![start..end])
                .to_owned();
            enhanced_system_identification(&block_input, &block_output, config)
        })
        .collect::<Result<Vec<_>, _>>()?;
    aggregate_block_results(&block_results)
}
/// Aggregate results from parallel blocks
#[allow(dead_code)]
fn aggregate_block_results(results: &[EnhancedSysIdResult]) -> SignalResult<EnhancedSysIdResult> {
    if results.is_empty() {
        return Err(SignalError::ValueError(
            "No _results to aggregate".to_string(),
        ));
    }
    let first = &results[0];
    let mut aggregated = first.clone();
    let weights: Vec<f64> = results
        .iter()
        .map(|r| 1.0 / (r.diagnostics.final_cost + 1e-10))
        .collect();
    let total_weight: f64 = weights.iter().sum();
    let mut weighted_params = Array1::zeros(first.parameters.values.len());
    for (result, &weight) in results.iter().zip(weights.iter()) {
        weighted_params = weighted_params + &result.parameters.values * (weight / total_weight);
    }
    aggregated.parameters.values = weighted_params;
    aggregated.diagnostics.final_cost = results
        .iter()
        .zip(weights.iter())
        .map(|(r, &w)| r.diagnostics.final_cost * w / total_weight)
        .sum();
    Ok(aggregated)
}
/// Robust system identification with outlier rejection
#[allow(dead_code)]
pub fn robust_system_identification(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<EnhancedSysIdResult> {
    let max_iterations = 10;
    let outlier_threshold = 3.0;
    let mut clean_input = input.clone();
    let mut clean_output = output.clone();
    for _iter in 0..max_iterations {
        let result = enhanced_system_identification(&clean_input, &clean_output, config)?;
        let y_pred = simulate_model(&result.model, &clean_input)?;
        let residuals = &clean_output - &y_pred;
        let outlier_mask = detect_outliers_mad(&residuals, outlier_threshold);
        let mut new_input = Vec::new();
        let mut new_output = Vec::new();
        for (i, &is_outlier) in outlier_mask.iter().enumerate() {
            if !is_outlier {
                new_input.push(clean_input[i]);
                new_output.push(clean_output[i]);
            }
        }
        if new_input.len() == clean_input.len() {
            return Ok(result);
        }
        clean_input = Array1::from_vec(new_input);
        clean_output = Array1::from_vec(new_output);
        if clean_input.len() < config.max_order * 3 {
            break;
        }
    }
    enhanced_system_identification(&clean_input, &clean_output, config)
}
/// Detect outliers using Median Absolute Deviation
#[allow(dead_code)]
fn detect_outliers_mad(data: &Array1<f64>, threshold: f64) -> Vec<bool> {
    let median = compute_median(&data.to_vec());
    let deviations: Vec<f64> = data.iter().map(|&x| (x - median).abs()).collect();
    let mad = compute_median(&deviations) / 0.6745;
    data.iter()
        .map(|&x| (x - median).abs() > threshold * mad)
        .collect()
}
/// Compute median of a vector
#[allow(dead_code)]
fn compute_median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
    let len = sorted.len();
    if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    }
}
/// Multi-Input Multi-Output (MIMO) system identification
#[allow(dead_code)]
pub fn mimo_system_identification(
    inputs: &Array2<f64>,
    outputs: &Array2<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<Vec<EnhancedSysIdResult>> {
    let n_inputs = inputs.ncols();
    let n_outputs = outputs.ncols();
    let mut results = Vec::with_capacity(n_outputs);
    for output_idx in 0..n_outputs {
        let output_signal = outputs.column(output_idx).to_owned();
        let combined_input = if n_inputs == 1 {
            inputs.column(0).to_owned()
        } else {
            let mut combined = Array1::zeros(inputs.nrows());
            for input_idx in 0..n_inputs {
                let input_col = inputs.column(input_idx);
                combined = combined + input_col;
            }
            combined / n_inputs as f64
        };
        let result = enhanced_system_identification(&combined_input, &output_signal, config)?;
        results.push(result);
    }
    Ok(results)
}
/// Advanced model selection using information-theoretic criteria
#[allow(dead_code)]
pub fn advanced_model_selection(
    input: &Array1<f64>,
    output: &Array1<f64>,
    candidate_structures: &[ModelStructure],
) -> SignalResult<(ModelStructure, EnhancedSysIdResult)> {
    let mut best_structure = candidate_structures[0];
    let mut best_result = None;
    let mut best_score = f64::INFINITY;
    for &structure in candidate_structures {
        let config = EnhancedSysIdConfig {
            model_structure: structure,
            order_selection: true,
            cv_folds: Some(5),
            ..Default::default()
        };
        if let Ok(result) = enhanced_system_identification(input, output, &config) {
            let score = compute_penalized_likelihood(&result);
            if score < best_score {
                best_score = score;
                best_structure = structure;
                best_result = Some(result);
            }
        }
    }
    match best_result {
        Some(result) => Ok((best_structure, result)),
        None => Err(SignalError::ComputationError(
            "No valid models found during selection".to_string(),
        )),
    }
}
/// Compute penalized likelihood for model selection
#[allow(dead_code)]
fn compute_penalized_likelihood(result: &EnhancedSysIdResult) -> f64 {
    let n = result.parameters.values.len() as f64;
    let k = get_model_parameters(&result.model) as f64;
    result.validation.aic + 2.0 * k * (k + 1.0) / (n - k - 1.0).max(1.0)
}
/// Compute condition number using Frobenius norms
#[allow(dead_code)]
pub(super) fn compute_condition_number(params: &ParameterEstimate) -> f64 {
    let frobenius_norm = |m: &Array2<f64>| -> f64 { m.iter().map(|&x| x * x).sum::<f64>().sqrt() };
    match inv(&params.covariance.view(), None) {
        Ok(inv_cov) => frobenius_norm(&params.covariance) * frobenius_norm(&inv_cov),
        Err(_) => f64::INFINITY,
    }
}
