//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;

/// Compute variance of an Array1<f64>
#[allow(dead_code)]
fn array_variance_local(arr: &Array1<f64>) -> f64 {
    let n = arr.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = arr.sum() / n;
    arr.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
}

use super::functions_4::{invert_matrix, solve_least_squares};
use super::types::{EnhancedSysIdConfig, ModelOrders, ModelValidationMetrics, ResidualAnalysis};

/// Enhanced order selection with multiple criteria
#[allow(dead_code)]
pub(super) fn enhanced_order_selection(
    input: &Array1<f64>,
    output: &Array1<f64>,
    config: &EnhancedSysIdConfig,
) -> SignalResult<ModelOrders> {
    let mut best_orders = ModelOrders {
        na: 1,
        nb: 1,
        nc: 1,
        nd: 1,
        nf: 1,
        nk: 1,
    };
    let mut best_criterion = f64::INFINITY;
    for na in 1..=config.max_order {
        for nb in 1..=config.max_order {
            if let Ok(result) = identify_arx_model(input, output, na, nb, 1) {
                let criterion = result.aic + 0.1 * (1.0 - result.fit_percentage / 100.0);
                if criterion < best_criterion {
                    best_criterion = criterion;
                    best_orders.na = na;
                    best_orders.nb = nb;
                }
            }
        }
    }
    Ok(best_orders)
}
/// Median calculation (helper)
#[allow(dead_code)]
pub(super) fn median_helper(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}
/// Median Absolute Deviation calculation
#[allow(dead_code)]
pub(super) fn mad(data: &[f64], medianval: f64) -> f64 {
    let deviations: Vec<f64> = data.iter().map(|&x| (x - medianval).abs()).collect();
    median_helper(&deviations) * 1.4826
}
/// Simplified ARX model identification for order selection
#[allow(dead_code)]
fn identify_arx_model(
    input: &Array1<f64>,
    output: &Array1<f64>,
    na: usize,
    nb: usize,
    nk: usize,
) -> SignalResult<ModelValidationMetrics> {
    let n = input.len();
    let max_order = na.max(nb + nk);
    if n <= max_order {
        return Err(SignalError::ValueError(
            "Insufficient data for model order".to_string(),
        ));
    }
    let n_params = na + nb;
    let n_data = n - max_order;
    let mut phi = Array2::zeros((n_data, n_params));
    let mut y = Array1::zeros(n_data);
    for i in 0..n_data {
        let idx = i + max_order;
        y[i] = output[idx];
        for j in 0..na {
            phi[[i, j]] = -output[idx - j - 1];
        }
        for j in 0..nb {
            if idx >= j + nk {
                phi[[i, na + j]] = input[idx - j - nk];
            }
        }
    }
    let theta = solve_least_squares(&phi, &y)?;
    let y_pred = phi.dot(&theta);
    let residuals = &y - &y_pred;
    let residual_variance = array_variance_local(&residuals);
    let output_variance = array_variance_local(&y);
    let fit_percentage = (1.0 - residual_variance / output_variance).max(0.0) * 100.0;
    let aic = (n_data as f64) * (residual_variance + 1e-15).ln() + 2.0 * n_params as f64;
    let bic = (n_data as f64) * (residual_variance + 1e-15).ln()
        + (n_params as f64) * (n_data as f64).ln();
    Ok(ModelValidationMetrics {
        fit_percentage,
        cv_fit: None,
        aic,
        bic,
        fpe: residual_variance * (n_data as f64 + n_params as f64)
            / (n_data as f64 - n_params as f64),
        residual_analysis: ResidualAnalysis {
            autocorrelation: Array1::zeros(10),
            cross_correlation: Array1::zeros(10),
            whiteness_pvalue: 0.5,
            independence_pvalue: 0.5,
            normality_pvalue: 0.5,
        },
        stability_margin: 1.0,
    })
}
/// Solve least squares problem (helper)
#[allow(dead_code)]
fn solve_least_squares_helper(a: &Array2<f64>, b: &Array1<f64>) -> SignalResult<Array1<f64>> {
    let at = a.t();
    let ata = at.dot(a);
    let atb = at.dot(b);
    let ata_inv = invert_matrix(&ata)?;
    Ok(ata_inv.dot(&atb))
}
