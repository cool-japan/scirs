//! Utility functions for Lomb-Scargle SciPy validation
//!
//! This module provides helper functions for error calculation, correlation,
//! peak finding, and other utility operations used throughout validation.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use std::collections::HashMap;

/// Calculate error metrics between two result arrays
#[allow(dead_code)]
pub fn calculate_error_metrics(result1: &[f64], result2: &[f64]) -> SignalResult<(f64, f64, f64)> {
    if result1.len() != result2.len() {
        return Err(SignalError::ValueError(
            "Result arrays must have same length".to_string(),
        ));
    }

    let mut max_abs_error: f64 = 0.0;
    let mut max_rel_error: f64 = 0.0;
    let mut mse_sum = 0.0;
    let n = result1.len();

    for (i, (&a, &b)) in result1.iter().zip(result2.iter()).enumerate() {
        let abs_error = (a - b).abs();
        let rel_error = if b.abs() > 1e-15 {
            abs_error / b.abs()
        } else {
            0.0
        };

        max_abs_error = max_abs_error.max(abs_error);
        max_rel_error = max_rel_error.max(rel_error);
        mse_sum += (a - b).powi(2);
    }

    let rmse = (mse_sum / n as f64).sqrt();

    Ok((max_abs_error, max_rel_error, rmse))
}

/// Calculate correlation coefficient between two arrays
#[allow(dead_code)]
pub fn calculate_correlation(x: &[f64], y: &[f64]) -> SignalResult<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Ok(0.0);
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x > 0.0 && var_y > 0.0 {
        Ok(cov / (var_x * var_y).sqrt())
    } else {
        Ok(0.0)
    }
}

/// Calculate normalization consistency across different methods
#[allow(dead_code)]
pub fn calculate_normalization_consistency(
    method_results: &HashMap<String, AccuracyValidationResult>,
) -> f64 {
    let correlations: Vec<f64> = method_results
        .values()
        .map(|result| result.correlation)
        .collect();

    if correlations.len() < 2 {
        return 1.0;
    }

    let mean_corr = correlations.iter().sum::<f64>() / correlations.len() as f64;
    let variance = correlations
        .iter()
        .map(|&c| (c - mean_corr).powi(2))
        .sum::<f64>()
        / correlations.len() as f64;

    (1.0 - variance).max(0.0) // High consistency = low variance
}

/// Find peaks in a signal above a threshold
#[allow(dead_code)]
pub fn find_peaks(data: &[f64], threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    let n = data.len();

    for i in 1..n - 1 {
        if data[i] > threshold && data[i] > data[i - 1] && data[i] > data[i + 1] {
            peaks.push(i);
        }
    }

    peaks
}
