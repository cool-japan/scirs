//! Validation utilities for 2D DWT operations
//!
//! This module provides comprehensive validation functions for 2D DWT decomposition
//! and reconstruction results. It includes dimension checking, coefficient validation,
//! energy conservation checks, and cross-subband property analysis.

use crate::dwt2d_enhanced::types::{Dwt2dConfig, EnhancedDwt2dResult};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;

/// Validate 2D DWT decomposition result
///
/// Performs basic validation of DWT decomposition results including dimension
/// consistency, finite value checks, and energy distribution analysis.
///
/// # Arguments
/// * `result` - The DWT decomposition result to validate
/// * `originalshape` - Original image dimensions (rows, cols)
/// * `config` - DWT configuration for tolerance and validation settings
///
/// # Returns
/// * `Ok(())` if validation passes, error otherwise
#[allow(dead_code)]
fn validate_dwt2d_result(
    result: &EnhancedDwt2dResult,
    originalshape: (usize, usize),
    config: &Dwt2dConfig,
) -> SignalResult<()> {
    let (orig_rows, orig_cols) = originalshape;
    let expected_rows = orig_rows.div_ceil(2);
    let expected_cols = orig_cols.div_ceil(2);

    // Check dimensions of all subbands
    if result.approx.dim() != (expected_rows, expected_cols) {
        return Err(SignalError::ComputationError(format!(
            "Approximation subband has incorrect dimensions: expected ({}, {}), got ({}, {})",
            expected_rows,
            expected_cols,
            result.approx.nrows(),
            result.approx.ncols()
        )));
    }

    if result.detail_h.dim() != (expected_rows, expected_cols) {
        return Err(SignalError::ComputationError(format!(
            "Horizontal detail subband has incorrect dimensions: expected ({}, {}), got ({}, {})",
            expected_rows,
            expected_cols,
            result.detail_h.nrows(),
            result.detail_h.ncols()
        )));
    }

    if result.detail_v.dim() != (expected_rows, expected_cols) {
        return Err(SignalError::ComputationError(format!(
            "Vertical detail subband has incorrect dimensions: expected ({}, {}), got ({}, {})",
            expected_rows,
            expected_cols,
            result.detail_v.nrows(),
            result.detail_v.ncols()
        )));
    }

    if result.detail_d.dim() != (expected_rows, expected_cols) {
        return Err(SignalError::ComputationError(format!(
            "Diagonal detail subband has incorrect dimensions: expected ({}, {}), got ({}, {})",
            expected_rows,
            expected_cols,
            result.detail_d.nrows(),
            result.detail_d.ncols()
        )));
    }

    // Check for finite values in all subbands
    let tolerance = config.tolerance;

    for &val in result.approx.iter() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value found in approximation subband: {}",
                val
            )));
        }
    }

    for &val in result.detail_h.iter() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value found in horizontal detail subband: {}",
                val
            )));
        }
    }

    for &val in result.detail_v.iter() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value found in vertical detail subband: {}",
                val
            )));
        }
    }

    for &val in result.detail_d.iter() {
        if !val.is_finite() {
            return Err(SignalError::ComputationError(format!(
                "Non-finite value found in diagonal detail subband: {}",
                val
            )));
        }
    }

    // Check for reasonable energy distribution
    let approx_energy: f64 = result.approx.iter().map(|&x| x * x).sum();
    let detail_h_energy: f64 = result.detail_h.iter().map(|&x| x * x).sum();
    let detail_v_energy: f64 = result.detail_v.iter().map(|&x| x * x).sum();
    let detail_d_energy: f64 = result.detail_d.iter().map(|&x| x * x).sum();

    let total_energy = approx_energy + detail_h_energy + detail_v_energy + detail_d_energy;

    if total_energy == 0.0 {
        eprintln!("Warning: All wavelet coefficients are zero. This may indicate a problem with the input or computation.");
    } else if total_energy < tolerance {
        eprintln!("Warning: Very low total energy in wavelet coefficients ({:.2e}). Results may be unreliable.", total_energy);
    }

    // Check energy distribution ratios
    if total_energy > 0.0 {
        let approx_ratio = approx_energy / total_energy;
        let detail_ratio = (detail_h_energy + detail_v_energy + detail_d_energy) / total_energy;

        if approx_ratio > 0.99 {
            eprintln!("Warning: Almost all energy is in the approximation subband ({:.1}%). The signal may be very smooth.", approx_ratio * 100.0);
        } else if detail_ratio > 0.99 {
            eprintln!("Warning: Almost all energy is in detail subbands ({:.1}%). The signal may be very noisy.", detail_ratio * 100.0);
        }
    }

    Ok(())
}

/// Production-ready enhanced validation of DWT2D result
///
/// Provides comprehensive validation with detailed error reporting, enhanced
/// tolerance checking, and cross-subband property analysis.
///
/// # Arguments
/// * `result` - The DWT decomposition result to validate
/// * `originalshape` - Original image dimensions (rows, cols)
/// * `config` - DWT configuration for validation settings
///
/// # Returns
/// * `Ok(())` if validation passes, error otherwise
#[allow(dead_code)]
fn validate_dwt2d_result_enhanced(
    result: &EnhancedDwt2dResult,
    originalshape: (usize, usize),
    config: &Dwt2dConfig,
) -> SignalResult<()> {
    // Check dimensions consistency
    let (orig_rows, orig_cols) = originalshape;
    let approx_rows = result.approx.nrows();
    let approx_cols = result.approx.ncols();

    // Expected dimensions (with downsampling)
    let expected_rows = orig_rows.div_ceil(2);
    let expected_cols = orig_cols.div_ceil(2);

    if approx_rows != expected_rows || approx_cols != expected_cols {
        return Err(SignalError::ComputationError(format!(
            "Dimension mismatch: expected {}x{}, got {}x{}",
            expected_rows, expected_cols, approx_rows, approx_cols
        )));
    }

    // Check all subbands have same dimensions
    if result.detail_h.dim() != (approx_rows, approx_cols)
        || result.detail_v.dim() != (approx_rows, approx_cols)
        || result.detail_d.dim() != (approx_rows, approx_cols)
    {
        return Err(SignalError::ComputationError(
            "Subband dimension mismatch".to_string(),
        ));
    }

    // Enhanced finite value validation with detailed error reporting
    let subbands = [
        (&result.approx, "approximation"),
        (&result.detail_h, "horizontal detail"),
        (&result.detail_v, "vertical detail"),
        (&result.detail_d, "diagonal detail"),
    ];

    for (subband, name) in subbands {
        for (idx, &val) in subband.iter().enumerate() {
            if !val.is_finite() {
                let (row, col) = (idx / subband.ncols(), idx % subband.ncols());
                return Err(SignalError::ComputationError(format!(
                    "Non-finite value {} found in {} subband at position ({}, {})",
                    val, name, row, col
                )));
            }
        }
    }

    // Enhanced tolerance-based validation
    if config.tolerance > 0.0 {
        // Comprehensive coefficient magnitude analysis
        let mut stats = WaveletCoefficientStats::new();

        for (subband, name) in subbands {
            let max_val = subband.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_val = subband.iter().cloned().fold(f64::INFINITY, f64::min);
            let mean_val = subband.iter().sum::<f64>() / subband.len() as f64;
            let variance =
                subband.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / subband.len() as f64;

            stats.update(name, max_val, min_val, mean_val, variance);

            // Check for extremely large coefficients
            if max_val > 1e12 {
                eprintln!(
                    "Warning: Very large coefficient detected in {}: {:.2e}",
                    name, max_val
                );
            }

            // Check for suspicious coefficient distributions
            if variance < 1e-15 && subband.len() > 1 {
                eprintln!(
                    "Warning: {} subband has near-zero variance, may indicate numerical issues",
                    name
                );
            }
        }

        // Cross-subband validation
        validate_cross_subband_properties(&stats, config)?;
    }

    // Energy conservation check
    if config.compute_metrics {
        validate_energy_conservation(result, originalshape)?;
    }

    Ok(())
}

/// Statistics for wavelet coefficient validation
#[derive(Debug)]
struct WaveletCoefficientStats {
    approx_max: f64,
    detail_max: f64,
    total_energy: f64,
    approx_energy: f64,
    detail_energy: f64,
}

impl WaveletCoefficientStats {
    fn new() -> Self {
        Self {
            approx_max: f64::NEG_INFINITY,
            detail_max: f64::NEG_INFINITY,
            total_energy: 0.0,
            approx_energy: 0.0,
            detail_energy: 0.0,
        }
    }

    fn update(
        &mut self,
        subband_name: &str,
        max_val: f64,
        _min_val: f64,
        _mean: f64,
        variance: f64,
    ) {
        let energy = variance;

        match subband_name {
            "approximation" => {
                self.approx_max = max_val;
                self.approx_energy = energy;
            }
            _ => {
                self.detail_max = self.detail_max.max(max_val);
                self.detail_energy += energy;
            }
        }

        self.total_energy += energy;
    }
}

/// Validate cross-subband properties
///
/// Analyzes the relationships between different wavelet subbands to detect
/// potential issues with energy distribution and coefficient magnitudes.
///
/// # Arguments
/// * `stats` - Wavelet coefficient statistics across subbands
/// * `config` - DWT configuration for validation settings
///
/// # Returns
/// * `Ok(())` if validation passes, error otherwise
#[allow(dead_code)]
fn validate_cross_subband_properties(
    stats: &WaveletCoefficientStats,
    config: &Dwt2dConfig,
) -> SignalResult<()> {
    // Check energy distribution
    if stats.total_energy > 0.0 {
        let approx_ratio = stats.approx_energy / stats.total_energy;
        let detail_ratio = stats.detail_energy / stats.total_energy;

        if approx_ratio > 0.999 {
            eprintln!(
                "Warning: {:.1}% of energy in approximation subband. Signal may be over-smoothed.",
                approx_ratio * 100.0
            );
        }

        if detail_ratio > 0.999 {
            eprintln!(
                "Warning: {:.1}% of energy in detail subbands. Signal may be very noisy.",
                detail_ratio * 100.0
            );
        }
    }

    // Check coefficient magnitude ratios
    if stats.detail_max > 0.0 && stats.approx_max > 0.0 {
        let magnitude_ratio = stats.detail_max / stats.approx_max;

        if magnitude_ratio > 100.0 {
            eprintln!(
                "Warning: Detail coefficients are {:.1}x larger than approximation. This may indicate numerical instability.",
                magnitude_ratio
            );
        }
    }

    Ok(())
}

/// Validate energy conservation
///
/// Checks that the energy is properly distributed across wavelet subbands
/// and detects potential energy conservation violations.
///
/// # Arguments
/// * `result` - The DWT decomposition result to analyze
/// * `originalshape` - Original image dimensions for reference
///
/// # Returns
/// * `Ok(())` if energy conservation is satisfied, error otherwise
#[allow(dead_code)]
fn validate_energy_conservation(
    result: &EnhancedDwt2dResult,
    originalshape: (usize, usize),
) -> SignalResult<()> {
    let approx_energy: f64 = result.approx.iter().map(|&x| x * x).sum();
    let detail_h_energy: f64 = result.detail_h.iter().map(|&x| x * x).sum();
    let detail_v_energy: f64 = result.detail_v.iter().map(|&x| x * x).sum();
    let detail_d_energy: f64 = result.detail_d.iter().map(|&x| x * x).sum();

    let total_wavelet_energy = approx_energy + detail_h_energy + detail_v_energy + detail_d_energy;

    // We can't directly compare with original energy without the original data,
    // but we can check for reasonable energy distribution
    if total_wavelet_energy == 0.0 {
        return Err(SignalError::ComputationError(
            "All wavelet coefficients are zero".to_string(),
        ));
    }

    // Check for reasonable energy distribution (this is a heuristic)
    let approx_ratio = approx_energy / total_wavelet_energy;
    if approx_ratio < 0.001 {
        eprintln!(
            "Warning: Approximation subband contains only {:.3}% of total energy",
            approx_ratio * 100.0
        );
    }

    Ok(())
}

/// Calculate edge variance to determine image characteristics
///
/// Analyzes the variance at the edges of the image to help select
/// appropriate boundary handling strategies.
///
/// # Arguments
/// * `data` - Input 2D array to analyze
///
/// # Returns
/// * Average edge variance across all four edges
#[allow(dead_code)]
fn calculate_edge_variance(data: &Array2<f64>) -> f64 {
    let (rows, cols) = data.dim();

    // Extract edges
    let top_edge = data.row(0);
    let bottom_edge = data.row(rows - 1);
    let left_edge = data.column(0);
    let right_edge = data.column(cols - 1);

    // Calculate variances using a simple variance computation
    let top_var = {
        let mean: f64 = top_edge.iter().sum::<f64>() / top_edge.len() as f64;
        top_edge.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / top_edge.len() as f64
    };

    let bottom_var = {
        let mean: f64 = bottom_edge.iter().sum::<f64>() / bottom_edge.len() as f64;
        bottom_edge.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / bottom_edge.len() as f64
    };

    let left_var = {
        let mean: f64 = left_edge.iter().sum::<f64>() / left_edge.len() as f64;
        left_edge.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / left_edge.len() as f64
    };

    let right_var = {
        let mean: f64 = right_edge.iter().sum::<f64>() / right_edge.len() as f64;
        right_edge.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / right_edge.len() as f64
    };

    // Return average edge variance
    (top_var + bottom_var + left_var + right_var) / 4.0
}

/// Calculate smoothness metric for the image
///
/// Computes a smoothness measure based on the Laplacian operator to
/// help determine the most appropriate boundary handling approach.
///
/// # Arguments
/// * `data` - Input 2D array to analyze
///
/// # Returns
/// * Smoothness metric (0.0 = rough, 1.0 = smooth)
#[allow(dead_code)]
fn calculate_smoothness(data: &Array2<f64>) -> f64 {
    let (rows, cols) = data.dim();

    if rows < 3 || cols < 3 {
        return 0.5; // Default for small images
    }

    // Calculate Laplacian to measure smoothness
    let mut total_laplacian = 0.0;
    let mut count = 0;

    for i in 1..(rows - 1) {
        for j in 1..(cols - 1) {
            let laplacian =
                data[[i - 1, j]] + data[[i + 1, j]] + data[[i, j - 1]] + data[[i, j + 1]]
                    - 4.0 * data[[i, j]];
            total_laplacian += laplacian.abs();
            count += 1;
        }
    }

    let avg_laplacian = total_laplacian / count as f64;
    let data_range = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - data.iter().cloned().fold(f64::INFINITY, f64::min);

    if data_range < 1e-12 {
        return 1.0; // Constant image is very smooth
    }

    // Normalize by data range and invert (higher value = smoother)
    1.0 / (1.0 + avg_laplacian / data_range)
}

/// Estimate periodicity of the image
///
/// Analyzes the image for periodic patterns using correlation analysis
/// to help select the most appropriate boundary extension method.
///
/// # Arguments
/// * `data` - Input 2D array to analyze
///
/// # Returns
/// * Periodicity measure (0.0 = non-periodic, 1.0 = highly periodic)
#[allow(dead_code)]
fn estimate_periodicity(data: &Array2<f64>) -> f64 {
    let (rows, cols) = data.dim();

    // Simple correlation-based periodicity detection
    let min_dim = rows.min(cols);
    if min_dim < 4 {
        return 0.0;
    }

    let half_rows = rows / 2;
    let half_cols = cols / 2;

    // Compare first half with second half
    let mut correlation = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..half_rows {
        for j in 0..half_cols {
            let val1 = data[[i, j]];
            let val2 = data[[i + half_rows, j + half_cols]];

            correlation += val1 * val2;
            norm1 += val1 * val1;
            norm2 += val2 * val2;
        }
    }

    if norm1 < 1e-12 || norm2 < 1e-12 {
        return 0.0;
    }

    (correlation / (norm1 * norm2).sqrt()).abs()
}
