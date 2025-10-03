//! Statistical analysis utilities for 2D DWT operations
//!
//! This module provides comprehensive statistical analysis functions for wavelet
//! decompositions including energy analysis, entropy computation, sparsity measures,
//! and various image quality metrics.

use crate::dwt2d_enhanced::types::{Dwt2dStatistics, MultilevelDwt2d};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;

/// Compute enhanced 2D DWT statistics for analysis
///
/// This function provides comprehensive statistical analysis of multilevel
/// wavelet decompositions including energy distribution, entropy measures,
/// and sparsity analysis across all decomposition levels.
///
/// # Arguments
/// * `decomp` - Multilevel 2D DWT decomposition to analyze
///
/// # Returns
/// * Result containing detailed statistics or an error
#[allow(dead_code)]
pub fn compute_enhanced_dwt2d_statistics(
    decomp: &MultilevelDwt2d,
) -> SignalResult<Dwt2dStatistics> {
    let mut level_energies = Vec::new();
    let mut level_entropies = Vec::new();
    let mut level_sparsities = Vec::new();

    // Analyze each decomposition level
    for (level, (detail_h, detail_v, detail_d)) in decomp.details.iter().enumerate() {
        // Energy analysis
        let h_energy: f64 = detail_h.iter().map(|&x| x * x).sum();
        let v_energy: f64 = detail_v.iter().map(|&x| x * x).sum();
        let d_energy: f64 = detail_d.iter().map(|&x| x * x).sum();
        let total_energy = h_energy + v_energy + d_energy;

        level_energies.push(total_energy);

        // Entropy analysis (Shannon entropy of coefficient magnitudes)
        let entropy = compute_coefficient_entropy(detail_h, detail_v, detail_d)?;
        level_entropies.push(entropy);

        // Sparsity analysis
        let sparsity = compute_coefficient_sparsity(detail_h, detail_v, detail_d, 1e-6);
        level_sparsities.push(sparsity);
    }

    // Analyze approximation coefficients
    let approx_energy: f64 = decomp.approx.iter().map(|&x| x * x).sum();

    Ok(Dwt2dStatistics {
        level_energies,
        level_entropies,
        level_sparsities,
        approx_energy,
        total_levels: decomp.details.len(),
    })
}

/// Compute Shannon entropy of wavelet coefficients
///
/// Calculates the Shannon entropy of wavelet detail coefficients, which provides
/// insight into the information content and compressibility of the decomposition.
///
/// # Arguments
/// * `detail_h` - Horizontal detail coefficients
/// * `detail_v` - Vertical detail coefficients
/// * `detail_d` - Diagonal detail coefficients
///
/// # Returns
/// * Shannon entropy value or an error
#[allow(dead_code)]
fn compute_coefficient_entropy(
    detail_h: &Array2<f64>,
    detail_v: &Array2<f64>,
    detail_d: &Array2<f64>,
) -> SignalResult<f64> {
    // Combine all detail coefficients
    let coeffs: Vec<f64> = detail_h.iter()
        .chain(detail_v.iter())
        .chain(detail_d.iter())
        .map(|&x: &f64| x.abs())
        .filter(|&x| x > 1e-12) // Filter near-zero values
        .collect();

    if coeffs.is_empty() {
        return Ok(0.0);
    }

    // Normalize to create probability distribution
    let sum: f64 = coeffs.iter().sum();
    if sum <= 0.0 {
        return Ok(0.0);
    }

    // Compute Shannon entropy
    let entropy = coeffs
        .iter()
        .map(|&x| {
            let p = x / sum;
            -p * p.log2()
        })
        .sum::<f64>();

    Ok(entropy)
}

/// Compute sparsity measure of wavelet coefficients
///
/// Calculates the fraction of coefficients below a given threshold, which
/// indicates how well the signal is concentrated in the wavelet domain.
///
/// # Arguments
/// * `detail_h` - Horizontal detail coefficients
/// * `detail_v` - Vertical detail coefficients
/// * `detail_d` - Diagonal detail coefficients
/// * `threshold` - Threshold below which coefficients are considered sparse
///
/// # Returns
/// * Sparsity ratio (0.0 = dense, 1.0 = completely sparse)
#[allow(dead_code)]
fn compute_coefficient_sparsity(
    detail_h: &Array2<f64>,
    detail_v: &Array2<f64>,
    detail_d: &Array2<f64>,
    threshold: f64,
) -> f64 {
    let total_coeffs = detail_h.len() + detail_v.len() + detail_d.len();

    let sparse_coeffs = detail_h
        .iter()
        .chain(detail_v.iter())
        .chain(detail_d.iter())
        .filter(|&&x| x.abs() < threshold)
        .count();

    sparse_coeffs as f64 / total_coeffs as f64
}

/// Calculate edge variance to determine image characteristics
///
/// Analyzes the variance at the edges of the image to help characterize
/// the boundary behavior and select appropriate boundary handling strategies.
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
/// Computes a smoothness measure based on the discrete Laplacian operator
/// to characterize the local variability in the image data.
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
/// Analyzes the image for periodic patterns using simple correlation analysis
/// between image quadrants to detect repeating structures.
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

/// Compute gradient magnitude using Sobel operator
///
/// Calculates the gradient magnitude at each pixel using the Sobel operator
/// for edge detection and image analysis purposes.
///
/// # Arguments
/// * `image` - Input 2D array
///
/// # Returns
/// * 2D array of gradient magnitudes or an error
#[allow(dead_code)]
fn compute_gradient_magnitude(image: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (rows, cols) = image.dim();
    let mut magnitude = Array2::zeros((rows, cols));

    // Sobel kernels
    let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    for i in 1..rows - 1 {
        for j in 1..cols - 1 {
            let mut gx = 0.0;
            let mut gy = 0.0;

            // Apply Sobel kernels
            for di in 0..3 {
                for dj in 0..3 {
                    let pixel = image[[i + di - 1, j + dj - 1]];
                    gx += pixel * sobel_x[di][dj];
                    gy += pixel * sobel_y[di][dj];
                }
            }

            magnitude[[i, j]] = (gx * gx + gy * gy).sqrt();
        }
    }

    Ok(magnitude)
}
