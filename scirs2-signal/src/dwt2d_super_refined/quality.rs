//! Quality metrics and analysis functions for advanced-refined 2D wavelet transforms
//!
//! This module provides comprehensive quality assessment including perceptual metrics,
//! compression analysis, frequency domain analysis, and edge preservation evaluation.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array2, Array3};
use statrs::statistics::Statistics;

/// Compute comprehensive quality metrics for advanced-refined wavelet analysis
pub fn compute_advanced_refined_quality_metrics(
    original_image: &Array2<f64>,
    processing_result: &ProcessingResult,
    decomposition_tree: &DecompositionTree,
    quality_config: &QualityConfig,
) -> SignalResult<AdvancedRefinedQualityMetrics> {
    // Compute basic energy preservation
    let original_energy = compute_image_energy(original_image);
    let coeffs_energy = compute_total_coefficients_energy(&processing_result.coefficients);
    let energy_preservation = if original_energy > 0.0 {
        coeffs_energy / original_energy
    } else {
        0.0
    };

    // Compute sparsity measure
    let sparsity = compute_sparsity(&processing_result.coefficients);

    // Compute compression ratio
    let compression_ratio = estimate_compression_ratio(&processing_result.coefficients);

    // Compute perceptual quality if enabled
    let perceptual_quality = if quality_config.compute_perceptual_metrics {
        compute_perceptual_quality(
            original_image,
            &processing_result.coefficients,
            quality_config,
        )?
    } else {
        0.0
    };

    // Compute edge preservation
    let edge_preservation =
        compute_multiscale_edge_preservation(original_image, &processing_result.coefficients)?;

    // Compute frequency analysis if enabled
    let frequency_analysis = if quality_config.compute_frequency_analysis {
        compute_frequency_analysis(&processing_result.coefficients)?
    } else {
        FrequencyAnalysis {
            spectral_energy_distribution: Vec::new(),
            dominant_frequencies: Vec::new(),
            frequency_content_preservation: 0.0,
            aliasing_artifacts: 0.0,
        }
    };

    // Compute compression metrics if enabled
    let compression_metrics = if quality_config.compute_compression_metrics {
        compute_compression_metrics(&processing_result.coefficients)?
    } else {
        CompressionMetrics {
            theoretical_compression_ratio: compression_ratio,
            effective_compression_ratio: compression_ratio,
            rate_distortion_score: 0.0,
        }
    };

    Ok(AdvancedRefinedQualityMetrics {
        perceptual_quality,
        energy_preservation,
        sparsity,
        compression_ratio,
        edge_preservation,
        frequency_analysis,
        compression_metrics,
    })
}

/// Compute energy of subband at specific level and index
pub fn compute_subband_energy(coefficients: &Array3<f64>, level: usize, index: usize) -> f64 {
    let shape = coefficients.shape();
    if level >= shape[0] {
        return 0.0;
    }

    let level_slice = coefficients.slice(scirs2_core::ndarray::s![level, .., ..]);
    let total_elements = level_slice.len();

    if total_elements == 0 {
        return 0.0;
    }

    // For simplicity, compute energy for the entire level
    // In practice, this would extract specific subband based on index
    level_slice.iter().map(|&x| x * x).sum()
}

/// Compute entropy of subband at specific level and index
pub fn compute_subband_entropy(coefficients: &Array3<f64>, level: usize, index: usize) -> f64 {
    let shape = coefficients.shape();
    if level >= shape[0] {
        return 0.0;
    }

    let level_slice = coefficients.slice(scirs2_core::ndarray::s![level, .., ..]);
    let level_vec: Vec<f64> = level_slice.iter().cloned().collect();

    if level_vec.is_empty() {
        return 0.0;
    }

    // Compute Shannon entropy
    let sum_abs: f64 = level_vec.iter().map(|&x| x.abs()).sum();
    if sum_abs == 0.0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &coeff in &level_vec {
        let prob = coeff.abs() / sum_abs;
        if prob > 1e-12 {
            entropy -= prob * prob.ln();
        }
    }

    entropy
}

/// Compute approximation energy from coefficients
pub fn compute_approximation_energy(coefficients: &Array3<f64>) -> f64 {
    // Approximation is typically at level 0
    compute_subband_energy(coefficients, 0, 0)
}

/// Compute detail energy from coefficients
pub fn compute_detail_energy(coefficients: &Array3<f64>) -> f64 {
    let shape = coefficients.shape();
    let mut total_energy = 0.0;

    // Sum energy from all detail subbands (all levels except level 0, index 0)
    for level in 0..shape[0] {
        for index in 1..4 {
            total_energy += compute_subband_energy(coefficients, level, index);
        }
    }

    total_energy
}

/// Compute total image energy
pub fn compute_image_energy(image: &Array2<f64>) -> f64 {
    image.iter().map(|&x| x * x).sum()
}

/// Estimate compression ratio from coefficients
pub fn estimate_compression_ratio(coefficients: &Array3<f64>) -> f64 {
    let total_coeffs = coefficients.len();
    if total_coeffs == 0 {
        return 1.0;
    }

    let non_zero_coeffs = coefficients.iter().filter(|&&x| x.abs() > 1e-12).count();

    if non_zero_coeffs == 0 {
        return f64::INFINITY;
    }

    total_coeffs as f64 / non_zero_coeffs as f64
}

/// Compute sparsity measure
pub fn compute_sparsity(coefficients: &Array3<f64>) -> f64 {
    let total_coeffs = coefficients.len();
    if total_coeffs == 0 {
        return 0.0;
    }

    let zero_coeffs = coefficients.iter().filter(|&&x| x.abs() <= 1e-12).count();
    zero_coeffs as f64 / total_coeffs as f64
}

/// Compute perceptual quality using reference image
pub fn compute_perceptual_quality(
    original_image: &Array2<f64>,
    coefficients: &Array3<f64>,
    quality_config: &QualityConfig,
) -> SignalResult<f64> {
    // Reconstruct image from coefficients
    let reconstructed = reconstruct_image_from_coefficients(coefficients)?;

    // Ensure same dimensions
    if reconstructed.dim() != original_image.dim() {
        let resized = resize_image_bilinear(&reconstructed, original_image.dim())?;
        return compute_structural_similarity(original_image, &resized);
    }

    compute_structural_similarity(original_image, &reconstructed)
}

/// Compute structural similarity (SSIM)
pub fn compute_structural_similarity(
    image1: &Array2<f64>,
    image2: &Array2<f64>,
) -> SignalResult<f64> {
    if image1.dim() != image2.dim() {
        return Err(SignalError::ValueError(
            "Images must have the same dimensions for SSIM calculation".to_string(),
        ));
    }

    let (height, width) = image1.dim();
    if height == 0 || width == 0 {
        return Ok(0.0);
    }

    // Compute means
    let mean1 = image1.mean().unwrap_or(0.0);
    let mean2 = image2.mean().unwrap_or(0.0);

    // Compute variances and covariance
    let var1 = image1.var(0.0);
    let var2 = image2.var(0.0);

    let mut covariance = 0.0;
    for ((i, j), &val1) in image1.indexed_iter() {
        let val2 = image2[[i, j]];
        covariance += (val1 - mean1) * (val2 - mean2);
    }
    covariance /= (height * width) as f64;

    // SSIM constants
    let c1 = 0.01 * 0.01;
    let c2 = 0.03 * 0.03;

    // Compute SSIM
    let numerator = (2.0 * mean1 * mean2 + c1) * (2.0 * covariance + c2);
    let denominator = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2);

    if denominator > 0.0 {
        Ok(numerator / denominator)
    } else {
        Ok(0.0)
    }
}

/// Compute Peak Signal-to-Noise Ratio
pub fn compute_peak_snr(image: &Array2<f64>, coefficients: &Array3<f64>) -> SignalResult<f64> {
    let reconstructed = reconstruct_image_from_coefficients(coefficients)?;

    if reconstructed.dim() != image.dim() {
        return Err(SignalError::ValueError(
            "Reconstructed image dimensions don't match original".to_string(),
        ));
    }

    // Find maximum possible pixel value
    let max_val = image.iter().cloned().fold(0.0f64, f64::max);
    if max_val == 0.0 {
        return Ok(f64::INFINITY);
    }

    // Compute MSE
    let mut mse = 0.0;
    let total_pixels = image.len();

    for (i, (&orig, &recon)) in image.iter().zip(reconstructed.iter()).enumerate() {
        let diff = orig - recon;
        mse += diff * diff;
    }
    mse /= total_pixels as f64;

    if mse == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(20.0 * (max_val * max_val / mse).log10())
    }
}

/// Compute multiscale edge preservation
pub fn compute_multiscale_edge_preservation(
    original_image: &Array2<f64>,
    coefficients: &Array3<f64>,
) -> SignalResult<f64> {
    let reconstructed = reconstruct_image_from_coefficients(coefficients)?;

    if reconstructed.dim() != original_image.dim() {
        return Ok(0.0);
    }

    let mut total_preservation = 0.0;
    let scales = vec![1, 2, 4];

    for &scale in &scales {
        // Detect edges at this scale
        let orig_edges = detect_edges_sobel(original_image, scale)?;
        let recon_edges = detect_edges_sobel(&reconstructed, scale)?;

        // Compute correlation between edge maps
        let correlation = compute_edge_correlation(&orig_edges, &recon_edges)?;
        total_preservation += correlation;
    }

    Ok(total_preservation / scales.len() as f64)
}

/// Compute frequency domain analysis
pub fn compute_frequency_analysis(coefficients: &Array3<f64>) -> SignalResult<FrequencyAnalysis> {
    let shape = coefficients.shape();
    let levels = shape[0];

    // Compute spectral energy distribution across levels
    let mut spectral_energy_distribution = Vec::with_capacity(levels);
    for level in 0..levels {
        let level_energy = compute_subband_energy(coefficients, level, 0)
            + compute_subband_energy(coefficients, level, 1)
            + compute_subband_energy(coefficients, level, 2)
            + compute_subband_energy(coefficients, level, 3);
        spectral_energy_distribution.push(level_energy);
    }

    // Find dominant frequencies (simplified - based on energy distribution)
    let mut dominant_frequencies = Vec::new();
    let total_energy: f64 = spectral_energy_distribution.iter().sum();

    for (level, &energy) in spectral_energy_distribution.iter().enumerate() {
        if energy / total_energy > 0.1 {
            // Frequency approximately inversely proportional to level
            let freq = 1.0 / (2.0_f64.powi(level as i32 + 1));
            dominant_frequencies.push(freq);
        }
    }

    // Estimate frequency content preservation (simplified)
    let frequency_content_preservation = if total_energy > 0.0 {
        let high_freq_energy: f64 = spectral_energy_distribution.iter().skip(levels / 2).sum();
        high_freq_energy / total_energy
    } else {
        0.0
    };

    // Estimate aliasing artifacts (simplified)
    let aliasing_artifacts = estimate_aliasing_artifacts(&spectral_energy_distribution);

    Ok(FrequencyAnalysis {
        spectral_energy_distribution,
        dominant_frequencies,
        frequency_content_preservation,
        aliasing_artifacts,
    })
}

/// Estimate aliasing artifacts from spectral distribution
fn estimate_aliasing_artifacts(spectral_distribution: &[f64]) -> f64 {
    if spectral_distribution.len() < 2 {
        return 0.0;
    }

    // Look for unexpected energy in high-frequency bands
    let total_energy: f64 = spectral_distribution.iter().sum();
    if total_energy == 0.0 {
        return 0.0;
    }

    let high_freq_start = spectral_distribution.len() / 2;
    let high_freq_energy: f64 = spectral_distribution.iter().skip(high_freq_start).sum();

    // Aliasing metric: excessive high-frequency energy relative to expected decay
    (high_freq_energy / total_energy).min(1.0)
}

/// Compute compression performance metrics
pub fn compute_compression_metrics(coefficients: &Array3<f64>) -> SignalResult<CompressionMetrics> {
    let theoretical_compression_ratio = estimate_compression_ratio(coefficients);

    // Effective compression considers quantization effects
    let effective_compression_ratio = theoretical_compression_ratio * 0.8; // Simplified

    // Rate-distortion score (simplified implementation)
    let sparsity = compute_sparsity(coefficients);
    let rate_distortion_score = sparsity * theoretical_compression_ratio;

    Ok(CompressionMetrics {
        theoretical_compression_ratio,
        effective_compression_ratio,
        rate_distortion_score,
    })
}

/// Reconstruct image from coefficients (simplified)
pub fn reconstruct_image_from_coefficients(
    coefficients: &Array3<f64>,
) -> SignalResult<Array2<f64>> {
    let shape = coefficients.shape();
    if shape.len() != 3 || shape[0] == 0 {
        return Err(SignalError::ValueError(
            "Invalid coefficients shape for reconstruction".to_string(),
        ));
    }

    let (_, height, width) = (shape[0], shape[1], shape[2]);

    if height == 0 || width == 0 {
        return Err(SignalError::ValueError(
            "Invalid image dimensions for reconstruction".to_string(),
        ));
    }

    // Simplified reconstruction: sum across levels (weighted by level)
    let mut reconstructed = Array2::zeros((height, width));

    for level in 0..shape[0] {
        let weight = 1.0 / (2.0_f64.powi(level as i32)); // Higher levels have less weight
        for y in 0..height {
            for x in 0..width {
                reconstructed[[y, x]] += weight * coefficients[[level, y, x]];
            }
        }
    }

    Ok(reconstructed)
}

/// Resize image using bilinear interpolation
pub fn resize_image_bilinear(
    image: &Array2<f64>,
    target_size: (usize, usize),
) -> SignalResult<Array2<f64>> {
    let (src_height, src_width) = image.dim();
    let (target_height, target_width) = target_size;

    if target_height == 0 || target_width == 0 {
        return Err(SignalError::ValueError(
            "Target size must be positive".to_string(),
        ));
    }

    if src_height == 0 || src_width == 0 {
        return Ok(Array2::zeros(target_size));
    }

    let mut resized = Array2::zeros(target_size);

    let scale_y = src_height as f64 / target_height as f64;
    let scale_x = src_width as f64 / target_width as f64;

    for y in 0..target_height {
        for x in 0..target_width {
            let src_y = y as f64 * scale_y;
            let src_x = x as f64 * scale_x;

            let y0 = src_y.floor() as usize;
            let x0 = src_x.floor() as usize;
            let y1 = (y0 + 1).min(src_height - 1);
            let x1 = (x0 + 1).min(src_width - 1);

            let dy = src_y - y0 as f64;
            let dx = src_x - x0 as f64;

            // Bilinear interpolation
            let val = (1.0 - dy) * (1.0 - dx) * image[[y0, x0]]
                + (1.0 - dy) * dx * image[[y0, x1]]
                + dy * (1.0 - dx) * image[[y1, x0]]
                + dy * dx * image[[y1, x1]];

            resized[[y, x]] = val;
        }
    }

    Ok(resized)
}

/// Detect edges using Sobel operator at given scale
pub fn detect_edges_sobel(image: &Array2<f64>, scale: usize) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();

    if height < 3 || width < 3 {
        return Ok(Array2::zeros((height, width)));
    }

    let mut edges = Array2::zeros((height, width));

    // Sobel kernels
    let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    for y in scale..(height - scale) {
        for x in scale..(width - scale) {
            let mut gx = 0.0;
            let mut gy = 0.0;

            // Apply Sobel operators
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = image[[y + ky - 1, x + kx - 1]];
                    gx += pixel * sobel_x[ky][kx];
                    gy += pixel * sobel_y[ky][kx];
                }
            }

            // Compute gradient magnitude
            edges[[y, x]] = (gx * gx + gy * gy).sqrt();
        }
    }

    Ok(edges)
}

/// Compute correlation between edge maps
pub fn compute_edge_correlation(edges1: &Array2<f64>, edges2: &Array2<f64>) -> SignalResult<f64> {
    if edges1.dim() != edges2.dim() {
        return Err(SignalError::ValueError(
            "Edge maps must have the same dimensions".to_string(),
        ));
    }

    let n = edges1.len();
    if n == 0 {
        return Ok(0.0);
    }

    let mean1 = edges1.mean().unwrap_or(0.0);
    let mean2 = edges2.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut var1 = 0.0;
    let mut var2 = 0.0;

    for (&val1, &val2) in edges1.iter().zip(edges2.iter()) {
        let diff1 = val1 - mean1;
        let diff2 = val2 - mean2;
        numerator += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }

    let denominator = (var1 * var2).sqrt();
    if denominator > 0.0 {
        Ok(numerator / denominator)
    } else {
        Ok(0.0)
    }
}

/// Compute total energy from all coefficients
fn compute_total_coefficients_energy(coefficients: &Array3<f64>) -> f64 {
    coefficients.iter().map(|&x| x * x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_compute_image_energy() {
        let image = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);
        let energy = compute_image_energy(&image);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_compute_sparsity() {
        let mut coefficients = Array3::zeros((2, 4, 4));
        coefficients[[0, 0, 0]] = 1.0;
        coefficients[[0, 1, 1]] = 2.0;

        let sparsity = compute_sparsity(&coefficients);
        assert!(sparsity > 0.9); // Most coefficients are zero
    }

    #[test]
    fn test_structural_similarity() {
        let image1 = Array2::from_shape_fn((8, 8), |(i, j)| (i + j) as f64);
        let image2 = image1.clone();

        let ssim = compute_structural_similarity(&image1, &image2).unwrap();
        assert!((ssim - 1.0).abs() < 0.01); // Should be very close to 1 for identical images
    }

    #[test]
    fn test_resize_image_bilinear() {
        let image = Array2::from_shape_fn((4, 4), |(i, j)| (i * j) as f64);
        let resized = resize_image_bilinear(&image, (8, 8)).unwrap();

        assert_eq!(resized.dim(), (8, 8));
    }

    #[test]
    fn test_detect_edges_sobel() {
        let image = Array2::from_shape_fn((8, 8), |(i, j)| {
            if i < 4 {
                0.0
            } else {
                1.0
            } // Step edge
        });

        let edges = detect_edges_sobel(&image, 1).unwrap();
        assert_eq!(edges.dim(), image.dim());

        // Check that edge is detected around the middle
        let middle_row_energy: f64 = edges.row(4).iter().map(|&x| x * x).sum();
        assert!(middle_row_energy > 0.0);
    }
}
