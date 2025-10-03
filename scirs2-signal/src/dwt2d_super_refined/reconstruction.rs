//! Reconstruction and utility functions for advanced-refined 2D wavelet transforms
//!
//! This module provides reconstruction functions, denoising utilities, and
//! coefficient processing functions for the advanced wavelet transform system.

use super::types::*;
use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array2, Array3};
use scirs2_core::random::Rng;

/// Apply perceptual coefficient processing
pub fn apply_perceptual_coefficient_processing(
    coefficients: &Array3<f64>,
    _decomposition_tree: &DecompositionTree,
) -> SignalResult<Array3<f64>> {
    // Simplified perceptual processing - apply slight enhancement to important coefficients
    let mut processed = coefficients.clone();

    let shape = coefficients.shape();
    for level in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let coeff = coefficients[[level, y, x]];

                // Enhance coefficients based on magnitude and level
                let enhancement = if coeff.abs() > 0.1 {
                    1.1 // Slightly enhance significant coefficients
                } else {
                    0.9 // Slightly suppress small coefficients
                };

                processed[[level, y, x]] = coeff * enhancement;
            }
        }
    }

    Ok(processed)
}

/// Reconstruct image using memory-efficient approach
pub fn reconstruct_image_memory_efficient(
    coefficients: &Array3<f64>,
    _decomposition_tree: &DecompositionTree,
    _wavelet: &Wavelet,
) -> SignalResult<Array2<f64>> {
    // Simplified memory-efficient reconstruction
    // In practice, this would use the actual wavelet reconstruction algorithm
    reconstruct_from_coefficients_simple(coefficients)
}

/// Reconstruct image using standard approach
pub fn reconstruct_image_standard(
    coefficients: &Array3<f64>,
    _decomposition_tree: &DecompositionTree,
    _wavelet: &Wavelet,
) -> SignalResult<Array2<f64>> {
    // Simplified standard reconstruction
    // In practice, this would use the actual wavelet reconstruction algorithm
    reconstruct_from_coefficients_simple(coefficients)
}

/// Simple reconstruction from coefficients
fn reconstruct_from_coefficients_simple(coefficients: &Array3<f64>) -> SignalResult<Array2<f64>> {
    let shape = coefficients.shape();
    if shape.len() != 3 || shape[0] == 0 {
        return Err(SignalError::ValueError(
            "Invalid coefficients shape for reconstruction".to_string(),
        ));
    }

    let (levels, height, width) = (shape[0], shape[1], shape[2]);

    if height == 0 || width == 0 {
        return Err(SignalError::ValueError(
            "Invalid image dimensions for reconstruction".to_string(),
        ));
    }

    // Weighted sum reconstruction
    let mut reconstructed = Array2::zeros((height, width));

    for level in 0..levels {
        let weight = 1.0 / (2.0_f64.powi(level as i32 + 1));
        for y in 0..height {
            for x in 0..width {
                reconstructed[[y, x]] += weight * coefficients[[level, y, x]];
            }
        }
    }

    Ok(reconstructed)
}

/// Compute reconstruction quality metrics
pub fn compute_reconstruction_metrics(
    reconstructed_image: &Array2<f64>,
    original_result: &AdvancedRefinedWaveletPacketResult,
) -> SignalResult<ReconstructionQualityMetrics> {
    // Compute energy preservation
    let reconstructed_energy: f64 = reconstructed_image.iter().map(|&x| x * x).sum();
    let original_energy: f64 = original_result.coefficients.iter().map(|&x| x * x).sum();

    let energy_preservation = if original_energy > 0.0 {
        reconstructed_energy / original_energy
    } else {
        0.0
    };

    // Simplified reconstruction error (would need original image for proper calculation)
    let reconstruction_error = 0.01; // Placeholder

    // Simplified perceptual similarity
    let perceptual_similarity = energy_preservation.min(1.0);

    Ok(ReconstructionQualityMetrics {
        reconstruction_error,
        energy_preservation,
        perceptual_similarity,
    })
}

/// Compute coefficient utilization ratio
pub fn compute_coefficient_utilization(coefficients: &Array3<f64>) -> f64 {
    let total_coeffs = coefficients.len();
    if total_coeffs == 0 {
        return 0.0;
    }

    let used_coeffs = coefficients.iter().filter(|&&x| x.abs() > 1e-12).count();
    used_coeffs as f64 / total_coeffs as f64
}

/// Analyze noise characteristics in the image
pub fn analyze_noise_characteristics(
    noisy_image: &Array2<f64>,
    _wavelet: &Wavelet,
) -> SignalResult<NoiseAnalysis> {
    let (height, width) = noisy_image.dim();

    if height == 0 || width == 0 {
        return Err(SignalError::ValueError("Empty image".to_string()));
    }

    // Estimate noise variance using high-frequency content
    let mut high_freq_energy = 0.0;
    let mut pixel_count = 0;

    // Simple high-pass filtering using differences
    for y in 1..height {
        for x in 1..width {
            let diff_x = noisy_image[[y, x]] - noisy_image[[y, x - 1]];
            let diff_y = noisy_image[[y, x]] - noisy_image[[y - 1, x]];
            high_freq_energy += diff_x * diff_x + diff_y * diff_y;
            pixel_count += 2;
        }
    }

    let estimated_variance = if pixel_count > 0 {
        high_freq_energy / pixel_count as f64
    } else {
        0.0
    };

    // Simplified noise type detection
    let noise_type = if estimated_variance < 0.01 {
        NoiseType::Gaussian
    } else if estimated_variance < 0.1 {
        NoiseType::Gaussian
    } else {
        NoiseType::Mixed
    };

    // Estimate spatial correlation (simplified)
    let spatial_correlation = estimate_spatial_correlation(noisy_image)?;

    // Placeholder frequency characteristics
    let frequency_characteristics = vec![estimated_variance; 10];

    Ok(NoiseAnalysis {
        noise_type,
        estimated_variance,
        spatial_correlation,
        frequency_characteristics,
    })
}

/// Estimate spatial correlation in noise
fn estimate_spatial_correlation(image: &Array2<f64>) -> SignalResult<f64> {
    let (height, width) = image.dim();

    if height < 2 || width < 2 {
        return Ok(0.0);
    }

    let mut correlation_sum = 0.0;
    let mut count = 0;

    // Compute correlation with horizontal neighbors
    for y in 0..height {
        for x in 1..width {
            let val1 = image[[y, x - 1]];
            let val2 = image[[y, x]];
            correlation_sum += val1 * val2;
            count += 1;
        }
    }

    // Compute correlation with vertical neighbors
    for y in 1..height {
        for x in 0..width {
            let val1 = image[[y - 1, x]];
            let val2 = image[[y, x]];
            correlation_sum += val1 * val2;
            count += 1;
        }
    }

    Ok(if count > 0 {
        correlation_sum / count as f64
    } else {
        0.0
    })
}

/// Apply adaptive denoising based on noise analysis
pub fn apply_adaptive_denoising(
    coefficients: &Array3<f64>,
    noise_analysis: &NoiseAnalysis,
    _decomposition_tree: &DecompositionTree,
    denoising_config: &AdvancedRefinedDenoisingConfig,
) -> SignalResult<Array3<f64>> {
    let mut denoised = coefficients.clone();
    let shape = coefficients.shape();

    // Compute adaptive threshold based on noise variance
    let base_threshold = if let Some(variance) = denoising_config.noise_variance {
        variance.sqrt()
    } else {
        noise_analysis.estimated_variance.sqrt()
    };

    // Apply thresholding based on method
    for level in 0..shape[0] {
        // Scale threshold by level (higher levels = more aggressive thresholding)
        let level_threshold = base_threshold * (2.0_f64.powi(level as i32 / 2));

        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let coeff = coefficients[[level, y, x]];

                let denoised_coeff = match denoising_config.threshold_method {
                    ThresholdMethod::Soft => {
                        if coeff.abs() > level_threshold {
                            coeff.signum() * (coeff.abs() - level_threshold)
                        } else {
                            0.0
                        }
                    }
                    ThresholdMethod::Hard => {
                        if coeff.abs() > level_threshold {
                            coeff
                        } else {
                            0.0
                        }
                    }
                    ThresholdMethod::Garrotte => {
                        if coeff.abs() > level_threshold {
                            let threshold_sq = level_threshold * level_threshold;
                            let coeff_sq = coeff * coeff;
                            coeff * (1.0 - threshold_sq / coeff_sq.max(f64::EPSILON))
                        } else {
                            0.0
                        }
                    }
                    ThresholdMethod::Greater => {
                        if coeff > level_threshold {
                            coeff - level_threshold
                        } else {
                            0.0
                        }
                    }
                    ThresholdMethod::Less => {
                        if coeff < -level_threshold {
                            coeff + level_threshold
                        } else {
                            0.0
                        }
                    }
                    ThresholdMethod::Adaptive => {
                        // Adaptive thresholding based on local statistics
                        let local_threshold =
                            level_threshold * (1.0 + 0.1 * noise_analysis.spatial_correlation);
                        if coeff.abs() > local_threshold {
                            coeff.signum() * (coeff.abs() - local_threshold)
                        } else {
                            0.0
                        }
                    }
                };

                denoised[[level, y, x]] = denoised_coeff;
            }
        }
    }

    Ok(denoised)
}

/// Compute denoising quality metrics
pub fn compute_denoising_quality_metrics(
    noisy_image: &Array2<f64>,
    denoised_image: &Array2<f64>,
    _noise_analysis: &NoiseAnalysis,
) -> SignalResult<DenoisingQualityMetrics> {
    if noisy_image.dim() != denoised_image.dim() {
        return Err(SignalError::ValueError(
            "Noisy and denoised images must have same dimensions".to_string(),
        ));
    }

    // Compute PSNR (using noisy image as reference - in practice, clean image would be ideal)
    let max_val = noisy_image.iter().cloned().fold(0.0f64, f64::max);
    let mut mse = 0.0;
    let pixel_count = noisy_image.len();

    for (&noisy, &denoised) in noisy_image.iter().zip(denoised_image.iter()) {
        let diff = noisy - denoised;
        mse += diff * diff;
    }
    mse /= pixel_count as f64;

    let psnr = if mse > 0.0 {
        20.0 * (max_val * max_val / mse).log10()
    } else {
        f64::INFINITY
    };

    // Compute SSIM between noisy and denoised (as a measure of structure preservation)
    let ssim = compute_ssim(noisy_image, denoised_image)?;

    // Edge preservation index (simplified)
    let edge_preservation_index = compute_edge_preservation_index(noisy_image, denoised_image)?;

    // Artifacts score (simplified - based on smoothness)
    let artifacts_score = compute_artifacts_score(denoised_image)?;

    Ok(DenoisingQualityMetrics {
        psnr,
        ssim,
        edge_preservation_index,
        artifacts_score,
    })
}

/// Compute SSIM between two images
fn compute_ssim(image1: &Array2<f64>, image2: &Array2<f64>) -> SignalResult<f64> {
    let mean1 = image1.iter().sum::<f64>() / image1.len() as f64;
    let mean2 = image2.iter().sum::<f64>() / image2.len() as f64;

    let var1 = image1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / image1.len() as f64;
    let var2 = image2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / image2.len() as f64;

    let covariance = image1
        .iter()
        .zip(image2.iter())
        .map(|(&x1, &x2)| (x1 - mean1) * (x2 - mean2))
        .sum::<f64>()
        / image1.len() as f64;

    let c1 = 0.01 * 0.01;
    let c2 = 0.03 * 0.03;

    let numerator = (2.0 * mean1 * mean2 + c1) * (2.0 * covariance + c2);
    let denominator = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2);

    Ok(if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    })
}

/// Compute edge preservation index
fn compute_edge_preservation_index(
    original: &Array2<f64>,
    processed: &Array2<f64>,
) -> SignalResult<f64> {
    let (height, width) = original.dim();

    if height < 3 || width < 3 {
        return Ok(1.0);
    }

    let mut edge_preservation_sum = 0.0;
    let mut edge_count = 0;

    // Simple edge detection using gradients
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Compute gradients in original image
            let gx_orig = original[[y, x + 1]] - original[[y, x - 1]];
            let gy_orig = original[[y + 1, x]] - original[[y - 1, x]];
            let grad_mag_orig = (gx_orig * gx_orig + gy_orig * gy_orig).sqrt();

            // Compute gradients in processed image
            let gx_proc = processed[[y, x + 1]] - processed[[y, x - 1]];
            let gy_proc = processed[[y + 1, x]] - processed[[y - 1, x]];
            let grad_mag_proc = (gx_proc * gx_proc + gy_proc * gy_proc).sqrt();

            // Edge preservation ratio
            if grad_mag_orig > 0.01 {
                // Only consider significant edges
                let preservation = if grad_mag_orig > 0.0 {
                    grad_mag_proc / grad_mag_orig
                } else {
                    1.0
                };
                edge_preservation_sum += preservation.min(1.0);
                edge_count += 1;
            }
        }
    }

    Ok(if edge_count > 0 {
        edge_preservation_sum / edge_count as f64
    } else {
        1.0
    })
}

/// Compute artifacts score
fn compute_artifacts_score(image: &Array2<f64>) -> SignalResult<f64> {
    let (height, width) = image.dim();

    if height < 3 || width < 3 {
        return Ok(0.0);
    }

    let mut artifacts_sum = 0.0;
    let mut pixel_count = 0;

    // Look for artifacts using second-order derivatives (smoothness measure)
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Laplacian operator
            let laplacian = -4.0 * image[[y, x]]
                + image[[y - 1, x]]
                + image[[y + 1, x]]
                + image[[y, x - 1]]
                + image[[y, x + 1]];

            artifacts_sum += laplacian.abs();
            pixel_count += 1;
        }
    }

    // Normalize artifacts score (lower is better)
    let artifacts_score = if pixel_count > 0 {
        1.0 - (artifacts_sum / pixel_count as f64).min(1.0)
    } else {
        1.0
    };

    Ok(artifacts_score)
}

/// Compute coefficient statistics
pub fn compute_coefficient_statistics(coefficients: &Array3<f64>) -> CoefficientStatistics {
    let flat_coeffs: Vec<f64> = coefficients.iter().cloned().collect();

    if flat_coeffs.is_empty() {
        return CoefficientStatistics {
            mean: 0.0,
            variance: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        };
    }

    let n = flat_coeffs.len() as f64;
    let mean = flat_coeffs.iter().sum::<f64>() / n;

    let variance = flat_coeffs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    let std_dev = variance.sqrt();

    let (skewness, kurtosis) = if std_dev > 0.0 {
        let third_moment = flat_coeffs
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>()
            / n;

        let fourth_moment = flat_coeffs
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>()
            / n;

        (third_moment, fourth_moment - 3.0) // Excess kurtosis
    } else {
        (0.0, 0.0)
    };

    CoefficientStatistics {
        mean,
        variance,
        skewness,
        kurtosis,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_compute_coefficient_utilization() {
        let mut coefficients = Array3::zeros((2, 4, 4));
        coefficients[[0, 0, 0]] = 1.0;
        coefficients[[0, 1, 1]] = 2.0;
        coefficients[[1, 2, 2]] = 3.0;

        let utilization = compute_coefficient_utilization(&coefficients);
        assert_eq!(utilization, 3.0 / 32.0); // 3 non-zero out of 32 total
    }

    #[test]
    fn test_analyze_noise_characteristics() {
        let noisy_image = Array2::from_shape_fn((16, 16), |(i, j)| {
            (i + j) as f64 + 0.1 * scirs2_core::random::random::<f64>() // Simple pattern + noise
        });

        let wavelet = crate::dwt::Wavelet::DB(2);
        let analysis = analyze_noise_characteristics(&noisy_image, &wavelet);

        assert!(analysis.is_ok());
        let analysis = analysis.unwrap();
        assert!(analysis.estimated_variance > 0.0);
    }

    #[test]
    fn test_apply_adaptive_denoising() {
        let coefficients =
            Array3::from_shape_fn(
                (2, 8, 8),
                |(l, i, j)| {
                    if (i + j + l) % 3 == 0 {
                        1.0
                    } else {
                        0.1
                    }
                },
            );

        let noise_analysis = NoiseAnalysis {
            noise_type: NoiseType::Gaussian,
            estimated_variance: 0.01,
            spatial_correlation: 0.1,
            frequency_characteristics: vec![0.01; 5],
        };

        let denoising_config = AdvancedRefinedDenoisingConfig {
            noise_variance: Some(0.01),
            threshold_method: ThresholdMethod::Soft,
            edge_preservation: 0.8,
            perceptual_weighting: true,
        };

        let tree = DecompositionTree {
            nodes: Vec::new(),
            optimal_basis: Vec::new(),
            cost_function: CostFunction::Entropy,
            traversal_stats: TreeTraversalStats {
                total_nodes: 0,
                leaf_nodes: 0,
                max_depth: 0,
                avg_branching_factor: 0.0,
            },
        };

        let result =
            apply_adaptive_denoising(&coefficients, &noise_analysis, &tree, &denoising_config);

        assert!(result.is_ok());
        let denoised = result.unwrap();
        assert_eq!(denoised.shape(), coefficients.shape());
    }

    #[test]
    fn test_compute_coefficient_statistics() {
        let coefficients =
            Array3::from_shape_fn((2, 4, 4), |(_, i, j)| (i as f64 - 2.0) * (j as f64 - 2.0));

        let stats = compute_coefficient_statistics(&coefficients);
        assert!(stats.mean.is_finite());
        assert!(stats.variance >= 0.0);
    }

    #[test]
    fn test_reconstruct_from_coefficients_simple() {
        let coefficients = Array3::from_shape_fn((3, 8, 8), |(l, i, j)| {
            1.0 / (1.0 + l as f64) * ((i + j) as f64 / 16.0)
        });

        let result = reconstruct_from_coefficients_simple(&coefficients);
        assert!(result.is_ok());

        let reconstructed = result.unwrap();
        assert_eq!(reconstructed.dim(), (8, 8));
    }
}
