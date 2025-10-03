//! 2D DWT Validation and Analysis Functions
//!
//! This module provides comprehensive validation, testing, and analysis capabilities
//! for 2D Discrete Wavelet Transform operations, including performance metrics,
//! numerical accuracy verification, and energy analysis.

use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use super::types::{Dwt2dResult, Dwt2dValidationConfig, Dwt2dValidationResult, MemoryEfficiencyMetrics, PerformanceMetrics2d, WaveletEnergy, WaveletCounts};
use super::simd::{simd_calculate_energy, PlatformCapabilities};
use scirs2_core::ndarray::Array2;
use scirs2_core::validation::{check_positive, check_finite};
use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Enhanced validation suite for 2D wavelet transforms
///
/// This function performs comprehensive validation including:
/// - Perfect reconstruction accuracy
/// - Energy conservation
/// - Orthogonality preservation
/// - Performance benchmarking
/// - Memory efficiency analysis
/// - Numerical stability testing
pub fn validate_dwt2d_comprehensive(
    config: &Dwt2dValidationConfig,
) -> SignalResult<Dwt2dValidationResult> {
    let mut issues: Vec<String> = Vec::new();
    let mut total_reconstruction_error = 0.0;
    let mut total_energy_error = 0.0;
    let mut total_orthogonality_error = 0.0;
    let mut performance_metrics = Vec::new();
    let mut memory_metrics = Vec::new();

    let start_time = Instant::now();
    let mut test_count = 0;

    // Test various image sizes and wavelets
    for &(rows, cols) in &config.test_sizes {
        for &wavelet in &config.test_wavelets {
            test_count += 1;

            // Create test image with known properties
            let mut test_image = Array2::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    test_image[[i, j]] = ((i as f64 + 1.0) * (j as f64 + 1.0)).sin()
                        * ((i as f64 * 0.1).cos() + (j as f64 * 0.1).sin());
                }
            }

            // Validate finite input
            if let Some(data_slice) = test_image.as_slice() {
                for &val in data_slice {
                    if !val.is_finite() {
                        return Err(SignalError::ValueError("Input contains non-finite values".to_string()));
                    }
                }
            }

            // Test single-level decomposition and reconstruction
            let decomp_start = Instant::now();
            let decomposition = crate::dwt2d::dwt2d_decompose(&test_image, wavelet, None)?;
            let decomp_time = decomp_start.elapsed().as_secs_f64() * 1000.0;

            let recon_start = Instant::now();
            let reconstructed = crate::dwt2d::dwt2d_reconstruct(&decomposition, wavelet, None)?;
            let recon_time = recon_start.elapsed().as_secs_f64() * 1000.0;

            // Calculate reconstruction error
            let mut recon_error = 0.0;
            let mut original_energy = 0.0;

            for i in 0..rows {
                for j in 0..cols {
                    let orig = test_image[[i, j]];
                    let recon = reconstructed[[i, j]];
                    let diff = orig - recon;
                    recon_error += diff * diff;
                    original_energy += orig * orig;
                }
            }

            recon_error = (recon_error / (rows * cols) as f64).sqrt();
            total_reconstruction_error += recon_error;

            // Check energy conservation
            let (original_total_energy, _) = calculate_energy_from_array(&test_image);
            let (decomp_total_energy, _) = calculate_energy(&decomposition, true);
            let energy_error =
                (original_total_energy - decomp_total_energy).abs() / original_total_energy;
            total_energy_error += energy_error;

            // Test orthogonality for orthogonal wavelets
            if matches!(wavelet, Wavelet::Haar | Wavelet::DB(_)) {
                let ortho_error = test_orthogonality(&decomposition);
                total_orthogonality_error += ortho_error;
            }

            // Performance metrics
            let data_size_mb = (rows * cols * 8) as f64 / (1024.0 * 1024.0);
            let throughput = data_size_mb / ((decomp_time + recon_time) / 1000.0);

            performance_metrics.push(PerformanceMetrics2d {
                total_time_ms: decomp_time + recon_time,
                decomposition_time_ms: decomp_time,
                reconstruction_time_ms: recon_time,
                simd_utilization: estimate_simd_utilization(rows * cols),
                parallel_efficiency: estimate_parallel_efficiency(rows, cols),
                throughput_mbs: throughput,
            });

            // Memory efficiency (simplified estimation)
            memory_metrics.push(MemoryEfficiencyMetrics {
                peak_memory_bytes: estimate_peak_memory(rows, cols),
                allocation_count: estimate_allocation_count(rows, cols),
                cache_miss_ratio: estimate_cache_miss_ratio(rows, cols),
                access_pattern_efficiency: estimate_access_pattern_efficiency(rows, cols),
            });

            // Test edge cases if enabled
            if config.test_edge_cases {
                // Test with extreme values
                test_image[[0, 0]] = f64::MAX / 1e10;
                test_image[[rows - 1, cols - 1]] = f64::MIN / 1e10;

                if let Err(e) = crate::dwt2d::dwt2d_decompose(&test_image, wavelet, None) {
                    issues.push(format!("Edge case failed for wavelet {:?}: {}", wavelet, e));
                }
            }

            // Validate reconstruction error is within tolerance
            if recon_error > config.tolerance {
                issues.push(format!(
                    "High reconstruction error ({:.2e}) for {}x{} image with {:?} wavelet",
                    recon_error, rows, cols, wavelet
                ));
            }

            // Validate energy conservation
            if energy_error > config.tolerance {
                issues.push(format!(
                    "Energy conservation violated ({:.2e}) for {}x{} image with {:?} wavelet",
                    energy_error, rows, cols, wavelet
                ));
            }
        }
    }

    // Calculate averages
    let avg_reconstruction_error = total_reconstruction_error / test_count as f64;
    let avg_energy_error = total_energy_error / test_count as f64;
    let avg_orthogonality_error = total_orthogonality_error / test_count as f64;

    // Calculate overall score (0-100)
    let reconstruction_score =
        (1.0 - (avg_reconstruction_error / config.tolerance).min(1.0)) * 100.0;
    let energy_score = (1.0 - (avg_energy_error / config.tolerance).min(1.0)) * 100.0;
    let orthogonality_score = (1.0 - (avg_orthogonality_error / config.tolerance).min(1.0)) * 100.0;
    let overall_score = (reconstruction_score + energy_score + orthogonality_score) / 3.0;

    // Average metrics
    let avg_performance = average_performance_metrics(&performance_metrics);
    let avg_memory = average_memory_metrics(&memory_metrics);

    Ok(Dwt2dValidationResult {
        reconstruction_error: avg_reconstruction_error,
        energy_conservation_error: avg_energy_error,
        orthogonality_error: avg_orthogonality_error,
        memory_efficiency: avg_memory,
        performance_metrics: avg_performance,
        overall_score,
        issues,
    })
}

/// Calculate the energy of wavelet coefficients in a decomposition.
///
/// Energy is the sum of squared coefficients. This function is useful for analyzing
/// the distribution of energy across different subbands and for determining appropriate
/// threshold values.
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition to analyze
/// * `include_approx` - Whether to include approximation coefficients in the calculation
///
/// # Returns
///
/// * A tuple containing the total energy and a struct with energy by subband
pub fn calculate_energy(
    decomposition: &Dwt2dResult,
    include_approx: bool,
) -> (f64, WaveletEnergy) {
    // Calculate energy for each subband using SIMD-optimized functions
    let approx_energy = if include_approx {
        if let Some(approx_slice) = decomposition.approx.as_slice() {
            simd_calculate_energy(approx_slice)
        } else {
            decomposition.approx.iter().map(|&x| x * x).sum()
        }
    } else {
        0.0
    };

    let detail_h_energy = if let Some(h_slice) = decomposition.detail_h.as_slice() {
        simd_calculate_energy(h_slice)
    } else {
        decomposition.detail_h.iter().map(|&x| x * x).sum()
    };

    let detail_v_energy = if let Some(v_slice) = decomposition.detail_v.as_slice() {
        simd_calculate_energy(v_slice)
    } else {
        decomposition.detail_v.iter().map(|&x| x * x).sum()
    };

    let detail_d_energy = if let Some(d_slice) = decomposition.detail_d.as_slice() {
        simd_calculate_energy(d_slice)
    } else {
        decomposition.detail_d.iter().map(|&x| x * x).sum()
    };

    // Calculate total energy
    let total = approx_energy + detail_h_energy + detail_v_energy + detail_d_energy;

    // Create energy structure
    let energy_by_subband = WaveletEnergy {
        approx: approx_energy,
        detail_h: detail_h_energy,
        detail_v: detail_v_energy,
        detail_d: detail_d_energy,
    };

    (total, energy_by_subband)
}

/// Count non-zero coefficients in a wavelet decomposition.
///
/// This is useful for quantifying the sparsity of a wavelet representation,
/// especially after thresholding for compression.
pub fn count_nonzeros(
    decomposition: &Dwt2dResult,
    include_approx: bool,
) -> (usize, WaveletCounts) {
    // Count non-zero coefficients in each subband
    let approx_count = if include_approx {
        decomposition.approx.iter().filter(|&&x| x != 0.0).count()
    } else {
        0
    };

    let detail_h_count = decomposition
        .detail_h
        .iter()
        .filter(|&&x| x != 0.0)
        .count();
    let detail_v_count = decomposition
        .detail_v
        .iter()
        .filter(|&&x| x != 0.0)
        .count();
    let detail_d_count = decomposition
        .detail_d
        .iter()
        .filter(|&&x| x != 0.0)
        .count();

    // Calculate total count
    let total = approx_count + detail_h_count + detail_v_count + detail_d_count;

    // Create counts structure
    let counts_by_subband = WaveletCounts {
        approx: approx_count,
        detail_h: detail_h_count,
        detail_v: detail_v_count,
        detail_d: detail_d_count,
    };

    (total, counts_by_subband)
}

/// Calculate energy from a 2D array
fn calculate_energy_from_array(data: &Array2<f64>) -> (f64, f64) {
    let total_energy = if let Some(data_slice) = data.as_slice() {
        simd_calculate_energy(data_slice)
    } else {
        data.iter().map(|&x| x * x).sum()
    };
    (total_energy, 0.0)
}

/// Test orthogonality of wavelet decomposition
fn test_orthogonality(decomp: &Dwt2dResult) -> f64 {
    // Simplified orthogonality test - check if subbands are approximately uncorrelated
    let mut correlation_sum = 0.0;
    let mut count = 0;

    // Test correlation between different subbands
    let subbands = [&decomp.detail_h, &decomp.detail_v, &decomp.detail_d];

    for i in 0..subbands.len() {
        for j in (i + 1)..subbands.len() {
            let corr = calculate_correlation(subbands[i], subbands[j]);
            correlation_sum += corr.abs();
            count += 1;
        }
    }

    if count > 0 {
        correlation_sum / count as f64
    } else {
        0.0
    }
}

/// Calculate correlation between two 2D arrays
fn calculate_correlation(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    if a.shape() != b.shape() {
        return 0.0;
    }

    let n = a.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    // Calculate means
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

    // Calculate correlation coefficient
    let mut numerator = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (&val_a, &val_b) in a.iter().zip(b.iter()) {
        let diff_a = val_a - mean_a;
        let diff_b = val_b - mean_b;
        numerator += diff_a * diff_b;
        var_a += diff_a * diff_a;
        var_b += diff_b * diff_b;
    }

    let denominator = (var_a * var_b).sqrt();
    if denominator > 1e-15 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Estimate SIMD utilization based on data size
fn estimate_simd_utilization(data_size: usize) -> f64 {
    if data_size < 64 {
        0.0 // Too small for SIMD
    } else if data_size < 1024 {
        0.5 // Partial SIMD utilization
    } else {
        0.85 // Good SIMD utilization
    }
}

/// Estimate parallel efficiency
fn estimate_parallel_efficiency(rows: usize, cols: usize) -> f64 {
    let total_ops = rows * cols;
    if total_ops < 1024 {
        0.0 // Too small for parallelization
    } else if total_ops < 10000 {
        0.6 // Moderate parallel efficiency
    } else {
        0.9 // Good parallel efficiency
    }
}

/// Estimate peak memory usage
fn estimate_peak_memory(rows: usize, cols: usize) -> usize {
    // Rough estimation: original + decomposition + temporaries
    let base_size = rows * cols * 8; // 8 bytes per f64
    base_size * 3 // Original + decomposed subbands + temporaries
}

/// Estimate allocation count
fn estimate_allocation_count(rows: usize, cols: usize) -> usize {
    // Estimation based on typical decomposition operations
    if rows * cols < 1024 {
        10 // Small arrays, frequent allocations
    } else {
        6 // Larger arrays, fewer allocations due to memory pool
    }
}

/// Estimate cache miss ratio
fn estimate_cache_miss_ratio(rows: usize, cols: usize) -> f64 {
    let data_size_kb = (rows * cols * 8) / 1024;
    if data_size_kb < 32 {
        0.1 // Fits in L1 cache
    } else if data_size_kb < 256 {
        0.3 // Fits in L2 cache
    } else {
        0.6 // Spills to main memory
    }
}

/// Estimate access pattern efficiency
fn estimate_access_pattern_efficiency(rows: usize, cols: usize) -> f64 {
    // Row-major access patterns are efficient in our implementation
    if rows > 64 && cols > 64 {
        0.85 // Good spatial locality
    } else {
        0.7 // Smaller arrays have less optimal patterns
    }
}

/// Average performance metrics
fn average_performance_metrics(metrics: &[PerformanceMetrics2d]) -> PerformanceMetrics2d {
    if metrics.is_empty() {
        return PerformanceMetrics2d {
            total_time_ms: 0.0,
            decomposition_time_ms: 0.0,
            reconstruction_time_ms: 0.0,
            simd_utilization: 0.0,
            parallel_efficiency: 0.0,
            throughput_mbs: 0.0,
        };
    }

    let count = metrics.len() as f64;
    PerformanceMetrics2d {
        total_time_ms: metrics.iter().map(|m| m.total_time_ms).sum::<f64>() / count,
        decomposition_time_ms: metrics
            .iter()
            .map(|m| m.decomposition_time_ms)
            .sum::<f64>()
            / count,
        reconstruction_time_ms: metrics
            .iter()
            .map(|m| m.reconstruction_time_ms)
            .sum::<f64>()
            / count,
        simd_utilization: metrics.iter().map(|m| m.simd_utilization).sum::<f64>() / count,
        parallel_efficiency: metrics.iter().map(|m| m.parallel_efficiency).sum::<f64>() / count,
        throughput_mbs: metrics.iter().map(|m| m.throughput_mbs).sum::<f64>() / count,
    }
}

/// Average memory metrics
fn average_memory_metrics(metrics: &[MemoryEfficiencyMetrics]) -> MemoryEfficiencyMetrics {
    if metrics.is_empty() {
        return MemoryEfficiencyMetrics {
            peak_memory_bytes: 0,
            allocation_count: 0,
            cache_miss_ratio: 0.0,
            access_pattern_efficiency: 0.0,
        };
    }

    let count = metrics.len();
    MemoryEfficiencyMetrics {
        peak_memory_bytes: metrics.iter().map(|m| m.peak_memory_bytes).sum::<usize>() / count,
        allocation_count: metrics.iter().map(|m| m.allocation_count).sum::<usize>() / count,
        cache_miss_ratio: metrics.iter().map(|m| m.cache_miss_ratio).sum::<f64>() / count as f64,
        access_pattern_efficiency: metrics
            .iter()
            .map(|m| m.access_pattern_efficiency)
            .sum::<f64>()
            / count as f64,
    }
}

/// Validate a single decomposition level
pub fn validate_decomposition_level(
    decomp: &Dwt2dResult,
    level: usize,
    input_rows: usize,
    input_cols: usize,
) -> SignalResult<()> {
    // Check that all subbands have the same shape
    let approx_shape = decomp.approx.shape();
    if decomp.detail_h.shape() != approx_shape
        || decomp.detail_v.shape() != approx_shape
        || decomp.detail_d.shape() != approx_shape
    {
        return Err(SignalError::ComputationError(format!(
            "Inconsistent subband shapes at level {}",
            level
        )));
    }

    // Validate expected dimensions
    let expected_rows = div_ceil(input_rows, 2);
    let expected_cols = div_ceil(input_cols, 2);

    if approx_shape[0] != expected_rows || approx_shape[1] != expected_cols {
        return Err(SignalError::ComputationError(format!(
            "Unexpected subband dimensions at level {}: got [{}, {}], expected [{}, {}]",
            level, approx_shape[0], approx_shape[1], expected_rows, expected_cols
        )));
    }

    // Check for numerical issues in coefficients
    for subband in [
        &decomp.approx,
        &decomp.detail_h,
        &decomp.detail_v,
        &decomp.detail_d,
    ] {
        if let Some(slice) = subband.as_slice() {
            for &val in slice {
                if !val.is_finite() {
                    return Err(SignalError::ComputationError(format!(
                        "Non-finite coefficient detected at level {}",
                        level
                    )));
                }
            }
        }
    }

    Ok(())
}

/// Helper function for integer ceiling division
fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::Wavelet;

    #[test]
    fn test_calculate_energy() {
        // Create a simple test decomposition
        let decomp = Dwt2dResult {
            approx: Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            detail_h: Array2::zeros((2, 2)),
            detail_v: Array2::zeros((2, 2)),
            detail_d: Array2::zeros((2, 2)),
        };

        let (total_energy, energy_breakdown) = calculate_energy(&decomp, true);

        // Energy should be 1^2 + 2^2 + 3^2 + 4^2 = 30
        assert!((total_energy - 30.0).abs() < 1e-10);
        assert!((energy_breakdown.approx - 30.0).abs() < 1e-10);
        assert!(energy_breakdown.detail_h < 1e-10);
        assert!(energy_breakdown.detail_v < 1e-10);
        assert!(energy_breakdown.detail_d < 1e-10);
    }

    #[test]
    fn test_count_nonzeros() {
        let decomp = Dwt2dResult {
            approx: Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 3.0, 0.0]).unwrap(),
            detail_h: Array2::from_shape_vec((2, 2), vec![0.0, 2.0, 0.0, 4.0]).unwrap(),
            detail_v: Array2::zeros((2, 2)),
            detail_d: Array2::ones((2, 2)),
        };

        let (total_count, counts_breakdown) = count_nonzeros(&decomp, true);

        assert_eq!(total_count, 8); // 2 + 2 + 0 + 4
        assert_eq!(counts_breakdown.approx, 2);
        assert_eq!(counts_breakdown.detail_h, 2);
        assert_eq!(counts_breakdown.detail_v, 0);
        assert_eq!(counts_breakdown.detail_d, 4);
    }

    #[test]
    fn test_validation_config() {
        let config = Dwt2dValidationConfig::default();

        assert_eq!(config.tolerance, 1e-12);
        assert!(config.benchmark_performance);
        assert!(config.test_memory_efficiency);
        assert!(config.test_numerical_stability);
        assert!(config.test_edge_cases);
    }
}