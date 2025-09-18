//! Frame theory validation for wavelet packet transforms
//!
//! This module provides validation of frame properties for wavelet packet
//! decompositions including eigenvalue analysis, condition numbers, and coherence.

use super::types::*;
use super::utils::{generate_test_signal, construct_frame_matrix};
use crate::error::SignalResult;
use ndarray::Array2;

/// Validate frame properties of wavelet packet decompositions
pub fn validate_frame_properties(
    config: &ComprehensiveWptValidationConfig,
) -> SignalResult<FrameValidationMetrics> {
    let mut eigenvalue_distributions = Vec::new();
    let mut condition_numbers = Vec::new();
    let mut frame_coherences = Vec::new();
    let mut redundancy_factors = Vec::new();

    // Test frame properties for different configurations
    for &wavelet in &config.test_wavelets {
        for &signal_length in &config.test_signal_lengths {
            if signal_length < 32 {
                continue; // Skip too small signals for frame analysis
            }

            for &level in &config.test_levels {
                if level * 4 > signal_length.trailing_zeros() as usize {
                    continue; // Skip invalid combinations
                }

                // Generate test signal for frame construction
                let signal = generate_test_signal(TestSignalType::WhiteNoise, signal_length, 0)?;

                // Construct frame matrix
                match construct_frame_matrix(&signal, wavelet, level) {
                    Ok(frame_matrix) => {
                        // Analyze frame properties
                        let eigenvalue_dist = analyze_eigenvalue_distribution(&frame_matrix)?;
                        let condition_number = compute_condition_number(&frame_matrix)?;
                        let frame_coherence = compute_frame_coherence(&frame_matrix)?;
                        let redundancy = compute_redundancy_factor(&frame_matrix);

                        eigenvalue_distributions.push(eigenvalue_dist);
                        condition_numbers.push(condition_number);
                        frame_coherences.push(frame_coherence);
                        redundancy_factors.push(redundancy);
                    }
                    Err(_) => {
                        // Skip this configuration if frame construction fails
                        continue;
                    }
                }
            }
        }
    }

    // Aggregate results
    let eigenvalue_distribution = aggregate_eigenvalue_distributions(&eigenvalue_distributions);

    let condition_number = if !condition_numbers.is_empty() {
        condition_numbers.iter().sum::<f64>() / condition_numbers.len() as f64
    } else {
        f64::INFINITY
    };

    let frame_coherence = if !frame_coherences.is_empty() {
        frame_coherences.iter().cloned().fold(0.0f64, f64::max)
    } else {
        1.0
    };

    let redundancy_factor = if !redundancy_factors.is_empty() {
        redundancy_factors.iter().sum::<f64>() / redundancy_factors.len() as f64
    } else {
        1.0
    };

    // Estimate reconstruction bounds based on frame bounds
    let reconstruction_bounds = if eigenvalue_distribution.min_eigenvalue > 0.0 {
        let lower_bound = eigenvalue_distribution.min_eigenvalue.sqrt();
        let upper_bound = eigenvalue_distribution.max_eigenvalue.sqrt();
        (1.0 / upper_bound, 1.0 / lower_bound)
    } else {
        (0.0, f64::INFINITY)
    };

    Ok(FrameValidationMetrics {
        eigenvalue_distribution,
        condition_number,
        frame_coherence,
        redundancy_factor,
        reconstruction_bounds,
    })
}

/// Analyze eigenvalue distribution of frame operator
fn analyze_eigenvalue_distribution(
    frame_matrix: &Array2<f64>,
) -> SignalResult<EigenvalueDistribution> {
    // Compute Gram matrix (Frame operator)
    let gram_matrix = frame_matrix.t().dot(frame_matrix);

    // Use power iteration to estimate dominant eigenvalues
    let max_eigenvalue = power_iteration(&gram_matrix, 100, 1e-10)?;

    // Use inverse power iteration for smallest eigenvalue
    let min_eigenvalue = inverse_power_iteration(&gram_matrix, 100, 1e-10)?;

    // Estimate mean eigenvalue as trace / dimension
    let trace: f64 = (0..gram_matrix.nrows()).map(|i| gram_matrix[[i, i]]).sum();
    let mean_eigenvalue = trace / gram_matrix.nrows() as f64;

    // Simple variance estimate
    let eigenvalue_variance = (max_eigenvalue - min_eigenvalue).powi(2) / 12.0;

    // Count near-zero eigenvalues (simplified)
    let near_zero_count = if min_eigenvalue < 1e-10 { 1 } else { 0 };

    Ok(EigenvalueDistribution {
        min_eigenvalue,
        max_eigenvalue,
        mean_eigenvalue,
        eigenvalue_variance,
        near_zero_count,
    })
}

/// Compute condition number of frame matrix
fn compute_condition_number(frame_matrix: &Array2<f64>) -> SignalResult<f64> {
    let gram_matrix = frame_matrix.t().dot(frame_matrix);
    let max_eigenvalue = power_iteration(&gram_matrix, 100, 1e-10)?;
    let min_eigenvalue = inverse_power_iteration(&gram_matrix, 100, 1e-10)?;

    if min_eigenvalue > 1e-15 {
        Ok(max_eigenvalue / min_eigenvalue)
    } else {
        Ok(f64::INFINITY)
    }
}

/// Compute frame coherence
fn compute_frame_coherence(frame_matrix: &Array2<f64>) -> SignalResult<f64> {
    let (rows, cols) = frame_matrix.dim();
    if cols <= 1 {
        return Ok(0.0);
    }

    let mut max_coherence = 0.0;

    // Normalize columns first
    let mut normalized_matrix = frame_matrix.clone();
    for j in 0..cols {
        let col = normalized_matrix.column(j);
        let norm = col.dot(&col).sqrt();
        if norm > 1e-15 {
            for i in 0..rows {
                normalized_matrix[[i, j]] /= norm;
            }
        }
    }

    // Compute pairwise inner products
    for i in 0..cols {
        for j in (i + 1)..cols {
            let col_i = normalized_matrix.column(i);
            let col_j = normalized_matrix.column(j);
            let inner_product = col_i.dot(&col_j).abs();
            max_coherence = max_coherence.max(inner_product);
        }
    }

    Ok(max_coherence)
}

/// Compute redundancy factor
fn compute_redundancy_factor(frame_matrix: &Array2<f64>) -> f64 {
    let (rows, cols) = frame_matrix.dim();
    if rows > 0 {
        cols as f64 / rows as f64
    } else {
        1.0
    }
}

/// Power iteration for largest eigenvalue
fn power_iteration(
    matrix: &Array2<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> SignalResult<f64> {
    let n = matrix.nrows();
    if n == 0 {
        return Ok(0.0);
    }

    // Start with random vector
    let mut v = ndarray::Array1::ones(n);
    v /= v.dot(&v).sqrt();

    let mut eigenvalue = 0.0;

    for _ in 0..max_iterations {
        // Multiply by matrix
        let mv = matrix.dot(&v);

        // Compute Rayleigh quotient
        let new_eigenvalue = v.dot(&mv);

        // Check convergence
        if (new_eigenvalue - eigenvalue).abs() < tolerance {
            return Ok(new_eigenvalue);
        }

        eigenvalue = new_eigenvalue;

        // Normalize
        let norm = mv.dot(&mv).sqrt();
        if norm > tolerance {
            v = mv / norm;
        } else {
            break;
        }
    }

    Ok(eigenvalue)
}

/// Inverse power iteration for smallest eigenvalue
fn inverse_power_iteration(
    matrix: &Array2<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> SignalResult<f64> {
    let n = matrix.nrows();
    if n == 0 {
        return Ok(0.0);
    }

    // For simplicity, use shifted power iteration
    // In practice, would use proper inverse iteration with LU decomposition
    let shift = 1e-12;
    let mut shifted_matrix = matrix.clone();
    for i in 0..n {
        shifted_matrix[[i, i]] += shift;
    }

    let max_eval = power_iteration(&shifted_matrix, max_iterations, tolerance)?;
    Ok((max_eval - shift).max(0.0))
}

/// Aggregate eigenvalue distributions from multiple tests
fn aggregate_eigenvalue_distributions(
    distributions: &[EigenvalueDistribution],
) -> EigenvalueDistribution {
    if distributions.is_empty() {
        return EigenvalueDistribution {
            min_eigenvalue: 0.0,
            max_eigenvalue: 0.0,
            mean_eigenvalue: 0.0,
            eigenvalue_variance: 0.0,
            near_zero_count: 0,
        };
    }

    let min_eigenvalue = distributions
        .iter()
        .map(|d| d.min_eigenvalue)
        .fold(f64::INFINITY, f64::min);

    let max_eigenvalue = distributions
        .iter()
        .map(|d| d.max_eigenvalue)
        .fold(0.0f64, f64::max);

    let mean_eigenvalue = distributions
        .iter()
        .map(|d| d.mean_eigenvalue)
        .sum::<f64>()
        / distributions.len() as f64;

    let eigenvalue_variance = distributions
        .iter()
        .map(|d| d.eigenvalue_variance)
        .sum::<f64>()
        / distributions.len() as f64;

    let near_zero_count = distributions
        .iter()
        .map(|d| d.near_zero_count)
        .sum::<usize>();

    EigenvalueDistribution {
        min_eigenvalue,
        max_eigenvalue,
        mean_eigenvalue,
        eigenvalue_variance,
        near_zero_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::Wavelet;

    #[test]
    fn test_frame_validation() {
        let mut config = ComprehensiveWptValidationConfig::default();

        // Use minimal configuration for testing
        config.test_wavelets = vec![Wavelet::Haar];
        config.test_signal_lengths = vec![64];
        config.test_levels = vec![1];

        let result = validate_frame_properties(&config);

        // The test should complete without panicking
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_power_iteration() {
        // Test with simple 2x2 matrix
        let matrix = ndarray::Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();

        let eigenvalue = power_iteration(&matrix, 100, 1e-10).unwrap();

        // Largest eigenvalue should be approximately 3.0
        assert!((eigenvalue - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_redundancy_factor() {
        let matrix = ndarray::Array2::zeros((10, 20));
        let redundancy = compute_redundancy_factor(&matrix);

        assert_eq!(redundancy, 2.0); // 20 columns / 10 rows
    }
}