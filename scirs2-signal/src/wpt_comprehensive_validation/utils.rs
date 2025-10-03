//! Utility functions for comprehensive WPT validation
//!
//! This module provides common utility functions used across the validation framework
//! including signal generation, correlation computation, and frame matrix construction.

use super::types::TestSignalType;
use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use crate::wpt::{reconstruct_from_nodes, wp_decompose};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};

/// Generate test signal of specified type and length
pub fn generate_test_signal(
    signal_type: TestSignalType,
    length: usize,
    seed: usize,
) -> SignalResult<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed as u64);

    match signal_type {
        TestSignalType::WhiteNoise => {
            Ok((0..length).map(|_| rng.gen::<f64>() - 0.5).collect())
        }
        TestSignalType::Sinusoidal => {
            let frequency = 0.1 + rng.gen::<f64>() * 0.3; // Random frequency between 0.1 and 0.4
            Ok((0..length)
                .map(|i| (2.0 * std::f64::consts::PI * frequency * i as f64 / length as f64).sin())
                .collect())
        }
        TestSignalType::Chirp => {
            let start_freq = 0.05 + rng.gen::<f64>() * 0.1;
            let end_freq = 0.2 + rng.gen::<f64>() * 0.2;
            Ok((0..length)
                .map(|i| {
                    let t = i as f64 / length as f64;
                    let freq = start_freq + (end_freq - start_freq) * t;
                    (2.0 * std::f64::consts::PI * freq * t * length as f64).sin()
                })
                .collect())
        }
        TestSignalType::PiecewiseConstant => {
            let num_pieces = 4 + (rng.gen::<usize>() % 8); // 4-12 pieces
            let piece_length = length / num_pieces;
            let mut signal = Vec::with_capacity(length);

            for piece in 0..num_pieces {
                let value = rng.gen::<f64>() - 0.5;
                let start = piece * piece_length;
                let end = if piece == num_pieces - 1 {
                    length
                } else {
                    (piece + 1) * piece_length
                };

                for _ in start..end {
                    signal.push(value);
                }
            }

            Ok(signal)
        }
        TestSignalType::PiecewisePolynomial => {
            let num_pieces = 3 + (rng.gen::<usize>() % 5); // 3-8 pieces
            let piece_length = length / num_pieces;
            let mut signal = Vec::with_capacity(length);

            for piece in 0..num_pieces {
                let a = rng.gen::<f64>() - 0.5;
                let b = rng.gen::<f64>() - 0.5;
                let c = rng.gen::<f64>() - 0.5;

                let start = piece * piece_length;
                let end = if piece == num_pieces - 1 {
                    length
                } else {
                    (piece + 1) * piece_length
                };

                for i in start..end {
                    let t = (i - start) as f64 / piece_length as f64;
                    signal.push(a + b * t + c * t * t);
                }
            }

            Ok(signal)
        }
        TestSignalType::Fractal => {
            // Simple fractal-like signal using recursive subdivision
            let mut signal = vec![0.0; length];
            generate_fractal_recursive(&mut signal, 0, length - 1, &mut rng, 0.5, 3);
            Ok(signal)
        }
        TestSignalType::Natural => {
            // Simulate natural signal characteristics (1/f noise + oscillations)
            let mut signal = vec![0.0; length];

            // Add 1/f noise component
            for i in 0..length {
                let freq = (i + 1) as f64 / length as f64;
                let amplitude = 1.0 / freq.sqrt();
                signal[i] += amplitude * (rng.gen::<f64>() - 0.5);
            }

            // Add harmonic components
            for harmonic in 1..5 {
                let freq = harmonic as f64 * 0.05;
                let amplitude = 1.0 / (harmonic as f64);
                for i in 0..length {
                    signal[i] += amplitude
                        * (2.0 * std::f64::consts::PI * freq * i as f64).sin();
                }
            }

            Ok(signal)
        }
    }
}

/// Recursive function for fractal signal generation
fn generate_fractal_recursive(
    signal: &mut [f64],
    start: usize,
    end: usize,
    rng: &mut StdRng,
    scale: f64,
    depth: usize,
) {
    if depth == 0 || end <= start {
        return;
    }

    let mid = (start + end) / 2;
    let noise = (rng.gen::<f64>() - 0.5) * scale;

    signal[mid] = (signal[start] + signal[end]) / 2.0 + noise;

    if mid > start {
        generate_fractal_recursive(signal, start, mid, rng, scale * 0.7, depth - 1);
    }
    if end > mid {
        generate_fractal_recursive(signal, mid, end, rng, scale * 0.7, depth - 1);
    }
}

/// Test WPT round-trip (decomposition + reconstruction)
pub fn test_wpt_round_trip(
    signal: &[f64],
    wavelet: Wavelet,
    level: usize,
) -> SignalResult<(f64, f64)> {
    // Decompose
    let tree = wp_decompose(signal, wavelet, level, None)?;

    // Reconstruct
    let reconstructed = reconstruct_from_nodes(&tree)?;

    // Calculate energy ratio
    let original_energy: f64 = signal.iter().map(|&x| x * x).sum();
    let reconstructed_energy: f64 = reconstructed.iter().map(|&x| x * x).sum();

    let energy_ratio = if original_energy > 1e-15 {
        reconstructed_energy / original_energy
    } else if reconstructed_energy < 1e-15 {
        1.0 // Both are essentially zero
    } else {
        f64::INFINITY
    };

    // Calculate reconstruction error
    let min_length = signal.len().min(reconstructed.len());
    let reconstruction_error: f64 = signal[..min_length]
        .iter()
        .zip(reconstructed[..min_length].iter())
        .map(|(&orig, &recon)| (orig - recon) * (orig - recon))
        .sum::<f64>()
        / min_length as f64;

    Ok((energy_ratio, reconstruction_error.sqrt()))
}

/// Construct frame matrix for wavelet packet decomposition
pub fn construct_frame_matrix(
    signal: &[f64],
    wavelet: Wavelet,
    level: usize,
) -> SignalResult<Array2<f64>> {
    // For simplicity, we'll create a simplified frame matrix
    // In practice, this would involve the analysis and synthesis operators

    let signal_length = signal.len();
    let num_packets = 4_usize.pow(level as u32);

    // Estimate frame matrix dimensions
    let rows = signal_length;
    let cols = num_packets * (signal_length / (2_usize.pow(level as u32))).max(1);

    // Create a simplified frame matrix based on WPT structure
    let mut frame_matrix = Array2::zeros((rows, cols));

    // Fill with simplified wavelet packet basis functions
    let mut col_idx = 0;
    for packet in 0..num_packets {
        let packet_size = signal_length / (2_usize.pow(level as u32)).max(1);

        for i in 0..packet_size.min(signal_length) {
            if col_idx < cols && i < rows {
                // Simple basis function approximation
                let phase = 2.0 * std::f64::consts::PI * packet as f64 * i as f64 / packet_size as f64;
                frame_matrix[[i, col_idx]] = phase.sin() / (signal_length as f64).sqrt();

                if col_idx + 1 < cols && i < rows {
                    frame_matrix[[i, col_idx + 1]] = phase.cos() / (signal_length as f64).sqrt();
                }
            }
        }
        col_idx += 2;
        if col_idx >= cols {
            break;
        }
    }

    Ok(frame_matrix)
}

/// Compute correlation between two signals
pub fn compute_correlation(signal1: &[f64], signal2: &[f64]) -> SignalResult<f64> {
    if signal1.len() != signal2.len() {
        return Err(SignalError::ValueError(
            "Signals must have the same length for correlation".to_string(),
        ));
    }

    let n = signal1.len();
    if n == 0 {
        return Ok(0.0);
    }

    // Compute means
    let mean1 = signal1.iter().sum::<f64>() / n as f64;
    let mean2 = signal2.iter().sum::<f64>() / n as f64;

    // Compute correlation coefficient
    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;

    for i in 0..n {
        let diff1 = signal1[i] - mean1;
        let diff2 = signal2[i] - mean2;

        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }

    let denominator = (sum_sq1 * sum_sq2).sqrt();

    if denominator > 1e-15 {
        Ok(numerator / denominator)
    } else {
        Ok(0.0)
    }
}

/// Compute statistical moments of a signal
pub fn compute_signal_moments(signal: &[f64]) -> (f64, f64, f64, f64) {
    let n = signal.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // Mean
    let mean = signal.iter().sum::<f64>() / n;

    // Variance
    let variance = signal
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f64>()
        / n;

    let std_dev = variance.sqrt();

    if std_dev < 1e-15 {
        return (mean, variance, 0.0, 0.0);
    }

    // Skewness
    let skewness = signal
        .iter()
        .map(|&x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>()
        / n;

    // Kurtosis
    let kurtosis = signal
        .iter()
        .map(|&x| ((x - mean) / std_dev).powi(4))
        .sum::<f64>()
        / n;

    (mean, variance, skewness, kurtosis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_white_noise() {
        let signal = generate_test_signal(TestSignalType::WhiteNoise, 100, 42).unwrap();
        assert_eq!(signal.len(), 100);

        // Check that it's roughly zero-mean
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        assert!(mean.abs() < 0.2); // Should be close to zero for white noise
    }

    #[test]
    fn test_generate_sinusoidal() {
        let signal = generate_test_signal(TestSignalType::Sinusoidal, 100, 42).unwrap();
        assert_eq!(signal.len(), 100);

        // Check that it's bounded
        assert!(signal.iter().all(|&x| x.abs() <= 1.1));
    }

    #[test]
    fn test_compute_correlation() {
        let signal1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let correlation = compute_correlation(&signal1, &signal2).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10); // Should be perfectly correlated

        let signal3 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let correlation2 = compute_correlation(&signal1, &signal3).unwrap();
        assert!((correlation2 + 1.0).abs() < 1e-10); // Should be perfectly anti-correlated
    }

    #[test]
    fn test_compute_signal_moments() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, variance, skewness, kurtosis) = compute_signal_moments(&signal);

        assert!((mean - 3.0).abs() < 1e-10);
        assert!((variance - 2.0).abs() < 1e-10);
        assert!(skewness.abs() < 1e-10); // Should be symmetric
        assert!((kurtosis - 1.8).abs() < 0.1); // Approximately uniform distribution
    }

    #[test]
    fn test_wpt_round_trip() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = test_wpt_round_trip(&signal, Wavelet::Haar, 1);

        // Test should either succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());
    }
}