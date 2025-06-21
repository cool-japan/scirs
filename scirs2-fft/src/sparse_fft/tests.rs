//! Tests for Sparse FFT implementations
//!
//! This module contains comprehensive tests for all sparse FFT algorithms and utilities.

use super::*;
use std::f64::consts::PI;

// Helper function to create a sparse signal
fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
    let mut signal = vec![0.0; n];

    for i in 0..n {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        for &(freq, amp) in frequencies {
            signal[i] += amp * (freq as f64 * t).sin();
        }
    }

    signal
}

#[test]
fn test_sparse_fft_basic() {
    // Create a signal with 3 frequency components
    let n = 256;
    let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);

    // Compute sparse FFT
    let result = sparse_fft(&signal, 6, None, None).unwrap();

    // Should find 6 components (positive and negative frequencies for each)
    assert_eq!(result.values.len(), 6);
}

#[test]
fn test_sparsity_estimation() {
    // Create a signal with 3 frequency components
    let n = 256;
    let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);

    // Create sparse FFT processor with threshold estimation
    let config = config::SparseFFTConfig {
        estimation_method: config::SparsityEstimationMethod::Threshold,
        threshold: 0.1,
        ..config::SparseFFTConfig::default()
    };

    let mut processor = algorithms::SparseFFT::new(config);

    // Estimate sparsity
    let estimated_k = processor.estimate_sparsity(&signal).unwrap();

    // Should estimate approximately 6 components (positive and negative frequencies)
    assert!(estimated_k >= 4 && estimated_k <= 8);
}

#[test]
fn test_frequency_pruning() {
    // Create a signal with 3 frequency components
    let n = 256;
    let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);

    // Create sparse FFT processor with frequency pruning
    let config = config::SparseFFTConfig {
        estimation_method: config::SparsityEstimationMethod::FrequencyPruning,
        algorithm: config::SparseFFTAlgorithm::FrequencyPruning,
        pruning_sensitivity: 2.0,
        ..config::SparseFFTConfig::default()
    };

    let mut processor = algorithms::SparseFFT::new(config);

    // Perform frequency-pruning sparse FFT
    let result = processor.sparse_fft(&signal).unwrap();

    // Should find the frequency components
    assert!(!result.values.is_empty());

    // Test standalone function too
    let result2 = frequency_pruning_sparse_fft(&signal, 2.0).unwrap();
    assert!(!result2.values.is_empty());
}

#[test]
fn test_spectral_flatness() {
    // Create a signal with 3 frequency components
    let n = 256;
    let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);

    // Add some noise
    let mut noisy_signal = signal.clone();
    for i in 0..n {
        noisy_signal[i] += 0.1 * (i as f64 / n as f64 - 0.5);
    }

    // Create sparse FFT processor with spectral flatness algorithm
    let config = config::SparseFFTConfig {
        estimation_method: config::SparsityEstimationMethod::SpectralFlatness,
        algorithm: config::SparseFFTAlgorithm::SpectralFlatness,
        flatness_threshold: 0.5,
        window_size: 16,
        ..config::SparseFFTConfig::default()
    };

    let mut processor = algorithms::SparseFFT::new(config);

    // Perform spectral flatness sparse FFT
    let result = processor.sparse_fft(&noisy_signal).unwrap();

    // Should find some frequency components
    assert!(!result.values.is_empty());

    // Test standalone function too
    let result2 = spectral_flatness_sparse_fft(&noisy_signal, 0.5, 16).unwrap();
    assert!(!result2.values.is_empty());
}

#[test]
fn test_windowing_functions() {
    let signal = vec![1.0, 2.0, 3.0, 4.0];

    // Test Hann window
    let result = windowing::apply_window(&signal, config::WindowFunction::Hann, 14.0).unwrap();
    assert_eq!(result.len(), 4);
    // First and last samples should be close to zero for Hann window
    assert!(result[0].re.abs() < 1e-10);
    assert!(result[3].re.abs() < 1e-10);

    // Test Hamming window
    let result = windowing::apply_window(&signal, config::WindowFunction::Hamming, 14.0).unwrap();
    assert_eq!(result.len(), 4);
    // Hamming window should not be zero at endpoints
    assert!(result[0].re > 0.0);
    assert!(result[3].re > 0.0);

    // Test no windowing
    let result = windowing::apply_window(&signal, config::WindowFunction::None, 14.0).unwrap();
    assert_eq!(result.len(), 4);
    assert_eq!(result[0].re, 1.0);
    assert_eq!(result[1].re, 2.0);
}

#[test]
fn test_reconstruction() {
    let n = 64;
    let frequencies = vec![(3, 1.0), (7, 0.5)];
    let signal = create_sparse_signal(n, &frequencies);

    // Compute sparse FFT
    let sparse_result = sparse_fft(&signal, 4, None, None).unwrap();

    // Test spectrum reconstruction
    let spectrum = reconstruction::reconstruct_spectrum(&sparse_result, n).unwrap();
    assert_eq!(spectrum.len(), n);

    // Test time domain reconstruction
    let reconstructed = reconstruction::reconstruct_time_domain(&sparse_result, n).unwrap();
    assert_eq!(reconstructed.len(), n);

    // Test high resolution reconstruction
    let high_res = reconstruction::reconstruct_high_resolution(&sparse_result, n, 2 * n).unwrap();
    assert_eq!(high_res.len(), 2 * n);
}

#[test]
fn test_adaptive_sparse_fft() {
    let n = 128;
    let frequencies = vec![(5, 1.0), (10, 0.5), (20, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);

    // Test adaptive sparse FFT
    let result = adaptive_sparse_fft(&signal, 0.1).unwrap();

    // Should find some components
    assert!(!result.values.is_empty());
    assert!(result.estimated_sparsity > 0);
}

#[test]
fn test_algorithm_variants() {
    let n = 64;
    let frequencies = vec![(3, 1.0), (7, 0.5)];
    let signal = create_sparse_signal(n, &frequencies);

    // Test different algorithms
    let algorithms = vec![
        config::SparseFFTAlgorithm::Sublinear,
        config::SparseFFTAlgorithm::CompressedSensing,
        config::SparseFFTAlgorithm::Iterative,
        config::SparseFFTAlgorithm::Deterministic,
    ];

    for algorithm in algorithms {
        let result = sparse_fft(&signal, 4, Some(algorithm), None).unwrap();
        assert_eq!(result.algorithm, algorithm);
        assert!(!result.values.is_empty());
        assert!(!result.indices.is_empty());
        assert_eq!(result.values.len(), result.indices.len());
    }
}

#[test]
fn test_performance_measurement() {
    let n = 64;
    let frequencies = vec![(3, 1.0), (7, 0.5)];
    let signal = create_sparse_signal(n, &frequencies);

    let result = sparse_fft(&signal, 4, None, None).unwrap();

    // Check that computation time was measured
    assert!(result.computation_time.as_nanos() > 0);
}

#[test]
fn test_edge_cases() {
    // Test empty signal
    let empty_signal: Vec<f64> = vec![];
    let result = sparse_fft(&empty_signal, 1, None, None);
    assert!(result.is_err());

    // Test single sample
    let single_sample = vec![1.0];
    let result = sparse_fft(&single_sample, 1, None, None).unwrap();
    assert_eq!(result.values.len(), 1);

    // Test k larger than signal length
    let small_signal = vec![1.0, 2.0];
    let result = sparse_fft(&small_signal, 10, None, None).unwrap();
    assert!(result.values.len() <= small_signal.len());
}

#[test]
fn test_sparsity_estimation_methods() {
    let n = 64;
    let frequencies = vec![(3, 1.0), (7, 0.5)];
    let signal = create_sparse_signal(n, &frequencies);

    // Test threshold method
    let estimated = estimation::estimate_sparsity_threshold(&signal, 0.1).unwrap();
    assert!(estimated > 0);

    // Test adaptive method
    let estimated = estimation::estimate_sparsity_adaptive(&signal, 0.25, 10).unwrap();
    assert!(estimated > 0);

    // Test frequency pruning method
    let estimated = estimation::estimate_sparsity_frequency_pruning(&signal, 2.0).unwrap();
    assert!(estimated > 0);

    // Test spectral flatness method
    let estimated = estimation::estimate_sparsity_spectral_flatness(&signal, 0.3, 8).unwrap();
    assert!(estimated > 0);
}

#[test]
fn test_configuration() {
    // Test default configuration
    let config = config::SparseFFTConfig::default();
    assert_eq!(config.sparsity, 10);
    assert_eq!(config.threshold, 0.01);
    assert_eq!(config.max_signal_size, 1024);

    // Test custom configuration
    let custom_config = config::SparseFFTConfig {
        sparsity: 5,
        threshold: 0.05,
        max_signal_size: 512,
        ..config::SparseFFTConfig::default()
    };
    assert_eq!(custom_config.sparsity, 5);
    assert_eq!(custom_config.threshold, 0.05);
    assert_eq!(custom_config.max_signal_size, 512);
}

#[test]
fn test_signal_size_limit() {
    // Create a large signal
    let n = 2048;
    let frequencies = vec![(10, 1.0)];
    let signal = create_sparse_signal(n, &frequencies);

    // Configure with small max signal size
    let config = config::SparseFFTConfig {
        max_signal_size: 64,
        ..config::SparseFFTConfig::default()
    };

    let mut processor = algorithms::SparseFFT::new(config);
    let result = processor.sparse_fft(&signal).unwrap();

    // Should still work, but process only the limited size
    assert!(!result.values.is_empty());
}

#[test]
fn test_complex_input_conversion() {
    // Test try_as_complex function
    let complex_val = num_complex::Complex64::new(1.0, 2.0);
    assert_eq!(config::try_as_complex(complex_val), Some(complex_val));

    let complex32_val = num_complex::Complex32::new(1.0f32, 2.0f32);
    assert_eq!(
        config::try_as_complex(complex32_val),
        Some(num_complex::Complex64::new(1.0, 2.0))
    );
}

#[test]
fn test_filtered_reconstruction() {
    let n = 64;
    let frequencies = vec![(3, 1.0), (7, 0.5), (15, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);

    let sparse_result = sparse_fft(&signal, 6, None, None).unwrap();

    // Test low-pass filter
    let low_pass = |freq_idx: usize, n: usize| -> f64 {
        if freq_idx <= n / 8 || freq_idx >= 7 * n / 8 {
            1.0
        } else {
            0.0
        }
    };

    let filtered = reconstruction::reconstruct_filtered(&sparse_result, n, low_pass).unwrap();
    assert_eq!(filtered.len(), n);
}

#[test]
fn test_multidimensional_placeholders() {
    // Test 2D sparse FFT placeholder
    let signal_2d = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let result = sparse_fft2(&signal_2d, 2, None);
    assert!(result.is_err()); // Should return NotImplemented error

    // Test N-D sparse FFT placeholder
    let signal_1d = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let result = sparse_fftn(&signal_1d, &shape, 2, None);
    assert!(result.is_err()); // Should return NotImplemented error
}
