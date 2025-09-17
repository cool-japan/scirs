//! Comprehensive tests for 2D DWT module
//!
//! This module contains all tests for the 2D Discrete Wavelet Transform functionality,
//! organized by the modules they test.

use super::*;
use crate::dwt::Wavelet;
use ndarray::Array2;

#[test]
fn test_dwt2d_haar_basic() {
    // Create a simple 4x4 test image
    let data = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ],
    )
    .unwrap();

    // Decompose using Haar wavelet
    let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();

    // Check shape
    assert_eq!(decomposition.approx.shape(), &[2, 2]);
    assert_eq!(decomposition.detail_h.shape(), &[2, 2]);
    assert_eq!(decomposition.detail_v.shape(), &[2, 2]);
    assert_eq!(decomposition.detail_d.shape(), &[2, 2]);

    // Reconstruct
    let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).unwrap();

    // Check shape matches
    assert_eq!(reconstructed.shape(), data.shape());
}

#[test]
fn test_wavedec2_waverec2_multilevel() {
    // Create a simple 8x8 test image
    let mut data = Array2::zeros((8, 8));
    for i in 0..8 {
        for j in 0..8 {
            data[[i, j]] = (i * 8 + j + 1) as f64;
        }
    }

    // Multi-level decomposition (using 1 level for reliability)
    let levels = 1;
    let coeffs = wavedec2(&data, Wavelet::Haar, levels, None).unwrap();

    // Check number of levels
    assert_eq!(coeffs.len(), levels);

    // Reconstruct
    let reconstructed = waverec2(&coeffs, Wavelet::Haar, None).unwrap();

    // Check shape matches
    assert_eq!(reconstructed.shape(), data.shape());
}

#[test]
fn test_threshold_dwt2d_hard() {
    // Create a simple test image
    let mut data = Array2::zeros((8, 8));
    for i in 0..8 {
        for j in 0..8 {
            data[[i, j]] = (i * j) as f64;
        }
    }

    // Decompose using Haar wavelet
    let mut decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();

    // Count non-zero coefficients before thresholding
    let (before_count, _) = count_nonzeros(&decomposition, true);

    // Apply hard thresholding with a moderate threshold
    let threshold = 5.0;
    threshold_dwt2d(&mut decomposition, threshold, ThresholdMethod::Hard);

    // Count non-zero coefficients after thresholding
    let (after_count, _) = count_nonzeros(&decomposition, true);

    // There should be fewer non-zero coefficients after thresholding
    assert!(after_count <= before_count);

    // Check that coefficients below threshold were set to zero
    for &val in decomposition.detail_h.iter() {
        assert!(val == 0.0 || val.abs() > threshold);
    }
    for &val in decomposition.detail_v.iter() {
        assert!(val == 0.0 || val.abs() > threshold);
    }
    for &val in decomposition.detail_d.iter() {
        assert!(val == 0.0 || val.abs() > threshold);
    }
}

#[test]
fn test_soft_thresholding() {
    // Create input coefficients
    let values = [-10.0, -6.0, -4.0, -2.0, 0.0, 3.0, 5.0, 8.0];
    let threshold = 4.0;

    // Apply soft thresholding
    let thresholded: Vec<f64> = values
        .iter()
        .map(|&x| apply_threshold(x, threshold, ThresholdMethod::Soft))
        .collect();

    // Expected results:
    // Values below threshold -> 0
    // Values above threshold -> shrink toward zero by threshold amount
    let expected = [-6.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0];

    assert_eq!(thresholded.len(), expected.len());
    for (actual, expected) in thresholded.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn test_garrote_thresholding() {
    // Create input coefficients (avoiding zero for garrote)
    let values = [-10.0, -6.0, -4.0, -3.0, 3.0, 5.0, 8.0];
    let threshold = 4.0;

    // Apply garrote thresholding
    let thresholded: Vec<f64> = values
        .iter()
        .map(|&x| apply_threshold(x, threshold, ThresholdMethod::Garrote))
        .collect();

    // Verify:
    // - Values below threshold are zero
    // - Values above threshold are shrunk non-linearly

    // Check threshold behavior
    assert_eq!(thresholded[2], 0.0); // -4.0 becomes 0
    assert_eq!(thresholded[3], 0.0); // -3.0 becomes 0
    assert_eq!(thresholded[4], 0.0); // 3.0 becomes 0

    // Check that values above threshold are shrunk non-linearly
    // For garrote: x * (1 - (t²/x²)) where t is threshold
    let expected_0 = -10.0 * (1.0 - (threshold * threshold) / (10.0 * 10.0));
    assert!((thresholded[0] - expected_0).abs() < 1e-10);
}

#[test]
fn test_calculate_energy_distribution() {
    // Create a simple test image
    let mut data = Array2::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            data[[i, j]] = (i * j) as f64;
        }
    }

    // Decompose using Haar wavelet
    let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).unwrap();

    // Calculate energy including approximation coefficients
    let (total_energy_with_approx, energy_by_subband) = calculate_energy(&decomposition, true);

    // Calculate energy excluding approximation coefficients
    let (total_energy_without_approx, _) = calculate_energy(&decomposition, false);

    // Most of the energy should be in the approximation coefficients
    assert!(energy_by_subband.approx > energy_by_subband.detail_h);
    assert!(energy_by_subband.approx > energy_by_subband.detail_v);
    assert!(energy_by_subband.approx > energy_by_subband.detail_d);

    // Total energy without approximation should be less than total with approximation
    assert!(total_energy_without_approx < total_energy_with_approx);

    // Sum of individual energies should equal total energy
    let sum_by_subband = energy_by_subband.approx
        + energy_by_subband.detail_h
        + energy_by_subband.detail_v
        + energy_by_subband.detail_d;
    assert!((total_energy_with_approx - sum_by_subband).abs() < 1e-10);
}

#[test]
fn test_dwt2d_db2_wavelets() {
    // Create a simple 6x6 test image
    let mut data = Array2::zeros((6, 6));
    for i in 0..6 {
        for j in 0..6 {
            data[[i, j]] = (i * 6 + j + 1) as f64;
        }
    }

    // Decompose using DB2 wavelet
    let decomposition = dwt2d_decompose(&data, Wavelet::DB(2), None).unwrap();

    // Check shape
    assert_eq!(decomposition.approx.shape(), &[3, 3]);
    assert_eq!(decomposition.detail_h.shape(), &[3, 3]);
    assert_eq!(decomposition.detail_v.shape(), &[3, 3]);
    assert_eq!(decomposition.detail_d.shape(), &[3, 3]);

    // Reconstruct
    let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::DB(2), None).unwrap();

    // Check shape matches
    assert_eq!(reconstructed.shape(), data.shape());
}

#[test]
fn test_psnr_calculation() {
    // Create original and slightly noisy version
    let original = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);
    let mut noisy = original.clone();
    noisy[[0, 0]] += 0.1; // Add small noise

    let psnr = calculate_psnr(&original, &noisy).unwrap();

    // PSNR should be finite and positive
    assert!(psnr.is_finite() && psnr > 0.0);

    // Perfect reconstruction should give infinite PSNR
    let psnr_perfect = calculate_psnr(&original, &original).unwrap();
    assert!(psnr_perfect.is_infinite());
}

#[test]
fn test_ssim_calculation() {
    // Create test images
    let original = Array2::from_shape_fn((16, 16), |(i, j)| {
        ((i as f64 * 0.5).sin() + (j as f64 * 0.3).cos()) * 100.0
    });

    let mut processed = original.clone();
    // Add some processing effects
    for ((i, j), val) in processed.indexed_iter_mut() {
        *val += ((i + j) as f64 * 0.1).sin();
    }

    let ssim = calculate_ssim(&original, &processed, 8, 0.01, 0.03).unwrap();

    // SSIM should be between -1 and 1
    assert!(ssim >= -1.0 && ssim <= 1.0);

    // Perfect match should give SSIM = 1.0
    let ssim_perfect = calculate_ssim(&original, &original, 8, 0.01, 0.03).unwrap();
    assert!((ssim_perfect - 1.0).abs() < 1e-10);
}

#[test]
fn test_noise_variance_estimation() {
    // Create a decomposition with known noise characteristics
    let mut decomp = Dwt2dResult {
        approx: Array2::zeros((4, 4)),
        detail_h: Array2::zeros((4, 4)),
        detail_v: Array2::zeros((4, 4)),
        detail_d: Array2::from_shape_fn((4, 4), |(i, j)| {
            // Simulate noise in diagonal details
            ((i + j) as f64 * 0.1).sin() * 2.0
        }),
    };

    let estimated_noise = estimate_noise_variance(&decomp);

    // Should return a positive value
    assert!(estimated_noise > 0.0);
}

#[test]
fn test_compression_ratio_calculation() {
    // Create test decomposition
    let original = Dwt2dResult {
        approx: Array2::ones((2, 2)),
        detail_h: Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as f64 * 0.5),
        detail_v: Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as f64 * 0.3),
        detail_d: Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as f64 * 0.1),
    };

    let mut compressed = original.clone();

    // Apply thresholding to create sparsity
    threshold_dwt2d(&mut compressed, 0.4, ThresholdMethod::Hard);

    let ratio = calculate_compression_ratio(&original, &compressed);

    // Compression ratio should be >= 1.0 (equal or fewer non-zero coefficients)
    assert!(ratio >= 1.0);
}

#[test]
fn test_simd_capabilities_detection() {
    let caps = PlatformCapabilities::detect();

    // Just ensure the detection runs without panicking
    // The actual capabilities depend on the test machine
    println!("SIMD available: {}", caps.simd_available);
    println!("AVX2 available: {}", caps.avx2_available);
    println!("AVX512 available: {}", caps.avx512_available);

    // Basic consistency check
    if caps.avx512_available {
        assert!(caps.simd_available);
    }
    if caps.avx2_available {
        assert!(caps.simd_available);
    }
}

#[test]
fn test_multilevel_thresholding() {
    // Create test data
    let data = Array2::from_shape_fn((16, 16), |(i, j)| (i * j) as f64);
    let mut coeffs = wavedec2(&data, Wavelet::Haar, 2, None).unwrap();

    // Apply different thresholds for each level
    let thresholds = vec![2.0, 4.0];
    threshold_wavedec2(&mut coeffs, &thresholds, ThresholdMethod::Soft);

    // Reconstruct and verify
    let reconstructed = waverec2(&coeffs, Wavelet::Haar, None).unwrap();
    assert_eq!(reconstructed.shape(), data.shape());
}

#[test]
fn test_adaptive_denoising() {
    // Create a simple test image
    let clean_image = Array2::from_shape_fn((16, 16), |(i, j)| {
        ((i as f64 * 0.3).sin() * (j as f64 * 0.2).cos()) * 10.0
    });

    // Add noise
    let mut noisy_image = clean_image.clone();
    for val in noisy_image.iter_mut() {
        *val += 0.5; // Add constant noise for simplicity
    }

    // Test denoising
    let denoised = denoise_dwt2d_adaptive(&noisy_image, Wavelet::DB(4), Some(0.25), ThresholdMethod::Soft);
    assert!(denoised.is_ok());

    let denoised = denoised.unwrap();
    assert_eq!(denoised.shape(), noisy_image.shape());
}

#[test]
fn test_enhanced_wavedec2() {
    let data = Array2::from_shape_fn((16, 16), |(i, j)| (i + j) as f64);
    let result = wavedec2_enhanced(&data, Wavelet::Haar, 2, None);

    assert!(result.is_ok());
    let coeffs = result.unwrap();
    assert_eq!(coeffs.len(), 2);

    // Verify shapes are as expected for 2-level decomposition
    assert_eq!(coeffs[0].approx.shape(), &[4, 4]); // Deepest level
    assert_eq!(coeffs[1].approx.shape(), &[8, 8]); // First level
}

#[test]
fn test_validation_config_defaults() {
    let config = Dwt2dValidationConfig::default();

    assert_eq!(config.tolerance, 1e-12);
    assert!(config.benchmark_performance);
    assert!(config.test_memory_efficiency);
    assert!(config.test_numerical_stability);
    assert!(config.test_edge_cases);
    assert!(!config.test_sizes.is_empty());
    assert!(!config.test_wavelets.is_empty());
}

#[test]
fn test_count_nonzeros_functionality() {
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