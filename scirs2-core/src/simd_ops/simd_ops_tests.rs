use super::*;
use ::ndarray::Array1;

use super::*;
use ::ndarray::arr1;

#[test]
fn test_simd_unified_ops_f32() {
    let a = arr1(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = arr1(&[5.0f32, 6.0, 7.0, 8.0]);

    let sum = f32::simd_add(&a.view(), &b.view());
    assert_eq!(sum, arr1(&[6.0f32, 8.0, 10.0, 12.0]));

    let product = f32::simd_mul(&a.view(), &b.view());
    assert_eq!(product, arr1(&[5.0f32, 12.0, 21.0, 32.0]));

    let dot = f32::simd_dot(&a.view(), &b.view());
    assert_eq!(dot, 70.0);
}

#[test]
fn test_platform_capabilities() {
    let caps = PlatformCapabilities::detect();
    println!("{}", caps.summary());
}

#[test]
fn test_auto_optimizer() {
    let optimizer = AutoOptimizer::new();

    // Small problem - should use scalar
    assert!(!optimizer.should_use_gpu(100));

    // Large problem - depends on GPU availability
    let large_size = 100000;
    if optimizer.capabilities.gpu_available {
        assert!(optimizer.should_use_gpu(large_size));
    }
}

#[test]
fn test_simd_unified_ops_variance_std() {
    let a = arr1(&[2.0f64, 4.0, 6.0, 8.0, 10.0]);
    let variance = f64::simd_variance(&a.view());
    let std = f64::simd_std(&a.view());

    // Mean = 6, sum of sq deviations = 40
    // Sample variance (Bessel's correction, n-1) = 40/4 = 10
    // Standard deviation = sqrt(10)
    assert!((variance - 10.0).abs() < 1e-10);
    assert!((std - 10.0f64.sqrt()).abs() < 1e-10);
}

#[test]
fn test_simd_unified_ops_norms() {
    let a = arr1(&[3.0f64, -4.0]);

    // L1 norm: |3| + |-4| = 7
    let l1 = f64::simd_norm_l1(&a.view());
    assert!((l1 - 7.0).abs() < 1e-10);

    // L2 norm: sqrt(9 + 16) = 5
    let l2 = f64::simd_norm(&a.view());
    assert!((l2 - 5.0).abs() < 1e-10);

    // Linf norm: max(|3|, |-4|) = 4
    let linf = f64::simd_norm_linf(&a.view());
    assert!((linf - 4.0).abs() < 1e-10);
}

#[test]
fn test_simd_unified_ops_cosine_similarity() {
    let a = arr1(&[1.0f64, 0.0, 0.0]);
    let b = arr1(&[0.0f64, 1.0, 0.0]);
    let c = arr1(&[1.0f64, 0.0, 0.0]);

    // Orthogonal vectors have cosine similarity 0
    let sim_ab = f64::simd_cosine_similarity(&a.view(), &b.view());
    assert!(sim_ab.abs() < 1e-10);

    // Identical vectors have cosine similarity 1
    let sim_ac = f64::simd_cosine_similarity(&a.view(), &c.view());
    assert!((sim_ac - 1.0).abs() < 1e-10);
}

#[test]
fn test_simd_unified_ops_distances() {
    let a = arr1(&[0.0f64, 0.0, 0.0]);
    let b = arr1(&[3.0f64, 4.0, 0.0]);

    // Euclidean distance: sqrt(9 + 16) = 5
    let euclidean = f64::simd_distance_euclidean(&a.view(), &b.view());
    assert!((euclidean - 5.0).abs() < 1e-10);

    // Manhattan distance: 3 + 4 = 7
    let manhattan = f64::simd_distance_manhattan(&a.view(), &b.view());
    assert!((manhattan - 7.0).abs() < 1e-10);

    // Chebyshev distance: max(3, 4) = 4
    let chebyshev = f64::simd_distance_chebyshev(&a.view(), &b.view());
    assert!((chebyshev - 4.0).abs() < 1e-10);
}

#[test]
fn test_simd_unified_ops_squared_euclidean_distance() {
    // f64: squared Euclidean distance: 3^2 + 4^2 = 25 (no sqrt, unlike simd_distance_euclidean)
    let a = arr1(&[0.0f64, 0.0, 0.0]);
    let b = arr1(&[3.0f64, 4.0, 0.0]);
    let squared = f64::simd_distance_squared_euclidean(&a.view(), &b.view());
    assert!((squared - 25.0).abs() < 1e-10);
    // sqrt of the squared distance must agree with the direct Euclidean distance
    let euclidean = f64::simd_distance_euclidean(&a.view(), &b.view());
    assert!((squared.sqrt() - euclidean).abs() < 1e-10);

    // Identical vectors -> zero distance
    let same = arr1(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
    assert!(f64::simd_distance_squared_euclidean(&same.view(), &same.view()).abs() < 1e-12);

    // f64, odd (non-SIMD-aligned) length: cross-check against an independent scalar reference
    let a_odd: Array1<f64> = (0..13).map(|i| (i as f64) * 0.37 - 1.1).collect();
    let b_odd: Array1<f64> = (0..13).map(|i| (i as f64) * 0.19 + 1.5).collect();
    let scalar_ref_f64: f64 = a_odd
        .iter()
        .zip(b_odd.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    let simd_f64 = f64::simd_distance_squared_euclidean(&a_odd.view(), &b_odd.view());
    assert!(
        (simd_f64 - scalar_ref_f64).abs() < 1e-12,
        "f64 mismatch: simd={simd_f64}, scalar={scalar_ref_f64}"
    );

    // f32, odd (non-SIMD-aligned) length: cross-check against an independent scalar reference.
    // Note: f32 has ~7 significant decimal digits, so a 1e-12 *absolute* bound is unreachable
    // for these magnitudes — 1e-4 is the tight bound consistent with f32 precision.
    let a_odd_f32: Array1<f32> = (0..17).map(|i| (i as f32) * 0.37 - 1.1).collect();
    let b_odd_f32: Array1<f32> = (0..17).map(|i| (i as f32) * 0.19 + 1.5).collect();
    let scalar_ref_f32: f32 = a_odd_f32
        .iter()
        .zip(b_odd_f32.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    let simd_f32 = f32::simd_distance_squared_euclidean(&a_odd_f32.view(), &b_odd_f32.view());
    assert!(
        (simd_f32 - scalar_ref_f32).abs() < 1e-4,
        "f32 mismatch: simd={simd_f32}, scalar={scalar_ref_f32}"
    );
}

#[test]
fn test_simd_unified_ops_weighted() {
    let values = arr1(&[1.0f64, 2.0, 3.0, 4.0]);
    let weights = arr1(&[0.1f64, 0.2, 0.3, 0.4]);

    // Weighted sum: 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 = 0.1 + 0.4 + 0.9 + 1.6 = 3.0
    let weighted_sum = f64::simd_weighted_sum(&values.view(), &weights.view());
    assert!((weighted_sum - 3.0).abs() < 1e-10);

    // Weighted mean: 3.0 / 1.0 = 3.0 (weights sum to 1.0)
    let weighted_mean = f64::simd_weighted_mean(&values.view(), &weights.view());
    assert!((weighted_mean - 3.0).abs() < 1e-10);
}

#[test]
fn test_simd_unified_ops_argmin() {
    let a = arr1(&[3.0f64, 1.0, 4.0, 1.5, 2.0]);
    let result = f64::simd_argmin(&a.view());
    assert_eq!(result, Some(1)); // Index of 1.0

    // Test f32
    let a_f32 = arr1(&[5.0f32, 2.0, 8.0, 1.0, 3.0]);
    let result_f32 = f32::simd_argmin(&a_f32.view());
    assert_eq!(result_f32, Some(3)); // Index of 1.0

    // Test empty array
    let empty: Array1<f64> = arr1(&[]);
    assert_eq!(f64::simd_argmin(&empty.view()), None);
}

#[test]
fn test_simd_unified_ops_argmax() {
    let a = arr1(&[3.0f64, 1.0, 4.0, 1.5, 2.0]);
    let result = f64::simd_argmax(&a.view());
    assert_eq!(result, Some(2)); // Index of 4.0

    // Test f32
    let a_f32 = arr1(&[5.0f32, 2.0, 8.0, 1.0, 3.0]);
    let result_f32 = f32::simd_argmax(&a_f32.view());
    assert_eq!(result_f32, Some(2)); // Index of 8.0

    // Test empty array
    let empty: Array1<f64> = arr1(&[]);
    assert_eq!(f64::simd_argmax(&empty.view()), None);
}

#[test]
fn test_simd_unified_ops_clip() {
    let a = arr1(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
    let result = f64::simd_clip(&a.view(), 2.0, 4.0);
    let expected = arr1(&[2.0f64, 2.0, 3.0, 4.0, 4.0]);
    assert_eq!(result, expected);

    // Test f32
    let a_f32 = arr1(&[-1.0f32, 0.0, 0.5, 1.0, 1.5]);
    let result_f32 = f32::simd_clip(&a_f32.view(), 0.0, 1.0);
    let expected_f32 = arr1(&[0.0f32, 0.0, 0.5, 1.0, 1.0]);
    assert_eq!(result_f32, expected_f32);
}

#[test]
fn test_simd_unified_ops_log_sum_exp() {
    let a = arr1(&[1.0f64, 2.0, 3.0]);
    let result = f64::simd_log_sum_exp(&a.view());
    let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
    assert!((result - expected).abs() < 1e-10);

    // Test f32
    let a_f32 = arr1(&[1.0f32, 2.0, 3.0]);
    let result_f32 = f32::simd_log_sum_exp(&a_f32.view());
    let expected_f32 = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    assert!((result_f32 - expected_f32).abs() < 1e-5);

    // Test numerical stability with large values
    let large = arr1(&[1000.0f64, 1000.1, 1000.2]);
    let result_large = f64::simd_log_sum_exp(&large.view());
    assert!(result_large.is_finite());

    // Test empty array
    let empty: Array1<f64> = arr1(&[]);
    let result_empty = f64::simd_log_sum_exp(&empty.view());
    assert!(result_empty.is_infinite() && result_empty < 0.0);
}

#[test]
fn test_simd_unified_ops_softmax_via_log_sum_exp() {
    // Softmax is computed as exp(x - log_sum_exp(x))
    let logits = arr1(&[2.0f64, 1.0, 0.1]);
    let lse = f64::simd_log_sum_exp(&logits.view());

    // Compute softmax probabilities
    let probs: Vec<f64> = logits.iter().map(|&x| (x - lse).exp()).collect();

    // Probabilities should sum to 1
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);

    // All probabilities should be positive
    for p in &probs {
        assert!(*p > 0.0 && *p <= 1.0);
    }
}

#[test]
fn test_simd_unified_ops_softmax() {
    let a = arr1(&[1.0f64, 2.0, 3.0]);
    let result = f64::simd_softmax(&a.view());

    // Check that probabilities sum to 1
    let sum: f64 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);

    // Test f32
    let a_f32 = arr1(&[1.0f32, 2.0, 3.0]);
    let result_f32 = f32::simd_softmax(&a_f32.view());
    let sum_f32: f32 = result_f32.iter().sum();
    assert!((sum_f32 - 1.0).abs() < 1e-5);
}

#[test]
fn test_simd_unified_ops_cumsum() {
    let a = arr1(&[1.0f64, 2.0, 3.0, 4.0]);
    let result = f64::simd_cumsum(&a.view());
    let expected = arr1(&[1.0f64, 3.0, 6.0, 10.0]);
    assert_eq!(result, expected);

    // Test f32
    let a_f32 = arr1(&[1.0f32, 2.0, 3.0, 4.0]);
    let result_f32 = f32::simd_cumsum(&a_f32.view());
    let expected_f32 = arr1(&[1.0f32, 3.0, 6.0, 10.0]);
    assert_eq!(result_f32, expected_f32);

    // Test empty
    let empty: Array1<f64> = arr1(&[]);
    let result_empty = f64::simd_cumsum(&empty.view());
    assert!(result_empty.is_empty());
}

#[test]
fn test_simd_unified_ops_cumprod() {
    let a = arr1(&[1.0f64, 2.0, 3.0, 4.0]);
    let result = f64::simd_cumprod(&a.view());
    let expected = arr1(&[1.0f64, 2.0, 6.0, 24.0]);
    assert_eq!(result, expected);

    // Test f32
    let a_f32 = arr1(&[1.0f32, 2.0, 3.0, 4.0]);
    let result_f32 = f32::simd_cumprod(&a_f32.view());
    let expected_f32 = arr1(&[1.0f32, 2.0, 6.0, 24.0]);
    assert_eq!(result_f32, expected_f32);

    // Test with zero
    let with_zero = arr1(&[1.0f64, 2.0, 0.0, 4.0]);
    let result_zero = f64::simd_cumprod(&with_zero.view());
    let expected_zero = arr1(&[1.0f64, 2.0, 0.0, 0.0]);
    assert_eq!(result_zero, expected_zero);
}

#[test]
fn test_simd_unified_ops_diff() {
    let a = arr1(&[1.0f64, 3.0, 6.0, 10.0, 15.0]);
    let result = f64::simd_diff(&a.view());
    let expected = arr1(&[2.0f64, 3.0, 4.0, 5.0]);
    assert_eq!(result, expected);

    // Test f32
    let a_f32 = arr1(&[1.0f32, 3.0, 6.0, 10.0]);
    let result_f32 = f32::simd_diff(&a_f32.view());
    let expected_f32 = arr1(&[2.0f32, 3.0, 4.0]);
    assert_eq!(result_f32, expected_f32);

    // Test single element
    let single = arr1(&[5.0f64]);
    let result_single = f64::simd_diff(&single.view());
    assert!(result_single.is_empty());
}

#[test]
fn test_simd_unified_ops_abs() {
    let a = arr1(&[-1.0f64, 2.0, -3.0, 4.0, -5.0]);
    let result = f64::simd_abs(&a.view());
    let expected = arr1(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(result, expected);

    // Test f32
    let a_f32 = arr1(&[-2.0f32, -1.0, 3.0]);
    let result_f32 = f32::simd_abs(&a_f32.view());
    let expected_f32 = arr1(&[2.0f32, 1.0, 3.0]);
    assert_eq!(result_f32, expected_f32);
}

#[test]
fn test_simd_unified_ops_sign() {
    let a = arr1(&[-5.0f64, 0.0, 3.0, -1.0, 0.0, 2.0]);
    let result = f64::simd_sign(&a.view());
    let expected = arr1(&[-1.0f64, 0.0, 1.0, -1.0, 0.0, 1.0]);
    assert_eq!(result, expected);

    // Test f32
    let a_f32 = arr1(&[1.0f32, -1.0, 0.0]);
    let result_f32 = f32::simd_sign(&a_f32.view());
    let expected_f32 = arr1(&[1.0f32, -1.0, 0.0]);
    assert_eq!(result_f32, expected_f32);
}

#[test]
fn test_simd_unified_ops_relu() {
    let a = arr1(&[-2.0f64, -1.0, 0.0, 1.0, 2.0]);
    let result = f64::simd_relu(&a.view());
    let expected = arr1(&[0.0f64, 0.0, 0.0, 1.0, 2.0]);
    assert_eq!(result, expected);

    // Test f32
    let a_f32 = arr1(&[-1.0f32, 0.0, 5.0]);
    let result_f32 = f32::simd_relu(&a_f32.view());
    let expected_f32 = arr1(&[0.0f32, 0.0, 5.0]);
    assert_eq!(result_f32, expected_f32);
}

#[test]
fn test_simd_unified_ops_leaky_relu() {
    let a = arr1(&[-2.0f64, -1.0, 0.0, 1.0, 2.0]);
    let result = f64::simd_leaky_relu(&a.view(), 0.1);
    let expected = arr1(&[-0.2f64, -0.1, 0.0, 1.0, 2.0]);
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-10);
    }

    // Test f32 with different alpha
    let a_f32 = arr1(&[-10.0f32, 0.0, 10.0]);
    let result_f32 = f32::simd_leaky_relu(&a_f32.view(), 0.01);
    let expected_f32 = arr1(&[-0.1f32, 0.0, 10.0]);
    for (r, e) in result_f32.iter().zip(expected_f32.iter()) {
        assert!((r - e).abs() < 1e-6);
    }
}

#[test]
fn test_simd_unified_ops_normalize() {
    let a = arr1(&[3.0f64, 4.0]);
    let result = f64::simd_normalize(&a.view());
    // norm = 5, so [0.6, 0.8]
    assert!((result[0] - 0.6).abs() < 1e-10);
    assert!((result[1] - 0.8).abs() < 1e-10);

    // Test unit norm
    let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10);

    // Test f32
    let a_f32 = arr1(&[3.0f32, 4.0]);
    let result_f32 = f32::simd_normalize(&a_f32.view());
    assert!((result_f32[0] - 0.6).abs() < 1e-6);
    assert!((result_f32[1] - 0.8).abs() < 1e-6);
}

#[test]
fn test_simd_unified_ops_standardize() {
    let a = arr1(&[2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
    let result = f64::simd_standardize(&a.view());

    // Check mean is ~0
    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    assert!(mean.abs() < 1e-10, "Mean should be ~0, got {}", mean);

    // Check std is ~1
    let variance: f64 = result.iter().map(|x| x * x).sum::<f64>() / (result.len() - 1) as f64;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 1e-10, "Std should be ~1, got {}", std);

    // Test f32
    let a_f32 = arr1(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let result_f32 = f32::simd_standardize(&a_f32.view());
    let mean_f32: f32 = result_f32.iter().sum::<f32>() / result_f32.len() as f32;
    assert!(mean_f32.abs() < 1e-5);
}
