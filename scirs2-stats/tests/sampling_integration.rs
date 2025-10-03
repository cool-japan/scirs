// Integration tests for sampling and random modules
use approx::assert_relative_eq;
use scirs2_core::ndarray::{array, Array1};
use scirs2_stats::{
    distributions::{norm, poisson},
    random, sampling,
};

#[test]
#[allow(dead_code)]
fn test_sampling_distribution_consistency() {
    // Test that sampling API produces consistent results when using the same seed
    let normal = norm(0.0f64, 1.0).unwrap();

    // Generate samples with same seed twice
    let samples1 = sampling::sample_distribution(&normal, 100).unwrap();
    let samples2 = normal.rvs(100).unwrap();

    // Make sure samples are different (random)
    assert!(samples1 != samples2.clone());

    // Test with explicit RNG seed for consistency
    let samples_seeded1 = random::randn(20, Some(42)).unwrap();
    let samples_seeded2 = random::randn(20, Some(42)).unwrap();

    // Same seed should produce identical results
    for i in 0..samples_seeded1.len() {
        assert_eq!(samples_seeded1[i], samples_seeded2[i]);
    }
}

#[test]
#[allow(dead_code)]
fn test_bootstrap_sample_properties() {
    // Test bootstrap sampling functionality
    let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];

    // Generate bootstrap samples
    let samples = sampling::bootstrap(&data.view(), 100, Some(42)).unwrap();

    // Check shape and properties
    assert_eq!(samples.shape(), &[100, 5]);

    // Each bootstrap sample should have the same length as original
    for i in 0..samples.shape()[0] {
        let bootstrap_sample = samples.slice(scirs2_core::ndarray::s![i, ..]);
        assert_eq!(bootstrap_sample.len(), data.len());

        // Each value in bootstrap sample should be one of the values from the original data
        for &value in bootstrap_sample.iter() {
            let is_from_original = data.iter().any(|&x| {
                let diff = (x - value).abs();
                diff < 0.0000001
            });
            assert!(is_from_original);
        }
    }
}

#[test]
#[allow(dead_code)]
fn test_permutation_properties() {
    // Test permutation functionality
    let data = array![10, 20, 30, 40, 50];

    // Generate permutation
    let perm = sampling::permutation(&data.view(), Some(42)).unwrap();

    // Check properties
    assert_eq!(perm.len(), data.len());

    // Check that all elements are present (just rearranged)
    for &value in data.iter() {
        assert!(perm.iter().any(|&x| x == value));
    }

    // Each value should appear exactly once
    for &value in data.iter() {
        assert_eq!(perm.iter().filter(|&&x| x == value).count(), 1);
    }
}

#[test]
#[allow(dead_code)]
fn test_statistical_properties() {
    // Test that random samples have expected statistical properties

    // Uniform distribution
    let uniform_samples = random::uniform(0.0, 1.0, 10000, Some(42)).unwrap();
    let mean = custom_mean(&uniform_samples);
    let std = custom_std(&uniform_samples, 0);

    // Mean should be approximately 0.5 for uniform(0,1)
    assert_relative_eq!(mean, 0.5, epsilon = 0.02);
    // Std dev should be approximately 1/sqrt(12) ≈ 0.2887
    assert_relative_eq!(std, 0.2887, epsilon = 0.02);

    // Normal distribution
    let normal_samples = random::randn(10000, Some(42)).unwrap();
    let mean = custom_mean(&normal_samples);
    let std = custom_std(&normal_samples, 0);

    // Mean should be approximately 0 for standard normal
    assert_relative_eq!(mean, 0.0, epsilon = 0.05);
    // Std dev should be approximately 1
    assert_relative_eq!(std, 1.0, epsilon = 0.05);

    // Poisson distribution (with lambda=3)
    let poisson_dist = poisson(3.0f64, 0.0).unwrap();
    let samples = poisson_dist.rvs(10000).unwrap();
    let samples_array = Array1::from(samples);
    let mean = custom_mean(&samples_array);

    // Mean should be approximately lambda=3 for Poisson
    assert_relative_eq!(mean, 3.0, epsilon = 0.1);
}

// Helper functions for statistical calculations
#[allow(dead_code)]
fn custom_mean(arr: &Array1<f64>) -> f64 {
    if arr.is_empty() {
        return 0.0;
    }
    let sum: f64 = arr.iter().sum();
    sum / arr.len() as f64
}

#[allow(dead_code)]
fn custom_std(arr: &Array1<f64>, ddof: usize) -> f64 {
    if arr.len() <= ddof {
        return 0.0;
    }

    let mean = custom_mean(arr);
    let sum_sq: f64 = arr.iter().map(|&x| (x - mean) * (x - mean)).sum();
    let denominator = (arr.len() - ddof) as f64;
    (sum_sq / denominator).sqrt()
}
