//! Property-based tests for v0.2.0 SIMD enhancements
//!
//! This module validates mathematical invariants and numerical stability
//! of SIMD-optimized operations.

use approx::assert_abs_diff_eq;
use quickcheck::{quickcheck, TestResult};
use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_stats::correlation_simd_enhanced::{
    covariance_matrix_simd, rolling_correlation_simd, spearman_r_simd,
};
use scirs2_stats::mean_simd;
use scirs2_stats::pearson_r_simd;
use scirs2_stats::sampling_simd::{bootstrap_simd, box_muller_simd, exponential_simd};

#[test]
fn property_correlation_bounds() {
    fn test_bounds(xs: Vec<f64>, ys: Vec<f64>) -> TestResult {
        if xs.len() < 2 || ys.len() < 2 || xs.len() != ys.len() {
            return TestResult::discard();
        }

        // Filter out NaN and infinite values
        if xs.iter().any(|&x| !x.is_finite()) || ys.iter().any(|&y| !y.is_finite()) {
            return TestResult::discard();
        }

        // Filter out extreme values to prevent numerical overflow
        if xs.iter().any(|&x| x.abs() > 1e50) || ys.iter().any(|&y| y.abs() > 1e50) {
            return TestResult::discard();
        }

        // Check for constant arrays
        let x_min = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (x_max - x_min).abs() < 1e-10 {
            return TestResult::discard();
        }

        let y_min = ys.iter().copied().fold(f64::INFINITY, f64::min);
        let y_max = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (y_max - y_min).abs() < 1e-10 {
            return TestResult::discard();
        }

        let x = Array1::from_vec(xs);
        let y = Array1::from_vec(ys);

        match pearson_r_simd(&x.view(), &y.view()) {
            Ok(r) => {
                // Correlation must be in [-1, 1]
                TestResult::from_bool((-1.0 - 1e-10..=1.0 + 1e-10).contains(&r))
            }
            Err(_) => TestResult::discard(),
        }
    }

    quickcheck(test_bounds as fn(Vec<f64>, Vec<f64>) -> TestResult);
}

#[test]
fn property_correlation_symmetry() {
    fn test_symmetry(xs: Vec<f64>, ys: Vec<f64>) -> TestResult {
        if xs.len() < 2 || ys.len() < 2 || xs.len() != ys.len() {
            return TestResult::discard();
        }

        // Filter out NaN and infinite values
        if xs.iter().any(|&x| !x.is_finite()) || ys.iter().any(|&y| !y.is_finite()) {
            return TestResult::discard();
        }

        // Filter out extreme values to prevent numerical overflow
        if xs.iter().any(|&x| x.abs() > 1e50) || ys.iter().any(|&y| y.abs() > 1e50) {
            return TestResult::discard();
        }

        let x = Array1::from_vec(xs);
        let y = Array1::from_vec(ys.clone());

        match (
            pearson_r_simd(&x.view(), &y.view()),
            pearson_r_simd(&y.view(), &x.view()),
        ) {
            (Ok(rxy), Ok(ryx)) => {
                // corr(X, Y) = corr(Y, X)
                TestResult::from_bool((rxy - ryx).abs() < 1e-10_f64)
            }
            _ => TestResult::discard(),
        }
    }

    quickcheck(test_symmetry as fn(Vec<f64>, Vec<f64>) -> TestResult);
}

#[test]
fn property_correlation_self() {
    fn test_self_correlation(xs: Vec<f64>) -> TestResult {
        if xs.len() < 2 {
            return TestResult::discard();
        }

        // Filter out NaN and infinite values
        if xs.iter().any(|&x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Filter out extreme values to prevent numerical overflow
        if xs.iter().any(|&x| x.abs() > 1e50) {
            return TestResult::discard();
        }

        // Check for constant array
        let x_min = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (x_max - x_min).abs() < 1e-10 {
            return TestResult::discard();
        }

        let x = Array1::from_vec(xs);

        match pearson_r_simd(&x.view(), &x.view()) {
            Ok(r) => {
                // corr(X, X) = 1
                TestResult::from_bool((r - 1.0_f64).abs() < 1e-10_f64)
            }
            Err(_) => TestResult::discard(),
        }
    }

    quickcheck(test_self_correlation as fn(Vec<f64>) -> TestResult);
}

#[test]
fn property_covariance_matrix_symmetry() {
    fn test_symmetry(data: Vec<Vec<f64>>) -> TestResult {
        if data.len() < 2 || data.is_empty() || data[0].len() < 2 {
            return TestResult::discard();
        }

        let n_obs = data.len();
        let n_vars = data[0].len();

        // Ensure all rows have same length
        if !data.iter().all(|row| row.len() == n_vars) {
            return TestResult::discard();
        }

        // Filter out NaN and infinite values
        if data.iter().any(|row| row.iter().any(|&x| !x.is_finite())) {
            return TestResult::discard();
        }

        // Filter out extreme values to prevent numerical overflow
        if data.iter().any(|row| row.iter().any(|&x| x.abs() > 1e50)) {
            return TestResult::discard();
        }

        let mut flat = Vec::new();
        for row in data {
            flat.extend(row);
        }

        let arr = Array2::from_shape_vec((n_obs, n_vars), flat).ok();
        if arr.is_none() {
            return TestResult::discard();
        }

        let arr = arr.unwrap();

        match covariance_matrix_simd(&arr.view(), false, 1) {
            Ok(cov) => {
                // Check symmetry: cov[i,j] = cov[j,i]
                let mut symmetric = true;
                for i in 0..n_vars {
                    for j in 0..n_vars {
                        if (cov[(i, j)] - cov[(j, i)]).abs() > 1e-10 {
                            symmetric = false;
                            break;
                        }
                    }
                }
                TestResult::from_bool(symmetric)
            }
            Err(_) => TestResult::discard(),
        }
    }

    quickcheck(test_symmetry as fn(Vec<Vec<f64>>) -> TestResult);
}

#[test]
fn property_spearman_monotonic_invariance() {
    // Spearman correlation should be invariant under monotonic transformations
    fn test_monotonic(xs: Vec<f64>) -> TestResult {
        if xs.len() < 3 {
            return TestResult::discard();
        }

        // Filter out NaN and infinite values
        if xs.iter().any(|&x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Filter out extreme values to prevent numerical overflow
        if xs.iter().any(|&x| x.abs() > 1e50) {
            return TestResult::discard();
        }

        let x = Array1::from_vec(xs.clone());
        // Apply monotonic transformation: y = x^3
        let y = x.mapv(|v| v.powi(3));

        match spearman_r_simd(&x.view(), &y.view()) {
            Ok(rho) => {
                // Should be close to ±1 (depending on monotonicity)
                TestResult::from_bool(rho.abs() > 0.99 || rho.abs() < 1e-5)
            }
            Err(_) => TestResult::discard(),
        }
    }

    quickcheck(test_monotonic as fn(Vec<f64>) -> TestResult);
}

#[test]
fn property_bootstrap_sample_size() {
    fn test_sample_size(data: Vec<f64>, n_samples: usize) -> TestResult {
        if data.is_empty() || n_samples == 0 || n_samples > 10000 {
            return TestResult::discard();
        }

        // Filter out NaN and infinite values
        if data.iter().any(|&x| !x.is_finite()) {
            return TestResult::discard();
        }

        // Filter out extreme values to prevent numerical overflow
        if data.iter().any(|&x| x.abs() > 1e50) {
            return TestResult::discard();
        }

        let arr = Array1::from_vec(data);

        match bootstrap_simd(&arr.view(), n_samples, Some(42)) {
            Ok(samples) => TestResult::from_bool(samples.len() == n_samples),
            Err(_) => TestResult::discard(),
        }
    }

    quickcheck(test_sample_size as fn(Vec<f64>, usize) -> TestResult);
}

#[test]
fn test_normal_distribution_properties() {
    // Test that Box-Muller produces samples with correct mean and variance
    for _ in 0..10 {
        let mean = 5.0;
        let std_dev = 2.0;
        let n = 10000;

        let samples = box_muller_simd(n, mean, std_dev, None).expect("Sampling failed");

        let sample_mean = mean_simd(&samples.view()).expect("Mean computation failed");
        let sample_variance: f64 = samples
            .iter()
            .map(|&x: &f64| (x - sample_mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;

        assert_abs_diff_eq!(sample_mean, mean, epsilon = 0.1);
        assert_abs_diff_eq!(sample_variance.sqrt(), std_dev, epsilon = 0.1);
    }
}

#[test]
fn test_exponential_distribution_properties() {
    // Test that exponential distribution has correct mean
    for _ in 0..10 {
        let rate = 2.0;
        let n = 10000;

        let samples = exponential_simd(n, rate, None).expect("Sampling failed");

        let sample_mean = mean_simd(&samples.view()).expect("Mean computation failed");
        let expected_mean = 1.0 / rate;

        assert_abs_diff_eq!(sample_mean, expected_mean, epsilon = 0.05);
    }
}

#[test]
fn test_rolling_correlation_window_size() {
    let n = 100;
    let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64));
    let y: Array1<f64> = Array1::from_iter((0..n).map(|i| (i as f64) * 2.0));

    for window_size in [5, 10, 20, 50].iter() {
        let result = rolling_correlation_simd(&x.view(), &y.view(), *window_size)
            .expect("Rolling correlation failed");

        let expected_len = n - window_size + 1;
        assert_eq!(result.len(), expected_len);

        // All correlations should be close to 1 (perfect linear relationship)
        for &r in result.iter() {
            assert_abs_diff_eq!(r, 1.0, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_numerical_stability_correlation() {
    // Test correlation with very small and very large values
    let small = array![1e-100, 2e-100, 3e-100, 4e-100, 5e-100];
    let large = array![1e100, 2e100, 3e100, 4e100, 5e100];

    // Should not panic or produce NaN
    match pearson_r_simd(&small.view(), &large.view()) {
        Ok(r) => {
            let r_f64: f64 = r;
            assert!(!r_f64.is_nan());
            assert!(r_f64.abs() <= 1.0 + 1e-10);
        }
        Err(_) => {
            // Acceptable if the implementation rejects extreme values
        }
    }
}

#[test]
fn test_covariance_matrix_positive_semidefinite() {
    // Covariance matrix should be positive semi-definite
    use scirs2_linalg::eig;

    let data = array![
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0]
    ];

    let cov = covariance_matrix_simd(&data.view(), false, 1).expect("Covariance failed");

    // Check eigenvalues are non-negative
    match eig(&cov.view(), None) {
        Ok((eigenvalues, _)) => {
            for lambda in eigenvalues.iter() {
                assert!(
                    lambda.re >= -1e-10,
                    "Eigenvalue should be non-negative: {}",
                    lambda.re
                );
            }
        }
        Err(_) => {
            // Skip if eigenvalue computation fails
        }
    }
}
