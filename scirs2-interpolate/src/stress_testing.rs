//! Stress testing module for interpolation methods with extreme inputs.
//!
//! This module provides comprehensive stress testing to ensure numerical stability,
//! proper error handling, and performance under extreme conditions.

use crate::{
    advanced::rbf::{RBFInterpolator, RBFKernel},
    error::{InterpolateError, InterpolateResult},
    linear_interpolate,
    spline::CubicSpline,
};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use std::f64::INFINITY;
use std::f64::{EPSILON as F64_EPSILON, MAX as F64_MAX, MIN_POSITIVE as F64_MIN_POSITIVE};

/// Configuration for stress testing
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Test with very large values (near f64::MAX)
    pub test_large_values: bool,
    /// Test with very small values (near f64::MIN_POSITIVE)  
    pub test_small_values: bool,
    /// Test with values spanning many orders of magnitude
    pub test_large_ranges: bool,
    /// Test with nearly collinear points
    pub test_near_collinear: bool,
    /// Test with duplicate points
    pub test_duplicates: bool,
    /// Test with NaN/Inf values
    pub test_special_values: bool,
    /// Test with extreme extrapolation
    pub test_extreme_extrapolation: bool,
    /// Test with very dense grids
    pub test_dense_grids: bool,
    /// Test with ill-conditioned matrices
    pub test_ill_conditioned: bool,
    /// Maximum dataset size to test
    pub max_dataset_size: usize,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            test_large_values: true,
            test_small_values: true,
            test_large_ranges: true,
            test_near_collinear: true,
            test_duplicates: true,
            test_special_values: true,
            test_extreme_extrapolation: true,
            test_dense_grids: true,
            test_ill_conditioned: true,
            max_dataset_size: 100_000,
        }
    }
}

/// Results from stress testing
#[derive(Debug)]
pub struct StressTestResults {
    /// Name of the method tested
    pub method_name: String,
    /// Configuration used
    pub config: StressTestConfig,
    /// Tests passed
    pub passed_tests: Vec<String>,
    /// Tests failed with reasons
    pub failed_tests: Vec<(String, String)>,
    /// Performance metrics
    pub performance_metrics: Vec<(String, f64)>,
    /// Numerical accuracy metrics
    pub accuracy_metrics: Vec<(String, f64)>,
}

impl StressTestResults {
    fn new(_methodname: String, config: StressTestConfig) -> Self {
        Self {
            method_name,
            config,
            passed_tests: Vec::new(),
            failed_tests: Vec::new(),
            performance_metrics: Vec::new(),
            accuracy_metrics: Vec::new(),
        }
    }

    fn add_pass(&mut self, testname: &str) {
        self.passed_tests.push(test_name.to_string());
    }

    fn add_fail(&mut self, testname: &str, reason: &str) {
        self.failed_tests
            .push((test_name.to_string(), reason.to_string()));
    }

    fn add_performance(&mut self, metricname: &str, value: f64) {
        self.performance_metrics
            .push((metric_name.to_string(), value));
    }

    fn add_accuracy(&mut self, metricname: &str, value: f64) {
        self.accuracy_metrics.push((metric_name.to_string(), value));
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.passed_tests.len() + self.failed_tests.len();
        if total == 0 {
            1.0
        } else {
            self.passed_tests.len() as f64 / total as f64
        }
    }

    /// Print summary report
    pub fn print_summary(&self) {
        println!("\n=== Stress Test Results for {} ===", self.method_name);
        println!("Success Rate: {:.1}%", self.success_rate() * 100.0);
        println!("\nPassed Tests ({}):", self.passed_tests.len());
        for test in &self.passed_tests {
            println!("  ✓ {}", test);
        }

        if !self.failed_tests.is_empty() {
            println!("\nFailed Tests ({}):", self.failed_tests.len());
            for (test, reason) in &self.failed_tests {
                println!("  ✗ {}: {}", test, reason);
            }
        }

        if !self.performance_metrics.is_empty() {
            println!("\nPerformance Metrics:");
            for (metric, value) in &self.performance_metrics {
                println!("  - {}: {:.3e}", metric, value);
            }
        }

        if !self.accuracy_metrics.is_empty() {
            println!("\nAccuracy Metrics:");
            for (metric, value) in &self.accuracy_metrics {
                println!("  - {}: {:.3e}", metric, value);
            }
        }
    }
}

/// Generate extreme test data
pub mod extreme_data {
    use super::*;

    /// Generate data with very large values
    pub fn large_values_1d(n: usize) -> (Array1<f64>, Array1<f64>) {
        let scale = F64_MAX.sqrt();
        let x = Array1::linspace(0.0, 1.0, n) * scale;
        let y = x.mapv(|xi: f64| (xi / scale).sin() * scale / 2.0);
        (x, y)
    }

    /// Generate data with very small values
    pub fn small_values_1d(n: usize) -> (Array1<f64>, Array1<f64>) {
        let scale = F64_MIN_POSITIVE.sqrt();
        let x = Array1::linspace(scale, scale * 100.0, n);
        let y = x.mapv(|xi| xi * 2.0);
        (x, y)
    }

    /// Generate data spanning many orders of magnitude
    pub fn large_range_1d(n: usize) -> (Array1<f64>, Array1<f64>) {
        let mut x = Array1::zeros(n);
        let mut y = Array1::zeros(n);

        for i in 0..n {
            let exp = -10.0 + 20.0 * (i as f64 / (n - 1) as f64);
            x[i] = 10.0_f64.powf(exp);
            y[i] = x[i].ln();
        }

        (x, y)
    }

    /// Generate nearly collinear points
    pub fn near_collinear_1d(n: usize) -> (Array1<f64>, Array1<f64>) {
        let x = Array1::linspace(0.0, 1.0, n);
        let mut y = x.clone() * 2.0 + 1.0;

        // Add tiny perturbations
        for i in 0..n {
            y[i] += F64_EPSILON * 100.0 * (i as f64).sin();
        }

        (x, y)
    }

    /// Generate data with duplicate x values
    pub fn with_duplicates_1d(n: usize) -> (Array1<f64>, Array1<f64>) {
        let mut x = Array1::zeros(n);
        let mut y = Array1::zeros(n);

        let unique_points = n / 3;
        for i in 0..n {
            let idx = i % unique_points;
            x[i] = idx as f64 / (unique_points - 1) as f64;
            y[i] = x[i].sin() + 0.1 * (i as f64 / n as f64);
        }

        (x, y)
    }

    /// Generate 2D data with extreme conditions
    pub fn extreme_2d(n: usize, condition: &str) -> (Array2<f64>, Array1<f64>) {
        let mut points = Array2::zeros((n, 2));
        let mut values = Array1::zeros(n);

        match condition {
            "large_values" => {
                let scale = F64_MAX.sqrt().sqrt();
                for i in 0..n {
                    points[[i, 0]] = (i as f64 / (n - 1) as f64) * scale;
                    points[[i, 1]] = ((i * 7) % n) as f64 / (n - 1) as f64 * scale;
                    values[i] =
                        (points[[i, 0]] / scale).sin() * (points[[i, 1]] / scale).cos() * scale
                            / 4.0;
                }
            }
            "clustered" => {
                // Create clustered points
                let n_clusters = 5;
                let points_per_cluster = n / n_clusters;

                for cluster in 0..n_clusters {
                    let center_x = cluster as f64 / (n_clusters - 1) as f64 * 10.0;
                    let center_y = (cluster as f64 * 1.7).sin() * 5.0 + 5.0;

                    for i in 0..points_per_cluster {
                        let idx = cluster * points_per_cluster + i;
                        if idx < n {
                            points[[idx, 0]] = center_x + 0.01 * scirs2_core::random::random::<f64>();
                            points[[idx, 1]] = center_y + 0.01 * scirs2_core::random::random::<f64>();
                            values[idx] = center_x.sin() * center_y.cos();
                        }
                    }
                }
            }
            "ill_conditioned" => {
                // Create points that lead to ill-conditioned matrices
                for i in 0..n {
                    let t = i as f64 / (n - 1) as f64;
                    points[[i, 0]] = t;
                    points[[i, 1]] = t + F64_EPSILON * 1000.0 * (i as f64).sin();
                    values[i] = t.sin();
                }
            }
            _ => {
                // Default: random points
                for i in 0..n {
                    points[[i, 0]] = scirs2_core::random::random::<f64>() * 10.0;
                    points[[i, 1]] = scirs2_core::random::random::<f64>() * 10.0;
                    values[i] = points[[i, 0]].sin() * points[[i, 1]].cos();
                }
            }
        }

        (points, values)
    }
}

/// Stress test implementations for different methods
pub mod stress_tests {
    use super::*;
    use std::time::Instant;

    /// Stress test linear interpolation
    pub fn test_linear_interpolation(config: &StressTestConfig) -> StressTestResults {
        let mut results =
            StressTestResults::new("Linear Interpolation".to_string(), config.clone());

        // Test with large values
        if config.test_large_values {
            let (x, y) = extreme_data::large_values_1d(1000);
            let queries = Array1::linspace(x[0], x[x.len() - 1], 100);

            match linear_interpolate(&x.view(), &y.view(), &queries.view()) {
                Ok(result) => {
                    if result.iter().all(|&v: &f64| v.is_finite()) {
                        results.add_pass("large_values");
                    } else {
                        results.add_fail("large_values", "Non-finite results");
                    }
                }
                Err(e) => results.add_fail("large_values", &e.to_string()),
            }
        }

        // Test with small values
        if config.test_small_values {
            let (x, y) = extreme_data::small_values_1d(1000);
            let queries = Array1::linspace(x[0], x[x.len() - 1], 100);

            match linear_interpolate(&x.view(), &y.view(), &queries.view()) {
                Ok(result) => {
                    let expected = queries.mapv(|q| q * 2.0);
                    let max_error = result
                        .iter()
                        .zip(expected.iter())
                        .map(|(&r, &e)| ((r - e) / e).abs())
                        .fold(0.0, f64::max);

                    if max_error < 1e-10 {
                        results.add_pass("small_values");
                        results.add_accuracy("small_values_rel_error", max_error);
                    } else {
                        results.add_fail(
                            "small_values",
                            &format!("Large relative error: {}", max_error),
                        );
                    }
                }
                Err(e) => results.add_fail("small_values", &e.to_string()),
            }
        }

        // Test with large ranges
        if config.test_large_ranges {
            let (x, y) = extreme_data::large_range_1d(100);
            let queries = Array1::from_vec(vec![1e-5, 1e-2, 1.0, 1e2, 1e5]);

            match linear_interpolate(&x.view(), &y.view(), &queries.view()) {
                Ok(result) => {
                    if result.iter().all(|&v: &f64| v.is_finite()) {
                        results.add_pass("large_ranges");
                    } else {
                        results.add_fail("large_ranges", "Non-finite results");
                    }
                }
                Err(e) => results.add_fail("large_ranges", &e.to_string()),
            }
        }

        // Test extreme extrapolation
        if config.test_extreme_extrapolation {
            let x = Array1::linspace(0.0, 1.0, 10);
            let y = x.mapv(|xi: f64| xi.sin());
            let queries = Array1::from_vec(vec![-100.0, -10.0, 10.0, 100.0]);

            match linear_interpolate(&x.view(), &y.view(), &queries.view()) {
                Ok(result) => {
                    if result.iter().all(|&v: &f64| v.is_finite()) {
                        results.add_pass("extreme_extrapolation");
                    } else {
                        results.add_fail("extreme_extrapolation", "Non-finite results");
                    }
                }
                Err(e) => results.add_fail("extreme_extrapolation", &e.to_string()),
            }
        }

        // Performance test with large dataset
        if config.max_dataset_size > 0 {
            let n = config.max_dataset_size.min(1_000_000);
            let x = Array1::linspace(0.0, 100.0, n);
            let y = x.mapv(|xi: f64| xi.sin());
            let queries = Array1::linspace(0.0, 100.0, 10000);

            let start = Instant::now();
            match linear_interpolate(&x.view(), &y.view(), &queries.view()) {
                Ok(_) => {
                    let elapsed = start.elapsed().as_secs_f64();
                    results.add_pass(&format!("large_dataset_{}", n));
                    results.add_performance(&format!("time_{}points_10k_queries", n), elapsed);
                }
                Err(e) => results.add_fail(&format!("large_dataset_{}", n), &e.to_string()),
            }
        }

        results
    }

    /// Stress test cubic spline interpolation
    pub fn test_cubic_spline(config: &StressTestConfig) -> StressTestResults {
        let mut results = StressTestResults::new("Cubic Spline".to_string(), config.clone());

        // Test with nearly collinear points
        if config.test_near_collinear {
            let (x, y) = extreme_data::near_collinear_1d(100);

            match CubicSpline::new(&x.view(), &y.view()) {
                Ok(spline) => {
                    let queries = Array1::linspace(0.0, 1.0, 50);
                    let mut all_finite = true;

                    for &q in queries.iter() {
                        let result = spline.evaluate(q);
                        if !result.unwrap_or(f64::NAN).is_finite() {
                            all_finite = false;
                            break;
                        }
                    }

                    if all_finite {
                        results.add_pass("near_collinear");
                    } else {
                        results.add_fail("near_collinear", "Non-finite results");
                    }
                }
                Err(e) => results.add_fail("near_collinear", &e.to_string()),
            }
        }

        // Test with duplicate detection
        if config.test_duplicates {
            let (x, y) = extreme_data::with_duplicates_1d(30);

            match CubicSpline::new(&x.view(), &y.view()) {
                Ok(_) => results.add_fail("duplicates", "Should have detected duplicate x values"),
                Err(_) => results.add_pass("duplicates"),
            }
        }

        // Test boundary conditions with extreme values
        if config.test_large_values {
            let n = 50;
            let x = Array1::linspace(0.0, 1.0, n);
            let scale = 1e100;
            let y = x.mapv(|xi: f64| xi.sin() * scale);

            match CubicSpline::new(&x.view(), &y.view()) {
                Ok(spline) => {
                    let test_points = vec![0.0, 0.5, 1.0];
                    let mut stable = true;

                    for &t in &test_points {
                        let val = spline.evaluate(t);
                        let deriv = spline.derivative(t);

                        if !val.unwrap_or(f64::NAN).is_finite()
                            || deriv.unwrap_or(f64::NAN).abs() > scale * 100.0
                        {
                            stable = false;
                            break;
                        }
                    }

                    if stable {
                        results.add_pass("large_values_stability");
                    } else {
                        results.add_fail("large_values_stability", "Numerical instability");
                    }
                }
                Err(e) => results.add_fail("large_values_stability", &e.to_string()),
            }
        }

        results
    }

    /// Stress test RBF interpolation
    pub fn test_rbf_interpolation(config: &StressTestConfig) -> StressTestResults {
        let mut results = StressTestResults::new("RBF Interpolation".to_string(), config.clone());

        // Test with ill-conditioned matrices
        if config.test_ill_conditioned {
            let (points, values) = extreme_data::extreme_2d(50, "ill_conditioned");

            let kernels = [
                ("gaussian", RBFKernel::Gaussian),
                ("multiquadric", RBFKernel::Multiquadric),
                ("thin_plate", RBFKernel::ThinPlateSpline),
            ];

            for (kernel_name, kernel) in &kernels {
                match RBFInterpolator::new(&points.view(), &values.view(), *kernel, 1.0) {
                    Ok(interpolator) => {
                        let queries = extreme_data::extreme_2d(10, "random").0;
                        match interpolator.interpolate(&queries.view()) {
                            Ok(result) => {
                                if result.iter().all(|&v: &f64| v.is_finite()) {
                                    results.add_pass(&format!("ill_conditioned_{}", kernel_name));
                                } else {
                                    results.add_fail(
                                        &format!("ill_conditioned_{}", kernel_name),
                                        "Non-finite results",
                                    );
                                }
                            }
                            Err(e) => results.add_fail(
                                &format!("ill_conditioned_{}", kernel_name),
                                &e.to_string(),
                            ),
                        }
                    }
                    Err(e) => results
                        .add_fail(&format!("ill_conditioned_{}", kernel_name), &e.to_string()),
                }
            }
        }

        // Test with clustered points
        if config.test_dense_grids {
            let (points, values) = extreme_data::extreme_2d(200, "clustered");

            match RBFInterpolator::new(&points.view(), &values.view(), RBFKernel::Gaussian, 0.1) {
                Ok(interpolator) => {
                    let queries = extreme_data::extreme_2d(50, "random").0;
                    let start = Instant::now();

                    match interpolator.interpolate(&queries.view()) {
                        Ok(_) => {
                            let elapsed = start.elapsed().as_secs_f64();
                            results.add_pass("clustered_points");
                            results.add_performance("clustered_200pts_50queries", elapsed);
                        }
                        Err(e) => results.add_fail("clustered_points", &e.to_string()),
                    }
                }
                Err(e) => results.add_fail("clustered_points", &e.to_string()),
            }
        }

        results
    }

    /// Comprehensive stress test suite
    pub fn run_all_stress_tests(config: &StressTestConfig) -> Vec<StressTestResults> {
        vec![
            test_linear_interpolation(_config),
            test_cubic_spline(_config),
            test_rbf_interpolation(_config),
            // Add more method tests as needed
        ]
    }
}

/// Helper functions for numerical stability testing
pub mod stability {
    use super::*;

    /// Check if a matrix is ill-conditioned
    pub fn condition_number_estimate(points: &ArrayView2<f64>) -> f64 {
        // Simple estimate based on point spacing
        let n = points.nrows();
        if n < 2 {
            return 1.0;
        }

        let mut min_dist = f64::MAX;
        let mut max_dist: f64 = 0.0;

        for i in 0..n {
            for j in i + 1..n {
                let dist: f64 = _points
                    .slice(scirs2_core::ndarray::s![i, ..])
                    .iter..].iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                min_dist = min_dist.min(dist);
                max_dist = max_dist.max(dist);
            }
        }

        if min_dist > 0.0 {
            max_dist / min_dist
        } else {
            INFINITY
        }
    }

    /// Test for numerical overflow/underflow
    pub fn test_numerical_limits(value: f64) -> InterpolateResult<()> {
        if !_value.is_finite() {
            return Err(InterpolateError::invalid_input(
                "Non-finite _value encountered".to_string(),
            ));
        }

        if value.abs() > F64_MAX / 100.0 {
            return Err(InterpolateError::invalid_input(
                "Value approaching overflow".to_string(),
            ));
        }

        if _value != 0.0 && value.abs() < F64_MIN_POSITIVE * 100.0 {
            return Err(InterpolateError::invalid_input(
                "Value approaching underflow".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_framework() {
        let config = StressTestConfig {
            test_large_values: true,
            test_small_values: true,
            test_large_ranges: false,
            test_near_collinear: true,
            test_duplicates: true,
            test_special_values: false,
            test_extreme_extrapolation: false,
            test_dense_grids: false,
            test_ill_conditioned: true,
            max_dataset_size: 10000,
        };

        let results = stress_tests::test_linear_interpolation(&config);
        assert!(results.success_rate() > 0.8);

        let results = stress_tests::test_cubic_spline(&config);
        results.print_summary();
    }
}
