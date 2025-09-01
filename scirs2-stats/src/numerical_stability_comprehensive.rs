//! Comprehensive numerical stability testing for statistical operations
//!
//! This module provides extensive testing for numerical stability, precision,
//! and edge case handling across all statistical functions. It ensures that
//! the library behaves correctly with extreme values, near-singular conditions,
//! and challenging numerical scenarios.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumCast, Zero, One};
use std::collections::HashMap;
use std::fmt::Debug;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

/// Configuration for numerical stability testing
#[derive(Debug, Clone)]
pub struct NumericalStabilityConfig {
    /// Tolerance for floating-point comparisons
    pub tolerance: f64,
    /// Number of test iterations for stochastic tests
    pub test_iterations: usize,
    /// Enable testing with extreme values
    pub test_extreme_values: bool,
    /// Enable testing with near-zero values
    pub test_near_zero: bool,
    /// Enable testing with large values
    pub test_large_values: bool,
    /// Enable testing with mixed-scale data
    pub test_mixed_scale: bool,
    /// Enable testing with special values (NaN, Inf)
    pub test_special_values: bool,
    /// Enable testing with ill-conditioned matrices
    pub test_ill_conditioned: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for NumericalStabilityConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            test_iterations: 100,
            test_extreme_values: true,
            test_near_zero: true,
            test_large_values: true,
            test_mixed_scale: true,
            test_special_values: true,
            test_ill_conditioned: true,
            random_seed: Some(42),
        }
    }
}

/// Results of numerical stability testing
#[derive(Debug, Clone)]
pub struct StabilityTestResults {
    /// Total number of tests run
    pub total_tests: usize,
    /// Number of tests passed
    pub passed_tests: usize,
    /// Number of tests failed
    pub failed_tests: usize,
    /// Detailed test results
    pub test_details: Vec<TestResult>,
    /// Summary by test category
    pub category_summary: HashMap<String, CategorySummary>,
    /// Overall stability score (0-100)
    pub stability_score: f64,
}

/// Individual test result
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test name
    pub test_name: String,
    /// Test category
    pub category: String,
    /// Whether the test passed
    pub passed: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Input data characteristics
    pub input_characteristics: InputCharacteristics,
    /// Measured precision loss
    pub precision_loss: Option<f64>,
    /// Expected vs actual result comparison
    pub result_comparison: Option<String>,
}

/// Summary for a test category
#[derive(Debug, Clone)]
pub struct CategorySummary {
    /// Total tests in category
    pub total: usize,
    /// Passed tests in category
    pub passed: usize,
    /// Average precision loss
    pub avg_precision_loss: f64,
    /// Worst precision loss
    pub worst_precision_loss: f64,
}

/// Characteristics of input data for a test
#[derive(Debug, Clone)]
pub struct InputCharacteristics {
    /// Data size
    pub size: usize,
    /// Minimum value
    pub min_value: f64,
    /// Maximum value
    pub max_value: f64,
    /// Range (max - min)
    pub range: f64,
    /// Condition number (for matrices)
    pub condition_number: Option<f64>,
    /// Contains special values (NaN, Inf)
    pub has_special_values: bool,
    /// Data scale (order of magnitude)
    pub scale: String,
}

/// Comprehensive numerical stability tester
pub struct NumericalStabilityTester {
    config: NumericalStabilityConfig,
    rng: StdRng,
    results: Vec<TestResult>,
}

impl NumericalStabilityTester {
    /// Create a new numerical stability tester
    pub fn new(config: NumericalStabilityConfig) -> Self {
        let rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(rand::rng()),
        };

        Self {
            config: config,
            rng,
            results: Vec::new(),
        }
    }

    /// Run comprehensive numerical stability tests
    pub fn run_comprehensive_tests(&mut self) -> StabilityTestResults {
        self.results.clear();

        // Test basic statistical functions
        self.test_basic_statistics();
        
        // Test correlation functions
        self.test_correlation_stability();
        
        // Test with extreme values
        if self.config.test_extreme_values {
            self.test_extreme_values();
        }
        
        // Test with near-zero values
        if self.config.test_near_zero {
            self.test_near_zero_values();
        }
        
        // Test with large values
        if self.config.test_large_values {
            self.test_large_values();
        }
        
        // Test with mixed-scale data
        if self.config.test_mixed_scale {
            self.test_mixed_scaledata();
        }
        
        // Test with special values
        if self.config.test_special_values {
            self.test_special_values();
        }
        
        // Test ill-conditioned scenarios
        if self.config.test_ill_conditioned {
            self.test_ill_conditioned_cases();
        }
        
        // Test iterative algorithms
        self.test_iterative_algorithms();
        
        // Test numerical derivatives
        self.test_numerical_derivatives();

        self.compile_results()
    }

    /// Test basic statistical functions for numerical stability
    #[ignore = "timeout"]
    fn test_basic_statistics(&mut self) {
        // Test mean with various data types
        self.test_mean_stability();
        
        // Test variance with edge cases
        self.test_variance_stability();
        
        // Test standard deviation
        self.test_standard_deviation_stability();
        
        // Test skewness and kurtosis
        self.test_higher_moments_stability();
        
        // Test quantiles
        self.test_quantile_stability();
    }

    /// Test mean calculation stability
    fn test_mean_stability(&mut self) {
        // Test with constant values
        let constantdata = vec![1e10; 1000];
        self.run_test("mean_constant_large", "basic_statistics", &constantdata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::mean(&arr.view()).map(|mean| {
                let expected = data[0];
                (mean - expected).abs() < self.config.tolerance * expected.abs()
            }).unwrap_or(false)
        });

        // Test with tiny values
        let tiny_data: Vec<f64> = (0..1000).map(|i| 1e-15 * (i as f64 + 1.0)).collect();
        self.run_test("mean_tiny_values", "basic_statistics", &tiny_data, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::mean(&arr.view()).map(|mean| {
                mean.is_finite() && mean > 0.0
            }).unwrap_or(false)
        });

        // Test with alternating signs and cancellation
        let alternatingdata: Vec<f64> = (0..1000).map(|i| if i % 2 == 0 { 1e15 } else { -1e15 + 1.0 }).collect();
        self.run_test("mean_alternating_cancellation", "basic_statistics", &alternatingdata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::mean(&arr.view()).map(|mean| {
                // Should be close to 0.5 due to the +1.0 in odd elements
                (mean - 0.5).abs() < 1.0
            }).unwrap_or(false)
        });
    }

    /// Test variance calculation stability
    fn test_variance_stability(&mut self) {
        // Test with nearly identical values (catastrophic cancellation risk)
        let base = 1e12;
        let epsilon = 1e-6;
        let near_identical: Vec<f64> = (0..100).map(|i| base + (i as f64) * epsilon).collect();
        
        self.run_test("variance_near_identical", "basic_statistics", &near_identical, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::var(&arr.view(), 1, None).map(|var| {
                var >= 0.0 && var.is_finite()
            }).unwrap_or(false)
        });

        // Test Welford's algorithm stability
        let large_scaledata: Vec<f64> = (0..1000).map(|i| 1e9 + rand::rng().random::<f64>()).collect();
        self.run_test("variance_welford_large_scale", "basic_statistics", &large_scaledata, |data| {
            let arr = Array1::from_vec(data.clone());
            let result1 = crate::descriptive::var(&arr.view(), 1, None);
            let result2 = crate::memory_efficient::welford_variance(&arr.view(), 1);
            
            match (result1, result2) {
                (Ok(var1), Ok(var2)) => {
                    let rel_error = ((var1 - var2) / var1.max(1e-15)).abs();
                    rel_error < 1e-10
                }
                _ => false,
            }
        });
    }

    /// Test standard deviation stability
    fn test_standard_deviation_stability(&mut self) {
        // Test that std = sqrt(variance) relationship holds
        for _ in 0..10 {
            let data: Vec<f64> = (0..500).map(|_| rand::rng().random_range(-1e6..1e6)).collect();
            
            self.run_test("std_sqrt_variance_consistency".."basic_statistics", &data, |data| {
                let arr = Array1::from_vec(data.clone());
                match (crate::descriptive::var(&arr.view(), 1, None), crate::descriptive::std(&arr.view(), 1, None)) {
                    (Ok(var), Ok(std)) => {
                        let expected_std = var.sqrt();
                        let rel_error = ((std - expected_std) / expected_std.max(1e-15)).abs();
                        rel_error < 1e-12
                    }
                    _ => false,
                }
            });
        }
    }

    /// Test higher moments (skewness, kurtosis) stability
    fn test_higher_moments_stability(&mut self) {
        // Test with symmetric data (skewness should be near zero)
        let symmetricdata: Vec<f64> = (-500..=500).map(|i| i as f64).collect();
        
        self.run_test("skewness_symmetricdata", "higher_moments", &symmetricdata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::skew(&arr.view(), false, None).map(|skew| {
                skew.abs() < 1e-10
            }).unwrap_or(false)
        });

        // Test with normal-like data (kurtosis should be near 3)
        let normaldata: Vec<f64> = (0..10000).map(|_| {
            let normal = Normal::new(0.0, 1.0).unwrap();
            normal.sample(&mut self.rng)
        }).collect();
        
        self.run_test("kurtosis_normaldata", "higher_moments", &normaldata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::kurtosis(&arr.view(), false, false).map(|kurt| {
                (kurt - 3.0).abs() < 0.5 // Allow some tolerance for finite samples
            }).unwrap_or(false)
        });
    }

    /// Test quantile calculation stability
    fn test_quantile_stability(&mut self) {
        // Test with sorted data (should be exact)
        let sorteddata: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        
        self.run_test("quantilesorteddata", "quantiles", &sorteddata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::quantile::quantile(&arr.view(), 0.5, crate::quantile::QuantileInterpolation::Linear).map(|median| {
                let expected = 499.5; // Midpoint of 0-999
                (median - expected).abs() < 1e-10
            }).unwrap_or(false)
        });

        // Test with duplicate values
        let duplicatedata = vec![42.0; 1000];
        
        self.run_test("quantile_duplicate_values", "quantiles", &duplicatedata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::quantile::quantile(&arr.view(), 0.75, crate::quantile::QuantileInterpolation::Linear).map(|q75| {
                (q75 - 42.0).abs() < 1e-15
            }).unwrap_or(false)
        });
    }

    /// Test correlation stability
    fn test_correlation_stability(&mut self) {
        // Test perfect correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2*x
        
        self.run_test("correlation_perfect_positive", "correlation", &x, |_| {
            let x_arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            let y_arr = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);
            crate::correlation::pearson_r(&x_arr.view(), &y_arr.view()).map(|corr| {
                (corr - 1.0).abs() < 1e-12
            }).unwrap_or(false)
        });

        // Test with high precision requirements
        for _ in 0..5 {
            let basedata: Vec<f64> = (0..1000).map(|_| rand::rng().random::<f64>()).collect();
            let scaleddata: Vec<f64> = basedata.iter().map(|&x| 1e15 * x + 1e10).collect();
            
            self.run_test("correlation_high_precision", "correlation", &basedata, |base| {
                let base_arr = Array1::from_vec(base.clone());
                let scaled_arr = Array1::from_vec(base.iter().map(|&x| 1e15 * x + 1e10).collect());
                
                match (
                    crate::correlation::pearson_r(&base_arr.view(), &base_arr.view()),
                    crate::correlation::pearson_r(&scaled_arr.view(), &scaled_arr.view())
                ) {
                    (Ok(corr1), Ok(corr2)) => {
                        // Both should be 1.0 (perfect self-correlation)
                        (corr1 - 1.0).abs() < 1e-12 && (corr2 - 1.0).abs() < 1e-12
                    }
                    _ => false,
                }
            });
        }
    }

    /// Test with extreme values
    fn test_extreme_values(&mut self) {
        // Test with very large values
        let largedata = vec![1e100, 2e100, 3e100, 1e99];
        
        self.run_test("extreme_large_values", "extreme_values", &largedata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::mean(&arr.view()).map(|mean| {
                mean.is_finite() && mean > 0.0
            }).unwrap_or(false)
        });

        // Test with very small values
        let smalldata = vec![1e-100, 2e-100, 3e-100, 1e-99];
        
        self.run_test("extreme_small_values", "extreme_values", &smalldata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::mean(&arr.view()).map(|mean| {
                mean.is_finite() && mean > 0.0
            }).unwrap_or(false)
        });
    }

    /// Test with near-zero values
    fn test_near_zero_values(&mut self) {
        let near_zerodata: Vec<f64> = (0..100).map(|i| 1e-15 * (i as f64)).collect();
        
        self.run_test("near_zero_variance", "near_zero", &near_zerodata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::var(&arr.view(), 1, None).map(|var| {
                var >= 0.0 && var.is_finite()
            }).unwrap_or(false)
        });
    }

    /// Test with large values
    fn test_large_values(&mut self) {
        let largedata: Vec<f64> = (0..100).map(|i| 1e12 + (i as f64)).collect();
        
        self.run_test("large_values_precision", "large_values", &largedata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::var(&arr.view(), 1, None).map(|var| {
                // Variance should be close to actual variance
                let expected_var = ((data.len() - 1) as f64 * (data.len() as f64)) / 12.0;
                let rel_error = ((var - expected_var) / expected_var).abs();
                rel_error < 0.1
            }).unwrap_or(false)
        });
    }

    /// Test with mixed-scale data
    fn test_mixed_scaledata(&mut self) {
        let mixeddata = vec![1e-10, 1.0, 1e10, 2e-10, 2.0, 2e10];
        
        self.run_test("mixed_scale_mean", "mixed_scale", &mixeddata, |data| {
            let arr = Array1::from_vec(data.clone());
            crate::descriptive::mean(&arr.view()).map(|mean| {
                mean.is_finite()
            }).unwrap_or(false)
        });
    }

    /// Test with special values (NaN, Inf)
    fn test_special_values(&mut self) {
        // Test with NaN values
        let nandata = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
        
        self.run_test("special_nan_handling", "special_values", &nandata, |data| {
            let arr = Array1::from_vec(data.clone());
            // Should either handle NaN gracefully or return an appropriate error
            match crate::descriptive::mean(&arr.view()) {
                Ok(mean) => !mean.is_nan(), // If it succeeds, result shouldn't be NaN
                Err(_) => true, // Error is acceptable for NaN input
            }
        });

        // Test with infinite values
        let infdata = vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0];
        
        self.run_test("special_inf_handling", "special_values", &infdata, |data| {
            let arr = Array1::from_vec(data.clone());
            match crate::descriptive::mean(&arr.view()) {
                Ok(mean) => mean.is_finite(), // Should handle infinity appropriately
                Err(_) => true, // Error is acceptable for infinite input
            }
        });
    }

    /// Test ill-conditioned cases
    fn test_ill_conditioned_cases(&mut self) {
        // Test correlation with nearly collinear data
        let x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&val| val + 1e-10 * rand::rng().random::<f64>()).collect();
        
        self.run_test("ill_conditioned_correlation", "ill_conditioned", &x, |_| {
            let x_arr = Array1::from_vec((0..1000).map(|i| i as f64).collect());
            let y_arr = Array1::from_vec(x_arr.iter().map(|&val| val + 1e-10 * 0.5).collect());
            
            crate::correlation::pearson_r(&x_arr.view(), &y_arr.view()).map(|corr| {
                corr > 0.999 && corr <= 1.0 // Should be very high correlation
            }).unwrap_or(false)
        });
    }

    /// Test iterative algorithms
    fn test_iterative_algorithms(&mut self) {
        // Test convergence properties of iterative statistical methods
        // This would test things like iterative PCA, EM algorithms, etc.
        
        // For now, test a simple iterative mean calculation
        let data: Vec<f64> = (0..1000).map(|_| rand::rng().random::<f64>()).collect();
        
        self.run_test("iterative_convergence", "iterative", &data, |data| {
            let arr = Array1::from_vec(data.clone());
            
            // Compare batch vs iterative mean calculation
            let batch_mean = crate::descriptive::mean(&arr.view()).unwrap_or(0.0);
            let mut iterative_mean = 0.0;
            let mut count = 0.0;
            
            for &value in data {
                count += 1.0;
                iterative_mean += (value - iterative_mean) / count;
            }
            
            let error = (batch_mean - iterative_mean).abs();
            error < 1e-12
        });
    }

    /// Test numerical derivatives (for testing optimization algorithms)
    fn test_numerical_derivatives(&mut self) {
        // Test finite difference approximations
        self.run_test("numerical_derivative_accuracy", "derivatives", &vec![1.0], |_| {
            let f = |x: f64| x * x; // f(x) = x^2, f'(x) = 2x
            let x = 1.0;
            let h = 1e-8;
            
            // Forward difference
            let forward_diff = (f(x + h) - f(x)) / h;
            
            // Central difference
            let central_diff = (f(x + h) - f(x - h)) / (2.0 * h);
            
            let true_derivative = 2.0 * x; // 2 * 1 = 2
            
            let forward_error = (forward_diff - true_derivative).abs();
            let central_error = (central_diff - true_derivative).abs();
            
            // Central difference should be more accurate
            central_error < forward_error && central_error < 1e-6
        });
    }

    /// Run a single test and record the result
    fn run_test<F>(&mut self, test_name: &str, category: &str, data: &[f64], testfn: F)
    where
        F: FnOnce(&[f64]) -> bool,
    {
        let input_characteristics = self.analyze_input_characteristics(data);
        
        let passed = test_fn(data);
        
        let test_result = TestResult {
            test_name: test_name.to_string(),
            category: category.to_string(),
            passed,
            error_message: if passed { None } else { Some("Test failed".to_string()) },
            input_characteristics,
            precision_loss: None, // Could be computed if needed
            result_comparison: None,
        };
        
        self.results.push(test_result);
    }

    /// Analyze characteristics of input data
    fn analyze_input_characteristics(&self, data: &[f64]) -> InputCharacteristics {
        let size = data.len();
        let min_value = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_value = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_value - min_value;
        
        let has_special_values = data.iter().any(|&x| x.is_nan() || x.is_infinite());
        
        let scale = if max_value.abs() > 1e6 {
            "large".to_string()
        } else if max_value.abs() < 1e-6 {
            "small".to_string()
        } else {
            "normal".to_string()
        };
        
        InputCharacteristics {
            size,
            min_value,
            max_value,
            range,
            condition_number: None, // Would compute for matrices
            has_special_values,
            scale,
        }
    }

    /// Compile final test results
    fn compile_results(&self) -> StabilityTestResults {
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;
        
        let mut category_summary = HashMap::new();
        
        // Group by category
        for result in &self.results {
            let entry = category_summary.entry(result.category.clone()).or_insert(CategorySummary {
                total: 0,
                passed: 0,
                avg_precision_loss: 0.0,
                worst_precision_loss: 0.0,
            });
            
            entry.total += 1;
            if result.passed {
                entry.passed += 1;
            }
        }
        
        let stability_score = if total_tests > 0 {
            (passed_tests as f64 / total_tests as f64) * 100.0
        } else {
            0.0
        };
        
        StabilityTestResults {
            total_tests,
            passed_tests,
            failed_tests,
            test_details: self.results.clone(),
            category_summary,
            stability_score,
        }
    }
}

/// Generate comprehensive numerical stability report
#[allow(dead_code)]
pub fn generate_stability_report(config: Option<NumericalStabilityConfig>) -> StabilityTestResults {
    let _config = config.unwrap_or_default();
    let mut tester = NumericalStabilityTester::new(_config);
    tester.run_comprehensive_tests()
}

/// Quick numerical stability check with default configuration
#[allow(dead_code)]
pub fn quick_stability_check() -> bool {
    let results = generate_stability_report(None);
    results.stability_score > 95.0 // Require 95% pass rate
}

/// Advanced numerical fixes and improvements
pub struct NumericalStabilityFixes;

impl NumericalStabilityFixes {
    /// Improved mean calculation with Kahan summation for better numerical stability
    pub fn stable_mean(data: &[f64]) -> StatsResult<f64> {
        if data.is_empty() {
            return Err(StatsError::InvalidArgument("Cannot compute mean of empty data".to_string()));
        }
        
        // Use Kahan summation algorithm for better numerical stability
        let mut sum = 0.0;
        let mut compensation = 0.0;
        
        for &value in data {
            if !value.is_finite() {
                return Err(StatsError::InvalidArgument(format!("Non-finite value detected: {}", value)));
            }
            
            let adjusted_value = value - compensation;
            let temp_sum = sum + adjusted_value;
            compensation = (temp_sum - sum) - adjusted_value;
            sum = temp_sum;
        }
        
        Ok(sum / data.len() as f64)
    }
    
    /// Numerically stable variance using Welford's online algorithm
    pub fn stable_variance(data: &[f64], ddof: usize) -> StatsResult<f64> {
        if data.len() <= ddof {
            return Err(StatsError::InvalidArgument(
                format!("Insufficient data points: {} <= {}", data.len(), ddof)
            ));
        }
        
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut count = 0;
        
        for &value in data {
            if !value.is_finite() {
                return Err(StatsError::InvalidArgument(format!("Non-finite value detected: {}", value)));
            }
            
            count += 1;
            let delta = value - mean;
            mean += delta / count as f64;
            let delta2 = value - mean;
            m2 += delta * delta2;
        }
        
        if count - ddof <= 0 {
            return Err(StatsError::InvalidArgument("Invalid degrees of freedom".to_string()));
        }
        
        Ok(m2 / (count - ddof) as f64)
    }
    
    /// Numerically stable correlation using enhanced precision
    pub fn stable_correlation(x: &[f64], y: &[f64]) -> StatsResult<f64> {
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch("Arrays must have same length".to_string()));
        }
        if x.len() < 2 {
            return Err(StatsError::InvalidArgument("Need at least 2 data points".to_string()));
        }
        
        // Check for finite values
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            if !xi.is_finite() || !yi.is_finite() {
                return Err(StatsError::InvalidArgument("Non-finite values detected".to_string()));
            }
        }
        
        // Use numerically stable calculation
        let n = x.len() as f64;
        let mean_x = Self::stable_mean(x)?;
        let mean_y = Self::stable_mean(y)?;
        
        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;
        
        // Use compensated summation for better stability
        let mut num_comp = 0.0;
        let mut den_x_comp = 0.0;
        let mut den_y_comp = 0.0;
        
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            
            // Numerator: sum of products of deviations
            let num_term = dx * dy - num_comp;
            let num_temp = num + num_term;
            num_comp = (num_temp - num) - num_term;
            num = num_temp;
            
            // Denominator X: sum of squared deviations
            let den_x_term = dx * dx - den_x_comp;
            let den_x_temp = den_x + den_x_term;
            den_x_comp = (den_x_temp - den_x) - den_x_term;
            den_x = den_x_temp;
            
            // Denominator Y: sum of squared deviations
            let den_y_term = dy * dy - den_y_comp;
            let den_y_temp = den_y + den_y_term;
            den_y_comp = (den_y_temp - den_y) - den_y_term;
            den_y = den_y_temp;
        }
        
        let denominator = (den_x * den_y).sqrt();
        if denominator < f64::EPSILON {
            return Ok(0.0); // One or both variables have zero variance
        }
        
        let correlation = num / denominator;
        
        // Clamp to valid range to handle floating-point errors
        Ok(correlation.max(-1.0).min(1.0))
    }
    
    /// Detect and fix common numerical issues in data
    pub fn diagnose_and_fixdata_issues(data: &[f64]) -> (Vec<f64>, Vec<String>) {
        let mut fixeddata = Vec::new();
        let mut issues_fixed = Vec::new();
        
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut extreme_count = 0;
        
        for &value in data {
            if value.is_nan() {
                nan_count += 1;
                // Skip NaN values (could also interpolate)
                continue;
            } else if value.is_infinite() {
                inf_count += 1;
                // Replace infinity with large finite value
                if value.is_sign_positive() {
                    fixeddata.push(f64::MAX / 2.0);
                } else {
                    fixeddata.push(f64::MIN / 2.0);
                }
            } else if value.abs() > 1e100 || (value != 0.0 && value.abs() < 1e-100) {
                extreme_count += 1;
                // Cap extreme values
                if value > 1e100 {
                    fixeddata.push(1e100);
                } else if value < -1e100 {
                    fixeddata.push(-1e100);
                } else if value.abs() < 1e-100 && value != 0.0 {
                    fixeddata.push(if value > 0.0 { 1e-100 } else { -1e-100 });
                } else {
                    fixeddata.push(value);
                }
            } else {
                fixeddata.push(value);
            }
        }
        
        if nan_count > 0 {
            issues_fixed.push(format!("Removed {} NaN values", nan_count));
        }
        if inf_count > 0 {
            issues_fixed.push(format!("Capped {} infinite values", inf_count));
        }
        if extreme_count > 0 {
            issues_fixed.push(format!("Capped {} extreme values", extreme_count));
        }
        
        (fixeddata, issues_fixed)
    }
    
    /// Test for numerical conditioning of a matrix (simplified)
    pub fn matrix_condition_number(matrix: &Array2<f64>) -> StatsResult<f64> {
        let (rows, cols) = matrix.dim();
        if rows != cols {
            return Err(StatsError::InvalidArgument("Matrix must be square".to_string()));
        }
        
        // Simplified condition number estimation using diagonal dominance
        let mut min_diag = f64::INFINITY;
        let mut max_off_diag_sum = 0.0;
        
        for i in 0..rows {
            let diag_val = matrix[[i, i]].abs();
            if diag_val < min_diag {
                min_diag = diag_val;
            }
            
            let mut off_diag_sum = 0.0;
            for j in 0..cols {
                if i != j {
                    off_diag_sum += matrix[[i, j]].abs();
                }
            }
            if off_diag_sum > max_off_diag_sum {
                max_off_diag_sum = off_diag_sum;
            }
        }
        
        if min_diag < f64::EPSILON {
            return Ok(f64::INFINITY); // Singular or near-singular
        }
        
        // Rough condition number estimate
        Ok(max_off_diag_sum / min_diag + 1.0)
    }
    
    /// Regularize a potentially ill-conditioned matrix
    pub fn regularize_matrix(matrix: &Array2<f64>, regularization: f64) -> Array2<f64> {
        let (rows, cols) = matrix.dim();
        let mut regularized = matrix.clone();
        
        // Add regularization to diagonal
        for i in 0..rows.min(cols) {
            regularized[[i, i]] += regularization;
        }
        
        regularized
    }
}

/// Enhanced precision arithmetic utilities
pub struct EnhancedPrecisionArithmetic;

impl EnhancedPrecisionArithmetic {
    /// Two-sum algorithm for exact floating-point addition
    pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let v = s - a;
        let e = (a - (s - v)) + (b - v);
        (s, e)
    }
    
    /// Compensated summation using Kahan algorithm
    pub fn kahan_sum(values: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut compensation = 0.0;
        
        for &value in _values {
            let adjusted_value = value - compensation;
            let temp_sum = sum + adjusted_value;
            compensation = (temp_sum - sum) - adjusted_value;
            sum = temp_sum;
        }
        
        sum
    }
    
    /// Pairwise summation for improved numerical stability
    pub fn pairwise_sum(values: &[f64]) -> f64 {
        match values.len() {
            0 => 0.0,
            1 => values[0],
            2 => values[0] + values[1],
            n => {
                let mid = n / 2;
                Self::pairwise_sum(&_values[..mid]) + Self::pairwise_sum(&_values[mid..])
            }
        }
    }
    
    /// Compute dot product with enhanced precision
    pub fn enhanced_dot_product(x: &[f64], y: &[f64]) -> StatsResult<f64> {
        if x.len() != y.len() {
            return Err(StatsError::DimensionMismatch("Arrays must have same length".to_string()));
        }
        
        let products: Vec<f64> = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).collect();
        Ok(Self::kahan_sum(&products))
    }
}

/// Numerical stability fixes integrated with existing functions
pub struct IntegratedStabilityFixes;

impl IntegratedStabilityFixes {
    /// Create a wrapper function that automatically applies stability fixes
    pub fn with_stability_checks<T, F>(data: &[f64], operation: F) -> StatsResult<T>
    where
        F: FnOnce(&[f64]) -> StatsResult<T>,
    {
        // First, diagnose and fix data issues
        let (fixeddata, issues) = NumericalStabilityFixes::diagnose_and_fixdata_issues(data);
        
        if !issues.is_empty() {
            eprintln!("Numerical stability fixes applied: {:?}", issues);
        }
        
        // Apply the operation to the fixed data
        operation(&fixeddata)
    }
    
    /// Stability-enhanced mean calculation
    pub fn enhanced_mean(data: &[f64]) -> StatsResult<f64> {
        Self::with_stability_checks(data, |fixeddata| {
            NumericalStabilityFixes::stable_mean(fixeddata)
        })
    }
    
    /// Stability-enhanced variance calculation
    pub fn enhanced_variance(data: &[f64], ddof: usize) -> StatsResult<f64> {
        Self::with_stability_checks(data, |fixeddata| {
            NumericalStabilityFixes::stable_variance(fixeddata, ddof)
        })
    }
    
    /// Stability-enhanced correlation calculation
    pub fn enhanced_correlation(x: &[f64], y: &[f64]) -> StatsResult<f64> {
        let (fixed_x_) = NumericalStabilityFixes::diagnose_and_fixdata_issues(x);
        let (fixed_y_) = NumericalStabilityFixes::diagnose_and_fixdata_issues(y);
        
        NumericalStabilityFixes::stable_correlation(&fixed_x, &fixed_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_stability_comprehensive() {
        let config = NumericalStabilityConfig {
            test_iterations: 10, // Reduce for faster testing
            ..Default::default()
        };
        
        let results = generate_stability_report(Some(config));
        
        // Should pass most tests
        assert!(results.stability_score > 80.0);
        assert!(results.passed_tests > 0);
        
        // Should test multiple categories
        assert!(results.category_summary.len() > 5);
    }

    #[test]
    fn test_quick_stability_check() {
        // This should pass for a well-implemented library
        assert!(quick_stability_check());
    }

    #[test]
    fn test_basic_statistics_stability() {
        let mut tester = NumericalStabilityTester::new(Default::default());
        tester.test_basic_statistics();
        
        let results = tester.compile_results();
        assert!(results.passed_tests > 0);
    }

    #[test]
    fn test_extreme_values_handling() {
        let mut tester = NumericalStabilityTester::new(Default::default());
        tester.test_extreme_values();
        
        let results = tester.compile_results();
        // Should handle extreme values gracefully
        assert!(results.stability_score > 50.0);
    }

    #[test]
    fn test_stable_mean_with_cancellation() {
        // Test mean with potential cancellation issues
        let data = vec![1e15, -1e15, 1.0, 2.0, 3.0];
        let result = NumericalStabilityFixes::stable_mean(&data).unwrap();
        let expected = 6.0 / 5.0; // (0 + 1 + 2 + 3) / 5
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_stable_variance() {
        let data = vec![1e12, 1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0];
        let result = NumericalStabilityFixes::stable_variance(&data, 1).unwrap();
        assert!(result > 0.0 && result.is_finite());
    }

    #[test]
    fn test_stable_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = NumericalStabilityFixes::stable_correlation(&x, &y).unwrap();
        assert!((result - 1.0).abs() < 1e-12);
    }

    #[test]
    fn testdata_issue_diagnosis() {
        let data = vec![1.0, f64::NAN, f64::INFINITY, -f64::INFINITY, 1e200, -1e200, 0.0];
        let (fixed, issues) = NumericalStabilityFixes::diagnose_and_fixdata_issues(&data);
        
        assert!(!issues.is_empty());
        assert!(fixed.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kahan_summation() {
        let data = vec![1.0, 1e15, 1.0, -1e15];
        let kahan_result = EnhancedPrecisionArithmetic::kahan_sum(&data);
        let naive_result: f64 = data.iter().sum();
        
        // Kahan should be more accurate
        assert!((kahan_result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_summation() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let pairwise_result = EnhancedPrecisionArithmetic::pairwise_sum(&data);
        let expected = 100.0 * 101.0 / 2.0; // Sum of 1 to 100
        
        assert!((pairwise_result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_enhanced_dot_product() {
        let x = vec![1e15, 1.0];
        let y = vec![1e-15, 1.0];
        
        let result = EnhancedPrecisionArithmetic::enhanced_dot_product(&x, &y).unwrap();
        let expected = 1.0 + 1.0; // 1e15 * 1e-15 + 1.0 * 1.0 = 1 + 1 = 2
        
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_condition_number() {
        let well_conditioned = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let condition = NumericalStabilityFixes::matrix_condition_number(&well_conditioned).unwrap();
        assert!(condition.is_finite() && condition > 0.0);
        
        let ill_conditioned = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0001]).unwrap();
        let condition2 = NumericalStabilityFixes::matrix_condition_number(&ill_conditioned).unwrap();
        assert!(condition2 > condition);
    }

    #[test]
    fn test_integrated_stability_fixes() {
        let problematicdata = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];
        
        let result = IntegratedStabilityFixes::enhanced_mean(&problematicdata);
        assert!(result.is_ok());
        assert!(result.unwrap().is_finite());
    }
}
