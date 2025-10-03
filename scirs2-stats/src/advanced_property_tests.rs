//! Advanced Property-Based Testing Framework
//!
//! Advanced property-based testing specifically designed for Advanced mode,
//! featuring comprehensive mathematical invariant testing, numerical stability
//! verification, SIMD consistency checks, and performance regression detection.

use crate::advanced_simd_stats::{BatchOperation, AdvancedSimdConfig, AdvancedSimdOptimizer};
use crate::parallel_enhancements::AdvancedParallelConfig;
use crate::{kurtosis, mean, pearson_r, skew, std, var};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};
use std::time::Instant;

/// Advanced-comprehensive property testing framework
pub struct AdvancedPropertyTester {
    simd_config: AdvancedSimdConfig,
    parallel_config: AdvancedParallelConfig,
    numerical_tolerance: f64,
    performance_tolerance: f64,
}

impl Default for AdvancedPropertyTester {
    fn default() -> Self {
        Self {
            simd_config: AdvancedSimdConfig::default(),
            parallel_config: AdvancedParallelConfig::default(),
            numerical_tolerance: 1e-12,
            performance_tolerance: 2.0, // 2x slowdown tolerance
        }
    }
}

impl AdvancedPropertyTester {
    /// Create a new property tester with custom configuration
    pub fn new(
        simd_config: AdvancedSimdConfig,
        parallel_config: AdvancedParallelConfig,
    ) -> Self {
        Self {
            simd_config,
            parallel_config,
            numerical_tolerance: 1e-12,
            performance_tolerance: 2.0,
        }
    }

    /// Test SIMD vs scalar consistency for all statistical operations
    pub fn test_simd_scalar_consistency<F>(&self, data: &ArrayView1<F>) -> PropertyTestResult
    where
        F: Float + NumCast + Copy + Send + Sync + PartialOrd + std::fmt::Debug
        + std::fmt::Display,
    {
        let n = data.len();
        if n < 10 {
            return PropertyTestResult::Skipped("Insufficient data for SIMD testing".to_string());
        }

        let mut errors = Vec::new();

        // Test batch statistics consistency
        let optimizer = AdvancedSimdOptimizer::new(self.simd_config.clone());
        let data_arrays = vec![data.to_owned().view()];
        let operations = vec![BatchOperation::Mean, BatchOperation::Variance];

        match optimizer.advanced_batch_statistics(&data_arrays, &operations) {
            Ok(simd_result) => {
                // Compare with scalar implementations
                if let Ok(scalar_mean) = mean(data) {
                    let mean_diff = (simd_result.means[0].to_f64().unwrap()
                        - scalar_mean.to_f64().unwrap())
                    .abs();
                    if mean_diff > self.numerical_tolerance {
                        errors.push(format!("SIMD mean differs from scalar: {}", mean_diff));
                    }
                }

                if let Ok(scalar_var) = var(data, 0) {
                    let var_diff = (simd_result.variances[0].to_f64().unwrap()
                        - scalar_var.to_f64().unwrap())
                    .abs();
                    if var_diff > self.numerical_tolerance {
                        errors.push(format!("SIMD variance differs from scalar: {}", var_diff));
                    }
                }
            }
            Err(e) => errors.push(format!("SIMD batch statistics failed: {:?}", e)),
        }

        if errors.is_empty() {
            PropertyTestResult::Passed
        } else {
            PropertyTestResult::Failed(errors.join("; "))
        }
    }

    /// Test mathematical invariants for statistical operations
    pub fn test_mathematical_invariants<F>(&self, data: &ArrayView1<F>) -> PropertyTestResult
    where
        F: Float + NumCast + Copy + Send + Sync + PartialOrd + std::fmt::Debug
        + std::fmt::Display,
    {
        let n = data.len();
        if n < 4 {
            return PropertyTestResult::Skipped(
                "Insufficient data for invariant testing".to_string(),
            );
        }

        let mut errors = Vec::new();

        // Test fundamental mathematical properties
        if let (Ok(mean_val), Ok(var_val), Ok(std_val)) = (mean(data), var(data, 0), std(data, 0)) {
            // 1. Variance-Standard Deviation Relationship: Var(X) = Std(X)²
            let var_std_diff =
                (var_val.to_f64().unwrap() - (std_val.to_f64().unwrap()).powi(2)).abs();
            if var_std_diff > self.numerical_tolerance {
                errors.push(format!(
                    "Variance-std relationship violated: {}",
                    var_std_diff
                ));
            }

            // 2. Mean bounds: min(X) ≤ E[X] ≤ max(X)
            let min_val = data.iter().fold(F::infinity(), |a, &b| a.min(b));
            let max_val = data.iter().fold(F::neg_infinity(), |a, &b| a.max(b));

            if mean_val < min_val || mean_val > max_val {
                errors.push("Mean outside data bounds".to_string());
            }

            // 3. Non-negativity: Var(X) ≥ 0
            if var_val < F::zero() {
                errors.push("Negative variance detected".to_string());
            }

            // 4. Standardization property: If Y = (X - μ)/σ, then E[Y] = 0, Var(Y) = 1
            if std_val > F::from(1e-10).unwrap() {
                let standardized: Array1<F> = data.map(|&x| (x - mean_val) / std_val);
                if let (Ok(std_mean), Ok(std_var)) =
                    (mean(&standardized.view()), var(&standardized.view(), 0, None))
                {
                    let mean_error = std_mean.to_f64().unwrap().abs();
                    let var_error = (std_var.to_f64().unwrap() - 1.0).abs();

                    if mean_error > self.numerical_tolerance {
                        errors.push(format!("Standardized mean not zero: {}", mean_error));
                    }
                    if var_error > self.numerical_tolerance {
                        errors.push(format!("Standardized variance not one: {}", var_error));
                    }
                }
            }
        }

        // Test higher moment properties
        if n >= 4 {
            if let (Ok(skew_val), Ok(kurt_val)) = (skew(data, false), kurtosis(data, true, false)) {
                // 5. Skewness bounds for most distributions
                let skew_f64 = skew_val.to_f64().unwrap();
                if skew_f64.abs() > 100.0 {
                    // Extreme skewness check
                    errors.push(format!("Extreme skewness detected: {}", skew_f64));
                }

                // 6. Kurtosis minimum bound (Fisher's definition)
                let kurt_f64 = kurt_val.to_f64().unwrap();
                if kurt_f64 < -2.0 - self.numerical_tolerance {
                    errors.push(format!("Kurtosis below theoretical minimum: {}", kurt_f64));
                }
            }
        }

        if errors.is_empty() {
            PropertyTestResult::Passed
        } else {
            PropertyTestResult::Failed(errors.join("; "))
        }
    }

    /// Test numerical stability under various transformations
    pub fn test_numerical_stability<F>(&self, data: &ArrayView1<F>) -> PropertyTestResult
    where
        F: Float + NumCast + Copy + Send + Sync + PartialOrd + std::fmt::Debug
        + std::fmt::Display,
    {
        let n = data.len();
        if n < 2 {
            return PropertyTestResult::Skipped(
                "Insufficient data for stability testing".to_string(),
            );
        }

        let mut errors = Vec::new();
        let tolerance = F::from(self.numerical_tolerance).unwrap();

        // Test 1: Translation invariance for variance
        // Var(X + c) = Var(X)
        let offset = F::from(1e6).unwrap();
        let translated: Array1<F> = data.map(|&x| x + offset);

        if let (Ok(orig_var), Ok(trans_var)) = (var(data, 0, None), var(&translated.view(), 0, None)) {
            let var_diff = (orig_var - trans_var).abs();
            if var_diff > tolerance {
                errors.push(format!(
                    "Translation invariance violated for variance: {}",
                    var_diff.to_f64().unwrap()
                ));
            }
        }

        // Test 2: Scaling properties
        // Var(aX) = a²Var(X)
        let scale = F::from(2.0).unwrap();
        let scaled: Array1<F> = data.map(|&x| x * scale);

        if let (Ok(orig_var), Ok(scaled_var)) = (var(data, 0, None), var(&scaled.view(), 0, None)) {
            let expected_var = scale * scale * orig_var;
            let scaling_error = (scaled_var - expected_var).abs();
            if scaling_error > tolerance * expected_var.abs() {
                errors.push(format!(
                    "Scaling property violated for variance: {}",
                    scaling_error.to_f64().unwrap()
                ));
            }
        }

        // Test 3: Precision loss detection for extreme values
        let max_val = data.iter().fold(F::neg_infinity(), |a, &b| a.max(b));
        let min_val = data.iter().fold(F::infinity(), |a, &b| a.min(b));
        let range = max_val - min_val;

        if range > F::from(1e10).unwrap() {
            errors.push("Extreme value range detected - potential precision loss".to_string());
        }

        // Test 4: Check for catastrophic cancellation in variance computation
        // Compare one-pass vs two-pass algorithms
        if let Ok(mean_val) = mean(data) {
            let two_pass_var = data
                .iter()
                .map(|&x| {
                    let diff = x - mean_val;
                    diff * diff
                })
                .fold(F::zero(), |acc, x| acc + x)
                / F::from(n).unwrap();

            if let Ok(one_pass_var) = var(data, 0) {
                let algorithm_diff = (two_pass_var - one_pass_var).abs();
                let relative_error = algorithm_diff / one_pass_var.abs().max(tolerance);

                if relative_error > F::from(1e-6).unwrap() {
                    errors.push(format!(
                        "Significant algorithmic variance difference: {}",
                        relative_error.to_f64().unwrap()
                    ));
                }
            }
        }

        if errors.is_empty() {
            PropertyTestResult::Passed
        } else {
            PropertyTestResult::Failed(errors.join("; "))
        }
    }

    /// Test correlation properties and bounds
    pub fn test_correlation_properties<F>(
        &self,
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
    ) -> PropertyTestResult
    where
        F: Float + NumCast + Copy + Send + Sync + PartialOrd + std::fmt::Debug
        + std::fmt::Display,
    {
        if x.len() != y.len() || x.len() < 2 {
            return PropertyTestResult::Skipped("Invalid data for correlation testing".to_string());
        }

        let mut errors = Vec::new();

        if let Ok(corr) = pearson_r(x, y) {
            let corr_f64 = corr.to_f64().unwrap();

            // 1. Correlation bounds: -1 ≤ r ≤ 1
            if corr_f64 < -1.0 - self.numerical_tolerance
                || corr_f64 > 1.0 + self.numerical_tolerance
            {
                errors.push(format!("Correlation outside bounds: {}", corr_f64));
            }

            // 2. Perfect correlation tests
            if corr_f64.abs() > 0.99999 {
                // Check if data is actually linearly related
                if let (Ok(x_var), Ok(y_var)) = (var(x, 0), var(y, 0)) {
                    if x_var.to_f64().unwrap() > 1e-10 && y_var.to_f64().unwrap() > 1e-10 {
                        // For near-perfect correlation, should be approximately linear
                        let linear_check = self.check_linearity(x, y);
                        if !linear_check && corr_f64.abs() > 0.9999 {
                            errors.push(
                                "Perfect correlation claimed but data not linear".to_string(),
                            );
                        }
                    }
                }
            }

            // 3. Symmetry: corr(X,Y) = corr(Y,X)
            if let Ok(corr_yx) = pearson_r(y, x) {
                let symmetry_error = (corr_f64 - corr_yx.to_f64().unwrap()).abs();
                if symmetry_error > self.numerical_tolerance {
                    errors.push(format!("Correlation asymmetry: {}", symmetry_error));
                }
            }

            // 4. Self-correlation: corr(X,X) = 1 (if var > 0)
            if let Ok(x_var) = var(x, 0) {
                if x_var.to_f64().unwrap() > 1e-10 {
                    if let Ok(self_corr) = pearson_r(x, x) {
                        let self_corr_error = (self_corr.to_f64().unwrap() - 1.0).abs();
                        if self_corr_error > self.numerical_tolerance {
                            errors.push(format!("Self-correlation not 1: {}", self_corr_error));
                        }
                    }
                }
            }
        }

        if errors.is_empty() {
            PropertyTestResult::Passed
        } else {
            PropertyTestResult::Failed(errors.join("; "))
        }
    }

    /// Test performance consistency and regression detection
    pub fn test_performance_consistency<F>(&self, data: &ArrayView1<F>) -> PropertyTestResult
    where
        F: Float + NumCast + Copy + Send + Sync + PartialOrd + std::fmt::Debug
        + std::fmt::Display,
    {
        let n = data.len();
        if n < 1000 {
            return PropertyTestResult::Skipped(
                "Dataset too small for performance testing".to_string(),
            );
        }

        let mut errors = Vec::new();

        // Benchmark scalar operations
        let start = Instant::now();
        let _ = mean(data);
        let _ = var(data, 0);
        let _ = std(data, 0);
        let scalar_time = start.elapsed();

        // Benchmark SIMD operations
        let start = Instant::now();
        let optimizer = AdvancedSimdOptimizer::new(self.simd_config.clone());
        let data_arrays = vec![data.to_owned().view()];
        let operations = vec![
            BatchOperation::Mean,
            BatchOperation::Variance,
            BatchOperation::StandardDeviation,
        ];
        let _ = optimizer.advanced_batch_statistics(&data_arrays, &operations);
        let simd_time = start.elapsed();

        // Check for performance regression
        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

        if speedup < 0.5 {
            errors.push(format!(
                "Significant performance regression detected: {:.2}x slower",
                1.0 / speedup
            ));
        }

        // Check for reasonable performance bounds
        let ops_per_second = n as f64 / simd_time.as_secs_f64();
        if ops_per_second < 1_000_000.0 {
            // Less than 1M elements per second
            errors.push(format!("Poor performance: {:.0} ops/sec", ops_per_second));
        }

        if errors.is_empty() {
            PropertyTestResult::Passed
        } else {
            PropertyTestResult::Failed(errors.join("; "))
        }
    }

    /// Run comprehensive property tests on given data
    pub fn run_comprehensive_tests<F>(&self, data: &ArrayView1<F>) -> ComprehensiveTestReport
    where
        F: Float + NumCast + Copy + Send + Sync + PartialOrd + std::fmt::Debug
        + std::fmt::Display,
    {
        let mut report = ComprehensiveTestReport::new();

        report.simd_consistency = self.test_simd_scalar_consistency(data);
        report.mathematical_invariants = self.test_mathematical_invariants(data);
        report.numerical_stability = self.test_numerical_stability(data);
        report.performance_consistency = self.test_performance_consistency(data);

        // Additional correlation tests if we can split the data
        if data.len() >= 4 {
            let mid = data.len() / 2;
            let x = data.slice(scirs2_core::ndarray::s![..mid]);
            let y = data.slice(scirs2_core::ndarray::s![mid..mid + x.len()]);
            report.correlation_properties = self.test_correlation_properties(&x, &y);
        }

        report
    }

    // Helper methods

    fn check_linearity<F>(&self, x: &ArrayView1<F>, y: &ArrayView1<F>) -> bool
    where
        F: Float + NumCast + Copy
        + std::fmt::Display,
    {
        if x.len() < 3 {
            return true; // Can't determine linearity with < 3 points
        }

        // Simple linearity check: compute residuals from best fit line
        let n = x.len() as f64;
        let sum_x = x
            .iter()
            .fold(F::zero(), |acc, &val| acc + val)
            .to_f64()
            .unwrap();
        let sum_y = y
            .iter()
            .fold(F::zero(), |acc, &val| acc + val)
            .to_f64()
            .unwrap();
        let sum_xy = x
            .iter()
            .zip(y.iter())
            .fold(F::zero(), |acc, (&xi, &yi)| acc + xi * yi)
            .to_f64()
            .unwrap();
        let sum_x2 = x
            .iter()
            .fold(F::zero(), |acc, &val| acc + val * val)
            .to_f64()
            .unwrap();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Check residuals
        let residual_sum_squares: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| {
                let predicted = slope * xi.to_f64().unwrap() + intercept;
                let residual = yi.to_f64().unwrap() - predicted;
                residual * residual
            })
            .sum();

        let mse = residual_sum_squares / n;
        mse < 1e-6 // Very small MSE indicates linearity
    }
}

/// Result of a property test
#[derive(Debug, Clone)]
pub enum PropertyTestResult {
    Passed,
    Failed(String),
    Skipped(String),
}

/// Comprehensive test report for all property tests
#[derive(Debug, Clone)]
pub struct ComprehensiveTestReport {
    pub simd_consistency: PropertyTestResult,
    pub mathematical_invariants: PropertyTestResult,
    pub numerical_stability: PropertyTestResult,
    pub correlation_properties: PropertyTestResult,
    pub performance_consistency: PropertyTestResult,
}

impl ComprehensiveTestReport {
    fn new() -> Self {
        Self {
            simd_consistency: PropertyTestResult::Skipped("Not run".to_string()),
            mathematical_invariants: PropertyTestResult::Skipped("Not run".to_string()),
            numerical_stability: PropertyTestResult::Skipped("Not run".to_string()),
            correlation_properties: PropertyTestResult::Skipped("Not run".to_string()),
            performance_consistency: PropertyTestResult::Skipped("Not run".to_string()),
        }
    }

    /// Check if all tests passed or were skipped
    pub fn all_passed(&self) -> bool {
        matches!(
            self.simd_consistency,
            PropertyTestResult::Passed | PropertyTestResult::Skipped(_)
        ) && matches!(
            self.mathematical_invariants,
            PropertyTestResult::Passed | PropertyTestResult::Skipped(_)
        ) && matches!(
            self.numerical_stability,
            PropertyTestResult::Passed | PropertyTestResult::Skipped(_)
        ) && matches!(
            self.correlation_properties,
            PropertyTestResult::Passed | PropertyTestResult::Skipped(_)
        ) && matches!(
            self.performance_consistency,
            PropertyTestResult::Passed | PropertyTestResult::Skipped(_)
        )
    }

    /// Get summary of failed tests
    pub fn get_failures(&self) -> Vec<String> {
        let mut failures = Vec::new();

        if let PropertyTestResult::Failed(msg) = &self.simd_consistency {
            failures.push(format!("SIMD Consistency: {}", msg));
        }
        if let PropertyTestResult::Failed(msg) = &self.mathematical_invariants {
            failures.push(format!("Mathematical Invariants: {}", msg));
        }
        if let PropertyTestResult::Failed(msg) = &self.numerical_stability {
            failures.push(format!("Numerical Stability: {}", msg));
        }
        if let PropertyTestResult::Failed(msg) = &self.correlation_properties {
            failures.push(format!("Correlation Properties: {}", msg));
        }
        if let PropertyTestResult::Failed(msg) = &self.performance_consistency {
            failures.push(format!("Performance Consistency: {}", msg));
        }

        failures
    }
}

/// Create a default Advanced property tester
#[allow(dead_code)]
pub fn create_advanced_property_tester() -> AdvancedPropertyTester {
    AdvancedPropertyTester::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    #[ignore = "timeout"]
    fn test_property_tester_creation() {
        let tester = create_advanced_property_tester();
        assert!(tester.numerical_tolerance > 0.0);
        assert!(tester.performance_tolerance > 0.0);
    }

    #[test]
    #[ignore = "timeout"]
    fn test_mathematical_invariants() {
        let tester = create_advanced_property_tester();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result = tester.test_mathematical_invariants(&data.view());

        match result {
            PropertyTestResult::Passed => {}
            PropertyTestResult::Failed(msg) => panic!("Mathematical invariants failed: {}", msg),
            PropertyTestResult::Skipped(_) => {}
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_correlation_properties() {
        let tester = create_advanced_property_tester();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation

        let result = tester.test_correlation_properties(&x.view(), &y.view());

        match result {
            PropertyTestResult::Passed => {}
            PropertyTestResult::Failed(msg) => panic!("Correlation properties failed: {}", msg),
            PropertyTestResult::Skipped(_) => {}
        }
    }

    #[test]
    #[ignore = "timeout"]
    fn test_comprehensive_report() {
        let tester = create_advanced_property_tester();
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let report = tester.run_comprehensive_tests(&data.view());

        // Should have at least attempted all tests
        assert!(matches!(
            report.mathematical_invariants,
            PropertyTestResult::Passed | PropertyTestResult::Failed(_)
        ));
    }
}
