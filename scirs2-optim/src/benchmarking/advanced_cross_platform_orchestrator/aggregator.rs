//! Result aggregation and analysis for cross-platform testing
//!
//! This module aggregates test results across platforms and provides
//! comprehensive analysis and reporting capabilities.

use crate::error::Result;
use std::collections::HashMap;

use super::types::*;

/// Result aggregator for cross-platform testing
#[derive(Debug)]
pub struct ResultAggregator {
    results: Vec<TestResult>,
    compatibility_matrix: CompatibilityMatrix,
}

impl ResultAggregator {
    /// Create new result aggregator
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            compatibility_matrix: CompatibilityMatrix::default(),
        }
    }

    /// Aggregate test results
    pub fn aggregate_results(&mut self, results: Vec<TestResult>) -> Result<()> {
        self.results.extend(results);
        self.update_compatibility_matrix()?;
        Ok(())
    }

    /// Update compatibility matrix
    fn update_compatibility_matrix(&mut self) -> Result<()> {
        let mut platform_results: HashMap<PlatformTarget, Vec<TestResult>> = HashMap::new();
        let mut performance_comparison: HashMap<PlatformTarget, PerformanceMetrics> = HashMap::new();

        // Group results by platform
        for result in &self.results {
            // Extract platform from test name (simplified)
            if let Some(platform) = self.extract_platform_from_test_name(&result.test_name) {
                platform_results.entry(platform.clone()).or_insert_with(Vec::new).push(result.clone());

                // Update performance comparison
                performance_comparison.insert(platform, result.performance_metrics.clone());
            }
        }

        // Calculate overall compatibility score
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| matches!(r.status, TestStatus::Passed)).count();
        let compatibility_score = if total_tests > 0 {
            passed_tests as f64 / total_tests as f64 * 100.0
        } else {
            0.0
        };

        self.compatibility_matrix = CompatibilityMatrix {
            platform_results,
            feature_compatibility: HashMap::new(),
            performance_comparison,
            issues: vec![],
            compatibility_score,
        };

        Ok(())
    }

    /// Extract platform from test name
    fn extract_platform_from_test_name(&self, test_name: &str) -> Option<PlatformTarget> {
        if test_name.contains("linux") {
            Some(PlatformTarget::LinuxX86_64)
        } else if test_name.contains("windows") {
            Some(PlatformTarget::WindowsX86_64)
        } else if test_name.contains("macos") {
            Some(PlatformTarget::MacOSX86_64)
        } else {
            None
        }
    }

    /// Get compatibility matrix
    pub fn get_compatibility_matrix(&self) -> CompatibilityMatrix {
        self.compatibility_matrix.clone()
    }

    /// Get aggregated results
    pub fn get_results(&self) -> &[TestResult] {
        &self.results
    }

    /// Generate summary statistics
    pub fn get_summary_statistics(&self) -> AggregationStatistics {
        let total_tests = self.results.len();
        let passed_tests = self.results.iter().filter(|r| matches!(r.status, TestStatus::Passed)).count();
        let failed_tests = self.results.iter().filter(|r| matches!(r.status, TestStatus::Failed)).count();
        let skipped_tests = self.results.iter().filter(|r| matches!(r.status, TestStatus::Skipped)).count();

        let mut platform_counts = HashMap::new();
        for result in &self.results {
            if let Some(platform) = self.extract_platform_from_test_name(&result.test_name) {
                *platform_counts.entry(platform).or_insert(0) += 1;
            }
        }

        AggregationStatistics {
            total_tests,
            passed_tests,
            failed_tests,
            skipped_tests,
            platform_counts,
            overall_success_rate: if total_tests > 0 { passed_tests as f64 / total_tests as f64 } else { 0.0 },
        }
    }
}

/// Aggregation statistics
#[derive(Debug, Clone)]
pub struct AggregationStatistics {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub skipped_tests: usize,
    pub platform_counts: HashMap<PlatformTarget, usize>,
    pub overall_success_rate: f64,
}

impl Default for ResultAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_result_aggregation() {
        let mut aggregator = ResultAggregator::new();

        let test_results = vec![
            TestResult {
                test_name: "test_linux_build".to_string(),
                status: TestStatus::Passed,
                execution_time: Duration::from_secs(10),
                performance_metrics: PerformanceMetrics::default(),
                error_message: None,
                platform_details: HashMap::new(),
                numerical_results: None,
            },
            TestResult {
                test_name: "test_windows_build".to_string(),
                status: TestStatus::Failed,
                execution_time: Duration::from_secs(15),
                performance_metrics: PerformanceMetrics::default(),
                error_message: Some("Build failed".to_string()),
                platform_details: HashMap::new(),
                numerical_results: None,
            },
        ];

        let result = aggregator.aggregate_results(test_results);
        assert!(result.is_ok());

        let stats = aggregator.get_summary_statistics();
        assert_eq!(stats.total_tests, 2);
        assert_eq!(stats.passed_tests, 1);
        assert_eq!(stats.failed_tests, 1);
        assert_eq!(stats.overall_success_rate, 0.5);
    }
}