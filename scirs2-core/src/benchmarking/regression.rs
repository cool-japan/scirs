//! # Performance Regression Testing
//!
//! This module provides automated regression testing to detect performance
//! degradation over time and across different versions of the codebase.

use crate::benchmarking::{BenchmarkResult, BenchmarkRunner};
use crate::error::{CoreError, CoreResult, ErrorContext};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Performance regression detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Threshold for considering a regression (e.g., 1.1 = 10% slower)
    pub regression_threshold: f64,
    /// Minimum number of historical results needed for comparison
    pub min_historical_samples: usize,
    /// Statistical confidence level for regression detection
    pub confidence_level: f64,
    /// Enable automatic baseline updates
    pub auto_update_baseline: bool,
    /// Directory to store historical results
    pub results_directory: PathBuf,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            regression_threshold: 1.1, // 10% slower
            min_historical_samples: 5,
            confidence_level: 0.95,
            auto_update_baseline: false,
            results_directory: PathBuf::from("benchmark_results"),
        }
    }
}

impl RegressionConfig {
    /// Create a new regression configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the regression threshold
    pub fn with_regression_threshold(mut self, threshold: f64) -> Self {
        self.regression_threshold = threshold;
        self
    }

    /// Set the minimum historical samples
    pub fn with_min_historical_samples(mut self, samples: usize) -> Self {
        self.min_historical_samples = samples;
        self
    }

    /// Set the confidence level
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Enable automatic baseline updates
    pub fn with_auto_update_baseline(mut self, enable: bool) -> Self {
        self.auto_update_baseline = enable;
        self
    }

    /// Set the results directory
    pub fn with_results_directory<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.results_directory = dir.as_ref().to_path_buf();
        self
    }
}

/// Historical benchmark result for regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalResult {
    /// Timestamp when benchmark was run
    pub timestamp: u64,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Version string
    pub version: Option<String>,
    /// Benchmark name
    pub benchmark_name: String,
    /// Mean execution time in nanoseconds
    pub mean_execution_time_nanos: u64,
    /// Standard deviation in nanoseconds
    pub std_dev_nanos: u64,
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
    /// Memory usage in bytes
    pub mean_memory_usage: usize,
    /// Sample count
    pub sample_count: usize,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl HistoricalResult {
    /// Create from a benchmark result
    pub fn from_benchmark_result(result: &BenchmarkResult) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            timestamp,
            commit_hash: Self::get_git_commit_hash(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            benchmark_name: result.name.clone(),
            mean_execution_time_nanos: result.statistics.mean_execution_time.as_nanos() as u64,
            std_dev_nanos: result.statistics.std_dev_execution_time.as_nanos() as u64,
            coefficient_of_variation: result.statistics.coefficient_of_variation,
            mean_memory_usage: result.statistics.mean_memory_usage,
            sample_count: result.statistics.sample_count,
            metadata: HashMap::new(),
        }
    }

    /// Get the current git commit hash
    fn get_git_commit_hash() -> Option<String> {
        // In a real implementation, this would execute `git rev-parse HEAD`
        // For now, we'll return None
        None
    }

    /// Get execution time as Duration
    pub fn execution_time(&self) -> Duration {
        Duration::from_nanos(self.mean_execution_time_nanos)
    }

    /// Get standard deviation as Duration
    pub fn std_dev(&self) -> Duration {
        Duration::from_nanos(self.std_dev_nanos)
    }
}

/// Regression detection result
#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    /// Benchmark name
    pub benchmark_name: String,
    /// Current result
    pub current_result: HistoricalResult,
    /// Baseline for comparison
    pub baseline: HistoricalResult,
    /// Historical results used for analysis
    pub historical_results: Vec<HistoricalResult>,
    /// Whether a regression was detected
    pub regression_detected: bool,
    /// Performance ratio (current / baseline)
    pub performance_ratio: f64,
    /// Statistical significance of the difference
    pub statistical_significance: f64,
    /// Trend analysis
    pub trend: PerformanceTrend,
    /// Confidence in the analysis
    pub confidence: f64,
}

/// Performance trend analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTrend {
    /// Performance is improving over time
    Improving,
    /// Performance is stable
    Stable,
    /// Performance is degrading
    Degrading,
    /// Insufficient data for trend analysis
    Unknown,
}

/// Regression detector for performance benchmarks
pub struct RegressionDetector {
    config: RegressionConfig,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new(config: RegressionConfig) -> Self {
        Self { config }
    }

    /// Analyze a benchmark result for regressions
    pub fn analyze_regression(&self, result: &BenchmarkResult) -> CoreResult<RegressionAnalysis> {
        let current_result = HistoricalResult::from_benchmark_result(result);

        // Load historical results
        let historical_results = self.load_historical_results(&result.name)?;

        if historical_results.len() < self.config.min_historical_samples {
            return Ok(RegressionAnalysis {
                benchmark_name: result.name.clone(),
                current_result: current_result.clone(),
                baseline: current_result.clone(),
                historical_results,
                regression_detected: false,
                performance_ratio: 1.0,
                statistical_significance: 0.0,
                trend: PerformanceTrend::Unknown,
                confidence: 0.0,
            });
        }

        // Calculate baseline from historical results
        let baseline = self.calculate_baseline(&historical_results)?;

        // Detect regression
        let performance_ratio = current_result.mean_execution_time_nanos as f64
            / baseline.mean_execution_time_nanos as f64;

        let regression_detected = performance_ratio > self.config.regression_threshold;

        // Calculate statistical significance
        let statistical_significance =
            self.calculate_statistical_significance(&current_result, &historical_results)?;

        // Analyze trend
        let trend = self.analyze_trend(&historical_results)?;

        // Calculate confidence based on sample size and variance
        let confidence = self.calculate_confidence(&historical_results, &current_result)?;

        Ok(RegressionAnalysis {
            benchmark_name: result.name.clone(),
            current_result,
            baseline,
            historical_results,
            regression_detected,
            performance_ratio,
            statistical_significance,
            trend,
            confidence,
        })
    }

    /// Run regression analysis on multiple benchmarks
    pub fn analyze_multiple_regressions(
        &self,
        results: &[BenchmarkResult],
    ) -> CoreResult<Vec<RegressionAnalysis>> {
        let mut analyses = Vec::new();

        for result in results {
            let analysis = self.analyze_regression(result)?;
            analyses.push(analysis);
        }

        Ok(analyses)
    }

    /// Store a benchmark result for future regression analysis
    pub fn store_result(&self, result: &BenchmarkResult) -> CoreResult<()> {
        let historical_result = HistoricalResult::from_benchmark_result(result);

        // Ensure results directory exists
        fs::create_dir_all(&self.config.results_directory).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to create results directory: {}",
                e
            )))
        })?;

        // Load existing results
        let mut historical_results = self.load_historical_results(&result.name)?;

        // Add new result
        historical_results.push(historical_result);

        // Sort by timestamp
        historical_results.sort_by_key(|r| r.timestamp);

        // Limit history size (keep last 1000 results)
        if historical_results.len() > 1000 {
            historical_results.drain(0..historical_results.len() - 1000);
        }

        // Save results
        let file_path = self.get_results_file_path(&result.name);
        let serialized = serde_json::to_string_pretty(&historical_results).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to serialize results: {}",
                e
            )))
        })?;

        fs::write(&file_path, serialized).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to write results file: {}",
                e
            )))
        })?;

        Ok(())
    }

    /// Load historical results for a benchmark
    fn load_historical_results(&self, benchmark_name: &str) -> CoreResult<Vec<HistoricalResult>> {
        let file_path = self.get_results_file_path(benchmark_name);

        if !file_path.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&file_path).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to read results file: {}",
                e
            )))
        })?;

        let results: Vec<HistoricalResult> = serde_json::from_str(&content).map_err(|e| {
            CoreError::IoError(ErrorContext::new(format!(
                "Failed to parse results file: {}",
                e
            )))
        })?;

        Ok(results)
    }

    /// Calculate baseline performance from historical results
    fn calculate_baseline(
        &self,
        historical_results: &[HistoricalResult],
    ) -> CoreResult<HistoricalResult> {
        if historical_results.is_empty() {
            return Err(CoreError::ValidationError(crate::error::ErrorContext::new(
                "No historical results for baseline calculation",
            )));
        }

        // Use the median of recent results as baseline
        let recent_count = (historical_results.len() / 3).max(self.config.min_historical_samples);
        let recent_results = &historical_results[historical_results.len() - recent_count..];

        let mut execution_times: Vec<u64> = recent_results
            .iter()
            .map(|r| r.mean_execution_time_nanos)
            .collect();
        execution_times.sort();

        let median_time = if execution_times.len() % 2 == 0 {
            let mid = execution_times.len() / 2;
            (execution_times[mid - 1] + execution_times[mid]) / 2
        } else {
            execution_times[execution_times.len() / 2]
        };

        // Create a synthetic baseline result
        let mut baseline = recent_results[recent_results.len() / 2].clone();
        baseline.mean_execution_time_nanos = median_time;

        Ok(baseline)
    }

    /// Calculate statistical significance of performance difference
    fn calculate_statistical_significance(
        &self,
        current: &HistoricalResult,
        historical: &[HistoricalResult],
    ) -> CoreResult<f64> {
        if historical.len() < 2 {
            return Ok(0.0);
        }

        // Calculate mean and standard deviation of historical results
        let historical_times: Vec<f64> = historical
            .iter()
            .map(|r| r.mean_execution_time_nanos as f64)
            .collect();

        let historical_mean = historical_times.iter().sum::<f64>() / historical_times.len() as f64;
        let historical_variance = historical_times
            .iter()
            .map(|&x| (x - historical_mean).powi(2))
            .sum::<f64>()
            / (historical_times.len() - 1) as f64;
        let historical_std = historical_variance.sqrt();

        // Calculate z-score
        let current_time = current.mean_execution_time_nanos as f64;
        let z_score =
            (current_time - historical_mean) / (historical_std / (historical.len() as f64).sqrt());

        // Convert to p-value (simplified normal distribution approximation)
        let p_value = if z_score > 0.0 {
            0.5 * (1.0 - erf(z_score / std::f64::consts::SQRT_2))
        } else {
            0.5 * (1.0 + erf(-z_score / std::f64::consts::SQRT_2))
        };

        Ok(1.0 - p_value) // Return significance level
    }

    /// Analyze performance trend over time
    fn analyze_trend(
        &self,
        historical_results: &[HistoricalResult],
    ) -> CoreResult<PerformanceTrend> {
        if historical_results.len() < 5 {
            return Ok(PerformanceTrend::Unknown);
        }

        // Calculate linear regression slope
        let n = historical_results.len() as f64;
        let sum_x: f64 = (0..historical_results.len()).map(|i| i as f64).sum();
        let sum_y: f64 = historical_results
            .iter()
            .map(|r| r.mean_execution_time_nanos as f64)
            .sum();
        let sum_xy: f64 = historical_results
            .iter()
            .enumerate()
            .map(|(i, r)| i as f64 * r.mean_execution_time_nanos as f64)
            .sum();
        let sum_x_sq: f64 = (0..historical_results.len())
            .map(|i| (i as f64).powi(2))
            .sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sq - sum_x.powi(2));

        // Classify trend based on slope
        let relative_slope = slope / (sum_y / n); // Normalize by mean

        if relative_slope > 0.01 {
            Ok(PerformanceTrend::Degrading)
        } else if relative_slope < -0.01 {
            Ok(PerformanceTrend::Improving)
        } else {
            Ok(PerformanceTrend::Stable)
        }
    }

    /// Calculate confidence in the regression analysis
    fn calculate_confidence(
        &self,
        historical_results: &[HistoricalResult],
        current: &HistoricalResult,
    ) -> CoreResult<f64> {
        let sample_size_factor = (historical_results.len() as f64 / 10.0).min(1.0);
        let variance_factor = if current.coefficient_of_variation < 0.1 {
            1.0
        } else {
            (0.1 / current.coefficient_of_variation).min(1.0)
        };

        Ok(sample_size_factor * variance_factor)
    }

    /// Get the file path for storing results
    fn get_results_file_path(&self, benchmark_name: &str) -> PathBuf {
        let safe_name = benchmark_name.replace(|c: char| !c.is_alphanumeric(), "_");
        self.config
            .results_directory
            .join(format!("{}.json", safe_name))
    }
}

/// Regression testing utilities
pub struct RegressionTestUtils;

impl RegressionTestUtils {
    /// Run a complete regression test suite
    pub fn run_regression_suite(
        benchmark_runner: &BenchmarkRunner,
        detector: &RegressionDetector,
        benchmark_names: &[&str],
    ) -> CoreResult<Vec<RegressionAnalysis>> {
        let mut analyses = Vec::new();

        for &name in benchmark_names {
            // Run benchmark (this is simplified - in practice you'd have the actual benchmark functions)
            let result = benchmark_runner.run(name, || {
                // Placeholder benchmark - replace with actual benchmark
                std::thread::sleep(Duration::from_micros(100));
                Ok(())
            })?;

            // Store result for future analysis
            detector.store_result(&result)?;

            // Analyze for regressions
            let analysis = detector.analyze_regression(&result)?;
            analyses.push(analysis);
        }

        Ok(analyses)
    }

    /// Generate a regression report
    pub fn generate_report(analyses: &[RegressionAnalysis]) -> String {
        let mut report = String::new();

        report.push_str("# Performance Regression Report\n\n");

        let regressions: Vec<_> = analyses.iter().filter(|a| a.regression_detected).collect();

        if regressions.is_empty() {
            report.push_str("✅ No performance regressions detected.\n\n");
        } else {
            report.push_str(&format!(
                "⚠️ {} performance regression(s) detected:\n\n",
                regressions.len()
            ));

            for regression in &regressions {
                report.push_str(&format!(
                    "- **{}**: {:.1}% slower (ratio: {:.3}, confidence: {:.0}%)\n",
                    regression.benchmark_name,
                    (regression.performance_ratio - 1.0) * 100.0,
                    regression.performance_ratio,
                    regression.confidence * 100.0
                ));
            }
            report.push('\n');
        }

        // Summary statistics
        report.push_str("## Summary\n\n");
        report.push_str(&format!("- Total benchmarks: {}\n", analyses.len()));
        report.push_str(&format!("- Regressions detected: {}\n", regressions.len()));

        let improving = analyses
            .iter()
            .filter(|a| a.trend == PerformanceTrend::Improving)
            .count();
        let stable = analyses
            .iter()
            .filter(|a| a.trend == PerformanceTrend::Stable)
            .count();
        let degrading = analyses
            .iter()
            .filter(|a| a.trend == PerformanceTrend::Degrading)
            .count();

        report.push_str(&format!("- Improving trends: {}\n", improving));
        report.push_str(&format!("- Stable trends: {}\n", stable));
        report.push_str(&format!("- Degrading trends: {}\n", degrading));

        report
    }
}

// Simplified error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_regression_config() {
        let config = RegressionConfig::new()
            .with_regression_threshold(1.2)
            .with_min_historical_samples(10)
            .with_confidence_level(0.99)
            .with_auto_update_baseline(true);

        assert_eq!(config.regression_threshold, 1.2);
        assert_eq!(config.min_historical_samples, 10);
        assert_eq!(config.confidence_level, 0.99);
        assert!(config.auto_update_baseline);
    }

    #[test]
    fn test_historical_result() {
        let benchmark_config = crate::benchmarking::BenchmarkConfig::default();
        let mut result = BenchmarkResult::new("test_benchmark".to_string(), benchmark_config);
        result.add_measurement(crate::benchmarking::BenchmarkMeasurement::new(
            Duration::from_millis(100),
        ));
        result.finalize().unwrap();

        let historical = HistoricalResult::from_benchmark_result(&result);

        assert_eq!(historical.benchmark_name, "test_benchmark");
        assert!(historical.mean_execution_time_nanos > 0);
        assert_eq!(historical.sample_count, 1);
    }

    #[test]
    fn test_regression_detector() {
        let temp_dir = TempDir::new().unwrap();
        let config = RegressionConfig::new()
            .with_results_directory(temp_dir.path())
            .with_min_historical_samples(1);

        let detector = RegressionDetector::new(config);

        // Create a test benchmark result
        let benchmark_config = crate::benchmarking::BenchmarkConfig::default();
        let mut result = BenchmarkResult::new("test_regression".to_string(), benchmark_config);
        result.add_measurement(crate::benchmarking::BenchmarkMeasurement::new(
            Duration::from_millis(100),
        ));
        result.finalize().unwrap();

        // Store and analyze
        detector.store_result(&result).unwrap();
        let analysis = detector.analyze_regression(&result).unwrap();

        assert_eq!(analysis.benchmark_name, "test_regression");
        assert!(!analysis.regression_detected); // First result can't be a regression
    }

    #[test]
    fn test_performance_trend() {
        assert_eq!(PerformanceTrend::Improving, PerformanceTrend::Improving);
        assert_ne!(PerformanceTrend::Improving, PerformanceTrend::Degrading);
    }

    #[test]
    fn test_erf_function() {
        // Test a few known values
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
        assert!((erf(1.0) - 0.8427007929).abs() < 1e-6);
        assert!((erf(-1.0) + 0.8427007929).abs() < 1e-6);
    }
}
