//! Core data types and structures for regression testing
//!
//! This module provides all the data structures used throughout the regression testing
//! framework, including performance metrics, analysis results, and statistical types.

use crate::benchmarking::regression_tester::config::TestEnvironment;
use crate::error::Result;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Performance metrics for regression testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics<A: Float> {
    /// Execution time metrics
    pub timing: TimingMetrics,
    /// Memory usage metrics
    pub memory: MemoryMetrics,
    /// Computational efficiency metrics
    pub efficiency: EfficiencyMetrics<A>,
    /// Convergence metrics
    pub convergence: ConvergenceMetrics<A>,
    /// Custom metrics
    pub custom: HashMap<String, f64>,
}

/// Timing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetrics {
    /// Mean execution time (nanoseconds)
    pub mean_time_ns: u64,
    /// Standard deviation of execution time
    pub std_time_ns: u64,
    /// Median execution time
    pub median_time_ns: u64,
    /// 95th percentile execution time
    pub p95_time_ns: u64,
    /// 99th percentile execution time
    pub p99_time_ns: u64,
    /// Minimum execution time
    pub min_time_ns: u64,
    /// Maximum execution time
    pub max_time_ns: u64,
}

/// Memory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Average memory usage (bytes)
    pub avg_memory_bytes: usize,
    /// Memory allocation count
    pub allocation_count: usize,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
    /// Memory efficiency score
    pub efficiency_score: f64,
}

/// Efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics<A: Float> {
    /// FLOPS achieved
    pub flops: f64,
    /// Arithmetic intensity
    pub arithmetic_intensity: f64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Overall efficiency score
    pub efficiency_score: f64,
    /// Custom efficiency metrics
    pub custom_metrics: HashMap<String, A>,
}

/// Convergence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics<A: Float> {
    /// Final objective value
    pub final_objective: A,
    /// Convergence rate
    pub convergence_rate: f64,
    /// Iterations to convergence
    pub iterations_to_convergence: Option<usize>,
    /// Convergence quality score
    pub quality_score: f64,
    /// Stability metrics
    pub stability_score: f64,
}

/// Individual performance record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecord<A: Float> {
    /// Timestamp of the record
    pub timestamp: u64,
    /// Git commit hash (if available)
    pub commit_hash: Option<String>,
    /// Branch name
    pub branch: Option<String>,
    /// Test environment information
    pub environment: TestEnvironment,
    /// Performance metrics
    pub metrics: PerformanceMetrics<A>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Database metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseMetadata {
    /// Database version
    pub version: String,
    /// Creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub last_updated: u64,
    /// Total number of records
    pub total_records: usize,
}

/// Performance baseline for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline<A: Float> {
    /// Baseline name/identifier
    pub name: String,
    /// Statistical summary of baseline performance
    pub baseline_stats: BaselineStatistics<A>,
    /// Confidence intervals
    pub confidence_intervals: ConfidenceIntervals,
    /// Sample count used for baseline
    pub sample_count: usize,
    /// Baseline creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
}

/// Statistical summary of baseline performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStatistics<A: Float> {
    /// Timing statistics
    pub timing: TimingStatistics,
    /// Memory statistics
    pub memory: MemoryStatistics,
    /// Efficiency statistics
    pub efficiency: EfficiencyStatistics<A>,
    /// Convergence statistics
    pub convergence: ConvergenceStatistics<A>,
}

/// Timing statistics for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStatistics {
    /// Mean execution time
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Median
    pub median: f64,
    /// Interquartile range
    pub iqr: f64,
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
}

/// Memory statistics for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Mean memory usage
    pub mean_memory: f64,
    /// Standard deviation
    pub std_dev_memory: f64,
    /// Peak memory percentiles
    pub peak_memory_percentiles: HashMap<String, f64>,
    /// Fragmentation statistics
    pub fragmentation_stats: FragmentationStatistics,
}

/// Fragmentation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationStatistics {
    /// Mean fragmentation ratio
    pub mean_ratio: f64,
    /// Standard deviation
    pub std_dev_ratio: f64,
    /// Trend analysis
    pub trend: f64,
}

/// Efficiency statistics for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyStatistics<A: Float> {
    /// Mean FLOPS
    pub mean_flops: f64,
    /// FLOPS variability
    pub flops_cv: f64,
    /// Mean efficiency score
    pub mean_efficiency: f64,
    /// Custom efficiency metrics
    pub custom_efficiency: HashMap<String, A>,
}

/// Convergence statistics for baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceStatistics<A: Float> {
    /// Mean final objective
    pub mean_objective: A,
    /// Objective standard deviation
    pub std_objective: A,
    /// Mean convergence rate
    pub mean_convergence_rate: f64,
    /// Convergence consistency
    pub convergence_consistency: f64,
}

/// Confidence intervals for baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// 95% confidence intervals for timing
    pub timing_ci_95: (f64, f64),
    /// 95% confidence intervals for memory
    pub memory_ci_95: (f64, f64),
    /// 99% confidence intervals for timing
    pub timing_ci_99: (f64, f64),
    /// 99% confidence intervals for memory
    pub memory_ci_99: (f64, f64),
}

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult<A: Float> {
    /// Test identifier
    pub test_id: String,
    /// Regression detected
    pub regression_detected: bool,
    /// Regression severity (0.0 to 1.0)
    pub severity: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Performance change percentage
    pub performance_change_percent: f64,
    /// Memory change percentage
    pub memory_change_percent: f64,
    /// Affected metrics
    pub affected_metrics: Vec<String>,
    /// Statistical test results
    pub statistical_tests: Vec<StatisticalTestResult>,
    /// Detailed analysis
    pub analysis: RegressionAnalysis<A>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    /// Test name
    pub test_name: String,
    /// Test statistic value
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: Option<usize>,
    /// Test conclusion
    pub conclusion: String,
}

/// Detailed regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis<A: Float> {
    /// Trend analysis
    pub trend_analysis: TrendAnalysis,
    /// Change point analysis
    pub change_point_analysis: ChangePointAnalysis,
    /// Outlier analysis
    pub outlier_analysis: OutlierAnalysis<A>,
    /// Root cause analysis hints
    pub root_cause_hints: Vec<String>,
}

/// Trend analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend magnitude
    pub magnitude: f64,
    /// Trend significance
    pub significance: f64,
    /// Trend starting point
    pub start_point: Option<usize>,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Performance is improving
    Improving,
    /// Performance is stable
    Stable,
    /// Performance is degrading
    Degrading,
    /// Performance is volatile/inconsistent
    Volatile,
}

/// Change point analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePointAnalysis {
    /// Change points detected
    pub change_points: Vec<usize>,
    /// Change magnitudes
    pub magnitudes: Vec<f64>,
    /// Confidence levels
    pub confidences: Vec<f64>,
}

/// Outlier analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAnalysis<A: Float> {
    /// Outlier indices
    pub outlier_indices: Vec<usize>,
    /// Outlier scores
    pub outlier_scores: Vec<A>,
    /// Outlier types
    pub outlier_types: Vec<OutlierType>,
}

/// Types of outliers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierType {
    /// Single extreme value
    Point,
    /// Shift in distribution
    Shift,
    /// Trend change
    Trend,
    /// Increased variance
    Variance,
}

/// Regression detection algorithm trait
pub trait RegressionDetector<A: Float>: Debug {
    /// Detect regression in performance data
    fn detect_regression(
        &self,
        baseline: &PerformanceBaseline<A>,
        current_metrics: &PerformanceMetrics<A>,
        history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>>;

    /// Get detector name
    fn name(&self) -> &str;

    /// Get detector configuration
    fn config(&self) -> HashMap<String, String>;
}

/// Statistical analysis trait
pub trait StatisticalAnalyzer<A: Float>: Debug {
    /// Perform statistical analysis on performance data
    fn analyze(
        &self,
        data: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<StatisticalAnalysisResult<A>>;

    /// Get analyzer name
    fn name(&self) -> &str;
}

/// Statistical analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisResult<A: Float> {
    /// Analysis summary
    pub summary: String,
    /// Statistical tests performed
    pub tests: Vec<StatisticalTestResult>,
    /// Detected patterns
    pub patterns: Vec<String>,
    /// Anomalies detected
    pub anomalies: Vec<A>,
}

impl<A: Float> Default for PerformanceMetrics<A> {
    fn default() -> Self {
        Self {
            timing: TimingMetrics::default(),
            memory: MemoryMetrics::default(),
            efficiency: EfficiencyMetrics::default(),
            convergence: ConvergenceMetrics::default(),
            custom: HashMap::new(),
        }
    }
}

impl Default for TimingMetrics {
    fn default() -> Self {
        Self {
            mean_time_ns: 0,
            std_time_ns: 0,
            median_time_ns: 0,
            p95_time_ns: 0,
            p99_time_ns: 0,
            min_time_ns: 0,
            max_time_ns: 0,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            peak_memory_bytes: 0,
            avg_memory_bytes: 0,
            allocation_count: 0,
            fragmentation_ratio: 0.0,
            efficiency_score: 0.0,
        }
    }
}

impl<A: Float> Default for EfficiencyMetrics<A> {
    fn default() -> Self {
        Self {
            flops: 0.0,
            arithmetic_intensity: 0.0,
            cache_hit_ratio: 0.0,
            cpu_utilization: 0.0,
            efficiency_score: 0.0,
            custom_metrics: HashMap::new(),
        }
    }
}

impl<A: Float> Default for ConvergenceMetrics<A> {
    fn default() -> Self {
        Self {
            final_objective: A::zero(),
            convergence_rate: 0.0,
            iterations_to_convergence: None,
            quality_score: 0.0,
            stability_score: 0.0,
        }
    }
}

impl Default for DatabaseMetadata {
    fn default() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            version: "1.0.0".to_string(),
            created_at: now,
            last_updated: now,
            total_records: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_default() {
        let metrics: PerformanceMetrics<f64> = PerformanceMetrics::default();
        assert_eq!(metrics.timing.mean_time_ns, 0);
        assert_eq!(metrics.memory.peak_memory_bytes, 0);
        assert_eq!(metrics.efficiency.flops, 0.0);
        assert_eq!(metrics.convergence.final_objective, 0.0);
        assert!(metrics.custom.is_empty());
    }

    #[test]
    fn test_database_metadata_default() {
        let metadata = DatabaseMetadata::default();
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.total_records, 0);
        assert!(metadata.created_at > 0);
        assert!(metadata.last_updated > 0);
    }

    #[test]
    fn test_trend_direction_variants() {
        assert!(matches!(TrendDirection::Improving, TrendDirection::Improving));
        assert!(matches!(TrendDirection::Stable, TrendDirection::Stable));
        assert!(matches!(TrendDirection::Degrading, TrendDirection::Degrading));
        assert!(matches!(TrendDirection::Volatile, TrendDirection::Volatile));
    }

    #[test]
    fn test_outlier_type_variants() {
        assert!(matches!(OutlierType::Point, OutlierType::Point));
        assert!(matches!(OutlierType::Shift, OutlierType::Shift));
        assert!(matches!(OutlierType::Trend, OutlierType::Trend));
        assert!(matches!(OutlierType::Variance, OutlierType::Variance));
    }
}