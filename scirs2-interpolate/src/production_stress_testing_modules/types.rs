//! Core type definitions for production stress testing
//!
//! This module contains all the fundamental types used throughout the stress testing system.

use crate::traits::InterpolationFloat;
use std::collections::HashMap;
use std::time::Duration;

/// Production stress testing suite
pub struct ProductionStressTester<T: InterpolationFloat> {
    /// Configuration for stress tests
    pub config: StressTestConfig,
    /// Results from completed tests
    pub results: Vec<StressTestResult>,
    /// Performance baseline for comparison
    pub baseline_performance: Option<BaselinePerformance>,
    /// Error tracking
    #[allow(dead_code)]
    pub error_patterns: HashMap<String, usize>,
    /// Phantom data for type parameter
    pub _phantom: std::marker::PhantomData<T>,
}

/// Configuration for stress testing
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Maximum data size for stress testing
    pub max_data_size: usize,
    /// Number of stress iterations
    pub stress_iterations: usize,
    /// Timeout for individual tests (seconds)
    pub test_timeout: u64,
    /// Memory limit for testing (bytes)
    pub memory_limit: Option<u64>,
    /// Enable extreme edge case testing
    pub test_extreme_cases: bool,
    /// Performance degradation threshold (factor)
    pub max_performance_degradation: f64,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            max_data_size: 1_000_000,
            stress_iterations: 100,
            test_timeout: 300,                          // 5 minutes
            memory_limit: Some(8 * 1024 * 1024 * 1024), // 8GB
            test_extreme_cases: true,
            max_performance_degradation: 10.0, // 10x slower max
        }
    }
}

/// Result of a stress test
#[derive(Debug, Clone)]
pub struct StressTestResult {
    /// Test name
    pub test_name: String,
    /// Test category
    pub category: StressTestCategory,
    /// Input characteristics description
    pub input_characteristics: String,
    /// Test status
    pub status: TestStatus,
    /// Execution time
    pub execution_time: Duration,
    /// Performance metrics
    pub performance: StressPerformanceMetrics,
    /// Error information (if any)
    pub error_info: Option<ErrorInfo>,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    /// Test duration
    pub duration: Duration,
    /// Issues detected
    pub issues: Vec<StressTestIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Categories of stress tests
#[derive(Debug, Clone)]
pub enum StressTestCategory {
    /// Extreme data size testing
    ExtremeDataSize,
    /// Pathological data patterns
    PathologicalData,
    /// Numerical edge cases
    NumericalEdgeCases,
    /// Memory pressure testing
    MemoryPressure,
    /// Error handling validation
    ErrorHandling,
    /// Performance under stress
    PerformanceStress,
    /// Concurrent access testing
    ConcurrentAccess,
    /// Numerical stability testing
    NumericalStability,
    /// Resource exhaustion testing
    ResourceExhaustion,
}

/// Test execution status
#[derive(Debug, Clone, PartialEq)]
pub enum TestStatus {
    /// Test passed successfully
    Passed,
    /// Test passed with warnings
    PassedWithWarnings,
    /// Test failed
    Failed,
    /// Test timed out
    TimedOut,
    /// Test was skipped
    Skipped,
    /// Test encountered an error
    Error,
}

/// Performance metrics for stress tests
#[derive(Debug, Clone)]
pub struct StressPerformanceMetrics {
    /// Mean execution time
    pub mean_execution_time: Duration,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Minimum execution time
    pub min_execution_time: Duration,
    /// Performance degradation factor vs baseline
    pub degradation_factor: Option<f64>,
    /// Throughput (operations per second)
    pub throughput: Option<f64>,
    /// Memory efficiency
    pub memory_efficiency: Option<f64>,
}

/// Error information for failed tests
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    /// Error type
    pub error_type: String,
    /// Error message
    pub error_message: String,
    /// Error occurred in which iteration
    pub iteration: Option<usize>,
    /// Data size when error occurred
    pub data_size: Option<usize>,
    /// Recovery attempted
    pub recovery_attempted: bool,
    /// Recovery successful
    pub recovery_successful: bool,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryUsageStats {
    /// Peak memory usage
    pub peak_usage: u64,
    /// Average memory usage
    pub average_usage: u64,
    /// Memory usage at start
    pub initial_usage: u64,
    /// Memory usage at end
    pub final_usage: u64,
    /// Memory leaked (final - initial)
    pub memory_leaked: i64,
}

/// Issues detected during stress testing
#[derive(Debug, Clone)]
pub struct StressTestIssue {
    /// Issue description
    pub description: String,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Production impact assessment
    pub production_impact: ProductionImpact,
    /// Suggested fix or mitigation
    pub suggested_fix: Option<String>,
    /// Test iteration where issue was detected
    pub iteration: Option<usize>,
}

/// Severity levels for issues
#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    /// Critical issue blocking production
    Critical,
    /// High severity issue
    High,
    /// Medium severity issue
    Medium,
    /// Low severity issue
    Low,
}

/// Production impact assessment
#[derive(Debug, Clone, PartialEq)]
pub enum ProductionImpact {
    /// Blocks production deployment
    Blocking,
    /// Major performance impact
    Major,
    /// Minor performance impact
    Minor,
    /// No significant impact
    None,
}

/// Baseline performance for comparison
#[derive(Debug, Clone)]
pub struct BaselinePerformance {
    /// Baseline execution time
    pub execution_time: Duration,
    /// Baseline memory usage
    pub memory_usage: u64,
    /// Baseline throughput
    pub throughput: f64,
}

/// Comprehensive stress test report
#[derive(Debug, Clone)]
pub struct StressTestReport {
    /// Overall test results
    pub results: Vec<StressTestResult>,
    /// Test configuration used
    pub config: StressTestConfig,
    /// Overall production readiness assessment
    pub production_readiness: ProductionReadiness,
    /// Summary statistics
    pub summary: StressTestSummary,
    /// Critical issues found
    pub critical_issues: Vec<StressTestIssue>,
    /// Performance analysis
    pub performance_analysis: PerformanceAnalysis,
    /// Recommendations for production deployment
    pub production_recommendations: Vec<String>,
}

/// Production readiness assessment
#[derive(Debug, Clone, PartialEq)]
pub enum ProductionReadiness {
    /// Ready for production
    Ready,
    /// Ready with monitoring recommendations
    ReadyWithMonitoring,
    /// Needs performance tuning
    NeedsPerformanceTuning,
    /// Needs bug fixes
    NeedsBugFixes,
    /// Not ready for production
    NotReady,
}

/// Summary statistics for stress tests
#[derive(Debug, Clone)]
pub struct StressTestSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Tests failed
    pub tests_failed: usize,
    /// Tests with warnings
    pub tests_with_warnings: usize,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Memory efficiency score (0-100)
    pub memory_efficiency_score: f64,
    /// Performance degradation factor
    pub overall_degradation_factor: f64,
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Performance bottlenecks identified
    pub bottlenecks: Vec<String>,
    /// Performance improvements suggested
    pub improvements: Vec<String>,
    /// Expected performance in production
    pub production_performance_estimate: Option<Duration>,
    /// Scalability assessment
    pub scalability_assessment: ScalabilityAssessment,
}

/// Scalability assessment
#[derive(Debug, Clone)]
pub struct ScalabilityAssessment {
    /// Can handle 10x current load
    pub can_handle_10x_load: bool,
    /// Can handle 100x current load
    pub can_handle_100x_load: bool,
    /// Estimated maximum scale factor
    pub max_scale_factor: f64,
    /// Limiting factors
    pub limiting_factors: Vec<String>,
}

impl Default for StressPerformanceMetrics {
    fn default() -> Self {
        Self {
            mean_execution_time: Duration::from_millis(0),
            max_execution_time: Duration::from_millis(0),
            min_execution_time: Duration::from_millis(0),
            degradation_factor: None,
            throughput: None,
            memory_efficiency: None,
        }
    }
}

impl<T: InterpolationFloat> ProductionStressTester<T> {
    /// Create a new stress tester with default configuration
    pub fn new() -> Self {
        Self {
            config: StressTestConfig::default(),
            results: Vec::new(),
            baseline_performance: None,
            error_patterns: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new stress tester with custom configuration
    pub fn with_config(config: StressTestConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            baseline_performance: None,
            error_patterns: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: InterpolationFloat> Default for ProductionStressTester<T> {
    fn default() -> Self {
        Self::new()
    }
}
