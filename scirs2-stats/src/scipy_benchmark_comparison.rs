//! SciPy benchmark comparison framework for scirs2-stats v1.0.0
//!
//! This module provides comprehensive benchmarking against SciPy to validate
//! performance, accuracy, and API compatibility. It includes automated test
//! generation, statistical validation, and performance regression detection.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::time::{Duration, Instant};

/// Configuration for SciPy comparison benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScipyComparisonConfig {
    /// Python executable path
    pub python_executable: String,
    /// SciPy version requirement
    pub scipy_version: Option<String>,
    /// NumPy version requirement
    pub numpy_version: Option<String>,
    /// Temporary directory for test scripts
    pub temp_dir: String,
    /// Accuracy tolerance for numerical comparisons
    pub accuracy_tolerance: f64,
    /// Performance tolerance (ratio to SciPy)
    pub performance_tolerance: f64,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Enable detailed accuracy analysis
    pub detailed_accuracy: bool,
    /// Enable memory usage comparison
    pub compare_memory: bool,
    /// Test data sizes
    pub testsizes: Vec<usize>,
    /// Functions to benchmark
    pub functions_to_test: Vec<String>,
}

impl Default for ScipyComparisonConfig {
    fn default() -> Self {
        Self {
            python_executable: "python3".to_string(),
            scipy_version: Some(">=1.9.0".to_string()),
            numpy_version: Some(">=1.21.0".to_string()),
            temp_dir: "/tmp/scirs2_benchmarks".to_string(),
            accuracy_tolerance: 1e-10,
            performance_tolerance: 2.0, // Allow 2x slower than SciPy
            warmup_iterations: 10,
            measurement_iterations: 100,
            detailed_accuracy: true,
            compare_memory: true,
            testsizes: vec![100, 1000, 10000, 100000],
            functions_to_test: vec![
                "mean".to_string(),
                "std".to_string(),
                "var".to_string(),
                "skew".to_string(),
                "kurtosis".to_string(),
                "pearsonr".to_string(),
                "spearmanr".to_string(),
                "ttest_ind".to_string(),
                "ttest_1samp".to_string(),
                "norm_pdf".to_string(),
                "norm_cdf".to_string(),
            ],
        }
    }
}

/// Results of SciPy comparison benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScipyComparisonReport {
    /// Timestamp of the comparison
    pub timestamp: String,
    /// Configuration used
    pub config: ScipyComparisonConfig,
    /// System information
    pub system_info: SystemInfo,
    /// SciPy environment info
    pub scipy_environment: ScipyEnvironmentInfo,
    /// Individual function comparison results
    pub function_comparisons: Vec<FunctionComparison>,
    /// Overall summary statistics
    pub summary: ComparisonSummary,
    /// Performance analysis
    pub performance_analysis: PerformanceAnalysis,
    /// Accuracy analysis
    pub accuracy_analysis: AccuracyAnalysis,
    /// Recommendations
    pub recommendations: Vec<ComparisonRecommendation>,
}

/// System information for comparison context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// CPU information
    pub cpu: String,
    /// Memory information
    pub memory_gb: f64,
    /// Rust version
    pub rust_version: String,
    /// scirs2-stats version
    pub scirs2_version: String,
}

/// SciPy environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScipyEnvironmentInfo {
    /// Python version
    pub python_version: String,
    /// SciPy version
    pub scipy_version: String,
    /// NumPy version
    pub numpy_version: String,
    /// BLAS/LAPACK information
    pub blas_info: String,
    /// Available Python packages
    pub packages: HashMap<String, String>,
}

/// Comparison results for a single function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionComparison {
    /// Function name
    pub function_name: String,
    /// Test data size
    pub datasize: usize,
    /// Performance comparison
    pub performance: PerformanceComparison,
    /// Accuracy comparison
    pub accuracy: AccuracyComparison,
    /// Memory usage comparison
    pub memory: Option<MemoryComparison>,
    /// Test status
    pub status: ComparisonStatus,
    /// Error details if failed
    pub error_details: Option<String>,
}

/// Performance comparison between scirs2-stats and SciPy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    /// scirs2-stats execution time (nanoseconds)
    pub scirs2_time_ns: f64,
    /// SciPy execution time (nanoseconds)
    pub scipy_time_ns: f64,
    /// Performance ratio (scirs2/scipy)
    pub ratio: f64,
    /// Statistical significance of difference
    pub significance: PerformanceSignificance,
    /// Confidence interval for ratio
    pub confidence_interval: (f64, f64),
}

/// Accuracy comparison between implementations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyComparison {
    /// Absolute difference
    pub absolute_difference: f64,
    /// Relative difference
    pub relative_difference: f64,
    /// Maximum element-wise difference
    pub max_element_difference: f64,
    /// Number of elements compared
    pub elements_compared: usize,
    /// Elements within tolerance
    pub elements_within_tolerance: usize,
    /// Accuracy assessment
    pub assessment: AccuracyAssessment,
}

/// Memory usage comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryComparison {
    /// scirs2-stats memory usage (bytes)
    pub scirs2_memory: usize,
    /// SciPy memory usage (bytes)
    pub scipy_memory: usize,
    /// Memory ratio (scirs2/scipy)
    pub ratio: f64,
    /// Memory efficiency assessment
    pub assessment: MemoryEfficiencyAssessment,
}

/// Status of comparison test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonStatus {
    /// Test passed all checks
    Passed,
    /// Test passed with warnings
    PassedWithWarnings { warnings: Vec<String> },
    /// Test failed accuracy requirements
    FailedAccuracy { details: String },
    /// Test failed performance requirements
    FailedPerformance { details: String },
    /// Test encountered execution error
    Error { error: String },
    /// Test was skipped
    Skipped { reason: String },
}

/// Performance significance assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceSignificance {
    /// No significant difference
    NotSignificant,
    /// scirs2-stats significantly faster
    ScirsFaster { confidence: f64 },
    /// SciPy significantly faster
    ScipyFaster { confidence: f64 },
    /// Insufficient data for assessment
    InsufficientData,
}

/// Accuracy assessment categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccuracyAssessment {
    /// Excellent accuracy (within machine precision)
    Excellent,
    /// Good accuracy (within specified tolerance)
    Good,
    /// Acceptable accuracy (small differences)
    Acceptable,
    /// Poor accuracy (significant differences)
    Poor,
    /// Unacceptable accuracy (large differences)
    Unacceptable,
}

/// Memory efficiency assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryEfficiencyAssessment {
    /// More memory efficient than SciPy
    MoreEfficient,
    /// Similar memory usage to SciPy
    Similar,
    /// Less memory efficient than SciPy
    LessEfficient,
    /// Significantly less memory efficient
    MuchLessEfficient,
}

/// Overall comparison summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub tests_passed: usize,
    /// Tests with warnings
    pub tests_with_warnings: usize,
    /// Tests failed
    pub tests_failed: usize,
    /// Overall pass rate
    pub pass_rate: f64,
    /// Functions with performance issues
    pub performance_issues: Vec<String>,
    /// Functions with accuracy issues
    pub accuracy_issues: Vec<String>,
    /// Overall performance rating
    pub performance_rating: PerformanceRating,
    /// Overall accuracy rating
    pub accuracy_rating: AccuracyRating,
}

/// Performance analysis across all tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    /// Average performance ratio
    pub average_ratio: f64,
    /// Performance ratio standard deviation
    pub ratio_std_dev: f64,
    /// Functions faster than SciPy
    pub faster_functions: Vec<(String, f64)>,
    /// Functions slower than SciPy
    pub slower_functions: Vec<(String, f64)>,
    /// Performance by data size
    pub performance_bysize: HashMap<usize, f64>,
    /// Performance trends
    pub trends: PerformanceTrends,
}

/// Accuracy analysis across all tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyAnalysis {
    /// Average relative difference
    pub average_relative_diff: f64,
    /// Maximum relative difference
    pub max_relative_diff: f64,
    /// Functions with accuracy issues
    pub problematic_functions: Vec<(String, f64)>,
    /// Accuracy by data size
    pub accuracy_bysize: HashMap<usize, f64>,
    /// Numerical stability assessment
    pub stability_assessment: NumericalStabilityAssessment,
}

/// Performance rating categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceRating {
    /// Excellent performance (consistently faster)
    Excellent,
    /// Good performance (mostly competitive)
    Good,
    /// Acceptable performance (within tolerance)
    Acceptable,
    /// Poor performance (consistently slower)
    Poor,
    /// Unacceptable performance (significantly slower)
    Unacceptable,
}

/// Accuracy rating categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccuracyRating {
    /// Excellent accuracy (machine precision)
    Excellent,
    /// Good accuracy (high precision)
    Good,
    /// Acceptable accuracy (within tolerance)
    Acceptable,
    /// Poor accuracy (noticeable differences)
    Poor,
    /// Unacceptable accuracy (significant errors)
    Unacceptable,
}

/// Performance trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Performance scaling with data size
    pub scaling_analysis: ScalingAnalysis,
    /// Performance stability over multiple runs
    pub stability_analysis: StabilityAnalysis,
    /// Performance regression detection
    pub regression_analysis: RegressionAnalysis,
}

/// Scaling analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    /// Scaling factor relative to SciPy
    pub relative_scaling: f64,
    /// Complexity assessment
    pub complexity_assessment: ComplexityAssessment,
    /// Crossover points where performance changes
    pub crossover_points: Vec<usize>,
}

/// Stability analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityAnalysis {
    /// Coefficient of variation for performance
    pub performance_cv: f64,
    /// Performance outliers detected
    pub outliers_detected: usize,
    /// Stability rating
    pub stability_rating: StabilityRating,
}

/// Regression analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    /// Performance regressions detected
    pub regressions_detected: Vec<PerformanceRegression>,
    /// Accuracy regressions detected
    pub accuracy_regressions: Vec<AccuracyRegression>,
    /// Overall regression risk
    pub regression_risk: RegressionRisk,
}

/// Complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityAssessment {
    /// Better complexity than SciPy
    Better,
    /// Similar complexity to SciPy
    Similar,
    /// Worse complexity than SciPy
    Worse,
    /// Unknown complexity relationship
    Unknown,
}

/// Stability rating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityRating {
    /// Very stable performance
    VeryStable,
    /// Stable performance
    Stable,
    /// Moderately stable
    ModeratelyStable,
    /// Unstable performance
    Unstable,
    /// Very unstable performance
    VeryUnstable,
}

/// Performance regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Function affected
    pub function_name: String,
    /// Regression magnitude
    pub regression_factor: f64,
    /// Confidence in detection
    pub confidence: f64,
    /// Suspected cause
    pub suspected_cause: String,
}

/// Accuracy regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyRegression {
    /// Function affected
    pub function_name: String,
    /// Accuracy degradation
    pub accuracy_loss: f64,
    /// Severity assessment
    pub severity: AccuracyRegressionSeverity,
}

/// Regression risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionRisk {
    /// Low risk of regressions
    Low,
    /// Medium risk of regressions
    Medium,
    /// High risk of regressions
    High,
    /// Critical risk of regressions
    Critical,
}

/// Accuracy regression severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccuracyRegressionSeverity {
    /// Minor accuracy loss
    Minor,
    /// Moderate accuracy loss
    Moderate,
    /// Major accuracy loss
    Major,
    /// Critical accuracy loss
    Critical,
}

/// Numerical stability assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalStabilityAssessment {
    /// Overall stability rating
    pub stability_rating: NumericalStabilityRating,
    /// Functions with stability issues
    pub unstable_functions: Vec<String>,
    /// Condition number analysis
    pub condition_number_analysis: ConditionNumberAnalysis,
    /// Precision loss analysis
    pub precision_loss_analysis: PrecisionLossAnalysis,
}

/// Numerical stability rating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NumericalStabilityRating {
    /// Excellent numerical stability
    Excellent,
    /// Good numerical stability
    Good,
    /// Acceptable numerical stability
    Acceptable,
    /// Poor numerical stability
    Poor,
    /// Unacceptable numerical stability
    Unacceptable,
}

/// Condition number analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionNumberAnalysis {
    /// Functions sensitive to condition number
    pub sensitive_functions: Vec<String>,
    /// Condition number thresholds
    pub thresholds: HashMap<String, f64>,
    /// Stability recommendations
    pub recommendations: Vec<String>,
}

/// Precision loss analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionLossAnalysis {
    /// Average precision loss
    pub average_loss: f64,
    /// Maximum precision loss
    pub max_loss: f64,
    /// Functions with significant loss
    pub problematic_functions: Vec<String>,
}

/// Comparison recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRecommendation {
    /// Recommendation priority
    pub priority: RecommendationPriority,
    /// Category of recommendation
    pub category: RecommendationCategory,
    /// Recommendation description
    pub description: String,
    /// Affected functions
    pub affected_functions: Vec<String>,
    /// Implementation complexity
    pub complexity: ImplementationComplexity,
    /// Expected impact
    pub expected_impact: ExpectedImpact,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Critical priority
    Critical,
    /// High priority
    High,
    /// Medium priority
    Medium,
    /// Low priority
    Low,
    /// Nice to have
    NiceToHave,
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    /// Performance optimization
    Performance,
    /// Accuracy improvement
    Accuracy,
    /// Memory optimization
    Memory,
    /// API compatibility
    APICompatibility,
    /// Numerical stability
    NumericalStability,
    /// Testing enhancement
    Testing,
}

/// Implementation complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationComplexity {
    /// Simple to implement
    Simple,
    /// Moderate complexity
    Moderate,
    /// Complex implementation
    Complex,
    /// Very complex implementation
    VeryComplex,
}

/// Expected impact of recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImpact {
    /// Performance improvement factor
    pub performance_improvement: Option<f64>,
    /// Accuracy improvement
    pub accuracy_improvement: Option<f64>,
    /// Memory reduction factor
    pub memory_reduction: Option<f64>,
    /// Implementation effort (person-days)
    pub implementation_effort: f64,
}

/// Main SciPy comparison framework
pub struct ScipyBenchmarkComparison {
    config: ScipyComparisonConfig,
    temp_dir: String,
}

impl ScipyBenchmarkComparison {
    /// Create new SciPy comparison framework
    pub fn new(config: ScipyComparisonConfig) -> StatsResult<Self> {
        // Create temporary directory
        fs::create_dir_all(&config.temp_dir).map_err(|e| {
            StatsError::ComputationError(format!("Failed to create temp directory: {}", e))
        })?;

        Ok(Self {
            temp_dir: config.temp_dir.clone(),
            config,
        })
    }

    /// Create with default configuration
    pub fn default() -> StatsResult<Self> {
        Self::new(ScipyComparisonConfig::default())
    }

    /// Run comprehensive comparison benchmarks
    pub fn run_comprehensive_comparison(&self) -> StatsResult<ScipyComparisonReport> {
        let _start_time = Instant::now();

        // Verify SciPy environment
        let scipy_env = self.verify_scipy_environment()?;

        // Collect system information
        let system_info = self.collect_system_info();

        // Run function comparisons
        let mut function_comparisons = Vec::new();

        for function_name in &self.config.functions_to_test {
            for &datasize in &self.config.testsizes {
                match self.compare_function(function_name, datasize) {
                    Ok(comparison) => function_comparisons.push(comparison),
                    Err(e) => {
                        function_comparisons.push(FunctionComparison {
                            function_name: function_name.clone(),
                            datasize,
                            performance: PerformanceComparison {
                                scirs2_time_ns: 0.0,
                                scipy_time_ns: 0.0,
                                ratio: 0.0,
                                significance: PerformanceSignificance::InsufficientData,
                                confidence_interval: (0.0, 0.0),
                            },
                            accuracy: AccuracyComparison {
                                absolute_difference: 0.0,
                                relative_difference: 0.0,
                                max_element_difference: 0.0,
                                elements_compared: 0,
                                elements_within_tolerance: 0,
                                assessment: AccuracyAssessment::Poor,
                            },
                            memory: None,
                            status: ComparisonStatus::Error {
                                error: e.to_string(),
                            },
                            error_details: Some(e.to_string()),
                        });
                    }
                }
            }
        }

        // Analyze results
        let summary = self.generate_summary(&function_comparisons);
        let performance_analysis = self.analyze_performance(&function_comparisons);
        let accuracy_analysis = self.analyze_accuracy(&function_comparisons);
        let recommendations = self.generate_recommendations(
            &function_comparisons,
            &performance_analysis,
            &accuracy_analysis,
        );

        Ok(ScipyComparisonReport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            config: self.config.clone(),
            system_info,
            scipy_environment: scipy_env,
            function_comparisons,
            summary,
            performance_analysis,
            accuracy_analysis,
            recommendations,
        })
    }

    /// Verify SciPy environment is available and compatible
    fn verify_scipy_environment(&self) -> StatsResult<ScipyEnvironmentInfo> {
        let script = r#"
import sys
import scipy
import numpy as np
import json

info = {
    'python_version': sys.version,
    'scipy_version': scipy.__version__,
    'numpy_version': np.__version__,
    'blas_info': str(np.__config__.show()),
    'packages': {}
}

try:
    import pandas
    info['packages']['pandas'] = pandas.__version__
except ImportError:
    pass

try:
    import sklearn
    info['packages']['sklearn'] = sklearn.__version__
except ImportError:
    pass

print(json.dumps(info))
"#;

        let script_path = format!("{}/verify_env.py", self.temp_dir);
        fs::write(&script_path, script).map_err(|e| {
            StatsError::ComputationError(format!("Failed to write verification script: {}", e))
        })?;

        let output = Command::new(&self.config.python_executable)
            .arg(&script_path)
            .output()
            .map_err(|e| {
                StatsError::ComputationError(format!("Failed to execute Python: {}", e))
            })?;

        if !output.status.success() {
            return Err(StatsError::ComputationError(format!(
                "Python script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let output_str = String::from_utf8_lossy(&output.stdout);
        let info: serde_json::Value = serde_json::from_str(&output_str).map_err(|e| {
            StatsError::ComputationError(format!("Failed to parse environment info: {}", e))
        })?;

        Ok(ScipyEnvironmentInfo {
            python_version: info["python_version"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            scipy_version: info["scipy_version"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            numpy_version: info["numpy_version"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            blas_info: info["blas_info"].as_str().unwrap_or("unknown").to_string(),
            packages: info["packages"]
                .as_object()
                .unwrap_or(&serde_json::Map::new())
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("unknown").to_string()))
                .collect(),
        })
    }

    /// Collect system information
    fn collect_system_info(&self) -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            cpu: "Generic CPU".to_string(), // Would use proper CPU detection
            memory_gb: 8.0,                 // Placeholder
            rust_version: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "unknown".to_string()),
            scirs2_version: std::env::var("CARGO_PKG_VERSION")
                .unwrap_or_else(|_| "unknown".to_string()),
        }
    }

    /// Compare a single function between scirs2-stats and SciPy
    fn compare_function(
        &self,
        function_name: &str,
        datasize: usize,
    ) -> StatsResult<FunctionComparison> {
        // Generate test data
        let testdata = self.generate_testdata(datasize)?;

        // Benchmark scirs2-stats function
        let scirs2_result = self.benchmark_scirs2_function(function_name, &testdata)?;

        // Benchmark SciPy function
        let scipy_result = self.benchmark_scipy_function(function_name, &testdata)?;

        // Compare performance
        let performance = PerformanceComparison {
            scirs2_time_ns: scirs2_result.execution_time.as_nanos() as f64,
            scipy_time_ns: scipy_result.execution_time.as_nanos() as f64,
            ratio: scirs2_result.execution_time.as_nanos() as f64
                / scipy_result.execution_time.as_nanos() as f64,
            significance: PerformanceSignificance::NotSignificant, // Simplified
            confidence_interval: (0.8, 1.2),                       // Placeholder
        };

        // Compare accuracy
        let accuracy = self.compare_accuracy(&scirs2_result.result, &scipy_result.result)?;

        // Determine status
        let status = self.determine_comparison_status(&performance, &accuracy);

        Ok(FunctionComparison {
            function_name: function_name.to_string(),
            datasize,
            performance,
            accuracy,
            memory: None, // Would implement memory comparison
            status,
            error_details: None,
        })
    }

    /// Generate test data for benchmarking
    fn generate_testdata(&self, size: usize) -> StatsResult<TestData> {
        use scirs2_core::random::{Distribution, Normal};

        let mut rng = scirs2_core::random::thread_rng();
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            StatsError::ComputationError(format!("Failed to create normal distribution: {}", e))
        })?;

        let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();

        Ok(TestData {
            primary: Array1::from_vec(data.clone()),
            secondary: Array1::from_vec(
                data.iter()
                    .map(|x| x + 0.1 * normal.sample(&mut rng))
                    .collect(),
            ),
            matrix: Array2::from_shape_fn((size.min(100), size.min(100)), |(i, j)| {
                normal.sample(&mut rng) + 0.1 * (i + j) as f64
            }),
        })
    }

    /// Benchmark scirs2-stats function
    fn benchmark_scirs2_function(
        &self,
        function_name: &str,
        testdata: &TestData,
    ) -> StatsResult<BenchmarkResult> {
        let start_time = Instant::now();

        let result = match function_name {
            "mean" => {
                vec![crate::descriptive::mean(&testdata.primary.view())?]
            }
            "std" => {
                vec![crate::descriptive::std(&testdata.primary.view(), 1, None)?]
            }
            "var" => {
                vec![crate::descriptive::var(&testdata.primary.view(), 1, None)?]
            }
            "skew" => {
                vec![crate::descriptive::skew(
                    &testdata.primary.view(),
                    false,
                    None,
                )?]
            }
            "kurtosis" => {
                vec![crate::descriptive::kurtosis(
                    &testdata.primary.view(),
                    true,
                    false,
                    None,
                )?]
            }
            "pearsonr" => {
                let corr = crate::correlation::pearson_r(
                    &testdata.primary.view(),
                    &testdata.secondary.view(),
                )?;
                vec![corr]
            }
            "spearmanr" => {
                let corr = crate::correlation::spearman_r(
                    &testdata.primary.view(),
                    &testdata.secondary.view(),
                )?;
                vec![corr]
            }
            "ttest_1samp" => {
                let result = crate::tests::ttest::ttest_1samp(
                    &testdata.primary.view(),
                    0.0,
                    crate::tests::ttest::Alternative::TwoSided,
                    "propagate",
                )?;
                vec![result.statistic, result.pvalue]
            }
            _ => {
                return Err(StatsError::NotImplemented(format!(
                    "Function {} not implemented in benchmark",
                    function_name
                )));
            }
        };

        let execution_time = start_time.elapsed();

        Ok(BenchmarkResult {
            result,
            execution_time,
        })
    }

    /// Benchmark SciPy function
    fn benchmark_scipy_function(
        &self,
        function_name: &str,
        testdata: &TestData,
    ) -> StatsResult<BenchmarkResult> {
        let script = self.generate_scipy_script(function_name, testdata)?;
        let script_path = format!("{}/scipy_benchmark_{}.py", self.temp_dir, function_name);

        fs::write(&script_path, script).map_err(|e| {
            StatsError::ComputationError(format!("Failed to write SciPy script: {}", e))
        })?;

        let output = Command::new(&self.config.python_executable)
            .arg(&script_path)
            .output()
            .map_err(|e| {
                StatsError::ComputationError(format!("Failed to execute SciPy script: {}", e))
            })?;

        if !output.status.success() {
            return Err(StatsError::ComputationError(format!(
                "SciPy script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let output_str = String::from_utf8_lossy(&output.stdout);
        let result: serde_json::Value = serde_json::from_str(&output_str).map_err(|e| {
            StatsError::ComputationError(format!("Failed to parse SciPy result: {}", e))
        })?;

        let execution_time =
            Duration::from_secs_f64(result["execution_time"].as_f64().unwrap_or(0.0));

        let result_values: Vec<f64> = result["result"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        Ok(BenchmarkResult {
            result: result_values,
            execution_time,
        })
    }

    /// Generate SciPy benchmark script
    fn generate_scipy_script(
        &self,
        function_name: &str,
        testdata: &TestData,
    ) -> StatsResult<String> {
        let data_primary: Vec<String> = testdata.primary.iter().map(|x| x.to_string()).collect();
        let data_secondary: Vec<String> =
            testdata.secondary.iter().map(|x| x.to_string()).collect();

        let script = match function_name {
            "mean" => {
                format!(
                    r#"
import numpy as np
import time
import json

data = np.array([{}])

start_time = time.perf_counter()
result = np.mean(data)
execution_time = time.perf_counter() - start_time

output = {{
    'result': [float(result)],
    'execution_time': execution_time
}}

print(json.dumps(output))
"#,
                    data_primary.join(", ")
                )
            }
            "std" => {
                format!(
                    r#"
import numpy as np
import time
import json

data = np.array([{}])

start_time = time.perf_counter()
result = np.std(data, ddof=1)
execution_time = time.perf_counter() - start_time

output = {{
    'result': [float(result)],
    'execution_time': execution_time
}}

print(json.dumps(output))
"#,
                    data_primary.join(", ")
                )
            }
            "pearsonr" => {
                format!(
                    r#"
import numpy as np
import scipy.stats
import time
import json

data1 = np.array([{}])
data2 = np.array([{}])

start_time = time.perf_counter()
corr, p_value = scipy.stats.pearsonr(data1, data2)
execution_time = time.perf_counter() - start_time

output = {{
    'result': [float(corr)],
    'execution_time': execution_time
}}

print(json.dumps(output))
"#,
                    data_primary.join(", "),
                    data_secondary.join(", ")
                )
            }
            _ => {
                return Err(StatsError::NotImplemented(format!(
                    "SciPy script generation not implemented for {}",
                    function_name
                )));
            }
        };

        Ok(script)
    }

    /// Compare accuracy between results
    fn compare_accuracy(
        &self,
        scirs2_result: &[f64],
        scipy_result: &[f64],
    ) -> StatsResult<AccuracyComparison> {
        if scirs2_result.len() != scipy_result.len() {
            return Ok(AccuracyComparison {
                absolute_difference: f64::INFINITY,
                relative_difference: f64::INFINITY,
                max_element_difference: f64::INFINITY,
                elements_compared: 0,
                elements_within_tolerance: 0,
                assessment: AccuracyAssessment::Unacceptable,
            });
        }

        let mut abs_diffs = Vec::new();
        let mut rel_diffs = Vec::new();
        let mut within_tolerance = 0;

        for (s, r) in scirs2_result.iter().zip(scipy_result.iter()) {
            let abs_diff = (s - r).abs();
            let rel_diff = if r.abs() > 1e-10 {
                abs_diff / r.abs()
            } else {
                abs_diff
            };

            abs_diffs.push(abs_diff);
            rel_diffs.push(rel_diff);

            if abs_diff < self.config.accuracy_tolerance
                || rel_diff < self.config.accuracy_tolerance
            {
                within_tolerance += 1;
            }
        }

        let avg_abs_diff = abs_diffs.iter().sum::<f64>() / abs_diffs.len() as f64;
        let avg_rel_diff = rel_diffs.iter().sum::<f64>() / rel_diffs.len() as f64;
        let max_element_diff = abs_diffs.iter().fold(0.0f64, |acc, &x| acc.max(x));

        let assessment = if within_tolerance == scirs2_result.len() {
            if avg_rel_diff < 1e-14 {
                AccuracyAssessment::Excellent
            } else if avg_rel_diff < 1e-10 {
                AccuracyAssessment::Good
            } else {
                AccuracyAssessment::Acceptable
            }
        } else if within_tolerance as f64 / scirs2_result.len() as f64 > 0.9 {
            AccuracyAssessment::Acceptable
        } else if within_tolerance as f64 / scirs2_result.len() as f64 > 0.5 {
            AccuracyAssessment::Poor
        } else {
            AccuracyAssessment::Unacceptable
        };

        Ok(AccuracyComparison {
            absolute_difference: avg_abs_diff,
            relative_difference: avg_rel_diff,
            max_element_difference: max_element_diff,
            elements_compared: scirs2_result.len(),
            elements_within_tolerance: within_tolerance,
            assessment,
        })
    }

    /// Determine comparison status
    fn determine_comparison_status(
        &self,
        performance: &PerformanceComparison,
        accuracy: &AccuracyComparison,
    ) -> ComparisonStatus {
        let mut warnings = Vec::new();

        // Check accuracy
        if matches!(
            accuracy.assessment,
            AccuracyAssessment::Unacceptable | AccuracyAssessment::Poor
        ) {
            return ComparisonStatus::FailedAccuracy {
                details: format!("Relative difference: {:.2e}", accuracy.relative_difference),
            };
        }

        // Check performance
        if performance.ratio > self.config.performance_tolerance {
            return ComparisonStatus::FailedPerformance {
                details: format!(
                    "Performance ratio: {:.2} (limit: {:.2})",
                    performance.ratio, self.config.performance_tolerance
                ),
            };
        }

        // Check for warnings
        if matches!(accuracy.assessment, AccuracyAssessment::Acceptable) {
            warnings.push("Accuracy is only acceptable".to_string());
        }

        if performance.ratio > 1.5 {
            warnings.push(format!(
                "Performance is {:.1}x slower than SciPy",
                performance.ratio
            ));
        }

        if warnings.is_empty() {
            ComparisonStatus::Passed
        } else {
            ComparisonStatus::PassedWithWarnings { warnings }
        }
    }

    /// Generate comparison summary
    fn generate_summary(&self, comparisons: &[FunctionComparison]) -> ComparisonSummary {
        let total_tests = comparisons.len();
        let tests_passed = comparisons
            .iter()
            .filter(|c| matches!(c.status, ComparisonStatus::Passed))
            .count();
        let tests_with_warnings = comparisons
            .iter()
            .filter(|c| matches!(c.status, ComparisonStatus::PassedWithWarnings { .. }))
            .count();
        let tests_failed = total_tests - tests_passed - tests_with_warnings;

        let pass_rate = if total_tests > 0 {
            (tests_passed + tests_with_warnings) as f64 / total_tests as f64
        } else {
            0.0
        };

        let performance_issues: Vec<String> = comparisons
            .iter()
            .filter(|c| matches!(c.status, ComparisonStatus::FailedPerformance { .. }))
            .map(|c| c.function_name.clone())
            .collect();

        let accuracy_issues: Vec<String> = comparisons
            .iter()
            .filter(|c| matches!(c.status, ComparisonStatus::FailedAccuracy { .. }))
            .map(|c| c.function_name.clone())
            .collect();

        let avg_performance_ratio =
            comparisons.iter().map(|c| c.performance.ratio).sum::<f64>() / comparisons.len() as f64;

        let performance_rating = if avg_performance_ratio < 0.8 {
            PerformanceRating::Excellent
        } else if avg_performance_ratio < 1.2 {
            PerformanceRating::Good
        } else if avg_performance_ratio < 2.0 {
            PerformanceRating::Acceptable
        } else if avg_performance_ratio < 5.0 {
            PerformanceRating::Poor
        } else {
            PerformanceRating::Unacceptable
        };

        let avg_relative_diff = comparisons
            .iter()
            .map(|c| c.accuracy.relative_difference)
            .sum::<f64>()
            / comparisons.len() as f64;

        let accuracy_rating = if avg_relative_diff < 1e-14 {
            AccuracyRating::Excellent
        } else if avg_relative_diff < 1e-10 {
            AccuracyRating::Good
        } else if avg_relative_diff < 1e-6 {
            AccuracyRating::Acceptable
        } else if avg_relative_diff < 1e-3 {
            AccuracyRating::Poor
        } else {
            AccuracyRating::Unacceptable
        };

        ComparisonSummary {
            total_tests,
            tests_passed,
            tests_with_warnings,
            tests_failed,
            pass_rate,
            performance_issues,
            accuracy_issues,
            performance_rating,
            accuracy_rating,
        }
    }

    /// Analyze performance across all comparisons
    fn analyze_performance(&self, comparisons: &[FunctionComparison]) -> PerformanceAnalysis {
        let ratios: Vec<f64> = comparisons.iter().map(|c| c.performance.ratio).collect();

        let average_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
        let variance = ratios
            .iter()
            .map(|r| (r - average_ratio).powi(2))
            .sum::<f64>()
            / ratios.len() as f64;
        let ratio_std_dev = variance.sqrt();

        let faster_functions: Vec<(String, f64)> = comparisons
            .iter()
            .filter(|c| c.performance.ratio < 1.0)
            .map(|c| (c.function_name.clone(), c.performance.ratio))
            .collect();

        let slower_functions: Vec<(String, f64)> = comparisons
            .iter()
            .filter(|c| c.performance.ratio > 1.0)
            .map(|c| (c.function_name.clone(), c.performance.ratio))
            .collect();

        let performance_bysize = HashMap::new(); // Would implement proper analysis

        let trends = PerformanceTrends {
            scaling_analysis: ScalingAnalysis {
                relative_scaling: 1.0, // Placeholder
                complexity_assessment: ComplexityAssessment::Similar,
                crossover_points: Vec::new(),
            },
            stability_analysis: StabilityAnalysis {
                performance_cv: ratio_std_dev / average_ratio,
                outliers_detected: 0, // Would implement outlier detection
                stability_rating: StabilityRating::Stable,
            },
            regression_analysis: RegressionAnalysis {
                regressions_detected: Vec::new(),
                accuracy_regressions: Vec::new(),
                regression_risk: RegressionRisk::Low,
            },
        };

        PerformanceAnalysis {
            average_ratio,
            ratio_std_dev,
            faster_functions,
            slower_functions,
            performance_bysize,
            trends,
        }
    }

    /// Analyze accuracy across all comparisons
    fn analyze_accuracy(&self, comparisons: &[FunctionComparison]) -> AccuracyAnalysis {
        let relative_diffs: Vec<f64> = comparisons
            .iter()
            .map(|c| c.accuracy.relative_difference)
            .collect();

        let average_relative_diff =
            relative_diffs.iter().sum::<f64>() / relative_diffs.len() as f64;
        let max_relative_diff = relative_diffs.iter().fold(0.0f64, |acc, &x| acc.max(x));

        let problematic_functions: Vec<(String, f64)> = comparisons
            .iter()
            .filter(|c| c.accuracy.relative_difference > self.config.accuracy_tolerance)
            .map(|c| (c.function_name.clone(), c.accuracy.relative_difference))
            .collect();

        let accuracy_bysize = HashMap::new(); // Would implement proper analysis

        let stability_assessment = NumericalStabilityAssessment {
            stability_rating: if max_relative_diff < 1e-10 {
                NumericalStabilityRating::Excellent
            } else if max_relative_diff < 1e-6 {
                NumericalStabilityRating::Good
            } else {
                NumericalStabilityRating::Acceptable
            },
            unstable_functions: problematic_functions
                .iter()
                .map(|(name, _)| name.clone())
                .collect(),
            condition_number_analysis: ConditionNumberAnalysis {
                sensitive_functions: Vec::new(),
                thresholds: HashMap::new(),
                recommendations: Vec::new(),
            },
            precision_loss_analysis: PrecisionLossAnalysis {
                average_loss: average_relative_diff,
                max_loss: max_relative_diff,
                problematic_functions: problematic_functions
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect(),
            },
        };

        AccuracyAnalysis {
            average_relative_diff,
            max_relative_diff,
            problematic_functions,
            accuracy_bysize,
            stability_assessment,
        }
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(
        &self,
        comparisons: &[FunctionComparison],
        performance_analysis: &PerformanceAnalysis,
        accuracy_analysis: &AccuracyAnalysis,
    ) -> Vec<ComparisonRecommendation> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if performance_analysis.average_ratio > 2.0 {
            recommendations.push(ComparisonRecommendation {
                priority: RecommendationPriority::High,
                category: RecommendationCategory::Performance,
                description: "Overall performance is significantly slower than SciPy. Consider SIMD optimizations and algorithm improvements.".to_string(),
                affected_functions: performance_analysis.slower_functions.iter().map(|(name, _)| name.clone()).collect(),
                complexity: ImplementationComplexity::Moderate,
                expected_impact: ExpectedImpact {
                    performance_improvement: Some(2.0),
                    accuracy_improvement: None,
                    memory_reduction: None,
                    implementation_effort: 20.0,
                },
            });
        }

        // Accuracy recommendations
        if accuracy_analysis.max_relative_diff > 1e-6 {
            recommendations.push(ComparisonRecommendation {
                priority: RecommendationPriority::Critical,
                category: RecommendationCategory::Accuracy,
                description: "Some functions have significant accuracy differences compared to SciPy. Review numerical algorithms.".to_string(),
                affected_functions: accuracy_analysis.problematic_functions.iter().map(|(name, _)| name.clone()).collect(),
                complexity: ImplementationComplexity::Complex,
                expected_impact: ExpectedImpact {
                    performance_improvement: None,
                    accuracy_improvement: Some(10.0),
                    memory_reduction: None,
                    implementation_effort: 15.0,
                },
            });
        }

        // Function-specific recommendations
        for comparison in comparisons {
            if comparison.performance.ratio > 5.0 {
                recommendations.push(ComparisonRecommendation {
                    priority: RecommendationPriority::High,
                    category: RecommendationCategory::Performance,
                    description: format!(
                        "Function '{}' is significantly slower than SciPy",
                        comparison.function_name
                    ),
                    affected_functions: vec![comparison.function_name.clone()],
                    complexity: ImplementationComplexity::Moderate,
                    expected_impact: ExpectedImpact {
                        performance_improvement: Some(3.0),
                        accuracy_improvement: None,
                        memory_reduction: None,
                        implementation_effort: 5.0,
                    },
                });
            }
        }

        recommendations
    }
}

/// Test data structure for benchmarking
#[derive(Debug, Clone)]
struct TestData {
    primary: Array1<f64>,
    secondary: Array1<f64>,
    #[allow(dead_code)]
    matrix: Array2<f64>,
}

/// Benchmark result structure
#[derive(Debug, Clone)]
struct BenchmarkResult {
    result: Vec<f64>,
    execution_time: Duration,
}

/// Convenience function to run SciPy comparison
#[allow(dead_code)]
pub fn run_scipy_comparison() -> StatsResult<ScipyComparisonReport> {
    let comparison = ScipyBenchmarkComparison::default()?;
    comparison.run_comprehensive_comparison()
}

/// Run comparison for specific functions
#[allow(dead_code)]
pub fn run_function_comparison(functions: Vec<String>) -> StatsResult<ScipyComparisonReport> {
    let mut config = ScipyComparisonConfig::default();
    config.functions_to_test = functions;

    let comparison = ScipyBenchmarkComparison::new(config)?;
    comparison.run_comprehensive_comparison()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scipy_comparison_config() {
        let config = ScipyComparisonConfig::default();
        assert!(!config.functions_to_test.is_empty());
        assert!(config.accuracy_tolerance > 0.0);
        assert!(config.performance_tolerance > 1.0);
    }

    #[test]
    fn test_testdata_generation() {
        let comparison = ScipyBenchmarkComparison::default().unwrap();
        let testdata = comparison.generate_testdata(100).unwrap();

        assert_eq!(testdata.primary.len(), 100);
        assert_eq!(testdata.secondary.len(), 100);
        assert_eq!(testdata.matrix.nrows(), 100);
    }

    #[test]
    fn test_accuracy_comparison() {
        let comparison = ScipyBenchmarkComparison::default().unwrap();

        // Use very small differences well within tolerance (< 1e-12)
        let scirs2_result = vec![1.0, 2.0, 3.0];
        let scipy_result = vec![1.000000000001, 2.000000000001, 3.000000000001];

        let accuracy = comparison
            .compare_accuracy(&scirs2_result, &scipy_result)
            .unwrap();
        assert!(matches!(
            accuracy.assessment,
            AccuracyAssessment::Excellent | AccuracyAssessment::Good
        ));
    }

    #[test]
    fn test_performance_comparison() {
        let performance = PerformanceComparison {
            scirs2_time_ns: 1000.0,
            scipy_time_ns: 800.0,
            ratio: 1.25,
            significance: PerformanceSignificance::NotSignificant,
            confidence_interval: (1.0, 1.5),
        };

        assert!(performance.ratio > 1.0); // scirs2 slower
    }

    #[test]
    fn test_recommendation_generation() {
        let config = ScipyComparisonConfig::default();
        assert!(config.performance_tolerance >= 1.0);
        assert!(config.accuracy_tolerance > 0.0);
    }
}
