//! Performance regression testing framework
//!
//! This module provides comprehensive performance regression detection capabilities
//! including baseline establishment, historical tracking, statistical analysis,
//! and automated CI/CD integration for continuous performance monitoring.

// Import all submodules
pub mod alerts;
pub mod config;
pub mod database;
pub mod detectors;
pub mod statistics;
pub mod types;

// Re-export commonly used types and functions
pub use alerts::{AlertSystem, AlertStatistics, SeverityCounts};
pub use config::{
    Alert, AlertConfig, AlertSeverity, AlertStatus, CiReportFormat, RegressionConfig,
    TestEnvironment,
};
pub use database::PerformanceDatabase;
pub use detectors::{
    ChangePointDetector, SlidingWindowDetector, StatisticalTestDetector,
};
pub use statistics::{OutlierAnalyzer, TrendAnalyzer};
pub use types::{
    BaselineStatistics, ChangePointAnalysis, ConfidenceIntervals, ConvergenceMetrics,
    ConvergenceStatistics, DatabaseMetadata, EfficiencyMetrics, EfficiencyStatistics,
    FragmentationStatistics, MemoryMetrics, MemoryStatistics, OutlierAnalysis, OutlierType,
    PerformanceBaseline, PerformanceMetrics, PerformanceRecord, RegressionAnalysis,
    RegressionDetector, RegressionResult, StatisticalAnalysisResult, StatisticalAnalyzer, StatisticalTestResult, TimingMetrics,
    TimingStatistics, TrendAnalysis, TrendDirection,
};

// For now, we'll include a simplified main framework here
// In a full refactoring, this would be split into core.rs and ci_integration.rs

use crate::benchmarking::BenchmarkResult;
use crate::error::Result;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;

/// Comprehensive performance regression testing framework
#[derive(Debug)]
pub struct RegressionTester<A: Float> {
    /// Configuration for regression testing
    config: RegressionConfig,
    /// Historical performance database
    performance_db: PerformanceDatabase<A>,
    /// Baseline performance metrics
    baselines: HashMap<String, PerformanceBaseline<A>>,
    /// Regression detection algorithms
    detectors: Vec<Box<dyn RegressionDetector<A>>>,
    /// Statistical analyzers
    analyzers: Vec<Box<dyn StatisticalAnalyzer<A>>>,
    /// Alert system
    alert_system: AlertSystem,
}

/// Regression test result summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestResult<A: Float> {
    /// Test identifier
    pub test_id: String,
    /// Test execution status
    pub status: String,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Number of regressions detected
    pub regression_count: usize,
    /// Detected regressions
    pub regressions: Vec<RegressionResult<A>>,
    /// Performance metrics
    pub metrics: PerformanceMetrics<A>,
    /// Baseline comparison results
    pub baseline_comparison: Option<BaselineComparison>,
}

/// Baseline comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Performance change percentage
    pub performance_change_percent: f64,
    /// Memory change percentage
    pub memory_change_percent: f64,
    /// Efficiency change percentage
    pub efficiency_change_percent: f64,
    /// Comparison status
    pub status: String,
}

/// CI report structure
#[derive(Debug, Serialize)]
pub struct CiReport {
    /// Report timestamp
    pub timestamp: u64,
    /// Total number of tests
    pub total_tests: usize,
    /// Number of passed tests
    pub passed_tests: usize,
    /// Number of failed tests
    pub failed_tests: usize,
    /// Individual test results
    pub test_results: Vec<CiTestResult>,
}

/// Individual test result for CI
#[derive(Debug, Serialize)]
pub struct CiTestResult {
    /// Test name
    pub name: String,
    /// Test status
    pub status: String,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Number of regressions detected
    pub regression_count: usize,
}

impl<A: Float + Debug + Serialize + for<'de> Deserialize<'de>> RegressionTester<A> {
    /// Create a new regression tester
    pub fn new(config: RegressionConfig) -> Result<Self> {
        // Ensure baseline directory exists
        fs::create_dir_all(&config.baseline_dir)?;

        let performance_db = PerformanceDatabase::load(&config.baseline_dir)
            .unwrap_or_else(|_| PerformanceDatabase::new());

        let mut tester = Self {
            config: config.clone(),
            performance_db,
            baselines: HashMap::new(),
            detectors: Vec::new(),
            analyzers: Vec::new(),
            alert_system: AlertSystem::new(),
        };

        // Initialize default detectors and analyzers
        tester.initialize_default_components()?;

        // Load existing baselines
        tester.load_baselines()?;

        Ok(tester)
    }

    /// Initialize default regression detectors and analyzers
    fn initialize_default_components(&mut self) -> Result<()> {
        // Add statistical test detector
        if self
            .config
            .detection_algorithms
            .contains(&"statistical_test".to_string())
        {
            self.detectors
                .push(Box::new(StatisticalTestDetector::new()));
        }

        // Add sliding window detector
        if self
            .config
            .detection_algorithms
            .contains(&"sliding_window".to_string())
        {
            self.detectors
                .push(Box::new(SlidingWindowDetector::new()));
        }

        // Add change point detector
        if self
            .config
            .detection_algorithms
            .contains(&"change_point".to_string())
        {
            self.detectors
                .push(Box::new(ChangePointDetector::new()));
        }

        // Add default statistical analyzers
        self.analyzers.push(Box::new(TrendAnalyzer::new()));
        self.analyzers.push(Box::new(OutlierAnalyzer::new()));

        Ok(())
    }

    /// Load existing baselines from disk
    fn load_baselines(&mut self) -> Result<()> {
        let baseline_path = self.config.baseline_dir.join("baselines.json");
        if baseline_path.exists() {
            let data = fs::read_to_string(&baseline_path)?;
            self.baselines = serde_json::from_str(&data)?;
        }
        Ok(())
    }

    /// Save baselines to disk
    fn save_baselines(&self) -> Result<()> {
        let baseline_path = self.config.baseline_dir.join("baselines.json");
        let data = serde_json::to_string_pretty(&self.baselines)?;
        fs::write(&baseline_path, data)?;
        Ok(())
    }

    /// Run regression test on benchmark result
    pub fn run_regression_test(
        &mut self,
        key: &str,
        result: &BenchmarkResult<A>,
    ) -> Result<RegressionTestResult<A>> {
        let start_time = std::time::Instant::now();

        // Extract performance metrics from benchmark result
        let metrics = self.extract_performance_metrics(result)?;

        // Create performance record
        let record = PerformanceRecord {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            commit_hash: None,
            branch: None,
            environment: TestEnvironment::default(),
            metrics: metrics.clone(),
            metadata: HashMap::new(),
        };

        // Add to database
        self.performance_db.add_record(key.to_string(), record);

        // Detect regressions
        let mut regressions = Vec::new();
        if let (Some(baseline), Some(history)) = (
            self.baselines.get(key),
            self.performance_db.get_history(key),
        ) {
            for detector in &self.detectors {
                match detector.detect_regression(baseline, &metrics, history) {
                    Ok(regression) => {
                        if regression.regression_detected {
                            regressions.push(regression);
                        }
                    }
                    Err(e) => eprintln!("Regression detection error: {}", e),
                }
            }
        }

        // Send alerts for regressions
        for regression in &regressions {
            if let Err(e) = self.alert_system.send_alert(regression) {
                eprintln!("Alert sending error: {}", e);
            }
        }

        // Update baselines if needed
        self.update_baselines(key)?;

        // Save database
        self.performance_db.save(&self.config.baseline_dir)?;
        self.save_baselines()?;

        let execution_time = start_time.elapsed().as_millis() as u64;

        Ok(RegressionTestResult {
            test_id: key.to_string(),
            status: if regressions.is_empty() {
                "passed".to_string()
            } else {
                "failed".to_string()
            },
            execution_time_ms: execution_time,
            regression_count: regressions.len(),
            regressions,
            metrics,
            baseline_comparison: self.baselines.get(key).map(|baseline| {
                self.compare_with_baseline(&metrics, baseline)
            }),
        })
    }

    /// Extract performance metrics from benchmark result
    fn extract_performance_metrics(
        &self,
        result: &BenchmarkResult<A>,
    ) -> Result<PerformanceMetrics<A>> {
        Ok(PerformanceMetrics {
            timing: TimingMetrics {
                mean_time_ns: (result.mean_time.as_nanos() as f64) as u64,
                std_time_ns: (result.std_dev.as_nanos() as f64) as u64,
                median_time_ns: (result.median_time.as_nanos() as f64) as u64,
                p95_time_ns: (result.mean_time.as_nanos() as f64 * 1.2) as u64,
                p99_time_ns: (result.mean_time.as_nanos() as f64 * 1.5) as u64,
                min_time_ns: (result.min_time.as_nanos() as f64) as u64,
                max_time_ns: (result.max_time.as_nanos() as f64) as u64,
            },
            memory: MemoryMetrics {
                peak_memory_bytes: result.memory_usage,
                avg_memory_bytes: result.memory_usage,
                allocation_count: 100, // Placeholder
                fragmentation_ratio: 0.1,
                efficiency_score: 0.9,
            },
            efficiency: EfficiencyMetrics {
                flops: 1000.0, // Placeholder
                arithmetic_intensity: 2.0,
                cache_hit_ratio: 0.95,
                cpu_utilization: 0.8,
                efficiency_score: 0.85,
                custom_metrics: HashMap::new(),
            },
            convergence: ConvergenceMetrics {
                final_objective: result.converged_value.unwrap_or(A::zero()),
                convergence_rate: 0.95,
                iterations_to_convergence: result.iterations,
                quality_score: 0.9,
                stability_score: 0.85,
            },
            custom: HashMap::new(),
        })
    }

    /// Compare metrics with baseline
    fn compare_with_baseline(
        &self,
        metrics: &PerformanceMetrics<A>,
        baseline: &PerformanceBaseline<A>,
    ) -> BaselineComparison {
        let timing_change = ((metrics.timing.mean_time_ns as f64 - baseline.baseline_stats.timing.mean)
            / baseline.baseline_stats.timing.mean)
            * 100.0;

        let memory_change = ((metrics.memory.peak_memory_bytes as f64
            - baseline.baseline_stats.memory.mean_memory)
            / baseline.baseline_stats.memory.mean_memory)
            * 100.0;

        let efficiency_change = ((metrics.efficiency.efficiency_score
            - baseline.baseline_stats.efficiency.mean_efficiency)
            / baseline.baseline_stats.efficiency.mean_efficiency)
            * 100.0;

        let status = if timing_change > self.config.degradation_threshold
            || memory_change > self.config.memory_threshold
        {
            "degraded".to_string()
        } else if timing_change < -self.config.degradation_threshold {
            "improved".to_string()
        } else {
            "stable".to_string()
        };

        BaselineComparison {
            performance_change_percent: timing_change,
            memory_change_percent: memory_change,
            efficiency_change_percent: efficiency_change,
            status,
        }
    }

    /// Update baselines based on new performance data
    fn update_baselines(&mut self, key: &str) -> Result<bool> {
        if let Some(history) = self.performance_db.get_history(key) {
            if history.len() >= self.config.min_baseline_samples {
                let new_baseline = self.calculate_baseline(history)?;
                self.baselines.insert(key.to_string(), new_baseline);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Calculate baseline from performance history
    fn calculate_baseline(
        &self,
        history: &std::collections::VecDeque<PerformanceRecord<A>>,
    ) -> Result<PerformanceBaseline<A>> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

        // Calculate timing statistics
        let timing_values: Vec<f64> = history
            .iter()
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let timing_mean = timing_values.iter().sum::<f64>() / timing_values.len() as f64;
        let timing_variance = timing_values
            .iter()
            .map(|x| (x - timing_mean).powi(2))
            .sum::<f64>()
            / (timing_values.len() - 1) as f64;
        let timing_std = timing_variance.sqrt();

        // Calculate memory statistics
        let memory_values: Vec<f64> = history
            .iter()
            .map(|r| r.metrics.memory.peak_memory_bytes as f64)
            .collect();

        let memory_mean = memory_values.iter().sum::<f64>() / memory_values.len() as f64;

        // Create baseline
        Ok(PerformanceBaseline {
            name: "auto_baseline".to_string(),
            baseline_stats: BaselineStatistics {
                timing: TimingStatistics {
                    mean: timing_mean,
                    std_dev: timing_std,
                    median: timing_mean, // Simplified
                    iqr: timing_std * 1.35,
                    coefficient_of_variation: timing_std / timing_mean,
                },
                memory: MemoryStatistics {
                    mean_memory: memory_mean,
                    std_dev_memory: memory_mean * 0.1, // Simplified
                    peak_memory_percentiles: HashMap::new(),
                    fragmentation_stats: FragmentationStatistics {
                        mean_ratio: 0.1,
                        std_dev_ratio: 0.05,
                        trend: 0.0,
                    },
                },
                efficiency: EfficiencyStatistics {
                    mean_flops: 1000.0,
                    flops_cv: 0.1,
                    mean_efficiency: 0.85,
                    custom_efficiency: HashMap::new(),
                },
                convergence: ConvergenceStatistics {
                    mean_objective: A::from(0.01).unwrap(),
                    std_objective: A::from(0.001).unwrap(),
                    mean_convergence_rate: 0.95,
                    convergence_consistency: 0.9,
                },
            },
            confidence_intervals: ConfidenceIntervals {
                timing_ci_95: (timing_mean - 1.96 * timing_std, timing_mean + 1.96 * timing_std),
                memory_ci_95: (memory_mean * 0.9, memory_mean * 1.1),
                timing_ci_99: (timing_mean - 2.58 * timing_std, timing_mean + 2.58 * timing_std),
                memory_ci_99: (memory_mean * 0.85, memory_mean * 1.15),
            },
            sample_count: history.len(),
            created_at: now,
            updated_at: now,
        })
    }

    /// Generate CI report
    pub fn generate_ci_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        match self.config.ci_report_format {
            CiReportFormat::Json => self.generate_json_report(results),
            CiReportFormat::JunitXml => self.generate_junit_xml_report(results),
            CiReportFormat::Markdown => self.generate_markdown_report(results),
            CiReportFormat::GitHubActions => self.generate_github_actions_report(results),
        }
    }

    /// Generate JSON CI report
    fn generate_json_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let passed_tests = results.iter().filter(|r| r.status == "passed").count();
        let failed_tests = results.len() - passed_tests;

        let report = CiReport {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            total_tests: results.len(),
            passed_tests,
            failed_tests,
            test_results: results.iter().map(|r| r.to_ci_test_result()).collect(),
        };

        Ok(serde_json::to_string_pretty(&report)?)
    }

    /// Generate JUnit XML report (simplified)
    fn generate_junit_xml_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let total_time: f64 = results.iter().map(|r| r.execution_time_ms as f64).sum();
        let failed_tests = results.iter().filter(|r| r.status == "failed").count();

        let mut xml = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="performance_regression" tests="{}" failures="{}" time="{:.3}">
"#,
            results.len(),
            failed_tests,
            total_time / 1000.0
        );

        for result in results {
            xml.push_str(&format!(
                r#"  <testcase name="{}" time="{:.3}""#,
                result.test_id,
                result.execution_time_ms as f64 / 1000.0
            ));

            if result.status == "failed" {
                xml.push_str(&format!(
                    r#">
    <failure message="Performance regression detected">{} regression(s) detected</failure>
  </testcase>
"#,
                    result.regression_count
                ));
            } else {
                xml.push_str(" />\n");
            }
        }

        xml.push_str("</testsuite>\n");
        Ok(xml)
    }

    /// Generate Markdown report (simplified)
    fn generate_markdown_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let mut md = String::from("# Performance Regression Test Report\n\n");

        let passed = results.iter().filter(|r| r.status == "passed").count();
        let failed = results.len() - passed;

        md.push_str(&format!(
            "- **Total Tests**: {}\n- **Passed**: {}\n- **Failed**: {}\n\n",
            results.len(),
            passed,
            failed
        ));

        md.push_str("## Test Results\n\n| Test | Status | Regressions | Time (ms) |\n|------|--------|-------------|----------|\n");

        for result in results {
            md.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                result.test_id, result.status, result.regression_count, result.execution_time_ms
            ));
        }

        Ok(md)
    }

    /// Generate GitHub Actions report (simplified)
    fn generate_github_actions_report(&self, results: &[RegressionTestResult<A>]) -> Result<String> {
        let failed_results: Vec<_> = results.iter().filter(|r| r.status == "failed").collect();

        if failed_results.is_empty() {
            Ok("::notice title=Performance Test::All performance tests passed".to_string())
        } else {
            let mut output = String::new();
            for result in failed_results {
                output.push_str(&format!(
                    "::error title=Performance Regression::{} detected {} regression(s)\n",
                    result.test_id, result.regression_count
                ));
            }
            Ok(output)
        }
    }
}

impl<A: Float> RegressionTestResult<A> {
    /// Convert to CI test result
    pub fn to_ci_test_result(&self) -> CiTestResult {
        CiTestResult {
            name: self.test_id.clone(),
            status: self.status.clone(),
            execution_time_ms: self.execution_time_ms,
            regression_count: self.regression_count,
        }
    }
}