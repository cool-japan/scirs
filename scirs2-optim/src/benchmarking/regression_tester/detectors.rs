//! Regression detection algorithms for performance testing
//!
//! This module provides various algorithms for detecting performance regressions,
//! including statistical tests, sliding window analysis, and change point detection.

use crate::benchmarking::regression_tester::types::{
    ChangePointAnalysis, OutlierAnalysis, OutlierType, PerformanceBaseline, PerformanceMetrics,
    PerformanceRecord, RegressionAnalysis, RegressionDetector, RegressionResult,
    StatisticalTestResult, TrendAnalysis, TrendDirection,
};
use crate::error::Result;
use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

/// Statistical test-based regression detector
///
/// Uses statistical hypothesis testing (t-test approximation) to detect
/// significant performance changes compared to a baseline.
#[derive(Debug)]
pub struct StatisticalTestDetector {
    /// Statistical significance threshold (alpha level)
    alpha: f64,
}

impl StatisticalTestDetector {
    /// Create a new statistical test detector with default significance level
    pub fn new() -> Self {
        Self { alpha: 0.05 }
    }

    /// Create a new statistical test detector with custom significance level
    pub fn with_alpha(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for StatisticalTestDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug> RegressionDetector<A> for StatisticalTestDetector {
    fn detect_regression(
        &self,
        baseline: &PerformanceBaseline<A>,
        current_metrics: &PerformanceMetrics<A>,
        _history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>> {
        // Simple t-test approximation for timing regression
        let current_time = current_metrics.timing.mean_time_ns as f64;
        let baseline_mean = baseline.baseline_stats.timing.mean;
        let baseline_std = baseline.baseline_stats.timing.std_dev;

        let change_percent = ((current_time - baseline_mean) / baseline_mean) * 100.0;

        // Calculate memory change percentage
        let current_memory = current_metrics.memory.peak_memory_bytes as f64;
        let baseline_memory_mean = baseline.baseline_stats.memory.mean_memory;
        let memory_change_percent = if baseline_memory_mean > 0.0 {
            ((current_memory - baseline_memory_mean) / baseline_memory_mean) * 100.0
        } else {
            0.0
        };

        // Simple z-score calculation
        let z_score =
            (current_time - baseline_mean) / (baseline_std / (baseline.sample_count as f64).sqrt());
        let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs())); // Two-tailed test

        let regression_detected =
            p_value < self.alpha && (change_percent > 0.0 || memory_change_percent > 10.0);

        Ok(RegressionResult {
            test_id: "statistical_test".to_string(),
            regression_detected,
            severity: if regression_detected {
                (change_percent / 100.0).min(1.0)
            } else {
                0.0
            },
            confidence: 1.0 - p_value,
            performance_change_percent: change_percent,
            memory_change_percent,
            affected_metrics: if regression_detected {
                vec!["timing".to_string()]
            } else {
                vec![]
            },
            statistical_tests: vec![StatisticalTestResult {
                test_name: "t_test".to_string(),
                test_statistic: z_score,
                p_value,
                degrees_of_freedom: Some(baseline.sample_count - 1),
                conclusion: if regression_detected {
                    "Significant regression detected".to_string()
                } else {
                    "No significant change".to_string()
                },
            }],
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: if change_percent > 0.0 {
                        TrendDirection::Degrading
                    } else {
                        TrendDirection::Improving
                    },
                    magnitude: change_percent.abs(),
                    significance: 1.0 - p_value,
                    start_point: None,
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: vec![],
                    magnitudes: vec![],
                    confidences: vec![],
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints: vec![],
            },
            recommendations: if regression_detected {
                vec![
                    "Check for recent code changes that might affect performance".to_string(),
                    "Review system load and resource availability".to_string(),
                    "Consider running additional test iterations for confirmation".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn name(&self) -> &str {
        "statistical_test"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("alpha".to_string(), self.alpha.to_string());
        config
    }
}

/// Sliding window regression detector
///
/// Compares current performance against a sliding window of recent measurements
/// to detect performance degradation trends.
#[derive(Debug)]
pub struct SlidingWindowDetector {
    /// Size of the sliding window for comparison
    window_size: usize,
    /// Performance degradation threshold (percentage)
    threshold: f64,
}

impl SlidingWindowDetector {
    /// Create a new sliding window detector with default parameters
    pub fn new() -> Self {
        Self {
            window_size: 10,
            threshold: 5.0, // 5% threshold
        }
    }

    /// Create a new sliding window detector with custom parameters
    pub fn with_params(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
        }
    }
}

impl Default for SlidingWindowDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug> RegressionDetector<A> for SlidingWindowDetector {
    fn detect_regression(
        &self,
        _baseline: &PerformanceBaseline<A>,
        current_metrics: &PerformanceMetrics<A>,
        history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>> {
        if history.len() < self.window_size {
            return Ok(RegressionResult {
                test_id: "sliding_window".to_string(),
                regression_detected: false,
                severity: 0.0,
                confidence: 0.0,
                performance_change_percent: 0.0,
                memory_change_percent: 0.0,
                affected_metrics: vec![],
                statistical_tests: vec![],
                analysis: RegressionAnalysis {
                    trend_analysis: TrendAnalysis {
                        direction: TrendDirection::Stable,
                        magnitude: 0.0,
                        significance: 0.0,
                        start_point: None,
                    },
                    change_point_analysis: ChangePointAnalysis {
                        change_points: vec![],
                        magnitudes: vec![],
                        confidences: vec![],
                    },
                    outlier_analysis: OutlierAnalysis {
                        outlier_indices: vec![],
                        outlier_scores: vec![],
                        outlier_types: vec![],
                    },
                    root_cause_hints: vec![
                        "Insufficient data for sliding window analysis".to_string()
                    ],
                },
                recommendations: vec![
                    "Collect more performance data for accurate analysis".to_string()
                ],
            });
        }

        // Calculate average of recent window
        let recent_times: Vec<f64> = history
            .iter()
            .rev()
            .take(self.window_size)
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let recent_avg = recent_times.iter().sum::<f64>() / recent_times.len() as f64;
        let current_time = current_metrics.timing.mean_time_ns as f64;

        let change_percent = ((current_time - recent_avg) / recent_avg) * 100.0;
        let regression_detected = change_percent > self.threshold;

        Ok(RegressionResult {
            test_id: "sliding_window".to_string(),
            regression_detected,
            severity: if regression_detected {
                (change_percent / 100.0).min(1.0)
            } else {
                0.0
            },
            confidence: if regression_detected { 0.8 } else { 0.2 },
            performance_change_percent: change_percent,
            memory_change_percent: 0.0,
            affected_metrics: if regression_detected {
                vec!["timing".to_string()]
            } else {
                vec![]
            },
            statistical_tests: vec![],
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: if change_percent > 0.0 {
                        TrendDirection::Degrading
                    } else {
                        TrendDirection::Improving
                    },
                    magnitude: change_percent.abs(),
                    significance: if regression_detected { 0.8 } else { 0.2 },
                    start_point: Some(history.len() - self.window_size),
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: vec![],
                    magnitudes: vec![],
                    confidences: vec![],
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints: vec![],
            },
            recommendations: if regression_detected {
                vec![
                    "Performance degradation detected in recent window".to_string(),
                    "Compare current run with recent baseline".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn name(&self) -> &str {
        "sliding_window"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("window_size".to_string(), self.window_size.to_string());
        config.insert("threshold".to_string(), self.threshold.to_string());
        config
    }
}

/// Change point detection regression detector
///
/// Detects significant changes in performance characteristics by analyzing
/// historical data for change points that indicate performance shifts.
#[derive(Debug)]
pub struct ChangePointDetector {
    /// Minimum segment size for change point analysis
    min_segment_size: usize,
    /// Statistical significance threshold for change detection
    significance_threshold: f64,
}

impl ChangePointDetector {
    /// Create a new change point detector with default parameters
    pub fn new() -> Self {
        Self {
            min_segment_size: 5,
            significance_threshold: 0.05,
        }
    }

    /// Create a new change point detector with custom parameters
    pub fn with_params(min_segment_size: usize, significance_threshold: f64) -> Self {
        Self {
            min_segment_size,
            significance_threshold,
        }
    }
}

impl Default for ChangePointDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug> RegressionDetector<A> for ChangePointDetector {
    fn detect_regression(
        &self,
        _baseline: &PerformanceBaseline<A>,
        _current_metrics: &PerformanceMetrics<A>,
        history: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<RegressionResult<A>> {
        // Simplified change point detection using variance change
        if history.len() < 2 * self.min_segment_size {
            return Ok(RegressionResult {
                test_id: "change_point".to_string(),
                regression_detected: false,
                severity: 0.0,
                confidence: 0.0,
                performance_change_percent: 0.0,
                memory_change_percent: 0.0,
                affected_metrics: vec![],
                statistical_tests: vec![],
                analysis: RegressionAnalysis {
                    trend_analysis: TrendAnalysis {
                        direction: TrendDirection::Stable,
                        magnitude: 0.0,
                        significance: 0.0,
                        start_point: None,
                    },
                    change_point_analysis: ChangePointAnalysis {
                        change_points: vec![],
                        magnitudes: vec![],
                        confidences: vec![],
                    },
                    outlier_analysis: OutlierAnalysis {
                        outlier_indices: vec![],
                        outlier_scores: vec![],
                        outlier_types: vec![],
                    },
                    root_cause_hints: vec![
                        "Insufficient data for change point detection".to_string()
                    ],
                },
                recommendations: vec![
                    "Collect more performance data for change point analysis".to_string()
                ],
            });
        }

        // Simple change point detection - compare first and second half
        let mid_point = history.len() / 2;

        let first_half: Vec<f64> = history
            .iter()
            .take(mid_point)
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let second_half: Vec<f64> = history
            .iter()
            .skip(mid_point)
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let first_mean = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_mean = second_half.iter().sum::<f64>() / second_half.len() as f64;

        let change_percent =
            A::from((second_mean - first_mean) / first_mean).unwrap() * A::from(100.0).unwrap();
        let change_detected = change_percent.abs() > A::from(5.0).unwrap(); // 5% change threshold

        Ok(RegressionResult {
            test_id: "change_point".to_string(),
            regression_detected: change_detected && change_percent > A::zero(),
            severity: if change_detected {
                (change_percent.abs() / A::from(100.0).unwrap())
                    .min(A::one())
                    .to_f64()
                    .unwrap_or(0.0)
            } else {
                0.0
            },
            confidence: if change_detected { 0.7 } else { 0.3 },
            performance_change_percent: change_percent.to_f64().unwrap_or(0.0),
            memory_change_percent: 0.0,
            affected_metrics: if change_detected {
                vec!["timing".to_string()]
            } else {
                vec![]
            },
            statistical_tests: vec![],
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: if change_percent > A::zero() {
                        TrendDirection::Degrading
                    } else {
                        TrendDirection::Improving
                    },
                    magnitude: change_percent.abs().to_f64().unwrap_or(0.0),
                    significance: if change_detected { 0.7 } else { 0.3 },
                    start_point: Some(mid_point),
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: if change_detected {
                        vec![mid_point]
                    } else {
                        vec![]
                    },
                    magnitudes: if change_detected {
                        vec![change_percent.to_f64().unwrap_or(0.0)]
                    } else {
                        vec![]
                    },
                    confidences: if change_detected { vec![0.7] } else { vec![] },
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints: if change_detected {
                    vec!["Significant performance change detected at mid-point".to_string()]
                } else {
                    vec![]
                },
            },
            recommendations: if change_detected {
                vec![
                    "Investigate changes that occurred around the detected change point"
                        .to_string(),
                    "Review commits and deployments near the change point".to_string(),
                ]
            } else {
                vec![]
            },
        })
    }

    fn name(&self) -> &str {
        "change_point"
    }

    fn config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert(
            "min_segment_size".to_string(),
            self.min_segment_size.to_string(),
        );
        config.insert(
            "significance_threshold".to_string(),
            self.significance_threshold.to_string(),
        );
        config
    }
}

/// Helper function to compute the cumulative distribution function of the standard normal distribution
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Simple error function approximation
///
/// This function provides a good approximation of the error function using
/// Abramowitz and Stegun's polynomial approximation.
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarking::regression_tester::config::TestEnvironment;
    use crate::benchmarking::regression_tester::types::{
        BaselineStatistics, ConfidenceIntervals, PerformanceMetrics, TimingMetrics, TimingStatistics,
        MemoryMetrics, MemoryStatistics, EfficiencyMetrics, ConvergenceMetrics,
        EfficiencyStatistics, ConvergenceStatistics, FragmentationStatistics,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_test_baseline() -> PerformanceBaseline<f64> {
        PerformanceBaseline {
            name: "test_baseline".to_string(),
            baseline_stats: BaselineStatistics {
                timing: TimingStatistics {
                    mean: 1000.0,
                    std_dev: 100.0,
                    median: 1000.0,
                    iqr: 200.0,
                    coefficient_of_variation: 0.1,
                },
                memory: MemoryStatistics {
                    mean_memory: 1000000.0,
                    std_dev_memory: 100000.0,
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
                    mean_efficiency: 0.8,
                    custom_efficiency: HashMap::new(),
                },
                convergence: ConvergenceStatistics {
                    mean_objective: 0.01,
                    std_objective: 0.001,
                    mean_convergence_rate: 0.95,
                    convergence_consistency: 0.9,
                },
            },
            confidence_intervals: ConfidenceIntervals {
                timing_ci_95: (900.0, 1100.0),
                memory_ci_95: (900000.0, 1100000.0),
                timing_ci_99: (850.0, 1150.0),
                memory_ci_99: (850000.0, 1150000.0),
            },
            sample_count: 100,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            updated_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    fn create_test_metrics(mean_time_ns: u64) -> PerformanceMetrics<f64> {
        PerformanceMetrics {
            timing: TimingMetrics {
                mean_time_ns,
                std_time_ns: 10,
                median_time_ns: mean_time_ns,
                p95_time_ns: mean_time_ns + 50,
                p99_time_ns: mean_time_ns + 100,
                min_time_ns: mean_time_ns - 20,
                max_time_ns: mean_time_ns + 200,
            },
            memory: MemoryMetrics {
                peak_memory_bytes: 1000000,
                avg_memory_bytes: 900000,
                allocation_count: 100,
                fragmentation_ratio: 0.1,
                efficiency_score: 0.9,
            },
            efficiency: EfficiencyMetrics {
                flops: 1000.0,
                arithmetic_intensity: 2.0,
                cache_hit_ratio: 0.95,
                cpu_utilization: 0.8,
                efficiency_score: 0.85,
                custom_metrics: HashMap::new(),
            },
            convergence: ConvergenceMetrics {
                final_objective: 0.01,
                convergence_rate: 0.95,
                iterations_to_convergence: Some(100),
                quality_score: 0.9,
                stability_score: 0.85,
            },
            custom: HashMap::new(),
        }
    }

    #[test]
    fn test_statistical_test_detector() {
        let detector = StatisticalTestDetector::new();
        let baseline = create_test_baseline();
        let current_metrics = create_test_metrics(1200); // 20% slower
        let history = VecDeque::new();

        let result = detector
            .detect_regression(&baseline, &current_metrics, &history)
            .unwrap();

        assert_eq!(result.test_id, "statistical_test");
        assert!(result.performance_change_percent > 15.0); // Should detect significant change
    }

    #[test]
    fn test_sliding_window_detector() {
        let detector = SlidingWindowDetector::new();
        let baseline = create_test_baseline();
        let current_metrics = create_test_metrics(1200);

        // Create history with consistent performance
        let mut history = VecDeque::new();
        for _ in 0..15 {
            let record = PerformanceRecord {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                commit_hash: None,
                branch: None,
                environment: TestEnvironment::default(),
                metrics: create_test_metrics(1000),
                metadata: HashMap::new(),
            };
            history.push_back(record);
        }

        let result = detector
            .detect_regression(&baseline, &current_metrics, &history)
            .unwrap();

        assert_eq!(result.test_id, "sliding_window");
        assert!(result.performance_change_percent > 15.0);
    }

    #[test]
    fn test_change_point_detector() {
        let detector = ChangePointDetector::new();
        let baseline = create_test_baseline();
        let current_metrics = create_test_metrics(1000);

        // Create history with a change point
        let mut history = VecDeque::new();

        // First half - good performance
        for _ in 0..10 {
            let record = PerformanceRecord {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                commit_hash: None,
                branch: None,
                environment: TestEnvironment::default(),
                metrics: create_test_metrics(1000),
                metadata: HashMap::new(),
            };
            history.push_back(record);
        }

        // Second half - degraded performance
        for _ in 0..10 {
            let record = PerformanceRecord {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                commit_hash: None,
                branch: None,
                environment: TestEnvironment::default(),
                metrics: create_test_metrics(1200), // 20% slower
                metadata: HashMap::new(),
            };
            history.push_back(record);
        }

        let result = detector
            .detect_regression(&baseline, &current_metrics, &history)
            .unwrap();

        assert_eq!(result.test_id, "change_point");
        assert!(result.analysis.change_point_analysis.change_points.len() > 0);
    }

    #[test]
    fn test_insufficient_data_handling() {
        let detector = SlidingWindowDetector::new();
        let baseline = create_test_baseline();
        let current_metrics = create_test_metrics(1000);
        let history = VecDeque::new(); // Empty history

        let result = detector
            .detect_regression(&baseline, &current_metrics, &history)
            .unwrap();

        assert!(!result.regression_detected);
        assert!(result
            .analysis
            .root_cause_hints
            .iter()
            .any(|hint| hint.contains("Insufficient data")));
    }

    #[test]
    fn test_normal_cdf_function() {
        // Test normal CDF at known points
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(-2.0) < 0.05);
        assert!(normal_cdf(2.0) > 0.95);
    }

    #[test]
    fn test_detector_configuration() {
        let detector = StatisticalTestDetector::with_alpha(0.01);
        let config = detector.config();
        assert_eq!(config.get("alpha"), Some(&"0.01".to_string()));

        let detector = SlidingWindowDetector::with_params(20, 10.0);
        let config = detector.config();
        assert_eq!(config.get("window_size"), Some(&"20".to_string()));
        assert_eq!(config.get("threshold"), Some(&"10".to_string()));
    }
}