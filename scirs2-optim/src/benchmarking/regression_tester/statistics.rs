//! Statistical analysis implementations for regression testing
//!
//! This module provides statistical analyzers for identifying trends, outliers,
//! and patterns in performance data to support regression detection.

use crate::benchmarking::regression_tester::types::{
    PerformanceRecord, StatisticalAnalysisResult, StatisticalAnalyzer,
};
use crate::error::Result;
use num_traits::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

/// Trend analysis implementation
///
/// Analyzes performance data to detect linear trends that may indicate
/// gradual performance improvements or degradations over time.
#[derive(Debug)]
pub struct TrendAnalyzer {
    /// Minimum number of data points required for analysis
    min_data_points: usize,
}

impl TrendAnalyzer {
    /// Create a new trend analyzer with default parameters
    pub fn new() -> Self {
        Self { min_data_points: 5 }
    }

    /// Create a new trend analyzer with custom minimum data points
    pub fn with_min_data_points(min_data_points: usize) -> Self {
        Self { min_data_points }
    }
}

impl Default for TrendAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug> StatisticalAnalyzer<A> for TrendAnalyzer {
    fn analyze(
        &self,
        data: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<StatisticalAnalysisResult<A>> {
        if data.len() < self.min_data_points {
            return Ok(StatisticalAnalysisResult {
                summary: "Insufficient data for trend analysis".to_string(),
                tests: vec![],
                patterns: vec!["Insufficient data".to_string()],
                anomalies: vec![],
            });
        }

        // Simple linear trend analysis using least squares regression
        let times: Vec<f64> = data
            .iter()
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        let n = times.len() as f64;
        let x_sum: f64 = (0..times.len()).map(|i| i as f64).sum();
        let y_sum: f64 = times.iter().sum();
        let xy_sum: f64 = times.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..times.len()).map(|i| (i as f64).powi(2)).sum();

        // Calculate slope using least squares formula
        let slope = if n * x2_sum - x_sum.powi(2) != 0.0 {
            (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2))
        } else {
            0.0
        };

        // Calculate correlation coefficient to assess trend strength
        let mean_x = x_sum / n;
        let mean_y = y_sum / n;

        let numerator: f64 = times.iter().enumerate()
            .map(|(i, &y)| (i as f64 - mean_x) * (y - mean_y))
            .sum();

        let denominator_x: f64 = (0..times.len())
            .map(|i| (i as f64 - mean_x).powi(2))
            .sum();

        let denominator_y: f64 = times.iter()
            .map(|&y| (y - mean_y).powi(2))
            .sum();

        let correlation = if denominator_x > 0.0 && denominator_y > 0.0 {
            numerator / (denominator_x * denominator_y).sqrt()
        } else {
            0.0
        };

        // Determine trend direction and strength
        let (trend_direction, trend_strength) = match (slope, correlation.abs()) {
            (s, c) if s > 1.0 && c > 0.7 => ("strongly increasing", "strong"),
            (s, c) if s > 0.1 && c > 0.5 => ("increasing", "moderate"),
            (s, c) if s > 0.0 && c > 0.3 => ("slightly increasing", "weak"),
            (s, c) if s < -1.0 && c > 0.7 => ("strongly decreasing", "strong"),
            (s, c) if s < -0.1 && c > 0.5 => ("decreasing", "moderate"),
            (s, c) if s < 0.0 && c > 0.3 => ("slightly decreasing", "weak"),
            _ => ("stable", "none"),
        };

        // Calculate relative change
        let relative_change = if !times.is_empty() && times[0] != 0.0 {
            ((times[times.len() - 1] - times[0]) / times[0]) * 100.0
        } else {
            0.0
        };

        Ok(StatisticalAnalysisResult {
            summary: format!(
                "Trend analysis: {} trend with slope {:.2} (correlation: {:.3}, change: {:.2}%)",
                trend_direction, slope, correlation, relative_change
            ),
            tests: vec![],
            patterns: vec![
                format!("Linear trend: {}", trend_direction),
                format!("Trend strength: {}", trend_strength),
                format!("Correlation coefficient: {:.3}", correlation),
                format!("Relative change: {:.2}%", relative_change),
            ],
            anomalies: vec![],
        })
    }

    fn name(&self) -> &str {
        "trend_analyzer"
    }
}

/// Outlier detection analyzer
///
/// Detects anomalous performance measurements that deviate significantly
/// from the expected distribution using statistical methods.
#[derive(Debug)]
pub struct OutlierAnalyzer {
    /// Z-score threshold for outlier detection (standard deviations)
    z_threshold: f64,
}

impl OutlierAnalyzer {
    /// Create a new outlier analyzer with default threshold
    pub fn new() -> Self {
        Self { z_threshold: 2.0 }
    }

    /// Create a new outlier analyzer with custom Z-score threshold
    pub fn with_threshold(z_threshold: f64) -> Self {
        Self { z_threshold }
    }
}

impl Default for OutlierAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Float + Debug> StatisticalAnalyzer<A> for OutlierAnalyzer {
    fn analyze(
        &self,
        data: &VecDeque<PerformanceRecord<A>>,
    ) -> Result<StatisticalAnalysisResult<A>> {
        if data.is_empty() {
            return Ok(StatisticalAnalysisResult {
                summary: "No data for outlier analysis".to_string(),
                tests: vec![],
                patterns: vec!["No data available".to_string()],
                anomalies: vec![],
            });
        }

        if data.len() < 3 {
            return Ok(StatisticalAnalysisResult {
                summary: "Insufficient data for reliable outlier detection".to_string(),
                tests: vec![],
                patterns: vec!["Insufficient data for statistical analysis".to_string()],
                anomalies: vec![],
            });
        }

        let times: Vec<f64> = data
            .iter()
            .map(|r| r.metrics.timing.mean_time_ns as f64)
            .collect();

        // Calculate statistical measures
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (times.len() - 1) as f64;
        let std_dev = variance.sqrt();

        // Prevent division by zero
        if std_dev == 0.0 {
            return Ok(StatisticalAnalysisResult {
                summary: "No variation in data - all values identical".to_string(),
                tests: vec![],
                patterns: vec!["Zero variance in performance data".to_string()],
                anomalies: vec![],
            });
        }

        // Detect outliers using Z-score method
        let mut outliers = Vec::new();
        let mut outlier_indices = Vec::new();
        let mut severe_outliers = 0;
        let mut moderate_outliers = 0;

        for (i, &time) in times.iter().enumerate() {
            let z_score = (time - mean) / std_dev;
            let abs_z_score = z_score.abs();

            if abs_z_score > self.z_threshold {
                outliers.push(A::from(z_score).unwrap());
                outlier_indices.push(i);

                if abs_z_score > 3.0 {
                    severe_outliers += 1;
                } else {
                    moderate_outliers += 1;
                }
            }
        }

        // Calculate additional statistics
        let outlier_percentage = (outliers.len() as f64 / times.len() as f64) * 100.0;

        // Calculate interquartile range (IQR) for additional context
        let mut sorted_times = times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1_idx = sorted_times.len() / 4;
        let q3_idx = 3 * sorted_times.len() / 4;
        let q1 = sorted_times[q1_idx];
        let q3 = sorted_times[q3_idx];
        let iqr = q3 - q1;

        // Determine outlier severity and patterns
        let severity = if severe_outliers > 0 {
            "severe"
        } else if moderate_outliers > 2 {
            "moderate"
        } else if outliers.len() > 1 {
            "mild"
        } else {
            "minimal"
        };

        let patterns = if outliers.is_empty() {
            vec!["No significant outliers detected".to_string()]
        } else {
            let mut patterns = vec![
                format!("{} potential outliers found ({:.1}% of data)", outliers.len(), outlier_percentage),
                format!("Outlier severity: {}", severity),
                format!("Mean: {:.2}, Std Dev: {:.2}", mean, std_dev),
                format!("IQR: {:.2} (Q1: {:.2}, Q3: {:.2})", iqr, q1, q3),
            ];

            if severe_outliers > 0 {
                patterns.push(format!("{} severe outliers (|z| > 3.0)", severe_outliers));
            }
            if moderate_outliers > 0 {
                patterns.push(format!("{} moderate outliers ({:.1} < |z| <= 3.0)", moderate_outliers, self.z_threshold));
            }

            patterns
        };

        Ok(StatisticalAnalysisResult {
            summary: format!(
                "Outlier analysis: {} outliers detected ({:.1}% of data, {} severity)",
                outliers.len(), outlier_percentage, severity
            ),
            tests: vec![],
            patterns,
            anomalies: outliers,
        })
    }

    fn name(&self) -> &str {
        "outlier_analyzer"
    }
}

/// Helper functions for statistical calculations
pub mod stats_utils {
    /// Calculate the median of a sorted slice
    pub fn median(sorted_data: &[f64]) -> f64 {
        let len = sorted_data.len();
        if len == 0 {
            return 0.0;
        }

        if len % 2 == 0 {
            (sorted_data[len / 2 - 1] + sorted_data[len / 2]) / 2.0
        } else {
            sorted_data[len / 2]
        }
    }

    /// Calculate percentile of sorted data
    pub fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }

        let index = (p / 100.0) * (sorted_data.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = index - lower as f64;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    }

    /// Calculate robust statistics (median absolute deviation)
    pub fn mad(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_val = median(&sorted_data);

        let mut deviations: Vec<f64> = data.iter()
            .map(|&x| (x - median_val).abs())
            .collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        median(&deviations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarking::regression_tester::config::TestEnvironment;
    use crate::benchmarking::regression_tester::types::PerformanceMetrics;
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_test_record(time_ns: u64) -> PerformanceRecord<f64> {
        PerformanceRecord {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            commit_hash: None,
            branch: None,
            environment: TestEnvironment::default(),
            metrics: PerformanceMetrics {
                timing: crate::benchmarking::regression_tester::types::TimingMetrics {
                    mean_time_ns: time_ns,
                    std_time_ns: 10,
                    median_time_ns: time_ns,
                    p95_time_ns: time_ns + 50,
                    p99_time_ns: time_ns + 100,
                    min_time_ns: time_ns - 20,
                    max_time_ns: time_ns + 200,
                },
                memory: Default::default(),
                efficiency: Default::default(),
                convergence: Default::default(),
                custom: HashMap::new(),
            },
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_trend_analyzer_increasing_trend() {
        let analyzer = TrendAnalyzer::new();
        let mut data = VecDeque::new();

        // Create increasing trend
        for i in 0..10 {
            data.push_back(create_test_record(1000 + i * 100));
        }

        let result = analyzer.analyze(&data).unwrap();

        assert!(result.summary.contains("increasing"));
        assert!(result.patterns.iter().any(|p| p.contains("increasing")));
    }

    #[test]
    fn test_trend_analyzer_decreasing_trend() {
        let analyzer = TrendAnalyzer::new();
        let mut data = VecDeque::new();

        // Create decreasing trend
        for i in 0..10 {
            data.push_back(create_test_record(2000 - i * 100));
        }

        let result = analyzer.analyze(&data).unwrap();

        assert!(result.summary.contains("decreasing"));
        assert!(result.patterns.iter().any(|p| p.contains("decreasing")));
    }

    #[test]
    fn test_trend_analyzer_insufficient_data() {
        let analyzer = TrendAnalyzer::new();
        let mut data = VecDeque::new();

        // Add only 3 data points (less than minimum of 5)
        for i in 0..3 {
            data.push_back(create_test_record(1000 + i * 10));
        }

        let result = analyzer.analyze(&data).unwrap();

        assert!(result.summary.contains("Insufficient data"));
        assert!(result.patterns.iter().any(|p| p.contains("Insufficient")));
    }

    #[test]
    fn test_outlier_analyzer_with_outliers() {
        let analyzer = OutlierAnalyzer::new();
        let mut data = VecDeque::new();

        // Create normal data with outliers
        for _ in 0..10 {
            data.push_back(create_test_record(1000));
        }

        // Add outliers
        data.push_back(create_test_record(2000)); // High outlier
        data.push_back(create_test_record(500));  // Low outlier

        let result = analyzer.analyze(&data).unwrap();

        assert!(result.anomalies.len() > 0);
        assert!(result.summary.contains("outliers detected"));
    }

    #[test]
    fn test_outlier_analyzer_no_outliers() {
        let analyzer = OutlierAnalyzer::new();
        let mut data = VecDeque::new();

        // Create consistent data
        for _ in 0..10 {
            data.push_back(create_test_record(1000));
        }

        let result = analyzer.analyze(&data).unwrap();

        assert_eq!(result.anomalies.len(), 0);
        assert!(result.patterns.iter().any(|p| p.contains("No significant outliers")));
    }

    #[test]
    fn test_outlier_analyzer_empty_data() {
        let analyzer = OutlierAnalyzer::new();
        let data = VecDeque::new();

        let result = analyzer.analyze(&data).unwrap();

        assert!(result.summary.contains("No data"));
        assert!(result.patterns.iter().any(|p| p.contains("No data")));
    }

    #[test]
    fn test_custom_parameters() {
        let trend_analyzer = TrendAnalyzer::with_min_data_points(3);
        assert_eq!(trend_analyzer.min_data_points, 3);

        let outlier_analyzer = OutlierAnalyzer::with_threshold(3.0);
        assert_eq!(outlier_analyzer.z_threshold, 3.0);
    }

    #[test]
    fn test_stats_utils_median() {
        use super::stats_utils::*;

        assert_eq!(median(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn test_stats_utils_percentile() {
        use super::stats_utils::*;

        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 50.0), 3.0);
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 100.0), 5.0);
    }

    #[test]
    fn test_analyzer_names() {
        let trend_analyzer = TrendAnalyzer::new();
        let outlier_analyzer = OutlierAnalyzer::new();

        assert_eq!(trend_analyzer.name(), "trend_analyzer");
        assert_eq!(outlier_analyzer.name(), "outlier_analyzer");
    }
}