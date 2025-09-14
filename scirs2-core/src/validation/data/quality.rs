//! Data quality assessment and reporting
//!
//! This module provides comprehensive data quality assessment capabilities,
//! including quality metrics calculation, issue detection, and recommendation generation.

use std::fmt;

// Core dependencies for array/matrix validation
use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive};

use super::config::{ErrorSeverity, QualityIssueType};
use crate::error::CoreError;

use serde::{Deserialize, Serialize};

/// Data quality assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityReport {
    /// Overall quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Detailed quality metrics
    pub metrics: QualityMetrics,
    /// Issues found during validation
    pub issues: Vec<QualityIssue>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Detailed quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Completeness (non-null/NaN ratio)
    pub completeness: f64,
    /// Consistency (pattern conformance)
    pub consistency: f64,
    /// Accuracy (constraint compliance)
    pub accuracy: f64,
    /// Validity (type/format correctness)
    pub validity: f64,
    /// Statistical properties
    pub statistical_summary: Option<StatisticalSummary>,
}

/// Statistical summary of numeric data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    /// Number of data points
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Number of outliers detected
    pub outliers: usize,
    /// Data distribution type (if detectable)
    pub distribution: Option<String>,
}

/// Quality issue found during validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: QualityIssueType,
    /// Location where issue was found
    pub location: String,
    /// Description of the issue
    pub description: String,
    /// Severity of the issue
    pub severity: ErrorSeverity,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Data quality analyzer
pub struct QualityAnalyzer;

impl QualityAnalyzer {
    /// Create new quality analyzer
    pub fn new() -> Self {
        Self
    }

    /// Generate comprehensive data quality report for arrays
    pub fn generate_quality_report<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        fieldname: &str,
    ) -> Result<DataQualityReport, CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + ScalarOperand + Send + Sync + FromPrimitive,
    {
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Calculate completeness (non-NaN ratio)
        let total_elements = array.len();
        let nan_count = array.iter().filter(|&&x| x.is_nan()).count();
        let completeness = if total_elements > 0 {
            (total_elements - nan_count) as f64 / total_elements as f64
        } else {
            1.0
        };

        if completeness < 0.95 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::MissingData,
                location: fieldname.to_string(),
                description: format!("Low data completeness: {:.1}%", completeness * 100.0),
                severity: if completeness < 0.8 {
                    ErrorSeverity::Error
                } else {
                    ErrorSeverity::Warning
                },
                suggestion: Some(
                    "Consider data imputation or removal of incomplete records".to_string(),
                ),
            });

            if completeness < 0.8 {
                recommendations.push("Critical: Data completeness is below 80%. Consider data quality improvement before analysis.".to_string());
            }
        }

        // Calculate validity (finite values ratio)
        let inf_count = array.iter().filter(|&&x| x.is_infinite()).count();
        let validity = if total_elements > 0 {
            (total_elements - nan_count - inf_count) as f64 / total_elements as f64
        } else {
            1.0
        };

        if validity < 1.0 {
            issues.push(QualityIssue {
                issue_type: QualityIssueType::InvalidNumeric,
                location: fieldname.to_string(),
                description: format!(
                    "Invalid numeric values detected: {:.1}% valid",
                    validity * 100.0
                ),
                severity: ErrorSeverity::Warning,
                suggestion: Some("Remove or replace NaN and infinite values".to_string()),
            });
        }

        // Statistical summary
        let statistical_summary = if total_elements > 0 && nan_count < total_elements {
            let finite_values: Vec<_> = array.iter().filter(|&&x| x.is_finite()).cloned().collect();
            if !finite_values.is_empty() {
                self.calculate_statistical_summary(&finite_values)?
            } else {
                None
            }
        } else {
            None
        };

        // Detect outliers if we have statistical summary
        if let Some(ref stats) = statistical_summary {
            let outlier_issues = self.detect_outliers(array, stats, fieldname)?;
            issues.extend(outlier_issues);
        }

        // Calculate overall quality score
        let consistency = self.calculate_consistency(array)?;
        let accuracy = if issues
            .iter()
            .any(|i| matches!(i.issue_type, QualityIssueType::ConstraintViolation))
        {
            0.8
        } else {
            1.0
        };

        let quality_score = (completeness + validity + consistency + accuracy) / 4.0;

        // Add performance recommendations
        if total_elements > 1_000_000 {
            recommendations.push(
                "Large dataset detected. Consider parallel processing for better performance."
                    .to_string(),
            );
        }

        if quality_score < 0.8 {
            recommendations.push(
                "Overall data quality is low. Review data collection and preprocessing procedures."
                    .to_string(),
            );
        }

        // Add specific recommendations based on issues
        self.add_specific_recommendations(&issues, &mut recommendations);

        Ok(DataQualityReport {
            quality_score,
            metrics: QualityMetrics {
                completeness,
                consistency,
                accuracy,
                validity,
                statistical_summary,
            },
            issues,
            recommendations,
        })
    }

    /// Calculate statistical summary for finite values
    fn calculate_statistical_summary<T>(
        &self,
        finite_values: &[T],
    ) -> Result<Option<StatisticalSummary>, CoreError>
    where
        T: Float + Copy + FromPrimitive,
    {
        if finite_values.is_empty() {
            return Ok(None);
        }

        let mean = finite_values.iter().fold(T::zero(), |acc, &x| acc + x)
            / num_traits::cast(finite_values.len()).unwrap_or(T::one());

        let variance = finite_values
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
            / num_traits::cast(finite_values.len()).unwrap_or(T::one());

        let std_dev = variance.sqrt();
        let min_val = finite_values
            .iter()
            .fold(finite_values[0], |acc, &x| if x < acc { x } else { acc });
        let max_val = finite_values
            .iter()
            .fold(finite_values[0], |acc, &x| if x > acc { x } else { acc });

        // Simple outlier detection using IQR method
        let mut sorted_values = finite_values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let outliers = self.count_outliers_iqr(&sorted_values);

        // Basic distribution detection
        let distribution = self.detect_distribution(&sorted_values);

        Ok(Some(StatisticalSummary {
            count: finite_values.len(),
            mean: num_traits::cast(mean).unwrap_or(0.0),
            std_dev: num_traits::cast(std_dev).unwrap_or(0.0),
            min: num_traits::cast(min_val).unwrap_or(0.0),
            max: num_traits::cast(max_val).unwrap_or(0.0),
            outliers,
            distribution,
        }))
    }

    /// Count outliers using IQR method
    fn count_outliers_iqr<T>(&self, sortedvalues: &[T]) -> usize
    where
        T: Float + Copy,
    {
        if sorted_values.len() < 4 {
            return 0;
        }

        let q1_index = sorted_values.len() / 4;
        let q3_index = 3 * sorted_values.len() / 4;
        let q1 = sorted_values[q1_index];
        let q3 = sorted_values[q3_index];
        let iqr = q3 - q1;
        let lower_bound = q1 - iqr * num_traits::cast(1.5).unwrap_or(T::one());
        let upper_bound = q3 + iqr * num_traits::cast(1.5).unwrap_or(T::one());

        sorted_values
            .iter()
            .filter(|&&x| x < lower_bound || x > upper_bound)
            .count()
    }

    /// Basic distribution detection
    fn detect_distribution<T>(&self, sortedvalues: &[T]) -> Option<String>
    where
        T: Float + Copy + FromPrimitive,
    {
        if sorted_values.len() < 10 {
            return None;
        }

        // Simple skewness calculation
        let mean = sorted_values.iter().fold(T::zero(), |acc, &x| acc + x)
            / num_traits::cast(sorted_values.len()).unwrap_or(T::one());

        let variance = sorted_values
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
            / num_traits::cast(sorted_values.len()).unwrap_or(T::one());

        let std_dev = variance.sqrt();

        if std_dev > T::zero() {
            let skewness = sorted_values
                .iter()
                .map(|&x| {
                    let diff = (x - mean) / std_dev;
                    diff * diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x)
                / num_traits::cast(sorted_values.len()).unwrap_or(T::one());

            let skewness_f64: f64 = num_traits::cast(skewness).unwrap_or(0.0);

            if skewness_f64.abs() < 0.5 {
                Some("approximately_normal".to_string())
            } else if skewness_f64 > 0.5 {
                Some("right_skewed".to_string())
            } else {
                Some("left_skewed".to_string())
            }
        } else {
            Some("constant".to_string())
        }
    }

    /// Detect outliers and create quality issues
    fn detect_outliers<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        stats: &StatisticalSummary,
        fieldname: &str,
    ) -> Result<Vec<QualityIssue>, CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug,
    {
        let mut issues = Vec::new();

        if stats.outliers > 0 {
            let outlier_percentage = (stats.outliers as f64 / stats.count as f64) * 100.0;

            if outlier_percentage > 5.0 {
                issues.push(QualityIssue {
                    issue_type: QualityIssueType::Outlier,
                    location: fieldname.to_string(),
                    description: format!(
                        "High number of outliers detected: {} ({:.1}%)",
                        stats.outliers, outlier_percentage
                    ),
                    severity: if outlier_percentage > 15.0 {
                        ErrorSeverity::Error
                    } else {
                        ErrorSeverity::Warning
                    },
                    suggestion: Some(
                        "Review outliers for data quality issues or consider outlier treatment"
                            .to_string(),
                    ),
                });
            }
        }

        Ok(issues)
    }

    /// Calculate data consistency score
    fn calculate_consistency<S, D>(&self, array: &ArrayBase<S, D>) -> Result<f64, CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float,
    {
        // Implement pattern consistency checking
        let array_size = array.len();

        if array_size < 3 {
            // Too small to check patterns
            return Ok(1.0);
        }

        let values: Vec<f64> = array.iter().filter_map(|&x| x.to_f64()).collect();

        if values.len() < 3 {
            // Not enough valid values to check patterns
            return Ok(1.0);
        }

        // Check for consistent differences (arithmetic progression)
        let mut diff_scores = Vec::new();
        for i in 1..values.len() {
            diff_scores.push(values[i] - values[i.saturating_sub(1)]);
        }

        // Calculate variance of differences
        let mean_diff = diff_scores.iter().sum::<f64>() / diff_scores.len() as f64;
        let variance = diff_scores
            .iter()
            .map(|&d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / diff_scores.len() as f64;

        // Check for periodic patterns
        let mut period_score = 1.0;
        for period in 2..((values.len() / 2).min(10)) {
            let mut matches = 0;
            let mut comparisons = 0;

            for i in period..values.len() {
                if (values[i] - values[i - period]).abs() < 1e-10 {
                    matches += 1;
                }
                comparisons += 1;
            }

            if comparisons > 0 {
                let current_score = matches as f64 / comparisons as f64;
                period_score = period_score.max(current_score);
            }
        }

        // Combine scores: lower variance in differences = higher consistency
        // Also consider periodic patterns
        let diff_consistency = if variance > 0.0 {
            (-variance.ln()).exp().clamp(0.0, 1.0)
        } else {
            1.0 // Perfect arithmetic progression
        };

        // Final score is weighted average
        let consistency_score = 0.7 * diff_consistency + 0.3 * period_score;

        Ok(consistency_score.clamp(0.0, 1.0))
    }

    /// Add specific recommendations based on detected issues
    fn add_specific_recommendations(
        &self,
        issues: &[QualityIssue],
        recommendations: &mut Vec<String>,
    ) {
        let has_missing_data = issues
            .iter()
            .any(|i| matches!(i.issue_type, QualityIssueType::MissingData));
        let has_invalid_numeric = issues
            .iter()
            .any(|i| matches!(i.issue_type, QualityIssueType::InvalidNumeric));
        let has_outliers = issues
            .iter()
            .any(|i| matches!(i.issue_type, QualityIssueType::Outlier));

        if has_missing_data {
            recommendations.push("Consider using imputation techniques (mean, median, mode, or forward-fill) for missing values.".to_string());
        }

        if has_invalid_numeric {
            recommendations
                .push("Remove or replace NaN and infinite values before analysis.".to_string());
        }

        if has_outliers {
            recommendations.push(
                "Investigate outliers - they may indicate data errors or interesting edge cases."
                    .to_string(),
            );
        }

        if has_missing_data && has_invalid_numeric {
            recommendations.push("Consider a comprehensive data cleaning pipeline to address multiple quality issues.".to_string());
        }
    }
}

impl Default for QualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl DataQualityReport {
    /// Get formatted report string
    pub fn formatted_report(&self) -> String {
        let mut report = "Data Quality Report\n".to_string();
        report.push_str("==================\n\n");
        report.push_str(&format!(
            "Overall Quality Score: {:.2}\n\n",
            self.quality_score
        ));

        report.push_str("Metrics:\n");
        report.push_str(&format!(
            "  Completeness: {:.1}%\n",
            self.metrics.completeness * 100.0
        ));
        report.push_str(&format!(
            "  Validity: {:.1}%\n",
            self.metrics.validity * 100.0
        ));
        report.push_str(&format!(
            "  Consistency: {:.1}%\n",
            self.metrics.consistency * 100.0
        ));
        report.push_str(&format!(
            "  Accuracy: {:.1}%\n\n",
            self.metrics.accuracy * 100.0
        ));

        if let Some(ref stats) = self.metrics.statistical_summary {
            report.push_str("Statistical Summary:\n");
            report.push_str(&format!("  Count: {}\n", stats.count));
            report.push_str(&format!("  Mean: {:.6}\n", stats.mean));
            report.push_str(&format!("  Std Dev: {:.6}\n", stats.std_dev));
            report.push_str(&format!("  Min: {:.6}\n", stats.min));
            report.push_str(&format!("  Max: {:.6}\n", stats.max));
            report.push_str(&format!("  Outliers: {}\n", stats.outliers));
            if let Some(ref dist) = stats.distribution {
                report.push_str(&format!("  Distribution: {}\n", dist));
            }
            report.push('\n');
        }

        if !self.issues.is_empty() {
            report.push_str("Issues Found:\n");
            for (i, issue) in self.issues.iter().enumerate() {
                report.push_str(&format!(
                    "  {}. [{:?}] {}: {}\n",
                    i + 1,
                    issue.severity,
                    issue.location,
                    issue.description
                ));
                if let Some(ref suggestion) = issue.suggestion {
                    report.push_str(&format!("     Suggestion: {}\n", suggestion));
                }
            }
            report.push('\n');
        }

        if !self.recommendations.is_empty() {
            report.push_str("Recommendations:\n");
            for (i, rec) in self.recommendations.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, rec));
            }
        }

        report
    }

    /// Check if quality is acceptable (score >= threshold)
    pub fn is_acceptable(&self, threshold: f64) -> bool {
        self.quality_score >= threshold
    }

    /// Get critical issues
    pub fn get_critical_issues(&self) -> Vec<&QualityIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == ErrorSeverity::Critical)
            .collect()
    }

    /// Get issues by type
    pub fn get_issues_by_type(&self, issuetype: QualityIssueType) -> Vec<&QualityIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.issue_type == issue_type)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_quality_analyzer() {
        let analyzer = QualityAnalyzer::new();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let report = analyzer
            .generate_quality_report(&array, "test_field")
            .unwrap();

        assert!(report.quality_score > 0.9); // Should be high quality
        assert_eq!(report.metrics.completeness, 1.0); // No missing values
        assert_eq!(report.metrics.validity, 1.0); // No invalid values
        assert!(report.issues.is_empty()); // No issues expected
    }

    #[test]
    fn test_quality_with_missing_data() {
        let analyzer = QualityAnalyzer::new();
        let array = Array1::from_vec(vec![1.0, f64::NAN, 3.0, 4.0, 5.0]);

        let report = analyzer
            .generate_quality_report(&array, "test_field")
            .unwrap();

        assert!(report.metrics.completeness < 1.0); // Has missing values
        assert!(!report.issues.is_empty()); // Should have issues

        let missing_issues = report.get_issues_by_type(QualityIssueType::MissingData);
        assert!(!missing_issues.is_empty());
    }

    #[test]
    fn test_quality_with_infinite_values() {
        let analyzer = QualityAnalyzer::new();
        let array = Array1::from_vec(vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0]);

        let report = analyzer
            .generate_quality_report(&array, "test_field")
            .unwrap();

        assert!(report.metrics.validity < 1.0); // Has invalid values

        let invalid_issues = report.get_issues_by_type(QualityIssueType::InvalidNumeric);
        assert!(!invalid_issues.is_empty());
    }

    #[test]
    fn test_statistical_summary() {
        let analyzer = QualityAnalyzer::new();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let report = analyzer
            .generate_quality_report(&array, "test_field")
            .unwrap();

        assert!(report.metrics.statistical_summary.is_some());
        let stats = report.metrics.statistical_summary.unwrap();
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_formatted_report() {
        let analyzer = QualityAnalyzer::new();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let report = analyzer
            .generate_quality_report(&array, "test_field")
            .unwrap();
        let formatted = report.formatted_report();

        assert!(formatted.contains("Data Quality Report"));
        assert!(formatted.contains("Overall Quality Score"));
        assert!(formatted.contains("Metrics:"));
        assert!(formatted.contains("Statistical Summary:"));
    }

    #[test]
    fn test_quality_acceptance() {
        let analyzer = QualityAnalyzer::new();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let report = analyzer
            .generate_quality_report(&array, "test_field")
            .unwrap();

        assert!(report.is_acceptable(0.8)); // Should pass 80% threshold
        assert!(report.is_acceptable(0.9)); // Should pass 90% threshold
        assert!(report.get_critical_issues().is_empty()); // No critical issues
    }
}
