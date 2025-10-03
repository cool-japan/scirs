//! Interactive dataset exploration and analysis tools
//!
//! This module provides CLI tools and utilities for exploring datasets interactively:
//! - Dataset summary and statistics
//! - Data visualization and plotting
//! - Interactive data filtering and querying
//! - Export functionality for exploration results

use std::collections::HashMap;
use std::io::{self, Write};

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;

/// Configuration for dataset exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExploreConfig {
    /// Output format for results
    pub output_format: OutputFormat,
    /// Number of decimal places for numerical output
    pub precision: usize,
    /// Whether to show detailed statistics
    pub show_detailed_stats: bool,
    /// Maximum number of unique values to show for categorical data
    pub max_unique_values: usize,
    /// Enable interactive mode
    pub interactive: bool,
}

impl Default for ExploreConfig {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::Table,
            precision: 3,
            show_detailed_stats: true,
            max_unique_values: 20,
            interactive: false,
        }
    }
}

/// Output format for exploration results
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Plain text table format
    Table,
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Markdown format
    Markdown,
}

/// Dataset exploration summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSummary {
    /// Basic dataset information
    pub info: DatasetInfo,
    /// Statistical summary of features
    pub statistics: FeatureStatistics,
    /// Missing data analysis
    pub missingdata: MissingDataAnalysis,
    /// Target variable analysis (if available)
    pub targetanalysis: Option<TargetAnalysis>,
    /// Data quality assessment
    pub quality_assessment: QualityAssessment,
}

/// Basic dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Feature names
    pub featurenames: Option<Vec<String>>,
    /// Target names
    pub targetnames: Option<Vec<String>>,
    /// Dataset description
    pub description: Option<String>,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

/// Statistical summary of features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    /// Per-feature statistics
    pub features: Vec<FeatureStats>,
    /// Correlation matrix (for numerical features)
    pub correlations: Option<Array2<f64>>,
}

/// Statistics for a single feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    /// Feature name
    pub name: String,
    /// Feature index
    pub index: usize,
    /// Data type inference
    pub data_type: InferredDataType,
    /// Basic statistics
    pub count: usize,
    /// Mean value (for numerical data)
    pub mean: Option<f64>,
    /// Standard deviation (for numerical data)
    pub std: Option<f64>,
    /// Minimum value (for numerical data)
    pub min: Option<f64>,
    /// Maximum value (for numerical data)
    pub max: Option<f64>,
    /// Median value (for numerical data)
    pub median: Option<f64>,
    /// Percentiles (25%, 75%)
    pub q25: Option<f64>,
    /// 75th percentile
    pub q75: Option<f64>,
    /// Unique values (for categorical data)
    pub unique_count: Option<usize>,
    /// List of unique values (for categorical data with few values)
    pub unique_values: Option<Vec<String>>,
    /// Missing data count
    pub missing_count: usize,
}

/// Inferred data type for a feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferredDataType {
    /// Continuous numerical data
    Numerical,
    /// Categorical/string data
    Categorical,
    /// Binary data (0/1 or true/false)
    Binary,
    /// Unknown data type
    Unknown,
}

/// Missing data analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingDataAnalysis {
    /// Total missing values
    pub total_missing: usize,
    /// Missing percentage
    pub missing_percentage: f64,
    /// Per-feature missing counts
    pub feature_missing: Vec<(String, usize, f64)>,
    /// Missing data patterns
    pub missing_patterns: Vec<MissingPattern>,
}

/// Missing data pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingPattern {
    /// Pattern description (which features are missing)
    pub pattern: Vec<bool>,
    /// Number of samples with this pattern
    pub count: usize,
    /// Percentage of samples with this pattern
    pub percentage: f64,
}

/// Target variable analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetAnalysis {
    /// Target variable statistics
    pub target_stats: FeatureStats,
    /// Class distribution (for classification)
    pub class_distribution: Option<HashMap<String, usize>>,
    /// Target-feature correlations
    pub correlations_with_features: Vec<(String, f64)>,
}

/// Data quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score (0-100)
    pub quality_score: f64,
    /// Identified issues
    pub issues: Vec<QualityIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Data quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: IssueType,
    /// Severity level
    pub severity: Severity,
    /// Description
    pub description: String,
    /// Affected features
    pub affected_features: Vec<String>,
}

/// Type of data quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    /// Missing data values
    MissingData,
    /// Statistical outliers
    Outliers,
    /// Duplicate rows
    Duplicates,
    /// Low variance features
    LowVariance,
    /// Highly correlated features
    HighCorrelation,
    /// Imbalanced class distribution
    ImbalancedClasses,
    /// Skewed data distribution
    SkewedDistribution,
}

/// Severity level of an issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    /// Low severity issue
    Low,
    /// Medium severity issue
    Medium,
    /// High severity issue
    High,
    /// Critical severity issue
    Critical,
}

/// Dataset explorer
pub struct DatasetExplorer {
    config: ExploreConfig,
}

impl DatasetExplorer {
    /// Create a new dataset explorer
    pub fn new(config: ExploreConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ExploreConfig::default())
    }

    /// Generate a comprehensive dataset summary
    pub fn summarize(&self, dataset: &Dataset) -> Result<DatasetSummary> {
        let info = self.collect_basic_info(dataset);
        let statistics = self.compute_feature_statistics(dataset)?;
        let missingdata = self.analyze_missingdata(dataset);
        let targetanalysis = self.analyze_target(dataset)?;
        let quality_assessment = self.assess_quality(dataset, &statistics, &missingdata)?;

        Ok(DatasetSummary {
            info,
            statistics,
            missingdata,
            targetanalysis,
            quality_assessment,
        })
    }

    /// Display dataset summary in the configured format
    pub fn display_summary(&self, summary: &DatasetSummary) -> Result<()> {
        match self.config.output_format {
            OutputFormat::Table => self.display_table(summary),
            OutputFormat::Json => self.display_json(summary),
            OutputFormat::Csv => self.display_csv(summary),
            OutputFormat::Markdown => self.display_markdown(summary),
        }
    }

    /// Start interactive exploration session
    pub fn interactive_explore(&self, dataset: &Dataset) -> Result<()> {
        if !self.config.interactive {
            return Err(DatasetsError::InvalidFormat(
                "Interactive mode not enabled".to_string(),
            ));
        }

        println!("🔍 Interactive Dataset Explorer");
        println!("==============================");

        let summary = self.summarize(dataset)?;
        self.display_basic_info(&summary.info);

        loop {
            println!("\nCommands:");
            println!("  1. Summary statistics");
            println!("  2. Feature details");
            println!("  3. Missing data analysis");
            println!("  4. Target analysis");
            println!("  5. Quality assessment");
            println!("  6. Export summary");
            println!("  q. Quit");

            print!("\nEnter command: ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            match input {
                "1" => self.display_statistics(&summary.statistics)?,
                "2" => self.interactive_feature_details(dataset, &summary.statistics)?,
                "3" => self.display_missingdata(&summary.missingdata)?,
                "4" => {
                    if let Some(ref targetanalysis) = summary.targetanalysis {
                        self.display_targetanalysis(targetanalysis)?;
                    } else {
                        println!("No target variable found in dataset.");
                    }
                }
                "5" => self.display_quality_assessment(&summary.quality_assessment)?,
                "6" => self.export_summary(&summary)?,
                "q" | "quit" | "exit" => break,
                _ => println!("Invalid command. Please try again."),
            }
        }

        Ok(())
    }

    // Implementation methods

    fn collect_basic_info(&self, dataset: &Dataset) -> DatasetInfo {
        let n_samples = dataset.n_samples();
        let n_features = dataset.n_features();

        // Estimate memory usage
        let data_size = n_samples * n_features * std::mem::size_of::<f64>();
        let target_size = dataset
            .target
            .as_ref()
            .map(|t| t.len() * std::mem::size_of::<f64>())
            .unwrap_or(0);
        let memory_usage = data_size + target_size;

        DatasetInfo {
            n_samples,
            n_features,
            featurenames: dataset.featurenames.clone(),
            targetnames: dataset.targetnames.clone(),
            description: dataset.description.clone(),
            memory_usage,
        }
    }

    fn compute_feature_statistics(&self, dataset: &Dataset) -> Result<FeatureStatistics> {
        let mut features = Vec::new();

        for (i, column) in dataset.data.columns().into_iter().enumerate() {
            let name = dataset
                .featurenames
                .as_ref()
                .and_then(|names| names.get(i))
                .cloned()
                .unwrap_or_else(|| format!("feature_{i}"));

            let stats = self.compute_single_feature_stats(&name, i, &column)?;
            features.push(stats);
        }

        // Compute correlation matrix for numerical features
        let correlations = if self.config.show_detailed_stats {
            Some(self.compute_correlation_matrix(dataset)?)
        } else {
            None
        };

        Ok(FeatureStatistics {
            features,
            correlations,
        })
    }

    fn compute_single_feature_stats(
        &self,
        name: &str,
        index: usize,
        column: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Result<FeatureStats> {
        let values: Vec<f64> = column.iter().copied().collect();
        let count = values.len();
        let missing_count = values.iter().filter(|&&x| x.is_nan()).count();
        let valid_values: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();

        let (mean, std, min, max, median, q25, q75) = if !valid_values.is_empty() {
            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;

            let variance = valid_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / valid_values.len() as f64;
            let std = variance.sqrt();

            let mut sorted_values = valid_values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let min = sorted_values.first().copied();
            let max = sorted_values.last().copied();

            let median = Self::percentile(&sorted_values, 0.5);
            let q25 = Self::percentile(&sorted_values, 0.25);
            let q75 = Self::percentile(&sorted_values, 0.75);

            (Some(mean), Some(std), min, max, median, q25, q75)
        } else {
            (None, None, None, None, None, None, None)
        };

        // Infer data type
        let data_type = self.infer_data_type(&valid_values);

        // For categorical-like data, compute unique values
        let (unique_count, unique_values) = if matches!(
            data_type,
            InferredDataType::Categorical | InferredDataType::Binary
        ) {
            let mut unique: std::collections::HashSet<String> = std::collections::HashSet::new();
            for &value in &valid_values {
                unique.insert(format!("{value:.0}"));
            }

            let unique_count = unique.len();
            let unique_values = if unique_count <= self.config.max_unique_values {
                let mut values: Vec<String> = unique.into_iter().collect();
                values.sort();
                Some(values)
            } else {
                None
            };

            (Some(unique_count), unique_values)
        } else {
            (None, None)
        };

        Ok(FeatureStats {
            name: name.to_string(),
            index,
            data_type,
            count,
            mean,
            std,
            min,
            max,
            median,
            q25,
            q75,
            unique_count,
            unique_values,
            missing_count,
        })
    }

    fn percentile(sorted_values: &[f64], p: f64) -> Option<f64> {
        if sorted_values.is_empty() {
            return None;
        }

        let index = p * (sorted_values.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            Some(sorted_values[lower])
        } else {
            let weight = index - lower as f64;
            Some(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)
        }
    }

    fn infer_data_type(&self, values: &[f64]) -> InferredDataType {
        if values.is_empty() {
            return InferredDataType::Unknown;
        }

        // Check if all values are integers
        let all_integers = values.iter().all(|&x| x.fract() == 0.0);

        if all_integers {
            let unique_values: std::collections::HashSet<i64> =
                values.iter().map(|&x| x as i64).collect();

            match unique_values.len() {
                1 => InferredDataType::Unknown, // Constant
                2 => InferredDataType::Binary,
                3..=20 => InferredDataType::Categorical,
                _ => InferredDataType::Numerical,
            }
        } else {
            InferredDataType::Numerical
        }
    }

    fn compute_correlation_matrix(&self, dataset: &Dataset) -> Result<Array2<f64>> {
        let n_features = dataset.n_features();
        let mut correlations = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    correlations[[i, j]] = 1.0;
                } else {
                    let col_i = dataset.data.column(i);
                    let col_j = dataset.data.column(j);

                    let corr = self.compute_correlation(&col_i, &col_j);
                    correlations[[i, j]] = corr;
                }
            }
        }

        Ok(correlations)
    }

    fn compute_correlation(
        &self,
        x: &scirs2_core::ndarray::ArrayView1<f64>,
        y: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        let x_vals: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
        let y_vals: Vec<f64> = y.iter().copied().filter(|v| !v.is_nan()).collect();

        if x_vals.len() != y_vals.len() || x_vals.len() < 2 {
            return 0.0;
        }

        let mean_x = x_vals.iter().sum::<f64>() / x_vals.len() as f64;
        let mean_y = y_vals.iter().sum::<f64>() / y_vals.len() as f64;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (x_val, y_val) in x_vals.iter().zip(y_vals.iter()) {
            let dx = x_val - mean_x;
            let dy = y_val - mean_y;

            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn analyze_missingdata(&self, dataset: &Dataset) -> MissingDataAnalysis {
        let n_samples = dataset.n_samples();
        let n_features = dataset.n_features();
        let total_values = n_samples * n_features;

        let mut total_missing = 0;
        let mut feature_missing = Vec::new();

        for (i, column) in dataset.data.columns().into_iter().enumerate() {
            let missing_count = column.iter().filter(|&&x| x.is_nan()).count();
            total_missing += missing_count;

            let featurename = dataset
                .featurenames
                .as_ref()
                .and_then(|names| names.get(i))
                .cloned()
                .unwrap_or_else(|| format!("feature_{i}"));

            let missing_percentage = missing_count as f64 / n_samples as f64 * 100.0;
            feature_missing.push((featurename, missing_count, missing_percentage));
        }

        let missing_percentage = total_missing as f64 / total_values as f64 * 100.0;

        // Analyze missing patterns (simplified)
        let missing_patterns = self.analyze_missing_patterns(dataset);

        MissingDataAnalysis {
            total_missing,
            missing_percentage,
            feature_missing,
            missing_patterns,
        }
    }

    fn analyze_missing_patterns(&self, dataset: &Dataset) -> Vec<MissingPattern> {
        let mut pattern_counts: HashMap<Vec<bool>, usize> = HashMap::new();

        for row in dataset.data.rows() {
            let pattern: Vec<bool> = row.iter().map(|&x| x.is_nan()).collect();
            *pattern_counts.entry(pattern).or_insert(0) += 1;
        }

        let total_samples = dataset.n_samples() as f64;
        let mut patterns: Vec<MissingPattern> = pattern_counts
            .into_iter()
            .map(|(pattern, count)| MissingPattern {
                pattern,
                count,
                percentage: count as f64 / total_samples * 100.0,
            })
            .collect();

        // Sort by frequency
        patterns.sort_by(|a, b| b.count.cmp(&a.count));

        // Keep only top 10 patterns
        patterns.truncate(10);

        patterns
    }

    fn analyze_target(&self, dataset: &Dataset) -> Result<Option<TargetAnalysis>> {
        let target = match &dataset.target {
            Some(target) => target,
            None => return Ok(None),
        };

        let target_column = target.view();
        let target_stats = self.compute_single_feature_stats("target", 0, &target_column)?;

        // Compute class distribution for classification
        let class_distribution = if matches!(
            target_stats.data_type,
            InferredDataType::Categorical | InferredDataType::Binary
        ) {
            let mut distribution = HashMap::new();
            for &value in target.iter() {
                if !value.is_nan() {
                    let classname = format!("{value:.0}");
                    *distribution.entry(classname).or_insert(0) += 1;
                }
            }
            Some(distribution)
        } else {
            None
        };

        // Compute correlations with features
        let mut correlations_with_features = Vec::new();
        for (i, column) in dataset.data.columns().into_iter().enumerate() {
            let featurename = dataset
                .featurenames
                .as_ref()
                .and_then(|names| names.get(i))
                .cloned()
                .unwrap_or_else(|| format!("feature_{i}"));

            let correlation = self.compute_correlation(&column, &target_column);
            correlations_with_features.push((featurename, correlation));
        }

        // Sort by absolute correlation
        correlations_with_features.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        Ok(Some(TargetAnalysis {
            target_stats,
            class_distribution,
            correlations_with_features,
        }))
    }

    fn assess_quality(
        &self,
        _dataset: &Dataset,
        statistics: &FeatureStatistics,
        missingdata: &MissingDataAnalysis,
    ) -> Result<QualityAssessment> {
        let mut issues = Vec::new();
        let mut quality_score = 100.0;

        // Check missing _data
        if missingdata.missing_percentage > 5.0 {
            let severity = if missingdata.missing_percentage > 20.0 {
                Severity::High
            } else if missingdata.missing_percentage > 10.0 {
                Severity::Medium
            } else {
                Severity::Low
            };

            issues.push(QualityIssue {
                issue_type: IssueType::MissingData,
                severity,
                description: format!("{:.1}% of _data is missing", missingdata.missing_percentage),
                affected_features: missingdata
                    .feature_missing
                    .iter()
                    .filter(|(_, _, pct)| *pct > 5.0)
                    .map(|(name, _, _)| name.clone())
                    .collect(),
            });

            quality_score -= missingdata.missing_percentage.min(30.0);
        }

        // Check for low variance features
        let low_variance_features: Vec<String> = statistics
            .features
            .iter()
            .filter(|f| f.std.is_some_and(|std| std < 1e-6))
            .map(|f| f.name.clone())
            .collect();

        if !low_variance_features.is_empty() {
            issues.push(QualityIssue {
                issue_type: IssueType::LowVariance,
                severity: Severity::Medium,
                description: format!(
                    "{} features have very low variance",
                    low_variance_features.len()
                ),
                affected_features: low_variance_features,
            });

            quality_score -= 10.0;
        }

        // Check for highly correlated features
        if let Some(ref correlations) = statistics.correlations {
            let mut high_corr_pairs = Vec::new();
            for i in 0..correlations.nrows() {
                for j in (i + 1)..correlations.ncols() {
                    if correlations[[i, j]].abs() > 0.9 {
                        let name_i = statistics.features[i].name.clone();
                        let name_j = statistics.features[j].name.clone();
                        high_corr_pairs.push(format!("{name_i} - {name_j}"));
                    }
                }
            }

            if !high_corr_pairs.is_empty() {
                issues.push(QualityIssue {
                    issue_type: IssueType::HighCorrelation,
                    severity: Severity::Medium,
                    description: format!(
                        "{} highly correlated feature pairs found",
                        high_corr_pairs.len()
                    ),
                    affected_features: high_corr_pairs,
                });

                quality_score -= 5.0;
            }
        }

        let recommendations = self.generate_recommendations(&issues);

        Ok(QualityAssessment {
            quality_score: quality_score.max(0.0),
            issues,
            recommendations,
        })
    }

    fn generate_recommendations(&self, issues: &[QualityIssue]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for issue in issues {
            match issue.issue_type {
                IssueType::MissingData => {
                    recommendations.push("Consider imputation strategies for missing data or remove features with excessive missing values".to_string());
                }
                IssueType::LowVariance => {
                    recommendations.push(
                        "Remove low variance features as they provide little information"
                            .to_string(),
                    );
                }
                IssueType::HighCorrelation => {
                    recommendations.push("Consider removing redundant highly correlated features or use dimensionality reduction".to_string());
                }
                _ => {}
            }
        }

        if recommendations.is_empty() {
            recommendations.push("Dataset appears to be of good quality".to_string());
        }

        recommendations
    }

    // Display methods

    fn display_table(&self, summary: &DatasetSummary) -> Result<()> {
        self.display_basic_info(&summary.info);
        self.display_statistics(&summary.statistics)?;
        self.display_missingdata(&summary.missingdata)?;

        if let Some(ref targetanalysis) = summary.targetanalysis {
            self.display_targetanalysis(targetanalysis)?;
        }

        self.display_quality_assessment(&summary.quality_assessment)?;

        Ok(())
    }

    fn display_basic_info(&self, info: &DatasetInfo) {
        println!("📊 Dataset Overview");
        println!("==================");
        println!("Samples: {}", info.n_samples);
        println!("Features: {}", info.n_features);
        println!(
            "Memory usage: {:.2} MB",
            info.memory_usage as f64 / 1_048_576.0
        );

        if let Some(ref description) = info.description {
            println!("Description: {description}");
        }

        println!();
    }

    fn display_statistics(&self, statistics: &FeatureStatistics) -> Result<()> {
        println!("📈 Feature Statistics");
        println!("====================");

        // Display table header
        println!(
            "{:<15} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "Feature", "Type", "Mean", "Std", "Min", "Max", "Missing"
        );
        let separator = "-".repeat(80);
        println!("{separator}");

        for feature in &statistics.features {
            let type_str = match feature.data_type {
                InferredDataType::Numerical => "num",
                InferredDataType::Categorical => "cat",
                InferredDataType::Binary => "bin",
                InferredDataType::Unknown => "unk",
            };

            println!(
                "{:<15} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
                feature.name.chars().take(15).collect::<String>(),
                type_str,
                feature
                    .mean
                    .map(|x| format!("{x:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                feature
                    .std
                    .map(|x| format!("{x:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                feature
                    .min
                    .map(|x| format!("{x:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                feature
                    .max
                    .map(|x| format!("{x:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                feature.missing_count
            );
        }

        println!();
        Ok(())
    }

    fn display_missingdata(&self, missingdata: &MissingDataAnalysis) -> Result<()> {
        println!("❌ Missing Data Analysis");
        println!("========================");
        println!(
            "Total missing: {} ({:.2}%)",
            missingdata.total_missing, missingdata.missing_percentage
        );

        if !missingdata.feature_missing.is_empty() {
            println!("\nMissing by feature:");
            for (feature, count, percentage) in &missingdata.feature_missing {
                if *count > 0 {
                    println!("  {feature}: {count} ({percentage:.1}%)");
                }
            }
        }

        println!();
        Ok(())
    }

    fn display_targetanalysis(&self, targetanalysis: &TargetAnalysis) -> Result<()> {
        println!("🎯 Target Analysis");
        println!("==================");

        let target = &targetanalysis.target_stats;
        println!("Target type: {:?}", target.data_type);

        if let Some(ref distribution) = targetanalysis.class_distribution {
            println!("\nClass distribution:");
            for (class, count) in distribution {
                println!("  {class}: {count}");
            }
        }

        println!("\nTop correlations with features:");
        for (feature, correlation) in targetanalysis.correlations_with_features.iter().take(5) {
            println!("  {feature}: {correlation:.3}");
        }

        println!();
        Ok(())
    }

    fn display_quality_assessment(&self, quality: &QualityAssessment) -> Result<()> {
        println!("✅ Quality Assessment");
        println!("=====================");
        println!("Quality score: {:.1}/100", quality.quality_score);

        if !quality.issues.is_empty() {
            println!("\nIssues found:");
            for issue in &quality.issues {
                let severity_icon = match issue.severity {
                    Severity::Low => "⚠️",
                    Severity::Medium => "🟡",
                    Severity::High => "🟠",
                    Severity::Critical => "🔴",
                };
                println!("  {} {}", severity_icon, issue.description);
            }
        }

        println!("\nRecommendations:");
        for recommendation in &quality.recommendations {
            println!("  • {recommendation}");
        }

        println!();
        Ok(())
    }

    fn display_json(&self, summary: &DatasetSummary) -> Result<()> {
        let json = serde_json::to_string_pretty(summary)
            .map_err(|e| DatasetsError::SerdeError(e.to_string()))?;
        println!("{json}");
        Ok(())
    }

    fn display_csv(&self, summary: &DatasetSummary) -> Result<()> {
        // CSV format for feature statistics
        println!("feature,type,count,mean,std,min,max,missing");
        for feature in &summary.statistics.features {
            println!(
                "{},{:?},{},{},{},{},{},{}",
                feature.name,
                feature.data_type,
                feature.count,
                feature
                    .mean
                    .map(|x| x.to_string())
                    .unwrap_or_else(|| "".to_string()),
                feature
                    .std
                    .map(|x| x.to_string())
                    .unwrap_or_else(|| "".to_string()),
                feature
                    .min
                    .map(|x| x.to_string())
                    .unwrap_or_else(|| "".to_string()),
                feature
                    .max
                    .map(|x| x.to_string())
                    .unwrap_or_else(|| "".to_string()),
                feature.missing_count
            );
        }
        Ok(())
    }

    fn display_markdown(&self, summary: &DatasetSummary) -> Result<()> {
        println!("# Dataset Summary\n");

        println!("## Overview\n");
        println!("- **Samples**: {}", summary.info.n_samples);
        println!("- **Features**: {}", summary.info.n_features);
        println!(
            "- **Memory usage**: {:.2} MB\n",
            summary.info.memory_usage as f64 / 1_048_576.0
        );

        println!("## Feature Statistics\n");
        println!("| Feature | Type | Mean | Std | Min | Max | Missing |");
        println!("|---------|------|------|-----|-----|-----|---------|");

        for feature in &summary.statistics.features {
            println!(
                "| {} | {:?} | {} | {} | {} | {} | {} |",
                feature.name,
                feature.data_type,
                feature
                    .mean
                    .map(|x| format!("{x:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                feature
                    .std
                    .map(|x| format!("{x:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                feature
                    .min
                    .map(|x| format!("{x:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                feature
                    .max
                    .map(|x| format!("{x:.3}"))
                    .unwrap_or_else(|| "-".to_string()),
                feature.missing_count
            );
        }

        println!(
            "\n## Quality Score: {:.1}/100\n",
            summary.quality_assessment.quality_score
        );

        Ok(())
    }

    fn interactive_feature_details(
        &self,
        dataset: &Dataset,
        statistics: &FeatureStatistics,
    ) -> Result<()> {
        println!("\nFeature Details");
        println!("===============");

        for (i, feature) in statistics.features.iter().enumerate() {
            println!("{}. {}", i + 1, feature.name);
        }

        print!("\nEnter feature number (or 'back'): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input == "back" {
            return Ok(());
        }

        if let Ok(index) = input.parse::<usize>() {
            if index > 0 && index <= statistics.features.len() {
                let feature = &statistics.features[index - 1];
                self.display_feature_detail(feature, dataset)?;
            } else {
                println!("Invalid feature number.");
            }
        } else {
            println!("Invalid input.");
        }

        Ok(())
    }

    fn display_feature_detail(&self, feature: &FeatureStats, _dataset: &Dataset) -> Result<()> {
        println!("\n📊 Feature: {}", feature.name);
        println!("==================");
        println!("Type: {:?}", feature.data_type);
        println!("Count: {}", feature.count);
        println!(
            "Missing: {} ({:.1}%)",
            feature.missing_count,
            feature.missing_count as f64 / feature.count as f64 * 100.0
        );

        if let Some(mean) = feature.mean {
            println!("Mean: {mean:.6}");
        }
        if let Some(std) = feature.std {
            println!("Std: {std:.6}");
        }
        if let Some(min) = feature.min {
            println!("Min: {min:.6}");
        }
        if let Some(max) = feature.max {
            println!("Max: {max:.6}");
        }
        if let Some(median) = feature.median {
            println!("Median: {median:.6}");
        }
        if let Some(q25) = feature.q25 {
            println!("Q25: {q25:.6}");
        }
        if let Some(q75) = feature.q75 {
            println!("Q75: {q75:.6}");
        }

        if let Some(ref unique_values) = feature.unique_values {
            println!("Unique values: {unique_values:?}");
        } else if let Some(unique_count) = feature.unique_count {
            println!("Unique count: {unique_count}");
        }

        Ok(())
    }

    fn export_summary(&self, summary: &DatasetSummary) -> Result<()> {
        print!("Export format (json/csv/markdown): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let format = input.trim();

        let filename = format!("dataset_summary.{format}");

        let content = match format {
            "json" => serde_json::to_string_pretty(summary)
                .map_err(|e| DatasetsError::SerdeError(e.to_string()))?,
            "csv" => {
                let mut csv_content = String::from("feature,type,count,mean,std,min,max,missing\n");
                for feature in &summary.statistics.features {
                    csv_content.push_str(&format!(
                        "{},{:?},{},{},{},{},{},{}\n",
                        feature.name,
                        feature.data_type,
                        feature.count,
                        feature
                            .mean
                            .map(|x| x.to_string())
                            .unwrap_or_else(|| "".to_string()),
                        feature
                            .std
                            .map(|x| x.to_string())
                            .unwrap_or_else(|| "".to_string()),
                        feature
                            .min
                            .map(|x| x.to_string())
                            .unwrap_or_else(|| "".to_string()),
                        feature
                            .max
                            .map(|x| x.to_string())
                            .unwrap_or_else(|| "".to_string()),
                        feature.missing_count
                    ));
                }
                csv_content
            }
            "markdown" => {
                // Generate markdown content
                format!(
                    "# Dataset Summary\n\nQuality Score: {:.1}/100\n",
                    summary.quality_assessment.quality_score
                )
            }
            _ => {
                return Err(DatasetsError::InvalidFormat(
                    "Unsupported export format".to_string(),
                ))
            }
        };

        std::fs::write(&filename, content).map_err(DatasetsError::IoError)?;

        println!("Summary exported to: {filename}");
        Ok(())
    }
}

/// Convenience functions for dataset exploration
pub mod convenience {
    use super::*;

    /// Quick dataset summary with default configuration
    pub fn quick_summary(dataset: &Dataset) -> Result<DatasetSummary> {
        let explorer = DatasetExplorer::default_config();
        explorer.summarize(dataset)
    }

    /// Display basic dataset information
    pub fn info(dataset: &Dataset) -> Result<()> {
        let explorer = DatasetExplorer::default_config();
        let summary = explorer.summarize(dataset)?;
        explorer.display_basic_info(&summary.info);
        Ok(())
    }

    /// Start interactive exploration
    pub fn explore(dataset: &Dataset) -> Result<()> {
        let config = ExploreConfig {
            interactive: true,
            ..Default::default()
        };

        let explorer = DatasetExplorer::new(config);
        explorer.interactive_explore(dataset)
    }

    /// Export dataset summary to file
    pub fn export_summary(dataset: &Dataset, format: OutputFormat, filename: &str) -> Result<()> {
        let config = ExploreConfig {
            output_format: format,
            ..Default::default()
        };
        let output_format = config.output_format;

        let explorer = DatasetExplorer::new(config);
        let summary = explorer.summarize(dataset)?;

        let content = match output_format {
            OutputFormat::Json => serde_json::to_string_pretty(&summary)
                .map_err(|e| DatasetsError::SerdeError(e.to_string()))?,
            _ => {
                return Err(DatasetsError::InvalidFormat(
                    "Only JSON export is currently supported in convenience function".to_string(),
                ));
            }
        };

        std::fs::write(filename, content).map_err(DatasetsError::IoError)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::make_classification;

    #[test]
    fn testdataset_explorer_creation() {
        let explorer = DatasetExplorer::default_config();
        assert_eq!(explorer.config.precision, 3);
        assert!(explorer.config.show_detailed_stats);
    }

    #[test]
    fn test_basic_summary() {
        let dataset = make_classification(100, 5, 2, 1, 1, Some(42)).unwrap();
        let summary = convenience::quick_summary(&dataset).unwrap();

        assert_eq!(summary.info.n_samples, 100);
        assert_eq!(summary.info.n_features, 5);
        assert_eq!(summary.statistics.features.len(), 5);
    }

    #[test]
    fn test_feature_statistics() {
        let dataset = make_classification(50, 3, 2, 1, 1, Some(42)).unwrap();
        let explorer = DatasetExplorer::default_config();
        let statistics = explorer.compute_feature_statistics(&dataset).unwrap();

        assert_eq!(statistics.features.len(), 3);

        for feature in &statistics.features {
            assert!(feature.mean.is_some());
            assert!(feature.std.is_some());
            assert!(feature.min.is_some());
            assert!(feature.max.is_some());
        }
    }

    #[test]
    fn test_quality_assessment() {
        let dataset = make_classification(100, 4, 2, 1, 1, Some(42)).unwrap();
        let explorer = DatasetExplorer::default_config();
        let summary = explorer.summarize(&dataset).unwrap();

        // Should have high quality score for synthetic data
        assert!(summary.quality_assessment.quality_score > 80.0);
    }

    #[test]
    fn test_data_type_inference() {
        let explorer = DatasetExplorer::default_config();

        // Test numerical data
        let numerical_data = vec![1.1, 2.3, 3.7, 4.2];
        assert!(matches!(
            explorer.infer_data_type(&numerical_data),
            InferredDataType::Numerical
        ));

        // Test binary data
        let binary_data = vec![0.0, 1.0, 0.0, 1.0];
        assert!(matches!(
            explorer.infer_data_type(&binary_data),
            InferredDataType::Binary
        ));

        // Test categorical data
        let categorical_data = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        assert!(matches!(
            explorer.infer_data_type(&categorical_data),
            InferredDataType::Categorical
        ));
    }
}
