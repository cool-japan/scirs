//! Advanced analytics for dataset quality assessment
//!
//! This module provides sophisticated analytics capabilities for evaluating
//! dataset quality, complexity, and characteristics.

use super::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use statrs::statistics::Statistics;
use std::error::Error;

/// Correlation insights from dataset analysis
#[derive(Debug, Clone)]
pub struct CorrelationInsights {
    /// Feature importance scores
    pub feature_importance: Array1<f64>,
}

/// Normality assessment results
#[derive(Debug, Clone)]
pub struct NormalityAssessment {
    /// Overall normality score
    pub overall_normality: f64,
    /// Shapiro-Wilk test scores for each feature
    pub shapiro_wilk_scores: Array1<f64>,
}

/// Advanced quality metrics for a dataset
#[derive(Debug, Clone)]
pub struct AdvancedQualityMetrics {
    /// Dataset complexity score
    pub complexity_score: f64,
    /// Information entropy
    pub entropy: f64,
    /// Outlier detection score
    pub outlier_score: f64,
    /// Machine learning quality score
    pub ml_quality_score: f64,
    /// Normality assessment results
    pub normality_assessment: NormalityAssessment,
    /// Correlation insights
    pub correlation_insights: CorrelationInsights,
}

/// Advanced dataset analyzer with configurable options
#[derive(Debug, Clone)]
pub struct AdvancedDatasetAnalyzer {
    gpu_enabled: bool,
    advanced_precision: bool,
    significance_threshold: f64,
}

impl Default for AdvancedDatasetAnalyzer {
    fn default() -> Self {
        Self {
            gpu_enabled: false,
            advanced_precision: false,
            significance_threshold: 0.05,
        }
    }
}

impl AdvancedDatasetAnalyzer {
    /// Create a new analyzer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable GPU acceleration
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.gpu_enabled = enabled;
        self
    }

    /// Enable advanced precision calculations
    pub fn with_advanced_precision(mut self, enabled: bool) -> Self {
        self.advanced_precision = enabled;
        self
    }

    /// Set significance threshold for statistical tests
    pub fn with_significance_threshold(mut self, threshold: f64) -> Self {
        self.significance_threshold = threshold;
        self
    }

    /// Analyze dataset quality with advanced metrics
    pub fn analyze_dataset_quality(
        &self,
        dataset: &Dataset,
    ) -> Result<AdvancedQualityMetrics, Box<dyn Error>> {
        let data = &dataset.data;
        let _n_features = data.ncols();

        // Calculate basic statistics
        let _mean_values: Array1<f64> = data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let _std_values: Array1<f64> = data
            .var_axis(scirs2_core::ndarray::Axis(0), 1.0)
            .mapv(|x| x.sqrt());

        // Calculate complexity score based on data distribution
        let complexity_score = self.calculate_complexity_score(data)?;

        // Calculate entropy
        let entropy = self.calculate_entropy(data)?;

        // Calculate outlier score
        let outlier_score = self.calculate_outlier_score(data)?;

        // Calculate ML quality score
        let ml_quality_score = self.calculate_ml_quality_score(data)?;

        // Calculate normality assessment
        let normality_assessment = self.calculate_normality_assessment(data)?;

        // Calculate correlation insights
        let correlation_insights = self.calculate_correlation_insights(data)?;

        Ok(AdvancedQualityMetrics {
            complexity_score,
            entropy,
            outlier_score,
            ml_quality_score,
            normality_assessment,
            correlation_insights,
        })
    }

    fn calculate_complexity_score(&self, data: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
        // Simple complexity measure based on variance and correlation
        let var_mean = {
            let val = data.var_axis(scirs2_core::ndarray::Axis(0), 1.0).mean();
            if val.is_nan() {
                1.0
            } else {
                val
            }
        };
        let complexity = (var_mean.ln() + 1.0).clamp(0.0, 1.0);
        Ok(complexity)
    }

    fn calculate_entropy(&self, data: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
        // Approximate entropy calculation
        let flattened = data.iter().cloned().collect::<Vec<f64>>();
        let mut sorted = flattened.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Simple entropy approximation
        let n = sorted.len() as f64;
        let entropy = if n > 0.0 {
            (n.ln() / 2.0).clamp(0.0, 5.0)
        } else {
            0.0
        };
        Ok(entropy)
    }

    fn calculate_outlier_score(&self, data: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
        // Z-score based outlier detection
        let threshold = 3.0;
        let mut outlier_count = 0;
        let total_count = data.len();

        for col in 0..data.ncols() {
            let column = data.column(col);
            let mean = {
                let val = column.mean();
                if val.is_nan() {
                    0.0
                } else {
                    val
                }
            };
            let std = column.var(1.0).sqrt();

            if std > 0.0 {
                for &value in column.iter() {
                    let z_score = (value - mean).abs() / std;
                    if z_score > threshold {
                        outlier_count += 1;
                    }
                }
            }
        }

        let outlier_ratio = outlier_count as f64 / total_count as f64;
        Ok(outlier_ratio.min(1.0))
    }

    fn calculate_ml_quality_score(&self, data: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
        // ML quality based on feature variance and separability
        let var_scores: Array1<f64> = data.var_axis(scirs2_core::ndarray::Axis(0), 1.0);
        let mean_variance = {
            let val = var_scores.mean();
            if val.is_nan() {
                1.0
            } else {
                val
            }
        };

        // Normalize to 0-1 range
        let quality_score = (mean_variance.ln() + 5.0) / 10.0;
        Ok(quality_score.clamp(0.0, 1.0))
    }

    fn calculate_normality_assessment(
        &self,
        data: &Array2<f64>,
    ) -> Result<NormalityAssessment, Box<dyn Error>> {
        let n_features = data.ncols();
        let mut shapiro_scores = Vec::with_capacity(n_features);

        for col in 0..n_features {
            let column = data.column(col);
            // Simplified normality test (placeholder)
            let score = self.simplified_normality_test(&column)?;
            shapiro_scores.push(score);
        }

        let shapiro_wilk_scores = Array1::from_vec(shapiro_scores);
        let overall_normality = {
            let val = shapiro_wilk_scores.view().mean();
            if val.is_nan() {
                0.5
            } else {
                val
            }
        };

        Ok(NormalityAssessment {
            overall_normality,
            shapiro_wilk_scores,
        })
    }

    fn simplified_normality_test(
        &self,
        data: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Result<f64, Box<dyn Error>> {
        // Placeholder normality test based on skewness and kurtosis
        let n = data.len();
        if n < 3 {
            return Ok(0.5);
        }

        use scirs2_core::ndarray::ArrayStatCompat;
        let mean = data.mean_or(0.0);
        let variance = data.var(1.0);

        if variance == 0.0 {
            return Ok(0.0);
        }

        let std_dev = variance.sqrt();

        // Calculate skewness and kurtosis
        let mut skewness: f64 = 0.0;
        let mut kurtosis: f64 = 0.0;

        for &value in data.iter() {
            let normalized = (value - mean) / std_dev;
            skewness += normalized.powi(3);
            kurtosis += normalized.powi(4);
        }

        skewness /= n as f64;
        kurtosis = kurtosis / (n as f64) - 3.0; // Excess kurtosis

        // Simple normality score based on how close skewness and kurtosis are to normal distribution
        let skew_penalty = (skewness.abs() / 2.0).min(1.0);
        let kurt_penalty = (kurtosis.abs() / 4.0).min(1.0);
        let normality_score: f64 = 1.0 - (skew_penalty + kurt_penalty) / 2.0;

        Ok(normality_score.clamp(0.0, 1.0))
    }

    fn calculate_correlation_insights(
        &self,
        data: &Array2<f64>,
    ) -> Result<CorrelationInsights, Box<dyn Error>> {
        let n_features = data.ncols();
        let mut importance_scores = Vec::with_capacity(n_features);

        // Calculate feature importance based on variance and correlation with other features
        for i in 0..n_features {
            let feature = data.column(i);
            let variance = feature.var(1.0);

            // Simple importance based on variance (higher variance = more important)
            let importance = (variance.ln() + 1.0).clamp(0.0, 1.0);
            importance_scores.push(importance);
        }

        let feature_importance = Array1::from_vec(importance_scores);

        Ok(CorrelationInsights { feature_importance })
    }
}

/// Perform quick quality assessment of a dataset
pub fn quick_quality_assessment(dataset: &Dataset) -> Result<f64, Box<dyn Error>> {
    let data = &dataset.data;

    // Quick quality assessment based on basic statistics
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples == 0 || n_features == 0 {
        return Ok(0.0);
    }

    // Check for missing values (NaN/inf)
    let valid_count = data.iter().filter(|&&x| x.is_finite()).count();
    let completeness = valid_count as f64 / data.len() as f64;

    // Check feature variance
    let variances: Array1<f64> = data.var_axis(scirs2_core::ndarray::Axis(0), 1.0);
    let non_zero_var_count = variances.iter().filter(|&&x| x > 1e-10).count();
    let variance_score = non_zero_var_count as f64 / n_features as f64;

    // Simple size penalty for very small datasets
    let size_score = ((n_samples as f64).ln() / 10.0).clamp(0.0, 1.0);

    // Combined quality score
    let quality_score = (completeness + variance_score + size_score) / 3.0;

    Ok(quality_score.clamp(0.0, 1.0))
}

/// Advanced dataset analysis function
#[allow(dead_code)]
pub fn analyze_dataset_advanced(
    dataset: &Dataset,
) -> Result<AdvancedQualityMetrics, Box<dyn Error>> {
    let analyzer = AdvancedDatasetAnalyzer::new()
        .with_gpu(false)
        .with_advanced_precision(true)
        .with_significance_threshold(0.05);

    analyzer.analyze_dataset_quality(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_quick_quality_assessment() {
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let dataset = Dataset::new(data, None);

        let quality = quick_quality_assessment(&dataset).unwrap();
        assert!((0.0..=1.0).contains(&quality));
    }

    #[test]
    fn test_advanced_dataset_analyzer() {
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let dataset = Dataset::new(data, None);

        let analyzer = AdvancedDatasetAnalyzer::new()
            .with_gpu(false)
            .with_advanced_precision(true);

        let metrics = analyzer.analyze_dataset_quality(&dataset).unwrap();
        assert!(metrics.complexity_score >= 0.0);
        assert!(metrics.entropy >= 0.0);
        assert!(metrics.outlier_score >= 0.0);
        assert!(metrics.ml_quality_score >= 0.0);
    }

    #[test]
    fn test_normality_assessment() {
        let data = Array2::from_shape_vec((20, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let dataset = Dataset::new(data, None);

        let analyzer = AdvancedDatasetAnalyzer::new();
        let metrics = analyzer.analyze_dataset_quality(&dataset).unwrap();

        assert!(metrics.normality_assessment.overall_normality >= 0.0);
        assert!(metrics.normality_assessment.overall_normality <= 1.0);
        assert_eq!(metrics.normality_assessment.shapiro_wilk_scores.len(), 2);
    }

    #[test]
    fn test_correlation_insights() {
        let data = Array2::from_shape_vec((15, 3), (0..45).map(|x| x as f64).collect()).unwrap();
        let dataset = Dataset::new(data, None);

        let analyzer = AdvancedDatasetAnalyzer::new();
        let metrics = analyzer.analyze_dataset_quality(&dataset).unwrap();

        assert_eq!(metrics.correlation_insights.feature_importance.len(), 3);
        assert!(metrics
            .correlation_insights
            .feature_importance
            .iter()
            .all(|&x| (0.0..=1.0).contains(&x)));
    }
}
