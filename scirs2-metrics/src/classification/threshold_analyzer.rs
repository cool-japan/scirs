//! Threshold analysis for binary classification
//!
//! This module provides tools for analyzing binary classification performance
//! across different thresholds and finding optimal thresholds based on various
//! metrics and strategies.

use ndarray::{ArrayBase, Data, Dimension, Ix1};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::classification::curves::roc_curve;
use crate::error::{MetricsError, Result};

/// Metrics calculated at a specific threshold
#[derive(Debug, Clone)]
pub struct ThresholdMetrics {
    /// Classification threshold value
    pub threshold: f64,
    /// True positive rate (sensitivity, recall)
    pub tpr: f64,
    /// False positive rate (1 - specificity)
    pub fpr: f64,
    /// Precision (positive predictive value)
    pub precision: f64,
    /// F1 score (harmonic mean of precision and recall)
    pub f1_score: f64,
    /// Accuracy (correct predictions / total predictions)
    pub accuracy: f64,
    /// Specificity (true negative rate)
    pub specificity: f64,
    /// Negative predictive value
    pub npv: f64,
    /// Matthews correlation coefficient
    pub mcc: f64,
    /// Cohen's kappa coefficient
    pub kappa: f64,
    /// Youden's J statistic (sensitivity + specificity - 1)
    pub youdens_j: f64,
    /// Balanced accuracy
    pub balanced_accuracy: f64,
    /// Count of true positives
    pub tp: usize,
    /// Count of false positives
    pub fp: usize,
    /// Count of true negatives
    pub tn: usize,
    /// Count of false negatives
    pub fn_: usize,
}

/// Strategies for finding optimal threshold
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimalThresholdStrategy {
    /// Maximize F1 score
    MaxF1,
    /// Maximize Youden's J statistic (sensitivity + specificity - 1)
    YoudensJ,
    /// Maximize accuracy
    MaxAccuracy,
    /// Maximize Matthews correlation coefficient
    MaxMCC,
    /// Maximize Cohen's kappa
    MaxKappa,
    /// Balance sensitivity and specificity
    BalancedSensSpec,
    /// Balance precision and recall
    BalancedPrecRecall,
    /// Minimize distance to perfect classifier (0,1) in ROC space
    MinDistanceToOptimal,
    /// Use a specific threshold value
    Manual(f64),
}

impl Hash for OptimalThresholdStrategy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            OptimalThresholdStrategy::MaxF1 => 0.hash(state),
            OptimalThresholdStrategy::YoudensJ => 1.hash(state),
            OptimalThresholdStrategy::MaxAccuracy => 2.hash(state),
            OptimalThresholdStrategy::MaxMCC => 3.hash(state),
            OptimalThresholdStrategy::MaxKappa => 4.hash(state),
            OptimalThresholdStrategy::BalancedSensSpec => 5.hash(state),
            OptimalThresholdStrategy::BalancedPrecRecall => 6.hash(state),
            OptimalThresholdStrategy::MinDistanceToOptimal => 7.hash(state),
            OptimalThresholdStrategy::Manual(val) => {
                8.hash(state);
                val.to_bits().hash(state);
            }
        }
    }
}

impl Eq for OptimalThresholdStrategy {}

/// Analyzer for binary classification thresholds
///
/// This struct provides tools for analyzing binary classification performance
/// across different thresholds and finding optimal thresholds based on various
/// metrics and strategies.
#[derive(Debug)]
pub struct ThresholdAnalyzer {
    /// True positive rates
    tpr: Vec<f64>,
    /// False positive rates
    fpr: Vec<f64>,
    /// Thresholds
    thresholds: Vec<f64>,
    /// Raw true labels
    y_true: Vec<f64>,
    /// Raw score predictions
    y_score: Vec<f64>,
    /// Metrics at each threshold
    metrics: Option<Vec<ThresholdMetrics>>,
    /// Optimal thresholds for each strategy
    optimal_thresholds: HashMap<OptimalThresholdStrategy, usize>,
}

impl ThresholdAnalyzer {
    /// Create a new threshold analyzer from true labels and scores
    ///
    /// # Arguments
    ///
    /// * `y_true` - Binary true labels (0 or 1)
    /// * `y_score` - Predicted scores (probabilities)
    ///
    /// # Returns
    ///
    /// * `Result<ThresholdAnalyzer>` - The analyzer or error
    pub fn new<D1, D2, S1, S2>(
        y_true: &ArrayBase<S1, D1>,
        y_score: &ArrayBase<S2, D2>,
    ) -> Result<Self>
    where
        S1: Data,
        S2: Data,
        D1: Dimension,
        D2: Dimension,
        S1::Elem: Clone + Into<f64> + PartialEq,
        S2::Elem: Clone + Into<f64> + PartialOrd,
    {
        // Compute ROC curve
        let (fpr, tpr, thresholds) = roc_curve(y_true, y_score)?;

        // Convert arrays to vectors for storage
        let fpr = fpr.to_vec();
        let tpr = tpr.to_vec();
        let thresholds = thresholds.to_vec();

        // Store original data
        let y_true = y_true
            .iter()
            .map(|x| x.clone().into())
            .collect::<Vec<f64>>();
        let y_score = y_score
            .iter()
            .map(|x| x.clone().into())
            .collect::<Vec<f64>>();

        Ok(Self {
            tpr,
            fpr,
            thresholds,
            y_true,
            y_score,
            metrics: None,
            optimal_thresholds: HashMap::new(),
        })
    }

    /// Create a new threshold analyzer from pre-computed ROC curve
    ///
    /// # Arguments
    ///
    /// * `fpr` - False positive rates
    /// * `tpr` - True positive rates
    /// * `thresholds` - Thresholds
    /// * `y_true` - Binary true labels (0 or 1)
    /// * `y_score` - Predicted scores (probabilities)
    ///
    /// # Returns
    ///
    /// * `Result<ThresholdAnalyzer>` - The analyzer or error
    pub fn from_roc_curve<D1, D2, S1, S2, S3, S4, S5, D3, D4, D5>(
        fpr: &ArrayBase<S1, D1>,
        tpr: &ArrayBase<S2, D2>,
        thresholds: &ArrayBase<S3, D3>,
        y_true: &ArrayBase<S4, D4>,
        y_score: &ArrayBase<S5, D5>,
    ) -> Result<Self>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
        S3: Data<Elem = f64>,
        S4: Data,
        S5: Data,
        D1: Dimension,
        D2: Dimension,
        D3: Dimension,
        D4: Dimension,
        D5: Dimension,
        S4::Elem: Clone + Into<f64>,
        S5::Elem: Clone + Into<f64>,
    {
        // Convert arrays to vectors for storage
        let fpr = fpr.iter().cloned().collect::<Vec<f64>>();
        let tpr = tpr.iter().cloned().collect::<Vec<f64>>();
        let thresholds = thresholds.iter().cloned().collect::<Vec<f64>>();

        // Store original data
        let y_true = y_true
            .iter()
            .map(|x| x.clone().into())
            .collect::<Vec<f64>>();
        let y_score = y_score
            .iter()
            .map(|x| x.clone().into())
            .collect::<Vec<f64>>();

        // Ensure proper shape
        if fpr.len() != tpr.len() || fpr.len() != thresholds.len() {
            return Err(MetricsError::ShapeMismatch {
                shape1: format!("fpr: {}", fpr.len()),
                shape2: format!("tpr: {}, thresholds: {}", tpr.len(), thresholds.len()),
            });
        }

        Ok(Self {
            tpr,
            fpr,
            thresholds,
            y_true,
            y_score,
            metrics: None,
            optimal_thresholds: HashMap::new(),
        })
    }

    /// Calculate metrics at all thresholds
    ///
    /// # Returns
    ///
    /// * `Result<&[ThresholdMetrics]>` - Metrics at all thresholds
    pub fn calculate_metrics(&mut self) -> Result<&[ThresholdMetrics]> {
        // If metrics are already calculated, return them
        if let Some(ref metrics) = self.metrics {
            return Ok(metrics);
        }

        // Calculate metrics for each threshold
        let mut metrics = Vec::with_capacity(self.thresholds.len());

        for &threshold in self.thresholds.iter() {
            // Count TP, FP, TN, FN
            let mut tp = 0;
            let mut fp = 0;
            let mut tn = 0;
            let mut fn_ = 0;

            for (&true_val, &score) in self.y_true.iter().zip(&self.y_score) {
                let pred = if score >= threshold { 1.0 } else { 0.0 };

                match (true_val, pred) {
                    (1.0, 1.0) => tp += 1,
                    (0.0, 1.0) => fp += 1,
                    (0.0, 0.0) => tn += 1,
                    (1.0, 0.0) => fn_ += 1,
                    _ => {
                        return Err(MetricsError::InvalidArgument(format!(
                            "Invalid true value: {}",
                            true_val
                        )));
                    }
                }
            }

            // Calculate metrics
            let tpr = if tp + fn_ > 0 {
                tp as f64 / (tp + fn_) as f64
            } else {
                0.0
            };
            let fpr = if fp + tn > 0 {
                fp as f64 / (fp + tn) as f64
            } else {
                0.0
            };
            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let f1_score = if precision + tpr > 0.0 {
                2.0 * precision * tpr / (precision + tpr)
            } else {
                0.0
            };
            let accuracy = (tp + tn) as f64 / (tp + fp + tn + fn_) as f64;
            let specificity = if tn + fp > 0 {
                tn as f64 / (tn + fp) as f64
            } else {
                0.0
            };
            let npv = if tn + fn_ > 0 {
                tn as f64 / (tn + fn_) as f64
            } else {
                0.0
            };
            let youdens_j = tpr + specificity - 1.0;
            let balanced_accuracy = (tpr + specificity) / 2.0;

            // Matthews correlation coefficient
            let mcc_numerator = (tp * tn) as f64 - (fp * fn_) as f64;
            let mcc_denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)) as f64;
            let mcc = if mcc_denominator > 0.0 {
                mcc_numerator / mcc_denominator.sqrt()
            } else {
                0.0
            };

            // Cohen's kappa
            let p_o = accuracy;
            let p_e = (((tp + fp) as f64 / (tp + fp + tn + fn_) as f64)
                * ((tp + fn_) as f64 / (tp + fp + tn + fn_) as f64))
                + (((tn + fn_) as f64 / (tp + fp + tn + fn_) as f64)
                    * ((tn + fp) as f64 / (tp + fp + tn + fn_) as f64));
            let kappa = if p_e < 1.0 {
                (p_o - p_e) / (1.0 - p_e)
            } else {
                0.0
            };

            metrics.push(ThresholdMetrics {
                threshold,
                tpr,
                fpr,
                precision,
                f1_score,
                accuracy,
                specificity,
                npv,
                mcc,
                kappa,
                youdens_j,
                balanced_accuracy,
                tp,
                fp,
                tn,
                fn_,
            });
        }

        self.metrics = Some(metrics);
        Ok(self.metrics.as_ref().unwrap())
    }

    /// Find optimal threshold based on a given strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - Strategy for finding optimal threshold
    ///
    /// # Returns
    ///
    /// * `Result<(f64, ThresholdMetrics)>` - Optimal threshold and its metrics
    pub fn find_optimal_threshold(
        &mut self,
        strategy: OptimalThresholdStrategy,
    ) -> Result<(f64, ThresholdMetrics)> {
        // Check if optimal threshold for this strategy is already calculated first
        if let Some(&idx) = self.optimal_thresholds.get(&strategy) {
            self.calculate_metrics()?;
            let threshold = self.thresholds[idx];
            let metrics = self.metrics.as_ref().unwrap();
            return Ok((threshold, metrics[idx].clone()));
        }

        // Calculate metrics for finding optimal
        self.calculate_metrics()?;
        let metrics = self.metrics.as_ref().unwrap();

        // Find optimal threshold based on strategy
        let optimal_idx = match strategy {
            OptimalThresholdStrategy::MaxF1 => metrics
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.f1_score.partial_cmp(&b.f1_score).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            OptimalThresholdStrategy::YoudensJ => metrics
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.youdens_j.partial_cmp(&b.youdens_j).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            OptimalThresholdStrategy::MaxAccuracy => metrics
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.accuracy.partial_cmp(&b.accuracy).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            OptimalThresholdStrategy::MaxMCC => metrics
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.mcc.partial_cmp(&b.mcc).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            OptimalThresholdStrategy::MaxKappa => metrics
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.kappa.partial_cmp(&b.kappa).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            OptimalThresholdStrategy::BalancedSensSpec => metrics
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let a_diff = (a.tpr - a.specificity).abs();
                    let b_diff = (b.tpr - b.specificity).abs();
                    a_diff.partial_cmp(&b_diff).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            OptimalThresholdStrategy::BalancedPrecRecall => metrics
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let a_diff = (a.precision - a.tpr).abs();
                    let b_diff = (b.precision - b.tpr).abs();
                    a_diff.partial_cmp(&b_diff).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            OptimalThresholdStrategy::MinDistanceToOptimal => metrics
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let a_dist = (a.fpr.powi(2) + (1.0 - a.tpr).powi(2)).sqrt();
                    let b_dist = (b.fpr.powi(2) + (1.0 - b.tpr).powi(2)).sqrt();
                    a_dist.partial_cmp(&b_dist).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            OptimalThresholdStrategy::Manual(threshold) => {
                // Find the closest threshold
                metrics
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let a_diff = (a.threshold - threshold).abs();
                        let b_diff = (b.threshold - threshold).abs();
                        a_diff.partial_cmp(&b_diff).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
        };

        // Store the threshold and metrics before borrowing conflicts
        let threshold = self.thresholds[optimal_idx];
        let metric = metrics[optimal_idx].clone();

        // Store optimal threshold for this strategy
        self.optimal_thresholds.insert(strategy, optimal_idx);

        Ok((threshold, metric))
    }

    /// Get metrics at a specific threshold
    ///
    /// # Arguments
    ///
    /// * `threshold` - Threshold value
    ///
    /// # Returns
    ///
    /// * `Result<ThresholdMetrics>` - Metrics at the threshold
    pub fn get_metrics_at_threshold(&mut self, threshold: f64) -> Result<ThresholdMetrics> {
        // Calculate metrics if not already calculated
        self.calculate_metrics()?;

        // Find the closest threshold index
        let idx = self
            .thresholds
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                let a_diff = (a - threshold).abs();
                let b_diff = (b - threshold).abs();
                a_diff.partial_cmp(&b_diff).unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        // Get metrics - we know it's calculated at this point
        let metrics = self.metrics.as_ref().unwrap();
        Ok(metrics[idx].clone())
    }

    /// Get all threshold metrics
    ///
    /// # Returns
    ///
    /// * `Result<&[ThresholdMetrics]>` - All threshold metrics
    pub fn get_all_metrics(&mut self) -> Result<&[ThresholdMetrics]> {
        self.calculate_metrics()
    }

    /// Get thresholds
    ///
    /// # Returns
    ///
    /// * `&[f64]` - Thresholds
    pub fn get_thresholds(&self) -> &[f64] {
        &self.thresholds
    }

    /// Get false positive rates
    ///
    /// # Returns
    ///
    /// * `&[f64]` - False positive rates
    pub fn get_fpr(&self) -> &[f64] {
        &self.fpr
    }

    /// Get true positive rates
    ///
    /// # Returns
    ///
    /// * `&[f64]` - True positive rates
    pub fn get_tpr(&self) -> &[f64] {
        &self.tpr
    }

    /// Convert metrics to a specific column data structure
    ///
    /// # Arguments
    ///
    /// * `metric_name` - Name of the metric to extract
    ///
    /// # Returns
    ///
    /// * `Result<Vec<f64>>` - Values of the specified metric
    pub fn get_metric_values(&mut self, metric_name: &str) -> Result<Vec<f64>> {
        let metrics = self.calculate_metrics()?;

        let values = match metric_name {
            "threshold" => metrics.iter().map(|m| m.threshold).collect(),
            "tpr" | "recall" | "sensitivity" => metrics.iter().map(|m| m.tpr).collect(),
            "fpr" => metrics.iter().map(|m| m.fpr).collect(),
            "precision" => metrics.iter().map(|m| m.precision).collect(),
            "f1_score" | "f1" => metrics.iter().map(|m| m.f1_score).collect(),
            "accuracy" => metrics.iter().map(|m| m.accuracy).collect(),
            "specificity" => metrics.iter().map(|m| m.specificity).collect(),
            "npv" => metrics.iter().map(|m| m.npv).collect(),
            "mcc" => metrics.iter().map(|m| m.mcc).collect(),
            "kappa" => metrics.iter().map(|m| m.kappa).collect(),
            "youdens_j" | "j" => metrics.iter().map(|m| m.youdens_j).collect(),
            "balanced_accuracy" => metrics.iter().map(|m| m.balanced_accuracy).collect(),
            _ => {
                return Err(MetricsError::InvalidArgument(format!(
                    "Unknown metric: {}",
                    metric_name
                )))
            }
        };

        Ok(values)
    }

    /// Get metric names
    ///
    /// # Returns
    ///
    /// * `Vec<String>` - Names of available metrics
    pub fn get_metric_names() -> Vec<String> {
        vec![
            "threshold".to_string(),
            "tpr".to_string(),
            "fpr".to_string(),
            "precision".to_string(),
            "f1_score".to_string(),
            "accuracy".to_string(),
            "specificity".to_string(),
            "npv".to_string(),
            "mcc".to_string(),
            "kappa".to_string(),
            "youdens_j".to_string(),
            "balanced_accuracy".to_string(),
        ]
    }
}

/// Find the optimal threshold for binary classification
///
/// # Arguments
///
/// * `y_true` - Binary true labels (0 or 1)
/// * `y_score` - Predicted scores (probabilities)
/// * `strategy` - Strategy for finding optimal threshold
///
/// # Returns
///
/// * `Result<(f64, ThresholdMetrics)>` - Optimal threshold and its metrics
pub fn find_optimal_threshold<S1, S2>(
    y_true: &ArrayBase<S1, Ix1>,
    y_score: &ArrayBase<S2, Ix1>,
    strategy: OptimalThresholdStrategy,
) -> Result<(f64, ThresholdMetrics)>
where
    S1: Data,
    S2: Data,
    S1::Elem: Clone + Into<f64> + PartialEq,
    S2::Elem: Clone + Into<f64> + PartialOrd,
{
    let mut analyzer = ThresholdAnalyzer::new(y_true, y_score)?;
    let (threshold, metrics) = analyzer.find_optimal_threshold(strategy)?;
    Ok((threshold, metrics.clone()))
}

/// Get metrics at a specific threshold
///
/// # Arguments
///
/// * `y_true` - Binary true labels (0 or 1)
/// * `y_score` - Predicted scores (probabilities)
/// * `threshold` - Specific threshold to evaluate
///
/// # Returns
///
/// * `Result<ThresholdMetrics>` - Metrics at the specified threshold
pub fn threshold_metrics<S1, S2>(
    y_true: &ArrayBase<S1, Ix1>,
    y_score: &ArrayBase<S2, Ix1>,
    threshold: f64,
) -> Result<ThresholdMetrics>
where
    S1: Data,
    S2: Data,
    S1::Elem: Clone + Into<f64> + PartialEq,
    S2::Elem: Clone + Into<f64> + PartialOrd,
{
    let mut analyzer = ThresholdAnalyzer::new(y_true, y_score)?;
    let metrics = analyzer.get_metrics_at_threshold(threshold)?;
    Ok(metrics.clone())
}

/// Calculate metrics at all possible thresholds
///
/// # Arguments
///
/// * `y_true` - Binary true labels (0 or 1)
/// * `y_score` - Predicted scores (probabilities)
///
/// # Returns
///
/// * `Result<Vec<ThresholdMetrics>>` - Metrics at all thresholds
pub fn all_threshold_metrics<S1, S2>(
    y_true: &ArrayBase<S1, Ix1>,
    y_score: &ArrayBase<S2, Ix1>,
) -> Result<Vec<ThresholdMetrics>>
where
    S1: Data,
    S2: Data,
    S1::Elem: Clone + Into<f64> + PartialEq,
    S2::Elem: Clone + Into<f64> + PartialOrd,
{
    let mut analyzer = ThresholdAnalyzer::new(y_true, y_score)?;
    let metrics = analyzer.get_all_metrics()?;
    Ok(metrics.to_vec())
}
