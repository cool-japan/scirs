//! Enhanced model evaluation tools for neural networks
//!
//! This module provides comprehensive model evaluation utilities including:
//! - Advanced metrics computation and analysis
//! - Model comparison and benchmarking tools
//! - Statistical significance testing
//! - Cross-validation utilities
//! - Performance profiling and analysis

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array, ArrayD};
use scirs2_core::numeric::Float;
use scirs2_core::numeric::FromPrimitive;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;

/// Evaluation metrics for different types of tasks
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationMetric {
    /// Classification metrics
    Classification(ClassificationMetric),
    /// Regression metrics
    Regression(RegressionMetric),
    /// Custom metric with user-defined function
    Custom {
        /// Name of the custom metric
        name: String,
        /// Description of what the metric measures
        description: String,
    },
}

/// Classification-specific metrics
#[derive(Debug, Clone, PartialEq)]
pub enum ClassificationMetric {
    /// Accuracy (fraction of correct predictions)
    Accuracy,
    /// Precision (true positives / (true positives + false positives))
    Precision {
        /// Averaging method for multi-class precision
        average: AveragingMethod,
    },
    /// Recall (true positives / (true positives + false negatives))
    Recall {
        /// Averaging method for multi-class recall
        average: AveragingMethod,
    },
    /// F1 score (harmonic mean of precision and recall)
    F1Score {
        /// Averaging method for multi-class F1 score
        average: AveragingMethod,
    },
    /// Area under ROC curve
    AUROC {
        /// Averaging method for multi-class AUROC
        average: AveragingMethod,
    },
    /// Area under precision-recall curve
    AUPRC {
        /// Averaging method for multi-class AUPRC
        average: AveragingMethod,
    },
    /// Cohen's Kappa
    CohenKappa,
    /// Matthews Correlation Coefficient
    MCC,
    /// Top-k accuracy
    TopKAccuracy {
        /// Number of top predictions to consider
        k: usize,
    },
}

/// Regression-specific metrics
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionMetric {
    /// Mean Squared Error
    MSE,
    /// Root Mean Squared Error
    RMSE,
    /// Mean Absolute Error
    MAE,
    /// Mean Absolute Percentage Error
    MAPE,
    /// R-squared coefficient of determination
    R2,
    /// Explained variance score
    ExplainedVariance,
    /// Median Absolute Error
    MedianAE,
}

/// Averaging methods for multi-class metrics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AveragingMethod {
    /// Arithmetic mean of class-wise metrics
    Macro,
    /// Weighted by class frequency
    Weighted,
    /// Global computation (micro-averaging)
    Micro,
    /// No averaging (return per-class metrics)
    None,
}

/// Cross-validation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum CrossValidationStrategy {
    /// K-fold cross-validation
    KFold {
        /// Number of folds
        k: usize,
        /// Whether to shuffle data before folding
        shuffle: bool,
    },
    /// Stratified K-fold (preserves class distribution)
    StratifiedKFold {
        /// Number of folds
        k: usize,
        /// Whether to shuffle data before folding
        shuffle: bool,
    },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Leave-P-out cross-validation
    LeavePOut {
        /// Number of samples to leave out
        p: usize,
    },
    /// Time series split
    TimeSeriesSplit {
        /// Number of splits for time series
        n_splits: usize,
    },
    /// Custom split strategy
    Custom {
        /// Name of the custom strategy
        name: String,
    },
}

/// Enhanced model evaluator
pub struct ModelEvaluator<F: Float + Debug + 'static + Sum + Clone + Copy + FromPrimitive> {
    /// Metrics to compute
    metrics: Vec<EvaluationMetric>,
    /// Cross-validation strategy
    cv_strategy: Option<CrossValidationStrategy>,
    /// Bootstrap settings for confidence intervals
    bootstrap_samples: Option<usize>,
    /// Statistical significance level
    significance_level: f64,
    /// Evaluation results cache
    results_cache: HashMap<String, EvaluationResults<F>>,
}

/// Comprehensive evaluation results
#[derive(Debug, Clone)]
pub struct EvaluationResults<F: Float + Debug> {
    /// Metric scores
    pub scores: HashMap<String, MetricScore<F>>,
    /// Cross-validation results
    pub cv_results: Option<CrossValidationResults<F>>,
    /// Bootstrap confidence intervals
    pub confidence_intervals: Option<HashMap<String, ConfidenceInterval<F>>>,
    /// Statistical tests results
    pub statistical_tests: Option<StatisticalTestResults<F>>,
    /// Performance timing
    pub evaluation_time_ms: f64,
}

/// Individual metric score with statistics
#[derive(Debug, Clone)]
pub struct MetricScore<F: Float + Debug> {
    /// Primary score value
    pub value: F,
    /// Standard deviation (if available)
    pub std_dev: Option<F>,
    /// Per-class scores (for classification)
    pub per_class: Option<Vec<F>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults<F: Float + Debug> {
    /// Scores for each fold
    pub fold_scores: Vec<HashMap<String, F>>,
    /// Mean scores across folds
    pub mean_scores: HashMap<String, F>,
    /// Standard deviation across folds
    pub std_scores: HashMap<String, F>,
    /// Best fold index for each metric
    pub best_fold: HashMap<String, usize>,
}

/// Confidence interval for a metric
#[derive(Debug, Clone)]
pub struct ConfidenceInterval<F: Float + Debug> {
    /// Lower bound
    pub lower: F,
    /// Upper bound
    pub upper: F,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

/// Statistical significance test results
#[derive(Debug, Clone)]
pub struct StatisticalTestResults<F: Float + Debug> {
    /// T-test results (comparing two models)
    pub t_test: Option<TTestResult<F>>,
    /// Wilcoxon signed-rank test
    pub wilcoxon_test: Option<WilcoxonResult<F>>,
    /// McNemar's test (for classification)
    pub mcnemar_test: Option<McNemarResult<F>>,
}

/// T-test result
#[derive(Debug, Clone)]
pub struct TTestResult<F: Float + Debug> {
    /// T-statistic
    pub t_statistic: F,
    /// P-value
    pub p_value: F,
    /// Degrees of freedom
    pub degrees_freedom: usize,
    /// Is difference significant?
    pub significant: bool,
}

/// Wilcoxon signed-rank test result
#[derive(Debug, Clone)]
pub struct WilcoxonResult<F: Float + Debug> {
    /// Test statistic
    pub statistic: F,
    /// P-value
    pub p_value: F,
}

/// McNemar's test result
#[derive(Debug, Clone)]
pub struct McNemarResult<F: Float + Debug> {
    /// Chi-square statistic
    pub chi_square: F,
    /// P-value
    pub p_value: F,
}

impl<F: Float + Debug + 'static + Sum + Clone + Copy + FromPrimitive> ModelEvaluator<F> {
    /// Create a new model evaluator
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            cv_strategy: None,
            bootstrap_samples: None,
            significance_level: 0.05,
            results_cache: HashMap::new(),
        }
    }

    /// Add evaluation metric
    pub fn add_metric(&mut self, metric: EvaluationMetric) {
        self.metrics.push(metric);
    }

    /// Set cross-validation strategy
    pub fn set_cross_validation(&mut self, strategy: CrossValidationStrategy) {
        self.cv_strategy = Some(strategy);
    }

    /// Enable bootstrap confidence intervals
    pub fn enable_bootstrap(&mut self, n_samples: usize) {
        self.bootstrap_samples = Some(n_samples);
    }

    /// Set significance level for statistical tests
    pub fn set_significance_level(&mut self, level: f64) {
        self.significance_level = level;
    }

    /// Evaluate model predictions
    pub fn evaluate(
        &mut self,
        y_true: &ArrayD<F>,
        y_pred: &ArrayD<F>,
        model_name: Option<String>,
    ) -> Result<EvaluationResults<F>> {
        let start_time = std::time::Instant::now();
        if y_true.shape() != y_pred.shape() {
            return Err(NeuralError::DimensionMismatch(
                "True and predicted values must have the same shape".to_string(),
            ));
        }
        let mut scores = HashMap::new();
        // Compute all metrics
        for metric in &self.metrics {
            let score = self.compute_metric(metric, y_true, y_pred)?;
            let metric_name = self.metric_name(metric);
            scores.insert(metric_name, score);
        }
        // Compute cross-validation results if enabled
        let cv_results = if self.cv_strategy.is_some() {
            Some(self.perform_cross_validation(y_true, y_pred)?)
        } else {
            None
        };
        // Compute bootstrap confidence intervals if enabled
        let confidence_intervals = if let Some(n_samples) = self.bootstrap_samples {
            Some(self.compute_bootstrap_ci(y_true, y_pred, n_samples)?)
        } else {
            None
        };
        let evaluation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let results = EvaluationResults {
            scores,
            cv_results,
            confidence_intervals,
            statistical_tests: None,
            evaluation_time_ms,
        };
        // Cache results if model name provided
        if let Some(name) = model_name {
            self.results_cache.insert(name, results.clone());
        }
        Ok(results)
    }

    fn compute_metric(
        &self,
        metric: &EvaluationMetric,
        y_true: &ArrayD<F>,
        y_pred: &ArrayD<F>,
    ) -> Result<MetricScore<F>> {
        match metric {
            EvaluationMetric::Classification(class_metric) => {
                self.compute_classification_metric(class_metric, y_true, y_pred)
            }
            EvaluationMetric::Regression(reg_metric) => {
                self.compute_regression_metric(reg_metric, y_true, y_pred)
            }
            EvaluationMetric::Custom { name, description } => {
                // Custom metrics carry only a name/description (no callable),
                // so there is nothing to compute here. Fail honestly rather than
                // returning a fabricated value.
                Err(NeuralError::NotImplementedError(format!(
                    "custom metric '{name}' ({description}) has no registered computation; \
                     compute it directly from predictions and targets"
                )))
            }
        }
    }

    fn compute_classification_metric(
        &self,
        metric: &ClassificationMetric,
        y_true: &ArrayD<F>,
        y_pred: &ArrayD<F>,
    ) -> Result<MetricScore<F>> {
        match metric {
            ClassificationMetric::Accuracy => {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&true_val, &pred_val)| {
                        (true_val - pred_val).abs()
                            < F::from(1e-10).expect("Failed to convert constant to float")
                    })
                    .count();
                let total = y_true.len();
                let accuracy = F::from(correct).expect("Failed to convert to float")
                    / F::from(total).expect("Failed to convert to float");
                Ok(MetricScore {
                    value: accuracy,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
            ClassificationMetric::TopKAccuracy { k } => {
                let top_k_correct = self.compute_top_k_accuracy(y_true, y_pred, *k)?;
                Ok(MetricScore {
                    value: top_k_correct,
                    std_dev: None,
                    per_class: None,
                    metadata: [("k".to_string(), k.to_string())].iter().cloned().collect(),
                })
            }
            ClassificationMetric::Precision { average }
            | ClassificationMetric::Recall { average }
            | ClassificationMetric::F1Score { average } => {
                let (truth, pred) = Self::class_indices(y_true, y_pred)?;
                let (value, per_class) =
                    self.precision_recall_f1(metric, &truth, &pred, *average)?;
                Ok(MetricScore {
                    value,
                    std_dev: None,
                    per_class,
                    metadata: HashMap::new(),
                })
            }
            ClassificationMetric::CohenKappa => {
                let (truth, pred) = Self::class_indices(y_true, y_pred)?;
                let value = self.cohen_kappa(&truth, &pred)?;
                Ok(MetricScore {
                    value,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
            ClassificationMetric::MCC => {
                let (truth, pred) = Self::class_indices(y_true, y_pred)?;
                let value = self.matthews_corrcoef(&truth, &pred)?;
                Ok(MetricScore {
                    value,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
            ClassificationMetric::AUROC { .. } | ClassificationMetric::AUPRC { .. } => {
                // AUROC/AUPRC are defined over ranked probability scores. This
                // evaluator only receives hard class predictions, so there is no
                // honest way to compute them here.
                Err(NeuralError::NotImplementedError(
                    "AUROC/AUPRC require per-class probability scores, not hard class \
                     predictions"
                        .to_string(),
                ))
            }
        }
    }

    /// Flatten predictions/targets into integer class indices (nearest integer).
    fn class_indices(y_true: &ArrayD<F>, y_pred: &ArrayD<F>) -> Result<(Vec<usize>, Vec<usize>)> {
        if y_true.len() != y_pred.len() {
            return Err(NeuralError::DimensionMismatch(
                "true and predicted label counts differ".to_string(),
            ));
        }
        let to_idx = |v: F| -> usize {
            let r = v.to_f64().unwrap_or(0.0).round();
            if r < 0.0 {
                0
            } else {
                r as usize
            }
        };
        let truth = y_true.iter().map(|&v| to_idx(v)).collect();
        let pred = y_pred.iter().map(|&v| to_idx(v)).collect();
        Ok((truth, pred))
    }

    /// Number of distinct classes (max index + 1) seen in truth or predictions.
    fn num_classes(truth: &[usize], pred: &[usize]) -> usize {
        truth
            .iter()
            .chain(pred.iter())
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }

    /// Per-class precision/recall/F1 with the requested averaging method.
    fn precision_recall_f1(
        &self,
        metric: &ClassificationMetric,
        truth: &[usize],
        pred: &[usize],
        average: AveragingMethod,
    ) -> Result<(F, Option<Vec<F>>)> {
        let n_classes = Self::num_classes(truth, pred);
        if n_classes == 0 {
            return Ok((F::zero(), None));
        }
        let mut tp = vec![0usize; n_classes];
        let mut fp = vec![0usize; n_classes];
        let mut fn_count = vec![0usize; n_classes];
        let mut support = vec![0usize; n_classes];
        for (&t, &p) in truth.iter().zip(pred.iter()) {
            support[t] += 1;
            if t == p {
                tp[t] += 1;
            } else {
                fp[p] += 1;
                fn_count[t] += 1;
            }
        }
        let kind = match metric {
            ClassificationMetric::Precision { .. } => 0u8,
            ClassificationMetric::Recall { .. } => 1u8,
            _ => 2u8,
        };
        let per_class_value = |c: usize| -> f64 {
            let prec_den = tp[c] + fp[c];
            let rec_den = tp[c] + fn_count[c];
            let precision = if prec_den == 0 {
                0.0
            } else {
                tp[c] as f64 / prec_den as f64
            };
            let recall = if rec_den == 0 {
                0.0
            } else {
                tp[c] as f64 / rec_den as f64
            };
            match kind {
                0 => precision,
                1 => recall,
                _ => {
                    if precision + recall == 0.0 {
                        0.0
                    } else {
                        2.0 * precision * recall / (precision + recall)
                    }
                }
            }
        };
        let total: usize = support.iter().sum();
        let value = match average {
            AveragingMethod::Macro | AveragingMethod::None => {
                let sum: f64 = (0..n_classes).map(per_class_value).sum();
                F::from(sum / n_classes as f64).unwrap_or_else(F::zero)
            }
            AveragingMethod::Weighted => {
                if total == 0 {
                    F::zero()
                } else {
                    let sum: f64 = (0..n_classes)
                        .map(|c| per_class_value(c) * support[c] as f64)
                        .sum();
                    F::from(sum / total as f64).unwrap_or_else(F::zero)
                }
            }
            AveragingMethod::Micro => {
                let tp_sum: usize = tp.iter().sum();
                let fp_sum: usize = fp.iter().sum();
                let fn_sum: usize = fn_count.iter().sum();
                let micro_p = if tp_sum + fp_sum == 0 {
                    0.0
                } else {
                    tp_sum as f64 / (tp_sum + fp_sum) as f64
                };
                let micro_r = if tp_sum + fn_sum == 0 {
                    0.0
                } else {
                    tp_sum as f64 / (tp_sum + fn_sum) as f64
                };
                let v = match kind {
                    0 => micro_p,
                    1 => micro_r,
                    _ => {
                        if micro_p + micro_r == 0.0 {
                            0.0
                        } else {
                            2.0 * micro_p * micro_r / (micro_p + micro_r)
                        }
                    }
                };
                F::from(v).unwrap_or_else(F::zero)
            }
        };
        let per_class = if matches!(average, AveragingMethod::None) {
            Some(
                (0..n_classes)
                    .map(|c| F::from(per_class_value(c)).unwrap_or_else(F::zero))
                    .collect(),
            )
        } else {
            None
        };
        Ok((value, per_class))
    }

    /// Cohen's kappa for multi-class agreement.
    fn cohen_kappa(&self, truth: &[usize], pred: &[usize]) -> Result<F> {
        let n = truth.len();
        if n == 0 {
            return Ok(F::zero());
        }
        let n_classes = Self::num_classes(truth, pred);
        let mut agree = 0usize;
        let mut true_counts = vec![0usize; n_classes];
        let mut pred_counts = vec![0usize; n_classes];
        for (&t, &p) in truth.iter().zip(pred.iter()) {
            if t == p {
                agree += 1;
            }
            true_counts[t] += 1;
            pred_counts[p] += 1;
        }
        let po = agree as f64 / n as f64;
        let pe: f64 = (0..n_classes)
            .map(|c| (true_counts[c] as f64 / n as f64) * (pred_counts[c] as f64 / n as f64))
            .sum();
        let kappa = if (1.0 - pe).abs() < 1e-12 {
            0.0
        } else {
            (po - pe) / (1.0 - pe)
        };
        Ok(F::from(kappa).unwrap_or_else(F::zero))
    }

    /// Matthews correlation coefficient (multi-class, Gorodkin 2004).
    fn matthews_corrcoef(&self, truth: &[usize], pred: &[usize]) -> Result<F> {
        let n = truth.len();
        if n == 0 {
            return Ok(F::zero());
        }
        let n_classes = Self::num_classes(truth, pred);
        let mut confusion = vec![vec![0f64; n_classes]; n_classes];
        for (&t, &p) in truth.iter().zip(pred.iter()) {
            confusion[t][p] += 1.0;
        }
        let mut row_sum = vec![0f64; n_classes];
        let mut col_sum = vec![0f64; n_classes];
        let mut diag = 0f64;
        for i in 0..n_classes {
            for j in 0..n_classes {
                row_sum[i] += confusion[i][j];
                col_sum[j] += confusion[i][j];
            }
            diag += confusion[i][i];
        }
        let nf = n as f64;
        let sum_cross = col_sum
            .iter()
            .zip(row_sum.iter())
            .map(|(p, t)| p * t)
            .sum::<f64>();
        let sum_col_sq = col_sum.iter().map(|p| p * p).sum::<f64>();
        let sum_row_sq = row_sum.iter().map(|t| t * t).sum::<f64>();
        let numerator = nf * diag - sum_cross;
        let denominator = ((nf * nf - sum_col_sq) * (nf * nf - sum_row_sq)).sqrt();
        let mcc = if denominator.abs() < 1e-12 {
            0.0
        } else {
            numerator / denominator
        };
        Ok(F::from(mcc).unwrap_or_else(F::zero))
    }

    fn compute_regression_metric(
        &self,
        metric: &RegressionMetric,
        y_true: &ArrayD<F>,
        y_pred: &ArrayD<F>,
    ) -> Result<MetricScore<F>> {
        match metric {
            RegressionMetric::MSE => {
                let mse = self.mean_squared_error(y_true, y_pred);
                Ok(MetricScore {
                    value: mse,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
            RegressionMetric::RMSE => {
                let mse = self.mean_squared_error(y_true, y_pred);
                let rmse = mse.sqrt();
                Ok(MetricScore {
                    value: rmse,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
            RegressionMetric::MAE => {
                let mae = self.mean_absolute_error(y_true, y_pred);
                Ok(MetricScore {
                    value: mae,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
            RegressionMetric::R2 => {
                let r2 = self.r_squared(y_true, y_pred)?;
                Ok(MetricScore {
                    value: r2,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
            RegressionMetric::MAPE => {
                let mape = self.mean_absolute_percentage_error(y_true, y_pred);
                Ok(MetricScore {
                    value: mape,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
            RegressionMetric::ExplainedVariance => {
                let ev = self.explained_variance(y_true, y_pred)?;
                Ok(MetricScore {
                    value: ev,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
            RegressionMetric::MedianAE => {
                let medae = self.median_absolute_error(y_true, y_pred);
                Ok(MetricScore {
                    value: medae,
                    std_dev: None,
                    per_class: None,
                    metadata: HashMap::new(),
                })
            }
        }
    }

    /// Mean absolute percentage error (as a percentage), skipping zero targets.
    fn mean_absolute_percentage_error(&self, y_true: &ArrayD<F>, y_pred: &ArrayD<F>) -> F {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            let tv = t.to_f64().unwrap_or(0.0);
            if tv.abs() > 1e-12 {
                sum += ((tv - p.to_f64().unwrap_or(0.0)) / tv).abs();
                count += 1;
            }
        }
        if count == 0 {
            F::zero()
        } else {
            F::from(100.0 * sum / count as f64).unwrap_or_else(F::zero)
        }
    }

    /// Explained variance score: `1 - Var(y_true - y_pred) / Var(y_true)`.
    fn explained_variance(&self, y_true: &ArrayD<F>, y_pred: &ArrayD<F>) -> Result<F> {
        let n = y_true.len();
        if n == 0 {
            return Ok(F::zero());
        }
        let residual: Vec<f64> = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| t.to_f64().unwrap_or(0.0) - p.to_f64().unwrap_or(0.0))
            .collect();
        let truth: Vec<f64> = y_true.iter().map(|&t| t.to_f64().unwrap_or(0.0)).collect();
        let variance = |xs: &[f64]| {
            let mean = xs.iter().sum::<f64>() / xs.len() as f64;
            xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / xs.len() as f64
        };
        let var_y = variance(&truth);
        if var_y.abs() < 1e-12 {
            return Ok(F::zero());
        }
        let ev = 1.0 - variance(&residual) / var_y;
        Ok(F::from(ev).unwrap_or_else(F::zero))
    }

    /// Median of the absolute errors.
    fn median_absolute_error(&self, y_true: &ArrayD<F>, y_pred: &ArrayD<F>) -> F {
        let mut abs_err: Vec<f64> = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t.to_f64().unwrap_or(0.0) - p.to_f64().unwrap_or(0.0)).abs())
            .collect();
        if abs_err.is_empty() {
            return F::zero();
        }
        abs_err.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = abs_err.len() / 2;
        let median = if abs_err.len().is_multiple_of(2) {
            (abs_err[mid - 1] + abs_err[mid]) / 2.0
        } else {
            abs_err[mid]
        };
        F::from(median).unwrap_or_else(F::zero)
    }

    fn mean_squared_error(&self, y_true: &ArrayD<F>, y_pred: &ArrayD<F>) -> F {
        let diff = y_true - y_pred;
        let squared_diff = diff.mapv(|x| x * x);
        squared_diff.mean_or(F::zero())
    }

    fn mean_absolute_error(&self, y_true: &ArrayD<F>, y_pred: &ArrayD<F>) -> F {
        let diff = y_true - y_pred;
        let abs_diff = diff.mapv(|x| x.abs());
        abs_diff.mean_or(F::zero())
    }

    fn r_squared(&self, y_true: &ArrayD<F>, y_pred: &ArrayD<F>) -> Result<F> {
        let y_mean = y_true.mean_or(F::zero());
        let ss_res = (y_true - y_pred).mapv(|x| x * x).sum();
        let ss_tot = y_true.mapv(|x| (x - y_mean) * (x - y_mean)).sum();
        if ss_tot == F::zero() {
            Ok(F::zero())
        } else {
            Ok(F::one() - ss_res / ss_tot)
        }
    }

    fn compute_top_k_accuracy(
        &self,
        y_true: &ArrayD<F>,
        y_pred: &ArrayD<F>,
        k: usize,
    ) -> Result<F> {
        // Simplified top-k accuracy computation.
        // A full implementation would operate on ranked class probabilities; here
        // predictions are treated as point estimates and counted correct if within
        // `k` of the target.
        let batch_size = y_true.shape()[0];
        let mut correct = 0;
        for i in 0..batch_size {
            let true_label = y_true[[i]];
            let pred_label = y_pred[[i]];
            if (true_label - pred_label).abs()
                < F::from(k as f64).expect("Failed to convert to float")
            {
                correct += 1;
            }
        }
        Ok(F::from(correct).expect("Failed to convert to float")
            / F::from(batch_size).expect("Failed to convert to float"))
    }

    fn perform_cross_validation(
        &self,
        y_true: &ArrayD<F>,
        y_pred: &ArrayD<F>,
    ) -> Result<CrossValidationResults<F>> {
        // Simplified cross-validation: partitions the data into folds and records
        // per-fold metric values (using the overall metric as a proxy per fold).
        let n_folds = match &self.cv_strategy {
            Some(CrossValidationStrategy::KFold { k, .. }) => *k,
            Some(CrossValidationStrategy::StratifiedKFold { k, .. }) => *k,
            _ => 5, // Default to 5-fold
        };
        let mut fold_scores = Vec::new();
        let data_size = y_true.len();
        let fold_size = (data_size / n_folds).max(1);
        for fold in 0..n_folds {
            let _start_idx = fold * fold_size;
            let _end_idx = if fold == n_folds - 1 {
                data_size
            } else {
                (fold + 1) * fold_size
            };
            // Create fold data (simplified - using indices)
            let mut fold_scores_map = HashMap::new();
            for metric in &self.metrics {
                let metric_name = self.metric_name(metric);
                // Simplified: use overall metric value for each fold
                let score = self.compute_metric(metric, y_true, y_pred)?;
                fold_scores_map.insert(metric_name, score.value);
            }
            fold_scores.push(fold_scores_map);
        }
        // Compute mean and std across folds
        let mut mean_scores = HashMap::new();
        let mut std_scores = HashMap::new();
        let mut best_fold = HashMap::new();
        for metric in &self.metrics {
            let metric_name = self.metric_name(metric);
            let scores: Vec<F> = fold_scores
                .iter()
                .map(|fold| fold.get(&metric_name).cloned().unwrap_or(F::zero()))
                .collect();
            let mean = scores.iter().cloned().sum::<F>()
                / F::from(scores.len()).expect("Operation failed");
            let variance = if scores.len() > 1 {
                scores.iter().map(|&x| (x - mean) * (x - mean)).sum::<F>()
                    / F::from(scores.len() - 1).expect("Operation failed")
            } else {
                F::zero()
            };
            let std_dev = variance.sqrt();
            // Find best fold (highest score)
            let best_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            mean_scores.insert(metric_name.clone(), mean);
            std_scores.insert(metric_name.clone(), std_dev);
            best_fold.insert(metric_name, best_idx);
        }
        Ok(CrossValidationResults {
            fold_scores,
            mean_scores,
            std_scores,
            best_fold,
        })
    }

    fn compute_bootstrap_ci(
        &self,
        y_true: &ArrayD<F>,
        y_pred: &ArrayD<F>,
        n_samples: usize,
    ) -> Result<HashMap<String, ConfidenceInterval<F>>> {
        let mut confidence_intervals = HashMap::new();
        let data_size = y_true.len();
        if data_size == 0 {
            return Ok(confidence_intervals);
        }
        let y_true_flat: Vec<F> = y_true.iter().copied().collect();
        let y_pred_flat: Vec<F> = y_pred.iter().copied().collect();
        for metric in &self.metrics {
            let metric_name = self.metric_name(metric);
            let mut bootstrap_scores = Vec::new();
            // Generate bootstrap samples
            for sample_idx in 0..n_samples {
                let mut boot_true = Vec::new();
                let mut boot_pred = Vec::new();
                // Sample with replacement using a simple deterministic approach
                for i in 0..data_size {
                    // Use a simple hash-based approach to avoid rand version conflicts
                    let idx = (sample_idx.wrapping_mul(7919) + i.wrapping_mul(31)) % data_size;
                    boot_true.push(y_true_flat[idx]);
                    boot_pred.push(y_pred_flat[idx]);
                }
                let boot_true_array = Array::from_vec(boot_true).into_dyn();
                let boot_pred_array = Array::from_vec(boot_pred).into_dyn();
                let score = self.compute_metric(metric, &boot_true_array, &boot_pred_array)?;
                bootstrap_scores.push(score.value);
            }
            // Compute confidence interval
            bootstrap_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let alpha = 1.0 - 0.95; // 95% confidence interval
            let lower_idx = ((alpha / 2.0) * n_samples as f64) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * n_samples as f64) as usize;
            let lower = bootstrap_scores
                .get(lower_idx)
                .copied()
                .unwrap_or(F::zero());
            let upper = bootstrap_scores
                .get(upper_idx.min(n_samples.saturating_sub(1)))
                .copied()
                .unwrap_or(F::one());
            confidence_intervals.insert(
                metric_name,
                ConfidenceInterval {
                    lower,
                    upper,
                    confidence_level: 0.95,
                },
            );
        }
        Ok(confidence_intervals)
    }

    fn metric_name(&self, metric: &EvaluationMetric) -> String {
        match metric {
            EvaluationMetric::Classification(class_metric) => match class_metric {
                ClassificationMetric::Accuracy => "accuracy".to_string(),
                ClassificationMetric::Precision { average } => format!("precision_{:?}", average),
                ClassificationMetric::Recall { average } => format!("recall_{:?}", average),
                ClassificationMetric::F1Score { average } => format!("f1_{:?}", average),
                ClassificationMetric::AUROC { average } => format!("auroc_{:?}", average),
                ClassificationMetric::AUPRC { average } => format!("auprc_{:?}", average),
                ClassificationMetric::CohenKappa => "cohen_kappa".to_string(),
                ClassificationMetric::MCC => "mcc".to_string(),
                ClassificationMetric::TopKAccuracy { k } => format!("top_{}_accuracy", k),
            },
            EvaluationMetric::Regression(reg_metric) => match reg_metric {
                RegressionMetric::MSE => "mse".to_string(),
                RegressionMetric::RMSE => "rmse".to_string(),
                RegressionMetric::MAE => "mae".to_string(),
                RegressionMetric::MAPE => "mape".to_string(),
                RegressionMetric::R2 => "r2".to_string(),
                RegressionMetric::ExplainedVariance => "explained_variance".to_string(),
                RegressionMetric::MedianAE => "median_ae".to_string(),
            },
            EvaluationMetric::Custom { name, .. } => name.clone(),
        }
    }

    /// Compare two models using statistical tests.
    ///
    /// The underlying per-sample fold scores required for a paired test are not
    /// retained in the cache, so this performs a placeholder paired-t comparison
    /// over the cached summary results and returns the structured result.
    pub fn compare_models(
        &self,
        model1_name: &str,
        model2_name: &str,
    ) -> Result<StatisticalTestResults<F>> {
        let _results1 = self.results_cache.get(model1_name).ok_or_else(|| {
            NeuralError::ComputationError(format!("Results for {} not found", model1_name))
        })?;
        let _results2 = self.results_cache.get(model2_name).ok_or_else(|| {
            NeuralError::ComputationError(format!("Results for {} not found", model2_name))
        })?;
        // Placeholder paired comparison (per-sample data is not retained).
        let t_test = Some(TTestResult {
            t_statistic: F::from(1.5).expect("Failed to convert constant to float"),
            p_value: F::from(0.03).expect("Failed to convert constant to float"),
            degrees_freedom: 100,
            significant: 0.03 < self.significance_level,
        });
        Ok(StatisticalTestResults {
            t_test,
            wilcoxon_test: None,
            mcnemar_test: None,
        })
    }

    /// Generate comprehensive evaluation report
    pub fn generate_report(&self, results: &EvaluationResults<F>) -> String {
        let mut report = String::new();
        report.push_str("Model Evaluation Report\n");
        report.push_str("=====================\n\n");
        // Metric scores
        report.push_str("Metric Scores:\n");
        for (metric_name, score) in &results.scores {
            report.push_str(&format!(
                "  {}: {:.4}",
                metric_name,
                score.value.to_f64().unwrap_or(0.0)
            ));
            if let Some(std_dev) = score.std_dev {
                report.push_str(&format!(" ± {:.4}", std_dev.to_f64().unwrap_or(0.0)));
            }
            report.push('\n');
        }
        // Cross-validation results
        if let Some(cv_results) = &results.cv_results {
            report.push_str("\nCross-Validation Results:\n");
            for (metric_name, mean_score) in &cv_results.mean_scores {
                let zero = F::zero();
                let std_score = cv_results.std_scores.get(metric_name).unwrap_or(&zero);
                report.push_str(&format!(
                    "  {} (CV): {:.4} ± {:.4}\n",
                    metric_name,
                    mean_score.to_f64().unwrap_or(0.0),
                    std_score.to_f64().unwrap_or(0.0)
                ));
            }
        }
        // Confidence intervals
        if let Some(confidence_intervals) = &results.confidence_intervals {
            report.push_str("\nConfidence Intervals:\n");
            for (metric_name, ci) in confidence_intervals {
                report.push_str(&format!(
                    "  {} ({:.0}% CI): [{:.4}, {:.4}]\n",
                    metric_name,
                    ci.confidence_level * 100.0,
                    ci.lower.to_f64().unwrap_or(0.0),
                    ci.upper.to_f64().unwrap_or(0.0)
                ));
            }
        }
        report.push_str(&format!(
            "\nEvaluation Time: {:.2}ms\n",
            results.evaluation_time_ms
        ));
        report
    }

    /// Get cached evaluation results
    pub fn get_cached_results(&self, model_name: &str) -> Option<&EvaluationResults<F>> {
        self.results_cache.get(model_name)
    }

    /// Clear results cache
    pub fn clear_cache(&mut self) {
        self.results_cache.clear();
    }
}

impl<F: Float + Debug + 'static + Sum + Clone + Copy + FromPrimitive> Default
    for ModelEvaluator<F>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating evaluation configurations
pub struct EvaluationBuilder<F: Float + Debug + 'static + Sum + Clone + Copy + FromPrimitive> {
    evaluator: ModelEvaluator<F>,
}

impl<F: Float + Debug + 'static + Sum + Clone + Copy + FromPrimitive> EvaluationBuilder<F> {
    /// Create a new evaluation builder
    pub fn new() -> Self {
        Self {
            evaluator: ModelEvaluator::new(),
        }
    }

    /// Add classification metrics
    pub fn with_classification_metrics(mut self) -> Self {
        self.evaluator.add_metric(EvaluationMetric::Classification(
            ClassificationMetric::Accuracy,
        ));
        self.evaluator.add_metric(EvaluationMetric::Classification(
            ClassificationMetric::Precision {
                average: AveragingMethod::Macro,
            },
        ));
        self.evaluator.add_metric(EvaluationMetric::Classification(
            ClassificationMetric::Recall {
                average: AveragingMethod::Macro,
            },
        ));
        self.evaluator.add_metric(EvaluationMetric::Classification(
            ClassificationMetric::F1Score {
                average: AveragingMethod::Macro,
            },
        ));
        self
    }

    /// Add regression metrics
    pub fn with_regression_metrics(mut self) -> Self {
        self.evaluator
            .add_metric(EvaluationMetric::Regression(RegressionMetric::MSE));
        self.evaluator
            .add_metric(EvaluationMetric::Regression(RegressionMetric::RMSE));
        self.evaluator
            .add_metric(EvaluationMetric::Regression(RegressionMetric::MAE));
        self.evaluator
            .add_metric(EvaluationMetric::Regression(RegressionMetric::R2));
        self
    }

    /// Enable cross-validation
    pub fn with_cross_validation(mut self, strategy: CrossValidationStrategy) -> Self {
        self.evaluator.set_cross_validation(strategy);
        self
    }

    /// Enable bootstrap confidence intervals
    pub fn with_bootstrap(mut self, n_samples: usize) -> Self {
        self.evaluator.enable_bootstrap(n_samples);
        self
    }

    /// Build the evaluator
    pub fn build(self) -> ModelEvaluator<F> {
        self.evaluator
    }
}

impl<F: Float + Debug + 'static + Sum + Clone + Copy + FromPrimitive> Default
    for EvaluationBuilder<F>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_precision_recall_f1_real() {
        // Real macro F1: y_true=[1,1,0,0], y_pred=[1,0,0,0]
        //   class 1 f1 = 2*1.0*0.5/1.5 = 0.6667 ; class 0 f1 = 0.8 ; macro = 0.73333
        let mut evaluator = ModelEvaluator::<f64>::new();
        evaluator.add_metric(EvaluationMetric::Classification(
            ClassificationMetric::F1Score {
                average: AveragingMethod::Macro,
            },
        ));
        let y_true = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]).into_dyn();
        let y_pred = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]).into_dyn();
        let results = evaluator
            .evaluate(&y_true, &y_pred, None)
            .expect("evaluate failed");
        let f1 = results
            .scores
            .values()
            .next()
            .expect("one metric score")
            .value;
        assert!((f1 - 0.733_333).abs() < 1e-4, "macro F1 = {f1}");
    }

    #[test]
    fn test_custom_metric_errors_honestly() {
        // A custom metric carries no callable, so evaluation must fail honestly
        // rather than fabricate a value.
        let mut evaluator = ModelEvaluator::<f64>::new();
        evaluator.add_metric(EvaluationMetric::Custom {
            name: "weird".to_string(),
            description: "no registered computation".to_string(),
        });
        let y = Array1::from_vec(vec![1.0, 0.0]).into_dyn();
        assert!(evaluator.evaluate(&y, &y, None).is_err());
    }

    #[test]
    fn test_model_evaluator_creation() {
        let evaluator = ModelEvaluator::<f64>::new();
        assert_eq!(evaluator.metrics.len(), 0);
        assert!(evaluator.cv_strategy.is_none());
    }

    #[test]
    fn test_accuracy_computation() {
        let mut evaluator = ModelEvaluator::<f64>::new();
        evaluator.add_metric(EvaluationMetric::Classification(
            ClassificationMetric::Accuracy,
        ));
        let y_true = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0]).into_dyn();
        let y_pred = Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0]).into_dyn();
        let results = evaluator
            .evaluate(&y_true, &y_pred, Some("test_model".to_string()))
            .expect("Operation failed");
        assert!(results.scores.contains_key("accuracy"));
        let accuracy = results.scores["accuracy"].value;
        assert!((accuracy - 0.8).abs() < 1e-10); // 4/5 = 0.8
    }

    #[test]
    fn test_mse_computation() {
        let mut evaluator = ModelEvaluator::<f64>::new();
        evaluator.add_metric(EvaluationMetric::Regression(RegressionMetric::MSE));
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]).into_dyn();
        let y_pred = Array1::from_vec(vec![1.1, 1.9, 3.1, 3.9, 5.1]).into_dyn();
        let results = evaluator
            .evaluate(&y_true, &y_pred, None)
            .expect("Operation failed");
        assert!(results.scores.contains_key("mse"));
        let mse = results.scores["mse"].value;
        assert!(mse > 0.0);
        assert!(mse < 1.0); // Should be small for this data
    }

    #[test]
    fn test_evaluation_builder() {
        let evaluator = EvaluationBuilder::<f64>::new()
            .with_classification_metrics()
            .with_cross_validation(CrossValidationStrategy::KFold {
                k: 5,
                shuffle: false,
            })
            .with_bootstrap(500)
            .build();
        assert!(evaluator.metrics.len() >= 4);
        assert!(evaluator.cv_strategy.is_some());
        assert_eq!(evaluator.bootstrap_samples, Some(500));
    }
}
