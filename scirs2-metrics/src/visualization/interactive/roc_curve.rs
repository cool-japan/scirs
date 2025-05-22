//! Interactive ROC curve visualization
//!
//! This module provides tools for creating interactive ROC curve visualizations with
//! threshold adjustment and performance metrics display.

use ndarray::{ArrayBase, Data, Ix1};
use std::collections::HashMap;
use std::error::Error;

use crate::classification::curves::roc_curve;
use crate::error::{MetricsError, Result};
use crate::visualization::{
    MetricVisualizer, PlotType, VisualizationData, VisualizationMetadata, VisualizationOptions,
};

/// Type alias for ROC curve computation result
pub(crate) type ROCComputeResult = (Vec<f64>, Vec<f64>, Vec<f64>, Option<f64>);

/// Type alias for confusion matrix values at a threshold
pub(crate) type ConfusionMatrixValues = (usize, usize, usize, usize); // (TP, FP, TN, FN)

/// Interactive ROC curve visualizer
///
/// This struct provides methods for visualizing interactive ROC curves with
/// threshold adjustment and performance metrics display.
#[derive(Debug, Clone)]
pub struct InteractiveROCVisualizer<'a, T, S>
where
    T: Clone + PartialOrd,
    S: Data<Elem = T>,
{
    /// True positive rates
    tpr: Option<Vec<f64>>,
    /// False positive rates
    fpr: Option<Vec<f64>>,
    /// Thresholds
    thresholds: Option<Vec<f64>>,
    /// AUC value
    auc: Option<f64>,
    /// Title for the plot
    title: String,
    /// Whether to display the AUC in the legend
    show_auc: bool,
    /// Whether to display the baseline
    show_baseline: bool,
    /// Original y_true data
    y_true: Option<&'a ArrayBase<S, Ix1>>,
    /// Original y_score data
    y_score: Option<&'a ArrayBase<S, Ix1>>,
    /// Class label for multi-class ROC curve
    pos_label: Option<T>,
    /// Current selected threshold index
    current_threshold_idx: Option<usize>,
    /// Display metrics on the plot
    show_metrics: bool,
    /// Layout options for the interactive plot
    interactive_options: InteractiveOptions,
}

/// Options for interactive visualization
#[derive(Debug, Clone)]
pub struct InteractiveOptions {
    /// Width of the plot
    pub width: usize,
    /// Height of the plot
    pub height: usize,
    /// Whether to show the threshold slider
    pub show_threshold_slider: bool,
    /// Whether to show metric values at the selected threshold
    pub show_metric_values: bool,
    /// Whether to show confusion matrix at the selected threshold
    pub show_confusion_matrix: bool,
    /// Custom layout options
    pub custom_layout: HashMap<String, String>,
}

impl Default for InteractiveOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            show_threshold_slider: true,
            show_metric_values: true,
            show_confusion_matrix: true,
            custom_layout: HashMap::new(),
        }
    }
}

impl<'a, T, S> InteractiveROCVisualizer<'a, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    /// Create a new InteractiveROCVisualizer from pre-computed ROC curve data
    ///
    /// # Arguments
    ///
    /// * `fpr` - False positive rates
    /// * `tpr` - True positive rates
    /// * `thresholds` - Optional thresholds
    /// * `auc` - Optional AUC value
    ///
    /// # Returns
    ///
    /// * A new InteractiveROCVisualizer
    pub fn new(
        fpr: Vec<f64>,
        tpr: Vec<f64>,
        thresholds: Option<Vec<f64>>,
        auc: Option<f64>,
    ) -> Self {
        InteractiveROCVisualizer {
            tpr: Some(tpr),
            fpr: Some(fpr),
            thresholds,
            auc,
            title: "Interactive ROC Curve".to_string(),
            show_auc: true,
            show_baseline: true,
            y_true: None,
            y_score: None,
            pos_label: None,
            current_threshold_idx: None,
            show_metrics: true,
            interactive_options: InteractiveOptions::default(),
        }
    }

    /// Create a ROCCurveVisualizer from true labels and scores
    ///
    /// # Arguments
    ///
    /// * `y_true` - True binary labels
    /// * `y_score` - Target scores (probabilities or decision function output)
    /// * `pos_label` - Label of the positive class
    ///
    /// # Returns
    ///
    /// * A new InteractiveROCVisualizer
    pub fn from_labels(
        y_true: &'a ArrayBase<S, Ix1>,
        y_score: &'a ArrayBase<S, Ix1>,
        pos_label: Option<T>,
    ) -> Self {
        InteractiveROCVisualizer {
            tpr: None,
            fpr: None,
            thresholds: None,
            auc: None,
            title: "Interactive ROC Curve".to_string(),
            show_auc: true,
            show_baseline: true,
            y_true: Some(y_true),
            y_score: Some(y_score),
            pos_label,
            current_threshold_idx: None,
            show_metrics: true,
            interactive_options: InteractiveOptions::default(),
        }
    }

    /// Set the title for the plot
    ///
    /// # Arguments
    ///
    /// * `title` - Title for the plot
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_title(mut self, title: String) -> Self {
        self.title = title;
        self
    }

    /// Set whether to display the AUC in the legend
    ///
    /// # Arguments
    ///
    /// * `show_auc` - Whether to display the AUC
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_show_auc(mut self, show_auc: bool) -> Self {
        self.show_auc = show_auc;
        self
    }

    /// Set whether to display the baseline
    ///
    /// # Arguments
    ///
    /// * `show_baseline` - Whether to display the baseline
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_show_baseline(mut self, show_baseline: bool) -> Self {
        self.show_baseline = show_baseline;
        self
    }

    /// Set the AUC value
    ///
    /// # Arguments
    ///
    /// * `auc` - AUC value
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_auc(mut self, auc: f64) -> Self {
        self.auc = Some(auc);
        self
    }

    /// Set whether to display metrics on the plot
    ///
    /// # Arguments
    ///
    /// * `show_metrics` - Whether to display metrics
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_show_metrics(mut self, show_metrics: bool) -> Self {
        self.show_metrics = show_metrics;
        self
    }

    /// Set interactive options
    ///
    /// # Arguments
    ///
    /// * `options` - Interactive options
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_interactive_options(mut self, options: InteractiveOptions) -> Self {
        self.interactive_options = options;
        self
    }

    /// Set current threshold index
    ///
    /// # Arguments
    ///
    /// * `idx` - Threshold index
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_threshold_index(mut self, idx: usize) -> Self {
        self.current_threshold_idx = Some(idx);
        self
    }

    /// Set current threshold value
    ///
    /// # Arguments
    ///
    /// * `threshold` - Threshold value
    ///
    /// # Returns
    ///
    /// * Result containing self for method chaining
    pub fn with_threshold_value(mut self, threshold: f64) -> Result<Self> {
        // Ensure thresholds are computed
        let (_, _, thresholds, _) = self.compute_roc()?;

        if thresholds.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No thresholds available".to_string(),
            ));
        }

        // Find the closest threshold index
        let mut closest_idx = 0;
        let mut min_diff = f64::INFINITY;

        for (i, &t) in thresholds.iter().enumerate() {
            let diff = (t - threshold).abs();
            if diff < min_diff {
                min_diff = diff;
                closest_idx = i;
            }
        }

        self.current_threshold_idx = Some(closest_idx);
        Ok(self)
    }

    /// Compute the ROC curve if not already computed
    ///
    /// # Returns
    ///
    /// * Result containing (fpr, tpr, thresholds, auc)
    fn compute_roc(&self) -> Result<ROCComputeResult> {
        if self.fpr.is_some() && self.tpr.is_some() {
            // Return pre-computed values
            return Ok((
                self.fpr.clone().unwrap(),
                self.tpr.clone().unwrap(),
                self.thresholds.clone().unwrap_or_default(),
                self.auc,
            ));
        }

        if self.y_true.is_none() || self.y_score.is_none() {
            return Err(MetricsError::InvalidInput(
                "No data provided for ROC curve computation".to_string(),
            ));
        }

        let y_true = self.y_true.unwrap();
        let y_score = self.y_score.unwrap();

        // Compute ROC curve
        let (fpr, tpr, thresholds) = roc_curve(y_true, y_score)?;

        // Compute AUC if not already provided
        let auc = if self.auc.is_none() {
            // AUC is the area under the ROC curve, which we can approximate
            // using the trapezoidal rule
            let n = fpr.len();

            let mut area = 0.0;
            for i in 1..n {
                // Trapezoidal area: (b - a) * (f(a) + f(b)) / 2
                area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0;
            }

            Some(area)
        } else {
            self.auc
        };

        Ok((fpr.to_vec(), tpr.to_vec(), thresholds.to_vec(), auc))
    }

    /// Calculate confusion matrix values at a given threshold index
    ///
    /// # Arguments
    ///
    /// * `threshold_idx` - Index of the threshold
    ///
    /// # Returns
    ///
    /// * Result containing confusion matrix values (TP, FP, TN, FN)
    pub fn calculate_confusion_matrix(
        &self,
        threshold_idx: usize,
    ) -> Result<ConfusionMatrixValues> {
        if self.y_true.is_none() || self.y_score.is_none() {
            return Err(MetricsError::InvalidInput(
                "Original data required for confusion matrix calculation".to_string(),
            ));
        }

        let (_, _, thresholds, _) = self.compute_roc()?;

        if threshold_idx >= thresholds.len() {
            return Err(MetricsError::InvalidArgument(
                "Threshold index out of range".to_string(),
            ));
        }

        let threshold = thresholds[threshold_idx];
        let y_true = self.y_true.unwrap();
        let y_score = self.y_score.unwrap();

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        // Convert positive label to f64 for comparison
        let pos_label_f64 = match &self.pos_label {
            Some(label) => f64::from(label.clone()),
            None => 1.0, // Default positive label is 1.0
        };

        for i in 0..y_true.len() {
            let true_val = f64::from(y_true[i].clone());
            let score = f64::from(y_score[i].clone());

            let pred = if score >= threshold {
                pos_label_f64
            } else {
                0.0
            };

            if pred == pos_label_f64 && true_val == pos_label_f64 {
                tp += 1;
            } else if pred == pos_label_f64 && true_val != pos_label_f64 {
                fp += 1;
            } else if pred != pos_label_f64 && true_val != pos_label_f64 {
                tn += 1;
            } else {
                fn_ += 1;
            }
        }

        Ok((tp, fp, tn, fn_))
    }

    /// Calculate metrics at a given threshold index
    ///
    /// # Arguments
    ///
    /// * `threshold_idx` - Index of the threshold
    ///
    /// # Returns
    ///
    /// * Result containing a HashMap with metric names and values
    pub fn calculate_metrics(&self, threshold_idx: usize) -> Result<HashMap<String, f64>> {
        let (tp, fp, tn, fn_) = self.calculate_confusion_matrix(threshold_idx)?;

        let mut metrics = HashMap::new();

        // Accuracy
        let accuracy = (tp + tn) as f64 / (tp + fp + tn + fn_) as f64;
        metrics.insert("accuracy".to_string(), accuracy);

        // Precision
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        metrics.insert("precision".to_string(), precision);

        // Recall (Sensitivity, True Positive Rate)
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        metrics.insert("recall".to_string(), recall);

        // Specificity (True Negative Rate)
        let specificity = if tn + fp > 0 {
            tn as f64 / (tn + fp) as f64
        } else {
            0.0
        };
        metrics.insert("specificity".to_string(), specificity);

        // F1 Score
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        metrics.insert("f1_score".to_string(), f1);

        // Add threshold value
        let (_, _, thresholds, _) = self.compute_roc()?;
        metrics.insert("threshold".to_string(), thresholds[threshold_idx]);

        Ok(metrics)
    }

    /// Get the current threshold index or a default
    ///
    /// # Returns
    ///
    /// * The current threshold index or the middle index if not set
    pub fn get_current_threshold_idx(&self) -> Result<usize> {
        let (_, _, thresholds, _) = self.compute_roc()?;

        if thresholds.is_empty() {
            return Err(MetricsError::InvalidInput(
                "No thresholds available".to_string(),
            ));
        }

        match self.current_threshold_idx {
            Some(idx) if idx < thresholds.len() => Ok(idx),
            _ => Ok(thresholds.len() / 2), // Default to middle threshold
        }
    }
}

impl<T, S> MetricVisualizer for InteractiveROCVisualizer<'_, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    fn prepare_data(&self) -> std::result::Result<VisualizationData, Box<dyn Error>> {
        let (fpr, tpr, thresholds, auc) = self
            .compute_roc()
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        // Prepare data for visualization
        let mut data = VisualizationData::new();

        // ROC curve points
        data.x = fpr.clone();
        data.y = tpr.clone();

        // Store thresholds in auxiliary data
        data.add_auxiliary_data("thresholds".to_string(), thresholds.clone());

        // Add AUC if available
        if let Some(auc_val) = auc {
            data.add_auxiliary_metadata("auc".to_string(), auc_val.to_string());
        }

        // Add current threshold point if available
        if let Ok(threshold_idx) = self.get_current_threshold_idx() {
            let current_point_x = vec![fpr[threshold_idx]];
            let current_point_y = vec![tpr[threshold_idx]];

            data.add_auxiliary_data("current_point_x".to_string(), current_point_x);
            data.add_auxiliary_data("current_point_y".to_string(), current_point_y);
            data.add_auxiliary_metadata(
                "current_threshold".to_string(),
                thresholds[threshold_idx].to_string(),
            );

            // Add metrics at current threshold if requested
            if self.show_metrics {
                if let Ok(metrics) = self.calculate_metrics(threshold_idx) {
                    for (name, value) in metrics {
                        data.add_auxiliary_metadata(format!("metric_{}", name), value.to_string());
                    }
                }
            }
        }

        // Add interactive options
        data.add_auxiliary_metadata(
            "interactive_width".to_string(),
            self.interactive_options.width.to_string(),
        );
        data.add_auxiliary_metadata(
            "interactive_height".to_string(),
            self.interactive_options.height.to_string(),
        );
        data.add_auxiliary_metadata(
            "show_threshold_slider".to_string(),
            self.interactive_options.show_threshold_slider.to_string(),
        );
        data.add_auxiliary_metadata(
            "show_metric_values".to_string(),
            self.interactive_options.show_metric_values.to_string(),
        );
        data.add_auxiliary_metadata(
            "show_confusion_matrix".to_string(),
            self.interactive_options.show_confusion_matrix.to_string(),
        );

        // Add custom layout options
        for (key, value) in &self.interactive_options.custom_layout {
            data.add_auxiliary_metadata(format!("layout_{}", key), value.clone());
        }

        // Add baseline if requested
        if self.show_baseline {
            data.add_auxiliary_data("baseline_x".to_string(), vec![0.0, 1.0]);
            data.add_auxiliary_data("baseline_y".to_string(), vec![0.0, 1.0]);
        }

        // Prepare series names
        let mut series_names = Vec::new();

        if self.show_auc && auc.is_some() {
            series_names.push(format!("ROC curve (AUC = {:.3})", auc.unwrap()));
        } else {
            series_names.push("ROC curve".to_string());
        }

        if self.show_baseline {
            series_names.push("Random classifier".to_string());
        }

        // Point at current threshold
        series_names.push("Current threshold".to_string());

        data.add_series_names(series_names);

        Ok(data)
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        let mut metadata = VisualizationMetadata::new(self.title.clone());
        metadata.set_plot_type(PlotType::Line);
        metadata.set_x_label("False Positive Rate".to_string());
        metadata.set_y_label("True Positive Rate".to_string());
        metadata.set_description("Interactive ROC curve showing the trade-off between true positive rate and false positive rate. Adjust the threshold to see performance metrics.".to_string());

        metadata
    }
}

/// Create an interactive ROC curve visualization from pre-computed ROC curve data
///
/// # Arguments
///
/// * `fpr` - False positive rates
/// * `tpr` - True positive rates
/// * `thresholds` - Optional thresholds
/// * `auc` - Optional AUC value
///
/// # Returns
///
/// * An InteractiveROCVisualizer
pub fn interactive_roc_curve_visualization(
    fpr: Vec<f64>,
    tpr: Vec<f64>,
    thresholds: Option<Vec<f64>>,
    auc: Option<f64>,
) -> InteractiveROCVisualizer<'static, f64, ndarray::OwnedRepr<f64>> {
    InteractiveROCVisualizer::new(fpr, tpr, thresholds, auc)
}

/// Create an interactive ROC curve visualization from true labels and scores
///
/// # Arguments
///
/// * `y_true` - True binary labels
/// * `y_score` - Target scores (probabilities or decision function output)
/// * `pos_label` - Optional label of the positive class
///
/// # Returns
///
/// * An InteractiveROCVisualizer
pub fn interactive_roc_curve_from_labels<'a, T, S>(
    y_true: &'a ArrayBase<S, Ix1>,
    y_score: &'a ArrayBase<S, Ix1>,
    pos_label: Option<T>,
) -> InteractiveROCVisualizer<'a, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    InteractiveROCVisualizer::from_labels(y_true, y_score, pos_label)
}
