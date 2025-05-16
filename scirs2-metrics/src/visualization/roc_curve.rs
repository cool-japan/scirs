//! ROC curve visualization
//!
//! This module provides tools for visualizing ROC (Receiver Operating Characteristic) curves.

use ndarray::{ArrayBase, Data, Ix1};
use std::error::Error;

use super::{MetricVisualizer, PlotType, VisualizationData, VisualizationMetadata};
use crate::classification::curves::roc_curve;
use crate::error::{MetricsError, Result};

/// Type alias for ROC curve computation result
pub(crate) type ROCComputeResult = (Vec<f64>, Vec<f64>, Vec<f64>, Option<f64>);

/// ROC curve visualizer
///
/// This struct provides methods for visualizing ROC curves.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ROCCurveVisualizer<'a, T, S>
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
}

impl<'a, T, S> ROCCurveVisualizer<'a, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    /// Create a new ROCCurveVisualizer from pre-computed ROC curve data
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
    /// * A new ROCCurveVisualizer
    pub fn new(
        fpr: Vec<f64>,
        tpr: Vec<f64>,
        thresholds: Option<Vec<f64>>,
        auc: Option<f64>,
    ) -> Self {
        ROCCurveVisualizer {
            tpr: Some(tpr),
            fpr: Some(fpr),
            thresholds,
            auc,
            title: "ROC Curve".to_string(),
            show_auc: true,
            show_baseline: true,
            y_true: None,
            y_score: None,
            pos_label: None,
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
    /// * A new ROCCurveVisualizer
    pub fn from_labels(
        y_true: &'a ArrayBase<S, Ix1>,
        y_score: &'a ArrayBase<S, Ix1>,
        pos_label: Option<T>,
    ) -> Self {
        ROCCurveVisualizer {
            tpr: None,
            fpr: None,
            thresholds: None,
            auc: None,
            title: "ROC Curve".to_string(),
            show_auc: true,
            show_baseline: true,
            y_true: Some(y_true),
            y_score: Some(y_score),
            pos_label,
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
}

impl<T, S> MetricVisualizer for ROCCurveVisualizer<'_, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    fn prepare_data(&self) -> std::result::Result<VisualizationData, Box<dyn Error>> {
        let (fpr, tpr, _, auc) = self
            .compute_roc()
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        // Prepare data for visualization
        let mut x = fpr;
        let mut y = tpr;

        // Add baseline if requested
        if self.show_baseline {
            // Add points for the diagonal baseline (random classifier)
            x.push(0.0);
            x.push(1.0);
            y.push(0.0);
            y.push(1.0);
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

        Ok(VisualizationData {
            x,
            y,
            z: None,
            series_names: Some(series_names),
            x_labels: None,
            y_labels: None,
            auxiliary_data: std::collections::HashMap::new(),
            auxiliary_metadata: std::collections::HashMap::new(),
            series: std::collections::HashMap::new(),
        })
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        VisualizationMetadata {
            title: self.title.clone(),
            x_label: "False Positive Rate".to_string(),
            y_label: "True Positive Rate".to_string(),
            plot_type: PlotType::Line,
            description: Some("ROC curve showing the trade-off between true positive rate and false positive rate".to_string()),
        }
    }
}

/// Create a ROC curve visualization from pre-computed ROC curve data
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
/// * A ROCCurveVisualizer
pub fn roc_curve_visualization(
    fpr: Vec<f64>,
    tpr: Vec<f64>,
    thresholds: Option<Vec<f64>>,
    auc: Option<f64>,
) -> ROCCurveVisualizer<'static, f64, ndarray::OwnedRepr<f64>> {
    ROCCurveVisualizer::new(fpr, tpr, thresholds, auc)
}

/// Create a ROC curve visualization from true labels and scores
///
/// # Arguments
///
/// * `y_true` - True binary labels
/// * `y_score` - Target scores (probabilities or decision function output)
/// * `pos_label` - Optional label of the positive class
///
/// # Returns
///
/// * A ROCCurveVisualizer
pub fn roc_curve_from_labels<'a, T, S>(
    y_true: &'a ArrayBase<S, Ix1>,
    y_score: &'a ArrayBase<S, Ix1>,
    pos_label: Option<T>,
) -> ROCCurveVisualizer<'a, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    ROCCurveVisualizer::from_labels(y_true, y_score, pos_label)
}
