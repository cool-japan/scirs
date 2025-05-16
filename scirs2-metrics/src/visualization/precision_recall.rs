//! Precision-Recall curve visualization
//!
//! This module provides tools for visualizing Precision-Recall curves.

use ndarray::{ArrayBase, Data, Ix1};
use std::error::Error;

use super::{MetricVisualizer, PlotType, VisualizationData, VisualizationMetadata};
use crate::classification::curves::precision_recall_curve;
use crate::error::{MetricsError, Result};

/// Type alias for PR curve computation result
pub(crate) type PRComputeResult = (Vec<f64>, Vec<f64>, Vec<f64>, Option<f64>);

/// Precision-Recall curve visualizer
///
/// This struct provides methods for visualizing Precision-Recall curves.
#[derive(Debug, Clone)]
pub struct PrecisionRecallVisualizer<'a, T, S>
where
    T: Clone + PartialOrd,
    S: Data<Elem = T>,
{
    /// Precision values
    precision: Option<Vec<f64>>,
    /// Recall values
    recall: Option<Vec<f64>>,
    /// Thresholds
    thresholds: Option<Vec<f64>>,
    /// Average precision score
    average_precision: Option<f64>,
    /// Title for the plot
    title: String,
    /// Whether to display the average precision in the legend
    show_ap: bool,
    /// Whether to display the baseline
    show_baseline: bool,
    /// Original y_true data
    y_true: Option<&'a ArrayBase<S, Ix1>>,
    /// Original y_score data
    y_score: Option<&'a ArrayBase<S, Ix1>>,
    /// Class label for multi-class PR curve
    pos_label: Option<T>,
}

impl<'a, T, S> PrecisionRecallVisualizer<'a, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    /// Create a new PrecisionRecallVisualizer from pre-computed PR curve data
    ///
    /// # Arguments
    ///
    /// * `precision` - Precision values
    /// * `recall` - Recall values
    /// * `thresholds` - Optional thresholds
    /// * `average_precision` - Optional average precision score
    ///
    /// # Returns
    ///
    /// * A new PrecisionRecallVisualizer
    pub fn new(
        precision: Vec<f64>,
        recall: Vec<f64>,
        thresholds: Option<Vec<f64>>,
        average_precision: Option<f64>,
    ) -> Self {
        PrecisionRecallVisualizer {
            precision: Some(precision),
            recall: Some(recall),
            thresholds,
            average_precision,
            title: "Precision-Recall Curve".to_string(),
            show_ap: true,
            show_baseline: true,
            y_true: None,
            y_score: None,
            pos_label: None,
        }
    }

    /// Create a PrecisionRecallVisualizer from true labels and scores
    ///
    /// # Arguments
    ///
    /// * `y_true` - True binary labels
    /// * `y_score` - Target scores (probabilities or decision function output)
    /// * `pos_label` - Label of the positive class
    ///
    /// # Returns
    ///
    /// * A new PrecisionRecallVisualizer
    pub fn from_labels(
        y_true: &'a ArrayBase<S, Ix1>,
        y_score: &'a ArrayBase<S, Ix1>,
        pos_label: Option<T>,
    ) -> Self {
        PrecisionRecallVisualizer {
            precision: None,
            recall: None,
            thresholds: None,
            average_precision: None,
            title: "Precision-Recall Curve".to_string(),
            show_ap: true,
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

    /// Set whether to display the average precision in the legend
    ///
    /// # Arguments
    ///
    /// * `show_ap` - Whether to display the average precision
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_show_ap(mut self, show_ap: bool) -> Self {
        self.show_ap = show_ap;
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

    /// Set the average precision score
    ///
    /// # Arguments
    ///
    /// * `average_precision` - Average precision score
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_average_precision(mut self, average_precision: f64) -> Self {
        self.average_precision = Some(average_precision);
        self
    }

    /// Compute the Precision-Recall curve if not already computed
    ///
    /// # Returns
    ///
    /// * Result containing (precision, recall, thresholds, average_precision)
    fn compute_pr(&self) -> Result<PRComputeResult> {
        if self.precision.is_some() && self.recall.is_some() {
            // Return pre-computed values
            return Ok((
                self.precision.clone().unwrap(),
                self.recall.clone().unwrap(),
                self.thresholds.clone().unwrap_or_default(),
                self.average_precision,
            ));
        }

        if self.y_true.is_none() || self.y_score.is_none() {
            return Err(MetricsError::InvalidInput(
                "No data provided for Precision-Recall curve computation".to_string(),
            ));
        }

        let y_true = self.y_true.unwrap();
        let y_score = self.y_score.unwrap();

        // Compute Precision-Recall curve
        let (precision, recall, thresholds) = precision_recall_curve(y_true, y_score)?;

        // Compute average precision if not already provided
        let ap = if self.average_precision.is_none() {
            // Average precision is the weighted mean of precisions at each threshold,
            // with the increase in recall from the previous threshold used as the weight
            let n = recall.len();

            let mut ap = 0.0;
            for i in 1..n {
                ap += (recall[i] - recall[i - 1]) * precision[i];
            }

            Some(ap)
        } else {
            self.average_precision
        };

        Ok((precision.to_vec(), recall.to_vec(), thresholds.to_vec(), ap))
    }
}

impl<T, S> MetricVisualizer for PrecisionRecallVisualizer<'_, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    fn prepare_data(&self) -> std::result::Result<VisualizationData, Box<dyn Error>> {
        let (precision, recall, _, ap) = self
            .compute_pr()
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        // Prepare data for visualization
        let mut x = recall;
        let mut y = precision;

        // Prepare series names
        let mut series_names = Vec::new();

        if self.show_ap && ap.is_some() {
            series_names.push(format!("Precision-Recall curve (AP = {:.3})", ap.unwrap()));
        } else {
            series_names.push("Precision-Recall curve".to_string());
        }

        // Add baseline if requested
        if self.show_baseline {
            // Compute the positive class ratio (baseline precision)
            let baseline_precision = if let (Some(y_true), Some(pos_label)) =
                (self.y_true, &self.pos_label)
            {
                // Count positive samples and total samples
                let total = y_true.len() as f64;
                let positive = y_true.iter().filter(|&label| *label == *pos_label).count() as f64;

                positive / total
            } else if !y.is_empty() {
                // Use the last precision value (at recall=0) as an approximation
                y[y.len() - 1]
            } else {
                0.5 // Default fallback
            };

            // Add the baseline (constant precision across all recall values)
            series_names.push(format!("Baseline (precision = {:.3})", baseline_precision));

            // Add points for the baseline
            x.push(0.0);
            x.push(1.0);
            y.push(baseline_precision);
            y.push(baseline_precision);
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
            x_label: "Recall".to_string(),
            y_label: "Precision".to_string(),
            plot_type: PlotType::Line,
            description: Some(
                "Precision-Recall curve showing the trade-off between precision and recall"
                    .to_string(),
            ),
        }
    }
}

/// Create a Precision-Recall curve visualization from pre-computed PR curve data
///
/// # Arguments
///
/// * `precision` - Precision values
/// * `recall` - Recall values
/// * `thresholds` - Optional thresholds
/// * `average_precision` - Optional average precision score
///
/// # Returns
///
/// * A PrecisionRecallVisualizer
pub fn precision_recall_visualization(
    precision: Vec<f64>,
    recall: Vec<f64>,
    thresholds: Option<Vec<f64>>,
    average_precision: Option<f64>,
) -> PrecisionRecallVisualizer<'static, f64, ndarray::OwnedRepr<f64>> {
    PrecisionRecallVisualizer::new(precision, recall, thresholds, average_precision)
}

/// Create a Precision-Recall curve visualization from true labels and scores
///
/// # Arguments
///
/// * `y_true` - True binary labels
/// * `y_score` - Target scores (probabilities or decision function output)
/// * `pos_label` - Optional label of the positive class
///
/// # Returns
///
/// * A PrecisionRecallVisualizer
pub fn precision_recall_from_labels<'a, T, S>(
    y_true: &'a ArrayBase<S, Ix1>,
    y_score: &'a ArrayBase<S, Ix1>,
    pos_label: Option<T>,
) -> PrecisionRecallVisualizer<'a, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    PrecisionRecallVisualizer::from_labels(y_true, y_score, pos_label)
}
