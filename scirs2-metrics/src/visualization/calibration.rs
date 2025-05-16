//! Calibration curve visualization
//!
//! This module provides tools for visualizing calibration curves (reliability diagrams).

use ndarray::{ArrayBase, Data, Ix1};
use std::error::Error;

use super::{MetricVisualizer, PlotType, VisualizationData, VisualizationMetadata};
use crate::classification::curves::calibration_curve;
use crate::error::{MetricsError, Result};

/// Calibration curve visualizer
///
/// This struct provides methods for visualizing calibration curves (reliability diagrams).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CalibrationVisualizer<'a, T, S>
where
    T: Clone + PartialOrd,
    S: Data<Elem = T>,
{
    /// Fraction of positive samples in each bin (empirical probabilities)
    fraction_of_positives: Option<Vec<f64>>,
    /// Mean predicted probability in each bin
    mean_predicted_value: Option<Vec<f64>>,
    /// Number of bins
    n_bins: usize,
    /// Strategy for binning
    strategy: String,
    /// Title for the plot
    title: String,
    /// Whether to display the perfect calibration line
    show_perfectly_calibrated: bool,
    /// Original y_true data
    y_true: Option<&'a ArrayBase<S, Ix1>>,
    /// Original y_prob data
    y_prob: Option<&'a ArrayBase<S, Ix1>>,
    /// Class label for multi-class calibration
    pos_label: Option<T>,
}

impl<'a, T, S> CalibrationVisualizer<'a, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    /// Create a new CalibrationVisualizer from pre-computed calibration curve data
    ///
    /// # Arguments
    ///
    /// * `fraction_of_positives` - Fraction of positive samples in each bin
    /// * `mean_predicted_value` - Mean predicted probability in each bin
    /// * `n_bins` - Number of bins used
    /// * `strategy` - Strategy used for binning ("uniform" or "quantile")
    ///
    /// # Returns
    ///
    /// * A new CalibrationVisualizer
    pub fn new(
        fraction_of_positives: Vec<f64>,
        mean_predicted_value: Vec<f64>,
        n_bins: usize,
        strategy: String,
    ) -> Self {
        CalibrationVisualizer {
            fraction_of_positives: Some(fraction_of_positives),
            mean_predicted_value: Some(mean_predicted_value),
            n_bins,
            strategy,
            title: "Calibration Curve".to_string(),
            show_perfectly_calibrated: true,
            y_true: None,
            y_prob: None,
            pos_label: None,
        }
    }

    /// Create a CalibrationVisualizer from true labels and probabilities
    ///
    /// # Arguments
    ///
    /// * `y_true` - True binary labels
    /// * `y_prob` - Predicted probabilities
    /// * `n_bins` - Number of bins to use
    /// * `strategy` - Strategy for binning
    /// * `pos_label` - Label of the positive class
    ///
    /// # Returns
    ///
    /// * A new CalibrationVisualizer
    pub fn from_labels(
        y_true: &'a ArrayBase<S, Ix1>,
        y_prob: &'a ArrayBase<S, Ix1>,
        n_bins: usize,
        strategy: String,
        pos_label: Option<T>,
    ) -> Self {
        CalibrationVisualizer {
            fraction_of_positives: None,
            mean_predicted_value: None,
            n_bins,
            strategy,
            title: "Calibration Curve".to_string(),
            show_perfectly_calibrated: true,
            y_true: Some(y_true),
            y_prob: Some(y_prob),
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

    /// Set whether to display the perfect calibration line
    ///
    /// # Arguments
    ///
    /// * `show_perfectly_calibrated` - Whether to display the perfect calibration line
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_show_perfectly_calibrated(mut self, show_perfectly_calibrated: bool) -> Self {
        self.show_perfectly_calibrated = show_perfectly_calibrated;
        self
    }

    /// Compute the calibration curve if not already computed
    ///
    /// # Returns
    ///
    /// * Result containing (fraction_of_positives, mean_predicted_value)
    fn compute_calibration(&self) -> Result<(Vec<f64>, Vec<f64>)> {
        if self.fraction_of_positives.is_some() && self.mean_predicted_value.is_some() {
            // Return pre-computed values
            return Ok((
                self.fraction_of_positives.clone().unwrap(),
                self.mean_predicted_value.clone().unwrap(),
            ));
        }

        if self.y_true.is_none() || self.y_prob.is_none() {
            return Err(MetricsError::InvalidInput(
                "No data provided for calibration curve computation".to_string(),
            ));
        }

        let y_true = self.y_true.unwrap();
        let y_prob = self.y_prob.unwrap();

        // Compute calibration curve
        let calib_result = calibration_curve(y_true, y_prob, Some(self.n_bins))?;

        Ok((calib_result.0.to_vec(), calib_result.1.to_vec()))
    }
}

impl<T, S> MetricVisualizer for CalibrationVisualizer<'_, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    fn prepare_data(&self) -> std::result::Result<VisualizationData, Box<dyn Error>> {
        let (fraction_of_positives, mean_predicted_value) = self
            .compute_calibration()
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        // Prepare data for visualization
        let mut x_values = mean_predicted_value;
        let mut y_values = fraction_of_positives;

        // Prepare series names
        let mut series_names = Vec::new();

        series_names.push(format!("Calibration curve (bins={})", self.n_bins));

        // Add perfect calibration line if requested
        if self.show_perfectly_calibrated {
            // Add the perfect calibration line (y = x)
            series_names.push("Perfectly calibrated".to_string());

            // Add points for the perfect calibration line
            x_values.push(0.0);
            x_values.push(1.0);
            y_values.push(0.0);
            y_values.push(1.0);
        }

        Ok(VisualizationData {
            x: x_values,
            y: y_values,
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
            x_label: "Mean predicted probability".to_string(),
            y_label: "Fraction of positives".to_string(),
            plot_type: PlotType::Line,
            description: Some("Calibration curve (reliability diagram) showing the relationship between predicted probabilities and the actual fraction of positive samples".to_string()),
        }
    }
}

/// Create a calibration curve visualization from pre-computed calibration curve data
///
/// # Arguments
///
/// * `fraction_of_positives` - Fraction of positive samples in each bin
/// * `mean_predicted_value` - Mean predicted probability in each bin
/// * `n_bins` - Number of bins used
/// * `strategy` - Strategy used for binning ("uniform" or "quantile")
///
/// # Returns
///
/// * A CalibrationVisualizer
pub fn calibration_visualization(
    fraction_of_positives: Vec<f64>,
    mean_predicted_value: Vec<f64>,
    n_bins: usize,
    strategy: String,
) -> CalibrationVisualizer<'static, f64, ndarray::OwnedRepr<f64>> {
    CalibrationVisualizer::new(
        fraction_of_positives,
        mean_predicted_value,
        n_bins,
        strategy,
    )
}

/// Create a calibration curve visualization from true labels and probabilities
///
/// # Arguments
///
/// * `y_true` - True binary labels
/// * `y_prob` - Predicted probabilities
/// * `n_bins` - Number of bins to use
/// * `strategy` - Strategy for binning
/// * `pos_label` - Optional label of the positive class
///
/// # Returns
///
/// * A CalibrationVisualizer
pub fn calibration_from_labels<'a, T, S>(
    y_true: &'a ArrayBase<S, Ix1>,
    y_prob: &'a ArrayBase<S, Ix1>,
    n_bins: usize,
    strategy: &str,
    pos_label: Option<T>,
) -> CalibrationVisualizer<'a, T, S>
where
    T: Clone + PartialOrd + 'static,
    S: Data<Elem = T>,
    f64: From<T>,
{
    CalibrationVisualizer::from_labels(y_true, y_prob, n_bins, strategy.to_string(), pos_label)
}
