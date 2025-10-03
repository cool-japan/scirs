//! Helper functions for visualization
//!
//! This module provides helper functions for creating visualizations for common
//! metrics result types.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::error::Error;

use crate::visualization::interactive::InteractiveOptions;
use crate::visualization::{ColorMap, PlotType, VisualizationData, VisualizationMetadata};

/// Create a confusion matrix visualization from a confusion matrix array
///
/// # Arguments
///
/// * `confusion_matrix` - The confusion matrix as a 2D array
/// * `class_names` - Optional class names
/// * `normalize` - Whether to normalize the confusion matrix
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - A visualizer for the confusion matrix
#[allow(dead_code)]
pub fn visualize_confusion_matrix<A>(
    confusion_matrix: ArrayView2<A>,
    class_names: Option<Vec<String>>,
    normalize: bool,
) -> Box<dyn crate::visualization::MetricVisualizer>
where
    A: Clone + Into<f64>,
{
    // Convert the confusion _matrix to f64
    let cm_f64 = Array2::from_shape_fn(confusion_matrix.dim(), |(i, j)| {
        confusion_matrix[[i, j]].clone().into()
    });

    crate::visualization::confusion_matrix::confusion_matrix_visualization(
        cm_f64,
        class_names,
        normalize,
    )
}

/// Create a ROC curve visualization
///
/// # Arguments
///
/// * `fpr` - False positive rates
/// * `tpr` - True positive rates
/// * `thresholds` - Optional thresholds
/// * `auc` - Optional area under the curve
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - A visualizer for the ROC curve
#[allow(dead_code)]
pub fn visualize_roc_curve<A>(
    fpr: ArrayView1<A>,
    tpr: ArrayView1<A>,
    thresholds: Option<ArrayView1<A>>,
    auc: Option<f64>,
) -> Box<dyn crate::visualization::MetricVisualizer>
where
    A: Clone + Into<f64>,
{
    // Convert the arrays to f64 vectors
    let fpr_vec = fpr.iter().map(|x| x.clone().into()).collect::<Vec<f64>>();
    let tpr_vec = tpr.iter().map(|x| x.clone().into()).collect::<Vec<f64>>();
    let thresholds_vec =
        thresholds.map(|t| t.iter().map(|x| x.clone().into()).collect::<Vec<f64>>());

    Box::new(crate::visualization::roc_curve::roc_curve_visualization(
        fpr_vec,
        tpr_vec,
        thresholds_vec,
        auc,
    ))
}

/// Create an interactive ROC curve visualization
///
/// # Arguments
///
/// * `fpr` - False positive rates
/// * `tpr` - True positive rates
/// * `thresholds` - Optional thresholds
/// * `auc` - Optional area under the curve
/// * `interactive_options` - Optional interactive options
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - An interactive visualizer for the ROC curve
#[allow(dead_code)]
pub fn visualize_interactive_roc_curve<A>(
    fpr: ArrayView1<A>,
    tpr: ArrayView1<A>,
    thresholds: Option<ArrayView1<A>>,
    auc: Option<f64>,
    interactive_options: Option<InteractiveOptions>,
) -> Box<dyn crate::visualization::MetricVisualizer>
where
    A: Clone + Into<f64>,
{
    // Convert the arrays to f64 vectors
    let fpr_vec = fpr.iter().map(|x| x.clone().into()).collect::<Vec<f64>>();
    let tpr_vec = tpr.iter().map(|x| x.clone().into()).collect::<Vec<f64>>();
    let thresholds_vec =
        thresholds.map(|t| t.iter().map(|x| x.clone().into()).collect::<Vec<f64>>());

    let mut visualizer = crate::visualization::interactive::interactive_roc_curve_visualization(
        fpr_vec,
        tpr_vec,
        thresholds_vec,
        auc,
    );

    if let Some(_options) = interactive_options {
        visualizer = visualizer.with_interactive_options(_options);
    }

    Box::new(visualizer)
}

/// Create an interactive ROC curve visualization from labels and scores
///
/// # Arguments
///
/// * `y_true` - True binary labels
/// * `y_score` - Target scores (probabilities or decision function output)
/// * `pos_label` - Optional positive class label
/// * `interactive_options` - Optional interactive options
///
/// # Returns
///
/// * `Result<Box<dyn crate::visualization::MetricVisualizer>, Box<dyn Error>>` - An interactive visualizer for the ROC curve
#[allow(dead_code)]
pub fn visualize_interactive_roc_from_labels<A, B>(
    y_true: ArrayView1<A>,
    y_score: ArrayView1<B>,
    _pos_label: Option<A>,
    interactive_options: Option<InteractiveOptions>,
) -> Result<Box<dyn crate::visualization::MetricVisualizer>, Box<dyn Error>>
where
    A: Clone + PartialOrd + 'static,
    B: Clone + PartialOrd + 'static,
    f64: From<A> + From<B>,
{
    // Compute ROC curve
    let (fpr, tpr, _thresholds) = crate::classification::curves::roc_curve(&y_true, &y_score)
        .map_err(|e| Box::new(e) as Box<dyn Error>)?;

    // Calculate AUC - simplified version
    let auc = {
        let n = fpr.len();
        let mut area = 0.0;
        for i in 1..n {
            area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0;
        }
        area
    };

    // Create an owned ROC curve visualizer using raw data
    let mut visualizer = crate::visualization::interactive::roc_curve::InteractiveROCVisualizer::<
        f64,
        scirs2_core::ndarray::OwnedRepr<f64>,
    >::new(fpr.to_vec(), tpr.to_vec(), None, Some(auc));

    if let Some(_options) = interactive_options {
        visualizer = visualizer.with_interactive_options(_options);
    }

    Ok(Box::new(visualizer))
}

/// Create a precision-recall curve visualization
///
/// # Arguments
///
/// * `precision` - Precision values
/// * `recall` - Recall values
/// * `thresholds` - Optional thresholds
/// * `average_precision` - Optional average precision
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - A visualizer for the precision-recall curve
#[allow(dead_code)]
pub fn visualize_precision_recall_curve<A>(
    precision: ArrayView1<A>,
    recall: ArrayView1<A>,
    thresholds: Option<ArrayView1<A>>,
    average_precision: Option<f64>,
) -> Box<dyn crate::visualization::MetricVisualizer>
where
    A: Clone + Into<f64>,
{
    // Convert the arrays to f64 vectors
    let precision_vec = precision
        .iter()
        .map(|x| x.clone().into())
        .collect::<Vec<f64>>();
    let recall_vec = recall
        .iter()
        .map(|x| x.clone().into())
        .collect::<Vec<f64>>();
    let thresholds_vec =
        thresholds.map(|t| t.iter().map(|x| x.clone().into()).collect::<Vec<f64>>());

    Box::new(
        crate::visualization::precision_recall::precision_recall_visualization(
            precision_vec,
            recall_vec,
            thresholds_vec,
            average_precision,
        ),
    )
}

/// Create a calibration curve visualization
///
/// # Arguments
///
/// * `prob_true` - True probabilities
/// * `prob_pred` - Predicted probabilities
/// * `n_bins` - Number of bins
/// * `strategy` - Binning strategy ("uniform" or "quantile")
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - A visualizer for the calibration curve
#[allow(dead_code)]
pub fn visualize_calibration_curve<A>(
    prob_true: ArrayView1<A>,
    prob_pred: ArrayView1<A>,
    n_bins: usize,
    strategy: impl Into<String>,
) -> Box<dyn crate::visualization::MetricVisualizer>
where
    A: Clone + Into<f64>,
{
    // Convert the arrays to f64 vectors
    let prob_true_vec = prob_true
        .iter()
        .map(|x| x.clone().into())
        .collect::<Vec<f64>>();
    let prob_pred_vec = prob_pred
        .iter()
        .map(|x| x.clone().into())
        .collect::<Vec<f64>>();

    Box::new(
        crate::visualization::calibration::calibration_visualization(
            prob_true_vec,
            prob_pred_vec,
            n_bins,
            strategy.into(),
        ),
    )
}

/// Create a learning curve visualization
///
/// # Arguments
///
/// * `train_sizes` - Training set sizes
/// * `train_scores` - Training scores (multiple runs for each size)
/// * `val_scores` - Validation scores (multiple runs for each size)
/// * `score_name` - Name of the score (e.g., "Accuracy")
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - A visualizer for the learning curve
/// * `Result<Box<dyn crate::visualization::MetricVisualizer>, Box<dyn Error>>` - A visualizer for the learning curve, or an error
#[allow(dead_code)]
pub fn visualize_learning_curve(
    train_sizes: Vec<usize>,
    train_scores: Vec<Vec<f64>>,
    val_scores: Vec<Vec<f64>>,
    score_name: impl Into<String>,
) -> Result<Box<dyn crate::visualization::MetricVisualizer>, Box<dyn Error>> {
    let visualizer = crate::visualization::learning_curve::learning_curve_visualization(
        train_sizes,
        train_scores,
        val_scores,
        score_name,
    )?;

    Ok(Box::new(visualizer))
}

/// Create a generic metric visualization
///
/// This function creates a visualization for generic metric data,
/// such as performance over time, hyperparameter tuning results, etc.
///
/// # Arguments
///
/// * `x_values` - X-axis values
/// * `y_values` - Y-axis values
/// * `title` - Plot title
/// * `x_label` - X-axis label
/// * `y_label` - Y-axis label
/// * `plot_type` - Plot type
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - A visualizer for the generic metric
#[allow(dead_code)]
pub fn visualize_metric<A, B>(
    x_values: ArrayView1<A>,
    y_values: ArrayView1<B>,
    title: impl Into<String>,
    x_label: impl Into<String>,
    y_label: impl Into<String>,
    plot_type: PlotType,
) -> Box<dyn crate::visualization::MetricVisualizer>
where
    A: Clone + Into<f64>,
    B: Clone + Into<f64>,
{
    let x_vec = x_values
        .iter()
        .map(|x| x.clone().into())
        .collect::<Vec<f64>>();
    let y_vec = y_values
        .iter()
        .map(|y| y.clone().into())
        .collect::<Vec<f64>>();

    Box::new(GenericMetricVisualizer::new(
        x_vec,
        y_vec,
        title.into(),
        x_label.into(),
        y_label.into(),
        plot_type,
    ))
}

/// A generic visualizer for metric data
pub struct GenericMetricVisualizer {
    /// X-axis values
    pub x: Vec<f64>,
    /// Y-axis values
    pub y: Vec<f64>,
    /// Title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Plot type
    pub plot_type: PlotType,
    /// Optional series names
    pub series_names: Option<Vec<String>>,
}

impl GenericMetricVisualizer {
    /// Create a new generic metric visualizer
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
        plot_type: PlotType,
    ) -> Self {
        Self {
            x,
            y,
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            plot_type,
            series_names: None,
        }
    }

    /// Add series names
    pub fn with_series_names(mut self, seriesnames: Vec<String>) -> Self {
        self.series_names = Some(seriesnames);
        self
    }
}

impl crate::visualization::MetricVisualizer for GenericMetricVisualizer {
    fn prepare_data(&self) -> Result<VisualizationData, Box<dyn Error>> {
        let mut data = VisualizationData::new();

        // Set x and y data
        data.x = self.x.clone();
        data.y = self.y.clone();

        // Add series names if available
        if let Some(series_names) = &self.series_names {
            data.series_names = Some(series_names.clone());
        }

        Ok(data)
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        let mut metadata = VisualizationMetadata::new(self.title.clone());
        metadata.set_plot_type(self.plot_type.clone());
        metadata.set_x_label(self.x_label.clone());
        metadata.set_y_label(self.y_label.clone());
        metadata
    }
}

/// Create a multi-curve visualization
///
/// This function creates a visualization with multiple curves,
/// such as performance comparisons between different models.
///
/// # Arguments
///
/// * `x_values` - X-axis values (common for all curves)
/// * `y_values_list` - List of Y-axis values, one for each curve
/// * `series_names` - Names for each curve
/// * `title` - Plot title
/// * `x_label` - X-axis label
/// * `y_label` - Y-axis label
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - A visualizer for the multi-curve plot
#[allow(dead_code)]
pub fn visualize_multi_curve<A, B>(
    x_values: ArrayView1<A>,
    y_values_list: Vec<ArrayView1<B>>,
    series_names: Vec<String>,
    title: impl Into<String>,
    x_label: impl Into<String>,
    y_label: impl Into<String>,
) -> Box<dyn crate::visualization::MetricVisualizer>
where
    A: Clone + Into<f64>,
    B: Clone + Into<f64>,
{
    let x_vec = x_values
        .iter()
        .map(|x| x.clone().into())
        .collect::<Vec<f64>>();

    // Set the first y-_values as the main y-axis data
    let y_vec = if !y_values_list.is_empty() {
        y_values_list[0]
            .iter()
            .map(|y| y.clone().into())
            .collect::<Vec<f64>>()
    } else {
        Vec::new()
    };

    // Create a visualizer
    let mut visualizer =
        MultiCurveVisualizer::new(x_vec, y_vec, title.into(), x_label.into(), y_label.into());

    // Add all series
    for (i, y_values) in y_values_list.iter().enumerate() {
        if i == 0 {
            // Skip the first one, already added as main y-axis
            continue;
        }

        let name = if i < series_names.len() {
            series_names[i].clone()
        } else {
            format!("Series {}", i + 1)
        };

        let y_vec = y_values
            .iter()
            .map(|y| y.clone().into())
            .collect::<Vec<f64>>();
        visualizer.add_series(name, y_vec);
    }

    // Set all series _names
    visualizer.set_series_names(series_names);

    Box::new(visualizer)
}

/// A visualizer for multi-curve plots
pub struct MultiCurveVisualizer {
    /// X-axis values
    pub x: Vec<f64>,
    /// Y-axis values for the main curve
    pub y: Vec<f64>,
    /// Additional Y-axis values for secondary curves
    pub secondary_y: Vec<(String, Vec<f64>)>,
    /// Title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Series names
    pub series_names: Vec<String>,
}

impl MultiCurveVisualizer {
    /// Create a new multi-curve visualizer
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        Self {
            x,
            y,
            secondary_y: Vec::new(),
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            series_names: Vec::new(),
        }
    }

    /// Add a secondary curve
    pub fn add_series(&mut self, name: impl Into<String>, y: Vec<f64>) {
        self.secondary_y.push((name.into(), y));
    }

    /// Set series names
    pub fn set_series_names(&mut self, names: Vec<String>) {
        self.series_names = names;
    }
}

impl crate::visualization::MetricVisualizer for MultiCurveVisualizer {
    fn prepare_data(&self) -> Result<VisualizationData, Box<dyn Error>> {
        let mut data = VisualizationData::new();

        // Set main x and y data
        data.x = self.x.clone();
        data.y = self.y.clone();

        // Add secondary curves
        for (name, y) in &self.secondary_y {
            data.series.insert(name.clone(), y.clone());
        }

        // Add series names
        if !self.series_names.is_empty() {
            data.series_names = Some(self.series_names.clone());
        }

        Ok(data)
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        let mut metadata = VisualizationMetadata::new(self.title.clone());
        metadata.set_plot_type(PlotType::Line);
        metadata.set_x_label(self.x_label.clone());
        metadata.set_y_label(self.y_label.clone());
        metadata
    }
}

/// Create a heatmap visualization
///
/// This function creates a heatmap visualization for 2D data,
/// such as correlation matrices, distance matrices, etc.
///
/// # Arguments
///
/// * `matrix` - 2D data matrix
/// * `x_labels` - Optional labels for x-axis
/// * `y_labels` - Optional labels for y-axis
/// * `title` - Plot title
/// * `color_map` - Optional color map
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - A visualizer for the heatmap
#[allow(dead_code)]
pub fn visualize_heatmap<A>(
    matrix: ArrayView2<A>,
    x_labels: Option<Vec<String>>,
    y_labels: Option<Vec<String>>,
    title: impl Into<String>,
    color_map: Option<ColorMap>,
) -> Box<dyn crate::visualization::MetricVisualizer>
where
    A: Clone + Into<f64>,
{
    // Convert matrix to Vec<Vec<f64>>
    let z = Array2::from_shape_fn(matrix.dim(), |(i, j)| matrix[[i, j]].clone().into());

    let z_vec = (0..z.shape()[0])
        .map(|i| (0..z.shape()[1]).map(|j| z[[i, j]]).collect::<Vec<f64>>())
        .collect::<Vec<Vec<f64>>>();

    // Create x and y coordinates for the heatmap
    let x = (0..z.shape()[1]).map(|i| i as f64).collect::<Vec<f64>>();
    let y = (0..z.shape()[0]).map(|i| i as f64).collect::<Vec<f64>>();

    Box::new(HeatmapVisualizer::new(
        x,
        y,
        z_vec,
        title.into(),
        x_labels,
        y_labels,
        color_map,
    ))
}

/// A visualizer for heatmaps
pub struct HeatmapVisualizer {
    /// X-axis values
    pub x: Vec<f64>,
    /// Y-axis values
    pub y: Vec<f64>,
    /// Z-axis values (2D matrix)
    pub z: Vec<Vec<f64>>,
    /// Title
    pub title: String,
    /// X-axis labels
    pub x_labels: Option<Vec<String>>,
    /// Y-axis labels
    pub y_labels: Option<Vec<String>>,
    /// Color map
    pub color_map: Option<ColorMap>,
}

impl HeatmapVisualizer {
    /// Create a new heatmap visualizer
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<Vec<f64>>,
        title: impl Into<String>,
        x_labels: Option<Vec<String>>,
        y_labels: Option<Vec<String>>,
        color_map: Option<ColorMap>,
    ) -> Self {
        Self {
            x,
            y,
            z,
            title: title.into(),
            x_labels,
            y_labels,
            color_map,
        }
    }
}

impl crate::visualization::MetricVisualizer for HeatmapVisualizer {
    fn prepare_data(&self) -> Result<VisualizationData, Box<dyn Error>> {
        let mut data = VisualizationData::new();

        // Set x, y, and z data
        data.x = self.x.clone();
        data.y = self.y.clone();
        data.z = Some(self.z.clone());

        // Add axis labels if available
        if let Some(x_labels) = &self.x_labels {
            data.x_labels = Some(x_labels.clone());
        }

        if let Some(y_labels) = &self.y_labels {
            data.y_labels = Some(y_labels.clone());
        }

        Ok(data)
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        let mut metadata = VisualizationMetadata::new(self.title.clone());
        metadata.set_plot_type(PlotType::Heatmap);

        // Set default axis labels if none are provided
        if self.x_labels.is_none() {
            metadata.set_x_label("X");
        } else {
            metadata.set_x_label(""); // Labels are provided directly
        }

        if self.y_labels.is_none() {
            metadata.set_y_label("Y");
        } else {
            metadata.set_y_label(""); // Labels are provided directly
        }

        metadata
    }
}

/// Create a histogram visualization
///
/// This function creates a histogram visualization for 1D data.
///
/// # Arguments
///
/// * `values` - Data values
/// * `bins` - Number of bins
/// * `title` - Plot title
/// * `x_label` - X-axis label
/// * `y_label` - Y-axis label (defaults to "Frequency")
///
/// # Returns
///
/// * `Box<dyn crate::visualization::MetricVisualizer>` - A visualizer for the histogram
#[allow(dead_code)]
pub fn visualize_histogram<A>(
    values: ArrayView1<A>,
    bins: usize,
    title: impl Into<String>,
    x_label: impl Into<String>,
    y_label: Option<String>,
) -> Box<dyn crate::visualization::MetricVisualizer>
where
    A: Clone + Into<f64>,
{
    // Convert values to f64 vector
    let values_vec = values
        .iter()
        .map(|x| x.clone().into())
        .collect::<Vec<f64>>();

    // Create histogram bins
    let (bin_edges, bin_counts) = create_histogram_bins(&values_vec, bins);

    Box::new(HistogramVisualizer::new(
        bin_edges,
        bin_counts,
        title.into(),
        x_label.into(),
        y_label.unwrap_or_else(|| "Frequency".to_string()),
    ))
}

/// Create histogram bins from data values
///
/// # Arguments
///
/// * `values` - Data values
/// * `bins` - Number of bins
///
/// # Returns
///
/// * `(Vec<f64>, Vec<f64>)` - Bin edges and bin counts
#[allow(dead_code)]
fn create_histogram_bins(values: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    // Ensure we have at least one value and valid bins
    if values.is_empty() || bins == 0 {
        return (Vec::new(), Vec::new());
    }

    // Find min and max _values
    let min_val = values.iter().fold(f64::INFINITY, |min, &val| min.min(val));
    let max_val = values
        .iter()
        .fold(f64::NEG_INFINITY, |max, &val| max.max(val));

    // Create bin edges
    let bin_width = (max_val - min_val) / bins as f64;
    let mut bin_edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        bin_edges.push(min_val + i as f64 * bin_width);
    }

    // Count _values in each bin
    let mut bin_counts = vec![0.0; bins];
    for &val in values {
        if val >= min_val && val <= max_val {
            let bin_idx = ((val - min_val) / bin_width).floor() as usize;
            // Handle the edge case where val is exactly max_val
            let bin_idx = bin_idx.min(bins - 1);
            bin_counts[bin_idx] += 1.0;
        }
    }

    (bin_edges, bin_counts)
}

/// A visualizer for histograms
pub struct HistogramVisualizer {
    /// Bin edges
    pub bin_edges: Vec<f64>,
    /// Bin counts
    pub bin_counts: Vec<f64>,
    /// Title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
}

impl HistogramVisualizer {
    /// Create a new histogram visualizer
    pub fn new(
        bin_edges: Vec<f64>,
        bin_counts: Vec<f64>,
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        Self {
            bin_edges,
            bin_counts,
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
        }
    }
}

impl crate::visualization::MetricVisualizer for HistogramVisualizer {
    fn prepare_data(&self) -> Result<VisualizationData, Box<dyn Error>> {
        let mut data = VisualizationData::new();

        // Use bin centers as x values
        if self.bin_edges.len() > 1 {
            let bin_centers = self
                .bin_edges
                .windows(2)
                .map(|w| (w[0] + w[1]) / 2.0)
                .collect::<Vec<f64>>();

            data.x = bin_centers;
        } else {
            data.x = Vec::new();
        }

        // Use bin counts as y values
        data.y = self.bin_counts.clone();

        // Store bin edges in auxiliary data
        data.add_auxiliary_data("bin_edges", self.bin_edges.clone());

        Ok(data)
    }

    fn get_metadata(&self) -> VisualizationMetadata {
        let mut metadata = VisualizationMetadata::new(self.title.clone());
        metadata.set_plot_type(PlotType::Histogram);
        metadata.set_x_label(self.x_label.clone());
        metadata.set_y_label(self.y_label.clone());
        metadata
    }
}
