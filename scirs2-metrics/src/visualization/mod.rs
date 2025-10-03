//! Visualization utilities for metrics
//!
//! This module provides tools for visualizing metrics results, including confusion
//! matrices, ROC curves, precision-recall curves, calibration plots, and learning curves.
//!
//! The visualization module is designed to work with popular Rust plotting libraries
//! and provides data structures that can be easily converted to formats used by those libraries.
//!
//! # Basic Usage
//!
//! ```
//! use scirs2_metrics::visualization::{
//!     VisualizationData, VisualizationMetadata, PlotType, PlottingBackend, backends
//! };
//!
//! // Create visualization data
//! let mut data = VisualizationData::new();
//! data.add_series("x", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! data.add_series("y", vec![2.0, 4.0, 1.0, 3.0, 7.0]);
//!
//! // Create visualization metadata
//! let mut metadata = VisualizationMetadata::new("Example Plot");
//! metadata.set_plot_type(PlotType::Line);
//! metadata.set_x_label("X Axis");
//! metadata.set_y_label("Y Axis");
//!
//! // Get the default backend
//! let backend = backends::default_backend();
//!
//! // Save the visualization to a file
//! backend.save_to_file(&data, &metadata, &Default::default(), "plot.html").unwrap();
//! ```
//!
//! # Using Specific Visualizers
//!
//! The module provides specialized visualizers for common metrics visualizations:
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_metrics::classification::confusion_matrix;
//! use scirs2_metrics::visualization::{
//!     confusion_matrix::confusion_matrix_visualization,
//!     PlottingBackend, backends
//! };
//!
//! // Example data
//! let y_true = array![0, 1, 2, 0, 1, 2];
//! let y_pred = array![0, 2, 1, 0, 0, 2];
//!
//! // Create confusion matrix
//! let (cm, _classes) = confusion_matrix(&y_true, &y_pred, None).unwrap();
//! let cm_f64 = cm.mapv(|x| x as f64);
//!
//! // Create visualizer
//! let visualizer = confusion_matrix_visualization(
//!     cm_f64,
//!     Some(vec!["Class 0".to_string(), "Class 1".to_string(), "Class 2".to_string()]),
//!     false
//! );
//!
//! // Get data and metadata
//! let viz_data = visualizer.prepare_data().unwrap();
//! let viz_metadata = visualizer.get_metadata();
//!
//! // Save the visualization
//! let backend = backends::default_backend();
//! backend.save_to_file(&viz_data, &viz_metadata, &Default::default(), "confusion_matrix.html").unwrap();
//! ```

// Re-export submodules
pub mod advanced_interactive;
pub mod backends;
pub mod calibration;
pub mod confusion_matrix;
pub mod helpers;
pub mod interactive;
pub mod learning_curve;
pub mod precision_recall;
pub mod roc_curve;

// Re-export common functionality
pub use advanced_interactive::{
    CollaborationConfig, CollaborationManager, DashboardConfig, DashboardState, DataSource,
    EventResponse, EventSystem, ExportConfig, InteractionConfig, InteractiveDashboard,
    InteractiveWidget, LayoutConfig, LayoutManager, RealtimeConfig, RenderingEngine, ThemeConfig,
    UpdateManager, WidgetConfig, WidgetEvent, WidgetType,
};
pub use backends::PlottingBackend;
pub use calibration::CalibrationVisualizer;
pub use confusion_matrix::ConfusionMatrixVisualizer;
pub use helpers::*;
pub use interactive::{
    interactive_roc_curve_from_labels, interactive_roc_curve_visualization, InteractiveOptions,
    InteractiveROCVisualizer,
};
pub use learning_curve::LearningCurveVisualizer;
pub use precision_recall::PrecisionRecallVisualizer;
pub use roc_curve::ROCCurveVisualizer;

use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

/// Common trait for metric visualizers
///
/// This trait provides a common interface for all metric visualizers.
pub trait MetricVisualizer {
    /// Prepare data for visualization
    fn prepare_data(&self) -> Result<VisualizationData, Box<dyn Error>>;

    /// Get visualization metadata
    fn get_metadata(&self) -> VisualizationMetadata;
}

/// Helper functions for saving visualizations
pub trait Visualization {
    /// Save the visualization to a file using the default backend
    ///
    /// # Arguments
    ///
    /// * `path` - The output file path
    /// * `options` - Optional visualization options
    ///
    /// # Returns
    ///
    /// * `Result<(), Box<dyn Error>>` - Ok if the visualization was successfully saved,
    ///   or an error if something went wrong
    fn save_to_file(
        &self,
        path: impl AsRef<Path>,
        options: Option<VisualizationOptions>,
    ) -> Result<(), Box<dyn Error>>;

    /// Render the visualization as SVG
    ///
    /// # Arguments
    ///
    /// * `options` - Optional visualization options
    ///
    /// # Returns
    ///
    /// * `Result<Vec<u8>, Box<dyn Error>>` - A byte array containing the SVG representation
    ///   of the visualization
    fn render_svg(&self, options: Option<VisualizationOptions>) -> Result<Vec<u8>, Box<dyn Error>>;

    /// Render the visualization as PNG
    ///
    /// # Arguments
    ///
    /// * `options` - Optional visualization options
    ///
    /// # Returns
    ///
    /// * `Result<Vec<u8>, Box<dyn Error>>` - A byte array containing the PNG representation
    ///   of the visualization
    fn render_png(&self, options: Option<VisualizationOptions>) -> Result<Vec<u8>, Box<dyn Error>>;
}

impl<T: MetricVisualizer> Visualization for T {
    fn save_to_file(
        &self,
        path: impl AsRef<Path>,
        options: Option<VisualizationOptions>,
    ) -> Result<(), Box<dyn Error>> {
        let data = self.prepare_data()?;
        let metadata = self.get_metadata();
        let options = options.unwrap_or_default();

        let backend = backends::default_backend();
        backend.save_to_file(&data, &metadata, &options, path)?;

        Ok(())
    }

    fn render_svg(&self, options: Option<VisualizationOptions>) -> Result<Vec<u8>, Box<dyn Error>> {
        let data = self.prepare_data()?;
        let metadata = self.get_metadata();
        let options = options.unwrap_or_default();

        let backend = backends::default_backend();
        backend.render_svg(&data, &metadata, &options)
    }

    fn render_png(&self, options: Option<VisualizationOptions>) -> Result<Vec<u8>, Box<dyn Error>> {
        let data = self.prepare_data()?;
        let metadata = self.get_metadata();
        let options = options.unwrap_or_default();

        let backend = backends::default_backend();
        backend.render_png(&data, &metadata, &options)
    }
}

/// Data structure for visualization
///
/// This structure contains the data needed for visualization, which can be
/// converted to formats used by popular plotting libraries.
#[derive(Debug, Clone)]
pub struct VisualizationData {
    /// X-axis data
    pub x: Vec<f64>,
    /// Y-axis data
    pub y: Vec<f64>,
    /// Optional Z-axis data for heatmaps
    pub z: Option<Vec<Vec<f64>>>,
    /// Optional series names for multi-series plots
    pub series_names: Option<Vec<String>>,
    /// Optional x-axis labels
    pub x_labels: Option<Vec<String>>,
    /// Optional y-axis labels
    pub y_labels: Option<Vec<String>>,
    /// Auxiliary data for enhanced plotting
    pub auxiliary_data: HashMap<String, Vec<f64>>,
    /// Auxiliary metadata (string key-value pairs)
    pub auxiliary_metadata: HashMap<String, String>,
    /// Multiple data series for complex plots
    pub series: HashMap<String, Vec<f64>>,
}

impl VisualizationData {
    /// Create a new visualization data structure
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: None,
            series_names: None,
            x_labels: None,
            y_labels: None,
            auxiliary_data: HashMap::new(),
            auxiliary_metadata: HashMap::new(),
            series: HashMap::new(),
        }
    }

    /// Add a data series
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the series
    /// * `data` - The data points
    pub fn add_series(&mut self, name: impl Into<String>, data: Vec<f64>) {
        let name = name.into();

        // If this is the first series, use it as x
        if self.x.is_empty() && name.to_lowercase().contains("x") {
            self.x = data;
            return;
        }

        // If this is the second series, use it as y
        if self.y.is_empty() && name.to_lowercase().contains("y") {
            self.y = data;
            return;
        }

        // Otherwise, add it to the series map
        self.series.insert(name, data);
    }

    /// Add a 2D data series for heatmaps
    ///
    /// # Arguments
    ///
    /// * `data` - The 2D data
    pub fn add_heatmap_data(&mut self, data: Vec<Vec<f64>>) {
        self.z = Some(data);
    }

    /// Add x-axis labels
    ///
    /// # Arguments
    ///
    /// * `labels` - The x-axis labels
    pub fn add_x_labels(&mut self, labels: Vec<String>) {
        self.x_labels = Some(labels);
    }

    /// Add y-axis labels
    ///
    /// # Arguments
    ///
    /// * `labels` - The y-axis labels
    pub fn add_y_labels(&mut self, labels: Vec<String>) {
        self.y_labels = Some(labels);
    }

    /// Add series names
    ///
    /// # Arguments
    ///
    /// * `names` - The series names
    pub fn add_series_names(&mut self, names: Vec<String>) {
        self.series_names = Some(names);
    }

    /// Add auxiliary data
    ///
    /// # Arguments
    ///
    /// * `key` - The data key
    /// * `data` - The data points
    pub fn add_auxiliary_data(&mut self, key: impl Into<String>, data: Vec<f64>) {
        self.auxiliary_data.insert(key.into(), data);
    }

    /// Add auxiliary metadata
    ///
    /// # Arguments
    ///
    /// * `key` - The metadata key
    /// * `value` - The metadata value
    pub fn add_auxiliary_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.auxiliary_metadata.insert(key.into(), value.into());
    }
}

impl Default for VisualizationData {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for visualization
///
/// This structure contains the metadata for visualization, including
/// titles, labels, and plot type.
#[derive(Debug, Clone)]
pub struct VisualizationMetadata {
    /// Plot title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Plot type
    pub plot_type: PlotType,
    /// Optional plot description
    pub description: Option<String>,
}

impl VisualizationMetadata {
    /// Create a new visualization metadata structure
    ///
    /// # Arguments
    ///
    /// * `title` - The plot title
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            plot_type: PlotType::Line,
            description: None,
        }
    }

    /// Set the plot type
    ///
    /// # Arguments
    ///
    /// * `plot_type` - The plot type
    pub fn set_plot_type(&mut self, plottype: PlotType) {
        self.plot_type = plottype;
    }

    /// Set the x-axis label
    ///
    /// # Arguments
    ///
    /// * `x_label` - The x-axis label
    pub fn set_x_label(&mut self, xlabel: impl Into<String>) {
        self.x_label = xlabel.into();
    }

    /// Set the y-axis label
    ///
    /// # Arguments
    ///
    /// * `y_label` - The y-axis label
    pub fn set_y_label(&mut self, ylabel: impl Into<String>) {
        self.y_label = ylabel.into();
    }

    /// Set the plot description
    ///
    /// # Arguments
    ///
    /// * `description` - The plot description
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.description = Some(description.into());
    }

    /// Create a line plot metadata
    ///
    /// # Arguments
    ///
    /// * `title` - The plot title
    /// * `x_label` - The x-axis label
    /// * `y_label` - The y-axis label
    pub fn line_plot(
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        let mut metadata = Self::new(title);
        metadata.set_plot_type(PlotType::Line);
        metadata.set_x_label(x_label);
        metadata.set_y_label(y_label);
        metadata
    }

    /// Create a scatter plot metadata
    ///
    /// # Arguments
    ///
    /// * `title` - The plot title
    /// * `x_label` - The x-axis label
    /// * `y_label` - The y-axis label
    pub fn scatter_plot(
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        let mut metadata = Self::new(title);
        metadata.set_plot_type(PlotType::Scatter);
        metadata.set_x_label(x_label);
        metadata.set_y_label(y_label);
        metadata
    }

    /// Create a bar chart metadata
    ///
    /// # Arguments
    ///
    /// * `title` - The plot title
    /// * `x_label` - The x-axis label
    /// * `y_label` - The y-axis label
    pub fn bar_chart(
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        let mut metadata = Self::new(title);
        metadata.set_plot_type(PlotType::Bar);
        metadata.set_x_label(x_label);
        metadata.set_y_label(y_label);
        metadata
    }

    /// Create a heatmap metadata
    ///
    /// # Arguments
    ///
    /// * `title` - The plot title
    /// * `x_label` - The x-axis label
    /// * `y_label` - The y-axis label
    pub fn heatmap(
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        let mut metadata = Self::new(title);
        metadata.set_plot_type(PlotType::Heatmap);
        metadata.set_x_label(x_label);
        metadata.set_y_label(y_label);
        metadata
    }

    /// Create a histogram metadata
    ///
    /// # Arguments
    ///
    /// * `title` - The plot title
    /// * `x_label` - The x-axis label
    /// * `y_label` - The y-axis label (usually "Frequency" or "Count")
    pub fn histogram(
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
    ) -> Self {
        let mut metadata = Self::new(title);
        metadata.set_plot_type(PlotType::Histogram);
        metadata.set_x_label(x_label);
        metadata.set_y_label(y_label);
        metadata
    }
}

/// Plot types supported by the visualizers
#[derive(Debug, Clone, PartialEq)]
pub enum PlotType {
    /// Line plot
    Line,
    /// Scatter plot
    Scatter,
    /// Bar chart
    Bar,
    /// Heatmap
    Heatmap,
    /// Histogram
    Histogram,
}

/// Color maps for heatmaps
#[derive(Debug, Clone, PartialEq)]
pub enum ColorMap {
    /// Blue to red
    BlueRed,
    /// Green to red
    GreenRed,
    /// Grayscale
    Grayscale,
    /// Viridis
    Viridis,
    /// Magma
    Magma,
}

/// Options for visualization
#[derive(Debug, Clone)]
pub struct VisualizationOptions {
    /// Figure width
    pub width: usize,
    /// Figure height
    pub height: usize,
    /// DPI
    pub dpi: usize,
    /// Color map for heatmaps
    pub color_map: Option<ColorMap>,
    /// Whether to show grid
    pub show_grid: bool,
    /// Whether to show legend
    pub show_legend: bool,
    /// Font size for title
    pub title_font_size: Option<f64>,
    /// Font size for labels
    pub label_font_size: Option<f64>,
    /// Font size for tick labels
    pub tick_font_size: Option<f64>,
    /// Line width for line plots
    pub line_width: Option<f64>,
    /// Marker size for scatter plots
    pub marker_size: Option<f64>,
    /// Whether to show colorbar for heatmaps
    pub show_colorbar: bool,
    /// Default color palette for multiple series
    pub color_palette: Option<String>,
}

impl Default for VisualizationOptions {
    fn default() -> Self {
        VisualizationOptions {
            width: 800,
            height: 600,
            dpi: 100,
            color_map: None,
            show_grid: true,
            show_legend: true,
            title_font_size: None,
            label_font_size: None,
            tick_font_size: None,
            line_width: None,
            marker_size: None,
            show_colorbar: true,
            color_palette: None,
        }
    }
}

impl VisualizationOptions {
    /// Create a new visualization options structure
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the figure width
    ///
    /// # Arguments
    ///
    /// * `width` - The figure width in pixels
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set the figure height
    ///
    /// # Arguments
    ///
    /// * `height` - The figure height in pixels
    pub fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    /// Set the figure DPI
    ///
    /// # Arguments
    ///
    /// * `dpi` - The figure DPI
    pub fn with_dpi(mut self, dpi: usize) -> Self {
        self.dpi = dpi;
        self
    }

    /// Set the color map
    ///
    /// # Arguments
    ///
    /// * `color_map` - The color map
    pub fn with_color_map(mut self, colormap: ColorMap) -> Self {
        self.color_map = Some(colormap);
        self
    }

    /// Set whether to show grid
    ///
    /// # Arguments
    ///
    /// * `show_grid` - Whether to show grid
    pub fn with_grid(mut self, showgrid: bool) -> Self {
        self.show_grid = showgrid;
        self
    }

    /// Set whether to show legend
    ///
    /// # Arguments
    ///
    /// * `show_legend` - Whether to show legend
    pub fn with_legend(mut self, showlegend: bool) -> Self {
        self.show_legend = showlegend;
        self
    }

    /// Set the font sizes
    ///
    /// # Arguments
    ///
    /// * `title` - The title font size
    /// * `label` - The label font size
    /// * `tick` - The tick font size
    pub fn with_font_sizes(
        mut self,
        title: Option<f64>,
        label: Option<f64>,
        tick: Option<f64>,
    ) -> Self {
        self.title_font_size = title;
        self.label_font_size = label;
        self.tick_font_size = tick;
        self
    }

    /// Set the line width
    ///
    /// # Arguments
    ///
    /// * `line_width` - The line width
    pub fn with_line_width(mut self, linewidth: f64) -> Self {
        self.line_width = Some(linewidth);
        self
    }

    /// Set the marker size
    ///
    /// # Arguments
    ///
    /// * `marker_size` - The marker size
    pub fn with_marker_size(mut self, markersize: f64) -> Self {
        self.marker_size = Some(markersize);
        self
    }

    /// Set whether to show colorbar
    ///
    /// # Arguments
    ///
    /// * `show_colorbar` - Whether to show colorbar
    pub fn with_colorbar(mut self, showcolorbar: bool) -> Self {
        self.show_colorbar = showcolorbar;
        self
    }

    /// Set the color palette
    ///
    /// # Arguments
    ///
    /// * `color_palette` - The color palette name
    pub fn with_color_palette(mut self, colorpalette: impl Into<String>) -> Self {
        self.color_palette = Some(colorpalette.into());
        self
    }
}
