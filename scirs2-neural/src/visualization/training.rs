//! Training metrics and curve visualization for neural networks
//!
//! This module provides comprehensive tools for visualizing training progress
//! including loss curves, accuracy metrics, learning rate schedules, and system performance.

use super::config::{DownsamplingStrategy, VisualizationConfig};
use crate::error::{NeuralError, Result};

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::path::PathBuf;

/// Training metrics visualizer
#[allow(dead_code)]
pub struct TrainingVisualizer<F: Float + Debug> {
    /// Training history
    metrics_history: Vec<TrainingMetrics<F>>,
    /// Visualization configuration
    config: VisualizationConfig,
    /// Active plots
    active_plots: HashMap<String, PlotConfig>,
}

/// Training metrics for a single epoch/step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics<F: Float + Debug> {
    /// Epoch number
    pub epoch: usize,
    /// Step number within epoch
    pub step: usize,
    /// Timestamp
    pub timestamp: String,
    /// Loss values
    pub losses: HashMap<String, F>,
    /// Accuracy metrics
    pub accuracies: HashMap<String, F>,
    /// Learning rate
    pub learning_rate: F,
    /// Other custom metrics
    pub custom_metrics: HashMap<String, F>,
    /// System metrics
    pub system_metrics: SystemMetrics,
}

/// System performance metrics during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// GPU memory usage in MB (if available)
    pub gpu_memory_mb: Option<f64>,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// GPU utilization percentage (if available)
    pub gpu_utilization: Option<f64>,
    /// Training step duration in milliseconds
    pub step_duration_ms: f64,
    /// Samples processed per second
    pub samples_per_second: f64,
}

/// Plot configuration
#[derive(Debug, Clone, Serialize)]
pub struct PlotConfig {
    /// Plot title
    pub title: String,
    /// X-axis configuration
    pub x_axis: AxisConfig,
    /// Y-axis configuration
    pub y_axis: AxisConfig,
    /// Series to plot
    pub series: Vec<SeriesConfig>,
    /// Plot type
    pub plot_type: PlotType,
    /// Update mode
    pub update_mode: UpdateMode,
}

/// Axis configuration
#[derive(Debug, Clone, Serialize)]
pub struct AxisConfig {
    /// Axis label
    pub label: String,
    /// Axis scale
    pub scale: AxisScale,
    /// Range (None for auto)
    pub range: Option<(f64, f64)>,
    /// Show grid lines
    pub show_grid: bool,
    /// Tick configuration
    pub ticks: TickConfig,
}

/// Axis scale type
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AxisScale {
    /// Linear scale
    Linear,
    /// Logarithmic scale
    Log,
    /// Square root scale
    Sqrt,
    /// Custom scale
    Custom(String),
}

/// Tick configuration
#[derive(Debug, Clone, Serialize)]
pub struct TickConfig {
    /// Tick interval (None for auto)
    pub interval: Option<f64>,
    /// Tick format
    pub format: TickFormat,
    /// Show tick labels
    pub show_labels: bool,
    /// Tick rotation angle
    pub rotation: f32,
}

/// Tick format options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum TickFormat {
    /// Automatic formatting
    Auto,
    /// Fixed decimal places
    Fixed(u32),
    /// Scientific notation
    Scientific,
    /// Percentage
    Percentage,
    /// Custom format string
    Custom(String),
}

/// Data series configuration
#[derive(Debug, Clone, Serialize)]
pub struct SeriesConfig {
    /// Series name
    pub name: String,
    /// Data source (metric name)
    pub data_source: String,
    /// Line style
    pub style: LineStyleConfig,
    /// Marker style
    pub markers: MarkerConfig,
    /// Series color
    pub color: String,
    /// Series opacity
    pub opacity: f32,
}

/// Line style configuration for series
#[derive(Debug, Clone, Serialize)]
pub struct LineStyleConfig {
    /// Line style
    pub style: LineStyle,
    /// Line width
    pub width: f32,
    /// Smoothing enabled
    pub smoothing: bool,
    /// Smoothing window size
    pub smoothing_window: usize,
}

/// Line style options (re-exported from network module)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum LineStyle {
    /// Solid line
    Solid,
    /// Dashed line
    Dashed,
    /// Dotted line
    Dotted,
    /// Dash-dot line
    DashDot,
}

/// Marker configuration for data points
#[derive(Debug, Clone, Serialize)]
pub struct MarkerConfig {
    /// Show markers
    pub show: bool,
    /// Marker shape
    pub shape: MarkerShape,
    /// Marker size
    pub size: f32,
    /// Marker fill color
    pub fill_color: String,
    /// Marker border color
    pub border_color: String,
}

/// Marker shape options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum MarkerShape {
    /// Circle marker
    Circle,
    /// Square marker
    Square,
    /// Triangle marker
    Triangle,
    /// Diamond marker
    Diamond,
    /// Cross marker
    Cross,
    /// Plus marker
    Plus,
}

/// Plot type options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum PlotType {
    /// Line plot
    Line,
    /// Scatter plot
    Scatter,
    /// Bar plot
    Bar,
    /// Area plot
    Area,
    /// Histogram
    Histogram,
    /// Box plot
    Box,
    /// Heatmap
    Heatmap,
}

/// Update mode for plots
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum UpdateMode {
    /// Append new data
    Append,
    /// Replace all data
    Replace,
    /// Rolling window
    Rolling(usize),
}

// Implementation for TrainingVisualizer

impl<F: Float + Debug + 'static + num_traits::FromPrimitive + Send + Sync> TrainingVisualizer<F> {
    /// Create a new training visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            metrics_history: Vec::new(),
            config,
            active_plots: HashMap::new(),
        }
    }

    /// Add training metrics for visualization
    pub fn add_metrics(&mut self, metrics: TrainingMetrics<F>) {
        self.metrics_history.push(metrics);

        // Apply downsampling if needed
        if self.metrics_history.len() > self.config.performance.max_points_per_plot
            && self.config.performance.enable_downsampling
        {
            self.downsample_metrics();
        }
    }

    /// Generate training curves visualization
    pub fn visualize_training_curves(&self) -> Result<Vec<PathBuf>> {
        let mut output_files = Vec::new();

        // Generate loss curves
        if let Some(loss_plot) = self.create_loss_plot()? {
            let loss_path = self.config.output_dir.join("training_loss.html");
            fs::write(&loss_path, loss_plot)
                .map_err(|e| NeuralError::IOError(format!("Failed to write loss plot: {}", e)))?;
            output_files.push(loss_path);
        }

        // Generate accuracy curves
        if let Some(accuracy_plot) = self.create_accuracy_plot()? {
            let accuracy_path = self.config.output_dir.join("training_accuracy.html");
            fs::write(&accuracy_path, accuracy_plot).map_err(|e| {
                NeuralError::IOError(format!("Failed to write accuracy plot: {}", e))
            })?;
            output_files.push(accuracy_path);
        }

        // Generate learning rate plot
        if let Some(lr_plot) = self.create_learning_rate_plot()? {
            let lr_path = self.config.output_dir.join("learning_rate.html");
            fs::write(&lr_path, lr_plot).map_err(|e| {
                NeuralError::IOError(format!("Failed to write learning rate plot: {}", e))
            })?;
            output_files.push(lr_path);
        }

        // Generate system metrics plot
        if let Some(system_plot) = self.create_system_metrics_plot()? {
            let system_path = self.config.output_dir.join("system_metrics.html");
            fs::write(&system_path, system_plot).map_err(|e| {
                NeuralError::IOError(format!("Failed to write system metrics plot: {}", e))
            })?;
            output_files.push(system_path);
        }

        Ok(output_files)
    }

    /// Get the current metrics history
    pub fn get_metrics_history(&self) -> &[TrainingMetrics<F>] {
        &self.metrics_history
    }

    /// Clear the metrics history
    pub fn clear_history(&mut self) {
        self.metrics_history.clear();
    }

    /// Add a custom plot configuration
    pub fn add_plot(&mut self, name: String, config: PlotConfig) {
        self.active_plots.insert(name, config);
    }

    /// Remove a plot configuration
    pub fn remove_plot(&mut self, name: &str) -> Option<PlotConfig> {
        self.active_plots.remove(name)
    }

    /// Update the visualization configuration
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config;
    }

    fn downsample_metrics(&mut self) {
        // TODO: Implement downsampling based on strategy
        match self.config.performance.downsampling_strategy {
            DownsamplingStrategy::Uniform => {
                // Keep every nth point
                let step = self.metrics_history.len() / self.config.performance.max_points_per_plot;
                if step > 1 {
                    let mut downsampled = Vec::new();
                    for (i, metric) in self.metrics_history.iter().enumerate() {
                        if i % step == 0 {
                            downsampled.push(metric.clone());
                        }
                    }
                    self.metrics_history = downsampled;
                }
            }
            _ => {
                // For now, just truncate to max size
                if self.metrics_history.len() > self.config.performance.max_points_per_plot {
                    let start =
                        self.metrics_history.len() - self.config.performance.max_points_per_plot;
                    self.metrics_history.drain(0..start);
                }
            }
        }
    }

    fn create_loss_plot(&self) -> Result<Option<String>> {
        if self.metrics_history.is_empty() {
            return Ok(None);
        }

        // TODO: Implement actual plotting library integration
        // For now, return a placeholder HTML
        let plot_html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Training Loss</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="lossPlot" style="width:100%;height:500px;"></div>
    <script>
        // TODO: Implement actual loss curve plotting
        var trace = {
            x: [1, 2, 3, 4],
            y: [0.8, 0.6, 0.4, 0.3],
            type: 'scatter',
            name: 'Training Loss'
        };
        
        var layout = {
            title: 'Training Loss Over Time',
            xaxis: { title: 'Epoch' },
            yaxis: { title: 'Loss' }
        };
        
        Plotly.newPlot('lossPlot', [trace], layout);
    </script>
</body>
</html>"#;

        Ok(Some(plot_html.to_string()))
    }

    fn create_accuracy_plot(&self) -> Result<Option<String>> {
        if self.metrics_history.is_empty() {
            return Ok(None);
        }

        // TODO: Implement accuracy plotting
        Ok(None)
    }

    fn create_learning_rate_plot(&self) -> Result<Option<String>> {
        if self.metrics_history.is_empty() {
            return Ok(None);
        }

        // TODO: Implement learning rate plotting
        Ok(None)
    }

    fn create_system_metrics_plot(&self) -> Result<Option<String>> {
        if self.metrics_history.is_empty() {
            return Ok(None);
        }

        // TODO: Implement system metrics plotting
        Ok(None)
    }
}

// Default implementations for configuration types

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: "Training Metrics".to_string(),
            x_axis: AxisConfig::default(),
            y_axis: AxisConfig::default(),
            series: Vec::new(),
            plot_type: PlotType::Line,
            update_mode: UpdateMode::Append,
        }
    }
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            label: "".to_string(),
            scale: AxisScale::Linear,
            range: None,
            show_grid: true,
            ticks: TickConfig::default(),
        }
    }
}

impl Default for TickConfig {
    fn default() -> Self {
        Self {
            interval: None,
            format: TickFormat::Auto,
            show_labels: true,
            rotation: 0.0,
        }
    }
}

impl Default for SeriesConfig {
    fn default() -> Self {
        Self {
            name: "Series".to_string(),
            data_source: "".to_string(),
            style: LineStyleConfig::default(),
            markers: MarkerConfig::default(),
            color: "#1f77b4".to_string(), // Default blue
            opacity: 1.0,
        }
    }
}

impl Default for LineStyleConfig {
    fn default() -> Self {
        Self {
            style: LineStyle::Solid,
            width: 2.0,
            smoothing: false,
            smoothing_window: 5,
        }
    }
}

impl Default for MarkerConfig {
    fn default() -> Self {
        Self {
            show: false,
            shape: MarkerShape::Circle,
            size: 6.0,
            fill_color: "#1f77b4".to_string(),
            border_color: "#1f77b4".to_string(),
        }
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            memory_usage_mb: 0.0,
            gpu_memory_mb: None,
            cpu_utilization: 0.0,
            gpu_utilization: None,
            step_duration_ms: 0.0,
            samples_per_second: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_visualizer_creation() {
        let config = VisualizationConfig::default();
        let visualizer = TrainingVisualizer::<f32>::new(config);

        assert!(visualizer.metrics_history.is_empty());
        assert!(visualizer.active_plots.is_empty());
    }

    #[test]
    fn test_add_metrics() {
        let config = VisualizationConfig::default();
        let mut visualizer = TrainingVisualizer::<f32>::new(config);

        let metrics = TrainingMetrics {
            epoch: 1,
            step: 100,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            losses: HashMap::from([("train_loss".to_string(), 0.5)]),
            accuracies: HashMap::from([("train_acc".to_string(), 0.8)]),
            learning_rate: 0.001,
            custom_metrics: HashMap::new(),
            system_metrics: SystemMetrics::default(),
        };

        visualizer.add_metrics(metrics);
        assert_eq!(visualizer.metrics_history.len(), 1);
    }

    #[test]
    fn test_plot_config_defaults() {
        let config = PlotConfig::default();
        assert_eq!(config.title, "Training Metrics");
        assert_eq!(config.plot_type, PlotType::Line);
        assert_eq!(config.update_mode, UpdateMode::Append);
    }

    #[test]
    fn test_axis_scale_variants() {
        assert_eq!(AxisScale::Linear, AxisScale::Linear);
        assert_eq!(AxisScale::Log, AxisScale::Log);
        assert_eq!(AxisScale::Sqrt, AxisScale::Sqrt);

        let custom = AxisScale::Custom("symlog".to_string());
        match custom {
            AxisScale::Custom(name) => assert_eq!(name, "symlog"),
            _ => panic!("Expected custom scale"),
        }
    }

    #[test]
    fn test_marker_shapes() {
        let shapes = vec![
            MarkerShape::Circle,
            MarkerShape::Square,
            MarkerShape::Triangle,
            MarkerShape::Diamond,
            MarkerShape::Cross,
            MarkerShape::Plus,
        ];

        assert_eq!(shapes.len(), 6);
        assert_eq!(shapes[0], MarkerShape::Circle);
    }

    #[test]
    fn test_plot_types() {
        let types = vec![
            PlotType::Line,
            PlotType::Scatter,
            PlotType::Bar,
            PlotType::Area,
            PlotType::Histogram,
            PlotType::Box,
            PlotType::Heatmap,
        ];

        assert_eq!(types.len(), 7);
        assert_eq!(types[0], PlotType::Line);
    }

    #[test]
    fn test_update_modes() {
        let append = UpdateMode::Append;
        let replace = UpdateMode::Replace;
        let rolling = UpdateMode::Rolling(100);

        assert_eq!(append, UpdateMode::Append);
        assert_eq!(replace, UpdateMode::Replace);

        match rolling {
            UpdateMode::Rolling(size) => assert_eq!(size, 100),
            _ => panic!("Expected rolling update mode"),
        }
    }

    #[test]
    fn test_clear_history() {
        let config = VisualizationConfig::default();
        let mut visualizer = TrainingVisualizer::<f32>::new(config);

        let metrics = TrainingMetrics {
            epoch: 1,
            step: 100,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            losses: HashMap::from([("train_loss".to_string(), 0.5)]),
            accuracies: HashMap::from([("train_acc".to_string(), 0.8)]),
            learning_rate: 0.001,
            custom_metrics: HashMap::new(),
            system_metrics: SystemMetrics::default(),
        };

        visualizer.add_metrics(metrics);
        assert_eq!(visualizer.metrics_history.len(), 1);

        visualizer.clear_history();
        assert!(visualizer.metrics_history.is_empty());
    }

    #[test]
    fn test_plot_management() {
        let config = VisualizationConfig::default();
        let mut visualizer = TrainingVisualizer::<f32>::new(config);

        let plot_config = PlotConfig::default();
        visualizer.add_plot("test_plot".to_string(), plot_config);

        assert!(visualizer.active_plots.contains_key("test_plot"));

        let removed = visualizer.remove_plot("test_plot");
        assert!(removed.is_some());
        assert!(!visualizer.active_plots.contains_key("test_plot"));
    }
}
