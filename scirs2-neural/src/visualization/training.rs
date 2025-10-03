//! Training metrics and curve visualization for neural networks
//!
//! This module provides comprehensive tools for visualizing training progress
//! including loss curves, accuracy metrics, learning rate schedules, and system performance.

use super::config::{DownsamplingStrategy, VisualizationConfig};
use crate::error::{NeuralError, Result};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
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
/// System performance metrics during training
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
/// Axis configuration
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
/// Tick configuration
pub struct TickConfig {
    /// Tick interval (None for auto)
    pub interval: Option<f64>,
    /// Tick format
    pub format: TickFormat,
    /// Show tick labels
    pub show_labels: bool,
    /// Tick rotation angle
    pub rotation: f32,
/// Tick format options
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
/// Data series configuration
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
/// Line style configuration for series
pub struct LineStyleConfig {
    pub style: LineStyle,
    /// Line width
    pub width: f32,
    /// Smoothing enabled
    pub smoothing: bool,
    /// Smoothing window size
    pub smoothing_window: usize,
/// Line style options (re-exported from network module)
pub enum LineStyle {
    /// Solid line
    Solid,
    /// Dashed line
    Dashed,
    /// Dotted line
    Dotted,
    /// Dash-dot line
    DashDot,
/// Marker configuration for data points
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
/// Marker shape options
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
/// Plot type options
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
/// Update mode for plots
pub enum UpdateMode {
    /// Append new data
    Append,
    /// Replace all data
    Replace,
    /// Rolling window
    Rolling(usize),
// Implementation for TrainingVisualizer
impl<F: Float + Debug + 'static + scirs2_core::numeric::FromPrimitive + Send + Sync> TrainingVisualizer<F> {
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
    /// Generate training curves visualization
    pub fn visualize_training_curves(&self) -> Result<Vec<PathBuf>> {
        let mut output_files = Vec::new();
        // Generate loss curves
        if let Some(loss_plot) = self.create_loss_plot()? {
            let loss_path = self.config.output_dir.join("training_loss.html");
            fs::write(&loss_path, loss_plot)
                .map_err(|e| NeuralError::IOError(format!("Failed to write loss plot: {}", e)))?;
            output_files.push(loss_path);
        // Generate accuracy curves
        if let Some(accuracy_plot) = self.create_accuracy_plot()? {
            let accuracy_path = self.config.output_dir.join("training_accuracy.html");
            fs::write(&accuracy_path, accuracy_plot).map_err(|e| {
                NeuralError::IOError(format!("Failed to write accuracy plot: {}", e))
            })?;
            output_files.push(accuracy_path);
        // Generate learning rate plot
        if let Some(lr_plot) = self.create_learning_rate_plot()? {
            let lr_path = self.config.output_dir.join("learning_rate.html");
            fs::write(&lr_path, lr_plot).map_err(|e| {
                NeuralError::IOError(format!("Failed to write learning rate plot: {}", e))
            output_files.push(lr_path);
        // Generate system metrics plot
        if let Some(system_plot) = self.create_system_metrics_plot()? {
            let system_path = self.config.output_dir.join("system_metrics.html");
            fs::write(&system_path, system_plot).map_err(|e| {
                NeuralError::IOError(format!("Failed to write system metrics plot: {}", e))
            output_files.push(system_path);
        Ok(output_files)
    /// Get the current metrics history
    pub fn get_metrics_history(&self) -> &[TrainingMetrics<F>] {
        &self.metrics_history
    /// Clear the metrics history
    pub fn clear_history(&mut self) {
        self.metrics_history.clear();
    /// Add a custom plot configuration
    pub fn add_plot(&mut self, name: String, config: PlotConfig) {
        self.active_plots.insert(name, config);
    /// Remove a plot configuration
    pub fn remove_plot(&mut self, name: &str) -> Option<PlotConfig> {
        self.active_plots.remove(name)
    /// Update the visualization configuration
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config;
    fn downsample_metrics(&mut self) {
        // Implement downsampling based on strategy
        if self.metrics_history.len() <= self.config.performance.max_points_per_plot {
            return; // No downsampling needed
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
            DownsamplingStrategy::LTTB => {
                // Largest Triangle Three Bucket algorithm - simplified implementation
                self.downsample_lttb();
            DownsamplingStrategy::MinMax => {
                // Min-max decimation - keep local minima and maxima
                self.downsample_minmax();
            DownsamplingStrategy::Statistical => {
                // Statistical sampling - sample based on variance/importance
                self.downsample_statistical();
    /// Largest Triangle Three Bucket (LTTB) downsampling algorithm
    fn downsample_lttb(&mut self) {
        let target_points = self.config.performance.max_points_per_plot;
        if self.metrics_history.len() <= target_points {
            return;
        let bucket_size = self.metrics_history.len() as f64 / target_points as f64;
        let mut downsampled = Vec::new();
        // Always keep first point
        downsampled.push(self.metrics_history[0].clone());
        // For each bucket, select the point that forms the largest triangle
        for bucket in 1..(target_points - 1) {
            let bucket_start = (bucket as f64 * bucket_size) as usize;
            let bucket_end =
                ((bucket + 1) as f64 * bucket_size).min(self.metrics_history.len() as f64) as usize;
            // Calculate average point of next bucket
            let next_bucket_start = bucket_end;
            let next_bucket_end =
                ((bucket + 2) as f64 * bucket_size).min(self.metrics_history.len() as f64) as usize;
            let avg_epoch = if next_bucket_end > next_bucket_start {
                let sum: usize = (next_bucket_start..next_bucket_end)
                    .map(|i| self.metrics_history[i].epoch)
                    .sum();
                sum as f64 / (next_bucket_end - next_bucket_start) as f64
            } else {
                self.metrics_history[self.metrics_history.len() - 1].epoch as f64
            };
            // Find point in current bucket that maximizes triangle area
            let mut max_area = 0.0f64;
            let mut selected_idx = bucket_start;
            let prev_epoch = downsampled.last().unwrap().epoch as f64;
            for i in bucket_start..bucket_end {
                let curr_epoch = self.metrics_history[i].epoch as f64;
                // Calculate triangle area (simplified - using epoch as primary metric)
                let area = ((prev_epoch - avg_epoch) * (curr_epoch - prev_epoch)).abs();
                if area > max_area {
                    max_area = area;
                    selected_idx = i;
            downsampled.push(self.metrics_history[selected_idx].clone());
        // Always keep last point
        downsampled.push(self.metrics_history[self.metrics_history.len() - 1].clone());
        self.metrics_history = downsampled;
    /// Min-max decimation downsampling
    fn downsample_minmax(&mut self) {
        let bucket_size = self.metrics_history.len() / (target_points / 2); // Divide by 2 because we keep min and max
        for chunk in self.metrics_history.chunks(bucket_size) {
            if chunk.is_empty() {
                continue;
            // Find min and max based on a primary loss metric
            let mut min_metric = &chunk[0];
            let mut max_metric = &chunk[0];
            for metric in chunk {
                // Use first loss value as comparison metric, or epoch if no losses
                let current_value = metric
                    .losses
                    .values()
                    .next()
                    .map(|v| v.to_f64().unwrap_or(0.0))
                    .unwrap_or(metric.epoch as f64);
                let min_value = min_metric
                    .unwrap_or(min_metric.epoch as f64);
                let max_value = max_metric
                    .unwrap_or(max_metric.epoch as f64);
                if current_value < min_value {
                    min_metric = metric;
                if current_value > max_value {
                    max_metric = metric;
            // Add min and max (avoid duplicates)
            if min_metric.epoch <= max_metric.epoch {
                downsampled.push(min_metric.clone());
                if min_metric.epoch != max_metric.epoch {
                    downsampled.push(max_metric.clone());
                downsampled.push(max_metric.clone());
        // Sort by epoch to maintain temporal order
        downsampled.sort_by_key(|m| m.epoch);
        // If still too many points, apply uniform sampling
        if downsampled.len() > target_points {
            let step = downsampled.len() / target_points;
            let mut final_downsampled = Vec::new();
            for (i, metric) in downsampled.iter().enumerate() {
                if i % step == 0 {
                    final_downsampled.push(metric.clone());
            self.metrics_history = final_downsampled;
        } else {
            self.metrics_history = downsampled;
    /// Statistical downsampling based on variance and importance
    fn downsample_statistical(&mut self) {
        // Calculate importance scores for each point
        let mut importance_scores: Vec<(usize, f64)> = Vec::new();
        for (i, metric) in self.metrics_history.iter().enumerate() {
            let mut score = 0.0f64;
            // Base importance: changes in loss values
            if i > 0 && i < self.metrics_history.len() - 1 {
                let prev_metric = &self.metrics_history[i - 1];
                let next_metric = &self.metrics_history[i + 1];
                // Calculate variance in loss values
                for (loss_name, &loss_value) in &metric.losses {
                    if let (Some(&prev_loss), Some(&next_loss)) = (
                        prev_metric.losses.get(loss_name),
                        next_metric.losses.get(loss_name),
                    ) {
                        let prev_val = prev_loss.to_f64().unwrap_or(0.0);
                        let curr_val = loss_value.to_f64().unwrap_or(0.0);
                        let next_val = next_loss.to_f64().unwrap_or(0.0);
                        // Second derivative (curvature) as importance measure
                        let curvature = ((next_val - curr_val) - (curr_val - prev_val)).abs();
                        score += curvature;
            // Always keep first and last points
            if i == 0 || i == self.metrics_history.len() - 1 {
                score += 1000.0; // High importance
            importance_scores.push((i, score));
        // Sort by importance score (descending)
        importance_scores
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        // Select top points and sort by original index to maintain temporal order
        let mut selected_indices: Vec<usize> = importance_scores
            .iter()
            .take(target_points)
            .map(|(idx_)| *idx)
            .collect();
        selected_indices.sort();
        for &idx in &selected_indices {
            downsampled.push(self.metrics_history[idx].clone());
    fn create_loss_plot(&self) -> Result<Option<String>> {
        if self.metrics_history.is_empty() {
            return Ok(None);
        // Extract loss data from metrics history
        let mut loss_data = std::collections::HashMap::new();
        let mut epochs = Vec::new();
        for metric in &self.metrics_history {
            epochs.push(metric.epoch);
            for (loss_name, loss_value) in &metric.losses {
                loss_data
                    .entry(loss_name.clone())
                    .or_insert_with(Vec::new)
                    .push(loss_value.to_f64().unwrap_or(0.0));
        if loss_data.is_empty() {
        // Generate HTML with Plotly.js
        let mut traces = Vec::new();
        let colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        ];
        for (i, (loss_name, values)) in loss_data.iter().enumerate() {
            let color = colors[i % colors.len()];
            let epochs_json = serde_json::to_string(&epochs).unwrap_or_default();
            let values_json = serde_json::to_string(values).unwrap_or_default();
            traces.push(format!(
                r#"{{
                    x: {},
                    y: {},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '{}',
                    line: {{ color: '{}', width: 2 }},
                    marker: {{ size: 6, color: '{}' }}
                }}"#,
                epochs_json, values_json, loss_name, color, color
            ));
        let traces_str = traces.join(",\n            ");
        let plot_html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Training Loss</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ width: 100%; height: 600px; }}
    </style>
</head>
<body>
    <h2>Training Loss Curves</h2>
    <div id="lossPlot" class="plot-container"></div>
    <script>
        var traces = [
            {}
        
        var layout = {{
            title: {{
                text: 'Training Loss Over Time',
                font: {{ size: 18 }}
            }},
            xaxis: {{ 
                title: 'Epoch',
                showgrid: true,
                gridcolor: '#e0e0e0'
            yaxis: {{ 
                title: 'Loss',
            hovermode: 'x unified',
            legend: {{
                x: 1,
                y: 1,
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#000',
                borderwidth: 1
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        }};
        var config = {{
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        Plotly.newPlot('lossPlot', traces, layout, config);
    </script>
</body>
</html>"#,
            traces_str
        );
        Ok(Some(plot_html))
    fn create_accuracy_plot(&self) -> Result<Option<String>> {
        // Extract accuracy data from metrics history
        let mut accuracy_data = std::collections::HashMap::new();
            for (acc_name, acc_value) in &metric.accuracies {
                accuracy_data
                    .entry(acc_name.clone())
                    .push(acc_value.to_f64().unwrap_or(0.0));
        if accuracy_data.is_empty() {
            "#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd", "#8c564b",
        for (i, (acc_name, values)) in accuracy_data.iter().enumerate() {
                epochs_json, values_json, acc_name, color, color
    <title>Training Accuracy</title>
    <h2>Training Accuracy Curves</h2>
    <div id="accuracyPlot" class="plot-container"></div>
                text: 'Training Accuracy Over Time',
                title: 'Accuracy',
                gridcolor: '#e0e0e0',
                range: [0, 1]
                y: 0,
        Plotly.newPlot('accuracyPlot', traces, layout, config);
    fn create_learning_rate_plot(&self) -> Result<Option<String>> {
        // Extract learning rate data from metrics history
        let mut learning_rates = Vec::new();
            learning_rates.push(metric.learning_rate.to_f64().unwrap_or(0.0));
        if learning_rates.is_empty() {
        let epochs_json = serde_json::to_string(&epochs).unwrap_or_default();
        let lr_json = serde_json::to_string(&learning_rates).unwrap_or_default();
    <title>Learning Rate Schedule</title>
    <h2>Learning Rate Schedule</h2>
    <div id="lrPlot" class="plot-container"></div>
        var trace = {{
            x: {},
            y: {},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Learning Rate',
            line: {{ color: '#d62728', width: 3 }},
            marker: {{ size: 8, color: '#d62728' }}
                text: 'Learning Rate Over Time',
                title: 'Learning Rate',
                type: 'log',
        Plotly.newPlot('lrPlot', [trace], layout, config);
            epochs_json, lr_json
    fn create_system_metrics_plot(&self) -> Result<Option<String>> {
        // Extract system metrics from history
        let mut memory_usage = Vec::new();
        let mut cpu_utilization = Vec::new();
        let mut gpu_utilization = Vec::new();
        let mut samples_per_second = Vec::new();
            memory_usage.push(metric.system_metrics.memory_usage_mb);
            cpu_utilization.push(metric.system_metrics.cpu_utilization);
            if let Some(gpu_util) = metric.system_metrics.gpu_utilization {
                gpu_utilization.push(gpu_util);
            samples_per_second.push(metric.system_metrics.samples_per_second);
        let memory_json = serde_json::to_string(&memory_usage).unwrap_or_default();
        let cpu_json = serde_json::to_string(&cpu_utilization).unwrap_or_default();
        let gpu_json = if !gpu_utilization.is_empty() {
            serde_json::to_string(&gpu_utilization).unwrap_or_default()
            "[]".to_string()
        };
        let sps_json = serde_json::to_string(&samples_per_second).unwrap_or_default();
    <title>System Metrics</title>
        .plot-container {{ width: 100%; height: 400px; margin-bottom: 20px; }}
    <h2>System Performance Metrics</h2>
    
    <h3>Memory Usage</h3>
    <div id="memoryPlot" class="plot-container"></div>
    <h3>CPU & GPU Utilization</h3>
    <div id="utilizationPlot" class="plot-container"></div>
    <h3>Training Throughput</h3>
    <div id="throughputPlot" class="plot-container"></div>
        // Memory usage plot
        var memoryTrace = {{
            name: 'Memory Usage (MB)',
            line: {{ color: '#ff7f0e', width: 2 }},
            marker: {{ size: 6, color: '#ff7f0e' }}
        var memoryLayout = {{
            title: 'Memory Usage Over Time',
            xaxis: {{ title: 'Epoch' }},
            yaxis: {{ title: 'Memory (MB)' }},
            showlegend: false
        Plotly.newPlot('memoryPlot', [memoryTrace], memoryLayout);
        // CPU and GPU utilization plot
        var traces = [{{
            name: 'CPU Utilization (%)',
            line: {{ color: '#1f77b4', width: 2 }},
            marker: {{ size: 6, color: '#1f77b4' }}
        }}];
        if ({}.length > 0) {{
            traces.push({{
                x: {},
                y: {},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'GPU Utilization (%)',
                line: {{ color: '#2ca02c', width: 2 }},
                marker: {{ size: 6, color: '#2ca02c' }}
            }});
        }}
        var utilizationLayout = {{
            title: 'CPU & GPU Utilization',
            yaxis: {{ title: 'Utilization (%)', range: [0, 100] }}
        Plotly.newPlot('utilizationPlot', traces, utilizationLayout);
        // Throughput plot
        var throughputTrace = {{
            name: 'Samples/Second',
            line: {{ color: '#9467bd', width: 2 }},
            marker: {{ size: 6, color: '#9467bd' }}
        var throughputLayout = {{
            title: 'Training Throughput',
            yaxis: {{ title: 'Samples per Second' }},
        Plotly.newPlot('throughputPlot', [throughputTrace], throughputLayout);
            epochs_json,
            memory_json,
            cpu_json,
            gpu_json,
            sps_json
// Default implementations for configuration types
impl Default for PlotConfig {
    fn default() -> Self {
            title: "Training Metrics".to_string(),
            x_axis: AxisConfig::default(),
            y_axis: AxisConfig::default(),
            series: Vec::new(),
            plot_type: PlotType::Line,
            update_mode: UpdateMode::Append,
impl Default for AxisConfig {
            label: "".to_string(),
            scale: AxisScale::Linear,
            range: None,
            show_grid: true,
            ticks: TickConfig::default(),
impl Default for TickConfig {
            interval: None,
            format: TickFormat::Auto,
            show_labels: true,
            rotation: 0.0,
impl Default for SeriesConfig {
            name: "Series".to_string(),
            data_source: "".to_string(),
            style: LineStyleConfig::default(),
            markers: MarkerConfig::default(),
            color: "#1f77b4".to_string(), // Default blue
            opacity: 1.0,
impl Default for LineStyleConfig {
            style: LineStyle::Solid,
            width: 2.0,
            smoothing: false,
            smoothing_window: 5,
impl Default for MarkerConfig {
            show: false,
            shape: MarkerShape::Circle,
            size: 6.0,
            fill_color: "#1f77b4".to_string(),
            border_color: "#1f77b4".to_string(),
impl Default for SystemMetrics {
            memory_usage_mb: 0.0,
            gpu_memory_mb: None,
            cpu_utilization: 0.0,
            gpu_utilization: None,
            step_duration_ms: 0.0,
            samples_per_second: 0.0,
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_training_visualizer_creation() {
        let config = VisualizationConfig::default();
        let visualizer = TrainingVisualizer::<f32>::new(config);
        assert!(visualizer.metrics_history.is_empty());
        assert!(visualizer.active_plots.is_empty());
    fn test_add_metrics() {
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
        visualizer.add_metrics(metrics);
        assert_eq!(visualizer.metrics_history.len(), 1);
    fn test_plot_config_defaults() {
        let config = PlotConfig::default();
        assert_eq!(config.title, "Training Metrics");
        assert_eq!(config.plot_type, PlotType::Line);
        assert_eq!(config.update_mode, UpdateMode::Append);
    fn test_axis_scale_variants() {
        assert_eq!(AxisScale::Linear, AxisScale::Linear);
        assert_eq!(AxisScale::Log, AxisScale::Log);
        assert_eq!(AxisScale::Sqrt, AxisScale::Sqrt);
        let custom = AxisScale::Custom("symlog".to_string());
        match custom {
            AxisScale::Custom(name) => assert_eq!(name, "symlog", _ => assert!(false, "Expected custom scale"),
    fn test_markershapes() {
        let shapes = [
            MarkerShape::Circle,
            MarkerShape::Square,
            MarkerShape::Triangle,
            MarkerShape::Diamond,
            MarkerShape::Cross,
            MarkerShape::Plus,
        assert_eq!(shapes.len(), 6);
        assert_eq!(shapes[0], MarkerShape::Circle);
    fn test_plot_types() {
        let types = [
            PlotType::Line,
            PlotType::Scatter,
            PlotType::Bar,
            PlotType::Area,
            PlotType::Histogram,
            PlotType::Box,
            PlotType::Heatmap,
        assert_eq!(types.len(), 7);
        assert_eq!(types[0], PlotType::Line);
    fn test_update_modes() {
        let append = UpdateMode::Append;
        let replace = UpdateMode::Replace;
        let rolling = UpdateMode::Rolling(100);
        assert_eq!(append, UpdateMode::Append);
        assert_eq!(replace, UpdateMode::Replace);
        match rolling {
            UpdateMode::Rolling(size) => assert_eq!(size, 100, _ => assert!(false, "Expected rolling update mode"),
    fn test_clear_history() {
        visualizer.clear_history();
    fn test_plot_management() {
        let plot_config = PlotConfig::default();
        visualizer.add_plot("test_plot".to_string(), plot_config);
        assert!(visualizer.active_plots.contains_key("test_plot"));
        let removed = visualizer.remove_plot("test_plot");
        assert!(removed.is_some());
        assert!(!visualizer.active_plots.contains_key("test_plot"));
