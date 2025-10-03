//! Comprehensive time series visualization module
//!
//! This module provides advanced visualization capabilities for time series data,
//! including interactive plotting, forecasting visualization with uncertainty bands,
//! and decomposition result visualization.
//!
//! # Features
//!
//! - Interactive time series plotting with zoom and pan
//! - Forecasting visualization with confidence intervals
//! - Decomposition result visualization (trend, seasonal, residual components)
//! - Multi-series plotting and comparison
//! - Seasonal pattern visualization
//! - Anomaly and change point highlighting
//! - Dashboard generation utilities
//! - Export capabilities (PNG, SVG, HTML)
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_series::visualization::{TimeSeriesPlot, PlotStyle, ExportFormat};
//! use scirs2_core::ndarray::Array1;
//!
//! let data = Array1::linspace(0.0, 10.0, 100);
//! let ts_data = data.mapv(|x| (x * 2.0 * std::f64::consts::PI).sin());
//!
//! let mut plot = TimeSeriesPlot::new("Sample Time Series");
//! plot.add_series("sine_wave", &data, &ts_data, PlotStyle::default());
//! plot.show();
//! ```

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for plot styling and appearance
#[derive(Debug, Clone)]
pub struct PlotStyle {
    /// Line color (RGB hex format, e.g., "#FF0000" for red)
    pub color: String,
    /// Line width in pixels
    pub line_width: f64,
    /// Line style (solid, dashed, dotted)
    pub line_style: LineStyle,
    /// Marker style for data points
    pub marker: MarkerStyle,
    /// Opacity (0.0 to 1.0)
    pub opacity: f64,
    /// Fill area under curve
    pub fill: bool,
    /// Fill color (if different from line color)
    pub fill_color: Option<String>,
}

impl Default for PlotStyle {
    fn default() -> Self {
        Self {
            color: "#1f77b4".to_string(), // Default blue
            line_width: 2.0,
            line_style: LineStyle::Solid,
            marker: MarkerStyle::None,
            opacity: 1.0,
            fill: false,
            fill_color: None,
        }
    }
}

/// Line style options
#[derive(Debug, Clone, Copy)]
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

/// Marker style options
#[derive(Debug, Clone, Copy)]
pub enum MarkerStyle {
    /// No marker
    None,
    /// Circle marker
    Circle,
    /// Square marker
    Square,
    /// Triangle marker
    Triangle,
    /// Cross marker
    Cross,
    /// Plus marker
    Plus,
}

/// Export format options
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    /// PNG image format
    PNG,
    /// SVG vector format
    SVG,
    /// HTML format
    HTML,
    /// PDF document format
    PDF,
}

/// Time series data point for plotting
#[derive(Debug, Clone)]
pub struct TimePoint {
    /// Time value (can be timestamp, index, etc.)
    pub time: f64,
    /// Data value
    pub value: f64,
    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// A single time series for plotting
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Series name
    pub name: String,
    /// Time points
    pub data: Vec<TimePoint>,
    /// Plot style
    pub style: PlotStyle,
    /// Series type
    pub series_type: SeriesType,
}

/// Type of time series data
#[derive(Debug, Clone, Copy)]
pub enum SeriesType {
    /// Regular time series data
    Line,
    /// Scatter plot points
    Scatter,
    /// Bar chart
    Bar,
    /// Filled area
    Area,
    /// Candlestick (OHLC data)
    Candlestick,
    /// Error bars with confidence intervals
    ErrorBars,
}

/// Main time series plotting structure
#[derive(Debug)]
pub struct TimeSeriesPlot {
    /// Plot title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Time series data
    series: Vec<TimeSeries>,
    /// Plot configuration
    config: PlotConfig,
    /// Annotations (text, arrows, shapes)
    annotations: Vec<Annotation>,
}

/// Plot configuration
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Plot width in pixels
    pub width: u32,
    /// Plot height in pixels
    pub height: u32,
    /// Show grid
    pub show_grid: bool,
    /// Show legend
    pub show_legend: bool,
    /// Legend position
    pub legend_position: LegendPosition,
    /// Enable interactivity (zoom, pan)
    pub interactive: bool,
    /// Background color
    pub background_color: String,
    /// Grid color
    pub grid_color: String,
    /// Axis color
    pub axis_color: String,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            show_grid: true,
            show_legend: true,
            legend_position: LegendPosition::TopRight,
            interactive: true,
            background_color: "#FFFFFF".to_string(),
            grid_color: "#E0E0E0".to_string(),
            axis_color: "#000000".to_string(),
        }
    }
}

/// Legend position options
#[derive(Debug, Clone, Copy)]
pub enum LegendPosition {
    /// Top left position
    TopLeft,
    /// Top right position
    TopRight,
    /// Bottom left position
    BottomLeft,
    /// Bottom right position
    BottomRight,
    /// Outside the plot area
    Outside,
}

/// Annotation for plots (text, arrows, shapes)
#[derive(Debug, Clone)]
pub struct Annotation {
    /// Annotation type
    pub annotation_type: AnnotationType,
    /// X position
    pub x: f64,
    /// Y position  
    pub y: f64,
    /// Text content (for text annotations)
    pub text: Option<String>,
    /// Style
    pub style: AnnotationStyle,
}

/// Types of annotations
#[derive(Debug, Clone)]
pub enum AnnotationType {
    /// Text annotation
    Text,
    /// Arrow annotation pointing to a target
    Arrow {
        /// X coordinate of arrow target
        target_x: f64,
        /// Y coordinate of arrow target
        target_y: f64,
    },
    /// Rectangle annotation with specified dimensions
    Rectangle {
        /// Width of rectangle
        width: f64,
        /// Height of rectangle
        height: f64,
    },
    /// Circle annotation with specified radius
    Circle {
        /// Radius of circle
        radius: f64,
    },
    /// Vertical line annotation
    VerticalLine,
    /// Horizontal line annotation
    HorizontalLine,
}

/// Annotation styling
#[derive(Debug, Clone)]
pub struct AnnotationStyle {
    /// Color of the annotation
    pub color: String,
    /// Font size for text annotations
    pub font_size: f64,
    /// Opacity/transparency level
    pub opacity: f64,
}

impl Default for AnnotationStyle {
    fn default() -> Self {
        Self {
            color: "#000000".to_string(),
            font_size: 12.0,
            opacity: 1.0,
        }
    }
}

impl TimeSeriesPlot {
    /// Create a new time series plot
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            x_label: "Time".to_string(),
            y_label: "Value".to_string(),
            series: Vec::new(),
            config: PlotConfig::default(),
            annotations: Vec::new(),
        }
    }

    /// Set axis labels
    pub fn set_labels(&mut self, x_label: &str, ylabel: &str) {
        self.x_label = x_label.to_string();
        self.y_label = ylabel.to_string();
    }

    /// Add a time series to the plot
    pub fn add_series(
        &mut self,
        name: &str,
        time: &Array1<f64>,
        values: &Array1<f64>,
        style: PlotStyle,
    ) -> Result<()> {
        if time.len() != values.len() {
            return Err(TimeSeriesError::InvalidInput(
                "Time and value arrays must have the same length".to_string(),
            ));
        }

        let data: Vec<TimePoint> = time
            .iter()
            .zip(values.iter())
            .map(|(&t, &v)| TimePoint {
                time: t,
                value: v,
                metadata: None,
            })
            .collect();

        let series = TimeSeries {
            name: name.to_string(),
            data,
            style,
            series_type: SeriesType::Line,
        };

        self.series.push(series);
        Ok(())
    }

    /// Add a series with error bars (confidence intervals)
    pub fn add_series_with_confidence(
        &mut self,
        name: &str,
        time: &Array1<f64>,
        values: &Array1<f64>,
        lower: &Array1<f64>,
        upper: &Array1<f64>,
        style: PlotStyle,
    ) -> Result<()> {
        if time.len() != values.len() || values.len() != lower.len() || lower.len() != upper.len() {
            return Err(TimeSeriesError::InvalidInput(
                "All arrays must have the same length".to_string(),
            ));
        }

        // Add main series
        self.add_series(name, time, values, style.clone())?;

        // Add confidence band as filled area
        let mut confidence_style = style.clone();
        confidence_style.fill = true;
        confidence_style.opacity = 0.3;
        confidence_style.line_style = LineStyle::Solid;

        // Create upper bound series
        let upper_data: Vec<TimePoint> = time
            .iter()
            .zip(upper.iter())
            .map(|(&t, &v)| TimePoint {
                time: t,
                value: v,
                metadata: None,
            })
            .collect();

        // Create lower bound series (reversed for proper filling)
        let lower_data: Vec<TimePoint> = time
            .iter()
            .zip(lower.iter())
            .rev()
            .map(|(&t, &v)| TimePoint {
                time: t,
                value: v,
                metadata: None,
            })
            .collect();

        // Combine for filled area
        let mut confidence_data = upper_data;
        confidence_data.extend(lower_data);

        let confidence_series = TimeSeries {
            name: format!("{name}_confidence"),
            data: confidence_data,
            style: confidence_style,
            series_type: SeriesType::Area,
        };

        self.series.push(confidence_series);
        Ok(())
    }

    /// Add annotation to the plot
    pub fn add_annotation(&mut self, annotation: Annotation) {
        self.annotations.push(annotation);
    }

    /// Highlight anomalies on the plot
    pub fn highlight_anomalies(
        &mut self,
        time: &Array1<f64>,
        anomaly_indices: &[usize],
    ) -> Result<()> {
        for &idx in anomaly_indices {
            if idx < time.len() {
                let annotation = Annotation {
                    annotation_type: AnnotationType::Circle { radius: 5.0 },
                    x: time[idx],
                    y: 0.0, // Will be adjusted to data value
                    text: Some("Anomaly".to_string()),
                    style: AnnotationStyle {
                        color: "#FF0000".to_string(), // Red
                        font_size: 10.0,
                        opacity: 0.7,
                    },
                };
                self.add_annotation(annotation);
            }
        }
        Ok(())
    }

    /// Highlight change points on the plot
    pub fn highlight_change_points(&mut self, changepoints: &[f64]) {
        for &cp in changepoints {
            let annotation = Annotation {
                annotation_type: AnnotationType::VerticalLine,
                x: cp,
                y: 0.0,
                text: Some("Change Point".to_string()),
                style: AnnotationStyle {
                    color: "#FFA500".to_string(), // Orange
                    font_size: 10.0,
                    opacity: 0.8,
                },
            };
            self.add_annotation(annotation);
        }
    }

    /// Configure plot appearance
    pub fn configure(&mut self, config: PlotConfig) {
        self.config = config;
    }

    /// Generate HTML output for the plot
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        // HTML header with CSS and JavaScript
        html.push_str(&format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ width: {}px; height: {}px; margin: auto; }}
        .title {{ text-align: center; font-size: 18px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="title">{}</div>
    <div id="plot" class="plot-container"></div>
    <script>
"#,
            self.title, self.config.width, self.config.height, self.title
        ));

        // Generate Plotly data
        html.push_str("var data = [\n");

        for (i, series) in self.series.iter().enumerate() {
            if i > 0 {
                html.push_str(",\n");
            }

            let x_values: Vec<f64> = series.data.iter().map(|p| p.time).collect();
            let y_values: Vec<f64> = series.data.iter().map(|p| p.value).collect();

            html.push_str(&format!(
                r#"
    {{
        x: {:?},
        y: {:?},
        type: '{}',
        mode: '{}',
        name: '{}',
        line: {{ color: '{}', width: {} }},
        opacity: {}
    }}"#,
                x_values,
                y_values,
                match series.series_type {
                    SeriesType::Line => "scatter",
                    SeriesType::Scatter => "scatter",
                    SeriesType::Bar => "bar",
                    SeriesType::Area => "scatter",
                    _ => "scatter",
                },
                match series.series_type {
                    SeriesType::Line => "lines",
                    SeriesType::Scatter => "markers",
                    SeriesType::Area => "lines",
                    _ => "lines+markers",
                },
                series.name,
                series.style.color,
                series.style.line_width,
                series.style.opacity
            ));
        }

        html.push_str("\n];\n");

        // Plot layout
        html.push_str(&format!(
            r#"
var layout = {{
    title: '{}',
    xaxis: {{ title: '{}' }},
    yaxis: {{ title: '{}' }},
    showlegend: {},
    plot_bgcolor: '{}',
    paper_bgcolor: '{}',
    font: {{ size: 12 }}
}};

var config = {{
    responsive: true,
    displayModeBar: {}
}};

Plotly.newPlot('plot', data, layout, config);
"#,
            self.title,
            self.x_label,
            self.y_label,
            self.config.show_legend,
            self.config.background_color,
            self.config.background_color,
            self.config.interactive
        ));

        html.push_str(
            r#"
    </script>
</body>
</html>
"#,
        );

        html
    }

    /// Save plot to file
    pub fn save<P: AsRef<Path>>(&self, path: P, format: ExportFormat) -> Result<()> {
        let path = path.as_ref();

        match format {
            ExportFormat::HTML => {
                let html_content = self.to_html();
                std::fs::write(path, html_content).map_err(|e| {
                    TimeSeriesError::IOError(format!("Failed to save HTML plot: {e}"))
                })?;
            }
            ExportFormat::SVG => {
                // Generate SVG output
                let svg_content = self.to_svg();
                std::fs::write(path, svg_content).map_err(|e| {
                    TimeSeriesError::IOError(format!("Failed to save SVG plot: {e}"))
                })?;
            }
            _ => {
                return Err(TimeSeriesError::NotImplemented(format!(
                    "Export format {format:?} not yet implemented"
                )));
            }
        }

        Ok(())
    }

    /// Generate SVG output for the plot
    fn to_svg(&self) -> String {
        let mut svg = String::new();

        svg.push_str(&format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="{}"/>
    <text x="{}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" font-weight="bold">{}</text>
"#, 
            self.config.width, self.config.height,
            self.config.background_color,
            self.config.width / 2,
            self.title
        ));

        // Plot area dimensions
        let margin = 60;
        let plot_width = self.config.width as i32 - 2 * margin;
        let plot_height = self.config.height as i32 - 2 * margin;

        // Find data ranges
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for series in &self.series {
            for point in &series.data {
                min_x = min_x.min(point.time);
                max_x = max_x.max(point.time);
                min_y = min_y.min(point.value);
                max_y = max_y.max(point.value);
            }
        }

        // Add some padding
        let x_range = max_x - min_x;
        let y_range = max_y - min_y;
        min_x -= x_range * 0.05;
        max_x += x_range * 0.05;
        min_y -= y_range * 0.05;
        max_y += y_range * 0.05;

        // Draw axes
        svg.push_str(&format!(
            r#"
    <!-- Axes -->
    <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
    <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
"#,
            margin,
            self.config.height as i32 - margin, // Y axis
            margin + plot_width,
            self.config.height as i32 - margin,
            self.config.axis_color,
            margin,
            margin, // X axis
            margin,
            self.config.height as i32 - margin,
            self.config.axis_color
        ));

        // Draw series
        for series in &self.series {
            if series.data.is_empty() {
                continue;
            }

            let mut path_data = String::from("M");

            for (i, point) in series.data.iter().enumerate() {
                let x = margin as f64 + (point.time - min_x) / (max_x - min_x) * plot_width as f64;
                let y = (self.config.height as f64 - margin as f64)
                    - (point.value - min_y) / (max_y - min_y) * plot_height as f64;

                if i == 0 {
                    path_data.push_str(&format!(" {x:.2} {y:.2}"));
                } else {
                    path_data.push_str(&format!(" L {x:.2} {y:.2}"));
                }
            }

            svg.push_str(&format!(
                r#"
    <path d="{}" fill="none" stroke="{}" stroke-width="{}" opacity="{}"/>
"#,
                path_data, series.style.color, series.style.line_width, series.style.opacity
            ));
        }

        // Add axis labels
        svg.push_str(&format!(r#"
    <text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{}</text>
    <text x="20" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" transform="rotate(-90 20 {})">{}</text>
"#,
            margin + plot_width / 2,
            self.config.height as i32 - 10,
            self.x_label,
            margin + plot_height / 2,
            margin + plot_height / 2,
            self.y_label
        ));

        svg.push_str("</svg>");
        svg
    }

    /// Display plot (opens in default browser)
    pub fn show(&self) -> Result<()> {
        let temp_path = std::env::temp_dir().join("scirs2_plot.html");
        self.save(&temp_path, ExportFormat::HTML)?;

        // Try to open in browser
        #[cfg(target_os = "windows")]
        std::process::Command::new("cmd")
            .args(["/c", "start", "", &temp_path.to_string_lossy()])
            .spawn()
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open plot: {e}")))?;

        #[cfg(target_os = "macos")]
        std::process::Command::new("open")
            .arg(&temp_path)
            .spawn()
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open plot: {e}")))?;

        #[cfg(target_os = "linux")]
        std::process::Command::new("xdg-open")
            .arg(&temp_path)
            .spawn()
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open plot: {e}")))?;

        Ok(())
    }
}

/// Specialized plotting functions for time series analysis results
pub struct SpecializedPlots;

impl SpecializedPlots {
    /// Plot decomposition results (trend, seasonal, residual)
    pub fn plot_decomposition(
        time: &Array1<f64>,
        original: &Array1<f64>,
        trend: &Array1<f64>,
        seasonal: &Array1<f64>,
        residual: &Array1<f64>,
        title: &str,
    ) -> Result<TimeSeriesPlot> {
        let mut plot = TimeSeriesPlot::new(title);
        plot.set_labels("Time", "Value");

        // Original series
        let original_style = PlotStyle {
            color: "#1f77b4".to_string(), // Blue
            ..Default::default()
        };
        plot.add_series("Original", time, original, original_style)?;

        // Trend component
        let trend_style = PlotStyle {
            color: "#ff7f0e".to_string(), // Orange
            line_width: 3.0,
            ..Default::default()
        };
        plot.add_series("Trend", time, trend, trend_style)?;

        // Seasonal component
        let seasonal_style = PlotStyle {
            color: "#2ca02c".to_string(),
            ..Default::default()
        };
        plot.add_series("Seasonal", time, seasonal, seasonal_style)?;

        // Residual component
        let residual_style = PlotStyle {
            color: "#d62728".to_string(), // Red
            opacity: 0.7,
            ..Default::default()
        };
        plot.add_series("Residual", time, residual, residual_style)?;

        Ok(plot)
    }

    /// Plot forecasting results with confidence intervals
    pub fn plot_forecast(
        historical_time: &Array1<f64>,
        historical_data: &Array1<f64>,
        forecast_time: &Array1<f64>,
        forecast_values: &Array1<f64>,
        confidence_lower: &Array1<f64>,
        confidence_upper: &Array1<f64>,
        title: &str,
    ) -> Result<TimeSeriesPlot> {
        let mut plot = TimeSeriesPlot::new(title);
        plot.set_labels("Time", "Value");

        // Historical _data
        let hist_style = PlotStyle {
            color: "#1f77b4".to_string(), // Blue
            line_width: 2.0,
            ..Default::default()
        };
        plot.add_series("Historical", historical_time, historical_data, hist_style)?;

        // Forecast with confidence intervals
        let forecast_style = PlotStyle {
            color: "#ff7f0e".to_string(), // Orange
            line_width: 2.5,
            line_style: LineStyle::Dashed,
            ..Default::default()
        };
        plot.add_series_with_confidence(
            "Forecast",
            forecast_time,
            forecast_values,
            confidence_lower,
            confidence_upper,
            forecast_style,
        )?;

        Ok(plot)
    }

    /// Plot seasonal patterns
    pub fn plot_seasonal_patterns(
        _time: &Array1<f64>,
        data: &Array1<f64>,
        period: usize,
        title: &str,
    ) -> Result<TimeSeriesPlot> {
        let mut plot = TimeSeriesPlot::new(title);
        plot.set_labels("Time within Period", "Value");

        // Group data by seasonal period
        let num_periods = data.len() / period;
        let mut seasonal_data = Array2::<f64>::zeros((period, num_periods));

        for i in 0..num_periods {
            for j in 0..period {
                let idx = i * period + j;
                if idx < data.len() {
                    seasonal_data[[j, i]] = data[idx];
                }
            }
        }

        // Create _time axis for one period
        let period_time = Array1::linspace(0.0, period as f64 - 1.0, period);

        // Plot each period as a separate series
        for i in 0..num_periods.min(10) {
            // Limit to 10 periods for clarity
            let period_values = seasonal_data.column(i).to_owned();
            let style = PlotStyle {
                opacity: 0.6,
                color: "#1f77b4".to_string(), // Use same color with varying opacity
                ..Default::default()
            };
            plot.add_series(
                &format!("Period {}", i + 1),
                &period_time,
                &period_values,
                style,
            )?;
        }

        // Add mean seasonal pattern
        let mean_seasonal: Array1<f64> = seasonal_data
            .mean_axis(scirs2_core::ndarray::Axis(1))
            .unwrap();
        let mean_style = PlotStyle {
            color: "#d62728".to_string(), // Red
            line_width: 3.0,
            ..Default::default()
        };
        plot.add_series("Mean Pattern", &period_time, &mean_seasonal, mean_style)?;

        Ok(plot)
    }
}

/// Dashboard generation utilities
pub struct Dashboard {
    /// Dashboard title
    pub title: String,
    /// Collection of plots
    plots: Vec<(String, TimeSeriesPlot)>,
    /// Layout configuration
    layout: DashboardLayout,
}

/// Dashboard layout configuration
#[derive(Debug, Clone)]
pub struct DashboardLayout {
    /// Number of columns
    pub columns: usize,
    /// Plot spacing
    pub spacing: u32,
    /// Overall width
    pub width: u32,
    /// Overall height
    pub height: u32,
}

impl Default for DashboardLayout {
    fn default() -> Self {
        Self {
            columns: 2,
            spacing: 20,
            width: 1200,
            height: 800,
        }
    }
}

impl Dashboard {
    /// Create a new dashboard
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            plots: Vec::new(),
            layout: DashboardLayout::default(),
        }
    }

    /// Add a plot to the dashboard
    pub fn add_plot(&mut self, sectiontitle: &str, plot: TimeSeriesPlot) {
        self.plots.push((sectiontitle.to_string(), plot));
    }

    /// Configure dashboard layout
    pub fn set_layout(&mut self, layout: DashboardLayout) {
        self.layout = layout;
    }

    /// Generate HTML dashboard
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str(&format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }}
        .dashboard-title {{ 
            text-align: center; 
            font-size: 24px; 
            margin-bottom: 30px; 
            color: #333;
        }}
        .dashboard-container {{ 
            display: grid; 
            grid-template-columns: repeat({}, 1fr); 
            gap: {}px; 
            max-width: {}px; 
            margin: 0 auto; 
        }}
        .plot-section {{ 
            background: white; 
            border-radius: 8px; 
            padding: 20px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }}
        .plot-title {{ 
            font-size: 16px; 
            margin-bottom: 15px; 
            color: #555; 
            border-bottom: 2px solid #e0e0e0; 
            padding-bottom: 10px; 
        }}
        .plot-container {{ 
            width: 100%; 
            height: 400px; 
        }}
    </style>
</head>
<body>
    <div class="dashboard-title">{}</div>
    <div class="dashboard-container">
"#,
            self.title, self.layout.columns, self.layout.spacing, self.layout.width, self.title
        ));

        // Add each plot section
        for (i, (section_title_plot, _plot)) in self.plots.iter().enumerate() {
            html.push_str(&format!(
                r#"
        <div class="plot-section">
            <div class="plot-title">{}</div>
            <div id="plot_{i}" class="plot-container"></div>
        </div>
"#,
                section_title_plot
            ));
        }

        html.push_str("    </div>\n");

        // Add JavaScript for each plot
        html.push_str("    <script>\n");

        for (i, (_, plot)) in self.plots.iter().enumerate() {
            // Generate plot data for each plot
            html.push_str(&format!("        // Plot {i}\n"));
            html.push_str(&format!("        var data_{i} = [\n"));

            for (j, series) in plot.series.iter().enumerate() {
                if j > 0 {
                    html.push_str(",\n");
                }

                let x_values: Vec<f64> = series.data.iter().map(|p| p.time).collect();
                let y_values: Vec<f64> = series.data.iter().map(|p| p.value).collect();

                html.push_str(&format!(
                    r#"
            {{
                x: {:?},
                y: {:?},
                type: 'scatter',
                mode: 'lines',
                name: '{}',
                line: {{ color: '{}', width: {} }}
            }}"#,
                    x_values, y_values, series.name, series.style.color, series.style.line_width
                ));
            }

            html.push_str("\n        ];\n");
            html.push_str(&format!(
                r#"
        var layout_{} = {{
            title: '{}',
            xaxis: {{ title: '{}' }},
            yaxis: {{ title: '{}' }},
            margin: {{ l: 50, r: 20, t: 50, b: 50 }},
            showlegend: true
        }};
        
        Plotly.newPlot('plot_{}', data_{}, layout_{}, {{responsive: true}});
"#,
                i, plot.title, plot.x_label, plot.y_label, i, i, i
            ));
        }

        html.push_str("    </script>\n</body>\n</html>");
        html
    }

    /// Save dashboard to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let html_content = self.to_html();
        std::fs::write(path, html_content)
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to save dashboard: {e}")))?;
        Ok(())
    }

    /// Display dashboard (opens in default browser)
    pub fn show(&self) -> Result<()> {
        let temp_path = std::env::temp_dir().join("scirs2_dashboard.html");
        self.save(&temp_path)?;

        // Try to open in browser
        #[cfg(target_os = "windows")]
        std::process::Command::new("cmd")
            .args(["/c", "start", "", &temp_path.to_string_lossy()])
            .spawn()
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open dashboard: {e}")))?;

        #[cfg(target_os = "macos")]
        std::process::Command::new("open")
            .arg(&temp_path)
            .spawn()
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open dashboard: {e}")))?;

        #[cfg(target_os = "linux")]
        std::process::Command::new("xdg-open")
            .arg(&temp_path)
            .spawn()
            .map_err(|e| TimeSeriesError::IOError(format!("Failed to open dashboard: {e}")))?;

        Ok(())
    }
}

/// Convenience functions for quick plotting
pub mod quick_plots {
    use super::*;

    /// Quick line plot
    pub fn line_plot(x: &Array1<f64>, y: &Array1<f64>, title: &str) -> Result<TimeSeriesPlot> {
        let mut plot = TimeSeriesPlot::new(title);
        plot.add_series("data", x, y, PlotStyle::default())?;
        Ok(plot)
    }

    /// Quick scatter plot
    pub fn scatter_plot(x: &Array1<f64>, y: &Array1<f64>, title: &str) -> Result<TimeSeriesPlot> {
        let mut plot = TimeSeriesPlot::new(title);
        let style = PlotStyle {
            marker: MarkerStyle::Circle,
            ..Default::default()
        };
        plot.add_series("data", x, y, style)?;
        Ok(plot)
    }

    /// Quick multi-series plot
    pub fn multi_plot(
        series_data: &[(String, Array1<f64>, Array1<f64>)],
        title: &str,
    ) -> Result<TimeSeriesPlot> {
        let mut plot = TimeSeriesPlot::new(title);

        let colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        ];

        for (i, (name, x, y)) in series_data.iter().enumerate() {
            let style = PlotStyle {
                color: colors[i % colors.len()].to_string(),
                ..Default::default()
            };
            plot.add_series(name, x, y, style)?;
        }

        Ok(plot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_plot_creation() {
        let plot = TimeSeriesPlot::new("Test Plot");
        assert_eq!(plot.title, "Test Plot");
        assert_eq!(plot.x_label, "Time");
        assert_eq!(plot.y_label, "Value");
    }

    #[test]
    fn test_add_series() {
        let mut plot = TimeSeriesPlot::new("Test Plot");
        let time = Array1::linspace(0.0, 10.0, 11);
        let values = time.mapv(|x: f64| x.sin());

        let result = plot.add_series("sine", &time, &values, PlotStyle::default());
        assert!(result.is_ok());
        assert_eq!(plot.series.len(), 1);
        assert_eq!(plot.series[0].name, "sine");
    }

    #[test]
    fn test_mismatched_arrays() {
        let mut plot = TimeSeriesPlot::new("Test Plot");
        let time = Array1::linspace(0.0, 10.0, 11);
        let values = Array1::linspace(0.0, 5.0, 6); // Different length

        let result = plot.add_series("test", &time, &values, PlotStyle::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_html_generation() {
        let mut plot = TimeSeriesPlot::new("Test Plot");
        let time = Array1::linspace(0.0, 10.0, 11);
        let values = time.mapv(|x: f64| x.sin());

        plot.add_series("sine", &time, &values, PlotStyle::default())
            .unwrap();
        let html = plot.to_html();

        assert!(html.contains("Test Plot"));
        assert!(html.contains("sine"));
        assert!(html.contains("Plotly.newPlot"));
    }

    #[test]
    fn test_dashboard_creation() {
        let mut dashboard = Dashboard::new("Test Dashboard");
        let mut plot = TimeSeriesPlot::new("Sub Plot");
        let time = Array1::linspace(0.0, 10.0, 11);
        let values = time.mapv(|x: f64| x.sin());

        plot.add_series("sine", &time, &values, PlotStyle::default())
            .unwrap();
        dashboard.add_plot("Section 1", plot);

        assert_eq!(dashboard.plots.len(), 1);
        assert_eq!(dashboard.plots[0].0, "Section 1");
    }
}
