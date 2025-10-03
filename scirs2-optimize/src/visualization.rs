//! Visualization tools for optimization trajectories and analysis
//!
//! This module provides comprehensive visualization capabilities for optimization
//! processes, including trajectory plotting, convergence analysis, and parameter
//! surface visualization.

use crate::error::{ScirsError, ScirsResult};
use scirs2_core::error_context;
use scirs2_core::ndarray::{Array1, ArrayView1}; // Unused import: Array2, ArrayView2
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Trajectory data collected during optimization
#[derive(Debug, Clone)]
pub struct OptimizationTrajectory {
    /// Parameter values at each iteration
    pub parameters: Vec<Array1<f64>>,
    /// Function values at each iteration
    pub function_values: Vec<f64>,
    /// Gradient norms at each iteration (if available)
    pub gradient_norms: Vec<f64>,
    /// Step sizes at each iteration (if available)
    pub step_sizes: Vec<f64>,
    /// Custom metrics at each iteration
    pub custom_metrics: HashMap<String, Vec<f64>>,
    /// Iteration numbers
    pub nit: Vec<usize>,
    /// Wall clock times (in seconds from start)
    pub times: Vec<f64>,
}

impl OptimizationTrajectory {
    /// Create a new empty trajectory
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            function_values: Vec::new(),
            gradient_norms: Vec::new(),
            step_sizes: Vec::new(),
            custom_metrics: HashMap::new(),
            nit: Vec::new(),
            times: Vec::new(),
        }
    }

    /// Add a new point to the trajectory
    pub fn add_point(
        &mut self,
        iteration: usize,
        params: &ArrayView1<f64>,
        function_value: f64,
        time: f64,
    ) {
        self.nit.push(iteration);
        self.parameters.push(params.to_owned());
        self.function_values.push(function_value);
        self.times.push(time);
    }

    /// Add gradient norm information
    pub fn add_gradient_norm(&mut self, grad_norm: f64) {
        self.gradient_norms.push(grad_norm);
    }

    /// Add step size information
    pub fn add_step_size(&mut self, step_size: f64) {
        self.step_sizes.push(step_size);
    }

    /// Add custom metric
    pub fn add_custom_metric(&mut self, name: &str, value: f64) {
        self.custom_metrics
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    /// Get the number of recorded points
    pub fn len(&self) -> usize {
        self.nit.len()
    }

    /// Check if trajectory is empty
    pub fn is_empty(&self) -> bool {
        self.nit.is_empty()
    }

    /// Get the final parameter values
    pub fn final_parameters(&self) -> Option<&Array1<f64>> {
        self.parameters.last()
    }

    /// Get the final function value
    pub fn final_function_value(&self) -> Option<f64> {
        self.function_values.last().copied()
    }

    /// Calculate convergence rate (linear convergence coefficient)
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.function_values.len() < 3 {
            return None;
        }

        let n = self.function_values.len();
        let mut rates = Vec::new();

        for i in 1..(n - 1) {
            let f_current = self.function_values[i];
            let f_next = self.function_values[i + 1];
            let f_prev = self.function_values[i - 1];

            if (f_current - f_next).abs() > 1e-14 && (f_prev - f_current).abs() > 1e-14 {
                let rate = (f_current - f_next).abs() / (f_prev - f_current).abs();
                if rate.is_finite() && rate > 0.0 {
                    rates.push(rate);
                }
            }
        }

        if rates.is_empty() {
            None
        } else {
            Some(rates.iter().sum::<f64>() / rates.len() as f64)
        }
    }
}

impl Default for OptimizationTrajectory {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for trajectory visualization
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Output format (svg, png, html)
    pub format: OutputFormat,
    /// Width of the plot in pixels
    pub width: u32,
    /// Height of the plot in pixels
    pub height: u32,
    /// Title for the plot
    pub title: Option<String>,
    /// Whether to show grid
    pub show_grid: bool,
    /// Whether to use logarithmic scale for y-axis
    pub log_scale_y: bool,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Whether to show legend
    pub show_legend: bool,
    /// Custom styling
    pub custom_style: Option<String>,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Svg,
            width: 800,
            height: 600,
            title: None,
            show_grid: true,
            log_scale_y: false,
            color_scheme: ColorScheme::Default,
            show_legend: true,
            custom_style: None,
        }
    }
}

/// Supported output formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Svg,
    Png,
    Html,
    Data, // Raw data output
}

/// Color schemes for visualization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorScheme {
    Default,
    Viridis,
    Plasma,
    Scientific,
    Monochrome,
}

/// Main visualization interface
pub struct OptimizationVisualizer {
    config: VisualizationConfig,
}

impl OptimizationVisualizer {
    /// Create a new visualizer with default configuration
    pub fn new() -> Self {
        Self {
            config: VisualizationConfig::default(),
        }
    }

    /// Create a new visualizer with custom configuration
    pub fn with_config(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Plot convergence curve (function value vs iteration)
    pub fn plot_convergence(
        &self,
        trajectory: &OptimizationTrajectory,
        output_path: &Path,
    ) -> ScirsResult<()> {
        if trajectory.is_empty() {
            return Err(ScirsError::InvalidInput(error_context!("Empty trajectory")));
        }

        match self.config.format {
            OutputFormat::Svg => self.plot_convergence_svg(trajectory, output_path),
            OutputFormat::Html => self.plot_convergence_html(trajectory, output_path),
            OutputFormat::Data => self.export_convergence_data(trajectory, output_path),
            _ => Err(ScirsError::NotImplementedError(error_context!(
                "PNG output not yet implemented"
            ))),
        }
    }

    /// Plot parameter trajectory (for 2D problems)
    pub fn plot_parameter_trajectory(
        &self,
        trajectory: &OptimizationTrajectory,
        output_path: &Path,
    ) -> ScirsResult<()> {
        if trajectory.is_empty() {
            return Err(ScirsError::InvalidInput(error_context!("Empty trajectory")));
        }

        if trajectory.parameters[0].len() != 2 {
            return Err(ScirsError::InvalidInput(error_context!(
                "Parameter trajectory visualization only supports 2D problems"
            )));
        }

        match self.config.format {
            OutputFormat::Svg => self.plot_trajectory_svg(trajectory, output_path),
            OutputFormat::Html => self.plot_trajectory_html(trajectory, output_path),
            OutputFormat::Data => self.export_trajectory_data(trajectory, output_path),
            _ => Err(ScirsError::NotImplementedError(error_context!(
                "PNG output not yet implemented"
            ))),
        }
    }

    /// Create a comprehensive optimization report
    pub fn create_optimization_report(
        &self,
        trajectory: &OptimizationTrajectory,
        output_dir: &Path,
    ) -> ScirsResult<()> {
        std::fs::create_dir_all(output_dir)?;

        // Generate convergence plot
        let convergence_path = output_dir.join("convergence.svg");
        self.plot_convergence(trajectory, &convergence_path)?;

        // Generate parameter trajectory if 2D
        if !trajectory.parameters.is_empty() && trajectory.parameters[0].len() == 2 {
            let trajectory_path = output_dir.join("trajectory.svg");
            self.plot_parameter_trajectory(trajectory, &trajectory_path)?;
        }

        // Generate summary statistics
        let summary_path = output_dir.join("summary.html");
        self.generate_summary_report(trajectory, &summary_path)?;

        // Export raw data
        let data_path = output_dir.join("data.csv");
        self.export_convergence_data(trajectory, &data_path)?;

        Ok(())
    }

    /// Generate summary statistics report
    fn generate_summary_report(
        &self,
        trajectory: &OptimizationTrajectory,
        output_path: &Path,
    ) -> ScirsResult<()> {
        let mut file = File::create(output_path)?;

        let html_content = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Optimization Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ margin: 10px 0; }}
        .value {{ font-weight: bold; color: #2E86AB; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Optimization Summary Report</h1>
    
    <h2>Basic Statistics</h2>
    <div class="metric">Total Iterations: <span class="value">{}</span></div>
    <div class="metric">Final Function Value: <span class="value">{:.6e}</span></div>
    <div class="metric">Initial Function Value: <span class="value">{:.6e}</span></div>
    <div class="metric">Function Improvement: <span class="value">{:.6e}</span></div>
    <div class="metric">Total Runtime: <span class="value">{:.3}s</span></div>
    {}
    
    <h2>Convergence Analysis</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Convergence Rate</td><td>{}</td></tr>
        <tr><td>Average Iteration Time</td><td>{:.6}s</td></tr>
        <tr><td>Function Evaluations per Second</td><td>{:.2}</td></tr>
    </table>
    
    {}
</body>
</html>"#,
            trajectory.len(),
            trajectory.final_function_value().unwrap_or(0.0),
            trajectory.function_values.first().cloned().unwrap_or(0.0),
            trajectory.function_values.first().cloned().unwrap_or(0.0)
                - trajectory.final_function_value().unwrap_or(0.0),
            trajectory.times.last().cloned().unwrap_or(0.0),
            if !trajectory.gradient_norms.is_empty() {
                format!("<div class=\"metric\">Final Gradient Norm: <span class=\"value\">{:.6e}</span></div>",
                       trajectory.gradient_norms.last().cloned().unwrap_or(0.0))
            } else {
                String::new()
            },
            trajectory
                .convergence_rate()
                .map(|r| format!("{:.6}", r))
                .unwrap_or_else(|| "N/A".to_string()),
            if trajectory.len() > 1 && !trajectory.times.is_empty() {
                trajectory.times.last().cloned().unwrap_or(0.0) / trajectory.len() as f64
            } else {
                0.0
            },
            if !trajectory.times.is_empty() && trajectory.times.last().cloned().unwrap_or(0.0) > 0.0
            {
                trajectory.len() as f64 / trajectory.times.last().cloned().unwrap_or(1.0)
            } else {
                0.0
            },
            self.generate_custom_metrics_table(trajectory)
        );

        file.write_all(html_content.as_bytes())?;
        Ok(())
    }

    fn generate_custom_metrics_table(&self, trajectory: &OptimizationTrajectory) -> String {
        if trajectory.custom_metrics.is_empty() {
            return String::new();
        }

        let mut table = String::from("<h2>Custom Metrics</h2>\n<table>\n<tr><th>Metric</th><th>Final Value</th><th>Min</th><th>Max</th><th>Mean</th></tr>\n");

        for (name, values) in &trajectory.custom_metrics {
            if let Some(final_val) = values.last() {
                let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mean_val = values.iter().sum::<f64>() / values.len() as f64;

                table.push_str(&format!(
                    "<tr><td>{}</td><td>{:.6e}</td><td>{:.6e}</td><td>{:.6e}</td><td>{:.6e}</td></tr>\n",
                    name, final_val, min_val, max_val, mean_val
                ));
            }
        }
        table.push_str("</table>\n");
        table
    }

    fn plot_convergence_svg(
        &self,
        trajectory: &OptimizationTrajectory,
        output_path: &Path,
    ) -> ScirsResult<()> {
        let mut file = File::create(output_path)?;

        let width = self.config.width;
        let height = self.config.height;
        let margin = 60;
        let plot_width = width - 2 * margin;
        let plot_height = height - 2 * margin;

        let min_y = if self.config.log_scale_y {
            trajectory
                .function_values
                .iter()
                .filter(|&&v| v > 0.0)
                .cloned()
                .fold(f64::INFINITY, f64::min)
                .ln()
        } else {
            trajectory
                .function_values
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min)
        };

        let max_y = if self.config.log_scale_y {
            trajectory
                .function_values
                .iter()
                .filter(|&&v| v > 0.0)
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
                .ln()
        } else {
            trajectory
                .function_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let max_x = trajectory.nit.len() as f64;

        let mut svg_content = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .axis {{ stroke: #333; stroke-width: 1; }}
            .grid {{ stroke: #ccc; stroke-width: 0.5; stroke-dasharray: 2,2; }}
            .line {{ fill: none; stroke: #2E86AB; stroke-width: 2; }}
            .text {{ font-family: Arial, sans-serif; font-size: 12px; fill: #333; }}
            .title {{ font-family: Arial, sans-serif; font-size: 16px; fill: #333; font-weight: bold; }}
        </style>
    </defs>
"#,
            width, height
        );

        // Grid lines
        if self.config.show_grid {
            for i in 0..=10 {
                let x = margin as f64 + (i as f64 / 10.0) * plot_width as f64;
                svg_content.push_str(&format!(
                    r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid" />
"#,
                    x,
                    margin,
                    x,
                    height - margin
                ));
            }

            for i in 0..=10 {
                let y = margin as f64 + (i as f64 / 10.0) * plot_height as f64;
                svg_content.push_str(&format!(
                    r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid" />
"#,
                    margin,
                    y,
                    width - margin,
                    y
                ));
            }
        }

        // Axes
        svg_content.push_str(&format!(
            r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis" />
    <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis" />
"#,
            margin,
            height - margin,
            width - margin,
            height - margin, // x-axis
            margin,
            margin,
            margin,
            height - margin // y-axis
        ));

        // Plot line
        svg_content.push_str("    <polyline points=\"");
        for (i, &f_val) in trajectory.function_values.iter().enumerate() {
            let x = margin as f64 + (i as f64 / max_x) * plot_width as f64;
            let y_val = if self.config.log_scale_y && f_val > 0.0 {
                f_val.ln()
            } else {
                f_val
            };
            let y = height as f64
                - margin as f64
                - ((y_val - min_y) / (max_y - min_y)) * plot_height as f64;
            svg_content.push_str(&format!("{},{} ", x, y));
        }
        svg_content.push_str("\" class=\"line\" />\n");

        // Title
        if let Some(ref title) = self.config.title {
            svg_content.push_str(&format!(
                r#"    <text x="{}" y="30" text-anchor="middle" class="title">{}</text>
"#,
                width / 2,
                title
            ));
        }

        // Labels
        svg_content.push_str(&format!(
            r#"    <text x="{}" y="{}" text-anchor="middle" class="text">Iteration</text>
    <text x="20" y="{}" text-anchor="middle" class="text" transform="rotate(-90 20 {})">Function Value{}</text>
"#,
            width / 2, height - 10,
            height / 2, height / 2,
            if self.config.log_scale_y { " (log)" } else { "" }
        ));

        svg_content.push_str("</svg>");

        file.write_all(svg_content.as_bytes())?;
        Ok(())
    }

    fn plot_convergence_html(
        &self,
        trajectory: &OptimizationTrajectory,
        output_path: &Path,
    ) -> ScirsResult<()> {
        let mut file = File::create(output_path)?;

        let html_content = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Optimization Convergence</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="convergence-plot" style="width:{}px;height:{}px;"></div>
    <script>
        var trace = {{
            x: [{}],
            y: [{}],
            type: 'scatter',
            mode: 'lines',
            name: 'Function Value',
            line: {{ color: '#2E86AB', width: 2 }}
        }};
        
        var layout = {{
            title: '{}',
            xaxis: {{ title: 'Iteration' }},
            yaxis: {{ 
                title: 'Function Value',
                type: '{}'
            }},
            showlegend: {}
        }};
        
        Plotly.newPlot('convergence-plot', [trace], layout);
    </script>
</body>
</html>"#,
            self.config.width,
            self.config.height,
            trajectory
                .nit
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
            trajectory
                .function_values
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(","),
            self.config
                .title
                .as_deref()
                .unwrap_or("Optimization Convergence"),
            if self.config.log_scale_y {
                "log"
            } else {
                "linear"
            },
            self.config.show_legend
        );

        file.write_all(html_content.as_bytes())?;
        Ok(())
    }

    fn plot_trajectory_svg(
        &self,
        trajectory: &OptimizationTrajectory,
        output_path: &Path,
    ) -> ScirsResult<()> {
        let mut file = File::create(output_path)?;

        let width = self.config.width;
        let height = self.config.height;
        let margin = 60;
        let plot_width = width - 2 * margin;
        let plot_height = height - 2 * margin;

        let x_coords: Vec<f64> = trajectory.parameters.iter().map(|p| p[0]).collect();
        let y_coords: Vec<f64> = trajectory.parameters.iter().map(|p| p[1]).collect();

        let min_x = x_coords.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_x = x_coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_y = y_coords.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_y = y_coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut svg_content = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <style>
            .axis {{ stroke: #333; stroke-width: 1; }}
            .grid {{ stroke: #ccc; stroke-width: 0.5; stroke-dasharray: 2,2; }}
            .trajectory {{ fill: none; stroke: #2E86AB; stroke-width: 2; }}
            .start {{ fill: #4CAF50; stroke: #333; stroke-width: 1; }}
            .end {{ fill: #F44336; stroke: #333; stroke-width: 1; }}
            .text {{ font-family: Arial, sans-serif; font-size: 12px; fill: #333; }}
            .title {{ font-family: Arial, sans-serif; font-size: 16px; fill: #333; font-weight: bold; }}
        </style>
    </defs>
"#,
            width, height
        );

        // Grid
        if self.config.show_grid {
            for i in 0..=10 {
                let x = margin as f64 + (i as f64 / 10.0) * plot_width as f64;
                svg_content.push_str(&format!(
                    r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid" />
"#,
                    x,
                    margin,
                    x,
                    height - margin
                ));
            }

            for i in 0..=10 {
                let y = margin as f64 + (i as f64 / 10.0) * plot_height as f64;
                svg_content.push_str(&format!(
                    r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" class="grid" />
"#,
                    margin,
                    y,
                    width - margin,
                    y
                ));
            }
        }

        // Axes
        svg_content.push_str(&format!(
            r#"    <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis" />
    <line x1="{}" y1="{}" x2="{}" y2="{}" class="axis" />
"#,
            margin,
            height - margin,
            width - margin,
            height - margin,
            margin,
            margin,
            margin,
            height - margin
        ));

        // Trajectory
        svg_content.push_str("    <polyline points=\"");
        for (x_val, y_val) in x_coords.iter().zip(y_coords.iter()) {
            let x = margin as f64 + ((x_val - min_x) / (max_x - min_x)) * plot_width as f64;
            let y = height as f64
                - margin as f64
                - ((y_val - min_y) / (max_y - min_y)) * plot_height as f64;
            svg_content.push_str(&format!("{},{} ", x, y));
        }
        svg_content.push_str("\" class=\"trajectory\" />\n");

        // Start and end points
        if !x_coords.is_empty() {
            let start_x =
                margin as f64 + ((x_coords[0] - min_x) / (max_x - min_x)) * plot_width as f64;
            let start_y = height as f64
                - margin as f64
                - ((y_coords[0] - min_y) / (max_y - min_y)) * plot_height as f64;

            let end_x = margin as f64
                + ((x_coords.last().unwrap() - min_x) / (max_x - min_x)) * plot_width as f64;
            let end_y = height as f64
                - margin as f64
                - ((y_coords.last().unwrap() - min_y) / (max_y - min_y)) * plot_height as f64;

            svg_content.push_str(&format!(
                r#"    <circle cx="{}" cy="{}" r="5" class="start" />
    <circle cx="{}" cy="{}" r="5" class="end" />
"#,
                start_x, start_y, end_x, end_y
            ));
        }

        // Title
        if let Some(ref title) = self.config.title {
            svg_content.push_str(&format!(
                r#"    <text x="{}" y="30" text-anchor="middle" class="title">{}</text>
"#,
                width / 2,
                title
            ));
        }

        svg_content.push_str("</svg>");

        file.write_all(svg_content.as_bytes())?;
        Ok(())
    }

    fn plot_trajectory_html(
        &self,
        trajectory: &OptimizationTrajectory,
        output_path: &Path,
    ) -> ScirsResult<()> {
        let mut file = File::create(output_path)?;

        let x_coords: Vec<f64> = trajectory.parameters.iter().map(|p| p[0]).collect();
        let y_coords: Vec<f64> = trajectory.parameters.iter().map(|p| p[1]).collect();

        let html_content = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Parameter Trajectory</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="trajectory-plot" style="width:{}px;height:{}px;"></div>
    <script>
        var trace = {{
            x: [{}],
            y: [{}],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Trajectory',
            line: {{ color: '#2E86AB', width: 2 }},
            marker: {{ 
                size: [{}],
                color: [{}],
                colorscale: 'Viridis',
                showscale: true
            }}
        }};
        
        var layout = {{
            title: '{}',
            xaxis: {{ title: 'Parameter 1' }},
            yaxis: {{ title: 'Parameter 2' }},
            showlegend: {}
        }};
        
        Plotly.newPlot('trajectory-plot', [trace], layout);
    </script>
</body>
</html>"#,
            self.config.width,
            self.config.height,
            x_coords
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(","),
            y_coords
                .iter()
                .map(|y| y.to_string())
                .collect::<Vec<_>>()
                .join(","),
            (0..x_coords.len())
                .map(|i| if i == 0 {
                    "10"
                } else if i == x_coords.len() - 1 {
                    "10"
                } else {
                    "6"
                })
                .collect::<Vec<_>>()
                .join(","),
            (0..x_coords.len())
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
            self.config
                .title
                .as_deref()
                .unwrap_or("Parameter Trajectory"),
            self.config.show_legend
        );

        file.write_all(html_content.as_bytes())?;
        Ok(())
    }

    fn export_convergence_data(
        &self,
        trajectory: &OptimizationTrajectory,
        output_path: &Path,
    ) -> ScirsResult<()> {
        let mut file = File::create(output_path)?;

        // CSV header
        let mut header = "iteration,function_value,time".to_string();
        if !trajectory.gradient_norms.is_empty() {
            header.push_str(",gradient_norm");
        }
        if !trajectory.step_sizes.is_empty() {
            header.push_str(",step_size");
        }

        // Add parameter columns
        if !trajectory.parameters.is_empty() {
            for i in 0..trajectory.parameters[0].len() {
                header.push_str(&format!(",param_{}", i));
            }
        }

        // Add custom metrics
        for name in trajectory.custom_metrics.keys() {
            header.push_str(&format!(",{}", name));
        }
        header.push('\n');

        file.write_all(header.as_bytes())?;

        // Data rows
        for i in 0..trajectory.len() {
            let mut row = format!(
                "{},{},{}",
                trajectory.nit[i], trajectory.function_values[i], trajectory.times[i]
            );

            if i < trajectory.gradient_norms.len() {
                row.push_str(&format!(",{}", trajectory.gradient_norms[i]));
            } else if !trajectory.gradient_norms.is_empty() {
                row.push(',');
            }

            if i < trajectory.step_sizes.len() {
                row.push_str(&format!(",{}", trajectory.step_sizes[i]));
            } else if !trajectory.step_sizes.is_empty() {
                row.push(',');
            }

            // Parameters
            if i < trajectory.parameters.len() {
                for param in trajectory.parameters[i].iter() {
                    row.push_str(&format!(",{}", param));
                }
            }

            // Custom metrics
            for name in trajectory.custom_metrics.keys() {
                if let Some(values) = trajectory.custom_metrics.get(name) {
                    if i < values.len() {
                        row.push_str(&format!(",{}", values[i]));
                    } else {
                        row.push(',');
                    }
                }
            }

            row.push('\n');
            file.write_all(row.as_bytes())?;
        }

        Ok(())
    }

    fn export_trajectory_data(
        &self,
        trajectory: &OptimizationTrajectory,
        output_path: &Path,
    ) -> ScirsResult<()> {
        self.export_convergence_data(trajectory, output_path)
    }
}

impl Default for OptimizationVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for creating trajectory trackers
pub mod tracking {
    use super::OptimizationTrajectory;
    use scirs2_core::ndarray::ArrayView1;
    use std::time::Instant;

    /// A callback-based trajectory tracker for use with optimization algorithms
    pub struct TrajectoryTracker {
        trajectory: OptimizationTrajectory,
        start_time: Instant,
    }

    impl TrajectoryTracker {
        /// Create a new trajectory tracker
        pub fn new() -> Self {
            Self {
                trajectory: OptimizationTrajectory::new(),
                start_time: Instant::now(),
            }
        }

        /// Record a new point in the optimization trajectory
        pub fn record(&mut self, iteration: usize, params: &ArrayView1<f64>, function_value: f64) {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            self.trajectory
                .add_point(iteration, params, function_value, elapsed);
        }

        /// Record gradient norm
        pub fn record_gradient_norm(&mut self, grad_norm: f64) {
            self.trajectory.add_gradient_norm(grad_norm);
        }

        /// Record step size
        pub fn record_step_size(&mut self, step_size: f64) {
            self.trajectory.add_step_size(step_size);
        }

        /// Record custom metric
        pub fn record_custom_metric(&mut self, name: &str, value: f64) {
            self.trajectory.add_custom_metric(name, value);
        }

        /// Get the recorded trajectory
        pub fn trajectory(&self) -> &OptimizationTrajectory {
            &self.trajectory
        }

        /// Consume the tracker and return the trajectory
        pub fn into_trajectory(self) -> OptimizationTrajectory {
            self.trajectory
        }
    }

    impl Default for TrajectoryTracker {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_trajectory_creation() {
        let mut trajectory = OptimizationTrajectory::new();
        assert!(trajectory.is_empty());

        let params = array![1.0, 2.0];
        trajectory.add_point(0, &params.view(), 5.0, 0.1);

        assert_eq!(trajectory.len(), 1);
        assert_eq!(trajectory.final_function_value(), Some(5.0));
    }

    #[test]
    fn test_convergence_rate_calculation() {
        let mut trajectory = OptimizationTrajectory::new();

        // Add points with known convergence pattern
        let function_values = vec![10.0, 5.0, 2.5, 1.25, 0.625];
        for (i, &f_val) in function_values.iter().enumerate() {
            let params = array![i as f64, i as f64];
            trajectory.add_point(i, &params.view(), f_val, i as f64 * 0.1);
        }

        let rate = trajectory.convergence_rate();
        assert!(rate.is_some());
        // Should be approximately 0.5 for this geometric sequence
        assert!((rate.unwrap() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_visualization_config() {
        let config = VisualizationConfig {
            format: OutputFormat::Svg,
            width: 1000,
            height: 800,
            title: Some("Test Plot".to_string()),
            show_grid: true,
            log_scale_y: true,
            color_scheme: ColorScheme::Viridis,
            show_legend: false,
            custom_style: None,
        };

        let visualizer = OptimizationVisualizer::with_config(config);
        assert_eq!(visualizer.config.width, 1000);
        assert_eq!(visualizer.config.height, 800);
    }

    #[test]
    fn test_trajectory_tracker() {
        let mut tracker = tracking::TrajectoryTracker::new();

        let params1 = array![0.0, 0.0];
        let params2 = array![1.0, 1.0];

        tracker.record(0, &params1.view(), 10.0);
        tracker.record_gradient_norm(2.5);
        tracker.record_step_size(0.1);

        tracker.record(1, &params2.view(), 5.0);
        tracker.record_gradient_norm(1.5);
        tracker.record_step_size(0.2);

        let trajectory = tracker.trajectory();
        assert_eq!(trajectory.len(), 2);
        assert_eq!(trajectory.gradient_norms.len(), 2);
        assert_eq!(trajectory.step_sizes.len(), 2);
        assert_eq!(trajectory.final_function_value(), Some(5.0));
    }
}
