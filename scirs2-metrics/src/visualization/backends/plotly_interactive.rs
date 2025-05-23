//! Plotly backend for interactive visualizations
//!
//! This module provides a backend adapter for creating interactive visualizations
//! using Plotly.

use plotly::Plot;
use plotly::{
    common::Mode,
    layout::{Axis, Layout},
    Scatter,
};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::visualization::PlotType;
use crate::visualization::{VisualizationData, VisualizationMetadata, VisualizationOptions};

/// A backend adapter for Plotly interactive visualizations
pub struct PlotlyInteractiveBackend;

impl Default for PlotlyInteractiveBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PlotlyInteractiveBackend {
    /// Create a new Plotly interactive backend
    pub fn new() -> Self {
        Self
    }

    /// Create an interactive ROC curve visualization
    pub fn create_interactive_roc(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<Plot, Box<dyn Error>> {
        let mut plot = Plot::new();

        // Set layout
        let width = options.width;
        let height = options.height;
        let mut layout = Layout::new()
            .title(plotly::common::Title::with_text(metadata.title.as_str()))
            .x_axis(
                plotly::layout::Axis::new()
                    .title(plotly::common::Title::with_text(metadata.x_label.as_str())),
            )
            .y_axis(
                plotly::layout::Axis::new()
                    .title(plotly::common::Title::with_text(metadata.y_label.as_str())),
            )
            .width(width)
            .height(height);

        if options.show_grid {
            layout = layout.show_legend(true);
        }

        // Main ROC curve trace
        let roc_trace = Scatter::new(data.x.clone(), data.y.clone())
            .mode(Mode::Lines)
            .name(
                data.series_names
                    .as_ref()
                    .map_or("ROC curve", |names| names[0].as_str()),
            );

        plot.add_trace(roc_trace);

        // Add baseline if available
        if let Some(baseline_x) = data.auxiliary_data.get("baseline_x") {
            if let Some(baseline_y) = data.auxiliary_data.get("baseline_y") {
                let name = data
                    .series_names
                    .as_ref()
                    .map_or("Random classifier", |names| {
                        if names.len() > 1 {
                            names[1].as_str()
                        } else {
                            "Random classifier"
                        }
                    });

                let baseline_trace = Scatter::new(baseline_x.clone(), baseline_y.clone())
                    .mode(Mode::Lines)
                    .name(name)
                    .line(
                        plotly::common::Line::new()
                            .dash(plotly::common::DashType::Dash)
                            .color("gray"),
                    );

                plot.add_trace(baseline_trace);
            }
        }

        // Add current threshold point if available
        if let Some(current_x) = data.auxiliary_data.get("current_point_x") {
            if let Some(current_y) = data.auxiliary_data.get("current_point_y") {
                let name = data
                    .series_names
                    .as_ref()
                    .map_or("Current threshold", |names| {
                        if names.len() > 2 {
                            names[2].as_str()
                        } else {
                            "Current threshold"
                        }
                    });

                let point_trace = Scatter::new(current_x.clone(), current_y.clone())
                    .mode(Mode::Markers)
                    .name(name);

                plot.add_trace(point_trace);

                // Add metrics annotation if available
                let mut annotations = Vec::new();

                if let Some(threshold) = data.auxiliary_metadata.get("current_threshold") {
                    let x = current_x[0];
                    let y = current_y[0];

                    // Collect metrics
                    let mut metrics_text = format!("Threshold: {}", threshold);

                    for (key, value) in &data.auxiliary_metadata {
                        if key.starts_with("metric_") {
                            let metric_name = key.strip_prefix("metric_").unwrap_or(key);
                            if metric_name != "threshold" {
                                metrics_text.push_str(&format!("<br>{}: {}", metric_name, value));
                            }
                        }
                    }

                    let annotation = plotly::layout::Annotation::new()
                        .x(x)
                        .y(y)
                        .text(&metrics_text)
                        .show_arrow(true);

                    annotations.push(annotation);
                }

                if !annotations.is_empty() {
                    layout = layout.annotations(annotations);
                }
            }
        }

        // Set layout
        plot.set_layout(layout);

        // Add threshold slider if requested
        if let Some(thresholds) = data.auxiliary_data.get("thresholds") {
            if thresholds.len() > 1 && options.show_grid {
                // In a real implementation, this would add a JavaScript-based interactive slider
                // Since this is a simplified version, we'll just add a comment here
                // Actual implementation would require adding JavaScript to the Plotly HTML output
            }
        }

        Ok(plot)
    }

    /// Save a Plotly visualization to an HTML file
    pub fn save_to_html(&self, plot: &Plot, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let html = plot.to_html();
        let mut file = File::create(path.as_ref())?;
        file.write_all(html.as_bytes())?;
        Ok(())
    }
}

/// Interface for Plotly interactive backend
pub trait PlotlyInteractiveBackendInterface {
    /// Create an interactive ROC curve visualization and save it to an HTML file
    fn save_interactive_roc(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn Error>>;

    /// Create an interactive ROC curve visualization and return the HTML
    fn render_interactive_roc_html(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<String, Box<dyn Error>>;
}

impl PlotlyInteractiveBackendInterface for PlotlyInteractiveBackend {
    fn save_interactive_roc(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn Error>> {
        let plot = self.create_interactive_roc(data, metadata, options)?;
        self.save_to_html(&plot, path)?;
        Ok(())
    }

    fn render_interactive_roc_html(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
    ) -> Result<String, Box<dyn Error>> {
        let plot = self.create_interactive_roc(data, metadata, options)?;
        Ok(plot.to_html())
    }
}

impl super::PlottingBackend for PlotlyInteractiveBackend {
    fn save_to_file(
        &self,
        data: &VisualizationData,
        metadata: &VisualizationMetadata,
        options: &VisualizationOptions,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn Error>> {
        match metadata.plot_type {
            PlotType::Line => {
                // Special handling for ROC curves
                if metadata.x_label == "False Positive Rate"
                    && metadata.y_label == "True Positive Rate"
                {
                    // This is an ROC curve
                    return self.save_interactive_roc(data, metadata, options, path);
                }

                // For other line plots, create a regular Plotly plot
                let mut plot = Plot::new();
                let trace = Scatter::new(data.x.clone(), data.y.clone())
                    .mode(Mode::Lines)
                    .name(&metadata.title);

                plot.add_trace(trace);

                let layout = Layout::new()
                    .title(plotly::common::Title::with_text(&metadata.title))
                    .x_axis(
                        plotly::layout::Axis::new()
                            .title(plotly::common::Title::with_text(&metadata.x_label)),
                    )
                    .y_axis(
                        plotly::layout::Axis::new()
                            .title(plotly::common::Title::with_text(&metadata.y_label)),
                    )
                    .width(options.width)
                    .height(options.height);

                plot.set_layout(layout);
                self.save_to_html(&plot, path)?;
            }
            _ => {
                // For other plot types, create a regular Plotly plot
                // (Implementation would depend on plot type)
                return Err(format!(
                    "Plot type {:?} not implemented for Plotly interactive backend",
                    metadata.plot_type
                )
                .into());
            }
        }

        Ok(())
    }

    fn render_svg(
        &self,
        _data: &VisualizationData,
        _metadata: &VisualizationMetadata,
        _options: &VisualizationOptions,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        Err("SVG rendering not supported by Plotly interactive backend".into())
    }

    fn render_png(
        &self,
        _data: &VisualizationData,
        _metadata: &VisualizationMetadata,
        _options: &VisualizationOptions,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        Err("PNG rendering not supported by Plotly interactive backend".into())
    }
}
