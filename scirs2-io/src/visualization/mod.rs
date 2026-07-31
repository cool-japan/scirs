//! Visualization tool integration for scientific data
//!
//! Provides interfaces and utilities for integrating with popular visualization
//! libraries and tools, enabling seamless data export and plotting capabilities.

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::{IoError, Result};
use crate::metadata::Metadata;
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

mod svg;
mod threed;

/// Visualization format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationFormat {
    /// Plotly JSON format
    PlotlyJson,
    /// Matplotlib script
    MatplotlibPython,
    /// Gnuplot script
    Gnuplot,
    /// D3.js JSON format
    D3Json,
    /// Vega-Lite specification
    VegaLite,
    /// Bokeh JSON format
    BokehJson,
    /// SVG format
    Svg,
    /// HTML with embedded visualization
    Html,
}

/// Plot type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlotType {
    Line,
    Scatter,
    Bar,
    Histogram,
    Heatmap,
    Surface,
    Contour,
    Box,
    Violin,
    Pie,
    Area,
    Stream,
}

/// Axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    pub title: Option<String>,
    pub range: Option<[f64; 2]>,
    pub scale: Option<ScaleType>,
    pub tick_format: Option<String>,
    pub grid: bool,
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            title: None,
            range: None,
            scale: None,
            tick_format: None,
            grid: true,
        }
    }
}

/// Scale types for axes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaleType {
    Linear,
    Log,
    SymLog,
    Sqrt,
    Power(f64),
}

/// Plot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotConfig {
    pub title: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub x_axis: AxisConfig,
    pub y_axis: AxisConfig,
    pub z_axis: Option<AxisConfig>,
    pub color_scale: Option<String>,
    pub theme: Option<String>,
    pub annotations: Vec<Annotation>,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: None,
            width: Some(800),
            height: Some(600),
            x_axis: AxisConfig::default(),
            y_axis: AxisConfig::default(),
            z_axis: None,
            color_scale: None,
            theme: None,
            annotations: Vec::new(),
        }
    }
}

/// Annotation for plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub arrow: bool,
}

/// Visualization builder
#[derive(Debug, Clone)]
pub struct VisualizationBuilder {
    data: Vec<DataSeries>,
    config: PlotConfig,
    metadata: Metadata,
}

/// Data series for plotting
#[derive(Debug, Clone)]
pub struct DataSeries {
    pub name: Option<String>,
    pub x: Option<Vec<f64>>,
    pub y: Vec<f64>,
    pub z: Option<Vec<f64>>,
    pub plot_type: PlotType,
    pub style: SeriesStyle,
}

/// Style configuration for data series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesStyle {
    pub color: Option<String>,
    pub line_style: Option<String>,
    pub marker: Option<String>,
    pub opacity: Option<f64>,
    pub size: Option<f64>,
}

impl Default for SeriesStyle {
    fn default() -> Self {
        Self {
            color: None,
            line_style: None,
            marker: None,
            opacity: Some(1.0),
            size: None,
        }
    }
}

impl Default for VisualizationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualizationBuilder {
    /// Create a new visualization builder
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            config: PlotConfig::default(),
            metadata: Metadata::new(),
        }
    }

    /// Set plot title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.config.title = Some(title.into());
        self
    }

    /// Set plot dimensions
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.config.width = Some(width);
        self.config.height = Some(height);
        self
    }

    /// Configure X axis
    pub fn x_axis(mut self, title: impl Into<String>) -> Self {
        self.config.x_axis.title = Some(title.into());
        self
    }

    /// Configure Y axis
    pub fn y_axis(mut self, title: impl Into<String>) -> Self {
        self.config.y_axis.title = Some(title.into());
        self
    }

    /// Add a line plot
    pub fn add_line(mut self, x: &[f64], y: &[f64], name: Option<&str>) -> Self {
        self.data.push(DataSeries {
            name: name.map(|s| s.to_string()),
            x: Some(x.to_vec()),
            y: y.to_vec(),
            z: None,
            plot_type: PlotType::Line,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Add a scatter plot
    pub fn add_scatter(mut self, x: &[f64], y: &[f64], name: Option<&str>) -> Self {
        self.data.push(DataSeries {
            name: name.map(|s| s.to_string()),
            x: Some(x.to_vec()),
            y: y.to_vec(),
            z: None,
            plot_type: PlotType::Scatter,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Add a histogram
    pub fn add_histogram(mut self, values: &[f64], name: Option<&str>) -> Self {
        self.data.push(DataSeries {
            name: name.map(|s| s.to_string()),
            x: None,
            y: values.to_vec(),
            z: None,
            plot_type: PlotType::Histogram,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Add a heatmap
    pub fn add_heatmap(mut self, z: Array2<f64>, name: Option<&str>) -> Self {
        let flat_z: Vec<f64> = z.iter().cloned().collect();
        self.data.push(DataSeries {
            name: name.map(|s| s.to_string()),
            x: Some(vec![z.shape()[1] as f64]), // Store dimensions
            y: vec![z.shape()[0] as f64],
            z: Some(flat_z),
            plot_type: PlotType::Heatmap,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Build and export to specified format
    pub fn export(self, format: VisualizationFormat, path: impl AsRef<Path>) -> Result<()> {
        let exporter = get_exporter(format);
        exporter.export(&self.data, &self.config, &self.metadata, path.as_ref())
    }

    /// Build and return as string
    pub fn to_string(self, format: VisualizationFormat) -> Result<String> {
        let exporter = get_exporter(format);
        exporter.to_string(&self.data, &self.config, &self.metadata)
    }
}

/// Trait for visualization exporters
trait VisualizationExporter {
    fn export(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()>;
    fn to_string(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
    ) -> Result<String>;
}

/// Get appropriate exporter for format
#[allow(dead_code)]
fn get_exporter(format: VisualizationFormat) -> Box<dyn VisualizationExporter> {
    match format {
        VisualizationFormat::PlotlyJson => Box::new(PlotlyExporter),
        VisualizationFormat::MatplotlibPython => Box::new(MatplotlibExporter),
        VisualizationFormat::Gnuplot => Box::new(GnuplotExporter),
        VisualizationFormat::VegaLite => Box::new(VegaLiteExporter),
        VisualizationFormat::D3Json => Box::new(D3Exporter),
        VisualizationFormat::BokehJson => Box::new(BokehExporter),
        VisualizationFormat::Svg => Box::new(svg::SvgExporter),
        VisualizationFormat::Html => Box::new(svg::HtmlExporter),
    }
}

/// Plotly JSON exporter
struct PlotlyExporter;

impl VisualizationExporter for PlotlyExporter {
    fn export(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let json_str = self.to_string(data, config, metadata)?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(json_str.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }

    fn to_string(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
    ) -> Result<String> {
        let mut traces = Vec::new();

        for series in data {
            let trace = match series.plot_type {
                PlotType::Line | PlotType::Scatter => {
                    serde_json::json!({
                        "type": "scatter",
                        "mode": if matches!(series.plot_type, PlotType::Line) { "lines" } else { "markers" },
                        "name": series.name,
                        "x": series.x,
                        "y": series.y,
                        "line": {
                            "color": series.style.color,
                            "dash": series.style.line_style,
                        },
                        "marker": {
                            "symbol": series.style.marker,
                            "size": series.style.size,
                        },
                        "opacity": series.style.opacity,
                    })
                }
                PlotType::Histogram => {
                    serde_json::json!({
                        "type": "histogram",
                        "name": series.name,
                        "x": series.y,
                        "opacity": series.style.opacity,
                    })
                }
                PlotType::Heatmap => {
                    let cols = series.x.as_ref().expect("Operation failed")[0] as usize;
                    let _rows = series.y[0] as usize;
                    let z_data: Vec<Vec<f64>> = series
                        .z
                        .as_ref()
                        .expect("Operation failed")
                        .chunks(cols)
                        .map(|chunk| chunk.to_vec())
                        .collect();

                    serde_json::json!({
                        "type": "heatmap",
                        "name": series.name,
                        "z": z_data,
                        "colorscale": config.color_scale,
                    })
                }
                _ => continue,
            };
            traces.push(trace);
        }

        let layout = serde_json::json!({
            "title": config.title,
            "width": config.width,
            "height": config.height,
            "xaxis": {
                "title": config.x_axis.title,
                "range": config.x_axis.range,
                "showgrid": config.x_axis.grid,
            },
            "yaxis": {
                "title": config.y_axis.title,
                "range": config.y_axis.range,
                "showgrid": config.y_axis.grid,
            },
            "annotations": config.annotations.iter().map(|ann| {
                serde_json::json!({
                    "text": ann.text,
                    "x": ann.x,
                    "y": ann.y,
                    "showarrow": ann.arrow,
                })
            }).collect::<Vec<_>>(),
        });

        let plot_data = serde_json::json!({
            "data": traces,
            "layout": layout,
        });

        serde_json::to_string_pretty(&plot_data)
            .map_err(|e| IoError::SerializationError(e.to_string()))
    }
}

/// Matplotlib Python script exporter
struct MatplotlibExporter;

impl VisualizationExporter for MatplotlibExporter {
    fn export(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let script = self.to_string(data, config, metadata)?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(script.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }

    fn to_string(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
    ) -> Result<String> {
        let mut script = String::from("import matplotlib.pyplot as plt\nimport numpy as np\n\n");

        // Create figure
        script.push_str(&format!(
            "fig, ax = plt.subplots(figsize=({}, {}))\n\n",
            config.width.unwrap_or(800) as f64 / 100.0,
            config.height.unwrap_or(600) as f64 / 100.0
        ));

        // Plot data
        for series in data {
            match series.plot_type {
                PlotType::Line => {
                    if let Some(x) = &series.x {
                        script.push_str(&format!("ax.plot({:?}, {:?}", x, series.y));
                        if let Some(name) = &series.name {
                            script.push_str(&format!(", label='{}'", name));
                        }
                        script.push_str(")\n");
                    }
                }
                PlotType::Scatter => {
                    if let Some(x) = &series.x {
                        script.push_str(&format!("ax.scatter({:?}, {:?}", x, series.y));
                        if let Some(name) = &series.name {
                            script.push_str(&format!(", label='{}'", name));
                        }
                        script.push_str(")\n");
                    }
                }
                PlotType::Histogram => {
                    script.push_str(&format!("ax.hist({:?}", series.y));
                    if let Some(name) = &series.name {
                        script.push_str(&format!(", label='{}'", name));
                    }
                    script.push_str(")\n");
                }
                _ => continue,
            }
        }

        // Configure axes
        if let Some(title) = &config.title {
            script.push_str(&format!("\nax.set_title('{}')\n", title));
        }
        if let Some(xlabel) = &config.x_axis.title {
            script.push_str(&format!("ax.set_xlabel('{}')\n", xlabel));
        }
        if let Some(ylabel) = &config.y_axis.title {
            script.push_str(&format!("ax.set_ylabel('{}')\n", ylabel));
        }

        script.push_str("\nax.grid(True)\n");
        script.push_str("ax.legend()\n");
        script.push_str("plt.tight_layout()\n");
        script.push_str("plt.show()\n");

        Ok(script)
    }
}

/// Gnuplot script exporter
struct GnuplotExporter;

impl VisualizationExporter for GnuplotExporter {
    fn export(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let script = self.to_string(data, config, metadata)?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(script.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }

    fn to_string(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
    ) -> Result<String> {
        let mut script = String::new();

        // Set terminal and output
        script.push_str("set terminal png size ");
        script.push_str(&format!(
            "{},{}\n",
            config.width.unwrap_or(800),
            config.height.unwrap_or(600)
        ));
        script.push_str("set output 'plot.png'\n\n");

        // Set title and labels
        if let Some(title) = &config.title {
            script.push_str(&format!("set title '{}'\n", title));
        }
        if let Some(xlabel) = &config.x_axis.title {
            script.push_str(&format!("set xlabel '{}'\n", xlabel));
        }
        if let Some(ylabel) = &config.y_axis.title {
            script.push_str(&format!("set ylabel '{}'\n", ylabel));
        }

        script.push_str("set grid\n\n");

        // Plot command
        script.push_str("plot ");
        let mut first = true;

        for (i, series) in data.iter().enumerate() {
            if !first {
                script.push_str(", ");
            }
            first = false;

            match series.plot_type {
                PlotType::Line => {
                    script.push_str(&format!(
                        "'-' using 1:2 with lines title '{}'",
                        series.name.as_deref().unwrap_or(&format!("Series {}", i))
                    ));
                }
                PlotType::Scatter => {
                    script.push_str(&format!(
                        "'-' using 1:2 with points title '{}'",
                        series.name.as_deref().unwrap_or(&format!("Series {}", i))
                    ));
                }
                _ => continue,
            }
        }
        script.push_str("\n\n");

        // Data blocks
        for series in data {
            if let Some(x) = &series.x {
                for (xi, yi) in x.iter().zip(series.y.iter()) {
                    script.push_str(&format!("{} {}\n", xi, yi));
                }
            }
            script.push_str("e\n");
        }

        Ok(script)
    }
}

/// Vega-Lite specification exporter
struct VegaLiteExporter;

impl VisualizationExporter for VegaLiteExporter {
    fn export(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let spec = self.to_string(data, config, metadata)?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(spec.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }

    fn to_string(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
    ) -> Result<String> {
        let mut data_values = Vec::new();

        for series in data {
            if let Some(x) = &series.x {
                for (xi, yi) in x.iter().zip(series.y.iter()) {
                    data_values.push(serde_json::json!({
                        "x": xi,
                        "y": yi,
                        "series": series.name.as_deref().unwrap_or("default"),
                    }));
                }
            }
        }

        let spec = serde_json::json!({
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": config.title,
            "width": config.width,
            "height": config.height,
            "data": {
                "values": data_values
            },
            "mark": "line",
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "title": config.x_axis.title,
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "title": config.y_axis.title,
                },
                "color": {
                    "field": "series",
                    "type": "nominal"
                }
            }
        });

        serde_json::to_string_pretty(&spec).map_err(|e| IoError::SerializationError(e.to_string()))
    }
}

/// Quick plotting functions
pub mod quick {
    use super::*;

    /// Quick line plot
    pub fn plot_line(x: &[f64], y: &[f64], output: impl AsRef<Path>) -> Result<()> {
        VisualizationBuilder::new()
            .title("Line Plot")
            .add_line(x, y, None)
            .export(VisualizationFormat::PlotlyJson, output)
    }

    /// Quick scatter plot
    pub fn plot_scatter(x: &[f64], y: &[f64], output: impl AsRef<Path>) -> Result<()> {
        VisualizationBuilder::new()
            .title("Scatter Plot")
            .add_scatter(x, y, None)
            .export(VisualizationFormat::PlotlyJson, output)
    }

    /// Quick histogram
    pub fn plot_histogram(values: &[f64], output: impl AsRef<Path>) -> Result<()> {
        VisualizationBuilder::new()
            .title("Histogram")
            .add_histogram(values, None)
            .export(VisualizationFormat::PlotlyJson, output)
    }

    /// Quick heatmap
    pub fn plot_heatmap(z: &Array2<f64>, output: impl AsRef<Path>) -> Result<()> {
        VisualizationBuilder::new()
            .title("Heatmap")
            .add_heatmap(z.clone(), None)
            .export(VisualizationFormat::PlotlyJson, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualization_builder() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];

        let result = VisualizationBuilder::new()
            .title("Test Plot")
            .x_axis("X values")
            .y_axis("Y values")
            .add_line(&x, &y, Some("y = x²"))
            .to_string(VisualizationFormat::PlotlyJson);

        assert!(result.is_ok());
        let json_str = result.expect("Operation failed");
        assert!(json_str.contains("Test Plot"));
        assert!(json_str.contains("y = x²"));
    }

    #[test]
    fn test_matplotlib_export() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];

        let result = VisualizationBuilder::new()
            .title("Matplotlib Test")
            .add_scatter(&x, &y, Some("data"))
            .to_string(VisualizationFormat::MatplotlibPython);

        assert!(result.is_ok());
        let script = result.expect("Operation failed");
        assert!(script.contains("import matplotlib.pyplot"));
        assert!(script.contains("ax.scatter"));
    }

    /// Extremely small well-formedness check: confirms every opening tag has a
    /// matching closing tag (ignoring self-closing tags and the `<?xml?>`/
    /// `<!DOCTYPE>` declarations) -- enough to catch unbalanced markup or
    /// unescaped user text without pulling in an XML-parsing dependency.
    fn assert_well_formed_xml(doc: &str) {
        let mut stack: Vec<String> = Vec::new();
        let mut pos = 0usize;
        while let Some(rel_start) = doc[pos..].find('<') {
            let start = pos + rel_start;
            let end = doc[start..]
                .find('>')
                .map(|p| start + p)
                .unwrap_or_else(|| panic!("unterminated '<' at byte {start} in: {doc}"));
            let tag = &doc[start + 1..end];
            pos = end + 1;

            if tag.starts_with('?') || tag.starts_with('!') {
                continue;
            }
            if let Some(name) = tag.strip_prefix('/') {
                let name = name.trim();
                let expected = stack
                    .pop()
                    .unwrap_or_else(|| panic!("unmatched closing tag '{name}' in: {doc}"));
                assert_eq!(expected, name, "mismatched closing tag in: {doc}");
            } else if tag.ends_with('/') {
                // self-closing tag: nothing to push
            } else {
                let name = tag.split_whitespace().next().unwrap_or(tag).to_string();
                stack.push(name);
            }
        }
        assert!(stack.is_empty(), "unclosed tags remain {stack:?} in: {doc}");
    }

    /// Non-constant sample data shared by the format-specific exporter tests
    /// below (an all-zero/all-constant series could "round trip" even through
    /// a fabricating stub).
    fn sample_builder() -> VisualizationBuilder {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![-3.5, 1.25, -0.5, 42.0, 7.75];
        VisualizationBuilder::new()
            .title("Sample Plot")
            .x_axis("time")
            .y_axis("value")
            .add_line(&x, &y, Some("series-a"))
    }

    #[test]
    fn test_get_exporter_formats_are_no_longer_all_plotly() {
        // Regression test for the original bug: D3Json/BokehJson/Svg/Html all
        // silently produced Plotly's own JSON schema. Each format must now
        // produce output structurally different from Plotly's.
        let plotly = sample_builder()
            .to_string(VisualizationFormat::PlotlyJson)
            .expect("Operation failed");
        let plotly_value: serde_json::Value =
            serde_json::from_str(&plotly).expect("Operation failed");
        assert!(plotly_value.get("data").is_some());
        assert!(plotly_value.get("layout").is_some());

        let d3 = sample_builder()
            .to_string(VisualizationFormat::D3Json)
            .expect("Operation failed");
        assert_ne!(d3, plotly);
        let d3_value: serde_json::Value = serde_json::from_str(&d3).expect("Operation failed");
        assert_eq!(d3_value["format"], "d3");
        assert!(
            d3_value.get("data").is_none(),
            "must not reuse Plotly's 'data' key"
        );

        let bokeh = sample_builder()
            .to_string(VisualizationFormat::BokehJson)
            .expect("Operation failed");
        assert_ne!(bokeh, plotly);
        assert_ne!(bokeh, d3);
        let bokeh_value: serde_json::Value =
            serde_json::from_str(&bokeh).expect("Operation failed");
        assert!(bokeh_value.get("doc").is_some());
        assert!(
            bokeh_value.get("data").is_none(),
            "must not reuse Plotly's 'data' key"
        );

        let svg = sample_builder()
            .to_string(VisualizationFormat::Svg)
            .expect("Operation failed");
        assert_ne!(svg, plotly);
        assert!(
            serde_json::from_str::<serde_json::Value>(&svg).is_err(),
            "SVG is not JSON"
        );
        assert!(svg.trim_start().starts_with("<svg"));

        let html = sample_builder()
            .to_string(VisualizationFormat::Html)
            .expect("Operation failed");
        assert_ne!(html, plotly);
        assert_ne!(html, svg);
        assert!(html.trim_start().starts_with("<!DOCTYPE html>"));
    }

    #[test]
    fn test_d3json_exporter_shape() {
        let json = sample_builder()
            .to_string(VisualizationFormat::D3Json)
            .expect("Operation failed");
        let value: serde_json::Value = serde_json::from_str(&json).expect("Operation failed");

        assert_eq!(value["format"], "d3");
        assert_eq!(value["title"], "Sample Plot");
        let series = value["series"].as_array().expect("Operation failed");
        assert_eq!(series.len(), 1);
        assert_eq!(series[0]["name"], "series-a");
        assert_eq!(series[0]["type"], "line");

        // D3-idiomatic array-of-records shape: each element is an {x, y} object,
        // NOT Plotly's parallel "x": [...], "y": [...] arrays.
        let values = series[0]["values"].as_array().expect("Operation failed");
        assert_eq!(values.len(), 5);
        assert_eq!(values[3]["x"], 3.0);
        assert_eq!(values[3]["y"], 42.0);
        assert!(series[0].get("x").is_none());
        assert!(series[0].get("y").is_none());
    }

    #[test]
    fn test_bokeh_json_exporter_shape() {
        let json = sample_builder()
            .to_string(VisualizationFormat::BokehJson)
            .expect("Operation failed");
        let value: serde_json::Value = serde_json::from_str(&json).expect("Operation failed");

        assert!(value.get("target_id").is_some());
        assert!(value.get("root_id").is_some());
        let references = value["doc"]["roots"]["references"]
            .as_array()
            .expect("Operation failed");
        assert!(
            !references.is_empty(),
            "must contain real model references, not an empty stub"
        );

        let source = references
            .iter()
            .find(|r| r["type"] == "ColumnDataSource")
            .expect("a ColumnDataSource reference must be present");
        let y_data: Vec<f64> = source["attributes"]["data"]["y"]
            .as_array()
            .expect("Operation failed")
            .iter()
            .map(|v| v.as_f64().expect("Operation failed"))
            .collect();
        assert_eq!(y_data, vec![-3.5, 1.25, -0.5, 42.0, 7.75]);

        assert!(references.iter().any(|r| r["type"] == "Plot"));
    }

    #[test]
    fn test_svg_exporter_is_well_formed_and_renders_line() {
        let svg = sample_builder()
            .to_string(VisualizationFormat::Svg)
            .expect("Operation failed");

        assert!(svg.starts_with("<svg"));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert!(
            svg.contains("<polyline"),
            "line plot must render a polyline"
        );
        assert!(
            svg.contains("Sample Plot"),
            "title must appear in the document"
        );
        assert_well_formed_xml(&svg);
    }

    #[test]
    fn test_svg_exporter_renders_scatter_and_histogram() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![5.0, -2.0, 3.0];
        let scatter_svg = VisualizationBuilder::new()
            .add_scatter(&x, &y, Some("pts"))
            .to_string(VisualizationFormat::Svg)
            .expect("Operation failed");
        assert!(scatter_svg.contains("<circle"));
        assert_well_formed_xml(&scatter_svg);

        let samples = vec![1.0, 1.5, 2.0, 2.5, 2.5, 3.0, 5.0, 5.5, 5.5, 5.5];
        let hist_svg = VisualizationBuilder::new()
            .add_histogram(&samples, Some("counts"))
            .to_string(VisualizationFormat::Svg)
            .expect("Operation failed");
        assert!(hist_svg.contains("<rect"));
        assert_well_formed_xml(&hist_svg);
    }

    #[test]
    fn test_svg_exporter_renders_heatmap_grid() {
        let z = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Operation failed");
        let svg = VisualizationBuilder::new()
            .add_heatmap(z, Some("grid"))
            .to_string(VisualizationFormat::Svg)
            .expect("Operation failed");

        // 2 rows x 3 cols = 6 cells, each rendered as its own <rect>, plus the
        // single background `<rect>` every `svg_document` emits.
        assert_eq!(svg.matches("<rect").count(), 7);
        assert_well_formed_xml(&svg);
    }

    #[test]
    fn test_html_exporter_is_self_contained() {
        let html = sample_builder()
            .to_string(VisualizationFormat::Html)
            .expect("Operation failed");

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
        assert!(html.contains("<polyline"));
        // The `xmlns="http://www.w3.org/2000/svg"` namespace declaration is a
        // required, purely symbolic XML identifier (never fetched over the
        // network), so check specifically for actual externally-loaded
        // resources rather than any occurrence of "http://".
        assert!(
            !html.contains("<script")
                && !html.contains("<link")
                && !html.contains("cdn.")
                && !html.contains("@import"),
            "self-contained HTML must not load any external script/stylesheet/CDN resource"
        );
    }

    #[test]
    fn test_svg_and_html_export_to_file() {
        let temp_dir = std::env::temp_dir();
        let svg_path = temp_dir.join(format!("test_viz_export_{}.svg", std::process::id()));
        let html_path = temp_dir.join(format!("test_viz_export_{}.html", std::process::id()));

        sample_builder()
            .export(VisualizationFormat::Svg, &svg_path)
            .expect("Operation failed");
        let svg_bytes = std::fs::read_to_string(&svg_path).expect("Operation failed");
        assert!(svg_bytes.starts_with("<svg"));

        sample_builder()
            .export(VisualizationFormat::Html, &html_path)
            .expect("Operation failed");
        let html_bytes = std::fs::read_to_string(&html_path).expect("Operation failed");
        assert!(html_bytes.contains("<!DOCTYPE html>"));

        let _ = std::fs::remove_file(svg_path);
        let _ = std::fs::remove_file(html_path);
    }

    fn sample_3d_builder() -> Visualization3DBuilder {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![-3.5, 1.25, -0.5, 42.0, 7.75];
        let z = vec![0.5, -2.0, 6.25, -1.5, 3.0];
        Visualization3DBuilder::new().add_scatter3d(&x, &y, &z, Some("cloud"))
    }

    #[test]
    fn test_get_3d_exporter_formats_are_no_longer_all_plotly() {
        // Regression test for the 3D counterpart of the original bug: every
        // format except PlotlyJson silently produced Plotly's own JSON
        // schema via `get_3d_exporter`'s `_ => Box::new(Plotly3DExporter)`
        // catch-all. Each real format must now produce output structurally
        // different from Plotly's.
        let temp_dir = std::env::temp_dir();
        let pid = std::process::id();

        let plotly_path = temp_dir.join(format!("test_3d_plotly_{pid}.json"));
        sample_3d_builder()
            .export(VisualizationFormat::PlotlyJson, &plotly_path)
            .expect("Operation failed");
        let plotly = std::fs::read_to_string(&plotly_path).expect("Operation failed");
        let plotly_value: serde_json::Value =
            serde_json::from_str(&plotly).expect("Operation failed");
        assert!(plotly_value.get("data").is_some());
        assert!(plotly_value["layout"].get("scene").is_some());

        let mpl_path = temp_dir.join(format!("test_3d_mpl_{pid}.py"));
        sample_3d_builder()
            .export(VisualizationFormat::MatplotlibPython, &mpl_path)
            .expect("Operation failed");
        let mpl = std::fs::read_to_string(&mpl_path).expect("Operation failed");
        assert_ne!(mpl, plotly);
        assert!(mpl.contains("projection='3d'"));
        assert!(mpl.contains("ax.scatter"));

        let gp_path = temp_dir.join(format!("test_3d_gnuplot_{pid}.gp"));
        sample_3d_builder()
            .export(VisualizationFormat::Gnuplot, &gp_path)
            .expect("Operation failed");
        let gp = std::fs::read_to_string(&gp_path).expect("Operation failed");
        assert_ne!(gp, plotly);
        assert!(gp.contains("splot"));
        assert!(gp.contains("using 1:2:3"));

        let d3_path = temp_dir.join(format!("test_3d_d3_{pid}.json"));
        sample_3d_builder()
            .export(VisualizationFormat::D3Json, &d3_path)
            .expect("Operation failed");
        let d3 = std::fs::read_to_string(&d3_path).expect("Operation failed");
        assert_ne!(d3, plotly);
        let d3_value: serde_json::Value = serde_json::from_str(&d3).expect("Operation failed");
        assert_eq!(d3_value["format"], "d3-3d");
        assert!(
            d3_value.get("data").is_none(),
            "must not reuse Plotly's 'data' key"
        );

        let svg_path = temp_dir.join(format!("test_3d_svg_{pid}.svg"));
        sample_3d_builder()
            .export(VisualizationFormat::Svg, &svg_path)
            .expect("Operation failed");
        let svg = std::fs::read_to_string(&svg_path).expect("Operation failed");
        assert_ne!(svg, plotly);
        assert!(svg.trim_start().starts_with("<svg"));
        assert!(serde_json::from_str::<serde_json::Value>(&svg).is_err());

        let html_path = temp_dir.join(format!("test_3d_html_{pid}.html"));
        sample_3d_builder()
            .export(VisualizationFormat::Html, &html_path)
            .expect("Operation failed");
        let html = std::fs::read_to_string(&html_path).expect("Operation failed");
        assert_ne!(html, svg);
        assert!(html.trim_start().starts_with("<!DOCTYPE html>"));

        for p in [plotly_path, mpl_path, gp_path, d3_path, svg_path, html_path] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn test_3d_vegalite_and_bokeh_are_honest_unsupported_errors() {
        // Vega-Lite and Bokeh have no native 3D/z concept in their grammar or
        // model system (see `threed` module docs); the fix must surface an
        // explicit error for these two formats rather than silently writing
        // mislabeled Plotly output.
        let temp_dir = std::env::temp_dir();
        let pid = std::process::id();

        let vega_path = temp_dir.join(format!("test_3d_vega_{pid}.json"));
        let result = sample_3d_builder().export(VisualizationFormat::VegaLite, &vega_path);
        assert!(result.is_err());
        assert!(
            !vega_path.exists(),
            "must not silently write a mislabeled file for an unsupported 3D format"
        );

        let bokeh_path = temp_dir.join(format!("test_3d_bokeh_{pid}.json"));
        let result = sample_3d_builder().export(VisualizationFormat::BokehJson, &bokeh_path);
        assert!(result.is_err());
        assert!(!bokeh_path.exists());
    }
}

// Advanced Visualization Features

#[cfg(feature = "async")]
use futures::StreamExt;
#[cfg(feature = "async")]
use tokio::sync::mpsc;

/// Interactive visualization server for real-time updates
#[cfg(feature = "async")]
pub struct VisualizationServer {
    port: u16,
    update_channel: mpsc::Sender<PlotUpdate>,
}

#[cfg(feature = "async")]
#[derive(Debug, Clone)]
pub struct PlotUpdate {
    pub plot_id: String,
    pub data: DataSeries,
    pub action: UpdateAction,
}

#[cfg(feature = "async")]
#[derive(Debug, Clone)]
pub enum UpdateAction {
    Append,
    Replace,
    Remove,
}

#[cfg(feature = "async")]
impl VisualizationServer {
    /// Create a new visualization server
    pub async fn new(port: u16) -> Result<Self> {
        let (tx, mut rx) = mpsc::channel(100);

        // Spawn server task
        tokio::spawn(async move {
            // In a real implementation, this would start an HTTP server
            // that serves interactive visualizations and handles WebSocket
            // connections for real-time updates
            while let Some(_update) = rx.recv().await {
                // Process update
            }
        });

        Ok(Self {
            port,
            update_channel: tx,
        })
    }

    /// Send update to a plot
    pub async fn update_plot(&self, update: PlotUpdate) -> Result<()> {
        self.update_channel
            .send(update)
            .await
            .map_err(|_| IoError::Other("Failed to send update".to_string()))
    }

    /// Get server URL
    pub fn url(&self) -> String {
        format!("http://localhost:{}", self.port)
    }
}

/// 3D Visualization support
#[derive(Debug, Clone)]
pub struct DataSeries3D {
    pub name: Option<String>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub plot_type: PlotType3D,
    pub style: SeriesStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlotType3D {
    Scatter3D,
    Surface,
    Mesh3D,
    Line3D,
    Isosurface,
    Volume,
}

/// 3D Visualization builder
pub struct Visualization3DBuilder {
    data: Vec<DataSeries3D>,
    config: Plot3DConfig,
    metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plot3DConfig {
    pub title: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub x_axis: AxisConfig,
    pub y_axis: AxisConfig,
    pub z_axis: AxisConfig,
    pub camera: CameraConfig,
    pub lighting: LightingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    pub eye: [f64; 3],
    pub center: [f64; 3],
    pub up: [f64; 3],
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            eye: [1.25, 1.25, 1.25],
            center: [0.0, 0.0, 0.0],
            up: [0.0, 0.0, 1.0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingConfig {
    pub ambient: f64,
    pub diffuse: f64,
    pub specular: f64,
    pub roughness: f64,
}

impl Default for LightingConfig {
    fn default() -> Self {
        Self {
            ambient: 0.8,
            diffuse: 0.8,
            specular: 0.2,
            roughness: 0.5,
        }
    }
}

impl Default for Plot3DConfig {
    fn default() -> Self {
        Self {
            title: None,
            width: Some(800),
            height: Some(600),
            x_axis: AxisConfig::default(),
            y_axis: AxisConfig::default(),
            z_axis: AxisConfig::default(),
            camera: CameraConfig::default(),
            lighting: LightingConfig::default(),
        }
    }
}

impl Default for Visualization3DBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Visualization3DBuilder {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            config: Plot3DConfig::default(),
            metadata: Metadata::new(),
        }
    }

    /// Add 3D scatter plot
    pub fn add_scatter3d(mut self, x: &[f64], y: &[f64], z: &[f64], name: Option<&str>) -> Self {
        self.data.push(DataSeries3D {
            name: name.map(|s| s.to_string()),
            x: x.to_vec(),
            y: y.to_vec(),
            z: z.to_vec(),
            plot_type: PlotType3D::Scatter3D,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Add surface plot
    pub fn add_surface(
        mut self,
        x: &[f64],
        y: &[f64],
        z: &Array2<f64>,
        name: Option<&str>,
    ) -> Self {
        let z_flat: Vec<f64> = z.iter().cloned().collect();
        self.data.push(DataSeries3D {
            name: name.map(|s| s.to_string()),
            x: x.to_vec(),
            y: y.to_vec(),
            z: z_flat,
            plot_type: PlotType3D::Surface,
            style: SeriesStyle::default(),
        });
        self
    }

    /// Export 3D visualization
    pub fn export(self, format: VisualizationFormat, path: impl AsRef<Path>) -> Result<()> {
        // Implementation would convert 3D data to appropriate format
        let exporter = get_3d_exporter(format);
        exporter.export_3d(&self.data, &self.config, &self.metadata, path.as_ref())
    }
}

/// Animation support
#[derive(Debug, Clone)]
pub struct AnimationFrame {
    pub time: f64,
    pub data: DataSeries,
}

#[derive(Debug, Clone)]
pub struct AnimatedVisualization {
    pub frames: Vec<AnimationFrame>,
    pub config: AnimationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    pub duration: f64,
    pub fps: u32,
    pub loop_mode: LoopMode,
    pub transition: TransitionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoopMode {
    Once,
    Loop,
    PingPong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransitionType {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
}

/// Dashboard builder for composite visualizations
pub struct DashboardBuilder {
    plots: Vec<DashboardPlot>,
    layout: DashboardLayout,
    config: DashboardConfig,
}

#[derive(Debug, Clone)]
pub struct DashboardPlot {
    pub plot: VisualizationBuilder,
    pub position: GridPosition,
}

#[derive(Debug, Clone)]
pub struct GridPosition {
    pub row: usize,
    pub col: usize,
    pub row_span: usize,
    pub col_span: usize,
}

#[derive(Debug, Clone)]
pub struct DashboardLayout {
    pub rows: usize,
    pub cols: usize,
    pub spacing: f64,
}

#[derive(Debug, Clone)]
pub struct DashboardConfig {
    pub title: Option<String>,
    pub width: u32,
    pub height: u32,
    pub theme: Option<String>,
    pub auto_refresh: Option<u32>, // seconds
}

impl DashboardBuilder {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            plots: Vec::new(),
            layout: DashboardLayout {
                rows,
                cols,
                spacing: 10.0,
            },
            config: DashboardConfig {
                title: None,
                width: 1200,
                height: 800,
                theme: None,
                auto_refresh: None,
            },
        }
    }

    /// Add plot to dashboard
    pub fn add_plot(mut self, plot: VisualizationBuilder, row: usize, col: usize) -> Self {
        self.plots.push(DashboardPlot {
            plot,
            position: GridPosition {
                row,
                col,
                row_span: 1,
                col_span: 1,
            },
        });
        self
    }

    /// Export dashboard as HTML
    pub fn export_html(self, path: impl AsRef<Path>) -> Result<()> {
        let html = self.generate_html()?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(html.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }

    fn generate_html(&self) -> Result<String> {
        let mut html = String::from(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat({cols}, 1fr);
            grid-template-rows: repeat({rows}, 1fr);
            gap: {spacing}px;
            width: {width}px;
            height: {height}px;
        }
        .plot-container {
            width: 100%;
            height: 100%;
        }
    </style>
</head>
<body>
    <div class="dashboard-grid">
"#,
        );

        // Add plots
        for (i, dashboard_plot) in self.plots.iter().enumerate() {
            let plot_data = dashboard_plot
                .plot
                .clone()
                .to_string(VisualizationFormat::PlotlyJson)?;

            html.push_str(&format!(
                r#"
        <div class="plot-container" style="grid-row: {}; grid-column: {};">
            <div id="plot{}" style="width: 100%; height: 100%;"></div>
            <script>
                Plotly.newPlot('plot{}', {});
            </script>
        </div>
"#,
                dashboard_plot.position.row + 1,
                dashboard_plot.position.col + 1,
                i,
                i,
                plot_data
            ));
        }

        html.push_str(
            r#"
    </div>
</body>
</html>
"#,
        );

        Ok(html
            .replace("{cols}", &self.layout.cols.to_string())
            .replace("{rows}", &self.layout.rows.to_string())
            .replace("{spacing}", &self.layout.spacing.to_string())
            .replace("{width}", &self.config.width.to_string())
            .replace("{height}", &self.config.height.to_string()))
    }
}

/// D3.js JSON exporter.
///
/// Emits a JSON document in the array-of-records shape D3.js data joins are
/// idiomatically written against (`selection.data(rows)`), rather than
/// Plotly's parallel-array trace schema -- e.g. `{"series": [{"values":
/// [{"x":.., "y":..}, ...]}]}` instead of Plotly's `{"data": [{"x": [..],
/// "y": [..]}]}`.
struct D3Exporter;

impl VisualizationExporter for D3Exporter {
    fn export(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let json_str = self.to_string(data, config, metadata)?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(json_str.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }

    fn to_string(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        _metadata: &Metadata,
    ) -> Result<String> {
        fn plot_type_name(plot_type: &PlotType) -> &'static str {
            match plot_type {
                PlotType::Line => "line",
                PlotType::Scatter => "scatter",
                PlotType::Bar => "bar",
                PlotType::Histogram => "histogram",
                PlotType::Heatmap => "heatmap",
                PlotType::Surface => "surface",
                PlotType::Contour => "contour",
                PlotType::Box => "box",
                PlotType::Violin => "violin",
                PlotType::Pie => "pie",
                PlotType::Area => "area",
                PlotType::Stream => "stream",
            }
        }

        let series: Vec<serde_json::Value> = data
            .iter()
            .map(|series| {
                let values: Vec<serde_json::Value> = match &series.x {
                    Some(xs) => xs
                        .iter()
                        .zip(series.y.iter())
                        .map(|(&x, &y)| serde_json::json!({"x": x, "y": y}))
                        .collect(),
                    None => series
                        .y
                        .iter()
                        .enumerate()
                        .map(|(i, &y)| serde_json::json!({"x": i as f64, "y": y}))
                        .collect(),
                };
                serde_json::json!({
                    "name": series.name,
                    "type": plot_type_name(&series.plot_type),
                    "values": values,
                    "style": {
                        "color": series.style.color,
                        "opacity": series.style.opacity,
                    },
                })
            })
            .collect();

        let doc = serde_json::json!({
            "format": "d3",
            "title": config.title,
            "width": config.width,
            "height": config.height,
            "series": series,
            "axes": {
                "x": { "title": config.x_axis.title, "grid": config.x_axis.grid },
                "y": { "title": config.y_axis.title, "grid": config.y_axis.grid },
            },
        });

        serde_json::to_string_pretty(&doc).map_err(|e| IoError::SerializationError(e.to_string()))
    }
}

/// Bokeh JSON exporter.
///
/// Emits a JSON document modelled on BokehJS's actual document format: a
/// `roots.references` list of typed model objects (`ColumnDataSource`,
/// glyph models like `Line`/`Circle`/`VBar`, `GlyphRenderer`, and the root
/// `Plot`) rather than Plotly's `{"data": [...], "layout": {...}}` trace
/// schema.
struct BokehExporter;

impl VisualizationExporter for BokehExporter {
    fn export(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let json = self.to_string(data, config, metadata)?;
        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(json.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }

    fn to_string(
        &self,
        data: &[DataSeries],
        config: &PlotConfig,
        _metadata: &Metadata,
    ) -> Result<String> {
        let mut references = Vec::new();
        let mut renderer_ids = Vec::new();

        for (i, series) in data.iter().enumerate() {
            let source_id = format!("cds_{i}");
            let glyph_id = format!("glyph_{i}");
            let renderer_id = format!("renderer_{i}");

            let xs: Vec<f64> = series
                .x
                .clone()
                .unwrap_or_else(|| (0..series.y.len()).map(|k| k as f64).collect());
            references.push(serde_json::json!({
                "id": source_id,
                "type": "ColumnDataSource",
                "attributes": { "data": { "x": xs, "y": series.y } },
            }));

            let glyph_type = match series.plot_type {
                PlotType::Line => "Line",
                PlotType::Scatter => "Circle",
                PlotType::Bar | PlotType::Histogram => "VBar",
                PlotType::Heatmap => "Image",
                _ => "Line",
            };
            references.push(serde_json::json!({
                "id": glyph_id,
                "type": glyph_type,
                "attributes": {
                    "line_color": series.style.color,
                    "fill_color": series.style.color,
                },
            }));
            references.push(serde_json::json!({
                "id": renderer_id,
                "type": "GlyphRenderer",
                "attributes": {
                    "data_source": source_id,
                    "glyph": glyph_id,
                    "name": series.name,
                },
            }));
            renderer_ids.push(serde_json::Value::String(renderer_id));
        }

        references.push(serde_json::json!({
            "id": "fig_0",
            "type": "Plot",
            "attributes": {
                "title": config.title,
                "width": config.width,
                "height": config.height,
                "renderers": renderer_ids,
            },
        }));

        let doc = serde_json::json!({
            "target_id": "bokeh_plot",
            "root_id": "fig_0",
            "doc": {
                "version": "3.4.0",
                "title": config.title,
                "roots": {
                    "root_ids": ["fig_0"],
                    "references": references,
                },
            },
        });

        serde_json::to_string_pretty(&doc).map_err(|e| IoError::SerializationError(e.to_string()))
    }
}

/// Get appropriate 3D exporter
#[allow(dead_code)]
fn get_3d_exporter(format: VisualizationFormat) -> Box<dyn Visualization3DExporter> {
    match format {
        VisualizationFormat::PlotlyJson => Box::new(Plotly3DExporter),
        VisualizationFormat::MatplotlibPython => Box::new(threed::Matplotlib3DExporter),
        VisualizationFormat::Gnuplot => Box::new(threed::Gnuplot3DExporter),
        VisualizationFormat::D3Json => Box::new(threed::D3ThreeDExporter),
        VisualizationFormat::VegaLite => Box::new(threed::VegaLite3DUnsupported),
        VisualizationFormat::BokehJson => Box::new(threed::Bokeh3DUnsupported),
        VisualizationFormat::Svg => Box::new(threed::Svg3DExporter),
        VisualizationFormat::Html => Box::new(threed::Html3DExporter),
    }
}

/// Trait for 3D visualization exporters
trait Visualization3DExporter {
    fn export_3d(
        &self,
        data: &[DataSeries3D],
        config: &Plot3DConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()>;
}

/// Plotly 3D exporter
struct Plotly3DExporter;

impl Visualization3DExporter for Plotly3DExporter {
    fn export_3d(
        &self,
        data: &[DataSeries3D],
        config: &Plot3DConfig,
        metadata: &Metadata,
        path: &Path,
    ) -> Result<()> {
        let mut traces = Vec::new();

        for series in data {
            let trace = match series.plot_type {
                PlotType3D::Scatter3D => {
                    serde_json::json!({
                        "type": "scatter3d",
                        "mode": "markers",
                        "name": series.name,
                        "x": series.x,
                        "y": series.y,
                        "z": series.z,
                        "marker": {
                            "size": series.style.size.unwrap_or(5.0),
                            "color": series.style.color,
                        }
                    })
                }
                PlotType3D::Surface => {
                    serde_json::json!({
                        "type": "surface",
                        "name": series.name,
                        "x": series.x,
                        "y": series.y,
                        "z": series.z,
                    })
                }
                _ => continue,
            };
            traces.push(trace);
        }

        let layout = serde_json::json!({
            "title": config.title,
            "width": config.width,
            "height": config.height,
            "scene": {
                "xaxis": {"title": config.x_axis.title},
                "yaxis": {"title": config.y_axis.title},
                "zaxis": {"title": config.z_axis.title},
                "camera": {
                    "eye": {"x": config.camera.eye[0], "y": config.camera.eye[1], "z": config.camera.eye[2]},
                    "center": {"x": config.camera.center[0], "y": config.camera.center[1], "z": config.camera.center[2]},
                    "up": {"x": config.camera.up[0], "y": config.camera.up[1], "z": config.camera.up[2]},
                }
            }
        });

        let plot_data = serde_json::json!({
            "data": traces,
            "layout": layout,
        });

        let json_str = serde_json::to_string_pretty(&plot_data)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;

        let mut file = File::create(path).map_err(IoError::Io)?;
        file.write_all(json_str.as_bytes()).map_err(IoError::Io)?;
        Ok(())
    }
}

/// Integration with external visualization services
pub mod external {
    use super::*;

    /// Plotly cloud integration
    pub struct PlotlyCloud {
        api_key: String,
        username: String,
    }

    impl PlotlyCloud {
        pub fn new(_apikey: String, username: String) -> Self {
            Self {
                api_key: _apikey,
                username,
            }
        }

        /// Upload visualization to Plotly cloud
        #[cfg(feature = "reqwest")]
        pub fn upload(&self, plotdata: &str, filename: &str) -> Result<String> {
            // Implementation would use reqwest to upload to Plotly API
            Ok(format!("https://plot.ly/~{}/{}", self.username, filename))
        }
    }

    /// Jupyter notebook integration
    pub struct JupyterIntegration;

    impl JupyterIntegration {
        /// Generate notebook cell with visualization
        pub fn create_cell(viz: &VisualizationBuilder) -> serde_json::Value {
            serde_json::json!({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generated visualization\n",
                    "import plotly.graph_objects as go\n",
                    "# ... visualization code ..."
                ]
            })
        }
    }
}
