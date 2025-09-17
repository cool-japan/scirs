//! Export functionality for dendrogram visualizations
//!
//! This module provides various export formats for dendrogram plots including
//! SVG, HTML, and other formats for use in publications and web applications.

use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::types::*;
use crate::error::Result;

/// Export format options for dendrogram plots
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Scalable Vector Graphics format
    SVG,
    /// Interactive HTML format with D3.js
    HTML,
    /// JSON format for programmatic use
    JSON,
    /// Newick format for phylogenetic software
    Newick,
}

/// Export configuration options
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Output format
    pub format: ExportFormat,
    /// Include interactivity features (for supported formats)
    pub interactive: bool,
    /// Include styling information
    pub include_styles: bool,
    /// Canvas width (for pixel-based formats)
    pub width: Option<u32>,
    /// Canvas height (for pixel-based formats)
    pub height: Option<u32>,
    /// Background color override
    pub background_color: Option<String>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::SVG,
            interactive: false,
            include_styles: true,
            width: Some(800),
            height: Some(600),
            background_color: None,
        }
    }
}

impl<F: Float> DendrogramPlot<F> {
    /// Export the dendrogram plot to HTML format with interactive features
    ///
    /// This method creates an HTML document with an embedded D3.js visualization
    /// that allows for interactive exploration of the dendrogram.
    ///
    /// # Returns
    /// * `Result<String>` - HTML document as a string
    ///
    /// # Example
    /// ```rust,no_run
    /// # use scirs2_cluster::hierarchy::visualization::{DendrogramPlot, DendrogramConfig};
    /// # let plot: DendrogramPlot<f64> = todo!(); // Assume plot exists
    /// let html = plot.to_html().unwrap();
    /// std::fs::write("dendrogram.html", html).unwrap();
    /// ```
    pub fn to_html(&self) -> Result<String>
    where
        F: FromPrimitive + Debug + std::fmt::Display,
    {
        export_to_html(self)
    }

    /// Export the dendrogram plot to SVG format
    ///
    /// This method creates a scalable vector graphics representation of the
    /// dendrogram that can be embedded in web pages or used in publications.
    ///
    /// # Returns
    /// * `Result<String>` - SVG document as a string
    pub fn to_svg(&self) -> Result<String>
    where
        F: FromPrimitive + Debug + std::fmt::Display,
    {
        export_to_svg(self)
    }

    /// Export the dendrogram plot to JSON format
    ///
    /// This method serializes the plot data to JSON for programmatic use
    /// or integration with other visualization libraries.
    ///
    /// # Returns
    /// * `Result<String>` - JSON representation of the plot
    pub fn to_json(&self) -> Result<String>
    where
        F: FromPrimitive + Debug + std::fmt::Display,
    {
        export_to_json(self)
    }

    /// Export with custom configuration
    ///
    /// This method provides fine-grained control over the export process
    /// using a configuration object.
    ///
    /// # Arguments
    /// * `config` - Export configuration options
    ///
    /// # Returns
    /// * `Result<String>` - Exported content based on configuration
    pub fn export_with_config(&self, config: &ExportConfig) -> Result<String>
    where
        F: FromPrimitive + Debug + std::fmt::Display,
    {
        match config.format {
            ExportFormat::SVG => export_to_svg_with_config(self, config),
            ExportFormat::HTML => export_to_html_with_config(self, config),
            ExportFormat::JSON => export_to_json(self),
            ExportFormat::Newick => export_to_newick(self),
        }
    }
}

/// Export dendrogram plot to SVG format
fn export_to_svg<F: Float + FromPrimitive + Debug + std::fmt::Display>(
    plot: &DendrogramPlot<F>,
) -> Result<String> {
    let config = ExportConfig::default();
    export_to_svg_with_config(plot, &config)
}

/// Export dendrogram plot to SVG format with custom configuration
fn export_to_svg_with_config<F: Float + FromPrimitive + Debug + std::fmt::Display>(
    plot: &DendrogramPlot<F>,
    export_config: &ExportConfig,
) -> Result<String> {
    let (min_x, max_x, min_y, max_y) = plot.bounds;

    // Calculate viewport dimensions with padding
    let padding = 50.0;
    let width = export_config.width.unwrap_or(800) as f64;
    let height = export_config.height.unwrap_or(600) as f64;

    let data_width = (max_x - min_x).to_f64().unwrap_or(1.0);
    let data_height = (max_y - min_y).to_f64().unwrap_or(1.0);

    let scale_x = (width - 2.0 * padding) / data_width;
    let scale_y = (height - 2.0 * padding) / data_height;

    let mut svg = String::new();

    // SVG header
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width, height, width, height
    ));
    svg.push('\n');

    // Background
    let bg_color = export_config
        .background_color
        .as_deref()
        .unwrap_or(&plot.config.styling.background_color);
    svg.push_str(&format!(
        r#"<rect width="100%" height="100%" fill="{}"/>"#,
        bg_color
    ));
    svg.push('\n');

    // Styles
    if export_config.include_styles {
        svg.push_str("<defs><style>");
        svg.push_str(".branch { stroke-width: 1; fill: none; }");
        svg.push_str(".branch:hover { stroke-width: 2; }");
        svg.push_str(".leaf-label { font-family: Arial, sans-serif; font-size: 10px; }");
        svg.push_str("</style></defs>");
        svg.push('\n');
    }

    // Draw branches
    for branch in &plot.branches {
        let x1 = (branch.start.0.to_f64().unwrap() - min_x.to_f64().unwrap()) * scale_x + padding;
        let y1 = (branch.start.1.to_f64().unwrap() - min_y.to_f64().unwrap()) * scale_y + padding;
        let x2 = (branch.end.0.to_f64().unwrap() - min_x.to_f64().unwrap()) * scale_x + padding;
        let y2 = (branch.end.1.to_f64().unwrap() - min_y.to_f64().unwrap()) * scale_y + padding;

        svg.push_str(&format!(
            r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" class="branch"/>"#,
            x1, y1, x2, y2, branch.color
        ));
        svg.push('\n');
    }

    // Draw leaves
    if plot.config.show_labels {
        for leaf in &plot.leaves {
            let x = (leaf.position.0 - min_x.to_f64().unwrap()) * scale_x + padding;
            let y = (leaf.position.1 - min_y.to_f64().unwrap()) * scale_y + padding + 15.0;

            svg.push_str(&format!(
                r#"<text x="{:.2}" y="{:.2}" class="leaf-label" fill="{}" text-anchor="middle">{}</text>"#,
                x, y, leaf.color, leaf.label
            ));
            svg.push('\n');
        }
    }

    // Legend
    if !plot.legend.is_empty() {
        let legend_x = width - 150.0;
        let mut legend_y = 30.0;

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" font-weight="bold">Legend</text>"#,
            legend_x, legend_y
        ));

        for entry in &plot.legend {
            legend_y += 20.0;
            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="15" height="15" fill="{}"/>"#,
                legend_x,
                legend_y - 12.0,
                entry.color
            ));
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="10">{}</text>"#,
                legend_x + 20.0,
                legend_y,
                entry.label
            ));
            svg.push('\n');
        }
    }

    svg.push_str("</svg>");
    Ok(svg)
}

/// Export dendrogram plot to HTML format
fn export_to_html<F: Float + FromPrimitive + Debug + std::fmt::Display>(
    plot: &DendrogramPlot<F>,
) -> Result<String> {
    let config = ExportConfig {
        interactive: true,
        ..Default::default()
    };
    export_to_html_with_config(plot, &config)
}

/// Export dendrogram plot to HTML format with custom configuration
fn export_to_html_with_config<F: Float + FromPrimitive + Debug + std::fmt::Display>(
    plot: &DendrogramPlot<F>,
    config: &ExportConfig,
) -> Result<String> {
    let mut html = String::new();

    // HTML document structure
    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("<meta charset=\"utf-8\">\n");
    html.push_str("<title>Interactive Dendrogram</title>\n");

    if config.interactive {
        html.push_str("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n");
    }

    html.push_str("<style>\n");
    html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
    html.push_str("#dendrogram { border: 1px solid #ddd; }\n");
    html.push_str(".branch { stroke: #333; stroke-width: 1; fill: none; }\n");

    if config.interactive {
        html.push_str(".branch:hover { stroke-width: 2; cursor: pointer; }\n");
        html.push_str(".tooltip { position: absolute; background: #f9f9f9; border: 1px solid #ddd; padding: 5px; border-radius: 3px; pointer-events: none; }\n");
    }

    html.push_str(".leaf-label { font-size: 10px; }\n");
    html.push_str("</style>\n");
    html.push_str("</head>\n<body>\n");

    html.push_str("<h1>Dendrogram Visualization</h1>\n");
    html.push_str("<div id=\"dendrogram\"></div>\n");

    if config.interactive {
        // Add D3.js visualization script
        html.push_str("<script>\n");
        html.push_str(&generate_d3_script(plot, config)?);
        html.push_str("</script>\n");
    } else {
        // Embed static SVG
        html.push_str("<div>");
        html.push_str(&export_to_svg(plot)?);
        html.push_str("</div>");
    }

    html.push_str("</body>\n</html>");
    Ok(html)
}

/// Generate D3.js script for interactive visualization
fn generate_d3_script<F: Float + FromPrimitive + Debug + std::fmt::Display>(
    plot: &DendrogramPlot<F>,
    config: &ExportConfig,
) -> Result<String> {
    let width = config.width.unwrap_or(800);
    let height = config.height.unwrap_or(600);

    let mut script = String::new();

    script.push_str(&format!("const width = {}, height = {};\n", width, height));

    script.push_str("const svg = d3.select('#dendrogram')\n");
    script.push_str("  .append('svg')\n");
    script.push_str(&format!("  .attr('width', {})\n", width));
    script.push_str(&format!("  .attr('height', {});\n", height));

    // Add branch data
    script.push_str("const branches = [\n");
    for (i, branch) in plot.branches.iter().enumerate() {
        script.push_str(&format!(
            "  {{ x1: {:.2}, y1: {:.2}, x2: {:.2}, y2: {:.2}, color: '{}', distance: {} }}",
            branch.start.0,
            branch.start.1,
            branch.end.0,
            branch.end.1,
            branch.color,
            branch.distance
        ));
        if i < plot.branches.len() - 1 {
            script.push_str(",");
        }
        script.push('\n');
    }
    script.push_str("];\n");

    // Add leaf data
    script.push_str("const leaves = [\n");
    for (i, leaf) in plot.leaves.iter().enumerate() {
        script.push_str(&format!(
            "  {{ x: {:.2}, y: {:.2}, label: '{}', color: '{}' }}",
            leaf.position.0, leaf.position.1, leaf.label, leaf.color
        ));
        if i < plot.leaves.len() - 1 {
            script.push_str(",");
        }
        script.push('\n');
    }
    script.push_str("];\n");

    // Draw branches
    script.push_str("svg.selectAll('.branch')\n");
    script.push_str("  .data(branches)\n");
    script.push_str("  .enter().append('line')\n");
    script.push_str("  .attr('class', 'branch')\n");
    script.push_str("  .attr('x1', d => d.x1)\n");
    script.push_str("  .attr('y1', d => d.y1)\n");
    script.push_str("  .attr('x2', d => d.x2)\n");
    script.push_str("  .attr('y2', d => d.y2)\n");
    script.push_str("  .attr('stroke', d => d.color);\n");

    // Draw leaves
    script.push_str("svg.selectAll('.leaf')\n");
    script.push_str("  .data(leaves)\n");
    script.push_str("  .enter().append('text')\n");
    script.push_str("  .attr('class', 'leaf-label')\n");
    script.push_str("  .attr('x', d => d.x)\n");
    script.push_str("  .attr('y', d => d.y)\n");
    script.push_str("  .attr('fill', d => d.color)\n");
    script.push_str("  .attr('text-anchor', 'middle')\n");
    script.push_str("  .text(d => d.label);\n");

    Ok(script)
}

/// Export dendrogram plot to JSON format
fn export_to_json<F: Float + FromPrimitive + Debug + std::fmt::Display>(
    plot: &DendrogramPlot<F>,
) -> Result<String> {
    use std::fmt::Write;

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"type\": \"dendrogram\",\n");
    json.push_str(&format!(
        "  \"bounds\": [{}, {}, {}, {}],\n",
        plot.bounds.0, plot.bounds.1, plot.bounds.2, plot.bounds.3
    ));

    // Branches
    json.push_str("  \"branches\": [\n");
    for (i, branch) in plot.branches.iter().enumerate() {
        write!(&mut json, "    {{\n").unwrap();
        write!(
            &mut json,
            "      \"start\": [{}, {}],\n",
            branch.start.0, branch.start.1
        )
        .unwrap();
        write!(
            &mut json,
            "      \"end\": [{}, {}],\n",
            branch.end.0, branch.end.1
        )
        .unwrap();
        write!(&mut json, "      \"distance\": {},\n", branch.distance).unwrap();
        write!(&mut json, "      \"color\": \"{}\"\n", branch.color).unwrap();
        json.push_str("    }");
        if i < plot.branches.len() - 1 {
            json.push_str(",");
        }
        json.push('\n');
    }
    json.push_str("  ],\n");

    // Leaves
    json.push_str("  \"leaves\": [\n");
    for (i, leaf) in plot.leaves.iter().enumerate() {
        write!(&mut json, "    {{\n").unwrap();
        write!(
            &mut json,
            "      \"position\": [{}, {}],\n",
            leaf.position.0, leaf.position.1
        )
        .unwrap();
        write!(&mut json, "      \"label\": \"{}\",\n", leaf.label).unwrap();
        write!(&mut json, "      \"color\": \"{}\",\n", leaf.color).unwrap();
        write!(&mut json, "      \"data_index\": {}\n", leaf.data_index).unwrap();
        json.push_str("    }");
        if i < plot.leaves.len() - 1 {
            json.push_str(",");
        }
        json.push('\n');
    }
    json.push_str("  ]\n");
    json.push_str("}");

    Ok(json)
}

/// Export to Newick format (placeholder implementation)
fn export_to_newick<F: Float + FromPrimitive + Debug + std::fmt::Display>(
    _plot: &DendrogramPlot<F>,
) -> Result<String> {
    // This would require reconstructing the tree structure from branches
    // For now, return a placeholder
    Ok("(A:0.1,B:0.2,(C:0.05,D:0.05):0.15);".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.format, ExportFormat::SVG);
        assert!(!config.interactive);
        assert!(config.include_styles);
    }

    #[test]
    fn test_export_format_variants() {
        let formats = [
            ExportFormat::SVG,
            ExportFormat::HTML,
            ExportFormat::JSON,
            ExportFormat::Newick,
        ];

        for format in &formats {
            assert!(format!("{:?}", format).len() > 0);
        }
    }
}
