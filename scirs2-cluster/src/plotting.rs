//! Native plotting capabilities for clustering results
//!
//! This module provides native plotting implementations using popular Rust visualization
//! libraries like plotters and egui. It bridges the visualization data structures with
//! actual plotting backends to create publication-ready plots.

use crate::error::{ClusteringError, Result};
use crate::hierarchy::visualization::{create_dendrogramplot, DendrogramConfig, DendrogramPlot};
use crate::visualization::{ScatterPlot2D, ScatterPlot3D, VisualizationConfig};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use std::path::Path;

#[cfg(feature = "egui")]
use egui::*;
#[cfg(feature = "plotters")]
use plotters::prelude::*;

/// Plot output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotFormat {
    /// PNG image format
    PNG,
    /// SVG vector format  
    SVG,
    /// PDF format (if supported)
    PDF,
    /// Interactive HTML
    HTML,
}

/// Plot output configuration
#[derive(Debug, Clone)]
pub struct PlotOutput {
    /// Output format
    pub format: PlotFormat,
    /// Output dimensions (width, height) in pixels
    pub dimensions: (u32, u32),
    /// DPI for raster formats
    pub dpi: u32,
    /// Background color (hex format)
    pub background_color: String,
    /// Whether to show grid
    pub show_grid: bool,
    /// Whether to show axes
    pub show_axes: bool,
    /// Plot title
    pub title: Option<String>,
    /// Axis labels (x, y, z)
    pub axis_labels: (Option<String>, Option<String>, Option<String>),
}

impl Default for PlotOutput {
    fn default() -> Self {
        Self {
            format: PlotFormat::PNG,
            dimensions: (800, 600),
            dpi: 300,
            background_color: "#FFFFFF".to_string(),
            show_grid: true,
            show_axes: true,
            title: None,
            axis_labels: (None, None, None),
        }
    }
}

/// Native dendrogram plot using plotters
#[cfg(feature = "plotters")]
#[allow(dead_code)]
pub fn plot_dendrogram<P: AsRef<Path>>(
    dendrogram_plot: &DendrogramPlot<f64>,
    output_path: P,
    output_config: &PlotOutput,
) -> Result<()> {
    let path = output_path.as_ref();

    match output_config.format {
        PlotFormat::PNG => plot_dendrogram_png(dendrogram_plot, path, output_config),
        PlotFormat::SVG => plot_dendrogram_svg(dendrogram_plot, path, output_config),
        _ => Err(ClusteringError::ComputationError(
            "Unsupported output format for plotters dendrogram".to_string(),
        )),
    }
}

#[cfg(feature = "plotters")]
#[allow(dead_code)]
fn plot_dendrogram_png<P: AsRef<Path>>(
    dendrogram_plot: &DendrogramPlot<f64>,
    output_path: P,
    output_config: &PlotOutput,
) -> Result<()> {
    let root = BitMapBackend::new(&output_path, output_config.dimensions).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        ClusteringError::ComputationError(format!("Failed to initialize plot: {}", e))
    })?;

    let bounds = dendrogram_plot.bounds;
    let margin = 0.1;
    let x_range = bounds.1 - bounds.0;
    let y_range = bounds.3 - bounds.2;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            output_config.title.as_deref().unwrap_or("Dendrogram"),
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (bounds.0 - margin * x_range)..(bounds.1 + margin * x_range),
            (bounds.2 - margin * y_range)..(bounds.3 + margin * y_range),
        )
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to build chart: {}", e)))?;

    // Configure chart
    chart
        .configure_mesh()
        .x_desc(
            output_config
                .axis_labels
                .0
                .as_deref()
                .unwrap_or("Sample Index"),
        )
        .y_desc(output_config.axis_labels.1.as_deref().unwrap_or("Distance"))
        .draw()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to draw mesh: {}", e)))?;

    // Draw branches
    for (i, branch) in dendrogram_plot.branches.iter().enumerate() {
        let colorhex = &dendrogram_plot.colors[i];
        let color = parsehex_color_plotters(colorhex).unwrap_or(BLACK);

        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![
                    (branch.start.0, branch.start.1),
                    (branch.end.0, branch.end.1),
                ],
                color.stroke_width(2),
            )))
            .map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to draw branches: {}", e))
            })?;
    }

    // Draw leaf labels
    for leaf in &dendrogram_plot.leaves {
        let text_style = ("sans-serif", 12).into_font().color(&BLACK);

        chart
            .draw_series(std::iter::once(Text::new(
                leaf.label.clone(),
                (leaf.position.0, leaf.position.1),
                text_style,
            )))
            .map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to draw labels: {}", e))
            })?;
    }

    root.present()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to save plot: {}", e)))?;

    Ok(())
}

#[cfg(feature = "plotters")]
#[allow(dead_code)]
fn plot_dendrogram_svg<P: AsRef<Path>>(
    dendrogram_plot: &DendrogramPlot<f64>,
    output_path: P,
    output_config: &PlotOutput,
) -> Result<()> {
    let root = SVGBackend::new(&output_path, output_config.dimensions).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        ClusteringError::ComputationError(format!("Failed to initialize plot: {}", e))
    })?;

    let bounds = dendrogram_plot.bounds;
    let margin = 0.1;
    let x_range = bounds.1 - bounds.0;
    let y_range = bounds.3 - bounds.2;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            output_config.title.as_deref().unwrap_or("Dendrogram"),
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (bounds.0 - margin * x_range)..(bounds.1 + margin * x_range),
            (bounds.2 - margin * y_range)..(bounds.3 + margin * y_range),
        )
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to build chart: {}", e)))?;

    // Configure chart
    chart
        .configure_mesh()
        .x_desc(
            output_config
                .axis_labels
                .0
                .as_deref()
                .unwrap_or("Sample Index"),
        )
        .y_desc(output_config.axis_labels.1.as_deref().unwrap_or("Distance"))
        .draw()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to draw mesh: {}", e)))?;

    // Draw branches
    for (i, branch) in dendrogram_plot.branches.iter().enumerate() {
        let colorhex = &dendrogram_plot.colors[i];
        let color = parsehex_color_plotters(colorhex).unwrap_or(BLACK);

        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![
                    (branch.start.0, branch.start.1),
                    (branch.end.0, branch.end.1),
                ],
                color.stroke_width(2),
            )))
            .map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to draw branches: {}", e))
            })?;
    }

    // Draw leaf labels
    for leaf in &dendrogram_plot.leaves {
        let text_style = ("sans-serif", 12).into_font().color(&BLACK);

        chart
            .draw_series(std::iter::once(Text::new(
                leaf.label.clone(),
                (leaf.position.0, leaf.position.1),
                text_style,
            )))
            .map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to draw labels: {}", e))
            })?;
    }

    root.present()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to save plot: {}", e)))?;

    Ok(())
}

/// Native 2D scatter plot using plotters
#[cfg(feature = "plotters")]
#[allow(dead_code)]
pub fn plot_scatter_2d<P: AsRef<Path>>(
    scatter_plot: &ScatterPlot2D,
    output_path: P,
    output_config: &PlotOutput,
) -> Result<()> {
    let path = output_path.as_ref();

    match output_config.format {
        PlotFormat::PNG => plot_scatter_2d_png(scatter_plot, path, output_config),
        PlotFormat::SVG => plot_scatter_2d_svg(scatter_plot, path, output_config),
        _ => Err(ClusteringError::ComputationError(
            "Unsupported output format for plotters backend".to_string(),
        )),
    }
}

#[cfg(feature = "plotters")]
#[allow(dead_code)]
fn plot_scatter_2d_png<P: AsRef<Path>>(
    scatter_plot: &ScatterPlot2D,
    output_path: P,
    output_config: &PlotOutput,
) -> Result<()> {
    let root = BitMapBackend::new(&output_path, output_config.dimensions).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        ClusteringError::ComputationError(format!("Failed to initialize plot: {}", e))
    })?;

    let (min_x, max_x, min_y, max_y) = scatter_plot.bounds;
    let margin = 0.1;
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            output_config
                .title
                .as_deref()
                .unwrap_or("Cluster Visualization"),
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (min_x - margin * x_range)..(max_x + margin * x_range),
            (min_y - margin * y_range)..(max_y + margin * y_range),
        )
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to build chart: {}", e)))?;

    // Configure chart
    chart
        .configure_mesh()
        .x_desc(output_config.axis_labels.0.as_deref().unwrap_or("X"))
        .y_desc(output_config.axis_labels.1.as_deref().unwrap_or("Y"))
        .draw()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to draw mesh: {}", e)))?;

    // Plot points
    for (i, point) in scatter_plot.points.rows().into_iter().enumerate() {
        let x = point[0];
        let y = point[1];
        let colorhex = &scatter_plot.colors[i];
        let size = scatter_plot.sizes[i] as i32;

        // Parse hex color
        let color = parsehex_color_plotters(colorhex).unwrap_or(RED);

        chart
            .draw_series(std::iter::once(Circle::new((x, y), size, color.filled())))
            .map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to draw points: {}", e))
            })?;
    }

    // Plot centroids if available
    if let Some(centroids) = &scatter_plot.centroids {
        for centroid in centroids.rows() {
            let x = centroid[0];
            let y = centroid[1];

            chart
                .draw_series(std::iter::once(Cross::new(
                    (x, y),
                    8,
                    BLACK.stroke_width(3),
                )))
                .map_err(|e| {
                    ClusteringError::ComputationError(format!("Failed to draw centroids: {}", e))
                })?;
        }
    }

    root.present()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to save plot: {}", e)))?;

    Ok(())
}

#[cfg(feature = "plotters")]
#[allow(dead_code)]
fn plot_scatter_2d_svg<P: AsRef<Path>>(
    scatter_plot: &ScatterPlot2D,
    output_path: P,
    output_config: &PlotOutput,
) -> Result<()> {
    let root = SVGBackend::new(&output_path, output_config.dimensions).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        ClusteringError::ComputationError(format!("Failed to initialize plot: {}", e))
    })?;

    let (min_x, max_x, min_y, max_y) = scatter_plot.bounds;
    let margin = 0.1;
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            output_config
                .title
                .as_deref()
                .unwrap_or("Cluster Visualization"),
            ("sans-serif", 30),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (min_x - margin * x_range)..(max_x + margin * x_range),
            (min_y - margin * y_range)..(max_y + margin * y_range),
        )
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to build chart: {}", e)))?;

    // Configure chart
    chart
        .configure_mesh()
        .x_desc(output_config.axis_labels.0.as_deref().unwrap_or("X"))
        .y_desc(output_config.axis_labels.1.as_deref().unwrap_or("Y"))
        .draw()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to draw mesh: {}", e)))?;

    // Plot points
    for (i, point) in scatter_plot.points.rows().into_iter().enumerate() {
        let x = point[0];
        let y = point[1];
        let colorhex = &scatter_plot.colors[i];
        let size = scatter_plot.sizes[i] as i32;

        // Parse hex color
        let color = parsehex_color_plotters(colorhex).unwrap_or(RED);

        chart
            .draw_series(std::iter::once(Circle::new((x, y), size, color.filled())))
            .map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to draw points: {}", e))
            })?;
    }

    // Plot centroids if available
    if let Some(centroids) = &scatter_plot.centroids {
        for centroid in centroids.rows() {
            let x = centroid[0];
            let y = centroid[1];

            chart
                .draw_series(std::iter::once(Cross::new(
                    (x, y),
                    8,
                    BLACK.stroke_width(3),
                )))
                .map_err(|e| {
                    ClusteringError::ComputationError(format!("Failed to draw centroids: {}", e))
                })?;
        }
    }

    root.present()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to save plot: {}", e)))?;

    Ok(())
}

/// Interactive clustering visualization using egui
#[cfg(feature = "egui")]
pub struct InteractiveClusteringApp {
    /// Current scatter plot data
    pub scatter_plot_2d: Option<ScatterPlot2D>,
    /// Visualization configuration
    pub config: VisualizationConfig,
    /// Show/hide elements
    pub show_centroids: bool,
    pub show_boundaries: bool,
    pub show_legend: bool,
    /// Zoom and pan state
    pub zoom: f32,
    pub pan_offset: (f32, f32),
    /// Selected cluster (for highlighting)
    pub selected_cluster: Option<i32>,
}

#[cfg(feature = "egui")]
impl Default for InteractiveClusteringApp {
    fn default() -> Self {
        Self {
            scatter_plot_2d: None,
            config: VisualizationConfig::default(),
            show_centroids: true,
            show_boundaries: false,
            show_legend: true,
            zoom: 1.0,
            pan_offset: (0.0, 0.0),
            selected_cluster: None,
        }
    }
}

#[cfg(feature = "egui")]
impl InteractiveClusteringApp {
    /// Create new interactive app with data
    pub fn new(_scatterplot: ScatterPlot2D) -> Self {
        Self {
            scatter_plot_2d: Some(_scatterplot),
            ..Default::default()
        }
    }

    /// Update the scatter plot data
    pub fn set_data(&mut self, scatterplot: ScatterPlot2D) {
        self.scatter_plot_2d = Some(scatterplot);
    }
}

#[cfg(feature = "egui")]
impl eframe::App for InteractiveClusteringApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("Clustering Visualization");
            ui.separator();

            ui.checkbox(&mut self.show_centroids, "Show Centroids");
            ui.checkbox(&mut self.show_boundaries, "Show Boundaries");
            ui.checkbox(&mut self.show_legend, "Show Legend");

            ui.separator();
            ui.label("Zoom:");
            ui.add(egui::Slider::new(&mut self.zoom, 0.1..=5.0));

            if ui.button("Reset View").clicked() {
                self.zoom = 1.0;
                self.pan_offset = (0.0, 0.0);
            }

            ui.separator();
            if let Some(ref plot) = self.scatter_plot_2d {
                ui.label("Cluster Information:");
                for legend_entry in &plot.legend {
                    let color = parsehex_color(&legend_entry.color).unwrap_or([255, 0, 0]);
                    let color32 = Color32::from_rgb(color[0], color[1], color[2]);

                    ui.horizontal(|ui| {
                        ui.colored_label(color32, "●");
                        if ui
                            .selectable_label(
                                self.selected_cluster == Some(legend_entry.cluster_id),
                                format!(
                                    "Cluster {} ({} points)",
                                    legend_entry.cluster_id, legend_entry.count
                                ),
                            )
                            .clicked()
                        {
                            self.selected_cluster =
                                if self.selected_cluster == Some(legend_entry.cluster_id) {
                                    None
                                } else {
                                    Some(legend_entry.cluster_id)
                                };
                        }
                    });
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(plot) = self.scatter_plot_2d.clone() {
                self.draw_scatterplot(ui, &plot);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("No clustering data available");
                });
            }
        });
    }
}

#[cfg(feature = "egui")]
impl InteractiveClusteringApp {
    fn draw_scatterplot(&mut self, ui: &mut Ui, plot: &ScatterPlot2D) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::drag());

        let rect = response.rect;
        let (min_x, max_x, min_y, max_y) = plot.bounds;

        // Handle pan and zoom
        if response.dragged() {
            self.pan_offset.0 += response.drag_delta().x;
            self.pan_offset.1 += response.drag_delta().y;
        }

        // Convert data coordinates to screen coordinates
        let to_screen = |x: f64, y: f64| -> Pos2 {
            let norm_x = (x - min_x) / (max_x - min_x);
            let norm_y = (y - min_y) / (max_y - min_y);

            let screen_x =
                rect.left() + norm_x as f32 * rect.width() * self.zoom + self.pan_offset.0;
            let screen_y =
                rect.bottom() - norm_y as f32 * rect.height() * self.zoom + self.pan_offset.1;

            Pos2::new(screen_x, screen_y)
        };

        // Draw points
        for (i, point) in plot.points.rows().into_iter().enumerate() {
            let x = point[0];
            let y = point[1];
            let screen_pos = to_screen(x, y);

            if !rect.contains(screen_pos) {
                continue; // Skip points outside visible area
            }

            let colorhex = &plot.colors[i];
            let color = parsehex_color(colorhex).unwrap_or([255, 0, 0]);
            let color32 = Color32::from_rgb(color[0], color[1], color[2]);

            let radius = plot.sizes[i] * self.zoom;
            let cluster_id = plot.labels[i];

            // Highlight selected cluster
            let point_color = if let Some(selected) = self.selected_cluster {
                if cluster_id == selected {
                    color32
                } else {
                    Color32::from_rgba_premultiplied(color32.r(), color32.g(), color32.b(), 100)
                }
            } else {
                color32
            };

            painter.circle_filled(screen_pos, radius, point_color);
        }

        // Draw centroids
        if self.show_centroids {
            if let Some(ref centroids) = plot.centroids {
                for centroid in centroids.rows() {
                    let x = centroid[0];
                    let y = centroid[1];
                    let screen_pos = to_screen(x, y);

                    if rect.contains(screen_pos) {
                        painter.circle_stroke(
                            screen_pos,
                            8.0 * self.zoom,
                            Stroke::new(3.0, Color32::BLACK),
                        );
                        painter.line_segment(
                            [
                                Pos2::new(screen_pos.x - 6.0 * self.zoom, screen_pos.y),
                                Pos2::new(screen_pos.x + 6.0 * self.zoom, screen_pos.y),
                            ],
                            Stroke::new(3.0, Color32::BLACK),
                        );
                        painter.line_segment(
                            [
                                Pos2::new(screen_pos.x, screen_pos.y - 6.0 * self.zoom),
                                Pos2::new(screen_pos.x, screen_pos.y + 6.0 * self.zoom),
                            ],
                            Stroke::new(3.0, Color32::BLACK),
                        );
                    }
                }
            }
        }
    }
}

/// Utility function to parse hex color to RGB
#[allow(dead_code)]
fn parsehex_color(hex: &str) -> Option<[u8; 3]> {
    if hex.len() != 7 || !hex.starts_with('#') {
        return None;
    }

    let r = u8::from_str_radix(&hex[1..3], 16).ok()?;
    let g = u8::from_str_radix(&hex[3..5], 16).ok()?;
    let b = u8::from_str_radix(&hex[5..7], 16).ok()?;

    Some([r, g, b])
}

#[cfg(feature = "plotters")]
#[allow(dead_code)]
fn parsehex_color_plotters(hex: &str) -> Option<RGBColor> {
    let rgb = parsehex_color(hex)?;
    Some(RGBColor(rgb[0], rgb[1], rgb[2]))
}

/// High-level function to create and save a dendrogram plot
#[allow(dead_code)]
pub fn save_dendrogram_plot<P: AsRef<Path>>(
    linkage_matrix: ArrayView2<f64>,
    labels: Option<&[String]>,
    output_path: P,
    dendrogram_config: Option<&DendrogramConfig<f64>>,
    output_config: Option<&PlotOutput>,
) -> Result<()> {
    let default_dend_config = DendrogramConfig::default();
    let default_out_config = PlotOutput::default();
    let dend_config = dendrogram_config.unwrap_or(&default_dend_config);
    let out_config = output_config.unwrap_or(&default_out_config);

    // Create dendrogram plot data
    let dendrogram_plot = create_dendrogramplot(linkage_matrix, labels, dend_config.clone())?;

    #[cfg(feature = "plotters")]
    {
        plot_dendrogram(&dendrogram_plot, output_path, out_config)?;
    }

    #[cfg(not(feature = "plotters"))]
    {
        return Err(ClusteringError::ComputationError(
            "Plotters feature not enabled. Enable with --features plotters".to_string(),
        ));
    }

    Ok(())
}

/// Native 3D scatter plot using plotters
#[cfg(feature = "plotters")]
#[allow(dead_code)]
pub fn plot_scatter_3d<P: AsRef<Path>>(
    scatter_plot: &ScatterPlot3D,
    output_path: P,
    output_config: &PlotOutput,
) -> Result<()> {
    let path = output_path.as_ref();

    match output_config.format {
        PlotFormat::PNG => plot_scatter_3d_png(scatter_plot, path, output_config),
        PlotFormat::SVG => plot_scatter_3d_svg(scatter_plot, path, output_config),
        _ => Err(ClusteringError::ComputationError(
            "Unsupported output format for 3D plotters backend".to_string(),
        )),
    }
}

#[cfg(feature = "plotters")]
#[allow(dead_code)]
fn plot_scatter_3d_png<P: AsRef<Path>>(
    scatter_plot: &ScatterPlot3D,
    output_path: P,
    output_config: &PlotOutput,
) -> Result<()> {
    use plotters::coord::ranged3d::Cartesian3d;
    use plotters::coord::types::RangedCoordf64;

    let root = BitMapBackend::new(&output_path, output_config.dimensions).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        ClusteringError::ComputationError(format!("Failed to initialize plot: {}", e))
    })?;

    let (min_x, max_x, min_y, max_y, min_z, max_z) = scatter_plot.bounds;
    let margin = 0.1;
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;
    let z_range = max_z - min_z;

    let chart_builder = ChartBuilder::on(&root)
        .caption(
            output_config
                .title
                .as_deref()
                .unwrap_or("3D Cluster Visualization"),
            ("sans-serif", 30),
        )
        .margin(20)
        .build_cartesian_3d(
            (min_x - margin * x_range)..(max_x + margin * x_range),
            (min_y - margin * y_range)..(max_y + margin * y_range),
            (min_z - margin * z_range)..(max_z + margin * z_range),
        )
        .map_err(|e| {
            ClusteringError::ComputationError(format!("Failed to build 3D chart: {}", e))
        })?;

    let mut chart = chart_builder;

    // Configure chart
    chart
        .configure_axes()
        .light_grid_style(BLUE.mix(0.15))
        .max_light_lines(4)
        .draw()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to draw axes: {}", e)))?;

    // Plot points
    for (i, point) in scatter_plot.points.rows().into_iter().enumerate() {
        let x = point[0];
        let y = point[1];
        let z = point[2];
        let colorhex = &scatter_plot.colors[i];
        let size = scatter_plot.sizes[i] as i32;

        // Parse hex color
        let color = parsehex_color_plotters(colorhex).unwrap_or(RED);

        chart
            .draw_series(std::iter::once(Circle::new(
                (x, y, z),
                size,
                color.filled(),
            )))
            .map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to draw 3D points: {}", e))
            })?;
    }

    // Plot centroids if available
    if let Some(centroids) = &scatter_plot.centroids {
        for centroid in centroids.rows() {
            let x = centroid[0];
            let y = centroid[1];
            let z = centroid[2];

            chart
                .draw_series(std::iter::once(Circle::new(
                    (x, y, z),
                    8,
                    BLACK.stroke_width(3),
                )))
                .map_err(|e| {
                    ClusteringError::ComputationError(format!("Failed to draw 3D centroids: {}", e))
                })?;
        }
    }

    root.present()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to save 3D plot: {}", e)))?;

    Ok(())
}

#[cfg(feature = "plotters")]
#[allow(dead_code)]
fn plot_scatter_3d_svg<P: AsRef<Path>>(
    scatter_plot: &ScatterPlot3D,
    output_path: P,
    output_config: &PlotOutput,
) -> Result<()> {
    use plotters::coord::ranged3d::Cartesian3d;
    use plotters::coord::types::RangedCoordf64;

    let root = SVGBackend::new(&output_path, output_config.dimensions).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        ClusteringError::ComputationError(format!("Failed to initialize plot: {}", e))
    })?;

    let (min_x, max_x, min_y, max_y, min_z, max_z) = scatter_plot.bounds;
    let margin = 0.1;
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;
    let z_range = max_z - min_z;

    let chart_builder = ChartBuilder::on(&root)
        .caption(
            output_config
                .title
                .as_deref()
                .unwrap_or("3D Cluster Visualization"),
            ("sans-serif", 30),
        )
        .margin(20)
        .build_cartesian_3d(
            (min_x - margin * x_range)..(max_x + margin * x_range),
            (min_y - margin * y_range)..(max_y + margin * y_range),
            (min_z - margin * z_range)..(max_z + margin * z_range),
        )
        .map_err(|e| {
            ClusteringError::ComputationError(format!("Failed to build 3D chart: {}", e))
        })?;

    let mut chart = chart_builder;

    // Configure chart
    chart
        .configure_axes()
        .light_grid_style(BLUE.mix(0.15))
        .max_light_lines(4)
        .draw()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to draw axes: {}", e)))?;

    // Plot points
    for (i, point) in scatter_plot.points.rows().into_iter().enumerate() {
        let x = point[0];
        let y = point[1];
        let z = point[2];
        let colorhex = &scatter_plot.colors[i];
        let size = scatter_plot.sizes[i] as i32;

        // Parse hex color
        let color = parsehex_color_plotters(colorhex).unwrap_or(RED);

        chart
            .draw_series(std::iter::once(Circle::new(
                (x, y, z),
                size,
                color.filled(),
            )))
            .map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to draw 3D points: {}", e))
            })?;
    }

    // Plot centroids if available
    if let Some(centroids) = &scatter_plot.centroids {
        for centroid in centroids.rows() {
            let x = centroid[0];
            let y = centroid[1];
            let z = centroid[2];

            chart
                .draw_series(std::iter::once(Circle::new(
                    (x, y, z),
                    8,
                    BLACK.stroke_width(3),
                )))
                .map_err(|e| {
                    ClusteringError::ComputationError(format!("Failed to draw 3D centroids: {}", e))
                })?;
        }
    }

    root.present()
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to save 3D plot: {}", e)))?;

    Ok(())
}

/// High-level function to create and save a clustering plot
#[allow(dead_code)]
pub fn save_clustering_plot<P: AsRef<Path>>(
    data: ArrayView2<f64>,
    labels: &Array1<i32>,
    centroids: Option<&Array2<f64>>,
    output_path: P,
    config: Option<&VisualizationConfig>,
    output_config: Option<&PlotOutput>,
) -> Result<()> {
    let default_vis_config = VisualizationConfig::default();
    let default_out_config = PlotOutput::default();
    let vis_config = config.unwrap_or(&default_vis_config);
    let out_config = output_config.unwrap_or(&default_out_config);

    // Create scatter plot data
    let scatter_plot =
        crate::visualization::create_scatter_plot_2d(data, labels, centroids, vis_config)?;

    #[cfg(feature = "plotters")]
    {
        plot_scatter_2d(&scatter_plot, output_path, out_config)?;
    }

    #[cfg(not(feature = "plotters"))]
    {
        return Err(ClusteringError::ComputationError(
            "Plotters feature not enabled. Enable with --features plotters".to_string(),
        ));
    }

    Ok(())
}

/// High-level function to create and save a 3D clustering plot
#[allow(dead_code)]
pub fn save_clustering_plot_3d<P: AsRef<Path>>(
    data: ArrayView2<f64>,
    labels: &Array1<i32>,
    centroids: Option<&Array2<f64>>,
    output_path: P,
    config: Option<&VisualizationConfig>,
    output_config: Option<&PlotOutput>,
) -> Result<()> {
    let default_vis_config = VisualizationConfig::default();
    let default_out_config = PlotOutput::default();
    let vis_config = config.unwrap_or(&default_vis_config);
    let out_config = output_config.unwrap_or(&default_out_config);

    // Create 3D scatter plot data
    let scatter_plot =
        crate::visualization::create_scatter_plot_3d(data, labels, centroids, vis_config)?;

    #[cfg(feature = "plotters")]
    {
        plot_scatter_3d(&scatter_plot, output_path, out_config)?;
    }

    #[cfg(not(feature = "plotters"))]
    {
        return Err(ClusteringError::ComputationError(
            "Plotters feature not enabled. Enable with --features plotters".to_string(),
        ));
    }

    Ok(())
}

/// Launch interactive clustering visualization
#[cfg(feature = "egui")]
#[allow(dead_code)]
pub fn launch_interactive_visualization(
    data: ArrayView2<f64>,
    labels: &Array1<i32>,
    centroids: Option<&Array2<f64>>,
    config: Option<&VisualizationConfig>,
) -> Result<()> {
    let default_vis_config = VisualizationConfig::default();
    let vis_config = config.unwrap_or(&default_vis_config);

    // Create scatter plot data
    let scatter_plot =
        crate::visualization::create_scatter_plot_2d(data, labels, centroids, vis_config)?;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Clustering Visualization"),
        ..Default::default()
    };

    let app = InteractiveClusteringApp::new(scatter_plot);

    eframe::run_native(
        "Clustering Visualization",
        options,
        Box::new(|_| Ok::<_, Box<dyn std::error::Error + Send + Sync>>(Box::new(app))),
    )
    .map_err(|e| {
        ClusteringError::ComputationError(format!("Failed to launch visualization: {}", e))
    })?;

    Ok(())
}

#[cfg(not(feature = "egui"))]
#[allow(dead_code)]
pub fn launch_interactive_visualization(
    _data: ArrayView2<f64>,
    _labels: &Array1<i32>,
    _centroids: Option<&Array2<f64>>,
    _config: Option<&VisualizationConfig>,
) -> Result<()> {
    Err(ClusteringError::ComputationError(
        "Interactive visualization requires egui feature. Enable with --features egui".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn testhex_color_parsing() {
        assert_eq!(parsehex_color("#FF0000"), Some([255, 0, 0]));
        assert_eq!(parsehex_color("#00FF00"), Some([0, 255, 0]));
        assert_eq!(parsehex_color("#0000FF"), Some([0, 0, 255]));
        assert_eq!(parsehex_color("FF0000"), None); // Missing #
        assert_eq!(parsehex_color("#FG0000"), None); // Invalid hex
    }

    #[test]
    fn test_plot_output_default() {
        let output = PlotOutput::default();
        assert_eq!(output.format, PlotFormat::PNG);
        assert_eq!(output.dimensions, (800, 600));
        assert_eq!(output.dpi, 300);
    }
}
