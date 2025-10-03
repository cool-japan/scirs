//! Network architecture visualization for neural networks
//!
//! This module provides comprehensive tools for visualizing neural network architectures
//! including layout algorithms, rendering capabilities, and interactive features.

use super::config::{ImageFormat, VisualizationConfig};
use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use scirs2_core::numeric::Float;
use serde::Serialize;
use std::fmt::Debug;
use std::fs;
/// Network architecture visualizer
#[allow(dead_code)]
pub struct NetworkVisualizer<F: Float + Debug + scirs2_core::ndarray::ScalarOperand> {
    /// Model to visualize
    model: Sequential<F>,
    /// Visualization configuration
    config: VisualizationConfig,
    /// Cached layout information
    layout_cache: Option<NetworkLayout>,
}
/// Network layout information
#[derive(Debug, Clone, Serialize)]
pub struct NetworkLayout {
    /// Layer positions
    pub layer_positions: Vec<LayerPosition>,
    /// Connection information
    pub connections: Vec<Connection>,
    /// Bounding box
    pub bounds: BoundingBox,
    /// Layout algorithm used
    pub algorithm: LayoutAlgorithm,
/// Layer position in the visualization
pub struct LayerPosition {
    /// Layer name/identifier
    pub name: String,
    /// Layer type
    pub layer_type: String,
    /// Position coordinates
    pub position: Point2D,
    /// Layer dimensions
    pub size: Size2D,
    /// Input/output information
    pub io_info: LayerIOInfo,
    /// Visual properties
    pub visual_props: LayerVisualProps,
/// Point in 2D space
pub struct Point2D {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
/// Size in 2D space
pub struct Size2D {
    /// Width
    pub width: f32,
    /// Height
    pub height: f32,
/// Layer input/output information
pub struct LayerIOInfo {
    /// Input shape
    pub inputshape: Vec<usize>,
    /// Output shape
    pub outputshape: Vec<usize>,
    /// Parameter count
    pub parameter_count: usize,
    /// Computation complexity (FLOPs)
    pub flops: u64,
/// Visual properties for layer rendering
pub struct LayerVisualProps {
    /// Fill color
    pub fill_color: String,
    /// Border color
    pub border_color: String,
    /// Border width
    pub border_width: f32,
    /// Opacity
    pub opacity: f32,
    /// Layer icon/symbol
    pub icon: Option<String>,
/// Connection between layers
pub struct Connection {
    /// Source layer index
    pub from_layer: usize,
    /// Target layer index
    pub to_layer: usize,
    /// Connection type
    pub connection_type: ConnectionType,
    pub visual_props: ConnectionVisualProps,
    /// Data flow information
    pub data_flow: DataFlowInfo,
/// Type of connection between layers
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ConnectionType {
    /// Standard forward connection
    Forward,
    /// Skip/residual connection
    Skip,
    /// Attention connection
    Attention,
    /// Recurrent connection
    Recurrent,
    /// Sequential connection
    Sequential,
    /// Lateral connection
    Lateral,
    /// Custom connection
    Custom(String),
/// Visual properties for connection rendering
pub struct ConnectionVisualProps {
    /// Line color
    pub color: String,
    /// Line width
    /// Line style
    pub style: LineStyle,
    /// Arrow style
    pub arrow: ArrowStyle,
/// Line style for connections
pub enum LineStyle {
    /// Solid line
    Solid,
    /// Dashed line
    Dashed,
    /// Dotted line
    Dotted,
    /// Dash-dot line
    DashDot,
/// Arrow style for connections
pub enum ArrowStyle {
    /// No arrow
    None,
    /// Simple arrow
    Simple,
    /// Block arrow
    Block,
    /// Curved arrow
    Curved,
/// Data flow information
pub struct DataFlowInfo {
    /// Tensor shape flowing through connection
    pub tensorshape: Vec<usize>,
    /// Data type
    pub data_type: String,
    /// Estimated memory usage in bytes
    pub memory_usage: usize,
    /// Batch size for data flow
    pub batch_size: Option<usize>,
    /// Throughput information
    pub throughput: Option<ThroughputInfo>,
/// Throughput information for data flow
pub struct ThroughputInfo {
    /// Samples per second
    pub samples_per_second: f64,
    /// Bytes per second
    pub bytes_per_second: u64,
    /// Latency in milliseconds
    pub latency_ms: f64,
/// Bounding box for layout
pub struct BoundingBox {
    /// Minimum X coordinate
    pub min_x: f32,
    /// Minimum Y coordinate
    pub min_y: f32,
    /// Maximum X coordinate
    pub max_x: f32,
    /// Maximum Y coordinate
    pub max_y: f32,
/// Layout algorithm for network visualization
pub enum LayoutAlgorithm {
    /// Hierarchical layout (top-down)
    Hierarchical,
    /// Force-directed layout
    ForceDirected,
    /// Circular layout
    Circular,
    /// Grid layout
    Grid,
    /// Custom layout
/// Layer information for analysis
pub struct LayerInfo {
    /// Layer name
    pub layer_name: String,
    /// Layer index
    pub layer_index: usize,
// Implementation for NetworkVisualizer
impl<
        F: Float + Debug + 'static + scirs2_core::numeric::FromPrimitive + scirs2_core::ndarray::ScalarOperand + Send + Sync,
    > NetworkVisualizer<F>
{
    /// Create a new network visualizer
    pub fn new(model: Sequential<F>, config: VisualizationConfig) -> Self {
        Self {
            model,
            config,
            layout_cache: None,
        }
    }
    /// Generate network architecture visualization
    pub fn visualize_architecture(&mut self) -> Result<PathBuf> {
        // Compute network layout
        let layout = self.compute_layout()?;
        self.layout_cache = Some(layout.clone());
        // Generate visualization based on format
        match self.config.image_format {
            ImageFormat::SVG => self.generate_svg_visualization(&layout),
            ImageFormat::HTML => self.generate_html_visualization(&layout),
            ImageFormat::JSON => self.generate_json_visualization(&layout, _ => self.generate_svg_visualization(&layout), // Default to SVG
    /// Compute network layout using specified algorithm
    fn compute_layout(&self) -> Result<NetworkLayout> {
        // Analyze model structure
        let layer_info = self.analyze_model_structure()?;
        // Choose layout algorithm based on network complexity
        let algorithm = self.select_layout_algorithm(&layer_info);
        // Compute positions using selected algorithm
        let (positions, connections) = match algorithm {
            LayoutAlgorithm::Hierarchical => self.compute_hierarchical_layout(&layer_info)?,
            LayoutAlgorithm::ForceDirected => self.compute_force_directed_layout(&layer_info)?,
            LayoutAlgorithm::Circular => self.compute_circular_layout(&layer_info)?,
            LayoutAlgorithm::Grid => self.compute_grid_layout(&layer_info)?,
            LayoutAlgorithm::Custom(_) => self.compute_hierarchical_layout(&layer_info)?, // Fallback
        };
        // Compute bounding box
        let bounds = self.compute_bounds(&positions);
        Ok(NetworkLayout {
            layer_positions: positions,
            connections,
            bounds,
            algorithm,
        })
    fn analyze_model_structure(&self) -> Result<Vec<LayerInfo>> {
        let mut layer_info = Vec::new();
        // For Sequential models, we can iterate through the layers
        let layers = self.model.layers();
        for (index, layer) in layers.iter().enumerate() {
            let layer_type = layer.layer_type().to_string();
            let layer_name = format!("{layer_type}_{index}");
            layer_info.push(LayerInfo {
                layer_name,
                layer_index: index,
                layer_type,
            });
        // If no layers found, return error
        if layer_info.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "Model has no layers".to_string(),
            ));
        Ok(layer_info)
    fn select_layout_algorithm(&self, _layerinfo: &[LayerInfo]) -> LayoutAlgorithm {
        // For now, default to hierarchical layout
        // In a full implementation, this would analyze the network structure
        // and choose the most appropriate layout algorithm
        LayoutAlgorithm::Hierarchical
    fn compute_hierarchical_layout(
        &self,
        layer_info: &[LayerInfo],
    ) -> Result<(Vec<LayerPosition>, Vec<Connection>)> {
            return Ok((Vec::new(), Vec::new()));
        let mut positions = Vec::new();
        let mut connections = Vec::new();
        // Layout parameters
        let layer_width = 120.0;
        let layer_height = 60.0;
        let vertical_spacing = 100.0;
        let horizontal_spacing = 150.0;
        // Calculate total width and starting position
        let total_width = (layer_info.len() as f32 - 1.0) * horizontal_spacing + layer_width;
        let start_x = -total_width / 2.0 + layer_width / 2.0;
        let start_y = -(layer_info.len() as f32 - 1.0) * vertical_spacing / 2.0;
        // Create layer positions
        for (i, layer) in layer_info.iter().enumerate() {
            let x = start_x;
            let y = start_y + i as f32 * vertical_spacing;
            // Determine layer visual properties based on type
            let (fill_color, border_color, icon) = match layer.layer_type.as_str() {
                "Dense" => (
                    "#4CAF50".to_string(),
                    "#2E7D32".to_string(),
                    Some("◯".to_string()),
                ),
                "Conv2D" => (
                    "#2196F3".to_string(),
                    "#1565C0".to_string(),
                    Some("⬜".to_string()),
                "Conv1D" => (
                    "#03A9F4".to_string(),
                    "#0277BD".to_string(),
                    Some("▬".to_string()),
                "MaxPool2D" | "AvgPool2D" => (
                    "#FF9800".to_string(),
                    "#E65100".to_string(),
                    Some("▣".to_string()),
                "Dropout" => (
                    "#9C27B0".to_string(),
                    "#6A1B9A".to_string(),
                    Some("×".to_string()),
                "BatchNorm" => (
                    "#607D8B".to_string(),
                    "#37474F".to_string(),
                    Some("∼".to_string()),
                "Activation" => (
                    "#FFC107".to_string(),
                    "#F57C00".to_string(),
                    Some("∘".to_string()),
                "LSTM" => (
                    "#E91E63".to_string(),
                    "#AD1457".to_string(),
                    Some("⟲".to_string()),
                "GRU" => (
                    "#F44336".to_string(),
                    "#C62828".to_string(),
                    Some("⟳".to_string()),
                "Attention" => (
                    "#673AB7".to_string(),
                    "#4527A0".to_string(),
                    Some("◉".to_string(), _ => (
                    "#9E9E9E".to_string(),
                    "#424242".to_string(),
                    Some("?".to_string()),
            };
            // Estimate parameter count (simplified)
            let parameter_count = match layer.layer_type.as_str() {
                "Dense" => 10000, // Placeholder
                "Conv2D" => 5000,
                "Conv1D" => 3000_ => 0,
            // Estimate FLOPs (simplified)
            let flops = match layer.layer_type.as_str() {
                "Dense" => 100000,
                "Conv2D" => 500000,
                "Conv1D" => 200000_ => 1000,
            let position = LayerPosition {
                name: layer.layer_name.clone(),
                layer_type: layer.layer_type.clone(),
                position: Point2D { x, y },
                size: Size2D {
                    width: layer_width,
                    height: layer_height,
                },
                io_info: LayerIOInfo {
                    inputshape: vec![32, 32, 3],  // Placeholder
                    outputshape: vec![32, 32, 3], // Placeholder
                    parameter_count,
                    flops,
                visual_props: LayerVisualProps {
                    fill_color,
                    border_color,
                    border_width: 2.0,
                    opacity: 0.9,
                    icon,
            positions.push(position);
        // Create connections between adjacent layers
        for i in 0..(layer_info.len().saturating_sub(1)) {
            let connection = Connection {
                from_layer: i,
                to_layer: i + 1,
                connection_type: ConnectionType::Forward,
                visual_props: ConnectionVisualProps {
                    color: "#666666".to_string(),
                    width: 2.0,
                    style: LineStyle::Solid,
                    arrow: ArrowStyle::Simple,
                    opacity: 0.8,
                data_flow: DataFlowInfo {
                    tensorshape: vec![32, 32, 3], // Placeholder
                    data_type: "f32".to_string(),
                    memory_usage: 4096,   // Placeholder
                    batch_size: Some(32), // Default batch size
                    throughput: Some(ThroughputInfo {
                        samples_per_second: 1000.0,
                        bytes_per_second: 4096000,
                        latency_ms: 1.0,
                    }),
            connections.push(connection);
        Ok((positions, connections))
    fn compute_force_directed_layout(
        // Force-directed layout parameters
        let area = 800.0 * 600.0; // Canvas area
        let k = (area / layer_info.len() as f32).sqrt(); // Optimal distance
        let iterations = 100;
        let cooling_factor = 0.95;
        let mut temperature = 100.0;
        // Initialize random positions
        let mut node_positions: Vec<Point2D> = (0..layer_info.len())
            .map(|i| Point2D {
                x: ((i % 4) as f32 - 1.5) * 100.0, // Rough grid start
                y: ((i / 4) as f32 - 1.5) * 100.0,
            })
            .collect();
        // Force-directed algorithm iterations
        for _iteration in 0..iterations {
            let mut forces: Vec<Point2D> = vec![Point2D { x: 0.0, y: 0.0 }; layer_info.len()];
            // Calculate repulsive forces between all pairs
            for i in 0..layer_info.len() {
                for j in (i + 1)..layer_info.len() {
                    let dx = node_positions[i].x - node_positions[j].x;
                    let dy = node_positions[i].y - node_positions[j].y;
                    let distance = (dx * dx + dy * dy).sqrt().max(1.0);
                    let repulsive_force = k * k / distance;
                    let fx = repulsive_force * dx / distance;
                    let fy = repulsive_force * dy / distance;
                    forces[i].x += fx;
                    forces[i].y += fy;
                    forces[j].x -= fx;
                    forces[j].y -= fy;
                }
            }
            // Calculate attractive forces for connected layers (sequential connections)
            for i in 0..(layer_info.len() - 1) {
                let dx = node_positions[i].x - node_positions[i + 1].x;
                let dy = node_positions[i].y - node_positions[i + 1].y;
                let distance = (dx * dx + dy * dy).sqrt().max(1.0);
                let attractive_force = distance * distance / k;
                let fx = attractive_force * dx / distance;
                let fy = attractive_force * dy / distance;
                forces[i].x -= fx;
                forces[i].y -= fy;
                forces[i + 1].x += fx;
                forces[i + 1].y += fy;
            // Apply forces with temperature cooling
                let force_magnitude =
                    (forces[i].x * forces[i].x + forces[i].y * forces[i].y).sqrt();
                if force_magnitude > 0.0 {
                    let displacement = temperature.min(force_magnitude);
                    node_positions[i].x += forces[i].x / force_magnitude * displacement;
                    node_positions[i].y += forces[i].y / force_magnitude * displacement;
            temperature *= cooling_factor;
        // Create layer positions with visual properties
                    "#8BC34A".to_string(),
                    "#558B2F".to_string(),
                    "#757575".to_string(),
                    Some("▢".to_string()),
                position: node_positions[i].clone(),
                    width: 120.0,
                    height: 60.0,
                    inputshape: vec![1, 32], // Placeholder
                    outputshape: vec![1, 32],
                    parameter_count: 1024,
                    flops: 2048,
        // Create connections between sequential layers
        for i in 0..(layer_info.len() - 1) {
                connection_type: ConnectionType::Sequential,
                    tensorshape: vec![1, 32],
                    data_type: "float32".to_string(),
                    memory_usage: 128, // 1 * 32 * 4 bytes
                    batch_size: Some(1),
                    throughput: None,
                    opacity: 0.7,
    fn compute_circular_layout(
        // Circular layout parameters
        let radius = if layer_info.len() == 1 {
            50.0
        } else {
            // Calculate radius to ensure layers don't overlap
            let circumference = layer_info.len() as f32 * 150.0; // 150px minimum spacing
            circumference / (2.0 * std::f32::consts::PI)
        let center_x = 0.0;
        let center_y = 0.0;
        // Create layer positions around the circle
            let angle = if layer_info.len() == 1 {
                0.0
            } else {
                2.0 * std::f32::consts::PI * i as f32 / layer_info.len() as f32
            let x = center_x + radius * angle.cos();
            let y = center_y + radius * angle.sin();
        // For circular layout, also connect the last layer back to the first (if more than 2 layers)
        if layer_info.len() > 2 {
                from_layer: layer_info.len() - 1,
                to_layer: 0,
                connection_type: ConnectionType::Recurrent,
                    color: "#999999".to_string(),
                    width: 1.5,
                    style: LineStyle::Dashed,
                    opacity: 0.5,
    fn compute_grid_layout(
        // Grid layout parameters
        let cell_width = 180.0;
        let cell_height = 120.0;
        let margin = 20.0;
        // Calculate grid dimensions (prefer square or wide rectangle)
        let total_layers = layer_info.len();
        let grid_cols = (total_layers as f32).sqrt().ceil() as usize;
        let grid_rows = (total_layers as f32 / grid_cols as f32).ceil() as usize;
        // Calculate starting position to center the grid
        let total_width = grid_cols as f32 * cell_width;
        let total_height = grid_rows as f32 * cell_height;
        let start_x = -total_width / 2.0 + cell_width / 2.0;
        let start_y = -total_height / 2.0 + cell_height / 2.0;
        // Create layer positions in grid formation
            let col = i % grid_cols;
            let row = i / grid_cols;
            let x = start_x + col as f32 * cell_width;
            let y = start_y + row as f32 * cell_height;
                    width: cell_width - margin,
                    height: cell_height - margin,
            let from_col = i % grid_cols;
            let from_row = i / grid_cols;
            let to_col = (i + 1) % grid_cols;
            let to_row = (i + 1) / grid_cols;
            // Determine connection visual style based on grid position relationship
            let (color, style, width) = if from_row == to_row {
                // Same row - horizontal connection
                ("#4CAF50".to_string(), LineStyle::Solid, 2.5)
            } else if from_col == to_col {
                // Same column - vertical connection
                ("#2196F3".to_string(), LineStyle::Solid, 2.5)
                // Diagonal connection
                ("#FF9800".to_string(), LineStyle::Dashed, 2.0)
                    color,
                    width,
                    style,
        // Add some additional connections for grid pattern visualization
        // Connect layers in the same row (if there are multiple rows)
        if grid_rows > 1 {
            for row in 0..grid_rows {
                for col in 0..(grid_cols - 1) {
                    let from_idx = row * grid_cols + col;
                    let to_idx = row * grid_cols + col + 1;
                    if from_idx < total_layers && to_idx < total_layers && from_idx + 1 != to_idx {
                        let connection = Connection {
                            from_layer: from_idx,
                            to_layer: to_idx,
                            connection_type: ConnectionType::Lateral,
                            data_flow: DataFlowInfo {
                                tensorshape: vec![1, 16],
                                data_type: "float32".to_string(),
                                memory_usage: 64, // 1 * 16 * 4 bytes
                                batch_size: Some(1),
                                throughput: None,
                            },
                            visual_props: ConnectionVisualProps {
                                color: "#9E9E9E".to_string(),
                                width: 1.0,
                                style: LineStyle::Dotted,
                                arrow: ArrowStyle::None,
                                opacity: 0.4,
                        };
                        connections.push(connection);
                    }
    fn compute_bounds(&self, positions: &[LayerPosition]) -> BoundingBox {
        if positions.is_empty() {
            return BoundingBox {
                min_x: 0.0,
                min_y: 0.0,
                max_x: 100.0,
                max_y: 100.0,
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for pos in positions {
            min_x = min_x.min(pos.position.x - pos.size.width / 2.0);
            min_y = min_y.min(pos.position.y - pos.size.height / 2.0);
            max_x = max_x.max(pos.position.x + pos.size.width / 2.0);
            max_y = max_y.max(pos.position.y + pos.size.height / 2.0);
        BoundingBox {
            min_x,
            min_y,
            max_x,
            max_y,
    fn generate_svg_visualization(&self, layout: &NetworkLayout) -> Result<PathBuf> {
        let output_path = self.config.output_dir.join("network_architecture.svg");
        // Generate SVG content
        let svg_content = self.create_svg_content(layout)?;
        // Write to file
        fs::write(&output_path, svg_content)
            .map_err(|e| NeuralError::IOError(format!("Failed to write SVG file: {e}")))?;
        Ok(output_path)
    fn generate_html_visualization(&self, layout: &NetworkLayout) -> Result<PathBuf> {
        let output_path = self.config.output_dir.join("network_architecture.html");
        // Generate HTML content with interactive features
        let html_content = self.create_html_content(layout)?;
        fs::write(&output_path, html_content)
            .map_err(|e| NeuralError::IOError(format!("Failed to write HTML file: {e}")))?;
    fn generate_json_visualization(&self, layout: &NetworkLayout) -> Result<PathBuf> {
        let output_path = self.config.output_dir.join("network_architecture.json");
        // Serialize layout to JSON
        let json_content = serde_json::to_string_pretty(&layout).map_err(|e| {
            NeuralError::SerializationError(format!("Failed to serialize layout: {e}"))
        })?;
        fs::write(&output_path, json_content)
            .map_err(|e| NeuralError::IOError(format!("Failed to write JSON file: {e}")))?;
    fn create_svg_content(&self, layout: &NetworkLayout) -> Result<String> {
        let bounds = &layout.bounds;
        let margin = 50.0;
        // Calculate SVG dimensions
        let svg_width = (bounds.max_x - bounds.min_x + 2.0 * margin) as u32;
        let svg_height = (bounds.max_y - bounds.min_y + 2.0 * margin) as u32;
        // Calculate viewBox to center the network
        let viewbox_x = bounds.min_x - margin;
        let viewbox_y = bounds.min_y - margin;
        let viewbox_width = bounds.max_x - bounds.min_x + 2.0 * margin;
        let viewbox_height = bounds.max_y - bounds.min_y + 2.0 * margin;
        let mut svg = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<svg width=\"{}\" height=\"{}\" viewBox=\"{} {} {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n\
  <title>Neural Network Architecture</title>\n\
  <defs>\n\
    <style>\n\
      .layer-rect {{ stroke-width: 2; }}\n\
      .connection {{ fill: none; marker-end: url(#arrowhead); }}\n\
      .layer-text {{ font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; fill: white; font-weight: bold; }}\n\
      .layer-info {{ font-family: Arial, sans-serif; font-size: 9px; text-anchor: middle; fill: #333; }}\n\
      .layer-icon {{ font-family: Arial, sans-serif; font-size: 16px; text-anchor: middle; fill: white; font-weight: bold; }}\n\
    </style>\n\
    <marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n\
      <polygon points=\"0 0, 10 3.5, 0 7\" fill=\"#{}\"/>\n\
    </marker>\n\
  </defs>\n\
  \n\
  <!-- Background -->\n\
  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#{}\" stroke=\"#{}\"/>\n\
  \n",
            svg_width, svg_height, viewbox_x, viewbox_y, viewbox_width, viewbox_height,
            "666666",
            viewbox_x, viewbox_y, viewbox_width, viewbox_height, "f8f9fa", "dee2e6"
        );
        // Draw connections first (so they appear behind layers)
        for connection in &layout.connections {
            if connection.from_layer < layout.layer_positions.len()
                && connection.to_layer < layout.layer_positions.len()
            {
                let from_pos = &layout.layer_positions[connection.from_layer];
                let to_pos = &layout.layer_positions[connection.to_layer];
                // Calculate connection points (bottom of source to top of target)
                let x1 = from_pos.position.x;
                let y1 = from_pos.position.y + from_pos.size.height / 2.0;
                let x2 = to_pos.position.x;
                let y2 = to_pos.position.y - to_pos.size.height / 2.0;
                let stroke_width = connection.visual_props.width;
                let stroke_color = &connection.visual_props.color;
                let opacity = connection.visual_props.opacity;
                svg.push_str(&format!(
                    r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" opacity="{}" class="connection"/>
"#,
                    x1, y1, x2, y2, stroke_color, stroke_width, opacity
                ));
        // Draw layers
        for (i, layer_pos) in layout.layer_positions.iter().enumerate() {
            let x = layer_pos.position.x - layer_pos.size.width / 2.0;
            let y = layer_pos.position.y - layer_pos.size.height / 2.0;
            let width = layer_pos.size.width;
            let height = layer_pos.size.height;
            let fill_color = &layer_pos.visual_props.fill_color;
            let border_color = &layer_pos.visual_props.border_color;
            let border_width = layer_pos.visual_props.border_width;
            let opacity = layer_pos.visual_props.opacity;
            // Draw layer rectangle
            svg.push_str(&format!(
                r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="{}" opacity="{}" rx="5" class="layer-rect"/>
                x, y, width, height, fill_color, border_color, border_width, opacity
            // Draw layer icon if available
            if let Some(ref icon) = layer_pos.visual_props.icon {
                    r#"  <text x="{}" y="{}" class="layer-icon">{}</text>
                    layer_pos.position.x,
                    layer_pos.position.y - 5.0,
                    icon
            // Draw layer name
                r#"  <text x="{}" y="{}" class="layer-text">{}</text>
                layer_pos.position.x,
                layer_pos.position.y + 8.0,
                layer_pos.layer_type
            // Draw parameter info below the layer
            let paramtext = if layer_pos.io_info.parameter_count > 0 {
                format!("{}K params", layer_pos.io_info.parameter_count / 1000)
                "No params".to_string()
                r#"  <text x="{}" y="{}" class="layer-info">{}</text>
                y + height + 15.0,
                paramtext
            // Draw layer index
                r#"  <text x="{}" y="{}" class="layer-info">Layer {}</text>
                y - 10.0,
                i
        // Add legend
        let legend_x = viewbox_x + 10.0;
        let legend_y = viewbox_y + viewbox_height - 100.0;
        svg.push_str(&format!(
            "  <!-- Legend -->\n\
  <rect x=\"{}\" y=\"{}\" width=\"200\" height=\"80\" fill=\"white\" stroke=\"#{}\" stroke-width=\"1\" opacity=\"0.9\" rx=\"5\"/>\n\
  <text x=\"{}\" y=\"{}\" font-family=\"Arial\" font-size=\"12\" font-weight=\"bold\" fill=\"#333\">Legend</text>\n\
  <text x=\"{}\" y=\"{}\" font-family=\"Arial\" font-size=\"10\" fill=\"#666\">◯ Dense Layer</text>\n\
  <text x=\"{}\" y=\"{}\" font-family=\"Arial\" font-size=\"10\" fill=\"#666\">⬜ Conv2D Layer</text>\n\
  <text x=\"{}\" y=\"{}\" font-family=\"Arial\" font-size=\"10\" fill=\"#666\">× Dropout Layer</text>\n\
  <text x=\"{}\" y=\"{}\" font-family=\"Arial\" font-size=\"10\" fill=\"#666\">∼ BatchNorm Layer</text>\n",
            legend_x, legend_y, "ccc",
            legend_x + 10.0, legend_y + 15.0,
            legend_x + 10.0, legend_y + 30.0,
            legend_x + 10.0, legend_y + 45.0,
            legend_x + 10.0, legend_y + 60.0,
            legend_x + 10.0, legend_y + 75.0
        ));
        svg.push_str("</svg>");
        Ok(svg)
    fn create_html_content(&self, layout: &NetworkLayout) -> Result<String> {
        // Generate SVG content for embedding
        // Create the interactive HTML with embedded SVG and JavaScript controls
        let html_content = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Neural Network Architecture</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #333;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        .controls {{
            margin-bottom: 20px;
        .control-group {{
            display: inline-block;
            margin-right: 20px;
            vertical-align: top;
        .control-group label {{
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
            color: #555;
        button {{
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        button:active {{
            transform: translateY(0);
        select {{
            padding: 8px 12px;
            border: 1px solid #ddd;
        #visualization {{
            overflow: hidden;
            position: relative;
        #network-svg {{
            width: 100%;
            height: 700px;
            transition: transform 0.3s ease;
        .layer-node {{
        .layer-node:hover {{
            stroke-width: 3;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3));
        .connection-line {{
        .connection-line:hover {{
            stroke-width: 4;
            opacity: 1;
        .info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.95);
            padding: 15px;
            border-radius: 8px;
            max-width: 300px;
            display: none;
        .info-panel h3 {{
            margin: 0 0 10px 0;
            color: #444;
        .info-panel p {{
            margin: 5px 0;
            font-size: 13px;
            color: #666;
        .layout-controls {{
            margin-bottom: 10px;
        .hidden {{
        .highlight {{
            stroke: #ff6b6b !important;
            stroke-width: 4 !important;
            filter: drop-shadow(0 0 10px #ff6b6b);
    </style>
</head>
<body>
    <div class="header">
        <h1>Interactive Neural Network Architecture</h1>
        <p>Algorithm: {algorithm} | Layers: {layer_count} | Connections: {connection_count}</p>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label>Zoom Controls:</label>
            <button onclick="zoomIn()">🔍+ Zoom In</button>
            <button onclick="zoomOut()">🔍- Zoom Out</button>
            <button onclick="resetView()">🎯 Reset View</button>
        </div>
            <label>Display Options:</label>
            <button onclick="toggleLabels()">🏷️ Toggle Labels</button>
            <button onclick="toggleConnections()">🔗 Toggle Connections</button>
            <button onclick="highlightPath()">⚡ Highlight Data Flow</button>
            <label>Layout Algorithm:</label>
            <select id="layoutSelect" onchange="changeLayout()">
                <option value="hierarchical">📊 Hierarchical</option>
                <option value="force-directed">🌟 Force-Directed</option>
                <option value="circular">⭕ Circular</option>
                <option value="grid">⬜ Grid</option>
            </select>
            <label>Animation:</label>
            <button onclick="animateDataFlow()">🎬 Animate Flow</button>
            <button onclick="showLayerDetails()">📋 Layer Details</button>
    <div id="visualization">
        <div id="network-svg-container">
            {svg_content}
        <div id="info-panel" class="info-panel">
            <h3 id="info-title">Layer Information</h3>
            <p><strong>Type:</strong> <span id="info-type">-</span></p>
            <p><strong>Input Shape:</strong> <span id="info-input">-</span></p>
            <p><strong>Output Shape:</strong> <span id="info-output">-</span></p>
            <p><strong>Parameters:</strong> <span id="info-params">-</span></p>
            <p><strong>FLOPs:</strong> <span id="info-flops">-</span></p>
    <script>
        // Global state
        let currentZoom = 1.0;
        let showLabels = true;
        let showConnections = true;
        let selectedLayer = null;
        let animationRunning = false;
        // SVG manipulation
        const svg = document.querySelector('#network-svg-container svg');
        const infoPanel = document.getElementById('info-panel');
        // Zoom functions
        function zoomIn() {{
            currentZoom = Math.min(currentZoom * 1.2, 3.0);
            updateZoom();
        function zoomOut() {{
            currentZoom = Math.max(currentZoom / 1.2, 0.3);
        function resetView() {{
            currentZoom = 1.0;
            clearHighlights();
            hideInfo();
        function updateZoom() {{
            if (svg) {{
                svg.style.transform = `scale(${{currentZoom}})`;
            }}
        // Label toggle
        function toggleLabels() {{
            showLabels = !showLabels;
            const labels = svg.querySelectorAll('text');
            labels.forEach(label => {{
                label.style.display = showLabels ? 'block' : 'none';
            }});
        // Connection toggle
        function toggleConnections() {{
            showConnections = !showConnections;
            const connections = svg.querySelectorAll('.connection-line, line[stroke]');
            connections.forEach(conn => {{
                conn.style.display = showConnections ? 'block' : 'none';
        // Highlight data flow path
        function highlightPath() {{
            const layers = svg.querySelectorAll('rect, circle, ellipse');
            const connections = svg.querySelectorAll('line[stroke], path[stroke]');
            
            // Sequential highlighting with delay
            layers.forEach((layer, index) => {{
                setTimeout(() => {{
                    layer.classList.add('highlight');
                    setTimeout(() => layer.classList.remove('highlight'), 1000);
                }}, index * 200);
            connections.forEach((conn, index) => {{
                    conn.classList.add('highlight');
                    setTimeout(() => conn.classList.remove('highlight'), 1000);
                }}, index * 200 + 100);
        // Animate data flow
        function animateDataFlow() {{
            if (animationRunning) return;
            animationRunning = true;
                    conn.style.strokeDasharray = '10,5';
                    conn.style.strokeDashoffset = '0';
                    conn.style.animation = 'flow 2s linear infinite';
                }}, index * 100);
            // Add CSS animation dynamically
            const style = document.createElement('style');
            style.textContent = `
                @keyframes flow {{
                    to {{ stroke-dashoffset: -15; }}
                }}
            `;
            document.head.appendChild(style);
            setTimeout(() => {{
                connections.forEach(conn => {{
                    conn.style.animation = '';
                    conn.style.strokeDasharray = '';
                    conn.style.strokeDashoffset = '';
                }});
                animationRunning = false;
            }}, 5000);
        // Layer details
        function showLayerDetails() {{
                layer.addEventListener('click', () => showLayerInfo(layer, index));
                layer.style.cursor = 'pointer';
        function showLayerInfo(layer, index) {{
            selectedLayer = layer;
            // Highlight selected layer
            layer.classList.add('highlight');
            // Show info panel with layer details
            document.getElementById('info-title').textContent = `Layer ${{index + 1}}`;
            document.getElementById('info-type').textContent = layer.getAttribute('data-type') || 'Unknown';
            document.getElementById('info-input').textContent = layer.getAttribute('data-input') || '[1, 32]';
            document.getElementById('info-output').textContent = layer.getAttribute('data-output') || '[1, 32]';
            document.getElementById('info-params').textContent = layer.getAttribute('data-params') || '1,024';
            document.getElementById('info-flops').textContent = layer.getAttribute('data-flops') || '2,048';
            infoPanel.style.display = 'block';
        function hideInfo() {{
            infoPanel.style.display = 'none';
            selectedLayer = null;
        function clearHighlights() {{
            const highlighted = svg.querySelectorAll('.highlight');
            highlighted.forEach(el => el.classList.remove('highlight'));
        // Layout change implementation
        function changeLayout() {{
            const select = document.getElementById('layoutSelect');
            const layout = select.value;
            console.log(`Switching to ${{layout}} layout`);
            // Apply different layout algorithms
            switch(layout) {{
                case 'hierarchical':
                    applyHierarchicalLayout();
                    break;
                case 'circular':
                    applyCircularLayout();
                case 'force':
                    applyForceDirectedLayout();
                case 'grid':
                    applyGridLayout();
                default:
                    applyDefaultLayout();
        function applyHierarchicalLayout() {{
            const width = svg.viewBox.baseVal.width || 800;
            const height = svg.viewBox.baseVal.height || 600;
            const margin = 50;
                const x = margin + (index % 4) * (width - 2 * margin) / 3;
                const y = margin + Math.floor(index / 4) * (height - 2 * margin) / 3;
                layer.setAttribute('x', x);
                layer.setAttribute('y', y);
        function applyCircularLayout() {{
            const centerX = (svg.viewBox.baseVal.width || 800) / 2;
            const centerY = (svg.viewBox.baseVal.height || 600) / 2;
            const radius = Math.min(centerX, centerY) - 100;
                const angle = (2 * Math.PI * index) / layers.length;
                const x = centerX + radius * Math.cos(angle);
                const y = centerY + radius * Math.sin(angle);
        function applyForceDirectedLayout() {{
            // Simple force-directed positioning
                const x = Math.random() * (width - 100) + 50;
                const y = Math.random() * (height - 100) + 50;
        function applyGridLayout() {{
            const cols = Math.ceil(Math.sqrt(layers.length));
            const rows = Math.ceil(layers.length / cols);
                const col = index % cols;
                const row = Math.floor(index / cols);
                const x = 50 + col * (width - 100) / cols;
                const y = 50 + row * (height - 100) / rows;
        function applyDefaultLayout() {{
                const x = 50 + (index * 100) % (width - 100);
                const y = 100 + Math.floor((index * 100) / (width - 100)) * 80;
        // Initialize interactive features
        document.addEventListener('DOMContentLoaded', function() {{
            // Add event listeners to existing SVG elements
            showLayerDetails();
            // Close info panel when clicking outside
            document.addEventListener('click', function(e) {{
                if (!infoPanel.contains(e.target) && !e.target.closest('rect, circle, ellipse')) {{
                    hideInfo();
                    clearHighlights();
            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {{
                switch(e.key) {{
                    case '+':
                    case '=':
                        zoomIn();
                        break;
                    case '-':
                        zoomOut();
                    case '0':
                        resetView();
                    case 'l':
                        toggleLabels();
                    case 'c':
                        toggleConnections();
                    case 'h':
                        highlightPath();
        }});
    </script>
</body>
</html>"#,
            algorithm = format!("{:?}", layout.algorithm),
            layer_count = layout.layer_positions.len(),
            connection_count = layout.connections.len(),
            svg_content = svg_content
        Ok(html_content)
    /// Get the cached layout if available
    pub fn get_cached_layout(&self) -> Option<&NetworkLayout> {
        self.layout_cache.as_ref()
    /// Clear the layout cache
    pub fn clear_cache(&mut self) {
        self.layout_cache = None;
    /// Update the visualization configuration
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config;
        self.clear_cache(); // Clear cache when config changes
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use scirs2_core::random::SeedableRng;
    #[test]
    fn test_network_visualizer_creation() {
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());
        let config = VisualizationConfig::default();
        let visualizer = NetworkVisualizer::new(model, config);
        assert!(visualizer.layout_cache.is_none());
    fn test_layout_algorithm_variants() {
        let hierarchical = LayoutAlgorithm::Hierarchical;
        let force_directed = LayoutAlgorithm::ForceDirected;
        let circular = LayoutAlgorithm::Circular;
        let grid = LayoutAlgorithm::Grid;
        assert_eq!(hierarchical, LayoutAlgorithm::Hierarchical);
        assert_eq!(force_directed, LayoutAlgorithm::ForceDirected);
        assert_eq!(circular, LayoutAlgorithm::Circular);
        assert_eq!(grid, LayoutAlgorithm::Grid);
    fn test_connection_types() {
        let forward = ConnectionType::Forward;
        let skip = ConnectionType::Skip;
        let attention = ConnectionType::Attention;
        let recurrent = ConnectionType::Recurrent;
        let custom = ConnectionType::Custom("test".to_string());
        assert_eq!(forward, ConnectionType::Forward);
        assert_eq!(skip, ConnectionType::Skip);
        assert_eq!(attention, ConnectionType::Attention);
        assert_eq!(recurrent, ConnectionType::Recurrent);
        match custom {
            ConnectionType::Custom(name) => assert_eq!(name, "test", _ => unreachable!("Expected custom connection type"),
    fn test_bounding_box_computation() {
        // Test empty positions
        let empty_positions = vec![];
        let bounds = visualizer.compute_bounds(&empty_positions);
        assert_eq!(bounds.min_x, 0.0);
        assert_eq!(bounds.min_y, 0.0);
        assert_eq!(bounds.max_x, 100.0);
        assert_eq!(bounds.max_y, 100.0);
    fn test_point_2d() {
        let point = Point2D { x: 10.0, y: 20.0 };
        assert_eq!(point.x, 10.0);
        assert_eq!(point.y, 20.0);
    fn test_size_2d() {
        let size = Size2D {
            width: 100.0,
            height: 50.0,
        assert_eq!(size.width, 100.0);
        assert_eq!(size.height, 50.0);
    fn test_line_style_variants() {
        assert_eq!(LineStyle::Solid, LineStyle::Solid);
        assert_eq!(LineStyle::Dashed, LineStyle::Dashed);
        assert_eq!(LineStyle::Dotted, LineStyle::Dotted);
        assert_eq!(LineStyle::DashDot, LineStyle::DashDot);
    fn test_arrow_style_variants() {
        assert_eq!(ArrowStyle::None, ArrowStyle::None);
        assert_eq!(ArrowStyle::Simple, ArrowStyle::Simple);
        assert_eq!(ArrowStyle::Block, ArrowStyle::Block);
        assert_eq!(ArrowStyle::Curved, ArrowStyle::Curved);
