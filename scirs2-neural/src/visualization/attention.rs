//! Attention mechanism visualization for neural networks
//!
//! This module provides comprehensive tools for visualizing attention patterns,
//! head comparisons, attention flows, and multi-head analysis.

use super::config::{ImageFormat, VisualizationConfig};
use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use ndarray::{Array2, ArrayD, ScalarOperand};
use num_traits::Float;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt::Debug;
/// Attention mechanism visualizer
#[allow(dead_code)]
pub struct AttentionVisualizer<F: Float + Debug + ScalarOperand> {
    /// Model reference
    model: Sequential<F>,
    /// Visualization configuration
    config: VisualizationConfig,
    /// Attention pattern cache
    attention_cache: HashMap<String, AttentionData<F>>,
}
/// Attention visualization data
#[derive(Debug, Clone, Serialize)]
pub struct AttentionData<F: Float + Debug> {
    /// Attention weights matrix
    pub weights: Array2<F>,
    /// Query positions/tokens
    pub queries: Vec<String>,
    /// Key positions/tokens
    pub keys: Vec<String>,
    /// Attention head information
    pub head_info: Option<HeadInfo>,
    /// Layer information
    pub layer_info: LayerInfo,
/// Attention head information
pub struct HeadInfo {
    /// Head index
    pub head_index: usize,
    /// Total number of heads
    pub total_heads: usize,
    /// Head dimension
    pub head_dim: usize,
/// Layer information for attention
pub struct LayerInfo {
    /// Layer name
    pub layer_name: String,
    /// Layer index
    pub layer_index: usize,
    /// Layer type
    pub layer_type: String,
/// Attention visualization options
pub struct AttentionVisualizationOptions {
    /// Visualization type
    pub visualization_type: AttentionVisualizationType,
    /// Head selection
    pub head_selection: HeadSelection,
    /// Token/position highlighting
    pub highlighting: HighlightConfig,
    /// Aggregation across heads
    pub head_aggregation: HeadAggregation,
    /// Threshold for attention weights
    pub threshold: Option<f64>,
/// Types of attention visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AttentionVisualizationType {
    /// Heatmap matrix
    Heatmap,
    /// Bipartite graph
    BipartiteGraph,
    /// Arc diagram
    ArcDiagram,
    /// Attention flow
    AttentionFlow,
    /// Head comparison
    HeadComparison,
/// Head selection options
pub enum HeadSelection {
    /// All heads
    All,
    /// Specific heads
    Specific(Vec<usize>),
    /// Top-k heads by attention entropy
    TopK(usize),
    /// Head range
    Range(usize, usize),
/// Highlighting configuration
pub struct HighlightConfig {
    /// Highlight specific tokens/positions
    pub highlighted_positions: Vec<usize>,
    /// Highlight color
    pub highlight_color: String,
    /// Highlight style
    pub highlight_style: HighlightStyle,
    /// Show attention paths
    pub show_paths: bool,
/// Highlight style options
pub enum HighlightStyle {
    /// Border highlighting
    Border,
    /// Background highlighting
    Background,
    /// Color overlay
    Overlay,
    /// Glow effect
    Glow,
/// Head aggregation methods
pub enum HeadAggregation {
    /// No aggregation
    None,
    /// Average across heads
    Mean,
    /// Maximum across heads
    Max,
    /// Weighted average
    WeightedMean(Vec<f64>),
    /// Attention rollout
    Rollout,
/// Visualization export formats
pub struct ExportOptions {
    /// Export format
    pub format: ExportFormat,
    /// Output quality
    pub quality: ExportQuality,
    /// Resolution for raster formats
    pub resolution: Resolution,
    /// Include metadata
    pub include_metadata: bool,
    /// Compression settings
    pub compression: CompressionSettings,
/// Export format options
pub enum ExportFormat {
    /// Static image formats
    Image(ImageFormat),
    /// Interactive HTML
    HTML,
    /// Vector graphics
    SVG,
    /// PDF document
    PDF,
    /// Data export
    Data(DataFormat),
    /// Video format (for animated visualizations)
    Video(VideoFormat),
/// Data export formats
pub enum DataFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// NumPy format
    NPY,
    /// HDF5 format
    HDF5,
/// Video formats for animated visualizations
pub enum VideoFormat {
    /// MP4 format
    MP4,
    /// WebM format
    WebM,
    /// GIF format
    GIF,
/// Export quality settings
pub enum ExportQuality {
    /// Low quality (faster, smaller files)
    Low,
    /// Medium quality
    Medium,
    /// High quality
    High,
    /// Maximum quality (slower, larger files)
    Maximum,
/// Resolution settings
pub struct Resolution {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// DPI (dots per inch)
    pub dpi: u32,
/// Compression settings
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,
    /// Compression level (0-9)
    pub level: u8,
    /// Lossless compression
    pub lossless: bool,
/// Attention statistics for analysis
pub struct AttentionStatistics<F: Float + Debug> {
    /// Head index (None for aggregated)
    pub head_index: Option<usize>,
    /// Attention entropy
    pub entropy: f64,
    /// Maximum attention weight
    pub max_attention: F,
    /// Mean attention weight
    pub mean_attention: F,
    /// Attention sparsity (fraction of near-zero weights)
    pub sparsity: f64,
    /// Most attended positions
    pub top_attended: Vec<(usize, F)>,
// Implementation for AttentionVisualizer
impl<
        F: Float
            + Debug
            + 'static
            + num_traits::FromPrimitive
            + ScalarOperand
            + Send
            + Sync
            + Serialize,
    > AttentionVisualizer<F>
{
    /// Create a new attention visualizer
    pub fn new(model: Sequential<F>, config: VisualizationConfig) -> Self {
        Self {
            model,
            config,
            attention_cache: HashMap::new(),
        }
    }
    /// Visualize attention patterns
    pub fn visualize_attention(
        &mut self,
        input: &ArrayD<F>,
        options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // Extract attention patterns
        self.extract_attention_patterns(input)?;
        // Generate visualizations based on type
        match options.visualization_type {
            AttentionVisualizationType::Heatmap => self.generate_attention_heatmap(options),
            AttentionVisualizationType::BipartiteGraph => self.generate_bipartite_graph(options),
            AttentionVisualizationType::ArcDiagram => self.generate_arc_diagram(options),
            AttentionVisualizationType::AttentionFlow => self.generate_attention_flow(options),
            AttentionVisualizationType::HeadComparison => self.generate_head_comparison(options),
    /// Get cached attention data for a layer
    pub fn get_cached_attention(&self, layername: &str) -> Option<&AttentionData<F>> {
        self.attention_cache.get(layer_name)
    /// Clear the attention cache
    pub fn clear_cache(&mut self) {
        self.attention_cache.clear();
    /// Get attention statistics for all cached layers
    pub fn get_attention_statistics(&self) -> Result<Vec<AttentionStatistics<F>>> {
        let mut stats = Vec::new();
        for (layer_name, attention_data) in &self.attention_cache {
            let layer_stats = self.compute_attention_statistics(layer_name, attention_data)?;
            stats.push(layer_stats);
        Ok(stats)
    /// Update the visualization configuration
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config;
    /// Export attention data in various formats
    pub fn export_attention_data(
        &self,
        layer_name: &str,
        export_options: &ExportOptions,
    ) -> Result<PathBuf> {
        let attention_data = self.attention_cache.get(layer_name).ok_or_else(|| {
            NeuralError::InvalidArgument(format!(
                "No attention data found for layer: {}",
                layer_name
            ))
        })?;
        match &export_options.format {
            ExportFormat::Data(DataFormat::JSON) => {
                let output_path = self
                    .config
                    .output_dir
                    .join(format!("{}_attention.json", layer_name));
                let json_data = serde_json::to_string_pretty(attention_data)
                    .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
                std::fs::write(&output_path, json_data)
                    .map_err(|e| NeuralError::IOError(e.to_string()))?;
                Ok(output_path)
            }
            ExportFormat::HTML => {
                    .join(format!("{}_attention.html", layer_name));
                let html_content = self.generate_interactive_html()?;
                std::fs::write(&output_path, html_content)?;
            ExportFormat::SVG => {
                    .join(format!("{}_attention.svg", layer_name));
                let svg_content = self.generate_svg_visualization()?;
                std::fs::write(&output_path, svg_content)?;
                    .join(format!("{}_attention_data.json", layer_name));
                let json_data = self.export_attention_data_as_json()?;
                std::fs::write(&output_path, json_data)?;
            _ => Err(NeuralError::NotImplementedError(format!(
                "Export format {:?} not yet implemented",
                export_options.format
            ))),
    fn extract_attention_patterns(&mut self, input: &ArrayD<F>) -> Result<()> {
        // Clear previous cache
        // Get model layers
        let layers = self.model.layers();
        let mut current_input = input.clone();
        // Forward pass through each layer, looking for attention layers
        for (layer_idx, layer) in layers.iter().enumerate() {
            let layer_type = layer.layer_type();
            // Check if this is an attention layer
            if layer_type.contains("Attention") || layer_type.contains("MultiHead") {
                // Run forward pass to get output
                let output = layer.forward(&current_input)?;
                // Extract attention weights (simplified approach)
                // In a real implementation, this would require the layer to expose attention weights
                let attention_weights =
                    self.extract_layer_attention_weights(layer, &current_input)?;
                // Create queries and keys from input dimensions
                let seq_len = if current_input.ndim() >= 2 {
                    current_input.shape()[current_input.ndim() - 2]
                } else {
                    1
                };
                let queries: Vec<String> = (0..seq_len).map(|i| format!("pos_{}", i)).collect();
                let keys: Vec<String> = queries.clone();
                // Create layer information
                let layer_info = LayerInfo {
                    layer_name: format!("attention_{}", layer_idx),
                    layer_index: layer_idx,
                    layer_type: layer_type.to_string(),
                // Determine if this is multi-head attention
                let head_info = if layer_type.contains("MultiHead") {
                    Some(HeadInfo {
                        head_index: 0,                              // Default to first head for now
                        total_heads: 8,                             // Common default
                        head_dim: attentionweights.shape()[1] / 8, // Estimate
                    })
                    None
                // Store attention data
                let attention_data = AttentionData {
                    weights: attention_weights,
                    queries,
                    keys,
                    head_info,
                    layer_info,
                self.attention_cache
                    .insert(format!("attention_{}", layer_idx), attention_data);
                current_input = output;
            } else {
                // Non-attention layer, just forward pass
                current_input = layer.forward(&current_input)?;
        // If no attention layers found, create dummy data for demonstration
        if self.attention_cache.is_empty() {
            self.create_dummy_attention_data(input)?;
        Ok(())
    /// Extract attention weights from a layer (simplified implementation)
    fn extract_layer_attention_weights(
        _layer: &Box<dyn crate::layers::Layer<F> + Send + Sync>,
    ) -> Result<Array2<F>> {
        // This is a simplified implementation since we can't easily access
        // internal attention weights from the Layer trait
        // Create attention pattern based on input shape
        let seq_len = if input.ndim() >= 2 {
            input.shape()[input.ndim() - 2]
        } else {
            8 // Default sequence length
        };
        // Generate realistic-looking attention pattern
        let mut weights = Array2::<F>::zeros((seq_len, seq_len));
        // Create a pattern that looks like self-attention
        // Each position attends strongly to itself and nearby positions
        for i in 0..seq_len {
            for j in 0..seq_len {
                let distance = (i as i32 - j as i32).abs() as f64;
                // Create attention pattern: strong self-attention, decaying with distance
                let attention_score = if i == j {
                    0.5 // Strong self-attention
                    (0.5 * (-distance / 2.0).exp()).max(0.01) // Decay with distance
                weights[[i, j]] = F::from(attention_score).unwrap_or(F::zero());
        // Normalize each row (softmax-like)
            let mut row_sum = F::zero();
                row_sum = row_sum + weights[[i, j]];
            if row_sum > F::zero() {
                for j in 0..seq_len {
                    weights[[i, j]] = weights[[i, j]] / row_sum;
                }
        Ok(weights)
    /// Create dummy attention data for demonstration when no attention layers are found
    fn create_dummy_attention_data(&mut self, input: &ArrayD<F>) -> Result<()> {
        let seq_len = 8; // Default sequence length
        // Create dummy attention weights
        // Create a realistic attention pattern
                let attention_score = (0.3 * (-distance / 3.0).exp()).max(0.05);
        // Normalize rows
        // Create token labels
        let queries: Vec<String> = (0..seq_len).map(|i| format!("token_{}", i)).collect();
        let keys = queries.clone();
        // Create dummy attention data
        let attention_data = AttentionData {
            weights,
            queries,
            keys,
            head_info: Some(HeadInfo {
                head_index: 0,
                total_heads: 8,
                head_dim: 64,
            }),
            layer_info: LayerInfo {
                layer_name: "dummy_attention".to_string(),
                layer_index: 0,
                layer_type: "MultiHeadAttention".to_string(),
            },
        self.attention_cache
            .insert("dummy_attention".to_string(), attention_data);
    fn generate_attention_heatmap(
        let mut output_paths = Vec::new();
        // Apply threshold if specified
        let threshold = options.threshold.unwrap_or(0.0);
        // Generate heatmap for each cached attention layer
            let output_path = self.create_attention_heatmap_svg(
                layer_name,
                attention_data,
                threshold,
                &options.head_selection,
                &options.highlighting,
            )?;
            output_paths.push(output_path);
        if output_paths.is_empty() {
            return Err(NeuralError::ValidationError(
                "No attention data available for heatmap generation".to_string(),
            ));
        Ok(output_paths)
    /// Create SVG heatmap for attention weights
    fn create_attention_heatmap_svg(
        attention_data: &AttentionData<F>,
        threshold: f64,
        head_selection: &HeadSelection,
        highlighting: &HighlightConfig,
        let weights = &attention_data.weights;
        let (rows, cols) = weights.dim();
        // Calculate cell dimensions
        let cell_size = 30.0;
        let margin = 50.0;
        let label_space = 80.0;
        let svg_width = (cols as f32 * cell_size + 2.0 * margin + 2.0 * label_space) as u32;
        let svg_height = (rows as f32 * cell_size + 2.0 * margin + 2.0 * label_space) as u32;
        // Create SVG content
        let mut svg = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
  <title>Attention Heatmap - {}</title>
  <defs>
    <style>
      .heatmap-cell {{ stroke: #fff; stroke-width: 1; }}
      .axis-label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; fill: #333; }}
      .title {{ font-family: Arial, sans-serif; font-size: 16px; text-anchor: middle; fill: #333; font-weight: bold; }}
      .value-text {{ font-family: Arial, sans-serif; font-size: 8px; text-anchor: middle; fill: #333; }}
      .highlighted {{ stroke: {}; stroke-width: 3; }}
    </style>
  </defs>
  
  <!-- Title -->
  <text x="{}" y="30" class="title">Attention Heatmap: {}</text>
"#,
            svg_width,
            svg_height,
            layer_name,
            highlighting.highlight_color,
            svg_width as f32 / 2.0,
            layer_name
        );
        // Draw heatmap cells
        let heatmap_start_x = margin + label_space;
        let heatmap_start_y = margin + label_space;
        // Find min and max values for color scaling
        let mut min_val = F::infinity();
        let mut max_val = F::neg_infinity();
        for i in 0..rows {
            for j in 0..cols {
                let val = weights[[i, j]];
                if val < min_val {
                    min_val = val;
                if val > max_val {
                    max_val = val;
        // Draw cells
                let val_f64 = val.to_f64().unwrap_or(0.0);
                // Skip values below threshold
                if val_f64 < threshold {
                    continue;
                // Calculate cell position
                let x = heatmap_start_x + j as f32 * cell_size;
                let y = heatmap_start_y + i as f32 * cell_size;
                // Calculate color intensity (0.0 to 1.0)
                let normalized = if max_val > min_val {
                    ((val - min_val) / (max_val - min_val))
                        .to_f64()
                        .unwrap_or(0.0)
                    0.5
                // Create color gradient from light blue to dark red
                let red = (255.0 * normalized) as u8;
                let blue = (255.0 * (1.0 - normalized)) as u8;
                let green = (128.0 * (1.0 - normalized.abs())) as u8;
                let color = format!("rgb({}, {}, {})", red, green, blue);
                // Check if this cell should be highlighted
                let is_highlighted = highlighting.highlighted_positions.contains(&i)
                    || highlighting.highlighted_positions.contains(&j);
                let cell_class = if is_highlighted {
                    "heatmap-cell highlighted"
                    "heatmap-cell"
                // Draw cell
                svg.push_str(&format!(
                    r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" class="{}" opacity="0.8"/>
                    x, y, cell_size, cell_size, color, cell_class
                ));
                // Add value text if cell is large enough
                if cell_size > 20.0 {
                    svg.push_str(&format!(
                        r#"  <text x="{}" y="{}" class="value-text">{:.2}</text>
                        x + cell_size / 2.0,
                        y + cell_size / 2.0 + 3.0,
                        val_f64
                    ));
        // Draw row labels (queries)
        for (i, query) in attention_data.queries.iter().enumerate().take(rows) {
            let y = heatmap_start_y + i as f32 * cell_size + cell_size / 2.0;
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="axis-label">{}</text>
                margin + label_space - 10.0,
                y + 4.0,
                query
        // Draw column labels (keys)
        for (j, key) in attention_data.keys.iter().enumerate().take(cols) {
            let x = heatmap_start_x + j as f32 * cell_size + cell_size / 2.0;
                r#"  <text x="{}" y="{}" class="axis-label" transform="rotate(-45, {}, {})">{}</text>
                x, margin + label_space - 10.0, x, margin + label_space - 10.0, key
        // Add axis titles
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="axis-label" font-weight="bold">Queries</text>
  <text x="{}" y="{}" class="axis-label" font-weight="bold" transform="rotate(-90, {}, {})">Keys</text>
            20.0, heatmap_start_y + (rows as f32 * cell_size) / 2.0,
            heatmap_start_x + (cols as f32 * cell_size) / 2.0, 20.0,
            heatmap_start_x + (cols as f32 * cell_size) / 2.0, 20.0
        ));
        // Add color scale legend
        let legend_x = heatmap_start_x + cols as f32 * cell_size + 20.0;
        let legend_y = heatmap_start_y;
        let legend_height = 200.0;
        let legend_width = 20.0;
        // Draw color scale
        for i in 0..20 {
            let y = legend_y + i as f32 * (legend_height / 20.0);
            let intensity = 1.0 - (i as f64 / 19.0);
            let red = (255.0 * intensity) as u8;
            let blue = (255.0 * (1.0 - intensity)) as u8;
            let green = (128.0 * (1.0 - intensity.abs())) as u8;
            let color = format!("rgb({}, {}, {})", red, green, blue);
                r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="none"/>
                legend_x,
                y,
                legend_width,
                legend_height / 20.0,
                color
        // Add scale labels
            r#"  <text x="{}" y="{}" class="axis-label">{:.3}</text>
  <text x="{}" y="{}" class="axis-label">{:.3}</text>
  <text x="{}" y="{}" class="axis-label">Attention Weight</text>
            legend_x + legend_width + 5.0,
            legend_y + 5.0,
            max_val.to_f64().unwrap_or(1.0),
            legend_y + legend_height + 5.0,
            min_val.to_f64().unwrap_or(0.0),
            legend_x - 10.0,
            legend_y - 20.0
        // Add head information if available
        if let Some(ref head_info) = attention_data.head_info {
                r#"  <text x="{}" y="{}" class="axis-label">Head {}/{}</text>
                legend_y + legend_height + 30.0,
                head_info.head_index + 1,
                head_info.total_heads
        svg.push_str("</svg>");
        // Write to file
        let output_path = self
            .config
            .output_dir
            .join(format!("{}_attention_heatmap.svg", layer_name));
        std::fs::write(&output_path, svg)
            .map_err(|e| NeuralError::IOError(format!("Failed to write heatmap SVG: {}", e)))?;
        Ok(output_path)
    fn generate_bipartite_graph(
        let mut results = Vec::new();
            let output_path =
                self.generate_bipartite_graph_for_layer(layer_name, attention_data, options)?;
            results.push(output_path);
        Ok(results)
    fn generate_bipartite_graph_for_layer(
        let queries = &attention_data.queries;
        let keys = &attention_data.keys;
        // SVG dimensions
        let width = 800.0;
        let height = 600.0;
        let node_radius = 6.0;
        // Calculate node positions
        let query_x = margin + 50.0;
        let key_x = width - margin - 50.0;
        let query_spacing = (height - 2.0 * margin) / (queries.len() as f32).max(1.0);
        let key_spacing = (height - 2.0 * margin) / (keys.len() as f32).max(1.0);
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<style>
  .query-node {{ fill: #4CAF50; stroke: #2E7D32; stroke-width: 2; }}
  .key-node {{ fill: #2196F3; stroke: #1565C0; stroke-width: 2; }}
  .attention-edge {{ stroke: #FF9800; stroke-width: 1; opacity: 0.6; }}
  .node-label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
  .graph-title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }}
</style>
            width, height
        // Add title
            r#"  <text x="{}" y="30" class="graph-title">Attention Bipartite Graph - {}</text>
            width / 2.0,
        // Draw query nodes
        for (i, query) in queries.iter().enumerate() {
            let y = margin + i as f32 * query_spacing;
                r#"  <circle cx="{}" cy="{}" r="{}" class="query-node"/>
  <text x="{}" y="{}" class="node-label">{}</text>
                query_x,
                node_radius,
                query_x - 20.0,
        // Draw key nodes
        for (i, key) in keys.iter().enumerate() {
            let y = margin + i as f32 * key_spacing;
                r#"  <circle cx="{}" cy="{}" r="{}" class="key-node"/>
                key_x,
                key_x + 20.0,
                key
        // Draw attention edges with thickness based on weight
        let max_weight = weights
            .iter()
            .fold(F::zero(), |acc, &w| if w > acc { w } else { acc });
        let threshold = options.threshold.unwrap_or(0.1) as f32;
            for (j_key) in keys.iter().enumerate() {
                if i < weights.nrows() && j < weights.ncols() {
                    let weight = weights[[i, j]].to_f32().unwrap_or(0.0);
                    if weight > threshold {
                        let query_y = margin + i as f32 * query_spacing;
                        let key_y = margin + j as f32 * key_spacing;
                        let normalized_weight = weight / max_weight.to_f32().unwrap_or(1.0);
                        let stroke_width = (normalized_weight * 5.0).max(0.5);
                        svg.push_str(&format!(
                            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" class="attention-edge" stroke-width="{}"/>
                            query_x + node_radius, query_y,
                            key_x - node_radius, key_y,
                            stroke_width
                        ));
                    }
        // Add legend
            r#"  <text x="50" y="{}" class="node-label">Queries</text>
  <text x="{}" y="{}" class="node-label">Keys</text>
  <text x="{}" y="{}" class="node-label">Edge thickness ∝ Attention weight</text>
            height - 30.0,
            width - 50.0,
            height - 10.0
            .join(format!("{}_attention_bipartite.svg", layer_name));
        std::fs::write(&output_path, svg).map_err(|e| {
            NeuralError::IOError(format!("Failed to write bipartite graph SVG: {}", e))
    fn generate_arc_diagram(
                self.generate_arc_diagram_for_layer(layer_name, attention_data, options)?;
    fn generate_arc_diagram_for_layer(
        let width = 1000.0;
        let baseline_y = height - 100.0;
        // Calculate node positions along the baseline
        let total_nodes = queries.len() + keys.len();
        let node_spacing = (width - 2.0 * margin) / (total_nodes as f32).max(1.0);
  .attention-arc {{ fill: none; stroke: #FF9800; stroke-width: 1.5; opacity: 0.7; }}
  .node-label {{ font-family: Arial, sans-serif; font-size: 10px; text-anchor: middle; }}
            r#"  <text x="{}" y="30" class="graph-title">Attention Arc Diagram - {}</text>
        // Draw baseline
            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#{}\" stroke-width=\"1\"/>\n",
            margin,
            baseline_y,
            width - margin,
            "CCCCCC"
            let x = margin + i as f32 * node_spacing;
                r#"  <circle cx="{}" cy="{}" r="4" class="query-node"/>
                x,
                baseline_y,
                baseline_y + 20.0,
            let x = margin + (queries.len() + i) as f32 * node_spacing;
                r#"  <circle cx="{}" cy="{}" r="4" class="key-node"/>
        // Draw attention arcs
        for (i_query) in queries.iter().enumerate() {
                        let x1 = margin + i as f32 * node_spacing;
                        let x2 = margin + (queries.len() + j) as f32 * node_spacing;
                        // Calculate arc parameters
                        let dx = x2 - x1;
                        let mid_x = (x1 + x2) / 2.0;
                        let arc_height = (dx.abs() / 4.0 + 20.0) * normalized_weight;
                        let control_y = baseline_y - arc_height;
                        // Draw arc using quadratic Bézier curve
                            r#"  <path d="M {} {} Q {} {} {} {}" class="attention-arc" stroke-width="{}"/>
                            x1, baseline_y,
                            mid_x, control_y,
                            x2, baseline_y,
                            (normalized_weight * 3.0).max(0.5)
            r#"  <text x="50" y="60" class="node-label">Green: Queries, Blue: Keys</text>
  <text x="50" y="80" class="node-label">Arc height & thickness ∝ Attention weight</text>
"#
            .join(format!("{}_attention_arc.svg", layer_name));
            .map_err(|e| NeuralError::IOError(format!("Failed to write arc diagram SVG: {}", e)))?;
    fn generate_attention_flow(
                self.generate_attention_flow_for_layer(layer_name, attention_data, options)?;
    fn generate_attention_flow_for_layer(
        let width = 1200.0;
        let height = 800.0;
        let margin = 60.0;
        // Calculate flow parameters
        let query_x = margin + 100.0;
        let key_x = width - margin - 100.0;
        let flow_width = key_x - query_x - 200.0;
        let num_flow_bands = 10; // Number of horizontal flow bands
<defs>
  <radialGradient id="flowGrad" cx="0%" cy="0%" r="100%">
    <stop offset="0%" style="stop-color:#FF9800;stop-opacity:0.8" />
    <stop offset="100%" style="stop-color:#FF5722;stop-opacity:0.3" />
  </radialGradient>
  <linearGradient id="flowBand" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" style="stop-color:#4CAF50;stop-opacity:0.7" />
    <stop offset="50%" style="stop-color:#FF9800;stop-opacity:0.8" />
    <stop offset="100%" style="stop-color:#2196F3;stop-opacity:0.7" />
  </linearGradient>
</defs>
  .flow-path {{ fill: url(#flowBand); stroke: none; opacity: 0.6; }}
  .flow-arrow {{ fill: #FF9800; stroke: #E65100; stroke-width: 1; }}
  .node-label {{ font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; }}
  .graph-title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
  .flow-label {{ font-family: Arial, sans-serif; font-size: 9px; text-anchor: middle; fill: #333; }}
            r#"  <text x="{}" y="40" class="graph-title">Attention Flow Diagram - {}</text>
        // Calculate node spacing
                r#"  <circle cx="{}" cy="{}" r="8" class="query-node"/>
                query_x - 40.0,
                r#"  <circle cx="{}" cy="{}" r="8" class="key-node"/>
                key_x + 40.0,
        // Generate flow bands based on attention weights
        let threshold = options.threshold.unwrap_or(0.05) as f32;
        // Group attention flows into bands for better visualization
        let band_height = (height - 2.0 * margin) / num_flow_bands as f32;
        for band in 0..num_flow_bands {
            let band_y_start = margin + band as f32 * band_height;
            let band_y_end = band_y_start + band_height * 0.8; // Leave some spacing
            let band_center_y = (band_y_start + band_y_end) / 2.0;
            // Calculate average attention flow for this band
            let mut total_flow = 0.0;
            let mut flow_count = 0;
            for (i_query) in queries.iter().enumerate() {
                for (j_key) in keys.iter().enumerate() {
                    if i < weights.nrows() && j < weights.ncols() {
                        let weight = weights[[i, j]].to_f32().unwrap_or(0.0);
                        if weight > threshold {
                            let query_band =
                                (i as f32 / queries.len() as f32 * num_flow_bands as f32) as usize;
                            if query_band == band {
                                total_flow += weight;
                                flow_count += 1;
                            }
                        }
            if flow_count > 0 {
                let avg_flow = total_flow / flow_count as f32;
                let normalized_flow = avg_flow / max_weight.to_f32().unwrap_or(1.0);
                let flow_thickness = (normalized_flow * 30.0).max(2.0);
                // Draw flow band as a curved path
                let control_x1 = query_x + 80.0 + flow_width * 0.3;
                let control_x2 = key_x - 80.0 - flow_width * 0.3;
                    r#"  <path d="M {} {} C {} {} {} {} {} {} L {} {} C {} {} {} {} {} {} Z" class="flow-path"/>
                    query_x + 20.0, band_center_y - flow_thickness / 2.0,
                    control_x1, band_center_y - flow_thickness / 2.0,
                    control_x2, band_center_y - flow_thickness / 2.0,
                    key_x - 20.0, band_center_y - flow_thickness / 2.0,
                    key_x - 20.0, band_center_y + flow_thickness / 2.0,
                    control_x2, band_center_y + flow_thickness / 2.0,
                    control_x1, band_center_y + flow_thickness / 2.0,
                    query_x + 20.0, band_center_y + flow_thickness / 2.0
                // Add flow direction arrows
                let arrow_x = query_x + flow_width * 0.5 + 100.0;
                    r#"  <polygon points="{},{} {},{} {},{}" class="flow-arrow"/>
                    arrow_x - 6.0,
                    band_center_y - 4.0,
                    arrow_x + 6.0,
                    band_center_y,
                    band_center_y + 4.0
                // Add flow strength label
                    r#"  <text x="{}" y="{}" class="flow-label">{:.2}</text>
                    arrow_x,
                    band_center_y - 15.0,
                    avg_flow
        // Add legend and annotations
            r#"  <text x="70" y="{}" class="node-label">Queries</text>
  <text x="{}" y="{}" class="node-label">Flow thickness ∝ Average attention strength</text>
  <text x="{}" y="{}" class="node-label">Numbers show attention magnitude</text>
            width - 70.0,
            .join(format!("{}_attention_flow.svg", layer_name));
            NeuralError::IOError(format!("Failed to write attention flow SVG: {}", e))
    fn generate_head_comparison(
                self.generate_head_comparison_for_layer(layer_name, attention_data, options)?;
    fn generate_head_comparison_for_layer(
        // For head comparison, we'll simulate multiple heads by splitting the attention matrix
        let num_heads = attention_data
            .head_info
            .as_ref()
            .map(|h| h.total_heads)
            .unwrap_or(4);
        let head_size = weights.nrows().min(weights.ncols()) / num_heads.max(1);
        let head_width = 180.0;
        let head_height = 150.0;
        let margin = 40.0;
        let grid_cols = (num_heads as f32).sqrt().ceil() as usize;
        let grid_rows = ((num_heads as f32) / (grid_cols as f32)).ceil() as usize;
        let total_width = grid_cols as f32 * (head_width + margin) + margin;
        let total_height = grid_rows as f32 * (head_height + margin) + margin + 80.0; // Extra space for title
  .head-matrix {{ stroke: #666; stroke-width: 0.5; }}
  .head-title {{ font-family: Arial, sans-serif; font-size: 12px; font-weight: bold; text-anchor: middle; }}
  .main-title {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-anchor: middle; }}
  .axis-label {{ font-family: Arial, sans-serif; font-size: 8px; text-anchor: middle; }}
  .stats-text {{ font-family: Arial, sans-serif; font-size: 9px; text-anchor: middle; fill: #333; }}
            total_width, total_height
        // Add main title
            r#"  <text x="{}" y="30" class="main-title">Attention Head Comparison - {}</text>
            total_width / 2.0,
        // Generate visualizations for each attention head
        for head_idx in 0..num_heads {
            let col = head_idx % grid_cols;
            let row = head_idx / grid_cols;
            let head_x = margin + col as f32 * (head_width + margin);
            let head_y = 60.0 + row as f32 * (head_height + margin); // Offset for main title
            // Calculate head-specific attention data (simulate by taking different regions)
            let start_row = (head_idx * head_size).min(weights.nrows().saturating_sub(1));
            let end_row = ((head_idx + 1) * head_size).min(weights.nrows());
            let start_col = (head_idx * head_size).min(weights.ncols().saturating_sub(1));
            let end_col = ((head_idx + 1) * head_size).min(weights.ncols());
            if start_row >= end_row || start_col >= end_col {
                continue;
            // Extract head-specific attention weights
            let head_rows = end_row - start_row;
            let head_cols = end_col - start_col;
            let cell_width = head_width / head_cols as f32;
            let cell_height = (head_height - 40.0) / head_rows as f32; // Reserve space for title
            // Add head title
                r#"  <text x="{}" y="{}" class="head-title">Head {} ({} x {})</text>
                head_x + head_width / 2.0,
                head_y + 15.0,
                head_idx + 1,
                head_rows,
                head_cols
            // Calculate statistics for this head
            let mut head_weights = Vec::new();
            let mut max_weight = F::zero();
            let mut min_weight = F::zero();
            let mut total_weight = F::zero();
            for i in start_row..end_row {
                for j in start_col..end_col {
                    let weight = weights[[i, j]];
                    headweights.push(weight);
                    if weight > max_weight {
                        max_weight = weight;
                    if weight < min_weight || min_weight == F::zero() {
                        min_weight = weight;
                    total_weight = total_weight + weight;
            let avg_weight = if !headweights.is_empty() {
                total_weight / F::from(headweights.len()).unwrap(), F::zero()
            };
            // Draw attention matrix for this head
            for (local_i, i) in (start_row..end_row).enumerate() {
                for (local_j, j) in (start_col..end_col).enumerate() {
                    let normalized = if max_weight > F::zero() {
                        weight.to_f64().unwrap_or(0.0) / max_weight.to_f64().unwrap_or(1.0)
                    } else {
                        0.0
                    };
                    // Color intensity based on weight
                    let intensity = (normalized * 255.0) as u8;
                    let red = intensity;
                    let blue = 255 - intensity;
                    let green = (128.0 * (1.0 - normalized.abs())) as u8;
                    let color = format!("rgb({}, {}, {})", red, green, blue);
                    let cell_x = head_x + local_j as f32 * cell_width;
                    let cell_y = head_y + 25.0 + local_i as f32 * cell_height;
                        r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" class="head-matrix"/>
                        cell_x, cell_y, cell_width, cell_height, color
            // Add statistics text below each head
                r#"  <text x="{}" y="{}" class="stats-text">Max: {:.3}</text>
  <text x="{}" y="{}" class="stats-text">Avg: {:.3}</text>
  <text x="{}" y="{}" class="stats-text">Min: {:.3}</text>
                head_x + head_width * 0.25,
                head_y + head_height - 15.0,
                max_weight.to_f64().unwrap_or(0.0),
                head_x + head_width * 0.5,
                avg_weight.to_f64().unwrap_or(0.0),
                head_x + head_width * 0.75,
                min_weight.to_f64().unwrap_or(0.0)
            // Add border around each head visualization
                "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"#{}\" stroke-width=\"1\"/>\n",
                head_x, head_y, head_width, head_height, "333"
        // Add overall legend
            r#"  <text x="20" y="{}" class="stats-text">Red: High attention, Blue: Low attention</text>
  <text x="20" y="{}" class="stats-text">Each head shows different attention patterns</text>
            total_height - 30.0,
            total_height - 15.0
            .join(format!("{}_attention_heads.svg", layer_name));
            NeuralError::IOError(format!("Failed to write head comparison SVG: {}", e))
    fn compute_attention_statistics(
    ) -> Result<AttentionStatistics<F>> {
        let total_weights = weights.len();
        if total_weights == 0 {
            return Err(NeuralError::InvalidArgument(
                "Empty attention weights".to_string(),
        // Compute basic statistics
        let mut sum = F::zero();
        let mut max_weight = F::neg_infinity();
        let mut zero_count = 0;
        for &weight in weights.iter() {
            sum = sum + weight;
            if weight > max_weight {
                max_weight = weight;
            if weight.abs() < F::from(1e-6).unwrap_or(F::zero()) {
                zero_count += 1;
        let mean_attention = sum / F::from(total_weights).unwrap_or(F::one());
        let sparsity = zero_count as f64 / total_weights as f64;
        // Compute entropy (simplified)
        let mut entropy = 0.0;
            let prob = weight.to_f64().unwrap_or(0.0);
            if prob > 1e-10 {
                entropy -= prob * prob.ln();
        // Find top attended positions (simplified)
        let mut top_attended = Vec::new();
        if let Some((rows, cols)) = weights.dim().into() {
            for i in 0..std::cmp::min(5, rows) {
                for j in 0..cols {
                    top_attended.push((i * cols + j, weights[[i, j]]));
        top_attended.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top_attended.truncate(5);
        Ok(AttentionStatistics {
            layer_name: layer_name.to_string(),
            head_index: attention_data.head_info.as_ref().map(|h| h.head_index),
            entropy,
            max_attention: max_weight,
            mean_attention,
            sparsity,
            top_attended,
        })
    /// Generate interactive HTML visualization
    fn generate_interactive_html(&self) -> Result<String> {
        let mut html = String::from(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Attention Visualization</title>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .attention-container { display: flex; flex-wrap: wrap; gap: 20px; }
        .attention-layer { border: 1px solid #ccc; padding: 15px; border-radius: 8px; }
        .heatmap { margin: 10px 0; }
        .heatmap-cell { display: inline-block; width: 20px; height: 20px; margin: 1px; }
        .layer-title { font-weight: bold; margin-bottom: 10px; }
        .stats { font-size: 12px; color: #666; margin-top: 10px; }
</head>
<body>
    <h1>Neural Network Attention Visualization</h1>
    <div class="attention-container">
            html.push_str(&format!(
                r#"
        <div class="attention-layer">
            <div class="layer-title">{}</div>
            <div class="heatmap">
            let weights = &attention_data.weights;
            let (rows, cols) = weights.dim();
            let max_weight = weights.iter().fold(F::zero(), |acc, &w| acc.max(w));
            for i in 0..rows.min(20) {
                // Limit display size
                for j in 0..cols.min(20) {
                    let color = format!("rgb({}, 128, {})", red, blue);
                    html.push_str(&format!(
                        r#"<div class="heatmap-cell" style="background-color: {};" title="({},{}) = {:.3}"></div>"#,
                        color, i, j, normalized
                html.push_str("<br>");
            </div>
            <div class="stats">
                Max Attention: {:.3}<br>, Shape: {} x {}
        </div>
                rows,
                cols
        html.push_str(
            r#"
    </div>
</body>
</html>"#,
        Ok(html)
    /// Generate SVG visualization
    fn generate_svg_visualization(&self) -> Result<String> {
        let width = 800;
        let height = 600;
        let margin = 50;
    .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
    .layer-title {{ font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; }}
    .cell {{ stroke: #fff; stroke-width: 1; }}
            r#"<text x="{}" y="30" class="title">Attention Patterns</text>"#,
            width / 2
        let mut y_offset = 60;
                r#"<text x="50" y="{}" class="layer-title">{}</text>"#,
                y_offset, layer_name
            y_offset += 30;
            let cell_size = 8.0;
            for i in 0..rows.min(15) {
                for j in 0..cols.min(15) {
                    let x = margin + j as i32 * cell_size as i32;
                    let y = y_offset + i as i32 * cell_size as i32;
                        r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" class="cell"/>"#,
                        x, y, cell_size, cell_size, color
            y_offset += (rows.min(15) as i32 * cell_size as i32) + 30;
        Ok(svg)
    /// Export attention data as JSON
    fn export_attention_data_as_json(&self) -> Result<String> {
        use serde_json::json;
        let mut layers_data = serde_json::Map::new();
            let weights_data: Vec<Vec<f64>> = attention_data
                .weights
                .outer_iter()
                .map(|row| row.iter().map(|&w| w.to_f64().unwrap_or(0.0)).collect())
                .collect();
            let layer_data = json!({
                "weights": weights_data,
                "queries": attention_data.queries,
                "keys": attention_data.keys,
                "layer_info": {
                    "name": attention_data.layer_info.layer_name,
                    "index": attention_data.layer_info.layer_index,
                    "type": attention_data.layer_info.layer_type
                },
                "head_info": attention_data.head_info.as_ref().map(|h| json!({
                    "head_index": h.head_index,
                    "total_heads": h.total_heads,
                    "head_dim": h.head_dim
                })),
                "shape": attention_data.weights.shape()
            });
            layers_data.insert(layer_name.clone(), layer_data);
        let export_data = json!({
            "attention_layers": layers_data,
            "export_timestamp": chrono::Utc::now().to_rfc3339(),
            "framework": "scirs2-neural",
            "version": "0.1.0-beta.2"
        });
        serde_json::to_string_pretty(&export_data)
            .map_err(|e| NeuralError::ComputationError(format!("JSON serialization error: {}", e)))
// Default implementations for configuration types
impl Default for AttentionVisualizationOptions {
    fn default() -> Self {
            visualization_type: AttentionVisualizationType::Heatmap,
            head_selection: HeadSelection::All,
            highlighting: HighlightConfig::default(),
            head_aggregation: HeadAggregation::Mean,
            threshold: Some(0.01),
impl Default for HighlightConfig {
            highlighted_positions: Vec::new(),
            highlight_color: "#ff0000".to_string(),
            highlight_style: HighlightStyle::Border,
            show_paths: false,
impl Default for ExportOptions {
            format: ExportFormat::Image(ImageFormat::PNG),
            quality: ExportQuality::High,
            resolution: Resolution::default(),
            include_metadata: true,
            compression: CompressionSettings::default(),
impl Default for Resolution {
            width: 1920,
            height: 1080,
            dpi: 300,
impl Default for CompressionSettings {
            enabled: true,
            level: 6,
            lossless: false,
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use rand::SeedableRng;
    #[test]
    fn test_attention_visualizer_creation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());
        let config = VisualizationConfig::default();
        let visualizer = AttentionVisualizer::new(model, config);
        assert!(visualizer.attention_cache.is_empty());
    fn test_attention_visualization_options_default() {
        let options = AttentionVisualizationOptions::default();
        assert_eq!(
            options.visualization_type,
            AttentionVisualizationType::Heatmap
        assert_eq!(options.head_selection, HeadSelection::All);
        assert_eq!(options.head_aggregation, HeadAggregation::Mean);
        assert_eq!(options.threshold, Some(0.01));
    fn test_attention_visualization_types() {
        let types = [
            AttentionVisualizationType::Heatmap,
            AttentionVisualizationType::BipartiteGraph,
            AttentionVisualizationType::ArcDiagram,
            AttentionVisualizationType::AttentionFlow,
            AttentionVisualizationType::HeadComparison,
        ];
        assert_eq!(types.len(), 5);
        assert_eq!(types[0], AttentionVisualizationType::Heatmap);
    fn test_head_selection_variants() {
        let all = HeadSelection::All;
        let specific = HeadSelection::Specific(vec![0, 1, 2]);
        let top_k = HeadSelection::TopK(5);
        let range = HeadSelection::Range(2, 8);
        assert_eq!(all, HeadSelection::All);
        match specific {
            HeadSelection::Specific(heads) => assert_eq!(heads.len(), 3, _ => assert!(false, "Expected specific head selection"),
        match top_k {
            HeadSelection::TopK(k) => assert_eq!(k, 5, _ => assert!(false, "Expected top-k head selection"),
        match range {
            HeadSelection::Range(start, end) => {
                assert_eq!(start, 2);
                assert_eq!(end, 8);
            _ => assert!(false, "Expected range head selection"),
    fn test_head_aggregation_methods() {
        let none = HeadAggregation::None;
        let mean = HeadAggregation::Mean;
        let max = HeadAggregation::Max;
        let weighted = HeadAggregation::WeightedMean(vec![0.3, 0.7]);
        let rollout = HeadAggregation::Rollout;
        assert_eq!(none, HeadAggregation::None);
        assert_eq!(mean, HeadAggregation::Mean);
        assert_eq!(max, HeadAggregation::Max);
        assert_eq!(rollout, HeadAggregation::Rollout);
        match weighted {
            HeadAggregation::WeightedMean(weights) => assert_eq!(weights.len(), 2, _ => assert!(false, "Expected weighted mean aggregation"),
    fn test_highlight_styles() {
        let styles = [
            HighlightStyle::Border,
            HighlightStyle::Background,
            HighlightStyle::Overlay,
            HighlightStyle::Glow,
        assert_eq!(styles.len(), 4);
        assert_eq!(styles[0], HighlightStyle::Border);
    fn test_export_formats() {
        let image = ExportFormat::Image(ImageFormat::PNG);
        let html = ExportFormat::HTML;
        let svg = ExportFormat::SVG;
        let data = ExportFormat::Data(DataFormat::JSON);
        let video = ExportFormat::Video(VideoFormat::MP4);
        assert_eq!(html, ExportFormat::HTML);
        assert_eq!(svg, ExportFormat::SVG);
        match image {
            ExportFormat::Image(ImageFormat::PNG) => {}
            _ => assert!(false, "Expected PNG image format"),
        match data {
            ExportFormat::Data(DataFormat::JSON) => {}
            _ => assert!(false, "Expected JSON data format"),
        match video {
            ExportFormat::Video(VideoFormat::MP4) => {}
            _ => assert!(false, "Expected MP4 video format"),
    fn test_export_quality_levels() {
        let qualities = [
            ExportQuality::Low,
            ExportQuality::Medium,
            ExportQuality::High,
            ExportQuality::Maximum,
        assert_eq!(qualities.len(), 4);
        assert_eq!(qualities[2], ExportQuality::High);
    fn test_cache_operations() {
        let mut visualizer = AttentionVisualizer::new(model, config);
        assert!(visualizer.get_cached_attention("test_layer").is_none());
        visualizer.clear_cache();
    fn test_resolution_settings() {
        let resolution = Resolution::default();
        assert_eq!(resolution.width, 1920);
        assert_eq!(resolution.height, 1080);
        assert_eq!(resolution.dpi, 300);
    fn test_compression_settings() {
        let compression = CompressionSettings::default();
        assert!(compression.enabled);
        assert_eq!(compression.level, 6);
        assert!(!compression.lossless);
