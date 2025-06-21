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
use std::path::PathBuf;

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
}

/// Attention head information
#[derive(Debug, Clone, Serialize)]
pub struct HeadInfo {
    /// Head index
    pub head_index: usize,
    /// Total number of heads
    pub total_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

/// Layer information for attention
#[derive(Debug, Clone, Serialize)]
pub struct LayerInfo {
    /// Layer name
    pub layer_name: String,
    /// Layer index
    pub layer_index: usize,
    /// Layer type
    pub layer_type: String,
}

/// Attention visualization options
#[derive(Debug, Clone, Serialize)]
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
}

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
}

/// Head selection options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum HeadSelection {
    /// All heads
    All,
    /// Specific heads
    Specific(Vec<usize>),
    /// Top-k heads by attention entropy
    TopK(usize),
    /// Head range
    Range(usize, usize),
}

/// Highlighting configuration
#[derive(Debug, Clone, Serialize)]
pub struct HighlightConfig {
    /// Highlight specific tokens/positions
    pub highlighted_positions: Vec<usize>,
    /// Highlight color
    pub highlight_color: String,
    /// Highlight style
    pub highlight_style: HighlightStyle,
    /// Show attention paths
    pub show_paths: bool,
}

/// Highlight style options
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum HighlightStyle {
    /// Border highlighting
    Border,
    /// Background highlighting
    Background,
    /// Color overlay
    Overlay,
    /// Glow effect
    Glow,
}

/// Head aggregation methods
#[derive(Debug, Clone, PartialEq, Serialize)]
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
}

/// Visualization export formats
#[derive(Debug, Clone, Serialize)]
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
}

/// Export format options
#[derive(Debug, Clone, PartialEq, Serialize)]
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
}

/// Data export formats
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DataFormat {
    /// JSON format
    JSON,
    /// CSV format
    CSV,
    /// NumPy format
    NPY,
    /// HDF5 format
    HDF5,
}

/// Video formats for animated visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum VideoFormat {
    /// MP4 format
    MP4,
    /// WebM format
    WebM,
    /// GIF format
    GIF,
}

/// Export quality settings
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ExportQuality {
    /// Low quality (faster, smaller files)
    Low,
    /// Medium quality
    Medium,
    /// High quality
    High,
    /// Maximum quality (slower, larger files)
    Maximum,
}

/// Resolution settings
#[derive(Debug, Clone, Serialize)]
pub struct Resolution {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// DPI (dots per inch)
    pub dpi: u32,
}

/// Compression settings
#[derive(Debug, Clone, Serialize)]
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,
    /// Compression level (0-9)
    pub level: u8,
    /// Lossless compression
    pub lossless: bool,
}

/// Attention statistics for analysis
#[derive(Debug, Clone, Serialize)]
pub struct AttentionStatistics<F: Float + Debug> {
    /// Layer name
    pub layer_name: String,
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
}

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
        }
    }

    /// Get cached attention data for a layer
    pub fn get_cached_attention(&self, layer_name: &str) -> Option<&AttentionData<F>> {
        self.attention_cache.get(layer_name)
    }

    /// Clear the attention cache
    pub fn clear_cache(&mut self) {
        self.attention_cache.clear();
    }

    /// Get attention statistics for all cached layers
    pub fn get_attention_statistics(&self) -> Result<Vec<AttentionStatistics<F>>> {
        let mut stats = Vec::new();

        for (layer_name, attention_data) in &self.attention_cache {
            let layer_stats = self.compute_attention_statistics(layer_name, attention_data)?;
            stats.push(layer_stats);
        }

        Ok(stats)
    }

    /// Update the visualization configuration
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config;
    }

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
            _ => Err(NeuralError::NotImplementedError(
                "Export format not yet implemented".to_string(),
            )),
        }
    }

    fn extract_attention_patterns(&mut self, _input: &ArrayD<F>) -> Result<()> {
        // TODO: Implement attention extraction
        // This would run forward pass and extract attention weights from attention layers
        Err(NeuralError::NotImplementedError(
            "Attention extraction not yet implemented".to_string(),
        ))
    }

    fn generate_attention_heatmap(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement attention heatmap generation
        Err(NeuralError::NotImplementedError(
            "Attention heatmap not yet implemented".to_string(),
        ))
    }

    fn generate_bipartite_graph(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement bipartite graph generation
        Err(NeuralError::NotImplementedError(
            "Bipartite graph not yet implemented".to_string(),
        ))
    }

    fn generate_arc_diagram(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement arc diagram generation
        Err(NeuralError::NotImplementedError(
            "Arc diagram not yet implemented".to_string(),
        ))
    }

    fn generate_attention_flow(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement attention flow generation
        Err(NeuralError::NotImplementedError(
            "Attention flow not yet implemented".to_string(),
        ))
    }

    fn generate_head_comparison(
        &self,
        _options: &AttentionVisualizationOptions,
    ) -> Result<Vec<PathBuf>> {
        // TODO: Implement head comparison generation
        Err(NeuralError::NotImplementedError(
            "Head comparison not yet implemented".to_string(),
        ))
    }

    fn compute_attention_statistics(
        &self,
        layer_name: &str,
        attention_data: &AttentionData<F>,
    ) -> Result<AttentionStatistics<F>> {
        let weights = &attention_data.weights;
        let total_weights = weights.len();

        if total_weights == 0 {
            return Err(NeuralError::InvalidArgument(
                "Empty attention weights".to_string(),
            ));
        }

        // Compute basic statistics
        let mut sum = F::zero();
        let mut max_weight = F::neg_infinity();
        let mut zero_count = 0;

        for &weight in weights.iter() {
            sum = sum + weight;
            if weight > max_weight {
                max_weight = weight;
            }
            if weight.abs() < F::from(1e-6).unwrap_or(F::zero()) {
                zero_count += 1;
            }
        }

        let mean_attention = sum / F::from(total_weights).unwrap_or(F::one());
        let sparsity = zero_count as f64 / total_weights as f64;

        // Compute entropy (simplified)
        let mut entropy = 0.0;
        for &weight in weights.iter() {
            let prob = weight.to_f64().unwrap_or(0.0);
            if prob > 1e-10 {
                entropy -= prob * prob.ln();
            }
        }

        // Find top attended positions (simplified)
        let mut top_attended = Vec::new();
        if let Some((rows, cols)) = weights.dim().into() {
            for i in 0..std::cmp::min(5, rows) {
                for j in 0..cols {
                    top_attended.push((i * cols + j, weights[[i, j]]));
                }
            }
        }
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
    }
}

// Default implementations for configuration types

impl Default for AttentionVisualizationOptions {
    fn default() -> Self {
        Self {
            visualization_type: AttentionVisualizationType::Heatmap,
            head_selection: HeadSelection::All,
            highlighting: HighlightConfig::default(),
            head_aggregation: HeadAggregation::Mean,
            threshold: Some(0.01),
        }
    }
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            highlighted_positions: Vec::new(),
            highlight_color: "#ff0000".to_string(),
            highlight_style: HighlightStyle::Border,
            show_paths: false,
        }
    }
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Image(ImageFormat::PNG),
            quality: ExportQuality::High,
            resolution: Resolution::default(),
            include_metadata: true,
            compression: CompressionSettings::default(),
        }
    }
}

impl Default for Resolution {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            dpi: 300,
        }
    }
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            level: 6,
            lossless: false,
        }
    }
}

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
    }

    #[test]
    fn test_attention_visualization_options_default() {
        let options = AttentionVisualizationOptions::default();
        assert_eq!(
            options.visualization_type,
            AttentionVisualizationType::Heatmap
        );
        assert_eq!(options.head_selection, HeadSelection::All);
        assert_eq!(options.head_aggregation, HeadAggregation::Mean);
        assert_eq!(options.threshold, Some(0.01));
    }

    #[test]
    fn test_attention_visualization_types() {
        let types = vec![
            AttentionVisualizationType::Heatmap,
            AttentionVisualizationType::BipartiteGraph,
            AttentionVisualizationType::ArcDiagram,
            AttentionVisualizationType::AttentionFlow,
            AttentionVisualizationType::HeadComparison,
        ];

        assert_eq!(types.len(), 5);
        assert_eq!(types[0], AttentionVisualizationType::Heatmap);
    }

    #[test]
    fn test_head_selection_variants() {
        let all = HeadSelection::All;
        let specific = HeadSelection::Specific(vec![0, 1, 2]);
        let top_k = HeadSelection::TopK(5);
        let range = HeadSelection::Range(2, 8);

        assert_eq!(all, HeadSelection::All);

        match specific {
            HeadSelection::Specific(heads) => assert_eq!(heads.len(), 3),
            _ => panic!("Expected specific head selection"),
        }

        match top_k {
            HeadSelection::TopK(k) => assert_eq!(k, 5),
            _ => panic!("Expected top-k head selection"),
        }

        match range {
            HeadSelection::Range(start, end) => {
                assert_eq!(start, 2);
                assert_eq!(end, 8);
            }
            _ => panic!("Expected range head selection"),
        }
    }

    #[test]
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
            HeadAggregation::WeightedMean(weights) => assert_eq!(weights.len(), 2),
            _ => panic!("Expected weighted mean aggregation"),
        }
    }

    #[test]
    fn test_highlight_styles() {
        let styles = vec![
            HighlightStyle::Border,
            HighlightStyle::Background,
            HighlightStyle::Overlay,
            HighlightStyle::Glow,
        ];

        assert_eq!(styles.len(), 4);
        assert_eq!(styles[0], HighlightStyle::Border);
    }

    #[test]
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
            _ => panic!("Expected PNG image format"),
        }

        match data {
            ExportFormat::Data(DataFormat::JSON) => {}
            _ => panic!("Expected JSON data format"),
        }

        match video {
            ExportFormat::Video(VideoFormat::MP4) => {}
            _ => panic!("Expected MP4 video format"),
        }
    }

    #[test]
    fn test_export_quality_levels() {
        let qualities = vec![
            ExportQuality::Low,
            ExportQuality::Medium,
            ExportQuality::High,
            ExportQuality::Maximum,
        ];

        assert_eq!(qualities.len(), 4);
        assert_eq!(qualities[2], ExportQuality::High);
    }

    #[test]
    fn test_cache_operations() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();
        let mut visualizer = AttentionVisualizer::new(model, config);

        assert!(visualizer.attention_cache.is_empty());
        assert!(visualizer.get_cached_attention("test_layer").is_none());

        visualizer.clear_cache();
        assert!(visualizer.attention_cache.is_empty());
    }

    #[test]
    fn test_resolution_settings() {
        let resolution = Resolution::default();
        assert_eq!(resolution.width, 1920);
        assert_eq!(resolution.height, 1080);
        assert_eq!(resolution.dpi, 300);
    }

    #[test]
    fn test_compression_settings() {
        let compression = CompressionSettings::default();
        assert!(compression.enabled);
        assert_eq!(compression.level, 6);
        assert!(!compression.lossless);
    }
}
