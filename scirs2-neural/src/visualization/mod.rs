//! Visualization tools for neural networks
//!
//! This module provides comprehensive visualization capabilities including:
//! - Network architecture visualization with interactive graphs
//! - Training curves and metrics plotting
//! - Layer activation maps and feature visualization
//! - Attention mechanisms visualization
//! - Interactive dashboards and real-time monitoring
//!
//! # Module Organization
//!
//! - [`config`] - Configuration types and settings for all visualization aspects
//! - [`network`] - Network architecture visualization and layout algorithms
//! - [`training`] - Training metrics, curves, and performance monitoring
//! - [`activations`] - Layer activation analysis and feature map visualization
//! - [`attention`] - Attention mechanism visualization and analysis
//!
//! # Basic Usage
//!
//! ```rust
//! use scirs2_neural::visualization::{VisualizationConfig, NetworkVisualizer};
//! use scirs2_neural::models::Sequential;
//! use scirs2_neural::layers::Dense;
//! use rand::SeedableRng;
//!
//! // Create a simple model
//! let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//! let mut model = Sequential::<f32>::new();
//! model.add_layer(Dense::new(784, 128, Some("relu"), &mut rng).unwrap());
//! model.add_layer(Dense::new(128, 10, Some("softmax"), &mut rng).unwrap());
//!
//! // Configure visualization
//! let config = VisualizationConfig::default();
//!
//! // Create network visualizer
//! let mut visualizer = NetworkVisualizer::new(model, config);
//!
//! // Generate architecture visualization
//! // Note: This is a placeholder - actual implementation coming soon
//! // let output_path = visualizer.visualize_architecture()?;
//! // println!("Network visualization saved to: {:?}", output_path);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Module declarations
pub mod activations;
pub mod attention;
pub mod config;
pub mod network;
pub mod training;

// Re-export main configuration types
pub use config::{
    ColorPalette, CustomTheme, DownsamplingStrategy, FontConfig, GridConfig, ImageFormat,
    InteractiveConfig, LayoutConfig, Margins, PerformanceConfig, StyleConfig, Theme,
    VisualizationConfig,
};

// Re-export network visualization types and components
pub use network::{
    ArrowStyle, BoundingBox, Connection, ConnectionType, ConnectionVisualProps, DataFlowInfo,
    LayerIOInfo, LayerInfo, LayerPosition, LayerVisualProps, LayoutAlgorithm, LineStyle,
    NetworkLayout, NetworkVisualizer, Point2D, Size2D, ThroughputInfo,
};

// Re-export training visualization types and components
pub use training::{
    AxisConfig, AxisScale, LineStyleConfig, MarkerConfig, MarkerShape, PlotConfig, PlotType,
    SeriesConfig, SystemMetrics, TickConfig, TickFormat, TrainingMetrics, TrainingVisualizer,
    UpdateMode,
};

// Re-export activation visualization types and components
pub use activations::{
    ActivationHistogram, ActivationNormalization, ActivationStatistics,
    ActivationVisualizationOptions, ActivationVisualizationType, ActivationVisualizer,
    ChannelAggregation, Colormap, FeatureMapInfo,
};

// Re-export attention visualization types and components
pub use attention::{
    AttentionData, AttentionStatistics, AttentionVisualizationOptions, AttentionVisualizationType,
    AttentionVisualizer, CompressionSettings, DataFormat, ExportFormat, ExportOptions,
    ExportQuality, HeadAggregation, HeadInfo, HeadSelection, HighlightConfig, HighlightStyle,
    Resolution, VideoFormat,
};

// Convenience type aliases for common use cases
/// Convenient type alias for network visualization
pub type NetworkViz<F> = NetworkVisualizer<F>;
/// Convenient type alias for training visualization  
pub type TrainingViz<F> = TrainingVisualizer<F>;
/// Convenient type alias for activation visualization
pub type ActivationViz<F> = ActivationVisualizer<F>;
/// Convenient type alias for attention visualization
pub type AttentionViz<F> = AttentionVisualizer<F>;

/// Combined visualization suite for comprehensive neural network analysis
pub struct VisualizationSuite<F>
where
    F: num_traits::Float
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + 'static
        + num_traits::FromPrimitive
        + Send
        + Sync
        + serde::Serialize,
{
    /// Network architecture visualizer
    pub network: NetworkVisualizer<F>,
    /// Training metrics visualizer
    pub training: TrainingVisualizer<F>,
    /// Activation visualizer
    pub activation: ActivationVisualizer<F>,
    /// Attention visualizer
    pub attention: AttentionVisualizer<F>,
    /// Shared configuration
    config: VisualizationConfig,
}

impl<F> VisualizationSuite<F>
where
    F: num_traits::Float
        + std::fmt::Debug
        + ndarray::ScalarOperand
        + 'static
        + num_traits::FromPrimitive
        + Send
        + Sync
        + serde::Serialize,
{
    /// Create a new comprehensive visualization suite
    pub fn new(
        model: crate::models::sequential::Sequential<F>,
        config: VisualizationConfig,
    ) -> Self {
        let training = TrainingVisualizer::new(config.clone());
        // Create default/empty models for visualizers that need them but don't use them actively
        let activation = ActivationVisualizer::new(
            crate::models::sequential::Sequential::default(),
            config.clone(),
        );
        let attention = AttentionVisualizer::new(
            crate::models::sequential::Sequential::default(),
            config.clone(),
        );
        let network = NetworkVisualizer::new(model, config.clone());

        Self {
            network,
            training,
            activation,
            attention,
            config,
        }
    }

    /// Update configuration for all visualizers
    pub fn update_config(&mut self, config: VisualizationConfig) {
        self.config = config.clone();
        self.network.update_config(config.clone());
        self.training.update_config(config.clone());
        self.activation.update_config(config.clone());
        self.attention.update_config(config);
    }

    /// Get the current configuration
    pub fn get_config(&self) -> &VisualizationConfig {
        &self.config
    }

    /// Clear all caches across visualizers
    pub fn clear_all_caches(&mut self) {
        self.network.clear_cache();
        self.training.clear_history();
        self.activation.clear_cache();
        self.attention.clear_cache();
    }
}

/// Builder pattern for creating visualization configurations
pub struct VisualizationConfigBuilder {
    config: VisualizationConfig,
}

impl VisualizationConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: VisualizationConfig::default(),
        }
    }

    /// Set the output directory
    pub fn output_dir<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
        self.config.output_dir = path.into();
        self
    }

    /// Set the image format
    pub fn image_format(mut self, format: ImageFormat) -> Self {
        self.config.image_format = format;
        self
    }

    /// Set the color palette
    pub fn color_palette(mut self, palette: ColorPalette) -> Self {
        self.config.style.color_palette = palette;
        self
    }

    /// Set the theme
    pub fn theme(mut self, theme: Theme) -> Self {
        self.config.style.theme = theme;
        self
    }

    /// Enable or disable interactive features
    pub fn interactive(mut self, enable: bool) -> Self {
        self.config.interactive.enable_interaction = enable;
        self
    }

    /// Set canvas dimensions
    pub fn canvas_size(mut self, width: u32, height: u32) -> Self {
        self.config.style.layout.width = width;
        self.config.style.layout.height = height;
        self
    }

    /// Set performance settings
    pub fn max_points(mut self, max_points: usize) -> Self {
        self.config.performance.max_points_per_plot = max_points;
        self
    }

    /// Enable or disable downsampling
    pub fn downsampling(mut self, strategy: DownsamplingStrategy) -> Self {
        self.config.performance.enable_downsampling = true;
        self.config.performance.downsampling_strategy = strategy;
        self
    }

    /// Build the configuration
    pub fn build(self) -> VisualizationConfig {
        self.config
    }
}

impl Default for VisualizationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for visualization
pub mod utils {
    use super::*;

    /// Create a quick visualization configuration for prototyping
    pub fn quick_config() -> VisualizationConfig {
        VisualizationConfigBuilder::new()
            .canvas_size(800, 600)
            .theme(Theme::Light)
            .color_palette(ColorPalette::Default)
            .interactive(false)
            .build()
    }

    /// Create a high-quality visualization configuration for publication
    pub fn publication_config() -> VisualizationConfig {
        VisualizationConfigBuilder::new()
            .canvas_size(1920, 1080)
            .image_format(ImageFormat::PDF)
            .theme(Theme::Light)
            .color_palette(ColorPalette::ColorblindFriendly)
            .interactive(false)
            .build()
    }

    /// Create an interactive visualization configuration for dashboards
    pub fn dashboard_config() -> VisualizationConfig {
        VisualizationConfigBuilder::new()
            .canvas_size(1200, 800)
            .image_format(ImageFormat::HTML)
            .theme(Theme::Dark)
            .color_palette(ColorPalette::HighContrast)
            .interactive(true)
            .max_points(50000)
            .downsampling(DownsamplingStrategy::LTTB)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use rand::SeedableRng;

    #[test]
    fn test_visualization_suite_creation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = crate::models::sequential::Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();
        let _suite = VisualizationSuite::new(model, config);

        // Test passes if no panic occurs during creation
    }

    #[test]
    fn test_config_builder() {
        let config = VisualizationConfigBuilder::new()
            .canvas_size(1920, 1080)
            .theme(Theme::Dark)
            .color_palette(ColorPalette::HighContrast)
            .interactive(true)
            .build();

        assert_eq!(config.style.layout.width, 1920);
        assert_eq!(config.style.layout.height, 1080);
        assert_eq!(config.style.theme, Theme::Dark);
        assert_eq!(config.style.color_palette, ColorPalette::HighContrast);
        assert!(config.interactive.enable_interaction);
    }

    #[test]
    fn test_utility_configs() {
        let quick = utils::quick_config();
        assert_eq!(quick.style.layout.width, 800);
        assert_eq!(quick.style.layout.height, 600);
        assert_eq!(quick.style.theme, Theme::Light);
        assert!(!quick.interactive.enable_interaction);

        let publication = utils::publication_config();
        assert_eq!(publication.image_format, ImageFormat::PDF);
        assert_eq!(
            publication.style.color_palette,
            ColorPalette::ColorblindFriendly
        );

        let dashboard = utils::dashboard_config();
        assert_eq!(dashboard.image_format, ImageFormat::HTML);
        assert_eq!(dashboard.style.theme, Theme::Dark);
        assert!(dashboard.interactive.enable_interaction);
    }

    #[test]
    fn test_type_aliases() {
        // Test that type aliases compile correctly
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = crate::models::sequential::Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();

        let _network_viz: NetworkViz<f32> = NetworkVisualizer::new(model.clone(), config.clone());
        let _training_viz: TrainingViz<f32> = TrainingVisualizer::new(config.clone());
        let _activation_viz: ActivationViz<f32> =
            ActivationVisualizer::new(model.clone(), config.clone());
        let _attention_viz: AttentionViz<f32> = AttentionVisualizer::new(model, config);
    }

    #[test]
    fn test_suite_operations() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut model = crate::models::sequential::Sequential::<f32>::new();
        model.add_layer(Dense::new(10, 5, Some("relu"), &mut rng).unwrap());

        let config = VisualizationConfig::default();
        let mut suite = VisualizationSuite::new(model, config.clone());

        assert_eq!(
            suite.get_config().style.layout.width,
            config.style.layout.width
        );

        // Test cache clearing
        suite.clear_all_caches();

        // Test config update
        let new_config = VisualizationConfigBuilder::new()
            .canvas_size(1024, 768)
            .build();
        suite.update_config(new_config);
        assert_eq!(suite.get_config().style.layout.width, 1024);
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules are properly accessible
        use super::activations::*;
        use super::attention::*;
        use super::config::*;
        use super::network::*;
        use super::training::*;

        // Test default configurations
        let _viz_config = VisualizationConfig::default();
        let _plot_config = PlotConfig::default();
        let _activation_options = ActivationVisualizationOptions::default();
        let _attention_options = AttentionVisualizationOptions::default();
        let _export_options = ExportOptions::default();

        // Test enums
        let _image_format = ImageFormat::SVG;
        let _layout_algo = LayoutAlgorithm::Hierarchical;
        let _plot_type = PlotType::Line;
        let _colormap = Colormap::Viridis;
        let _attention_type = AttentionVisualizationType::Heatmap;
    }
}
