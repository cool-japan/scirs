//! Configuration types and settings for neural network visualization
//!
//! This module provides comprehensive configuration structures for all aspects
//! of visualization including output formats, styling, interactivity, and performance.

use serde::Serialize;
use std::path::PathBuf;

/// Visualization configuration
#[derive(Debug, Clone, Serialize)]
pub struct VisualizationConfig {
    /// Output directory for generated visualizations
    pub output_dir: PathBuf,
    /// Image format for static outputs
    pub image_format: ImageFormat,
    /// Interactive visualization settings
    pub interactive: InteractiveConfig,
    /// Color scheme and styling
    pub style: StyleConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
}

/// Supported image formats for visualization output
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum ImageFormat {
    /// PNG format (recommended for web)
    PNG,
    /// SVG format (vector graphics)
    SVG,
    /// PDF format (publication quality)
    PDF,
    /// HTML format (interactive)
    HTML,
    /// JSON format (data export)
    JSON,
}

/// Interactive visualization configuration
#[derive(Debug, Clone, Serialize)]
pub struct InteractiveConfig {
    /// Enable interactive features
    pub enable_interaction: bool,
    /// Web server port for live visualization
    pub server_port: u16,
    /// Auto-refresh interval in milliseconds
    pub refresh_interval_ms: u32,
    /// Enable real-time updates
    pub real_time_updates: bool,
    /// Maximum data points to display
    pub max_data_points: usize,
}

/// Style configuration for visualizations
#[derive(Debug, Clone, Serialize)]
pub struct StyleConfig {
    /// Color palette
    pub color_palette: ColorPalette,
    /// Font settings
    pub font: FontConfig,
    /// Layout settings
    pub layout: LayoutConfig,
    /// Theme selection
    pub theme: Theme,
}

/// Color palette for visualizations
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ColorPalette {
    /// Default SciRS2 colors
    Default,
    /// Colorblind-friendly palette
    ColorblindFriendly,
    /// High contrast palette
    HighContrast,
    /// Grayscale palette
    Grayscale,
    /// Custom palette
    Custom(Vec<String>),
}

/// Font configuration
#[derive(Debug, Clone, Serialize)]
pub struct FontConfig {
    /// Font family
    pub family: String,
    /// Base font size
    pub size: u32,
    /// Title font size multiplier
    pub title_scale: f32,
    /// Label font size multiplier
    pub label_scale: f32,
}

/// Layout configuration
#[derive(Debug, Clone, Serialize)]
pub struct LayoutConfig {
    /// Canvas width
    pub width: u32,
    /// Canvas height
    pub height: u32,
    /// Margin settings
    pub margins: Margins,
    /// Grid settings
    pub grid: GridConfig,
}

/// Margin configuration
#[derive(Debug, Clone, Serialize)]
pub struct Margins {
    /// Top margin
    pub top: u32,
    /// Bottom margin
    pub bottom: u32,
    /// Left margin
    pub left: u32,
    /// Right margin
    pub right: u32,
}

/// Grid configuration
#[derive(Debug, Clone, Serialize)]
pub struct GridConfig {
    /// Show grid lines
    pub show_grid: bool,
    /// Grid line color
    pub grid_color: String,
    /// Grid line width
    pub grid_width: u32,
    /// Grid opacity
    pub grid_opacity: f32,
}

/// Visualization theme
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Theme {
    /// Light theme
    Light,
    /// Dark theme
    Dark,
    /// Auto theme (system preference)
    Auto,
    /// Custom theme
    Custom(CustomTheme),
}

/// Custom theme configuration
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CustomTheme {
    /// Background color
    pub background: String,
    /// Text color
    pub text: String,
    /// Primary accent color
    pub primary: String,
    /// Secondary accent color
    pub secondary: String,
    /// Success color
    pub success: String,
    /// Warning color
    pub warning: String,
    /// Error color
    pub error: String,
}

/// Performance configuration for visualizations
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceConfig {
    /// Maximum number of points per plot
    pub max_points_per_plot: usize,
    /// Enable data downsampling
    pub enable_downsampling: bool,
    /// Downsampling strategy
    pub downsampling_strategy: DownsamplingStrategy,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size limit in MB
    pub cache_size_mb: usize,
}

/// Downsampling strategy for large datasets
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum DownsamplingStrategy {
    /// Take every nth point
    Uniform,
    /// Largest triangle three bucket algorithm
    LTTB,
    /// Min-max decimation
    MinMax,
    /// Statistical sampling
    Statistical,
}

// Default implementations for all configuration types

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            output_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            image_format: ImageFormat::SVG,
            interactive: InteractiveConfig::default(),
            style: StyleConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            enable_interaction: true,
            server_port: 8080,
            refresh_interval_ms: 1000,
            real_time_updates: true,
            max_data_points: 10000,
        }
    }
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            color_palette: ColorPalette::Default,
            font: FontConfig::default(),
            layout: LayoutConfig::default(),
            theme: Theme::Light,
        }
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "Arial, sans-serif".to_string(),
            size: 12,
            title_scale: 1.5,
            label_scale: 0.9,
        }
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            margins: Margins::default(),
            grid: GridConfig::default(),
        }
    }
}

impl Default for Margins {
    fn default() -> Self {
        Self {
            top: 40,
            bottom: 60,
            left: 80,
            right: 40,
        }
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            show_grid: true,
            grid_color: "#e0e0e0".to_string(),
            grid_width: 1,
            grid_opacity: 0.5,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_points_per_plot: 10000,
            enable_downsampling: true,
            downsampling_strategy: DownsamplingStrategy::Uniform,
            enable_caching: true,
            cache_size_mb: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VisualizationConfig::default();
        assert_eq!(config.image_format, ImageFormat::SVG);
        assert_eq!(config.interactive.server_port, 8080);
        assert_eq!(config.style.theme, Theme::Light);
    }

    #[test]
    fn test_custom_color_palette() {
        let custom_colors = vec![
            "#ff0000".to_string(),
            "#00ff00".to_string(),
            "#0000ff".to_string(),
        ];
        let palette = ColorPalette::Custom(custom_colors.clone());
        match palette {
            ColorPalette::Custom(colors) => assert_eq!(colors, custom_colors),
            _ => panic!("Expected custom color palette"),
        }
    }

    #[test]
    fn test_interactive_config() {
        let mut config = InteractiveConfig::default();
        config.enable_interaction = false;
        config.server_port = 3000;

        assert!(!config.enable_interaction);
        assert_eq!(config.server_port, 3000);
    }

    #[test]
    fn test_performance_config() {
        let config = PerformanceConfig::default();
        assert!(config.enable_downsampling);
        assert_eq!(config.downsampling_strategy, DownsamplingStrategy::Uniform);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_theme_variants() {
        let light = Theme::Light;
        let dark = Theme::Dark;
        let auto = Theme::Auto;

        assert_eq!(light, Theme::Light);
        assert_eq!(dark, Theme::Dark);
        assert_eq!(auto, Theme::Auto);
    }

    #[test]
    fn test_custom_theme() {
        let custom = CustomTheme {
            background: "#ffffff".to_string(),
            text: "#000000".to_string(),
            primary: "#007bff".to_string(),
            secondary: "#6c757d".to_string(),
            success: "#28a745".to_string(),
            warning: "#ffc107".to_string(),
            error: "#dc3545".to_string(),
        };

        let theme = Theme::Custom(custom.clone());
        match theme {
            Theme::Custom(t) => {
                assert_eq!(t.background, "#ffffff");
                assert_eq!(t.primary, "#007bff");
            }
            _ => panic!("Expected custom theme"),
        }
    }
}
