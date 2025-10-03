//! Core types for dendrogram visualization
//!
//! This module contains all the fundamental types and configurations
//! used for visualizing hierarchical clustering results.

use scirs2_core::numeric::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};

/// Color scheme options for dendrogram visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    /// Default color scheme with standard cluster colors
    Default,
    /// High contrast colors for better visibility
    HighContrast,
    /// Viridis color map (blue to green to yellow)
    Viridis,
    /// Plasma color map (purple to pink to yellow)
    Plasma,
    /// Grayscale for black and white publications
    Grayscale,
}

/// Color threshold configuration for dendrogram visualization
#[derive(Debug, Clone)]
pub struct ColorThreshold<F: Float> {
    /// Threshold value for coloring clusters
    pub threshold: F,
    /// Color to use above threshold
    pub above_color: String,
    /// Color to use below threshold
    pub below_color: String,
    /// Whether to use automatic threshold based on cluster count
    pub auto_threshold: bool,
    /// Number of clusters for automatic threshold (if auto_threshold is true)
    pub target_clusters: Option<usize>,
}

impl<F: Float + FromPrimitive> Default for ColorThreshold<F> {
    fn default() -> Self {
        Self {
            threshold: F::zero(),
            above_color: "#1f77b4".to_string(), // Blue
            below_color: "#ff7f0e".to_string(), // Orange
            auto_threshold: true,
            target_clusters: Some(4),
        }
    }
}

/// Enhanced dendrogram visualization configuration
#[derive(Debug, Clone)]
pub struct DendrogramConfig<F: Float> {
    /// Color scheme to use
    pub color_scheme: ColorScheme,
    /// Color threshold configuration
    pub color_threshold: ColorThreshold<F>,
    /// Whether to show cluster labels
    pub show_labels: bool,
    /// Whether to show distance labels on branches
    pub show_distances: bool,
    /// Orientation of the dendrogram
    pub orientation: DendrogramOrientation,
    /// Line width for dendrogram branches
    pub line_width: f32,
    /// Font size for labels
    pub font_size: f32,
    /// Whether to truncate the dendrogram at a certain level
    pub truncate_mode: Option<TruncateMode>,
    /// Advanced styling options
    pub styling: DendrogramStyling,
}

/// Advanced styling options for dendrograms
#[derive(Debug, Clone)]
pub struct DendrogramStyling {
    /// Background color
    pub background_color: String,
    /// Branch style (solid, dashed, dotted)
    pub branch_style: BranchStyle,
    /// Node marker style
    pub node_markers: NodeMarkerStyle,
    /// Label styling
    pub label_style: LabelStyle,
    /// Grid options
    pub grid: Option<GridStyle>,
    /// Shadow effects
    pub shadows: bool,
    /// Border around the plot
    pub border: Option<BorderStyle>,
}

/// Branch styling options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

/// Node marker styling
#[derive(Debug, Clone)]
pub struct NodeMarkerStyle {
    /// Show markers at internal nodes
    pub show_internal_nodes: bool,
    /// Show markers at leaf nodes
    pub show_leaf_nodes: bool,
    /// Marker shape
    pub markershape: MarkerShape,
    /// Marker size
    pub marker_size: f32,
    /// Marker color
    pub marker_color: String,
}

/// Available marker shapes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Cross,
}

/// Label styling options
#[derive(Debug, Clone)]
pub struct LabelStyle {
    /// Label font family
    pub font_family: String,
    /// Label font weight
    pub font_weight: FontWeight,
    /// Label color
    pub color: String,
    /// Label rotation angle in degrees
    pub rotation: f32,
    /// Label background
    pub background: Option<String>,
    /// Label padding
    pub padding: f32,
}

/// Font weight options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
}

/// Grid styling options
#[derive(Debug, Clone)]
pub struct GridStyle {
    /// Show horizontal grid lines
    pub show_horizontal: bool,
    /// Show vertical grid lines
    pub show_vertical: bool,
    /// Grid line color
    pub color: String,
    /// Grid line width
    pub line_width: f32,
    /// Grid line style
    pub style: BranchStyle,
}

/// Border styling options
#[derive(Debug, Clone)]
pub struct BorderStyle {
    /// Border color
    pub color: String,
    /// Border width
    pub width: f32,
    /// Border radius
    pub radius: f32,
}

impl Default for DendrogramStyling {
    fn default() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            branch_style: BranchStyle::Solid,
            node_markers: NodeMarkerStyle::default(),
            label_style: LabelStyle::default(),
            grid: None,
            shadows: false,
            border: None,
        }
    }
}

impl Default for NodeMarkerStyle {
    fn default() -> Self {
        Self {
            show_internal_nodes: false,
            show_leaf_nodes: true,
            markershape: MarkerShape::Circle,
            marker_size: 4.0,
            marker_color: "#333333".to_string(),
        }
    }
}

impl Default for LabelStyle {
    fn default() -> Self {
        Self {
            font_family: "Arial, sans-serif".to_string(),
            font_weight: FontWeight::Normal,
            color: "#000000".to_string(),
            rotation: 0.0,
            background: None,
            padding: 2.0,
        }
    }
}

impl Default for GridStyle {
    fn default() -> Self {
        Self {
            show_horizontal: true,
            show_vertical: false,
            color: "#e0e0e0".to_string(),
            line_width: 0.5,
            style: BranchStyle::Solid,
        }
    }
}

impl Default for BorderStyle {
    fn default() -> Self {
        Self {
            color: "#cccccc".to_string(),
            width: 1.0,
            radius: 0.0,
        }
    }
}

/// Dendrogram orientation options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DendrogramOrientation {
    /// Top to bottom (leaves at bottom)
    Top,
    /// Bottom to top (leaves at top)
    Bottom,
    /// Left to right (leaves on right)
    Left,
    /// Right to left (leaves on left)
    Right,
}

/// Truncation modes for large dendrograms
#[derive(Debug, Clone)]
pub enum TruncateMode {
    /// Show only the last N merges
    LastMerges(usize),
    /// Show only merges above a distance threshold
    DistanceThreshold(f64),
    /// Show only the top N levels of the tree
    TopLevels(usize),
}

impl<F: Float + FromPrimitive> Default for DendrogramConfig<F> {
    fn default() -> Self {
        Self {
            color_scheme: ColorScheme::Default,
            color_threshold: ColorThreshold::default(),
            show_labels: true,
            show_distances: false,
            orientation: DendrogramOrientation::Top,
            line_width: 1.0,
            font_size: 10.0,
            truncate_mode: None,
            styling: DendrogramStyling::default(),
        }
    }
}

/// Enhanced dendrogram visualization data structure
#[derive(Debug, Clone)]
pub struct DendrogramPlot<F: Float> {
    /// Branch coordinates for drawing
    pub branches: Vec<Branch<F>>,
    /// Leaf positions and labels
    pub leaves: Vec<Leaf>,
    /// Color assignments for each branch
    pub colors: Vec<String>,
    /// Legend information
    pub legend: Vec<LegendEntry>,
    /// Plot bounds (min_x, max_x, min_y, max_y)
    pub bounds: (F, F, F, F),
    /// Configuration used to create this plot
    pub config: DendrogramConfig<F>,
}

/// Branch representation for visualization
#[derive(Debug, Clone)]
pub struct Branch<F: Float> {
    /// Starting point coordinates (x, y)
    pub start: (F, F),
    /// Ending point coordinates (x, y)
    pub end: (F, F),
    /// Distance value at this merge
    pub distance: F,
    /// Cluster ID or merge index
    pub cluster_id: Option<usize>,
    /// Color for this branch
    pub color: String,
    /// Line width override (if different from default)
    pub line_width: Option<f32>,
}

/// Leaf node representation for visualization
#[derive(Debug, Clone)]
pub struct Leaf {
    /// Position coordinates (x, y)
    pub position: (f64, f64),
    /// Label text to display
    pub label: String,
    /// Color for the leaf
    pub color: String,
    /// Original data index
    pub data_index: usize,
}

/// Legend entry for cluster visualization
#[derive(Debug, Clone)]
pub struct LegendEntry {
    /// Color swatch
    pub color: String,
    /// Descriptive label
    pub label: String,
    /// Distance threshold (if applicable)
    pub threshold: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_scheme_variants() {
        let schemes = [
            ColorScheme::Default,
            ColorScheme::HighContrast,
            ColorScheme::Viridis,
            ColorScheme::Plasma,
            ColorScheme::Grayscale,
        ];

        for scheme in &schemes {
            assert!(format!("{:?}", scheme).len() > 0);
        }
    }

    #[test]
    fn test_color_threshold_default() {
        let threshold: ColorThreshold<f64> = ColorThreshold::default();
        assert_eq!(threshold.threshold, 0.0);
        assert!(threshold.auto_threshold);
        assert_eq!(threshold.target_clusters, Some(4));
    }

    #[test]
    fn test_dendrogram_config_default() {
        let config: DendrogramConfig<f64> = DendrogramConfig::default();
        assert_eq!(config.color_scheme, ColorScheme::Default);
        assert!(config.show_labels);
        assert!(!config.show_distances);
        assert_eq!(config.orientation, DendrogramOrientation::Top);
    }

    #[test]
    fn test_styling_defaults() {
        let styling = DendrogramStyling::default();
        assert_eq!(styling.background_color, "#ffffff");
        assert_eq!(styling.branch_style, BranchStyle::Solid);
        assert!(!styling.shadows);
    }

    #[test]
    fn test_branch_style_variants() {
        let styles = [
            BranchStyle::Solid,
            BranchStyle::Dashed,
            BranchStyle::Dotted,
            BranchStyle::DashDot,
        ];

        for style in &styles {
            assert!(format!("{:?}", style).len() > 0);
        }
    }

    #[test]
    fn test_marker_shape_variants() {
        let shapes = [
            MarkerShape::Circle,
            MarkerShape::Square,
            MarkerShape::Triangle,
            MarkerShape::Diamond,
            MarkerShape::Cross,
        ];

        for shape in &shapes {
            assert!(format!("{:?}", shape).len() > 0);
        }
    }

    #[test]
    fn test_truncate_mode_variants() {
        let modes = [
            TruncateMode::LastMerges(10),
            TruncateMode::DistanceThreshold(0.5),
            TruncateMode::TopLevels(5),
        ];

        for mode in &modes {
            assert!(format!("{:?}", mode).len() > 0);
        }
    }
}
