//! Data types for native plotting — structs, enums, and their Default impls.

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for native plotting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativePlotConfig {
    /// Canvas width in pixels
    pub width: usize,
    /// Canvas height in pixels
    pub height: usize,
    /// Enable interactive features
    pub enable_interactivity: bool,
    /// Enable animations
    pub enable_animations: bool,
    /// Animation frame rate (FPS)
    pub animation_fps: f64,
    /// Color scheme
    pub color_scheme: PlotColorScheme,
    /// Export quality
    pub export_quality: ExportQuality,
}

/// Color schemes for native plotting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlotColorScheme {
    /// Quantum theme (blue-cyan-purple)
    Quantum,
    /// Neuromorphic theme (green-yellow-red)
    Neuromorphic,
    /// AI theme (gold-orange-red)
    AI,
    /// Scientific theme (grayscale with highlights)
    Scientific,
    /// Custom color palette
    Custom(Vec<[u8; 3]>),
}

/// Export quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportQuality {
    /// Draft quality (fast rendering)
    Draft,
    /// Standard quality
    Standard,
    /// High quality (detailed rendering)
    High,
    /// Publication quality (maximum detail)
    Publication,
}

/// SVG canvas for native rendering
#[derive(Debug)]
pub struct SvgCanvas {
    /// Canvas dimensions
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// SVG elements
    pub(crate) elements: Vec<SvgElement>,
    /// Style definitions
    pub(crate) styles: HashMap<String, String>,
}

/// SVG element types
#[derive(Debug, Clone)]
pub enum SvgElement {
    /// Circle element
    Circle {
        cx: f64,
        cy: f64,
        r: f64,
        fill: String,
        stroke: String,
        stroke_width: f64,
        opacity: f64,
    },
    /// Line element
    Line {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        stroke: String,
        stroke_width: f64,
        opacity: f64,
    },
    /// Path element (for complex shapes)
    Path {
        d: String,
        fill: String,
        stroke: String,
        stroke_width: f64,
        opacity: f64,
    },
    /// Text element
    Text {
        x: f64,
        y: f64,
        content: String,
        font_size: f64,
        fill: String,
        text_anchor: String,
    },
    /// Group element (for hierarchical organization)
    Group {
        id: String,
        elements: Vec<SvgElement>,
        transform: String,
    },
}

/// Animation engine for dynamic visualizations
#[derive(Debug)]
pub struct AnimationEngine {
    /// Animation frames
    pub(crate) frames: Vec<AnimationFrame>,
    /// Current frame index
    pub(crate) current_frame: usize,
    /// Frame duration in milliseconds
    pub(crate) frame_duration: f64,
    /// Total animation duration
    pub(crate) total_duration: f64,
}

/// Animation frame data
#[derive(Debug, Clone)]
pub struct AnimationFrame {
    /// Frame timestamp
    pub timestamp: f64,
    /// Frame elements
    pub elements: Vec<SvgElement>,
    /// Frame-specific transformations
    pub transformations: Vec<Transformation>,
}

/// Animation transformations
#[derive(Debug, Clone)]
pub enum Transformation {
    /// Translation
    Translate { dx: f64, dy: f64 },
    /// Rotation
    Rotate { angle: f64, cx: f64, cy: f64 },
    /// Scale
    Scale { sx: f64, sy: f64 },
    /// Opacity fade
    Fade { from: f64, to: f64 },
    /// Color transition
    ColorTransition { from: String, to: String },
}

/// Interactive controller for user interaction
#[derive(Debug)]
pub struct InteractiveController {
    /// Zoom level
    pub(crate) zoom_level: f64,
    /// Pan offset
    pub(crate) pan_offset: (f64, f64),
    /// Selected elements
    pub(crate) selected_elements: Vec<String>,
    /// Hover state
    pub(crate) hover_element: Option<String>,
}

/// Native dendrogram plot
#[derive(Debug, Serialize, Deserialize)]
pub struct NativeDendrogramPlot {
    /// Dendrogram tree structure
    pub tree: DendrogramTree,
    /// Node positions
    pub node_positions: HashMap<String, (f64, f64)>,
    /// Branch lengths
    pub branch_lengths: HashMap<String, f64>,
    /// Quantum enhancement data
    pub quantum_enhancements: HashMap<String, f64>,
    /// Interactive features
    pub interactive_features: Vec<InteractiveFeature>,
}

/// Dendrogram tree structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DendrogramTree {
    /// Root node
    pub root: DendrogramNode,
    /// Total height
    pub height: f64,
    /// Leaf count
    pub leaf_count: usize,
}

/// Dendrogram node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DendrogramNode {
    /// Node ID
    pub id: String,
    /// Node height
    pub height: f64,
    /// Child nodes
    pub children: Vec<DendrogramNode>,
    /// Data point indices (for leaf nodes)
    pub data_indices: Vec<usize>,
    /// Quantum coherence at this node
    pub quantum_coherence: f64,
    /// Neuromorphic activity
    pub neuromorphic_activity: f64,
}

/// Interactive features for plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractiveFeature {
    /// Zoom and pan
    ZoomPan,
    /// Node selection
    NodeSelection,
    /// Tooltip on hover
    Tooltip,
    /// Real-time filtering
    RealTimeFilter,
    /// Animation controls
    AnimationControls,
    /// Export options
    ExportOptions,
}

/// 3D cluster plot for high-dimensional visualization
#[derive(Debug, Serialize, Deserialize)]
pub struct Native3DClusterPlot {
    /// 3D data points
    pub points_3d: Array2<f64>,
    /// Point colors based on clustering
    pub point_colors: Vec<[u8; 3]>,
    /// 3D centroids
    pub centroids_3d: Array2<f64>,
    /// Camera position
    pub camera: Camera3D,
    /// Lighting setup
    pub lighting: Lighting3D,
    /// Quantum field visualization
    pub quantum_field: QuantumField3D,
}

/// 3D camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera3D {
    /// Camera position
    pub position: [f64; 3],
    /// Look-at target
    pub target: [f64; 3],
    /// Up vector
    pub up: [f64; 3],
    /// Field of view
    pub fov: f64,
    /// Near and far clipping planes
    pub near: f64,
    pub far: f64,
}

/// 3D lighting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lighting3D {
    /// Ambient light intensity
    pub ambient: f64,
    /// Directional lights
    pub directional_lights: Vec<DirectionalLight>,
    /// Point lights
    pub point_lights: Vec<PointLight>,
}

/// Directional light
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalLight {
    /// Light direction
    pub direction: [f64; 3],
    /// Light intensity
    pub intensity: f64,
    /// Light color
    pub color: [f64; 3],
}

/// Point light
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointLight {
    /// Light position
    pub position: [f64; 3],
    /// Light intensity
    pub intensity: f64,
    /// Light color
    pub color: [f64; 3],
    /// Attenuation
    pub attenuation: f64,
}

/// 3D quantum field visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumField3D {
    /// Field strength at grid points
    pub field_strength: Array2<f64>,
    /// Field coherence
    pub coherence: Array2<f64>,
    /// Phase information
    pub phase: Array2<f64>,
    /// Entanglement connections
    pub entanglement_lines: Vec<([f64; 3], [f64; 3], f64)>,
}

// Supporting output data structures

/// Native cluster plot output
#[derive(Debug)]
pub struct NativeClusterPlot {
    /// Plot data
    pub data: Array2<f64>,
    /// Point elements
    pub point_elements: Vec<SvgElement>,
    /// Centroid elements
    pub centroid_elements: Vec<SvgElement>,
    /// Quantum enhancements per point
    pub quantum_enhancements: Vec<f64>,
    /// Plot bounds
    pub bounds: (f64, f64, f64, f64),
    /// Scale factors
    pub scale: (f64, f64),
}

/// Complete native visualization output
#[derive(Debug)]
pub struct NativeVisualizationOutput {
    /// Main cluster plot
    pub cluster_plot: NativeClusterPlot,
    /// Dendrogram (if applicable)
    pub dendrogram: Option<NativeDendrogramPlot>,
    /// 3D plot (if applicable)
    pub plot_3d: Option<Native3DClusterPlot>,
    /// Quantum coherence animation
    pub quantum_animation: Option<QuantumCoherenceAnimation>,
    /// Neuromorphic activity plot
    pub neuromorphic_plot: NeuromorphicActivityPlot,
    /// Interactive performance dashboard
    pub performance_dashboard: InteractivePerformanceDashboard,
    /// SVG content as string
    pub svg_content: String,
    /// Interactive JavaScript
    pub interactive_script: String,
}

/// Quantum coherence animation data
#[derive(Debug)]
pub struct QuantumCoherenceAnimation {
    /// Animation frames
    pub frames: Vec<QuantumCoherenceFrame>,
    /// Total duration in seconds
    pub duration: f64,
    /// Frames per second
    pub fps: f64,
}

/// Single quantum coherence frame
#[derive(Debug, Clone)]
pub struct QuantumCoherenceFrame {
    /// Frame timestamp
    pub timestamp: f64,
    /// Coherence visualization elements
    pub elements: Vec<SvgElement>,
    /// Quantum field strength
    pub field_strength: Array2<f64>,
}

/// Neuromorphic activity plot
#[derive(Debug)]
pub struct NeuromorphicActivityPlot {
    /// Activity matrix (time x neurons)
    pub activity_matrix: Array2<f64>,
    /// Spike trains
    pub spike_trains: Array2<f64>,
    /// Plasticity changes
    pub plasticity_changes: Array2<f64>,
    /// Time resolution
    pub time_resolution: f64,
}

/// Interactive performance dashboard
#[derive(Debug)]
pub struct InteractivePerformanceDashboard {
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Improvement factors
    pub improvements: HashMap<String, f64>,
    /// Metrics timeline
    pub metrics_timeline: Vec<MetricTimelinePoint>,
    /// Execution summary
    pub execution_summary: ExecutionSummary,
}

/// Timeline point for metrics
#[derive(Debug, Clone)]
pub struct MetricTimelinePoint {
    /// Timestamp
    pub timestamp: f64,
    /// Quantum coherence at this time
    pub quantum_coherence: f64,
    /// Neural adaptation rate
    pub neural_adaptation: f64,
    /// AI confidence
    pub ai_confidence: f64,
}

/// Execution summary
#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    /// Total execution time
    pub total_time: f64,
    /// Memory usage
    pub memory_usage: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Selected algorithm
    pub algorithm: String,
    /// Final confidence
    pub confidence: f64,
}

impl Default for NativePlotConfig {
    fn default() -> Self {
        Self {
            width: 1200,
            height: 800,
            enable_interactivity: true,
            enable_animations: true,
            animation_fps: 30.0,
            color_scheme: PlotColorScheme::Quantum,
            export_quality: ExportQuality::High,
        }
    }
}
