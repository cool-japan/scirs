//! Basic plot types and data structures for visualization
//!
//! This module contains the fundamental data structures used across the visualization system.

use crate::analysis::BifurcationPoint;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

/// Data structure for plotting 2D phase space
#[derive(Debug, Clone)]
pub struct PhaseSpacePlot {
    /// X coordinates
    pub x: Vec<f64>,
    /// Y coordinates
    pub y: Vec<f64>,
    /// Optional color values for each point
    pub colors: Option<Vec<f64>>,
    /// Trajectory metadata
    pub metadata: PlotMetadata,
}

/// Data structure for bifurcation diagrams
#[derive(Debug, Clone)]
pub struct BifurcationDiagram {
    /// Parameter values
    pub parameters: Vec<f64>,
    /// State values (can be multiple branches)
    pub states: Vec<Vec<f64>>,
    /// Stability information for each point
    pub stability: Vec<bool>,
    /// Bifurcation points
    pub bifurcation_points: Vec<BifurcationPoint>,
}

/// Visualization metadata
#[derive(Debug, Clone)]
pub struct PlotMetadata {
    /// Plot title
    pub title: String,
    /// X-axis label
    pub xlabel: String,
    /// Y-axis label
    pub ylabel: String,
    /// Additional annotations
    pub annotations: HashMap<String, String>,
}

impl Default for PlotMetadata {
    fn default() -> Self {
        Self {
            title: "Numerical Integration Result".to_string(),
            xlabel: "X".to_string(),
            ylabel: "Y".to_string(),
            annotations: HashMap::new(),
        }
    }
}

/// Field visualization for 2D vector fields
#[derive(Debug, Clone)]
pub struct VectorFieldPlot {
    /// Grid x coordinates
    pub x_grid: Array2<f64>,
    /// Grid y coordinates
    pub y_grid: Array2<f64>,
    /// X components of vectors
    pub u: Array2<f64>,
    /// Y components of vectors
    pub v: Array2<f64>,
    /// Magnitude for color coding
    pub magnitude: Array2<f64>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Heat map visualization for scalar fields
#[derive(Debug, Clone)]
pub struct HeatMapPlot {
    /// X coordinates
    pub x: Array1<f64>,
    /// Y coordinates
    pub y: Array1<f64>,
    /// Z values (scalar field)
    pub z: Array2<f64>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// 3D surface plot data
#[derive(Debug, Clone)]
pub struct SurfacePlot {
    /// X grid
    pub x: Array2<f64>,
    /// Y grid
    pub y: Array2<f64>,
    /// Z values
    pub z: Array2<f64>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Output format options
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// ASCII art for terminal
    ASCII,
    /// CSV data
    CSV,
    /// JSON data
    JSON,
    /// SVG graphics
    SVG,
}

/// Color scheme options
#[derive(Debug, Clone, Copy)]
pub enum ColorScheme {
    /// Viridis (default)
    Viridis,
    /// Plasma
    Plasma,
    /// Inferno
    Inferno,
    /// Grayscale
    Grayscale,
}

/// Statistical summary of plot data
#[derive(Debug, Clone)]
pub struct PlotStatistics {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

/// Parameter exploration plot for 2D parameter spaces
#[derive(Debug, Clone)]
pub struct ParameterExplorationPlot {
    /// X parameter grid
    pub x_grid: Array2<f64>,
    /// Y parameter grid  
    pub y_grid: Array2<f64>,
    /// Function values at each grid point
    pub z_values: Array2<f64>,
    /// Parameter ranges
    pub param_ranges: Vec<(f64, f64)>,
    /// Parameter names
    pub param_names: Vec<String>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Stability classification for attractors
#[derive(Debug, Clone, PartialEq)]
pub enum AttractorStability {
    /// Fixed point attractor
    FixedPoint,
    /// Period-2 cycle
    PeriodTwo,
    /// Higher-order periodic cycle
    Periodic(usize),
    /// Quasi-periodic attractor
    QuasiPeriodic,
    /// Chaotic attractor
    Chaotic,
    /// Unknown/undetermined
    Unknown,
}

/// Real-time bifurcation diagram
#[derive(Debug, Clone)]
pub struct RealTimeBifurcationPlot {
    /// Parameter values
    pub parameter_values: Vec<f64>,
    /// Attractor data for each parameter value and initial condition
    pub attractor_data: Vec<Vec<Vec<f64>>>,
    /// Stability classification for each attractor
    pub stability_data: Vec<Vec<AttractorStability>>,
    /// Parameter range
    pub parameter_range: (f64, f64),
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// 3D phase space trajectory
#[derive(Debug, Clone)]
pub struct PhaseSpace3D {
    /// X coordinates
    pub x: Vec<f64>,
    /// Y coordinates
    pub y: Vec<f64>,
    /// Z coordinates
    pub z: Vec<f64>,
    /// Optional color values for each point
    pub colors: Option<Vec<f64>>,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Sensitivity analysis plot
#[derive(Debug, Clone)]
pub struct SensitivityPlot {
    /// Parameter names
    pub parameter_names: Vec<String>,
    /// Sensitivity values for each parameter
    pub sensitivities: Vec<f64>,
    /// Base parameter values
    pub base_parameters: Vec<f64>,
    /// Base function value
    pub base_value: f64,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Interactive plot controls
#[derive(Debug, Clone)]
pub struct InteractivePlotControls {
    /// Zoom level
    pub zoom: f64,
    /// Pan offset (x, y)
    pub pan_offset: (f64, f64),
    /// Selected parameter ranges
    pub selected_ranges: Vec<(f64, f64)>,
    /// Animation frame
    pub current_frame: usize,
    /// Animation speed
    pub animation_speed: f64,
    /// Show/hide elements
    pub visibility_flags: std::collections::HashMap<String, bool>,
}

/// Parameter exploration methods
#[derive(Debug, Clone, Copy)]
pub enum ExplorationMethod {
    /// Grid-based scanning
    GridScan,
    /// Random sampling
    RandomSampling,
    /// Adaptive sampling with refinement
    AdaptiveSampling,
    /// Gradient-guided exploration
    GradientGuided,
}

/// Parameter region for refinement
#[derive(Debug, Clone)]
pub struct ParameterRegion {
    /// Center of the region
    pub center: Array1<f64>,
    /// Radius of the region (relative to parameter bounds)
    pub radius: f64,
}

/// Results of parameter exploration
#[derive(Debug, Clone)]
pub struct ParameterExplorationResult {
    /// Explored parameter points
    pub exploration_points: Vec<Array1<f64>>,
    /// System responses at each point
    pub response_values: Vec<Array1<f64>>,
    /// Full parameter grid (including failed evaluations)
    pub parameter_grid: Vec<Array1<f64>>,
    /// Convergence history for iterative methods
    pub convergence_history: Vec<Array1<f64>>,
    /// Exploration method used
    pub exploration_method: ExplorationMethod,
    /// Optimization metrics
    pub optimization_metrics: ExplorationMetrics,
}

/// Exploration performance metrics
#[derive(Debug, Clone)]
pub struct ExplorationMetrics {
    /// Maximum response norm found
    pub max_response_norm: f64,
    /// Minimum response norm found
    pub min_response_norm: f64,
    /// Mean response norm
    pub mean_response_norm: f64,
    /// Response variance
    pub response_variance: f64,
    /// Coverage efficiency (0-1)
    pub coverage_efficiency: f64,
}

/// Attractor analysis information
#[derive(Debug, Clone)]
pub struct AttractorInfo {
    /// Representative states of the attractor
    pub representative_states: Vec<f64>,
    /// Stability flag
    pub is_stable: bool,
    /// Period (0 for aperiodic)
    pub period: usize,
}

/// Dimension reduction methods
#[derive(Debug, Clone, Copy)]
pub enum DimensionReductionMethod {
    /// Principal Component Analysis
    PCA,
    /// t-Distributed Stochastic Neighbor Embedding
    TSNE,
    /// Uniform Manifold Approximation and Projection
    UMAP,
    /// Linear Discriminant Analysis
    LDA,
    /// Multidimensional Scaling
    MDS,
}

/// Clustering methods for visualization
#[derive(Debug, Clone, Copy)]
pub enum ClusteringMethod {
    /// K-means clustering
    KMeans { k: usize },
    /// DBSCAN clustering
    DBSCAN { eps: f64, min_samples: usize },
    /// Hierarchical clustering
    Hierarchical { n_clusters: usize },
    /// No clustering
    None,
}

/// High-dimensional data visualization plot
#[derive(Debug, Clone)]
pub struct HighDimensionalPlot {
    /// X coordinates (first component)
    pub x: Vec<f64>,
    /// Y coordinates (second component)
    pub y: Vec<f64>,
    /// Z coordinates (third component, if 3D)
    pub z: Option<Vec<f64>>,
    /// Colors for points
    pub colors: Vec<f64>,
    /// Cluster labels
    pub cluster_labels: Option<Vec<usize>>,
    /// Original data dimensions
    pub original_dimensions: usize,
    /// Reduced dimensions
    pub reduced_dimensions: usize,
    /// Reduction method used
    pub reduction_method: DimensionReductionMethod,
    /// Plot metadata
    pub metadata: PlotMetadata,
}

/// Animation settings
#[derive(Debug, Clone)]
pub struct AnimationSettings {
    /// Frames per second
    pub fps: f64,
    /// Loop animation
    pub loop_animation: bool,
    /// Frame interpolation
    pub interpolate_frames: bool,
    /// Fade trail length
    pub trail_length: usize,
}

impl Default for AnimationSettings {
    fn default() -> Self {
        Self {
            fps: 30.0,
            loop_animation: true,
            interpolate_frames: false,
            trail_length: 50,
        }
    }
}

/// Fluid state for 2D visualization
#[derive(Debug, Clone)]
pub struct FluidState {
    /// Velocity components [u, v] as 2D arrays
    pub velocity: Vec<Array2<f64>>,
    /// Pressure field
    pub pressure: Array2<f64>,
    /// Temperature field (optional)
    pub temperature: Option<Array2<f64>>,
    /// Current time
    pub time: f64,
    /// Grid spacing in x-direction
    pub dx: f64,
    /// Grid spacing in y-direction
    pub dy: f64,
}

/// Fluid state for 3D visualization
#[derive(Debug, Clone)]
pub struct FluidState3D {
    /// Velocity components [u, v, w] as 3D arrays
    pub velocity: Vec<Array3<f64>>,
    /// Pressure field
    pub pressure: Array3<f64>,
    /// Temperature field (optional)
    pub temperature: Option<Array3<f64>>,
    /// Current time
    pub time: f64,
    /// Grid spacing in x-direction
    pub dx: f64,
    /// Grid spacing in y-direction
    pub dy: f64,
    /// Grid spacing in z-direction
    pub dz: f64,
}

/// Error visualization options
#[derive(Debug, Clone)]
pub struct ErrorVisualizationOptions {
    /// Show absolute errors
    pub show_absolute: bool,
    /// Show relative errors
    pub show_relative: bool,
    /// Show error distribution
    pub show_distribution: bool,
    /// Show convergence history
    pub show_convergence: bool,
    /// Error threshold for highlighting
    pub error_threshold: f64,
}

impl Default for ErrorVisualizationOptions {
    fn default() -> Self {
        Self {
            show_absolute: true,
            show_relative: true,
            show_distribution: true,
            show_convergence: true,
            error_threshold: 1e-6,
        }
    }
}

/// Error types for visualization
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ErrorType {
    /// Absolute error
    Absolute,
    /// Relative error
    Relative,
    /// Truncation error
    Truncation,
    /// Roundoff error
    Roundoff,
    /// Discretization error
    Discretization,
}

/// Error distribution plot data
#[derive(Debug, Clone)]
pub struct ErrorDistributionPlot {
    /// Bin center values
    pub bin_centers: Array1<f64>,
    /// Histogram values
    pub histogram: Array1<f64>,
    /// Type of error
    pub error_type: ErrorType,
    /// Statistical summary
    pub statistics: ErrorStatistics,
    /// Color scheme
    pub color_scheme: ColorScheme,
}

/// Error statistics
#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    /// Mean error
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum error
    pub min: f64,
    /// Maximum error
    pub max: f64,
    /// Median error
    pub median: f64,
    /// 95th percentile
    pub percentile_95: f64,
}

/// Convergence plot data structure
#[derive(Debug, Clone)]
pub struct ConvergencePlot {
    pub iterations: Array1<f64>,
    pub residuals: Array1<f64>,
    pub convergence_iteration: Option<usize>,
    pub convergence_rate: f64,
    pub theoretical_line: Option<Array1<f64>>,
    pub algorithm_name: String,
    pub tolerance_line: f64,
}

/// Multi-metric convergence plot
#[derive(Debug, Clone)]
pub struct MultiMetricConvergencePlot {
    pub iterations: Array1<f64>,
    pub curves: Vec<ConvergenceCurve>,
    pub convergence_rates: Vec<(String, f64)>,
    pub tolerance_line: f64,
}

/// Individual convergence curve
#[derive(Debug, Clone)]
pub struct ConvergenceCurve {
    pub name: String,
    pub data: Array1<f64>,
    pub convergence_rate: f64,
    pub color: [f64; 3],
}

/// Step size analysis plot
#[derive(Debug, Clone)]
pub struct StepSizeAnalysisPlot {
    pub log_step_sizes: Array1<f64>,
    pub log_errors: Array1<f64>,
    pub theoretical_errors: Option<Array1<f64>>,
    pub order_of_accuracy: f64,
    pub method_name: String,
}

/// Phase space density plot
#[derive(Debug, Clone)]
pub struct PhaseDensityPlot {
    pub x_grid: Array1<f64>,
    pub y_grid: Array1<f64>,
    pub density_grid: Array2<f64>,
    pub x_bounds: (f64, f64),
    pub y_bounds: (f64, f64),
    pub n_points: usize,
}
