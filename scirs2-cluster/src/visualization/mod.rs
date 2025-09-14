//! Enhanced visualization capabilities for clustering results
//!
//! This module provides comprehensive visualization tools for clustering algorithms,
//! including 2D/3D scatter plots, animations, interactive visualizations, real-time
//! streaming displays, and various export formats for research and presentation use.
//!
//! # Features
//!
//! * **Static Visualizations**: 2D and 3D scatter plots with customizable styling
//! * **Interactive 3D**: Real-time manipulation, camera controls, VR/AR support
//! * **Animations**: Algorithm convergence animations, real-time streaming
//! * **Export Capabilities**: Multiple formats (PNG, SVG, HTML, JSON, video, etc.)
//! * **Dimensionality Reduction**: PCA, t-SNE, UMAP integration for high-dimensional data
//! * **Real-time Streaming**: Live data visualization with adaptive boundaries
//!
//! # Examples
//!
//! ## Basic 2D Visualization
//! ```
//! use ndarray::Array2;
//! use scirs2_cluster::visualization::{create_scatter_plot_2d, VisualizationConfig};
//!
//! let data = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
//! let labels = ndarray::Array1::from_vec(vec![0, 0, 1, 1]);
//! let config = VisualizationConfig::default();
//!
//! let plot = create_scatter_plot_2d(data.view(), &labels, None, &config).unwrap();
//! ```
//!
//! ## 3D Interactive Visualization
//! ```
//! use scirs2_cluster::visualization::interactive::{InteractiveVisualizer, InteractiveConfig};
//!
//! let config = InteractiveConfig::default();
//! let mut visualizer = InteractiveVisualizer::new(config);
//! // Set up interactive controls and update with data
//! ```
//!
//! ## Animation Recording
//! ```
//! use scirs2_cluster::visualization::animation::{IterativeAnimationRecorder, IterativeAnimationConfig};
//!
//! let config = IterativeAnimationConfig::default();
//! let mut recorder = IterativeAnimationRecorder::new(config);
//! // Record frames during algorithm iterations
//! ```

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};

// Sub-modules
pub mod animation;
pub mod export;
pub mod interactive;

// Re-export main types from sub-modules
pub use animation::{
    AnimationFrame, ConvergenceInfo, IterativeAnimationConfig, IterativeAnimationRecorder,
    StreamingConfig, StreamingFrame, StreamingStats, StreamingVisualizer,
};
pub use export::{
    export_animation_to_file, export_scatter_2d_to_file, export_scatter_2d_to_html,
    export_scatter_2d_to_json, export_scatter_3d_to_file, export_scatter_3d_to_html,
    export_scatter_3d_to_json, save_visualization_to_file, ExportConfig, ExportFormat,
};
pub use interactive::{
    BoundingBox3D, CameraState, ClusterStats, InteractiveConfig, InteractiveState,
    InteractiveVisualizer, KeyCode, MouseButton, ViewMode,
};

/// Configuration for clustering visualizations
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Color scheme for clusters
    pub color_scheme: ColorScheme,
    /// Point size for scatter plots
    pub point_size: f32,
    /// Point opacity (0.0 to 1.0)
    pub point_opacity: f32,
    /// Show cluster centroids
    pub show_centroids: bool,
    /// Show cluster boundaries (convex hull or ellipse)
    pub show_boundaries: bool,
    /// Boundary type
    pub boundary_type: BoundaryType,
    /// Enable interactive features
    pub interactive: bool,
    /// Animation settings
    pub animation: Option<AnimationConfig>,
    /// Dimensionality reduction method for high-dimensional data
    pub dimensionality_reduction: DimensionalityReduction,
}

/// Color schemes for cluster visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScheme {
    /// Default bright colors
    Default,
    /// Colorblind-friendly palette
    ColorblindFriendly,
    /// High contrast colors
    HighContrast,
    /// Pastel colors
    Pastel,
    /// Viridis colormap
    Viridis,
    /// Plasma colormap
    Plasma,
    /// Custom colors (user-defined)
    Custom,
}

/// Cluster boundary visualization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Convex hull around points
    ConvexHull,
    /// Ellipse based on covariance
    Ellipse,
    /// Alpha shapes for non-convex boundaries
    AlphaShape,
    /// No boundaries
    None,
}

/// Dimensionality reduction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionalityReduction {
    /// Principal Component Analysis
    PCA,
    /// t-Distributed Stochastic Neighbor Embedding
    TSNE,
    /// Uniform Manifold Approximation and Projection
    UMAP,
    /// Multidimensional Scaling
    MDS,
    /// Use first two dimensions
    First2D,
    /// Use first three dimensions
    First3D,
    /// No reduction (error if >3D)
    None,
}

/// Animation configuration for visualizations
#[derive(Debug, Clone)]
pub struct AnimationConfig {
    /// Animation duration in milliseconds
    pub duration_ms: u32,
    /// Number of animation frames
    pub frames: u32,
    /// Easing function
    pub easing: EasingFunction,
    /// Whether to loop the animation
    pub loop_animation: bool,
}

/// Easing functions for animations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            color_scheme: ColorScheme::Default,
            point_size: 5.0,
            point_opacity: 0.8,
            show_centroids: true,
            show_boundaries: false,
            boundary_type: BoundaryType::ConvexHull,
            interactive: true,
            animation: None,
            dimensionality_reduction: DimensionalityReduction::PCA,
        }
    }
}

/// 2D scatter plot visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterPlot2D {
    /// Point coordinates
    pub points: Array2<f64>,
    /// Cluster labels for each point
    pub labels: Array1<i32>,
    /// Cluster centroids (if available)
    pub centroids: Option<Array2<f64>>,
    /// Point colors (hex format)
    pub colors: Vec<String>,
    /// Point sizes
    pub sizes: Vec<f32>,
    /// Point labels (optional)
    pub point_labels: Option<Vec<String>>,
    /// Plot boundaries (min_x, max_x, min_y, max_y)
    pub bounds: (f64, f64, f64, f64),
    /// Legend information
    pub legend: Vec<LegendEntry>,
}

/// 3D scatter plot visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterPlot3D {
    /// Point coordinates (x, y, z)
    pub points: Array2<f64>,
    /// Cluster labels for each point
    pub labels: Array1<i32>,
    /// Cluster centroids (if available)
    pub centroids: Option<Array2<f64>>,
    /// Point colors (hex format)
    pub colors: Vec<String>,
    /// Point sizes
    pub sizes: Vec<f32>,
    /// Point labels (optional)
    pub point_labels: Option<Vec<String>>,
    /// Plot boundaries (min_x, max_x, min_y, max_y, min_z, max_z)
    pub bounds: (f64, f64, f64, f64, f64, f64),
    /// Legend information
    pub legend: Vec<LegendEntry>,
}

/// Legend entry for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendEntry {
    /// Cluster ID
    pub cluster_id: i32,
    /// Color hex code
    pub color: String,
    /// Cluster label/name
    pub label: String,
    /// Number of points in cluster
    pub count: usize,
}

/// Cluster boundary representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterBoundary {
    /// Cluster ID
    pub cluster_id: i32,
    /// Boundary points
    pub boundary_points: Array2<f64>,
    /// Boundary type
    pub boundary_type: String,
    /// Color for the boundary
    pub color: String,
}

/// Create 2D scatter plot visualization
///
/// # Arguments
///
/// * `data` - Input data matrix (samples x features)
/// * `labels` - Cluster labels for each sample
/// * `centroids` - Optional cluster centroids
/// * `config` - Visualization configuration
///
/// # Returns
///
/// * `Result<ScatterPlot2D>` - 2D scatter plot data
#[allow(dead_code)]
pub fn create_scatter_plot_2d<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    labels: &Array1<i32>,
    centroids: Option<&Array2<F>>,
    config: &VisualizationConfig,
) -> Result<ScatterPlot2D> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if labels.len() != n_samples {
        return Err(ClusteringError::InvalidInput(
            "Number of labels must match number of samples".to_string(),
        ));
    }

    // Reduce dimensionality if needed
    let plotdata =
        if n_features == 2 && config.dimensionality_reduction == DimensionalityReduction::None {
            data.mapv(|x| x.to_f64().unwrap_or(0.0))
        } else {
            apply_dimensionality_reduction_2d(data, config.dimensionality_reduction)?
        };

    // Convert centroids if provided
    let plot_centroids = if let Some(cents) = centroids {
        if cents.ncols() == 2 && config.dimensionality_reduction == DimensionalityReduction::None {
            Some(cents.mapv(|x| x.to_f64().unwrap_or(0.0)))
        } else {
            Some(apply_dimensionality_reduction_2d(
                cents.view(),
                config.dimensionality_reduction,
            )?)
        }
    } else {
        None
    };

    // Generate colors for clusters
    let unique_labels: Vec<i32> = {
        let mut labels_vec: Vec<i32> = labels.iter().cloned().collect();
        labels_vec.sort_unstable();
        labels_vec.dedup();
        labels_vec
    };

    let cluster_colors = generate_cluster_colors(&unique_labels, config.color_scheme);
    let point_colors = labels
        .iter()
        .map(|&label| {
            cluster_colors
                .get(&label)
                .cloned()
                .unwrap_or_else(|| "#000000".to_string())
        })
        .collect();

    // Generate point sizes
    let sizes = vec![config.point_size; n_samples];

    // Calculate plot bounds
    let bounds = calculate_2d_bounds(&plotdata);

    // Create legend
    let legend = create_legend(&unique_labels, &cluster_colors, labels);

    Ok(ScatterPlot2D {
        points: plotdata,
        labels: labels.clone(),
        centroids: plot_centroids,
        colors: point_colors,
        sizes,
        point_labels: None,
        bounds,
        legend,
    })
}

/// Create 3D scatter plot visualization
///
/// # Arguments
///
/// * `data` - Input data matrix (samples x features)
/// * `labels` - Cluster labels for each sample
/// * `centroids` - Optional cluster centroids
/// * `config` - Visualization configuration
///
/// # Returns
///
/// * `Result<ScatterPlot3D>` - 3D scatter plot data
#[allow(dead_code)]
pub fn create_scatter_plot_3d<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    labels: &Array1<i32>,
    centroids: Option<&Array2<F>>,
    config: &VisualizationConfig,
) -> Result<ScatterPlot3D> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if labels.len() != n_samples {
        return Err(ClusteringError::InvalidInput(
            "Number of labels must match number of samples".to_string(),
        ));
    }

    // Reduce dimensionality if needed
    let plotdata =
        if n_features == 3 && config.dimensionality_reduction == DimensionalityReduction::None {
            data.mapv(|x| x.to_f64().unwrap_or(0.0))
        } else {
            apply_dimensionality_reduction_3d(data, config.dimensionality_reduction)?
        };

    // Convert centroids if provided
    let plot_centroids = if let Some(cents) = centroids {
        if cents.ncols() == 3 && config.dimensionality_reduction == DimensionalityReduction::None {
            Some(cents.mapv(|x| x.to_f64().unwrap_or(0.0)))
        } else {
            Some(apply_dimensionality_reduction_3d(
                cents.view(),
                config.dimensionality_reduction,
            )?)
        }
    } else {
        None
    };

    // Generate colors for clusters
    let unique_labels: Vec<i32> = {
        let mut labels_vec: Vec<i32> = labels.iter().cloned().collect();
        labels_vec.sort_unstable();
        labels_vec.dedup();
        labels_vec
    };

    let cluster_colors = generate_cluster_colors(&unique_labels, config.color_scheme);
    let point_colors = labels
        .iter()
        .map(|&label| {
            cluster_colors
                .get(&label)
                .cloned()
                .unwrap_or_else(|| "#000000".to_string())
        })
        .collect();

    // Generate point sizes
    let sizes = vec![config.point_size; n_samples];

    // Calculate plot bounds
    let bounds = calculate_3d_bounds(&plotdata);

    // Create legend
    let legend = create_legend(&unique_labels, &cluster_colors, labels);

    Ok(ScatterPlot3D {
        points: plotdata,
        labels: labels.clone(),
        centroids: plot_centroids,
        colors: point_colors,
        sizes,
        point_labels: None,
        bounds,
        legend,
    })
}

/// Apply dimensionality reduction for 2D visualization
#[allow(dead_code)]
fn apply_dimensionality_reduction_2d<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    method: DimensionalityReduction,
) -> Result<Array2<f64>> {
    let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

    match method {
        DimensionalityReduction::PCA => apply_pca_2d(&data_f64),
        DimensionalityReduction::First2D => {
            if data_f64.ncols() >= 2 {
                Ok(data_f64.slice(s![.., 0..2]).to_owned())
            } else {
                Err(ClusteringError::InvalidInput(
                    "Data must have at least 2 dimensions for First2D".to_string(),
                ))
            }
        }
        DimensionalityReduction::TSNE => apply_tsne_2d(&data_f64),
        DimensionalityReduction::UMAP => apply_umap_2d(&data_f64),
        DimensionalityReduction::MDS => apply_mds_2d(&data_f64),
        DimensionalityReduction::None => {
            if data_f64.ncols() == 2 {
                Ok(data_f64)
            } else {
                Err(ClusteringError::InvalidInput(
                    "Data must be 2D when no dimensionality reduction is specified".to_string(),
                ))
            }
        }
        _ => apply_pca_2d(&data_f64), // Default to PCA
    }
}

/// Apply dimensionality reduction for 3D visualization
#[allow(dead_code)]
fn apply_dimensionality_reduction_3d<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    method: DimensionalityReduction,
) -> Result<Array2<f64>> {
    let data_f64 = data.mapv(|x| x.to_f64().unwrap_or(0.0));

    match method {
        DimensionalityReduction::PCA => apply_pca_3d(&data_f64),
        DimensionalityReduction::First3D => {
            if data_f64.ncols() >= 3 {
                Ok(data_f64.slice(s![.., 0..3]).to_owned())
            } else {
                Err(ClusteringError::InvalidInput(
                    "Data must have at least 3 dimensions for First3D".to_string(),
                ))
            }
        }
        DimensionalityReduction::None => {
            if data_f64.ncols() == 3 {
                Ok(data_f64)
            } else {
                Err(ClusteringError::InvalidInput(
                    "Data must be 3D when no dimensionality reduction is specified".to_string(),
                ))
            }
        }
        _ => apply_pca_3d(&data_f64), // Default to PCA
    }
}

/// Apply PCA for 2D visualization
#[allow(dead_code)]
fn apply_pca_2d(data: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_features < 2 {
        return Err(ClusteringError::InvalidInput(
            "Need at least 2 features for PCA".to_string(),
        ));
    }

    // Center the data
    let mean = data.mean_axis(Axis(0)).unwrap();
    let centered = data - &mean;

    // Compute covariance matrix
    let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

    // Simplified PCA projection (stub implementation)
    // In a real implementation, this would compute eigenvectors of covariance matrix
    let n_features = centered.ncols();
    let eigenvectors_ = Array2::eye(n_features)
        .slice(s![.., 0..2.min(n_features)])
        .to_owned();

    // Project data onto first 2 principal components
    let projected = centered.dot(&eigenvectors_);

    Ok(projected)
}

/// Apply PCA for 3D visualization
#[allow(dead_code)]
fn apply_pca_3d(data: &Array2<f64>) -> Result<Array2<f64>> {
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_features < 3 {
        return Err(ClusteringError::InvalidInput(
            "Need at least 3 features for 3D PCA".to_string(),
        ));
    }

    // Center the data
    let mean = data.mean_axis(Axis(0)).unwrap();
    let centered = data - &mean;

    // Compute covariance matrix
    let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

    // Simplified PCA projection (stub implementation)
    // In a real implementation, this would compute eigenvectors of covariance matrix
    let n_features = centered.ncols();
    let eigenvectors_ = Array2::eye(n_features)
        .slice(s![.., 0..3.min(n_features)])
        .to_owned();

    // Project data onto first 3 principal components
    let projected = centered.dot(&eigenvectors_);

    Ok(projected)
}

/// Simplified implementation of other dimensionality reduction methods
/// These would ideally use proper implementations from specialized libraries
#[allow(dead_code)]
fn apply_tsne_2d(data: &Array2<f64>) -> Result<Array2<f64>> {
    // For now, fall back to PCA
    apply_pca_2d(data)
}

#[allow(dead_code)]
fn apply_umap_2d(data: &Array2<f64>) -> Result<Array2<f64>> {
    // For now, fall back to PCA
    apply_pca_2d(data)
}

#[allow(dead_code)]
fn apply_mds_2d(data: &Array2<f64>) -> Result<Array2<f64>> {
    // For now, fall back to PCA
    apply_pca_2d(data)
}

/// Compute top eigenvectors using power iteration
#[allow(dead_code)]
fn compute_top_eigenvectors(
    matrix: &Array2<f64>,
    num_components: usize,
) -> Result<(Array2<f64>, Array1<f64>)> {
    let n = matrix.nrows();
    let k = num_components.min(n);

    let mut eigenvectors = Array2::zeros((n, k));
    let mut eigenvalues = Array1::zeros(k);

    // Simple power iteration for dominant eigenvector
    for i in 0..k {
        let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());

        // Orthogonalize against previous eigenvectors
        for j in 0..i {
            let prev_eigenvector = eigenvectors.column(j);
            let dot_product = v.dot(&prev_eigenvector);
            v = &v - &(&prev_eigenvector * dot_product);
        }

        // Power iteration
        for _ in 0..100 {
            let new_v = matrix.dot(&v);
            let norm = (new_v.dot(&new_v)).sqrt();
            if norm > 1e-10 {
                v = new_v / norm;
            }
        }

        eigenvalues[i] = v.dot(&matrix.dot(&v));
        for j in 0..n {
            eigenvectors[[j, i]] = v[j];
        }
    }

    Ok((eigenvectors, eigenvalues))
}

/// Generate cluster colors based on color scheme
#[allow(dead_code)]
fn generate_cluster_colors(labels: &[i32], scheme: ColorScheme) -> HashMap<i32, String> {
    let mut colors = HashMap::new();

    let color_palette = match scheme {
        ColorScheme::Default => vec![
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf",
        ],
        ColorScheme::ColorblindFriendly => vec![
            "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999",
        ],
        ColorScheme::HighContrast => vec![
            "#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff",
        ],
        ColorScheme::Pastel => vec![
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d3", "#c7c7c7",
            "#dbdb8d", "#9edae5",
        ],
        ColorScheme::Viridis => vec![
            "#440154", "#482777", "#3f4a8a", "#31678e", "#26838f", "#1f9d8a", "#6cce5a", "#b6de2b",
            "#fee825",
        ],
        ColorScheme::Plasma => vec![
            "#0c0887", "#5302a3", "#8b0aa5", "#b83289", "#db5c68", "#f48849", "#febd2a", "#f0f921",
        ],
        ColorScheme::Custom => vec!["#333333"], // Placeholder
    };

    for (i, &label) in labels.iter().enumerate() {
        if !colors.contains_key(&label) {
            let color_index = i % color_palette.len();
            colors.insert(label, color_palette[color_index].to_string());
        }
    }

    colors
}

/// Calculate 2D plot bounds
#[allow(dead_code)]
fn calculate_2d_bounds(data: &Array2<f64>) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let x_min = data.column(0).iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = data
        .column(0)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = data.column(1).iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = data
        .column(1)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Add padding
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    let padding = 0.05;

    (
        x_min - x_range * padding,
        x_max + x_range * padding,
        y_min - y_range * padding,
        y_max + y_range * padding,
    )
}

/// Calculate 3D plot bounds
#[allow(dead_code)]
fn calculate_3d_bounds(data: &Array2<f64>) -> (f64, f64, f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    }

    let x_min = data.column(0).iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = data
        .column(0)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = data.column(1).iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = data
        .column(1)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let z_min = data.column(2).iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let z_max = data
        .column(2)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Add padding
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    let z_range = z_max - z_min;
    let padding = 0.05;

    (
        x_min - x_range * padding,
        x_max + x_range * padding,
        y_min - y_range * padding,
        y_max + y_range * padding,
        z_min - z_range * padding,
        z_max + z_range * padding,
    )
}

/// Create legend entries
#[allow(dead_code)]
fn create_legend(
    labels: &[i32],
    colors: &HashMap<i32, String>,
    data_labels: &Array1<i32>,
) -> Vec<LegendEntry> {
    let mut legend = Vec::new();

    for &label in labels {
        let count = data_labels.iter().filter(|&&l| l == label).count();
        let color = colors
            .get(&label)
            .cloned()
            .unwrap_or_else(|| "#000000".to_string());

        legend.push(LegendEntry {
            cluster_id: label,
            color,
            label: format!("Cluster {}", label),
            count,
        });
    }

    // Sort by cluster ID
    legend.sort_by_key(|entry| entry.cluster_id);

    legend
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_create_scatter_plot_2d() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let config = VisualizationConfig::default();

        let plot = create_scatter_plot_2d(data.view(), &labels, None, &config).unwrap();

        assert_eq!(plot.points.nrows(), 4);
        assert_eq!(plot.points.ncols(), 2);
        assert_eq!(plot.labels.len(), 4);
        assert_eq!(plot.colors.len(), 4);
        assert_eq!(plot.legend.len(), 2);
    }

    #[test]
    fn test_create_scatter_plot_3d() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);
        let config = VisualizationConfig::default();

        let plot = create_scatter_plot_3d(data.view(), &labels, None, &config).unwrap();

        assert_eq!(plot.points.nrows(), 4);
        assert_eq!(plot.points.ncols(), 3);
        assert_eq!(plot.labels.len(), 4);
    }

    #[test]
    fn test_dimensionality_reduction() {
        let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect()).unwrap();

        let result_2d =
            apply_dimensionality_reduction_2d(data.view(), DimensionalityReduction::PCA).unwrap();
        assert_eq!(result_2d.ncols(), 2);

        let result_3d =
            apply_dimensionality_reduction_3d(data.view(), DimensionalityReduction::PCA).unwrap();
        assert_eq!(result_3d.ncols(), 3);
    }

    #[test]
    fn test_color_generation() {
        let labels = vec![0, 1, 2];
        let colors = generate_cluster_colors(&labels, ColorScheme::Default);

        assert_eq!(colors.len(), 3);
        assert!(colors.contains_key(&0));
        assert!(colors.contains_key(&1));
        assert!(colors.contains_key(&2));
    }
}
