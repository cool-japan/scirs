//! Enhanced visualization utilities for dendrograms
//!
//! This module provides advanced tools for visualizing hierarchical clustering results,
//! including color threshold controls, cluster highlighting, and enhanced plotting utilities.

use ndarray::ArrayView2;
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

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
}

/// Represents a branch in the dendrogram
#[derive(Debug, Clone)]
pub struct Branch<F: Float> {
    /// Starting position (x, y)
    pub start: (F, F),
    /// Ending position (x, y)
    pub end: (F, F),
    /// Height of this branch
    pub height: F,
    /// Cluster ID associated with this branch
    pub cluster_id: usize,
    /// Whether this branch is above the color threshold
    pub above_threshold: bool,
}

/// Represents a leaf (terminal node) in the dendrogram
#[derive(Debug, Clone)]
pub struct Leaf {
    /// Position (x, y)
    pub position: (f64, f64),
    /// Label text
    pub label: String,
    /// Original sample index
    pub sample_id: usize,
}

/// Legend entry for the dendrogram
#[derive(Debug, Clone)]
pub struct LegendEntry {
    /// Color hex code
    pub color: String,
    /// Description text
    pub description: String,
}

/// Create enhanced dendrogram visualization data
///
/// # Arguments
///
/// * `linkage_matrix` - The linkage matrix from hierarchical clustering
/// * `labels` - Optional labels for the leaves (defaults to sample indices)
/// * `config` - Visualization configuration
///
/// # Returns
///
/// * `Result<DendrogramPlot<F>>` - The dendrogram plot data
pub fn create_dendrogram_plot<F: Float + FromPrimitive + PartialOrd + Debug>(
    linkage_matrix: ArrayView2<F>,
    labels: Option<&[String]>,
    config: DendrogramConfig<F>,
) -> Result<DendrogramPlot<F>> {
    let n_samples = linkage_matrix.shape()[0] + 1;

    if n_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "Need at least 2 samples to create dendrogram".into(),
        ));
    }

    // Calculate color threshold if using automatic mode
    let actual_threshold = if config.color_threshold.auto_threshold {
        calculate_auto_threshold(&linkage_matrix, config.color_threshold.target_clusters)?
    } else {
        config.color_threshold.threshold
    };

    // Build the dendrogram tree structure
    let tree = build_dendrogram_tree(&linkage_matrix)?;

    // Calculate positions for nodes
    let positions = calculate_node_positions(&tree, n_samples, config.orientation);

    // Create branches
    let branches = create_branches(&tree, &positions, actual_threshold, &config)?;

    // Create leaves
    let leaves = create_leaves(&positions, labels, n_samples, config.orientation);

    // Assign colors to branches
    let colors = assign_branch_colors(&branches, &config);

    // Create legend
    let legend = create_legend(&config, actual_threshold);

    // Calculate plot bounds
    let bounds = calculate_plot_bounds(&branches, &leaves);

    Ok(DendrogramPlot {
        branches,
        leaves,
        colors,
        legend,
        bounds,
    })
}

/// Node in the dendrogram tree
#[derive(Debug, Clone)]
struct TreeNode<F: Float> {
    /// Node ID (sample index for leaves, or cluster index for internal nodes)
    id: usize,
    /// Height of this node
    height: F,
    /// Left child (None for leaves)
    left: Option<Box<TreeNode<F>>>,
    /// Right child (None for leaves)
    right: Option<Box<TreeNode<F>>>,
    /// Number of leaves under this node
    #[allow(dead_code)]
    count: usize,
    /// Whether this is a leaf node
    is_leaf: bool,
}

/// Build the dendrogram tree structure from linkage matrix
fn build_dendrogram_tree<F: Float + FromPrimitive + Debug>(
    linkage_matrix: &ArrayView2<F>,
) -> Result<TreeNode<F>> {
    let n_samples = linkage_matrix.shape()[0] + 1;
    let mut nodes: HashMap<usize, TreeNode<F>> = HashMap::new();

    // Create leaf nodes
    for i in 0..n_samples {
        nodes.insert(
            i,
            TreeNode {
                id: i,
                height: F::zero(),
                left: None,
                right: None,
                count: 1,
                is_leaf: true,
            },
        );
    }

    // Build internal nodes from linkage matrix
    for (step, row) in linkage_matrix.outer_iter().enumerate() {
        let left_id = row[0].to_usize().unwrap();
        let right_id = row[1].to_usize().unwrap();
        let height = row[2];
        let count = row[3].to_usize().unwrap();

        let left_node = nodes.remove(&left_id).ok_or_else(|| {
            ClusteringError::ComputationError(format!("Missing node {}", left_id))
        })?;

        let right_node = nodes.remove(&right_id).ok_or_else(|| {
            ClusteringError::ComputationError(format!("Missing node {}", right_id))
        })?;

        let new_node = TreeNode {
            id: n_samples + step,
            height,
            left: Some(Box::new(left_node)),
            right: Some(Box::new(right_node)),
            count,
            is_leaf: false,
        };

        nodes.insert(n_samples + step, new_node);
    }

    // Return the root node (should be the only remaining node)
    let root_id = n_samples + linkage_matrix.shape()[0] - 1;
    nodes
        .into_iter()
        .find(|(_, node)| node.id == root_id)
        .map(|(_, node)| node)
        .ok_or_else(|| ClusteringError::ComputationError("Failed to build dendrogram tree".into()))
}

/// Calculate positions for all nodes in the dendrogram
fn calculate_node_positions<F: Float + FromPrimitive>(
    root: &TreeNode<F>,
    n_samples: usize,
    orientation: DendrogramOrientation,
) -> HashMap<usize, (F, F)> {
    let mut positions = HashMap::new();
    let mut leaf_counter = 0;

    calculate_positions_recursive(
        root,
        &mut positions,
        &mut leaf_counter,
        n_samples,
        orientation,
    );

    positions
}

/// Recursive helper for calculating node positions
fn calculate_positions_recursive<F: Float + FromPrimitive>(
    node: &TreeNode<F>,
    positions: &mut HashMap<usize, (F, F)>,
    leaf_counter: &mut usize,
    _n_samples: usize,
    orientation: DendrogramOrientation,
) -> F {
    if node.is_leaf {
        let x = F::from_usize(*leaf_counter).unwrap();
        let y = F::zero();

        let (final_x, final_y) = match orientation {
            DendrogramOrientation::Top | DendrogramOrientation::Bottom => (x, y),
            DendrogramOrientation::Left | DendrogramOrientation::Right => (y, x),
        };

        positions.insert(node.id, (final_x, final_y));
        *leaf_counter += 1;
        x
    } else {
        let left_x = if let Some(ref left) = node.left {
            calculate_positions_recursive(left, positions, leaf_counter, _n_samples, orientation)
        } else {
            F::zero()
        };

        let right_x = if let Some(ref right) = node.right {
            calculate_positions_recursive(right, positions, leaf_counter, _n_samples, orientation)
        } else {
            F::zero()
        };

        let x = (left_x + right_x) / F::from(2).unwrap();
        let y = node.height;

        let (final_x, final_y) = match orientation {
            DendrogramOrientation::Top => (x, y),
            DendrogramOrientation::Bottom => (x, -y),
            DendrogramOrientation::Left => (y, x),
            DendrogramOrientation::Right => (-y, x),
        };

        positions.insert(node.id, (final_x, final_y));
        x
    }
}

/// Create branches for the dendrogram
fn create_branches<F: Float + FromPrimitive + PartialOrd>(
    tree: &TreeNode<F>,
    positions: &HashMap<usize, (F, F)>,
    threshold: F,
    config: &DendrogramConfig<F>,
) -> Result<Vec<Branch<F>>> {
    let mut branches = Vec::new();
    create_branches_recursive(tree, positions, &mut branches, threshold, config);
    Ok(branches)
}

/// Recursive helper for creating branches
fn create_branches_recursive<F: Float + FromPrimitive + PartialOrd>(
    node: &TreeNode<F>,
    positions: &HashMap<usize, (F, F)>,
    branches: &mut Vec<Branch<F>>,
    threshold: F,
    _config: &DendrogramConfig<F>,
) {
    if node.is_leaf {
        return;
    }

    let (node_x, node_y) = positions[&node.id];
    let above_threshold = node.height > threshold;

    // Create branches to children
    if let Some(ref left) = node.left {
        let (left_x, left_y) = positions[&left.id];

        // Vertical line from child to parent height
        branches.push(Branch {
            start: (left_x, left_y),
            end: (left_x, node_y),
            height: node.height,
            cluster_id: node.id,
            above_threshold,
        });

        // Horizontal line from child to node
        branches.push(Branch {
            start: (left_x, node_y),
            end: (node_x, node_y),
            height: node.height,
            cluster_id: node.id,
            above_threshold,
        });

        create_branches_recursive(left, positions, branches, threshold, _config);
    }

    if let Some(ref right) = node.right {
        let (right_x, right_y) = positions[&right.id];

        // Vertical line from child to parent height
        branches.push(Branch {
            start: (right_x, right_y),
            end: (right_x, node_y),
            height: node.height,
            cluster_id: node.id,
            above_threshold,
        });

        // Horizontal line from child to node
        branches.push(Branch {
            start: (right_x, node_y),
            end: (node_x, node_y),
            height: node.height,
            cluster_id: node.id,
            above_threshold,
        });

        create_branches_recursive(right, positions, branches, threshold, _config);
    }
}

/// Create leaves for the dendrogram
fn create_leaves<F: Float>(
    positions: &HashMap<usize, (F, F)>,
    labels: Option<&[String]>,
    n_samples: usize,
    _orientation: DendrogramOrientation,
) -> Vec<Leaf> {
    let mut leaves = Vec::new();

    for i in 0..n_samples {
        if let Some(&(x, y)) = positions.get(&i) {
            let label = if let Some(labels) = labels {
                labels.get(i).cloned().unwrap_or_else(|| i.to_string())
            } else {
                i.to_string()
            };

            leaves.push(Leaf {
                position: (x.to_f64().unwrap_or(0.0), y.to_f64().unwrap_or(0.0)),
                label,
                sample_id: i,
            });
        }
    }

    leaves
}

/// Assign colors to branches based on configuration
fn assign_branch_colors<F: Float>(
    branches: &[Branch<F>],
    config: &DendrogramConfig<F>,
) -> Vec<String> {
    branches
        .iter()
        .map(|branch| {
            if branch.above_threshold {
                config.color_threshold.above_color.clone()
            } else {
                config.color_threshold.below_color.clone()
            }
        })
        .collect()
}

/// Create legend for the dendrogram
fn create_legend<F: Float>(config: &DendrogramConfig<F>, threshold: F) -> Vec<LegendEntry> {
    vec![
        LegendEntry {
            color: config.color_threshold.above_color.clone(),
            description: format!("Distance > {:.3}", threshold.to_f64().unwrap_or(0.0)),
        },
        LegendEntry {
            color: config.color_threshold.below_color.clone(),
            description: format!("Distance ≤ {:.3}", threshold.to_f64().unwrap_or(0.0)),
        },
    ]
}

/// Calculate automatic threshold based on desired number of clusters
fn calculate_auto_threshold<F: Float + FromPrimitive + PartialOrd>(
    linkage_matrix: &ArrayView2<F>,
    target_clusters: Option<usize>,
) -> Result<F> {
    let n_samples = linkage_matrix.shape()[0] + 1;
    let clusters = target_clusters.unwrap_or(4).min(n_samples);

    if clusters <= 1 || clusters > n_samples {
        return Ok(F::zero());
    }

    // Find the threshold that gives us the desired number of clusters
    // This is the distance at the (n_samples - clusters)th merge
    let merge_index = n_samples - clusters;

    if merge_index < linkage_matrix.shape()[0] {
        Ok(linkage_matrix[[merge_index, 2]])
    } else {
        Ok(F::zero())
    }
}

/// Calculate plot bounds
fn calculate_plot_bounds<F: Float>(branches: &[Branch<F>], leaves: &[Leaf]) -> (F, F, F, F) {
    let mut min_x = F::infinity();
    let mut max_x = F::neg_infinity();
    let mut min_y = F::infinity();
    let mut max_y = F::neg_infinity();

    // Check branch bounds
    for branch in branches {
        let points = [branch.start, branch.end];
        for (x, y) in points.iter() {
            if *x < min_x {
                min_x = *x;
            }
            if *x > max_x {
                max_x = *x;
            }
            if *y < min_y {
                min_y = *y;
            }
            if *y > max_y {
                max_y = *y;
            }
        }
    }

    // Check leaf bounds
    for leaf in leaves {
        let x = F::from(leaf.position.0).unwrap_or(F::zero());
        let y = F::from(leaf.position.1).unwrap_or(F::zero());

        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
    }

    (min_x, max_x, min_y, max_y)
}

/// Get color palette for a given color scheme
pub fn get_color_palette(scheme: ColorScheme, n_colors: usize) -> Vec<String> {
    match scheme {
        ColorScheme::Default => get_default_colors(n_colors),
        ColorScheme::HighContrast => get_high_contrast_colors(n_colors),
        ColorScheme::Viridis => get_viridis_colors(n_colors),
        ColorScheme::Plasma => get_plasma_colors(n_colors),
        ColorScheme::Grayscale => get_grayscale_colors(n_colors),
    }
}

/// Default color palette
fn get_default_colors(n_colors: usize) -> Vec<String> {
    let base_colors = vec![
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(n_colors)
        .map(|s| s.to_string())
        .collect()
}

/// High contrast color palette
fn get_high_contrast_colors(n_colors: usize) -> Vec<String> {
    let base_colors = vec![
        "#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#800000", "#008000",
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(n_colors)
        .map(|s| s.to_string())
        .collect()
}

/// Viridis color palette (approximation)
fn get_viridis_colors(n_colors: usize) -> Vec<String> {
    let base_colors = vec![
        "#440154", "#482777", "#3f4a8a", "#31678e", "#26838f", "#1f9d8a", "#6cce5a", "#b6de2b",
        "#fee825", "#fff200",
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(n_colors)
        .map(|s| s.to_string())
        .collect()
}

/// Plasma color palette (approximation)
fn get_plasma_colors(n_colors: usize) -> Vec<String> {
    let base_colors = vec![
        "#0c0887", "#5c01a6", "#900da4", "#bf3984", "#e16462", "#f89441", "#fdc328", "#f0f921",
        "#fcffa4", "#ffffff",
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(n_colors)
        .map(|s| s.to_string())
        .collect()
}

/// Grayscale color palette
fn get_grayscale_colors(n_colors: usize) -> Vec<String> {
    (0..n_colors)
        .map(|i| {
            let intensity = 255 * i / n_colors.max(1);
            format!("#{:02x}{:02x}{:02x}", intensity, intensity, intensity)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::{linkage, LinkageMethod, Metric};
    use ndarray::Array2;

    #[test]
    fn test_create_dendrogram_plot() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0]).unwrap();

        let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();
        let config = DendrogramConfig::default();

        let plot = create_dendrogram_plot(linkage_matrix.view(), None, config).unwrap();

        // Should have branches and leaves
        assert!(!plot.branches.is_empty());
        assert_eq!(plot.leaves.len(), 4);

        // Should have colors for each branch
        assert_eq!(plot.colors.len(), plot.branches.len());

        // Should have legend entries
        assert_eq!(plot.legend.len(), 2);
    }

    #[test]
    fn test_color_threshold_auto() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2],
        )
        .unwrap();

        let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();

        // Test automatic threshold calculation
        let threshold = calculate_auto_threshold(&linkage_matrix.view(), Some(2)).unwrap();
        assert!(threshold > 0.0);

        // Should be finite and reasonable
        assert!(threshold.is_finite());
    }

    #[test]
    fn test_color_palettes() {
        let schemes = vec![
            ColorScheme::Default,
            ColorScheme::HighContrast,
            ColorScheme::Viridis,
            ColorScheme::Plasma,
            ColorScheme::Grayscale,
        ];

        for scheme in schemes {
            let colors = get_color_palette(scheme, 5);
            assert_eq!(colors.len(), 5);

            // Each color should be a valid hex color
            for color in colors {
                assert!(color.starts_with('#'));
                assert_eq!(color.len(), 7);
            }
        }
    }

    #[test]
    fn test_dendrogram_orientations() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.5, 1.0]).unwrap();

        let linkage_matrix =
            linkage(data.view(), LinkageMethod::Single, Metric::Euclidean).unwrap();

        let orientations = vec![
            DendrogramOrientation::Top,
            DendrogramOrientation::Bottom,
            DendrogramOrientation::Left,
            DendrogramOrientation::Right,
        ];

        for orientation in orientations {
            let config = DendrogramConfig {
                orientation,
                ..Default::default()
            };

            let plot = create_dendrogram_plot(linkage_matrix.view(), None, config).unwrap();

            // Should create valid plot for each orientation
            assert!(!plot.branches.is_empty());
            assert_eq!(plot.leaves.len(), 3);
        }
    }

    #[test]
    fn test_custom_labels() {
        let data = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();

        let linkage_matrix =
            linkage(data.view(), LinkageMethod::Single, Metric::Euclidean).unwrap();
        let config = DendrogramConfig::default();

        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let plot = create_dendrogram_plot(linkage_matrix.view(), Some(&labels), config).unwrap();

        // Should use custom labels
        assert_eq!(plot.leaves.len(), 3);
        assert_eq!(plot.leaves[0].label, "A");
        assert_eq!(plot.leaves[1].label, "B");
        assert_eq!(plot.leaves[2].label, "C");
    }

    #[test]
    fn test_bounds_calculation() {
        let branches = vec![
            Branch {
                start: (0.0, 0.0),
                end: (1.0, 1.0),
                height: 1.0,
                cluster_id: 0,
                above_threshold: false,
            },
            Branch {
                start: (2.0, 2.0),
                end: (3.0, 3.0),
                height: 2.0,
                cluster_id: 1,
                above_threshold: true,
            },
        ];

        let leaves = vec![
            Leaf {
                position: (-1.0, -1.0),
                label: "A".to_string(),
                sample_id: 0,
            },
            Leaf {
                position: (4.0, 4.0),
                label: "B".to_string(),
                sample_id: 1,
            },
        ];

        let bounds = calculate_plot_bounds(&branches, &leaves);

        // Should encompass all points
        assert_eq!(bounds.0, -1.0); // min_x
        assert_eq!(bounds.1, 4.0); // max_x
        assert_eq!(bounds.2, -1.0); // min_y
        assert_eq!(bounds.3, 4.0); // max_y
    }
}
