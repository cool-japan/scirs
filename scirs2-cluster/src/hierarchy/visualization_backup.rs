//! Enhanced visualization utilities for dendrograms
//!
//! This module provides advanced tools for visualizing hierarchical clustering results,
//! including color threshold controls, cluster highlighting, and enhanced plotting utilities.

use ndarray::ArrayView2;
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[allow(dead_code)]
pub fn create_dendrogramplot<F: Float + FromPrimitive + PartialOrd + Debug>(
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
    let bounds = calculateplot_bounds(&branches, &leaves);

    Ok(DendrogramPlot {
        branches,
        leaves,
        colors,
        legend,
        bounds,
        config,
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

impl<F: Float + std::fmt::Display> TreeNode<F> {
    /// Convert this tree node to Newick format string
    fn to_newick(&self) -> String {
        if self.is_leaf {
            format!("{}", self.id)
        } else {
            let left_str = if let Some(ref left) = self.left {
                left.to_newick()
            } else {
                "".to_string()
            };
            let right_str = if let Some(ref right) = self.right {
                right.to_newick()
            } else {
                "".to_string()
            };
            format!(
                "({}:{},{}:{})",
                left_str, self.height, right_str, self.height
            )
        }
    }
}

/// Build the dendrogram tree structure from linkage matrix
#[allow(dead_code)]
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

    // Build internal nodes from linkage _matrix
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
fn create_branches_recursive<F: Float + FromPrimitive + PartialOrd>(
    node: &TreeNode<F>,
    positions: &HashMap<usize, (F, F)>,
    branches: &mut Vec<Branch<F>>,
    threshold: F,
    config: &DendrogramConfig<F>,
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

        create_branches_recursive(left, positions, branches, threshold, config);
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

        create_branches_recursive(right, positions, branches, threshold, config);
    }
}

/// Create leaves for the dendrogram
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
fn calculateplot_bounds<F: Float>(branches: &[Branch<F>], leaves: &[Leaf]) -> (F, F, F, F) {
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
#[allow(dead_code)]
pub fn get_color_palette(_scheme: ColorScheme, ncolors: usize) -> Vec<String> {
    match _scheme {
        ColorScheme::Default => get_default_colors(ncolors),
        ColorScheme::HighContrast => get_high_contrast_colors(ncolors),
        ColorScheme::Viridis => get_viridis_colors(ncolors),
        ColorScheme::Plasma => get_plasma_colors(ncolors),
        ColorScheme::Grayscale => get_grayscale_colors(ncolors),
    }
}

/// Default color palette
#[allow(dead_code)]
fn get_default_colors(_ncolors: usize) -> Vec<String> {
    let base_colors = vec![
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(_ncolors)
        .map(|s| s.to_string())
        .collect()
}

/// High contrast color palette
#[allow(dead_code)]
fn get_high_contrast_colors(_ncolors: usize) -> Vec<String> {
    let base_colors = vec![
        "#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#800000", "#008000",
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(_ncolors)
        .map(|s| s.to_string())
        .collect()
}

/// Viridis color palette (approximation)
#[allow(dead_code)]
fn get_viridis_colors(_ncolors: usize) -> Vec<String> {
    let base_colors = vec![
        "#440154", "#482777", "#3f4a8a", "#31678e", "#26838f", "#1f9d8a", "#6cce5a", "#b6de2b",
        "#fee825", "#fff200",
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(_ncolors)
        .map(|s| s.to_string())
        .collect()
}

/// Plasma color palette (approximation)
#[allow(dead_code)]
fn get_plasma_colors(_ncolors: usize) -> Vec<String> {
    let base_colors = vec![
        "#0c0887", "#5c01a6", "#900da4", "#bf3984", "#e16462", "#f89441", "#fdc328", "#f0f921",
        "#fcffa4", "#ffffff",
    ];

    base_colors
        .into_iter()
        .cycle()
        .take(_ncolors)
        .map(|s| s.to_string())
        .collect()
}

/// Grayscale color palette
#[allow(dead_code)]
fn get_grayscale_colors(_ncolors: usize) -> Vec<String> {
    (0.._ncolors)
        .map(|i| {
            let intensity = 255 * i / _ncolors.max(1);
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
    fn test_create_dendrogramplot() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0]).unwrap();

        let linkage_matrix = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean).unwrap();
        let config = DendrogramConfig::default();

        let plot = create_dendrogramplot(linkage_matrix.view(), None, config).unwrap();

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

            let plot = create_dendrogramplot(linkage_matrix.view(), None, config).unwrap();

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
        let plot = create_dendrogramplot(linkage_matrix.view(), Some(&labels), config).unwrap();

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

        let bounds = calculateplot_bounds(&branches, &leaves);

        // Should encompass all points
        assert_eq!(bounds.0, -1.0); // min_x
        assert_eq!(bounds.1, 4.0); // max_x
        assert_eq!(bounds.2, -1.0); // min_y
        assert_eq!(bounds.3, 4.0); // max_y
    }
}

/// Tree node representation for Newick export
#[derive(Debug, Clone)]
enum NewickTreeNode {
    Leaf {
        id: usize,
        label: String,
    },
    Internal {
        id: usize,
        left: Box<NewickTreeNode>,
        right: Box<NewickTreeNode>,
        distance: f64,
    },
}

impl NewickTreeNode {
    /// Convert tree node to Newick format string
    fn to_newick(&self) -> String {
        match self {
            NewickTreeNode::Leaf { label, .. } => label.clone(),
            NewickTreeNode::Internal {
                left,
                right,
                distance,
                ..
            } => {
                format!(
                    "({}:{},{}:{})",
                    left.to_newick(),
                    distance,
                    right.to_newick(),
                    distance
                )
            }
        }
    }
}

/// Interactive dendrogram features for enhanced user experience
pub mod interactive {
    use super::*;
    use std::collections::BTreeMap;

    /// Interactive dendrogram configuration
    #[derive(Debug, Clone)]
    pub struct InteractiveDendrogramConfig<F: Float> {
        /// Base configuration
        pub base_config: DendrogramConfig<F>,
        /// Enable zooming functionality
        pub enable_zoom: bool,
        /// Enable cluster highlighting on hover
        pub enable_hover: bool,
        /// Enable click-to-cut functionality
        pub enable_click_cut: bool,
        /// Show tooltips with cluster information
        pub show_tooltips: bool,
        /// Animation duration in milliseconds
        pub animation_duration: u32,
    }

    impl<F: Float + FromPrimitive> Default for InteractiveDendrogramConfig<F> {
        fn default() -> Self {
            Self {
                base_config: DendrogramConfig::default(),
                enable_zoom: true,
                enable_hover: true,
                enable_click_cut: true,
                show_tooltips: true,
                animation_duration: 300,
            }
        }
    }

    /// Tooltip information for dendrogram nodes
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TooltipInfo {
        /// Cluster ID
        pub cluster_id: usize,
        /// Height/distance at this node
        pub height: f64,
        /// Number of samples in this cluster
        pub sample_count: usize,
        /// List of sample IDs in this cluster
        pub sample_ids: Vec<usize>,
        /// Additional metadata
        pub metadata: BTreeMap<String, String>,
    }

    /// Interactive dendrogram with enhanced features
    #[derive(Debug, Clone)]
    pub struct InteractiveDendrogram<F: Float> {
        /// Base dendrogram plot
        pub plot: DendrogramPlot<F>,
        /// Tooltip information for each node
        pub tooltips: Vec<TooltipInfo>,
        /// Interactive configuration
        pub config: InteractiveDendrogramConfig<F>,
        /// Cut points for dynamic cluster extraction
        pub cut_points: Vec<F>,
    }

    impl<F: Float + FromPrimitive + PartialOrd + Debug> InteractiveDendrogram<F> {
        /// Create a new interactive dendrogram
        pub fn new(
            linkage_matrix: ArrayView2<F>,
            labels: Option<&[String]>,
            config: InteractiveDendrogramConfig<F>,
        ) -> Result<Self> {
            let plot = create_dendrogramplot(linkage_matrix, labels, config.base_config.clone())?;
            let tooltips = Self::generate_tooltips(&linkage_matrix, &plot)?;

            Ok(Self {
                plot,
                tooltips,
                config,
                cut_points: Vec::new(),
            })
        }

        /// Generate tooltip information for each node
        fn generate_tooltips(
            linkage_matrix: &ArrayView2<F>,
            plot: &DendrogramPlot<F>,
        ) -> Result<Vec<TooltipInfo>> {
            let mut tooltips = Vec::new();
            let n_samples = linkage_matrix.shape()[0] + 1;

            // Generate tooltips for internal nodes
            for (i, row) in linkage_matrix.rows().into_iter().enumerate() {
                let cluster_id1 = row[0].to_usize().unwrap();
                let cluster_id2 = row[1].to_usize().unwrap();
                let height = row[2].to_f64().unwrap();
                let sample_count = row[3].to_usize().unwrap();

                let mut sample_ids = Vec::new();
                // In a full implementation, we would track which samples belong to each cluster
                // For now, we'll use a simplified approach
                if cluster_id1 < n_samples {
                    sample_ids.push(cluster_id1);
                }
                if cluster_id2 < n_samples {
                    sample_ids.push(cluster_id2);
                }

                let mut metadata = BTreeMap::new();
                metadata.insert("merge_order".to_string(), i.to_string());
                metadata.insert("left_child".to_string(), cluster_id1.to_string());
                metadata.insert("right_child".to_string(), cluster_id2.to_string());

                tooltips.push(TooltipInfo {
                    cluster_id: n_samples + i,
                    height,
                    sample_count,
                    sample_ids,
                    metadata,
                });
            }

            Ok(tooltips)
        }

        /// Add a cut point for dynamic cluster extraction
        pub fn add_cut_point(&mut self, height: F) {
            self.cut_points.push(height);
            self.cut_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }

        /// Remove a cut point
        pub fn remove_cut_point(&mut self, height: F) {
            self.cut_points.retain(|&h| h != height);
        }

        /// Get clusters at a specific cut height
        pub fn get_clusters_at_height(&self, height: F) -> Result<Vec<Vec<usize>>> {
            // Simplified cluster extraction - in a full implementation,
            // this would traverse the dendrogram tree
            Ok(vec![vec![0, 1], vec![2, 3]]) // Placeholder
        }

        /// Export to HTML with interactive features
        pub fn to_html(&self) -> Result<String> {
            let mut html = String::new();

            html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
            html.push_str("<title>Interactive Dendrogram</title>\n");
            html.push_str("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n");
            html.push_str("<style>\n");
            html.push_str(".branch { stroke: #333; stroke-width: 1; fill: none; }\n");
            html.push_str(".branch:hover { stroke-width: 2; }\n");
            html.push_str(".tooltip { position: absolute; background: #f9f9f9; border: 1px solid #ddd; padding: 5px; border-radius: 3px; }\n");
            html.push_str("</style>\n");
            html.push_str("</head>\n<body>\n");
            html.push_str("<div id=\"dendrogram\"></div>\n");

            // Add JavaScript for interactivity
            html.push_str("<script>\n");
            html.push_str("const width = 800, height = 600;\n");
            html.push_str("const svg = d3.select('#dendrogram').append('svg').attr('width', width).attr('height', height);\n");

            // Add branches
            html.push_str("const branches = [\n");
            for branch in &self.plot.branches {
                html.push_str(&format!(
                    "  {{start: [{}, {}], end: [{}, {}], height: {}}},\n",
                    branch.start.0.to_f64().unwrap(),
                    branch.start.1.to_f64().unwrap(),
                    branch.end.0.to_f64().unwrap(),
                    branch.end.1.to_f64().unwrap(),
                    branch.height.to_f64().unwrap()
                ));
            }
            html.push_str("];\n");

            // Add D3 visualization code
            html.push_str("svg.selectAll('.branch').data(branches).enter().append('line')\n");
            html.push_str("  .attr('class', 'branch')\n");
            html.push_str("  .attr('x1', d => d.start[0])\n");
            html.push_str("  .attr('y1', d => d.start[1])\n");
            html.push_str("  .attr('x2', d => d.end[0])\n");
            html.push_str("  .attr('y2', d => d.end[1]);\n");

            html.push_str("</script>\n");
            html.push_str("</body>\n</html>");

            Ok(html)
        }
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    #[allow(dead_code)]
    struct DummyStruct; // To satisfy the Serialize/Deserialize requirement
}

/// Animation support for dendrogram visualization
pub mod animation {
    use super::*;
    use std::time::Duration;

    /// Animation keyframe for dendrogram transitions
    #[derive(Debug, Clone)]
    pub struct AnimationKeyframe<F: Float> {
        /// Time offset in the animation (0.0 to 1.0)
        pub time: f64,
        /// Dendrogram state at this keyframe
        pub plot: DendrogramPlot<F>,
        /// Easing function to use
        pub easing: EasingFunction,
    }

    /// Easing functions for smooth animations
    #[derive(Debug, Clone, Copy)]
    pub enum EasingFunction {
        Linear,
        EaseIn,
        EaseOut,
        EaseInOut,
        Bounce,
        Elastic,
    }

    /// Animated dendrogram sequence
    #[derive(Debug, Clone)]
    pub struct AnimatedDendrogram<F: Float> {
        /// Animation keyframes
        pub keyframes: Vec<AnimationKeyframe<F>>,
        /// Total animation duration
        pub duration: Duration,
        /// Whether to loop the animation
        pub loop_animation: bool,
    }

    impl<F: Float + FromPrimitive> AnimatedDendrogram<F> {
        /// Create a new animated dendrogram
        pub fn new(duration: Duration) -> Self {
            Self {
                keyframes: Vec::new(),
                duration,
                loop_animation: false,
            }
        }

        /// Add a keyframe to the animation
        pub fn add_keyframe(&mut self, time: f64, plot: DendrogramPlot<F>, easing: EasingFunction) {
            self.keyframes
                .push(AnimationKeyframe { time, plot, easing });
            self.keyframes
                .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        }

        /// Interpolate dendrogram state at a specific time
        pub fn interpolate_at_time(&self, time: f64) -> Result<DendrogramPlot<F>> {
            if self.keyframes.is_empty() {
                return Err(ClusteringError::InvalidInput(
                    "No keyframes available".to_string(),
                ));
            }

            let normalized_time = time.clamp(0.0, 1.0);

            // Find surrounding keyframes
            let (before, after) = self.find_surrounding_keyframes(normalized_time);

            if before == after {
                return Ok(self.keyframes[before].plot.clone());
            }

            // Interpolate between keyframes
            let before_frame = &self.keyframes[before];
            let after_frame = &self.keyframes[after];

            let local_time =
                (normalized_time - before_frame.time) / (after_frame.time - before_frame.time);
            let eased_time = apply_easing(local_time, after_frame.easing);

            self.interpolateplots(&before_frame.plot, &after_frame.plot, eased_time)
        }

        fn find_surrounding_keyframes(&self, time: f64) -> (usize, usize) {
            for i in 0..self.keyframes.len() - 1 {
                if time >= self.keyframes[i].time && time <= self.keyframes[i + 1].time {
                    return (i, i + 1);
                }
            }
            (self.keyframes.len() - 1, self.keyframes.len() - 1)
        }

        fn interpolateplots(
            &self,
            plot1: &DendrogramPlot<F>,
            plot2: &DendrogramPlot<F>,
            t: f64,
        ) -> Result<DendrogramPlot<F>> {
            // Interpolate branch positions
            let mut interpolated_branches = Vec::new();
            let min_branches = plot1.branches.len().min(plot2.branches.len());

            for i in 0..min_branches {
                let branch1 = &plot1.branches[i];
                let branch2 = &plot2.branches[i];

                let start_x = lerp(
                    branch1.start.0.to_f64().unwrap(),
                    branch2.start.0.to_f64().unwrap(),
                    t,
                );
                let start_y = lerp(
                    branch1.start.1.to_f64().unwrap(),
                    branch2.start.1.to_f64().unwrap(),
                    t,
                );
                let end_x = lerp(
                    branch1.end.0.to_f64().unwrap(),
                    branch2.end.0.to_f64().unwrap(),
                    t,
                );
                let end_y = lerp(
                    branch1.end.1.to_f64().unwrap(),
                    branch2.end.1.to_f64().unwrap(),
                    t,
                );
                let height = lerp(
                    branch1.height.to_f64().unwrap(),
                    branch2.height.to_f64().unwrap(),
                    t,
                );

                interpolated_branches.push(Branch {
                    start: (F::from(start_x).unwrap(), F::from(start_y).unwrap()),
                    end: (F::from(end_x).unwrap(), F::from(end_y).unwrap()),
                    height: F::from(height).unwrap(),
                    cluster_id: branch1.cluster_id,
                    above_threshold: branch1.above_threshold,
                });
            }

            // Use leaves from the target plot
            let leaves = plot2.leaves.clone();
            let colors = plot2.colors.clone();
            let legend = plot2.legend.clone();

            // Interpolate bounds
            let bounds = (
                F::from(lerp(
                    plot1.bounds.0.to_f64().unwrap(),
                    plot2.bounds.0.to_f64().unwrap(),
                    t,
                ))
                .unwrap(),
                F::from(lerp(
                    plot1.bounds.1.to_f64().unwrap(),
                    plot2.bounds.1.to_f64().unwrap(),
                    t,
                ))
                .unwrap(),
                F::from(lerp(
                    plot1.bounds.2.to_f64().unwrap(),
                    plot2.bounds.2.to_f64().unwrap(),
                    t,
                ))
                .unwrap(),
                F::from(lerp(
                    plot1.bounds.3.to_f64().unwrap(),
                    plot2.bounds.3.to_f64().unwrap(),
                    t,
                ))
                .unwrap(),
            );

            Ok(DendrogramPlot {
                branches: interpolated_branches,
                leaves,
                colors,
                legend,
                bounds,
                config: plot1.config.clone(),
            })
        }
    }

    /// Linear interpolation
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t
    }

    /// Apply easing function to time value
    fn apply_easing(t: f64, easing: EasingFunction) -> f64 {
        match easing {
            EasingFunction::Linear => t,
            EasingFunction::EaseIn => t * t,
            EasingFunction::EaseOut => 1.0 - (1.0 - t).powi(2),
            EasingFunction::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - 2.0 * (1.0 - t).powi(2)
                }
            }
            EasingFunction::Bounce => {
                let n1 = 7.5625;
                let d1 = 2.75;

                if t < 1.0 / d1 {
                    n1 * t * t
                } else if t < 2.0 / d1 {
                    let t = t - 1.5 / d1;
                    n1 * t * t + 0.75
                } else if t < 2.5 / d1 {
                    let t = t - 2.25 / d1;
                    n1 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / d1;
                    n1 * t * t + 0.984375
                }
            }
            EasingFunction::Elastic => {
                let c4 = (2.0 * std::f64::consts::PI) / 3.0;
                if t == 0.0 {
                    0.0
                } else if t == 1.0 {
                    1.0
                } else {
                    -(2.0_f64.powf(10.0 * (t - 1.0)))
                        * ((t - 1.0) * c4 - std::f64::consts::PI / 2.0).sin()
                }
            }
        }
    }
}

/// Performance optimization utilities for large dendrograms
pub mod performance {
    use super::*;

    /// Performance configuration for large dendrogram rendering
    #[derive(Debug, Clone)]
    pub struct PerformanceConfig {
        /// Maximum number of branches to render
        pub max_branches: usize,
        /// Enable level-of-detail rendering
        pub enable_lod: bool,
        /// Minimum branch length threshold for rendering
        pub min_branch_length: f64,
        /// Enable branch culling based on viewport
        pub enable_culling: bool,
        /// Use simplified rendering for distant branches
        pub use_simplified_rendering: bool,
        /// Viewport bounds for culling
        pub viewport: Option<(f64, f64, f64, f64)>,
    }

    impl Default for PerformanceConfig {
        fn default() -> Self {
            Self {
                max_branches: 10000,
                enable_lod: true,
                min_branch_length: 0.001,
                enable_culling: true,
                use_simplified_rendering: true,
                viewport: None,
            }
        }
    }

    /// Optimized dendrogram for large datasets
    #[derive(Debug, Clone)]
    pub struct OptimizedDendrogram<F: Float> {
        /// Original dendrogram plot
        pub original: DendrogramPlot<F>,
        /// Optimized version for rendering
        pub optimized: DendrogramPlot<F>,
        /// Performance configuration
        pub config: PerformanceConfig,
        /// Rendering statistics
        pub stats: RenderingStats,
    }

    /// Rendering performance statistics
    #[derive(Debug, Clone, Default)]
    pub struct RenderingStats {
        /// Number of branches in original
        pub original_branches: usize,
        /// Number of branches after optimization
        pub optimized_branches: usize,
        /// Number of culled branches
        pub culled_branches: usize,
        /// Rendering complexity reduction ratio
        pub complexity_reduction: f64,
    }

    impl<F: Float + FromPrimitive + PartialOrd> OptimizedDendrogram<F> {
        /// Create optimized dendrogram from original
        pub fn new(plot: DendrogramPlot<F>, config: PerformanceConfig) -> Self {
            let mut optimized = plot.clone();
            let original_count = plot.branches.len();

            // Apply level-of-detail optimization
            if config.enable_lod {
                optimized = Self::apply_lod(&optimized, &config);
            }

            // Apply viewport culling
            if config.enable_culling && config.viewport.is_some() {
                optimized = Self::apply_culling(&optimized, &config);
            }

            // Limit maximum branches
            if optimized.branches.len() > config.max_branches {
                optimized.branches.truncate(config.max_branches);
                optimized.colors.truncate(config.max_branches);
            }

            let stats = RenderingStats {
                original_branches: original_count,
                optimized_branches: optimized.branches.len(),
                culled_branches: original_count - optimized.branches.len(),
                complexity_reduction: 1.0
                    - (optimized.branches.len() as f64 / original_count as f64),
            };

            Self {
                original: plot,
                optimized,
                config,
                stats,
            }
        }

        /// Apply level-of-detail optimization
        fn apply_lod(plot: &DendrogramPlot<F>, config: &PerformanceConfig) -> DendrogramPlot<F> {
            let mut optimized = plot.clone();

            // Remove branches shorter than minimum length
            optimized.branches.retain(|branch| {
                let length = ((branch.end.0 - branch.start.0).to_f64().unwrap().powi(2)
                    + (branch.end.1 - branch.start.1).to_f64().unwrap().powi(2))
                .sqrt();
                length >= config.min_branch_length
            });

            // Update colors to match retained branches
            optimized.colors.truncate(optimized.branches.len());

            optimized
        }

        /// Apply viewport culling
        fn apply_culling(
            plot: &DendrogramPlot<F>,
            config: &PerformanceConfig,
        ) -> DendrogramPlot<F> {
            if let Some((vx_min, vy_min, vx_max, vy_max)) = config.viewport {
                let mut optimized = plot.clone();
                let mut retained_indices = Vec::new();

                // Check which branches intersect with viewport
                for (i, branch) in plot.branches.iter().enumerate() {
                    let x1 = branch.start.0.to_f64().unwrap();
                    let y1 = branch.start.1.to_f64().unwrap();
                    let x2 = branch.end.0.to_f64().unwrap();
                    let y2 = branch.end.1.to_f64().unwrap();

                    // Simple bounding box intersection test
                    if (x1 <= vx_max && x2 >= vx_min && y1 <= vy_max && y2 >= vy_min)
                        || (x2 <= vx_max && x1 >= vx_min && y2 <= vy_max && y1 >= vy_min)
                    {
                        retained_indices.push(i);
                    }
                }

                // Keep only visible branches
                optimized.branches = retained_indices
                    .iter()
                    .map(|&i| plot.branches[i].clone())
                    .collect();
                optimized.colors = retained_indices
                    .iter()
                    .map(|&i| {
                        plot.colors
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| "#000000".to_string())
                    })
                    .collect();

                optimized
            } else {
                plot.clone()
            }
        }

        /// Update viewport for real-time culling
        pub fn update_viewport(&mut self, viewport: (f64, f64, f64, f64)) {
            self.config.viewport = Some(viewport);
            self.optimized = Self::apply_culling(&self.original, &self.config);
        }

        /// Get rendering complexity ratio
        pub fn get_complexity_ratio(&self) -> f64 {
            if self.stats.original_branches == 0 {
                1.0
            } else {
                self.stats.optimized_branches as f64 / self.stats.original_branches as f64
            }
        }
    }
}

/// Advanced export utilities for visualization
pub mod export {
    use super::*;
    use std::io::Write;

    /// Export format for dendrogram visualizations
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum VisualizationFormat {
        /// SVG vector format
        Svg,
        /// HTML with interactive features
        Html,
        /// JSON data for web frameworks
        Json,
        /// DOT format for Graphviz
        Dot,
        /// CSV coordinates for external plotting
        Csv,
        /// Newick tree format
        Newick,
    }

    /// Export dendrogram to various formats
    pub fn export_dendrogram<F: Float + FromPrimitive + Debug>(
        plot: &DendrogramPlot<F>,
        format: VisualizationFormat,
    ) -> Result<String> {
        match format {
            VisualizationFormat::Svg => export_to_svg(plot),
            VisualizationFormat::Html => export_to_html(plot),
            VisualizationFormat::Json => export_to_json(plot),
            VisualizationFormat::Dot => export_to_dot(plot),
            VisualizationFormat::Csv => export_to_csv(plot),
            VisualizationFormat::Newick => Err(ClusteringError::InvalidInput(
                "Newick export requires linkage matrix. Use export_dendrogram_newick instead."
                    .to_string(),
            )),
        }
    }

    /// Export dendrogram to Newick format
    pub fn export_dendrogram_newick<F: Float + FromPrimitive + Debug + std::fmt::Display>(
        linkage_matrix: &ArrayView2<F>,
        labels: Option<&[String]>,
    ) -> Result<String> {
        export_to_newick(linkage_matrix, labels)
    }

    /// Export to SVG format
    fn export_to_svg<F: Float + FromPrimitive + Debug>(plot: &DendrogramPlot<F>) -> Result<String> {
        let config = &plot.config;
        let mut svg = String::new();
        let (min_x, max_x, min_y, max_y) = plot.bounds;

        // Calculate viewport dimensions with padding
        let padding = 50.0;
        let width = (max_x - min_x).to_f64().unwrap() + 2.0 * padding;
        let height = (max_y - min_y).to_f64().unwrap() + 2.0 * padding;

        // Start SVG with enhanced styling
        svg.push_str(&format!(
            "<svg width=\"800\" height=\"600\" viewBox=\"{} {} {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            min_x.to_f64().unwrap() - padding,
            min_y.to_f64().unwrap() - padding,
            width,
            height
        ));

        // Add styles
        svg.push_str("<defs>\n");

        // Add shadow filter if enabled
        if config.styling.shadows {
            svg.push_str(
                r#"
                <filter id="drop-shadow">
                    <feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.3"/>
                </filter>
            "#,
            );
        }

        svg.push_str("</defs>\n");

        // Add background
        svg.push_str(&format!(
            "<rect width=\"100%\" height=\"100%\" fill=\"{}\"/>\n",
            config.styling.background_color
        ));

        // Add border if specified
        if let Some(ref border) = config.styling.border {
            svg.push_str(&format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\" rx=\"{}\"/>\n",
                min_x.to_f64().unwrap() - padding + border.width as f64,
                min_y.to_f64().unwrap() - padding + border.width as f64,
                width - 2.0 * border.width as f64,
                height - 2.0 * border.width as f64,
                border.color,
                border.width,
                border.radius
            ));
        }

        // Add grid if specified
        if let Some(ref grid) = config.styling.grid {
            let grid_style = match grid.style {
                BranchStyle::Solid => "solid",
                BranchStyle::Dashed => "dashed",
                BranchStyle::Dotted => "dotted",
                BranchStyle::DashDot => "dash-dot",
            };

            if grid.show_horizontal {
                // Add horizontal grid lines
                let grid_spacing = (max_y - min_y).to_f64().unwrap() / 10.0;
                for i in 1..10 {
                    let y = min_y.to_f64().unwrap() + i as f64 * grid_spacing;
                    svg.push_str(&format!(
                        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\" stroke-dasharray=\"{}\"/>\n",
                        min_x.to_f64().unwrap(),
                        y,
                        max_x.to_f64().unwrap(),
                        y,
                        grid.color,
                        grid.line_width,
                        if grid_style == "dashed" { "5,5" } else if grid_style == "dotted" { "2,2" } else { "none" }
                    ));
                }
            }

            if grid.show_vertical {
                // Add vertical grid lines
                let grid_spacing = (max_x - min_x).to_f64().unwrap() / 10.0;
                for i in 1..10 {
                    let x = min_x.to_f64().unwrap() + i as f64 * grid_spacing;
                    svg.push_str(&format!(
                        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\" stroke-dasharray=\"{}\"/>\n",
                        x,
                        min_y.to_f64().unwrap(),
                        x,
                        max_y.to_f64().unwrap(),
                        grid.color,
                        grid.line_width,
                        if grid_style == "dashed" { "5,5" } else if grid_style == "dotted" { "2,2" } else { "none" }
                    ));
                }
            }
        }

        // Add branches with enhanced styling
        let branch_style = match config.styling.branch_style {
            BranchStyle::Solid => "none",
            BranchStyle::Dashed => "10,5",
            BranchStyle::Dotted => "3,3",
            BranchStyle::DashDot => "10,5,3,5",
        };

        for (i, branch) in plot.branches.iter().enumerate() {
            let color = if i < plot.colors.len() {
                &plot.colors[i]
            } else {
                "#000000"
            };

            let shadow_filter = if config.styling.shadows {
                " filter=\"url(#drop-shadow)\""
            } else {
                ""
            };

            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\" stroke-dasharray=\"{}\"{}/>\n",
                branch.start.0.to_f64().unwrap(),
                branch.start.1.to_f64().unwrap(),
                branch.end.0.to_f64().unwrap(),
                branch.end.1.to_f64().unwrap(),
                color,
                config.line_width,
                branch_style,
                shadow_filter
            ));
        }

        // Add node markers
        if config.styling.node_markers.show_leaf_nodes {
            for leaf in &plot.leaves {
                let marker_svg = match config.styling.node_markers.markershape {
                    MarkerShape::Circle => format!(
                        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\"/>\n",
                        leaf.position.0,
                        leaf.position.1,
                        config.styling.node_markers.marker_size / 2.0,
                        config.styling.node_markers.marker_color
                    ),
                    MarkerShape::Square => format!(
                        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\"/>\n",
                        leaf.position.0 - config.styling.node_markers.marker_size as f64 / 2.0,
                        leaf.position.1 - config.styling.node_markers.marker_size as f64 / 2.0,
                        config.styling.node_markers.marker_size,
                        config.styling.node_markers.marker_size,
                        config.styling.node_markers.marker_color
                    ),
                    _ => format!(
                        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\"/>\n",
                        leaf.position.0,
                        leaf.position.1,
                        config.styling.node_markers.marker_size / 2.0,
                        config.styling.node_markers.marker_color
                    ),
                };
                svg.push_str(&marker_svg);
            }
        }

        // Add leaves with enhanced label styling
        for leaf in &plot.leaves {
            let font_weight = match config.styling.label_style.font_weight {
                FontWeight::Normal => "normal",
                FontWeight::Bold => "bold",
                FontWeight::Light => "300",
            };

            let transform = if config.styling.label_style.rotation != 0.0 {
                format!(
                    " transform=\"rotate({}, {}, {})\"",
                    config.styling.label_style.rotation, leaf.position.0, leaf.position.1
                )
            } else {
                String::new()
            };

            // Add label background if specified
            if let Some(ref bg_color) = config.styling.label_style.background {
                svg.push_str(&format!(
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" rx=\"2\"/>\n",
                    leaf.position.0 - config.styling.label_style.padding as f64 * 3.0,
                    leaf.position.1
                        - config.font_size as f64 / 2.0
                        - config.styling.label_style.padding as f64,
                    config.styling.label_style.padding as f64 * 6.0,
                    config.font_size as f64 + 2.0 * config.styling.label_style.padding as f64,
                    bg_color
                ));
            }

            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" font-size=\"{}\" font-family=\"{}\" font-weight=\"{}\" fill=\"{}\" text-anchor=\"middle\"{}>{}</text>\n",
                leaf.position.0,
                leaf.position.1,
                config.font_size,
                config.styling.label_style.font_family,
                font_weight,
                config.styling.label_style.color,
                transform,
                leaf.label
            ));
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Export to HTML format
    fn export_to_html<F: Float + FromPrimitive + Debug>(
        plot: &DendrogramPlot<F>,
    ) -> Result<String> {
        Ok("<html><body>HTML export not yet implemented</body></html>".to_string())
    }

    /// Export to JSON format
    /// Export dendrogram to Newick format
    fn export_to_newick<F: Float + FromPrimitive + Debug + std::fmt::Display>(
        linkage_matrix: &ArrayView2<F>,
        labels: Option<&[String]>,
    ) -> Result<String> {
        let n_samples = linkage_matrix.shape()[0] + 1;

        // Build tree structure from linkage _matrix
        let mut tree_nodes = Vec::new();

        // Create leaf nodes
        for i in 0..n_samples {
            let label = if let Some(labels) = labels {
                labels
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("node_{}", i))
            } else {
                format!("node_{}", i)
            };
            tree_nodes.push(TreeNode {
                id: i,
                height: F::zero(),
                left: None,
                right: None,
                count: 1,
                is_leaf: true,
            });
        }

        // Create internal nodes from linkage _matrix
        for (merge_idx, row) in linkage_matrix.outer_iter().enumerate() {
            let left_id = row[0].to_usize().unwrap();
            let right_id = row[1].to_usize().unwrap();
            let distance = row[2].to_f64().unwrap_or(0.0);

            let left_child = tree_nodes[left_id].clone();
            let right_child = tree_nodes[right_id].clone();

            let internal_node = TreeNode {
                id: n_samples + merge_idx,
                height: F::from(distance).unwrap(),
                left: Some(Box::new(left_child)),
                right: Some(Box::new(right_child)),
                count: tree_nodes[left_id].count + tree_nodes[right_id].count,
                is_leaf: false,
            };

            tree_nodes.push(internal_node);
        }

        // The root is the last node created
        let root = &tree_nodes[tree_nodes.len() - 1];

        // Convert to Newick format
        let newick = format!("{};", root.to_newick());
        Ok(newick)
    }

    fn export_to_json<F: Float + FromPrimitive + Debug>(
        plot: &DendrogramPlot<F>,
    ) -> Result<String> {
        #[cfg(feature = "serde")]
        {
            #[derive(serde::Serialize)]
            struct JsonBranch {
                start: (f64, f64),
                end: (f64, f64),
                height: f64,
                cluster_id: usize,
            }

            #[derive(serde::Serialize)]
            struct JsonPlot {
                branches: Vec<JsonBranch>,
                leaves: Vec<Leaf>,
                colors: Vec<String>,
                bounds: (f64, f64, f64, f64),
            }

            let json_branches: Vec<JsonBranch> = plot
                .branches
                .iter()
                .map(|b| JsonBranch {
                    start: (b.start.0.to_f64().unwrap(), b.start.1.to_f64().unwrap()),
                    end: (b.end.0.to_f64().unwrap(), b.end.1.to_f64().unwrap()),
                    height: b.height.to_f64().unwrap(),
                    cluster_id: b.cluster_id,
                })
                .collect();

            let jsonplot = JsonPlot {
                branches: json_branches,
                leaves: plot.leaves.clone(),
                colors: plot.colors.clone(),
                bounds: (
                    plot.bounds.0.to_f64().unwrap(),
                    plot.bounds.1.to_f64().unwrap(),
                    plot.bounds.2.to_f64().unwrap(),
                    plot.bounds.3.to_f64().unwrap(),
                ),
            };

            return serde_json::to_string_pretty(&jsonplot).map_err(|e| {
                ClusteringError::InvalidInput(format!("JSON serialization failed: {}", e))
            });
        }

        #[cfg(not(feature = "serde"))]
        {
            Err(ClusteringError::InvalidInput(
                "JSON export requires 'serde' feature to be enabled".to_string(),
            ))
        }
    }

    /// Export to DOT format for Graphviz
    fn export_to_dot<F: Float + FromPrimitive + Debug>(plot: &DendrogramPlot<F>) -> Result<String> {
        let mut dot = String::new();
        dot.push_str("digraph dendrogram {\n");
        dot.push_str("  rankdir=TB;\n");

        // Add nodes
        for (i, leaf) in plot.leaves.iter().enumerate() {
            dot.push_str(&format!("  leaf_{} [label=\"{}\"];\n", i, leaf.label));
        }

        // Add internal nodes and edges (simplified)
        for (i, branch) in plot.branches.iter().enumerate() {
            dot.push_str(&format!(
                "  internal_{} [label=\"{:.2}\"];\n",
                i,
                branch.height.to_f64().unwrap()
            ));
        }

        dot.push_str("}\n");
        Ok(dot)
    }

    /// Export to CSV format
    fn export_to_csv<F: Float + FromPrimitive + Debug>(plot: &DendrogramPlot<F>) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("type,start_x,start_y,end_x,end_y,height,cluster_id,label\n");

        // Add branches
        for branch in &plot.branches {
            csv.push_str(&format!(
                "branch,{},{},{},{},{},{},\n",
                branch.start.0.to_f64().unwrap(),
                branch.start.1.to_f64().unwrap(),
                branch.end.0.to_f64().unwrap(),
                branch.end.1.to_f64().unwrap(),
                branch.height.to_f64().unwrap(),
                branch.cluster_id
            ));
        }

        // Add leaves
        for leaf in &plot.leaves {
            csv.push_str(&format!(
                "leaf,{},{},,,,,{}\n",
                leaf.position.0, leaf.position.1, leaf.label
            ));
        }

        Ok(csv)
    }

    /// Save dendrogram to file
    pub fn save_dendrogram_to_file<F: Float + FromPrimitive + Debug, P: AsRef<std::path::Path>>(
        plot: &DendrogramPlot<F>,
        path: P,
        format: VisualizationFormat,
    ) -> Result<()> {
        let content = export_dendrogram(plot, format)?;
        let mut file = std::fs::File::create(path)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to create file: {}", e)))?;
        file.write_all(content.as_bytes())
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to write file: {}", e)))?;
        Ok(())
    }
}

/// Real-time clustering visualization for progressive algorithms
pub mod realtime {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Real-time dendrogram that updates as clustering progresses
    pub struct RealtimeDendrogram<F: Float> {
        /// Current dendrogram state
        currentplot: Arc<Mutex<Option<DendrogramPlot<F>>>>,
        /// Configuration for real-time updates
        config: RealtimeConfig,
        /// Update callback function
        update_callback: Option<Arc<dyn Fn(&DendrogramPlot<F>) + Send + Sync>>,
        /// Whether the visualization is active
        is_active: Arc<Mutex<bool>>,
    }

    impl<F: Float> std::fmt::Debug for RealtimeDendrogram<F> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RealtimeDendrogram")
                .field("currentplot", &"<Arc<Mutex<Option<DendrogramPlot<F>>>>>")
                .field("config", &self.config)
                .field("update_callback", &"<closure>")
                .field("is_active", &self.is_active)
                .finish()
        }
    }

    /// Configuration for real-time visualization
    #[derive(Debug, Clone)]
    pub struct RealtimeConfig {
        /// Update frequency in milliseconds
        pub update_interval_ms: u64,
        /// Maximum number of updates per second
        pub max_fps: u32,
        /// Buffer size for smooth animations
        pub buffer_size: usize,
        /// Enable automatic scaling
        pub auto_scale: bool,
        /// Show progress indicators
        pub show_progress: bool,
    }

    impl Default for RealtimeConfig {
        fn default() -> Self {
            Self {
                update_interval_ms: 100,
                max_fps: 30,
                buffer_size: 10,
                auto_scale: true,
                show_progress: true,
            }
        }
    }

    impl<F: Float + FromPrimitive + PartialOrd + Debug + Send + Sync + 'static> RealtimeDendrogram<F> {
        /// Create a new real-time dendrogram
        pub fn new(config: RealtimeConfig) -> Self {
            Self {
                currentplot: Arc::new(Mutex::new(None)),
                config,
                update_callback: None,
                is_active: Arc::new(Mutex::new(false)),
            }
        }

        /// Set update callback function
        pub fn set_update_callback<Callback>(&mut self, callback: Callback)
        where
            Callback: Fn(&DendrogramPlot<F>) + Send + Sync + 'static,
        {
            self.update_callback = Some(Arc::new(callback));
        }

        /// Start real-time visualization
        pub fn start(&self) {
            let mut is_active = self.is_active.lock().unwrap();
            *is_active = true;

            let plot_ref = Arc::clone(&self.currentplot);
            let callback_ref = self.update_callback.clone();
            let active_ref = Arc::clone(&self.is_active);
            let update_interval = Duration::from_millis(self.config.update_interval_ms);

            thread::spawn(move || {
                while *active_ref.lock().unwrap() {
                    let start_time = Instant::now();

                    if let Some(ref callback) = callback_ref {
                        if let Some(ref plot) = *plot_ref.lock().unwrap() {
                            callback(plot);
                        }
                    }

                    let elapsed = start_time.elapsed();
                    if elapsed < update_interval {
                        thread::sleep(update_interval - elapsed);
                    }
                }
            });
        }

        /// Stop real-time visualization
        pub fn stop(&self) {
            let mut is_active = self.is_active.lock().unwrap();
            *is_active = false;
        }

        /// Update the dendrogram with new data
        pub fn updateplot(&self, newplot: DendrogramPlot<F>) {
            let mut current = self.currentplot.lock().unwrap();
            *current = Some(newplot);
        }

        /// Get current plot snapshot
        pub fn get_currentplot(&self) -> Option<DendrogramPlot<F>> {
            self.currentplot.lock().unwrap().clone()
        }
    }

    /// Live clustering monitor for tracking algorithm progress
    #[derive(Debug)]
    pub struct LiveClusteringMonitor {
        /// Metrics buffer for real-time display
        metrics_buffer: Arc<Mutex<Vec<ClusteringMetrics>>>,
        /// Configuration
        config: MonitorConfig,
        /// Start time
        start_time: Instant,
    }

    /// Configuration for live monitoring
    #[derive(Debug, Clone)]
    pub struct MonitorConfig {
        /// Maximum metrics to keep in buffer
        pub max_buffer_size: usize,
        /// Update frequency
        pub update_frequency_ms: u64,
        /// Metrics to track
        pub tracked_metrics: Vec<MetricType>,
    }

    /// Types of metrics to track
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum MetricType {
        /// Number of clusters formed
        ClusterCount,
        /// Within-cluster sum of squares
        WCSS,
        /// Between-cluster sum of squares
        BCSS,
        /// Silhouette score
        Silhouette,
        /// Davies-Bouldin index
        DaviesBouldin,
        /// Calinski-Harabasz index
        CalinskiHarabasz,
        /// Processing time
        ProcessingTime,
        /// Memory usage
        MemoryUsage,
    }

    /// Clustering metrics at a point in time
    #[derive(Debug, Clone)]
    pub struct ClusteringMetrics {
        /// Timestamp
        pub timestamp: Instant,
        /// Metric values
        pub values: HashMap<MetricType, f64>,
        /// Number of iterations completed
        pub iteration: usize,
        /// Convergence status
        pub converged: bool,
    }

    impl Default for MonitorConfig {
        fn default() -> Self {
            Self {
                max_buffer_size: 1000,
                update_frequency_ms: 50,
                tracked_metrics: vec![
                    MetricType::ClusterCount,
                    MetricType::WCSS,
                    MetricType::ProcessingTime,
                ],
            }
        }
    }

    impl LiveClusteringMonitor {
        /// Create a new live monitoring instance
        pub fn new(config: MonitorConfig) -> Self {
            Self {
                metrics_buffer: Arc::new(Mutex::new(Vec::new())),
                config,
                start_time: Instant::now(),
            }
        }

        /// Record new metrics
        pub fn record_metrics(
            &self,
            iteration: usize,
            values: HashMap<MetricType, f64>,
            converged: bool,
        ) {
            let metrics = ClusteringMetrics {
                timestamp: Instant::now(),
                values,
                iteration,
                converged,
            };

            let mut buffer = self.metrics_buffer.lock().unwrap();
            buffer.push(metrics);

            // Keep buffer size under limit
            if buffer.len() > self.config.max_buffer_size {
                buffer.remove(0);
            }
        }

        /// Get latest metrics
        pub fn get_latest_metrics(&self) -> Option<ClusteringMetrics> {
            self.metrics_buffer.lock().unwrap().last().cloned()
        }

        /// Get all metrics in buffer
        pub fn get_all_metrics(&self) -> Vec<ClusteringMetrics> {
            self.metrics_buffer.lock().unwrap().clone()
        }

        /// Get metrics for a specific metric type over time
        pub fn get_metric_history(&self, metrictype: MetricType) -> Vec<(f64, f64)> {
            let buffer = self.metrics_buffer.lock().unwrap();
            buffer
                .iter()
                .filter_map(|m| {
                    m.values.get(&metrictype).map(|&value| {
                        let time_sec = m.timestamp.duration_since(self.start_time).as_secs_f64();
                        (time_sec, value)
                    })
                })
                .collect()
        }

        /// Clear metrics buffer
        pub fn clear_buffer(&self) {
            self.metrics_buffer.lock().unwrap().clear();
        }

        /// Export metrics to CSV format
        pub fn export_metrics_csv(&self) -> String {
            let mut csv = String::new();
            csv.push_str("timestamp,iteration,converged");

            // Add headers for tracked metrics
            for metric_type in &self.config.tracked_metrics {
                csv.push_str(&format!(",{:?}", metric_type));
            }
            csv.push('\n');

            let buffer = self.metrics_buffer.lock().unwrap();
            for metrics in buffer.iter() {
                let time_sec = metrics
                    .timestamp
                    .duration_since(self.start_time)
                    .as_secs_f64();
                csv.push_str(&format!(
                    "{:.3},{},{}",
                    time_sec, metrics.iteration, metrics.converged
                ));

                for metric_type in &self.config.tracked_metrics {
                    let value = metrics.values.get(metric_type).unwrap_or(&0.0);
                    csv.push_str(&format!(",{:.6}", value));
                }
                csv.push('\n');
            }

            csv
        }
    }
}
