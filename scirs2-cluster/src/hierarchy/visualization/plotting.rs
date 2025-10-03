//! Dendrogram plotting functionality
//!
//! This module contains the core functionality for creating and positioning
//! dendrogram plots from linkage matrices.

use scirs2_core::ndarray::ArrayView2;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use super::types::*;
use crate::error::{ClusteringError, Result};

/// Internal tree node structure for dendrogram construction
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
    leaf_count: usize,
}

impl<F: Float + std::fmt::Display> TreeNode<F> {
    /// Create a new leaf node
    fn new_leaf(id: usize) -> Self {
        Self {
            id,
            height: F::zero(),
            left: None,
            right: None,
            leaf_count: 1,
        }
    }

    /// Create a new internal node
    fn new_internal(id: usize, height: F, left: TreeNode<F>, right: TreeNode<F>) -> Self {
        let leaf_count = left.leaf_count + right.leaf_count;
        Self {
            id,
            height,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            leaf_count,
        }
    }

    /// Check if this node is a leaf
    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }
}

/// Create an enhanced dendrogram plot from a linkage matrix
///
/// This function takes a linkage matrix (as produced by hierarchical clustering)
/// and creates a comprehensive dendrogram visualization with advanced styling options.
///
/// # Arguments
/// * `linkage_matrix` - The linkage matrix from hierarchical clustering
/// * `labels` - Optional labels for the leaf nodes
/// * `config` - Configuration options for the dendrogram
///
/// # Returns
/// * `Result<DendrogramPlot<F>>` - The complete dendrogram plot structure
///
/// # Example
/// ```rust
/// use scirs2_core::ndarray::Array2;
/// use scirs2_cluster::hierarchy::visualization::{create_dendrogramplot, DendrogramConfig};
///
/// let linkage = Array2::from_shape_vec((3, 4), vec![
///     0.0, 1.0, 0.1, 2.0,
///     2.0, 3.0, 0.2, 2.0,
///     4.0, 5.0, 0.3, 4.0,
/// ]).unwrap();
/// let labels = Some(vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]);
/// let config = DendrogramConfig::default();
/// let plot = create_dendrogramplot(linkage.view(), labels.as_deref(), config).unwrap();
/// ```
pub fn create_dendrogramplot<F: Float + FromPrimitive + PartialOrd + Debug + std::fmt::Display>(
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
        calculate_auto_threshold(linkage_matrix, config.color_threshold.target_clusters)?
    } else {
        config.color_threshold.threshold
    };

    // Build the dendrogram tree structure
    let tree = build_dendrogram_tree(linkage_matrix)?;

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
        config,
    })
}

/// Build the tree structure from a linkage matrix
fn build_dendrogram_tree<F: Float + FromPrimitive + Debug + std::fmt::Display>(
    linkage_matrix: ArrayView2<F>,
) -> Result<TreeNode<F>> {
    let n_samples = linkage_matrix.shape()[0] + 1;
    let mut nodes: HashMap<usize, TreeNode<F>> = HashMap::new();

    // Create leaf nodes
    for i in 0..n_samples {
        nodes.insert(i, TreeNode::new_leaf(i));
    }

    // Create internal nodes from linkage matrix
    for (i, row) in linkage_matrix.outer_iter().enumerate() {
        let left_id = row[0].to_usize().unwrap();
        let right_id = row[1].to_usize().unwrap();
        let distance = row[2];

        let left_node = nodes.remove(&left_id).ok_or_else(|| {
            ClusteringError::InvalidInput(format!("Invalid left cluster ID: {}", left_id))
        })?;

        let right_node = nodes.remove(&right_id).ok_or_else(|| {
            ClusteringError::InvalidInput(format!("Invalid right cluster ID: {}", right_id))
        })?;

        let internal_id = n_samples + i;
        let internal_node = TreeNode::new_internal(internal_id, distance, left_node, right_node);

        nodes.insert(internal_id, internal_node);
    }

    // Return the root node (should be the only remaining node)
    let root_id = 2 * n_samples - 2;
    nodes.remove(&root_id).ok_or_else(|| {
        ClusteringError::ComputationError("Failed to construct dendrogram tree".to_string())
    })
}

/// Calculate positions for all nodes in the dendrogram
fn calculate_node_positions<F: Float + FromPrimitive + std::fmt::Display>(
    root: &TreeNode<F>,
    n_samples: usize,
    orientation: DendrogramOrientation,
) -> HashMap<usize, (F, F)> {
    let mut positions = HashMap::new();
    let mut leaf_counter = 0;

    calculate_positions_recursive(root, &mut positions, &mut leaf_counter, orientation);
    positions
}

/// Recursively calculate positions for nodes
fn calculate_positions_recursive<F: Float + FromPrimitive + std::fmt::Display>(
    node: &TreeNode<F>,
    positions: &mut HashMap<usize, (F, F)>,
    leaf_counter: &mut usize,
    orientation: DendrogramOrientation,
) -> F {
    if node.is_leaf() {
        let x_pos = F::from(*leaf_counter).unwrap();
        let y_pos = F::zero();

        let pos = match orientation {
            DendrogramOrientation::Top => (x_pos, y_pos),
            DendrogramOrientation::Bottom => (x_pos, -y_pos),
            DendrogramOrientation::Left => (y_pos, x_pos),
            DendrogramOrientation::Right => (-y_pos, x_pos),
        };

        positions.insert(node.id, pos);
        *leaf_counter += 1;
        x_pos
    } else {
        let left = node.left.as_ref().unwrap();
        let right = node.right.as_ref().unwrap();

        let left_x = calculate_positions_recursive(left, positions, leaf_counter, orientation);
        let right_x = calculate_positions_recursive(right, positions, leaf_counter, orientation);

        let x_pos = (left_x + right_x) / F::from(2).unwrap();
        let y_pos = node.height;

        let pos = match orientation {
            DendrogramOrientation::Top => (x_pos, y_pos),
            DendrogramOrientation::Bottom => (x_pos, -y_pos),
            DendrogramOrientation::Left => (y_pos, x_pos),
            DendrogramOrientation::Right => (-y_pos, x_pos),
        };

        positions.insert(node.id, pos);
        x_pos
    }
}

/// Create branch structures for visualization
fn create_branches<F: Float + FromPrimitive + PartialOrd + std::fmt::Display>(
    root: &TreeNode<F>,
    positions: &HashMap<usize, (F, F)>,
    threshold: F,
    config: &DendrogramConfig<F>,
) -> Result<Vec<Branch<F>>> {
    let mut branches = Vec::new();
    create_branches_recursive(root, positions, threshold, config, &mut branches)?;
    Ok(branches)
}

/// Recursively create branches
fn create_branches_recursive<F: Float + FromPrimitive + PartialOrd + std::fmt::Display>(
    node: &TreeNode<F>,
    positions: &HashMap<usize, (F, F)>,
    threshold: F,
    config: &DendrogramConfig<F>,
    branches: &mut Vec<Branch<F>>,
) -> Result<()> {
    if !node.is_leaf() {
        let left = node.left.as_ref().unwrap();
        let right = node.right.as_ref().unwrap();

        let node_pos = positions.get(&node.id).unwrap();
        let left_pos = positions.get(&left.id).unwrap();
        let right_pos = positions.get(&right.id).unwrap();

        // Determine color based on threshold
        let color = if node.height > threshold {
            config.color_threshold.above_color.clone()
        } else {
            config.color_threshold.below_color.clone()
        };

        // Create horizontal line from left child to right child
        let horizontal_branch = Branch {
            start: *left_pos,
            end: *right_pos,
            distance: node.height,
            cluster_id: Some(node.id),
            color: color.clone(),
            line_width: Some(config.line_width),
        };
        branches.push(horizontal_branch);

        // Create vertical line from horizontal line to node
        let mid_x = (left_pos.0 + right_pos.0) / F::from(2).unwrap();
        let vertical_start = (mid_x, left_pos.1.max(right_pos.1));
        let vertical_branch = Branch {
            start: vertical_start,
            end: *node_pos,
            distance: node.height,
            cluster_id: Some(node.id),
            color,
            line_width: Some(config.line_width),
        };
        branches.push(vertical_branch);

        // Recursively process children
        create_branches_recursive(left, positions, threshold, config, branches)?;
        create_branches_recursive(right, positions, threshold, config, branches)?;
    }

    Ok(())
}

/// Create leaf representations
fn create_leaves<F: Float + FromPrimitive>(
    positions: &HashMap<usize, (F, F)>,
    labels: Option<&[String]>,
    n_samples: usize,
    orientation: DendrogramOrientation,
) -> Vec<Leaf> {
    let mut leaves = Vec::new();

    for i in 0..n_samples {
        if let Some(pos) = positions.get(&i) {
            let label = if let Some(labels) = labels {
                labels
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("Sample {}", i))
            } else {
                format!("Sample {}", i)
            };

            let leaf = Leaf {
                position: (pos.0.to_f64().unwrap(), pos.1.to_f64().unwrap()),
                label,
                color: "#333333".to_string(),
                data_index: i,
            };

            leaves.push(leaf);
        }
    }

    leaves
}

/// Assign colors to branches
fn assign_branch_colors<F: Float>(
    branches: &[Branch<F>],
    config: &DendrogramConfig<F>,
) -> Vec<String> {
    branches.iter().map(|branch| branch.color.clone()).collect()
}

/// Create legend for the plot
fn create_legend<F: Float + std::fmt::Display>(
    config: &DendrogramConfig<F>,
    threshold: F,
) -> Vec<LegendEntry> {
    if config.color_threshold.auto_threshold || config.color_threshold.threshold > F::zero() {
        vec![
            LegendEntry {
                color: config.color_threshold.above_color.clone(),
                label: format!("Distance > {}", threshold),
                threshold: Some(threshold.to_f64().unwrap_or(0.0)),
            },
            LegendEntry {
                color: config.color_threshold.below_color.clone(),
                label: format!("Distance d {}", threshold),
                threshold: Some(threshold.to_f64().unwrap_or(0.0)),
            },
        ]
    } else {
        Vec::new()
    }
}

/// Calculate plot bounds
fn calculate_plot_bounds<F: Float>(branches: &[Branch<F>], leaves: &[Leaf]) -> (F, F, F, F) {
    if branches.is_empty() && leaves.is_empty() {
        return (F::zero(), F::zero(), F::zero(), F::zero());
    }

    let mut min_x = F::infinity();
    let mut max_x = F::neg_infinity();
    let mut min_y = F::infinity();
    let mut max_y = F::neg_infinity();

    // Consider branch bounds
    for branch in branches {
        min_x = min_x.min(branch.start.0).min(branch.end.0);
        max_x = max_x.max(branch.start.0).max(branch.end.0);
        min_y = min_y.min(branch.start.1).min(branch.end.1);
        max_y = max_y.max(branch.start.1).max(branch.end.1);
    }

    // Consider leaf bounds
    for leaf in leaves {
        let leaf_x = F::from(leaf.position.0).unwrap();
        let leaf_y = F::from(leaf.position.1).unwrap();
        min_x = min_x.min(leaf_x);
        max_x = max_x.max(leaf_x);
        min_y = min_y.min(leaf_y);
        max_y = max_y.max(leaf_y);
    }

    (min_x, max_x, min_y, max_y)
}

/// Calculate automatic threshold based on desired number of clusters
fn calculate_auto_threshold<F: Float + FromPrimitive + PartialOrd>(
    linkage_matrix: ArrayView2<F>,
    target_clusters: Option<usize>,
) -> Result<F> {
    let target = target_clusters.unwrap_or(4);
    let n_merges = linkage_matrix.shape()[0];

    if target > n_merges {
        return Ok(F::zero());
    }

    // Get the distance at which we have the target number of clusters
    let merge_index = n_merges - target;
    let threshold = linkage_matrix[[merge_index, 2]];

    Ok(threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_tree_node_creation() {
        let leaf = TreeNode::<f64>::new_leaf(0);
        assert!(leaf.is_leaf());
        assert_eq!(leaf.id, 0);
        assert_eq!(leaf.leaf_count, 1);

        let left = TreeNode::new_leaf(0);
        let right = TreeNode::new_leaf(1);
        let internal = TreeNode::new_internal(2, 0.5, left, right);
        assert!(!internal.is_leaf());
        assert_eq!(internal.leaf_count, 2);
    }

    #[test]
    fn test_build_dendrogram_tree() {
        let linkage = Array2::from_shape_vec(
            (3, 4),
            vec![0.0, 1.0, 0.1, 2.0, 2.0, 3.0, 0.2, 2.0, 4.0, 5.0, 0.3, 4.0],
        )
        .unwrap();

        let tree = build_dendrogram_tree(linkage.view()).unwrap();
        assert!(!tree.is_leaf());
        assert_eq!(tree.leaf_count, 4);
    }

    #[test]
    fn test_calculate_auto_threshold() {
        let linkage = Array2::from_shape_vec(
            (3, 4),
            vec![0.0, 1.0, 0.1, 2.0, 2.0, 3.0, 0.2, 2.0, 4.0, 5.0, 0.3, 4.0],
        )
        .unwrap();

        let threshold = calculate_auto_threshold(linkage.view(), Some(2)).unwrap();
        assert!((threshold - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_create_dendrogramplot() {
        let linkage = Array2::from_shape_vec((1, 4), vec![0.0, 1.0, 0.1, 2.0]).unwrap();

        let labels = Some(vec!["A".to_string(), "B".to_string()]);
        let config = DendrogramConfig::default();

        let plot = create_dendrogramplot(linkage.view(), labels.as_deref(), config).unwrap();
        assert!(!plot.branches.is_empty());
        assert_eq!(plot.leaves.len(), 2);
    }
}
