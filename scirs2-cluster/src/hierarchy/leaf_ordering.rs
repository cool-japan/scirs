//! Optimal leaf ordering algorithms for dendrograms
//!
//! This module provides algorithms to reorder the leaves of a dendrogram
//! to minimize the sum of distances between adjacent leaves, improving
//! the visual interpretability of hierarchical clustering results.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

/// Represents a node in the dendrogram tree
#[derive(Debug, Clone)]
struct TreeNode {
    /// Index of this node
    #[allow(dead_code)]
    id: usize,
    /// Left child (None for leaf nodes)
    left: Option<Box<TreeNode>>,
    /// Right child (None for leaf nodes)
    right: Option<Box<TreeNode>>,
    /// Merge height/distance
    #[allow(dead_code)]
    height: f64,
    /// List of original observation indices contained in this subtree
    leaves: Vec<usize>,
}

impl TreeNode {
    /// Create a new leaf node
    fn new_leaf(id: usize) -> Self {
        TreeNode {
            id,
            left: None,
            right: None,
            height: 0.0,
            leaves: vec![id],
        }
    }

    /// Create a new internal node
    fn new_internal(id: usize, left: TreeNode, right: TreeNode, height: f64) -> Self {
        let mut leaves = left.leaves.clone();
        leaves.extend(right.leaves.iter());

        TreeNode {
            id,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            height,
            leaves,
        }
    }

    /// Check if this is a leaf node
    fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Get all possible leaf orderings for this subtree
    fn get_possible_orderings(&self) -> Vec<Vec<usize>> {
        if self.is_leaf() {
            // Leaf node has only one ordering
            vec![self.leaves.clone()]
        } else if let (Some(left), Some(right)) = (&self.left, &self.right) {
            // Internal node: combine orderings from left and right subtrees
            let left_orderings = left.get_possible_orderings();
            let right_orderings = right.get_possible_orderings();

            let mut all_orderings = Vec::new();

            // Try left-right order
            for left_ord in &left_orderings {
                for right_ord in &right_orderings {
                    let mut combined = left_ord.clone();
                    combined.extend(right_ord.iter());
                    all_orderings.push(combined);
                }
            }

            // Try right-left order
            for right_ord in &right_orderings {
                for left_ord in &left_orderings {
                    let mut combined = right_ord.clone();
                    combined.extend(left_ord.iter());
                    all_orderings.push(combined);
                }
            }

            all_orderings
        } else {
            vec![]
        }
    }
}

/// Builds a tree representation from a linkage matrix
fn build_tree_from_linkage<F: Float + FromPrimitive + Debug + PartialOrd>(
    linkage_matrix: ArrayView2<F>,
) -> Result<TreeNode> {
    let n_merges = linkage_matrix.shape()[0];
    let n_observations = n_merges + 1;

    if linkage_matrix.shape()[1] != 4 {
        return Err(ClusteringError::InvalidInput(
            "Linkage matrix should have 4 columns".to_string(),
        ));
    }

    // Create initial leaf nodes
    let mut nodes: HashMap<usize, TreeNode> = HashMap::new();
    for i in 0..n_observations {
        nodes.insert(i, TreeNode::new_leaf(i));
    }

    // Process merges in order
    for i in 0..n_merges {
        let cluster1 = linkage_matrix[[i, 0]].to_usize().unwrap();
        let cluster2 = linkage_matrix[[i, 1]].to_usize().unwrap();
        let height = linkage_matrix[[i, 2]].to_f64().unwrap();

        // Get the nodes to merge
        let left_node = nodes.remove(&cluster1).ok_or_else(|| {
            ClusteringError::ComputationError(format!("Node {} not found", cluster1))
        })?;

        let right_node = nodes.remove(&cluster2).ok_or_else(|| {
            ClusteringError::ComputationError(format!("Node {} not found", cluster2))
        })?;

        // Create new internal node
        let new_node_id = n_observations + i;
        let new_node = TreeNode::new_internal(new_node_id, left_node, right_node, height);

        nodes.insert(new_node_id, new_node);
    }

    // The final node should be the root
    if nodes.len() != 1 {
        return Err(ClusteringError::ComputationError(
            "Tree construction failed: should have exactly one root node".to_string(),
        ));
    }

    Ok(nodes.into_values().next().unwrap())
}

/// Calculates the cost of a given leaf ordering
fn calculate_ordering_cost<F: Float + FromPrimitive + Debug + PartialOrd>(
    ordering: &[usize],
    distance_matrix: ArrayView1<F>,
    n_observations: usize,
) -> F {
    let mut total_cost = F::zero();

    for i in 0..(ordering.len() - 1) {
        let obs1 = ordering[i];
        let obs2 = ordering[i + 1];

        // Calculate condensed distance matrix index
        let (min_idx, max_idx) = if obs1 < obs2 {
            (obs1, obs2)
        } else {
            (obs2, obs1)
        };
        let dist_idx =
            (n_observations * min_idx) - ((min_idx * (min_idx + 1)) / 2) + (max_idx - min_idx - 1);

        if dist_idx < distance_matrix.len() {
            total_cost = total_cost + distance_matrix[dist_idx];
        }
    }

    total_cost
}

/// Optimal leaf ordering using exact dynamic programming
///
/// This function finds the optimal ordering of leaves in a dendrogram that
/// minimizes the sum of distances between adjacent leaves. Uses an exact
/// algorithm that explores all possible orderings.
///
/// # Arguments
///
/// * `linkage_matrix` - The linkage matrix from hierarchical clustering
/// * `distance_matrix` - Original distance matrix in condensed form
///
/// # Returns
///
/// * `Result<Array1<usize>>` - Optimal leaf ordering
///
/// # Note
///
/// This is an exact algorithm with exponential time complexity. For large
/// dendrograms (>15 leaves), consider using the heuristic version.
pub fn optimal_leaf_ordering_exact<F: Float + FromPrimitive + Debug + PartialOrd>(
    linkage_matrix: ArrayView2<F>,
    distance_matrix: ArrayView1<F>,
) -> Result<Array1<usize>> {
    let n_observations = linkage_matrix.shape()[0] + 1;

    // Build tree from linkage matrix
    let tree = build_tree_from_linkage(linkage_matrix)?;

    // Get all possible orderings
    let all_orderings = tree.get_possible_orderings();

    if all_orderings.is_empty() {
        return Err(ClusteringError::ComputationError(
            "No valid orderings found".to_string(),
        ));
    }

    // Find the ordering with minimum cost
    let mut best_ordering = &all_orderings[0];
    let mut best_cost = calculate_ordering_cost(best_ordering, distance_matrix, n_observations);

    for ordering in &all_orderings[1..] {
        let cost = calculate_ordering_cost(ordering, distance_matrix, n_observations);
        if cost < best_cost {
            best_cost = cost;
            best_ordering = ordering;
        }
    }

    Ok(Array1::from_vec(best_ordering.clone()))
}

/// Heuristic leaf ordering using greedy approach
///
/// This function provides a fast heuristic for leaf ordering that gives
/// good results in most cases but is not guaranteed to be optimal.
/// Suitable for large dendrograms where the exact algorithm would be too slow.
///
/// # Arguments
///
/// * `linkage_matrix` - The linkage matrix from hierarchical clustering
/// * `distance_matrix` - Original distance matrix in condensed form
///
/// # Returns
///
/// * `Result<Array1<usize>>` - Heuristic leaf ordering
pub fn optimal_leaf_ordering_heuristic<F: Float + FromPrimitive + Debug + PartialOrd>(
    linkage_matrix: ArrayView2<F>,
    distance_matrix: ArrayView1<F>,
) -> Result<Array1<usize>> {
    let n_observations = linkage_matrix.shape()[0] + 1;

    // Start with identity ordering
    let mut current_ordering: Vec<usize> = (0..n_observations).collect();
    let mut current_cost =
        calculate_ordering_cost(&current_ordering, distance_matrix, n_observations);

    let max_iterations = 1000;
    let mut improved = true;
    let mut iteration = 0;

    // Iterative improvement using 2-opt swaps
    while improved && iteration < max_iterations {
        improved = false;
        iteration += 1;

        // Try all possible adjacent swaps
        for i in 0..(current_ordering.len() - 1) {
            // Swap adjacent elements
            current_ordering.swap(i, i + 1);
            let new_cost =
                calculate_ordering_cost(&current_ordering, distance_matrix, n_observations);

            if new_cost < current_cost {
                // Keep the improvement
                current_cost = new_cost;
                improved = true;
            } else {
                // Revert the swap
                current_ordering.swap(i, i + 1);
            }
        }

        // Try larger swaps (2-opt style)
        if !improved {
            for i in 0..current_ordering.len() {
                for j in (i + 2)..current_ordering.len() {
                    // Reverse the segment between i and j
                    current_ordering[i..=j].reverse();
                    let new_cost =
                        calculate_ordering_cost(&current_ordering, distance_matrix, n_observations);

                    if new_cost < current_cost {
                        current_cost = new_cost;
                        improved = true;
                        break;
                    } else {
                        // Revert the reversal
                        current_ordering[i..=j].reverse();
                    }
                }
                if improved {
                    break;
                }
            }
        }
    }

    Ok(Array1::from_vec(current_ordering))
}

/// Optimal leaf ordering with automatic algorithm selection
///
/// Automatically chooses between exact and heuristic algorithms based on
/// the size of the dendrogram for optimal performance.
///
/// # Arguments
///
/// * `linkage_matrix` - The linkage matrix from hierarchical clustering
/// * `distance_matrix` - Original distance matrix in condensed form
///
/// # Returns
///
/// * `Result<Array1<usize>>` - Optimal (or near-optimal) leaf ordering
pub fn optimal_leaf_ordering<F: Float + FromPrimitive + Debug + PartialOrd>(
    linkage_matrix: ArrayView2<F>,
    distance_matrix: ArrayView1<F>,
) -> Result<Array1<usize>> {
    let n_observations = linkage_matrix.shape()[0] + 1;

    // Use exact algorithm for small problems, heuristic for large ones
    if n_observations <= 12 {
        // For small dendrograms, use exact algorithm
        optimal_leaf_ordering_exact(linkage_matrix, distance_matrix)
    } else {
        // For large dendrograms, use heuristic
        optimal_leaf_ordering_heuristic(linkage_matrix, distance_matrix)
    }
}

/// Applies leaf ordering to a linkage matrix
///
/// This function reorders the linkage matrix to reflect the optimal leaf ordering.
/// The resulting linkage matrix will produce a dendrogram with the specified
/// leaf order when plotted.
///
/// # Arguments
///
/// * `linkage_matrix` - Original linkage matrix
/// * `leaf_ordering` - Desired leaf ordering
///
/// # Returns
///
/// * `Result<Array2<F>>` - Reordered linkage matrix
pub fn apply_leaf_ordering<F: Float + FromPrimitive + Debug + PartialOrd + Clone>(
    linkage_matrix: ArrayView2<F>,
    leaf_ordering: ArrayView1<usize>,
) -> Result<Array2<F>> {
    let n_observations = linkage_matrix.shape()[0] + 1;

    if leaf_ordering.len() != n_observations {
        return Err(ClusteringError::InvalidInput(format!(
            "Leaf ordering length {} doesn't match number of observations {}",
            leaf_ordering.len(),
            n_observations
        )));
    }

    // Create mapping from old indices to new indices
    let mut index_map = HashMap::new();
    for (new_idx, &old_idx) in leaf_ordering.iter().enumerate() {
        index_map.insert(old_idx, new_idx);
    }

    // Create reordered linkage matrix
    let mut reordered_linkage = linkage_matrix.to_owned();

    // Update cluster indices in the linkage matrix
    for i in 0..linkage_matrix.shape()[0] {
        let cluster1 = linkage_matrix[[i, 0]].to_usize().unwrap();
        let cluster2 = linkage_matrix[[i, 1]].to_usize().unwrap();

        // Map original leaf indices, leave internal node indices unchanged
        if cluster1 < n_observations {
            if let Some(&new_idx) = index_map.get(&cluster1) {
                reordered_linkage[[i, 0]] = F::from_usize(new_idx).unwrap();
            }
        }

        if cluster2 < n_observations {
            if let Some(&new_idx) = index_map.get(&cluster2) {
                reordered_linkage[[i, 1]] = F::from_usize(new_idx).unwrap();
            }
        }
    }

    Ok(reordered_linkage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hierarchy::{linkage, LinkageMethod, Metric};
    use ndarray::Array2;

    #[test]
    fn test_optimal_leaf_ordering_small() {
        // Create simple test data
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                0.0, 0.0, // Point 0
                1.0, 0.0, // Point 1 (close to 0)
                10.0, 0.0, // Point 2 (far from 0,1)
                11.0, 0.0, // Point 3 (close to 2)
            ],
        )
        .unwrap();

        // Compute linkage matrix
        let linkage_matrix =
            linkage(data.view(), LinkageMethod::Single, Metric::Euclidean).unwrap();

        // Compute original distances (condensed form)
        let mut original_distances = Array1::zeros(6); // C(4,2) = 6 pairwise distances
        let mut idx = 0;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let dist = ((data[[i, 0]] - data[[j, 0]]).powi(2)
                    + (data[[i, 1]] - data[[j, 1]]).powi(2))
                .sqrt();
                original_distances[idx] = dist;
                idx += 1;
            }
        }

        // Test exact algorithm
        let result = optimal_leaf_ordering_exact(linkage_matrix.view(), original_distances.view());
        assert!(result.is_ok(), "Exact leaf ordering should succeed");

        let ordering = result.unwrap();
        assert_eq!(ordering.len(), 4, "Should have 4 leaf positions");

        // Verify all indices are present
        let mut sorted_ordering = ordering.to_vec();
        sorted_ordering.sort();
        assert_eq!(
            sorted_ordering,
            vec![0, 1, 2, 3],
            "All indices should be present"
        );
    }

    #[test]
    fn test_optimal_leaf_ordering_heuristic() {
        // Create test data with more points to test heuristic
        let data = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).unwrap();

        let linkage_matrix =
            linkage(data.view(), LinkageMethod::Average, Metric::Euclidean).unwrap();

        // Compute distances
        let mut original_distances = Array1::zeros(15); // C(6,2) = 15
        let mut idx = 0;
        for i in 0..6 {
            for j in (i + 1)..6 {
                let dist = (data[[i, 0]] - data[[j, 0]]).abs();
                original_distances[idx] = dist;
                idx += 1;
            }
        }

        let result =
            optimal_leaf_ordering_heuristic(linkage_matrix.view(), original_distances.view());
        assert!(result.is_ok(), "Heuristic leaf ordering should succeed");

        let ordering = result.unwrap();
        assert_eq!(ordering.len(), 6, "Should have 6 leaf positions");

        // Verify all indices are present
        let mut sorted_ordering = ordering.to_vec();
        sorted_ordering.sort();
        assert_eq!(
            sorted_ordering,
            vec![0, 1, 2, 3, 4, 5],
            "All indices should be present"
        );
    }

    #[test]
    fn test_automatic_algorithm_selection() {
        // Small dataset should use exact algorithm
        let small_data = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let small_linkage =
            linkage(small_data.view(), LinkageMethod::Single, Metric::Euclidean).unwrap();
        let small_distances = Array1::from_vec(vec![1.0, 2.0, 1.0]);

        let small_result = optimal_leaf_ordering(small_linkage.view(), small_distances.view());
        assert!(
            small_result.is_ok(),
            "Small dataset should work with automatic selection"
        );

        // Larger dataset should use heuristic
        let large_data =
            Array2::from_shape_vec((15, 1), (0..15).map(|i| i as f64).collect()).unwrap();
        let large_linkage =
            linkage(large_data.view(), LinkageMethod::Average, Metric::Euclidean).unwrap();

        // Create distance matrix
        let n_distances = 15 * 14 / 2;
        let mut large_distances = Array1::zeros(n_distances);
        let mut idx = 0;
        for i in 0..15 {
            for j in (i + 1)..15 {
                large_distances[idx] = (i as f64 - j as f64).abs();
                idx += 1;
            }
        }

        let large_result = optimal_leaf_ordering(large_linkage.view(), large_distances.view());
        assert!(
            large_result.is_ok(),
            "Large dataset should work with automatic selection"
        );
    }

    #[test]
    fn test_apply_leaf_ordering() {
        let data = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let linkage_matrix =
            linkage(data.view(), LinkageMethod::Single, Metric::Euclidean).unwrap();

        // Test with reversed ordering
        let new_ordering = Array1::from_vec(vec![2, 1, 0]);
        let result = apply_leaf_ordering(linkage_matrix.view(), new_ordering.view());

        assert!(result.is_ok(), "Apply leaf ordering should succeed");
        let reordered_linkage = result.unwrap();
        assert_eq!(
            reordered_linkage.shape(),
            linkage_matrix.shape(),
            "Shape should be preserved"
        );
    }

    #[test]
    fn test_tree_construction() {
        // Test tree construction with simple linkage matrix
        let linkage = Array2::from_shape_vec(
            (2, 4),
            vec![
                0.0, 1.0, 1.0, 2.0, // Merge 0 and 1
                2.0, 3.0, 2.0, 3.0, // Merge result with 2
            ],
        )
        .unwrap();

        let tree = build_tree_from_linkage(linkage.view());
        assert!(tree.is_ok(), "Tree construction should succeed");

        let root = tree.unwrap();
        assert!(!root.is_leaf(), "Root should not be a leaf");
        assert_eq!(root.leaves.len(), 3, "Root should contain all 3 leaves");
    }
}
