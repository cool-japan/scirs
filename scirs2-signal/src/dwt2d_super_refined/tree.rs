//! Decomposition tree building and optimization for wavelet packets
//!
//! This module provides functions for building optimal wavelet packet decomposition
//! trees, basis selection, and tree traversal optimization.

use super::types::*;
use crate::error::{SignalError, SignalResult};
use ndarray::Array3;
use statrs::statistics::Statistics;

/// Build optimal decomposition tree for wavelet packet coefficients
pub fn build_optimal_decomposition_tree(
    coefficients: &Array3<f64>,
    cost_function: CostFunction,
    max_levels: usize,
    min_subband_size: usize,
) -> SignalResult<DecompositionTree> {
    if coefficients.is_empty() {
        return Err(SignalError::ValueError(
            "Empty coefficients array".to_string(),
        ));
    }

    let mut nodes = Vec::new();
    let mut node_id = 0;

    // Build tree recursively
    build_tree_recursive(
        coefficients,
        0,
        0,
        None,
        &mut nodes,
        &mut node_id,
        max_levels,
        min_subband_size,
    )?;

    // Find optimal basis
    let optimal_basis = find_optimal_basis(&nodes, cost_function)?;

    // Compute traversal statistics
    let traversal_stats = compute_traversal_statistics(&nodes);

    Ok(DecompositionTree {
        nodes,
        optimal_basis,
        cost_function,
        traversal_stats,
    })
}

/// Recursively build decomposition tree
fn build_tree_recursive(
    coefficients: &Array3<f64>,
    level: usize,
    index: usize,
    parent: Option<usize>,
    nodes: &mut Vec<TreeNode>,
    node_id: &mut usize,
    max_levels: usize,
    min_subband_size: usize,
) -> SignalResult<()> {
    let current_id = *node_id;
    *node_id += 1;

    // Extract coefficients for this node
    let node_coefficients = extract_node_coefficients(coefficients, level, index)?;

    // Compute node statistics
    let (energy, entropy) = compute_node_statistics(&node_coefficients)?;

    // Determine if this should be a leaf node
    let estimated_size = estimate_subband_size(coefficients, level);
    let is_leaf = level >= max_levels || estimated_size < min_subband_size;

    // Classify subband type
    let subband_type = classify_subband(index);

    let mut children = Vec::new();

    // If not a leaf, create children
    if !is_leaf && level < max_levels {
        // Create 4 children for 2D wavelet decomposition
        for child_index in 0..4 {
            children.push(*node_id);
            build_tree_recursive(
                coefficients,
                level + 1,
                index * 4 + child_index,
                Some(current_id),
                nodes,
                node_id,
                max_levels,
                min_subband_size,
            )?;
        }
    }

    // Create tree node
    let node = TreeNode {
        level,
        index,
        parent,
        children,
        energy,
        entropy,
        is_leaf,
        subband_type,
    };

    nodes.push(node);
    Ok(())
}

/// Compute statistics for a tree node
fn compute_node_statistics(coefficients: &[f64]) -> SignalResult<(f64, f64)> {
    if coefficients.is_empty() {
        return Ok((0.0, 0.0));
    }

    // Compute energy
    let energy = coefficients.iter().map(|&x| x * x).sum::<f64>();

    // Compute entropy
    let entropy = compute_shannon_entropy(coefficients)?;

    Ok((energy, entropy))
}

/// Extract coefficients for a specific tree node
fn extract_node_coefficients(
    coefficients: &Array3<f64>,
    level: usize,
    index: usize,
) -> SignalResult<Vec<f64>> {
    let shape = coefficients.shape();

    if level >= shape[0] {
        return Ok(Vec::new());
    }

    let level_coeffs = coefficients.slice(ndarray::s![level, .., ..]);
    let flat_coeffs: Vec<f64> = level_coeffs.iter().cloned().collect();

    // For simplicity, return all coefficients for the level
    // In practice, this would extract specific subband coefficients
    let start_idx = index * (flat_coeffs.len() / (4_usize.pow(level as u32)).max(1));
    let end_idx = ((index + 1) * (flat_coeffs.len() / (4_usize.pow(level as u32)).max(1)))
        .min(flat_coeffs.len());

    if start_idx < flat_coeffs.len() {
        Ok(flat_coeffs[start_idx..end_idx].to_vec())
    } else {
        Ok(Vec::new())
    }
}

/// Compute Shannon entropy of coefficients
fn compute_shannon_entropy(coefficients: &[f64]) -> SignalResult<f64> {
    if coefficients.is_empty() {
        return Ok(0.0);
    }

    // Normalize coefficients to probabilities
    let sum_abs: f64 = coefficients.iter().map(|&x| x.abs()).sum();

    if sum_abs == 0.0 {
        return Ok(0.0);
    }

    let mut entropy = 0.0;
    for &coeff in coefficients {
        let prob = coeff.abs() / sum_abs;
        if prob > 1e-12 {
            entropy -= prob * prob.ln();
        }
    }

    Ok(entropy)
}

/// Find optimal basis using dynamic programming
fn find_optimal_basis(nodes: &[TreeNode], cost_function: CostFunction) -> SignalResult<Vec<usize>> {
    if nodes.is_empty() {
        return Ok(Vec::new());
    }

    let mut optimal_basis = Vec::new();
    find_optimal_basis_recursive(nodes, 0, cost_function, &mut optimal_basis);
    Ok(optimal_basis)
}

/// Recursively find optimal basis
fn find_optimal_basis_recursive(
    nodes: &[TreeNode],
    node_index: usize,
    cost_function: CostFunction,
    optimal_basis: &mut Vec<usize>,
) {
    if node_index >= nodes.len() {
        return;
    }

    let node = &nodes[node_index];

    if node.is_leaf || node.children.is_empty() {
        // This is a leaf node, include it in the basis
        optimal_basis.push(node_index);
        return;
    }

    // Compute cost of keeping this node vs expanding to children
    let node_cost = compute_node_cost(node, cost_function);

    let mut children_cost = 0.0;
    for &child_index in &node.children {
        if child_index < nodes.len() {
            children_cost += compute_node_cost(&nodes[child_index], cost_function);
        }
    }

    if node_cost <= children_cost {
        // Keep this node
        optimal_basis.push(node_index);
    } else {
        // Expand to children
        for &child_index in &node.children {
            find_optimal_basis_recursive(nodes, child_index, cost_function, optimal_basis);
        }
    }
}

/// Compute cost for a tree node based on the cost function
fn compute_node_cost(_node: &TreeNode, cost_function: CostFunction) -> f64 {
    match cost_function {
        CostFunction::Entropy => _node.entropy,
        CostFunction::Energy => _node.energy,
        CostFunction::LogEntropy => {
            if _node.entropy > 0.0 {
                _node.entropy.ln()
            } else {
                0.0
            }
        }
        CostFunction::Sure => {
            // Simplified SURE estimate
            _node.energy + 2.0 * _node.entropy
        }
        CostFunction::Minimax => {
            // Simplified minimax criterion
            _node.energy.max(_node.entropy)
        }
        CostFunction::Adaptive => {
            // Adaptive cost combining multiple criteria
            0.5 * _node.entropy + 0.3 * _node.energy + 0.2 * _node.entropy.ln().max(0.0)
        }
    }
}

/// Compute traversal statistics for the tree
fn compute_traversal_statistics(nodes: &[TreeNode]) -> TreeTraversalStats {
    let total_nodes = nodes.len();
    let leaf_nodes = nodes.iter().filter(|n| n.is_leaf).count();

    let max_depth = nodes.iter().map(|n| n.level).max().unwrap_or(0);

    // Compute average branching factor
    let non_leaf_nodes: Vec<_> = nodes.iter().filter(|n| !n.is_leaf).collect();
    let avg_branching_factor = if !non_leaf_nodes.is_empty() {
        non_leaf_nodes
            .iter()
            .map(|n| n.children.len())
            .sum::<usize>() as f64
            / non_leaf_nodes.len() as f64
    } else {
        0.0
    };

    TreeTraversalStats {
        total_nodes,
        leaf_nodes,
        max_depth,
        avg_branching_factor,
    }
}

/// Estimate subband size at a given level
fn estimate_subband_size(coefficients: &Array3<f64>, level: usize) -> usize {
    let shape = coefficients.shape();
    if level < shape[0] {
        // Estimate based on coefficient array dimensions
        let level_size = shape[1] * shape[2];
        level_size / (4_usize.pow(level as u32)).max(1)
    } else {
        0
    }
}

/// Classify subband type based on index
fn classify_subband(index: usize) -> SubbandType {
    match index % 4 {
        0 => SubbandType::Approximation,
        1 => SubbandType::HorizontalDetail,
        2 => SubbandType::VerticalDetail,
        3 => SubbandType::DiagonalDetail,
        _ => SubbandType::Approximation, // Should never happen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_build_decomposition_tree() {
        let coefficients = Array3::zeros((3, 16, 16));
        let tree = build_optimal_decomposition_tree(&coefficients, CostFunction::Entropy, 2, 4);

        assert!(tree.is_ok());
        let tree = tree.unwrap();
        assert!(!tree.nodes.is_empty());
        assert!(tree.traversal_stats.total_nodes > 0);
    }

    #[test]
    fn test_shannon_entropy() {
        let coefficients = vec![1.0, 2.0, 3.0, 4.0];
        let entropy = compute_shannon_entropy(&coefficients);

        assert!(entropy.is_ok());
        let entropy = entropy.unwrap();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_node_statistics() {
        let coefficients = vec![1.0, -2.0, 3.0, -4.0];
        let (energy, entropy) = compute_node_statistics(&coefficients).unwrap();

        assert_eq!(energy, 30.0); // 1 + 4 + 9 + 16
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_empty_coefficients() {
        let empty_coefficients = Array3::zeros((0, 0, 0));
        let result =
            build_optimal_decomposition_tree(&empty_coefficients, CostFunction::Entropy, 2, 4);

        assert!(result.is_err());
    }

    #[test]
    fn test_subband_classification() {
        assert_eq!(classify_subband(0), SubbandType::Approximation);
        assert_eq!(classify_subband(1), SubbandType::HorizontalDetail);
        assert_eq!(classify_subband(2), SubbandType::VerticalDetail);
        assert_eq!(classify_subband(3), SubbandType::DiagonalDetail);
        assert_eq!(classify_subband(4), SubbandType::Approximation);
    }
}
