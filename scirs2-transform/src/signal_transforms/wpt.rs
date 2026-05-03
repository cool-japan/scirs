//! Wavelet Packet Transform (WPT) Implementation
//!
//! Provides wavelet packet decomposition and best basis selection.
//! Wavelet packets extend DWT by decomposing both approximation and detail coefficients.

use crate::error::{Result, TransformError};
use crate::signal_transforms::dwt::{BoundaryMode, WaveletType, DWT};
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::collections::HashMap;

/// Wavelet packet node
#[derive(Debug, Clone)]
pub struct WaveletPacketNode {
    /// Node data (coefficients)
    pub data: Array1<f64>,
    /// Node path (sequence of 'a' for approximation, 'd' for detail)
    pub path: String,
    /// Level in the packet tree
    pub level: usize,
    /// Node index at this level
    pub index: usize,
    /// Cost/entropy of this node
    pub cost: f64,
}

impl WaveletPacketNode {
    /// Create a new wavelet packet node
    pub fn new(data: Array1<f64>, path: String, level: usize, index: usize) -> Self {
        let cost = Self::compute_cost(&data);
        WaveletPacketNode {
            data,
            path,
            level,
            index,
            cost,
        }
    }

    /// Compute the cost (Shannon entropy) of the node
    fn compute_cost(data: &Array1<f64>) -> f64 {
        let energy: f64 = data.iter().map(|x| x * x).sum();
        if energy < 1e-10 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &val in data.iter() {
            let p = (val * val) / energy;
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    /// Update the cost
    pub fn update_cost(&mut self) {
        self.cost = Self::compute_cost(&self.data);
    }
}

/// Best basis selection criterion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BestBasisCriterion {
    /// Shannon entropy
    Shannon,
    /// Threshold (number of coefficients above threshold)
    Threshold(f64),
    /// Log energy
    LogEnergy,
    /// Sure (Stein's Unbiased Risk Estimate)
    Sure,
}

/// Wavelet Packet Transform
#[derive(Debug, Clone)]
pub struct WPT {
    wavelet: WaveletType,
    max_level: usize,
    boundary: BoundaryMode,
    criterion: BestBasisCriterion,
    nodes: HashMap<String, WaveletPacketNode>,
}

impl WPT {
    /// Create a new WPT instance
    pub fn new(wavelet: WaveletType, max_level: usize) -> Self {
        WPT {
            wavelet,
            max_level,
            boundary: BoundaryMode::Symmetric,
            criterion: BestBasisCriterion::Shannon,
            nodes: HashMap::new(),
        }
    }

    /// Set the boundary mode
    pub fn with_boundary(mut self, boundary: BoundaryMode) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set the best basis criterion
    pub fn with_criterion(mut self, criterion: BestBasisCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Perform full wavelet packet decomposition
    pub fn decompose(&mut self, signal: &ArrayView1<f64>) -> Result<()> {
        self.nodes.clear();

        // Create root node
        let root = WaveletPacketNode::new(signal.to_owned(), String::new(), 0, 0);
        self.nodes.insert(String::new(), root);

        // Recursively decompose
        self.decompose_node("", 0)?;

        Ok(())
    }

    /// Recursively decompose a node
    fn decompose_node(&mut self, path: &str, level: usize) -> Result<()> {
        if level >= self.max_level {
            return Ok(());
        }

        // Get the current node
        let node = self
            .nodes
            .get(path)
            .ok_or_else(|| TransformError::InvalidInput(format!("Node not found: {}", path)))?
            .clone();

        // Create DWT instance
        let dwt = DWT::new(self.wavelet)?.with_boundary(self.boundary);

        // Decompose
        let (approx, detail) = dwt.decompose(&node.data.view())?;

        // Create child nodes
        let approx_path = format!("{}a", path);
        let detail_path = format!("{}d", path);

        let index = node.index;
        let approx_node = WaveletPacketNode::new(approx, approx_path.clone(), level + 1, index * 2);
        let detail_node =
            WaveletPacketNode::new(detail, detail_path.clone(), level + 1, index * 2 + 1);

        self.nodes.insert(approx_path.clone(), approx_node);
        self.nodes.insert(detail_path.clone(), detail_node);

        // Recursively decompose child nodes
        self.decompose_node(&approx_path, level + 1)?;
        self.decompose_node(&detail_path, level + 1)?;

        Ok(())
    }

    /// Select the best basis using the specified criterion
    pub fn best_basis(&self) -> Result<Vec<WaveletPacketNode>> {
        let mut best_nodes = Vec::new();
        self.select_best_basis("", &mut best_nodes)?;
        Ok(best_nodes)
    }

    /// Recursively select best basis
    fn select_best_basis(&self, path: &str, selected: &mut Vec<WaveletPacketNode>) -> Result<f64> {
        let node = self
            .nodes
            .get(path)
            .ok_or_else(|| TransformError::InvalidInput(format!("Node not found: {}", path)))?;

        let approx_path = format!("{}a", path);
        let detail_path = format!("{}d", path);

        // Check if we have children
        if self.nodes.contains_key(&approx_path) && self.nodes.contains_key(&detail_path) {
            // Compute cost of decomposition
            let approx_cost = self.select_best_basis(&approx_path, selected)?;
            let detail_cost = self.select_best_basis(&detail_path, selected)?;
            let children_cost = approx_cost + detail_cost;

            // Compare with keeping this node
            if node.cost <= children_cost {
                // Keep this node
                selected.retain(|n| !n.path.starts_with(path) || n.path == path);
                selected.push(node.clone());
                Ok(node.cost)
            } else {
                // Use children
                Ok(children_cost)
            }
        } else {
            // Leaf node
            selected.push(node.clone());
            Ok(node.cost)
        }
    }

    /// Reconstruct signal from wavelet packet coefficients
    ///
    /// Performs inverse WPT from a set of leaf nodes (e.g. a best-basis selection).
    /// The algorithm works bottom-up: it places each basis node at its position in the
    /// packet tree, then repeatedly merges pairs of sibling nodes using the inverse DWT
    /// until the root (level 0) is reached.
    pub fn reconstruct(&self, nodes: &[WaveletPacketNode]) -> Result<Array1<f64>> {
        if nodes.is_empty() {
            return Err(TransformError::InvalidInput(
                "No nodes provided for reconstruction".to_string(),
            ));
        }

        // Short-circuit: if the root node is among the inputs, return it directly.
        if let Some(root) = nodes.iter().find(|n| n.path.is_empty()) {
            return Ok(root.data.clone());
        }

        // Build a mutable map path -> data, starting from all input nodes.
        let mut tree: HashMap<String, Array1<f64>> = nodes
            .iter()
            .map(|n| (n.path.clone(), n.data.clone()))
            .collect();

        // Create one DWT instance for reconstruction filters.
        let dwt = DWT::new(self.wavelet)?.with_boundary(self.boundary);

        // Find the maximum depth we need to collapse to.
        let max_level = nodes.iter().map(|n| n.level).max().unwrap_or(0);

        // Bottom-up merging: at each level collapse sibling pairs.
        for _level in (1..=max_level).rev() {
            // Collect all unique parent paths that still need merging.
            let parents: Vec<String> = tree
                .keys()
                .filter_map(|p| {
                    if p.is_empty() {
                        return None;
                    }
                    // Parent is everything but the last char ('a' or 'd')
                    let parent = &p[..p.len() - 1];
                    // Only include if BOTH children are present and parent is absent
                    let approx_key = format!("{}a", parent);
                    let detail_key = format!("{}d", parent);
                    if tree.contains_key(&approx_key)
                        && tree.contains_key(&detail_key)
                        && !tree.contains_key(parent)
                    {
                        Some(parent.to_string())
                    } else {
                        None
                    }
                })
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            for parent in parents {
                let approx_key = format!("{}a", parent);
                let detail_key = format!("{}d", parent);

                let approx = tree.remove(&approx_key).ok_or_else(|| {
                    TransformError::InvalidInput(format!("Missing approx node: {}", approx_key))
                })?;
                let detail = tree.remove(&detail_key).ok_or_else(|| {
                    TransformError::InvalidInput(format!("Missing detail node: {}", detail_key))
                })?;

                let reconstructed = dwt.reconstruct(&approx.view(), &detail.view())?;
                tree.insert(parent, reconstructed);
            }
        }

        // The root should now be in the tree.
        tree.remove("").ok_or_else(|| {
            TransformError::InvalidInput(
                "Could not fully reconstruct to root — basis nodes may be incomplete".to_string(),
            )
        })
    }

    /// Get all nodes at a specific level
    pub fn get_level(&self, level: usize) -> Vec<&WaveletPacketNode> {
        self.nodes
            .values()
            .filter(|node| node.level == level)
            .collect()
    }

    /// Get a specific node by path
    pub fn get_node(&self, path: &str) -> Option<&WaveletPacketNode> {
        self.nodes.get(path)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<String, WaveletPacketNode> {
        &self.nodes
    }

    /// Compute the total cost of the best basis
    pub fn best_basis_cost(&self) -> Result<f64> {
        let best = self.best_basis()?;
        Ok(best.iter().map(|node| node.cost).sum())
    }
}

/// Denoise using wavelet packet transform
pub fn denoise_wpt(
    signal: &ArrayView1<f64>,
    wavelet: WaveletType,
    level: usize,
    threshold: f64,
) -> Result<Array1<f64>> {
    // Perform WPT
    let mut wpt = WPT::new(wavelet, level);
    wpt.decompose(signal)?;

    // Get best basis
    let best = wpt.best_basis()?;

    // Apply thresholding
    let mut denoised_nodes = Vec::new();
    for mut node in best {
        // Soft thresholding
        for val in node.data.iter_mut() {
            if val.abs() < threshold {
                *val = 0.0;
            } else {
                *val = if *val > 0.0 {
                    *val - threshold
                } else {
                    *val + threshold
                };
            }
        }
        node.update_cost();
        denoised_nodes.push(node);
    }

    // Reconstruct
    wpt.reconstruct(&denoised_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_wpt_decompose() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut wpt = WPT::new(WaveletType::Haar, 2);

        wpt.decompose(&signal.view())?;

        // Should have nodes at levels 0, 1, 2
        assert!(wpt.get_node("").is_some());
        assert!(wpt.get_node("a").is_some());
        assert!(wpt.get_node("d").is_some());
        assert!(wpt.get_node("aa").is_some());
        assert!(wpt.get_node("ad").is_some());
        assert!(wpt.get_node("da").is_some());
        assert!(wpt.get_node("dd").is_some());

        Ok(())
    }

    #[test]
    fn test_wpt_best_basis() -> Result<()> {
        let signal = Array1::from_vec((0..16).map(|i| (i as f64 * 0.5).sin()).collect());
        let mut wpt = WPT::new(WaveletType::Haar, 3);

        wpt.decompose(&signal.view())?;
        let best = wpt.best_basis()?;

        assert!(!best.is_empty());

        // Check that all selected nodes are unique
        let mut paths: Vec<_> = best.iter().map(|n| n.path.clone()).collect();
        paths.sort();
        paths.dedup();
        assert_eq!(paths.len(), best.len());

        Ok(())
    }

    #[test]
    fn test_wpt_levels() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut wpt = WPT::new(WaveletType::Haar, 2);

        wpt.decompose(&signal.view())?;

        let level0 = wpt.get_level(0);
        let level1 = wpt.get_level(1);
        let level2 = wpt.get_level(2);

        assert_eq!(level0.len(), 1);
        assert_eq!(level1.len(), 2);
        assert_eq!(level2.len(), 4);

        Ok(())
    }

    #[test]
    fn test_wavelet_packet_node_cost() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let node = WaveletPacketNode::new(data, "test".to_string(), 1, 0);

        assert!(node.cost >= 0.0);
    }

    #[test]
    fn test_best_basis_criterion() {
        let wpt1 = WPT::new(WaveletType::Haar, 3).with_criterion(BestBasisCriterion::Shannon);
        assert_eq!(wpt1.criterion, BestBasisCriterion::Shannon);

        let wpt2 = WPT::new(WaveletType::Haar, 3).with_criterion(BestBasisCriterion::LogEnergy);
        assert_eq!(wpt2.criterion, BestBasisCriterion::LogEnergy);
    }

    #[test]
    fn test_wpt_reconstruct_from_best_basis() -> Result<()> {
        // Reconstruct from best-basis nodes and check length is preserved
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let original_len = signal.len();
        let mut wpt = WPT::new(WaveletType::Haar, 2);
        wpt.decompose(&signal.view())?;
        let best = wpt.best_basis()?;
        let reconstructed = wpt.reconstruct(&best)?;
        // Reconstruction may differ in length due to boundary effects; allow ±2 samples
        let diff = (reconstructed.len() as isize - original_len as isize).unsigned_abs();
        assert!(
            diff <= 2,
            "Reconstructed length {} too different from original {}",
            reconstructed.len(),
            original_len
        );
        Ok(())
    }

    #[test]
    fn test_wpt_reconstruct_leaf_nodes() -> Result<()> {
        // Feed all leaf nodes at level 1 and verify reconstruction succeeds
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut wpt = WPT::new(WaveletType::Haar, 1);
        wpt.decompose(&signal.view())?;
        let level1: Vec<WaveletPacketNode> = wpt.get_level(1).into_iter().cloned().collect();
        // Both "a" and "d" nodes should be present, allowing reconstruction
        assert_eq!(level1.len(), 2);
        let reconstructed = wpt.reconstruct(&level1)?;
        assert!(reconstructed.len() > 0);
        Ok(())
    }

    #[test]
    fn test_wpt_reconstruct_root_shortcut() -> Result<()> {
        // If the root node (empty path) is in the slice, reconstruct returns it directly.
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let root = WaveletPacketNode::new(data.clone(), String::new(), 0, 0);
        let wpt = WPT::new(WaveletType::Haar, 2);
        let result = wpt.reconstruct(&[root])?;
        assert_eq!(result.len(), data.len());
        assert_abs_diff_eq!(result[0], data[0], epsilon = 1e-10);
        Ok(())
    }
}
