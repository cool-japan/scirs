//! Graph cuts segmentation algorithm
//!
//! This module implements the graph cuts segmentation algorithm, which formulates
//! image segmentation as a min-cut/max-flow problem on a graph.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;

use crate::error::{NdimageError, NdimageResult};

/// Graph structure for max-flow/min-cut algorithm
struct Graph {
    nodes: Vec<Node>,
    edges: HashMap<(usize, usize), f64>,
    source: usize,
    sink: usize,
}

/// Node in the graph
#[derive(Clone, Debug)]
struct Node {
    id: usize,
    neighbors: Vec<usize>,
}

impl Graph {
    /// Create a new graph with specified number of nodes
    fn new(_numnodes: usize) -> Self {
        let mut _nodes = Vec::with_capacity(_numnodes + 2);
        for i in 0.._numnodes + 2 {
            _nodes.push(Node {
                id: i,
                neighbors: Vec::new(),
            });
        }

        Self {
            nodes: _nodes,
            edges: HashMap::new(),
            source: _numnodes,
            sink: _numnodes + 1,
        }
    }

    /// Add an edge between two nodes
    fn add_edge(&mut self, from: usize, to: usize, capacity: f64) {
        if from != to && capacity > 0.0 {
            self.nodes[from].neighbors.push(to);
            self.nodes[to].neighbors.push(from);
            self.edges.insert((from, to), capacity);
            self.edges.insert((to, from), 0.0); // Reverse edge with 0 capacity
        }
    }

    /// Find augmenting path using BFS
    fn bfs(
        &self,
        parent: &mut Vec<Option<usize>>,
        residual: &HashMap<(usize, usize), f64>,
    ) -> bool {
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();

        queue.push_back(self.source);
        visited[self.source] = true;
        parent[self.source] = None;

        while let Some(u) = queue.pop_front() {
            for &v in &self.nodes[u].neighbors {
                let capacity = residual.get(&(u, v)).unwrap_or(&0.0);
                if !visited[v] && *capacity > 0.0 {
                    visited[v] = true;
                    parent[v] = Some(u);

                    if v == self.sink {
                        return true;
                    }

                    queue.push_back(v);
                }
            }
        }

        false
    }

    /// Compute maximum flow using Ford-Fulkerson algorithm
    fn max_flow(&mut self) -> (f64, Vec<bool>) {
        let mut residual = self.edges.clone();
        let mut parent = vec![None; self.nodes.len()];
        let mut max_flow = 0.0;

        // Find augmenting paths
        while self.bfs(&mut parent, &residual) {
            // Find minimum capacity along the path
            let mut path_flow = f64::INFINITY;
            let mut v = self.sink;

            while v != self.source {
                let u = parent[v].unwrap();
                let capacity = residual.get(&(u, v)).unwrap_or(&0.0);
                path_flow = path_flow.min(*capacity);
                v = u;
            }

            // Update residual capacities
            v = self.sink;
            while v != self.source {
                let u = parent[v].unwrap();
                *residual.get_mut(&(u, v)).unwrap() -= path_flow;
                *residual.get_mut(&(v, u)).unwrap() += path_flow;
                v = u;
            }

            max_flow += path_flow;
        }

        // Find minimum cut
        let mut cut = vec![false; self.nodes.len()];
        let mut visited = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();

        queue.push_back(self.source);
        visited[self.source] = true;
        cut[self.source] = true;

        while let Some(u) = queue.pop_front() {
            for &v in &self.nodes[u].neighbors {
                let capacity = residual.get(&(u, v)).unwrap_or(&0.0);
                if !visited[v] && *capacity > 0.0 {
                    visited[v] = true;
                    cut[v] = true;
                    queue.push_back(v);
                }
            }
        }

        (max_flow, cut)
    }
}

/// Parameters for graph cuts segmentation
#[derive(Clone)]
pub struct GraphCutsParams {
    /// Weight for smoothness term (pairwise potentials)
    pub lambda: f64,
    /// Sigma for Gaussian similarity in smoothness term
    pub sigma: f64,
    /// Neighborhood system: 4 or 8 connectivity
    pub connectivity: u8,
}

impl Default for GraphCutsParams {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            sigma: 50.0,
            connectivity: 8,
        }
    }
}

/// Perform graph cuts segmentation on an image
///
/// # Arguments
/// * `image` - Input image
/// * `foreground_seeds` - Mask indicating definite foreground pixels
/// * `background_seeds` - Mask indicating definite background pixels
/// * `params` - Segmentation parameters
///
/// # Returns
/// Binary segmentation mask where true indicates foreground
#[allow(dead_code)]
pub fn graph_cuts<T>(
    image: &ArrayView2<T>,
    foreground_seeds: &ArrayView2<bool>,
    background_seeds: &ArrayView2<bool>,
    params: Option<GraphCutsParams>,
) -> NdimageResult<Array2<bool>>
where
    T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static,
{
    let params = params.unwrap_or_default();
    let (height, width) = image.dim();
    let num_pixels = height * width;

    // Validate inputs
    if foreground_seeds.dim() != image.dim() || background_seeds.dim() != image.dim() {
        return Err(NdimageError::DimensionError(
            "Seed masks must have same dimensions as image".into(),
        ));
    }

    // Check for overlapping _seeds
    for i in 0..height {
        for j in 0..width {
            if foreground_seeds[[i, j]] && background_seeds[[i, j]] {
                return Err(NdimageError::InvalidInput(
                    "Foreground and background _seeds cannot overlap".into(),
                ));
            }
        }
    }

    // Create graph
    let mut graph = Graph::new(num_pixels);

    // Helper function to convert 2D coordinates to node index
    let coord_to_idx = |y: usize, x: usize| -> usize { y * width + x };

    // Add terminal edges (data term)
    let k = compute_k_constant(image);

    for i in 0..height {
        for j in 0..width {
            let idx = coord_to_idx(i, j);

            if foreground_seeds[[i, j]] {
                // Definite foreground
                graph.add_edge(graph.source, idx, k);
                graph.add_edge(idx, graph.sink, 0.0);
            } else if background_seeds[[i, j]] {
                // Definite background
                graph.add_edge(graph.source, idx, 0.0);
                graph.add_edge(idx, graph.sink, k);
            } else {
                // Unknown - use data-driven weights
                let (fg_weight, bg_weight) =
                    compute_data_weights(image, i, j, foreground_seeds, background_seeds);
                graph.add_edge(graph.source, idx, fg_weight);
                graph.add_edge(idx, graph.sink, bg_weight);
            }
        }
    }

    // Add neighbor edges (smoothness term)
    let neighbors = get_neighbors(params.connectivity);

    for i in 0..height {
        for j in 0..width {
            let idx1 = coord_to_idx(i, j);
            let val1 = image[[i, j]];

            for (di, dj) in &neighbors {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;

                if ni >= 0 && ni < height as i32 && nj >= 0 && nj < width as i32 {
                    let ni = ni as usize;
                    let nj = nj as usize;
                    let idx2 = coord_to_idx(ni, nj);

                    if idx1 < idx2 {
                        // Avoid duplicate edges
                        let val2 = image[[ni, nj]];
                        let weight =
                            compute_smoothness_weight(val1, val2, params.lambda, params.sigma);
                        graph.add_edge(idx1, idx2, weight);
                    }
                }
            }
        }
    }

    // Solve max-flow/min-cut
    let (_, cut) = graph.max_flow();

    // Convert cut to segmentation mask
    let mut result = Array2::default((height, width));
    for i in 0..height {
        for j in 0..width {
            let idx = coord_to_idx(i, j);
            result[[i, j]] = cut[idx];
        }
    }

    Ok(result)
}

/// Compute K constant for terminal edges
#[allow(dead_code)]
fn compute_k_constant<T: Float>(image: &ArrayView2<T>) -> f64 {
    // K should be larger than any possible sum of edge weights
    let max_val = image
        .iter()
        .map(|&v| v.to_f64().unwrap_or(0.0))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);

    1.0 + max_val * 8.0 // Conservative estimate
}

/// Compute data weights for a pixel
#[allow(dead_code)]
fn compute_data_weights<T: Float>(
    image: &ArrayView2<T>,
    y: usize,
    x: usize,
    foreground_seeds: &ArrayView2<bool>,
    background_seeds: &ArrayView2<bool>,
) -> (f64, f64) {
    let pixel_val = image[[y, x]].to_f64().unwrap_or(0.0);
    let (height, width) = image.dim();

    // Compute mean intensity of seed regions
    let mut fg_sum = 0.0;
    let mut fg_count = 0;
    let mut bg_sum = 0.0;
    let mut bg_count = 0;

    for i in 0..height {
        for j in 0..width {
            if foreground_seeds[[i, j]] {
                fg_sum += image[[i, j]].to_f64().unwrap_or(0.0);
                fg_count += 1;
            } else if background_seeds[[i, j]] {
                bg_sum += image[[i, j]].to_f64().unwrap_or(0.0);
                bg_count += 1;
            }
        }
    }

    let fg_mean = if fg_count > 0 {
        fg_sum / fg_count as f64
    } else {
        0.0
    };
    let bg_mean = if bg_count > 0 {
        bg_sum / bg_count as f64
    } else {
        255.0
    };

    // Simple Gaussian model
    let fg_diff = pixel_val - fg_mean;
    let bg_diff = pixel_val - bg_mean;

    let fg_prob = (-fg_diff * fg_diff / 100.0).exp();
    let bg_prob = (-bg_diff * bg_diff / 100.0).exp();

    let epsilon = 1e-10;
    let fg_weight = -((bg_prob + epsilon).ln());
    let bg_weight = -((fg_prob + epsilon).ln());

    (fg_weight.max(0.0), bg_weight.max(0.0))
}

/// Compute smoothness weight between neighboring pixels
#[allow(dead_code)]
fn compute_smoothness_weight<T: Float>(val1: T, val2: T, lambda: f64, sigma: f64) -> f64 {
    let diff = (val1 - val2).to_f64().unwrap_or(0.0);
    let weight = lambda * (-diff * diff / (2.0 * sigma * sigma)).exp();
    weight
}

/// Get neighbor offsets based on connectivity
#[allow(dead_code)]
fn get_neighbors(connectivity: u8) -> Vec<(i32, i32)> {
    match connectivity {
        4 => vec![(0, 1), (1, 0), (0, -1), (-1, 0)],
        8 => vec![
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ],
        _ => vec![(0, 1), (1, 0), (0, -1), (-1, 0)], // Default to 4-_connectivity
    }
}

/// Interactive graph cuts segmentation with iterative refinement
pub struct InteractiveGraphCuts<T> {
    image: Array2<T>,
    foreground_seeds: Array2<bool>,
    background_seeds: Array2<bool>,
    current_segmentation: Option<Array2<bool>>,
    params: GraphCutsParams,
}

impl<T: Float + FromPrimitive + Debug + std::ops::AddAssign + std::ops::DivAssign + 'static>
    InteractiveGraphCuts<T>
{
    /// Create new interactive segmentation session
    pub fn new(image: Array2<T>, params: Option<GraphCutsParams>) -> Self {
        let shape = image.dim();
        Self {
            image,
            foreground_seeds: Array2::default(shape),
            background_seeds: Array2::default(shape),
            current_segmentation: None,
            params: params.unwrap_or_default(),
        }
    }

    /// Add foreground seeds
    pub fn add_foreground_seeds(&mut self, seeds: &[(usize, usize)]) {
        for &(y, x) in seeds {
            if y < self.foreground_seeds.dim().0 && x < self.foreground_seeds.dim().1 {
                self.foreground_seeds[[y, x]] = true;
                self.background_seeds[[y, x]] = false; // Ensure no overlap
            }
        }
    }

    /// Add background seeds
    pub fn add_background_seeds(&mut self, seeds: &[(usize, usize)]) {
        for &(y, x) in seeds {
            if y < self.background_seeds.dim().0 && x < self.background_seeds.dim().1 {
                self.background_seeds[[y, x]] = true;
                self.foreground_seeds[[y, x]] = false; // Ensure no overlap
            }
        }
    }

    /// Clear all seeds
    pub fn clear_seeds(&mut self) {
        self.foreground_seeds.fill(false);
        self.background_seeds.fill(false);
    }

    /// Run segmentation with current seeds
    pub fn segment(&mut self) -> NdimageResult<&Array2<bool>> {
        let result = graph_cuts(
            &self.image.view(),
            &self.foreground_seeds.view(),
            &self.background_seeds.view(),
            Some(self.params.clone()),
        )?;

        self.current_segmentation = Some(result);
        Ok(self.current_segmentation.as_ref().unwrap())
    }

    /// Get current segmentation result
    pub fn get_segmentation(&self) -> Option<&Array2<bool>> {
        self.current_segmentation.as_ref()
    }
}

impl GraphCutsParams {
    /// Create parameters optimized for grayscale images
    pub fn for_grayscale() -> Self {
        Self {
            lambda: 10.0,
            sigma: 30.0,
            connectivity: 8,
        }
    }

    /// Create parameters optimized for color images
    pub fn for_color() -> Self {
        Self {
            lambda: 5.0,
            sigma: 50.0,
            connectivity: 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_graph_cuts_simple() {
        // Create simple test image
        let image = arr2(&[
            [0.0, 0.0, 100.0, 100.0],
            [0.0, 0.0, 100.0, 100.0],
            [0.0, 0.0, 100.0, 100.0],
            [0.0, 0.0, 100.0, 100.0],
        ]);

        // Create seed masks
        let mut fg_seeds = Array2::default((4, 4));
        let mut bg_seeds = Array2::default((4, 4));

        // Mark some foreground seeds (right side)
        fg_seeds[[1, 2]] = true;
        fg_seeds[[2, 3]] = true;

        // Mark some background seeds (left side)
        bg_seeds[[1, 0]] = true;
        bg_seeds[[2, 1]] = true;

        // Run segmentation
        let result = graph_cuts(&image.view(), &fg_seeds.view(), &bg_seeds.view(), None).unwrap();

        // Check that right side is segmented as foreground
        assert!(result[[0, 2]] || result[[0, 3]]);
        assert!(result[[1, 2]] || result[[1, 3]]);

        // Check that left side is segmented as background
        assert!(!result[[0, 0]] && !result[[0, 1]]);
        assert!(!result[[1, 0]] && !result[[1, 1]]);
    }

    #[test]
    fn test_interactive_graph_cuts() {
        let image = arr2(&[
            [10.0, 20.0, 80.0, 90.0],
            [15.0, 25.0, 85.0, 95.0],
            [12.0, 22.0, 82.0, 92.0],
            [18.0, 28.0, 88.0, 98.0],
        ]);

        let mut interactive = InteractiveGraphCuts::new(image, None);

        // Add seeds
        interactive.add_foreground_seeds(&[(0, 3), (1, 2)]);
        interactive.add_background_seeds(&[(0, 0), (1, 1)]);

        // Segment
        let result = interactive.segment().unwrap();
        assert_eq!(result.dim(), (4, 4));

        // Add more seeds and re-segment
        interactive.add_foreground_seeds(&[(2, 3)]);
        let result2 = interactive.segment().unwrap();
        assert_eq!(result2.dim(), (4, 4));
    }
}
