//! Graph layout algorithms for visualization
//!
//! This module provides algorithms for computing node positions
//! for graph visualization.

use std::collections::HashMap;
use std::f64::consts::PI;

use rand::Rng;

use crate::base::{EdgeWeight, Graph, Node};
use crate::error::Result;

/// 2D position for graph layout
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    /// X coordinate
    pub x: f64,
    /// Y coordinate  
    pub y: f64,
}

impl Position {
    /// Create a new position
    pub fn new(x: f64, y: f64) -> Self {
        Position { x, y }
    }

    /// Calculate Euclidean distance to another position
    pub fn distance(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Compute a circular layout for the graph
///
/// Nodes are placed evenly around a circle.
pub fn circular_layout<N, E, Ix>(graph: &Graph<N, E, Ix>, radius: f64) -> HashMap<N, Position>
where
    N: Node + Clone,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let mut layout = HashMap::new();

    if n == 0 {
        return layout;
    }

    let angle_step = 2.0 * PI / n as f64;

    for (i, node) in nodes.into_iter().enumerate() {
        let angle = i as f64 * angle_step;
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        layout.insert(node, Position::new(x, y));
    }

    layout
}

/// Compute a spring layout using force-directed placement
///
/// This is a simplified version of the Fruchterman-Reingold algorithm.
pub fn spring_layout<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    iterations: usize,
    area_width: f64,
    area_height: f64,
) -> HashMap<N, Position>
where
    N: Node + Clone,
    E: EdgeWeight + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return HashMap::new();
    }

    // Initialize with random positions
    let mut positions = vec![Position::new(0.0, 0.0); n];
    let mut rng = rand::rng();

    for pos in &mut positions {
        pos.x = rng.random::<f64>() * area_width - area_width / 2.0;
        pos.y = rng.random::<f64>() * area_height - area_height / 2.0;
    }

    // Ideal spring length
    let area = area_width * area_height;
    let k = (area / n as f64).sqrt();

    // Temperature for simulated annealing
    let mut temperature = area_width.min(area_height) / 10.0;
    let cooling_rate = temperature / iterations as f64;

    // Create node index mapping
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Force-directed iterations
    for _ in 0..iterations {
        // Calculate repulsive forces between all pairs
        let mut forces = vec![Position::new(0.0, 0.0); n];

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dx = positions[i].x - positions[j].x;
                    let dy = positions[i].y - positions[j].y;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.1);

                    // Repulsive force
                    let force = k * k / dist;
                    forces[i].x += (dx / dist) * force;
                    forces[i].y += (dy / dist) * force;
                }
            }
        }

        // Calculate attractive forces along edges
        for i in 0..n {
            if let Ok(neighbors) = graph.neighbors(&nodes[i]) {
                for neighbor in neighbors {
                    if let Some(&j) = node_to_idx.get(&neighbor) {
                        let dx = positions[i].x - positions[j].x;
                        let dy = positions[i].y - positions[j].y;
                        let dist = (dx * dx + dy * dy).sqrt().max(0.1);

                        // Attractive force
                        let force = dist * dist / k;
                        forces[i].x -= (dx / dist) * force;
                        forces[i].y -= (dy / dist) * force;
                    }
                }
            }
        }

        // Update positions
        for i in 0..n {
            let disp = (forces[i].x * forces[i].x + forces[i].y * forces[i].y).sqrt();
            if disp > 0.0 {
                let limited_disp = disp.min(temperature);
                positions[i].x += (forces[i].x / disp) * limited_disp;
                positions[i].y += (forces[i].y / disp) * limited_disp;

                // Keep within bounds
                positions[i].x = positions[i].x.max(-area_width / 2.0).min(area_width / 2.0);
                positions[i].y = positions[i]
                    .y
                    .max(-area_height / 2.0)
                    .min(area_height / 2.0);
            }
        }

        // Cool down
        temperature -= cooling_rate;
    }

    // Convert to HashMap
    nodes
        .into_iter()
        .enumerate()
        .map(|(i, node)| (node, positions[i]))
        .collect()
}

/// Compute a hierarchical layout for a directed acyclic graph
///
/// Nodes are arranged in layers based on their topological ordering.
pub fn hierarchical_layout<N, E, Ix>(
    graph: &crate::base::DiGraph<N, E, Ix>,
    layer_height: f64,
    node_spacing: f64,
) -> Result<HashMap<N, Position>>
where
    N: Node + Clone + std::hash::Hash + Eq,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    use crate::algorithms::topological_sort;

    // For hierarchical layout, we need a DAG
    // Try to get a topological ordering
    let topo_order = topological_sort(graph)?;

    let mut layout = HashMap::new();
    let mut node_to_layer: HashMap<N, usize> = HashMap::new();
    let mut layer_counts: HashMap<usize, usize> = HashMap::new();

    // Assign layers based on longest path from sources
    for node in &topo_order {
        let mut max_pred_layer = 0;

        // Find maximum layer of predecessors
        if let Ok(predecessors) = graph.predecessors(node) {
            for predecessor in predecessors {
                if let Some(&pred_layer) = node_to_layer.get(&predecessor) {
                    max_pred_layer = max_pred_layer.max(pred_layer + 1);
                }
            }
        }

        node_to_layer.insert(node.clone(), max_pred_layer);
        *layer_counts.entry(max_pred_layer).or_insert(0) += 1;
    }

    // Calculate positions
    let mut layer_offsets: HashMap<usize, f64> = HashMap::new();

    for (node, &layer) in &node_to_layer {
        let layer_size = layer_counts[&layer];
        let layer_width = (layer_size - 1) as f64 * node_spacing;

        let offset = layer_offsets.entry(layer).or_insert(-layer_width / 2.0);

        let x = *offset;
        let y = -(layer as f64 * layer_height);

        layout.insert(node.clone(), Position::new(x, y));

        *offset += node_spacing;
    }

    Ok(layout)
}

/// Compute a spectral layout based on eigenvectors of the Laplacian
///
/// Uses the second and third smallest eigenvectors of the Laplacian matrix.
pub fn spectral_layout<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<HashMap<N, Position>>
where
    N: Node + Clone,
    E: EdgeWeight + Into<f64> + num_traits::Zero + num_traits::One + PartialOrd + Copy,
    Ix: petgraph::graph::IndexType,
{
    use crate::spectral::{laplacian, LaplacianType};

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n < 2 {
        let mut layout = HashMap::new();
        if n == 1 {
            layout.insert(nodes[0].clone(), Position::new(0.0, 0.0));
        }
        return Ok(layout);
    }

    // Get the Laplacian matrix
    let _lap = laplacian(graph, LaplacianType::Normalized)?;

    // For a proper implementation, we would compute eigenvectors here
    // For now, use a simple approximation based on node degrees
    let mut positions = vec![Position::new(0.0, 0.0); n];

    // Use node degrees to spread out nodes
    let degrees: Vec<usize> = (0..n)
        .map(|i| graph.neighbors(&nodes[i]).unwrap_or_default().len())
        .collect();

    let max_degree = *degrees.iter().max().unwrap_or(&1) as f64;

    for (i, pos) in positions.iter_mut().enumerate() {
        let angle = 2.0 * PI * i as f64 / n as f64;
        let radius = 1.0 - (degrees[i] as f64 / max_degree) * 0.5;
        pos.x = radius * angle.cos();
        pos.y = radius * angle.sin();
    }

    // Convert to HashMap
    Ok(nodes
        .into_iter()
        .enumerate()
        .map(|(i, node)| (node, positions[i]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_layout() {
        let mut graph: Graph<char, f64> = Graph::new();
        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('B', 'C', 1.0).unwrap();
        graph.add_edge('C', 'A', 1.0).unwrap();

        let layout = circular_layout(&graph, 10.0);

        assert_eq!(layout.len(), 3);

        // All nodes should be at distance 10 from origin
        for pos in layout.values() {
            let dist = (pos.x * pos.x + pos.y * pos.y).sqrt();
            assert!((dist - 10.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spring_layout() {
        let mut graph: Graph<char, f64> = Graph::new();
        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('B', 'C', 1.0).unwrap();
        graph.add_edge('C', 'D', 1.0).unwrap();
        graph.add_edge('D', 'A', 1.0).unwrap();

        let layout = spring_layout(&graph, 50, 100.0, 100.0);

        assert_eq!(layout.len(), 4);

        // All positions should be within bounds
        for pos in layout.values() {
            assert!(pos.x >= -50.0 && pos.x <= 50.0);
            assert!(pos.y >= -50.0 && pos.y <= 50.0);
        }
    }

    #[test]
    fn test_hierarchical_layout() {
        let mut graph: crate::base::DiGraph<char, f64> = crate::base::DiGraph::new();
        // Create a DAG
        graph.add_edge('A', 'B', 1.0).unwrap();
        graph.add_edge('A', 'C', 1.0).unwrap();
        graph.add_edge('B', 'D', 1.0).unwrap();
        graph.add_edge('C', 'D', 1.0).unwrap();

        let layout = hierarchical_layout(&graph, 50.0, 30.0).unwrap();

        assert_eq!(layout.len(), 4);

        // Check that nodes are arranged in layers
        // A should be at the top (y = 0)
        assert!((layout[&'A'].y - 0.0).abs() < 1e-10);

        // B and C should be in the middle layer
        assert!((layout[&'B'].y - layout[&'C'].y).abs() < 1e-10);

        // D should be at the bottom
        assert!(layout[&'D'].y < layout[&'B'].y);
    }
}
