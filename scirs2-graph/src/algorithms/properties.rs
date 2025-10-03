//! Graph property algorithms
//!
//! This module contains algorithms for computing various properties of graphs.

use crate::algorithms::shortest_path::dijkstra_path;
use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::hash::Hash;

/// Computes the diameter of a graph
///
/// The diameter is the maximum shortest path distance between any two nodes in the graph.
/// Returns None if the graph is disconnected.
#[allow(dead_code)]
pub fn diameter<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Option<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight
        + Into<f64>
        + Clone
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::cmp::PartialOrd
        + std::fmt::Debug
        + Copy
        + Default,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return None;
    }

    let mut max_distance = 0.0;

    // Compute all-pairs shortest paths
    for i in 0..n {
        for j in i + 1..n {
            match dijkstra_path(graph, &nodes[i], &nodes[j]) {
                Ok(Some(path)) => {
                    let distance: f64 = path.total_weight.into();
                    if distance > max_distance {
                        max_distance = distance;
                    }
                }
                Ok(None) => return None, // No path exists
                Err(_) => return None,   // Graph is disconnected
            }
        }
    }

    Some(max_distance)
}

/// Computes the radius of a graph
///
/// The radius is the minimum eccentricity over all nodes, where eccentricity of a node
/// is the maximum distance from that node to any other node.
/// Returns None if the graph is disconnected.
#[allow(dead_code)]
pub fn radius<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Option<f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight
        + Into<f64>
        + Clone
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::cmp::PartialOrd
        + std::fmt::Debug
        + Copy
        + Default,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return None;
    }

    let mut min_eccentricity = f64::INFINITY;

    // Compute eccentricity for each node
    for i in 0..n {
        let mut max_distance_from_i = 0.0;

        for j in 0..n {
            if i != j {
                match dijkstra_path(graph, &nodes[i], &nodes[j]) {
                    Ok(Some(path)) => {
                        let distance: f64 = path.total_weight.into();
                        if distance > max_distance_from_i {
                            max_distance_from_i = distance;
                        }
                    }
                    Ok(None) => return None, // No path exists
                    Err(_) => return None,   // Graph is disconnected
                }
            }
        }

        if max_distance_from_i < min_eccentricity {
            min_eccentricity = max_distance_from_i;
        }
    }

    Some(min_eccentricity)
}

/// Find the center nodes of a graph
///
/// Center nodes are those with minimum eccentricity (equal to the radius).
/// Returns empty vector if the graph is disconnected.
#[allow(dead_code)]
pub fn center_nodes<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight
        + Into<f64>
        + Clone
        + scirs2_core::numeric::Zero
        + scirs2_core::numeric::One
        + std::cmp::PartialOrd
        + std::fmt::Debug
        + Copy
        + Default,
    Ix: IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return vec![];
    }

    let mut eccentricities = vec![0.0; n];
    let mut min_eccentricity = f64::INFINITY;

    // Compute eccentricity for each node
    for i in 0..n {
        let mut max_distance_from_i = 0.0;

        for j in 0..n {
            if i != j {
                match dijkstra_path(graph, &nodes[i], &nodes[j]) {
                    Ok(Some(path)) => {
                        let distance: f64 = path.total_weight.into();
                        if distance > max_distance_from_i {
                            max_distance_from_i = distance;
                        }
                    }
                    Ok(None) => return vec![], // No path exists
                    Err(_) => return vec![],   // Graph is disconnected
                }
            }
        }

        eccentricities[i] = max_distance_from_i;

        if max_distance_from_i < min_eccentricity {
            min_eccentricity = max_distance_from_i;
        }
    }

    // Collect all nodes with minimum eccentricity
    nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| (eccentricities[*i] - min_eccentricity).abs() < f64::EPSILON)
        .map(|(_, node)| node)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_diameter_radius_center() -> GraphResult<()> {
        // Create a path graph: 0 - 1 - 2 - 3 - 4
        let mut path = create_graph::<i32, f64>();
        path.add_edge(0, 1, 1.0)?;
        path.add_edge(1, 2, 1.0)?;
        path.add_edge(2, 3, 1.0)?;
        path.add_edge(3, 4, 1.0)?;

        // Diameter should be 4 (from 0 to 4)
        assert_eq!(diameter(&path), Some(4.0));

        // Radius should be 2 (from center node 2)
        assert_eq!(radius(&path), Some(2.0));

        // Center should be node 2
        let centers = center_nodes(&path);
        assert_eq!(centers.len(), 1);
        assert_eq!(centers[0], 2);

        Ok(())
    }

    #[test]
    fn test_stargraph() -> GraphResult<()> {
        // Create a star graph with center 0
        let mut star = create_graph::<i32, f64>();
        star.add_edge(0, 1, 1.0)?;
        star.add_edge(0, 2, 1.0)?;
        star.add_edge(0, 3, 1.0)?;
        star.add_edge(0, 4, 1.0)?;

        // Diameter should be 2 (from any leaf to another)
        assert_eq!(diameter(&star), Some(2.0));

        // Radius should be 1 (from center)
        assert_eq!(radius(&star), Some(1.0));

        // Center should be node 0
        let centers = center_nodes(&star);
        assert_eq!(centers.len(), 1);
        assert_eq!(centers[0], 0);

        Ok(())
    }

    #[test]
    fn test_disconnectedgraph() -> GraphResult<()> {
        // Create a disconnected graph
        let mut graph = create_graph::<i32, f64>();
        graph.add_edge(0, 1, 1.0)?;
        graph.add_edge(2, 3, 1.0)?;

        // All functions should return None/empty for disconnected graphs
        assert_eq!(diameter(&graph), None);
        assert_eq!(radius(&graph), None);
        assert!(center_nodes(&graph).is_empty());

        Ok(())
    }
}
