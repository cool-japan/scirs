//! Graph motif finding algorithms
//!
//! This module contains algorithms for finding small recurring subgraph patterns (motifs).

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Finds all occurrences of a specific motif pattern in a graph
///
/// A motif is a small recurring subgraph pattern. This function finds all
/// instances of common motifs like triangles, squares, or stars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotifType {
    /// Triangle (3-cycle)
    Triangle,
    /// Square (4-cycle)
    Square,
    /// Star with 3 leaves
    Star3,
    /// Clique of size 4
    Clique4,
    /// Path of length 3 (4 nodes)
    Path3,
    /// Bi-fan motif (2 nodes connected to 2 other nodes)
    BiFan,
    /// Feed-forward loop
    FeedForwardLoop,
    /// Bi-directional motif
    BiDirectional,
}

/// Find all occurrences of a specified motif in the graph
#[allow(dead_code)]
pub fn find_motifs<N, E, Ix>(graph: &Graph<N, E, Ix>, motiftype: MotifType) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    match motiftype {
        MotifType::Triangle => find_triangles(graph),
        MotifType::Square => find_squares(graph),
        MotifType::Star3 => find_star3s(graph),
        MotifType::Clique4 => find_clique4s(graph),
        MotifType::Path3 => find_path3s(graph),
        MotifType::BiFan => find_bi_fans(graph),
        MotifType::FeedForwardLoop => find_feed_forward_loops(graph),
        MotifType::BiDirectional => find_bidirectional_motifs(graph),
    }
}

#[allow(dead_code)]
fn find_triangles<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let triangles = Mutex::new(Vec::new());

    // Parallel triangle finding using edge-based approach for better performance
    nodes.par_iter().enumerate().for_each(|(_i, node_i)| {
        if let Ok(neighbors_i) = graph.neighbors(node_i) {
            let neighbors_i: Vec<_> = neighbors_i;

            for (j, node_j) in neighbors_i.iter().enumerate() {
                for node_k in neighbors_i.iter().skip(j + 1) {
                    if graph.has_edge(node_j, node_k) {
                        let mut triangles_guard = triangles.lock().unwrap();
                        let mut triangle = vec![node_i.clone(), node_j.clone(), node_k.clone()];
                        triangle.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                        // Avoid duplicates
                        if !triangles_guard.iter().any(|t| t == &triangle) {
                            triangles_guard.push(triangle);
                        }
                    }
                }
            }
        }
    });

    triangles.into_inner().unwrap()
}

#[allow(dead_code)]
fn find_squares<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let mut squares = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // For each quadruplet of nodes, check if they form a square
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if !graph.has_edge(&nodes[i], &nodes[j]) {
                continue;
            }
            for k in j + 1..nodes.len() {
                if !graph.has_edge(&nodes[j], &nodes[k]) {
                    continue;
                }
                for l in k + 1..nodes.len() {
                    if graph.has_edge(&nodes[k], &nodes[l])
                        && graph.has_edge(&nodes[l], &nodes[i])
                        && !graph.has_edge(&nodes[i], &nodes[k])
                        && !graph.has_edge(&nodes[j], &nodes[l])
                    {
                        squares.push(vec![
                            nodes[i].clone(),
                            nodes[j].clone(),
                            nodes[k].clone(),
                            nodes[l].clone(),
                        ]);
                    }
                }
            }
        }
    }

    squares
}

#[allow(dead_code)]
fn find_star3s<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let mut stars = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // For each node as center, find if it has exactly 3 neighbors that aren't connected
    for center in &nodes {
        if let Ok(neighbors) = graph.neighbors(center) {
            let neighbor_list: Vec<N> = neighbors;

            if neighbor_list.len() >= 3 {
                // Check all combinations of 3 neighbors
                for i in 0..neighbor_list.len() {
                    for j in i + 1..neighbor_list.len() {
                        for k in j + 1..neighbor_list.len() {
                            // Check that the neighbors aren't connected to each other
                            if !graph.has_edge(&neighbor_list[i], &neighbor_list[j])
                                && !graph.has_edge(&neighbor_list[j], &neighbor_list[k])
                                && !graph.has_edge(&neighbor_list[i], &neighbor_list[k])
                            {
                                stars.push(vec![
                                    center.clone(),
                                    neighbor_list[i].clone(),
                                    neighbor_list[j].clone(),
                                    neighbor_list[k].clone(),
                                ]);
                            }
                        }
                    }
                }
            }
        }
    }

    stars
}

#[allow(dead_code)]
fn find_clique4s<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let mut cliques = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // For each quadruplet of nodes, check if they form a complete graph
    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if !graph.has_edge(&nodes[i], &nodes[j]) {
                continue;
            }
            for k in j + 1..nodes.len() {
                if !graph.has_edge(&nodes[i], &nodes[k]) || !graph.has_edge(&nodes[j], &nodes[k]) {
                    continue;
                }
                for l in k + 1..nodes.len() {
                    if graph.has_edge(&nodes[i], &nodes[l])
                        && graph.has_edge(&nodes[j], &nodes[l])
                        && graph.has_edge(&nodes[k], &nodes[l])
                    {
                        cliques.push(vec![
                            nodes[i].clone(),
                            nodes[j].clone(),
                            nodes[k].clone(),
                            nodes[l].clone(),
                        ]);
                    }
                }
            }
        }
    }

    cliques
}

/// Find all path motifs of length 3 (4 nodes in a line)
#[allow(dead_code)]
fn find_path3s<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let paths = Mutex::new(Vec::new());

    nodes.par_iter().for_each(|start_node| {
        if let Ok(neighbors1) = graph.neighbors(start_node) {
            for middle1 in neighbors1 {
                if let Ok(neighbors2) = graph.neighbors(&middle1) {
                    for middle2 in neighbors2 {
                        if middle2 == *start_node {
                            continue;
                        }

                        if let Ok(neighbors3) = graph.neighbors(&middle2) {
                            for end_node in neighbors3 {
                                if end_node == middle1 || end_node == *start_node {
                                    continue;
                                }

                                // Check it's a path (no shortcuts)
                                if !graph.has_edge(start_node, &middle2)
                                    && !graph.has_edge(start_node, &end_node)
                                    && !graph.has_edge(&middle1, &end_node)
                                {
                                    let mut path = vec![
                                        start_node.clone(),
                                        middle1.clone(),
                                        middle2.clone(),
                                        end_node.clone(),
                                    ];
                                    path.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                                    let mut paths_guard = paths.lock().unwrap();
                                    if !paths_guard.iter().any(|p| p == &path) {
                                        paths_guard.push(path);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    paths.into_inner().unwrap()
}

/// Find bi-fan motifs (2 nodes connected to the same 2 other nodes)
#[allow(dead_code)]
fn find_bi_fans<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let bi_fans = Mutex::new(Vec::new());

    nodes.par_iter().enumerate().for_each(|(i, node1)| {
        for node2 in nodes.iter().skip(i + 1) {
            if let (Ok(neighbors1), Ok(neighbors2)) =
                (graph.neighbors(node1), graph.neighbors(node2))
            {
                let neighbors1: HashSet<_> = neighbors1.into_iter().collect();
                let neighbors2: HashSet<_> = neighbors2.into_iter().collect();

                // Find common neighbors (excluding node1 and node2)
                let common: Vec<_> = neighbors1
                    .intersection(&neighbors2)
                    .filter(|&n| n != node1 && n != node2)
                    .cloned()
                    .collect();

                if common.len() >= 2 {
                    // For each pair of common neighbors, create a bi-fan
                    for (j, fan1) in common.iter().enumerate() {
                        for fan2 in common.iter().skip(j + 1) {
                            let mut bi_fan =
                                vec![node1.clone(), node2.clone(), fan1.clone(), fan2.clone()];
                            bi_fan.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                            let mut bi_fans_guard = bi_fans.lock().unwrap();
                            if !bi_fans_guard.iter().any(|bf| bf == &bi_fan) {
                                bi_fans_guard.push(bi_fan);
                            }
                        }
                    }
                }
            }
        }
    });

    bi_fans.into_inner().unwrap()
}

/// Find feed-forward loop motifs (3 nodes with specific directed pattern)
#[allow(dead_code)]
fn find_feed_forward_loops<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let ffls = Mutex::new(Vec::new());

    // Feed-forward loop: A->B, A->C, B->C (but not A<-B, A<-C, B<-C)
    nodes.par_iter().for_each(|node_a| {
        if let Ok(out_neighbors_a) = graph.neighbors(node_a) {
            let out_neighbors_a: Vec<_> = out_neighbors_a;

            for (i, node_b) in out_neighbors_a.iter().enumerate() {
                for node_c in out_neighbors_a.iter().skip(i + 1) {
                    // Check if B->C exists and no back edges exist
                    if graph.has_edge(node_b, node_c) {
                        // Ensure it's a true feed-forward (no cycles back)
                        if !graph.has_edge(node_b, node_a)
                            && !graph.has_edge(node_c, node_a)
                            && !graph.has_edge(node_c, node_b)
                        {
                            let mut ffl = vec![node_a.clone(), node_b.clone(), node_c.clone()];
                            ffl.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                            let mut ffls_guard = ffls.lock().unwrap();
                            if !ffls_guard.iter().any(|f| f == &ffl) {
                                ffls_guard.push(ffl);
                            }
                        }
                    }
                }
            }
        }
    });

    ffls.into_inner().unwrap()
}

/// Find bi-directional motifs (mutual connections between pairs of nodes)
#[allow(dead_code)]
fn find_bidirectional_motifs<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let bidirectionals = Mutex::new(Vec::new());

    nodes.par_iter().enumerate().for_each(|(i, node1)| {
        for node2 in nodes.iter().skip(i + 1) {
            // Check for bidirectional connection
            if graph.has_edge(node1, node2) && graph.has_edge(node2, node1) {
                let mut motif = vec![node1.clone(), node2.clone()];
                motif.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                let mut bidirectionals_guard = bidirectionals.lock().unwrap();
                if !bidirectionals_guard.iter().any(|m| m == &motif) {
                    bidirectionals_guard.push(motif);
                }
            }
        }
    });

    bidirectionals.into_inner().unwrap()
}

/// Advanced motif counting with frequency analysis
/// Returns a map of motif patterns to their occurrence counts
#[allow(dead_code)]
pub fn count_motif_frequencies<N, E, Ix>(graph: &Graph<N, E, Ix>) -> HashMap<MotifType, usize>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    let motif_types = vec![
        MotifType::Triangle,
        MotifType::Square,
        MotifType::Star3,
        MotifType::Clique4,
        MotifType::Path3,
        MotifType::BiFan,
        MotifType::FeedForwardLoop,
        MotifType::BiDirectional,
    ];

    motif_types
        .par_iter()
        .map(|motif_type| {
            let count = find_motifs(graph, *motif_type).len();
            (*motif_type, count)
        })
        .collect()
}

/// Efficient motif detection using sampling for large graphs
/// Returns estimated motif counts based on random sampling
#[allow(dead_code)]
pub fn sample_motif_frequencies<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    sample_size: usize,
    rng: &mut impl scirs2_core::random::Rng,
) -> HashMap<MotifType, f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::random::seq::SliceRandom;

    let all_nodes: Vec<_> = graph.nodes().into_iter().cloned().collect();
    if all_nodes.len() <= sample_size {
        // If graph is small, do exact counting
        return count_motif_frequencies(graph)
            .into_iter()
            .map(|(k, v)| (k, v as f64))
            .collect();
    }

    // Sample nodes
    let mut sampled_nodes = all_nodes.clone();
    sampled_nodes.shuffle(rng);
    sampled_nodes.truncate(sample_size);

    // Create subgraph from sampled nodes
    let mut subgraph = crate::generators::create_graph::<N, E>();
    for node in &sampled_nodes {
        let _ = subgraph.add_node(node.clone());
    }

    // Add edges between sampled nodes
    for node1 in &sampled_nodes {
        if let Ok(neighbors) = graph.neighbors(node1) {
            for node2 in neighbors {
                if sampled_nodes.contains(&node2) && node1 != &node2 {
                    if let Ok(weight) = graph.edge_weight(node1, &node2) {
                        let _ = subgraph.add_edge(node1.clone(), node2, weight);
                    }
                }
            }
        }
    }

    // Count motifs in subgraph and extrapolate
    let subgraph_counts = count_motif_frequencies(&subgraph);
    let scaling_factor = (all_nodes.len() as f64) / (sample_size as f64);

    subgraph_counts
        .into_iter()
        .map(|(motif_type, count)| (motif_type, count as f64 * scaling_factor))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_find_triangles() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a triangle ABC
        graph.add_edge("A", "B", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("C", "A", ())?;

        // Add another node D connected to A (not forming new triangles)
        graph.add_edge("A", "D", ())?;

        let triangles = find_motifs(&graph, MotifType::Triangle);
        assert_eq!(triangles.len(), 1);

        // The triangle should contain A, B, and C
        let triangle = &triangles[0];
        assert_eq!(triangle.len(), 3);
        assert!(triangle.contains(&"A"));
        assert!(triangle.contains(&"B"));
        assert!(triangle.contains(&"C"));

        Ok(())
    }

    #[test]
    fn test_find_squares() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a square ABCD
        graph.add_edge("A", "B", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("C", "D", ())?;
        graph.add_edge("D", "A", ())?;

        let squares = find_motifs(&graph, MotifType::Square);
        assert_eq!(squares.len(), 1);

        let square = &squares[0];
        assert_eq!(square.len(), 4);

        Ok(())
    }

    #[test]
    fn test_find_star3() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a star with center A and leaves B, C, D
        graph.add_edge("A", "B", ())?;
        graph.add_edge("A", "C", ())?;
        graph.add_edge("A", "D", ())?;

        let stars = find_motifs(&graph, MotifType::Star3);
        assert_eq!(stars.len(), 1);

        let star = &stars[0];
        assert_eq!(star.len(), 4);
        assert!(star.contains(&"A")); // Center should be included

        Ok(())
    }

    #[test]
    fn test_find_clique4() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();

        // Create a complete graph K4
        let nodes = ["A", "B", "C", "D"];
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                graph.add_edge(nodes[i], nodes[j], ())?;
            }
        }

        let cliques = find_motifs(&graph, MotifType::Clique4);
        assert_eq!(cliques.len(), 1);

        let clique = &cliques[0];
        assert_eq!(clique.len(), 4);

        Ok(())
    }
}
