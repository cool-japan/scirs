//! Graph generation algorithms
//!
//! This module provides functions for generating various types of graphs:
//! - Random graphs (Erdős–Rényi, Barabási–Albert, etc.)
//! - Regular graphs (complete, star, path, cycle)
//! - Lattice graphs
//! - Small-world networks

use rand::prelude::*;
use std::collections::HashSet;

use crate::base::{DiGraph, Graph};
use crate::error::{GraphError, Result};
use rand::seq::SliceRandom;

/// Create a new empty undirected graph
#[allow(dead_code)]
pub fn create_graph<N: crate::base::Node + std::fmt::Debug, E: crate::base::EdgeWeight>(
) -> Graph<N, E> {
    Graph::new()
}

/// Create a new empty directed graph
#[allow(dead_code)]
pub fn create_digraph<N: crate::base::Node + std::fmt::Debug, E: crate::base::EdgeWeight>(
) -> DiGraph<N, E> {
    DiGraph::new()
}

/// Generates an Erdős–Rényi random graph
///
/// # Arguments
/// * `n` - Number of nodes
/// * `p` - Probability of edge creation between any two nodes
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph with node IDs 0..n-1
#[allow(dead_code)]
pub fn erdos_renyi_graph<R: Rng>(n: usize, p: f64, rng: &mut R) -> Result<Graph<usize, f64>> {
    if !(0.0..=1.0).contains(&p) {
        return Err(GraphError::InvalidGraph(
            "Probability must be between 0 and 1".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Add edges with probability p
    for i in 0..n {
        for j in i + 1..n {
            if rng.random::<f64>() < p {
                graph.add_edge(i, j, 1.0)?;
            }
        }
    }

    Ok(graph)
}

/// Generates a Barabási–Albert preferential attachment graph
///
/// # Arguments
/// * `n` - Total number of nodes
/// * `m` - Number of edges to attach from a new node to existing nodes
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph with node IDs 0..n-1
#[allow(dead_code)]
pub fn barabasi_albert_graph<R: Rng>(n: usize, m: usize, rng: &mut R) -> Result<Graph<usize, f64>> {
    if m >= n {
        return Err(GraphError::InvalidGraph(
            "m must be less than n".to_string(),
        ));
    }
    if m == 0 {
        return Err(GraphError::InvalidGraph("m must be positive".to_string()));
    }

    let mut graph = Graph::new();

    // Start with a complete graph of m+1 nodes
    for i in 0..=m {
        graph.add_node(i);
    }

    for i in 0..=m {
        for j in i + 1..=m {
            graph.add_edge(i, j, 1.0)?;
        }
    }

    // Keep track of node degrees for preferential attachment
    let mut degrees = vec![m; m + 1];
    let mut total_degree = m * (m + 1);

    // Add remaining nodes
    for new_node in (m + 1)..n {
        graph.add_node(new_node);

        let mut targets = HashSet::new();

        // Select m nodes to connect to based on preferential attachment
        while targets.len() < m {
            let mut cumulative_prob = 0.0;
            let random_value = rng.random::<f64>() * total_degree as f64;

            for (node_id, &degree) in degrees.iter().enumerate() {
                cumulative_prob += degree as f64;
                if random_value <= cumulative_prob && !targets.contains(&node_id) {
                    targets.insert(node_id);
                    break;
                }
            }
        }

        // Add edges to selected targets
        for &target in &targets {
            graph.add_edge(new_node, target, 1.0)?;
            degrees[target] += 1;
            total_degree += 2; // Each edge adds 2 to total degree
        }

        degrees.push(m); // New node has degree m
    }

    Ok(graph)
}

/// Generates a complete graph (clique)
///
/// # Arguments
/// * `n` - Number of nodes
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A complete graph with n nodes
#[allow(dead_code)]
pub fn complete_graph(n: usize) -> Result<Graph<usize, f64>> {
    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Add all possible edges
    for i in 0..n {
        for j in i + 1..n {
            graph.add_edge(i, j, 1.0)?;
        }
    }

    Ok(graph)
}

/// Generates a star graph with one central node connected to all others
///
/// # Arguments
/// * `n` - Total number of nodes (must be >= 1)
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A star graph with node 0 as the center
#[allow(dead_code)]
pub fn star_graph(n: usize) -> Result<Graph<usize, f64>> {
    if n == 0 {
        return Err(GraphError::InvalidGraph(
            "Star graph must have at least 1 node".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Connect center (node 0) to all other nodes
    for i in 1..n {
        graph.add_edge(0, i, 1.0)?;
    }

    Ok(graph)
}

/// Generates a path graph (nodes connected in a line)
///
/// # Arguments
/// * `n` - Number of nodes
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A path graph with nodes 0, 1, ..., n-1
#[allow(dead_code)]
pub fn path_graph(n: usize) -> Result<Graph<usize, f64>> {
    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Connect consecutive nodes
    for i in 0..n.saturating_sub(1) {
        graph.add_edge(i, i + 1, 1.0)?;
    }

    Ok(graph)
}

/// Generates a random tree with n nodes
///
/// Uses a random process to connect nodes while maintaining the tree property
/// (connected and acyclic). Each tree has exactly n-1 edges.
///
/// # Arguments
/// * `n` - Number of nodes
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A random tree with nodes 0, 1, ..., n-1
#[allow(dead_code)]
pub fn tree_graph<R: Rng>(n: usize, rng: &mut R) -> Result<Graph<usize, f64>> {
    if n == 0 {
        return Ok(Graph::new());
    }
    if n == 1 {
        let mut graph = Graph::new();
        graph.add_node(0);
        return Ok(graph);
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Use Prim's algorithm variation to build a random tree
    let mut in_tree = vec![false; n];
    let mut tree_nodes = Vec::new();

    // Start with a random node
    let start = rng.gen_range(0..n);
    in_tree[start] = true;
    tree_nodes.push(start);

    // Add n-1 edges to complete the tree
    for _ in 1..n {
        // Pick a random node already in the tree
        let tree_node = tree_nodes[rng.gen_range(0..tree_nodes.len())];

        // Pick a random node not yet in the tree
        let candidates: Vec<usize> = (0..n).filter(|&i| !in_tree[i]).collect();
        if candidates.is_empty() {
            break;
        }

        let new_node = candidates[rng.gen_range(0..candidates.len())];

        // Add edge and mark node as in tree
        graph.add_edge(tree_node, new_node, 1.0)?;
        in_tree[new_node] = true;
        tree_nodes.push(new_node);
    }

    Ok(graph)
}

/// Generates a random spanning tree from an existing graph
///
/// Uses Kruskal's algorithm with randomized edge selection to produce
/// a random spanning tree of the input graph.
///
/// # Arguments
/// * `graph` - The input graph to extract a spanning tree from
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<N, E>>` - A spanning tree of the input graph
#[allow(dead_code)]
pub fn random_spanning_tree<N, E, Ix, R>(
    graph: &Graph<N, E, Ix>,
    rng: &mut R,
) -> Result<Graph<N, E, Ix>>
where
    N: crate::base::Node + std::fmt::Debug,
    E: crate::base::EdgeWeight + Clone,
    Ix: petgraph::graph::IndexType,
    R: Rng,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    if nodes.is_empty() {
        return Ok(Graph::new());
    }
    if nodes.len() == 1 {
        let mut tree = Graph::new();
        tree.add_node(nodes[0].clone());
        return Ok(tree);
    }

    // Get all edges and shuffle them randomly
    let mut edges: Vec<_> = graph.edges().into_iter().collect();
    edges.shuffle(rng);

    let mut tree = Graph::new();

    // Add all nodes to the tree
    for node in &nodes {
        tree.add_node(node.clone());
    }

    // Use Union-Find to track components
    let mut parent: std::collections::HashMap<N, N> =
        nodes.iter().map(|n| (n.clone(), n.clone())).collect();
    let mut rank: std::collections::HashMap<N, usize> =
        nodes.iter().map(|n| (n.clone(), 0)).collect();

    fn find<N: crate::base::Node>(parent: &mut std::collections::HashMap<N, N>, node: &N) -> N {
        if parent[node] != *node {
            let root = find(parent, &parent[node].clone());
            parent.insert(node.clone(), root.clone());
        }
        parent[node].clone()
    }

    fn union<N: crate::base::Node>(
        parent: &mut std::collections::HashMap<N, N>,
        rank: &mut std::collections::HashMap<N, usize>,
        x: &N,
        y: &N,
    ) -> bool {
        let root_x = find(parent, x);
        let root_y = find(parent, y);

        if root_x == root_y {
            return false; // Already in same component
        }

        // Union by rank
        match rank[&root_x].cmp(&rank[&root_y]) {
            std::cmp::Ordering::Less => {
                parent.insert(root_x, root_y);
            }
            std::cmp::Ordering::Greater => {
                parent.insert(root_y, root_x);
            }
            std::cmp::Ordering::Equal => {
                parent.insert(root_y, root_x.clone());
                *rank.get_mut(&root_x).unwrap() += 1;
            }
        }
        true
    }

    let mut edges_added = 0;

    // Add edges without creating cycles until we have n-1 edges
    for edge in edges {
        if union(&mut parent, &mut rank, &edge.source, &edge.target) {
            tree.add_edge(edge.source, edge.target, edge.weight)?;
            edges_added += 1;
            if edges_added == nodes.len() - 1 {
                break;
            }
        }
    }

    // Check if we have a spanning tree (connected graph)
    if edges_added != nodes.len() - 1 {
        return Err(GraphError::InvalidGraph(
            "Input graph is not connected - cannot create spanning tree".to_string(),
        ));
    }

    Ok(tree)
}

/// Generates a random forest (collection of trees)
///
/// Creates a forest by generating multiple random trees and combining them
/// into a single graph. The trees are disjoint (no edges between different trees).
///
/// # Arguments
/// * `tree_sizes` - Vector specifying the size of each tree in the forest
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A forest containing the specified trees
#[allow(dead_code)]
pub fn forest_graph<R: Rng>(
    _tree_sizes: &[usize],
    sizes: &[usize],
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    let mut forest = Graph::new();
    let mut node_offset = 0;

    for &tree_size in _tree_sizes {
        if tree_size == 0 {
            continue;
        }

        // Generate a tree with nodes starting from node_offset
        let tree = tree_graph(tree_size, rng)?;

        // Add nodes to forest with offset
        for i in 0..tree_size {
            forest.add_node(node_offset + i);
        }

        // Add edges with offset
        for edge in tree.edges() {
            forest.add_edge(
                node_offset + edge.source,
                node_offset + edge.target,
                edge.weight,
            )?;
        }

        node_offset += tree_size;
    }

    Ok(forest)
}

/// Generates a cycle graph (circular arrangement of nodes)
///
/// # Arguments
/// * `n` - Number of nodes (must be >= 3 for a meaningful cycle)
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A cycle graph with nodes 0, 1, ..., n-1
#[allow(dead_code)]
pub fn cycle_graph(n: usize) -> Result<Graph<usize, f64>> {
    if n < 3 {
        return Err(GraphError::InvalidGraph(
            "Cycle graph must have at least 3 nodes".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Connect consecutive nodes
    for i in 0..n {
        graph.add_edge(i, (i + 1) % n, 1.0)?;
    }

    Ok(graph)
}

/// Generates a 2D grid/lattice graph
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A grid graph where node ID = row * cols + col
#[allow(dead_code)]
pub fn grid_2d_graph(rows: usize, cols: usize) -> Result<Graph<usize, f64>> {
    if rows == 0 || cols == 0 {
        return Err(GraphError::InvalidGraph(
            "Grid dimensions must be positive".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..(rows * cols) {
        graph.add_node(i);
    }

    // Add edges to adjacent nodes (4-connectivity)
    for row in 0..rows {
        for col in 0..cols {
            let node_id = row * cols + col;

            // Connect to right neighbor
            if col + 1 < cols {
                let right_neighbor = row * cols + (col + 1);
                graph.add_edge(node_id, right_neighbor, 1.0)?;
            }

            // Connect to bottom neighbor
            if row + 1 < rows {
                let bottom_neighbor = (row + 1) * cols + col;
                graph.add_edge(node_id, bottom_neighbor, 1.0)?;
            }
        }
    }

    Ok(graph)
}

/// Generates a 3D grid/lattice graph
///
/// # Arguments
/// * `x_dim` - Size in x dimension
/// * `y_dim` - Size in y dimension  
/// * `z_dim` - Size in z dimension
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A 3D grid graph where node ID = z*x_dim*y_dim + y*x_dim + x
#[allow(dead_code)]
pub fn grid_3d_graph(x_dim: usize, y_dim: usize, z_dim: usize) -> Result<Graph<usize, f64>> {
    if x_dim == 0 || y_dim == 0 || z_dim == 0 {
        return Err(GraphError::InvalidGraph(
            "Grid dimensions must be positive".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..(x_dim * y_dim * z_dim) {
        graph.add_node(i);
    }

    // Connect neighbors in 3D grid
    for z in 0..z_dim {
        for y in 0..y_dim {
            for x in 0..x_dim {
                let node_id = z * x_dim * y_dim + y * x_dim + x;

                // Connect to right neighbor
                if x + 1 < x_dim {
                    let right_neighbor = z * x_dim * y_dim + y * x_dim + (x + 1);
                    graph.add_edge(node_id, right_neighbor, 1.0)?;
                }

                // Connect to front neighbor
                if y + 1 < y_dim {
                    let front_neighbor = z * x_dim * y_dim + (y + 1) * x_dim + x;
                    graph.add_edge(node_id, front_neighbor, 1.0)?;
                }

                // Connect to top neighbor
                if z + 1 < z_dim {
                    let top_neighbor = (z + 1) * x_dim * y_dim + y * x_dim + x;
                    graph.add_edge(node_id, top_neighbor, 1.0)?;
                }
            }
        }
    }

    Ok(graph)
}

/// Generates a triangular lattice graph
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A triangular lattice where each node has up to 6 neighbors
#[allow(dead_code)]
pub fn triangular_lattice_graph(rows: usize, cols: usize) -> Result<Graph<usize, f64>> {
    if rows == 0 || cols == 0 {
        return Err(GraphError::InvalidGraph(
            "Lattice dimensions must be positive".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..(rows * cols) {
        graph.add_node(i);
    }

    for row in 0..rows {
        for col in 0..cols {
            let node_id = row * cols + col;

            // Standard grid connections (4-connected)
            // Right neighbor
            if col + 1 < cols {
                let right_neighbor = row * cols + (col + 1);
                graph.add_edge(node_id, right_neighbor, 1.0)?;
            }

            // Bottom neighbor
            if row + 1 < rows {
                let bottom_neighbor = (row + 1) * cols + col;
                graph.add_edge(node_id, bottom_neighbor, 1.0)?;
            }

            // Diagonal connections for triangular lattice
            // Bottom-right diagonal
            if row + 1 < rows && col + 1 < cols {
                let diag_neighbor = (row + 1) * cols + (col + 1);
                graph.add_edge(node_id, diag_neighbor, 1.0)?;
            }

            // Bottom-left diagonal (for even rows)
            if row + 1 < rows && col > 0 && row % 2 == 0 {
                let diag_neighbor = (row + 1) * cols + (col - 1);
                graph.add_edge(node_id, diag_neighbor, 1.0)?;
            }
        }
    }

    Ok(graph)
}

/// Generates a hexagonal lattice graph (honeycomb structure)
///
/// # Arguments
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A hexagonal lattice where each node has exactly 3 neighbors
#[allow(dead_code)]
pub fn hexagonal_lattice_graph(rows: usize, cols: usize) -> Result<Graph<usize, f64>> {
    if rows == 0 || cols == 0 {
        return Err(GraphError::InvalidGraph(
            "Lattice dimensions must be positive".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..(rows * cols) {
        graph.add_node(i);
    }

    for row in 0..rows {
        for col in 0..cols {
            let node_id = row * cols + col;

            // Hexagonal lattice connections (3 neighbors per node in honeycomb pattern)
            // This creates a simplified hexagonal structure

            // Right neighbor (horizontal)
            if col + 1 < cols {
                let right_neighbor = row * cols + (col + 1);
                graph.add_edge(node_id, right_neighbor, 1.0)?;
            }

            // Connect in honeycomb pattern
            if row % 2 == 0 {
                // Even _rows: connect down-left and down-right
                if row + 1 < rows {
                    if col > 0 {
                        let down_left = (row + 1) * cols + (col - 1);
                        graph.add_edge(node_id, down_left, 1.0)?;
                    }
                    if col < cols {
                        let down_right = (row + 1) * cols + col;
                        graph.add_edge(node_id, down_right, 1.0)?;
                    }
                }
            } else {
                // Odd _rows: connect down-left and down-right with offset
                if row + 1 < rows {
                    let down_left = (row + 1) * cols + col;
                    graph.add_edge(node_id, down_left, 1.0)?;

                    if col + 1 < cols {
                        let down_right = (row + 1) * cols + (col + 1);
                        graph.add_edge(node_id, down_right, 1.0)?;
                    }
                }
            }
        }
    }

    Ok(graph)
}

/// Generates a Watts-Strogatz small-world graph
///
/// # Arguments
/// * `n` - Number of nodes
/// * `k` - Each node is connected to k nearest neighbors in ring topology (must be even)
/// * `p` - Probability of rewiring each edge
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - A small-world graph
#[allow(dead_code)]
pub fn watts_strogatz_graph<R: Rng>(
    n: usize,
    k: usize,
    p: f64,
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    if k >= n || k % 2 != 0 {
        return Err(GraphError::InvalidGraph(
            "k must be even and less than n".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&p) {
        return Err(GraphError::InvalidGraph(
            "Probability must be between 0 and 1".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Create regular ring lattice
    for i in 0..n {
        for j in 1..=(k / 2) {
            let neighbor = (i + j) % n;
            graph.add_edge(i, neighbor, 1.0)?;
        }
    }

    // Rewire edges with probability p
    let edges_to_process: Vec<_> = graph.edges().into_iter().collect();

    for edge in edges_to_process {
        if rng.random::<f64>() < p {
            // Remove the original edge (we'll recreate the graph to do this)
            let mut new_graph = Graph::new();

            // Add all nodes
            for i in 0..n {
                new_graph.add_node(i);
            }

            // Add all edges except the one we're rewiring
            for existing_edge in graph.edges() {
                if (existing_edge.source != edge.source || existing_edge.target != edge.target)
                    && (existing_edge.source != edge.target || existing_edge.target != edge.source)
                {
                    new_graph.add_edge(
                        existing_edge.source,
                        existing_edge.target,
                        existing_edge.weight,
                    )?;
                }
            }

            // Add rewired edge to a random node
            let mut new_target = rng.gen_range(0..n);
            while new_target == edge.source || new_graph.has_node(&new_target) {
                new_target = rng.gen_range(0..n);
            }

            new_graph.add_edge(edge.source, new_target, 1.0)?;
            graph = new_graph;
        }
    }

    Ok(graph)
}

/// Generates a graph using the Stochastic Block Model (SBM)
///
/// The SBM generates a graph where nodes are divided into communities (blocks)
/// and edge probabilities depend on which communities the nodes belong to.
///
/// # Arguments
/// * `block_sizes` - Vector specifying the size of each block/community
/// * `block_matrix` - Probability matrix where entry (i,j) is the probability
///   of an edge between nodes in block i and block j
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph with node IDs 0..n-1
///   where nodes 0..block_sizes[0]-1 are in block 0, etc.
#[allow(dead_code)]
pub fn stochastic_block_model<R: Rng>(
    block_sizes: &[usize],
    block_matrix: &[Vec<f64>],
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    if block_sizes.is_empty() {
        return Err(GraphError::InvalidGraph(
            "At least one block must be specified".to_string(),
        ));
    }

    if block_matrix.len() != block_sizes.len() {
        return Err(GraphError::InvalidGraph(
            "Block _matrix dimensions must match number of blocks".to_string(),
        ));
    }

    for row in block_matrix {
        if row.len() != block_sizes.len() {
            return Err(GraphError::InvalidGraph(
                "Block _matrix must be square".to_string(),
            ));
        }
        for &prob in row {
            if !(0.0..=1.0).contains(&prob) {
                return Err(GraphError::InvalidGraph(
                    "All probabilities must be between 0 and 1".to_string(),
                ));
            }
        }
    }

    let total_nodes: usize = block_sizes.iter().sum();
    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..total_nodes {
        graph.add_node(i);
    }

    // Create mapping from node to block
    let mut node_to_block = vec![0; total_nodes];
    let mut current_node = 0;
    for (block_id, &block_size) in block_sizes.iter().enumerate() {
        for _ in 0..block_size {
            node_to_block[current_node] = block_id;
            current_node += 1;
        }
    }

    // Generate edges based on block probabilities
    for i in 0..total_nodes {
        for j in (i + 1)..total_nodes {
            let block_i = node_to_block[i];
            let block_j = node_to_block[j];
            let prob = block_matrix[block_i][block_j];

            if rng.random::<f64>() < prob {
                graph.add_edge(i, j, 1.0)?;
            }
        }
    }

    Ok(graph)
}

/// Generates a simple stochastic block model with two communities
///
/// This is a convenience function for creating a two-community SBM with
/// high intra-community probability and low inter-community probability.
///
/// # Arguments
/// * `n1` - Size of first community
/// * `n2` - Size of second community
/// * `p_in` - Probability of edges within communities
/// * `p_out` - Probability of edges between communities
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph
#[allow(dead_code)]
pub fn two_community_sbm<R: Rng>(
    n1: usize,
    n2: usize,
    p_in: f64,
    p_out: f64,
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    let block_sizes = vec![n1, n2];
    let block_matrix = vec![vec![p_in, p_out], vec![p_out, p_in]];

    stochastic_block_model(&block_sizes, &block_matrix, rng)
}

/// Generates a planted partition model (special case of SBM)
///
/// In this model, there are k communities of equal size, with high
/// intra-community probability and low inter-community probability.
///
/// # Arguments
/// * `n` - Total number of nodes (must be divisible by k)
/// * `k` - Number of communities
/// * `p_in` - Probability of edges within communities
/// * `p_out` - Probability of edges between communities
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph
#[allow(dead_code)]
pub fn planted_partition_model<R: Rng>(
    n: usize,
    k: usize,
    p_in: f64,
    p_out: f64,
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    if n % k != 0 {
        return Err(GraphError::InvalidGraph(
            "Number of nodes must be divisible by number of communities".to_string(),
        ));
    }

    let community_size = n / k;
    let block_sizes = vec![community_size; k];

    // Create block matrix
    let mut block_matrix = vec![vec![p_out; k]; k];
    for (i, row) in block_matrix.iter_mut().enumerate().take(k) {
        row[i] = p_in;
    }

    stochastic_block_model(&block_sizes, &block_matrix, rng)
}

/// Generates a random graph using the Configuration Model
///
/// The Configuration Model generates a random graph where each node has a specified degree.
/// The degree sequence is the sequence of degrees for all nodes. The algorithm creates
/// "stubs" (half-edges) for each node according to its degree, then randomly connects
/// the stubs to form edges.
///
/// # Arguments
/// * `degree_sequence` - Vector specifying the degree of each node
/// * `rng` - Random number generator
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated graph with node IDs 0..n-1
///
/// # Notes
/// * The sum of all degrees must be even (since each edge contributes 2 to the total degree)
/// * Self-loops and multiple edges between the same pair of nodes are possible
/// * If you want a simple graph (no self-loops or multiple edges), you may need to
///   regenerate or post-process the result
#[allow(dead_code)]
pub fn configuration_model<R: Rng>(
    degree_sequence: &[usize],
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    if degree_sequence.is_empty() {
        return Ok(Graph::new());
    }

    // Check that sum of degrees is even
    let total_degree: usize = degree_sequence.iter().sum();
    if total_degree % 2 != 0 {
        return Err(GraphError::InvalidGraph(
            "Sum of degrees must be even".to_string(),
        ));
    }

    let n = degree_sequence.len();
    let mut graph = Graph::new();

    // Add all nodes
    for i in 0..n {
        graph.add_node(i);
    }

    // Create stubs (half-edges) for each node
    let mut stubs = Vec::new();
    for (node_id, &degree) in degree_sequence.iter().enumerate() {
        for _ in 0..degree {
            stubs.push(node_id);
        }
    }

    // Randomly connect stubs to form edges
    while stubs.len() >= 2 {
        // Pick two random stubs
        let idx1 = rng.gen_range(0..stubs.len());
        let stub1 = stubs.remove(idx1);

        let idx2 = rng.gen_range(0..stubs.len());
        let stub2 = stubs.remove(idx2);

        // Connect the nodes (allow self-loops and multiple edges)
        graph.add_edge(stub1, stub2, 1.0)?;
    }

    Ok(graph)
}

/// Generates a simple random graph using the Configuration Model
///
/// This variant attempts to generate a simple graph (no self-loops or multiple edges)
/// by rejecting problematic edge attempts. If too many rejections occur, it returns
/// an error indicating that the degree sequence may not be realizable as a simple graph.
///
/// # Arguments
/// * `degree_sequence` - Vector specifying the degree of each node
/// * `rng` - Random number generator
/// * `max_attempts` - Maximum number of attempts before giving up
///
/// # Returns
/// * `Result<Graph<usize, f64>>` - The generated simple graph
#[allow(dead_code)]
pub fn simple_configuration_model<R: Rng>(
    degree_sequence: &[usize],
    rng: &mut R,
    max_attempts: usize,
) -> Result<Graph<usize, f64>> {
    if degree_sequence.is_empty() {
        return Ok(Graph::new());
    }

    // Check that sum of degrees is even
    let total_degree: usize = degree_sequence.iter().sum();
    if total_degree % 2 != 0 {
        return Err(GraphError::InvalidGraph(
            "Sum of degrees must be even".to_string(),
        ));
    }

    let n = degree_sequence.len();

    // Check for degree _sequence constraints for simple graphs
    for &degree in degree_sequence {
        if degree >= n {
            return Err(GraphError::InvalidGraph(
                "Node degree cannot exceed n-1 in a simple graph".to_string(),
            ));
        }
    }

    let mut _attempts = 0;

    while _attempts < max_attempts {
        let mut graph = Graph::new();

        // Add all nodes
        for i in 0..n {
            graph.add_node(i);
        }

        // Create stubs (half-edges) for each node
        let mut stubs = Vec::new();
        for (node_id, &degree) in degree_sequence.iter().enumerate() {
            for _ in 0..degree {
                stubs.push(node_id);
            }
        }

        let mut success = true;

        // Randomly connect stubs to form edges
        while stubs.len() >= 2 && success {
            // Pick two random stubs
            let idx1 = rng.gen_range(0..stubs.len());
            let stub1 = stubs[idx1];

            let idx2 = rng.gen_range(0..stubs.len());
            let stub2 = stubs[idx2];

            // Check for self-loop or existing edge
            if stub1 == stub2 || graph.has_edge(&stub1, &stub2) {
                // Try a few more times before giving up on this attempt
                let mut retries = 0;
                let mut found_valid = false;

                while retries < 50 && !found_valid {
                    let new_idx2 = rng.gen_range(0..stubs.len());
                    let new_stub2 = stubs[new_idx2];

                    if stub1 != new_stub2 && !graph.has_edge(&stub1, &new_stub2) {
                        // Remove stubs and add edge
                        // Remove the larger index first to avoid index shifting issues
                        if idx1 > new_idx2 {
                            stubs.remove(idx1);
                            stubs.remove(new_idx2);
                        } else {
                            stubs.remove(new_idx2);
                            stubs.remove(idx1);
                        }
                        graph.add_edge(stub1, new_stub2, 1.0)?;
                        found_valid = true;
                    }
                    retries += 1;
                }

                if !found_valid {
                    success = false;
                }
            } else {
                // Remove stubs and add edge
                // Remove the larger index first to avoid index shifting issues
                if idx1 > idx2 {
                    stubs.remove(idx1);
                    stubs.remove(idx2);
                } else {
                    stubs.remove(idx2);
                    stubs.remove(idx1);
                }
                graph.add_edge(stub1, stub2, 1.0)?;
            }
        }

        if success && stubs.is_empty() {
            return Ok(graph);
        }

        _attempts += 1;
    }

    Err(GraphError::InvalidGraph(
        "Could not generate simple graph with given degree _sequence after maximum _attempts"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erdos_renyi_graph() {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = erdos_renyi_graph(10, 0.3, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 10);
        // With p=0.3 and 45 possible edges, we expect around 13-14 edges
        // but this is random, so we just check it's reasonable
        assert!(graph.edge_count() <= 45);
    }

    #[test]
    fn test_complete_graph() {
        let graph = complete_graph(5).unwrap();

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 10); // n*(n-1)/2 = 5*4/2 = 10
    }

    #[test]
    fn test_star_graph() {
        let graph = star_graph(6).unwrap();

        assert_eq!(graph.node_count(), 6);
        assert_eq!(graph.edge_count(), 5); // n-1 edges
    }

    #[test]
    fn test_path_graph() {
        let graph = path_graph(5).unwrap();

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4); // n-1 edges
    }

    #[test]
    fn test_cycle_graph() {
        let graph = cycle_graph(5).unwrap();

        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 5); // n edges

        // Test error case
        assert!(cycle_graph(2).is_err());
    }

    #[test]
    fn test_grid_2d_graph() {
        let graph = grid_2d_graph(3, 4).unwrap();

        assert_eq!(graph.node_count(), 12); // 3*4 = 12 nodes
        assert_eq!(graph.edge_count(), 17); // (3-1)*4 + 3*(4-1) = 8 + 9 = 17 edges
    }

    #[test]
    fn test_grid_3d_graph() {
        let graph = grid_3d_graph(2, 2, 2).unwrap();

        assert_eq!(graph.node_count(), 8); // 2*2*2 = 8 nodes
                                           // Each internal node connects to 3 neighbors in 3D grid
                                           // Expected edges: 3 faces × 2 edges per face + 3 additional connections = 12 edges
        assert_eq!(graph.edge_count(), 12);
    }

    #[test]
    fn test_triangular_lattice_graph() {
        let graph = triangular_lattice_graph(3, 3).unwrap();

        assert_eq!(graph.node_count(), 9); // 3*3 = 9 nodes
                                           // Triangular lattice has more edges than regular grid due to diagonal connections
        assert!(graph.edge_count() > 12); // More than standard 2D grid edges
    }

    #[test]
    fn test_hexagonal_lattice_graph() {
        let graph = hexagonal_lattice_graph(3, 3).unwrap();

        assert_eq!(graph.node_count(), 9); // 3*3 = 9 nodes
                                           // Hexagonal lattice should have fewer edges than triangular due to honeycomb structure
        assert!(graph.edge_count() >= 6);
    }

    #[test]
    fn test_barabasi_albert_graph() {
        let mut rng = StdRng::seed_from_u64(42);
        let graph = barabasi_albert_graph(10, 2, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 10);
        // Should have 3 + 2*7 = 17 edges (3 initial edges + 2 for each of the 7 new nodes)
        assert_eq!(graph.edge_count(), 17);
    }

    #[test]
    fn test_stochastic_block_model() {
        let mut rng = StdRng::seed_from_u64(42);

        // Two blocks of size 3 and 4
        let block_sizes = vec![3, 4];
        // High intra-block probability, low inter-block probability
        let block_matrix = vec![vec![0.8, 0.1], vec![0.1, 0.8]];

        let graph = stochastic_block_model(&block_sizes, &block_matrix, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 7); // 3 + 4 = 7 nodes

        // Check that all nodes are present
        for i in 0..7 {
            assert!(graph.has_node(&i));
        }
    }

    #[test]
    fn test_two_community_sbm() {
        let mut rng = StdRng::seed_from_u64(42);

        let graph = two_community_sbm(5, 5, 0.8, 0.1, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 10);

        // Should have some edges within communities and fewer between
        // This is probabilistic so we can't test exact numbers
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_planted_partition_model() {
        let mut rng = StdRng::seed_from_u64(42);

        let graph = planted_partition_model(12, 3, 0.7, 0.1, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 12); // 12 nodes total

        // 3 communities of size 4 each
        // Should have some edges
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_stochastic_block_model_errors() {
        let mut rng = StdRng::seed_from_u64(42);

        // Empty blocks
        assert!(stochastic_block_model(&[], &[], &mut rng).is_err());

        // Mismatched dimensions
        let block_sizes = vec![3, 4];
        let wrong_matrix = vec![vec![0.5]];
        assert!(stochastic_block_model(&block_sizes, &wrong_matrix, &mut rng).is_err());

        // Invalid probabilities
        let bad_matrix = vec![vec![1.5, 0.5], vec![0.5, 0.5]];
        assert!(stochastic_block_model(&block_sizes, &bad_matrix, &mut rng).is_err());

        // Non-divisible nodes for planted partition
        assert!(planted_partition_model(10, 3, 0.5, 0.1, &mut rng).is_err());
    }

    #[test]
    fn test_configuration_model() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test valid degree sequence (even sum)
        let degree_sequence = vec![2, 2, 2, 2]; // Sum = 8 (even)
        let graph = configuration_model(&degree_sequence, &mut rng).unwrap();

        assert_eq!(graph.node_count(), 4);
        // Should have 4 edges (sum of degrees / 2)
        assert_eq!(graph.edge_count(), 4);

        // Check that each node has the correct degree
        for (i, &expected_degree) in degree_sequence.iter().enumerate() {
            let actual_degree = graph.degree(&i);
            assert_eq!(actual_degree, expected_degree);
        }
    }

    #[test]
    fn test_configuration_model_errors() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test odd degree sum (should fail)
        let odd_degree_sequence = vec![1, 2, 2]; // Sum = 5 (odd)
        assert!(configuration_model(&odd_degree_sequence, &mut rng).is_err());

        // Test empty sequence
        let empty_sequence = vec![];
        let graph = configuration_model(&empty_sequence, &mut rng).unwrap();
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_simple_configuration_model() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test valid degree sequence for simple graph
        let degree_sequence = vec![2, 2, 2, 2]; // Sum = 8 (even)
        let graph = simple_configuration_model(&degree_sequence, &mut rng, 100).unwrap();

        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 4);

        // Check that graph is simple (no self-loops)
        for i in 0..4 {
            assert!(!graph.has_edge(&i, &i), "Graph should not have self-loops");
        }

        // Check degrees
        for (i, &expected_degree) in degree_sequence.iter().enumerate() {
            let actual_degree = graph.degree(&i);
            assert_eq!(actual_degree, expected_degree);
        }
    }

    #[test]
    fn test_simple_configuration_model_errors() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test degree too large for simple graph
        let invalid_degree_sequence = vec![4, 2, 2, 2]; // Node 0 has degree 4, but n=4, so max degree is 3
        assert!(simple_configuration_model(&invalid_degree_sequence, &mut rng, 10).is_err());

        // Test odd degree sum
        let odd_degree_sequence = vec![1, 2, 2]; // Sum = 5 (odd)
        assert!(simple_configuration_model(&odd_degree_sequence, &mut rng, 10).is_err());
    }

    #[test]
    fn test_tree_graph() {
        let mut rng = StdRng::seed_from_u64(42);

        // Test empty tree
        let empty_tree = tree_graph(0, &mut rng).unwrap();
        assert_eq!(empty_tree.node_count(), 0);
        assert_eq!(empty_tree.edge_count(), 0);

        // Test single node tree
        let single_tree = tree_graph(1, &mut rng).unwrap();
        assert_eq!(single_tree.node_count(), 1);
        assert_eq!(single_tree.edge_count(), 0);

        // Test tree with multiple nodes
        let tree = tree_graph(5, &mut rng).unwrap();
        assert_eq!(tree.node_count(), 5);
        assert_eq!(tree.edge_count(), 4); // n-1 edges for a tree

        // Verify all nodes are present
        for i in 0..5 {
            assert!(tree.has_node(&i));
        }
    }

    #[test]
    fn test_random_spanning_tree() {
        let mut rng = StdRng::seed_from_u64(42);

        // Create a complete graph
        let complete = complete_graph(4).unwrap();

        // Generate spanning tree
        let spanning_tree = random_spanning_tree(&complete, &mut rng).unwrap();

        assert_eq!(spanning_tree.node_count(), 4);
        assert_eq!(spanning_tree.edge_count(), 3); // n-1 edges for spanning tree

        // Verify all nodes are present
        for i in 0..4 {
            assert!(spanning_tree.has_node(&i));
        }
    }

    #[test]
    fn test_forest_graph() {
        let mut rng = StdRng::seed_from_u64(42);

        // Create forest with trees of sizes [3, 2, 4]
        let tree_sizes = vec![3, 2, 4];
        let forest = forest_graph(&tree_sizes, &tree_sizes, &mut rng).unwrap();

        assert_eq!(forest.node_count(), 9); // 3 + 2 + 4 = 9 nodes
        assert_eq!(forest.edge_count(), 6); // (3-1) + (2-1) + (4-1) = 6 edges

        // Verify all nodes are present
        for i in 0..9 {
            assert!(forest.has_node(&i));
        }

        // Test empty forest
        let empty_forest = forest_graph(&[], &[], &mut rng).unwrap();
        assert_eq!(empty_forest.node_count(), 0);
        assert_eq!(empty_forest.edge_count(), 0);

        // Test forest with empty trees
        let forest_with_zeros = forest_graph(&[0, 3, 0, 2], &[0, 3, 0, 2], &mut rng).unwrap();
        assert_eq!(forest_with_zeros.node_count(), 5); // 3 + 2 = 5 nodes
        assert_eq!(forest_with_zeros.edge_count(), 3); // (3-1) + (2-1) = 3 edges
    }
}
