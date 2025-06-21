//! Graph transformation algorithms
//!
//! This module provides algorithms for transforming graphs into different
//! representations and extracting subgraphs.

use crate::base::{DiGraph, EdgeWeight, Graph, IndexType, Node};
use std::collections::HashSet;

/// Creates a line graph from the input graph
///
/// In a line graph, each edge of the original graph becomes a vertex,
/// and two vertices in the line graph are connected if the corresponding
/// edges in the original graph share a common vertex.
///
/// # Arguments
/// * `graph` - The input graph
///
/// # Returns
/// * A new graph representing the line graph
pub fn line_graph<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Graph<(N, N), (), Ix>
where
    N: Node + Clone,
    E: EdgeWeight + Clone,
    Ix: IndexType,
{
    let mut line_graph = Graph::new();
    let edges = graph.edges();

    // Each edge becomes a node in the line graph
    let edge_nodes: Vec<(N, N)> = edges
        .iter()
        .map(|e| (e.source.clone(), e.target.clone()))
        .collect();

    // Add all edge nodes to the line graph
    for edge_node in &edge_nodes {
        line_graph.add_node(edge_node.clone());
    }

    // Connect edges that share a vertex
    for (i, edge1) in edge_nodes.iter().enumerate() {
        for (_j, edge2) in edge_nodes.iter().enumerate().skip(i + 1) {
            // Check if edges share a vertex
            if edge1.0 == edge2.0 || edge1.0 == edge2.1 || edge1.1 == edge2.0 || edge1.1 == edge2.1
            {
                // Connect the corresponding nodes in the line graph
                let _ = line_graph.add_edge(edge1.clone(), edge2.clone(), ());
            }
        }
    }

    line_graph
}

/// Creates a line graph from a directed graph
///
/// For directed graphs, two vertices in the line graph are connected
/// if the head of one edge equals the tail of another.
pub fn line_digraph<N, E, Ix>(digraph: &DiGraph<N, E, Ix>) -> DiGraph<(N, N), (), Ix>
where
    N: Node + Clone,
    E: EdgeWeight + Clone,
    Ix: IndexType,
{
    let mut line_digraph = DiGraph::new();
    let edges = digraph.edges();

    // Each edge becomes a node in the line graph
    let edge_nodes: Vec<(N, N)> = edges
        .iter()
        .map(|e| (e.source.clone(), e.target.clone()))
        .collect();

    // Add all edge nodes to the line graph
    for edge_node in &edge_nodes {
        line_digraph.add_node(edge_node.clone());
    }

    // Connect edges where head of first equals tail of second
    for edge1 in &edge_nodes {
        for edge2 in &edge_nodes {
            if edge1 != edge2 && edge1.1 == edge2.0 {
                // Head of edge1 equals tail of edge2
                let _ = line_digraph.add_edge(edge1.clone(), edge2.clone(), ());
            }
        }
    }

    line_digraph
}

/// Extracts a subgraph containing only the specified nodes
///
/// # Arguments
/// * `graph` - The input graph
/// * `nodes` - The set of nodes to include in the subgraph
///
/// # Returns
/// * A new graph containing only the specified nodes and edges between them
pub fn subgraph<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &HashSet<N>) -> Graph<N, E, Ix>
where
    N: Node + Clone,
    E: EdgeWeight + Clone,
    Ix: IndexType,
{
    let mut sub = Graph::new();

    // Add specified nodes
    for node in nodes {
        if graph.has_node(node) {
            sub.add_node(node.clone());
        }
    }

    // Add edges between included nodes
    for edge in graph.edges() {
        if nodes.contains(&edge.source) && nodes.contains(&edge.target) {
            let _ = sub.add_edge(
                edge.source.clone(),
                edge.target.clone(),
                edge.weight.clone(),
            );
        }
    }

    sub
}

/// Extracts a subgraph from a directed graph
pub fn subdigraph<N, E, Ix>(digraph: &DiGraph<N, E, Ix>, nodes: &HashSet<N>) -> DiGraph<N, E, Ix>
where
    N: Node + Clone,
    E: EdgeWeight + Clone,
    Ix: IndexType,
{
    let mut sub = DiGraph::new();

    // Add specified nodes
    for node in nodes {
        if digraph.has_node(node) {
            sub.add_node(node.clone());
        }
    }

    // Add edges between included nodes
    for edge in digraph.edges() {
        if nodes.contains(&edge.source) && nodes.contains(&edge.target) {
            let _ = sub.add_edge(
                edge.source.clone(),
                edge.target.clone(),
                edge.weight.clone(),
            );
        }
    }

    sub
}

/// Extracts an edge-induced subgraph
///
/// Creates a subgraph containing the specified edges and their endpoints.
///
/// # Arguments
/// * `graph` - The input graph
/// * `edges` - The set of edges to include
///
/// # Returns
/// * A new graph containing the specified edges and their endpoints
pub fn edge_subgraph<N, E, Ix>(graph: &Graph<N, E, Ix>, edges: &[(N, N)]) -> Graph<N, E, Ix>
where
    N: Node + Clone,
    E: EdgeWeight + Clone,
    Ix: IndexType,
{
    let mut sub = Graph::new();
    let mut included_nodes = HashSet::new();

    // Collect all nodes from the specified edges
    for (u, v) in edges {
        included_nodes.insert(u.clone());
        included_nodes.insert(v.clone());
    }

    // Add nodes
    for node in &included_nodes {
        if graph.has_node(node) {
            sub.add_node(node.clone());
        }
    }

    // Add specified edges
    for (u, v) in edges {
        if graph.has_edge(u, v) {
            if let Ok(weight) = graph.edge_weight(u, v) {
                let _ = sub.add_edge(u.clone(), v.clone(), weight);
            }
        }
    }

    sub
}

/// Computes the Cartesian product of two graphs
///
/// The Cartesian product G □ H has vertex set V(G) × V(H) and
/// edge set {((u₁,v₁),(u₂,v₂)) : (u₁=u₂ and (v₁,v₂) ∈ E(H)) or (v₁=v₂ and (u₁,u₂) ∈ E(G))}.
///
/// # Arguments
/// * `graph1` - First input graph
/// * `graph2` - Second input graph
///
/// # Returns
/// * The Cartesian product graph
pub fn cartesian_product<N1, N2, E1, E2, Ix>(
    graph1: &Graph<N1, E1, Ix>,
    graph2: &Graph<N2, E2, Ix>,
) -> Graph<(N1, N2), (), Ix>
where
    N1: Node + Clone,
    N2: Node + Clone,
    E1: EdgeWeight,
    E2: EdgeWeight,
    Ix: IndexType,
{
    let mut product = Graph::new();

    let nodes1: Vec<N1> = graph1.nodes().into_iter().cloned().collect();
    let nodes2: Vec<N2> = graph2.nodes().into_iter().cloned().collect();

    // Add all combinations of nodes
    for n1 in &nodes1 {
        for n2 in &nodes2 {
            product.add_node((n1.clone(), n2.clone()));
        }
    }

    // Add edges according to Cartesian product rules
    for n1 in &nodes1 {
        for n2 in &nodes2 {
            // Connect (n1, n2) to (n1, m2) if (n2, m2) is an edge in graph2
            if let Ok(neighbors2) = graph2.neighbors(n2) {
                for m2 in neighbors2 {
                    if n2 != &m2 {
                        // Avoid self-loops from undirected graph
                        let _ = product.add_edge((n1.clone(), n2.clone()), (n1.clone(), m2), ());
                    }
                }
            }

            // Connect (n1, n2) to (m1, n2) if (n1, m1) is an edge in graph1
            if let Ok(neighbors1) = graph1.neighbors(n1) {
                for m1 in neighbors1 {
                    if n1 != &m1 {
                        // Avoid self-loops from undirected graph
                        let _ = product.add_edge((n1.clone(), n2.clone()), (m1, n2.clone()), ());
                    }
                }
            }
        }
    }

    product
}

/// Computes the tensor product (Kronecker product) of two graphs
///
/// The tensor product G ⊗ H has vertex set V(G) × V(H) and
/// edge set {((u₁,v₁),(u₂,v₂)) : (u₁,u₂) ∈ E(G) and (v₁,v₂) ∈ E(H)}.
pub fn tensor_product<N1, N2, E1, E2, Ix>(
    graph1: &Graph<N1, E1, Ix>,
    graph2: &Graph<N2, E2, Ix>,
) -> Graph<(N1, N2), (), Ix>
where
    N1: Node + Clone,
    N2: Node + Clone,
    E1: EdgeWeight,
    E2: EdgeWeight,
    Ix: IndexType,
{
    let mut product = Graph::new();

    let nodes1: Vec<N1> = graph1.nodes().into_iter().cloned().collect();
    let nodes2: Vec<N2> = graph2.nodes().into_iter().cloned().collect();

    // Add all combinations of nodes
    for n1 in &nodes1 {
        for n2 in &nodes2 {
            product.add_node((n1.clone(), n2.clone()));
        }
    }

    // Add edges according to tensor product rules
    for n1 in &nodes1 {
        for n2 in &nodes2 {
            if let Ok(neighbors1) = graph1.neighbors(n1) {
                if let Ok(neighbors2) = graph2.neighbors(n2) {
                    for m1 in neighbors1 {
                        for m2 in &neighbors2 {
                            if n1 != &m1 && n2 != m2 {
                                // Avoid self-loops
                                let _ = product.add_edge(
                                    (n1.clone(), n2.clone()),
                                    (m1.clone(), m2.clone()),
                                    (),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    product
}

/// Computes the complement of a graph
///
/// The complement G̅ of a graph G has the same vertex set as G,
/// but edge (u,v) is in G̅ if and only if (u,v) is not in G.
pub fn complement<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Graph<N, (), Ix>
where
    N: Node + Clone,
    E: EdgeWeight,
    Ix: IndexType,
{
    let mut comp = Graph::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // Add all nodes
    for node in &nodes {
        comp.add_node(node.clone());
    }

    // Add edges that are NOT in the original graph
    for (i, u) in nodes.iter().enumerate() {
        for v in nodes.iter().skip(i + 1) {
            if !graph.has_edge(u, v) {
                let _ = comp.add_edge(u.clone(), v.clone(), ());
            }
        }
    }

    comp
}

/// Creates a spanning subgraph with only edges of specified weights
///
/// # Arguments
/// * `graph` - The input graph
/// * `valid_weights` - Set of edge weights to include
///
/// # Returns
/// * A subgraph containing only edges with specified weights
pub fn weight_filtered_subgraph<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    valid_weights: &HashSet<E>,
) -> Graph<N, E, Ix>
where
    N: Node + Clone,
    E: EdgeWeight + Clone + std::hash::Hash + Eq,
    Ix: IndexType,
{
    let mut filtered = Graph::new();

    // Add all nodes
    for node in graph.nodes() {
        filtered.add_node(node.clone());
    }

    // Add edges with valid weights
    for edge in graph.edges() {
        if valid_weights.contains(&edge.weight) {
            let _ = filtered.add_edge(
                edge.source.clone(),
                edge.target.clone(),
                edge.weight.clone(),
            );
        }
    }

    filtered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::create_graph;

    #[test]
    fn test_line_graph() {
        let mut graph = create_graph::<&str, ()>();
        graph.add_edge("A", "B", ()).unwrap();
        graph.add_edge("B", "C", ()).unwrap();
        graph.add_edge("C", "A", ()).unwrap();

        let line = line_graph(&graph);

        // Line graph of triangle should have 3 nodes (one per edge)
        assert_eq!(line.node_count(), 3);

        // Each pair of edges shares a vertex, so line graph should have 3 edges
        assert_eq!(line.edge_count(), 3);
    }

    #[test]
    fn test_subgraph() {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(1, 2, ()).unwrap();
        graph.add_edge(2, 3, ()).unwrap();
        graph.add_edge(3, 4, ()).unwrap();
        graph.add_edge(1, 4, ()).unwrap();

        let mut nodes = HashSet::new();
        nodes.insert(1);
        nodes.insert(2);
        nodes.insert(3);

        let sub = subgraph(&graph, &nodes);

        // Should have 3 nodes
        assert_eq!(sub.node_count(), 3);

        // Should have 2 edges: (1,2) and (2,3)
        assert_eq!(sub.edge_count(), 2);
        assert!(sub.has_edge(&1, &2));
        assert!(sub.has_edge(&2, &3));
        assert!(!sub.has_edge(&3, &4)); // Not included since 4 is not in subgraph
    }

    #[test]
    fn test_edge_subgraph() {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(1, 2, ()).unwrap();
        graph.add_edge(2, 3, ()).unwrap();
        graph.add_edge(3, 4, ()).unwrap();
        graph.add_edge(1, 4, ()).unwrap();

        let edges = vec![(1, 2), (3, 4)];
        let sub = edge_subgraph(&graph, &edges);

        // Should have 4 nodes (endpoints of selected edges)
        assert_eq!(sub.node_count(), 4);

        // Should have 2 edges
        assert_eq!(sub.edge_count(), 2);
        assert!(sub.has_edge(&1, &2));
        assert!(sub.has_edge(&3, &4));
        assert!(!sub.has_edge(&2, &3));
        assert!(!sub.has_edge(&1, &4));
    }

    #[test]
    fn test_cartesian_product() {
        // Create K2 (complete graph on 2 vertices)
        let mut k2 = create_graph::<i32, ()>();
        k2.add_edge(1, 2, ()).unwrap();

        // Create P2 (path graph on 2 vertices)
        let mut p2 = create_graph::<char, ()>();
        p2.add_edge('A', 'B', ()).unwrap();

        let product = cartesian_product(&k2, &p2);

        // Should have 4 nodes: (1,A), (1,B), (2,A), (2,B)
        assert_eq!(product.node_count(), 4);

        // Cartesian product K2 □ P2 creates a 4-cycle with 8 directed edges in undirected representation
        // Since we're using undirected graph, edges are bidirectional
        assert_eq!(product.edge_count(), 8);
    }

    #[test]
    fn test_tensor_product() {
        // Create K2
        let mut k2_1 = create_graph::<i32, ()>();
        k2_1.add_edge(1, 2, ()).unwrap();

        // Create another K2
        let mut k2_2 = create_graph::<char, ()>();
        k2_2.add_edge('A', 'B', ()).unwrap();

        let product = tensor_product(&k2_1, &k2_2);

        // Should have 4 nodes
        assert_eq!(product.node_count(), 4);

        // Tensor product of K2 ⊗ K2 creates 4 edges since each direction is counted
        // ((1,A),(2,B)), ((1,B),(2,A)), ((2,A),(1,B)), ((2,B),(1,A))
        assert_eq!(product.edge_count(), 4);
    }

    #[test]
    fn test_complement() {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(1, 2, ()).unwrap();
        graph.add_edge(2, 3, ()).unwrap();

        let comp = complement(&graph);

        // Should have same number of nodes
        assert_eq!(comp.node_count(), 3);

        // Complement should have edge (1,3) but not (1,2) or (2,3)
        assert_eq!(comp.edge_count(), 1);
        assert!(comp.has_edge(&1, &3));
        assert!(!comp.has_edge(&1, &2));
        assert!(!comp.has_edge(&2, &3));
    }

    #[test]
    fn test_weight_filtered_subgraph() {
        let mut graph = create_graph::<i32, i32>();
        graph.add_edge(1, 2, 10).unwrap();
        graph.add_edge(2, 3, 20).unwrap();
        graph.add_edge(3, 4, 10).unwrap();

        let mut valid_weights = HashSet::new();
        valid_weights.insert(10);

        let filtered = weight_filtered_subgraph(&graph, &valid_weights);

        // Should have all 4 nodes
        assert_eq!(filtered.node_count(), 4);

        // Should have 2 edges with weight 10
        assert_eq!(filtered.edge_count(), 2);
        assert!(filtered.has_edge(&1, &2));
        assert!(filtered.has_edge(&3, &4));
        assert!(!filtered.has_edge(&2, &3)); // Has weight 20
    }
}
