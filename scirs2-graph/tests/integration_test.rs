//! Integration tests for scirs2-graph
//!
//! These tests verify that the refactored module works correctly

use scirs2_graph::algorithms::*;
use scirs2_graph::generators::create_graph;

#[test]
fn test_basic_graph_operations() {
    let mut graph = create_graph::<i32, f64>();

    // Add nodes and edges
    graph.add_edge(1, 2, 1.0).unwrap();
    graph.add_edge(2, 3, 2.0).unwrap();
    graph.add_edge(3, 4, 1.5).unwrap();
    graph.add_edge(1, 4, 3.0).unwrap();

    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 4);
}

#[test]
fn test_shortest_path_algorithm() {
    let mut graph = create_graph::<i32, f64>();

    graph.add_edge(1, 2, 1.0).unwrap();
    graph.add_edge(2, 3, 2.0).unwrap();
    graph.add_edge(1, 3, 5.0).unwrap();

    if let Ok(Some(path)) = shortest_path(&graph, &1, &3) {
        // Should find path 1->2->3 with total weight 3.0, not direct path 1->3 with weight 5.0
        assert_eq!(path.total_weight, 3.0);
        assert_eq!(path.nodes, vec![1, 2, 3]);
    } else {
        panic!("Expected to find a path");
    }
}

#[test]
fn test_connectivity_algorithms() {
    let mut graph = create_graph::<i32, f64>();

    // Create two disconnected components
    graph.add_edge(1, 2, 1.0).unwrap();
    graph.add_edge(2, 3, 1.0).unwrap();
    graph.add_edge(4, 5, 1.0).unwrap();

    let components = connected_components(&graph);
    assert_eq!(components.len(), 2);
}

#[test]
fn test_minimum_spanning_tree() {
    let mut graph = create_graph::<i32, f64>();

    graph.add_edge(1, 2, 1.0).unwrap();
    graph.add_edge(2, 3, 2.0).unwrap();
    graph.add_edge(1, 3, 3.0).unwrap();
    graph.add_edge(3, 4, 1.0).unwrap();

    let mst = minimum_spanning_tree(&graph).unwrap();

    // MST should have n-1 edges
    assert_eq!(mst.len(), 3);

    // Total weight should be 4.0 (edges: 1-2: 1.0, 2-3: 2.0, 3-4: 1.0)
    let total_weight: f64 = mst.iter().map(|e| e.weight).sum();
    assert_eq!(total_weight, 4.0);
}

#[test]
fn test_pagerank() {
    // Create a simple directed graph for PageRank
    use scirs2_graph::generators::create_digraph;
    let mut digraph = create_digraph::<i32, f64>();

    digraph.add_edge(1, 2, 1.0).unwrap();
    digraph.add_edge(2, 3, 1.0).unwrap();
    digraph.add_edge(3, 1, 1.0).unwrap();

    let pagerank_scores = pagerank(&digraph, 0.85, 1e-6, 100);

    // Should have scores for all nodes
    assert_eq!(pagerank_scores.len(), 3);

    // All scores should be positive
    for score in pagerank_scores.values() {
        assert!(*score > 0.0);
    }
}
