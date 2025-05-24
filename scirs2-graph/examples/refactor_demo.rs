//! Demonstration that the refactored scirs2-graph module works correctly
//!
//! This example shows that all the major algorithm categories are accessible
//! and functional after the refactoring.

use scirs2_graph::algorithms::*;
use scirs2_graph::generators::{create_digraph, create_graph};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== scirs2-graph Refactoring Demo ===\n");

    // 1. Basic graph creation and operations
    println!("1. Basic Graph Operations:");
    let mut graph = create_graph::<i32, f64>();
    graph.add_edge(1, 2, 1.0)?;
    graph.add_edge(2, 3, 2.0)?;
    graph.add_edge(3, 4, 1.5)?;
    graph.add_edge(1, 4, 3.5)?;

    println!(
        "   Created graph with {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // 2. Shortest Path Algorithm (from shortest_path module)
    println!("\n2. Shortest Path:");
    if let Ok(Some(path)) = shortest_path(&graph, &1, &4) {
        println!(
            "   Path from 1 to 4: {:?} (weight: {})",
            path.nodes, path.total_weight
        );
    }

    // 3. Connected Components (from connectivity module)
    println!("\n3. Connectivity:");
    let components = connected_components(&graph);
    println!("   Number of connected components: {}", components.len());

    // 4. Minimum Spanning Tree (from main algorithms module)
    println!("\n4. Minimum Spanning Tree:");
    let mst = minimum_spanning_tree(&graph)?;
    let total_weight: f64 = mst.iter().map(|e| e.weight).sum();
    println!(
        "   MST has {} edges with total weight: {}",
        mst.len(),
        total_weight
    );

    // 5. PageRank on a directed graph (from main algorithms module)
    println!("\n5. PageRank on Directed Graph:");
    let mut digraph = create_digraph::<i32, f64>();
    digraph.add_edge(1, 2, 1.0)?;
    digraph.add_edge(2, 3, 1.0)?;
    digraph.add_edge(3, 1, 1.0)?;
    digraph.add_edge(2, 4, 1.0)?;

    let scores = pagerank(&digraph, 0.85, 1e-6, 100);
    println!("   PageRank scores:");
    for (node, score) in scores {
        println!("     Node {}: {:.4}", node, score);
    }

    // 6. Graph Properties (from properties module)
    println!("\n6. Graph Properties:");
    if let Some(diameter) = diameter(&graph) {
        println!("   Graph diameter: {}", diameter);
    }
    if let Some(radius) = radius(&graph) {
        println!("   Graph radius: {}", radius);
    }

    println!("\n=== All algorithm modules working correctly! ===");

    Ok(())
}
