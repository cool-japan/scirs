//! Demonstration of the Infomap algorithm for community detection
//!
//! This example shows how to use the Infomap algorithm to detect community structure
//! in graphs using information-theoretic principles.

use scirs2_graph::{generators::create_graph, infomap_communities, louvain_communities};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a graph with two clear communities connected by a weak bridge
    let mut graph = create_graph::<&str, f64>();

    // Community 1: Densely connected triangle
    graph.add_edge("A", "B", 1.0)?;
    graph.add_edge("B", "C", 1.0)?;
    graph.add_edge("C", "A", 1.0)?;

    // Community 2: Densely connected triangle
    graph.add_edge("D", "E", 1.0)?;
    graph.add_edge("E", "F", 1.0)?;
    graph.add_edge("F", "D", 1.0)?;

    // Weak bridge between communities
    graph.add_edge("C", "D", 0.1)?;

    println!("Graph Analysis: Communities in a Two-Triangle Network");
    println!("=====================================================");
    println!("Graph structure:");
    println!("  Community 1: A-B-C (triangle)");
    println!("  Community 2: D-E-F (triangle)");
    println!("  Bridge: C-D (weak connection)");
    println!();

    // Run Infomap algorithm
    let infomap_result = infomap_communities(&graph, 100, 1e-6);

    println!("Infomap Results:");
    println!("================");
    println!(
        "Communities found: {}",
        infomap_result
            .node_communities
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    println!(
        "Code length (description cost): {:.6}",
        infomap_result.code_length
    );
    println!("Modularity: {:.6}", infomap_result.modularity);
    println!();

    println!("Node assignments:");
    for (node, community) in &infomap_result.node_communities {
        println!("  Node {}: Community {}", node, community);
    }
    println!();

    // Compare with Louvain method
    let louvain_result = louvain_communities(&graph);

    println!("Louvain Method (for comparison):");
    println!("================================");
    println!(
        "Communities found: {}",
        louvain_result
            .node_communities
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    println!("Modularity: {:.6}", louvain_result.modularity);
    println!();

    println!("Node assignments:");
    for (node, community) in &louvain_result.node_communities {
        println!("  Node {}: Community {}", node, community);
    }
    println!();

    // Analysis
    println!("Algorithm Comparison:");
    println!("====================");
    println!("The Infomap algorithm uses information theory (specifically the map equation)");
    println!("to find communities that minimize the description length of random walks.");
    println!("This often provides better resolution for detecting small communities");
    println!("compared to modularity-based methods like Louvain.");
    println!();
    println!("Lower code length indicates better compression and community structure.");
    println!(
        "Both algorithms should ideally separate the two triangles into distinct communities."
    );

    Ok(())
}
