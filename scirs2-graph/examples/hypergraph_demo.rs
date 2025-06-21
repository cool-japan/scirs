//! Demonstration of hypergraph functionality
//!
//! This example shows how to create and manipulate hypergraphs,
//! which are generalizations of graphs where edges can connect
//! any number of vertices.

use scirs2_graph::{Hypergraph, Result};

fn main() -> Result<()> {
    println!("=== Hypergraph Demo ===\n");

    // Create a new hypergraph
    let mut hypergraph: Hypergraph<&str, f64> = Hypergraph::new();

    println!("1. Creating hypergraph and adding nodes:");
    // Add individual nodes (optional - they're added automatically with hyperedges)
    hypergraph.add_node("Alice");
    hypergraph.add_node("Bob");
    hypergraph.add_node("Charlie");
    hypergraph.add_node("Diana");
    hypergraph.add_node("Eve");

    println!(
        "   Initial nodes: {:?}",
        hypergraph.nodes().collect::<Vec<_>>()
    );
    println!("   Node count: {}", hypergraph.node_count());

    println!("\n2. Adding hyperedges:");

    // Add a 3-way collaboration hyperedge
    let he1 = hypergraph.add_hyperedge_from_vec(vec!["Alice", "Bob", "Charlie"], 1.0)?;
    println!(
        "   Added collaboration hyperedge {}: Alice-Bob-Charlie (weight: 1.0)",
        he1
    );

    // Add a 2-way friendship hyperedge (like a regular edge)
    let he2 = hypergraph.add_hyperedge_from_vec(vec!["Alice", "Diana"], 0.8)?;
    println!(
        "   Added friendship hyperedge {}: Alice-Diana (weight: 0.8)",
        he2
    );

    // Add a 4-way meeting hyperedge
    let he3 = hypergraph.add_hyperedge_from_vec(vec!["Bob", "Charlie", "Diana", "Eve"], 1.5)?;
    println!(
        "   Added meeting hyperedge {}: Bob-Charlie-Diana-Eve (weight: 1.5)",
        he3
    );

    println!("\n3. Hypergraph statistics:");
    println!("   Total nodes: {}", hypergraph.node_count());
    println!("   Total hyperedges: {}", hypergraph.hyperedge_count());

    let (min_size, max_size, avg_size) = hypergraph.hyperedge_size_stats();
    println!(
        "   Hyperedge sizes - Min: {}, Max: {}, Avg: {:.2}",
        min_size, max_size, avg_size
    );
    println!("   Is uniform: {}", hypergraph.is_uniform());

    println!("\n4. Analyzing connections:");

    // Check neighbors for each person
    for person in &["Alice", "Bob", "Charlie", "Diana", "Eve"] {
        let neighbors = hypergraph.neighbors(person);
        let degree = hypergraph.degree(person);
        println!(
            "   {}: degree = {}, neighbors = {:?}",
            person,
            degree,
            neighbors.iter().collect::<Vec<_>>()
        );
    }

    println!("\n5. Connectivity analysis:");

    // Check if specific pairs are connected
    let pairs = [("Alice", "Bob"), ("Alice", "Eve"), ("Diana", "Charlie")];
    for (person1, person2) in pairs.iter() {
        let connected = hypergraph.are_connected(person1, person2);
        let connecting_edges = hypergraph.connecting_hyperedges(person1, person2);
        println!(
            "   {} and {} connected: {} (via hyperedges: {:?})",
            person1, person2, connected, connecting_edges
        );
    }

    println!("\n6. Hyperedge details:");

    for hyperedge in hypergraph.hyperedges() {
        println!(
            "   Hyperedge {}: nodes = {:?}, weight = {}, size = {}",
            hyperedge.id,
            hyperedge.nodes.iter().collect::<Vec<_>>(),
            hyperedge.weight,
            hyperedge.nodes.len()
        );
    }

    println!("\n7. Converting to regular graph (2-section):");

    // Convert hypergraph to regular graph by connecting all pairs within each hyperedge
    let regular_graph = hypergraph.to_graph();
    println!("   Regular graph nodes: {}", regular_graph.node_count());
    println!("   Regular graph edges: {}", regular_graph.edge_count());

    // Show the edges in the regular graph
    println!("   Edges in 2-section:");
    for edge in regular_graph.edges() {
        println!(
            "     {} -- {} (weight: {})",
            edge.source, edge.target, edge.weight
        );
    }

    println!("\n8. Incidence matrix:");

    let incidence = hypergraph.incidence_matrix();
    println!("   Incidence matrix shape: {:?}", incidence.shape());
    println!("   (rows = nodes, columns = hyperedges)");

    // Print a simplified view of the incidence matrix
    let nodes: Vec<_> = hypergraph.nodes().cloned().collect();
    let mut sorted_nodes = nodes;
    sorted_nodes.sort();

    for (i, node) in sorted_nodes.iter().enumerate() {
        print!("   {}: ", node);
        for j in 0..incidence.shape()[1] {
            print!("{} ", incidence[[i, j]]);
        }
        println!();
    }

    println!("\n9. Maximal cliques:");

    let cliques = hypergraph.maximal_cliques();
    for (i, clique) in cliques.iter().enumerate() {
        println!(
            "   Clique {}: {:?}",
            i + 1,
            clique.iter().collect::<Vec<_>>()
        );
    }

    println!("\n10. Removing a hyperedge:");

    println!("   Removing hyperedge {} (Alice-Diana friendship)", he2);
    let removed = hypergraph.remove_hyperedge(he2)?;
    println!(
        "   Removed: {:?} with weight {}",
        removed.nodes.iter().collect::<Vec<_>>(),
        removed.weight
    );

    println!(
        "   Are Alice and Diana still connected? {}",
        hypergraph.are_connected(&"Alice", &"Diana")
    );
    println!("   Remaining hyperedges: {}", hypergraph.hyperedge_count());

    println!("\n=== Demo Complete ===");
    Ok(())
}
