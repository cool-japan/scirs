//! Comprehensive example demonstrating scirs2-graph capabilities

use scirs2_graph::algorithms::*;
use scirs2_graph::generators::*;

fn main() {
    println!("=== SciRS2 Graph Module Demo ===\n");

    // 1. Creating a simple social network graph
    println!("1. Creating a Social Network Graph");
    let mut graph = create_graph::<&str, f64>();

    // Add nodes (people) and edges (friendships with weights as interaction strength)
    graph.add_edge("Alice", "Bob", 0.8).unwrap();
    graph.add_edge("Bob", "Charlie", 0.6).unwrap();
    graph.add_edge("Charlie", "David", 0.7).unwrap();
    graph.add_edge("David", "Alice", 0.5).unwrap();
    graph.add_edge("Alice", "Charlie", 0.3).unwrap();

    println!(
        "   Created graph with {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // 2. Basic traversal algorithms
    println!("\n2. Graph Traversal");

    // Breadth-first search
    let bfs_result = breadth_first_search(&graph, &"Alice").unwrap();
    println!("   BFS from Alice: {} nodes visited", bfs_result.len());

    // Depth-first search
    let dfs_result = depth_first_search(&graph, &"Alice").unwrap();
    println!("   DFS from Alice: {} nodes visited", dfs_result.len());

    // 3. Shortest path algorithms
    println!("\n3. Shortest Path Algorithms");
    match shortest_path(&graph, &"Alice", &"Charlie") {
        Ok(Some(path)) => {
            println!("   Shortest path from Alice to Charlie:");
            println!("   Path: {:?}", path.nodes);
            println!("   Total weight: {:.2}", path.total_weight);
        }
        Ok(None) => println!("   No path found"),
        Err(e) => println!("   Error: {:?}", e),
    }

    // 4. Connectivity analysis
    println!("\n4. Connectivity Analysis");
    let components = connected_components(&graph);
    println!("   Number of connected components: {}", components.len());

    // Check for articulation points (critical nodes)
    let articulation_pts = articulation_points(&graph);
    println!("   Articulation points: {:?}", articulation_pts);

    // 5. Community detection
    println!("\n5. Community Detection");
    let communities = louvain_communities(&graph);
    println!("   Modularity: {:.4}", communities.modularity);
    println!("   Community assignments:");
    for (node, community) in &communities.node_communities {
        println!("     {} belongs to community {}", node, community);
    }

    // 6. Graph properties
    println!("\n6. Graph Properties");

    // Diameter and radius
    if let Some(d) = diameter(&graph) {
        println!("   Graph diameter: {:.2}", d);
    }
    if let Some(r) = radius(&graph) {
        println!("   Graph radius: {:.2}", r);
    }

    // Center nodes
    let centers = center_nodes(&graph);
    println!("   Center nodes: {:?}", centers);

    // 7. Minimum spanning tree
    println!("\n7. Minimum Spanning Tree");
    match minimum_spanning_tree(&graph) {
        Ok(mst) => {
            let total_weight: f64 = mst.iter().map(|e| e.weight).sum();
            println!("   MST edges: {}", mst.len());
            println!("   Total weight: {:.2}", total_weight);
        }
        Err(e) => println!("   Error: {:?}", e),
    }

    // 8. Random graph generation
    println!("\n8. Random Graph Generation");

    let mut rng = rand::rng();

    // Erdős-Rényi random graph
    let random_graph = erdos_renyi_graph(10, 0.3, &mut rng).unwrap();
    println!(
        "   Erdős-Rényi graph: {} nodes, {} edges",
        random_graph.node_count(),
        random_graph.edge_count()
    );

    // Barabási-Albert preferential attachment
    let ba_graph = barabasi_albert_graph(20, 2, &mut rng).unwrap();
    println!(
        "   Barabási-Albert graph: {} nodes, {} edges",
        ba_graph.node_count(),
        ba_graph.edge_count()
    );

    // 9. Directed graph operations
    println!("\n9. Directed Graph Operations");
    let mut digraph = create_digraph::<&str, f64>();

    // Create a directed graph with cycles
    digraph.add_edge("A", "B", 1.0).unwrap();
    digraph.add_edge("B", "C", 1.0).unwrap();
    digraph.add_edge("C", "A", 1.0).unwrap();
    digraph.add_edge("B", "D", 1.0).unwrap();

    // Topological sort (will fail due to cycle)
    match topological_sort(&digraph) {
        Ok(sorted) => println!("   Topological order: {:?}", sorted),
        Err(_) => println!("   Graph has cycles, cannot perform topological sort"),
    }

    // PageRank
    let pagerank = pagerank(&digraph, 0.85, 1e-6, 100);
    println!("   PageRank scores:");
    for (node, score) in pagerank {
        println!("     {}: {:.4}", node, score);
    }

    // 10. K-core decomposition
    println!("\n10. K-Core Decomposition");
    let k_cores = k_core_decomposition(&graph);
    println!("   K-core values:");
    for (node, k) in k_cores {
        println!("     {}: {}-core", node, k);
    }

    println!("\n=== Demo Complete ===");
}
