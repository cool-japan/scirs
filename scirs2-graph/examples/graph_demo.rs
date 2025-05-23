//! Comprehensive example demonstrating scirs2-graph capabilities

use scirs2_graph::{
    barabasi_albert_graph,
    breadth_first_search,
    center_nodes,
    // Centrality
    centrality,
    circular_layout,
    // Measures
    clustering_coefficient,
    connected_components,
    cosine_similarity,
    depth_first_search,
    diameter,
    // Generators
    erdos_renyi_graph,
    find_motifs,
    graph_density,
    hits_algorithm,
    // Graph properties
    is_bipartite,
    // Similarity
    jaccard_similarity,
    k_core_decomposition,
    label_propagation,
    // Community detection
    louvain_communities,
    max_flow,
    minimum_spanning_tree,
    pagerank_centrality,
    personalized_pagerank,
    radius,
    // Random walks
    random_walk,
    // Algorithms
    shortest_path,
    // Layout
    spring_layout,
    strongly_connected_components,
    watts_strogatz_graph,
    CentralityType,
    DiGraph,
    Graph,
    MotifType,
};
use std::collections::HashMap;

fn main() {
    println!("=== SciRS2 Graph Module Demo ===\n");

    // 1. Basic graph creation and operations
    println!("1. Creating a simple graph:");
    let mut graph: Graph<&str, f64> = Graph::new();

    // Add edges (nodes are created automatically)
    graph.add_edge("Alice", "Bob", 1.0).unwrap();
    graph.add_edge("Bob", "Charlie", 2.0).unwrap();
    graph.add_edge("Charlie", "David", 1.5).unwrap();
    graph.add_edge("David", "Alice", 3.0).unwrap();
    graph.add_edge("Bob", "David", 0.5).unwrap();

    println!(
        "   Graph has {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // 2. Shortest path algorithms
    println!("\n2. Shortest path from Alice to Charlie:");
    match shortest_path(&graph, &"Alice", &"Charlie") {
        Ok((distance, path)) => {
            println!("   Distance: {}", distance);
            println!("   Path: {:?}", path);
        }
        Err(e) => println!("   Error: {:?}", e),
    }

    // 3. Graph traversal
    println!("\n3. Breadth-first search from Alice:");
    let bfs_result = breadth_first_search(&graph, &"Alice").unwrap();
    println!("   Visited order: {:?}", bfs_result);

    // 4. Centrality measures
    println!("\n4. Centrality measures:");
    let degree_centrality = centrality(&graph, CentralityType::Degree).unwrap();
    println!("   Degree centrality:");
    for (node, score) in &degree_centrality {
        println!("     {}: {:.3}", node, score);
    }

    // 5. Community detection
    println!("\n5. Community detection (Louvain method):");
    let communities = louvain_communities(&graph, 1.0, 1e-6, 10).unwrap();
    println!("   Found {} communities", communities.num_communities());
    println!("   Modularity: {:.3}", communities.modularity);

    // 6. Graph properties
    println!("\n6. Graph properties:");
    match diameter(&graph) {
        Some(d) => println!("   Diameter: {}", d),
        None => println!("   Graph is disconnected"),
    }
    match radius(&graph) {
        Some(r) => println!("   Radius: {}", r),
        None => println!("   Graph is disconnected"),
    }

    let density = graph_density(&graph).unwrap();
    println!("   Density: {:.3}", density);

    // 7. Bipartite check
    println!("\n7. Bipartite check:");
    match is_bipartite(&graph) {
        Ok(result) => match result {
            scirs2_graph::BipartiteResult::Bipartite { set_a, set_b } => {
                println!("   Graph is bipartite!");
                println!("   Set A: {:?}", set_a);
                println!("   Set B: {:?}", set_b);
            }
            scirs2_graph::BipartiteResult::NotBipartite => {
                println!("   Graph is not bipartite");
            }
        },
        Err(e) => println!("   Error: {:?}", e),
    }

    // 8. K-core decomposition
    println!("\n8. K-core decomposition:");
    let k_cores = k_core_decomposition(&graph);
    for (node, core_num) in &k_cores {
        println!("   {}: {}-core", node, core_num);
    }

    // 9. Random graph generation
    println!("\n9. Generating random graphs:");
    let random_graph = erdos_renyi_graph::<f64>(10, 0.3);
    println!(
        "   Erdős-Rényi graph: {} nodes, {} edges",
        random_graph.node_count(),
        random_graph.edge_count()
    );

    let ba_graph = barabasi_albert_graph::<f64>(20, 2);
    println!(
        "   Barabási-Albert graph: {} nodes, {} edges",
        ba_graph.node_count(),
        ba_graph.edge_count()
    );

    // 10. Graph similarity
    println!("\n10. Node similarity (Jaccard):");
    let similarity = jaccard_similarity(&graph, &"Alice", &"Bob").unwrap();
    println!(
        "   Jaccard similarity between Alice and Bob: {:.3}",
        similarity
    );

    // 11. Random walks
    println!("\n11. Random walk from Alice (10 steps):");
    let walk = random_walk(&graph, &"Alice", 10, 0.1).unwrap();
    println!("   Walk: {:?}", walk);

    // 12. Directed graph example
    println!("\n12. Directed graph example:");
    let mut digraph: DiGraph<&str, f64> = DiGraph::new();

    digraph.add_edge("A", "B", 1.0).unwrap();
    digraph.add_edge("B", "C", 1.0).unwrap();
    digraph.add_edge("C", "D", 1.0).unwrap();
    digraph.add_edge("D", "B", 1.0).unwrap();

    // PageRank on directed graph
    let pagerank = pagerank_centrality(&digraph, 0.85, 1e-6).unwrap();
    println!("   PageRank scores:");
    for (node, score) in &pagerank {
        println!("     {}: {:.3}", node, score);
    }

    // HITS algorithm
    let hits = hits_algorithm(&digraph, 100, 1e-6).unwrap();
    println!("   HITS authority scores:");
    for (node, score) in &hits.authorities {
        println!("     {}: {:.3}", node, score);
    }

    // 13. Motif finding
    println!("\n13. Finding triangles in the graph:");
    let triangles = find_motifs(&graph, MotifType::Triangle);
    println!("   Found {} triangles", triangles.len());
    for (i, triangle) in triangles.iter().enumerate() {
        println!("   Triangle {}: {:?}", i + 1, triangle);
    }

    // 14. Graph layout
    println!("\n14. Computing graph layout:");
    let layout = spring_layout(&graph, 50, 100.0, 100.0);
    println!("   Spring layout positions:");
    for (node, pos) in &layout {
        println!("     {}: ({:.2}, {:.2})", node, pos.x, pos.y);
    }

    println!("\n=== Demo completed! ===");
}
