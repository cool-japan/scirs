//! Stress tests for large graphs (>1M nodes)
//!
//! These tests verify that scirs2-graph can handle production-scale graphs
//! with millions of nodes and edges.

use rand::{rng, Rng};
use scirs2_core::error::CoreResult;
use scirs2_graph::{algorithms, generators, measures, DiGraph, Graph};
use std::time::Instant;

#[test]
#[ignore] // Run with: cargo test stress_tests -- --ignored --test-threads=1
#[allow(dead_code)]
fn test_large_erdos_renyi_graph() -> CoreResult<()> {
    println!("\n=== Erdős-Rényi Graph Stress Test ===");

    let sizes = vec![100_000, 500_000, 1_000_000];
    let edge_probability = 0.00001; // Keep sparse to avoid memory issues

    for n in sizes {
        println!("\nTesting with {n} nodes");
        let start = Instant::now();

        // Generate graph
        let gen_start = Instant::now();
        let mut rng = rand::rng();
        let graph = generators::erdos_renyi_graph(n, edge_probability, &mut rng)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        let gen_time = gen_start.elapsed();

        println!("  Generation time: {:.2}s", gen_time.as_secs_f64());
        println!("  Nodes: {}", graph.node_count());
        println!("  Edges: {}", graph.edge_count());
        let density = measures::graph_density(&graph)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        println!("  Density: {density:.6}");

        // Test basic operations
        let ops_start = Instant::now();

        // Degree calculation
        let max_degree = (0..n).map(|i| graph.degree(&i)).max().unwrap_or(0);
        println!("  Max degree: {max_degree}");

        // Connected components (sample for very large graphs)
        if n <= 500_000 {
            let cc_start = Instant::now();
            let components = algorithms::connected_components(&graph);
            println!(
                "  Connected components: {} ({:.2}s)",
                components.len(),
                cc_start.elapsed().as_secs_f64()
            );
        }

        println!(
            "  Operations time: {:.2}s",
            ops_start.elapsed().as_secs_f64()
        );
        println!("  Total time: {:.2}s", start.elapsed().as_secs_f64());
    }

    Ok(())
}

#[test]
#[ignore]
#[allow(dead_code)]
fn test_large_barabasi_albert_graph() -> CoreResult<()> {
    println!("\n=== Barabási-Albert Graph Stress Test ===");

    let sizes = vec![100_000, 500_000, 1_000_000];
    let m = 3; // Number of edges to attach from a new node

    for n in sizes {
        println!("\nTesting with {n} nodes");
        let start = Instant::now();

        // Generate graph
        let gen_start = Instant::now();
        let mut rng = rand::rng();
        let graph = generators::barabasi_albert_graph(n, m, &mut rng)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        let gen_time = gen_start.elapsed();

        println!("  Generation time: {:.2}s", gen_time.as_secs_f64());
        println!("  Nodes: {}", graph.node_count());
        println!("  Edges: {}", graph.edge_count());

        // Test algorithms
        let ops_start = Instant::now();

        // Degree distribution
        let degrees: Vec<usize> = (0..graph.node_count()).map(|i| graph.degree(&i)).collect();
        let max_degree = degrees.iter().max().unwrap_or(&0);
        let avg_degree = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;
        println!("  Max degree: {max_degree}");
        println!("  Avg degree: {avg_degree:.2}");

        // Sample node for local algorithms
        let sample_node = n / 2;

        // Local clustering coefficient
        let clustering_start = Instant::now();
        let coefficients = measures::clustering_coefficient(&graph)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        let local_clustering = coefficients.get(&sample_node).copied().unwrap_or(0.0);
        println!(
            "  Local clustering (node {}): {:.4} ({:.2}s)",
            sample_node,
            local_clustering,
            clustering_start.elapsed().as_secs_f64()
        );

        println!(
            "  Operations time: {:.2}s",
            ops_start.elapsed().as_secs_f64()
        );
        println!("  Total time: {:.2}s", start.elapsed().as_secs_f64());
    }

    Ok(())
}

#[test]
#[ignore]
#[allow(dead_code)]
fn test_large_grid_graph() -> CoreResult<()> {
    println!("\n=== Grid Graph Stress Test ===");

    let dimensions = vec![(316, 316), (500, 500), (1000, 1000)]; // ~100k, 250k, 1M nodes

    for (rows, cols) in dimensions {
        println!("\nTesting {}x{} grid ({} nodes)", rows, cols, rows * cols);
        let start = Instant::now();

        // Generate graph
        let gen_start = Instant::now();
        let graph = generators::grid_2d_graph(rows, cols)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        let gen_time = gen_start.elapsed();

        println!("  Generation time: {:.2}s", gen_time.as_secs_f64());
        println!("  Nodes: {}", graph.node_count());
        println!("  Edges: {}", graph.edge_count());

        // Test algorithms suitable for grid graphs
        let ops_start = Instant::now();

        // Shortest path from corner to corner
        let source = 0;
        let target = graph.node_count() - 1;
        let path_start = Instant::now();
        let path_result = algorithms::dijkstra_path(&graph, &source, &target)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        let path = path_result.unwrap().nodes;
        println!(
            "  Shortest path (corner to corner): {} steps ({:.2}s)",
            path.len() - 1,
            path_start.elapsed().as_secs_f64()
        );

        // Diameter estimation (sample-based for large graphs)
        if rows * cols <= 250_000 {
            let diam_start = Instant::now();
            let diameter = estimate_diameter(&graph, 10)?;
            println!(
                "  Estimated diameter: {} ({:.2}s)",
                diameter,
                diam_start.elapsed().as_secs_f64()
            );
        }

        println!(
            "  Operations time: {:.2}s",
            ops_start.elapsed().as_secs_f64()
        );
        println!("  Total time: {:.2}s", start.elapsed().as_secs_f64());
    }

    Ok(())
}

#[test]
#[ignore]
#[allow(dead_code)]
fn test_large_directed_graph_algorithms() -> CoreResult<()> {
    println!("\n=== Large Directed Graph Algorithms Test ===");

    let n = 100_000;
    let _edge_probability = 0.00002;

    println!("\nGenerating directed graph with {n} nodes");
    let start = Instant::now();

    // Create directed graph
    let mut graph: DiGraph<usize, f64, u32> = DiGraph::new();
    for i in 0..n {
        graph.add_node(i);
    }

    // Add edges with preferential attachment pattern
    use rand::prelude::*;
    let mut rng = rand::rng();

    for i in 1..n {
        // Add edges from new nodes to existing nodes
        let num_edges = rng.random_range(1..=5);
        for _ in 0..num_edges {
            let target = rng.random_range(0..i);
            graph
                .add_edge(i, target, 1.0)
                .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        }
    }

    println!("  Generation time: {:.2}s", start.elapsed().as_secs_f64());
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());

    // Test directed graph algorithms
    let ops_start = Instant::now();

    // Strongly connected components
    let scc_start = Instant::now();
    let sccs = algorithms::strongly_connected_components(&graph);
    println!(
        "  Strongly connected components: {} ({:.2}s)",
        sccs.len(),
        scc_start.elapsed().as_secs_f64()
    );

    // Find largest SCC
    let largest_scc_size = sccs.iter().map(|scc| scc.len()).max().unwrap_or(0);
    println!("  Largest SCC size: {largest_scc_size}");

    // Topological sort (if DAG)
    match algorithms::topological_sort(&graph) {
        Ok(_topo) => println!("  Graph is DAG, topological sort computed"),
        Err(_) => println!("  Graph contains cycles"),
    }

    println!(
        "  Operations time: {:.2}s",
        ops_start.elapsed().as_secs_f64()
    );
    println!("  Total time: {:.2}s", start.elapsed().as_secs_f64());

    Ok(())
}

#[test]
#[ignore]
#[allow(dead_code)]
fn test_memory_efficient_operations() -> CoreResult<()> {
    println!("\n=== Memory Efficient Operations Test ===");

    let n = 1_000_000;
    let m = 5; // Average degree of 10

    println!("\nGenerating graph with {n} nodes");
    let start = Instant::now();

    // Generate a sparse graph
    let mut rng = rand::rng();
    let graph = generators::barabasi_albert_graph(n, m, &mut rng)
        .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
    println!("  Generation time: {:.2}s", start.elapsed().as_secs_f64());
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());

    // Test memory-efficient operations

    // 1. Streaming degree calculation
    let degree_start = Instant::now();
    let mut degree_sum = 0u64;
    let mut max_degree = 0usize;
    for i in 0..graph.node_count() {
        let degree = graph.degree(&i);
        degree_sum += degree as u64;
        max_degree = max_degree.max(degree);
    }
    let avg_degree = degree_sum as f64 / graph.node_count() as f64;
    println!(
        "  Degree stats: max={}, avg={:.2} ({:.2}s)",
        max_degree,
        avg_degree,
        degree_start.elapsed().as_secs_f64()
    );

    // 2. Sample-based clustering coefficient
    let clustering_start = Instant::now();
    let sample_size = 1000;
    let mut clustering_sum = 0.0;
    if let Ok(coefficients) = measures::clustering_coefficient(&graph) {
        for _ in 0..sample_size {
            let node = rng.random_range(0..graph.node_count());
            if let Some(cc) = coefficients.get(&node) {
                clustering_sum += cc;
            }
        }
    }
    let avg_clustering = clustering_sum / sample_size as f64;
    println!(
        "  Avg clustering coefficient (sampled): {:.4} ({:.2}s)",
        avg_clustering,
        clustering_start.elapsed().as_secs_f64()
    );

    // 3. BFS with depth limit
    let bfs_start = Instant::now();
    let start_node = n / 2;
    let depth_limit = 3;
    let reachable = bfs_with_depth_limit(&graph, start_node, depth_limit)?;
    println!(
        "  Nodes within {} hops of {}: {} ({:.2}s)",
        depth_limit,
        start_node,
        reachable,
        bfs_start.elapsed().as_secs_f64()
    );

    Ok(())
}

#[test]
#[ignore]
#[allow(dead_code)]
fn test_parallel_algorithms_on_large_graphs() -> CoreResult<()> {
    println!("\n=== Parallel Algorithms on Large Graphs Test ===");

    #[cfg(feature = "parallel")]
    {
        use scirs2_core::parallel_ops::*;

        let n = 500_000;
        let edge_probability = 0.00002;

        println!("\nGenerating graph with {} nodes", n);
        let mut rng = rand::rng();
        let graph = generators::erdos_renyi_graph(n, edge_probability, &mut rng)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        println!("  Nodes: {}", graph.node_count());
        println!("  Edges: {}", graph.edge_count());

        // Test parallel degree calculation
        let degree_start = Instant::now();
        let degrees: Vec<usize> = (0..graph.node_count())
            .into_par_iter()
            .map(|i| graph.degree(&i))
            .collect();

        let max_degree = degrees.par_iter().max().unwrap_or(&0);
        println!(
            "  Parallel degree calculation: max={} ({:.2}s)",
            max_degree,
            degree_start.elapsed().as_secs_f64()
        );

        // Test parallel PageRank (skipped - requires DiGraph)
        println!("  Parallel PageRank: Skipped (requires DiGraph, current test uses Graph)");
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("  Parallel feature not enabled, skipping parallel tests");
    }

    Ok(())
}

// Helper functions

#[allow(dead_code)]
fn estimate_diameter(graph: &Graph<usize, f64>, samples: usize) -> CoreResult<usize> {
    use rand::prelude::*;
    let mut rng = rand::rng();
    let mut max_distance = 0;

    for _ in 0..samples {
        let source = rng.random_range(0..graph.node_count());
        // Use BFS to compute distances manually
        let bfs_nodes = algorithms::breadth_first_search(&graph, &source)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;

        // For BFS, distance is the position in the traversal order
        for (dist_count_, _node) in bfs_nodes.into_iter().enumerate() {
            max_distance = max_distance.max(dist_count_);
        }
    }

    Ok(max_distance)
}

#[allow(dead_code)]
fn bfs_with_depth_limit(
    graph: &Graph<usize, f64>,
    start: usize,
    depth_limit: usize,
) -> CoreResult<usize> {
    use std::collections::{HashSet, VecDeque};

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    visited.insert(start);
    queue.push_back((start, 0));

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= depth_limit {
            continue;
        }

        if let Ok(neighbors) = graph.neighbors(&node) {
            for neighbor in neighbors {
                if visited.insert(neighbor) {
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
    }

    Ok(visited.len())
}

#[test]
#[ignore]
#[allow(dead_code)]
fn test_extreme_scale_graph() -> CoreResult<()> {
    println!("\n=== Extreme Scale Graph Test (5M nodes) ===");

    let n = 5_000_000;
    let m = 2; // Very sparse to fit in memory

    println!("\nGenerating graph with {n} nodes");
    let start = Instant::now();

    // Generate graph
    let mut rng = rand::rng();
    let graph = generators::barabasi_albert_graph(n, m, &mut rng)
        .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
    let gen_time = start.elapsed();

    println!("  Generation time: {:.2}s", gen_time.as_secs_f64());
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());
    println!(
        "  Memory estimate: ~{:.1} MB",
        estimate_memory_usage(graph.node_count(), graph.edge_count())
    );

    // Only test lightweight operations at this scale
    let ops_start = Instant::now();

    // Degree distribution sampling
    let sample_size = 10_000;
    let mut degree_samples = Vec::with_capacity(sample_size);
    let step = n / sample_size;

    for i in (0..n).step_by(step) {
        degree_samples.push(graph.degree(&i));
    }

    let max_sampled = degree_samples.iter().max().unwrap_or(&0);
    let avg_sampled = degree_samples.iter().sum::<usize>() as f64 / degree_samples.len() as f64;

    println!("  Sampled degree stats: max={max_sampled}, avg={avg_sampled:.2}");
    println!(
        "  Operations time: {:.2}s",
        ops_start.elapsed().as_secs_f64()
    );
    println!("  Total time: {:.2}s", start.elapsed().as_secs_f64());

    Ok(())
}

#[allow(dead_code)]
fn estimate_memory_usage(nodes: usize, edges: usize) -> f64 {
    // Rough estimate based on adjacency list representation
    let node_overhead = nodes * 24; // Vec pointer + capacity + length
    let edge_storage = edges * 16; // Two usize values per edge
    let total_bytes = node_overhead + edge_storage;
    total_bytes as f64 / (1024.0 * 1024.0)
}

#[test]
#[ignore]
#[allow(dead_code)]
fn test_algorithm_scaling() -> CoreResult<()> {
    println!("\n=== Algorithm Scaling Analysis ===");

    let sizes = vec![10_000, 50_000, 100_000, 200_000];

    println!("\nTesting algorithm scaling with graph sizes: {sizes:?}");

    for n in sizes {
        println!("\n--- Graph size: {n} nodes ---");

        // Generate test graph
        let mut rng = rand::rng();
        let graph = generators::barabasi_albert_graph(n, 3, &mut rng)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        println!("  Edges: {}", graph.edge_count());

        // Test BFS scaling
        let bfs_start = Instant::now();
        let _ = algorithms::breadth_first_search(&graph, &0)
            .map_err(|e| scirs2_core::error::CoreError::from(e.to_string()))?;
        let bfs_time = bfs_start.elapsed();
        println!("  BFS time: {:.3}s", bfs_time.as_secs_f64());

        // Test connected components scaling
        let cc_start = Instant::now();
        let _ = algorithms::connected_components(&graph);
        let cc_time = cc_start.elapsed();
        println!("  Connected components time: {:.3}s", cc_time.as_secs_f64());

        // Test PageRank scaling (fixed iterations) - Skip for undirected graphs
        // let pr_start = Instant::now();
        // let _ = algorithms::pagerank(&graph, 0.85, 1e-6, 5); // PageRank requires DiGraph
        // let pr_time = pr_start.elapsed();
        // println!("  PageRank (5 iter) time: {:.3}s", pr_time.as_secs_f64());
    }

    Ok(())
}
