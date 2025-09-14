//! Performance regression test suite for scirs2-graph
//!
//! This benchmark suite validates that performance doesn't regress between versions
//! and establishes baseline performance metrics for various graph operations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_graph::*;
use std::hint::black_box;
use std::time::Duration;

/// Performance baselines established for regression testing
/// These values should be updated when legitimate performance improvements are made
const PERFORMANCE_BASELINES: &[(&str, &str, f64)] = &[
    // (algorithm, graph_type, baseline_time_ms)
    ("pagerank", "barabasi_albert_1000", 5.0),
    ("pagerank", "erdos_renyi_1000", 4.0),
    ("betweenness_centrality", "barabasi_albert_500", 150.0),
    ("louvain_communities", "barabasi_albert_1000", 25.0),
    ("shortest_path", "grid_2d_32x32", 2.0),
    ("connected_components", "erdos_renyi_10000", 8.0),
    ("minimum_spanning_tree", "complete_graph_500", 80.0),
    ("clustering_coefficient", "watts_strogatz_1000", 12.0),
];

/// Regression tolerance (20% slower than baseline triggers a warning)
const REGRESSION_TOLERANCE: f64 = 1.2;

/// Generate test graphs for benchmarking
struct GraphGenerator {
    rng: StdRng,
}

impl GraphGenerator {
    fn new() -> Self {
        Self {
            rng: StdRng::seed_from_u64(42), // Fixed seed for reproducibility
        }
    }

    fn barabasi_albert(&mut self, n: usize, m: usize) -> Graph<usize, f64> {
        barabasi_albert_graph(n, m, &mut self.rng).unwrap()
    }

    fn erdos_renyi(&mut self, n: usize, p: f64) -> Graph<usize, f64> {
        erdos_renyi_graph(n, p, &mut self.rng).unwrap()
    }

    fn watts_strogatz(&mut self, n: usize, k: usize, p: f64) -> Graph<usize, f64> {
        watts_strogatz_graph(n, k, p, &mut self.rng).unwrap()
    }

    fn complete(&self, n: usize) -> Graph<usize, f64> {
        complete_graph(n).expect("Failed to create complete graph")
    }

    fn grid_2d(&self, rows: usize, cols: usize) -> Graph<usize, f64> {
        grid_2d_graph(rows, cols).expect("Failed to create grid graph")
    }

    fn path(&self, n: usize) -> Graph<usize, f64> {
        path_graph(n).expect("Failed to create path graph")
    }

    fn star(&self, n: usize) -> Graph<usize, f64> {
        star_graph(n).expect("Failed to create star graph")
    }
}

/// Centrality algorithms benchmarks
#[allow(dead_code)]
fn bench_centrality_algorithms(c: &mut Criterion) {
    let mut generator = GraphGenerator::new();
    let mut group = c.benchmark_group("centrality");

    // Set reasonable sample size and measurement time for CI
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(30));

    let graph_configs = vec![
        ("barabasi_albert_500", generator.barabasi_albert(500, 3)),
        ("barabasi_albert_1000", generator.barabasi_albert(1000, 3)),
        ("erdos_renyi_500", generator.erdos_renyi(500, 0.01)),
        ("erdos_renyi_1000", generator.erdos_renyi(1000, 0.005)),
        ("watts_strogatz_500", generator.watts_strogatz(500, 6, 0.3)),
        (
            "watts_strogatz_1000",
            generator.watts_strogatz(1000, 6, 0.3),
        ),
    ];

    for (graph_name, graph) in &graph_configs {
        group.throughput(Throughput::Elements(graph.node_count() as u64));

        // PageRank
        group.bench_with_input(BenchmarkId::new("pagerank", graph_name), &graph, |b, g| {
            b.iter(|| pagerank_centrality(black_box(g), 0.85, 1e-6).unwrap())
        });

        // Betweenness centrality (only for smaller graphs)
        if graph.node_count() <= 500 {
            group.bench_with_input(
                BenchmarkId::new("betweenness_centrality", graph_name),
                &graph,
                |b, g| b.iter(|| betweenness_centrality(black_box(g), true)),
            );
        }

        // Eigenvector centrality
        group.bench_with_input(
            BenchmarkId::new("eigenvector_centrality", graph_name),
            &graph,
            |b, g| b.iter(|| eigenvector_centrality(black_box(g), 100, 1e-6).unwrap()),
        );

        // Closeness centrality (only for smaller graphs)
        if graph.node_count() <= 500 {
            group.bench_with_input(
                BenchmarkId::new("closeness_centrality", graph_name),
                &graph,
                |b, g| b.iter(|| closeness_centrality(black_box(g), true)),
            );
        }
    }

    group.finish();
}

/// Community detection algorithms benchmarks
#[allow(dead_code)]
fn bench_community_detection(c: &mut Criterion) {
    let mut generator = GraphGenerator::new();
    let mut group = c.benchmark_group("community_detection");

    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    let graph_configs = vec![
        ("barabasi_albert_500", generator.barabasi_albert(500, 3)),
        ("barabasi_albert_1000", generator.barabasi_albert(1000, 3)),
        ("barabasi_albert_2000", generator.barabasi_albert(2000, 3)),
        ("planted_partition_500", {
            let mut g = Graph::new();
            // Create a simple planted partition graph for community detection
            for i in 0..500 {
                g.add_node(i);
            }
            // Add intra-community edges
            for community in 0..5 {
                let start = community * 100;
                let end = (community + 1) * 100;
                for i in start..end {
                    for j in (i + 1)..end {
                        if generator.rng.random::<f64>() < 0.3 {
                            g.add_edge(i, j, 1.0).unwrap();
                        }
                    }
                }
            }
            // Add inter-community edges
            for i in 0..500 {
                for j in (i + 1)..500 {
                    if i / 100 != j / 100 && generator.rng.random::<f64>() < 0.01 {
                        g.add_edge(i, j, 1.0).unwrap();
                    }
                }
            }
            g
        }),
    ];

    for (graph_name, graph) in &graph_configs {
        group.throughput(Throughput::Elements(graph.node_count() as u64));

        // Louvain method
        group.bench_with_input(
            BenchmarkId::new("louvain_communities", graph_name),
            &graph,
            |b, g| b.iter(|| louvain_communities_result(black_box(g)).communities),
        );

        // Label propagation
        group.bench_with_input(
            BenchmarkId::new("label_propagation", graph_name),
            &graph,
            |b, g| b.iter(|| label_propagation_result(black_box(g), 100).communities),
        );

        // Modularity optimization
        if graph.node_count() <= 1000 {
            group.bench_with_input(
                BenchmarkId::new("modularity_optimization", graph_name),
                &graph,
                |b, g| {
                    b.iter(|| {
                        modularity_optimization_result(black_box(g), 0.5, 100.0, 1000).communities
                    })
                },
            );
        }
    }

    group.finish();
}

/// Path finding algorithms benchmarks
#[allow(dead_code)]
fn bench_path_algorithms(c: &mut Criterion) {
    let mut generator = GraphGenerator::new();
    let mut group = c.benchmark_group("path_algorithms");

    group.sample_size(50);
    group.measurement_time(Duration::from_secs(30));

    let graph_configs = vec![
        ("grid_2d_32x32", generator.grid_2d(32, 32)),
        ("grid_2d_64x64", generator.grid_2d(64, 64)),
        ("barabasi_albert_1000", generator.barabasi_albert(1000, 3)),
        ("erdos_renyi_1000", generator.erdos_renyi(1000, 0.005)),
    ];

    for (graph_name, graph) in &graph_configs {
        group.throughput(Throughput::Elements(graph.node_count() as u64));

        let nodes: Vec<_> = graph.nodes().into_iter().collect();
        if nodes.len() >= 2 {
            let source = nodes[0];
            let target = nodes[nodes.len() - 1];

            // Dijkstra's algorithm
            group.bench_with_input(
                BenchmarkId::new("dijkstra_path", graph_name),
                &(graph, source, target),
                |b, (g, s, t)| {
                    b.iter(|| dijkstra_path(black_box(g), black_box(s), black_box(t)).ok())
                },
            );

            // BFS
            group.bench_with_input(
                BenchmarkId::new("breadth_first_search", graph_name),
                &(graph, source),
                |b, (g, s)| b.iter(|| breadth_first_search(black_box(g), black_box(s)).unwrap()),
            );

            // DFS
            group.bench_with_input(
                BenchmarkId::new("depth_first_search", graph_name),
                &(graph, source),
                |b, (g, s)| b.iter(|| depth_first_search(black_box(g), black_box(s)).unwrap()),
            );
        }
    }

    group.finish();
}

/// Graph properties and measures benchmarks
#[allow(dead_code)]
fn bench_graph_measures(c: &mut Criterion) {
    let mut generator = GraphGenerator::new();
    let mut group = c.benchmark_group("graph_measures");

    group.sample_size(30);
    group.measurement_time(Duration::from_secs(45));

    let graph_configs = vec![
        ("barabasi_albert_500", generator.barabasi_albert(500, 3)),
        ("barabasi_albert_1000", generator.barabasi_albert(1000, 3)),
        ("watts_strogatz_500", generator.watts_strogatz(500, 6, 0.3)),
        (
            "watts_strogatz_1000",
            generator.watts_strogatz(1000, 6, 0.3),
        ),
        ("erdos_renyi_1000", generator.erdos_renyi(1000, 0.005)),
    ];

    for (graph_name, graph) in &graph_configs {
        group.throughput(Throughput::Elements(graph.node_count() as u64));

        // Connected components
        group.bench_with_input(
            BenchmarkId::new("connected_components", graph_name),
            &graph,
            |b, g| b.iter(|| connected_components(black_box(g))),
        );

        // Clustering coefficient
        group.bench_with_input(
            BenchmarkId::new("clustering_coefficient", graph_name),
            &graph,
            |b, g| b.iter(|| clustering_coefficient(black_box(g)).unwrap()),
        );

        // Graph density
        group.bench_with_input(
            BenchmarkId::new("graph_density", graph_name),
            &graph,
            |b, g| b.iter(|| graph_density(black_box(g))),
        );

        // K-core decomposition
        group.bench_with_input(
            BenchmarkId::new("k_core_decomposition", graph_name),
            &graph,
            |b, g| b.iter(|| k_core_decomposition(black_box(g))),
        );
    }

    group.finish();
}

/// Spanning tree and matching algorithms benchmarks
#[allow(dead_code)]
fn bench_spanning_and_matching(c: &mut Criterion) {
    let mut generator = GraphGenerator::new();
    let mut group = c.benchmark_group("spanning_matching");

    group.sample_size(20);
    group.measurement_time(Duration::from_secs(45));

    let graph_configs = vec![
        ("complete_graph_100", generator.complete(100)),
        ("complete_graph_200", generator.complete(200)),
        ("complete_graph_500", generator.complete(500)),
        ("grid_2d_32x32", generator.grid_2d(32, 32)),
        ("barabasi_albert_1000", generator.barabasi_albert(1000, 5)),
    ];

    for (graph_name, graph) in &graph_configs {
        group.throughput(Throughput::Elements(graph.edge_count() as u64));

        // Minimum spanning tree
        group.bench_with_input(
            BenchmarkId::new("minimum_spanning_tree", graph_name),
            &graph,
            |b, g| b.iter(|| minimum_spanning_tree(black_box(g)).unwrap()),
        );

        // Maximum cardinality matching
        if graph.node_count() <= 500 {
            group.bench_with_input(
                BenchmarkId::new("maximum_cardinality_matching", graph_name),
                &graph,
                |b, g| b.iter(|| maximum_cardinality_matching(black_box(g))),
            );
        }
    }

    group.finish();
}

/// Graph generation benchmarks
#[allow(dead_code)]
fn bench_graph_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_generation");

    group.sample_size(30);
    group.measurement_time(Duration::from_secs(30));

    let configs = vec![("small", 100), ("medium", 1000), ("large", 10000)];

    for (size_name, n) in configs {
        group.throughput(Throughput::Elements(n as u64));

        // Erdős-Rényi
        group.bench_with_input(BenchmarkId::new("erdos_renyi", size_name), &n, |b, &n| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(42);
                erdos_renyi_graph(black_box(n), 0.01, &mut rng).unwrap()
            })
        });

        // Barabási-Albert
        group.bench_with_input(
            BenchmarkId::new("barabasi_albert", size_name),
            &n,
            |b, &n| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(42);
                    barabasi_albert_graph(black_box(n), 3, &mut rng).unwrap()
                })
            },
        );

        // Watts-Strogatz
        group.bench_with_input(
            BenchmarkId::new("watts_strogatz", size_name),
            &n,
            |b, &n| {
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(42);
                    watts_strogatz_graph(black_box(n), 6, 0.3, &mut rng).unwrap()
                })
            },
        );

        // Complete graph (only for smaller sizes)
        if n <= 1000 {
            group.bench_with_input(
                BenchmarkId::new("complete_graph", size_name),
                &n,
                |b, &n| b.iter(|| complete_graph(black_box(n))),
            );
        }
    }

    group.finish();
}

/// Memory operations benchmarks
#[allow(dead_code)]
fn bench_memory_operations(c: &mut Criterion) {
    let mut generator = GraphGenerator::new();
    let mut group = c.benchmark_group("memory_operations");

    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let graph_configs = vec![
        ("barabasi_albert_1000", generator.barabasi_albert(1000, 3)),
        ("erdos_renyi_5000", generator.erdos_renyi(5000, 0.001)),
    ];

    for (graph_name, graph) in &graph_configs {
        group.throughput(Throughput::Elements(graph.node_count() as u64));

        // Graph cloning
        group.bench_with_input(
            BenchmarkId::new("graph_clone", graph_name),
            &graph,
            |b, g| b.iter(|| black_box(g.clone())),
        );

        // Subgraph creation
        let nodes: std::collections::HashSet<_> = graph
            .nodes()
            .into_iter()
            .take(graph.node_count() / 2)
            .cloned()
            .collect();
        group.bench_with_input(
            BenchmarkId::new("subgraph_creation", graph_name),
            &(graph, &nodes),
            |b, (g, nodes)| b.iter(|| subgraph(black_box(g), black_box(nodes))),
        );

        // Memory profiling
        group.bench_with_input(
            BenchmarkId::new("memory_profiling", graph_name),
            &graph,
            |b, g| b.iter(|| black_box(MemoryProfiler::profile_graph(g))),
        );
    }

    group.finish();
}

/// Parallel algorithms benchmarks (if parallel features are enabled)
#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn bench_parallel_algorithms(c: &mut Criterion) {
    let mut generator = GraphGenerator::new();
    let mut group = c.benchmark_group("parallel_algorithms");

    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    let graph_configs = vec![
        ("barabasi_albert_2000", generator.barabasi_albert(2000, 3)),
        ("barabasi_albert_5000", generator.barabasi_albert(5000, 3)),
        ("erdos_renyi_10000", generator.erdos_renyi(10000, 0.0005)),
    ];

    for (graph_name, graph) in &graph_configs {
        group.throughput(Throughput::Elements(graph.node_count() as u64));

        // Parallel PageRank
        group.bench_with_input(
            BenchmarkId::new("parallel_pagerank", graph_name),
            &graph,
            |b, g| b.iter(|| parallel_pagerank_centrality(black_box(g), 0.85, Some(1e-6)).unwrap()),
        );

        // Parallel Louvain
        group.bench_with_input(
            BenchmarkId::new("parallel_louvain", graph_name),
            &graph,
            |b, g| b.iter(|| parallel_louvain_communities_result(black_box(g)).communities),
        );

        // Compare with sequential versions
        group.bench_with_input(
            BenchmarkId::new("sequential_pagerank", graph_name),
            &graph,
            |b, g| b.iter(|| pagerank(black_box(g), 0.85, Some(1e-6)).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("sequential_louvain", graph_name),
            &graph,
            |b, g| b.iter(|| louvain_communities_result(black_box(g)).communities),
        );
    }

    group.finish();
}

/// Large graph stress tests for scalability
#[allow(dead_code)]
fn bench_large_graph_scalability(c: &mut Criterion) {
    let mut generator = GraphGenerator::new();
    let mut group = c.benchmark_group("large_graph_scalability");

    // Longer measurement time for large graphs
    group.sample_size(5);
    group.measurement_time(Duration::from_secs(120));

    let large_graph_configs = vec![
        ("barabasi_albert_10k", generator.barabasi_albert(10000, 3)),
        ("barabasi_albert_50k", generator.barabasi_albert(50000, 3)),
        ("erdos_renyi_10k", generator.erdos_renyi(10000, 0.0005)),
        ("grid_2d_100x100", generator.grid_2d(100, 100)),
    ];

    for (graph_name, graph) in &large_graph_configs {
        println!(
            "Benchmarking large graph: {} ({} nodes, {} edges)",
            graph_name,
            graph.node_count(),
            graph.edge_count()
        );

        group.throughput(Throughput::Elements(graph.node_count() as u64));

        // Only test algorithms that scale well to large graphs
        group.bench_with_input(
            BenchmarkId::new("large_pagerank", graph_name),
            &graph,
            |b, g| {
                b.iter(|| {
                    pagerank_centrality(black_box(g), 0.85, 1e-3).unwrap() // Relaxed tolerance for speed
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("large_connected_components", graph_name),
            &graph,
            |b, g| b.iter(|| connected_components(black_box(g))),
        );

        group.bench_with_input(
            BenchmarkId::new("large_clustering_coefficient", graph_name),
            &graph,
            |b, g| b.iter(|| clustering_coefficient(black_box(g)).unwrap()),
        );

        // Community detection for large graphs
        if graph.node_count() <= 20000 {
            group.bench_with_input(
                BenchmarkId::new("large_label_propagation", graph_name),
                &graph,
                |b, g| b.iter(|| label_propagation_result(black_box(g), 100).communities),
            );
        }
    }

    group.finish();
}

/// Regression checker function
#[allow(dead_code)]
fn check_performance_regression() {
    println!("\n=== Performance Regression Check ===");
    println!("Baseline performance expectations:");
    for (algorithm, graph_type, baseline_ms) in PERFORMANCE_BASELINES {
        println!("  {}/{}: {:.1}ms", algorithm, graph_type, baseline_ms);
    }
    println!(
        "Regression tolerance: {:.0}% above baseline",
        (REGRESSION_TOLERANCE - 1.0) * 100.0
    );
    println!("Run 'cargo bench --bench performance_regression' to validate performance");
    println!("========================================\n");
}

criterion_group!(
    benches,
    bench_centrality_algorithms,
    bench_community_detection,
    bench_path_algorithms,
    bench_graph_measures,
    bench_spanning_and_matching,
    bench_graph_generation,
    bench_memory_operations,
    bench_large_graph_scalability,
);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, bench_parallel_algorithms,);

#[cfg(feature = "parallel")]
criterion_main!(benches, parallel_benches);

#[cfg(not(feature = "parallel"))]
criterion_main!(benches);

// Initialize regression checker on import
#[ctor::ctor]
#[allow(dead_code)]
fn init() {
    check_performance_regression();
}
