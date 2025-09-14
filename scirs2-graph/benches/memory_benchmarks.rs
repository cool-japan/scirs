//! Memory usage and performance benchmarks for different graph representations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_graph::{
    generators,
    memory::{BitPackedGraph, CSRGraph, CompressedAdjacencyList, HybridGraph, MemoryProfiler},
    Graph,
};
use std::hint::black_box;
use std::time::Duration;

/// Benchmark memory usage for different graph representations
#[allow(dead_code)]
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));

    let sizes = vec![1_000, 10_000, 50_000];
    let edge_probability = 0.01; // 1% density

    for size in sizes {
        // Standard Graph representation
        group.bench_with_input(BenchmarkId::new("standard_graph", size), &size, |b, &n| {
            b.iter(|| {
                let graph =
                    generators::erdos_renyi_graph(n, edge_probability, &mut rand::rng()).unwrap();
                let stats = MemoryProfiler::profile_graph(&graph);
                black_box(stats.total_bytes)
            });
        });

        // CSR representation
        group.bench_with_input(BenchmarkId::new("csr_graph", size), &size, |b, &n| {
            b.iter(|| {
                let graph =
                    generators::erdos_renyi_graph(n, edge_probability, &mut rand::rng()).unwrap();
                let edges: Vec<_> = (0..graph.node_count())
                    .flat_map(|u| {
                        graph
                            .neighbors(&u)
                            .unwrap()
                            .into_iter()
                            .map(move |v| (u, v, 1.0))
                    })
                    .collect();
                let csr = CSRGraph::from_edges(n, edges).unwrap();
                black_box(csr.memory_usage())
            });
        });

        // Bit-packed representation (for unweighted)
        group.bench_with_input(BenchmarkId::new("bitpacked_graph", size), &size, |b, &n| {
            b.iter(|| {
                let mut bitpacked = BitPackedGraph::new(n, false);
                let graph =
                    generators::erdos_renyi_graph(n, edge_probability, &mut rand::rng()).unwrap();

                for u in 0..graph.node_count() {
                    for v in graph.neighbors(&u).unwrap() {
                        if u <= v {
                            // Avoid duplicates for undirected
                            bitpacked.add_edge(u, v).unwrap();
                        }
                    }
                }
                black_box(bitpacked.memory_usage())
            });
        });
    }

    group.finish();
}

/// Benchmark neighbor iteration performance
#[allow(dead_code)]
fn bench_neighbor_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_iteration");

    let n = 10_000;
    let m = 5; // Average degree of 10

    // Generate test graph
    let graph = generators::barabasi_albert_graph(n, m, &mut rand::rng()).unwrap();

    // Create different representations
    let edges: Vec<_> = (0..graph.node_count())
        .flat_map(|u| {
            graph
                .neighbors(&u)
                .unwrap()
                .into_iter()
                .map(move |v| (u, v, 1.0))
        })
        .collect();

    let csr = CSRGraph::from_edges(n, edges.clone()).unwrap();

    let mut bitpacked = BitPackedGraph::new(n, false);
    for (u, v, _) in &edges {
        if u <= v {
            bitpacked.add_edge(*u, *v).unwrap();
        }
    }

    let adj_lists: Vec<Vec<_>> = (0..n)
        .map(|u| graph.neighbors(&u).unwrap().into_iter().collect())
        .collect();
    let compressed = CompressedAdjacencyList::from_adjacency(adj_lists);

    // Benchmark standard graph
    group.bench_function("standard_graph", |b| {
        b.iter(|| {
            let mut sum = 0;
            for node in 0..n {
                for neighbor in graph.neighbors(&node).unwrap() {
                    sum += neighbor;
                }
            }
            black_box(sum)
        });
    });

    // Benchmark CSR
    group.bench_function("csr_graph", |b| {
        b.iter(|| {
            let mut sum = 0;
            for node in 0..n {
                for neighbor in csr.neighbors(node) {
                    sum += neighbor.0;
                }
            }
            black_box(sum)
        });
    });

    // Benchmark bit-packed
    group.bench_function("bitpacked_graph", |b| {
        b.iter(|| {
            let mut sum = 0;
            for node in 0..n {
                for neighbor in bitpacked.neighbors(node) {
                    sum += neighbor;
                }
            }
            black_box(sum)
        });
    });

    // Benchmark compressed adjacency
    group.bench_function("compressed_adjacency", |b| {
        b.iter(|| {
            let mut sum = 0;
            for node in 0..n {
                for neighbor in compressed.neighbors(node) {
                    sum += neighbor;
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

/// Benchmark edge query performance
#[allow(dead_code)]
fn bench_edge_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_queries");

    let n = 5_000;
    let edge_probability = 0.02;

    // Generate test graph
    let graph = generators::erdos_renyi_graph(n, edge_probability, &mut rand::rng()).unwrap();

    // Create bit-packed representation
    let mut bitpacked = BitPackedGraph::new(n, false);
    for u in 0..graph.node_count() {
        for v in graph.neighbors(&u).unwrap() {
            if u <= v {
                bitpacked.add_edge(u, v).unwrap();
            }
        }
    }

    // Generate random query pairs
    use rand::prelude::*;
    let mut rng = rand::rng();
    let query_pairs: Vec<(usize, usize)> = (0..1000)
        .map(|_| (rng.gen_range(0..n), rng.gen_range(0..n)))
        .collect();

    // Benchmark standard graph
    group.bench_function("standard_graph", |b| {
        b.iter(|| {
            let mut count = 0;
            for &(u, v) in &query_pairs {
                if graph.has_edge(&u, &v) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    // Benchmark bit-packed
    group.bench_function("bitpacked_graph", |b| {
        b.iter(|| {
            let mut count = 0;
            for &(u, v) in &query_pairs {
                if bitpacked.has_edge(u, v) {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.finish();
}

/// Benchmark graph construction time
#[allow(dead_code)]
fn bench_construction_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_time");

    let sizes = vec![1_000, 5_000, 10_000];

    for size in sizes {
        let edges: Vec<(usize, usize, f64)> =
            generators::barabasi_albert_graph(size, 3, &mut rand::rng())
                .unwrap()
                .edges()
                .into_iter()
                .map(|edge| (edge.source, edge.target, edge.weight))
                .collect();

        // Standard graph construction
        group.bench_with_input(
            BenchmarkId::new("standard_graph", size),
            &edges,
            |b, edges| {
                b.iter(|| {
                    let mut graph = Graph::<usize, f64>::new();
                    for i in 0..size {
                        graph.add_node(i);
                    }
                    for &(u, v, w) in edges {
                        graph.add_edge(u, v, w).unwrap();
                    }
                    black_box(graph)
                });
            },
        );

        // CSR construction
        group.bench_with_input(
            BenchmarkId::new("csr_graph", size),
            &edges,
            |b, edges: &Vec<(usize, usize, f64)>| {
                b.iter(|| {
                    let csr = CSRGraph::from_edges(size, edges.clone()).unwrap();
                    black_box(csr)
                });
            },
        );

        // Hybrid auto-selection
        group.bench_with_input(
            BenchmarkId::new("hybrid_auto", size),
            &edges,
            |b, edges: &Vec<(usize, usize, f64)>| {
                b.iter(|| {
                    let edges_opt: Vec<_> =
                        edges.iter().map(|&(u, v, w)| (u, v, Some(w))).collect();
                    let hybrid = HybridGraph::auto_select(size, edges_opt, false).unwrap();
                    black_box(hybrid)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory fragmentation analysis
#[allow(dead_code)]
fn bench_fragmentation_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragmentation_analysis");

    let sizes = vec![1_000, 5_000, 10_000];

    for size in sizes {
        group.bench_with_input(
            BenchmarkId::new("analyze_fragmentation", size),
            &size,
            |b, &n| {
                let graph = generators::barabasi_albert_graph(n, 5, &mut rand::rng()).unwrap();
                b.iter(|| {
                    let report = MemoryProfiler::analyze_fragmentation(&graph);
                    black_box(report.fragmentation_ratio)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_usage,
    bench_neighbor_iteration,
    bench_edge_queries,
    bench_construction_time,
    bench_fragmentation_analysis
);
criterion_main!(benches);
