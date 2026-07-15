# SciRS2 Graph

[![crates.io](https://img.shields.io/crates/v/scirs2-graph.svg)](https://crates.io/crates/scirs2-graph)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-graph)](https://docs.rs/scirs2-graph)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)]()

**scirs2-graph** is the graph theory and network analysis crate for the [SciRS2](https://github.com/cool-japan/scirs) scientific computing library. It provides a comprehensive suite of graph algorithms, data structures, graph neural networks, embeddings, and visualization tools for scientific computing, machine learning, and network science applications.

**Status (2026-07-15, v0.6.1)**: Stable. No `todo!()`/`unimplemented!()` stubs found anywhere in `src/`. Test suite: 1418/1418 passing with default features (8 skipped), 1479/1479 passing with `--all-features` (8 skipped, including GPU-dispatch smoke tests under the `wgpu` feature) — 0 failures in either configuration.

## What scirs2-graph Provides

Use scirs2-graph when you need to:

- Analyze social, biological, or infrastructure networks
- Run community detection or graph clustering
- Compute shortest paths, centrality, or flow in large graphs
- Train graph neural networks (GCN, GAT, GraphSAGE, Graph Transformers)
- Generate graph embeddings with Node2Vec or spectral methods
- Work with temporal, heterogeneous, or knowledge graphs
- Visualize graphs as SVG or DOT output
- Detect graph isomorphism or subgraph patterns

## Features (v0.6.1)

### Core Graph Representations
- Directed and undirected graphs with efficient adjacency storage
- Multi-graphs with parallel edges
- Bipartite graphs with specialized bipartite algorithms
- Hypergraphs for higher-order relationships
- Attributed graphs with rich vertex and edge metadata
- Temporal graphs and dynamic graph algorithms
- Heterogeneous graphs and knowledge graphs
- CSR (Compressed Sparse Row) graph representation for large-scale graphs

### Graph Traversal and Search
- Breadth-first search (BFS), depth-first search (DFS)
- Bidirectional search, priority-first search
- A* pathfinding with custom heuristics

### Shortest Paths
- Dijkstra's algorithm (single-source)
- Bellman-Ford (handles negative weights)
- Floyd-Warshall (all-pairs)
- k-shortest paths

### Connectivity and Structure
- Connected components, strongly connected components
- Articulation points and bridges
- Topological sorting
- k-core decomposition
- Clique enumeration and motif detection (triangles, stars)

### Network Flow
- Ford-Fulkerson, Dinic's algorithm, push-relabel
- Minimum-cost flow
- Minimum cut

### Matching
- Bipartite matching (Hopcroft-Karp)
- Maximum cardinality matching
- Stable marriage problem

### Centrality Measures
- Degree, betweenness, closeness, eigenvector centrality
- PageRank, personalized PageRank, Katz centrality
- HITS algorithm (hubs and authorities)

### Community Detection
- Louvain method (modularity optimization)
- Girvan-Newman (edge betweenness)
- Label propagation
- Infomap algorithm
- Fluid communities
- Hierarchical clustering

### Spectral Graph Theory
- Graph Laplacian and normalized Laplacian
- Spectral clustering
- Algebraic connectivity (Fiedler value)
- Normalized cut

### Graph Neural Networks
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- GraphSAGE (inductive representation learning)
- Message-passing framework
- Graph Transformers (Graphormer, GPS) with positional encodings
- Equivariant GNNs (E(n)-GNN, SE(3)-Transformer) for molecular applications
- Relational GCN (R-GCN) and Heterogeneous Graph Transformer (HGT) for knowledge graphs

### Graph Self-Supervised Learning
- Contrastive learning (GraphCL-style augmentation, SimGRACE weight perturbation, NT-Xent loss)
- Graph masked autoencoders (GraphMAE)
- Node-masking and graph-context pre-training objectives

### Hypergraph Neural Networks
- Hypergraph convolution layers (HGNN)
- Hypergraph attention networks
- Hyperedge prediction

### Signed and Directed Graph Learning
- Signed Laplacian, SPONGE embedding, signed ratio-cut clustering
- Balance-theory Signed GCN (SGCN)
- HOPE and APP directed graph embeddings

### Graph Embeddings
- Node2Vec random walk embeddings
- DeepWalk
- Spectral embeddings
- Diffusion-based embeddings

### Graph Isomorphism and Matching
- VF2 algorithm for graph/subgraph isomorphism
- Subgraph matching with constraints

### Graph Alignment
- IsoRank spectral alignment (Kronecker product power iteration)
- GRASP (greedy randomized construction + local search) alignment

### Graph Signal Processing
- Graph Fourier transform
- Graph wavelets
- Graph filtering in spectral domain

### Network Analysis
- Graph diameter, radius, density
- Clustering coefficient (local and global)
- Average shortest path length
- Small-world metrics
- Network robustness and reliability analysis
- Social network analysis (influence propagation, role detection)

### Graph Generators
- Erdos-Renyi random graphs
- Barabasi-Albert (preferential attachment)
- Watts-Strogatz (small-world)
- Regular graphs, complete graphs, path graphs, cycle graphs
- Grid graphs, tree generators

### Graph I/O
- GraphML, GML, DOT (Graphviz), JSON
- Edge list and adjacency list formats
- Matrix Market for sparse representations

### Graph Visualization
- SVG output with customizable layouts
- DOT format for Graphviz rendering
- Force-directed, circular, hierarchical layouts

### Large-Scale Partitioning and Distributed Storage
- Multilevel k-way partitioning (METIS-style coarsen/uncoarsen/refine)
- Streaming partitioning for dynamic graphs
- Distributed graph shards with hash and Fennel partitioning

### Graph Condensation
- Coreset selection (k-center, importance sampling, kernel herding)
- Gradient-matching graph distillation
- Condensation-quality evaluation (degree/spectral distance, label coverage)

### Additional Features
- Domination problems (dominating sets, independent sets)
- Planarity testing
- Algebraic graph theory
- Reliability and robustness analysis
- Sampling algorithms for large graphs

## Installation

```toml
[dependencies]
scirs2-graph = "0.6.1"
```

For parallel processing support:

```toml
[dependencies]
scirs2-graph = { version = "0.6.1", features = ["parallel"] }
```

## Quick Start

### Basic Graph Operations

```rust
use scirs2_graph::{Graph, connected_components, betweenness_centrality, Result};

fn main() -> Result<()> {
    let mut g: Graph<i32, f64> = Graph::new();
    g.add_node(0);
    g.add_node(1);
    g.add_node(2);
    g.add_edge(0, 1, 1.0)?;
    g.add_edge(1, 2, 2.0)?;
    g.add_edge(0, 2, 4.0)?;

    println!("Nodes: {}, Edges: {}", g.node_count(), g.edge_count());

    // connected_components/betweenness_centrality are infallible for a built graph
    let components = connected_components(&g);
    println!("Connected components: {}", components.len());

    let centrality = betweenness_centrality(&g, true); // true = normalized
    println!("Betweenness centrality: {:?}", centrality);

    Ok(())
}
```

### Community Detection

```rust
use scirs2_graph::{louvain_communities_result, label_propagation_result, Result};
use scirs2_graph::generators::barabasi_albert_graph;

fn community_example() -> Result<()> {
    let mut rng = scirs2_core::random::rng();
    let graph = barabasi_albert_graph(200, 3, &mut rng)?;

    // Louvain community detection
    let communities = louvain_communities_result(&graph);
    println!("Louvain found {} communities", communities.num_communities);

    // Label propagation (max_iterations = 100)
    let lp_communities = label_propagation_result(&graph, 100);
    println!("Label propagation found {} communities", lp_communities.num_communities);

    Ok(())
}
```

### Graph Neural Networks

```rust
use scirs2_graph::gnn::{GCNLayer, GATLayer, GraphSAGELayer};

fn gnn_example() {
    // Graph Convolutional Network layer (in_dim=64, out_dim=32)
    let gcn = GCNLayer::new(64, 32);

    // Graph Attention Network layer (in_dim=64, out_dim=32)
    let gat = GATLayer::new(64, 32);

    // GraphSAGE layer (in_dim=64, out_dim=32)
    let sage = GraphSAGELayer::new(64, 32);

    println!(
        "GNN layers initialized (out_dims: {}, {}, {})",
        gcn.out_dim, gat.out_dim, sage.out_dim
    );
}
```

### Node2Vec Embeddings

```rust
use scirs2_graph::embeddings::{Node2Vec, Node2VecConfig};
use scirs2_graph::Result;

fn embedding_example() -> Result<()> {
    // Build graph first...
    // let graph = ...;

    let config = Node2VecConfig {
        dimensions: 128,
        walk_length: 80,
        num_walks: 10,
        p: 1.0,  // return parameter
        q: 0.5,  // in-out parameter
        ..Default::default()
    };

    // let mut node2vec = Node2Vec::new(config);
    // node2vec.train(&graph)?;       // fits skip-gram over generated random walks
    // let model = node2vec.model();  // -> &EmbeddingModel<N>
    Ok(())
}
```

### Subgraph Isomorphism (VF2)

```rust
use scirs2_graph::isomorphism::vf2_subgraph_isomorphism;
use scirs2_graph::Result;

fn isomorphism_example() -> Result<()> {
    // Query whether pattern is a subgraph of target (max_matches = 0 means "find all")
    // let result = vf2_subgraph_isomorphism(&pattern, &target, 0);
    // println!("Found {} subgraph matches", result.mappings.len());
    Ok(())
}
```

### Graph Visualization

```rust
use scirs2_graph::visualization::{compute_layout, render_svg, LayoutAlgorithm, SvgConfig};
use scirs2_graph::{Graph, Result};

fn viz_example() -> Result<()> {
    let mut g: Graph<&str, f64> = Graph::new();
    g.add_node("A");
    g.add_node("B");
    g.add_edge("A", "B", 1.0)?;

    let layout = compute_layout(&g, &LayoutAlgorithm::Circular { radius: 100.0 }, 400.0, 300.0)?;
    let svg = render_svg(&g, &layout, &SvgConfig::default());
    // std::fs::write("graph.svg", svg)?;
    println!("Rendered {} bytes of SVG", svg.len());
    Ok(())
}
```

### Temporal Graphs

```rust
use scirs2_graph::temporal::{TemporalGraph, TemporalEdge};

fn temporal_example() {
    let mut tg = TemporalGraph::new(4); // 4 nodes, indices 0..4
    tg.add_edge(TemporalEdge::new(0, 1, 1.0)); // (from, to, timestamp)
    tg.add_edge(TemporalEdge::new(1, 2, 2.0));
    tg.add_edge(TemporalEdge::new(2, 3, 3.0));

    // Snapshot of interactions within the time window [0.5, 2.5)
    let snapshot = tg.snapshot(0.5, 2.5);
    println!("Snapshot has {} edges", snapshot.edge_count());
}
```

## GPU Acceleration (v0.5.0)

`scirs2-graph` provides real wgpu GPU dispatch for large-scale graph traversal and shortest paths (feature `wgpu`):

| Algorithm | Implementation |
|-----------|---------------|
| BFS | WGSL level-sync kernel with `atomicCompareExchange` on distances; CPU fallback for n_edges < 4096 |
| Bellman-Ford SSSP | WGSL edge-parallel kernel with `atomicMin` (f32-bits trick); handles non-negative weights |
| Delta-stepping SSSP | True WGSL bucket-based relaxation: `DELTA_LIGHT_WGSL` + `DELTA_APPLY_WGSL` + `DELTA_HEAVY_WGSL` with `changed_flag` convergence; adaptive delta heuristic |
| CPU-parallel BFS | `parallel_bfs_atomic` via rayon + `AtomicI32` distance array |
| CPU-parallel Bellman-Ford | `parallel_bellman_ford_atomic` via rayon + `AtomicU32` (f32 bits) |

All GPU algorithms gracefully skip to CPU when no wgpu adapter is available (tested on Darwin Metal and Linux Vulkan).

## Feature Flags

| Flag | Description |
|------|-------------|
| `parallel` | Enable Rayon-based parallel processing for large graph algorithms |
| `simd` | Enable SIMD-accelerated numerical operations |
| `wgpu` | Enable wgpu GPU dispatch for BFS, SSSP, delta-stepping (requires wgpu/pollster/bytemuck) |
| `cuda` | Off-by-default, NVIDIA-only CUDA CSR SpMV for graph adjacency matrices via the pure-Rust oxicuda-* crates; runtime-probed, falls back to CPU when no NVIDIA device is present (e.g. macOS) |

## Performance

- Multi-threaded algorithms via Rayon for large graphs (millions of nodes/edges)
- GPU dispatch (wgpu) for BFS and SSSP on large graphs (n_edges ≥ 4096)
- CSR representation for cache-efficient traversal
- Memory profiling tools built in
- Validated against NetworkX and igraph reference implementations
- 10-50x faster than NetworkX for most core operations

## Documentation

Full API reference: [docs.rs/scirs2-graph](https://docs.rs/scirs2-graph)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
