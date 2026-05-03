# SciRS2 Graph

[![crates.io](https://img.shields.io/crates/v/scirs2-graph.svg)](https://crates.io/crates/scirs2-graph)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-graph)](https://docs.rs/scirs2-graph)

**scirs2-graph** is the graph theory and network analysis crate for the [SciRS2](https://github.com/cool-japan/scirs) scientific computing library. It provides a comprehensive suite of graph algorithms, data structures, graph neural networks, embeddings, and visualization tools for scientific computing, machine learning, and network science applications.

## What scirs2-graph Provides

Use scirs2-graph when you need to:

- Analyze social, biological, or infrastructure networks
- Run community detection or graph clustering
- Compute shortest paths, centrality, or flow in large graphs
- Train graph neural networks (GCN, GAT, GraphSAGE, GIN)
- Generate graph embeddings with Node2Vec or spectral methods
- Work with temporal, heterogeneous, or knowledge graphs
- Visualize graphs as SVG or DOT output
- Detect graph isomorphism or subgraph patterns

## Features (v0.4.3)

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
- Graph Isomorphism Network (GIN)
- Message-passing framework

### Graph Embeddings
- Node2Vec random walk embeddings
- DeepWalk
- Spectral embeddings
- Diffusion-based embeddings

### Graph Isomorphism and Matching
- VF2 algorithm for graph/subgraph isomorphism
- Subgraph matching with constraints

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

### Additional Features
- Domination problems (dominating sets, independent sets)
- Planarity testing
- Algebraic graph theory
- Reliability and robustness analysis
- Sampling algorithms for large graphs

## Installation

```toml
[dependencies]
scirs2-graph = "0.4.3"
```

For parallel processing support:

```toml
[dependencies]
scirs2-graph = { version = "0.4.3", features = ["parallel"] }
```

## Quick Start

### Basic Graph Operations

```rust
use scirs2_graph::{Graph, connected_components, betweenness_centrality};
use scirs2_core::error::CoreResult;

fn main() -> CoreResult<()> {
    let mut g: Graph<i32, f64> = Graph::new();
    let n0 = g.add_node(0);
    let n1 = g.add_node(1);
    let n2 = g.add_node(2);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 2.0);
    g.add_edge(0, 2, 4.0);

    println!("Nodes: {}, Edges: {}", g.node_count(), g.edge_count());

    let components = connected_components(&g)?;
    println!("Connected components: {}", components.len());

    let centrality = betweenness_centrality(&g)?;
    println!("Betweenness centrality: {:?}", centrality);

    Ok(())
}
```

### Community Detection

```rust
use scirs2_graph::{louvain_communities, label_propagation_communities};
use scirs2_graph::generators::barabasi_albert_graph;
use scirs2_core::error::CoreResult;

fn community_example() -> CoreResult<()> {
    let graph = barabasi_albert_graph(200, 3, None)?;

    // Louvain community detection
    let communities = louvain_communities(&graph, None)?;
    println!("Louvain found {} communities", communities.len());

    // Label propagation
    let lp_communities = label_propagation_communities(&graph, None)?;
    println!("Label propagation found {} communities", lp_communities.len());

    Ok(())
}
```

### Graph Neural Networks

```rust
use scirs2_graph::gnn::{GCNLayer, GATLayer, GraphSAGELayer};
use scirs2_core::error::CoreResult;

fn gnn_example() -> CoreResult<()> {
    // Graph Convolutional Network layer
    let gcn = GCNLayer::new(64, 32)?;

    // Graph Attention Network layer
    let gat = GATLayer::new(64, 32, 4)?; // 4 attention heads

    // GraphSAGE layer with mean aggregation
    let sage = GraphSAGELayer::new(64, 32, "mean")?;

    println!("GNN layers initialized");
    Ok(())
}
```

### Node2Vec Embeddings

```rust
use scirs2_graph::embeddings::{Node2Vec, Node2VecConfig};
use scirs2_core::error::CoreResult;

fn embedding_example() -> CoreResult<()> {
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

    // let embeddings = Node2Vec::fit(&graph, config)?;
    // println!("Embedding shape: {:?}", embeddings.shape());
    Ok(())
}
```

### Subgraph Isomorphism (VF2)

```rust
use scirs2_graph::isomorphism::vf2_subgraph_isomorphism;
use scirs2_core::error::CoreResult;

fn isomorphism_example() -> CoreResult<()> {
    // Query whether pattern is a subgraph of target
    // let matches = vf2_subgraph_isomorphism(&pattern, &target, None)?;
    // println!("Found {} subgraph matches", matches.len());
    Ok(())
}
```

### Graph Visualization

```rust
use scirs2_graph::visualization::{render_svg, SvgConfig, Layout};
use scirs2_core::error::CoreResult;

fn viz_example() -> CoreResult<()> {
    // let graph = ...;
    // let config = SvgConfig {
    //     layout: Layout::ForceDirected,
    //     width: 800,
    //     height: 600,
    //     ..Default::default()
    // };
    // let svg = render_svg(&graph, &config)?;
    // std::fs::write("graph.svg", svg)?;
    Ok(())
}
```

### Temporal Graphs

```rust
use scirs2_graph::temporal::TemporalGraph;
use scirs2_core::error::CoreResult;

fn temporal_example() -> CoreResult<()> {
    let mut tg = TemporalGraph::new();
    // Add edges with timestamps
    // tg.add_temporal_edge(0, 1, 1.0, 0.0)?;  // (from, to, weight, time)
    // tg.add_temporal_edge(1, 2, 1.0, 5.0)?;

    // Query graph at specific time
    // let snapshot = tg.snapshot_at(3.0)?;
    Ok(())
}
```

## Feature Flags

| Flag | Description |
|------|-------------|
| `parallel` | Enable Rayon-based parallel processing for large graph algorithms |
| `simd` | Enable SIMD-accelerated numerical operations |

## Performance

- Multi-threaded algorithms via Rayon for large graphs (millions of nodes/edges)
- CSR representation for cache-efficient traversal
- Memory profiling tools built in
- Validated against NetworkX and igraph reference implementations
- 10-50x faster than NetworkX for most core operations

## Documentation

Full API reference: [docs.rs/scirs2-graph](https://docs.rs/scirs2-graph)

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.
