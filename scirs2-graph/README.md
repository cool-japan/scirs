# SciRS2 Graph

[![crates.io](https://img.shields.io/crates/v/scirs2-graph.svg)](https://crates.io/crates/scirs2-graph)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-graph)](https://docs.rs/scirs2-graph)

**Production-ready graph processing module for the SciRS2 scientific computing library.**

This is the **final alpha release (0.1.0-alpha.6)** featuring a comprehensive, high-performance graph theory and network analysis library designed for scientific computing and machine learning applications.

## 🚀 Production Features

### Core Graph Types
- **Standard Graphs**: Efficient directed and undirected graph implementations
- **Specialized Graphs**: Bipartite, multi-graphs with parallel edges, hypergraphs
- **Temporal Graphs**: Time-based graph operations and analysis
- **Attributed Graphs**: Rich metadata support with flexible attribute systems

### Comprehensive Algorithm Suite (90+ algorithms implemented)

**Graph Traversal & Search**
- Breadth-first search (BFS), Depth-first search (DFS)
- Bidirectional search, Priority-first search, A* pathfinding

**Shortest Paths & Connectivity**
- Dijkstra, Floyd-Warshall, k-shortest paths algorithms
- Connected components, strongly connected components
- Articulation points, bridges, topological sorting

**Network Flow & Matching**
- Maximum flow (Ford-Fulkerson, Dinic, push-relabel)
- Minimum cut algorithms, bipartite matching
- Maximum cardinality matching, stable marriage problem

**Centrality & Importance**
- Degree, betweenness, closeness, eigenvector centrality  
- PageRank, personalized PageRank, Katz centrality, HITS algorithm

**Community Detection**
- Modularity optimization, Louvain method, label propagation
- Infomap algorithm, fluid communities, hierarchical clustering

**Graph Analytics**
- Spectral clustering, Laplacian matrix operations
- Graph isomorphism (VF2 algorithm), subgraph matching
- Motif detection (triangles, cliques, stars), k-core decomposition

### Performance & Scale
- **Multi-threaded**: Rayon-powered parallel processing for large graphs
- **Memory Efficient**: Cache-friendly data structures and streaming algorithms  
- **SIMD Accelerated**: Optimized numerical operations where applicable
- **Large Graph Support**: Handles graphs with millions of nodes/edges

### Rich I/O Support
Multiple format support with robust parsing:
- **GraphML**, **GML**, **DOT** (Graphviz), **JSON**
- **Edge lists**, **Adjacency lists**, **Matrix Market** format
- **Bidirectional conversion** with comprehensive error handling

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-graph = "0.1.0-alpha.6"
```

For performance features:
```toml
[dependencies]
scirs2-graph = { version = "0.1.0-alpha.6", features = ["parallel"] }
```

## Quick Start

### Basic Graph Operations

```rust
use scirs2_graph::{Graph, algorithms, measures};
use scirs2_core::error::CoreResult;

fn main() -> CoreResult<()> {
    // Create and populate graph
    let mut graph = Graph::new();
    graph.add_node(1);
    graph.add_node(2); 
    graph.add_node(3);
    graph.add_edge(1, 2, 1.0)?;
    graph.add_edge(2, 3, 2.0)?;
    graph.add_edge(1, 3, 3.0)?;

    // Graph analysis
    println!("Nodes: {}, Edges: {}", graph.node_count(), graph.edge_count());
    
    // Shortest path
    let path = algorithms::shortest_path(&graph, 1, 3)?;
    println!("Shortest path 1→3: {:?}", path);
    
    // Centrality analysis
    let centrality = measures::centrality::degree_centrality(&graph)?;
    println!("Degree centrality: {:?}", centrality);
    
    Ok(())
}
```

### Advanced Analytics

```rust
use scirs2_graph::{algorithms, spectral, generators};

fn advanced_analysis() -> CoreResult<()> {
    // Generate test network
    let graph = generators::barabasi_albert_graph(100, 3)?;
    
    // Community detection
    let communities = algorithms::louvain_communities(&graph)?;
    println!("Found {} communities", communities.len());
    
    // Spectral clustering  
    let adj_matrix = graph.adjacency_matrix();
    let clusters = spectral::spectral_clustering(&adj_matrix, 5, None, None)?;
    
    // Centrality analysis
    let pagerank = algorithms::pagerank(&graph, 0.85, None)?;
    let betweenness = algorithms::betweenness_centrality(&graph)?;
    
    Ok(())
}
```

### Graph I/O Operations

```rust
use scirs2_graph::io;

fn graph_io_example() -> CoreResult<()> {
    // Read from various formats
    let graph1 = io::read_graphml("network.graphml")?;
    let graph2 = io::read_gml("network.gml")?;
    let graph3 = io::read_edgelist("edges.txt", false)?;
    
    // Write to different formats
    io::write_dot(&graph1, "output.dot")?;
    io::write_json(&graph1, "output.json")?;
    
    Ok(())
}
```

## API Overview

### Core Modules

```rust
use scirs2_graph::{
    // Graph types
    Graph, DiGraph, BipartiteGraph, Hypergraph, TemporalGraph,
    
    // Algorithms by category
    algorithms::{
        shortest_path::dijkstra,
        connectivity::connected_components,
        community::louvain_communities,
        centrality::pagerank,
        flow::dinic_max_flow,
        matching::maximum_bipartite_matching,
    },
    
    // Graph measures
    measures::{
        degree_centrality,
        clustering_coefficient, 
        graph_density,
    },
    
    // Spectral methods
    spectral::{
        laplacian_matrix,
        spectral_clustering,
        normalized_cut,
    },
    
    // Graph generation
    generators::{
        erdos_renyi_graph,
        barabasi_albert_graph,
        watts_strogatz_graph,
    },
    
    // I/O operations
    io::{
        read_graphml, write_graphml,
        read_gml, write_gml,
        read_edgelist, write_edgelist,
    },
};
```

## Performance

This library is designed for production use with:
- **High Performance**: Competitive with NetworkX and igraph
- **Memory Efficiency**: Optimized data structures for large graphs
- **Parallel Processing**: Multi-threaded algorithms scale with available cores
- **Test Coverage**: 269+ unit tests ensuring reliability

## Compatibility

- **SciPy Compatibility**: API designed to match SciPy's graph functionality where possible
- **NetworkX Migration**: Similar interface patterns for easy migration
- **Rust Ecosystem**: Integrates well with ndarray, rayon, and other scientific Rust crates

## Project Status

✅ **Production Ready**: Comprehensive feature set with extensive testing  
✅ **API Stable**: Final alpha with stable public interface  
✅ **Well Documented**: Complete API documentation with examples  
✅ **Performance Optimized**: Benchmarked and optimized for real-world use  

**Next**: Version 1.0 release with additional performance optimizations and extended documentation.

## Contributing

See the [project root CLAUDE.md](../CLAUDE.md) for development guidelines and contribution instructions.

## License

This project is dual-licensed under:
- [MIT License](../LICENSE-MIT)  
- [Apache License Version 2.0](../LICENSE-APACHE)

See the [LICENSE](../LICENSE) file for details.