# SciRS2 Graph

[![crates.io](https://img.shields.io/crates/v/scirs2-graph.svg)](https://crates.io/crates/scirs2-graph)
[![License](https://img.shields.io/crates/l/scirs2-graph.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-graph)](https://docs.rs/scirs2-graph)

Graph theory and network analysis module for the SciRS2 scientific computing library. This module provides data structures and algorithms for working with graphs and networks.

## Features

- **Graph Data Structures**: Efficient representations for directed and undirected graphs
- **Graph Algorithms**: Shortest paths, minimum spanning trees, flow algorithms
- **Graph Measures**: Centrality measures, clustering coefficients, graph similarity
- **Spectral Methods**: Spectral clustering, graph Laplacian, eigenvalue decomposition
- **I/O Functions**: Tools for reading and writing graphs in various formats

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-graph = "0.1.0-alpha.1"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-graph = { version = "0.1.0-alpha.1", features = ["parallel"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_graph::{base, algorithms, measures, spectral};
use scirs2_core::error::CoreResult;

// Create an undirected graph
fn create_undirected_graph() -> CoreResult<()> {
    let mut graph = base::UndirectedGraph::new();
    
    // Add nodes
    graph.add_node(1);
    graph.add_node(2);
    graph.add_node(3);
    
    // Add edges
    graph.add_edge(1, 2, 1.0)?;
    graph.add_edge(2, 3, 2.0)?;
    graph.add_edge(1, 3, 3.0)?;
    
    println!("Graph has {} nodes and {} edges", graph.node_count(), graph.edge_count());
    
    // Find shortest path
    let path = algorithms::shortest_path::dijkstra(&graph, 1, 3)?;
    println!("Shortest path from 1 to 3: {:?}", path);
    
    // Calculate degree centrality
    let centrality = measures::centrality::degree_centrality(&graph)?;
    println!("Degree centrality: {:?}", centrality);
    
    Ok(())
}

// Spectral clustering example
fn spectral_clustering_example() -> CoreResult<()> {
    // Create adjacency matrix for a graph
    let adj_matrix = ndarray::arr2(&[
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    ]);
    
    // Perform spectral clustering
    let n_clusters = 2;
    let clusters = spectral::spectral_clustering(&adj_matrix, n_clusters, None, None)?;
    
    println!("Cluster assignments: {:?}", clusters);
    
    Ok(())
}
```

## Components

### Graph Base

Core graph data structures:

```rust
use scirs2_graph::base::{
    Graph,                  // Generic graph trait
    DirectedGraph,          // Directed graph implementation
    UndirectedGraph,        // Undirected graph implementation
    WeightedGraph,          // Weighted graph trait
    Node,                   // Node trait
    Edge,                   // Edge trait
    Path,                   // Path trait
};
```

### Graph Algorithms

Various graph algorithms:

```rust
use scirs2_graph::algorithms::{
    // Shortest Paths
    dijkstra,               // Dijkstra's shortest path algorithm
    bellman_ford,           // Bellman-Ford algorithm
    floyd_warshall,         // Floyd-Warshall algorithm for all pairs shortest paths
    a_star,                 // A* search algorithm
    
    // Minimum Spanning Trees
    prim,                   // Prim's algorithm
    kruskal,                // Kruskal's algorithm
    
    // Flow Algorithms
    ford_fulkerson,         // Ford-Fulkerson max flow algorithm
    edmonds_karp,           // Edmonds-Karp algorithm
    dinic,                  // Dinic's algorithm
    
    // Connectivity
    is_connected,           // Check if graph is connected
    connected_components,   // Find connected components
    strongly_connected_components, // Find strongly connected components
    
    // Traversal
    breadth_first_search,   // BFS traversal
    depth_first_search,     // DFS traversal
    topological_sort,       // Topological sorting
    
    // Matching and Covering
    maximum_bipartite_matching, // Find maximum bipartite matching
    minimum_vertex_cover,   // Find minimum vertex cover
};
```

### Graph Measures

Functions for measuring graph properties:

```rust
use scirs2_graph::measures::{
    // Centrality Measures
    degree_centrality,      // Degree centrality
    betweenness_centrality, // Betweenness centrality
    closeness_centrality,   // Closeness centrality
    eigenvector_centrality, // Eigenvector centrality
    pagerank,               // PageRank algorithm
    
    // Clustering
    clustering_coefficient, // Clustering coefficient
    transitivity,           // Transitivity
    
    // Distance Measures
    eccentricity,           // Eccentricity
    diameter,               // Graph diameter
    radius,                 // Graph radius
    
    // Graph Similarity
    graph_edit_distance,    // Graph edit distance
    graph_kernel,           // Graph kernel methods
};
```

### Spectral Methods

Spectral graph theory algorithms:

```rust
use scirs2_graph::spectral::{
    laplacian_matrix,       // Compute the Laplacian matrix
    normalized_laplacian,   // Compute normalized Laplacian
    spectral_clustering,    // Spectral clustering algorithm
    spectral_embedding,     // Spectral embedding
    fiedler_vector,         // Compute the Fiedler vector
    algebraic_connectivity, // Compute algebraic connectivity
};
```

### Graph I/O

Functions for reading and writing graphs:

```rust
use scirs2_graph::io::{
    read_edgelist,          // Read edge list format
    write_edgelist,         // Write edge list format
    read_adjacency_matrix,  // Read adjacency matrix
    write_adjacency_matrix, // Write adjacency matrix
    read_gml,               // Read GML format
    write_gml,              // Write GML format
    read_graphml,           // Read GraphML format
    write_graphml,          // Write GraphML format
};
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.