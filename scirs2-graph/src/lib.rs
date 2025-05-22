//! Graph processing module for SciRS2
//!
//! This module provides graph algorithms and data structures
//! for scientific computing and machine learning applications.
//!
//! ## Features
//!
//! - Basic graph representations and operations
//! - Graph algorithms (traversal, shortest paths, etc.)
//! - Network analysis (centrality measures, community detection)
//! - Spectral graph theory
//! - Support for graph neural networks

#![warn(missing_docs)]

extern crate blas;
extern crate openblas_src;

pub mod algorithms;
pub mod base;
pub mod error;
pub mod generators;
pub mod io;
pub mod measures;
pub mod spectral;

// Re-export important types and functions
pub use algorithms::{
    articulation_points, astar_search, astar_search_digraph, breadth_first_search,
    breadth_first_search_digraph, bridges, chromatic_number, connected_components,
    depth_first_search, depth_first_search_digraph, eulerian_type, find_eulerian_circuit,
    find_subgraph_matches, floyd_warshall, floyd_warshall_digraph, greedy_coloring,
    has_hamiltonian_circuit, has_hamiltonian_path, is_bipartite, is_isomorphic,
    k_core_decomposition, louvain_communities, max_flow, maximum_bipartite_matching,
    minimum_spanning_tree, shortest_path, shortest_path_digraph, strongly_connected_components,
    topological_sort, AStarResult, BipartiteMatching, BipartiteResult, CommunityStructure,
    EulerianType, GraphColoring, MaxFlowResult,
};
pub use base::{DiGraph, Edge, EdgeWeight, Graph, Node};
pub use error::{GraphError, Result};
pub use generators::{
    barabasi_albert_graph, complete_graph, cycle_graph, erdos_renyi_graph, grid_2d_graph,
    path_graph, star_graph, watts_strogatz_graph,
};
pub use measures::{
    centrality, clustering_coefficient, graph_density, katz_centrality, katz_centrality_digraph,
    pagerank_centrality, pagerank_centrality_digraph, CentralityType,
};
pub use spectral::{laplacian, normalized_cut, spectral_radius};
