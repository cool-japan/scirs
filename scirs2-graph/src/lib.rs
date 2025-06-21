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

// Temporarily commenting out OpenBLAS to fix build issues
// extern crate blas;
// extern crate openblas_src;

pub mod algorithms;
pub mod attributes;
pub mod base;
pub mod embeddings;
pub mod error;
pub mod generators;
pub mod io;
pub mod layout;
pub mod measures;
pub mod performance;
pub mod spectral;
pub mod temporal;
pub mod weighted;

// Re-export important types and functions
pub use algorithms::{
    are_graphs_isomorphic, are_graphs_isomorphic_enhanced, articulation_points, astar_search,
    astar_search_digraph, betweenness_centrality, bidirectional_search,
    bidirectional_search_digraph, breadth_first_search, breadth_first_search_digraph, bridges,
    cartesian_product, center_nodes, chromatic_number, closeness_centrality, complement,
    connected_components, cosine_similarity, depth_first_search, depth_first_search_digraph,
    diameter, dinic_max_flow, edge_subgraph, eigenvector_centrality, eulerian_type,
    find_isomorphism, find_isomorphism_vf2, find_motifs, find_subgraph_matches, floyd_warshall,
    floyd_warshall_digraph, fluid_communities, graph_edit_distance, greedy_coloring,
    greedy_modularity_optimization, has_hamiltonian_circuit, has_hamiltonian_path,
    hierarchical_communities, infomap_communities, is_bipartite, jaccard_similarity,
    k_core_decomposition, k_shortest_paths, label_propagation, line_digraph, line_graph,
    louvain_communities, maximal_matching, maximum_bipartite_matching,
    maximum_cardinality_matching, minimum_cut, minimum_spanning_tree,
    minimum_weight_bipartite_matching, modularity, modularity_optimization, pagerank,
    personalized_pagerank, push_relabel_max_flow, radius, random_walk, shortest_path,
    shortest_path_digraph, stable_marriage, strongly_connected_components, subdigraph, subgraph,
    tensor_product, topological_sort, transition_matrix, weight_filtered_subgraph, AStarResult,
    BipartiteMatching, BipartiteResult, CommunityStructure, EulerianType, GraphColoring,
    InfomapResult, MaximumMatching, MotifType,
};
pub use attributes::{
    AttributeSummary, AttributeValue, AttributeView, AttributedDiGraph, AttributedGraph, Attributes,
};
pub use base::{
    BipartiteGraph, DiGraph, Edge, EdgeWeight, Graph, Hyperedge, Hypergraph, IndexType,
    MultiDiGraph, MultiGraph, Node,
};
pub use embeddings::{
    DeepWalk, DeepWalkConfig, Embedding, EmbeddingModel, Node2Vec, Node2VecConfig, RandomWalk,
    RandomWalkGenerator,
};
pub use error::{GraphError, Result};
pub use generators::{
    barabasi_albert_graph, complete_graph, cycle_graph, erdos_renyi_graph, grid_2d_graph,
    grid_3d_graph, hexagonal_lattice_graph, path_graph, planted_partition_model, star_graph,
    stochastic_block_model, triangular_lattice_graph, two_community_sbm, watts_strogatz_graph,
};
pub use layout::{circular_layout, hierarchical_layout, spectral_layout, spring_layout, Position};
pub use measures::{
    centrality, clustering_coefficient, graph_density, hits_algorithm, katz_centrality,
    katz_centrality_digraph, pagerank_centrality, pagerank_centrality_digraph, CentralityType,
    HitsScores,
};
pub use performance::{
    LargeGraphIterator, LargeGraphOps, ParallelConfig, PerformanceMonitor, StreamingGraphProcessor,
};
pub use spectral::{laplacian, normalized_cut, spectral_radius};
pub use temporal::{
    temporal_betweenness_centrality, temporal_reachability, TemporalGraph, TemporalPath,
    TimeInstant, TimeInterval,
};
pub use weighted::{
    MultiWeight, NormalizationMethod, WeightStatistics, WeightTransform, WeightedOps,
};
