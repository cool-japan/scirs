#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]
//! # SciRS2 Graph - Graph Algorithms and Network Analysis
//!
//! **scirs2-graph** provides comprehensive graph algorithms and data structures for network analysis,
//! offering shortest paths, centrality measures, community detection, spectral methods, and graph
//! embeddings with parallel processing and memory-efficient implementations.
//!
//! ## 🎯 Key Features
//!
//! - **Graph Representations**: Directed, undirected, weighted, multi-graphs
//! - **Classic Algorithms**: BFS, DFS, Dijkstra, A*, Bellman-Ford, Floyd-Warshall
//! - **Centrality Measures**: Betweenness, closeness, eigenvector, PageRank
//! - **Community Detection**: Louvain, label propagation, spectral clustering
//! - **Spectral Methods**: Graph Laplacian, spectral clustering, graph embeddings
//! - **Network Metrics**: Clustering coefficient, diameter, average path length
//! - **Performance**: Parallel algorithms, memory profiling
//!
//! ## 📦 Module Overview
//!
//! | SciRS2 Module | NetworkX/igraph Equivalent | Description |
//! |---------------|----------------------------|-------------|
//! | `algorithms` | `networkx.algorithms` | Core graph algorithms (BFS, DFS, shortest paths) |
//! | `measures` | `networkx.centrality` | Centrality and network metrics |
//! | `spectral` | `scipy.sparse.linalg` | Spectral graph theory and embeddings |
//! | `generators` | `networkx.generators` | Graph generation (random, regular, etc.) |
//! | `io` | `networkx.readwrite` | Graph I/O (GML, GraphML, edge lists) |
//!
//! ## 🚀 Quick Start
//!
//! ```toml
//! [dependencies]
//! scirs2-graph = "0.5.1"
//! ```
//!
//! ```rust,no_run
//! use scirs2_graph::{Graph, breadth_first_search, betweenness_centrality};
//!
//! // Create graph and run BFS
//! let mut g: Graph<i32, f64> = Graph::new();
//! let n0 = g.add_node(0);
//! let n1 = g.add_node(1);
//! g.add_edge(0, 1, 1.0);
//! ```
//!
//! ## 🔒 Version: 0.5.1
//!
//! ## API Stability and Versioning
//!
//! scirs2-graph follows strict semantic versioning with clear stability guarantees:
//!
//! ### Stability Classifications
//! - ✅ **Stable**: Core APIs guaranteed until next major version (2.0.0)
//! - ⚠️ **Experimental**: May change in minor versions, marked with `#[cfg(feature = "experimental")]`
//! - 📋 **Deprecated**: Will be removed in next major version, use alternatives
//!
//! ### Version Guarantees
//! - **MAJOR** (1.x.x → 2.x.x): Breaking changes to stable APIs allowed
//! - **MINOR** (1.0.x → 1.1.x): New features, deprecations only (no breaks to stable APIs)
//! - **PATCH** (1.0.0 → 1.0.1): Bug fixes only, no API changes
//!
//! ### Stable Core APIs (v0.4.0+)
//! - Graph data structures (`Graph`, `DiGraph`, `MultiGraph`)
//! - Basic algorithms (traversal, shortest paths, connectivity)
//! - Graph generators and I/O operations
//! - Community detection with `_result` suffix functions
//! - Error handling and core types

#![warn(missing_docs)]

pub mod advanced;
pub mod algorithms;
pub mod attributes;
pub mod base;
pub mod compressed;
pub mod embeddings;
pub mod error;
pub mod generators;
pub mod graph_memory_profiler;
pub mod io;
pub mod layout;
pub mod link_prediction;
pub mod measures;
pub mod memory;
pub mod numerical_accuracy_validation;
pub mod parallel_algorithms;
pub mod performance;
pub mod spectral;
pub mod spectral_graph;
pub mod streaming;
pub mod temporal;
pub mod temporal_graph;
pub mod temporal_interval;
pub mod weighted;

// Graph alignment (IsoRank)
pub mod alignment;
// Graph condensation (coreset, distillation, evaluation)
pub mod condensation;
// Distributed graph algorithms
pub mod distributed;
// GPU-accelerated graph operations
pub mod gpu;
// Graph transformers (GraphGPS, Graphormer)
pub mod graph_transformer;
// Graph partitioning (METIS, FENNEL)
pub mod partitioning;
// Signed and directed graph embeddings
pub mod signed_directed;
// Self-supervised learning on graphs
pub mod ssl;

// Graph Neural Network layers and transformers
pub mod gnn;

// SIMD-accelerated graph operations
#[cfg(feature = "simd")]
pub mod simd_ops;

// Re-export stable APIs for 1.0
pub use algorithms::{
    articulation_points,
    astar_search,
    astar_search_digraph,
    // Centrality measures - stable for 1.0
    betweenness_centrality,
    bidirectional_search,
    bidirectional_search_digraph,

    // Core traversal algorithms - stable for 1.0
    breadth_first_search,
    breadth_first_search_digraph,
    bridges,
    // Flow algorithms - stable for 1.0
    capacity_scaling_max_flow,
    center_nodes,
    closeness_centrality,
    complement,
    // Connectivity analysis - stable for 1.0
    connected_components,
    cosine_similarity,

    depth_first_search,
    depth_first_search_digraph,
    // Graph properties - stable for 1.0
    diameter,
    // Shortest path algorithms - stable for 1.0
    dijkstra_path,
    dinic_max_flow,
    dinic_max_flow_full,
    edge_subgraph,
    edmonds_karp_max_flow,
    eigenvector_centrality,
    eulerian_type,
    floyd_warshall,
    floyd_warshall_digraph,
    fluid_communities_result,
    ford_fulkerson_max_flow,
    // Girvan-Newman community detection
    girvan_newman_communities_result,
    girvan_newman_result,
    greedy_coloring,
    greedy_modularity_optimization_result,
    hierarchical_communities_result,
    hopcroft_karp,
    infomap_communities,
    is_bipartite,

    isap_max_flow,
    // Similarity measures - stable for 1.0
    jaccard_similarity,
    k_core_decomposition,

    k_shortest_paths,

    label_propagation_result,
    line_digraph,
    line_graph,
    // Community detection algorithms - stable for 1.0
    louvain_communities_result,
    maximal_matching,
    // Matching algorithms - stable for 1.0
    maximum_bipartite_matching,
    maximum_cardinality_matching,
    min_cost_max_flow,
    min_cost_max_flow_graph,
    minimum_cut,

    // Spanning tree algorithms - stable for 1.0
    minimum_spanning_tree,

    minimum_st_cut,
    minimum_weight_bipartite_matching,
    modularity,

    modularity_optimization_result,
    multi_commodity_flow,
    multi_source_multi_sink_max_flow,
    pagerank,
    parallel_max_flow,
    personalized_pagerank,

    push_relabel_max_flow,
    push_relabel_max_flow_full,
    radius,
    random_walk,
    stable_marriage,

    strongly_connected_components,
    subdigraph,
    // Graph transformations - stable for 1.0
    subgraph,
    tensor_product,
    // Other algorithms - stable for 1.0
    topological_sort,
    transition_matrix,

    weight_filtered_subgraph,

    // Result types - stable for 1.0
    AStarResult,
    BipartiteMatching,
    BipartiteResult,
    CommunityResult,
    CommunityStructure,
    CostEdge,
    DendrogramLevel,
    EulerianType,
    GirvanNewmanConfig,
    GirvanNewmanResult,
    GraphColoring,
    HopcroftKarpResult,
    InfomapResult,
    MaxFlowResult,
    MaximumMatching,
    MinCostFlowResult,
    MotifType,
    MultiCommodityFlowResult,
};

// Parallel algorithms - stable for 1.0 when parallel feature is enabled
#[cfg(feature = "parallel")]
pub use algorithms::{
    parallel_label_propagation_result, parallel_louvain_communities_result, parallel_modularity,
};

// Parallel spectral operations - stable for 1.0 when parallel feature is enabled
#[cfg(feature = "parallel")]
pub use spectral::{parallel_laplacian, parallel_spectral_clustering};

// Experimental algorithms - unstable, may change in future versions
pub use algorithms::{
    // Isomorphism and advanced matching - experimental
    are_graphs_isomorphic,
    are_graphs_isomorphic_enhanced,
    // Complex graph products - experimental
    cartesian_product,

    // NP-hard problems - experimental (may be moved or optimized)
    chromatic_number,
    find_isomorphism,
    find_isomorphism_vf2,
    find_motifs,
    find_subgraph_matches,
    graph_edit_distance,

    has_hamiltonian_circuit,
    has_hamiltonian_path,
};

// Advanced coloring algorithms
pub use algorithms::coloring::{
    chromatic_bounds, dsatur_coloring, edge_coloring, greedy_coloring_with_order, list_coloring,
    verify_coloring, welsh_powell, ChromaticBounds, ColoringOrder, EdgeColoring, ListColoring,
};

// Advanced motif and subgraph analysis
pub use algorithms::motifs::{
    count_3node_motifs, count_4node_motifs, count_motif_frequencies, frequent_subgraph_mining,
    graphlet_degree_distribution, sample_motif_frequencies, vf2_subgraph_isomorphism,
    wl_subtree_kernel, FrequentPattern, GraphletDDResult, GraphletDegreeVector, VF2Result,
    WLKernelResult,
};

// Link prediction algorithms
pub use link_prediction::{
    adamic_adar_all, adamic_adar_index, common_neighbors_all, common_neighbors_score, compute_auc,
    evaluate_link_prediction, jaccard_coefficient, jaccard_coefficient_all, katz_similarity,
    katz_similarity_all, preferential_attachment, preferential_attachment_all,
    resource_allocation_all, resource_allocation_index, simrank, simrank_score,
    LinkPredictionConfig, LinkPredictionEval, LinkScore,
};

// Core graph types - stable for 1.0
pub use base::{
    BipartiteGraph, DiGraph, Edge, EdgeWeight, Graph, Hyperedge, Hypergraph, IndexType,
    MultiDiGraph, MultiGraph, Node,
};

// Error handling - stable for 1.0
pub use error::{ErrorContext, GraphError, Result};

// Graph generators - stable for 1.0
pub use generators::{
    barabasi_albert_graph, complete_graph, cycle_graph, erdos_renyi_graph, grid_2d_graph,
    grid_3d_graph, hexagonal_lattice_graph, path_graph, planted_partition_model,
    power_law_cluster_graph, random_geometric_graph, star_graph, stochastic_block_model,
    triangular_lattice_graph, two_community_sbm, watts_strogatz_graph,
};

// Graph measures - stable for 1.0
pub use measures::{
    centrality, clustering_coefficient, graph_density, hits_algorithm, katz_centrality,
    katz_centrality_digraph, pagerank_centrality, pagerank_centrality_digraph, CentralityType,
    HitsScores,
};

// Parallel measures - stable for 1.0 when parallel feature is enabled
#[cfg(feature = "parallel")]
pub use measures::parallel_pagerank_centrality;

// Spectral analysis - stable for 1.0
pub use spectral::{laplacian, normalized_cut, spectral_radius};

// Weighted operations - stable for 1.0
pub use weighted::{
    MultiWeight, NormalizationMethod, WeightStatistics, WeightTransform, WeightedOps,
};

// Attribute system - stable for 1.0
pub use attributes::{
    AttributeSummary, AttributeValue, AttributeView, AttributedDiGraph, AttributedGraph, Attributes,
};

// Memory optimization - stable for 1.0
pub use memory::{
    suggest_optimizations, BitPackedGraph, CSRGraph, CompressedAdjacencyList, FragmentationReport,
    HybridGraph, MemoryProfiler, MemorySample, MemoryStats, OptimizationSuggestions,
    OptimizedGraphBuilder,
};

// Performance monitoring - stable for 1.0
pub use performance::{
    LargeGraphIterator, LargeGraphOps, MemoryMetrics, ParallelConfig, PerformanceMonitor,
    PerformanceReport, StreamingGraphProcessor,
};

// I/O operations - stable for 1.0
pub use io::*;

// Graph embedding algorithms
pub use embeddings::{
    DeepWalk, DeepWalkConfig, DeepWalkMode, Embedding, EmbeddingModel, LINEConfig, LINEOrder,
    Node2Vec, Node2VecConfig, RandomWalk, RandomWalkGenerator, SpectralEmbedding,
    SpectralEmbeddingConfig, SpectralLaplacianType, LINE,
};

pub use layout::{circular_layout, hierarchical_layout, spectral_layout, spring_layout, Position};

// Stream-model temporal graph (f64 timestamps, centrality, motifs, community)
pub use temporal::{
    count_temporal_triangles, evolutionary_clustering, temporal_betweenness, temporal_closeness,
    temporal_motif_count, temporal_pagerank, DynamicCommunity, TemporalEdge as StreamTemporalEdge,
    TemporalGraph as StreamTemporalGraph, TemporalMotifCounts,
};

// Interval-model temporal graph (generic typed nodes, TimeInstant/TimeInterval)
pub use temporal_interval::{
    temporal_betweenness_centrality, temporal_reachability, TemporalGraph, TemporalPath,
    TimeInstant, TimeInterval,
};

// Advanced mode optimizations - experimental but stable API
pub use advanced::{
    create_advanced_processor, execute_with_advanced, AdvancedConfig, AdvancedProcessor,
    AdvancedStats, AlgorithmMetrics, GPUAccelerationContext, NeuralRLAgent, NeuromorphicProcessor,
};

// Graph memory profiling - experimental
pub use graph_memory_profiler::{
    AdvancedMemoryProfiler,
    EfficiencyAnalysis,
    MemoryProfile,
    MemoryProfilerConfig,
    MemoryStats as GraphMemoryStats, // Renamed to avoid conflict
    OptimizationOpportunity,
    OptimizationType,
};

// Numerical accuracy validation - experimental
pub use numerical_accuracy_validation::{
    create_comprehensive_validation_suite, run_quick_validation, AdvancedNumericalValidator,
    ValidationAlgorithm, ValidationConfig, ValidationReport, ValidationResult,
    ValidationTolerances,
};

// Compressed sparse row graph representation
pub use compressed::{AdjacencyList, CsrGraph, CsrGraphBuilder, NeighborIter};

// Parallel graph algorithms on CSR graphs
pub use parallel_algorithms::{
    bfs, connected_components as csr_connected_components, pagerank as csr_pagerank,
    triangle_count, BfsResult, ComponentsResult, PageRankConfig, PageRankResult,
    TriangleCountResult,
};

#[cfg(feature = "parallel")]
pub use parallel_algorithms::{
    parallel_bfs, parallel_connected_components, parallel_pagerank, parallel_triangle_count,
};

// Streaming graph processing
pub use streaming::{
    DegreeDistribution, DoulionTriangleCounter, EvictionStrategy, MascotTriangleCounter,
    MemoryBoundedConfig, MemoryBoundedProcessor, SlidingWindowGraph, StreamEdge, StreamEvent,
    StreamOp, StreamingGraph, TriangleCounterStats,
};

// Additional modules (previously orphaned, now wired)
pub mod attributed_graph;
pub mod compat;
pub mod heterogeneous;
pub mod knowledge_graph;
pub mod network_statistics;
pub mod sampling;
pub mod social;
pub mod traversal;
pub mod visualization;

// Previously orphaned subdirectory modules
pub mod algebraic;
pub mod centrality;
pub mod community;
pub mod diffusion;
pub mod domination;
pub mod dynamic;
pub mod flow;
pub mod hypergraph;
pub mod isomorphism;
pub mod planarity;
pub mod reliability;
pub mod signal_processing;
