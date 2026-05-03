//! Hypergraph algorithms and simplicial complexes.
//!
//! This module provides four sub-modules:
//!
//! | Sub-module | Contents |
//! |---|---|
//! | [`core`] | Core data structures: [`IndexedHypergraph`], generic [`Hypergraph<N,E>`], clique/star/bipartite expansions |
//! | [`algorithms`] | Spectral clustering, hyperedge cuts, stationary distribution, betweenness centrality, s-walks |
//! | [`simplicial`] | [`SimplicialComplex`], boundary matrices, Betti numbers, Vietoris-Rips/Čech/nerve complexes |
//! | [`higher_order`] | [`MotifTensor`], topological features, [`CellularSheaf`] and Hodge Laplacian |
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_graph::hypergraph::{IndexedHypergraph, clique_expansion, SpectralClusteringResult};
//! use scirs2_graph::hypergraph::{SimplicialComplex, CellularSheaf};
//!
//! // Build a hypergraph
//! let mut hg = IndexedHypergraph::new(5);
//! hg.add_hyperedge(vec![0,1,2], 1.0).unwrap();
//! hg.add_hyperedge(vec![2,3,4], 1.0).unwrap();
//!
//! // Clique expansion → ordinary graph
//! let g = clique_expansion(&hg);
//! assert_eq!(g.node_count(), 5);
//!
//! // Simplicial complex topology
//! let mut sc = SimplicialComplex::new();
//! sc.add_simplex(vec![0,1,2]);
//! let betti = sc.betti_numbers();
//! assert_eq!(betti[0], 1); // connected
//! ```

pub mod algorithms;
pub mod attention;
pub mod core;
pub mod edge_prediction;
pub mod higher_order;
pub mod neural;
pub mod simplicial;

// Core structures and free functions
pub use core::{
    clique_expansion, hyperedge_centrality, hypergraph_clustering_coefficient,
    hypergraph_random_walk, hypergraph_random_walk_seeded, line_graph, Hyperedge, Hypergraph,
    IndexedHypergraph,
};

// Algorithms
pub use algorithms::{
    betweenness_centrality as hypergraph_betweenness_centrality, hyperedge_cut,
    s_betweenness_centrality, s_diameter, s_distance, s_reachability, spectral_clustering,
    stationary_distribution, CutResult, SpectralClusteringResult,
};

// Simplicial complexes
pub use simplicial::SimplicialComplex;

// Higher-order analysis
pub use higher_order::{
    directed_motif_tensor, trivial_sheaf_from_graph, CellularSheaf, MotifTensor,
    TopologicalFeatures,
};
