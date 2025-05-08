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
pub mod io;
pub mod measures;
pub mod spectral;

// Re-export important types and functions
pub use algorithms::{connected_components, minimum_spanning_tree, shortest_path};
pub use base::{DiGraph, Edge, EdgeWeight, Graph, Node};
pub use error::{GraphError, Result};
pub use measures::{centrality, clustering_coefficient, graph_density};
pub use spectral::laplacian;
