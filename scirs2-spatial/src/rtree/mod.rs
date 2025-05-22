//! R-tree implementation for efficient spatial indexing
//!
//! This module provides an implementation of the R-tree data structure
//! for efficient spatial indexing and querying in 2D and higher dimensional spaces.
//!
//! R-trees are tree data structures used for spatial access methods that group
//! nearby objects and represent them with minimum bounding rectangles (MBRs)
//! in the next higher level of the tree. They are useful for spatial databases,
//! GIS systems, and other applications involving multidimensional data.
//!
//! This implementation supports:
//! - Insertion and deletion of data points
//! - Range queries
//! - Nearest neighbor queries
//! - Spatial joins

mod deletion;
mod insertion;
mod node;
mod optimization;
mod query;

// Re-export all public components from submodules
pub use node::{RTree, Rectangle};

// Imports used across modules
// Module uses individual imports in submodules

// Module-wide common type definitions and helper functions can go here
