//! Utility modules for clustering algorithms
//!
//! This module contains various utility functions and helpers used throughout
//! the clustering library.

pub mod contingency;

// Re-export key utilities
pub use contingency::build_contingency_matrix;
