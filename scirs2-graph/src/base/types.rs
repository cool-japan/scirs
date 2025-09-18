//! Core types for graph structures

use std::hash::Hash;

/// A trait representing a node in a graph
pub trait Node: Clone + Eq + Hash + Send + Sync {}

/// Implements Node for common types
impl<T: Clone + Eq + Hash + Send + Sync> Node for T {}

/// A trait for edge weights in a graph
pub trait EdgeWeight: Clone + PartialOrd + Send + Sync {}

/// Implements EdgeWeight for common types
impl<T: Clone + PartialOrd + Send + Sync> EdgeWeight for T {}

/// Represents an edge in a graph
#[derive(Debug, Clone)]
pub struct Edge<N: Node, E: EdgeWeight> {
    /// Source node
    pub source: N,
    /// Target node
    pub target: N,
    /// Edge weight
    pub weight: E,
}
