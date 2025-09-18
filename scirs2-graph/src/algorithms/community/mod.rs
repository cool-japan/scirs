//! Community detection algorithms
//!
//! This module contains algorithms for detecting community structure in graphs.

// Community detection algorithm modules
pub mod fluid;
pub mod hierarchical;
pub mod infomap;
pub mod label_propagation;
pub mod louvain;
pub mod modularity;
pub mod parallel;
pub mod types;

// Re-export core types for backward compatibility and convenience
pub use types::{CommunityResult, CommunityStructure};

// Re-export all main algorithm functions
pub use fluid::{fluid_communities, fluid_communities_result};
pub use hierarchical::{hierarchical_communities, hierarchical_communities_result};
pub use infomap::{infomap_communities, InfomapResult};
pub use label_propagation::{label_propagation, label_propagation_result};
pub use louvain::{louvain_communities, louvain_communities_result};
pub use modularity::{
    greedy_modularity_optimization, greedy_modularity_optimization_result, modularity,
    modularity_optimization, modularity_optimization_result,
};
pub use parallel::{parallel_louvain_communities, parallel_louvain_communities_result};

#[cfg(feature = "parallel")]
pub use parallel::{parallel_label_propagation_result, parallel_modularity};
