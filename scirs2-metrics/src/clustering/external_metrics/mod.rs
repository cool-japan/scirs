//! External clustering metrics module
//!
//! This module provides metrics for evaluating clustering results against known ground truth labels.
//! These metrics compare clustering assignments with reference labels to measure agreement between partitions.
//!
//! ## Available Metrics
//!
//! - **Adjusted Rand Index (ARI)**: Measures similarity between clusterings, adjusted for chance
//! - **Normalized Mutual Information (NMI)**: Measures shared information between clusterings, normalized to [0,1]
//! - **Adjusted Mutual Information (AMI)**: Similar to NMI, but adjusted for chance agreements
//! - **Homogeneity, Completeness, V-Measure**: Related metrics measuring cluster purity and assignment completeness
//! - **Fowlkes-Mallows Score**: Geometric mean of precision and recall for cluster pairs

mod adjusted_rand;
mod fowlkes_mallows;
mod homogeneity;
mod mutual_information;

pub use adjusted_rand::*;
pub use fowlkes_mallows::*;
pub use homogeneity::*;
pub use mutual_information::*;

// Re-export all public items for backward compatibility
