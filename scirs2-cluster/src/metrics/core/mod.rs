//! Core clustering evaluation metrics
//!
//! This module contains the fundamental metrics for evaluating clustering quality,
//! organized into focused submodules for maintainability and clarity.

pub mod advanced;
pub mod basic;
pub mod information_theory;
pub mod stability;

// Re-export core metrics for backward compatibility
pub use basic::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score,
    homogeneity_completeness_v_measure, mean_silhouette_score, normalized_mutual_info,
};

pub use advanced::{aic_score, bic_score, dunn_index};

pub use information_theory::{
    information_cluster_quality, jensen_shannon_divergence, variation_of_information,
};

pub use stability::{
    bootstrap_confidence_interval, comprehensive_stability_analysis, StabilityConfig,
    StabilityResult,
};
