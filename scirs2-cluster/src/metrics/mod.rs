//! Clustering evaluation metrics
//!
//! This module provides comprehensive metrics for evaluating clustering algorithm performance.
//! Metrics are organized into focused submodules for better maintainability.
//!
//! # Basic Metrics
//! - Silhouette coefficient for measuring cluster cohesion and separation
//! - Davies-Bouldin index for evaluating cluster separation
//! - Calinski-Harabasz index for measuring between-cluster vs within-cluster variance
//! - Adjusted Rand Index and Normalized Mutual Information for comparing clusterings
//!
//! # Advanced Metrics
//! - Dunn index for cluster separation quality
//! - BIC/AIC scores for model selection
//!
//! # Information-Theoretic Metrics
//! - Jensen-Shannon divergence between clusterings
//! - Variation of Information
//! - Information-theoretic cluster quality measures
//!
//! # Stability Analysis
//! - Bootstrap confidence intervals
//! - Multi-scale stability analysis
//! - Noise perturbation analysis

// Core metrics organized into focused modules
pub mod core;

// Specialized metrics modules
pub mod information_theoretic;
mod silhouette;

// Re-export core metrics for backward compatibility and convenience
pub use core::{
    // Basic metrics
    adjusted_rand_index,
    // Advanced metrics
    aic_score,
    bic_score,
    // Stability analysis
    bootstrap_confidence_interval,
    calinski_harabasz_score,
    comprehensive_stability_analysis,
    davies_bouldin_score,
    dunn_index,
    homogeneity_completeness_v_measure,
    // Information-theoretic metrics
    information_cluster_quality,
    jensen_shannon_divergence,
    mean_silhouette_score,
    normalized_mutual_info,
    variation_of_information,
    StabilityConfig,
    StabilityResult,
};

// Re-export silhouette metrics
pub use silhouette::{silhouette_samples, silhouette_score};

// Re-export information-theoretic metrics from existing module
pub use information_theoretic::{
    adjusted_mutual_info_score, adjusted_rand_score, completeness_score, homogeneity_score,
    mutual_info_score, normalized_mutual_info_score, normalized_variation_of_information,
    v_measure_score,
};

// Re-export specialized modules
pub use core::advanced;
pub use core::basic;
pub use core::information_theory;
pub use core::stability;
