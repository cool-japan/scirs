//! Information-theoretic clustering evaluation metrics
//!
//! This module provides comprehensive information-theoretic metrics for evaluating
//! clustering algorithms when ground truth labels are available. The metrics are
//! organized into focused submodules for better maintainability.
//!
//! # Modules
//!
//! - [`mutual_information`] - Core mutual information metrics and variants
//! - [`rand_metrics`] - Rand-based metrics (Rand index, ARI, Fowlkes-Mallows)
//! - [`v_measure`] - V-measure score and its components (homogeneity, completeness)
//! - [`advanced`] - Advanced information measures (Jensen-Shannon, variation of information)
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_cluster::metrics::information_theoretic::*;
//! use scirs2_core::ndarray::Array1;
//!
//! let true_labels = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
//! let pred_labels = Array1::from_vec(vec![0, 0, 1, 1, 1, 2]);
//!
//! // Basic metrics
//! let mi: f64 = mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
//! let nmi: f64 = normalized_mutual_info_score(true_labels.view(), pred_labels.view()).unwrap();
//! let ari: f64 = adjusted_rand_score(true_labels.view(), pred_labels.view()).unwrap();
//!
//! // V-measure components
//! let (h, c, v): (f64, f64, f64) = homogeneity_completeness_v_measure(
//!     true_labels.view(), pred_labels.view()
//! ).unwrap();
//!
//! // Advanced measures
//! let js: f64 = jensen_shannon_divergence(true_labels.view(), pred_labels.view()).unwrap();
//! let vi: f64 = normalized_variation_of_information(true_labels.view(), pred_labels.view()).unwrap();
//! ```

// Re-export submodules
pub mod advanced;
pub mod mutual_information;
pub mod rand_metrics;
pub mod v_measure;

// Re-export key functions for convenience
pub use advanced::{
    information_gain_ratio, jensen_shannon_divergence, normalized_variation_of_information,
    symmetric_information_coefficient, symmetric_uncertainty_coefficient, uncertainty_coefficient,
};

pub use mutual_information::{
    adjusted_mutual_info_score, conditional_mutual_information, entropy, mutual_info_score,
    normalized_mutual_info_score, normalized_mutual_info_score_with_method,
};

pub use rand_metrics::{
    adjusted_rand_score, fowlkes_mallows_score, pair_confusion_matrix, pair_precision_recall,
    rand_score,
};

pub use v_measure::{
    completeness_score, homogeneity_completeness_v_measure, homogeneity_score, v_measure_score,
    weighted_v_measure_score,
};
