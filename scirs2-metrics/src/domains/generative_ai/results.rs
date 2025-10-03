//! Result structures for generative AI metrics
//!
//! This module contains all the result structures returned by
//! the various generative AI and deep learning metrics.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Result of Inception Score computation
#[derive(Debug, Clone)]
pub struct InceptionScoreResult<F: Float> {
    pub mean_score: F,
    pub std_score: F,
    pub split_scores: Vec<F>,
}

/// Result of KID computation
#[derive(Debug, Clone)]
pub struct KIDResult<F: Float> {
    pub kid_estimate: F,
    pub kid_corrected: F,
    pub bias_correction: F,
    pub n_samples_real: usize,
    pub n_samples_fake: usize,
}

/// Result of InfoNCE computation
#[derive(Debug, Clone)]
pub struct InfoNCEResult<F: Float> {
    pub loss: F,
    pub accuracy: F,
    pub n_pairs: usize,
    pub temperature: F,
}

/// Result of linear probing evaluation
#[derive(Debug, Clone)]
pub struct LinearProbingResult<F: Float> {
    pub overall_accuracy: F,
    pub balanced_accuracy: F,
    pub per_class_accuracies: Vec<F>,
    pub n_classes: usize,
    pub n_test_samples: usize,
}

/// Result of representation rank analysis
#[derive(Debug, Clone)]
pub struct RepresentationRankResult<F: Float> {
    pub effective_rank: usize,
    pub participation_ratio: F,
    pub eigenvalues: Vec<F>,
    pub total_variance: F,
    pub explained_variance_ratio: F,
}

/// Result of clustering evaluation
#[derive(Debug, Clone)]
pub struct ClusteringResult<F: Float> {
    pub normalized_mutual_information: F,
    pub adjusted_rand_index: F,
    pub silhouette_score: F,
    pub cluster_assignments: Vec<usize>,
    pub n_clusters: usize,
}

/// Result of few-shot learning evaluation
#[derive(Debug, Clone)]
pub struct FewShotResult<F: Float> {
    pub overall_accuracy: F,
    pub balanced_accuracy: F,
    pub per_class_accuracies: Vec<F>,
    pub n_shot: usize,
    pub n_classes: usize,
    pub n_query_samples: usize,
}

/// Result of cross-modal retrieval evaluation
#[derive(Debug, Clone)]
pub struct CrossModalRetrievalResult<F: Float> {
    pub recall_at_k: HashMap<usize, F>,
    pub precision_at_k: HashMap<usize, F>,
    pub mean_reciprocal_rank: F,
    pub n_queries: usize,
    pub n_candidates: usize,
}

/// Result of multimodal alignment evaluation
#[derive(Debug, Clone)]
pub struct MultimodalAlignmentResult<F: Float> {
    pub mean_positive_similarity: F,
    pub mean_negative_similarity: F,
    pub alignment_gap: F,
    pub positive_std: F,
    pub negative_std: F,
    pub n_positive_pairs: usize,
    pub n_negative_pairs: usize,
}
