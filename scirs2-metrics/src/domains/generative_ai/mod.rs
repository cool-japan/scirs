//! Advanced generative AI and deep learning metrics
//!
//! This module provides comprehensive evaluation metrics for modern deep learning
//! approaches including:
//! - **GAN Evaluation**: Inception Score, FID, KID, IS, LPIPS
//! - **Contrastive Learning**: Uniformity, Alignment, InfoNCE
//! - **Self-Supervised Learning**: Linear probing, clustering metrics, representation quality
//! - **Foundation Models**: Zero-shot evaluation, few-shot performance
//! - **Multimodal Models**: Cross-modal retrieval, alignment metrics

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

// Module declarations
pub mod contrastive_learning;
pub mod core;
pub mod foundation_models;
pub mod gan_evaluation;
pub mod multimodal;
pub mod results;
pub mod self_supervised;

// Re-export core types for backward compatibility
pub use core::GenerativeAISuite;

// Re-export all metric types
pub use contrastive_learning::ContrastiveLearningMetrics;
pub use foundation_models::FoundationModelMetrics;
pub use gan_evaluation::GANEvaluationMetrics;
pub use multimodal::MultimodalMetrics;
pub use self_supervised::SelfSupervisedMetrics;

// Re-export all result types
pub use results::{
    ClusteringResult, CrossModalRetrievalResult, FewShotResult, InceptionScoreResult,
    InfoNCEResult, KIDResult, LinearProbingResult, MultimodalAlignmentResult,
    RepresentationRankResult,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::DomainMetrics;
    use scirs2_core::ndarray::array;

    fn mock_inception_features() -> scirs2_core::ndarray::Array2<f64> {
        array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ]
    }

    #[test]
    fn test_inception_score() {
        let gan_metrics = GANEvaluationMetrics::<f64>::new();
        let features = mock_inception_features();

        let result = gan_metrics.inception_score(&features, 2).unwrap();

        assert!(result.mean_score > 0.0);
        assert!(result.std_score >= 0.0);
        assert_eq!(result.split_scores.len(), 2);
    }

    #[test]
    fn test_fid_score() {
        let gan_metrics = GANEvaluationMetrics::<f64>::new();
        let real_features = mock_inception_features();
        let fake_features = array![
            [1.1, 2.1, 3.1, 4.1],
            [2.1, 3.1, 4.1, 5.1],
            [3.1, 4.1, 5.1, 6.1],
            [4.1, 5.1, 6.1, 7.1],
        ];

        let fid = gan_metrics
            .frechet_inception_distance(&real_features, &fake_features)
            .unwrap();

        assert!(fid >= 0.0);
    }

    #[test]
    fn test_uniformity() {
        let contrastive_metrics = ContrastiveLearningMetrics::<f64>::new();
        let representations = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],];

        let uniformity = contrastive_metrics
            .uniformity(&representations, 2.0)
            .unwrap();

        assert!(uniformity.is_finite());
    }

    #[test]
    fn test_alignment() {
        let contrastive_metrics = ContrastiveLearningMetrics::<f64>::new();
        let anchors = array![[1.0, 0.0], [0.0, 1.0],];
        let positives = array![[0.9, 0.1], [0.1, 0.9],];

        let alignment = contrastive_metrics
            .alignment(&anchors, &positives, 2.0)
            .unwrap();

        assert!(alignment >= 0.0);
    }

    #[test]
    fn test_linear_probing() {
        let ssl_metrics = SelfSupervisedMetrics::<f64>::new();

        let train_repr = array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0],];
        let train_labels = array![0, 1, 0, 1];

        let test_repr = array![[1.1, 0.1], [0.1, 1.1],];
        let test_labels = array![0, 1];

        let result = ssl_metrics
            .linear_probing_accuracy(&train_repr, &train_labels, &test_repr, &test_labels)
            .unwrap();

        assert!(result.overall_accuracy >= 0.0);
        assert!(result.overall_accuracy <= 1.0);
        assert_eq!(result.n_classes, 2);
    }

    #[test]
    fn test_zero_shot_accuracy() {
        let foundation_metrics = FoundationModelMetrics::<f64>::new();

        let predictions = array![0.8, 0.3, 0.9, 0.1];
        let targets = array![1, 0, 1, 0];

        let accuracy = foundation_metrics
            .zero_shot_accuracy(&predictions, &targets)
            .unwrap();

        assert!(accuracy >= 0.0);
        assert!(accuracy <= 1.0);
    }

    #[test]
    fn test_cross_modal_retrieval() {
        let multimodal_metrics = MultimodalMetrics::<f64>::new();

        let query_emb = array![[1.0, 0.0], [0.0, 1.0],];
        let candidate_emb = array![
            [0.9, 0.1], // Similar to query 0
            [0.1, 0.9], // Similar to query 1
            [0.5, 0.5], // Neutral
        ];
        let gt_pairs = vec![(0, 0), (1, 1)];

        let result = multimodal_metrics
            .cross_modal_retrieval(&query_emb, &candidate_emb, &gt_pairs)
            .unwrap();

        assert!(result.mean_reciprocal_rank >= 0.0);
        assert!(result.mean_reciprocal_rank <= 1.0);
        assert_eq!(result.n_queries, 2);
        assert_eq!(result.n_candidates, 3);
    }

    #[test]
    fn test_generative_ai_suite() {
        let suite = GenerativeAISuite::<f64>::new();

        assert_eq!(suite.domain_name(), "Generative AI & Deep Learning");
        assert!(!suite.available_metrics().is_empty());
        assert!(!suite.metric_descriptions().is_empty());
    }
}
