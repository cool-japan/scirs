//! Core generative AI metrics suite and shared utilities
//!
//! This module provides the main GenerativeAISuite that orchestrates
//! all the different types of deep learning and generative AI metrics.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::domains::{DomainEvaluationResult, DomainMetrics};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::iter::Sum;

use super::contrastive_learning::ContrastiveLearningMetrics;
use super::foundation_models::FoundationModelMetrics;
use super::gan_evaluation::GANEvaluationMetrics;
use super::multimodal::MultimodalMetrics;
use super::self_supervised::SelfSupervisedMetrics;

/// Comprehensive generative AI metrics suite
pub struct GenerativeAISuite<F: Float> {
    /// GAN evaluation metrics
    pub gan_metrics: GANEvaluationMetrics<F>,
    /// Contrastive learning metrics
    pub contrastive_metrics: ContrastiveLearningMetrics<F>,
    /// Self-supervised learning metrics
    pub ssl_metrics: SelfSupervisedMetrics<F>,
    /// Foundation model metrics
    pub foundation_metrics: FoundationModelMetrics<F>,
    /// Multimodal metrics
    pub multimodal_metrics: MultimodalMetrics<F>,
}

impl<
        F: Float + scirs2_core::numeric::FromPrimitive + Sum + scirs2_core::ndarray::ScalarOperand,
    > Default for GenerativeAISuite<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        F: Float + scirs2_core::numeric::FromPrimitive + Sum + scirs2_core::ndarray::ScalarOperand,
    > GenerativeAISuite<F>
{
    /// Create new generative AI metrics suite
    pub fn new() -> Self {
        Self {
            gan_metrics: GANEvaluationMetrics::new(),
            contrastive_metrics: ContrastiveLearningMetrics::new(),
            ssl_metrics: SelfSupervisedMetrics::new(),
            foundation_metrics: FoundationModelMetrics::new(),
            multimodal_metrics: MultimodalMetrics::new(),
        }
    }

    /// Get GAN evaluation metrics
    pub fn gan_metrics(&self) -> &GANEvaluationMetrics<F> {
        &self.gan_metrics
    }

    /// Get contrastive learning metrics
    pub fn contrastive_metrics(&self) -> &ContrastiveLearningMetrics<F> {
        &self.contrastive_metrics
    }

    /// Get self-supervised learning metrics
    pub fn ssl_metrics(&self) -> &SelfSupervisedMetrics<F> {
        &self.ssl_metrics
    }

    /// Get foundation model metrics
    pub fn foundation_metrics(&self) -> &FoundationModelMetrics<F> {
        &self.foundation_metrics
    }

    /// Get multimodal metrics
    pub fn multimodal_metrics(&self) -> &MultimodalMetrics<F> {
        &self.multimodal_metrics
    }
}

impl<
        F: Float + scirs2_core::numeric::FromPrimitive + Sum + scirs2_core::ndarray::ScalarOperand,
    > DomainMetrics for GenerativeAISuite<F>
{
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Generative AI & Deep Learning"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "inception_score",
            "fid_score",
            "kid_score",
            "lpips_distance",
            "uniformity",
            "alignment",
            "infonce_loss",
            "linear_probing_accuracy",
            "clustering_nmi",
            "representation_rank",
            "zero_shot_accuracy",
            "few_shot_accuracy",
            "cross_modal_retrieval",
            "multimodal_alignment",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "inception_score",
            "Inception Score for evaluating GAN quality",
        );
        descriptions.insert(
            "fid_score",
            "Fréchet Inception Distance for distribution comparison",
        );
        descriptions.insert("kid_score", "Kernel Inception Distance for sample quality");
        descriptions.insert(
            "lpips_distance",
            "Learned Perceptual Image Patch Similarity",
        );
        descriptions.insert("uniformity", "Uniformity of learned representations");
        descriptions.insert(
            "alignment",
            "Alignment between positive pairs in contrastive learning",
        );
        descriptions.insert("infonce_loss", "InfoNCE contrastive loss value");
        descriptions.insert(
            "linear_probing_accuracy",
            "Linear probe classification accuracy",
        );
        descriptions.insert(
            "clustering_nmi",
            "Normalized Mutual Information for clustering",
        );
        descriptions.insert(
            "representation_rank",
            "Effective rank of representation matrix",
        );
        descriptions.insert("zero_shot_accuracy", "Zero-shot classification accuracy");
        descriptions.insert("few_shot_accuracy", "Few-shot learning performance");
        descriptions.insert("cross_modal_retrieval", "Cross-modal retrieval performance");
        descriptions.insert(
            "multimodal_alignment",
            "Multimodal representation alignment",
        );
        descriptions
    }
}
