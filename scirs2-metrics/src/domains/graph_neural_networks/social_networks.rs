//! Social network analysis metrics for Graph Neural Networks
//!
//! This module provides metrics for evaluating social network analysis tasks
//! including influence prediction, social role classification, and recommendation systems.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Social network analysis metrics
#[derive(Debug, Clone)]
pub struct SocialNetworkMetrics {
    /// Influence prediction metrics
    pub influence_prediction: InfluencePredictionMetrics,
    /// Social role classification metrics
    pub social_role: SocialRoleMetrics,
    /// Social recommendation metrics
    pub social_recommendation: SocialRecommendationMetrics,
    /// Information diffusion metrics
    pub information_diffusion: InformationDiffusionMetrics,
}

/// Influence prediction evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluencePredictionMetrics {
    /// Influence spread accuracy
    pub spread_accuracy: f64,
    /// Influence ranking correlation
    pub ranking_correlation: f64,
    /// Top-K influencer precision
    pub top_k_precision: HashMap<usize, f64>,
}

impl Default for InfluencePredictionMetrics {
    fn default() -> Self {
        Self {
            spread_accuracy: 0.0,
            ranking_correlation: 0.0,
            top_k_precision: HashMap::new(),
        }
    }
}

impl InfluencePredictionMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Social role classification metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialRoleMetrics {
    /// Role classification accuracy
    pub role_accuracy: f64,
    /// Role assignment F1 score
    pub role_f1: f64,
    /// Per-role performance
    pub per_role_metrics: HashMap<String, (f64, f64, f64)>, // (precision, recall, f1)
}

impl Default for SocialRoleMetrics {
    fn default() -> Self {
        Self {
            role_accuracy: 0.0,
            role_f1: 0.0,
            per_role_metrics: HashMap::new(),
        }
    }
}

impl SocialRoleMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Social recommendation evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialRecommendationMetrics {
    /// Recommendation precision
    pub precision: f64,
    /// Recommendation recall
    pub recall: f64,
    /// Recommendation F1 score
    pub f1_score: f64,
    /// NDCG score
    pub ndcg: f64,
    /// Diversity score
    pub diversity: f64,
}

impl Default for SocialRecommendationMetrics {
    fn default() -> Self {
        Self {
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            ndcg: 0.0,
            diversity: 0.0,
        }
    }
}

impl SocialRecommendationMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Information diffusion prediction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationDiffusionMetrics {
    /// Cascade prediction accuracy
    pub cascade_accuracy: f64,
    /// Adoption time prediction MAE
    pub adoption_time_mae: f64,
    /// Diffusion size prediction error
    pub diffusion_size_error: f64,
}

impl Default for InformationDiffusionMetrics {
    fn default() -> Self {
        Self {
            cascade_accuracy: 0.0,
            adoption_time_mae: 0.0,
            diffusion_size_error: 0.0,
        }
    }
}

impl InformationDiffusionMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SocialNetworkMetrics {
    pub fn new() -> Self {
        Self {
            influence_prediction: InfluencePredictionMetrics::new(),
            social_role: SocialRoleMetrics::new(),
            social_recommendation: SocialRecommendationMetrics::new(),
            information_diffusion: InformationDiffusionMetrics::new(),
        }
    }
}

impl Default for SocialNetworkMetrics {
    fn default() -> Self {
        Self::new()
    }
}
