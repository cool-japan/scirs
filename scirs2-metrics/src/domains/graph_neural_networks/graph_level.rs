//! Graph-level task evaluation metrics for Graph Neural Networks
//!
//! This module provides metrics for evaluating graph-level tasks such as
//! graph classification, regression, property prediction, and similarity assessment.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::ClassMetrics;
use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Graph-level task evaluation metrics
#[derive(Debug, Clone)]
pub struct GraphLevelMetrics {
    /// Graph classification metrics
    pub classification: GraphClassificationMetrics,
    /// Graph regression metrics
    pub regression: GraphRegressionMetrics,
    /// Graph property prediction metrics
    pub property_prediction: GraphPropertyMetrics,
    /// Graph similarity metrics
    pub similarity_metrics: GraphSimilarityMetrics,
}

/// Graph classification evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphClassificationMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Macro F1 score
    pub macro_f1: f64,
    /// Micro F1 score
    pub micro_f1: f64,
    /// Per-class metrics
    pub per_class_metrics: HashMap<String, ClassMetrics>,
    /// ROC AUC (for binary classification)
    pub roc_auc: Option<f64>,
    /// Cross-validation scores
    pub cv_scores: Vec<f64>,
}

impl Default for GraphClassificationMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            macro_f1: 0.0,
            micro_f1: 0.0,
            per_class_metrics: HashMap::new(),
            roc_auc: None,
            cv_scores: Vec::new(),
        }
    }
}

impl GraphClassificationMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Graph regression evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRegressionMetrics {
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// R-squared score
    pub r2_score: f64,
    /// Mean Absolute Percentage Error
    pub mape: f64,
    /// Explained variance score
    pub explained_variance: f64,
}

impl Default for GraphRegressionMetrics {
    fn default() -> Self {
        Self {
            mse: 0.0,
            rmse: 0.0,
            mae: 0.0,
            r2_score: 0.0,
            mape: 0.0,
            explained_variance: 0.0,
        }
    }
}

impl GraphRegressionMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Graph property prediction metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphPropertyMetrics {
    /// Structural property prediction accuracy
    pub structural_accuracy: HashMap<String, f64>, // property -> accuracy
    /// Spectral property prediction accuracy
    pub spectral_accuracy: HashMap<String, f64>,
    /// Topological property prediction accuracy
    pub topological_accuracy: HashMap<String, f64>,
}

impl GraphPropertyMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Graph similarity evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSimilarityMetrics {
    /// Graph edit distance correlation
    pub ged_correlation: f64,
    /// Subgraph isomorphism accuracy
    pub subgraph_isomorphism_accuracy: f64,
    /// Maximum common subgraph ratio
    pub mcs_ratio: f64,
    /// Spectral distance correlation
    pub spectral_distance_correlation: f64,
}

impl Default for GraphSimilarityMetrics {
    fn default() -> Self {
        Self {
            ged_correlation: 0.0,
            subgraph_isomorphism_accuracy: 0.0,
            mcs_ratio: 0.0,
            spectral_distance_correlation: 0.0,
        }
    }
}

impl GraphSimilarityMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

impl GraphLevelMetrics {
    /// Create new graph-level metrics
    pub fn new() -> Self {
        Self {
            classification: GraphClassificationMetrics::new(),
            regression: GraphRegressionMetrics::new(),
            property_prediction: GraphPropertyMetrics::new(),
            similarity_metrics: GraphSimilarityMetrics::new(),
        }
    }
}

impl Default for GraphLevelMetrics {
    fn default() -> Self {
        Self::new()
    }
}
