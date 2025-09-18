//! Machine Learning Components for Deadlock Detection
//!
//! This module contains machine learning models and feature extraction
//! components for advanced deadlock detection and prediction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Machine learning model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLModelType {
    /// Neural network
    NeuralNetwork { architecture: Vec<usize> },
    /// Support vector machine
    SVM { kernel: String },
    /// Random forest
    RandomForest { trees: usize },
    /// Gradient boosting
    GradientBoosting { estimators: usize },
    /// Deep learning
    DeepLearning { model: String },
}

/// Feature extraction for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtraction {
    /// Graph features
    pub graph_features: Vec<GraphFeature>,
    /// Temporal features
    pub temporal_features: Vec<TemporalFeature>,
    /// Resource features
    pub resource_features: Vec<ResourceFeature>,
    /// Custom features
    pub custom_features: Vec<String>,
}

/// Graph features for ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphFeature {
    /// Node count
    NodeCount,
    /// Edge count
    EdgeCount,
    /// Graph density
    Density,
    /// Clustering coefficient
    ClusteringCoefficient,
    /// Path lengths
    PathLengths,
    /// Centrality measures
    Centrality,
}

/// Temporal features for ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalFeature {
    /// Wait times
    WaitTimes,
    /// Request patterns
    RequestPatterns,
    /// Resource utilization patterns
    UtilizationPatterns,
    /// Seasonal patterns
    SeasonalPatterns,
}

/// Resource features for ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceFeature {
    /// Resource types
    ResourceTypes,
    /// Resource availability
    Availability,
    /// Resource contention
    Contention,
    /// Resource hierarchy
    Hierarchy,
}

/// Algorithm combination strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CombinationStrategy {
    /// Voting-based combination
    Voting { weights: HashMap<String, f64> },
    /// Threshold-based combination
    Threshold { threshold: f64 },
    /// Priority-based combination
    Priority { priorities: Vec<String> },
    /// Ensemble methods
    Ensemble { method: EnsembleMethod },
}

/// Ensemble methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleMethod {
    /// Majority voting
    MajorityVoting,
    /// Weighted voting
    WeightedVoting,
    /// Stacking
    Stacking,
    /// Boosting
    Boosting,
    /// Bagging
    Bagging,
}