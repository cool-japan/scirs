//! Learning Components for AI-Driven Adaptive Processing
//!
//! This module contains neural networks, learning algorithms, and training components
//! for the AI-driven adaptive processing system.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use std::collections::{HashMap, VecDeque};

use super::config::{AlgorithmType, ImagePattern, PatternType};

/// Neural Model for AI-driven processing
#[derive(Debug, Clone)]
pub struct NeuralModel {
    /// Network weights
    pub weights: Array2<f64>,
    /// Network biases
    pub biases: Array1<f64>,
    /// Model architecture metadata
    pub architecture: String,
}

/// Processing Experience (for reinforcement learning)
#[derive(Debug, Clone)]
pub struct ProcessingExperience {
    /// Input image characteristics
    pub inputfeatures: Array1<f64>,
    /// Processing action taken
    pub action: ProcessingAction,
    /// Quality reward achieved
    pub reward: f64,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Next state features
    pub nextfeatures: Array1<f64>,
    /// Context information
    pub context: String,
}

/// Processing Action (for reinforcement learning)
#[derive(Debug, Clone)]
pub struct ProcessingAction {
    /// Primary algorithm to use
    pub primary_algorithm: AlgorithmType,
    /// Secondary algorithms (if any)
    pub secondary_algorithms: Vec<AlgorithmType>,
    /// Parameter modifications
    pub parameter_adjustments: HashMap<String, f64>,
    /// Processing order
    pub processing_order: Vec<usize>,
}

/// Performance Metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Processing speed (pixels per second)
    pub speed: f64,
    /// Quality score (0-1)
    pub quality: f64,
    /// Memory usage (MB)
    pub memory_usage: f64,
    /// Energy consumption (estimated)
    pub energy_consumption: f64,
    /// User satisfaction (if available)
    pub user_satisfaction: Option<f64>,
}

/// Continual Learning State
#[derive(Debug, Clone)]
pub struct ContinualLearningState {
    /// Task-specific knowledge
    pub task_knowledge: Vec<TaskKnowledge>,
    /// Catastrophic forgetting prevention
    pub forgetting_prevention: ForgettingPreventionState,
    /// Meta-learning parameters
    pub meta_learning_params: Array1<f64>,
    /// Adaptation history
    pub adaptationhistory: Vec<AdaptationRecord>,
}

/// Task Knowledge
#[derive(Debug, Clone)]
pub struct TaskKnowledge {
    /// Task identifier
    pub task_id: String,
    /// Task-specific parameters
    pub parameters: Array1<f64>,
    /// Importance weights
    pub importance_weights: Array1<f64>,
    /// Performance on this task
    pub task_performance: f64,
}

/// Forgetting Prevention State
#[derive(Debug, Clone)]
pub struct ForgettingPreventionState {
    /// Elastic weight consolidation parameters
    pub ewc_params: Array1<f64>,
    /// Fisher information matrix
    pub fisher_information: Array2<f64>,
    /// Important parameter mask
    pub importance_mask: Array1<bool>,
    /// Memory strength
    pub memory_strength: f64,
}

/// Adaptation Record
#[derive(Debug, Clone)]
pub struct AdaptationRecord {
    /// Timestamp
    pub timestamp: u64,
    /// Adaptation type
    pub adaptation_type: String,
    /// Parameters changed
    pub parameters_changed: Vec<String>,
    /// Performance improvement
    pub improvement: f64,
}

/// Transfer Learning Model
#[derive(Debug, Clone)]
pub struct TransferLearningModel {
    /// Source domain
    pub source_domain: String,
    /// Target domain
    pub target_domain: String,
    /// Transfer weights
    pub transfer_weights: Array2<f64>,
    /// Transfer effectiveness
    pub effectiveness: f64,
    /// Number of successful transfers
    pub transfer_count: usize,
}

/// Few-Shot Learning Entry
#[derive(Debug, Clone)]
pub struct FewShotLearningEntry {
    /// Few-shot examples
    pub examples: Vec<Array1<f64>>,
    /// Associated labels/strategies
    pub labels: Vec<String>,
    /// Model adaptation parameters
    pub adaptation_params: Array1<f64>,
    /// Learning progress
    pub learning_progress: f64,
}

/// Explanation Tracker
#[derive(Debug, Clone)]
pub struct ExplanationTracker {
    /// Decision explanations
    pub decision_explanations: VecDeque<String>,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f64>,
    /// Processing justifications
    pub justifications: HashMap<String, String>,
    /// Confidence scores
    pub confidence_scores: HashMap<String, f64>,
}

/// Prediction Model
#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Model weights
    pub weights: Array2<f64>,
    /// Model type
    pub model_type: String,
    /// Prediction accuracy
    pub accuracy: f64,
    /// Training epochs
    pub epochs_trained: usize,
}
