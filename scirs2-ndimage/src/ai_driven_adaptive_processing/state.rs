//! State Management for AI-Driven Adaptive Processing
//!
//! This module handles the main processing state, experience buffers,
//! and state transitions for the AI-driven adaptive processing system.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use std::collections::{HashMap, VecDeque};

use super::config::ImagePattern;
use super::knowledge::{MultiModalKnowledgeBase, ProcessingContext};
use super::learning::{
    ContinualLearningState, ExplanationTracker, FewShotLearningEntry, NeuralModel,
    ProcessingExperience, TransferLearningModel,
};
use super::strategies::ProcessingStrategy;

/// AI-Driven Processing State
#[derive(Debug, Clone)]
pub struct AIProcessingState {
    /// Neural network weights for decision making
    pub decision_network: Array3<f64>,
    /// Experience replay buffer
    pub experience_buffer: VecDeque<ProcessingExperience>,
    /// Learned processing strategies
    pub processing_strategies: HashMap<ImagePattern, ProcessingStrategy>,
    /// Performance history
    pub performancehistory: VecDeque<f64>,
    /// Multi-modal knowledge base
    pub knowledge_base: MultiModalKnowledgeBase,
    /// Current processing context
    pub currentcontext: ProcessingContext,
    /// Continual learning state
    pub continual_learningstate: ContinualLearningState,
    /// Explainability tracking
    pub explanation_tracker: ExplanationTracker,
    /// Transfer learning models
    pub transfer_models: Vec<TransferLearningModel>,
    /// Few-shot learning cache
    pub few_shot_cache: HashMap<String, FewShotLearningEntry>,
    /// Algorithm confidence levels
    pub algorithm_confidence: HashMap<String, f64>,
    /// Neural network for pattern classification
    pub neural_network: NeuralModel,
    /// Pattern to strategy mapping
    pub pattern_strategy_mapping: HashMap<String, String>,
    /// Algorithm usage count tracking
    pub algorithm_usage_count: HashMap<String, usize>,
    /// Strategy performance tracking
    pub strategy_performance: HashMap<String, f64>,
    /// Pattern processing history
    pub patternhistory: VecDeque<ImagePattern>,
    /// Learned feature representations
    pub learnedfeatures: HashMap<String, Array1<f64>>,
}

impl AIProcessingState {
    /// Create a new AI processing state
    pub fn new() -> Self {
        Self {
            decision_network: Array3::zeros((10, 10, 5)), // Default small network
            experience_buffer: VecDeque::new(),
            processing_strategies: HashMap::new(),
            performancehistory: VecDeque::new(),
            knowledge_base: MultiModalKnowledgeBase::new(),
            currentcontext: ProcessingContext::default(),
            continual_learningstate: ContinualLearningState {
                task_knowledge: Vec::new(),
                forgetting_prevention: super::learning::ForgettingPreventionState {
                    ewc_params: Array1::zeros(100),
                    fisher_information: Array2::zeros((100, 100)),
                    importance_mask: Array1::from_elem(100, false),
                    memory_strength: 1.0,
                },
                meta_learning_params: Array1::zeros(50),
                adaptationhistory: Vec::new(),
            },
            explanation_tracker: ExplanationTracker {
                decision_explanations: VecDeque::new(),
                feature_importance: HashMap::new(),
                justifications: HashMap::new(),
                confidence_scores: HashMap::new(),
            },
            transfer_models: Vec::new(),
            few_shot_cache: HashMap::new(),
            algorithm_confidence: HashMap::new(),
            neural_network: NeuralModel {
                weights: Array2::zeros((100, 50)),
                biases: Array1::zeros(50),
                architecture: "feedforward".to_string(),
            },
            pattern_strategy_mapping: HashMap::new(),
            algorithm_usage_count: HashMap::new(),
            strategy_performance: HashMap::new(),
            patternhistory: VecDeque::new(),
            learnedfeatures: HashMap::new(),
        }
    }

    /// Add experience to the replay buffer
    pub fn add_experience(&mut self, experience: ProcessingExperience) {
        self.experience_buffer.push_back(experience);

        // Limit buffer size to prevent excessive memory usage
        if self.experience_buffer.len() > 10000 {
            self.experience_buffer.pop_front();
        }
    }

    /// Get a batch of experiences for learning
    pub fn get_experience_batch(&self, batch_size: usize) -> Vec<ProcessingExperience> {
        if self.experience_buffer.is_empty() {
            return Vec::new();
        }

        let buffer_size = self.experience_buffer.len();
        let actual_batch_size = batch_size.min(buffer_size);
        let mut batch = Vec::new();

        // Simple random sampling (in production, use proper randomization)
        let step = buffer_size / actual_batch_size;
        for i in 0..actual_batch_size {
            let index = (i * step) % buffer_size;
            if let Some(experience) = self.experience_buffer.get(index) {
                batch.push(experience.clone());
            }
        }

        batch
    }

    /// Update strategy for a given pattern
    pub fn update_strategy(&mut self, pattern: ImagePattern, strategy: ProcessingStrategy) {
        self.processing_strategies.insert(pattern, strategy);
    }

    /// Get best strategy for a given pattern
    pub fn get_best_strategy(&self, pattern: &ImagePattern) -> Option<&ProcessingStrategy> {
        self.processing_strategies.get(pattern)
    }

    /// Add performance record
    pub fn add_performance_record(&mut self, performance: f64) {
        self.performancehistory.push_back(performance);

        // Limit history size
        if self.performancehistory.len() > 1000 {
            self.performancehistory.pop_front();
        }
    }

    /// Get average performance over recent history
    pub fn get_average_performance(&self, recent_count: usize) -> f64 {
        if self.performancehistory.is_empty() {
            return 0.0;
        }

        let count = recent_count.min(self.performancehistory.len());
        let start_index = self.performancehistory.len() - count;

        let sum: f64 = self.performancehistory.iter().skip(start_index).sum();

        sum / count as f64
    }

    /// Update algorithm confidence
    pub fn update_algorithm_confidence(&mut self, algorithm: &str, confidence: f64) {
        self.algorithm_confidence
            .insert(algorithm.to_string(), confidence);
    }

    /// Get algorithm confidence
    pub fn get_algorithm_confidence(&self, algorithm: &str) -> f64 {
        self.algorithm_confidence
            .get(algorithm)
            .copied()
            .unwrap_or(0.5)
    }

    /// Add pattern to history
    pub fn add_pattern_to_history(&mut self, pattern: ImagePattern) {
        self.patternhistory.push_back(pattern);

        // Limit pattern history size
        if self.patternhistory.len() > 500 {
            self.patternhistory.pop_front();
        }
    }

    /// Check if pattern has been seen recently
    pub fn has_seen_pattern_recently(&self, pattern: &ImagePattern, within_last: usize) -> bool {
        let check_count = within_last.min(self.patternhistory.len());
        let start_index = self.patternhistory.len() - check_count;

        self.patternhistory
            .iter()
            .skip(start_index)
            .any(|p| p == pattern)
    }

    /// Get state summary for monitoring
    pub fn get_state_summary(&self) -> StateSummary {
        StateSummary {
            experience_count: self.experience_buffer.len(),
            strategy_count: self.processing_strategies.len(),
            average_performance: self.get_average_performance(100),
            patterns_learned: self.patternhistory.len(),
            algorithms_used: self.algorithm_usage_count.len(),
        }
    }
}

/// Summary of AI processing state for monitoring
#[derive(Debug, Clone)]
pub struct StateSummary {
    /// Number of experiences in buffer
    pub experience_count: usize,
    /// Number of learned strategies
    pub strategy_count: usize,
    /// Average performance over recent history
    pub average_performance: f64,
    /// Number of patterns processed
    pub patterns_learned: usize,
    /// Number of different algorithms used
    pub algorithms_used: usize,
}
