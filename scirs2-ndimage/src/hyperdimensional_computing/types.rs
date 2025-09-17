//! Core Types and Configuration for Hyperdimensional Computing
//!
//! This module defines the fundamental data structures and configuration
//! for hyperdimensional computing operations, including hypervectors and
//! various result types used throughout the HDC system.

use std::collections::HashMap;

/// Configuration for hyperdimensional computing
#[derive(Debug, Clone)]
pub struct HDCConfig {
    /// Dimensionality of hypervectors (typically 10,000+)
    pub hypervector_dim: usize,
    /// Sparsity level (fraction of dimensions that are non-zero)
    pub sparsity: f64,
    /// Similarity threshold for recognition
    pub similarity_threshold: f64,
    /// Number of training iterations
    pub training_iterations: usize,
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Bundling capacity (number of vectors to bundle)
    pub bundling_capacity: usize,
    /// Binding strength for compositional operations
    pub binding_strength: f64,
    /// Cleanup threshold for memory operations
    pub cleanup_threshold: f64,
}

impl Default for HDCConfig {
    fn default() -> Self {
        Self {
            hypervector_dim: 10000,
            sparsity: 0.01, // 1% sparsity
            similarity_threshold: 0.8,
            training_iterations: 10,
            learning_rate: 0.1,
            bundling_capacity: 100,
            binding_strength: 1.0,
            cleanup_threshold: 0.7,
        }
    }
}

/// Hyperdimensional vector representation
#[derive(Debug, Clone)]
pub struct Hypervector {
    /// Sparse representation: (index, value) pairs
    pub sparse_data: Vec<(usize, f64)>,
    /// Dimensionality
    pub dimension: usize,
    /// Norm for normalization
    pub norm: f64,
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Matched pattern label
    pub label: String,
    /// Confidence score [0, 1]
    pub confidence: f64,
    /// Position in the image (y, x)
    pub position: (usize, usize),
    /// Size of the matched region (height, width)
    pub size: (usize, usize),
}

/// Feature detection result
#[derive(Debug, Clone)]
pub struct FeatureDetection {
    /// Type of detected feature
    pub feature_type: String,
    /// Position of the feature (y, x)
    pub position: (usize, usize),
    /// Strength/confidence of the detection
    pub strength: f64,
    /// Encoded hypervector representation
    pub hypervector: Hypervector,
    /// Size of the feature patch (height, width)
    pub patch_size: (usize, usize),
}

/// Sequence encoding result
#[derive(Debug, Clone)]
pub struct SequenceEncoding {
    /// Encoded sequence as hypervector
    pub encoding: Hypervector,
    /// Temporal position mappings
    pub temporal_positions: Vec<usize>,
    /// Sequence confidence score
    pub confidence: f64,
}

/// Experience for continual learning
#[derive(Debug, Clone)]
pub struct Experience {
    /// Encoded representation
    pub encoding: Hypervector,
    /// Associated label
    pub label: String,
    /// Timestamp of experience
    pub timestamp: usize,
    /// Importance score
    pub importance: f64,
}

/// Consolidation result for memory operations
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Number of interference cases prevented
    pub interference_prevented: usize,
    /// Effectiveness score of consolidation
    pub effectiveness_score: f64,
    /// Number of replay cycles used
    pub replay_cycles_used: usize,
}

/// Prediction result
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Predicted label
    pub predicted_label: String,
    /// Confidence in prediction
    pub confidence: f64,
    /// Alternative predictions with scores
    pub alternatives: Vec<(String, f64)>,
}

/// Update result for online learning
#[derive(Debug, Clone)]
pub struct UpdateResult {
    /// Whether memory was updated
    pub memory_updated: bool,
    /// Learning rate used for update
    pub learning_rate_used: f64,
    /// Change in performance metrics
    pub performance_change: f64,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Learning speed
    pub learning_speed: f64,
    /// Memory efficiency
    pub memory_efficiency: f64,
    /// Adaptation effectiveness
    pub adaptation_effectiveness: f64,
}

/// Result of online learning step
#[derive(Debug, Clone)]
pub struct OnlineLearningResult {
    /// Prediction made
    pub prediction: PredictionResult,
    /// Learning update information
    pub learning_update: UpdateResult,
    /// Current system performance
    pub system_performance: PerformanceMetrics,
    /// Current adaptation rate
    pub adaptation_rate: f64,
}

/// Result of compositional reasoning
#[derive(Debug, Clone)]
pub struct CompositionResult {
    /// Similarity between query and image
    pub query_similarity: f64,
    /// Presence strength of individual concepts
    pub concept_presence: HashMap<String, f64>,
    /// Composed query representation
    pub composed_representation: Hypervector,
    /// Image representation
    pub image_representation: Hypervector,
}

/// Abstraction level for hierarchical reasoning
#[derive(Debug, Clone)]
pub struct AbstractionLevel {
    /// Level identifier
    pub level: usize,
    /// Concepts at this level
    pub concepts: HashMap<String, Hypervector>,
    /// Resolution/granularity
    pub resolution: f64,
    /// Complexity score
    pub complexity: f64,
}

/// Reasoning chain for logical inference
#[derive(Debug, Clone)]
pub struct ReasoningChain {
    /// Chain identifier
    pub chain_id: String,
    /// Sequence of concepts
    pub concepts: Vec<String>,
    /// Overall confidence
    pub confidence: f64,
    /// Supporting evidence strength
    pub support_evidence: f64,
}

/// Meta-cognitive assessment
#[derive(Debug, Clone)]
pub struct MetaCognitiveAssessment {
    /// Confidence in reasoning
    pub confidence_score: f64,
    /// Depth of reasoning
    pub reasoning_depth: usize,
    /// Uncertainty estimate
    pub uncertainty_estimate: f64,
}

/// Multi-modal fusion configuration
#[derive(Debug, Clone)]
pub struct MultiModalFusionConfig {
    /// Weight for visual modality
    pub visual_weight: f64,
    /// Weight for temporal modality
    pub temporal_weight: f64,
    /// Weight for semantic modality
    pub semantic_weight: f64,
    /// Attention mechanism strength
    pub attention_strength: f64,
    /// Fusion method to use
    pub fusion_method: FusionMethod,
}

impl Default for MultiModalFusionConfig {
    fn default() -> Self {
        Self {
            visual_weight: 0.4,
            temporal_weight: 0.3,
            semantic_weight: 0.3,
            attention_strength: 0.8,
            fusion_method: FusionMethod::WeightedBundle,
        }
    }
}

/// Fusion method enumeration
#[derive(Debug, Clone)]
pub enum FusionMethod {
    /// Simple weighted bundling
    WeightedBundle,
    /// Attention-based fusion
    AttentionFusion,
    /// Hierarchical fusion
    HierarchicalFusion,
}

/// Adaptation parameters for online learning
#[derive(Debug, Clone)]
pub struct AdaptationParameters {
    /// Base learning rate
    pub base_rate: f64,
    /// Current learning rate
    pub current_rate: f64,
    /// Minimum learning rate
    pub min_rate: f64,
    /// Maximum learning rate
    pub max_rate: f64,
    /// Adaptation speed
    pub adaptation_speed: f64,
    /// Performance threshold for adaptation
    pub performance_threshold: f64,
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        Self {
            base_rate: 0.1,
            current_rate: 0.1,
            min_rate: 0.001,
            max_rate: 0.5,
            adaptation_speed: 0.05,
            performance_threshold: 0.8,
        }
    }
}

impl AdaptationParameters {
    /// Adjust learning rate based on performance
    pub fn adjust_based_on_performance(
        &mut self,
        tracker: &crate::hyperdimensional_computing::memory::PerformanceTracker,
    ) {
        let recent_change = tracker.get_recent_performance_change();

        if recent_change > 0.05 {
            // Performance improving - increase rate
            self.current_rate =
                (self.current_rate * (1.0 + self.adaptation_speed)).min(self.max_rate);
        } else if recent_change < -0.05 {
            // Performance degrading - decrease rate
            self.current_rate =
                (self.current_rate * (1.0 - self.adaptation_speed)).max(self.min_rate);
        }
        // Otherwise maintain current rate
    }

    /// Reset to base rate
    pub fn reset(&mut self) {
        self.current_rate = self.base_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdc_config_default() {
        let config = HDCConfig::default();
        assert_eq!(config.hypervector_dim, 10000);
        assert_eq!(config.sparsity, 0.01);
        assert_eq!(config.similarity_threshold, 0.8);
        assert_eq!(config.training_iterations, 10);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.bundling_capacity, 100);
        assert_eq!(config.binding_strength, 1.0);
        assert_eq!(config.cleanup_threshold, 0.7);
    }

    #[test]
    fn test_hypervector_creation() {
        let hv = Hypervector {
            sparse_data: vec![(0, 1.0), (100, -1.0), (200, 1.0)],
            dimension: 1000,
            norm: 1.732, // sqrt(3)
        };

        assert_eq!(hv.dimension, 1000);
        assert_eq!(hv.sparse_data.len(), 3);
        assert!((hv.norm - 1.732).abs() < 0.01);
    }

    #[test]
    fn test_pattern_match() {
        let pattern_match = PatternMatch {
            label: "test_pattern".to_string(),
            confidence: 0.95,
            position: (10, 20),
            size: (32, 32),
        };

        assert_eq!(pattern_match.label, "test_pattern");
        assert_eq!(pattern_match.confidence, 0.95);
        assert_eq!(pattern_match.position, (10, 20));
        assert_eq!(pattern_match.size, (32, 32));
    }

    #[test]
    fn test_multimodal_fusion_config_default() {
        let config = MultiModalFusionConfig::default();

        // Weights should sum approximately to 1.0
        let sum = config.visual_weight + config.temporal_weight + config.semantic_weight;
        assert!((sum - 1.0).abs() < 0.1);

        assert!(config.attention_strength > 0.0);
        assert!(config.attention_strength <= 1.0);

        matches!(config.fusion_method, FusionMethod::WeightedBundle);
    }

    #[test]
    fn test_adaptation_parameters() {
        let mut params = AdaptationParameters::default();
        let original_rate = params.current_rate;

        // Test reset functionality
        params.current_rate = 0.3;
        params.reset();
        assert_eq!(params.current_rate, params.base_rate);

        // Test bounds
        assert!(params.min_rate < params.max_rate);
        assert!(params.current_rate >= params.min_rate);
        assert!(params.current_rate <= params.max_rate);
    }

    #[test]
    fn test_prediction_result() {
        let prediction = PredictionResult {
            predicted_label: "cat".to_string(),
            confidence: 0.9,
            alternatives: vec![("dog".to_string(), 0.1), ("bird".to_string(), 0.05)],
        };

        assert_eq!(prediction.predicted_label, "cat");
        assert_eq!(prediction.confidence, 0.9);
        assert_eq!(prediction.alternatives.len(), 2);
        assert_eq!(prediction.alternatives[0].0, "dog");
        assert_eq!(prediction.alternatives[0].1, 0.1);
    }

    #[test]
    fn test_reasoning_chain() {
        let chain = ReasoningChain {
            chain_id: "chain_1".to_string(),
            concepts: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            confidence: 0.8,
            support_evidence: 0.75,
        };

        assert_eq!(chain.chain_id, "chain_1");
        assert_eq!(chain.concepts.len(), 3);
        assert_eq!(chain.concepts[0], "A");
        assert_eq!(chain.confidence, 0.8);
        assert_eq!(chain.support_evidence, 0.75);
    }
}
