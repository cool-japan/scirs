//! Configuration and Core Types for AI-Driven Adaptive Processing
//!
//! This module contains configuration structures, enums, and core data types
//! used throughout the AI-driven adaptive processing system.

use crate::advanced_fusion_algorithms::AdvancedConfig;

/// AI-Driven Adaptive Processing Configuration
#[derive(Debug, Clone)]
pub struct AIAdaptiveConfig {
    /// Base Advanced configuration
    pub base_config: AdvancedConfig,
    /// Learning rate for AI adaptation
    pub learning_rate: f64,
    /// Experience replay buffer size
    pub replay_buffer_size: usize,
    /// Multi-modal learning enabled
    pub multi_modal_learning: bool,
    /// Continual learning enabled
    pub continual_learning: bool,
    /// Explainable AI features enabled
    pub explainable_ai: bool,
    /// Transfer learning enabled
    pub transfer_learning: bool,
    /// Few-shot learning threshold
    pub few_shot_threshold: usize,
    /// Performance optimization target
    pub optimization_target: OptimizationTarget,
    /// AI model complexity level
    pub model_complexity: ModelComplexity,
    /// Prediction horizon (for predictive processing)
    pub prediction_horizon: usize,
    /// Adaptation speed (how fast to adapt to new patterns)
    pub adaptation_speed: f64,
}

impl Default for AIAdaptiveConfig {
    fn default() -> Self {
        Self {
            base_config: AdvancedConfig::default(),
            learning_rate: 0.001,
            replay_buffer_size: 10000,
            multi_modal_learning: true,
            continual_learning: true,
            explainable_ai: true,
            transfer_learning: true,
            few_shot_threshold: 5,
            optimization_target: OptimizationTarget::Balanced,
            model_complexity: ModelComplexity::High,
            prediction_horizon: 10,
            adaptation_speed: 0.1,
        }
    }
}

/// Optimization Target Preferences
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationTarget {
    Speed,
    Quality,
    Balanced,
    MemoryEfficient,
    EnergyEfficient,
    UserCustom(Vec<f64>), // Custom weights for different objectives
}

/// AI Model Complexity Levels
#[derive(Debug, Clone)]
pub enum ModelComplexity {
    Low,
    Medium,
    High,
    Advanced,
    Adaptive, // Automatically adjusts complexity based on available resources
}

/// Pattern Types
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum PatternType {
    Natural,
    Synthetic,
    Medical,
    Satellite,
    Scientific,
    Artistic,
    Document,
    Industrial,
    Security,
    Gaming,
    Educational,
    Research,
    Unknown,
}

/// Complexity Levels
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ComplexityLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
    Extreme,
}

/// Noise Levels
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum NoiseLevel {
    Clean,
    Low,
    Medium,
    High,
    Extreme,
}

/// Algorithm Types
#[derive(Debug, Clone)]
pub enum AlgorithmType {
    GaussianFilter,
    MedianFilter,
    BilateralFilter,
    EdgeDetection,
    MorphologyOperation,
    QuantumProcessing,
    NeuromorphicProcessing,
    ConsciousnessSimulation,
    AdvancedFusion,
    CustomAI,
}

/// Feature Types
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum FeatureType {
    Edges,
    Textures,
    Shapes,
    Colors,
    Gradients,
    Corners,
    Lines,
    Curves,
    Patterns,
    Objects,
    Faces,
    Text,
}

/// Image Pattern Recognition
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ImagePattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Complexity level
    pub complexity: ComplexityLevel,
    /// Noise level
    pub noise_level: NoiseLevel,
    /// Dominant features
    pub dominantfeatures: Vec<FeatureType>,
}
