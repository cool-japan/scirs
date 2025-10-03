//! Core type definitions for advanced interpolation coordination
//!
//! This module contains all the foundational data types used throughout
//! the advanced interpolation coordinator system.

use scirs2_core::numeric::Float;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::time::Instant;

/// Key for method identification
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MethodKey {
    /// Interpolation method type
    pub method_type: InterpolationMethodType,
    /// Data size characteristics
    pub size_class: DataSizeClass,
    /// Data pattern type
    pub pattern_type: DataPatternType,
    /// Dimensionality
    pub dimensionality: u8,
}

/// Interpolation method types
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum InterpolationMethodType {
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    CubicSpline,
    /// B-spline interpolation
    BSpline,
    /// Radial basis function interpolation
    RadialBasisFunction,
    /// Kriging interpolation
    Kriging,
    /// Polynomial interpolation
    Polynomial,
    /// Piecewise cubic Hermite interpolation
    PchipInterpolation,
    /// Akima spline interpolation
    AkimaSpline,
    /// Thin plate spline interpolation
    ThinPlateSpline,
    /// Natural neighbor interpolation
    NaturalNeighbor,
    /// Shepard's method
    ShepardsMethod,
    /// Quantum-inspired interpolation
    QuantumInspired,
}

/// Data size classification
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum DataSizeClass {
    /// Small datasets (< 1K points)
    Small,
    /// Medium datasets (1K - 100K points)
    Medium,
    /// Large datasets (100K - 10M points)
    Large,
    /// Massive datasets (> 10M points)
    Massive,
}

/// Data pattern classification
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum DataPatternType {
    /// Smooth, continuous data
    Smooth,
    /// Oscillatory data
    Oscillatory,
    /// Noisy data
    Noisy,
    /// Sparse data
    Sparse,
    /// Piecewise continuous data
    PiecewiseContinuous,
    /// Monotonic data
    Monotonic,
    /// Irregular/scattered data
    Irregular,
    /// Highly structured data
    Structured,
}

/// Performance data for methods
#[derive(Debug, Clone)]
pub struct MethodPerformanceData {
    /// Average execution time (microseconds)
    pub avg_execution_time: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Interpolation accuracy (RMS error)
    pub accuracy: f64,
    /// Robustness to noise
    pub noise_robustness: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Last update time
    pub last_update: Instant,
}

/// Data profile for analysis
#[derive(Debug, Clone)]
pub struct DataProfile<F: Float> {
    /// Number of data points
    pub size: usize,
    /// Data dimensionality
    pub dimensionality: usize,
    /// Value range (min, max)
    pub value_range: (F, F),
    /// Gradient statistics
    pub gradient_stats: GradientStatistics<F>,
    /// Frequency content analysis
    pub frequency_content: FrequencyContent<F>,
    /// Noise characteristics
    pub noise_level: F,
    /// Sparsity ratio
    pub sparsity: F,
    /// Smoothness measure
    pub smoothness: F,
    /// Pattern type classification
    pub pattern_type: DataPatternType,
}

/// Gradient statistics
#[derive(Debug, Clone)]
pub struct GradientStatistics<F: Float> {
    /// Mean gradient magnitude
    pub mean_magnitude: F,
    /// Gradient variance
    pub variance: F,
    /// Maximum gradient
    pub max_gradient: F,
    /// Gradient distribution characteristics
    pub distribution_skew: F,
}

/// Frequency content analysis
#[derive(Debug, Clone)]
pub struct FrequencyContent<F: Float> {
    /// Dominant frequency
    pub dominant_frequency: F,
    /// Frequency spread
    pub frequency_spread: F,
    /// High frequency content ratio
    pub high_freq_ratio: F,
    /// Low frequency content ratio
    pub low_freq_ratio: F,
}

/// Method performance record for history tracking
#[derive(Debug, Clone)]
pub struct MethodPerformanceRecord {
    /// Timestamp of record
    pub timestamp: Instant,
    /// Method used
    pub method: InterpolationMethodType,
    /// Execution time (microseconds)
    pub execution_time: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Achieved accuracy
    pub accuracy: f64,
    /// Data size processed
    pub data_size: usize,
    /// Success flag
    pub success: bool,
}

/// Pattern signature for data analysis
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct PatternSignature {
    /// Primary pattern type
    pub primary_pattern: DataPatternType,
    /// Secondary patterns
    pub secondary_patterns: Vec<DataPatternType>,
    /// Pattern confidence score (0-100)
    pub confidence: u8,
    /// Pattern complexity level
    pub complexity: PatternComplexity,
    /// Recommended method classes
    pub recommended_methods: Vec<InterpolationMethodType>,
}

/// Pattern complexity levels
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum PatternComplexity {
    /// Simple, predictable patterns
    Simple,
    /// Moderate complexity
    Moderate,
    /// Complex, multi-faceted patterns
    Complex,
    /// Extremely complex, chaotic patterns
    Chaotic,
}

/// Pattern analysis data
#[derive(Debug, Clone)]
pub struct PatternData<F: Float> {
    /// Statistical features
    pub features: Vec<F>,
    /// Spectral characteristics
    pub spectral_features: Vec<F>,
    /// Geometric properties
    pub geometric_features: Vec<F>,
    /// Correlation structure
    pub correlation_matrix: Vec<Vec<F>>,
}

/// Performance characteristics for tuning
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    /// Throughput (points per second)
    pub throughput: f64,
    /// Memory efficiency (points per MB)
    pub memory_efficiency: f64,
    /// Accuracy score (0-1)
    pub accuracy_score: f64,
    /// Stability measure
    pub stability: f64,
    /// Scalability factor
    pub scalability: f64,
}

/// Analysis state for pattern recognition
#[derive(Debug, Clone)]
pub struct AnalysisState<F: Float> {
    /// Current analysis stage
    pub stage: AnalysisStage,
    /// Intermediate results
    pub intermediate_results: Vec<F>,
    /// Analysis progress (0-1)
    pub progress: f64,
    /// Confidence in current analysis
    pub confidence: f64,
}

/// Analysis stages
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisStage {
    /// Initial data loading
    DataLoading,
    /// Feature extraction
    FeatureExtraction,
    /// Pattern recognition
    PatternRecognition,
    /// Method recommendation
    MethodRecommendation,
    /// Performance prediction
    PerformancePrediction,
    /// Optimization
    Optimization,
    /// Completed
    Completed,
}

/// Performance targets for optimization
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    /// Target accuracy (0-1)
    pub target_accuracy: f64,
    /// Maximum acceptable time (microseconds)
    pub max_time: f64,
    /// Maximum memory usage (bytes)
    pub max_memory: usize,
    /// Priority weights for different metrics
    pub priority_weights: PerformancePriorities,
}

/// Performance priority weights
#[derive(Debug, Clone)]
pub struct PerformancePriorities {
    /// Accuracy weight (0-1)
    pub accuracy_weight: f64,
    /// Speed weight (0-1)
    pub speed_weight: f64,
    /// Memory weight (0-1)
    pub memory_weight: f64,
    /// Stability weight (0-1)
    pub stability_weight: f64,
}

/// Adaptive parameters for performance tuning
#[derive(Debug, Clone)]
pub struct AdaptiveParameters<F: Float> {
    /// Learning rate
    pub learning_rate: F,
    /// Adaptation speed
    pub adaptation_speed: F,
    /// Convergence threshold
    pub convergence_threshold: F,
    /// Momentum factor
    pub momentum: F,
    /// Regularization strength
    pub regularization: F,
}

/// Tuning result
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// Achieved performance improvement
    pub improvement: f64,
    /// Number of iterations required
    pub iterations: usize,
    /// Final parameters
    pub final_parameters: Vec<f64>,
    /// Convergence achieved
    pub converged: bool,
    /// Time taken (microseconds)
    pub time_taken: f64,
}

/// Input validation result
#[derive(Debug, Clone)]
pub struct InputValidationResult<F: Float> {
    /// Validation passed
    pub is_valid: bool,
    /// Error messages
    pub errors: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Data quality score (0-1)
    pub quality_score: F,
    /// Recommended preprocessing steps
    pub preprocessing_recommendations: Vec<PreprocessingStep>,
}

/// Preprocessing steps
#[derive(Debug, Clone, PartialEq)]
pub enum PreprocessingStep {
    /// Remove outliers
    RemoveOutliers,
    /// Smooth data
    SmoothData,
    /// Normalize values
    NormalizeValues,
    /// Interpolate missing values
    InterpolateMissing,
    /// Resample data
    ResampleData,
    /// Apply filtering
    ApplyFiltering,
}

/// Accuracy prediction result
#[derive(Debug, Clone)]
pub struct AccuracyPrediction<F: Float> {
    /// Predicted accuracy
    pub predicted_accuracy: F,
    /// Confidence interval
    pub confidence_interval: (F, F),
    /// Prediction confidence (0-1)
    pub prediction_confidence: F,
    /// Factors affecting accuracy
    pub accuracy_factors: Vec<AccuracyFactor<F>>,
}

/// Factors affecting accuracy
#[derive(Debug, Clone)]
pub struct AccuracyFactor<F: Float> {
    /// Factor name
    pub name: String,
    /// Impact on accuracy (-1 to 1)
    pub impact: F,
    /// Confidence in impact assessment
    pub confidence: F,
    /// Mitigation suggestions
    pub mitigations: Vec<String>,
}

/// Method performance estimate
#[derive(Debug, Clone)]
pub struct MethodPerformanceEstimate {
    /// Estimated execution time (microseconds)
    pub estimated_time: f64,
    /// Estimated memory usage (bytes)
    pub estimated_memory: usize,
    /// Estimated accuracy
    pub estimated_accuracy: f64,
    /// Confidence in estimates
    pub estimate_confidence: f64,
    /// Risk factors
    pub risk_factors: Vec<String>,
}

/// Interpolation recommendation
#[derive(Debug, Clone)]
pub struct InterpolationRecommendation<F: Float> {
    /// Primary recommendation
    pub primary_method: MethodRecommendation,
    /// Alternative methods
    pub alternatives: Vec<MethodRecommendation>,
    /// Expected performance
    pub expected_performance: ExpectedPerformance,
    /// Confidence in recommendation
    pub confidence: F,
    /// Reasoning behind recommendation
    pub reasoning: String,
}

/// Method recommendation details
#[derive(Debug, Clone)]
pub struct MethodRecommendation {
    /// Recommended method
    pub method: InterpolationMethodType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Expected performance characteristics
    pub expected_performance: ExpectedPerformance,
    /// Recommended parameters
    pub parameters: Vec<f64>,
    /// Configuration suggestions
    pub configuration: String,
    /// Expected benefits
    pub benefits: Vec<String>,
    /// Potential limitations
    pub limitations: Vec<String>,
}

/// Expected performance metrics
#[derive(Debug, Clone)]
pub struct ExpectedPerformance {
    /// Expected accuracy range
    pub accuracy_range: (f64, f64),
    /// Expected time range (microseconds)
    pub time_range: (f64, f64),
    /// Expected memory usage range (bytes)
    pub memory_range: (usize, usize),
    /// Performance score (0-1)
    pub performance_score: f64,
}

/// Actual performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Achieved accuracy
    pub accuracy: f64,
}

impl<F: Float> Default for AdaptiveParameters<F> {
    fn default() -> Self {
        Self {
            learning_rate: F::from(0.01).unwrap(),
            adaptation_speed: F::from(0.1).unwrap(),
            convergence_threshold: F::from(1e-6).unwrap(),
            momentum: F::from(0.9).unwrap(),
            regularization: F::from(1e-4).unwrap(),
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_accuracy: 0.99,
            max_time: 1_000_000.0,          // 1 second
            max_memory: 1024 * 1024 * 1024, // 1 GB
            priority_weights: PerformancePriorities::default(),
        }
    }
}

impl Default for PerformancePriorities {
    fn default() -> Self {
        Self {
            accuracy_weight: 0.4,
            speed_weight: 0.3,
            memory_weight: 0.2,
            stability_weight: 0.1,
        }
    }
}

impl<F: Float + std::fmt::Display> InputValidationResult<F> {
    /// Check if validation passed
    pub fn is_high_quality(&self) -> bool {
        self.is_valid && self.quality_score > F::from(0.8).unwrap()
    }

    /// Get validation summary
    pub fn get_summary(&self) -> String {
        format!(
            "Valid: {}, Quality: {:.2}, Errors: {}, Warnings: {}",
            self.is_valid,
            self.quality_score,
            self.errors.len(),
            self.warnings.len()
        )
    }
}
