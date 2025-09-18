//! Advanced Mode Coordinator for Interpolation Operations
//!
//! This module provides an advanced AI-driven coordination system for interpolation
//! operations, featuring intelligent method selection, adaptive parameter tuning,
//! real-time accuracy optimization, and cross-domain interpolation intelligence.
//!
//! # API Consistency
//!
//! This coordinator follows the standardized Advanced API patterns:
//! - Consistent naming: `enable_method_selection`, `enable_adaptive_optimization`
//! - Unified configuration fields across all Advanced coordinators
//! - Standard factory functions: `create_advanced_interpolation_coordinator()`
//!
//! # Features
//!
//! - **Intelligent Method Selection**: AI-driven selection of optimal interpolation methods
//! - **Adaptive Parameter Tuning**: Real-time optimization based on data characteristics
//! - **Multi-dimensional Coordination**: Unified optimization across 1D, 2D, and N-D interpolation
//! - **Error-Aware Optimization**: Smart accuracy vs. performance trade-off management
//! - **Pattern Recognition**: Advanced data pattern analysis for method recommendations
//! - **Quantum-Inspired Optimization**: Next-generation parameter optimization
//! - **Cross-Domain Knowledge Transfer**: Learning from diverse interpolation tasks
//! - **Memory-Efficient Processing**: Intelligent memory management for large datasets
//!
//! # Modular Architecture
//!
//! The advanced coordinator has been refactored into focused modules:
//! - `types`: Core type definitions and data structures
//! - `config`: Configuration management with builder patterns
//! - `method_selection`: AI-driven method selection engine
//! - `accuracy_optimization`: Accuracy prediction and optimization
//! - `pattern_analysis`: Data pattern recognition and analysis
//! - `performance_tuning`: Performance optimization and resource management
//! - `quantum_optimization`: Quantum-inspired parameter optimization
//! - `knowledge_transfer`: Cross-domain learning and knowledge transfer
//! - `memory_management`: Memory tracking, caching, and performance monitoring
//! - `core_coordinator`: Main coordinator orchestrating all subsystems
//!
//! # Usage
//!
//! ```rust
//! use scirs2_interpolate::advanced_coordinator::{
//!     create_advanced_interpolation_coordinator,
//!     AdvancedInterpolationConfig
//! };
//! use ndarray::Array1;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // Sample data
//! let x_data = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
//! let y_data = Array1::from(vec![0.0, 1.0, 4.0, 9.0, 16.0]);
//! let x_new = Array1::from(vec![1.5, 2.5]);
//!
//! // Create coordinator with default configuration
//! let coordinator = create_advanced_interpolation_coordinator::<f64>(None)?;
//!
//! // Or with custom configuration
//! let config = AdvancedInterpolationConfig {
//!     enable_method_selection: true,
//!     enable_adaptive_optimization: true,
//!     enable_quantum_optimization: true,
//!     target_accuracy: 1e-8,
//!     max_memory_mb: 2048,
//!     ..Default::default()
//! };
//! let coordinator = create_advanced_interpolation_coordinator(Some(config))?;
//!
//! // Analyze data and get recommendations
//! let recommendation = coordinator.analyze_and_recommend(&x_data, &y_data)?;
//!
//! // Execute optimized interpolation
//! let result = coordinator.execute_optimized_interpolation(
//!     &x_data, &y_data, &x_new, &recommendation
//! )?;
//! # Ok(())
//! # }
//! ```

// Re-export all functionality from the modular implementation
pub use crate::advanced_coordinator_modules::*;

// Maintain backward compatibility with any existing imports
pub use crate::advanced_coordinator_modules::{
    create_advanced_interpolation_coordinator,

    // Accuracy optimization
    AccuracyOptimizationEngine,
    AccuracyPrediction,

    AdaptiveInterpolationCache,
    AdvancedInterpolationConfig,
    // Core functionality
    AdvancedInterpolationCoordinator,
    CacheOptimizationResult,

    // Knowledge transfer
    CrossDomainInterpolationKnowledge,
    // Pattern analysis
    DataPatternAnalyzer,
    DataPatternType,
    DataProfile,
    FrequencyContent,

    GradientStatistics,
    // Method selection
    IntelligentMethodSelector,
    // Memory management
    InterpolationMemoryManager,
    // Core data types
    InterpolationMethodType,
    InterpolationPerformanceMetrics,
    InterpolationPerformanceTracker,
    // Recommendation and performance types
    InterpolationRecommendation,
    MemoryStatistics,
    MethodPerformanceEstimate,
    MethodRecommendation,

    PatternAnalysisResult,

    PerformanceMetrics,
    PerformanceOptimizationResult,

    PerformanceSummary,
    PerformanceTargets,
    // Performance tuning
    PerformanceTuningSystem,
    QuantumOptimizationResult,

    // Quantum optimization
    QuantumParameterOptimizer,
    SystemOptimizationResult,
    TransferKnowledgeResult,
};

/// Convenience function for creating a coordinator with default settings
pub fn create_default_advanced_coordinator<
    F: num_traits::Float
        + std::fmt::Debug
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::default::Default,
>() -> crate::error::InterpolateResult<AdvancedInterpolationCoordinator<F>> {
    create_advanced_interpolation_coordinator(None)
}

/// Convenience function for creating a high-performance coordinator
pub fn create_high_performance_coordinator<
    F: num_traits::Float
        + std::fmt::Debug
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::default::Default,
>() -> crate::error::InterpolateResult<AdvancedInterpolationCoordinator<F>> {
    let config = AdvancedInterpolationConfig {
        enable_method_selection: true,
        enable_adaptive_optimization: true,
        enable_quantum_optimization: true,
        enable_knowledge_transfer: true,
        target_accuracy: 1e-8,
        max_memory_mb: 8192,     // 8GB for high-performance scenarios
        monitoring_interval: 10, // More frequent monitoring
        enable_real_time_learning: true,
        enable_error_prediction: true,
        cache_size_limit: 10000,    // Larger cache
        adaptation_threshold: 0.05, // More sensitive adaptation
        enable_hardware_optimization: true,
    };

    create_advanced_interpolation_coordinator(Some(config))
}

/// Convenience function for creating a memory-efficient coordinator
pub fn create_memory_efficient_coordinator<
    F: num_traits::Float
        + std::fmt::Debug
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::default::Default,
>() -> crate::error::InterpolateResult<AdvancedInterpolationCoordinator<F>> {
    let config = AdvancedInterpolationConfig {
        enable_method_selection: true,
        enable_adaptive_optimization: false, // Reduce memory usage
        enable_quantum_optimization: false,  // Disable memory-intensive features
        enable_knowledge_transfer: false,
        target_accuracy: 1e-6,    // Slightly relaxed accuracy
        max_memory_mb: 512,       // Limited memory
        monitoring_interval: 100, // Less frequent monitoring
        enable_real_time_learning: false,
        enable_error_prediction: false,
        cache_size_limit: 100,     // Smaller cache
        adaptation_threshold: 0.2, // Less sensitive adaptation
        enable_hardware_optimization: false,
    };

    create_advanced_interpolation_coordinator(Some(config))
}

/// Convenience function for creating a balanced coordinator
pub fn create_balanced_coordinator<
    F: num_traits::Float
        + std::fmt::Debug
        + std::ops::MulAssign
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::default::Default,
>() -> crate::error::InterpolateResult<AdvancedInterpolationCoordinator<F>> {
    let config = AdvancedInterpolationConfig {
        enable_method_selection: true,
        enable_adaptive_optimization: true,
        enable_quantum_optimization: true,
        enable_knowledge_transfer: true,
        target_accuracy: 1e-6,
        max_memory_mb: 2048,     // 2GB balanced
        monitoring_interval: 50, // Balanced monitoring
        enable_real_time_learning: true,
        enable_error_prediction: true,
        cache_size_limit: 1000,    // Balanced cache
        adaptation_threshold: 0.1, // Balanced sensitivity
        enable_hardware_optimization: true,
    };

    create_advanced_interpolation_coordinator(Some(config))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_create_default_coordinator() {
        let coordinator = create_default_advanced_coordinator::<f64>();
        assert!(coordinator.is_ok());
    }

    #[test]
    fn test_create_high_performance_coordinator() {
        let coordinator = create_high_performance_coordinator::<f64>();
        assert!(coordinator.is_ok());

        if let Ok(coord) = coordinator {
            let config = coord.get_config();
            assert_eq!(config.max_memory_mb, 8192);
            assert!(config.enable_quantum_optimization);
        }
    }

    #[test]
    fn test_create_memory_efficient_coordinator() {
        let coordinator = create_memory_efficient_coordinator::<f64>();
        assert!(coordinator.is_ok());

        if let Ok(coord) = coordinator {
            let config = coord.get_config();
            assert_eq!(config.max_memory_mb, 512);
            assert!(!config.enable_quantum_optimization);
        }
    }

    #[test]
    fn test_create_balanced_coordinator() {
        let coordinator = create_balanced_coordinator::<f64>();
        assert!(coordinator.is_ok());

        if let Ok(coord) = coordinator {
            let config = coord.get_config();
            assert_eq!(config.max_memory_mb, 2048);
            assert!(config.enable_adaptive_optimization);
        }
    }

    #[test]
    fn test_basic_recommendation() {
        let coordinator = create_default_advanced_coordinator::<f64>().unwrap();

        let x_data = Array1::linspace(0.0, 10.0, 11);
        let y_data = x_data.mapv(|x: f64| x.sin());

        let recommendation = coordinator.analyze_and_recommend(&x_data, &y_data);
        assert!(recommendation.is_ok());

        if let Ok(rec) = recommendation {
            assert!(rec.confidence_score >= 0.0 && rec.confidence_score <= 1.0);
            // Note: Cannot test specific method type without implementing pattern matching
        }
    }
}
