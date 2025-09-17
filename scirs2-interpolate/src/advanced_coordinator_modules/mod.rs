//! Advanced Interpolation Coordinator Modules
//!
//! This module contains the refactored components of the advanced interpolation
//! coordinator, broken down into focused, maintainable modules.

// Core types and configuration
pub mod config;
pub mod types;

// Core coordinator functionality
pub mod core_coordinator;

// Specialized optimization engines
pub mod accuracy_optimization;
pub mod method_selection;
pub mod pattern_analysis;
pub mod performance_tuning;
pub mod quantum_optimization;

// Knowledge and memory systems
pub mod knowledge_transfer;
pub mod memory_management;

// Public API re-exports
pub use config::*;
pub use core_coordinator::{
    create_advanced_interpolation_coordinator, AdvancedInterpolationCoordinator,
    CacheOptimizationResult, InterpolationPerformanceMetrics, InterpolationRecommendation,
    MethodPerformanceEstimate, SystemOptimizationResult,
};
pub use types::*;

// Re-export key types from specialized modules
pub use accuracy_optimization::AccuracyOptimizationEngine;
pub use knowledge_transfer::{CrossDomainInterpolationKnowledge, TransferKnowledgeResult};
pub use memory_management::{
    AdaptiveInterpolationCache, InterpolationMemoryManager, InterpolationPerformanceTracker,
    MemoryStatistics, PerformanceSummary,
};
pub use method_selection::IntelligentMethodSelector;
pub use pattern_analysis::{DataPatternAnalyzer, PatternAnalysisResult};
pub use performance_tuning::{PerformanceOptimizationResult, PerformanceTuningSystem};
pub use quantum_optimization::{QuantumOptimizationResult, QuantumParameterOptimizer};
