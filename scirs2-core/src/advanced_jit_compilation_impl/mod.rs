//! Advanced JIT Compilation Framework
//!
//! This module provides a comprehensive Just-In-Time (JIT) compilation framework
//! with LLVM integration for runtime optimization in Advanced mode. It enables
//! dynamic code generation, runtime optimization, and adaptive compilation strategies
//! to maximize performance for scientific computing workloads.
//!
//! # Features
//!
//! - **LLVM-based Code Generation**: Advanced optimization through LLVM infrastructure
//! - **Runtime Kernel Compilation**: JIT compilation of computational kernels
//! - **Adaptive Optimization**: Dynamic optimization based on runtime characteristics
//! - **Cross-platform Support**: Native code generation for multiple architectures
//! - **Intelligent Caching**: Smart caching of compiled code with automatic invalidation
//! - **Performance Profiling**: Integrated profiling for continuous optimization
//! - **Template-based Specialization**: Automatic code specialization for specific data types
//! - **Vectorization**: Automatic SIMD optimization for mathematical operations

// Module declarations
pub mod analytics;
pub mod cache;
pub mod code_generator;
pub mod compiler;
pub mod config;
pub mod llvm_engine;
pub mod neuromorphic;
pub mod optimizer;
pub mod profiler;

// Re-export main types for public API
pub use analytics::{CompilationStatistics, JitAnalytics, MemoryUsageStats};
pub use cache::{
    CacheHitRates, CacheStatistics, CachedKernel, CompiledKernel, KernelMetadata,
    KernelPerformance, MemoryAccessPattern,
};
pub use code_generator::{
    AdaptiveCodeGenerator, CodeGenerationRule, CodeTemplate, GenerationStatistics, SpecializedCode,
    TargetCodeGenerator, TemplateParameter,
};
pub use compiler::AdvancedJitCompiler;
pub use config::{
    CacheConfig, EvictionPolicy, JitCompilerConfig, NeuromorphicConfig, PatternCacheConfig,
    ProfilerConfig, ProfilingSessionConfig,
};
pub use llvm_engine::{
    CodeModel, CodeSizeMetrics, CompilationError, CompilationMetadata, CompilationStatus,
    CompiledModule, CustomPass, ErrorLocation, ErrorSeverity, FunctionPass, LlvmCompilationEngine,
    LlvmContext, LoopPass, ModulePass, ModulePerformance, OptimizationPasses, RelocationModel,
    TargetMachine,
};
pub use neuromorphic::{
    BurstPattern, CompiledSNN, Connection, ConnectionPattern, DynamicsModel, DynamicsType,
    EventDrivenOptimizer, EventPerformanceMetrics, EventQueue, EventType, FourierComponent,
    FrequencySpectrum, Layer, LayerType, LearningEvent, NetworkTopology, NeuralNetwork,
    NeuromorphicJitCompiler, NeuronModel, NeuronType, PatternOptimization, PatternUsage,
    PlasticityRule, PlasticityStatistics, PlasticityType, PopulationStatistics,
    SpikeCharacteristics, SpikeEvent, SpikeOptimizationResult, SpikePattern, SpikePatternCache,
    SpikePerformancePrediction, SpikingNeuralNetworkCompiler, SynapseModel, SynapseType,
    SynapticPlasticityEngine, TemporalDynamicsCompiler, TemporalPattern, TemporalStatistics,
};
pub use optimizer::{
    AdaptationEvent, AdaptationRule, OptimizationCandidate, OptimizationFailure,
    OptimizationResults, OptimizationState, OptimizationStrategy, PerformanceAnalysis,
    PerformanceFeedback, PerformanceImprovement, RuntimeOptimizer,
};
pub use profiler::{
    CompilationProfile, ComplexityLevel, ExecutionProfile, Hotspot, JitProfiler, OpportunityType,
    OptimizationOpportunity, PerformanceCounters, ProfilerAnalytics, ProfilingSample,
    ProfilingSession,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_compiler_creation() {
        let compiler = AdvancedJitCompiler::new();
        assert!(compiler.is_ok());
    }

    #[test]
    fn test_jit_compiler_config() {
        let config = JitCompilerConfig::default();
        assert!(config.enable_aggressive_optimization);
        assert!(config.enable_vectorization);
        assert_eq!(config.optimization_level, 3);
    }

    #[test]
    fn test_llvm_engine_creation() {
        let config = JitCompilerConfig::default();
        let engine = LlvmCompilationEngine::new(&config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_kernel_cache_creation() {
        let config = JitCompilerConfig::default();
        let cache = cache::KernelCache::new(&config);
        assert!(cache.is_ok());
    }

    #[test]
    fn test_profiler_creation() {
        let config = JitCompilerConfig::default();
        let profiler = JitProfiler::new(&config);
        assert!(profiler.is_ok());
    }
}
