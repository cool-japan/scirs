//! GPU-accelerated linear algebra operations
//!
//! This module provides a comprehensive set of GPU-accelerated linear algebra operations
//! with automatic CPU/GPU dispatching, performance optimization, and intelligent workload
//! analysis.

// Module declarations
pub mod advanced;
pub mod dispatcher;
pub mod hardware;
pub mod intelligent;
pub mod kernels;
pub mod metrics;
pub mod optimization;
pub mod profiling;

#[cfg(test)]
mod tests;

// Re-export main public APIs
pub use advanced::AdvancedGpuOperations;
pub use dispatcher::{GpuOperationDispatcher, DEFAULT_GPU_THRESHOLD};
pub use hardware::{DeviceProfile, HardwareCapabilityProfiler};
pub use intelligent::{
    AdvancedIntelligentGpuDispatcher, DataCharacteristics, DispatchDecision,
    GpuPerformancePredictor, MemoryAccessPattern, ModelCoefficients, OptimalChoice,
    PerformancePrediction, WorkloadAnalysis, WorkloadAnalyzer,
};
pub use kernels::{AutoTuneResults, BenchmarkResults, GpuKernelManager, OptimizationLevel};
pub use metrics::{
    EnergyMetrics, MemoryMetrics, MultiDimensionalMetrics, RunningStats, ThroughputMetrics,
    TimeMetrics,
};
pub use optimization::{BatchPerformanceRecord, BatchSizeOptimizer};
pub use profiling::GpuPerformanceProfiler;
