//! Optimization and performance enhancements for metrics computation

#![allow(clippy::manual_div_ceil)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::arc_with_non_send_sync)]
#![allow(clippy::await_holding_lock)]
#![allow(clippy::type_complexity)]
#![allow(clippy::manual_map)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]
//!
//! This module provides optimized implementations of metrics calculations
//! for improved performance, memory efficiency, and numerical stability.
//!
//! The optimization module contains four main components:
//!
//! 1. `parallel` - Tools for parallel computation of metrics
//! 2. `memory` - Tools for memory-efficient metrics computation
//! 3. `numeric` - Tools for numerically stable metrics computation
//! 4. `quantum_acceleration` - Quantum-inspired algorithms for exponential speedups
//!
//! # Examples
//!
//! ## Using StableMetrics for numerical stability
//!
//! ```
//! use scirs2_metrics::optimization::numeric::StableMetrics;
//! use scirs2_metrics::error::Result;
//!
//! fn compute_kl_divergence(p: &[f64], q: &[f64]) -> Result<f64> {
//!     let stable = StableMetrics::<f64>::new()
//!         .with_epsilon(1e-10)
//!         .with_clip_values(true);
//!     stable.kl_divergence(p, q)
//! }
//! ```
//!
//! ## Using parallel computation for batch metrics
//!
//! ```
//! use scirs2_metrics::optimization::parallel::{ParallelConfig, compute_metrics_batch};
//! use scirs2_metrics::error::Result;
//! use scirs2_core::ndarray::Array1;
//!
//! fn compute_multiple_metrics(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Vec<f64>> {
//!     let config = ParallelConfig {
//!         parallel_enabled: true,
//!         min_chunk_size: 1000,
//!         num_threads: None,
//!     };
//!
//!     let metrics: Vec<Box<dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<f64> + Send + Sync>> = vec![
//!         // Define your metric functions here
//!     ];
//!
//!     compute_metrics_batch(y_true, y_pred, &metrics, &config)
//! }
//! ```
//!
//! ## Using memory-efficient metrics for large datasets
//!
//! ```
//! use scirs2_metrics::optimization::memory::{StreamingMetric, ChunkedMetrics};
//! use scirs2_metrics::error::Result;
//!
//! struct StreamingMeanAbsoluteError;
//!
//! impl StreamingMetric<f64> for StreamingMeanAbsoluteError {
//!     type State = (f64, usize); // Running sum and count
//!     
//!     fn init_state(&self) -> Self::State {
//!         (0.0, 0)
//!     }
//!     
//!     fn update_state(&self, state: &mut Self::State, batch_true: &[f64], batch_pred: &[f64]) -> Result<()> {
//!         for (y_t, y_p) in batch_true.iter().zip(batch_pred.iter()) {
//!             state.0 += (y_t - y_p).abs();
//!             state.1 += 1;
//!         }
//!         Ok(())
//!     }
//!     
//!     fn finalize(&self, state: &Self::State) -> Result<f64> {
//!         if state.1 == 0 {
//!             return Err(scirs2_metrics::error::MetricsError::InvalidInput(
//!                 "No data processed".to_string()
//!             ));
//!         }
//!         Ok(state.0 / state.1 as f64)
//!     }
//! }
//! ```
//!
//! ## Using quantum-inspired acceleration for large-scale computations
//!
//! ```
//! use scirs2_metrics::optimization::quantum_acceleration::{QuantumMetricsComputer, QuantumConfig};
//! use scirs2_metrics::error::Result;
//! use scirs2_core::ndarray::Array1;
//!
//! fn compute_quantum_correlation(x: &Array1<f64>, y: &Array1<f64>) -> Result<f64> {
//!     let config = QuantumConfig::default();
//!     let mut quantum_computer = QuantumMetricsComputer::new(config)?;
//!     quantum_computer.quantum_correlation(&x.view(), &y.view())
//! }
//! ```

// Re-export submodules
pub mod advanced_memory_optimization;
pub mod distributed;
pub mod distributed_advanced;
pub mod enhanced_gpu_kernels;
pub mod gpu_acceleration;
pub mod gpu_kernels;
pub mod hardware;
pub mod memory;
pub mod numeric;
pub mod parallel;
pub mod quantum_acceleration;
pub mod simd_gpu;

// Re-export common functionality
pub use advanced_memory_optimization::{
    AdvancedMemoryPool, AllocationStrategy, BlockType, MemoryBlock, MemoryPoolConfig, MemoryStats,
    StrategyBenchmark,
};
pub use distributed::{
    DistributedConfig, DistributedMetricsBuilder, DistributedMetricsCoordinator,
};
pub use distributed_advanced::{
    AdvancedClusterConfig, AdvancedDistributedCoordinator, AutoScalingConfig, ClusterState,
    ConsensusAlgorithm, ConsensusConfig, DistributedTask, FaultToleranceConfig, LocalityConfig,
    NodeInfo, NodeRole, NodeStatus, OptimizationConfig, ResourceRequirements, ShardingConfig,
    ShardingStrategy, TaskPriority, TaskType,
};
pub use gpu_acceleration::{BenchmarkResults, GpuAccelConfig, GpuInfo, GpuMetricsComputer};
pub use gpu_kernels::{
    AdvancedGpuComputer, BatchSettings, CudaContext, ErrorHandling, GpuApi, GpuComputeConfig,
    GpuComputeResults, GpuPerformanceStats, KernelMetrics, KernelOptimization, MemoryStrategy,
    OpenClContext, TransferMetrics, VectorizationLevel,
};
pub use hardware::{
    HardwareAccelConfig, HardwareAcceleratedMatrix, HardwareCapabilities, SimdDistanceMetrics,
    SimdStatistics, VectorWidth,
};
pub use memory::{ChunkedMetrics, StreamingMetric};
pub use numeric::{StableMetric, StableMetrics};
pub use parallel::ParallelConfig;
pub use quantum_acceleration::{
    InterferencePatterns, QuantumBenchmarkResults, QuantumConfig, QuantumMetricsComputer,
    QuantumProcessor, SuperpositionManager, VqeParameters,
};
pub use simd_gpu::SimdMetrics;
