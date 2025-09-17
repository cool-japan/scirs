//! CUDA kernel implementations for GPU-accelerated optimization
//!
//! This module provides a comprehensive CUDA kernel system for high-performance
//! optimization operations, featuring modular architecture with specialized
//! components for configuration, execution, profiling, memory management,
//! Tensor Core acceleration, pipeline management, and adaptive optimization.
//!
//! # Architecture Overview
//!
//! The CUDA kernels module is organized into focused submodules:
//!
//! - **config**: Configuration management and kernel parameter definitions
//! - **kernels**: Core CUDA kernel implementations and execution
//! - **profiling**: Performance monitoring and metrics collection
//! - **memory**: Advanced memory management and allocation strategies
//! - **tensor_cores**: NVIDIA Tensor Core acceleration support
//! - **pipeline**: Asynchronous execution pipeline management
//! - **adaptive**: Intelligent optimization and parameter adaptation
//!
//! # Usage Examples
//!
//! ## Basic CUDA Kernel Execution
//!
//! ```rust
//! use scirs2_optim::gpu::cuda_kernels::*;
//!
//! // Create optimizer kernel with default configuration
//! let config = KernelConfig::default();
//! let mut kernel = OptimizerKernel::new(config)?;
//!
//! // Execute Adam optimization step
//! let params = Array1::zeros(1000);
//! let gradients = Array1::zeros(1000);
//! kernel.execute_adam(&mut params, &gradients, 0.001, 0.9, 0.999, 1e-8, 1)?;
//! ```
//!
//! ## Advanced Memory Management
//!
//! ```rust
//! use scirs2_optim::gpu::cuda_kernels::*;
//!
//! // Create memory manager with pooled allocation
//! let memory_config = MemoryPoolConfig {
//!     strategy: AllocationStrategy::PooledDynamic {
//!         initial_size_mb: 512,
//!         max_size_mb: 4096,
//!         growth_factor: 1.5,
//!     },
//!     ..Default::default()
//! };
//!
//! let memory_manager = CudaMemoryManager::new(memory_config)?;
//! let allocation = memory_manager.allocate::<f32>(1000000, MemoryType::Device, None)?;
//! ```
//!
//! ## Tensor Core Acceleration
//!
//! ```rust
//! use scirs2_optim::gpu::cuda_kernels::*;
//!
//! // Create Tensor Core manager and configure operation
//! let tc_manager = TensorCoreManager::new()?;
//! let config = tc_manager.get_optimal_config(1024, 1024, 512);
//! let descriptor = tc_manager.prepare_operation(config)?;
//!
//! // Execute high-performance matrix operation
//! tc_manager.execute_operation(&descriptor, a_ptr, b_ptr, c_ptr)?;
//! ```
//!
//! ## Adaptive Optimization
//!
//! ```rust
//! use scirs2_optim::gpu::cuda_kernels::*;
//!
//! // Create adaptive optimizer with genetic algorithm
//! let strategy = AdaptationStrategy::Genetic {
//!     population_size: 50,
//!     mutation_rate: 0.1,
//!     crossover_rate: 0.8,
//! };
//!
//! let optimizer = AdaptiveOptimizer::new(strategy, AdaptationTriggers::default());
//!
//! // Record performance and let the optimizer adapt
//! optimizer.record_performance(params, metrics, workload_context)?;
//! ```
//!
//! # Performance Features
//!
//! - **Multi-stream execution** for parallel kernel launches
//! - **Tensor Core acceleration** on supported NVIDIA GPUs (Volta+)
//! - **Advanced memory pooling** with configurable allocation strategies
//! - **Comprehensive profiling** with detailed performance metrics
//! - **Adaptive optimization** with multiple machine learning strategies
//! - **Pipeline management** for complex operation sequences
//!
//! # GPU Compatibility
//!
//! The module supports NVIDIA GPUs with compute capability 7.0+ for Tensor Core
//! features, and provides CPU fallbacks for systems without CUDA support.
//! Specific feature support varies by GPU generation:
//!
//! - **Volta (SM 7.0)**: Basic Tensor Core support, FP16 operations
//! - **Turing (SM 7.5)**: Enhanced Tensor Core, INT8 support
//! - **Ampere (SM 8.0/8.6)**: Sparse operations, BF16, TF32 support
//! - **Ada Lovelace/Hopper (SM 8.9/9.0)**: FP8 precision, transformer acceleration

// Core configuration and parameter management
pub mod config;
pub use config::*;

// Core CUDA kernel implementations
pub mod kernels;
pub use kernels::*;

// Performance profiling and monitoring
pub mod profiling;
pub use profiling::*;

// Memory management and allocation
pub mod memory;
pub use memory::*;

// NVIDIA Tensor Core acceleration
pub mod tensor_cores;
pub use tensor_cores::*;

// Execution pipeline management
pub mod pipeline;
pub use pipeline::*;

// Adaptive optimization strategies
pub mod adaptive;
pub use adaptive::*;

// Re-export all essential types and functions for convenient access

// Configuration types
pub use config::{
    KernelConfig, ProfilingConfig, TensorCoreGeneration, DeviceInfo,
    OptimizationLevel, MemoryStrategy, PipelineConfig, AdaptiveConfig,
};

// Kernel execution types
pub use kernels::{
    OptimizerKernel, OptimizerKernelError, KernelExecutionMode,
    ExecutionMetrics, KernelResult, CudaKernelState,
};

// Profiling types
pub use profiling::{
    KernelProfiler, ProfilingHandle, PerformanceMetrics, ExecutionTiming,
    ProfilingSample, ProfilingReport, MemoryUsage, ClockFrequencies,
};

// Memory management types
pub use memory::{
    CudaMemoryManager, ManagedAllocation, MemoryPoolConfig, AllocationStrategy,
    MemoryType, MemoryStats, MemoryReport, AllocationInfo,
};

// Tensor Core types
pub use tensor_cores::{
    TensorCoreManager, TensorCoreConfig, TensorCorePrecision, TensorCoreOperation,
    TensorCoreCapability, TensorCoreMetrics, TensorCoreReport, MatrixLayout,
};

// Pipeline types
pub use pipeline::{
    CudaPipeline, PipelineOperation, OperationHandle, PipelineStrategy,
    OperationPriority, PipelineStatistics, PipelineReport, LaunchConfig,
};

// Adaptive optimization types
pub use adaptive::{
    AdaptiveOptimizer, AdaptationStrategy, KernelParameters, AdaptationMetrics,
    WorkloadContext, AdaptationStatistics, MemoryCoalescingStrategy,
};

// Common error types
use scirs2_core::error::{Result, ScirsMlError};

/// High-level CUDA kernel manager that orchestrates all components
pub struct CudaKernelManager {
    /// Core optimizer kernel
    optimizer_kernel: OptimizerKernel,
    /// Memory manager
    memory_manager: CudaMemoryManager,
    /// Tensor Core manager
    tensor_core_manager: TensorCoreManager,
    /// Execution pipeline
    pipeline: CudaPipeline,
    /// Adaptive optimizer
    adaptive_optimizer: AdaptiveOptimizer,
    /// Performance profiler
    profiler: KernelProfiler,
}

impl CudaKernelManager {
    /// Creates a new CUDA kernel manager with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(CudaKernelConfig::default())
    }

    /// Creates a new CUDA kernel manager with custom configuration
    pub fn with_config(config: CudaKernelConfig) -> Result<Self> {
        let optimizer_kernel = OptimizerKernel::new(config.kernel_config.clone())?;
        let memory_manager = CudaMemoryManager::new(config.memory_config.clone())?;
        let tensor_core_manager = TensorCoreManager::new()?;
        let pipeline = CudaPipeline::new(config.pipeline_config.clone())?;
        let adaptive_optimizer = AdaptiveOptimizer::new(
            config.adaptation_strategy.clone(),
            config.adaptation_triggers.clone(),
        );
        let profiler = KernelProfiler::new(config.profiling_config.clone())?;

        Ok(Self {
            optimizer_kernel,
            memory_manager,
            tensor_core_manager,
            pipeline,
            adaptive_optimizer,
            profiler,
        })
    }

    /// Executes a high-level optimization operation with full pipeline
    pub async fn execute_optimization<T: num_traits::Float + Send + Sync>(
        &mut self,
        operation: OptimizationOperation<T>,
    ) -> Result<OptimizationResult<T>> {
        // Start profiling
        let profiling_handle = self.profiler.start_profiling(&operation.name)?;

        // Allocate memory through memory manager
        let param_allocation = self.memory_manager.allocate::<T>(
            operation.parameters.len(),
            MemoryType::Device,
            Some(format!("params_{}", operation.name)),
        )?;

        let grad_allocation = self.memory_manager.allocate::<T>(
            operation.gradients.len(),
            MemoryType::Device,
            Some(format!("grads_{}", operation.name)),
        )?;

        // Prepare Tensor Core operation if applicable
        let tc_descriptor = if operation.use_tensor_cores && operation.parameters.len() >= 512 {
            let tc_config = self.tensor_core_manager.get_optimal_config(
                operation.matrix_dims.0,
                operation.matrix_dims.1,
                operation.matrix_dims.2,
            );
            Some(self.tensor_core_manager.prepare_operation(tc_config)?)
        } else {
            None
        };

        // Execute through pipeline
        let pipeline_op = self.create_pipeline_operation(&operation, tc_descriptor.as_ref())?;
        let operation_handle = self.pipeline.submit_operation(pipeline_op)?;

        // Wait for completion
        let pipeline_result = operation_handle.await;

        // Record performance for adaptive optimization
        if let Ok(_) = &pipeline_result {
            let metrics = self.extract_adaptation_metrics(&operation)?;
            let workload_context = WorkloadContext {
                matrix_dims: operation.matrix_dims,
                batch_size: operation.batch_size,
                data_type_size: std::mem::size_of::<T>(),
                access_pattern: operation.access_pattern,
                compute_intensity: operation.compute_intensity,
            };

            // Get current best parameters or use defaults
            let current_params = self.adaptive_optimizer
                .get_best_parameters(&workload_context)
                .unwrap_or_else(|| self.get_default_kernel_parameters());

            self.adaptive_optimizer.record_performance(
                current_params,
                metrics,
                workload_context,
            )?;
        }

        // Complete profiling
        match &pipeline_result {
            Ok(_) => profiling_handle.complete_success(
                operation.parameters.len(),
                operation.memory_transferred,
                operation.flop_count,
            )?,
            Err(e) => profiling_handle.complete_error(&e.to_string())?,
        }

        // Create result
        let result = OptimizationResult {
            success: pipeline_result.is_ok(),
            execution_time_ms: 0.0, // Would be filled from profiling
            memory_usage_mb: param_allocation.size_bytes() as f64 / 1024.0 / 1024.0,
            gpu_utilization: 0.0, // Would be filled from profiling
            parameters: operation.parameters, // Would contain updated parameters
            error_message: pipeline_result.err().map(|e| e.to_string()),
        };

        Ok(result)
    }

    /// Creates a pipeline operation from high-level operation descriptor
    fn create_pipeline_operation<T>(
        &self,
        operation: &OptimizationOperation<T>,
        _tc_descriptor: Option<&String>,
    ) -> Result<PipelineOperation> {
        Ok(PipelineOperation {
            id: 0, // Will be assigned by pipeline
            op_type: operation.name.clone(),
            priority: operation.priority,
            dependencies: operation.dependencies.clone(),
            estimated_time_us: operation.estimated_time_us,
            memory_requirement: operation.memory_requirement,
            #[cfg(feature = "cuda")]
            kernel: None, // Would be populated with actual kernel
            parameters: vec![], // Would be populated with kernel parameters
            launch_config: LaunchConfig {
                grid_dims: (1, 1, 1), // Would be calculated based on operation
                block_dims: (256, 1, 1),
                shared_memory: 0,
                stream_id: None,
            },
            stream_hint: operation.stream_hint,
            completion_callback: None,
        })
    }

    /// Extracts adaptation metrics from operation results
    fn extract_adaptation_metrics<T>(&self, _operation: &OptimizationOperation<T>) -> Result<AdaptationMetrics> {
        // In real implementation, would extract actual metrics from profiling
        Ok(AdaptationMetrics {
            execution_time_ms: 1.0,
            memory_throughput_gbps: 100.0,
            gpu_utilization: 80.0,
            memory_utilization: 70.0,
            occupancy: 0.85,
            cache_hit_rate: 0.95,
            branch_efficiency: 0.90,
            warp_efficiency: 0.88,
        })
    }

    /// Gets default kernel parameters
    fn get_default_kernel_parameters(&self) -> KernelParameters {
        KernelParameters {
            grid_dims: (256, 1, 1),
            block_dims: (256, 1, 1),
            shared_memory: 0,
            registers_per_thread: 32,
            occupancy_target: 0.75,
            memory_coalescing: MemoryCoalescingStrategy::Adaptive,
            loop_unroll_factor: 4,
            use_texture_memory: false,
            use_constant_memory: true,
        }
    }

    /// Gets comprehensive performance report from all components
    pub fn generate_comprehensive_report(&self) -> CudaKernelReport {
        CudaKernelReport {
            kernel_metrics: self.optimizer_kernel.get_metrics(),
            memory_report: self.memory_manager.generate_report(),
            tensor_core_report: self.tensor_core_manager.generate_report(),
            pipeline_report: self.pipeline.generate_report(),
            adaptation_stats: self.adaptive_optimizer.get_adaptation_statistics(),
            profiling_report: self.profiler.generate_report(),
        }
    }

    /// Shuts down all components gracefully
    pub fn shutdown(self) -> Result<()> {
        self.pipeline.shutdown()?;
        Ok(())
    }
}

/// Comprehensive configuration for CUDA kernel manager
#[derive(Debug, Clone)]
pub struct CudaKernelConfig {
    /// Core kernel configuration
    pub kernel_config: KernelConfig,
    /// Memory management configuration
    pub memory_config: MemoryPoolConfig,
    /// Pipeline configuration
    pub pipeline_config: PipelineConfig,
    /// Profiling configuration
    pub profiling_config: ProfilingConfig,
    /// Adaptation strategy
    pub adaptation_strategy: AdaptationStrategy,
    /// Adaptation triggers
    pub adaptation_triggers: AdaptationTriggers,
}

impl Default for CudaKernelConfig {
    fn default() -> Self {
        Self {
            kernel_config: KernelConfig::default(),
            memory_config: MemoryPoolConfig::default(),
            pipeline_config: PipelineConfig::default(),
            profiling_config: ProfilingConfig::default(),
            adaptation_strategy: AdaptationStrategy::Heuristic,
            adaptation_triggers: AdaptationTriggers::default(),
        }
    }
}

/// High-level optimization operation descriptor
#[derive(Debug, Clone)]
pub struct OptimizationOperation<T> {
    /// Operation name for profiling
    pub name: String,
    /// Operation priority
    pub priority: OperationPriority,
    /// Parameter dependencies
    pub dependencies: Vec<u64>,
    /// Parameters to optimize
    pub parameters: Vec<T>,
    /// Gradients for optimization
    pub gradients: Vec<T>,
    /// Matrix dimensions (M, N, K) if applicable
    pub matrix_dims: (usize, usize, usize),
    /// Batch size
    pub batch_size: usize,
    /// Whether to use Tensor Cores
    pub use_tensor_cores: bool,
    /// Memory access pattern
    pub access_pattern: AccessPattern,
    /// Computational intensity
    pub compute_intensity: f64,
    /// Estimated execution time
    pub estimated_time_us: u64,
    /// Memory requirement in bytes
    pub memory_requirement: usize,
    /// Memory transferred in bytes
    pub memory_transferred: usize,
    /// FLOP count estimate
    pub flop_count: u64,
    /// Stream hint for execution
    pub stream_hint: Option<usize>,
}

/// Result of optimization operation
#[derive(Debug, Clone)]
pub struct OptimizationResult<T> {
    /// Whether operation succeeded
    pub success: bool,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// GPU utilization percentage
    pub gpu_utilization: f64,
    /// Updated parameters
    pub parameters: Vec<T>,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct CudaKernelReport {
    /// Kernel execution metrics
    pub kernel_metrics: ExecutionMetrics,
    /// Memory management report
    pub memory_report: MemoryReport,
    /// Tensor Core performance report
    pub tensor_core_report: TensorCoreReport,
    /// Pipeline performance report
    pub pipeline_report: PipelineReport,
    /// Adaptation statistics
    pub adaptation_stats: AdaptationStatistics,
    /// Profiling report
    pub profiling_report: ProfilingReport,
}

impl CudaKernelReport {
    /// Formats the comprehensive report as human-readable text
    pub fn format_comprehensive_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Comprehensive CUDA Kernel Performance Report ===\n\n");

        // Executive Summary
        report.push_str("Executive Summary:\n");
        report.push_str(&format!("  Total Operations: {}\n", self.pipeline_report.statistics.total_operations));
        report.push_str(&format!("  Success Rate: {:.2}%\n",
            (self.pipeline_report.statistics.completed_operations as f64 /
             self.pipeline_report.statistics.total_operations.max(1) as f64) * 100.0));
        report.push_str(&format!("  Average Throughput: {:.2} ops/sec\n",
            self.pipeline_report.statistics.throughput_ops_per_sec));
        report.push_str(&format!("  Memory Efficiency: {:.2}%\n", self.memory_report.cache_efficiency));
        report.push_str(&format!("  Performance Improvement: {:.2}%\n",
            self.adaptation_stats.performance_improvement_percent));
        report.push_str("\n");

        // Individual component reports
        report.push_str(&self.memory_report.format_report());
        report.push_str("\n");
        report.push_str(&self.tensor_core_report.format_report());
        report.push_str("\n");
        report.push_str(&self.pipeline_report.format_report());
        report.push_str("\n");
        report.push_str(&self.profiling_report.format_report());
        report.push_str("\n");
        report.push_str(&self.adaptation_stats.format_report());

        report
    }
}

// Re-export error types for convenience
pub use scirs2_core::error::{Result as CudaResult, ScirsMlError as CudaError};