//! Configuration and core types for CUDA kernel optimization
//!
//! This module contains configuration structures and fundamental types
//! used throughout the CUDA kernel system for GPU-accelerated optimization.

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Profiling configuration for CUDA kernels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable detailed profiling
    pub enable_profiling: bool,

    /// Profiling sample rate (0.0-1.0)
    pub sample_rate: f32,

    /// Maximum profiling samples to retain
    pub max_samples: usize,

    /// Profile memory transfers
    pub profile_memory_transfers: bool,

    /// Profile kernel execution times
    pub profile_kernel_times: bool,

    /// Profile GPU utilization
    pub profile_gpu_utilization: bool,

    /// Profiling output format
    pub output_format: ProfilingOutputFormat,
}

/// Profiling output formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProfilingOutputFormat {
    Json,
    Csv,
    Binary,
    Nvprof,
}

/// Performance metrics for kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Kernel execution time
    pub execution_time: Duration,

    /// Memory transfer time
    pub memory_transfer_time: Duration,

    /// GPU utilization percentage
    pub gpu_utilization: f32,

    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f32,

    /// Tensor core utilization (if available)
    pub tensor_core_utilization: Option<f32>,

    /// Power consumption in watts
    pub power_consumption: Option<f32>,

    /// Achieved memory throughput (GB/s)
    pub memory_throughput: f32,

    /// Achieved compute throughput (FLOPS)
    pub compute_throughput: f64,

    /// Cache hit rate
    pub cache_hit_rate: Option<f32>,

    /// Occupancy percentage
    pub occupancy: f32,

    /// Number of registers used per thread
    pub registers_per_thread: u32,

    /// Shared memory usage per block
    pub shared_memory_per_block: u32,

    /// Grid dimensions
    pub grid_dimensions: (u32, u32, u32),

    /// Block dimensions
    pub block_dimensions: (u32, u32, u32),
}

/// Tensor Core support configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCoreSupport {
    /// Tensor Core availability
    pub available: bool,

    /// Supported generations
    pub supported_generations: Vec<TensorCoreGeneration>,

    /// Current generation in use
    pub current_generation: Option<TensorCoreGeneration>,

    /// Supported data types
    pub supported_data_types: Vec<TensorCoreDataType>,

    /// Optimal matrix sizes
    pub optimal_sizes: Vec<(u32, u32, u32)>, // (M, N, K)

    /// Performance characteristics
    pub performance_characteristics: TensorCorePerformanceProfile,
}

/// Tensor Core generations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorCoreGeneration {
    V1,  // Volta
    V2,  // Turing
    V3,  // Ampere
    V4,  // Ada Lovelace / Hopper
}

/// Supported data types for Tensor Cores
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorCoreDataType {
    FP16,
    BF16,
    INT8,
    INT4,
    FP8,
    TF32,
}

/// Tensor Core performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCorePerformanceProfile {
    /// Peak TOPS (Tera Operations Per Second)
    pub peak_tops: f64,

    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f32,

    /// Optimal tile sizes for different data types
    pub optimal_tile_sizes: HashMap<TensorCoreDataType, (u32, u32, u32)>,

    /// Efficiency factors
    pub efficiency_factors: HashMap<TensorCoreDataType, f32>,
}

/// Mixed precision training support configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionSupport {
    /// Enable mixed precision
    pub enabled: bool,

    /// Loss scaling factor
    pub loss_scale: f32,

    /// Automatic loss scaling
    pub auto_loss_scaling: bool,

    /// Growth factor for loss scaling
    pub growth_factor: f32,

    /// Backoff factor for loss scaling
    pub backoff_factor: f32,

    /// Growth interval
    pub growth_interval: u32,

    /// Supported precisions
    pub supported_precisions: Vec<Precision>,

    /// Default precision for forward pass
    pub forward_precision: Precision,

    /// Default precision for backward pass
    pub backward_precision: Precision,

    /// Precision for master weights
    pub master_weights_precision: Precision,
}

/// Precision types for mixed precision training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    TF32,
    FP8,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Default CUDA allocator
    Default,

    /// Memory pool allocation
    Pool,

    /// Unified memory allocation
    Unified,

    /// Pinned memory allocation
    Pinned,

    /// Managed memory allocation
    Managed,

    /// Custom allocation strategy
    Custom,
}

/// Pipeline configuration for kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Number of pipeline stages
    pub num_stages: usize,

    /// Enable asynchronous execution
    pub async_execution: bool,

    /// Stream priority
    pub stream_priority: i32,

    /// Enable memory prefetching
    pub enable_prefetch: bool,

    /// Overlap computation with communication
    pub overlap_compute_comm: bool,

    /// Buffer size for each stage
    pub stage_buffer_sizes: Vec<usize>,

    /// Synchronization points
    pub sync_points: Vec<PipelineSyncPoint>,
}

/// Pipeline synchronization points
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PipelineSyncPoint {
    /// Synchronize before memory transfer
    BeforeMemoryTransfer,

    /// Synchronize after memory transfer
    AfterMemoryTransfer,

    /// Synchronize before kernel execution
    BeforeKernelExecution,

    /// Synchronize after kernel execution
    AfterKernelExecution,

    /// Custom synchronization point
    Custom(u32),
}

/// Pipeline execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatistics {
    /// Total execution time
    pub total_execution_time: Duration,

    /// Time per stage
    pub stage_execution_times: Vec<Duration>,

    /// Pipeline efficiency
    pub pipeline_efficiency: f32,

    /// Bottleneck stage
    pub bottleneck_stage: Option<usize>,

    /// Throughput (operations per second)
    pub throughput: f64,

    /// Memory transfer overhead
    pub memory_transfer_overhead: f32,
}

/// Adaptive kernel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveKernelConfig<T: Float> {
    /// Enable adaptive optimization
    pub enable_adaptive: bool,

    /// Learning rate for adaptation
    pub adaptation_learning_rate: T,

    /// Minimum performance threshold
    pub min_performance_threshold: T,

    /// Maximum adaptation iterations
    pub max_adaptation_iterations: usize,

    /// Adaptation strategy
    pub adaptation_strategy: AdaptationStrategy,

    /// Performance metrics to optimize for
    pub target_metrics: Vec<TargetMetric>,

    /// Constraints on adaptation
    pub constraints: AdaptationConstraints<T>,
}

/// Adaptation strategies for kernel optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Gradient-based optimization
    GradientBased,

    /// Genetic algorithm
    Genetic,

    /// Simulated annealing
    SimulatedAnnealing,

    /// Bayesian optimization
    Bayesian,

    /// Random search
    RandomSearch,

    /// Grid search
    GridSearch,
}

/// Target metrics for optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TargetMetric {
    /// Minimize execution time
    ExecutionTime,

    /// Maximize throughput
    Throughput,

    /// Minimize power consumption
    PowerConsumption,

    /// Maximize GPU utilization
    GpuUtilization,

    /// Minimize memory usage
    MemoryUsage,

    /// Composite metric
    Composite,
}

/// Constraints for adaptive optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConstraints<T: Float> {
    /// Maximum memory usage
    pub max_memory_usage: Option<usize>,

    /// Maximum execution time
    pub max_execution_time: Option<Duration>,

    /// Minimum accuracy
    pub min_accuracy: Option<T>,

    /// Maximum power consumption
    pub max_power_consumption: Option<f32>,

    /// Resource limits
    pub resource_limits: ResourceLimits,
}

/// Resource limits for kernel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum number of threads per block
    pub max_threads_per_block: u32,

    /// Maximum shared memory per block
    pub max_shared_memory_per_block: u32,

    /// Maximum registers per thread
    pub max_registers_per_thread: u32,

    /// Maximum grid size
    pub max_grid_size: (u32, u32, u32),

    /// Maximum texture memory
    pub max_texture_memory: Option<usize>,

    /// Maximum constant memory
    pub max_constant_memory: Option<usize>,
}

/// Adaptive precision selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AdaptivePrecision {
    /// Fixed precision
    Fixed(Precision),

    /// Dynamic precision selection
    Dynamic {
        /// Available precisions
        available: &'static [Precision],
        /// Selection criteria
        criteria: PrecisionSelectionCriteria,
    },

    /// Learning-based precision selection
    Learned {
        /// Initial precision
        initial: Precision,
        /// Adaptation rate
        adaptation_rate: f32,
    },
}

/// Criteria for precision selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PrecisionSelectionCriteria {
    /// Based on accuracy requirements
    Accuracy,

    /// Based on performance requirements
    Performance,

    /// Based on memory constraints
    Memory,

    /// Balanced approach
    Balanced,
}

/// Tensor Core performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCoreMetrics {
    /// Utilization percentage
    pub utilization: f32,

    /// Achieved TOPS
    pub achieved_tops: f64,

    /// Efficiency compared to peak
    pub efficiency: f32,

    /// Optimal tile size usage
    pub optimal_tile_usage: f32,
}

/// Comprehensive kernel execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelStatistics {
    /// Total kernels executed
    pub total_kernels_executed: u64,

    /// Average execution time
    pub average_execution_time: Duration,

    /// Peak performance achieved
    pub peak_performance: f64,

    /// Memory efficiency
    pub memory_efficiency: f32,

    /// Error rate
    pub error_rate: f32,
}

/// Performance report for optimization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report generation timestamp
    pub timestamp: std::time::SystemTime,

    /// Overall performance summary
    pub summary: PerformanceMetrics,

    /// Detailed kernel statistics
    pub kernel_stats: KernelStatistics,

    /// Tensor Core metrics (if available)
    pub tensor_core_metrics: Option<TensorCoreMetrics>,

    /// Recommendations for optimization
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,

    /// Description
    pub description: String,

    /// Expected performance improvement
    pub expected_improvement: f32,

    /// Implementation difficulty
    pub implementation_difficulty: DifficultyLevel,

    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Adjust block size
    BlockSizeOptimization,

    /// Optimize memory access patterns
    MemoryAccessOptimization,

    /// Use Tensor Cores
    TensorCoreUtilization,

    /// Enable mixed precision
    MixedPrecisionOptimization,

    /// Improve pipeline efficiency
    PipelineOptimization,

    /// Reduce memory footprint
    MemoryFootprintReduction,

    /// Increase parallelism
    ParallelismIncrease,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Resource requirements for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Additional memory required
    pub additional_memory: Option<usize>,

    /// Additional compute resources
    pub additional_compute: Option<f32>,

    /// Development time estimate
    pub development_time: Option<Duration>,

    /// Expertise level required
    pub expertise_level: ExpertiseLevel,
}

/// Expertise levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    Beginner,
    Intermediate,
    Advanced,
    Expert,
}

/// Kernel optimization errors
#[derive(Debug, Clone)]
pub enum OptimizerKernelError {
    /// CUDA context creation failed
    CudaContextCreationFailed(String),

    /// Kernel compilation failed
    KernelCompilationFailed(String),

    /// Memory allocation failed
    MemoryAllocationFailed(String),

    /// Kernel execution failed
    KernelExecutionFailed(String),

    /// Unsupported operation
    UnsupportedOperation(String),

    /// Invalid configuration
    InvalidConfiguration(String),

    /// Performance degradation detected
    PerformanceDegradation(String),

    /// Resource limit exceeded
    ResourceLimitExceeded(String),

    /// Precision conversion error
    PrecisionConversionError(String),

    /// Pipeline synchronization error
    PipelineSynchronizationError(String),
}

impl std::fmt::Display for OptimizerKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerKernelError::CudaContextCreationFailed(msg) => {
                write!(f, "CUDA context creation failed: {}", msg)
            }
            OptimizerKernelError::KernelCompilationFailed(msg) => {
                write!(f, "Kernel compilation failed: {}", msg)
            }
            OptimizerKernelError::MemoryAllocationFailed(msg) => {
                write!(f, "Memory allocation failed: {}", msg)
            }
            OptimizerKernelError::KernelExecutionFailed(msg) => {
                write!(f, "Kernel execution failed: {}", msg)
            }
            OptimizerKernelError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
            OptimizerKernelError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            OptimizerKernelError::PerformanceDegradation(msg) => {
                write!(f, "Performance degradation detected: {}", msg)
            }
            OptimizerKernelError::ResourceLimitExceeded(msg) => {
                write!(f, "Resource limit exceeded: {}", msg)
            }
            OptimizerKernelError::PrecisionConversionError(msg) => {
                write!(f, "Precision conversion error: {}", msg)
            }
            OptimizerKernelError::PipelineSynchronizationError(msg) => {
                write!(f, "Pipeline synchronization error: {}", msg)
            }
        }
    }
}

impl std::error::Error for OptimizerKernelError {}

// Default implementations

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enable_profiling: true,
            sample_rate: 0.1,
            max_samples: 10000,
            profile_memory_transfers: true,
            profile_kernel_times: true,
            profile_gpu_utilization: true,
            output_format: ProfilingOutputFormat::Json,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time: Duration::from_millis(0),
            memory_transfer_time: Duration::from_millis(0),
            gpu_utilization: 0.0,
            memory_bandwidth_utilization: 0.0,
            tensor_core_utilization: None,
            power_consumption: None,
            memory_throughput: 0.0,
            compute_throughput: 0.0,
            cache_hit_rate: None,
            occupancy: 0.0,
            registers_per_thread: 0,
            shared_memory_per_block: 0,
            grid_dimensions: (1, 1, 1),
            block_dimensions: (1, 1, 1),
        }
    }
}

impl Default for TensorCoreSupport {
    fn default() -> Self {
        Self {
            available: false,
            supported_generations: Vec::new(),
            current_generation: None,
            supported_data_types: Vec::new(),
            optimal_sizes: Vec::new(),
            performance_characteristics: TensorCorePerformanceProfile::default(),
        }
    }
}

impl Default for TensorCorePerformanceProfile {
    fn default() -> Self {
        Self {
            peak_tops: 0.0,
            memory_bandwidth: 0.0,
            optimal_tile_sizes: HashMap::new(),
            efficiency_factors: HashMap::new(),
        }
    }
}

impl Default for MixedPrecisionSupport {
    fn default() -> Self {
        Self {
            enabled: false,
            loss_scale: 65536.0,
            auto_loss_scaling: true,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            supported_precisions: vec![Precision::FP32, Precision::FP16],
            forward_precision: Precision::FP16,
            backward_precision: Precision::FP32,
            master_weights_precision: Precision::FP32,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_stages: 3,
            async_execution: true,
            stream_priority: 0,
            enable_prefetch: true,
            overlap_compute_comm: true,
            stage_buffer_sizes: vec![1024 * 1024; 3], // 1MB per stage
            sync_points: vec![
                PipelineSyncPoint::BeforeKernelExecution,
                PipelineSyncPoint::AfterKernelExecution,
            ],
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 48 * 1024, // 48KB
            max_registers_per_thread: 255,
            max_grid_size: (65535, 65535, 65535),
            max_texture_memory: None,
            max_constant_memory: Some(64 * 1024), // 64KB
        }
    }
}

impl<T: Float> Default for AdaptiveKernelConfig<T> {
    fn default() -> Self {
        Self {
            enable_adaptive: false,
            adaptation_learning_rate: T::from(0.01).unwrap_or_else(|| T::zero()),
            min_performance_threshold: T::from(0.8).unwrap_or_else(|| T::zero()),
            max_adaptation_iterations: 100,
            adaptation_strategy: AdaptationStrategy::GradientBased,
            target_metrics: vec![TargetMetric::ExecutionTime, TargetMetric::GpuUtilization],
            constraints: AdaptationConstraints::default(),
        }
    }
}

impl<T: Float> Default for AdaptationConstraints<T> {
    fn default() -> Self {
        Self {
            max_memory_usage: None,
            max_execution_time: None,
            min_accuracy: None,
            max_power_consumption: None,
            resource_limits: ResourceLimits::default(),
        }
    }
}

// Utility functions

impl TensorCoreGeneration {
    /// Get the compute capability for this generation
    pub fn compute_capability(&self) -> (u32, u32) {
        match self {
            TensorCoreGeneration::V1 => (7, 0), // Volta
            TensorCoreGeneration::V2 => (7, 5), // Turing
            TensorCoreGeneration::V3 => (8, 0), // Ampere
            TensorCoreGeneration::V4 => (9, 0), // Ada Lovelace / Hopper
        }
    }

    /// Check if this generation supports a specific data type
    pub fn supports_data_type(&self, data_type: TensorCoreDataType) -> bool {
        match (self, data_type) {
            (TensorCoreGeneration::V1, TensorCoreDataType::FP16) => true,
            (TensorCoreGeneration::V2, TensorCoreDataType::FP16) => true,
            (TensorCoreGeneration::V2, TensorCoreDataType::INT8) => true,
            (TensorCoreGeneration::V3, TensorCoreDataType::FP16) => true,
            (TensorCoreGeneration::V3, TensorCoreDataType::BF16) => true,
            (TensorCoreGeneration::V3, TensorCoreDataType::TF32) => true,
            (TensorCoreGeneration::V3, TensorCoreDataType::INT8) => true,
            (TensorCoreGeneration::V4, _) => true, // Supports all types
            _ => false,
        }
    }
}

impl Precision {
    /// Get the number of bits for this precision
    pub fn bits(&self) -> u8 {
        match self {
            Precision::FP32 => 32,
            Precision::FP16 => 16,
            Precision::BF16 => 16,
            Precision::TF32 => 19, // 19-bit mantissa + 8-bit exponent + 1 sign bit = 28 total
            Precision::FP8 => 8,
        }
    }

    /// Check if this precision supports Tensor Cores
    pub fn supports_tensor_cores(&self) -> bool {
        matches!(self, Precision::FP16 | Precision::BF16 | Precision::TF32)
    }

    /// Get the memory footprint multiplier compared to FP32
    pub fn memory_multiplier(&self) -> f32 {
        self.bits() as f32 / 32.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiling_config_default() {
        let config = ProfilingConfig::default();
        assert!(config.enable_profiling);
        assert_eq!(config.sample_rate, 0.1);
        assert_eq!(config.max_samples, 10000);
    }

    #[test]
    fn test_tensor_core_generation_capabilities() {
        let v1 = TensorCoreGeneration::V1;
        assert_eq!(v1.compute_capability(), (7, 0));
        assert!(v1.supports_data_type(TensorCoreDataType::FP16));
        assert!(!v1.supports_data_type(TensorCoreDataType::BF16));

        let v4 = TensorCoreGeneration::V4;
        assert_eq!(v4.compute_capability(), (9, 0));
        assert!(v4.supports_data_type(TensorCoreDataType::FP8));
    }

    #[test]
    fn test_precision_properties() {
        assert_eq!(Precision::FP32.bits(), 32);
        assert_eq!(Precision::FP16.bits(), 16);
        assert_eq!(Precision::FP8.bits(), 8);

        assert!(Precision::FP16.supports_tensor_cores());
        assert!(!Precision::FP32.supports_tensor_cores());

        assert_eq!(Precision::FP16.memory_multiplier(), 0.5);
        assert_eq!(Precision::FP8.memory_multiplier(), 0.25);
    }

    #[test]
    fn test_adaptive_kernel_config_default() {
        let config = AdaptiveKernelConfig::<f32>::default();
        assert!(!config.enable_adaptive);
        assert_eq!(config.adaptation_learning_rate, 0.01);
        assert_eq!(config.max_adaptation_iterations, 100);
    }

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_threads_per_block, 1024);
        assert_eq!(limits.max_shared_memory_per_block, 48 * 1024);
        assert_eq!(limits.max_registers_per_thread, 255);
    }

    #[test]
    fn test_mixed_precision_default() {
        let config = MixedPrecisionSupport::default();
        assert!(!config.enabled);
        assert_eq!(config.loss_scale, 65536.0);
        assert!(config.auto_loss_scaling);
        assert_eq!(config.forward_precision, Precision::FP16);
        assert_eq!(config.master_weights_precision, Precision::FP32);
    }
}