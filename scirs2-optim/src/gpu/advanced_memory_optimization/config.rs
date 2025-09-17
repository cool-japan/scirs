//! Configuration and basic types for advanced GPU memory optimization
//!
//! This module contains the main configuration structures and fundamental types
//! used throughout the advanced memory optimization system.

use std::time::{Duration, Instant};
use num_traits::Float;

/// Advanced memory optimization configuration
#[derive(Debug, Clone)]
pub struct AdvancedMemoryConfig {
    /// Enable gradient checkpointing
    pub enable_gradient_checkpointing: bool,

    /// Enable memory-efficient attention
    pub enable_memory_efficient_attention: bool,

    /// Enable dynamic batch sizing
    pub enable_dynamic_batching: bool,

    /// Enable parameter offloading
    pub enable_parameter_offloading: bool,

    /// Enable activation recomputation
    pub enable_activation_recomputation: bool,

    /// Maximum memory usage (percentage of total GPU memory)
    pub max_memory_usage: f32,

    /// Memory pressure threshold for triggering optimizations
    pub memory_pressure_threshold: f32,

    /// Checkpoint interval (number of layers)
    pub checkpoint_interval: usize,

    /// Offload threshold (parameter size in bytes)
    pub offload_threshold: usize,

    /// Enable memory profiling
    pub enable_profiling: bool,

    /// Microbatch size for gradient accumulation
    pub microbatch_size: usize,

    /// Maximum gradient accumulation steps
    pub max_accumulation_steps: usize,

    /// Enable zero redundancy optimizer
    pub enable_zero_redundancy: bool,

    /// Enable mixed precision training
    pub enable_mixed_precision: bool,

    /// Enable memory mapped I/O for large models
    pub enable_memory_mapping: bool,
}

impl Default for AdvancedMemoryConfig {
    fn default() -> Self {
        Self {
            enable_gradient_checkpointing: true,
            enable_memory_efficient_attention: true,
            enable_dynamic_batching: true,
            enable_parameter_offloading: true,
            enable_activation_recomputation: true,
            max_memory_usage: 0.85,
            memory_pressure_threshold: 0.8,
            checkpoint_interval: 4,
            offload_threshold: 1024 * 1024, // 1MB
            enable_profiling: true,
            microbatch_size: 1,
            max_accumulation_steps: 32,
            enable_zero_redundancy: true,
            enable_mixed_precision: true,
            enable_memory_mapping: false,
        }
    }
}

impl AdvancedMemoryConfig {
    /// Create configuration optimized for large models (>1B parameters)
    pub fn for_large_models() -> Self {
        Self {
            enable_gradient_checkpointing: true,
            enable_memory_efficient_attention: true,
            enable_dynamic_batching: true,
            enable_parameter_offloading: true,
            enable_activation_recomputation: true,
            max_memory_usage: 0.95,
            memory_pressure_threshold: 0.9,
            checkpoint_interval: 2,
            offload_threshold: 512 * 1024, // 512KB
            enable_profiling: false, // Disable for performance
            microbatch_size: 1,
            max_accumulation_steps: 64,
            enable_zero_redundancy: true,
            enable_mixed_precision: true,
            enable_memory_mapping: true,
        }
    }

    /// Create configuration optimized for training efficiency
    pub fn for_training_efficiency() -> Self {
        Self {
            enable_gradient_checkpointing: false,
            enable_memory_efficient_attention: true,
            enable_dynamic_batching: true,
            enable_parameter_offloading: false,
            enable_activation_recomputation: false,
            max_memory_usage: 0.8,
            memory_pressure_threshold: 0.75,
            checkpoint_interval: 8,
            offload_threshold: 2 * 1024 * 1024, // 2MB
            enable_profiling: true,
            microbatch_size: 4,
            max_accumulation_steps: 16,
            enable_zero_redundancy: false,
            enable_mixed_precision: true,
            enable_memory_mapping: false,
        }
    }

    /// Create configuration optimized for inference
    pub fn for_inference() -> Self {
        Self {
            enable_gradient_checkpointing: false,
            enable_memory_efficient_attention: true,
            enable_dynamic_batching: true,
            enable_parameter_offloading: true,
            enable_activation_recomputation: false,
            max_memory_usage: 0.9,
            memory_pressure_threshold: 0.85,
            checkpoint_interval: 0, // No checkpointing needed
            offload_threshold: 1024 * 1024, // 1MB
            enable_profiling: false,
            microbatch_size: 1,
            max_accumulation_steps: 1, // No accumulation for inference
            enable_zero_redundancy: false,
            enable_mixed_precision: true,
            enable_memory_mapping: true,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.max_memory_usage <= 0.0 || self.max_memory_usage > 1.0 {
            return Err("max_memory_usage must be between 0.0 and 1.0".to_string());
        }

        if self.memory_pressure_threshold <= 0.0 || self.memory_pressure_threshold > 1.0 {
            return Err("memory_pressure_threshold must be between 0.0 and 1.0".to_string());
        }

        if self.memory_pressure_threshold > self.max_memory_usage {
            return Err("memory_pressure_threshold cannot exceed max_memory_usage".to_string());
        }

        if self.microbatch_size == 0 {
            return Err("microbatch_size must be positive".to_string());
        }

        if self.max_accumulation_steps == 0 {
            return Err("max_accumulation_steps must be positive".to_string());
        }

        Ok(())
    }
}

/// Types of memory allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationType {
    /// Model parameters
    Parameters,

    /// Gradient tensors
    Gradients,

    /// Activation tensors
    Activations,

    /// Checkpoint data
    Checkpoint,

    /// Temporary computation buffers
    Temporary,

    /// I/O buffers
    Buffer,
}

impl AllocationType {
    /// Get the priority for memory allocation (lower number = higher priority)
    pub fn priority(&self) -> u8 {
        match self {
            AllocationType::Parameters => 1,
            AllocationType::Gradients => 2,
            AllocationType::Activations => 3,
            AllocationType::Buffer => 4,
            AllocationType::Checkpoint => 5,
            AllocationType::Temporary => 6,
        }
    }

    /// Check if this allocation type is critical (cannot be offloaded easily)
    pub fn is_critical(&self) -> bool {
        matches!(self, AllocationType::Parameters | AllocationType::Gradients)
    }
}

/// Actions taken under memory pressure
#[derive(Debug, Clone)]
pub enum PressureAction {
    /// Checkpoint activations (number of layers)
    CheckpointActivations(usize),

    /// Offload parameters (number of parameters)
    OffloadParameters(usize),

    /// Reduce batch size
    ReduceBatchSize { from: usize, to: usize },

    /// Clear temporary buffers (bytes freed)
    ClearBuffers(usize),

    /// Trigger garbage collection
    GarbageCollection,

    /// Recompute activations (number of layers)
    RecomputeActivations(usize),
}

impl PressureAction {
    /// Get the estimated memory savings in bytes
    pub fn estimated_savings(&self) -> usize {
        match self {
            PressureAction::CheckpointActivations(layers) => layers * 1024 * 1024, // Rough estimate
            PressureAction::OffloadParameters(params) => params * 4, // Assuming f32
            PressureAction::ReduceBatchSize { from, to } => (from - to) * 1024 * 1024,
            PressureAction::ClearBuffers(bytes) => *bytes,
            PressureAction::GarbageCollection => 0, // Unknown
            PressureAction::RecomputeActivations(layers) => layers * 512 * 1024,
        }
    }

    /// Get the cost associated with this action (higher = more expensive)
    pub fn cost(&self) -> f32 {
        match self {
            PressureAction::CheckpointActivations(_) => 0.3,
            PressureAction::OffloadParameters(_) => 0.7,
            PressureAction::ReduceBatchSize { .. } => 0.9, // High cost due to batch size change
            PressureAction::ClearBuffers(_) => 0.1,
            PressureAction::GarbageCollection => 0.2,
            PressureAction::RecomputeActivations(_) => 0.5,
        }
    }
}

/// Checkpoint strategies for activation recomputation
#[derive(Debug, Clone, Copy)]
pub enum CheckpointStrategy {
    /// Uniform checkpointing every N layers
    Uniform(usize),

    /// Adaptive checkpointing based on memory pressure
    Adaptive,

    /// Optimal checkpointing using dynamic programming
    Optimal,

    /// User-defined checkpoints
    Manual,
}

impl Default for CheckpointStrategy {
    fn default() -> Self {
        CheckpointStrategy::Adaptive
    }
}

/// Eviction policies for checkpoint management
#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,

    /// Least Frequently Used
    LFU,

    /// Cost-based eviction
    CostBased,

    /// First In First Out
    FIFO,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        EvictionPolicy::CostBased
    }
}

/// Storage locations for offloaded parameters
#[derive(Debug, Clone)]
pub enum StorageLocation {
    /// CPU memory
    CpuMemory { ptr: *mut u8, size: usize },

    /// Disk storage
    DiskStorage { file_path: String, offset: usize },

    /// Remote storage
    RemoteStorage { url: String, checksum: String },

    /// Compressed storage
    Compressed { data: Vec<u8>, compression_type: CompressionType },
}

/// Compression algorithms
#[derive(Debug, Clone, Copy)]
pub enum CompressionType {
    /// Lossless compression
    LZ4,
    ZSTD,
    GZIP,

    /// Lossy compression for parameters
    Quantization8Bit,
    Quantization4Bit,
    PruningBased,

    /// Learned compression
    NeuralCompression,
}

impl CompressionType {
    /// Get expected compression ratio (output size / input size)
    pub fn expected_ratio(&self) -> f32 {
        match self {
            CompressionType::LZ4 => 0.6,
            CompressionType::ZSTD => 0.5,
            CompressionType::GZIP => 0.4,
            CompressionType::Quantization8Bit => 0.25,
            CompressionType::Quantization4Bit => 0.125,
            CompressionType::PruningBased => 0.3,
            CompressionType::NeuralCompression => 0.2,
        }
    }

    /// Check if compression is lossy
    pub fn is_lossy(&self) -> bool {
        matches!(
            self,
            CompressionType::Quantization8Bit
                | CompressionType::Quantization4Bit
                | CompressionType::PruningBased
                | CompressionType::NeuralCompression
        )
    }
}

/// Parameter offloading strategies
#[derive(Debug, Clone, Copy)]
pub enum OffloadStrategy {
    /// Size-based offloading (threshold in bytes)
    SizeBased(usize),

    /// Access frequency based
    FrequencyBased,

    /// Cost-benefit analysis
    CostBenefit,

    /// Least recently used
    LRU,

    /// Memory pressure driven
    PressureDriven,
}

impl Default for OffloadStrategy {
    fn default() -> Self {
        OffloadStrategy::CostBenefit
    }
}

/// Transfer cost analysis
#[derive(Debug, Clone)]
pub struct TransferCost {
    /// GPU to CPU transfer cost (nanoseconds per byte)
    pub gpu_to_cpu_cost: f32,

    /// CPU to GPU transfer cost (nanoseconds per byte)
    pub cpu_to_gpu_cost: f32,

    /// Disk I/O cost (nanoseconds per byte)
    pub disk_io_cost: f32,

    /// Network transfer cost (nanoseconds per byte)
    pub network_cost: f32,

    /// Compression cost (nanoseconds per byte)
    pub compression_cost: f32,

    /// Decompression cost (nanoseconds per byte)
    pub decompression_cost: f32,
}

impl Default for TransferCost {
    fn default() -> Self {
        Self {
            gpu_to_cpu_cost: 0.5,
            cpu_to_gpu_cost: 0.7,
            disk_io_cost: 100.0,
            network_cost: 1000.0,
            compression_cost: 50.0,
            decompression_cost: 30.0,
        }
    }
}

/// Compression information
#[derive(Debug, Clone)]
pub struct CompressionInfo {
    /// Compression algorithm used
    pub algorithm: CompressionType,

    /// Compression ratio achieved
    pub ratio: f32,

    /// Compression time
    pub compression_time: Duration,

    /// Decompression time
    pub decompression_time: Duration,

    /// Quality loss (for lossy compression)
    pub quality_loss: Option<f32>,
}

/// Recomputation cost analysis
#[derive(Debug, Clone)]
pub struct RecomputationCost {
    /// Computational cost (FLOPs)
    pub compute_cost: u64,

    /// Memory cost (bytes)
    pub memory_cost: usize,

    /// Time cost (nanoseconds)
    pub time_cost: u64,

    /// Cost-benefit ratio
    pub cost_benefit_ratio: f32,
}

impl Default for RecomputationCost {
    fn default() -> Self {
        Self {
            compute_cost: 0,
            memory_cost: 0,
            time_cost: 0,
            cost_benefit_ratio: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let mut config = AdvancedMemoryConfig::default();
        assert!(config.validate().is_ok());

        config.max_memory_usage = 1.5;
        assert!(config.validate().is_err());

        config.max_memory_usage = 0.8;
        config.memory_pressure_threshold = 0.9;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_allocation_type_priority() {
        assert!(AllocationType::Parameters.priority() < AllocationType::Temporary.priority());
        assert!(AllocationType::Gradients.priority() < AllocationType::Checkpoint.priority());
    }

    #[test]
    fn test_pressure_action_estimates() {
        let action = PressureAction::CheckpointActivations(4);
        assert!(action.estimated_savings() > 0);
        assert!(action.cost() > 0.0);
    }

    #[test]
    fn test_compression_type_properties() {
        assert!(CompressionType::LZ4.expected_ratio() < 1.0);
        assert!(!CompressionType::LZ4.is_lossy());
        assert!(CompressionType::Quantization8Bit.is_lossy());
    }

    #[test]
    fn test_config_presets() {
        let large_config = AdvancedMemoryConfig::for_large_models();
        assert!(large_config.validate().is_ok());
        assert!(large_config.enable_parameter_offloading);
        assert!(large_config.enable_memory_mapping);

        let inference_config = AdvancedMemoryConfig::for_inference();
        assert!(inference_config.validate().is_ok());
        assert!(!inference_config.enable_gradient_checkpointing);
        assert_eq!(inference_config.max_accumulation_steps, 1);
    }
}