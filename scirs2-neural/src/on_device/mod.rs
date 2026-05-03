//! On-device training optimisations module
//!
//! This module provides optimisations for training neural networks on edge devices
//! with limited compute and memory resources.
//!
//! ## Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`model_compression`] | Model compression: pruning, quantization, distillation |
//! | [`gradient_checkpointing`] | Gradient checkpointing for memory-efficient backprop |
//! | [`memory_efficient_training`] | Gradient accumulation, activation checkpointing, memory pool |
//! | [`quantization_aware_training`] | Fake-quantization training for post-training quant |
//! | [`sparse_training`] | Magnitude/structured/random pruning + dynamic sparse networks |

pub mod gradient_checkpointing;
pub mod memory_efficient_training;
pub mod model_compression;
pub mod quantization_aware_training;
pub mod sparse_training;

// ── model_compression re-exports ─────────────────────────────────────────────
pub use model_compression::{
    CompressionResult, CompressionStrategy, DistillationConfig, HuffmanConfig,
    LayerCompressionStat, LowRankConfig, ModelCompressor, PruningConfig, PruningScope,
    QuantizationConfig, QuantizationPrecision,
};

// ── gradient_checkpointing re-exports ────────────────────────────────────────
pub use gradient_checkpointing::{
    Checkpoint, CheckpointStrategy, CheckpointedModel, GradientCheckpointing, LayerInfo,
    MemoryStats,
};

// ── memory_efficient_training re-exports ─────────────────────────────────────
pub use memory_efficient_training::{
    ActivationCheckpointing, BufferReuseStrategy, EfficientDataLoader, GradientAccumulator,
    MemoryEfficientTrainer,
};

// ── quantization_aware_training re-exports ───────────────────────────────────
pub use quantization_aware_training::{
    CalibrationMethod, QATConfig, QuantizationAwareTraining, QuantizationScheme, QuantizedTensor,
};

// ── sparse_training re-exports ────────────────────────────────────────────────
pub use sparse_training::{
    DynamicSparseNetwork, PruningMethod, SparseTrainer, SparsitySchedule, SparsityStats,
};
