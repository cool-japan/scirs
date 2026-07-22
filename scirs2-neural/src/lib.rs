#![allow(rustdoc::broken_intra_doc_links)]
//! # SciRS2 Neural Networks
//!
//! **scirs2-neural** provides PyTorch-style neural network building blocks for Rust,
//! with automatic differentiation integration and production-ready training utilities.
//!
//! ## 🎯 Key Features
//!
//! - **Layer-based Architecture**: Modular neural network layers (Dense, Conv2D, LSTM, etc.)
//! - **Activation Functions**: Common activations (ReLU, Sigmoid, Tanh, GELU, etc.)
//! - **Loss Functions**: Classification and regression losses
//! - **Training Utilities**: Training loops, callbacks, and metrics
//! - **Autograd Integration**: Automatic differentiation via scirs2-autograd
//! - **Type Safety**: Compile-time shape and type checking where possible
//!
//! ## 📦 Module Overview
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`activations_minimal`] | Activation functions (ReLU, Sigmoid, Tanh, GELU, etc.) |
//! | [`layers`] | Neural network layers (Dense, Conv2D, LSTM, Dropout, etc.) |
//! | [`losses`] | Loss functions (MSE, CrossEntropy, Focal, Contrastive, etc.) |
//! | [`training`] | Training loops and utilities |
//! | [`autograd`] | Automatic differentiation integration |
//! | [`error`] | Error types and handling |
//! | [`utils`] | Helper utilities |
//!
//! ## 🚀 Quick Start
//!
//! ### Installation
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! scirs2-neural = "0.6.2"
//! ```
//!
//! ### Building a Simple Neural Network
//!
//! ```rust
//! use scirs2_neural::prelude::*;
//! use scirs2_core::ndarray::Array2;
//! use scirs2_core::random::rng;
//!
//! let mut rng = rng();
//!
//! // Build a 3-layer MLP for MNIST
//! let mut model = Sequential::<f32>::new();
//! model.add(Dense::new(784, 256, Some("relu"), &mut rng).expect("failed to create dense layer"));
//! model.add(Dense::new(256, 128, Some("relu"), &mut rng).expect("failed to create dense layer"));
//! model.add(Dense::new(128, 10, None, &mut rng).expect("failed to create dense layer"));
//!
//! println!("Model created with {} layers", model.len());
//! assert_eq!(model.len(), 3);
//! ```
//!
//! ### Using Individual Layers
//!
//! ```rust
//! use scirs2_neural::prelude::*;
//! use scirs2_core::ndarray::Array2;
//! use scirs2_core::random::rng;
//!
//! let mut rng = rng();
//!
//! // Dense layer
//! let dense = Dense::<f32>::new(10, 5, None, &mut rng).expect("failed to create dense layer");
//!
//! // Activation functions
//! let relu = ReLU::new();
//! let sigmoid = Sigmoid::new();
//! let tanh_act = Tanh::new();
//! let gelu = GELU::new();
//!
//! // Normalization layers
//! let batch_norm = BatchNorm::<f32>::new(5, 0.1, 1e-5, &mut rng).expect("failed to create batch norm");
//! let layer_norm = LayerNorm::<f32>::new(5, 1e-5, &mut rng).expect("failed to create layer norm");
//! ```
//!
//! ### Convolutional Networks
//!
//! ```rust
//! use scirs2_neural::prelude::*;
//! use scirs2_core::random::rng;
//!
//! let mut rng = rng();
//!
//! // Build a simple CNN
//! let mut model = Sequential::<f32>::new();
//!
//! // Conv layers (in_channels, out_channels, kernel_size, stride, name)
//! model.add(Conv2D::new(1, 32, (3, 3), (1, 1), Some("relu")).expect("conv2d failed"));
//! model.add(Conv2D::new(32, 64, (3, 3), (1, 1), Some("relu")).expect("conv2d failed"));
//!
//! // Flatten and classify
//! model.add(Dense::new(64 * 28 * 28, 10, None, &mut rng).expect("dense failed"));
//!
//! assert_eq!(model.len(), 3);
//! ```
//!
//! ### Recurrent Networks (LSTM)
//!
//! ```rust
//! use scirs2_neural::prelude::*;
//! use scirs2_core::random::rng;
//!
//! let mut rng = rng();
//!
//! // Build an LSTM-based model
//! let mut model = Sequential::<f32>::new();
//!
//! // LSTM (input_size, hidden_size, rng)
//! model.add(LSTM::new(100, 256, &mut rng).expect("lstm failed"));
//! model.add(Dense::new(256, 10, None, &mut rng).expect("dense failed"));
//!
//! assert_eq!(model.len(), 2);
//! ```
//!
//! ### Loss Functions
//!
//! ```rust
//! use scirs2_neural::prelude::*;
//!
//! // Mean Squared Error (regression)
//! let mse = MeanSquaredError::new();
//!
//! // Cross Entropy (classification)
//! let ce = CrossEntropyLoss::new(1e-7);
//!
//! // Focal Loss (imbalanced classes)
//! let focal = FocalLoss::new(2.0, None, 1e-7);
//!
//! // Contrastive Loss (metric learning)
//! let contrastive = ContrastiveLoss::new(1.0);
//!
//! // Triplet Loss (metric learning)
//! let triplet = TripletLoss::new(1.0);
//! ```
//!
//! ### Training a Model
//!
//! ```rust
//! use scirs2_neural::prelude::*;
//! use scirs2_core::random::rng;
//!
//! let mut rng = rng();
//!
//! // Build model
//! let mut model = Sequential::<f32>::new();
//! model.add(Dense::new(784, 128, Some("relu"), &mut rng).expect("dense failed"));
//! model.add(Dense::new(128, 10, None, &mut rng).expect("dense failed"));
//!
//! // Training configuration
//! let config = TrainingConfig {
//!     learning_rate: 0.001,
//!     batch_size: 32,
//!     epochs: 10,
//!     validation: Some(ValidationSettings {
//!         enabled: true,
//!         validation_split: 0.2,
//!         batch_size: 32,
//!         num_workers: 0,
//!     }),
//!     ..Default::default()
//! };
//!
//! // Create training session
//! let session = TrainingSession::<f32>::new(config);
//! assert_eq!(model.len(), 2);
//! ```
//!
//! ## 🧠 Available Layers
//!
//! ### Core Layers
//!
//! - **`Dense`**: Fully connected (linear) layer
//! - **`Conv2D`**: 2D convolutional layer
//! - **`LSTM`**: Long Short-Term Memory recurrent layer
//!
//! ### Activation Layers
//!
//! - **`ReLU`**: Rectified Linear Unit
//! - **`Sigmoid`**: Sigmoid activation
//! - **`Tanh`**: Hyperbolic tangent
//! - **`GELU`**: Gaussian Error Linear Unit
//! - **`Softmax`**: Softmax for classification
//!
//! ### Normalization Layers
//!
//! - **`BatchNorm`**: Batch normalization
//! - **`LayerNorm`**: Layer normalization
//!
//! ### Regularization Layers
//!
//! - **`Dropout`**: Random dropout for regularization
//!
//! ## 📊 Loss Functions
//!
//! ### Regression
//!
//! - **`MeanSquaredError`**: L2 loss for regression
//!
//! ### Classification
//!
//! - **`CrossEntropyLoss`**: Standard classification loss
//! - **`FocalLoss`**: For imbalanced classification
//!
//! ### Metric Learning
//!
//! - **`ContrastiveLoss`**: Pairwise similarity learning
//! - **`TripletLoss`**: Triplet-based metric learning
//!
//! ## 🎨 Design Philosophy
//!
//! scirs2-neural follows PyTorch's design philosophy:
//!
//! - **Layer-based**: Composable building blocks
//! - **Explicit**: Clear forward/backward passes
//! - **Flexible**: Easy to extend with custom layers
//! - **Type-safe**: Leverage Rust's type system
//!
//! ## 🔗 Integration with SciRS2 Ecosystem
//!
//! - **scirs2-autograd**: Automatic differentiation support
//! - **scirs2-linalg**: Matrix operations and decompositions
//! - **scirs2-metrics**: Model evaluation metrics
//! - **scirs2-datasets**: Sample datasets for training
//! - **scirs2-vision**: Computer vision utilities
//! - **scirs2-text**: Text processing for NLP models
//!
//! ## 🚀 Performance
//!
//! scirs2-neural provides multiple optimization paths:
//!
//! - **Pure Rust**: Fast, safe implementations
//! - **SIMD**: Vectorized operations where applicable
//! - **Parallel**: Multi-threaded training
//! - **GPU**: CUDA/Metal support (via scirs2-core)
//!
//! ## 📚 Comparison with PyTorch
//!
//! | Feature | PyTorch | scirs2-neural |
//! |---------|---------|---------------|
//! | Layer-based API | ✅ | ✅ |
//! | Autograd | ✅ | ✅ (via scirs2-autograd) |
//! | GPU Support | ✅ | ✅ (limited) |
//! | Dynamic Graphs | ✅ | ✅ |
//! | JIT Compilation | ✅ | ⚠️ (planned) |
//! | Production Deployment | ⚠️ | ✅ (native Rust) |
//! | Type Safety | ❌ | ✅ |
//!
//! ## 📜 Examples
//!
//! See the `examples/` directory for complete examples:
//!
//! - `mnist_mlp.rs` - Multi-layer perceptron for MNIST
//! - `cifar_cnn.rs` - Convolutional network for CIFAR-10
//! - `sentiment_lstm.rs` - LSTM for sentiment analysis
//! - `custom_layer.rs` - Creating custom layers
//!
//! ## 🔒 Version
//!
//! Current version: **0.6.2**

pub mod activations;
pub mod activations_minimal;
pub mod autograd;
pub mod callbacks;
pub mod data;
pub mod distillation;
pub mod error;
pub mod gpu; // GPU acceleration (honest device detection; real GPU via scirs2_core::gpu)
pub mod layers;
pub mod linalg; // Re-enabled - fixing errors
pub mod losses;
pub mod model_evaluation;
pub mod models;
pub mod ops;
pub mod optimizers;
pub mod quantization;
pub mod serialization;
pub mod serving; // Model packaging/deployment + real host-target code generation
pub mod tensor_ops;
pub mod training;
pub mod transformer;
pub mod utils;
pub mod visualization;

// Attention mechanisms (flash attention, sparse attention)
pub mod attention;
// Model export (ONNX, WeightStore)
pub mod export;
// Inference optimization (speculative decoding)
pub mod inference;
// LoRA and adapter layers
pub mod lora;
// Neural architecture search (DARTS, GDAS, SNAS)
pub mod nas;
// Speculative decoding
pub mod speculative;
// Model tracing and static graph
pub mod tracing;
// NLP utilities (tokenizers, beam search, language model evaluation)
pub mod nlp;
// Capsule Networks (CapsNet with dynamic routing)
pub mod capsule;
// Generative models (VAE, GAN, normalizing flows)
pub mod generative;
// Graph Neural Networks (GCN, GAT, GraphSAGE)
pub mod gnn;
// Reinforcement learning utilities (Q-learning, policy gradient)
pub mod rl;
// Spiking Neural Networks
pub mod snn;

// The following modules have incomplete/stub implementations and are pending repair:
// pub mod bindings;     // FFI bindings - pending full reconstruction; all 5 sub-modules (config, generator, header_generation, source_generation, examples_docs) have pervasive syntax damage (missing braces, wrong params, unterminated raw strings) dating to the initial commit
pub mod config; // Model configuration - fixed
pub mod continual; // Continual learning - fixed
pub mod evaluation; // Model evaluation
pub mod federated; // Federated learning - fixed
pub mod hardware; // Hardware acceleration (fixed)
pub mod interpretation; // Model interpretation (fixed)
pub mod mobile; // Mobile optimization (fixed)
pub mod on_device; // On-device model compression (model_compression only; other sub-modules need rewrite)
pub mod performance; // Performance profiling (fixed)
pub mod reinforcement; // RL algorithms
pub mod wasm; // WASM bindings (fixed)

#[cfg(feature = "symbolic")]
pub use activations::SymbolicActivation;
pub use activations_minimal::{Activation, ReLU, Sigmoid, Softmax, Tanh, GELU};
pub use error::{Error, NeuralError, Result};
// Enhanced model evaluation tools (classification/regression metrics, CV, bootstrap CIs).
// `CrossValidationStrategy` is intentionally not re-exported here to avoid a name
// clash with `evaluation::CrossValidationStrategy`; use the full path
// `model_evaluation::CrossValidationStrategy` when needed.
#[cfg(feature = "symbolic")]
pub use losses::SymbolicLoss;
#[cfg(feature = "symbolic")]
pub use losses::SymbolicPriorLoss;
pub use model_evaluation::{
    AveragingMethod, ClassificationMetric, EvaluationBuilder, EvaluationMetric, EvaluationResults,
    MetricScore, ModelEvaluator, RegressionMetric,
};
#[cfg(feature = "symbolic")]
pub mod symbolic;
pub use layers::{
    BatchNorm, Conv2D, Dense, Dropout, Layer, LayerNorm, LayerNorm2D, PatchEmbedding, Sequential,
    LSTM,
};
pub use losses::{
    ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, MeanSquaredError, TripletLoss,
};
pub use training::{TrainingConfig, TrainingSession};

// Re-export enhanced training (v0.2.0)
pub use training::{
    EarlyStoppingConfig, EnhancedTrainer, EnhancedTrainingConfig, GradientAccumulationSettings,
    LRWarmupConfig, OptimizedDataLoader, OptimizedLoaderConfig, ProfilingConfig, ProfilingResults,
    ProgressConfig, TrainingState, ValidationConfig, WarmupSchedule,
};

// Re-export serialization (v0.3.0)
pub use serialization::{
    ExtractParameters, ModelDeserialize, ModelFormat, ModelMetadata, ModelSerialize,
    NamedParameters, SafeTensorsReader, SafeTensorsWriter, TensorInfo,
};

// Re-export checkpoint (v0.3.0)
pub use training::{
    best_checkpoint, checkpoint_dir_name, latest_checkpoint, list_checkpoints, load_checkpoint,
    save_checkpoint, CheckpointMetadata, OptimizerStateMetadata, ParamGroupState,
};

// Re-export distillation (v0.3.0)
pub use distillation::{
    DistanceMetric, DistillationConfig, DistillationMethod, DistillationResult,
    DistillationStatistics, DistillationTrainer, EnsembleAggregation, FeatureAdaptation,
};

// Re-export quantization (v0.3.0)
pub use quantization::{
    DynamicQuantizer, MixedBitWidthQuantizer, PostTrainingQuantizer, QuantizationAwareTraining,
    QuantizationConfig, QuantizationMode, QuantizationParams, QuantizationScheme, QuantizedTensor,
};

// Re-export LR finder (v0.3.0+)
pub use training::{
    find_optimal_lr, LRFinder, LRFinderConfig, LRFinderResult, LRFinderStatus, LRScheduleType,
};

// Re-export curriculum learning (v0.3.0+)
pub use training::{CompetenceSchedule, CurriculumConfig, CurriculumLearner, CurriculumStrategy};

// Re-export federated learning (v0.3.0+)
pub use training::{
    AggregationMethod, ClientSelectionStrategy, ClientUpdate, FederatedConfig, FederatedServer,
};

// Re-export training profiler (v0.3.0+)
pub use training::{Bottleneck, LayerProfile, ProfilePhase, ProfileSummary, TrainingProfiler};

// Re-export hyperparameter tuner (v0.3.0+)
pub use training::{HParamSpace, HParamTuner, HParamValue, SearchStrategy, TrialResult};

/// Prelude module with core functionality
///
/// Import everything you need to get started:
///
/// ```rust
/// use scirs2_neural::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        activations_minimal::{Activation, ReLU, Sigmoid, Softmax, Tanh, GELU},
        callbacks::{Callback, CallbackManager, CallbackTiming, EarlyStopping},
        data::{DataLoader, Dataset, InMemoryDataset},
        error::{Error, NeuralError, Result},
        layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, LayerNorm, Sequential, LSTM},
        losses::{
            ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, MeanSquaredError, TripletLoss,
        },
        training::{
            EnhancedTrainer, EnhancedTrainingConfig, TrainingConfig, TrainingSession,
            ValidationConfig, ValidationSettings,
        },
    };
}
