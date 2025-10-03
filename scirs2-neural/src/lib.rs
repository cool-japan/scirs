#![allow(deprecated)]
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
//! scirs2-neural = "0.1.0-rc.1"
//! ```
//!
//! ### Building a Simple Neural Network
//!
//! ```rust,ignore
//! # IGNORED: Waiting for 0.1.0 - API needs migration to scirs2_core abstractions
//! # Current implementation uses ndarray_rand::rand directly (POLICY violation)
//! # TODO: Migrate all layer APIs to use scirs2_core::random::Random
//! use scirs2_neural::prelude::*;
//! use scirs2_core::ndarray::Array2;
//!
//! fn main() -> Result<()> {
//!     let mut rng = scirs2_core::random::Random::seed(42);
//!
//!     // Build a 3-layer MLP for MNIST
//!     let mut model = Sequential::<f32>::new();
//!     model.add(Dense::new(784, 256, Some("relu"), &mut rng)?);
//!     model.add(Dropout::new(0.2, &mut rng)?);
//!     model.add(Dense::new(256, 128, Some("relu"), &mut rng)?);
//!     model.add(Dense::new(128, 10, None, &mut rng)?);
//!
//!     // Forward pass
//!     let input = Array2::<f32>::zeros((32, 784));
//!     // let output = model.forward(&input)?;
//!
//!     println!("Model created with {} layers", model.len());
//!     Ok(())
//! }
//! ```
//!
//! ### Using Individual Layers
//!
//! ```rust,ignore
//! # IGNORED: Waiting for 0.1.0 - API needs migration to scirs2_core abstractions
//! use scirs2_neural::prelude::*;
//! use scirs2_core::ndarray::Array2;
//!
//! fn main() -> Result<()> {
//!     let mut rng = scirs2_core::random::Random::seed(42);
//!
//!     // Dense layer
//!     let mut dense = Dense::new(10, 5, None, &mut rng)?;
//!     let input = Array2::<f32>::zeros((2, 10));
//!     // let output = dense.forward(&input)?;
//!
//!     // Activation functions
//!     let relu = ReLU::new();
//!     let sigmoid = Sigmoid::new();
//!     let tanh = Tanh::new();
//!     let gelu = GELU::new();
//!
//!     // Normalization layers
//!     let batch_norm = BatchNorm::new(5, 0.1, 1e-5, &mut rng)?;
//!     let layer_norm = LayerNorm::new(5, 1e-5, &mut rng)?;
//!
//!     // Regularization
//!     let dropout = Dropout::new(0.5, &mut rng)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Convolutional Networks
//!
//! ```rust,ignore
//! # IGNORED: Waiting for 0.1.0 - API needs migration to scirs2_core abstractions
//! use scirs2_neural::prelude::*;
//!
//! fn main() -> Result<()> {
//!     let mut rng = scirs2_core::random::Random::seed(42);
//!
//!     // Build a simple CNN
//!     let mut model = Sequential::<f32>::new();
//!
//!     // Conv layers (in_channels, out_channels, kernel_size, stride, name)
//!     model.add(Conv2D::new(1, 32, (3, 3), (1, 1), Some("relu"))?);
//!     model.add(Conv2D::new(32, 64, (3, 3), (1, 1), Some("relu"))?);
//!
//!     // Flatten and classify
//!     model.add(Dense::new(64 * 28 * 28, 10, None, &mut rng)?);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Recurrent Networks (LSTM)
//!
//! ```rust,ignore
//! # IGNORED: Waiting for 0.1.0 - API needs migration to scirs2_core abstractions
//! use scirs2_neural::prelude::*;
//!
//! fn main() -> Result<()> {
//!     let mut rng = scirs2_core::random::Random::seed(42);
//!
//!     // Build an LSTM-based model
//!     let mut model = Sequential::<f32>::new();
//!
//!     // LSTM (input_size, hidden_size, rng)
//!     model.add(LSTM::new(100, 256, &mut rng)?);
//!     model.add(Dense::new(256, 10, None, &mut rng)?);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Loss Functions
//!
//! ```rust,ignore
//! # IGNORED: Consistent with other examples pending 0.1.0 API migration
//! use scirs2_neural::prelude::*;
//! use scirs2_core::ndarray::array;
//!
//! fn main() -> Result<()> {
//!     // Mean Squared Error (regression)
//!     let mse = MeanSquaredError::new();
//!
//!     // Cross Entropy (classification)
//!     let ce = CrossEntropyLoss::new(1e-7);
//!
//!     // Focal Loss (imbalanced classes)
//!     let focal = FocalLoss::new(2.0, None, 1e-7);
//!
//!     // Contrastive Loss (metric learning)
//!     let contrastive = ContrastiveLoss::new(1.0);
//!
//!     // Triplet Loss (metric learning)
//!     let triplet = TripletLoss::new(1.0);
//!
//!     // Compute loss
//!     let predictions = array![[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]];
//!     let targets = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
//!     // let loss = mse.compute(&predictions.view(), &targets.view())?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Training a Model
//!
//! ```rust,ignore
//! # IGNORED: Waiting for 0.1.0 - API needs migration to scirs2_core abstractions
//! use scirs2_neural::prelude::*;
//! use scirs2_neural::training::ValidationSettings;
//! use scirs2_core::ndarray::Array2;
//!
//! fn main() -> Result<()> {
//!     let mut rng = scirs2_core::random::Random::seed(42);
//!
//!     // Build model
//!     let mut model = Sequential::<f32>::new();
//!     model.add(Dense::new(784, 128, Some("relu"), &mut rng)?);
//!     model.add(Dense::new(128, 10, None, &mut rng)?);
//!
//!     // Training configuration
//!     let config = TrainingConfig {
//!         learning_rate: 0.001,
//!         batch_size: 32,
//!         epochs: 10,
//!         validation: Some(ValidationSettings {
//!             enabled: true,
//!             validation_split: 0.2,
//!             batch_size: 32,
//!             num_workers: 0,
//!         }),
//!         ..Default::default()
//!     };
//!
//!     // Create training session
//!     let session = TrainingSession::<f32>::new(config);
//!
//!     // Prepare data
//!     let x_train = Array2::<f32>::zeros((1000, 784));
//!     let y_train = Array2::<f32>::zeros((1000, 10));
//!
//!     // Train
//!     // session.fit(&x_train, &y_train)?;
//!
//!     Ok(())
//! }
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
//! Current version: **0.1.0-rc.1** (Released October 03, 2025)

pub mod activations_minimal;
pub mod autograd;
pub mod error;
// pub mod gpu; // Disabled in minimal version - has syntax errors
pub mod layers;
pub mod losses;
pub mod training;
pub mod utils;

pub use activations_minimal::{Activation, ReLU, Sigmoid, Softmax, Tanh, GELU};
pub use error::{Error, NeuralError, Result};
pub use layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, LayerNorm, Sequential, LSTM};
pub use losses::{
    ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, MeanSquaredError, TripletLoss,
};
pub use training::{TrainingConfig, TrainingSession};

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
        error::{Error, NeuralError, Result},
        layers::{BatchNorm, Conv2D, Dense, Dropout, Layer, LayerNorm, Sequential, LSTM},
        losses::{
            ContrastiveLoss, CrossEntropyLoss, FocalLoss, Loss, MeanSquaredError, TripletLoss,
        },
        training::{TrainingConfig, TrainingSession},
    };
}
