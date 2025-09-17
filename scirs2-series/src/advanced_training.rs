//! Advanced Training Methods for Time Series
//!
//! This module has been refactored into a modular structure for better maintainability
//! and organization. All original functionality is preserved through re-exports.
//!
//! ## Refactored Module Structure
//!
//! The advanced training functionality has been organized into focused sub-modules:
//!
//! - **Configuration**: Common data structures and configurations
//! - **Meta-Learning**: MAML and other meta-learning algorithms
//! - **Neural ODEs**: Continuous-time neural networks with ODE solvers
//! - **Variational Methods**: VAEs for probabilistic time series modeling
//! - **Transformers**: Attention-based sequence modeling
//! - **Hyperparameter Optimization**: Automated parameter tuning
//! - **Few-Shot Learning**: Prototypical Networks and REPTILE
//! - **Memory-Augmented Networks**: External memory architectures
//! - **Meta-Optimization**: Learned optimizers for adaptive updates
//!
//! ## Usage
//!
//! All original APIs are preserved. You can continue to use this module exactly
//! as before:
//!
//! ```rust
//! use scirs2_series::advanced_training::{MAML, TimeSeriesVAE, TimeSeriesTransformer};
//!
//! // Create a MAML instance
//! let mut maml = MAML::<f64>::new(4, 8, 2, 0.01, 0.1, 5);
//!
//! // Create a VAE for time series
//! let vae = TimeSeriesVAE::<f64>::new(10, 3, 5, 16, 16);
//!
//! // Create a transformer for forecasting
//! let transformer = TimeSeriesTransformer::<f64>::new(12, 6, 64, 8, 4, 256);
//! ```
//!
//! ## Advanced Usage
//!
//! For more specific functionality, you can also import directly from sub-modules:
//!
//! ```rust
//! use scirs2_series::advanced_training::few_shot::{PrototypicalNetworks, REPTILE};
//! use scirs2_series::advanced_training::hyperparameter_optimization::{
//!     HyperparameterOptimizer, OptimizationMethod
//! };
//! ```

// Re-export the entire modular structure
pub use self::advanced_training_modules::*;

// Import the modular implementation
#[path = "advanced_training_modules/mod.rs"]
mod advanced_training_modules;
