//! Integration with other scirs2 modules
//!
//! This module provides adapters and utilities for integrating scirs2-metrics
//! with other modules in the scirs2 ecosystem, such as scirs2-neural and scirs2-optim.
//!
//! # Neural Integration
//!
//! The [`neural`] module provides integration with the scirs2-neural module, allowing
//! metrics to be used during model training, evaluation, and visualization.
//!
//! ```toml
//! [dependencies]
//! scirs2-metrics = { version = "0.1.0-alpha.3", features = ["neural_common"] }
//! ```
//!
//! ## Basic Usage
//!
//! ```no_run
//! # #[cfg(feature = "neural_common")]
//! # {
//! use scirs2_metrics::integration::neural::NeuralMetricAdapter;
//! use ndarray::Array;
//!
//! // Create metric adapters for common metrics
//! let accuracy = NeuralMetricAdapter::<f64>::accuracy();
//! let precision = NeuralMetricAdapter::<f64>::precision();
//! let recall = NeuralMetricAdapter::<f64>::recall();
//! let f1_score = NeuralMetricAdapter::<f64>::f1_score();
//! let mse = NeuralMetricAdapter::<f64>::mse();
//! let r2 = NeuralMetricAdapter::<f64>::r2();
//!
//! // Custom metric adapter
//! let custom_metric = NeuralMetricAdapter::new(
//!     "custom_metric",
//!     Box::new(|preds, targets| {
//!         // Custom metric calculation
//!         Ok(0.5)
//!     }),
//! );
//! # }
//! ```
//!
//! ## Training Callbacks
//!
//! The [`MetricsCallback`] can be used to track metrics during neural network training.
//!
//! ## Visualizations
//!
//! Visualization utilities are provided for common model evaluation plots:
//!
//! - Training history curves
//! - ROC curves
//! - Precision-Recall curves
//! - Confusion matrices
//!
//! # Optimization Integration
//!
//! The [`optim`] module provides integration with the scirs2-optim module, allowing
//! metrics to be used for hyperparameter optimization and learning rate scheduling.
//!
//! ```toml
//! [dependencies]
//! scirs2-metrics = { version = "0.1.0-alpha.3", features = ["optim_integration"] }
//! ```
//!
//! ## Basic Usage
//!
//! ```no_run
//! # #[cfg(feature = "optim_integration")]
//! # {
//! use scirs2_metrics::integration::optim::{MetricOptimizer, MetricLRScheduler};
//! use ndarray::Array1;
//!
//! // Create a metric optimizer for accuracy
//! let metric_optimizer = MetricOptimizer::new("accuracy", true);
//!
//! // Create scheduler configuration for external use
//! let scheduler_config = metric_optimizer.create_scheduler_config(0.1, 0.1, 5, 1e-6);
//!
//! // Create an actual scheduler using the configuration
//! let mut scheduler = MetricLRScheduler::new(
//!     scheduler_config.initial_lr,
//!     scheduler_config.factor,
//!     scheduler_config.patience,
//!     scheduler_config.min_lr,
//!     &scheduler_config.metric_name,
//!     true // maximize
//! );
//!
//! // Update scheduler based on a metric value
//! let metric_value = 0.85;
//! let new_lr = scheduler.step_with_metric(metric_value);
//! # }
//! ```
//!

// Core integration traits (no dependencies on other modules)
pub mod traits;

// Feature-gated integration modules
#[cfg(feature = "neural_common")]
pub mod neural;

#[cfg(feature = "optim_integration")]
pub mod optim;
