//! Integration with scirs2-neural
//!
//! This module provides adapters and utilities for integrating scirs2-metrics
//! with scirs2-neural, allowing metrics to be used during model training,
//! validation, and evaluation.
//!
//! # Feature Flag
//!
//! This integration requires the `neural_common` feature flag to be enabled:
//!
//! ```toml
//! [dependencies]
//! scirs2-metrics = { version = "0.1.0-alpha.3", features = ["neural_common"] }
//! ```
//!
//! # Metric Adapters
//!
//! The [`NeuralMetricAdapter`] provides an adapter for using scirs2-metrics metrics
//! with scirs2-neural models:
//!
//! ```no_run
//! # #[cfg(feature = "neural_common")]
//! # {
//! use scirs2_metrics::integration::neural::NeuralMetricAdapter;
//! use scirs2_metrics::integration::traits::MetricComputation;
//! use ndarray::{Array, IxDyn};
//!
//! // Create metric adapters
//! let accuracy = NeuralMetricAdapter::<f64>::accuracy();
//! let precision = NeuralMetricAdapter::<f64>::precision();
//! let f1_score = NeuralMetricAdapter::<f64>::f1_score();
//!
//! // Use with neural network predictions and targets
//! let predictions = Array::<f64, _>::zeros(IxDyn(&[10, 1]));
//! let targets = Array::<f64, _>::zeros(IxDyn(&[10, 1]));
//!
//! // Compute metrics
//! let acc = accuracy.compute(&predictions, &targets).unwrap();
//! let prec = precision.compute(&predictions, &targets).unwrap();
//! let f1 = f1_score.compute(&predictions, &targets).unwrap();
//! # }
//! ```
//!
//! # Metrics Callback
//!
//! The [`MetricsCallback`] can be used to track metrics during neural network training:
//!
//! ```no_run
//! # #[cfg(feature = "neural_common")]
//! # {
//! use scirs2_metrics::integration::neural::{NeuralMetricAdapter, MetricsCallback};
//!
//! // Create metric adapters
//! let metrics = vec![
//!     NeuralMetricAdapter::<f32>::accuracy(),
//!     NeuralMetricAdapter::<f32>::precision(),
//!     NeuralMetricAdapter::<f32>::f1_score(),
//! ];
//!
//! // Create callback
//! let mut callback = MetricsCallback::new(metrics, true);
//!
//! // In scirs2-neural, use with model training:
//! // model.fit(..., callbacks: &[&callback], ...);
//! # }
//! ```
//!
//! # Visualization
//!
//! Visualization utilities are provided for neural network metrics:
//!
//! ```no_run
//! # #[cfg(feature = "neural_common")]
//! # {
//! use scirs2_metrics::integration::neural::{
//!     neural_roc_curve_visualization,
//!     neural_precision_recall_curve_visualization,
//!     neural_confusion_matrix_visualization,
//!     training_history_visualization,
//! };
//! use ndarray::{Array, IxDyn};
//! use std::collections::HashMap;
//!
//! // Example data
//! let y_true = Array::<f64, _>::zeros(IxDyn(&[100]));
//! let y_score = Array::<f64, _>::zeros(IxDyn(&[100]));
//! let history = vec![HashMap::from([
//!     ("loss".to_string(), 0.5),
//!     ("accuracy".to_string(), 0.85),
//! ])];
//!
//! // Create visualizations
//! let roc_viz = neural_roc_curve_visualization(&y_true, &y_score, Some(0.8)).unwrap();
//! let pr_viz = neural_precision_recall_curve_visualization(&y_true, &y_score, Some(0.75)).unwrap();
//! let cm_viz = neural_confusion_matrix_visualization(
//!     &y_true, &y_score, Some(vec!["Class 0".to_string(), "Class 1".to_string()]), false
//! ).unwrap();
//! let history_viz = training_history_visualization(
//!     vec!["loss".to_string(), "accuracy".to_string()],
//!     history,
//!     None,
//! );
//! # }
//! ```

// Core implementation that's available with neural_common feature
#[cfg(feature = "neural_common")]
mod neural_adapter;
#[cfg(feature = "neural_common")]
pub use neural_adapter::*;

// Neural-integration specific implementations that depend on neural crate
#[cfg(feature = "neural_common")]
mod callback;
#[cfg(feature = "neural_common")]
mod visualization;

// Re-export only when feature is enabled
#[cfg(feature = "neural_common")]
pub use callback::*;
#[cfg(feature = "neural_common")]
pub use visualization::*;
