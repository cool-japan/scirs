//! Evaluation utilities and visualizations for model performance
//!
//! This module provides visualization and evaluation tools for neural networks,
//! including confusion matrices, ROC curves, learning curves, and feature importance.
//! These tools help visualize and analyze model performance and training metrics.

mod confusion_matrix;
mod feature_importance;
mod helpers;
mod learning_curve;
mod roc_curve;

pub use confusion_matrix::ConfusionMatrix;
pub use feature_importance::FeatureImportance;
pub use learning_curve::LearningCurve;
pub use roc_curve::ROCCurve;
// Helper functions are kept internal to the module
