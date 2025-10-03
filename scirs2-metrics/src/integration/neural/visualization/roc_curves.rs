//! ROC curve visualizations for neural networks
//!
//! This module provides utilities for visualizing ROC curves from neural
//! network evaluation, especially for binary classification problems.

use crate::classification::curves::roc_curve;
use crate::visualization::roc_curve::roc_curve_visualization;
use crate::visualization::MetricVisualizer;
use scirs2_core::ndarray::{Array, Ix1, IxDyn};
use std::error::Error;

/// Create a ROC curve visualizer from neural network predictions and targets
#[allow(dead_code)]
pub fn neural_roc_curve_visualization<F: scirs2_core::numeric::Float + std::fmt::Debug>(
    y_true: &Array<F, IxDyn>,
    y_pred: &Array<F, IxDyn>,
    auc: Option<f64>,
) -> Result<Box<dyn MetricVisualizer>, Box<dyn Error>> {
    // Convert to f64 arrays and ensure 1D shape
    let y_true_f64 = y_true
        .clone()
        .mapv(|x| x.to_f64().unwrap_or(0.0))
        .into_dimensionality::<Ix1>()?;

    let y_pred_f64 = y_pred
        .clone()
        .mapv(|x| x.to_f64().unwrap_or(0.0))
        .into_dimensionality::<Ix1>()?;

    // Compute ROC curve
    let (fpr, tpr, thresholds) = roc_curve(&y_true_f64, &y_pred_f64)?;

    // Create visualization
    let visualizer =
        roc_curve_visualization(fpr.to_vec(), tpr.to_vec(), Some(thresholds.to_vec()), auc);

    Ok(Box::new(visualizer))
}
