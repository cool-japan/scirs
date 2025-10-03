//! Precision-Recall curve visualizations for neural networks
//!
//! This module provides utilities for visualizing Precision-Recall curves from neural
//! network evaluation, especially for binary classification problems.

use crate::classification::curves::precision_recall_curve;
use crate::classification::threshold::average_precision_score;
use crate::visualization::precision_recall::precision_recall_visualization;
use crate::visualization::MetricVisualizer;
use scirs2_core::ndarray::{Array, Ix1, IxDyn};
use std::error::Error;

/// Create a Precision-Recall curve visualizer from neural network predictions and targets
#[allow(dead_code)]
pub fn neural_precision_recall_curve_visualization<
    F: scirs2_core::numeric::Float + std::fmt::Debug,
>(
    y_true: &Array<F, IxDyn>,
    y_pred: &Array<F, IxDyn>,
    average_precision: Option<f64>,
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

    // Compute Precision-Recall curve
    let (precision, recall, thresholds) = precision_recall_curve(&y_true_f64, &y_pred_f64)?;

    // Compute average precision if not provided
    let ap = match average_precision {
        Some(ap) => ap,
        None => average_precision_score(&y_true_f64, &y_pred_f64, None, None)?,
    };

    // Create visualization
    let visualizer = precision_recall_visualization(
        precision.to_vec(),
        recall.to_vec(),
        Some(thresholds.to_vec()),
        Some(ap),
    );

    Ok(Box::new(visualizer))
}
