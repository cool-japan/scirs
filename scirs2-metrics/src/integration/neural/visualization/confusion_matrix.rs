//! Confusion matrix visualizations for neural networks
//!
//! This module provides utilities for visualizing confusion matrices from neural
//! network evaluation, especially for classification problems.

use crate::classification::confusion_matrix;
use crate::visualization::confusion_matrix::confusion_matrix_visualization;
use crate::visualization::MetricVisualizer;
use scirs2_core::ndarray::{Array, Ix1, IxDyn};
use std::error::Error;

/// Create a confusion matrix visualizer from neural network predictions and targets
#[allow(dead_code)]
pub fn neural_confusion_matrix_visualization<F: scirs2_core::numeric::Float + std::fmt::Debug>(
    y_true: &Array<F, IxDyn>,
    y_pred: &Array<F, IxDyn>,
    labels: Option<Vec<String>>,
    normalize: bool,
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

    // We need to convert to integers since confusion_matrix requires Ord + Hash
    let y_true_i32 = y_true_f64.mapv(|x| x.round() as i32);
    let y_pred_i32 = y_pred_f64.mapv(|x| x.round() as i32);

    // Compute confusion matrix
    let (cm, classes) = confusion_matrix(&y_true_i32, &y_pred_i32, None)?;

    // Convert classes to labels if not provided
    let class_labels = match labels {
        Some(l) => l,
        None => classes.iter().map(|c| format!("Class {}", c)).collect(),
    };

    // Convert to f64 for visualization
    let cm_f64 = cm.mapv(|x| x as f64);

    // Create visualization
    let visualizer = confusion_matrix_visualization(cm_f64, Some(class_labels), normalize);

    Ok(visualizer)
}
