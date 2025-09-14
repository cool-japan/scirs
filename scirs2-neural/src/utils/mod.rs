//! Utility functions for neural networks
//!
//! This module provides various utility functions for neural networks,
//! such as weight initialization strategies, metric calculations, positional encoding
//! for transformer models, etc.

use crate::error::{NeuralError, Result};
use ndarray::{Array, Dimension};
use num_traits::Float;
use rand::Rng;
use std::fmt::Debug;
/// Terminal color utilities for visualization
pub mod colors;
pub mod datasets;
/// Evaluation utilities and visualizations for model performance
pub mod evaluation;
pub mod initializers;
pub mod metrics;
/// Model architecture visualization utilities
pub mod model_viz;
// pub mod positional_encoding; // Disabled - file is broken
/// Visualization utilities for neural networks
// pub mod visualization; // Disabled - file is broken
pub use colors::{
    color_legend, colored_metric_cell, colorize, colorize_and_style, colorize_bg, gradient_color,
    stylize, Color, ColorOptions, Style,
};
pub use evaluation::{ConfusionMatrix, FeatureImportance, LearningCurve, ROCCurve};
pub use initializers::*;
pub use metrics::*;
pub use model_viz::{sequential_model_dataflow, sequential_model_summary, ModelVizOptions};
// pub use positional__encoding::{
//     LearnedPositionalEncoding, PositionalEncoding, PositionalEncodingFactory,
//     PositionalEncodingType, RelativePositionalEncoding, SinusoidalPositionalEncoding,
// }; // Disabled - module is broken
// pub use visualization::{
//     analyze_training_history, ascii_plot, export_history_to_csv, LearningRateSchedule, PlotOptions,
// }; // Disabled - module is broken
/// Generate a random vector or matrix with values from a normal distribution
///
/// # Arguments
/// * `shape` - The shape of the array to generate
/// * `mean` - The mean of the normal distribution
/// * `std` - The standard deviation of the normal distribution
/// * `rng` - Random number generator
/// # Returns
/// * A random array with the specified shape and distribution
/// # Examples
/// ```
/// use scirs2_neural::utils::random_normal;
/// use ndarray::IxDyn;
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
/// let mut rng = rand::rng();
/// let shape = IxDyn(&[2, 3]);
/// let random_matrix = random_normal(shape, 0.0, 1.0, &mut rng).unwrap();
/// assert_eq!(random_matrix.shape(), &[2, 3]);
#[allow(dead_code)]
pub fn random_normal<F: Float + Debug, R: Rng>(
    shape: ndarray::IxDyn,
    mean: F,
    std: F,
    rng: &mut R,
) -> Result<Array<F, ndarray::IxDyn>> {
    let size = shape.as_array_view().iter().product();
    let mean_f64 = mean.to_f64().ok_or_else(|| {
        NeuralError::InvalidArchitecture("Failed to convert mean to f64".to_string())
    })?;
    let std_f64 = std.to_f64().ok_or_else(|| {
        NeuralError::InvalidArchitecture("Failed to convert std to f64".to_string())
    })?;
    // Generate random values from normal distribution
    let values: Vec<F> = (0..size)
        .map(|_| {
            // Box-Muller transform to generate normal distribution
            let u1 = rng.gen_range(0.0..1.0);
            let u2 = rng.gen_range(0.0..1.0);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let val = mean_f64 + std_f64 * z;
            F::from(val).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Failed to convert random value".to_string())
            })
        })
        .collect::<Result<Vec<F>>>()?;
    // Create ndarray from values
    Array::from_shape_vec(shape, values)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("Failed to create array: {e}")))
}
/// Calculate the one-hot encoding of a vector of indices
/// * `indices` - Vector of class indices
/// * `num_classes` - Number of classes
/// * A 2D array where each row is a one-hot encoded vector
///   use scirs2_neural::utils::one_hot_encode;
///   use ndarray::arr1;
///   let indices = arr1(&[0, 2, 1]);
///   let one_hot = one_hot_encode::<f64>(&indices, 3).unwrap();
///   assert_eq!(one_hot.shape(), &[3, 3]);
///   assert_eq!(one_hot[[0, 0]], 1.0f64); // First sample, class 0
///   assert_eq!(one_hot[[1, 2]], 1.0f64); // Second sample, class 2
///   assert_eq!(one_hot[[2, 1]], 1.0f64); // Third sample, class 1
#[allow(dead_code)]
pub fn one_hot_encode<F: Float + Debug>(
    indices: &ndarray::Array1<usize>,
    num_classes: usize,
) -> Result<ndarray::Array2<F>> {
    let n_samples = indices.len();
    let mut one_hot = ndarray::Array2::zeros((n_samples, num_classes));
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= num_classes {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Index {idx} is out of bounds for {num_classes} _classes"
            )));
        }
        one_hot[[i, idx]] = F::one();
    }
    Ok(one_hot)
}

/// Split data into training and testing sets
/// * `x` - Input data array
/// * `y` - Target data array
/// * `test_size` - Fraction of data to use for testing (between 0 and 1)
/// * `shuffle` - Whether to shuffle the data before splitting
/// * `rng` - Random number generator (only used if shuffle is true)
/// * A tuple of (x_train, x_test, y_train, y_test)
///   use scirs2_neural::utils::train_test_split;
///   use ndarray::{Array, arr2};
///   let x = arr2(&[[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).into_dyn();
///   let y = arr2(&[[0.0f64], [1.0], [0.0], [1.0]]).into_dyn();
///   let (x_train, x_test, y_train, y_test) =
///   train_test_split::<f64>(&x, &y, 0.25, true, &mut rng).unwrap();
///   // Note: Since the implementation is incomplete (TODO in the code),
///   // we're just checking that the shapes are what we expect
///   assert_eq!(x_train.shape()[0], 3);
///   assert_eq!(x_train.shape()[1], 2);
///   assert_eq!(x_test.shape()[0], 1);
///   assert_eq!(x_test.shape()[1], 2);
///   Result type for train/test split
pub type TrainTestSplitResult<F> = (
    ndarray::Array<F, ndarray::IxDyn>, // x_train
    ndarray::Array<F, ndarray::IxDyn>, // x_test
    ndarray::Array<F, ndarray::IxDyn>, // y_train
    ndarray::Array<F, ndarray::IxDyn>, // y_test
);
/// This function splits input and target data into training and testing sets.
#[allow(dead_code)]
pub fn train_test_split<F: Float + Debug, R: Rng>(
    x: &ndarray::Array<F, ndarray::IxDyn>,
    y: &ndarray::Array<F, ndarray::IxDyn>,
    test_size: f64,
    shuffle: bool,
    rng: &mut R,
) -> Result<TrainTestSplitResult<F>> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "test_size must be between 0 and 1, got {test_size}"
        )));
    }

    let n_samples = x.shape()[0];
    if n_samples != y.shape()[0] {
        return Err(NeuralError::InvalidArchitecture(format!(
            "x and y must have the same number of samples, got {} and {}",
            n_samples,
            y.shape()[0]
        )));
    }

    // Calculate split indices
    let n_test = (n_samples as f64 * test_size).round() as usize;
    if n_test == 0 || n_test == n_samples {
        return Err(NeuralError::InvalidArchitecture(format!(
            "test_size {test_size} results in {n_test} test samples, which is invalid"
        )));
    }
    let n_train = n_samples - n_test;
    // Prepare indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    if shuffle {
        // Fisher-Yates shuffle
        for i in (1..n_samples).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }

    // Actually split the data using the shuffled indices
    let train_indices = &indices[..n_train];
    let test_indices = &indices[n_train..];
    // Create output arrays
    let mut xshape = x.shape().to_vec();
    let mut yshape = y.shape().to_vec();
    xshape[0] = n_train;
    let mut x_train = ndarray::Array::zeros(xshape.clone());
    xshape[0] = n_test;
    let mut x_test = ndarray::Array::zeros(xshape);
    yshape[0] = n_train;
    let mut y_train = ndarray::Array::zeros(yshape.clone());
    yshape[0] = n_test;
    let mut y_test = ndarray::Array::zeros(yshape);
    // Copy training data
    for (new_idx, &orig_idx) in train_indices.iter().enumerate() {
        // Copy x data
        let x_slice = x.slice(ndarray::s![orig_idx, ..]);
        let mut x_train_slice = x_train.slice_mut(ndarray::s![new_idx, ..]);
        x_train_slice.assign(&x_slice);
        // Copy y data
        let y_slice = y.slice(ndarray::s![orig_idx, ..]);
        let mut y_train_slice = y_train.slice_mut(ndarray::s![new_idx, ..]);
        y_train_slice.assign(&y_slice);
    }

    // Copy test data
    for (new_idx, &orig_idx) in test_indices.iter().enumerate() {
        // Copy x data
        let x_slice = x.slice(ndarray::s![orig_idx, ..]);
        let mut x_test_slice = x_test.slice_mut(ndarray::s![new_idx, ..]);
        x_test_slice.assign(&x_slice);
        // Copy y data
        let y_slice = y.slice(ndarray::s![orig_idx, ..]);
        let mut y_test_slice = y_test.slice_mut(ndarray::s![new_idx, ..]);
        y_test_slice.assign(&y_slice);
    }

    Ok((x_train, x_test, y_train, y_test))
}
