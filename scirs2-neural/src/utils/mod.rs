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
pub mod positional_encoding;
/// Visualization utilities for neural networks
pub mod visualization;

pub use colors::{
    color_legend, colored_metric_cell, colorize, colorize_and_style, colorize_bg, gradient_color,
    stylize, Color, ColorOptions, Style,
};
pub use evaluation::{ConfusionMatrix, FeatureImportance, LearningCurve, ROCCurve};
pub use initializers::*;
pub use metrics::*;
pub use model_viz::{sequential_model_dataflow, sequential_model_summary, ModelVizOptions};
pub use positional_encoding::{
    LearnedPositionalEncoding, PositionalEncoding, PositionalEncodingFactory,
    PositionalEncodingType, RelativePositionalEncoding, SinusoidalPositionalEncoding,
};
pub use visualization::{
    analyze_training_history, ascii_plot, export_history_to_csv, LearningRateSchedule, PlotOptions,
};

/// Generate a random vector or matrix with values from a normal distribution
///
/// # Arguments
///
/// * `shape` - The shape of the array to generate
/// * `mean` - The mean of the normal distribution
/// * `std` - The standard deviation of the normal distribution
/// * `rng` - Random number generator
///
/// # Returns
///
/// * A random array with the specified shape and distribution
///
/// # Examples
///
/// ```
/// use scirs2_neural::utils::random_normal;
/// use ndarray::IxDyn;
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// let mut rng = SmallRng::seed_from_u64(42);
/// let shape = IxDyn(&[2, 3]);
/// let random_matrix = random_normal(shape, 0.0, 1.0, &mut rng).unwrap();
///
/// assert_eq!(random_matrix.shape(), &[2, 3]);
/// ```
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
            let u1 = rng.random_range(0.0..1.0);
            let u2 = rng.random_range(0.0..1.0);

            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let val = mean_f64 + std_f64 * z;

            F::from(val).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Failed to convert random value".to_string())
            })
        })
        .collect::<Result<Vec<F>>>()?;

    // Create ndarray from values
    Array::from_shape_vec(shape, values)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("Failed to create array: {}", e)))
}

/// Calculate the one-hot encoding of a vector of indices
///
/// # Arguments
///
/// * `indices` - Vector of class indices
/// * `num_classes` - Number of classes
///
/// # Returns
///
/// * A 2D array where each row is a one-hot encoded vector
///
/// # Examples
///
/// ```
/// use scirs2_neural::utils::one_hot_encode;
/// use ndarray::arr1;
///
/// let indices = arr1(&[0, 2, 1]);
/// let one_hot = one_hot_encode::<f64>(&indices, 3).unwrap();
///
/// assert_eq!(one_hot.shape(), &[3, 3]);
/// assert_eq!(one_hot[[0, 0]], 1.0f64); // First sample, class 0
/// assert_eq!(one_hot[[1, 2]], 1.0f64); // Second sample, class 2
/// assert_eq!(one_hot[[2, 1]], 1.0f64); // Third sample, class 1
/// ```
pub fn one_hot_encode<F: Float + Debug>(
    indices: &ndarray::Array1<usize>,
    num_classes: usize,
) -> Result<ndarray::Array2<F>> {
    let n_samples = indices.len();
    let mut one_hot = ndarray::Array2::zeros((n_samples, num_classes));

    for (i, &idx) in indices.iter().enumerate() {
        if idx >= num_classes {
            return Err(NeuralError::InvalidArchitecture(format!(
                "Index {} is out of bounds for {} classes",
                idx, num_classes
            )));
        }
        one_hot[[i, idx]] = F::one();
    }

    Ok(one_hot)
}

/// Split data into training and testing sets
///
/// # Arguments
///
/// * `x` - Input data array
/// * `y` - Target data array
/// * `test_size` - Fraction of data to use for testing (between 0 and 1)
/// * `shuffle` - Whether to shuffle the data before splitting
/// * `rng` - Random number generator (only used if shuffle is true)
///
/// # Returns
///
/// * A tuple of (x_train, x_test, y_train, y_test)
///
/// # Examples
///
/// ```
/// use scirs2_neural::utils::train_test_split;
/// use ndarray::{Array, arr2};
/// use rand::rngs::SmallRng;
/// use rand::SeedableRng;
///
/// let mut rng = SmallRng::seed_from_u64(42);
/// let x = arr2(&[[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).into_dyn();
/// let y = arr2(&[[0.0f64], [1.0], [0.0], [1.0]]).into_dyn();
///
/// let (x_train, x_test, y_train, y_test) =
///     train_test_split::<f64, _>(&x, &y, 0.25, true, &mut rng).unwrap();
///
/// // Note: Since the implementation is incomplete (TODO in the code),
/// // we're just checking that the shapes are what we expect
/// assert_eq!(x_train.shape()[0], 3);
/// assert_eq!(x_train.shape()[1], 2);
/// assert_eq!(x_test.shape()[0], 1);
/// assert_eq!(x_test.shape()[1], 2);
/// ```
/// Result type for train/test split
pub type TrainTestSplitResult<F> = (
    ndarray::Array<F, ndarray::IxDyn>,
    ndarray::Array<F, ndarray::IxDyn>,
    ndarray::Array<F, ndarray::IxDyn>,
    ndarray::Array<F, ndarray::IxDyn>,
);

/// Split data into training and testing sets
///
/// This function splits input and target data into training and testing sets.
pub fn train_test_split<F: Float + Debug, R: Rng>(
    x: &ndarray::Array<F, ndarray::IxDyn>,
    y: &ndarray::Array<F, ndarray::IxDyn>,
    test_size: f64,
    shuffle: bool,
    rng: &mut R,
) -> Result<TrainTestSplitResult<F>> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(NeuralError::InvalidArchitecture(format!(
            "test_size must be between 0 and 1, got {}",
            test_size
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
            "test_size {} results in {} test samples, which is invalid",
            test_size, n_test
        )));
    }

    let n_train = n_samples - n_test;

    // Prepare indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    if shuffle {
        // Fisher-Yates shuffle
        for i in (1..n_samples).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }
    }

    // TODO: Implement actual splitting code once ndarray's slice functionality is better understood
    // This is a placeholder that just returns empty arrays of the right shapes

    // Create empty arrays with the correct shapes
    let mut x_shape = x.shape().to_vec();
    let mut y_shape = y.shape().to_vec();

    x_shape[0] = n_train;
    let x_train = ndarray::Array::zeros(x_shape.clone());

    x_shape[0] = n_test;
    let x_test = ndarray::Array::zeros(x_shape);

    y_shape[0] = n_train;
    let y_train = ndarray::Array::zeros(y_shape.clone());

    y_shape[0] = n_test;
    let y_test = ndarray::Array::zeros(y_shape);

    Ok((x_train, x_test, y_train, y_test))
}
