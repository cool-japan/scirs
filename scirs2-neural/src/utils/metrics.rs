//! Evaluation metrics for neural networks

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, Ix1, Ix2, IxDyn, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Trait for metrics that evaluate model performance
pub trait Metric<F: Float> {
    /// Compute the metric for the given predictions and targets
    fn compute(&self, predictions: &Array<F, IxDyn>, targets: &Array<F, IxDyn>) -> Result<F>;
}

/// Mean squared error metric
pub struct MeanSquaredError;

impl<F: Float + Debug> Metric<F> for MeanSquaredError {
    fn compute(&self, predictions: &Array<F, IxDyn>, targets: &Array<F, IxDyn>) -> Result<F> {
        mean_squared_error(predictions, targets)
    }
}

/// Binary accuracy metric for classification
pub struct BinaryAccuracy {
    /// Threshold for classifying a prediction as positive
    pub threshold: f64,
}

impl BinaryAccuracy {
    /// Create a new binary accuracy metric with the given threshold
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Default for BinaryAccuracy {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl<F: Float> Metric<F> for BinaryAccuracy {
    fn compute(&self, predictions: &Array<F, IxDyn>, targets: &Array<F, IxDyn>) -> Result<F> {
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::InferenceError(format!(
                "Predictions shape {:?} does not match targets shape {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let threshold = F::from(self.threshold).ok_or_else(|| {
            NeuralError::Other("Could not convert threshold to the required float type".to_string())
        })?;

        let mut correct = 0;
        let n_elements = predictions.len();
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_class = if *pred >= threshold {
                F::one()
            } else {
                F::zero()
            };
            if pred_class == *target {
                correct += 1;
            }
        }

        Ok(F::from(correct).unwrap_or(F::zero()) / F::from(n_elements).unwrap_or(F::one()))
    }
}

/// Categorical accuracy metric for multi-class classification
pub struct CategoricalAccuracy;

impl<F: Float + Debug> Metric<F> for CategoricalAccuracy {
    fn compute(&self, predictions: &Array<F, IxDyn>, targets: &Array<F, IxDyn>) -> Result<F> {
        if predictions.ndim() >= 2 && targets.ndim() >= 2 {
            categorical_accuracy(
                &predictions.to_owned().into_dimensionality::<Ix2>().unwrap(),
                &targets.to_owned().into_dimensionality::<Ix2>().unwrap(),
            )
        } else {
            Err(NeuralError::Other(
                "Predictions and targets must have at least 2 dimensions for categorical accuracy"
                    .to_string(),
            ))
        }
    }
}

/// Coefficient of determination (R²) metric
pub struct R2Score;

impl<F: Float> Metric<F> for R2Score {
    fn compute(&self, predictions: &Array<F, IxDyn>, targets: &Array<F, IxDyn>) -> Result<F> {
        let n_elements = F::from(targets.len()).unwrap_or(F::one());

        // Calculate mean of targets
        let target_mean = targets.iter().fold(F::zero(), |acc, &x| acc + x) / n_elements;

        // Calculate total sum of squares
        let mut ss_tot = F::zero();
        for target in targets.iter() {
            let diff = *target - target_mean;
            ss_tot = ss_tot + diff * diff;
        }

        // Calculate residual sum of squares
        let mut ss_res = F::zero();
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let diff = *target - *pred;
            ss_res = ss_res + diff * diff;
        }

        // Calculate R²
        let r2 = F::one() - ss_res / ss_tot;

        // R² can be negative if the model is worse than predicting the mean
        Ok(r2)
    }
}

/// Calculate the mean squared error between predictions and targets
///
/// # Arguments
/// * `predictions` - Predicted values
/// * `targets` - Target values
///
/// # Returns
/// * The mean squared error
///
/// # Examples
/// ```
/// use scirs2_neural::utils::mean_squared_error;
/// use scirs2_core::ndarray::arr1;
///
/// let predictions = arr1(&[1.0f64, 2.0, 3.0]).into_dyn();
/// let targets = arr1(&[1.5f64, 1.8, 2.5]).into_dyn();
/// let mse = mean_squared_error(&predictions, &targets).unwrap();
/// assert!(mse > 0.0f64);
/// ```
#[allow(dead_code)]
pub fn mean_squared_error<F: Float + Debug>(
    predictions: &Array<F, IxDyn>,
    targets: &Array<F, IxDyn>,
) -> Result<F> {
    if predictions.shape() != targets.shape() {
        return Err(NeuralError::InferenceError(format!(
            "Shape mismatch in mean_squared_error: predictions {:?} vs targets {:?}",
            predictions.shape(),
            targets.shape()
        )));
    }

    let n = F::from(predictions.len())
        .ok_or_else(|| NeuralError::Other("Could not convert array length to float".to_string()))?;

    let mut sum_squared_diff = F::zero();
    for (p, t) in predictions.iter().zip(targets.iter()) {
        let diff = *p - *t;
        sum_squared_diff = sum_squared_diff + diff * diff;
    }

    Ok(sum_squared_diff / n)
}

/// Calculate the binary accuracy between predictions and targets
///
/// # Arguments
/// * `predictions` - Predicted values (should be between 0 and 1)
/// * `targets` - Target values (should be either 0 or 1)
/// * `threshold` - Threshold value for binary classification (default: 0.5)
///
/// # Returns
/// * The accuracy (proportion of correct predictions)
///
/// # Examples
/// ```
/// use scirs2_neural::utils::binary_accuracy;
/// use scirs2_core::ndarray::arr1;
///
/// let predictions = arr1(&[0.7f64, 0.3, 0.8, 0.2]);
/// let targets = arr1(&[1.0f64, 0.0, 1.0, 0.0]);
/// let accuracy = binary_accuracy(&predictions, &targets, 0.5f64).unwrap();
/// assert_eq!(accuracy, 1.0f64); // All predictions are correct
/// ```
#[allow(dead_code)]
pub fn binary_accuracy<F: Float + Debug>(
    predictions: &Array<F, Ix1>,
    targets: &Array<F, Ix1>,
    threshold: F,
) -> Result<F> {
    if predictions.shape() != targets.shape() {
        return Err(NeuralError::InferenceError(format!(
            "Shape mismatch in binary_accuracy: predictions {:?} vs targets {:?}",
            predictions.shape(),
            targets.shape()
        )));
    }

    let n = F::from(predictions.len()).ok_or_else(|| {
        NeuralError::InferenceError("Could not convert array length to float".to_string())
    })?;

    let mut correct = F::zero();
    Zip::from(predictions).and(targets).for_each(|&p, &t| {
        let pred_class = if p >= threshold { F::one() } else { F::zero() };
        if pred_class == t {
            correct = correct + F::one();
        }
    });

    Ok(correct / n)
}

/// Calculate the categorical accuracy between predictions and targets
///
/// # Arguments
/// * `predictions` - Predicted class probabilities (each row sums to 1)
/// * `targets` - One-hot encoded target classes (each row has a single 1)
///
/// # Returns
/// * The accuracy (proportion of correct predictions)
///
/// # Examples
/// ```
/// use scirs2_neural::utils::categorical_accuracy;
/// use scirs2_core::ndarray::arr2;
///
/// let predictions = arr2(&[
///     [0.7f64, 0.2, 0.1],  // Predicted class: 0
///     [0.3f64, 0.6, 0.1],  // Predicted class: 1
///     [0.2f64, 0.3, 0.5]   // Predicted class: 2
/// ]);
/// let targets = arr2(&[
///     [1.0f64, 0.0, 0.0],  // True class: 0
///     [0.0f64, 1.0, 0.0],  // True class: 1
///     [0.0f64, 0.0, 1.0]   // True class: 2
/// ]);
/// let accuracy = categorical_accuracy(&predictions, &targets).unwrap();
/// assert_eq!(accuracy, 1.0f64); // All predictions are correct
/// ```
#[allow(dead_code)]
pub fn categorical_accuracy<F: Float + Debug>(
    predictions: &Array<F, Ix2>,
    targets: &Array<F, Ix2>,
) -> Result<F> {
    if predictions.shape() != targets.shape() {
        return Err(NeuralError::InferenceError(format!(
            "Shape mismatch in categorical_accuracy: predictions {:?} vs targets {:?}",
            predictions.shape(),
            targets.shape()
        )));
    }

    let n = F::from(predictions.shape()[0]).ok_or_else(|| {
        NeuralError::InferenceError("Could not convert sample count to float".to_string())
    })?;

    let mut correct = F::zero();
    for i in 0..predictions.shape()[0] {
        // Find predicted class (index of max value)
        let mut pred_class = 0;
        let mut max_prob = predictions[[i, 0]];
        for j in 1..predictions.shape()[1] {
            if predictions[[i, j]] > max_prob {
                max_prob = predictions[[i, j]];
                pred_class = j;
            }
        }

        // Find true class (index of 1 in one-hot encoding)
        let mut true_class = 0;
        for j in 0..targets.shape()[1] {
            if targets[[i, j]] == F::one() {
                true_class = j;
                break;
            }
        }

        // Check if prediction is correct
        if pred_class == true_class {
            correct = correct + F::one();
        }
    }

    Ok(correct / n)
}
