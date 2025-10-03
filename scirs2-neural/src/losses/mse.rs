//! Mean squared error loss function implementation

use crate::error::{NeuralError, Result};
use crate::losses::Loss;
use scirs2_core::ndarray::{Array, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Mean squared error loss function.
///
/// The MSE is calculated as the average of squared differences between
/// predictions and targets: MSE = mean((predictions - targets)^2)
/// It is commonly used for regression problems.
///
/// # Examples
/// ```
/// use scirs2_neural::losses::MeanSquaredError;
/// use scirs2_neural::losses::Loss;
/// use scirs2_core::ndarray::{Array, arr1};
///
/// let mse = MeanSquaredError::new();
/// let predictions = arr1(&[1.0, 2.0, 3.0]).into_dyn();
/// let targets = arr1(&[1.5, 1.8, 2.5]).into_dyn();
/// // Forward pass to calculate loss
/// let loss = mse.forward(&predictions, &targets).unwrap();
/// // Backward pass to calculate gradients
/// let gradients = mse.backward(&predictions, &targets).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MeanSquaredError;

impl MeanSquaredError {
    /// Create a new mean squared error loss function
    pub fn new() -> Self {
        Self
    }
}

impl Default for MeanSquaredError {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + Debug> Loss<F> for MeanSquaredError {
    fn forward(
        &self,
        predictions: &Array<F, scirs2_core::ndarray::IxDyn>,
        targets: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<F> {
        // Check shape compatibility
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::InferenceError(format!(
                "Shape mismatch in MSE: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let n = F::from(predictions.len()).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert array length to float".to_string())
        })?;

        // Calculate squared differences and mean
        let mut sum_squared_diff = F::zero();
        Zip::from(predictions).and(targets).for_each(|&p, &t| {
            let diff = p - t;
            sum_squared_diff = sum_squared_diff + diff * diff;
        });
        let mse = sum_squared_diff / n;
        Ok(mse)
    }

    fn backward(
        &self,
        predictions: &Array<F, scirs2_core::ndarray::IxDyn>,
        targets: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Check shape compatibility
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::InferenceError(format!(
                "Shape mismatch in MSE gradient: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let n = F::from(predictions.len()).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert array length to float".to_string())
        })?;

        // dL/dp = 2 * (p - t) / n
        let two = F::one() + F::one();
        let mut gradients = predictions.clone();
        Zip::from(&mut gradients).and(targets).for_each(|grad, &t| {
            *grad = two * (*grad - t) / n;
        });
        Ok(gradients)
    }
}
