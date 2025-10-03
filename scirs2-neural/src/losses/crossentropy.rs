//! Cross-entropy loss function implementation

use crate::error::{NeuralError, Result};
use crate::losses::Loss;
use scirs2_core::ndarray::{Array, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// Cross-entropy loss function.
///
/// The cross-entropy loss is defined as:
/// L = -sum(y_true * log(y_pred)) where y_true are target probabilities and y_pred are predicted probabilities.
/// It is commonly used for classification problems.
/// # Examples
/// ```
/// use scirs2_neural::losses::CrossEntropyLoss;
/// use scirs2_neural::losses::Loss;
/// use scirs2_core::ndarray::{Array, arr1, arr2};
/// let ce = CrossEntropyLoss::new(1e-10);
/// // One-hot encoded targets and softmax'd predictions for a 3-class problem
/// let predictions = arr2(&[
///     [0.7, 0.2, 0.1],  // First sample, class probabilities
///     [0.3, 0.6, 0.1]   // Second sample, class probabilities
/// ]).into_dyn();
/// let targets = arr2(&[
///     [1.0, 0.0, 0.0],  // First sample, true class is 0
///     [0.0, 1.0, 0.0]   // Second sample, true class is 1
/// ]).into_dyn();
/// // Forward pass to calculate loss
/// let loss = ce.forward(&predictions, &targets).unwrap();
/// // Backward pass to calculate gradients
/// let gradients = ce.backward(&predictions, &targets).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss {
    /// Small epsilon value to avoid log(0)
    epsilon: f64,
}
impl CrossEntropyLoss {
    /// Create a new cross-entropy loss function
    ///
    /// # Arguments
    /// * `epsilon` - Small value to add to predictions to avoid log(0)
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new(1e-10)
    }
}

impl<F: Float + Debug> Loss<F> for CrossEntropyLoss {
    fn forward(
        &self,
        predictions: &Array<F, scirs2_core::ndarray::IxDyn>,
        targets: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<F> {
        // Check shape compatibility
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::InferenceError(format!(
                "Shape mismatch in CrossEntropy: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }
        let epsilon = F::from(self.epsilon).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert epsilon to float".to_string())
        })?;
        let n = F::from(if predictions.ndim() > 1 {
            predictions.shape()[0]
        } else {
            1
        })
        .ok_or_else(|| {
            NeuralError::InferenceError("Could not convert batch size to float".to_string())
        })?;
        // Calculate cross-entropy loss
        let mut loss = F::zero();
        // If we have a batch (first dimension is batch size)
        if predictions.ndim() > 1 {
            for i in 0..predictions.shape()[0] {
                let mut sample_loss = F::zero();
                for j in 0..predictions.shape()[1] {
                    let y_pred = predictions[[i, j]].max(epsilon).min(F::one() - epsilon);
                    let y_true = targets[[i, j]];
                    // Only add to loss if target is non-zero (for sparse targets)
                    if y_true > F::zero() {
                        sample_loss = sample_loss - y_true * y_pred.ln();
                    }
                }
                loss = loss + sample_loss;
            }
            // Average over batch
            loss = loss / n;
        } else {
            // Single sample case
            Zip::from(predictions).and(targets).for_each(|&p, &t| {
                let p_safe = p.max(epsilon).min(F::one() - epsilon);
                if t > F::zero() {
                    loss = loss - t * p_safe.ln();
                }
            });
        }
        Ok(loss)
    }
    fn backward(
        &self,
        predictions: &Array<F, scirs2_core::ndarray::IxDyn>,
        targets: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<Array<F, scirs2_core::ndarray::IxDyn>> {
        // Check shape compatibility
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::InferenceError(format!(
                "Shape mismatch in CrossEntropy gradient: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let epsilon = F::from(self.epsilon).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert epsilon to float".to_string())
        })?;
        let n = F::from(if predictions.ndim() > 1 {
            predictions.shape()[0]
        } else {
            1
        })
        .ok_or_else(|| {
            NeuralError::InferenceError("Could not convert batch size to float".to_string())
        })?;

        // For cross-entropy with softmax, the gradient is (p - t)
        let mut gradients = predictions.clone();
        Zip::from(&mut gradients).and(targets).for_each(|grad, &t| {
            let _p_safe = grad.max(epsilon).min(F::one() - epsilon);
            // Note: this simplification works because we assume predictions come from a softmax
            *grad = (*grad - t) / n;
        });
        Ok(gradients)
    }
}
