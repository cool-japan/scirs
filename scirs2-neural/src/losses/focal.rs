//! Focal loss function implementation

use crate::error::{NeuralError, Result};
use crate::losses::Loss;
use scirs2_core::ndarray::Array;
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Focal loss function.
///
/// The focal loss is designed to address class imbalance problems by down-weighting
/// easy examples and focusing training on hard examples. For a single class,
/// the focal loss is defined as:
/// FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
/// where p_t is the model's estimated probability for the true class.
///
/// # Examples
/// ```
/// use scirs2_neural::losses::FocalLoss;
/// use scirs2_neural::losses::Loss;
/// use scirs2_core::ndarray::{Array, arr2};
///
/// let focal = FocalLoss::new(2.0, Some(0.25), 1e-10);
/// // Predictions and targets for a 3-class problem
/// let predictions = arr2(&[
///     [0.7, 0.2, 0.1],  // First sample, class probabilities
///     [0.3, 0.6, 0.1]   // Second sample, class probabilities
/// ]).into_dyn();
/// let targets = arr2(&[
///     [1.0, 0.0, 0.0],  // First sample, true class is 0
///     [0.0, 1.0, 0.0]   // Second sample, true class is 1
/// ]).into_dyn();
/// // Forward pass to calculate loss
/// let loss = focal.forward(&predictions, &targets).unwrap();
/// // Backward pass to calculate gradients
/// let gradients = focal.backward(&predictions, &targets).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FocalLoss {
    /// Focusing parameter gamma (reduces the relative loss for well-classified examples)
    gamma: f64,
    /// Optional weighting factor alpha
    alpha: Option<f64>,
    /// Optional class-specific alpha weights
    alpha_per_class: Option<Vec<f64>>,
    /// Small epsilon value to avoid log(0)
    epsilon: f64,
}

impl FocalLoss {
    /// Create a new focal loss function with a single alpha value for all classes
    ///
    /// # Arguments
    /// * `gamma` - Focusing parameter, gamma >= 0. Higher gamma means more focus on misclassified examples.
    /// * `alpha` - Optional weighting factor, typically between 0 and 1.
    /// * `epsilon` - Small value to add to predictions to avoid log(0)
    pub fn new(gamma: f64, alpha: Option<f64>, epsilon: f64) -> Self {
        Self {
            gamma,
            alpha,
            alpha_per_class: None,
            epsilon,
        }
    }

    /// Create a focal loss with class-specific alpha weights
    /// * `alpha_per_class` - Vector of weighting factors, one per class
    pub fn with_class_weights(gamma: f64, alpha_perclass: Vec<f64>, epsilon: f64) -> Self {
        Self {
            gamma,
            alpha: None,
            alpha_per_class: Some(alpha_perclass),
            epsilon,
        }
    }
}

impl Default for FocalLoss {
    fn default() -> Self {
        // Default values typically used in literature: gamma=2.0, alpha=0.25
        Self::new(2.0, Some(0.25), 1e-10)
    }
}

impl<F: Float + Debug> Loss<F> for FocalLoss {
    fn forward(
        &self,
        predictions: &Array<F, scirs2_core::ndarray::IxDyn>,
        targets: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<F> {
        // Check shape compatibility
        if predictions.shape() != targets.shape() {
            return Err(NeuralError::InferenceError(format!(
                "Shape mismatch in FocalLoss: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let gamma = F::from(self.gamma).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert gamma to float".to_string())
        })?;
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

        // Check if we have per-class alpha values and if it matches the number of classes
        if let Some(ref alpha_per_class) = self.alpha_per_class {
            let num_classes = if predictions.ndim() > 1 {
                predictions.shape()[1]
            } else {
                predictions.len()
            };
            if alpha_per_class.len() != num_classes {
                return Err(NeuralError::InferenceError(format!(
                    "Number of alpha values ({}) does not match number of classes ({})",
                    alpha_per_class.len(),
                    num_classes
                )));
            }
        }

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
                        // Get the alpha value for this class
                        let alpha = if let Some(ref alpha_per_class) = self.alpha_per_class {
                            F::from(alpha_per_class[j]).ok_or_else(|| {
                                NeuralError::InferenceError(
                                    "Could not convert alpha to float".to_string(),
                                )
                            })?
                        } else if let Some(alpha) = self.alpha {
                            F::from(alpha).ok_or_else(|| {
                                NeuralError::InferenceError(
                                    "Could not convert alpha to float".to_string(),
                                )
                            })?
                        } else {
                            F::one()
                        };
                        // Probability for the target class
                        let p_t = y_pred;
                        // Focal weight: (1 - p_t)^gamma
                        let focal_weight = (F::one() - p_t).powf(gamma);
                        // Term added to loss
                        sample_loss = sample_loss - alpha * focal_weight * y_true * p_t.ln();
                    }
                }
                loss = loss + sample_loss;
            }
            // Average over batch
            loss = loss / n;
        } else {
            // Single sample case
            for j in 0..predictions.len() {
                let p = predictions[j];
                let t = targets[j];
                if t > F::zero() {
                    // Get the alpha value for this class
                    let alpha = if let Some(ref alpha_per_class) = self.alpha_per_class {
                        F::from(alpha_per_class[j]).unwrap_or(F::one())
                    } else if let Some(a) = self.alpha {
                        F::from(a).unwrap_or(F::one())
                    } else {
                        F::one()
                    };
                    let p_safe = p.max(epsilon).min(F::one() - epsilon);
                    let focal_weight = (F::one() - p_safe).powf(gamma);
                    loss = loss - alpha * focal_weight * t * p_safe.ln();
                }
            }
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
                "Shape mismatch in FocalLoss gradient: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let gamma = F::from(self.gamma).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert gamma to float".to_string())
        })?;
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

        // Initialize gradients
        let mut gradients = Array::zeros(predictions.raw_dim());

        // Calculate gradients for focal loss
        if predictions.ndim() == 1 {
            // For 1D arrays (single sample)
            for idx in 0..predictions.len() {
                let p = predictions[idx];
                let t = targets[idx];
                if t > F::zero() {
                    let p_safe = p.max(epsilon).min(F::one() - epsilon);
                    // Get alpha for this class
                    let alpha = if let Some(ref alpha_per_class) = self.alpha_per_class {
                        F::from(alpha_per_class[idx]).unwrap_or(F::one())
                    } else if let Some(a) = self.alpha {
                        F::from(a).unwrap_or(F::one())
                    } else {
                        F::one()
                    };
                    // Term 1: -alpha * gamma * (1-p_t)^(gamma-1) * log(p_t)
                    let term1 =
                        -alpha * gamma * (F::one() - p_safe).powf(gamma - F::one()) * p_safe.ln();
                    // Term 2: -alpha * (1-p_t)^gamma * (1/p_t)
                    let term2 = -alpha * (F::one() - p_safe).powf(gamma) * (F::one() / p_safe);
                    gradients[idx] = (term1 + term2) * t / n;
                } else {
                    gradients[idx] = F::zero();
                }
            }
        } else {
            // For multi-dimensional arrays (batched)
            let batch_size = predictions.shape()[0];
            for i in 0..batch_size {
                for j in 0..predictions.shape()[1] {
                    let p = predictions[[i, j]];
                    let t = targets[[i, j]];
                    if t > F::zero() {
                        let p_safe = p.max(epsilon).min(F::one() - epsilon);
                        // Get alpha for this class
                        let alpha = if let Some(ref alpha_per_class) = self.alpha_per_class {
                            F::from(alpha_per_class[j]).unwrap_or(F::one())
                        } else if let Some(a) = self.alpha {
                            F::from(a).unwrap_or(F::one())
                        } else {
                            F::one()
                        };
                        // Term 1: -alpha * gamma * (1-p_t)^(gamma-1) * log(p_t)
                        let term1 = -alpha
                            * gamma
                            * (F::one() - p_safe).powf(gamma - F::one())
                            * p_safe.ln();
                        // Term 2: -alpha * (1-p_t)^gamma * (1/p_t)
                        let term2 = -alpha * (F::one() - p_safe).powf(gamma) * (F::one() / p_safe);
                        gradients[[i, j]] = (term1 + term2) * t / n;
                    } else {
                        gradients[[i, j]] = F::zero();
                    }
                }
            }
        }
        Ok(gradients)
    }
}
