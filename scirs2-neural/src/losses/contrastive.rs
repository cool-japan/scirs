//! Contrastive loss function implementation

use crate::error::{NeuralError, Result};
use crate::losses::Loss;
use ndarray::Array;
use num_traits::Float;
use std::fmt::Debug;

/// Contrastive loss function.
///
/// The contrastive loss is designed to learn embeddings where similar pairs are closer
/// together and dissimilar pairs are further apart in the embedding space.
///
/// For a pair of samples (xi, xj) with label y (1 for similar, 0 for dissimilar),
/// the contrastive loss is defined as:
///
/// L(xi, xj, y) = y * d(xi, xj)^2 + (1 - y) * max(0, margin - d(xi, xj))^2
///
/// where d(xi, xj) is the Euclidean distance between the embeddings.
///
/// # Examples
///
/// ```
/// use scirs2_neural::losses::ContrastiveLoss;
/// use scirs2_neural::losses::Loss;
/// use ndarray::{Array, arr2};
///
/// // Create contrastive loss with margin=1.0
/// let contrastive = ContrastiveLoss::new(1.0);
///
/// // Embedding pairs (batch_size x 2 x embedding_dim)
/// let embeddings = arr2(&[
///     [0.1, 0.2, 0.3],  // First pair, first embedding
///     [0.1, 0.3, 0.3],  // First pair, second embedding (similar)
///     [0.5, 0.5, 0.5],  // Second pair, first embedding
///     [0.9, 0.8, 0.7],  // Second pair, second embedding (dissimilar)
/// ]).into_shape((2, 2, 3)).unwrap().into_dyn();
///
/// // Labels: 1 for similar pairs, 0 for dissimilar
/// let labels = arr2(&[
///     [1.0],  // First pair is similar
///     [0.0],  // Second pair is dissimilar
/// ]).into_dyn();
///
/// // Forward pass to calculate loss
/// let loss = contrastive.forward(&embeddings, &labels).unwrap();
///
/// // Backward pass to calculate gradients
/// let gradients = contrastive.backward(&embeddings, &labels).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ContrastiveLoss {
    /// Margin for dissimilar pairs
    margin: f64,
}

impl ContrastiveLoss {
    /// Create a new contrastive loss function
    ///
    /// # Arguments
    ///
    /// * `margin` - Minimum distance margin for dissimilar pairs
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }
}

impl Default for ContrastiveLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<F: Float + Debug> Loss<F> for ContrastiveLoss {
    fn forward(
        &self,
        predictions: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
    ) -> Result<F> {
        // Verify predictions shape: should be (batch_size, 2, embedding_dim)
        if predictions.ndim() != 3 || predictions.shape()[1] != 2 {
            return Err(NeuralError::InferenceError(format!(
                "Expected predictions shape (batch_size, 2, embedding_dim), got {:?}",
                predictions.shape()
            )));
        }

        // Verify targets shape: should be (batch_size, 1)
        if targets.ndim() != 2 || targets.shape()[1] != 1 {
            return Err(NeuralError::InferenceError(format!(
                "Expected targets shape (batch_size, 1), got {:?}",
                targets.shape()
            )));
        }

        // Verify batch sizes match
        if predictions.shape()[0] != targets.shape()[0] {
            return Err(NeuralError::InferenceError(format!(
                "Batch size mismatch: predictions {} vs targets {}",
                predictions.shape()[0],
                targets.shape()[0]
            )));
        }

        let batch_size = predictions.shape()[0];
        let embedding_dim = predictions.shape()[2];
        let margin = F::from(self.margin).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert margin to float".to_string())
        })?;

        let mut total_loss = F::zero();
        let n = F::from(batch_size).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert batch size to float".to_string())
        })?;

        for i in 0..batch_size {
            // Extract pair of embeddings
            let x1 = predictions.slice(ndarray::s![i, 0, ..]);
            let x2 = predictions.slice(ndarray::s![i, 1, ..]);

            // Compute Euclidean distance
            let mut distance_squared = F::zero();
            for j in 0..embedding_dim {
                let diff = x1[j] - x2[j];
                distance_squared = distance_squared + diff * diff;
            }
            let distance = distance_squared.sqrt();

            // Extract label (1 for similar, 0 for dissimilar)
            let y = targets[[i, 0]];

            // Calculate loss for this pair
            let pair_loss = if y > F::zero() {
                // Similar pair: loss = d^2
                distance_squared
            } else {
                // Dissimilar pair: loss = max(0, margin - d)^2
                let zero = F::zero();
                let margin_term = (margin - distance).max(zero);
                margin_term * margin_term
            };

            total_loss = total_loss + pair_loss;
        }

        // Average loss over the batch
        let loss = total_loss / n;
        Ok(loss)
    }

    fn backward(
        &self,
        predictions: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        // Verify predictions shape: should be (batch_size, 2, embedding_dim)
        if predictions.ndim() != 3 || predictions.shape()[1] != 2 {
            return Err(NeuralError::InferenceError(format!(
                "Expected predictions shape (batch_size, 2, embedding_dim), got {:?}",
                predictions.shape()
            )));
        }

        // Verify targets shape: should be (batch_size, 1)
        if targets.ndim() != 2 || targets.shape()[1] != 1 {
            return Err(NeuralError::InferenceError(format!(
                "Expected targets shape (batch_size, 1), got {:?}",
                targets.shape()
            )));
        }

        // Verify batch sizes match
        if predictions.shape()[0] != targets.shape()[0] {
            return Err(NeuralError::InferenceError(format!(
                "Batch size mismatch: predictions {} vs targets {}",
                predictions.shape()[0],
                targets.shape()[0]
            )));
        }

        let batch_size = predictions.shape()[0];
        let embedding_dim = predictions.shape()[2];
        let margin = F::from(self.margin).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert margin to float".to_string())
        })?;

        let n = F::from(batch_size).ok_or_else(|| {
            NeuralError::InferenceError("Could not convert batch size to float".to_string())
        })?;

        // Initialize gradients with zeros
        let mut gradients = Array::zeros(predictions.raw_dim());

        for i in 0..batch_size {
            // Extract pair of embeddings
            let x1 = predictions.slice(ndarray::s![i, 0, ..]);
            let x2 = predictions.slice(ndarray::s![i, 1, ..]);

            // Compute Euclidean distance
            let mut distance_squared = F::zero();
            for j in 0..embedding_dim {
                let diff = x1[j] - x2[j];
                distance_squared = distance_squared + diff * diff;
            }
            let distance = distance_squared.sqrt();

            // To avoid division by zero
            let distance_safe = distance.max(F::from(1e-10).unwrap());

            // Extract label (1 for similar, 0 for dissimilar)
            let y = targets[[i, 0]];

            // Calculate gradients for this pair
            if y > F::zero() {
                // Similar pair: d(loss)/d(x1) = 2 * (x1 - x2) / n
                // Similar pair: d(loss)/d(x2) = 2 * (x2 - x1) / n
                for j in 0..embedding_dim {
                    gradients[[i, 0, j]] = F::from(2.0).unwrap() * (x1[j] - x2[j]) / n;
                    gradients[[i, 1, j]] = F::from(2.0).unwrap() * (x2[j] - x1[j]) / n;
                }
            } else {
                // Dissimilar pair: only contribute to gradient if within margin
                if distance < margin {
                    // d(loss)/d(x1) = -2 * (margin - d) * (x1 - x2) / d / n
                    // d(loss)/d(x2) = -2 * (margin - d) * (x2 - x1) / d / n
                    let factor = F::from(-2.0).unwrap() * (margin - distance) / distance_safe / n;
                    for j in 0..embedding_dim {
                        gradients[[i, 0, j]] = factor * (x1[j] - x2[j]);
                        gradients[[i, 1, j]] = factor * (x2[j] - x1[j]);
                    }
                }
            }
        }

        Ok(gradients)
    }
}
