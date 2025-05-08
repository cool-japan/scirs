//! Triplet loss function implementation

use crate::error::{NeuralError, Result};
use crate::losses::Loss;
use ndarray::Array;
use num_traits::Float;
use std::fmt::Debug;

/// Triplet loss function.
///
/// The triplet loss is used for learning embeddings where the distance between
/// an anchor and a positive sample (same class) is smaller than the distance
/// between the anchor and a negative sample (different class) by at least a margin.
///
/// For a triplet (a, p, n) of anchor, positive, and negative samples,
/// the triplet loss is defined as:
///
/// L(a, p, n) = max(0, d(a, p) - d(a, n) + margin)
///
/// where d(x, y) is the distance between embeddings x and y (typically Euclidean).
///
/// # Examples
///
/// ```
/// use scirs2_neural::losses::TripletLoss;
/// use scirs2_neural::losses::Loss;
/// use ndarray::{Array, arr2};
///
/// // Create triplet loss with margin=0.5
/// let triplet = TripletLoss::new(0.5);
///
/// // Embedding triplets (batch_size x 3 x embedding_dim)
/// let embeddings = arr2(&[
///     [0.1, 0.2, 0.3],  // First triplet, anchor
///     [0.1, 0.3, 0.3],  // First triplet, positive
///     [0.5, 0.5, 0.5],  // First triplet, negative
///     [0.6, 0.6, 0.6],  // Second triplet, anchor
///     [0.5, 0.6, 0.6],  // Second triplet, positive
///     [0.1, 0.1, 0.1],  // Second triplet, negative
/// ]).into_shape((2, 3, 3)).unwrap().into_dyn();
///
/// // No labels needed for triplet loss calculation
/// let dummy_labels = Array::zeros(ndarray::IxDyn(&[2, 1]));
///
/// // Forward pass to calculate loss
/// let loss = triplet.forward(&embeddings, &dummy_labels).unwrap();
///
/// // Backward pass to calculate gradients
/// let gradients = triplet.backward(&embeddings, &dummy_labels).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TripletLoss {
    /// Margin between positive and negative distances
    margin: f64,
}

impl TripletLoss {
    /// Create a new triplet loss function
    ///
    /// # Arguments
    ///
    /// * `margin` - Margin between positive and negative distances
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }
}

impl Default for TripletLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<F: Float + Debug> Loss<F> for TripletLoss {
    fn forward(
        &self,
        predictions: &Array<F, ndarray::IxDyn>,
        _targets: &Array<F, ndarray::IxDyn>,
    ) -> Result<F> {
        // Verify predictions shape: should be (batch_size, 3, embedding_dim)
        // Where the triplets are in the order: anchor, positive, negative
        if predictions.ndim() != 3 || predictions.shape()[1] != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected predictions shape (batch_size, 3, embedding_dim), got {:?}",
                predictions.shape()
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
            // Extract triplet of embeddings
            let anchor = predictions.slice(ndarray::s![i, 0, ..]);
            let positive = predictions.slice(ndarray::s![i, 1, ..]);
            let negative = predictions.slice(ndarray::s![i, 2, ..]);

            // Compute distances
            let mut pos_distance_squared = F::zero();
            let mut neg_distance_squared = F::zero();

            for j in 0..embedding_dim {
                // Anchor-positive distance
                let pos_diff = anchor[j] - positive[j];
                pos_distance_squared = pos_distance_squared + pos_diff * pos_diff;

                // Anchor-negative distance
                let neg_diff = anchor[j] - negative[j];
                neg_distance_squared = neg_distance_squared + neg_diff * neg_diff;
            }

            let pos_distance = pos_distance_squared.sqrt();
            let neg_distance = neg_distance_squared.sqrt();

            // Calculate loss for this triplet
            // max(0, pos_distance - neg_distance + margin)
            let zero = F::zero();
            let triplet_loss = (pos_distance - neg_distance + margin).max(zero);

            total_loss = total_loss + triplet_loss;
        }

        // Average loss over the batch
        let loss = total_loss / n;
        Ok(loss)
    }

    fn backward(
        &self,
        predictions: &Array<F, ndarray::IxDyn>,
        _targets: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>> {
        // Verify predictions shape: should be (batch_size, 3, embedding_dim)
        if predictions.ndim() != 3 || predictions.shape()[1] != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected predictions shape (batch_size, 3, embedding_dim), got {:?}",
                predictions.shape()
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
            // Extract triplet of embeddings
            let anchor = predictions.slice(ndarray::s![i, 0, ..]);
            let positive = predictions.slice(ndarray::s![i, 1, ..]);
            let negative = predictions.slice(ndarray::s![i, 2, ..]);

            // Compute distances
            let mut pos_distance_squared = F::zero();
            let mut neg_distance_squared = F::zero();

            for j in 0..embedding_dim {
                // Anchor-positive distance
                let pos_diff = anchor[j] - positive[j];
                pos_distance_squared = pos_distance_squared + pos_diff * pos_diff;

                // Anchor-negative distance
                let neg_diff = anchor[j] - negative[j];
                neg_distance_squared = neg_distance_squared + neg_diff * neg_diff;
            }

            let pos_distance = pos_distance_squared.sqrt();
            let neg_distance = neg_distance_squared.sqrt();

            // To avoid division by zero
            let pos_distance_safe = pos_distance.max(F::from(1e-10).unwrap());
            let neg_distance_safe = neg_distance.max(F::from(1e-10).unwrap());

            // Only compute gradients if the triplet is active (loss > 0)
            if pos_distance - neg_distance + margin > F::zero() {
                // Gradients for anchor
                for j in 0..embedding_dim {
                    // Positive direction: push away from positive
                    let pos_grad = (anchor[j] - positive[j]) / pos_distance_safe;
                    // Negative direction: pull towards negative
                    let neg_grad = (anchor[j] - negative[j]) / neg_distance_safe;

                    // Combined gradient for anchor
                    gradients[[i, 0, j]] = (pos_grad - neg_grad) / n;
                }

                // Gradients for positive: pull towards anchor
                for j in 0..embedding_dim {
                    gradients[[i, 1, j]] =
                        -F::one() * (anchor[j] - positive[j]) / pos_distance_safe / n;
                }

                // Gradients for negative: push away from anchor
                for j in 0..embedding_dim {
                    gradients[[i, 2, j]] = (anchor[j] - negative[j]) / neg_distance_safe / n;
                }
            }
        }

        Ok(gradients)
    }
}
