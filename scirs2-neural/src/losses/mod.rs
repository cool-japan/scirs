//! Loss functions for neural networks
//!
//! This module provides common loss functions used in neural networks for training.
//! Loss functions measure how well the model's predictions match the target values,
//! providing a signal for optimization during training.
//!
//! # Overview
//!
//! Loss functions serve two main purposes in neural networks:
//! 1. **Forward pass**: Calculate the scalar loss value between predictions and targets
//! 2. **Backward pass**: Calculate gradients needed for backpropagation
//!
//! The choice of loss function depends on your task type and requirements.
//!
//! # Available Loss Functions
//!
//! - **MeanSquaredError**: For regression tasks, measures squared differences
//! - **CrossEntropyLoss**: For classification tasks, combines softmax and negative log-likelihood
//! - **FocalLoss**: For imbalanced classification, focuses on hard examples
//! - **ContrastiveLoss**: For learning embeddings, brings similar items closer
//! - **TripletLoss**: For metric learning, enforces relative distances
//!
//! # Examples
//!
//! ## Regression with Mean Squared Error
//!
//! ```rust
//! use scirs2_neural::losses::{Loss, MeanSquaredError};
//! use ndarray::Array;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! let mse = MeanSquaredError::new();
//!
//! // Predictions and targets for regression
//! let predictions = Array::from_vec(vec![1.5, 2.3, 0.8]).into_dyn();
//! let targets = Array::from_vec(vec![1.0, 2.0, 1.0]).into_dyn();
//!
//! // Calculate loss
//! let loss_value = mse.forward(&predictions, &targets)?;
//! println!("MSE Loss: {:.4}", loss_value);
//!
//! // Calculate gradients for backpropagation
//! let gradients = mse.backward(&predictions, &targets)?;
//! println!("Gradients: {:?}", gradients);
//! # Ok(())
//! # }
//! ```
//!
//! ## Classification with Cross Entropy
//!
//! ```rust
//! use scirs2_neural::losses::{Loss, CrossEntropyLoss};
//! use ndarray::Array;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! let ce_loss = CrossEntropyLoss::default();
//!
//! // Logits (raw predictions) for 3-class classification
//! let logits = Array::from_shape_vec((2, 3), vec![
//!     1.0, 2.0, 0.5,  // Sample 1
//!     0.8, 0.2, 1.5,  // Sample 2
//! ])?.into_dyn();
//!
//! // One-hot encoded targets
//! let targets = Array::from_shape_vec((2, 3), vec![
//!     0.0, 1.0, 0.0,  // Sample 1: class 1
//!     0.0, 0.0, 1.0,  // Sample 2: class 2
//! ])?.into_dyn();
//!
//! // Calculate cross-entropy loss
//! let loss_value = ce_loss.forward(&logits, &targets)?;
//! println!("Cross-Entropy Loss: {:.4}", loss_value);
//!
//! // Get gradients for backpropagation
//! let gradients = ce_loss.backward(&logits, &targets)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Handling Class Imbalance with Focal Loss
//!
//! ```rust
//! use scirs2_neural::losses::{Loss, FocalLoss};
//! use ndarray::Array;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! // Focal loss with gamma=2.0, alpha=0.25 (typical values)
//! let focal_loss = FocalLoss::new(2.0, Some(0.25), 1e-10);
//!
//! let predictions = Array::from_shape_vec((2, 2), vec![
//!     0.9, 0.1,  // High confidence prediction
//!     0.6, 0.4,  // Lower confidence prediction
//! ])?.into_dyn();
//!
//! let targets = Array::from_shape_vec((2, 2), vec![
//!     1.0, 0.0,  // Correct prediction
//!     1.0, 0.0,  // Incorrect prediction (will get higher loss)
//! ])?.into_dyn();
//!
//! let loss_value = focal_loss.forward(&predictions, &targets)?;
//! // Focal loss will be higher for the incorrect, hard example
//! println!("Focal Loss: {:.4}", loss_value);
//! # Ok(())
//! # }
//! ```
//!
//! ## Metric Learning with Triplet Loss
//!
//! ```rust
//! use scirs2_neural::losses::{Loss, TripletLoss};
//! use ndarray::Array;
//!
//! # fn example() -> scirs2_neural::error::Result<()> {
//! let triplet_loss = TripletLoss::new(1.0); // margin = 1.0
//!
//! // Embeddings: [anchor, positive, negative]
//! let anchor = Array::from_vec(vec![1.0, 0.0, 0.0]).into_dyn();
//! let positive = Array::from_vec(vec![0.9, 0.1, 0.0]).into_dyn(); // Similar to anchor
//! let negative = Array::from_vec(vec![0.0, 0.0, 1.0]).into_dyn(); // Different from anchor
//!
//! // Stack into batch format
//! let embeddings = Array::from_shape_vec((3, 3), vec![
//!     1.0, 0.0, 0.0,  // anchor
//!     0.9, 0.1, 0.0,  // positive
//!     0.0, 0.0, 1.0,  // negative
//! ])?.into_dyn();
//!
//! // Dummy targets (not used in triplet loss)
//! let targets = Array::zeros((3,)).into_dyn();
//!
//! let loss_value = triplet_loss.forward(&embeddings, &targets)?;
//! println!("Triplet Loss: {:.4}", loss_value);
//! # Ok(())
//! # }
//! ```
//!
//! # Choosing the Right Loss Function
//!
//! ## For Regression:
//! - **MeanSquaredError**: Most common, penalizes large errors heavily
//! - **Mean Absolute Error**: More robust to outliers (implement if needed)
//! - **Huber Loss**: Combines benefits of MSE and MAE (implement if needed)
//!
//! ## For Classification:
//! - **CrossEntropyLoss**: Standard choice for balanced multi-class problems
//! - **FocalLoss**: Better for imbalanced datasets, focuses on hard examples
//! - **Binary Cross-Entropy**: For binary classification (implement if needed)
//!
//! ## For Similarity Learning:
//! - **ContrastiveLoss**: For learning embeddings with positive/negative pairs
//! - **TripletLoss**: For learning relative similarities with triplets
//!
//! # Training Tips
//!
//! 1. **Scale your loss**: If loss values are too large/small, consider scaling
//! 2. **Monitor gradients**: Use backward() to check gradient magnitudes
//! 3. **Combine losses**: You can combine multiple loss functions for complex objectives
//! 4. **Loss scheduling**: Consider changing loss function or parameters during training

use crate::error::Result;
use ndarray::Array;
use num_traits::Float;
use std::fmt::Debug;

/// Trait for loss functions used in neural networks
///
/// Loss functions compute the error between model predictions and target values,
/// providing both the scalar loss value and gradients needed for training.
///
/// # Examples
///
/// ```rust
/// use scirs2_neural::losses::{Loss, MeanSquaredError};
/// use ndarray::Array;
///
/// # fn example() -> scirs2_neural::error::Result<()> {
/// let loss_fn = MeanSquaredError::new();
///
/// let predictions = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
/// let targets = Array::from_vec(vec![1.1, 1.9, 3.2]).into_dyn();
///
/// // Forward pass: compute loss value
/// let loss_value = loss_fn.forward(&predictions, &targets)?;
/// assert!(loss_value >= 0.0); // Loss is non-negative
///
/// // Backward pass: compute gradients
/// let gradients = loss_fn.backward(&predictions, &targets)?;
/// assert_eq!(gradients.shape(), predictions.shape());
/// # Ok(())
/// # }
/// ```
pub trait Loss<F: Float + Debug> {
    /// Calculate the loss between predictions and targets
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (can be logits or probabilities depending on loss)
    /// * `targets` - Ground truth target values
    ///
    /// # Returns
    /// A scalar loss value (typically averaged over the batch)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::losses::{Loss, MeanSquaredError};
    /// use ndarray::Array;
    ///
    /// # fn example() -> scirs2_neural::error::Result<()> {
    /// let mse = MeanSquaredError::new();
    /// let predictions = Array::from_vec(vec![1.0, 2.0]).into_dyn();
    /// let targets = Array::from_vec(vec![1.5, 1.8]).into_dyn();
    ///
    /// let loss = mse.forward(&predictions, &targets)?;
    /// // MSE = mean((predictions - targets)^2) = mean([0.25, 0.04]) = 0.145
    /// assert!((loss - 0.145f64).abs() < 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    fn forward(
        &self,
        predictions: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
    ) -> Result<F>;

    /// Calculate the gradient of the loss with respect to the predictions
    ///
    /// This method computes ∂Loss/∂predictions, which is used in backpropagation
    /// to update the model parameters.
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (same as used in forward pass)
    /// * `targets` - Ground truth target values (same as used in forward pass)
    ///
    /// # Returns
    /// Gradient tensor with the same shape as predictions
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::losses::{Loss, MeanSquaredError};
    /// use ndarray::Array;
    ///
    /// # fn example() -> scirs2_neural::error::Result<()> {
    /// let mse = MeanSquaredError::new();
    /// let predictions = Array::from_vec(vec![1.0, 2.0]).into_dyn();
    /// let targets = Array::from_vec(vec![0.0, 1.0]).into_dyn();
    ///
    /// let gradients = mse.backward(&predictions, &targets)?;
    /// // MSE gradient = 2 * (predictions - targets) / n
    /// // For our example: [2*(1-0)/2, 2*(2-1)/2] = [1.0, 1.0]
    /// assert_eq!(gradients.as_slice().unwrap(), &[1.0, 1.0]);
    /// # Ok(())
    /// # }
    /// ```
    fn backward(
        &self,
        predictions: &Array<F, ndarray::IxDyn>,
        targets: &Array<F, ndarray::IxDyn>,
    ) -> Result<Array<F, ndarray::IxDyn>>;
}

mod contrastive;
mod crossentropy;
mod focal;
mod mse;
mod triplet;

pub use contrastive::ContrastiveLoss;
pub use crossentropy::CrossEntropyLoss;
pub use focal::FocalLoss;
pub use mse::MeanSquaredError;
pub use triplet::TripletLoss;
