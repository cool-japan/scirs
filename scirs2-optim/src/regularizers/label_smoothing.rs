//! Label Smoothing regularization
//!
//! Label smoothing is a regularization technique that prevents the model from
//! becoming over-confident by replacing hard one-hot encoded targets with
//! soft targets that include some probability for incorrect classes.

use ndarray::{Array, Array1, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::regularizers::Regularizer;

/// Label Smoothing regularization
///
/// Implements label smoothing by replacing one-hot encoded target vectors with
/// "smoother" target distributions, where some probability mass is assigned to
/// non-target classes.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_optim::regularizers::LabelSmoothing;
///
/// let label_smooth = LabelSmoothing::new(0.1, 3).unwrap();
/// let one_hot_target = array![0.0, 1.0, 0.0];
///
/// // Apply label smoothing to one-hot targets
/// let smoothed_target = label_smooth.smooth_labels(&one_hot_target).unwrap();
/// // Result will be [0.033..., 0.933..., 0.033...]
/// ```
#[derive(Debug, Clone)]
pub struct LabelSmoothing<A: Float> {
    /// Smoothing factor (between 0 and 1)
    alpha: A,
    /// Number of classes
    num_classes: usize,
}

impl<A: Float + Debug + ScalarOperand + FromPrimitive> LabelSmoothing<A> {
    /// Create a new label smoothing regularizer
    ///
    /// # Arguments
    ///
    /// * `alpha` - Smoothing factor, where 0 gives one-hot encoding and 1 gives uniform distribution
    /// * `num_classes` - Number of classes in the classification task
    ///
    /// # Errors
    ///
    /// Returns an error if alpha is not between 0 and 1
    pub fn new(alpha: A, num_classes: usize) -> Result<Self> {
        if alpha < A::zero() || alpha > A::one() {
            return Err(OptimError::InvalidConfig(
                "Alpha must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self { alpha, num_classes })
    }

    /// Smooth the one-hot encoded target labels
    ///
    /// # Arguments
    ///
    /// * `labels` - One-hot encoded target labels
    ///
    /// # Returns
    ///
    /// The smoothed labels
    ///
    /// # Example
    ///
    /// For a 3-class problem with smoothing factor 0.1:
    /// [0, 1, 0] -> [0.033..., 0.933..., 0.033...]
    pub fn smooth_labels(&self, labels: &Array1<A>) -> Result<Array1<A>> {
        if labels.len() != self.num_classes {
            return Err(OptimError::InvalidConfig(format!(
                "Expected {} classes, got {} in label vector",
                self.num_classes,
                labels.len()
            )));
        }

        let uniform_val = A::one() / A::from_usize(self.num_classes).unwrap();
        let smooth_coef = self.alpha;
        let one_minus_alpha = A::one() - smooth_coef;

        // Compute (1 - alpha) * y + alpha * uniform
        let smoothed = labels.map(|&y| one_minus_alpha * y + smooth_coef * uniform_val);

        Ok(smoothed)
    }

    /// Apply label smoothing to a batch of one-hot encoded targets
    ///
    /// # Arguments
    ///
    /// * `labels` - Batch of one-hot encoded target labels
    ///
    /// # Returns
    ///
    /// The smoothed labels for the batch
    pub fn smooth_batch<D>(&self, labels: &Array<A, D>) -> Result<Array<A, D>>
    where
        D: Dimension,
    {
        // Ensure the last dimension is the class dimension
        if labels.shape().last().unwrap_or(&0) != &self.num_classes {
            return Err(OptimError::InvalidConfig(
                "Last dimension must match number of classes".to_string(),
            ));
        }

        // Apply smoothing to each label vector
        let uniform_val = A::one() / A::from_usize(self.num_classes).unwrap();
        let smooth_coef = self.alpha;
        let one_minus_alpha = A::one() - smooth_coef;

        // Compute (1 - alpha) * y + alpha * uniform for each element
        let smoothed = labels.map(|&y| one_minus_alpha * y + smooth_coef * uniform_val);

        Ok(smoothed)
    }

    /// Compute cross-entropy loss with label smoothing
    ///
    /// # Arguments
    ///
    /// * `logits` - Raw model outputs (unnormalized)
    /// * `labels` - One-hot encoded target labels
    /// * `eps` - Small value for numerical stability
    ///
    /// # Returns
    ///
    /// The smoothed cross-entropy loss
    pub fn cross_entropy_loss(&self, logits: &Array1<A>, labels: &Array1<A>, eps: A) -> Result<A> {
        if logits.len() != self.num_classes || labels.len() != self.num_classes {
            return Err(OptimError::InvalidConfig(
                "Logits and labels must match number of classes".to_string(),
            ));
        }

        // Compute softmax probabilities
        let max_logit = logits.fold(A::neg_infinity(), |max, &v| if v > max { v } else { max });
        let exp_logits = logits.map(|&l| (l - max_logit).exp());
        let sum_exp = exp_logits.sum();
        let probs = exp_logits.map(|&e| e / (sum_exp + eps));

        // Smooth the labels
        let smoothed_labels = self.smooth_labels(labels)?;

        // Compute cross-entropy with smoothed labels
        let mut loss = A::zero();
        for (p, y) in probs.iter().zip(smoothed_labels.iter()) {
            loss = loss - *y * (*p + eps).ln();
        }

        Ok(loss)
    }
}

// Implement Regularizer trait (though it's not the primary interface for label smoothing)
impl<A: Float + Debug + ScalarOperand + FromPrimitive, D: Dimension> Regularizer<A, D>
    for LabelSmoothing<A>
{
    fn apply(&self, _params: &Array<A, D>, _gradients: &mut Array<A, D>) -> Result<A> {
        // Label smoothing is not applied to model parameters directly
        // It's applied to the target labels during loss computation
        Ok(A::zero())
    }

    fn penalty(&self, _params: &Array<A, D>) -> Result<A> {
        // Label smoothing doesn't add a parameter penalty term
        Ok(A::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_label_smoothing_creation() {
        let ls = LabelSmoothing::<f64>::new(0.1, 3).unwrap();
        assert_eq!(ls.alpha, 0.1);
        assert_eq!(ls.num_classes, 3);

        // Alpha out of range should fail
        assert!(LabelSmoothing::<f64>::new(-0.1, 3).is_err());
        assert!(LabelSmoothing::<f64>::new(1.1, 3).is_err());
    }

    #[test]
    fn test_smooth_labels() {
        let ls = LabelSmoothing::new(0.1, 3).unwrap();
        let one_hot = array![0.0, 1.0, 0.0];

        let smoothed = ls.smooth_labels(&one_hot).unwrap();

        // Expected: [0.033..., 0.933..., 0.033...]
        let uniform_val = 1.0 / 3.0;
        let expected_1 = 0.9 * 1.0 + 0.1 * uniform_val;
        let expected_0 = 0.9 * 0.0 + 0.1 * uniform_val;

        assert_relative_eq!(smoothed[0], expected_0, epsilon = 1e-5);
        assert_relative_eq!(smoothed[1], expected_1, epsilon = 1e-5);
        assert_relative_eq!(smoothed[2], expected_0, epsilon = 1e-5);

        // Sum should still be 1
        assert_relative_eq!(smoothed.sum(), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_full_smoothing() {
        let ls = LabelSmoothing::new(1.0, 4).unwrap();
        let one_hot = array![0.0, 0.0, 1.0, 0.0];

        let smoothed = ls.smooth_labels(&one_hot).unwrap();

        // With alpha=1, should be uniform distribution [0.25, 0.25, 0.25, 0.25]
        for i in 0..4 {
            assert_relative_eq!(smoothed[i], 0.25, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_no_smoothing() {
        let ls = LabelSmoothing::new(0.0, 3).unwrap();
        let one_hot = array![0.0, 1.0, 0.0];

        let smoothed = ls.smooth_labels(&one_hot).unwrap();

        // With alpha=0, should be identical to input
        for i in 0..3 {
            assert_relative_eq!(smoothed[i], one_hot[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_smooth_batch() {
        let ls = LabelSmoothing::new(0.2, 2).unwrap();
        let batch = array![[1.0, 0.0], [0.0, 1.0]];

        let smoothed = ls.smooth_batch(&batch).unwrap();

        // With alpha=0.2 and 2 classes, uniform_val = 0.5
        // For label 1.0: (1 - 0.2) * 1.0 + 0.2 * 0.5 = 0.8 + 0.1 = 0.9
        // For label 0.0: (1 - 0.2) * 0.0 + 0.2 * 0.5 = 0.0 + 0.1 = 0.1
        assert_relative_eq!(smoothed[[0, 0]], 0.9, epsilon = 1e-5);
        assert_relative_eq!(smoothed[[0, 1]], 0.1, epsilon = 1e-5);
        assert_relative_eq!(smoothed[[1, 0]], 0.1, epsilon = 1e-5);
        assert_relative_eq!(smoothed[[1, 1]], 0.9, epsilon = 1e-5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let ls = LabelSmoothing::new(0.1, 3).unwrap();
        let labels = array![0.0, 1.0, 0.0];
        let logits = array![1.0, 2.0, 0.5];

        let loss = ls.cross_entropy_loss(&logits, &labels, 1e-8).unwrap();

        // Loss should be positive and finite
        assert!(loss > 0.0 && loss.is_finite());
    }

    #[test]
    fn test_regularizer_trait() {
        let ls = LabelSmoothing::new(0.1, 3).unwrap();
        let params = array![[1.0, 2.0], [3.0, 4.0]];
        let mut gradients = array![[0.1, 0.2], [0.3, 0.4]];
        let original_gradients = gradients.clone();

        let penalty = ls.apply(&params, &mut gradients).unwrap();

        // Penalty should be zero
        assert_eq!(penalty, 0.0);

        // Gradients should be unchanged
        assert_eq!(gradients, original_gradients);
    }
}
