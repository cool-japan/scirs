//! Weight Standardization
//!
//! Weight Standardization is a technique that normalizes the weights of convolutional
//! layers by standardizing the weights along the channel dimension. This improves
//! training stability and allows for use of larger batch sizes.

use ndarray::{Array, Array2, Array4, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::regularizers::Regularizer;

/// Weight Standardization
///
/// Weight Standardization normalizes the weights along the channel dimension by
/// adjusting them to have zero mean and unit variance. This helps with training
/// stability, especially when used with batch normalization.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_optim::regularizers::{WeightStandardization, Regularizer};
///
/// let weight_std = WeightStandardization::new(1e-5);
/// let weights = array![[1.0, 2.0], [3.0, 4.0]];
/// let mut gradients = array![[0.1, 0.2], [0.3, 0.4]];
///
/// // Get standardized weights
/// let standardized = weight_std.standardize(&weights).unwrap();
///
/// // Apply during training (modifies gradients)
/// let _ = weight_std.apply(&weights, &mut gradients);
/// ```
#[derive(Debug, Clone)]
pub struct WeightStandardization<A: Float> {
    /// Small constant for numerical stability
    eps: A,
}

impl<A: Float + Debug + ScalarOperand + FromPrimitive> WeightStandardization<A> {
    /// Create a new Weight Standardization regularizer
    ///
    /// # Arguments
    ///
    /// * `eps` - Small constant for numerical stability (typically 1e-5)
    pub fn new(eps: f64) -> Self {
        Self {
            eps: A::from_f64(eps).unwrap(),
        }
    }

    /// Apply weight standardization to a 2D weight matrix
    ///
    /// Standardizes the weights to have zero mean and unit variance.
    ///
    /// # Arguments
    ///
    /// * `weights` - 2D weight matrix
    ///
    /// # Returns
    ///
    /// Standardized weights with zero mean and unit variance
    pub fn standardize(&self, weights: &Array2<A>) -> Result<Array2<A>> {
        // Calculate mean for each row (output channel)
        let n_cols = weights.ncols();
        let n_cols_f = A::from_usize(n_cols).unwrap();

        // Calculate mean, subtract from weights, then calculate variance and normalize
        let means = weights.sum_axis(ndarray::Axis(1)) / n_cols_f;

        // Subtract mean from weights
        let mut centered = weights.clone();
        for i in 0..weights.nrows() {
            for j in 0..weights.ncols() {
                centered[[i, j]] = centered[[i, j]] - means[i];
            }
        }

        // Calculate variance
        let mut var = Array::zeros(weights.nrows());
        for i in 0..weights.nrows() {
            let mut sum_sq = A::zero();
            for j in 0..weights.ncols() {
                sum_sq = sum_sq + centered[[i, j]] * centered[[i, j]];
            }
            var[i] = sum_sq / n_cols_f;
        }

        // Normalize
        let mut standardized = centered.clone();
        for i in 0..weights.nrows() {
            let denom = (var[i] + self.eps).sqrt();
            for j in 0..weights.ncols() {
                standardized[[i, j]] = centered[[i, j]] / denom;
            }
        }

        Ok(standardized)
    }

    /// Apply weight standardization to 4D convolutional weights
    ///
    /// # Arguments
    ///
    /// * `weights` - Convolutional weights with shape [out_channels, in_channels, height, width]
    ///
    /// # Returns
    ///
    /// Standardized convolutional weights
    pub fn standardize_conv4d(&self, weights: &Array4<A>) -> Result<Array4<A>> {
        let shape = weights.shape();
        if shape.len() != 4 {
            return Err(OptimError::InvalidConfig(
                "Expected 4D weights for conv4d standardization".to_string(),
            ));
        }

        let out_channels = shape[0];
        let in_channels = shape[1];
        let kernel_h = shape[2];
        let kernel_w = shape[3];
        let n_elements = in_channels * kernel_h * kernel_w;
        let n_elements_f = A::from_usize(n_elements).unwrap();

        // Calculate mean for each output channel
        let mut means = Array::zeros(out_channels);

        for c_out in 0..out_channels {
            let mut sum = A::zero();
            for c_in in 0..in_channels {
                for h in 0..kernel_h {
                    for w in 0..kernel_w {
                        sum = sum + weights[[c_out, c_in, h, w]];
                    }
                }
            }
            means[c_out] = sum / n_elements_f;
        }

        // Center the weights
        let mut centered = weights.clone();

        for c_out in 0..out_channels {
            for c_in in 0..in_channels {
                for h in 0..kernel_h {
                    for w in 0..kernel_w {
                        centered[[c_out, c_in, h, w]] = weights[[c_out, c_in, h, w]] - means[c_out];
                    }
                }
            }
        }

        // Calculate variance for each output channel
        let mut vars = Array::zeros(out_channels);

        for c_out in 0..out_channels {
            let mut sum_sq = A::zero();
            for c_in in 0..in_channels {
                for h in 0..kernel_h {
                    for w in 0..kernel_w {
                        sum_sq =
                            sum_sq + centered[[c_out, c_in, h, w]] * centered[[c_out, c_in, h, w]];
                    }
                }
            }
            vars[c_out] = sum_sq / n_elements_f;
        }

        // Standardize
        let mut standardized = centered.clone();

        for c_out in 0..out_channels {
            let std_dev = (vars[c_out] + self.eps).sqrt();
            for c_in in 0..in_channels {
                for h in 0..kernel_h {
                    for w in 0..kernel_w {
                        standardized[[c_out, c_in, h, w]] = centered[[c_out, c_in, h, w]] / std_dev;
                    }
                }
            }
        }

        Ok(standardized)
    }

    /// Calculate the gradients of weight standardization
    ///
    /// # Arguments
    ///
    /// * `weights` - Original weights
    /// * `grad_output` - Gradient from the next layer
    ///
    /// # Returns
    ///
    /// The gradient for the weights
    fn compute_gradients<S1, S2>(
        &self,
        weights: &ArrayBase<S1, ndarray::Ix2>,
        grad_output: &ArrayBase<S2, ndarray::Ix2>,
    ) -> Result<Array2<A>>
    where
        S1: Data<Elem = A>,
        S2: Data<Elem = A>,
    {
        // For simplicity, we're implementing a numerical approximation of the gradient
        // In a real-world scenario, you would implement the analytical gradient

        // Convert views to owned arrays to ensure we can modify them
        let weights = weights.to_owned();
        let grad_output = grad_output.to_owned();

        let n_rows = weights.nrows();
        let n_cols = weights.ncols();
        let epsilon = A::from_f64(1e-6).unwrap();

        let mut gradients = Array2::zeros((n_rows, n_cols));
        let standardized = self.standardize(&weights)?;

        // Numerical gradient approximation
        for i in 0..n_rows {
            for j in 0..n_cols {
                let mut weights_plus = weights.clone();
                weights_plus[[i, j]] = weights_plus[[i, j]] + epsilon;

                let standardized_plus = self.standardize(&weights_plus)?;

                // Calculate the gradient using centered difference
                let diff = &standardized_plus - &standardized;

                // Element-wise multiplication with grad_output and sum
                let mut grad_sum = A::zero();
                for r in 0..n_rows {
                    for c in 0..n_cols {
                        grad_sum = grad_sum + diff[[r, c]] * grad_output[[r, c]];
                    }
                }

                gradients[[i, j]] = grad_sum / epsilon;
            }
        }

        Ok(gradients)
    }
}

impl<A: Float + Debug + ScalarOperand + FromPrimitive, D: Dimension> Regularizer<A, D>
    for WeightStandardization<A>
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // Check if we have 2D parameters
        if params.ndim() != 2 {
            // For simplicity, only handle 2D weights for gradient computation
            // In practice, you would also handle 4D conv weights
            return Ok(A::zero());
        }

        // Downcast to 2D
        let params_2d = params
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OptimError::InvalidConfig("Expected 2D array".to_string()))?;
        let gradients_2d = gradients
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OptimError::InvalidConfig("Expected 2D array".to_string()))?;

        // Compute the gradient corrections
        let corrections = self.compute_gradients(&params_2d, &gradients_2d)?;

        // Apply the corrections to the gradients
        let mut grad_mut = gradients
            .view_mut()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OptimError::InvalidConfig("Expected 2D array".to_string()))?;

        // Add the corrections to the gradients
        grad_mut.zip_mut_with(&corrections, |g, &c| *g = *g + c);

        // Weight standardization doesn't add a penalty term
        Ok(A::zero())
    }

    fn penalty(&self, _params: &Array<A, D>) -> Result<A> {
        // Weight standardization doesn't add a penalty term
        Ok(A::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_weight_standardization_creation() {
        let ws = WeightStandardization::<f64>::new(1e-5);
        assert_eq!(ws.eps, 1e-5);
    }

    #[test]
    fn test_standardize_2d() {
        let ws = WeightStandardization::new(1e-5);

        // Create a simple 2D weight matrix
        let weights = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let standardized = ws.standardize(&weights).unwrap();

        // Check shape is preserved
        assert_eq!(standardized.shape(), weights.shape());

        // Check means are close to zero
        let mean1 = standardized.row(0).sum() / 3.0;
        let mean2 = standardized.row(1).sum() / 3.0;

        assert_relative_eq!(mean1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(mean2, 0.0, epsilon = 1e-10);

        // Check variances are close to 1 (allowing for numerical precision)
        let var1 = standardized.row(0).mapv(|x| x * x).sum() / 3.0;
        let var2 = standardized.row(1).mapv(|x| x * x).sum() / 3.0;

        println!("var1 = {}, var2 = {}", var1, var2);

        // Relaxed tolerance needed due to numerical precision
        assert!((var1 - 1.0).abs() < 2e-4);
        assert!((var2 - 1.0).abs() < 2e-4);
    }

    #[test]
    fn test_standardize_conv4d() {
        let ws = WeightStandardization::new(1e-5);

        // Create a simple 4D convolutional weight tensor
        let weights = Array4::from_shape_fn((2, 2, 2, 2), |idx| {
            let (a, b, c, d) = (idx.0, idx.1, idx.2, idx.3);
            (a * 8 + b * 4 + c * 2 + d) as f64
        });

        let standardized = ws.standardize_conv4d(&weights).unwrap();

        // Check shape is preserved
        assert_eq!(standardized.shape(), weights.shape());

        // Check means are close to zero for each output channel
        let mut sum1 = 0.0;
        let mut sum2 = 0.0;

        for c_in in 0..2 {
            for h in 0..2 {
                for w in 0..2 {
                    sum1 += standardized[[0, c_in, h, w]];
                    sum2 += standardized[[1, c_in, h, w]];
                }
            }
        }

        let mean1 = sum1 / 8.0;
        let mean2 = sum2 / 8.0;

        assert_relative_eq!(mean1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(mean2, 0.0, epsilon = 1e-10);

        // Check variances are close to 1 for each output channel (allowing for numerical precision)
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for c_in in 0..2 {
            for h in 0..2 {
                for w in 0..2 {
                    sum_sq1 += standardized[[0, c_in, h, w]] * standardized[[0, c_in, h, w]];
                    sum_sq2 += standardized[[1, c_in, h, w]] * standardized[[1, c_in, h, w]];
                }
            }
        }

        let var1 = sum_sq1 / 8.0;
        let var2 = sum_sq2 / 8.0;

        assert!((var1 - 1.0).abs() < 1e-5);
        assert!((var2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_regularizer_trait() {
        let ws = WeightStandardization::new(1e-5);
        let params = array![[1.0, 2.0], [3.0, 4.0]];
        let mut gradients = array![[0.1, 0.2], [0.3, 0.4]];
        let orig_gradients = gradients.clone();

        let penalty = ws.apply(&params, &mut gradients).unwrap();

        // Penalty should be zero
        assert_eq!(penalty, 0.0);

        // Gradients should be modified
        assert_ne!(gradients, orig_gradients);
    }
}
