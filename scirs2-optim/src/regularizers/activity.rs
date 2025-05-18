use ndarray::{Array, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::Result;
use crate::regularizers::Regularizer;

/// Different norms for Activity regularization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivityNorm {
    /// L1 norm (sum of absolute values)
    L1,
    /// L2 norm (square root of sum of squares)
    L2,
    /// Squared L2 norm (sum of squares)
    L2Squared,
}

/// Activity regularization
///
/// Activity regularization penalizes high activation values in neural networks.
/// It encourages sparse activations by adding a penalty based on the magnitude
/// of activation values.
///
/// # Parameters
///
/// * `lambda`: Regularization strength parameter
/// * `norm`: Type of norm to use for measuring activation magnitudes (L1, L2, or L2Squared)
///
/// # References
///
/// * Nowlan, S. J., & Hinton, G. E. (1992). Simplifying neural networks by soft
///   weight-sharing. Neural Computation, 4(4), 473-493.
///
#[derive(Debug, Clone, Copy)]
pub struct ActivityRegularization<A: Float + FromPrimitive + Debug> {
    /// Regularization strength
    pub lambda: A,
    /// Norm to use for activity regularization
    pub norm: ActivityNorm,
}

impl<A: Float + FromPrimitive + Debug> ActivityRegularization<A> {
    /// Create a new activity regularizer with L1 norm
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength parameter
    ///
    /// # Returns
    ///
    /// A new activity regularizer with L1 norm
    pub fn l1(lambda: A) -> Self {
        Self {
            lambda,
            norm: ActivityNorm::L1,
        }
    }

    /// Create a new activity regularizer with L2 norm
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength parameter
    ///
    /// # Returns
    ///
    /// A new activity regularizer with L2 norm
    pub fn l2(lambda: A) -> Self {
        Self {
            lambda,
            norm: ActivityNorm::L2,
        }
    }

    /// Create a new activity regularizer with squared L2 norm
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength parameter
    ///
    /// # Returns
    ///
    /// A new activity regularizer with squared L2 norm
    pub fn l2_squared(lambda: A) -> Self {
        Self {
            lambda,
            norm: ActivityNorm::L2Squared,
        }
    }

    /// Create a new activity regularizer with custom norm
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength parameter
    /// * `norm` - Norm to use (L1, L2, or L2Squared)
    ///
    /// # Returns
    ///
    /// A new activity regularizer with specified norm
    pub fn new(lambda: A, norm: ActivityNorm) -> Self {
        Self { lambda, norm }
    }

    /// Calculate the activity penalty
    ///
    /// # Arguments
    ///
    /// * `activations` - The activations to regularize
    ///
    /// # Returns
    ///
    /// The regularization penalty value
    fn calculate_penalty<S, D>(&self, activations: &ArrayBase<S, D>) -> A
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        match self.norm {
            ActivityNorm::L1 => {
                // L1 norm: sum of absolute values
                let sum_abs = activations.mapv(|x| x.abs()).sum();
                self.lambda * sum_abs
            }
            ActivityNorm::L2 => {
                // L2 norm: sqrt of sum of squares
                let sum_squared = activations.mapv(|x| x * x).sum();
                self.lambda * sum_squared.sqrt()
            }
            ActivityNorm::L2Squared => {
                // Squared L2 norm: sum of squares
                let sum_squared = activations.mapv(|x| x * x).sum();
                self.lambda * sum_squared
            }
        }
    }

    /// Calculate gradients for activity regularization
    ///
    /// # Arguments
    ///
    /// * `activations` - The activations to regularize
    ///
    /// # Returns
    ///
    /// The gradient array with respect to the activations
    fn calculate_gradients<S, D>(&self, activations: &ArrayBase<S, D>) -> Array<A, D>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        match self.norm {
            ActivityNorm::L1 => {
                // Derivative of L1: sign of the value
                activations.mapv(|x| {
                    if x > A::zero() {
                        self.lambda
                    } else if x < A::zero() {
                        -self.lambda
                    } else {
                        A::zero()
                    }
                })
            }
            ActivityNorm::L2 => {
                // Derivative of L2: x / sqrt(sum(x^2))
                let sum_squared = activations.mapv(|x| x * x).sum();

                // Handle the case where sum_squared is zero to avoid division by zero
                if sum_squared <= A::epsilon() {
                    return Array::zeros(activations.raw_dim());
                }

                let norm = sum_squared.sqrt();
                activations.mapv(|x| self.lambda * x / norm)
            }
            ActivityNorm::L2Squared => {
                // Derivative of squared L2: 2 * x
                let two = A::one() + A::one();
                activations.mapv(|x| self.lambda * two * x)
            }
        }
    }
}

impl<A, D> Regularizer<A, D> for ActivityRegularization<A>
where
    A: Float + ScalarOperand + Debug + FromPrimitive,
    D: Dimension,
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // Calculate penalty
        let penalty = self.calculate_penalty(params);

        // Calculate and apply gradients
        let activity_grads = self.calculate_gradients(params);
        gradients.zip_mut_with(&activity_grads, |g, &a| *g = *g + a);

        Ok(penalty)
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        Ok(self.calculate_penalty(params))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_activity_regularization_creation() {
        let ar = ActivityRegularization::l1(0.1f64);
        assert_eq!(ar.lambda, 0.1);
        assert_eq!(ar.norm, ActivityNorm::L1);

        let ar = ActivityRegularization::l2(0.2f64);
        assert_eq!(ar.lambda, 0.2);
        assert_eq!(ar.norm, ActivityNorm::L2);

        let ar = ActivityRegularization::l2_squared(0.3f64);
        assert_eq!(ar.lambda, 0.3);
        assert_eq!(ar.norm, ActivityNorm::L2Squared);

        let ar = ActivityRegularization::new(0.4f64, ActivityNorm::L1);
        assert_eq!(ar.lambda, 0.4);
        assert_eq!(ar.norm, ActivityNorm::L1);
    }

    #[test]
    fn test_l1_penalty() {
        let lambda = 0.1f64;
        let ar = ActivityRegularization::l1(lambda);

        let activations = Array1::from_vec(vec![1.0f64, -2.0, 3.0]);
        let penalty = ar.penalty(&activations).unwrap();

        // L1 penalty = lambda * sum(|x|) = 0.1 * (1 + 2 + 3) = 0.1 * 6 = 0.6
        assert_abs_diff_eq!(penalty, lambda * 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_penalty() {
        let lambda = 0.1f64;
        let ar = ActivityRegularization::l2(lambda);

        let activations = Array1::from_vec(vec![3.0f64, 4.0]);
        let penalty = ar.penalty(&activations).unwrap();

        // L2 penalty = lambda * sqrt(sum(x^2)) = 0.1 * sqrt(9 + 16) = 0.1 * 5 = 0.5
        assert_abs_diff_eq!(penalty, lambda * 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_squared_penalty() {
        let lambda = 0.1f64;
        let ar = ActivityRegularization::l2_squared(lambda);

        let activations = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        let penalty = ar.penalty(&activations).unwrap();

        // L2 squared penalty = lambda * sum(x^2) = 0.1 * (1 + 4 + 9) = 0.1 * 14 = 1.4
        assert_abs_diff_eq!(penalty, lambda * 14.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l1_gradients() {
        let lambda = 0.1f64;
        let ar = ActivityRegularization::l1(lambda);

        let activations = Array1::from_vec(vec![1.0f64, -2.0, 0.0]);
        let mut gradients = Array1::zeros(3);

        let penalty = ar.apply(&activations, &mut gradients).unwrap();

        // L1 gradients = lambda * sign(x)
        assert_abs_diff_eq!(gradients[0], lambda, epsilon = 1e-10); // sign(1) = 1
        assert_abs_diff_eq!(gradients[1], -lambda, epsilon = 1e-10); // sign(-2) = -1
        assert_abs_diff_eq!(gradients[2], 0.0, epsilon = 1e-10); // sign(0) = 0

        // L1 penalty = lambda * sum(|x|) = 0.1 * (1 + 2 + 0) = 0.1 * 3 = 0.3
        assert_abs_diff_eq!(penalty, lambda * 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_gradients() {
        let lambda = 0.1f64;
        let ar = ActivityRegularization::l2(lambda);

        let activations = Array1::from_vec(vec![3.0f64, 4.0]);
        let mut gradients = Array1::zeros(2);

        let penalty = ar.apply(&activations, &mut gradients).unwrap();

        // Norm = sqrt(9 + 16) = 5
        // L2 gradients = lambda * x / norm
        assert_abs_diff_eq!(gradients[0], lambda * 3.0 / 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gradients[1], lambda * 4.0 / 5.0, epsilon = 1e-10);

        // L2 penalty = lambda * sqrt(sum(x^2)) = 0.1 * sqrt(9 + 16) = 0.1 * 5 = 0.5
        assert_abs_diff_eq!(penalty, lambda * 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_gradients_zero_activations() {
        let lambda = 0.1f64;
        let ar = ActivityRegularization::l2(lambda);

        let activations = Array1::from_vec(vec![0.0f64, 0.0]);
        let mut gradients = Array1::zeros(2);

        let penalty = ar.apply(&activations, &mut gradients).unwrap();

        // When all activations are zero, gradients should be zero to avoid division by zero
        assert_abs_diff_eq!(gradients[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gradients[1], 0.0, epsilon = 1e-10);

        // L2 penalty = lambda * sqrt(sum(x^2)) = 0.1 * sqrt(0) = 0
        assert_abs_diff_eq!(penalty, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_squared_gradients() {
        let lambda = 0.1f64;
        let ar = ActivityRegularization::l2_squared(lambda);

        let activations = Array1::from_vec(vec![2.0f64, 3.0]);
        let mut gradients = Array1::zeros(2);

        let penalty = ar.apply(&activations, &mut gradients).unwrap();

        // L2 squared gradients = lambda * 2 * x
        assert_abs_diff_eq!(gradients[0], lambda * 2.0 * 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gradients[1], lambda * 2.0 * 3.0, epsilon = 1e-10);

        // L2 squared penalty = lambda * sum(x^2) = 0.1 * (4 + 9) = 0.1 * 13 = 1.3
        assert_abs_diff_eq!(penalty, lambda * 13.0, epsilon = 1e-10);
    }

    #[test]
    fn test_2d_activations() {
        let lambda = 0.1f64;
        let ar = ActivityRegularization::l1(lambda);

        let activations = Array2::from_shape_vec((2, 2), vec![1.0f64, 2.0, -3.0, 4.0]).unwrap();
        let penalty = ar.penalty(&activations).unwrap();

        // L1 penalty = lambda * sum(|x|) = 0.1 * (1 + 2 + 3 + 4) = 0.1 * 10 = 1.0
        assert_abs_diff_eq!(penalty, lambda * 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_regularizer_trait() {
        let lambda = 0.1f64;
        let ar = ActivityRegularization::l1(lambda);

        let activations = array![1.0f64, 2.0, 3.0];
        let mut gradients = Array1::zeros(3);

        // Both penalty() and apply() should return the same penalty value
        let penalty1 = ar.penalty(&activations).unwrap();
        let penalty2 = ar.apply(&activations, &mut gradients).unwrap();

        assert_abs_diff_eq!(penalty1, penalty2, epsilon = 1e-10);

        // Check that gradients have been modified correctly
        assert_abs_diff_eq!(gradients[0], lambda, epsilon = 1e-10);
        assert_abs_diff_eq!(gradients[1], lambda, epsilon = 1e-10);
        assert_abs_diff_eq!(gradients[2], lambda, epsilon = 1e-10);
    }
}
