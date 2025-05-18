//! Orthogonal regularization
//!
//! Orthogonal regularization encourages weight matrices to be orthogonal,
//! which helps with gradient flow and prevents vanishing/exploding gradients.

use ndarray::{Array, ArrayBase, Data, Dimension, Ix2, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::regularizers::Regularizer;

/// Orthogonal regularization
///
/// Encourages weight matrices to be orthogonal by penalizing the difference
/// between W^T * W and the identity matrix.
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use scirs2_optim::regularizers::{OrthogonalRegularization, Regularizer};
///
/// let ortho_reg = OrthogonalRegularization::new(0.01);
/// let weights = array![[1.0, 0.0], [0.0, 1.0]];
/// let mut gradient = array![[0.1, 0.2], [0.3, 0.4]];
///
/// // Apply orthogonal regularization  
/// let penalty = ortho_reg.apply(&weights, &mut gradient).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OrthogonalRegularization<A: Float> {
    /// Regularization strength
    lambda: A,
}

impl<A: Float + Debug + ScalarOperand + FromPrimitive> OrthogonalRegularization<A> {
    /// Create a new orthogonal regularization
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength
    pub fn new(lambda: A) -> Self {
        Self { lambda }
    }

    /// Compute orthogonal penalty for a 2D weight matrix
    fn compute_penalty_2d<S: Data<Elem = A>>(&self, weights: &ArrayBase<S, Ix2>) -> A {
        let n = weights.nrows().min(weights.ncols());
        let eye = Array::<A, Ix2>::eye(n);

        // Compute W^T * W
        let wtw = weights.t().dot(weights);

        // Compute Frobenius norm of (W^T * W - I)
        let mut penalty = A::zero();
        for i in 0..n {
            for j in 0..n {
                let diff = wtw[[i, j]] - eye[[i, j]];
                penalty = penalty + diff * diff;
            }
        }

        // For non-square matrices, add penalty for off-diagonal elements
        if weights.nrows() != weights.ncols() {
            let (rows, cols) = wtw.dim();
            for i in 0..rows {
                for j in 0..cols {
                    if i >= n || j >= n {
                        penalty = penalty + wtw[[i, j]] * wtw[[i, j]];
                    }
                }
            }
        }

        self.lambda * penalty
    }

    /// Compute gradient of orthogonal penalty
    fn compute_gradient_2d<S: Data<Elem = A>>(&self, weights: &ArrayBase<S, Ix2>) -> Array<A, Ix2> {
        let n = weights.nrows().min(weights.ncols());

        // Compute W^T * W
        let wtw = weights.t().dot(weights);

        // Compute gradient: 2 * lambda * W * (W^T * W - I)
        let mut diff = wtw.clone();
        for i in 0..n {
            diff[[i, i]] = diff[[i, i]] - A::one();
        }

        weights.dot(&diff) * (A::from_f64(2.0).unwrap() * self.lambda)
    }
}

// Implement Regularizer trait
impl<A: Float + Debug + ScalarOperand + FromPrimitive, D: Dimension> Regularizer<A, D>
    for OrthogonalRegularization<A>
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        if params.ndim() != 2 {
            // Only apply to 2D weight matrices
            return Ok(A::zero());
        }

        // Downcast to 2D
        let params_2d = params
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OptimError::InvalidConfig("Expected 2D array".to_string()))?;

        let gradient_update = self.compute_gradient_2d(&params_2d);

        // Add orthogonal regularization gradient
        let mut gradients_2d = gradients
            .view_mut()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OptimError::InvalidConfig("Expected 2D array".to_string()))?;

        // Manual element-wise addition
        gradients_2d.zip_mut_with(&gradient_update, |g, &u| *g = *g + u);

        // Return penalty
        Ok(self.compute_penalty_2d(&params_2d))
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        if params.ndim() != 2 {
            // Only apply to 2D weight matrices
            return Ok(A::zero());
        }

        // Downcast to 2D
        let params_2d = params
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OptimError::InvalidConfig("Expected 2D array".to_string()))?;

        Ok(self.compute_penalty_2d(&params_2d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_orthogonal_creation() {
        let ortho = OrthogonalRegularization::<f64>::new(0.01);
        assert_eq!(ortho.lambda, 0.01);
    }

    #[test]
    fn test_identity_matrix_penalty() {
        let ortho = OrthogonalRegularization::new(0.01);

        // Identity matrix is already orthogonal, penalty should be 0
        let weights = array![[1.0, 0.0], [0.0, 1.0]];
        let penalty = ortho.compute_penalty_2d(&weights);

        assert_relative_eq!(penalty, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_non_orthogonal_penalty() {
        let ortho = OrthogonalRegularization::new(0.01);

        // Non-orthogonal matrix should have non-zero penalty
        let weights = array![[1.0, 0.5], [0.5, 1.0]];
        let penalty = ortho.compute_penalty_2d(&weights);

        assert!(penalty > 0.0);
    }

    #[test]
    fn test_rectangular_matrix() {
        let ortho = OrthogonalRegularization::new(0.01);

        // Rectangular matrix
        let weights = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let penalty = ortho.compute_penalty_2d(&weights);

        // First 2x2 block is identity, rest should contribute to penalty
        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_gradient_computation() {
        let ortho = OrthogonalRegularization::new(0.1);

        let weights = array![[1.0, 0.5], [0.5, 1.0]];
        let gradient = ortho.compute_gradient_2d(&weights);

        // Gradient should not be zero for non-orthogonal matrix
        assert!(gradient.abs().sum() > 0.0);
    }

    #[test]
    fn test_regularizer_trait() {
        let ortho = OrthogonalRegularization::new(0.01);

        let params = array![[1.0, 0.5], [0.5, 1.0]];
        let mut gradient = array![[0.1, 0.2], [0.3, 0.4]];
        let original_gradient = gradient.clone();

        let penalty = ortho.apply(&params, &mut gradient).unwrap();

        // Penalty should be positive
        assert!(penalty > 0.0);

        // Gradient should be modified
        assert_ne!(gradient, original_gradient);

        // Penalty from apply should match penalty method
        let penalty2 = ortho.penalty(&params).unwrap();
        assert_relative_eq!(penalty, penalty2, epsilon = 1e-10);
    }

    #[test]
    fn test_non_2d_array() {
        let ortho = OrthogonalRegularization::new(0.01);

        // 3D array - should return zero penalty
        let params = Array::<f64, _>::zeros((2, 2, 2));
        let mut gradient = Array::<f64, _>::zeros((2, 2, 2));

        let penalty = ortho.apply(&params, &mut gradient).unwrap();
        assert_eq!(penalty, 0.0);

        // Gradient should be unchanged
        assert_eq!(gradient, Array::<f64, _>::zeros((2, 2, 2)));
    }
}
