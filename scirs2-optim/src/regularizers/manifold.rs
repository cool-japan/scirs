//! Manifold regularization
//!
//! Manifold regularization assumes that data lies on a low-dimensional manifold
//! and encourages smoothness along this manifold structure.

use ndarray::{Array, Array2, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::regularizers::Regularizer;

/// Manifold regularization
///
/// Regularizes based on the assumption that data lies on a low-dimensional manifold,
/// encouraging smooth predictions along the manifold structure.
///
/// # Example
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_optim::regularizers::{ManifoldRegularization, Regularizer};
///
/// let mut manifold_reg = ManifoldRegularization::new(0.01);
///
/// // Set up similarity matrix based on data structure
/// let similarity = array![[1.0, 0.8], [0.8, 1.0]];
/// manifold_reg.set_similarity_matrix(similarity).unwrap();
///
/// let params = array![[1.0, 2.0], [3.0, 4.0]];
/// let mut gradient = array![[0.1, 0.2], [0.3, 0.4]];
///
/// // Apply manifold regularization
/// let penalty = manifold_reg.apply(&params, &mut gradient).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ManifoldRegularization<A: Float> {
    /// Regularization strength
    lambda: A,
    /// Similarity/adjacency matrix encoding manifold structure  
    similarity_matrix: Option<Array2<A>>,
    /// Degree matrix for normalization
    degree_matrix: Option<Array2<A>>,
    /// Laplacian matrix  
    laplacian: Option<Array2<A>>,
}

impl<A: Float + Debug + ScalarOperand + FromPrimitive> ManifoldRegularization<A> {
    /// Create a new manifold regularization
    ///
    /// # Arguments
    ///
    /// * `lambda` - Regularization strength
    pub fn new(lambda: A) -> Self {
        Self {
            lambda,
            similarity_matrix: None,
            degree_matrix: None,
            laplacian: None,
        }
    }

    /// Set the similarity matrix encoding manifold structure
    ///
    /// # Arguments
    ///
    /// * `similarity` - Symmetric similarity/adjacency matrix
    pub fn set_similarity_matrix(&mut self, similarity: Array2<A>) -> Result<()> {
        let (rows, cols) = similarity.dim();
        if rows != cols {
            return Err(OptimError::InvalidConfig(
                "Similarity matrix must be square".to_string(),
            ));
        }

        // Compute degree matrix (diagonal matrix of row sums)
        let mut degree = Array2::zeros((rows, rows));
        for i in 0..rows {
            let row_sum = similarity.row(i).sum();
            degree[[i, i]] = row_sum;
        }

        // Compute Laplacian: L = D - W  (not normalized for simplicity)
        let laplacian = &degree - &similarity;

        self.similarity_matrix = Some(similarity);
        self.degree_matrix = Some(degree);
        self.laplacian = Some(laplacian);

        Ok(())
    }

    /// Compute manifold regularization penalty
    fn compute_penalty<S>(&self, params: &ArrayBase<S, ndarray::Ix2>) -> Result<A>
    where
        S: Data<Elem = A>,
    {
        let laplacian = self
            .laplacian
            .as_ref()
            .ok_or_else(|| OptimError::InvalidConfig("Similarity matrix not set".to_string()))?;

        // Penalty = λ * tr(F^T * L * F) where F are the parameters
        // For efficiency, compute as λ * sum(F .* (L * F))
        let lf = laplacian.dot(params);
        let penalty = params
            .iter()
            .zip(lf.iter())
            .map(|(p, lf)| *p * *lf)
            .fold(A::zero(), |acc, val| acc + val);

        Ok(self.lambda * penalty)
    }

    /// Compute gradient of manifold regularization
    fn compute_gradient<S>(&self, params: &ArrayBase<S, ndarray::Ix2>) -> Result<Array2<A>>
    where
        S: Data<Elem = A>,
    {
        let laplacian = self
            .laplacian
            .as_ref()
            .ok_or_else(|| OptimError::InvalidConfig("Similarity matrix not set".to_string()))?;

        // Gradient = 2 * λ * L * F
        let gradient = laplacian.dot(params) * (A::from_f64(2.0).unwrap() * self.lambda);
        Ok(gradient)
    }
}

// Implement Regularizer trait
impl<A: Float + Debug + ScalarOperand + FromPrimitive, D: Dimension> Regularizer<A, D>
    for ManifoldRegularization<A>
{
    fn apply(&self, params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        if params.ndim() != 2 {
            // Only apply to 2D parameter matrices
            return Ok(A::zero());
        }

        // Downcast to 2D
        let params_2d = params
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OptimError::InvalidConfig("Expected 2D array".to_string()))?;

        let gradient_update = self.compute_gradient(&params_2d)?;

        // Add manifold regularization gradient
        let mut gradients_2d = gradients
            .view_mut()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OptimError::InvalidConfig("Expected 2D array".to_string()))?;

        gradients_2d.zip_mut_with(&gradient_update, |g, &u| *g = *g + u);

        // Return penalty
        self.compute_penalty(&params_2d)
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        if params.ndim() != 2 {
            // Only apply to 2D parameter matrices
            return Ok(A::zero());
        }

        // Downcast to 2D
        let params_2d = params
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| OptimError::InvalidConfig("Expected 2D array".to_string()))?;

        self.compute_penalty(&params_2d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_manifold_creation() {
        let manifold = ManifoldRegularization::<f64>::new(0.01);
        assert_eq!(manifold.lambda, 0.01);
        assert!(manifold.similarity_matrix.is_none());
    }

    #[test]
    fn test_set_similarity_matrix() {
        let mut manifold = ManifoldRegularization::new(0.01);

        // Simple 2x2 similarity matrix
        let similarity = array![[1.0, 0.5], [0.5, 1.0]];

        assert!(manifold.set_similarity_matrix(similarity).is_ok());
        assert!(manifold.laplacian.is_some());

        // Check Laplacian computation
        let laplacian = manifold.laplacian.as_ref().unwrap();
        // For matrix [[1.0, 0.5], [0.5, 1.0]]:
        // Degree matrix is [[1.5, 0.0], [0.0, 1.5]]
        // Laplacian is [[1.5, 0.0], [0.0, 1.5]] - [[1.0, 0.5], [0.5, 1.0]] = [[0.5, -0.5], [-0.5, 0.5]]
        assert_relative_eq!(laplacian[[0, 0]], 0.5, epsilon = 1e-10);
        assert_relative_eq!(laplacian[[0, 1]], -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_similarity_matrix() {
        let mut manifold = ManifoldRegularization::<f64>::new(0.01);

        // Non-square matrix should fail
        let similarity = array![[1.0, 0.5, 0.3], [0.5, 1.0, 0.4]];
        assert!(manifold.set_similarity_matrix(similarity).is_err());
    }

    #[test]
    fn test_penalty_without_similarity() {
        let manifold = ManifoldRegularization::<f64>::new(0.01);
        let params = array![[1.0, 2.0], [3.0, 4.0]];

        // Should fail without similarity matrix
        assert!(manifold.compute_penalty(&params).is_err());
    }

    #[test]
    fn test_penalty_computation() {
        let mut manifold = ManifoldRegularization::new(0.1);

        // Set up similarity matrix
        let similarity = array![[1.0, 0.8], [0.8, 1.0]];
        manifold.set_similarity_matrix(similarity).unwrap();

        // Test penalty computation
        let params = array![[1.0, 0.0], [0.0, 1.0]];
        let penalty = manifold.compute_penalty(&params).unwrap();

        // Penalty should be positive for non-similar parameters
        assert!(penalty > 0.0);
    }

    #[test]
    fn test_gradient_computation() {
        let mut manifold = ManifoldRegularization::new(0.1);

        // Set up similarity matrix
        let similarity = array![[1.0, 0.8], [0.8, 1.0]];
        manifold.set_similarity_matrix(similarity).unwrap();

        let params = array![[1.0, 2.0], [3.0, 4.0]];
        let gradient = manifold.compute_gradient(&params).unwrap();

        // Gradient should be non-zero
        assert!(gradient.abs().sum() > 0.0);
    }

    #[test]
    fn test_regularizer_trait() {
        let mut manifold = ManifoldRegularization::new(0.01);

        // Set up similarity matrix
        let similarity = array![[1.0, 0.6], [0.6, 1.0]];
        manifold.set_similarity_matrix(similarity).unwrap();

        let params = array![[1.0, 2.0], [3.0, 4.0]];
        let mut gradient = array![[0.1, 0.2], [0.3, 0.4]];
        let original_gradient = gradient.clone();

        let penalty = manifold.apply(&params, &mut gradient).unwrap();

        // Penalty should be positive
        assert!(penalty > 0.0);

        // Gradient should be modified
        assert_ne!(gradient, original_gradient);
    }

    #[test]
    fn test_identity_similarity() {
        let mut manifold = ManifoldRegularization::new(0.1);

        // Identity similarity matrix (no connections)
        let similarity = array![[1.0, 0.0], [0.0, 1.0]];
        manifold.set_similarity_matrix(similarity).unwrap();

        let params = array![[1.0, 2.0], [3.0, 4.0]];
        let penalty = manifold.compute_penalty(&params).unwrap();

        // With identity similarity, penalty depends on diagonal structure
        assert!(penalty >= 0.0);
    }
}
