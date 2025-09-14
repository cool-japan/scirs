//! Spectral normalization regularization
//!
//! Spectral normalization is a weight normalization technique that controls the
//! Lipschitz constant of the neural network by normalizing the spectral norm
//! (largest singular value) of weight matrices.

use ndarray::{Array, Array2, Array4, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use rand::Rng;
use scirs2_core::random::Random;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::regularizers::Regularizer;

/// Spectral normalization regularizer
///
/// Normalizes weight matrices by their spectral norm to ensure the Lipschitz
/// constant is bounded. This helps with training stability and generalization.
///
/// # Example
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_optim::regularizers::SpectralNorm;
///
/// let mut spec_norm = SpectralNorm::new(1);
/// let weights = array![[1.0, 2.0], [3.0, 4.0]];
///
/// // Normalize weights by spectral norm
/// let normalized_weights = spec_norm.normalize(&weights).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SpectralNorm<A: Float> {
    /// Number of power iterations for SVD approximation
    n_power_iterations: usize,
    /// Epsilon for numerical stability
    eps: A,
    /// Cached left singular vector
    u: Option<Array<A, ndarray::Ix1>>,
    /// Cached right singular vector  
    v: Option<Array<A, ndarray::Ix1>>,
    /// Random number generator
    rng: Random<rand::rngs::StdRng>,
}

impl<A: Float + Debug + ScalarOperand + FromPrimitive> SpectralNorm<A> {
    /// Create a new spectral normalization regularizer
    ///
    /// # Arguments
    ///
    /// * `n_power_iterations` - Number of power iterations for SVD approximation
    pub fn new(n_poweriterations: usize) -> Self {
        Self {
            n_power_iterations: n_poweriterations,
            eps: A::from_f64(1e-12).unwrap(),
            u: None,
            v: None,
            rng: Random::seed(42),
        }
    }

    /// Compute the spectral norm (largest singular value) using power iteration
    fn compute_spectral_norm(&mut self, weights: &Array2<A>) -> Result<A> {
        let (m, n) = (weights.nrows(), weights.ncols());

        // Initialize u and v if not already done
        if self.u.is_none() || self.u.as_ref().unwrap().len() != m {
            self.u = Some(Array::from_shape_fn((m,), |_| {
                let val: f64 = self.rng.gen_range(0.0..1.0);
                A::from_f64(val).unwrap()
            }));
        }

        if self.v.is_none() || self.v.as_ref().unwrap().len() != n {
            self.v = Some(Array::from_shape_fn((n,), |_| {
                let val: f64 = self.rng.gen_range(0.0..1.0);
                A::from_f64(val).unwrap()
            }));
        }

        let mut u = self.u.as_ref().unwrap().clone();
        let mut v = self.v.as_ref().unwrap().clone();

        // Power iteration
        for _ in 0..self.n_power_iterations {
            // v = W^T u / ||W^T u||
            let wt_u = weights.t().dot(&u);
            let v_norm = (wt_u.dot(&wt_u) + self.eps).sqrt();
            v = wt_u / v_norm;

            // u = W v / ||W v||
            let w_v = weights.dot(&v);
            let u_norm = (w_v.dot(&w_v) + self.eps).sqrt();
            u = w_v / u_norm;
        }

        // Update cached vectors
        self.u = Some(u.clone());
        self.v = Some(v.clone());

        // Compute spectral norm as u^T W v
        let w_v = weights.dot(&v);
        let spectral_norm = u.dot(&w_v);

        Ok(spectral_norm)
    }

    /// Normalize weights by spectral norm
    pub fn normalize(&mut self, weights: &Array2<A>) -> Result<Array2<A>> {
        let spectral_norm = self.compute_spectral_norm(weights)?;

        if spectral_norm > self.eps {
            Ok(weights / spectral_norm)
        } else {
            Ok(weights.clone())
        }
    }

    /// Apply spectral normalization to 4D convolutional weights
    pub fn normalize_conv4d(&mut self, weights: &Array4<A>) -> Result<Array4<A>> {
        // Reshape to 2D for spectral norm computation
        let shape = weights.shape();
        let out_channels = shape[0];
        let in_channels = shape[1];
        let kernel_h = shape[2];
        let kernel_w = shape[3];

        let weights_2d = weights
            .to_shape((out_channels, in_channels * kernel_h * kernel_w))
            .map_err(|e| OptimError::InvalidConfig(format!("Cannot reshape weights: {}", e)))?;
        let weights_2d_owned = weights_2d.to_owned();
        let normalized_2d = self.normalize(&weights_2d_owned)?;

        // Reshape back to 4D
        let normalized_4d = normalized_2d
            .to_shape((out_channels, in_channels, kernel_h, kernel_w))
            .map_err(|e| {
                OptimError::InvalidConfig(format!("Cannot reshape normalized weights: {}", e))
            })?;
        Ok(normalized_4d.to_owned())
    }
}

// Implement Regularizer trait
impl<A: Float + Debug + ScalarOperand + FromPrimitive, D: Dimension> Regularizer<A, D>
    for SpectralNorm<A>
{
    fn apply(&self, _params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // For spectral normalization, we don't modify _gradients directly
        // Instead, the normalization is typically applied during the forward pass
        Ok(A::zero())
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        // Spectral normalization doesn't add a penalty term
        Ok(A::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_spectral_norm_creation() {
        let sn = SpectralNorm::<f64>::new(5);
        assert_eq!(sn.n_power_iterations, 5);
    }

    #[test]
    fn test_spectral_norm_2d() {
        let mut sn = SpectralNorm::new(10);

        // Create a simple matrix with known singular values
        // For a 2x2 matrix [[1, 0], [0, 2]], the singular values are 1 and 2
        let weights = array![[1.0, 0.0], [0.0, 2.0]];

        let spectral_norm = sn.compute_spectral_norm(&weights).unwrap();

        // The spectral norm should be close to 2.0 (largest singular value)
        assert_relative_eq!(spectral_norm, 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_normalize_2d() {
        let mut sn = SpectralNorm::new(10);

        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let normalized = sn.normalize(&weights).unwrap();

        // After normalization, the spectral norm should be close to 1
        let spec_norm = sn.compute_spectral_norm(&normalized).unwrap();
        assert_relative_eq!(spec_norm, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_conv4d_normalization() {
        let mut sn = SpectralNorm::new(5);

        // Create a 4D tensor (out_channels, in_channels, height, width)
        let weights = Array::from_shape_fn((2, 3, 3, 3), |(o, i, h, w)| {
            (o * 27 + i * 9 + h * 3 + w) as f64
        });

        let normalized = sn.normalize_conv4d(&weights).unwrap();

        // Check that the shape is preserved
        assert_eq!(normalized.shape(), weights.shape());
    }

    #[test]
    fn test_invalid_conv4d() {
        let mut sn = SpectralNorm::<f64>::new(5);

        // Create a 4D tensor (which is valid)
        let weights = Array::zeros((2, 3, 4, 4));

        // Should succeed for 4D tensors
        assert!(sn.normalize_conv4d(&weights).is_ok());
    }

    #[test]
    fn test_regularizer_trait() {
        let sn = SpectralNorm::new(5);
        let params = array![[1.0, 2.0], [3.0, 4.0]];
        let mut gradient = array![[0.1, 0.2], [0.3, 0.4]];

        // Spectral norm doesn't modify gradients or add penalties
        let penalty = sn.penalty(&params).unwrap();
        assert_eq!(penalty, 0.0);

        let apply_result = sn.apply(&params, &mut gradient).unwrap();
        assert_eq!(apply_result, 0.0);

        // Gradients should be unchanged
        assert_eq!(gradient, array![[0.1, 0.2], [0.3, 0.4]]);
    }
}
