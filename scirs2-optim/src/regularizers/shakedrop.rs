use ndarray::{Array, ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use rand::Rng;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::regularizers::Regularizer;

/// ShakeDrop regularization
///
/// ShakeDrop is a regularization method that extends Stochastic Depth and
/// is often used in very deep neural networks. It randomly scales activations
/// during training.
///
/// # Parameters
///
/// * `p` - The probability of activating ShakeDrop (probability of activating the forward
///   pass transformation), value between 0 and 1.
/// * `alpha_range` - The range for the alpha parameter used in forward pass (default: [-1.0, 1.0]).
/// * `beta_range` - The range for the beta parameter used in backward pass (default: [0.0, 1.0]).
///
/// # References
///
/// * Yamada, Y., Iwamura, M., & Kise, K. (2018). ShakeDrop regularization.
///   arXiv preprint arXiv:1802.02375.
///
#[derive(Debug, Clone)]
pub struct ShakeDrop<A: Float + FromPrimitive + Debug> {
    /// Probability of applying ShakeDrop
    pub p: A,
    /// Range for the alpha parameter
    pub alpha_range: (A, A),
    /// Range for the beta parameter
    pub beta_range: (A, A),
    /// Random number generator
    rng: scirs2_core::random::Random<rand::rngs::StdRng>,
}

impl<A: Float + FromPrimitive + Debug> ShakeDrop<A> {
    /// Create a new ShakeDrop regularizer
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of applying ShakeDrop, between 0 and 1
    /// * `alpha_range` - Range for the alpha parameter (default: [-1.0, 1.0])
    /// * `beta_range` - Range for the beta parameter (default: [0.0, 1.0])
    ///
    /// # Returns
    ///
    /// A ShakeDrop regularizer
    pub fn new(p: A) -> Self {
        let zero = A::zero();
        let one = A::one();
        let neg_one = zero - one;

        Self {
            p,
            alpha_range: (neg_one, one),
            beta_range: (zero, one),
            rng: scirs2_core::random::Random::seed(42),
        }
    }

    /// Create a new ShakeDrop regularizer with custom ranges
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of applying ShakeDrop, between 0 and 1
    /// * `alpha_range` - Range for the alpha parameter
    /// * `beta_range` - Range for the beta parameter
    ///
    /// # Returns
    ///
    /// A ShakeDrop regularizer
    pub fn new_with_ranges(p: A, alpharange: (A, A), beta_range: (A, A)) -> Self {
        Self {
            p,
            alpha_range: alpharange,
            beta_range,
            rng: scirs2_core::random::Random::seed(42),
        }
    }

    /// Get a random value between the given range
    fn random_in_range(&mut self, range: (A, A)) -> A {
        let (min, max) = range;
        let min_f = min.to_f64().unwrap();
        let max_f = max.to_f64().unwrap();

        // Handle equal min and max to avoid "empty range" error
        if (max_f - min_f).abs() < 1e-10 {
            return min;
        }

        let random_val = self.rng.gen_range(min_f..max_f);
        A::from_f64(random_val).unwrap()
    }

    /// Get a forward pass gate for the ShakeDrop
    ///
    /// # Returns
    ///
    /// A tuple (b, alpha, beta):
    /// - b: Binary gate (1 or 0) based on the probability p
    /// - alpha: Random value within alpha_range if b is 1, otherwise 0
    /// - beta: Random value within beta_range
    fn get_gate(&mut self) -> (A, A, A) {
        let zero = A::zero();
        let one = A::one();

        // Determine if the gate is active
        let u: f64 = self.rng.gen_range(0.0..1.0);
        let b = if u < self.p.to_f64().unwrap() {
            one
        } else {
            zero
        };

        // Get random alpha if gate is active..otherwise 0
        let alpha = if b > zero {
            self.random_in_range(self.alpha_range)
        } else {
            zero
        };

        // Get random beta regardless of gate
        let beta = self.random_in_range(self.beta_range);

        (b, alpha, beta)
    }

    /// Apply ShakeDrop to input activations
    ///
    /// # Arguments
    ///
    /// * `x` - Input activation tensor
    ///
    /// # Returns
    ///
    /// The transformed activations and gate parameters for use in backward pass
    pub fn forward<S, D>(&mut self, x: &ArrayBase<S, D>) -> (Array<A, D>, (A, A, A))
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        // Get the gate values
        let (b, alpha, beta) = self.get_gate();

        // Apply ShakeDrop transformation
        // During forward pass: x' = x * (b + alpha - b*alpha)
        let factor = b + alpha - b * alpha;
        let result = x.mapv(|v| v * factor);

        (result, (b, alpha, beta))
    }

    /// Backward pass for ShakeDrop
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient from the next layer
    /// * `gate_params` - The gate parameters (b, alpha, beta) from the forward pass
    ///
    /// # Returns
    ///
    /// The modified gradients
    pub fn backward<S, D>(
        &self,
        grad_output: &ArrayBase<S, D>,
        gate_params: (A, A, A),
    ) -> Array<A, D>
    where
        S: Data<Elem = A>,
        D: Dimension,
    {
        let (b, alpha, beta) = gate_params;

        // During backward pass: grad_x = grad_output * (b + beta - b*beta)
        let factor = b + beta - b * beta;
        grad_output.mapv(|g| g * factor)
    }
}

impl<A: Float + FromPrimitive + Debug + ScalarOperand, D: Dimension> Regularizer<A, D>
    for ShakeDrop<A>
{
    fn apply(&self, _params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        // ShakeDrop is typically applied to activations, not parameters
        // In this implementation, apply() isn't the primary usage pattern
        // Instead, users would call forward() during the forward pass
        // and backward() during the backward pass
        Err(OptimError::InvalidConfig(
            "ShakeDrop should be applied to activations during forward/backward passes, \
             not through the Regularizer trait's apply method"
                .to_string(),
        ))
    }

    fn penalty(&self, params: &Array<A, D>) -> Result<A> {
        // ShakeDrop doesn't add a penalty term to the loss function
        Ok(A::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_shakedrop_new() {
        let sd = ShakeDrop::new(0.5f64);
        assert_eq!(sd.p, 0.5);
        assert_eq!(sd.alpha_range, (-1.0, 1.0));
        assert_eq!(sd.beta_range, (0.0, 1.0));
    }

    #[test]
    fn test_shakedrop_new_with_ranges() {
        let sd = ShakeDrop::new_with_ranges(0.7f64, (-0.5, 0.5), (0.2, 0.8));
        assert_eq!(sd.p, 0.7);
        assert_eq!(sd.alpha_range, (-0.5, 0.5));
        assert_eq!(sd.beta_range, (0.2, 0.8));
    }

    #[test]
    fn test_shakedrop_forward_backward() {
        // Create a simple 2D array
        let x = Array2::from_elem((2, 3), 1.0f64);

        // Initialize ShakeDrop with p=1.0 to ensure gate is always active
        // Use slightly different values for min and max to avoid empty range error
        let mut sd = ShakeDrop::new_with_ranges(1.0f64, (0.5, 0.500001), (0.5, 0.500001));

        // Forward pass
        let (output, gate_params) = sd.forward(&x);

        // Verify the gate parameters
        assert_eq!(gate_params.0, 1.0); // b should be 1 since p=1.0
        assert_abs_diff_eq!(gate_params.1, 0.5, epsilon = 1e-5); // alpha should be approximately 0.5
        assert_abs_diff_eq!(gate_params.2, 0.5, epsilon = 1e-5); // beta should be approximately 0.5

        // The expected output is x * (b + alpha - b*alpha) = x * (1 + 0.5 - 1*0.5) = x * 1
        for &val in output.iter() {
            assert_abs_diff_eq!(val, 1.0, epsilon = 1e-5);
        }

        // Backward pass
        let grad_output = Array2::from_elem((2, 3), 2.0f64);
        let grad_input = sd.backward(&grad_output, gate_params);

        // The expected gradient is grad_output * (b + beta - b*beta) = grad_output * (1 + 0.5 - 1*0.5) = grad_output * 1
        for &val in grad_input.iter() {
            assert_abs_diff_eq!(val, 2.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_shakedrop_forward_inactive() {
        // Create a simple 1D array
        let x = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);

        // Initialize ShakeDrop with p=0.0 to ensure gate is always inactive
        // Use slightly different values for min and max to avoid empty range error
        let mut sd = ShakeDrop::new_with_ranges(0.0f64, (-0.5, -0.499999), (0.5, 0.500001));

        // Forward pass - gate should be inactive
        let (output, gate_params) = sd.forward(&x);

        // Verify the gate parameters
        assert_eq!(gate_params.0, 0.0); // b should be 0 since p=0.0
        assert_eq!(gate_params.1, 0.0); // alpha should be 0 when gate is inactive
        assert_abs_diff_eq!(gate_params.2, 0.5, epsilon = 1e-5); // beta should be approximately 0.5

        // The expected output is x * (b + alpha - b*alpha) = x * (0 + 0 - 0*0) = x * 0
        for &val in output.iter() {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_shakedrop_gen_range() {
        let mut sd = ShakeDrop::new(0.5f64);

        // Test random value generation within range
        for _ in 0..100 {
            let value = sd.random_in_range((-0.5, 0.5));
            assert!((-0.5..=0.5).contains(&value));
        }

        // Test with very small range (should not panic)
        let value = sd.random_in_range((0.5, 0.5));
        assert_eq!(value, 0.5);
    }

    #[test]
    fn test_shakedrop_get_gate() {
        // Test with p=1.0 - gate should always be active
        let mut sd = ShakeDrop::new(1.0f64);
        for _ in 0..10 {
            let (b, alpha, beta) = sd.get_gate();
            assert_eq!(b, 1.0);
            assert!((-1.0..=1.0).contains(&alpha));
            assert!((0.0..=1.0).contains(&beta));
        }

        // Test with p=0.0 - gate should always be inactive
        let mut sd = ShakeDrop::new(0.0f64);
        for _ in 0..10 {
            let (b, alpha, beta) = sd.get_gate();
            assert_eq!(b, 0.0);
            assert_eq!(alpha, 0.0);
            assert!((0.0..=1.0).contains(&beta));
        }
    }

    #[test]
    fn test_regularizer_trait() {
        let sd = ShakeDrop::new(0.5f64);
        let params = Array2::from_elem((2, 3), 1.0f64);
        let mut grads = Array2::from_elem((2, 3), 1.0f64);

        // apply() should return an error for ShakeDrop
        assert!(sd.apply(&params, &mut grads).is_err());

        // penalty() should return zero
        let penalty = sd.penalty(&params).unwrap();
        assert_eq!(penalty, 0.0);
    }
}
