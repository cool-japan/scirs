//! Lion optimizer implementation
//!
//! Based on the paper "Symbolic Discovery of Optimization Algorithms"
//! by Chen et al. (2023).

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// Lion optimizer
///
/// Implements the Lion (Evolved Sign Momentum) optimization algorithm.
/// Lion is a memory-efficient optimizer that achieves strong performance
/// with only momentum state and uses the sign of the momentum for updates.
///
/// Formula:
/// u_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// theta_t = theta_{t-1} - alpha * (sign(u_t) + lambda * theta_{t-1})
/// m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{Lion, Optimizer};
///
/// // Initialize parameters and gradients
/// let params = Array1::zeros(5);
/// let gradients = Array1::from_vec(vec![0.1, 0.2, -0.3, 0.0, 0.5]);
///
/// // Create a Lion optimizer with default hyperparameters
/// let mut optimizer = Lion::new(0.001);
///
/// // Update parameters
/// let new_params = optimizer.step(&params, &gradients).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Lion<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Exponential decay rate for the momentum
    beta1: A,
    /// Exponential decay rate for the momentum update
    beta2: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// Momentum vector
    m: Option<Vec<Array<A, ndarray::IxDyn>>>,
}

impl<A: Float + ScalarOperand + Debug> Lion<A> {
    /// Creates a new Lion optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).unwrap(),
            beta2: A::from(0.99).unwrap(),
            weight_decay: A::zero(),
            m: None,
        }
    }

    /// Creates a new Lion optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for computing the interpolated update (default: 0.9)
    /// * `beta2` - Exponential decay rate for updating the momentum (default: 0.99)
    /// * `weight_decay` - Weight decay factor for L2 regularization (default: 0.0)
    pub fn new_with_config(learning_rate: A, beta1: A, beta2: A, weight_decay: A) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            weight_decay,
            m: None,
        }
    }

    /// Sets the beta1 parameter
    pub fn set_beta1(&mut self, beta1: A) -> &mut Self {
        self.beta1 = beta1;
        self
    }

    /// Gets the beta1 parameter
    pub fn get_beta1(&self) -> A {
        self.beta1
    }

    /// Sets the beta2 parameter
    pub fn set_beta2(&mut self, beta2: A) -> &mut Self {
        self.beta2 = beta2;
        self
    }

    /// Gets the beta2 parameter
    pub fn get_beta2(&self) -> A {
        self.beta2
    }

    /// Sets the weight decay parameter
    pub fn set_weight_decay(&mut self, weight_decay: A) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Gets the weight decay parameter
    pub fn get_weight_decay(&self) -> A {
        self.weight_decay
    }

    /// Gets the current learning rate
    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    /// Sets the learning rate
    pub fn set_lr(&mut self, lr: A) {
        self.learning_rate = lr;
    }

    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.m = None;
    }
}

impl<A, D> Optimizer<A, D> for Lion<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Convert to dynamic dimension for storage in state vectors
        let params_dyn = params.to_owned().into_dyn();
        let gradients_dyn = gradients.to_owned().into_dyn();

        // Initialize state if this is the first step
        if self.m.is_none() {
            self.m = Some(vec![Array::zeros(params_dyn.raw_dim())]);
        }

        let m = self.m.as_mut().unwrap();

        // Ensure we have state for this parameter set
        if m.is_empty() {
            m.push(Array::zeros(params_dyn.raw_dim()));
        } else if m[0].raw_dim() != params_dyn.raw_dim() {
            // If the parameter dimensions have changed, reset state
            m[0] = Array::zeros(params_dyn.raw_dim());
        }

        // Step 1: Compute interpolated update using beta1
        let interpolated_update = &m[0] * self.beta1 + &gradients_dyn * (A::one() - self.beta1);

        // Step 2: Compute sign of interpolated update
        let sign_update = interpolated_update.mapv(|x| {
            if x > A::zero() {
                A::one()
            } else if x < A::zero() {
                -A::one()
            } else {
                A::zero()
            }
        });

        // Step 3: Update parameters
        let mut updated_params = params_dyn.clone();

        // Apply weight decay if specified
        if self.weight_decay > A::zero() {
            updated_params = &updated_params * (A::one() - self.weight_decay * self.learning_rate);
        }

        // Apply the sign update
        updated_params = &updated_params - &sign_update * self.learning_rate;

        // Step 4: Update momentum using beta2
        m[0] = &m[0] * self.beta2 + &gradients_dyn * (A::one() - self.beta2);

        // Convert back to original dimension
        Ok(updated_params.into_dimensionality::<D>().unwrap())
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_lion_basic_creation() {
        let optimizer: Lion<f64> = Lion::new(0.001);
        assert_abs_diff_eq!(optimizer.learning_rate(), 0.001);
        assert_abs_diff_eq!(optimizer.get_beta1(), 0.9);
        assert_abs_diff_eq!(optimizer.get_beta2(), 0.99);
        assert_abs_diff_eq!(optimizer.get_weight_decay(), 0.0);
    }

    #[test]
    fn test_lion_convergence() {
        let mut optimizer: Lion<f64> = Lion::new(0.1); // Higher learning rate for testing

        // Minimize a simple quadratic function: f(x) = x^2
        let mut params = Array1::from_vec(vec![5.0]);

        // Lion converges linearly with sign updates
        for _ in 0..40 {
            // Fewer iterations with higher learning rate
            // Gradient of x^2 is 2x
            let gradients = Array1::from_vec(vec![2.0 * params[0]]);
            params = optimizer.step(&params, &gradients).unwrap();
        }

        // With learning rate 0.1 and 40 iterations, should reach close to 1.0
        assert!(params[0].abs() < 1.1);
    }

    #[test]
    fn test_lion_reset() {
        let mut optimizer: Lion<f64> = Lion::new(0.1);

        // Perform a step to initialize state
        let params = Array1::from_vec(vec![1.0]);
        let gradients = Array1::from_vec(vec![0.1]);
        let _ = optimizer.step(&params, &gradients).unwrap();

        // Reset optimizer
        optimizer.reset();

        // Next step should behave like the first
        let next_step = optimizer.step(&params, &gradients).unwrap();

        // Create fresh optimizer for comparison
        let mut fresh_optimizer: Lion<f64> = Lion::new(0.1);
        let fresh_step = fresh_optimizer.step(&params, &gradients).unwrap();

        assert_abs_diff_eq!(next_step[0], fresh_step[0], epsilon = 1e-10);
    }
}
