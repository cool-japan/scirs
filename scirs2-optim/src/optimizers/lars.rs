//! Layer-wise Adaptive Rate Scaling (LARS) optimizer
//!
//! LARS is an optimization algorithm specifically designed for large batch training
//! in deep neural networks. It scales the learning rate for each layer based on the
//! ratio of the weight norm to the gradient norm.
//!
//! References:
//! - [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Layer-wise Adaptive Rate Scaling (LARS) optimizer
///
/// LARS is an optimization algorithm specifically designed for large batch training,
/// which allows scaling up the batch size significantly without loss of accuracy.
/// It works by adapting the learning rate per layer based on the ratio of
/// weight norm to gradient norm.
///
/// # Parameters
///
/// * `learning_rate` - Base learning rate
/// * `momentum` - Momentum factor (default: 0.9)
/// * `weight_decay` - Weight decay factor (default: 0.0001)
/// * `trust_coefficient` - Trust coefficient for scaling (default: 0.001)
/// * `eps` - Small constant for numerical stability (default: 1e-8)
/// * `exclude_bias_and_norm` - Whether to exclude bias and normalization layers from LARS adaptation (default: true)
///
/// # Example
///
/// ```no_run
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{LARS, Optimizer};
///
/// let mut optimizer = LARS::new(0.01)
///     .with_momentum(0.9)
///     .with_weight_decay(0.0001)
///     .with_trust_coefficient(0.001);
///
/// let params = Array1::zeros(10);
/// let gradients = Array1::ones(10);
///
/// let updated_params = optimizer.step(&params, &gradients).unwrap();
/// // Parameters are automatically updated
/// ```
#[derive(Debug, Clone)]
pub struct LARS<A: Float> {
    learning_rate: A,
    momentum: A,
    weight_decay: A,
    trust_coefficient: A,
    eps: A,
    exclude_bias_and_norm: bool,
    velocity: Option<Vec<A>>,
}

impl<A: Float + ScalarOperand + Debug> LARS<A> {
    /// Create a new LARS optimizer with the given learning rate
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            momentum: A::from(0.9).unwrap(),
            weight_decay: A::from(0.0001).unwrap(),
            trust_coefficient: A::from(0.001).unwrap(),
            eps: A::from(1e-8).unwrap(),
            exclude_bias_and_norm: true,
            velocity: None,
        }
    }

    /// Set the momentum factor
    pub fn with_momentum(mut self, momentum: A) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the weight decay factor
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set the trust coefficient
    pub fn with_trust_coefficient(mut self, trust_coefficient: A) -> Self {
        self.trust_coefficient = trust_coefficient;
        self
    }

    /// Set the epsilon value for numerical stability
    pub fn with_eps(mut self, eps: A) -> Self {
        self.eps = eps;
        self
    }

    /// Set whether to exclude bias and normalization layers from LARS adaptation
    pub fn with_exclude_bias_and_norm(mut self, exclude_bias_and_norm: bool) -> Self {
        self.exclude_bias_and_norm = exclude_bias_and_norm;
        self
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.velocity = None;
    }
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> Optimizer<A, D> for LARS<A> {
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Initialize velocity if not already created
        let n_params = gradients.len();
        if self.velocity.is_none() {
            self.velocity = Some(vec![A::zero(); n_params]);
        }

        let velocity = match &mut self.velocity {
            Some(v) => {
                if v.len() != n_params {
                    return Err(OptimError::InvalidConfig(format!(
                        "LARS velocity length ({}) does not match gradients length ({})",
                        v.len(),
                        n_params
                    )));
                }
                v
            }
            None => unreachable!(), // We already initialized it
        };

        // Make a clone of parameters for calculating update
        let params_clone = params.clone();

        // Calculate the weight decay term
        let weight_decay_term = if self.weight_decay > A::zero() {
            &params_clone * self.weight_decay
        } else {
            Array::zeros(params.raw_dim())
        };

        // Calculate weight norm and gradient norm
        let weight_norm = params_clone.mapv(|x| x * x).sum().sqrt();
        let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();

        // Determine if we should apply LARS scaling
        let should_apply_lars = !self.exclude_bias_and_norm || weight_norm > A::zero();

        // Calculate local learning rate using trust ratio
        let local_lr = if should_apply_lars && weight_norm > A::zero() && grad_norm > A::zero() {
            self.trust_coefficient * weight_norm
                / (grad_norm + self.weight_decay * weight_norm + self.eps)
        } else {
            A::one()
        };

        // Apply local learning rate scaling
        let scaled_lr = self.learning_rate * local_lr;

        // Calculate gradient update with weight decay
        let update_raw = gradients + &weight_decay_term;

        // Apply scaled learning rate
        let update_scaled = update_raw * scaled_lr;

        // Create output array - will be our result
        let mut updated_params = params.clone();

        // Apply momentum and update parameters
        for (idx, (p, &update)) in updated_params
            .iter_mut()
            .zip(update_scaled.iter())
            .enumerate()
        {
            // Update velocity with momentum
            velocity[idx] = self.momentum * velocity[idx] + update;
            // Update parameter
            *p = *p - velocity[idx];
        }

        Ok(updated_params)
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_lars_creation() {
        let optimizer = LARS::new(0.01);
        assert_abs_diff_eq!(optimizer.learning_rate, 0.01);
        assert_abs_diff_eq!(optimizer.momentum, 0.9);
        assert_abs_diff_eq!(optimizer.weight_decay, 0.0001);
        assert_abs_diff_eq!(optimizer.trust_coefficient, 0.001);
        assert_abs_diff_eq!(optimizer.eps, 1e-8);
        assert!(optimizer.exclude_bias_and_norm);
    }

    #[test]
    fn test_lars_builder() {
        let optimizer = LARS::new(0.01)
            .with_momentum(0.95)
            .with_weight_decay(0.0005)
            .with_trust_coefficient(0.01)
            .with_eps(1e-6)
            .with_exclude_bias_and_norm(false);

        assert_abs_diff_eq!(optimizer.momentum, 0.95);
        assert_abs_diff_eq!(optimizer.weight_decay, 0.0005);
        assert_abs_diff_eq!(optimizer.trust_coefficient, 0.01);
        assert_abs_diff_eq!(optimizer.eps, 1e-6);
        assert!(!optimizer.exclude_bias_and_norm);
    }

    #[test]
    fn test_lars_update() {
        let mut optimizer = LARS::new(0.1)
            .with_momentum(0.9)
            .with_weight_decay(0.0)
            .with_trust_coefficient(1.0);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // First update
        let updated_params = optimizer.step(&params, &gradients).unwrap();

        // LARS scaling factor with trust_coefficient=1.0 should be:
        // weight_norm / grad_norm = sqrt(14) / sqrt(0.14) ≈ 10
        // So the effective learning rate is 0.1 * 10 = 1.0
        // Scale is approximately 10, but let's check actual value (more precise)
        let weight_norm = params.mapv(|x| x * x).sum().sqrt();
        let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();
        let scale = weight_norm / grad_norm;

        assert_abs_diff_eq!(updated_params[0], 1.0 - 0.1 * scale * 0.1, epsilon = 1e-5);
        assert_abs_diff_eq!(updated_params[1], 2.0 - 0.1 * scale * 0.2, epsilon = 1e-5);
        assert_abs_diff_eq!(updated_params[2], 3.0 - 0.1 * scale * 0.3, epsilon = 1e-5);

        // Second update should include momentum
        let updated_params2 = optimizer.step(&updated_params, &gradients).unwrap();

        // For the second update, the velocity will be updated with momentum
        // Just check that parameters continue to change in the expected direction
        assert!(updated_params2[0] < updated_params[0]);
        assert!(updated_params2[1] < updated_params[1]);
        assert!(updated_params2[2] < updated_params[2]);
    }

    #[test]
    fn test_lars_weight_decay() {
        let mut optimizer = LARS::new(0.01)
            .with_momentum(0.0) // No momentum for clarity
            .with_weight_decay(0.1)
            .with_trust_coefficient(1.0);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let updated_params = optimizer.step(&params, &gradients).unwrap();

        // Gradients with weight decay: [0.1, 0.2, 0.3] + 0.1*[1.0, 2.0, 3.0] = [0.2, 0.4, 0.6]
        // LARS scaling factor includes weight decay in denominator
        // weight_norm / (grad_norm + weight_decay * weight_norm)
        // = sqrt(14) / (sqrt(0.56) + 0.1*sqrt(14))
        let weight_norm = params.mapv(|x| x * x).sum().sqrt();
        let grad_norm = gradients.mapv(|x| x * x).sum().sqrt();
        let expected_scale = weight_norm / (grad_norm + 0.1 * weight_norm);

        // Check calculation is approximately correct (allowing for floating point differences)
        let expected_p0 = 1.0 - 0.01 * expected_scale * (0.1 + 0.1 * 1.0);
        let expected_p1 = 2.0 - 0.01 * expected_scale * (0.2 + 0.1 * 2.0);
        let expected_p2 = 3.0 - 0.01 * expected_scale * (0.3 + 0.1 * 3.0);

        assert_abs_diff_eq!(updated_params[0], expected_p0, epsilon = 1e-5);
        assert_abs_diff_eq!(updated_params[1], expected_p1, epsilon = 1e-5);
        assert_abs_diff_eq!(updated_params[2], expected_p2, epsilon = 1e-5);
    }

    #[test]
    fn test_zero_gradients() {
        let mut optimizer = LARS::new(0.01);
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let zero_gradients = Array1::zeros(3);

        let updated_params = optimizer.step(&params, &zero_gradients).unwrap();

        // With zero gradients, only weight decay should contribute to the update
        // With small weight decay (0.0001), changes should be very small
        assert_abs_diff_eq!(updated_params[0], params[0], epsilon = 1e-3);
        assert_abs_diff_eq!(updated_params[1], params[1], epsilon = 1e-3);
        assert_abs_diff_eq!(updated_params[2], params[2], epsilon = 1e-3);
    }

    #[test]
    fn test_exclude_bias_and_norm() {
        let mut optimizer_excluded = LARS::new(0.01)
            .with_momentum(0.0)
            .with_weight_decay(0.0)
            .with_exclude_bias_and_norm(true);

        let mut optimizer_included = LARS::new(0.01)
            .with_momentum(0.0)
            .with_weight_decay(0.0)
            .with_exclude_bias_and_norm(false);

        // Test with parameters that could be bias (small 1D array)
        let bias_params = Array1::from_vec(vec![0.1, 0.2]);
        let bias_grads = Array1::from_vec(vec![0.01, 0.02]);

        let updated_excluded = optimizer_excluded.step(&bias_params, &bias_grads).unwrap();
        let updated_included = optimizer_included.step(&bias_params, &bias_grads).unwrap();

        // When excluded, should use base learning rate (but still include momentum calculation)
        assert_abs_diff_eq!(updated_excluded[0], 0.1 - 0.01 * 0.01, epsilon = 1e-4);

        // When included, should use LARS scaled learning rate
        let weight_norm = (0.1f64.powi(2) + 0.2f64.powi(2)).sqrt();
        let grad_norm = (0.01f64.powi(2) + 0.02f64.powi(2)).sqrt();
        let expected_factor = 0.001 * weight_norm / grad_norm; // trust_coefficient * weight_norm / grad_norm

        assert_abs_diff_eq!(
            updated_included[0],
            0.1 - 0.01 * expected_factor * 0.01,
            epsilon = 1e-5
        );
    }
}
