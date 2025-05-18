//! Sharpness-Aware Minimization (SAM) optimizer
//!
//! Implements the SAM optimization algorithm from:
//! "Sharpness-Aware Minimization for Efficiently Improving Generalization" (Foret et al., 2020)

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Sharpness-Aware Minimization (SAM) optimizer
///
/// SAM is an optimization technique that seeks parameters that lie in neighborhoods
/// having uniformly low loss values, which improves generalization. It achieves this by
/// performing a two-step update process:
///
/// 1. Compute and take a step in the direction of the "sharpness" gradient (perturbed parameters)
/// 2. Compute the gradient at these perturbed parameters and use it to update the original parameters
///
/// This implementation wraps around a base optimizer and modifies its behavior to implement
/// the SAM algorithm.
///
/// # Parameters
///
/// * `inner_optimizer` - The optimizer to use for the parameter updates
/// * `rho` - The neighborhood size for perturbation (default: 0.05)
/// * `epsilon` - Small constant for numerical stability (default: 1e-12)
/// * `adaptive` - Whether to use adaptive perturbation size (SAM-A) (default: false)
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{SAM, SGD};
/// use scirs2_optim::Optimizer;
///
/// // Create a base optimizer
/// let sgd = SGD::new(0.1);
///
/// // Wrap it with SAM
/// let mut optimizer = SAM::new(sgd);
///
/// // First step to compute perturbed parameters and store perturbed gradients
/// let params = Array1::zeros(10);
/// let gradients = Array1::ones(10);
/// let (perturbed_params, _) = optimizer.first_step(&params, &gradients).unwrap();
///
/// // Second step to update original parameters using gradients at perturbed parameters
/// // Normally, you would compute new gradients at perturbed_params
/// let new_gradients = Array1::ones(10) * 0.5; // Example new gradients
/// let updated_params = optimizer.second_step(&params, &new_gradients).unwrap();
/// ```
pub struct SAM<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone,
    D: Dimension,
{
    /// Inner optimizer for parameter updates
    inner_optimizer: O,
    /// Neighborhood size for perturbation (ρ)
    rho: A,
    /// Small constant for numerical stability (ε)
    epsilon: A,
    /// Whether to use adaptive perturbation size (SAM-A)
    adaptive: bool,
    /// Perturbed parameters from first step
    perturbed_params: Option<Array<A, D>>,
    /// Original parameters from first step
    original_params: Option<Array<A, D>>,
    /// Dimension type marker
    _phantom: PhantomData<D>,
}

impl<A, O, D> SAM<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone,
    D: Dimension,
{
    /// Creates a new SAM optimizer with the given inner optimizer and default settings
    pub fn new(inner_optimizer: O) -> Self {
        Self {
            inner_optimizer,
            rho: A::from(0.05).unwrap(),
            epsilon: A::from(1e-12).unwrap(),
            adaptive: false,
            perturbed_params: None,
            original_params: None,
            _phantom: PhantomData,
        }
    }

    /// Creates a new SAM optimizer with the given inner optimizer and configuration
    pub fn with_config(inner_optimizer: O, rho: A, adaptive: bool) -> Self {
        Self {
            inner_optimizer,
            rho,
            epsilon: A::from(1e-12).unwrap(),
            adaptive,
            perturbed_params: None,
            original_params: None,
            _phantom: PhantomData,
        }
    }

    /// Set the rho parameter (neighborhood size)
    pub fn with_rho(mut self, rho: A) -> Self {
        self.rho = rho;
        self
    }

    /// Set the epsilon parameter (numerical stability)
    pub fn with_epsilon(mut self, epsilon: A) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set whether to use adaptive perturbation size (SAM-A)
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Get the inner optimizer
    pub fn inner_optimizer(&self) -> &O {
        &self.inner_optimizer
    }

    /// Get a mutable reference to the inner optimizer
    pub fn inner_optimizer_mut(&mut self) -> &mut O {
        &mut self.inner_optimizer
    }

    /// Get the rho parameter
    pub fn rho(&self) -> A {
        self.rho
    }

    /// Get the epsilon parameter
    pub fn epsilon(&self) -> A {
        self.epsilon
    }

    /// Check if using adaptive perturbation size
    pub fn is_adaptive(&self) -> bool {
        self.adaptive
    }

    /// First step of SAM: compute perturbed parameters by moving in the direction of the gradient
    ///
    /// # Arguments
    ///
    /// * `params` - Current parameters
    /// * `gradients` - Gradients of the loss with respect to the parameters
    ///
    /// # Returns
    ///
    /// Tuple containing (perturbed_parameters, perturbation_size)
    pub fn first_step(
        &mut self,
        params: &Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<(Array<A, D>, A)> {
        // Store original parameters
        self.original_params = Some(params.clone());

        // Calculate gradient norm for scaling
        let grad_norm = calculate_norm(gradients)?;

        if grad_norm.is_zero() || !grad_norm.is_finite() {
            return Err(OptimError::OptimizationError(
                "Gradient norm is zero or not finite".to_string(),
            ));
        }

        // Calculate perturbation size
        let e_w = if self.adaptive {
            // Adaptive SAM: scale perturbation by parameter-wise gradient magnitude
            // Note: We need to be careful with parameter scaling to avoid numerical issues
            let param_norm = calculate_norm(params)?;
            if param_norm.is_zero() || !param_norm.is_finite() {
                // Fall back to standard SAM if parameter norm is problematic
                let perturb = gradients / (grad_norm + self.epsilon);
                &perturb * self.rho
            } else {
                // Use a more stable calculation for adaptive SAM
                let mut perturb = params.mapv(|p| p.abs() + self.epsilon);
                perturb = &perturb / param_norm; // Normalize first
                                                 // Element-wise multiply and scale by rho
                gradients * &perturb * self.rho
            }
        } else {
            // Standard SAM: scale perturbation by gradient norm
            let perturb = gradients / (grad_norm + self.epsilon);
            &perturb * self.rho
        };

        // Create perturbed parameters
        let perturbed_params = params + &e_w;
        self.perturbed_params = Some(perturbed_params.clone());

        // Return perturbed parameters and perturbation norm
        Ok((perturbed_params, calculate_norm(&e_w)?))
    }

    /// Second step of SAM: update the original parameters using gradients at the perturbed point
    ///
    /// # Arguments
    ///
    /// * `_params` - Original parameters (used for validation)
    /// * `gradients` - Gradients of the loss with respect to the perturbed parameters
    ///
    /// # Returns
    ///
    /// Updated parameters after applying the "sharpness-aware" update
    pub fn second_step(
        &mut self,
        _params: &Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<Array<A, D>> {
        // Get original parameters
        let original_params = match &self.original_params {
            Some(params) => params,
            None => {
                return Err(OptimError::OptimizationError(
                    "Must call first_step before second_step".to_string(),
                ))
            }
        };

        // Use the inner optimizer to update the original parameters with the perturbed gradients
        let updated_params = self.inner_optimizer.step(original_params, gradients)?;

        // Reset stored parameters
        self.perturbed_params = None;
        self.original_params = None;

        Ok(updated_params)
    }

    /// Reset the internal state
    pub fn reset(&mut self) {
        self.perturbed_params = None;
        self.original_params = None;
    }
}

impl<A, O, D> Clone for SAM<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone,
    D: Dimension,
{
    fn clone(&self) -> Self {
        Self {
            inner_optimizer: self.inner_optimizer.clone(),
            rho: self.rho,
            epsilon: self.epsilon,
            adaptive: self.adaptive,
            perturbed_params: self.perturbed_params.clone(),
            original_params: self.original_params.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<A, O, D> Debug for SAM<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone + Debug,
    D: Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SAM")
            .field("inner_optimizer", &self.inner_optimizer)
            .field("rho", &self.rho)
            .field("epsilon", &self.epsilon)
            .field("adaptive", &self.adaptive)
            .finish()
    }
}

impl<A, O, D> Optimizer<A, D> for SAM<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // This single-step implementation is a convenience that combines first_step and second_step
        // However, it's less efficient than using the two-step process because it uses the same gradients
        // for both steps, which doesn't fully implement the SAM algorithm

        // First step: compute perturbed parameters
        let (_, _) = self.first_step(params, gradients)?;

        // Second step: update with the same gradients
        // Note: In a real implementation, you should compute new gradients at the perturbed point
        self.second_step(params, gradients)
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.inner_optimizer.set_learning_rate(learning_rate);
    }

    fn get_learning_rate(&self) -> A {
        self.inner_optimizer.get_learning_rate()
    }
}

/// Calculate the L2 norm of an array
fn calculate_norm<A, D>(array: &Array<A, D>) -> Result<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    let squared_sum = array.iter().fold(A::zero(), |acc, &x| acc + x * x);
    let norm = squared_sum.sqrt();

    if !norm.is_finite() {
        return Err(OptimError::OptimizationError(
            "Norm calculation resulted in non-finite value".to_string(),
        ));
    }

    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::sgd::SGD;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_sam_creation() {
        let sgd = SGD::new(0.01);
        let optimizer: SAM<f64, _, ndarray::Ix1> = SAM::new(sgd);

        assert_abs_diff_eq!(optimizer.rho(), 0.05);
        assert_abs_diff_eq!(optimizer.get_learning_rate(), 0.01);
        assert!(!optimizer.is_adaptive());
    }

    #[test]
    fn test_sam_with_config() {
        let sgd = SGD::new(0.01);
        let optimizer: SAM<f64, _, ndarray::Ix1> = SAM::with_config(sgd, 0.1.into(), true);

        assert_abs_diff_eq!(optimizer.rho(), 0.1);
        assert!(optimizer.is_adaptive());
    }

    #[test]
    fn test_sam_first_step() {
        let sgd = SGD::new(0.1);
        let mut optimizer: SAM<f64, _, ndarray::Ix1> = SAM::new(sgd);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Calculate expected perturbation for standard SAM
        let grad_norm = (0.1f64.powi(2) + 0.2f64.powi(2) + 0.3f64.powi(2)).sqrt();
        let normalized_grads = gradients.mapv(|g| g / grad_norm);
        let expected_perturb = normalized_grads.mapv(|g| g * 0.05);
        let expected_params = &params + &expected_perturb;

        let (perturbed_params, perturb_size) = optimizer.first_step(&params, &gradients).unwrap();

        // Verify perturbed parameters
        assert_abs_diff_eq!(perturbed_params[0], expected_params[0], epsilon = 1e-6);
        assert_abs_diff_eq!(perturbed_params[1], expected_params[1], epsilon = 1e-6);
        assert_abs_diff_eq!(perturbed_params[2], expected_params[2], epsilon = 1e-6);

        // Verify perturbation size
        assert_abs_diff_eq!(perturb_size, 0.05, epsilon = 1e-6);
    }

    #[test]
    fn test_sam_adaptive() {
        let sgd = SGD::new(0.1);
        let mut optimizer: SAM<f64, _, ndarray::Ix1> = SAM::with_config(sgd, 0.05.into(), true);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // With the more stable implementation, we'll just verify the behavior makes sense
        let (perturbed_params, perturb_size) = optimizer.first_step(&params, &gradients).unwrap();

        // Verify perturbed parameters make sense
        assert!(perturb_size > 0.0 && perturb_size < 1.0); // Perturbation size should be reasonable

        // Params should be perturbed in a way that relates to both param magnitude and gradients
        assert!(perturbed_params[0] != params[0]);
        assert!(perturbed_params[1] != params[1]);
        assert!(perturbed_params[2] != params[2]);

        // The perturbation on larger parameters should be larger (relative to their gradients)
        let delta0 = (perturbed_params[0] - params[0]).abs();
        let delta2 = (perturbed_params[2] - params[2]).abs();
        assert!(delta2 > delta0);
    }

    #[test]
    fn test_sam_second_step() {
        let sgd = SGD::new(0.1);
        let mut optimizer: SAM<f64, _, ndarray::Ix1> = SAM::new(sgd);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // First step to set up perturbed parameters
        let (_, _) = optimizer.first_step(&params, &gradients).unwrap();

        // Simulate computing new gradients at perturbed point
        let new_gradients = Array1::from_vec(vec![0.15, 0.25, 0.35]);

        // Second step should update original parameters with new gradients
        let updated_params = optimizer.second_step(&params, &new_gradients).unwrap();

        // Expected update: params - lr * new_gradients
        let expected_params =
            Array1::from_vec(vec![1.0 - 0.1 * 0.15, 2.0 - 0.1 * 0.25, 3.0 - 0.1 * 0.35]);

        assert_abs_diff_eq!(updated_params[0], expected_params[0], epsilon = 1e-6);
        assert_abs_diff_eq!(updated_params[1], expected_params[1], epsilon = 1e-6);
        assert_abs_diff_eq!(updated_params[2], expected_params[2], epsilon = 1e-6);
    }

    #[test]
    fn test_sam_reset() {
        let sgd = SGD::new(0.1);
        let mut optimizer: SAM<f64, _, ndarray::Ix1> = SAM::new(sgd);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // First step
        let _ = optimizer.first_step(&params, &gradients).unwrap();

        // Reset
        optimizer.reset();

        // Should fail because first_step needs to be called before second_step
        let result = optimizer.second_step(&params, &gradients);
        assert!(result.is_err());
    }

    #[test]
    fn test_sam_error_handling() {
        let sgd = SGD::new(0.1);
        let mut optimizer: SAM<f64, _, ndarray::Ix1> = SAM::new(sgd);

        // Gradient with all zeros should return error
        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let zero_gradients = Array1::zeros(3);

        let result = optimizer.first_step(&params, &zero_gradients);
        assert!(result.is_err());
    }
}
