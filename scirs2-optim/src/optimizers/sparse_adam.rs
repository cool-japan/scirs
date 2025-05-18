//! SparseAdam optimizer implementation for sparse gradients
//!
//! This module implements a variant of the Adam optimizer that efficiently
//! handles sparse gradients by only updating the parameters and moments
//! for indices that have non-zero gradients.

use ndarray::{Array, Ix1, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;

/// A struct representing a sparse gradient with indices and values
///
/// This provides a convenient interface for working with sparse gradients
/// where most elements are zero.
pub struct SparseGradient<A: Float + ScalarOperand + Debug> {
    /// The indices of non-zero elements
    pub indices: Vec<usize>,
    /// The values at the non-zero elements
    pub values: Vec<A>,
    /// The total dimension of the gradient (including zero elements)
    pub dim: usize,
}

impl<A: Float + ScalarOperand + Debug> SparseGradient<A> {
    /// Create a new sparse gradient from indices and values
    pub fn new(indices: Vec<usize>, values: Vec<A>, dim: usize) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Indices and values must have the same length"
        );
        // Ensure no index is out of bounds
        if let Some(&max_idx) = indices.iter().max() {
            assert!(
                max_idx < dim,
                "Index {} is out of bounds for dimension {}",
                max_idx,
                dim
            );
        }
        Self {
            indices,
            values,
            dim,
        }
    }

    /// Create a sparse gradient from a dense array, keeping only non-zero entries
    pub fn from_array(array: &Array<A, Ix1>) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (idx, &val) in array.iter().enumerate() {
            if !val.is_zero() {
                indices.push(idx);
                values.push(val);
            }
        }

        Self {
            indices,
            values,
            dim: array.len(),
        }
    }

    /// Convert the sparse gradient to a dense array
    pub fn to_array(&self) -> Array<A, Ix1> {
        let mut array = Array::zeros(self.dim);
        for (&idx, &val) in self.indices.iter().zip(&self.values) {
            array[idx] = val;
        }
        array
    }

    /// Check if this sparse gradient is empty (all zeros)
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// SparseAdam optimizer for sparse gradients
///
/// Implements a variant of the Adam optimization algorithm that's optimized for
/// sparse gradients. It only updates the parameters and momentum vectors
/// for indices that have non-zero gradients, saving computation and memory.
///
/// This optimizer is particularly useful for large embedding layers or
/// models with sparse input features.
///
/// Formula (for non-zero gradient indices):
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m_hat_t = m_t / (1 - beta1^t)
/// v_hat_t = v_t / (1 - beta2^t)
/// theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{SparseAdam, SparseGradient, Optimizer};
///
/// // Initialize parameters
/// let params = Array1::zeros(5);
///
/// // Create sparse gradient with non-zero values at indices 1 and 3
/// let sparse_grad = SparseGradient::new(
///     vec![1, 3],             // Indices
///     vec![0.2, 0.5],         // Values
///     5                       // Total dimension
/// );
///
/// // Create a SparseAdam optimizer
/// let mut optimizer = SparseAdam::new(0.001);
///
/// // Update parameters with sparse gradient
/// let new_params = optimizer.step_sparse(&params, &sparse_grad).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseAdam<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// Exponential decay rate for the first moment estimates
    beta1: A,
    /// Exponential decay rate for the second moment estimates
    beta2: A,
    /// Small constant for numerical stability
    epsilon: A,
    /// Weight decay factor (L2 regularization)
    weight_decay: A,
    /// First moment vector stored as a hash map for sparse updates
    m: HashMap<usize, A>,
    /// Second moment vector stored as a hash map for sparse updates
    v: HashMap<usize, A>,
    /// Current timestep
    t: usize,
}

impl<A: Float + ScalarOperand + Debug> SparseAdam<A> {
    /// Creates a new SparseAdam optimizer with the given learning rate and default settings
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).unwrap(),
            beta2: A::from(0.999).unwrap(),
            epsilon: A::from(1e-8).unwrap(),
            weight_decay: A::zero(),
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Creates a new SparseAdam optimizer with the full configuration
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for the first moment estimates (default: 0.9)
    /// * `beta2` - Exponential decay rate for the second moment estimates (default: 0.999)
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    /// * `weight_decay` - Weight decay factor for L2 regularization (default: 0.0)
    pub fn new_with_config(
        learning_rate: A,
        beta1: A,
        beta2: A,
        epsilon: A,
        weight_decay: A,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Sets the beta1 parameter
    pub fn set_beta1(&mut self, beta1: A) -> &mut Self {
        self.beta1 = beta1;
        self
    }

    /// Builder method to set beta1 and return self
    pub fn with_beta1(mut self, beta1: A) -> Self {
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

    /// Builder method to set beta2 and return self
    pub fn with_beta2(mut self, beta2: A) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Gets the beta2 parameter
    pub fn get_beta2(&self) -> A {
        self.beta2
    }

    /// Sets the epsilon parameter
    pub fn set_epsilon(&mut self, epsilon: A) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Builder method to set epsilon and return self
    pub fn with_epsilon(mut self, epsilon: A) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Gets the epsilon parameter
    pub fn get_epsilon(&self) -> A {
        self.epsilon
    }

    /// Sets the weight decay parameter
    pub fn set_weight_decay(&mut self, weight_decay: A) -> &mut Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Builder method to set weight decay and return self
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Gets the weight decay parameter
    pub fn get_weight_decay(&self) -> A {
        self.weight_decay
    }

    /// Updates parameters using sparse gradients
    ///
    /// This method efficiently updates only the parameters corresponding to
    /// non-zero gradient entries, saving computation and memory.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters to update
    /// * `gradient` - The sparse gradient
    ///
    /// # Returns
    ///
    /// The updated parameters
    pub fn step_sparse(
        &mut self,
        params: &Array<A, Ix1>,
        gradient: &SparseGradient<A>,
    ) -> Result<Array<A, Ix1>> {
        // Verify dimensions match
        if params.len() != gradient.dim {
            return Err(OptimError::InvalidConfig(format!(
                "Parameter dimension ({}) doesn't match gradient dimension ({})",
                params.len(),
                gradient.dim
            )));
        }

        // If gradient is empty, just return the parameters unchanged
        if gradient.is_empty() {
            return Ok(params.clone());
        }

        // Increment timestep
        self.t += 1;

        // Compute the bias correction terms
        let bias_correction1 = A::one() - self.beta1.powi(self.t as i32);
        let bias_correction2 = A::one() - self.beta2.powi(self.t as i32);

        // Create a copy of the parameters that we'll update
        let mut updated_params = params.clone();

        // Update only the parameters for which we have non-zero gradients
        for (&idx, &grad_val) in gradient.indices.iter().zip(&gradient.values) {
            // Apply weight decay if needed
            let adjusted_grad = if self.weight_decay > A::zero() {
                grad_val + params[idx] * self.weight_decay
            } else {
                grad_val
            };

            // Update first moment (m)
            let m_prev = *self.m.get(&idx).unwrap_or(&A::zero());
            let m_t = self.beta1 * m_prev + (A::one() - self.beta1) * adjusted_grad;
            self.m.insert(idx, m_t);

            // Update second moment (v)
            let v_prev = *self.v.get(&idx).unwrap_or(&A::zero());
            let v_t = self.beta2 * v_prev + (A::one() - self.beta2) * adjusted_grad * adjusted_grad;
            self.v.insert(idx, v_t);

            // Bias-corrected first and second moment estimates
            let m_hat = m_t / bias_correction1;
            let v_hat = v_t / bias_correction2;

            // Update parameter
            let step = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            updated_params[idx] = params[idx] - step;
        }

        Ok(updated_params)
    }

    /// Resets the internal state of the optimizer
    pub fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

impl<A> Optimizer<A, Ix1> for SparseAdam<A>
where
    A: Float + ScalarOperand + Debug,
{
    fn step(&mut self, params: &Array<A, Ix1>, gradients: &Array<A, Ix1>) -> Result<Array<A, Ix1>> {
        // Convert dense gradient to sparse
        let sparse_gradient = SparseGradient::from_array(gradients);

        // Call step_sparse with the converted gradient
        self.step_sparse(params, &sparse_gradient)
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
    fn test_sparse_gradient_creation() {
        let indices = vec![0, 2, 4];
        let values = vec![1.0, 2.0, 3.0];
        let dim = 5;

        let sparse_grad = SparseGradient::new(indices, values, dim);

        assert_eq!(sparse_grad.indices, vec![0, 2, 4]);
        assert_eq!(sparse_grad.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(sparse_grad.dim, 5);
    }

    #[test]
    fn test_sparse_gradient_from_array() {
        let dense = Array1::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let sparse_grad = SparseGradient::from_array(&dense);

        assert_eq!(sparse_grad.indices, vec![0, 2, 4]);
        assert_eq!(sparse_grad.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(sparse_grad.dim, 5);
    }

    #[test]
    fn test_sparse_gradient_to_array() {
        let indices = vec![0, 2, 4];
        let values = vec![1.0, 2.0, 3.0];
        let dim = 5;

        let sparse_grad = SparseGradient::new(indices, values, dim);
        let dense = sparse_grad.to_array();

        let expected = Array1::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        assert_eq!(dense, expected);
    }

    #[test]
    fn test_sparse_adam_creation() {
        let optimizer = SparseAdam::<f64>::new(0.001);

        assert_eq!(optimizer.get_learning_rate(), 0.001);
        assert_eq!(optimizer.get_beta1(), 0.9);
        assert_eq!(optimizer.get_beta2(), 0.999);
        assert_eq!(optimizer.get_epsilon(), 1e-8);
        assert_eq!(optimizer.get_weight_decay(), 0.0);
    }

    #[test]
    fn test_sparse_adam_step() {
        let mut optimizer = SparseAdam::<f64>::new(0.1);

        // Initialize parameters
        let params = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        // Create sparse gradient with non-zero values at indices 1 and 3
        let sparse_grad = SparseGradient::new(
            vec![1, 3],     // Indices
            vec![0.2, 0.5], // Values
            5,              // Total dimension
        );

        // First update
        let updated_params = optimizer.step_sparse(&params, &sparse_grad).unwrap();

        // Only the parameters at indices 1 and 3 should be updated
        assert_abs_diff_eq!(updated_params[0], 0.0);
        assert!(updated_params[1] < 0.0); // Should be negative due to gradient descent
        assert_abs_diff_eq!(updated_params[2], 0.0);
        assert!(updated_params[3] < 0.0); // Should be negative due to gradient descent
        assert_abs_diff_eq!(updated_params[4], 0.0);

        // The parameter at index 3 should have a larger update due to larger gradient
        assert!(updated_params[3].abs() > updated_params[1].abs());
    }

    #[test]
    fn test_sparse_adam_vs_dense_adam() {
        let mut sparse_optimizer = SparseAdam::<f64>::new(0.1);
        let mut dense_optimizer = crate::optimizers::adam::Adam::<f64>::new(0.1);

        // Initialize parameters
        let params = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        // Create a dense gradient with only some non-zero values
        let dense_grad = Array1::from_vec(vec![0.0, 0.2, 0.0, 0.5, 0.0]);

        // Create equivalent sparse gradient
        let sparse_grad = SparseGradient::from_array(&dense_grad);

        // Update with both optimizers
        let sparse_result = sparse_optimizer.step_sparse(&params, &sparse_grad).unwrap();
        let dense_result = dense_optimizer.step(&params, &dense_grad).unwrap();

        // Results should be nearly identical
        assert_abs_diff_eq!(sparse_result[0], dense_result[0]);
        assert_abs_diff_eq!(sparse_result[1], dense_result[1], epsilon = 1e-10);
        assert_abs_diff_eq!(sparse_result[2], dense_result[2]);
        assert_abs_diff_eq!(sparse_result[3], dense_result[3], epsilon = 1e-10);
        assert_abs_diff_eq!(sparse_result[4], dense_result[4]);
    }

    #[test]
    fn test_sparse_adam_multiple_steps() {
        let mut optimizer = SparseAdam::<f64>::new(0.1);
        let mut params = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);

        // First step - update indices 1 and 3
        let sparse_grad1 = SparseGradient::new(
            vec![1, 3],     // Indices
            vec![0.2, 0.5], // Values
            5,              // Total dimension
        );

        params = optimizer.step_sparse(&params, &sparse_grad1).unwrap();

        // Second step - update indices 0 and 2
        let sparse_grad2 = SparseGradient::new(
            vec![0, 2],     // Indices
            vec![0.3, 0.4], // Values
            5,              // Total dimension
        );

        params = optimizer.step_sparse(&params, &sparse_grad2).unwrap();

        // All parameters except index 4 should now be updated
        assert!(params[0] < 0.0);
        assert!(params[1] < 0.0);
        assert!(params[2] < 0.0);
        assert!(params[3] < 0.0);
        assert_abs_diff_eq!(params[4], 0.0);

        // Third step - update the same indices again (accumulates momentum)
        params = optimizer.step_sparse(&params, &sparse_grad2).unwrap();

        // Parameters at indices 0 and 2 should have larger updates now
        let prev_param0 = params[0];
        let prev_param2 = params[2];

        params = optimizer.step_sparse(&params, &sparse_grad2).unwrap();

        assert!(params[0].abs() > prev_param0.abs());
        assert!(params[2].abs() > prev_param2.abs());
    }

    #[test]
    fn test_sparse_adam_with_weight_decay() {
        let mut optimizer = SparseAdam::<f64>::new(0.1).with_weight_decay(0.01);

        // Initialize parameters with non-zero values
        let params = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        // Create sparse gradient
        let sparse_grad = SparseGradient::new(
            vec![1, 3],     // Indices
            vec![0.2, 0.5], // Values
            5,              // Total dimension
        );

        // Create a version without weight decay for comparison
        let mut optimizer_no_decay = SparseAdam::<f64>::new(0.1);

        let with_decay = optimizer.step_sparse(&params, &sparse_grad).unwrap();
        let without_decay = optimizer_no_decay
            .step_sparse(&params, &sparse_grad)
            .unwrap();

        // Parameters with non-zero gradients should be different when weight decay is applied
        assert!(with_decay[1] != without_decay[1]);
        assert!(with_decay[3] != without_decay[3]);

        // Parameters without gradients should remain the same
        assert_abs_diff_eq!(with_decay[0], params[0]);
        assert_abs_diff_eq!(with_decay[2], params[2]);
        assert_abs_diff_eq!(with_decay[4], params[4]);
    }

    #[test]
    fn test_sparse_adam_empty_gradient() {
        let mut optimizer = SparseAdam::<f64>::new(0.1);

        // Initialize parameters
        let params = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        // Create an empty sparse gradient
        let sparse_grad = SparseGradient::new(
            vec![], // Empty indices
            vec![], // Empty values
            5,      // Total dimension
        );

        // No parameters should change
        let result = optimizer.step_sparse(&params, &sparse_grad).unwrap();
        assert_eq!(result, params);
    }

    #[test]
    fn test_sparse_adam_reset() {
        let mut optimizer = SparseAdam::<f64>::new(0.1);

        // Initialize parameters
        let params = Array1::from_vec(vec![0.0; 5]);

        // Create sparse gradient
        let sparse_grad = SparseGradient::new(
            vec![1, 3],     // Indices
            vec![0.2, 0.5], // Values
            5,              // Total dimension
        );

        // Do several steps to build up momentum
        for _ in 0..10 {
            optimizer.step_sparse(&params, &sparse_grad).unwrap();
        }

        // Reset optimizer
        optimizer.reset();

        // The next step should be the same as the first step with a new optimizer
        let mut new_optimizer = SparseAdam::<f64>::new(0.1);
        let reset_result = optimizer.step_sparse(&params, &sparse_grad).unwrap();
        let new_result = new_optimizer.step_sparse(&params, &sparse_grad).unwrap();

        assert_abs_diff_eq!(reset_result[1], new_result[1], epsilon = 1e-10);
        assert_abs_diff_eq!(reset_result[3], new_result[3], epsilon = 1e-10);
    }
}
