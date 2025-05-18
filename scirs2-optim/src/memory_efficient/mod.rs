//! Memory-efficient optimizers and utilities
//!
//! This module provides in-place parameter update capabilities and
//! memory-efficient implementations of optimization algorithms.

use crate::error::{OptimError, Result};
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign, SubAssign};

/// Trait for in-place parameter updates
pub trait InPlaceOptimizer<A: Float + ScalarOperand + Debug, D: Dimension> {
    /// Update parameters in-place using the given gradients
    ///
    /// This method modifies the parameters directly rather than returning new arrays,
    /// which can significantly reduce memory usage for large models.
    fn step_inplace(&mut self, params: &mut Array<A, D>, gradients: &Array<A, D>) -> Result<()>;

    /// Update multiple parameter arrays in-place
    fn step_list_inplace(
        &mut self,
        params_list: &mut [&mut Array<A, D>],
        gradients_list: &[&Array<A, D>],
    ) -> Result<()> {
        if params_list.len() != gradients_list.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Number of parameter arrays ({}) does not match number of gradient arrays ({})",
                params_list.len(),
                gradients_list.len()
            )));
        }

        for (params, grads) in params_list.iter_mut().zip(gradients_list.iter()) {
            self.step_inplace(params, grads)?;
        }
        Ok(())
    }
}

/// Memory-efficient SGD optimizer with in-place updates
#[derive(Debug, Clone)]
pub struct InPlaceSGD<A: Float> {
    learning_rate: A,
    momentum: A,
    weight_decay: A,
}

impl<A: Float + ScalarOperand + Debug> InPlaceSGD<A> {
    /// Create a new in-place SGD optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            momentum: A::zero(),
            weight_decay: A::zero(),
        }
    }

    /// Set momentum
    pub fn with_momentum(mut self, momentum: A) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> InPlaceOptimizer<A, D> for InPlaceSGD<A> {
    fn step_inplace(&mut self, params: &mut Array<A, D>, gradients: &Array<A, D>) -> Result<()> {
        // Apply weight decay if configured
        if self.weight_decay > A::zero() {
            params.zip_mut_with(gradients, |p, &g| {
                *p = *p - self.learning_rate * (g + *p * self.weight_decay);
            });
        } else {
            // Simple gradient descent
            params.zip_mut_with(gradients, |p, &g| {
                *p = *p - self.learning_rate * g;
            });
        }
        Ok(())
    }
}

/// Memory-efficient Adam optimizer with in-place updates
#[derive(Debug)]
pub struct InPlaceAdam<A: Float, D: Dimension> {
    learning_rate: A,
    beta1: A,
    beta2: A,
    epsilon: A,
    weight_decay: A,
    t: i32,
    /// First moment estimate (momentum)
    m: Option<Array<A, D>>,
    /// Second moment estimate (RMSprop)
    v: Option<Array<A, D>>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> InPlaceAdam<A, D> {
    /// Create a new in-place Adam optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).unwrap(),
            beta2: A::from(0.999).unwrap(),
            epsilon: A::from(1e-8).unwrap(),
            weight_decay: A::zero(),
            t: 0,
            m: None,
            v: None,
        }
    }

    /// Set beta1 (momentum decay)
    pub fn with_beta1(mut self, beta1: A) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (RMSprop decay)
    pub fn with_beta2(mut self, beta2: A) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set epsilon
    pub fn with_epsilon(mut self, epsilon: A) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.t = 0;
        self.m = None;
        self.v = None;
    }
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> InPlaceOptimizer<A, D> for InPlaceAdam<A, D> {
    fn step_inplace(&mut self, params: &mut Array<A, D>, gradients: &Array<A, D>) -> Result<()> {
        self.t += 1;
        let _t = A::from(self.t).unwrap();

        // Initialize momentum and variance if needed
        if self.m.is_none() {
            self.m = Some(Array::zeros(params.raw_dim()));
        }
        if self.v.is_none() {
            self.v = Some(Array::zeros(params.raw_dim()));
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Apply weight decay if configured
        let grad_with_decay = if self.weight_decay > A::zero() {
            // Create temporary with weight decay
            let mut temp = gradients.clone();
            temp.zip_mut_with(params, |g, &p| {
                *g = *g + p * self.weight_decay;
            });
            temp
        } else {
            gradients.clone()
        };

        // Update biased first moment estimate
        m.zip_mut_with(&grad_with_decay, |m_i, &g| {
            *m_i = self.beta1 * *m_i + (A::one() - self.beta1) * g;
        });

        // Update biased second raw moment estimate
        v.zip_mut_with(&grad_with_decay, |v_i, &g| {
            *v_i = self.beta2 * *v_i + (A::one() - self.beta2) * g * g;
        });

        // Compute bias-corrected moments
        let bias1 = A::one() - self.beta1.powi(self.t);
        let bias2 = A::one() - self.beta2.powi(self.t);

        // Update parameters in-place
        let m_iter = m.iter();
        let v_iter = v.iter();
        let params_iter = params.iter_mut();

        for ((p, &m_i), &v_i) in params_iter.zip(m_iter).zip(v_iter) {
            let m_hat = m_i / bias1;
            let v_hat = v_i / bias2;
            *p = *p - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }

        Ok(())
    }
}

/// Utility functions for memory-efficient operations
pub mod utils {
    use super::*;

    /// Apply a scalar operation in-place
    pub fn scale_inplace<A, D>(array: &mut Array<A, D>, scalar: A)
    where
        A: Float + ScalarOperand + MulAssign,
        D: Dimension,
    {
        array.map_inplace(|x| *x *= scalar);
    }

    /// Add arrays in-place (a += b)
    pub fn add_inplace<A, D>(a: &mut Array<A, D>, b: &Array<A, D>)
    where
        A: Float + ScalarOperand + AddAssign,
        D: Dimension,
    {
        a.zip_mut_with(b, |x, &y| *x += y);
    }

    /// Subtract arrays in-place (a -= b)
    pub fn subtract_inplace<A, D>(a: &mut Array<A, D>, b: &Array<A, D>)
    where
        A: Float + ScalarOperand + SubAssign,
        D: Dimension,
    {
        a.zip_mut_with(b, |x, &y| *x -= y);
    }

    /// Apply element-wise operation in-place
    pub fn apply_inplace<A, D, F>(array: &mut Array<A, D>, f: F)
    where
        A: Float + ScalarOperand,
        D: Dimension,
        F: Fn(&mut A),
    {
        array.map_inplace(f);
    }

    /// Clip values in-place
    pub fn clip_inplace<A, D>(array: &mut Array<A, D>, min: A, max: A)
    where
        A: Float + ScalarOperand,
        D: Dimension,
    {
        array.map_inplace(|x| {
            if *x < min {
                *x = min;
            } else if *x > max {
                *x = max;
            }
        });
    }

    /// Normalize array in-place (divide by its norm)
    pub fn normalize_inplace<A, D>(array: &mut Array<A, D>)
    where
        A: Float + ScalarOperand + MulAssign,
        D: Dimension,
    {
        let norm = array.mapv(|x| x * x).sum().sqrt();
        if norm > A::zero() {
            array.map_inplace(|x| *x *= A::one() / norm);
        }
    }
}

// Re-export utility functions at module level for convenience
pub use utils::{
    add_inplace, apply_inplace, clip_inplace, normalize_inplace, scale_inplace, subtract_inplace,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_inplace_sgd() {
        let mut optimizer = InPlaceSGD::new(0.1);
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        optimizer.step_inplace(&mut params, &gradients).unwrap();

        assert_relative_eq!(params[0], 0.99, epsilon = 1e-6);
        assert_relative_eq!(params[1], 1.98, epsilon = 1e-6);
        assert_relative_eq!(params[2], 2.97, epsilon = 1e-6);
    }

    #[test]
    fn test_inplace_adam() {
        let mut optimizer = InPlaceAdam::new(0.001);
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Multiple steps to see momentum effects
        for _ in 0..5 {
            optimizer.step_inplace(&mut params, &gradients).unwrap();
        }

        // Verify parameters have been updated
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert!(params[2] < 3.0);
    }

    #[test]
    fn test_utils_scale_inplace() {
        let mut array = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        utils::scale_inplace(&mut array, 2.0);

        assert_eq!(array.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_utils_clip_inplace() {
        let mut array = Array1::from_vec(vec![0.5, 1.5, 2.5]);
        utils::clip_inplace(&mut array, 1.0, 2.0);

        assert_eq!(array.as_slice().unwrap(), &[1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_memory_efficiency() {
        // Test that in-place operations don't allocate new arrays
        let mut params = Array1::from_vec(vec![1.0; 1000]);
        let gradients = Array1::from_vec(vec![0.01; 1000]);
        let params_ptr = params.as_ptr();

        let mut optimizer = InPlaceSGD::new(0.1);
        optimizer.step_inplace(&mut params, &gradients).unwrap();

        // Verify the same memory is being used
        assert_eq!(params_ptr, params.as_ptr());
    }
}
