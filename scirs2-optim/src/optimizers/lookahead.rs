//! Lookahead optimizer
//!
//! Implements the Lookahead optimization algorithm from:
//! "Lookahead Optimizer: k steps forward, 1 step back" (Zhang et al., 2019)

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Lookahead optimizer
///
/// Implements the "Lookahead Optimizer: k steps forward, 1 step back" algorithm.
/// This optimizer maintains two sets of weights: "fast" weights that are updated by
/// an inner optimizer, and "slow" weights that follow behind at a controlled pace.
///
/// The algorithm proceeds by:
/// 1. Starting with both sets of weights synchronized
/// 2. Letting the fast weights explore using the inner optimizer for k steps
/// 3. Then updating the slow weights to move partially toward the fast weights
/// 4. Resetting the fast weights back to the slow weights
/// 5. Repeating this process
///
/// This provides more stable optimization by allowing aggressive exploration while
/// maintaining a conservative trajectory.
///
/// # Parameters
///
/// * `inner_optimizer` - The optimizer to use for fast weight updates
/// * `alpha` - The step size for slow weight updates (default: 0.5)
/// * `k` - The number of fast weight updates before updating slow weights (default: 5)
///
/// # Example
///
/// ```
/// use ndarray::Array1;
/// use scirs2_optim::optimizers::{Lookahead, SGD};
/// use scirs2_optim::Optimizer;
///
/// // Create an inner optimizer
/// let sgd = SGD::new(0.01);
///
/// // Wrap it with Lookahead
/// let mut optimizer = Lookahead::new(sgd);
///
/// // Use like any other optimizer
/// let params = Array1::zeros(10);
/// let gradients = Array1::ones(10);
/// let updated_params = optimizer.step(&params, &gradients).unwrap();
/// ```
pub struct Lookahead<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone,
    D: Dimension,
{
    /// Inner optimizer for fast weights
    inner_optimizer: O,
    /// Step size for slow weights update (alpha)
    alpha: A,
    /// Synchronization period (k)
    k: usize,
    /// Current step counter
    current_step: usize,
    /// Slow weights
    slow_weights: Option<Array<A, D>>,
    /// Fast weights
    fast_weights: Option<Array<A, D>>,
    /// Use slow weights for evaluation
    use_slow_weights: bool,
    /// Dimension type marker
    _phantom: PhantomData<D>,
}

impl<A, O, D> Lookahead<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone,
    D: Dimension,
{
    /// Creates a new Lookahead optimizer with the given inner optimizer and default settings
    pub fn new(inner_optimizer: O) -> Self {
        Self {
            inner_optimizer,
            alpha: A::from(0.5).unwrap(), // Default alpha is 0.5
            k: 5,                         // Default k is 5
            current_step: 0,
            slow_weights: None,
            fast_weights: None,
            use_slow_weights: false,
            _phantom: PhantomData,
        }
    }

    /// Creates a new Lookahead optimizer with the specified alpha and k values
    pub fn with_config(inner_optimizer: O, alpha: A, k: usize) -> Self {
        Self {
            inner_optimizer,
            alpha,
            k,
            current_step: 0,
            slow_weights: None,
            fast_weights: None,
            use_slow_weights: false,
            _phantom: PhantomData,
        }
    }

    /// Set the alpha parameter (slow weights step size)
    pub fn with_alpha(mut self, alpha: A) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the k parameter (synchronization period)
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
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

    /// Get the alpha parameter (slow weights step size)
    pub fn alpha(&self) -> A {
        self.alpha
    }

    /// Get the k parameter (synchronization period)
    pub fn k(&self) -> usize {
        self.k
    }

    /// Switches to using slow weights for evaluation
    /// Call this before evaluation to get better performance
    pub fn use_slow_weights_for_eval(&mut self) {
        self.use_slow_weights = true;
    }

    /// Switches to using fast weights for training
    /// Call this after evaluation to resume training
    pub fn use_fast_weights_for_train(&mut self) {
        self.use_slow_weights = false;
    }

    /// Resets the internal state
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.slow_weights = None;
        self.fast_weights = None;
    }
}

impl<A, O, D> Clone for Lookahead<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone,
    D: Dimension,
{
    fn clone(&self) -> Self {
        Self {
            inner_optimizer: self.inner_optimizer.clone(),
            alpha: self.alpha,
            k: self.k,
            current_step: self.current_step,
            slow_weights: self.slow_weights.clone(),
            fast_weights: self.fast_weights.clone(),
            use_slow_weights: self.use_slow_weights,
            _phantom: PhantomData,
        }
    }
}

impl<A, O, D> Debug for Lookahead<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone + Debug,
    D: Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lookahead")
            .field("inner_optimizer", &self.inner_optimizer)
            .field("alpha", &self.alpha)
            .field("k", &self.k)
            .field("current_step", &self.current_step)
            .field("use_slow_weights", &self.use_slow_weights)
            .finish()
    }
}

impl<A, O, D> Optimizer<A, D> for Lookahead<A, O, D>
where
    A: Float + ScalarOperand + Debug,
    O: Optimizer<A, D> + Clone,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        // Initialize weights if first step
        if self.slow_weights.is_none() {
            self.slow_weights = Some(params.clone());
            self.fast_weights = Some(params.clone());
        }

        // Get mutable references to weights
        let fast_weights = match &mut self.fast_weights {
            Some(w) => w,
            None => {
                return Err(OptimError::OptimizationError(
                    "Fast weights not initialized".to_string(),
                ))
            }
        };

        let slow_weights = match &mut self.slow_weights {
            Some(w) => w,
            None => {
                return Err(OptimError::OptimizationError(
                    "Slow weights not initialized".to_string(),
                ))
            }
        };

        // Update fast weights using inner optimizer
        *fast_weights = self.inner_optimizer.step(fast_weights, gradients)?;

        // Increment step counter
        self.current_step += 1;

        // If we've reached k steps, update slow weights and reset fast weights
        if self.current_step >= self.k {
            // Update slow weights: φₜ ← φₜ₋₁ + α(θₜ,ₖ - φₜ₋₁)
            // Compute difference between fast and slow weights
            let diff = &*fast_weights - &*slow_weights;

            // Update slow weights by moving alpha of the way toward fast weights
            *slow_weights = &*slow_weights + &(diff * self.alpha);

            // Reset fast weights to slow weights
            *fast_weights = slow_weights.clone();

            // Reset step counter
            self.current_step = 0;
        }

        // Return the appropriate weights (slow for evaluation, fast for training)
        if self.use_slow_weights {
            Ok(slow_weights.clone())
        } else {
            Ok(fast_weights.clone())
        }
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.inner_optimizer.set_learning_rate(learning_rate);
    }

    fn get_learning_rate(&self) -> A {
        self.inner_optimizer.get_learning_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::sgd::SGD;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_lookahead_creation() {
        let sgd = SGD::new(0.01);
        let optimizer: Lookahead<f64, _, ndarray::Ix1> = Lookahead::new(sgd);

        assert_abs_diff_eq!(optimizer.alpha(), 0.5);
        assert_eq!(optimizer.k(), 5);
        assert_abs_diff_eq!(optimizer.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_lookahead_with_config() {
        let sgd = SGD::new(0.01);
        let optimizer: Lookahead<f64, _, ndarray::Ix1> =
            Lookahead::with_config(sgd, 0.8.into(), 10);

        assert_abs_diff_eq!(optimizer.alpha(), 0.8);
        assert_eq!(optimizer.k(), 10);
    }

    #[test]
    fn test_lookahead_step() {
        let mut sgd = SGD::new(0.1);
        sgd.set_momentum(0.0);
        let mut optimizer: Lookahead<f64, _, ndarray::Ix1> =
            Lookahead::with_config(sgd, 0.5.into(), 2);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // First step
        let updated_params = optimizer.step(&params, &gradients).unwrap();

        // After first step, fast weights should be updated by SGD but slow weights unchanged
        // SGD update: params - lr * gradients = [1.0, 2.0, 3.0] - 0.1 * [0.1, 0.2, 0.3] = [0.99, 1.98, 2.97]
        assert_abs_diff_eq!(updated_params[0], 0.99, epsilon = 1e-6);
        assert_abs_diff_eq!(updated_params[1], 1.98, epsilon = 1e-6);
        assert_abs_diff_eq!(updated_params[2], 2.97, epsilon = 1e-6);

        // Second step
        let updated_params2 = optimizer.step(&updated_params, &gradients).unwrap();

        // After second step (which is k), slow weights should be updated and fast weights reset to slow weights
        // SGD update on fast weights: [0.99, 1.98, 2.97] - 0.1 * [0.1, 0.2, 0.3] = [0.98, 1.96, 2.94]
        // Slow weights update: [1.0, 2.0, 3.0] + 0.5 * ([0.98, 1.96, 2.94] - [1.0, 2.0, 3.0])
        //                    = [1.0, 2.0, 3.0] + 0.5 * [-0.02, -0.04, -0.06]
        //                    = [0.99, 1.98, 2.97]
        // Fast weights are reset to slow weights = [0.99, 1.98, 2.97]

        // The returned value should be the fast weights (which are now reset to slow weights)
        assert_abs_diff_eq!(updated_params2[0], 0.99, epsilon = 1e-6);
        assert_abs_diff_eq!(updated_params2[1], 1.98, epsilon = 1e-6);
        assert_abs_diff_eq!(updated_params2[2], 2.97, epsilon = 1e-6);

        // Third step (starting a new cycle)
        let updated_params3 = optimizer.step(&updated_params2, &gradients).unwrap();

        // SGD update on fast weights: [0.99, 1.98, 2.97] - 0.1 * [0.1, 0.2, 0.3] = [0.98, 1.96, 2.94]
        assert_abs_diff_eq!(updated_params3[0], 0.98, epsilon = 1e-6);
        assert_abs_diff_eq!(updated_params3[1], 1.96, epsilon = 1e-6);
        assert_abs_diff_eq!(updated_params3[2], 2.94, epsilon = 1e-6);
    }

    #[test]
    fn test_slow_weights_for_eval() {
        let mut sgd = SGD::new(0.1);
        sgd.set_momentum(0.0);
        let mut optimizer: Lookahead<f64, _, ndarray::Ix1> =
            Lookahead::with_config(sgd, 0.5.into(), 2);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // First step
        let updated_params = optimizer.step(&params, &gradients).unwrap();

        // Switch to slow weights for evaluation
        optimizer.use_slow_weights_for_eval();

        // Get the parameters when using slow weights
        let eval_params = optimizer.step(&updated_params, &gradients).unwrap();

        // First step already updated both fast and slow weights
        // When using slow weights, we should get the slow weights which were initialized with
        // values from params: [1.0, 2.0, 3.0] but then updated by the first step
        assert_abs_diff_eq!(eval_params[0], 0.99, epsilon = 1e-6);
        assert_abs_diff_eq!(eval_params[1], 1.98, epsilon = 1e-6);
        assert_abs_diff_eq!(eval_params[2], 2.97, epsilon = 1e-6);

        // Switch back to fast weights for training
        optimizer.use_fast_weights_for_train();

        // Should be back to fast weights
        let train_params = optimizer.step(&eval_params, &gradients).unwrap();
        assert!(train_params[0] < 1.0);
    }

    #[test]
    fn test_reset() {
        let sgd = SGD::new(0.1);
        let mut optimizer: Lookahead<f64, _, ndarray::Ix1> = Lookahead::new(sgd);

        let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Do a step to initialize weights
        let _ = optimizer.step(&params, &gradients).unwrap();

        // Reset
        optimizer.reset();

        // Both fast and slow weights should be None, verified by new initialization
        let updated_params = optimizer.step(&params, &gradients).unwrap();
        // First step after reset should be equivalent to first step on a new optimizer
        assert_abs_diff_eq!(updated_params[0], 0.99, epsilon = 1e-6);
    }
}
