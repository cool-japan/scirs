//! Exponential learning rate scheduler

use crate::schedulers::LRScheduler;
use crate::Float;

/// Exponential learning rate scheduler
///
/// Decays the learning rate exponentially by a factor `gamma` at each step.
/// This provides a smooth, continuous decay of the learning rate.
///
/// The learning rate is calculated as:
/// `lr = initial_lr * gamma^step`
///
/// # Example
/// ```
/// use scirs2_autograd::schedulers::{ExponentialLR, LRScheduler};
///
/// let scheduler = ExponentialLR::new(0.1f32, 0.95f32);
///
/// // Initial learning rate
/// assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-6);
///
/// // After 1 step
/// assert!((scheduler.get_lr(1) - 0.095).abs() < 1e-6);
///
/// // After 10 steps
/// // lr ≈ 0.1 * 0.95^10 ≈ 0.0599
/// let lr_10 = scheduler.get_lr(10);
/// assert!((lr_10 - 0.0599).abs() < 1e-3);
/// ```
pub struct ExponentialLR<F: Float> {
    /// Initial learning rate
    pub initial_lr: F,
    /// Decay factor applied at each step
    pub gamma: F,
}

impl<F: Float> ExponentialLR<F> {
    /// Create a new exponential learning rate scheduler
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    /// * `gamma` - Multiplicative factor of learning rate decay at each step (should be < 1.0 for decay)
    ///
    /// # Panics
    /// Panics if `gamma` is not positive
    pub fn new(initial_lr: F, gamma: F) -> Self {
        assert!(gamma > F::zero(), "gamma must be positive");

        Self { initial_lr, gamma }
    }

    /// Create an ExponentialLR scheduler with slow decay
    ///
    /// This creates a scheduler with gamma = 0.99, providing very slow decay
    /// suitable for long training runs where you want gradual adjustment.
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    pub fn slow_decay(initial_lr: F) -> Self {
        Self::new(initial_lr, F::from(0.99).unwrap())
    }

    /// Create an ExponentialLR scheduler with moderate decay
    ///
    /// This creates a scheduler with gamma = 0.95, providing moderate decay
    /// suitable for most training scenarios.
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    pub fn moderate_decay(initial_lr: F) -> Self {
        Self::new(initial_lr, F::from(0.95).unwrap())
    }

    /// Create an ExponentialLR scheduler with fast decay
    ///
    /// This creates a scheduler with gamma = 0.9, providing faster decay
    /// suitable for quick convergence scenarios.
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    pub fn fast_decay(initial_lr: F) -> Self {
        Self::new(initial_lr, F::from(0.9).unwrap())
    }

    /// Create an ExponentialLR scheduler from half-life
    ///
    /// This creates a scheduler where the learning rate will decay to half
    /// its current value after `half_life` steps.
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    /// * `half_life` - Number of steps for learning rate to decay to half
    ///
    /// # Returns
    /// ExponentialLR scheduler with appropriate gamma
    pub fn from_half_life(initial_lr: F, half_life: usize) -> Self {
        assert!(half_life > 0, "half_life must be greater than 0");

        // gamma = 0.5^(1/half_life)
        let gamma = F::from(0.5)
            .unwrap()
            .powf(F::one() / F::from(half_life).unwrap());

        Self::new(initial_lr, gamma)
    }

    /// Calculate the learning rate after a specific number of steps
    ///
    /// This is equivalent to `get_lr(step)` but provides a more explicit interface.
    ///
    /// # Arguments
    /// * `step` - Number of steps
    ///
    /// # Returns
    /// Learning rate after the specified number of steps
    pub fn lr_after_steps(&self, step: usize) -> F {
        self.get_lr(step)
    }

    /// Calculate how many steps it takes for learning rate to decay to a target value
    ///
    /// # Arguments
    /// * `target_lr` - Target learning rate
    ///
    /// # Returns
    /// Number of steps needed to reach target learning rate (rounded up)
    pub fn steps_to_reach(&self, target_lr: F) -> Option<usize> {
        if target_lr <= F::zero() || target_lr > self.initial_lr {
            return None;
        }

        // Solve: target_lr = initial_lr * gamma^steps
        // steps = log(target_lr / initial_lr) / log(gamma)
        let ratio = target_lr / self.initial_lr;
        let steps = ratio.ln() / self.gamma.ln();

        Some(steps.ceil().to_usize().unwrap_or(usize::MAX))
    }

    /// Get the effective decay rate per epoch
    ///
    /// Returns the fraction of learning rate that remains after one step.
    pub fn decay_rate(&self) -> F {
        self.gamma
    }

    /// Get the percentage decay per step
    ///
    /// Returns the percentage by which learning rate decreases each step.
    pub fn decay_percentage(&self) -> F {
        (F::one() - self.gamma) * F::from(100.0).unwrap()
    }
}

impl<F: Float> LRScheduler<F> for ExponentialLR<F> {
    fn get_lr(&self, step: usize) -> F {
        if step == 0 {
            self.initial_lr
        } else {
            // Calculate gamma^step
            let decay_factor = self.gamma.powf(F::from(step).unwrap());
            self.initial_lr * decay_factor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_lr_basic() {
        let scheduler = ExponentialLR::new(1.0f32, 0.5);

        assert_eq!(scheduler.get_lr(0), 1.0);
        assert_eq!(scheduler.get_lr(1), 0.5);
        assert_eq!(scheduler.get_lr(2), 0.25);
        assert_eq!(scheduler.get_lr(3), 0.125);
    }

    #[test]
    fn test_exponential_lr_presets() {
        let slow = ExponentialLR::slow_decay(0.1f32);
        assert!((slow.gamma - 0.99).abs() < 1e-6);

        let moderate = ExponentialLR::moderate_decay(0.1f32);
        assert!((moderate.gamma - 0.95).abs() < 1e-6);

        let fast = ExponentialLR::fast_decay(0.1f32);
        assert!((fast.gamma - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr_half_life() {
        let scheduler = ExponentialLR::from_half_life(1.0f32, 10);

        // After 10 steps, learning rate should be approximately 0.5
        let lr_at_half_life = scheduler.get_lr(10);
        assert!((lr_at_half_life - 0.5).abs() < 1e-3);

        // After 20 steps, learning rate should be approximately 0.25
        let lr_at_double_half_life = scheduler.get_lr(20);
        assert!((lr_at_double_half_life - 0.25).abs() < 1e-3);
    }

    #[test]
    fn test_exponential_lr_steps_to_reach() {
        let scheduler = ExponentialLR::new(1.0f32, 0.5);

        // To reach 0.5, should take 1 step
        assert_eq!(scheduler.steps_to_reach(0.5), Some(1));

        // To reach 0.25, should take 2 steps
        assert_eq!(scheduler.steps_to_reach(0.25), Some(2));

        // Invalid targets
        assert_eq!(scheduler.steps_to_reach(0.0), None);
        assert_eq!(scheduler.steps_to_reach(2.0), None);
    }

    #[test]
    fn test_exponential_lr_decay_info() {
        let scheduler = ExponentialLR::new(1.0f32, 0.9);

        assert!((scheduler.decay_rate() - 0.9).abs() < 1e-6);
        assert!((scheduler.decay_percentage() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_lr_sequence() {
        let scheduler = ExponentialLR::new(1.0f32, 0.5);
        let sequence = scheduler.get_lr_sequence(0, 5);

        let expected = [1.0, 0.5, 0.25, 0.125, 0.0625];

        for (actual, expected) in sequence.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Expected {}, got {}",
                expected,
                actual
            );
        }
    }

    #[test]
    #[should_panic(expected = "gamma must be positive")]
    fn test_exponential_lr_negative_gamma() {
        ExponentialLR::new(0.1f32, -0.1);
    }

    #[test]
    #[should_panic(expected = "gamma must be positive")]
    fn test_exponential_lr_zero_gamma() {
        ExponentialLR::new(0.1f32, 0.0);
    }

    #[test]
    #[should_panic(expected = "half_life must be greater than 0")]
    fn test_exponential_lr_zero_half_life() {
        ExponentialLR::from_half_life(0.1f32, 0);
    }
}
