//! Step learning rate scheduler

use crate::schedulers::LRScheduler;
use crate::Float;

/// Step learning rate scheduler
///
/// Decays the learning rate by a multiplicative factor (`gamma`) every `step_size` epochs.
/// This is one of the most commonly used learning rate schedules.
///
/// The learning rate is calculated as:
/// `lr = initial_lr * gamma^(step // step_size)`
///
/// # Example
/// ```
/// use scirs2_autograd::schedulers::{StepLR, LRScheduler};
///
/// let scheduler = StepLR::new(0.1f32, 30, 0.1f32);
///
/// // Initial learning rate
/// assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-6);
///
/// // After 30 steps, learning rate is reduced by gamma
/// assert!((scheduler.get_lr(30) - 0.01).abs() < 1e-6);
///
/// // After 60 steps, learning rate is reduced again
/// assert!((scheduler.get_lr(60) - 0.001).abs() < 1e-6);
/// ```
pub struct StepLR<F: Float> {
    /// Initial learning rate
    pub initial_lr: F,
    /// Number of steps between learning rate reductions
    pub step_size: usize,
    /// Multiplicative factor for learning rate decay
    pub gamma: F,
}

impl<F: Float> StepLR<F> {
    /// Create a new step learning rate scheduler
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    /// * `step_size` - Period of learning rate decay (in steps/epochs)
    /// * `gamma` - Multiplicative factor of learning rate decay (should be < 1.0 for decay)
    ///
    /// # Panics
    /// Panics if `step_size` is 0
    pub fn new(initial_lr: F, step_size: usize, gamma: F) -> Self {
        assert!(step_size > 0, "step_size must be greater than 0");

        Self {
            initial_lr,
            step_size,
            gamma,
        }
    }

    /// Create a StepLR scheduler with commonly used parameters
    ///
    /// This creates a scheduler that reduces learning rate by factor of 10 every 30 epochs.
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    pub fn default_decay(initial_lr: F) -> Self {
        Self::new(
            initial_lr,
            30,                    // step_size = 30 epochs
            F::from(0.1).unwrap(), // gamma = 0.1 (reduce by 10x)
        )
    }

    /// Create a StepLR scheduler for fine-tuning
    ///
    /// This creates a scheduler with more conservative decay, suitable for fine-tuning
    /// pre-trained models where we want smaller, more frequent adjustments.
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    pub fn for_fine_tuning(initial_lr: F) -> Self {
        Self::new(
            initial_lr,
            10,                    // step_size = 10 epochs (more frequent)
            F::from(0.5).unwrap(), // gamma = 0.5 (reduce by 2x, more conservative)
        )
    }

    /// Create a StepLR scheduler for aggressive decay
    ///
    /// This creates a scheduler with aggressive decay, useful when training
    /// is converging quickly and you want to reduce learning rate more aggressively.
    ///
    /// # Arguments
    /// * `initial_lr` - The initial learning rate
    pub fn aggressive_decay(initial_lr: F) -> Self {
        Self::new(
            initial_lr,
            20,                     // step_size = 20 epochs
            F::from(0.01).unwrap(), // gamma = 0.01 (reduce by 100x)
        )
    }

    /// Get the number of decay steps that have occurred for a given step
    ///
    /// # Arguments
    /// * `step` - Current step number
    ///
    /// # Returns
    /// Number of times the learning rate has been decayed
    pub fn num_decays(&self, step: usize) -> usize {
        step / self.step_size
    }

    /// Check if a decay should occur at the given step
    ///
    /// # Arguments
    /// * `step` - Current step number
    ///
    /// # Returns
    /// `true` if learning rate should decay at this step
    pub fn should_decay_at_step(&self, step: usize) -> bool {
        step > 0 && step % self.step_size == 0
    }
}

impl<F: Float> LRScheduler<F> for StepLR<F> {
    fn get_lr(&self, step: usize) -> F {
        let num_decays = self.num_decays(step);

        if num_decays == 0 {
            self.initial_lr
        } else {
            // Calculate gamma^num_decays
            let decay_factor = self.gamma.powf(F::from(num_decays).unwrap());
            self.initial_lr * decay_factor
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_lr_basic() {
        let scheduler = StepLR::new(0.1f32, 10, 0.1);

        // Before first decay
        assert_eq!(scheduler.get_lr(0), 0.1);
        assert_eq!(scheduler.get_lr(5), 0.1);
        assert_eq!(scheduler.get_lr(9), 0.1);

        // After first decay
        assert!((scheduler.get_lr(10) - 0.01).abs() < 1e-6);
        assert!((scheduler.get_lr(15) - 0.01).abs() < 1e-6);
        assert!((scheduler.get_lr(19) - 0.01).abs() < 1e-6);

        // After second decay
        assert!((scheduler.get_lr(20) - 0.001).abs() < 1e-6);
        assert!((scheduler.get_lr(25) - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_step_lr_num_decays() {
        let scheduler = StepLR::new(0.1f32, 5, 0.5);

        assert_eq!(scheduler.num_decays(0), 0);
        assert_eq!(scheduler.num_decays(4), 0);
        assert_eq!(scheduler.num_decays(5), 1);
        assert_eq!(scheduler.num_decays(9), 1);
        assert_eq!(scheduler.num_decays(10), 2);
        assert_eq!(scheduler.num_decays(15), 3);
    }

    #[test]
    fn test_step_lr_should_decay() {
        let scheduler = StepLR::new(0.1f32, 5, 0.5);

        assert!(!scheduler.should_decay_at_step(0));
        assert!(!scheduler.should_decay_at_step(4));
        assert!(scheduler.should_decay_at_step(5));
        assert!(!scheduler.should_decay_at_step(6));
        assert!(scheduler.should_decay_at_step(10));
        assert!(scheduler.should_decay_at_step(15));
    }

    #[test]
    fn test_step_lr_presets() {
        let default_scheduler = StepLR::default_decay(0.1f32);
        assert_eq!(default_scheduler.step_size, 30);
        assert!((default_scheduler.gamma - 0.1).abs() < 1e-6);

        let fine_tune_scheduler = StepLR::for_fine_tuning(0.01f32);
        assert_eq!(fine_tune_scheduler.step_size, 10);
        assert!((fine_tune_scheduler.gamma - 0.5).abs() < 1e-6);

        let aggressive_scheduler = StepLR::aggressive_decay(0.1f32);
        assert_eq!(aggressive_scheduler.step_size, 20);
        assert!((aggressive_scheduler.gamma - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_step_lr_sequence() {
        let scheduler = StepLR::new(1.0f32, 3, 0.5);
        let sequence = scheduler.get_lr_sequence(0, 10);

        let expected = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.125];

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
    #[should_panic(expected = "step_size must be greater than 0")]
    fn test_step_lr_zero_step_size() {
        StepLR::new(0.1f32, 0, 0.1);
    }
}
