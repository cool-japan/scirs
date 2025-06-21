//! Learning rate schedulers
//!
//! This module provides various learning rate scheduling strategies that can be used
//! to adjust the learning rate during training to improve convergence and performance.

pub mod cosine_annealing;
pub mod exponential;
pub mod step;

pub use cosine_annealing::CosineAnnealingLR;
pub use exponential::ExponentialLR;
pub use step::StepLR;

use crate::Float;

/// Trait for learning rate schedulers
///
/// Learning rate schedulers implement policies for adjusting the learning rate
/// during training. This can help with convergence and prevent overfitting.
pub trait LRScheduler<F: Float> {
    /// Get the current learning rate for the given step/epoch
    ///
    /// # Arguments
    /// * `step` - Current step/epoch number (0-indexed)
    ///
    /// # Returns
    /// The learning rate that should be used for this step
    fn get_lr(&self, step: usize) -> F;

    /// Update the scheduler's internal state for the next step
    ///
    /// Some schedulers may need to track internal state. This method
    /// should be called after each step/epoch.
    ///
    /// # Arguments
    /// * `step` - Current step/epoch number that was just completed
    fn step(&mut self, step: usize) {
        // Default implementation does nothing
        let _ = step;
    }

    /// Reset the scheduler to its initial state
    ///
    /// This can be useful when restarting training or changing phases.
    fn reset(&mut self) {
        // Default implementation does nothing
    }

    /// Get a sequence of learning rates for a range of steps
    ///
    /// This is a convenience method for plotting or analyzing the schedule.
    ///
    /// # Arguments
    /// * `start_step` - Starting step number
    /// * `end_step` - Ending step number (exclusive)
    ///
    /// # Returns
    /// Vector of learning rates for each step in the range
    fn get_lr_sequence(&self, start_step: usize, end_step: usize) -> Vec<F> {
        (start_step..end_step)
            .map(|step| self.get_lr(step))
            .collect()
    }
}

/// Constant learning rate scheduler (baseline)
///
/// This scheduler maintains a constant learning rate throughout training.
/// It's useful as a baseline or when no learning rate decay is desired.
pub struct ConstantLR<F: Float> {
    pub lr: F,
}

impl<F: Float> ConstantLR<F> {
    /// Create a new constant learning rate scheduler
    ///
    /// # Arguments
    /// * `lr` - The constant learning rate to use
    pub fn new(lr: F) -> Self {
        Self { lr }
    }
}

impl<F: Float> LRScheduler<F> for ConstantLR<F> {
    fn get_lr(&self, _step: usize) -> F {
        self.lr
    }
}

/// Linear learning rate scheduler
///
/// This scheduler linearly interpolates the learning rate between a start and end value.
/// It's useful for warming up or cooling down the learning rate.
pub struct LinearLR<F: Float> {
    pub start_lr: F,
    pub end_lr: F,
    pub total_steps: usize,
}

impl<F: Float> LinearLR<F> {
    /// Create a new linear learning rate scheduler
    ///
    /// # Arguments
    /// * `start_lr` - Initial learning rate
    /// * `end_lr` - Final learning rate
    /// * `total_steps` - Total number of steps over which to interpolate
    pub fn new(start_lr: F, end_lr: F, total_steps: usize) -> Self {
        Self {
            start_lr,
            end_lr,
            total_steps,
        }
    }
}

impl<F: Float> LRScheduler<F> for LinearLR<F> {
    fn get_lr(&self, step: usize) -> F {
        if self.total_steps == 0 {
            return self.start_lr;
        }

        let progress =
            F::from(step.min(self.total_steps)).unwrap() / F::from(self.total_steps).unwrap();
        let lr_diff = self.end_lr - self.start_lr;
        self.start_lr + lr_diff * progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let scheduler = ConstantLR::new(0.001f32);

        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(100), 0.001);
        assert_eq!(scheduler.get_lr(1000), 0.001);
    }

    #[test]
    fn test_linear_lr() {
        let scheduler = LinearLR::new(0.0f32, 1.0f32, 100);

        assert_eq!(scheduler.get_lr(0), 0.0);
        assert_eq!(scheduler.get_lr(50), 0.5);
        assert_eq!(scheduler.get_lr(100), 1.0);
        assert_eq!(scheduler.get_lr(150), 1.0); // Clamps at end_lr
    }

    #[test]
    fn test_linear_lr_sequence() {
        let scheduler = LinearLR::new(0.0f32, 1.0f32, 10);
        let sequence = scheduler.get_lr_sequence(0, 11);

        assert_eq!(sequence.len(), 11);
        assert_eq!(sequence[0], 0.0);
        assert_eq!(sequence[5], 0.5);
        assert_eq!(sequence[10], 1.0);
    }
}
