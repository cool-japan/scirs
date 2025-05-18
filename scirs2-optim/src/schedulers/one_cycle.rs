//! One-cycle learning rate policy
//!
//! This module implements the one-cycle learning rate policy as described by Leslie N. Smith
//! in "A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate,
//! batch size, momentum, and weight decay"

use crate::schedulers::LearningRateScheduler;
use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::{self, Debug};

/// One-cycle learning rate policy
///
/// The one-cycle policy combines triangular learning rate policy with momentum cycling.
/// It consists of two phases:
/// 1. A warm-up phase where learning rate increases and momentum decreases
/// 2. A cool-down phase where learning rate decreases and momentum increases
///
/// # Example
///
/// ```
/// use scirs2_optim::schedulers::{OneCycle, LearningRateScheduler};
///
/// let mut scheduler = OneCycle::new(
///     0.0001,  // initial learning rate
///     0.001,   // max learning rate
///     1000,    // total steps
///     0.25,    // warm-up percentage
/// );
///
/// // The learning rate will increase from 0.0001 to 0.001 in first 250 steps,
/// // then decrease to a value lower than initial in remaining 750 steps
/// for _ in 0..1000 {
///     let lr = scheduler.get_learning_rate();
///     // Use lr for optimization
///     scheduler.step();
/// }
/// ```
pub struct OneCycle<A: Float> {
    initial_lr: A,
    max_lr: A,
    final_lr: Option<A>,
    total_steps: usize,
    warmup_steps: usize,
    current_step: usize,
    max_momentum: Option<A>,
    min_momentum: Option<A>,
    base_momentum: Option<A>,
    anneal_strategy: AnnealStrategy,
    #[allow(dead_code)]
    div_factor: A,
    final_div_factor: A,
}

/// Annealing strategy for the cool-down phase
#[derive(Debug, Clone, Copy)]
pub enum AnnealStrategy {
    /// Linear annealing
    Linear,
    /// Cosine annealing
    Cosine,
}

impl<A: Float + ScalarOperand + std::fmt::Debug> OneCycle<A> {
    /// Create a new one-cycle scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Starting learning rate
    /// * `max_lr` - Maximum learning rate reached after warm-up
    /// * `total_steps` - Total number of training steps
    /// * `warmup_frac` - Fraction of total steps used for warm-up (typically 0.2-0.3)
    pub fn new(initial_lr: A, max_lr: A, total_steps: usize, warmup_frac: f64) -> Self {
        let warmup_steps = (total_steps as f64 * warmup_frac) as usize;
        let div_factor = max_lr / initial_lr;
        let final_div_factor = A::from(10000.0).unwrap(); // Very small final LR

        Self {
            initial_lr,
            max_lr,
            final_lr: None,
            total_steps,
            warmup_steps,
            current_step: 0,
            max_momentum: None,
            min_momentum: None,
            base_momentum: None,
            anneal_strategy: AnnealStrategy::Cosine,
            div_factor,
            final_div_factor,
        }
    }

    /// Create with specific final learning rate
    pub fn with_final_lr(mut self, final_lr: A) -> Self {
        self.final_lr = Some(final_lr);
        self.final_div_factor = self.initial_lr / final_lr;
        self
    }

    /// Set momentum cycling parameters
    pub fn with_momentum(mut self, min_momentum: A, max_momentum: A, base_momentum: A) -> Self {
        self.min_momentum = Some(min_momentum);
        self.max_momentum = Some(max_momentum);
        self.base_momentum = Some(base_momentum);
        self
    }

    /// Set annealing strategy for cool-down phase
    pub fn with_anneal_strategy(mut self, strategy: AnnealStrategy) -> Self {
        self.anneal_strategy = strategy;
        self
    }

    /// Get current momentum value
    pub fn get_momentum(&self) -> Option<A> {
        match (self.min_momentum, self.max_momentum) {
            (Some(min_mom), Some(max_mom)) => {
                if self.current_step < self.warmup_steps {
                    // During warm-up: momentum decreases
                    let progress =
                        A::from(self.current_step).unwrap() / A::from(self.warmup_steps).unwrap();
                    Some(max_mom - (max_mom - min_mom) * progress)
                } else {
                    // During cool-down: momentum increases
                    let remaining_steps = self.total_steps - self.warmup_steps;
                    let cool_progress = A::from(self.current_step - self.warmup_steps).unwrap()
                        / A::from(remaining_steps).unwrap();

                    match self.anneal_strategy {
                        AnnealStrategy::Linear => {
                            Some(min_mom + (max_mom - min_mom) * cool_progress)
                        }
                        AnnealStrategy::Cosine => {
                            let cos_out =
                                ((cool_progress * A::from(std::f64::consts::PI).unwrap()).cos()
                                    + A::one())
                                    / A::from(2.0).unwrap();
                            Some(min_mom + (max_mom - min_mom) * (A::one() - cos_out))
                        }
                    }
                }
            }
            _ => self.base_momentum,
        }
    }

    /// Get percentage of completion
    pub fn get_percentage_complete(&self) -> A {
        A::from(self.current_step).unwrap() / A::from(self.total_steps).unwrap()
    }
}

impl<A: Float + ScalarOperand + Debug> LearningRateScheduler<A> for OneCycle<A> {
    fn get_learning_rate(&self) -> A {
        if self.current_step < self.warmup_steps {
            // Warm-up phase: increase from initial to max
            let progress =
                A::from(self.current_step).unwrap() / A::from(self.warmup_steps).unwrap();
            self.initial_lr + (self.max_lr - self.initial_lr) * progress
        } else {
            // Cool-down phase: decrease from max to final
            let remaining_steps = self.total_steps - self.warmup_steps;
            let cool_progress = A::from(self.current_step - self.warmup_steps).unwrap()
                / A::from(remaining_steps).unwrap();

            let final_lr = self
                .final_lr
                .unwrap_or(self.initial_lr / self.final_div_factor);

            match self.anneal_strategy {
                AnnealStrategy::Linear => self.max_lr - (self.max_lr - final_lr) * cool_progress,
                AnnealStrategy::Cosine => {
                    let cos_out = ((cool_progress * A::from(std::f64::consts::PI).unwrap()).cos()
                        + A::one())
                        / A::from(2.0).unwrap();
                    final_lr + (self.max_lr - final_lr) * cos_out
                }
            }
        }
    }

    fn step(&mut self) -> A {
        self.current_step += 1;
        self.get_learning_rate()
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

impl<A: Float + Debug> fmt::Debug for OneCycle<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OneCycle")
            .field("initial_lr", &self.initial_lr)
            .field("max_lr", &self.max_lr)
            .field("final_lr", &self.final_lr)
            .field("total_steps", &self.total_steps)
            .field("warmup_steps", &self.warmup_steps)
            .field("current_step", &self.current_step)
            .field("anneal_strategy", &self.anneal_strategy)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_one_cycle_basic() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25);

        // Initial learning rate
        assert_relative_eq!(scheduler.get_learning_rate(), 0.0001, epsilon = 1e-6);

        // At end of warm-up (25% = 25 steps)
        for _ in 0..25 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-6);

        // Final learning rate should be very small
        for _ in 25..100 {
            scheduler.step();
        }
        assert!(scheduler.get_learning_rate() < 0.0001);
    }

    #[test]
    fn test_one_cycle_momentum() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25).with_momentum(0.85, 0.95, 0.9);

        // Initial momentum (max during warm-up)
        assert_relative_eq!(scheduler.get_momentum().unwrap(), 0.95, epsilon = 1e-6);

        // At end of warm-up (min momentum)
        for _ in 0..25 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_momentum().unwrap(), 0.85, epsilon = 1e-6);

        // Final momentum (back to max)
        for _ in 25..100 {
            scheduler.step();
        }
        let final_momentum = scheduler.get_momentum().unwrap();
        assert!(final_momentum > 0.94); // Should be close to max
    }

    #[test]
    fn test_one_cycle_linear_anneal() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25)
            .with_anneal_strategy(AnnealStrategy::Linear)
            .with_final_lr(0.00001);

        // Move past warm-up
        for _ in 0..25 {
            scheduler.step();
        }

        let lr_at_warmup = scheduler.get_learning_rate();
        assert_relative_eq!(lr_at_warmup, 0.001, epsilon = 1e-6);

        // Check linear decrease
        for _ in 0..37 {
            // Halfway through cool-down
            scheduler.step();
        }

        let lr_halfway = scheduler.get_learning_rate();
        assert!(lr_halfway < 0.001);
        assert!(lr_halfway > 0.00001);

        // Should decrease linearly
        let expected = 0.001 - (0.001 - 0.00001) * 0.5;
        assert_relative_eq!(lr_halfway, expected, epsilon = 1e-4);
    }

    #[test]
    fn test_percentage_complete() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25);

        assert_relative_eq!(scheduler.get_percentage_complete(), 0.0, epsilon = 1e-6);

        for _ in 0..50 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_percentage_complete(), 0.5, epsilon = 1e-6);

        for _ in 50..100 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_percentage_complete(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut scheduler = OneCycle::new(0.0001, 0.001, 100, 0.25);

        // Advance scheduler
        for _ in 0..50 {
            scheduler.step();
        }

        let lr_mid = scheduler.get_learning_rate();
        assert!(lr_mid != 0.0001);

        // Reset
        scheduler.reset();
        assert_eq!(scheduler.current_step, 0);
        assert_relative_eq!(scheduler.get_learning_rate(), 0.0001, epsilon = 1e-6);
    }
}
