//! Linear warmup with decay learning rate scheduler
//!
//! This scheduler combines linear warmup with a decay strategy.
//! It first linearly increases the learning rate from a minimum value to the
//! initial learning rate during the warmup phase, then applies a decay strategy.

use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;

use crate::schedulers::{
    CosineAnnealing, ExponentialDecay, LearningRateScheduler, LinearDecay, StepDecay,
};

/// Decay strategy to use after the warmup phase
#[derive(Debug, Clone)]
pub enum DecayStrategy<A: Float + Debug> {
    /// Linear decay to a final value
    Linear {
        /// Final learning rate
        final_lr: A,
    },
    /// Exponential decay with a given decay rate
    Exponential {
        /// Decay rate per step
        decay_rate: A,
    },
    /// Step decay with a given step size and decay rate
    Step {
        /// Decay rate at each step
        decay_rate: A,
        /// Step size (number of steps between decays)
        step_size: usize,
    },
    /// Cosine annealing to a minimum value
    Cosine {
        /// Minimum learning rate
        min_lr: A,
    },
    /// No decay - constant learning rate after warmup
    Constant,
}

/// Linear warmup with decay learning rate scheduler
///
/// This scheduler first linearly increases the learning rate from a minimum value to the
/// initial learning rate during the warmup phase, then applies a decay strategy.
///
/// Formula during warmup:
/// lr = min_lr + (initial_lr - min_lr) * (step / warmup_steps)
///
/// After warmup, the specified decay strategy is applied.
///
/// # Examples
///
/// ```
/// use scirs2_optim::schedulers::{LinearWarmupDecay, DecayStrategy, LearningRateScheduler};
///
/// // Create a scheduler with linear warmup for 10 steps, followed by linear decay
/// // to 0.001 over 90 steps
/// let mut scheduler = LinearWarmupDecay::new(
///     0.1f64,                          // initial_lr (peak learning rate)
///     0.01,                            // min_lr (starting learning rate)
///     10,                              // warmup_steps
///     90,                              // total_decay_steps
///     DecayStrategy::Linear { final_lr: 0.001 }
/// );
///
/// // Verify initial learning rate is the minimum
/// assert_eq!(scheduler.get_learning_rate(), 0.01);
///
/// // Run for warmup period
/// for _ in 0..10 {
///     let lr = scheduler.step();
///     // Learning rate should increase during warmup
///     println!("Warmup LR: {}", lr);
/// }
///
/// // After warmup, we should be at the initial (peak) learning rate
/// assert_eq!(scheduler.get_learning_rate(), 0.1);
///
/// // Run for decay period
/// for _ in 0..90 {
///     let lr = scheduler.step();
///     // Learning rate should decrease during decay
///     println!("Decay LR: {}", lr);
/// }
///
/// // Verify learning rate has decayed to the target
/// assert!(scheduler.get_learning_rate() <= 0.001 + 1e-6);
/// ```
#[derive(Debug)]
pub struct LinearWarmupDecay<A: Float + Debug> {
    /// Initial learning rate (the peak learning rate after warmup)
    initial_lr: A,
    /// Minimum learning rate (starting point for warmup)
    min_lr: A,
    /// Number of warmup steps
    warmup_steps: usize,
    /// Number of decay steps after warmup
    total_decay_steps: usize,
    /// Current step
    step: usize,
    /// Current learning rate
    current_lr: A,
    /// Decay strategy to use after warmup
    decay_strategy: DecayStrategy<A>,
    /// Whether warmup phase is complete
    warmup_complete: bool,
    /// Inner scheduler for decay phase (initialized after warmup)
    #[allow(clippy::missing_docs_in_private_items)]
    inner_scheduler: Option<InnerScheduler<A>>,
}

/// Inner scheduler types for LinearWarmupDecay
#[allow(clippy::missing_docs_in_private_items)]
#[derive(Debug)]
enum InnerScheduler<A: Float + Debug> {
    /// Linear decay scheduler
    Linear(LinearDecay<A>),
    /// Exponential decay scheduler
    Exponential(ExponentialDecay<A>),
    /// Step decay scheduler
    Step(StepDecay<A>),
    /// Cosine annealing scheduler
    Cosine(CosineAnnealing<A>),
}

impl<A: Float + Debug + ScalarOperand> LinearWarmupDecay<A> {
    /// Create a new linear warmup with decay scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate (the peak learning rate after warmup)
    /// * `min_lr` - Minimum learning rate (starting point for warmup)
    /// * `warmup_steps` - Number of warmup steps
    /// * `total_decay_steps` - Number of decay steps after warmup
    /// * `decay_strategy` - Decay strategy to use after warmup
    pub fn new(
        initial_lr: A,
        min_lr: A,
        warmup_steps: usize,
        total_decay_steps: usize,
        decay_strategy: DecayStrategy<A>,
    ) -> Self {
        Self {
            initial_lr,
            min_lr,
            warmup_steps,
            total_decay_steps,
            step: 0,
            current_lr: min_lr,
            decay_strategy,
            warmup_complete: false,
            inner_scheduler: None,
        }
    }

    /// Initialize the inner decay scheduler once warmup is complete
    fn initialize_decay_scheduler(&mut self) {
        let scheduler = match self.decay_strategy {
            DecayStrategy::Linear { final_lr } => InnerScheduler::Linear(LinearDecay::new(
                self.initial_lr,
                final_lr,
                self.total_decay_steps,
            )),
            DecayStrategy::Exponential { decay_rate } => InnerScheduler::Exponential(
                ExponentialDecay::new(self.initial_lr, decay_rate, self.total_decay_steps),
            ),
            DecayStrategy::Step {
                decay_rate,
                step_size,
            } => InnerScheduler::Step(StepDecay::new(self.initial_lr, decay_rate, step_size)),
            DecayStrategy::Cosine { min_lr } => InnerScheduler::Cosine(CosineAnnealing::new(
                self.initial_lr,
                min_lr,
                self.total_decay_steps,
                false, // No warm restarts in this scheduler
            )),
            DecayStrategy::Constant => {
                // For constant strategy, we use a linear decay with the same start and end values
                InnerScheduler::Linear(LinearDecay::new(
                    self.initial_lr,
                    self.initial_lr,
                    self.total_decay_steps,
                ))
            }
        };

        self.inner_scheduler = Some(scheduler);
    }
}

impl<A: Float + Debug + ScalarOperand> LearningRateScheduler<A> for LinearWarmupDecay<A> {
    fn get_learning_rate(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) -> A {
        self.step += 1;

        // Special case: if warmup_steps is 0, go straight to decay
        if self.warmup_steps == 0 && !self.warmup_complete {
            self.warmup_complete = true;
            self.initialize_decay_scheduler();
        }

        if !self.warmup_complete && self.step <= self.warmup_steps {
            // Warmup phase: linear increase from min_lr to initial_lr
            let progress = if self.warmup_steps > 0 {
                A::from(self.step).unwrap() / A::from(self.warmup_steps).unwrap()
            } else {
                A::one()
            };

            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * progress;

            // Check if warmup is complete after this step
            if self.step == self.warmup_steps {
                self.warmup_complete = true;
                self.initialize_decay_scheduler();
            }
        } else if self.warmup_complete {
            // Decay phase: use inner scheduler
            if let Some(scheduler) = &mut self.inner_scheduler {
                self.current_lr = match scheduler {
                    InnerScheduler::Linear(s) => s.step(),
                    InnerScheduler::Exponential(s) => s.step(),
                    InnerScheduler::Step(s) => s.step(),
                    InnerScheduler::Cosine(s) => s.step(),
                };
            }
        }

        self.current_lr
    }

    fn reset(&mut self) {
        self.step = 0;
        self.current_lr = self.min_lr;
        self.warmup_complete = false;
        self.inner_scheduler = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_linear_warmup_linear_decay() {
        // Create scheduler with 10 warmup steps and 90 decay steps
        let mut scheduler = LinearWarmupDecay::new(
            0.1f64,
            0.01,
            10,
            90,
            DecayStrategy::Linear { final_lr: 0.001 },
        );

        // Check initial learning rate
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.01);

        // Test warmup phase
        let mut warmup_lrs = Vec::new();
        for _ in 0..10 {
            warmup_lrs.push(scheduler.step());
        }

        // After warmup, we should be at the initial (peak) learning rate
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // Verify warmup is linearly increasing
        for i in 1..warmup_lrs.len() {
            assert!(warmup_lrs[i] > warmup_lrs[i - 1]);
        }

        // Test decay phase
        let mut decay_lrs = Vec::new();
        for _ in 0..90 {
            decay_lrs.push(scheduler.step());
        }

        // Verify final learning rate
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-6);

        // Verify decay is linearly decreasing
        for i in 1..decay_lrs.len() {
            assert!(decay_lrs[i] < decay_lrs[i - 1]);
        }
    }

    #[test]
    fn test_linear_warmup_exponential_decay() {
        // Create scheduler with 10 warmup steps and 10 decay steps
        // Using a very extreme decay rate to make the effect very obvious in few steps
        let mut scheduler = LinearWarmupDecay::new(
            0.1f64,
            0.01,
            10,
            10,
            DecayStrategy::Exponential { decay_rate: 0.1 },
        );

        // Check warmup phase
        for _ in 0..10 {
            scheduler.step();
        }

        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // Collect learning rates for the entire decay phase
        let mut lrs = Vec::new();
        for _ in 0..10 {
            lrs.push(scheduler.step());
        }

        // Verify the final learning rate is significantly lower than initial
        let final_lr = *lrs.last().unwrap();
        assert!(
            final_lr < 0.05,
            "Final learning rate {:.6} should be significantly less than initial 0.1",
            final_lr
        );
    }

    #[test]
    fn test_linear_warmup_step_decay() {
        // Create scheduler with 10 warmup steps and 40 decay steps
        let mut scheduler = LinearWarmupDecay::new(
            0.1f64,
            0.01,
            10,
            40,
            DecayStrategy::Step {
                decay_rate: 0.5,
                step_size: 10,
            },
        );

        // Check warmup phase
        for _ in 0..10 {
            scheduler.step();
        }

        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // First 10 steps of decay should maintain the learning rate
        for _ in 0..9 {
            scheduler.step();
        }
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // After step_size, the learning rate should decay by decay_rate
        scheduler.step(); // Step 10 of decay (20 overall)
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.05);

        // Another 10 steps should maintain 0.05
        for _ in 0..9 {
            scheduler.step();
        }
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.05);

        // Another decay step
        scheduler.step(); // Step 20 of decay (30 overall)
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.025);
    }

    #[test]
    fn test_linear_warmup_cosine_decay() {
        // Create scheduler with 10 warmup steps and 90 decay steps
        let mut scheduler = LinearWarmupDecay::new(
            0.1f64,
            0.01,
            10,
            90,
            DecayStrategy::Cosine { min_lr: 0.001 },
        );

        // Check warmup phase
        for _ in 0..10 {
            scheduler.step();
        }

        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // Run for a while and collect learning rates
        let mut lrs = Vec::new();
        for _ in 0..90 {
            lrs.push(scheduler.step());
        }

        // Verify the learning rate is decreasing
        assert!(lrs[0] < 0.1); // Should decrease from initial

        // The curve should eventually approach the minimum
        let min_lr = lrs.iter().fold(1.0, |a, &b| a.min(b));
        assert_abs_diff_eq!(min_lr, 0.001, epsilon = 1e-2);
    }

    #[test]
    fn test_linear_warmup_constant() {
        // Create scheduler with 10 warmup steps and constant after
        let mut scheduler = LinearWarmupDecay::new(0.1f64, 0.01, 10, 90, DecayStrategy::Constant);

        // Check warmup phase
        for _ in 0..10 {
            scheduler.step();
        }

        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // After decay, we should still have the initial_lr
        for _ in 0..90 {
            scheduler.step();
        }

        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);
    }

    #[test]
    fn test_reset() {
        // Create scheduler with 5 warmup steps and 15 decay steps
        let mut scheduler = LinearWarmupDecay::new(
            0.1f64,
            0.01,
            5,
            15,
            DecayStrategy::Linear { final_lr: 0.001 },
        );

        // Go through a few steps
        for _ in 0..15 {
            scheduler.step();
        }

        // Reset the scheduler
        scheduler.reset();

        // Verify state has been reset
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.01);
        assert_eq!(scheduler.step, 0);
        assert!(!scheduler.warmup_complete);
        assert!(scheduler.inner_scheduler.is_none());

        // Check we can perform warmup again
        for _ in 0..5 {
            scheduler.step();
        }
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);
    }

    #[test]
    fn test_zero_warmup() {
        // Create scheduler with 0 warmup steps
        let mut scheduler = LinearWarmupDecay::new(
            0.1f64,
            0.01,
            0,
            10,
            DecayStrategy::Linear { final_lr: 0.001 },
        );

        // Initial learning rate should be the peak rate
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.01);

        // First step should go directly to decay phase
        scheduler.step();
        assert!(scheduler.warmup_complete);

        // Continue decay for 9 more steps
        for _ in 0..9 {
            scheduler.step();
        }

        // Verify final learning rate
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-6);
    }
}
