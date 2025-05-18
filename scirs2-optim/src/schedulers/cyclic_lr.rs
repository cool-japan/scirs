//! Cyclic learning rate scheduler
//!
//! This module provides cyclic learning rate scheduling, which cycles the learning rate
//! between two boundaries with a constant frequency.

use crate::schedulers::LearningRateScheduler;
use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt;

/// Cyclic learning rate policy
#[derive(Debug, Clone, Copy)]
pub enum CyclicMode {
    /// Triangular mode: linear scaling between min and max
    Triangular,
    /// Triangular2 mode: linear scaling with halved amplitude each cycle
    Triangular2,
    /// Exponential range: exponential scaling
    ExpRange(f64),
}

/// Cyclic learning rate scheduler
///
/// This scheduler cycles the learning rate between two boundaries with a constant frequency.
/// It's based on the paper "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.
///
/// # Example
///
/// ```
/// use scirs2_optim::schedulers::{CyclicLR, CyclicMode, LearningRateScheduler};
///
/// let mut scheduler = CyclicLR::new(0.001, 0.01, 2000, CyclicMode::Triangular);
///
/// // Learning rate cycles between 0.001 and 0.01 over 2000 steps
/// for step in 0..6000 {
///     let lr = scheduler.get_learning_rate();
///     // Use lr for optimization
///     scheduler.step();
/// }
/// ```
pub struct CyclicLR<A: Float> {
    base_lr: A,
    max_lr: A,
    step_size: usize,
    mode: CyclicMode,
    gamma: A,
    current_step: usize,
    scale_fn: Box<dyn Fn(usize, usize, A, A) -> A + Send + Sync>,
}

impl<A: Float + std::fmt::Debug> fmt::Debug for CyclicLR<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CyclicLR")
            .field("base_lr", &self.base_lr)
            .field("max_lr", &self.max_lr)
            .field("step_size", &self.step_size)
            .field("mode", &self.mode)
            .field("gamma", &self.gamma)
            .field("current_step", &self.current_step)
            .field("scale_fn", &"<function>")
            .finish()
    }
}

impl<A: Float + ScalarOperand + std::fmt::Debug> CyclicLR<A> {
    /// Create a new cyclic learning rate scheduler
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Minimum learning rate
    /// * `max_lr` - Maximum learning rate
    /// * `step_size` - Number of training iterations per half cycle
    /// * `mode` - Cycling mode (Triangular, Triangular2, or ExpRange)
    pub fn new(base_lr: A, max_lr: A, step_size: usize, mode: CyclicMode) -> Self {
        let gamma = match mode {
            CyclicMode::ExpRange(g) => A::from(g).unwrap(),
            _ => A::one(),
        };

        let scale_fn: Box<dyn Fn(usize, usize, A, A) -> A + Send + Sync> = match mode {
            CyclicMode::Triangular => Box::new(|_, _, _, _| A::one()),
            CyclicMode::Triangular2 => Box::new(|current, cycle_half, _, _| {
                A::one()
                    / (A::from(2)
                        .unwrap()
                        .powi(current as i32 / (2 * cycle_half) as i32))
            }),
            CyclicMode::ExpRange(_) => Box::new(|current, cycle_half, gamma, _| {
                gamma.powi((current % (2 * cycle_half)) as i32)
            }),
        };

        Self {
            base_lr,
            max_lr,
            step_size,
            mode,
            gamma,
            current_step: 0,
            scale_fn,
        }
    }

    /// Create a new triangular cyclic scheduler
    pub fn triangular(base_lr: A, max_lr: A, step_size: usize) -> Self {
        Self::new(base_lr, max_lr, step_size, CyclicMode::Triangular)
    }

    /// Create a new triangular2 cyclic scheduler
    pub fn triangular2(base_lr: A, max_lr: A, step_size: usize) -> Self {
        Self::new(base_lr, max_lr, step_size, CyclicMode::Triangular2)
    }

    /// Create a new exponential range cyclic scheduler
    pub fn exp_range(base_lr: A, max_lr: A, step_size: usize, gamma: f64) -> Self {
        Self::new(base_lr, max_lr, step_size, CyclicMode::ExpRange(gamma))
    }

    /// Set custom scale function
    pub fn with_scale_fn<F>(mut self, scale_fn: F) -> Self
    where
        F: Fn(usize, usize, A, A) -> A + Send + Sync + 'static,
    {
        self.scale_fn = Box::new(scale_fn);
        self
    }

    /// Get the current cycle number
    pub fn get_cycle(&self) -> usize {
        self.current_step / (2 * self.step_size)
    }

    /// Get position within current cycle (0.0 to 1.0)
    pub fn get_cycle_position(&self) -> A {
        let cycle_position = self.current_step % (2 * self.step_size);
        if cycle_position < self.step_size {
            // First half: increasing
            A::from(cycle_position).unwrap() / A::from(self.step_size).unwrap()
        } else {
            // Second half: decreasing
            A::from(2 * self.step_size - cycle_position).unwrap() / A::from(self.step_size).unwrap()
        }
    }
}

impl<A: Float + ScalarOperand + std::fmt::Debug> LearningRateScheduler<A> for CyclicLR<A> {
    fn get_learning_rate(&self) -> A {
        let position = self.get_cycle_position();
        let scale = (self.scale_fn)(self.current_step, self.step_size, self.gamma, A::one());

        let amplitude = (self.max_lr - self.base_lr) * scale;
        self.base_lr + amplitude * position
    }

    fn step(&mut self) -> A {
        self.current_step += 1;
        self.get_learning_rate()
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_triangular_cyclic() {
        let base_lr = 0.001;
        let max_lr = 0.01;
        let step_size = 100;

        let mut scheduler = CyclicLR::triangular(base_lr, max_lr, step_size);

        // At start, should be base_lr
        assert_relative_eq!(scheduler.get_learning_rate(), base_lr, epsilon = 1e-6);

        // At half cycle, should be max_lr
        for _ in 0..step_size {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_learning_rate(), max_lr, epsilon = 1e-6);

        // At full cycle, should be back to base_lr
        for _ in 0..step_size {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_learning_rate(), base_lr, epsilon = 1e-6);
    }

    #[test]
    fn test_triangular2_cyclic() {
        let base_lr = 0.001;
        let max_lr = 0.01;
        let step_size = 100;

        let mut scheduler = CyclicLR::triangular2(base_lr, max_lr, step_size);

        // First cycle
        for _ in 0..step_size {
            scheduler.step();
        }
        let first_max = scheduler.get_learning_rate();
        assert_relative_eq!(first_max, max_lr, epsilon = 1e-6);

        // Move to second cycle max
        for _ in 0..(2 * step_size) {
            scheduler.step();
        }
        let second_max = scheduler.get_learning_rate();

        // Second cycle should have half amplitude
        assert_relative_eq!(
            second_max,
            base_lr + (max_lr - base_lr) / 2.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_exp_range_cyclic() {
        let base_lr = 0.001;
        let max_lr = 0.01;
        let step_size = 100;
        let gamma = 0.99995;

        let mut scheduler = CyclicLR::exp_range(base_lr, max_lr, step_size, gamma);

        // Test that learning rate decreases exponentially within each cycle
        let lr_start = scheduler.get_learning_rate();

        for _ in 0..10 {
            scheduler.step();
        }

        let lr_10_steps = scheduler.get_learning_rate();

        // Learning rate should increase in first half of cycle
        assert!(lr_10_steps > lr_start);

        // But the increase should be modulated by gamma
        assert!(lr_10_steps < base_lr + (max_lr - base_lr) * 0.1);
    }

    #[test]
    fn test_cycle_counting() {
        let mut scheduler = CyclicLR::triangular(0.001, 0.01, 100);

        assert_eq!(scheduler.get_cycle(), 0);

        // Complete one cycle
        for _ in 0..200 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_cycle(), 1);

        // Half way through second cycle
        for _ in 0..100 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_cycle(), 1);
    }

    #[test]
    fn test_reset() {
        let mut scheduler = CyclicLR::triangular(0.001, 0.01, 100);

        // Move forward
        for _ in 0..50 {
            scheduler.step();
        }

        let lr_before_reset = scheduler.get_learning_rate();
        assert!(lr_before_reset > 0.001);

        // Reset
        scheduler.reset();
        assert_relative_eq!(scheduler.get_learning_rate(), 0.001, epsilon = 1e-6);
        assert_eq!(scheduler.current_step, 0);
    }
}
