//! Constant learning rate scheduler
//!
//! This module provides a simple scheduler that maintains a constant learning rate.
//! It's useful as a base for other schedulers or for testing.

use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;

use super::LearningRateScheduler;

/// A scheduler that maintains a constant learning rate
#[derive(Debug, Clone, Copy)]
pub struct ConstantScheduler<A: Float + Debug + ScalarOperand> {
    /// The constant learning rate
    learning_rate: A,
}

impl<A: Float + Debug + ScalarOperand> ConstantScheduler<A> {
    /// Create a new constant scheduler with the given learning rate
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The constant learning rate to maintain
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_optim::schedulers::{ConstantScheduler, LearningRateScheduler};
    ///
    /// let mut scheduler = ConstantScheduler::new(0.1);
    /// assert_eq!(scheduler.get_learning_rate(), 0.1);
    ///
    /// // Learning rate stays constant after stepping
    /// scheduler.step();
    /// assert_eq!(scheduler.get_learning_rate(), 0.1);
    /// ```
    pub fn new(learning_rate: A) -> Self {
        Self { learning_rate }
    }
}

impl<A: Float + Debug + ScalarOperand> LearningRateScheduler<A> for ConstantScheduler<A> {
    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }

    fn step(&mut self) -> A {
        self.learning_rate
    }

    fn reset(&mut self) {
        // Nothing to reset for a constant scheduler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let mut scheduler = ConstantScheduler::new(0.1);
        assert_eq!(scheduler.get_learning_rate(), 0.1);

        // Step multiple times and check that learning rate remains constant
        for _ in 0..10 {
            assert_eq!(scheduler.step(), 0.1);
        }

        // Reset shouldn't change the learning rate
        scheduler.reset();
        assert_eq!(scheduler.get_learning_rate(), 0.1);
    }
}
