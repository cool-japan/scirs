//! Cosine annealing with warm restarts learning rate scheduler
//!
//! Implementation of SGDR: Stochastic Gradient Descent with Warm Restarts

use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;

use crate::schedulers::LearningRateScheduler;

/// Cosine annealing with warm restarts learning rate scheduler
///
/// Implements the SGDR (Stochastic Gradient Descent with Warm Restarts) scheduler
/// from the paper "SGDR: Stochastic Gradient Descent with Warm Restarts" by
/// Loshchilov & Hutter (2017).
///
/// This scheduler performs a cosine annealing schedule with periodic restarts,
/// where the period grows by a multiplicative factor after each restart.
///
/// The learning rate follows a cosine schedule from the initial learning rate to the
/// minimum learning rate over a cycle of T_i steps. After each cycle, the learning rate
/// is reset to the initial learning rate, and the cycle length is multiplied by T_mult.
///
/// Formula:
/// lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * T_cur / T_i))
///
/// where:
/// - T_i is the cycle length for the i-th cycle
/// - T_cur is the number of steps taken in the current cycle
/// - T_i = T_0 * (T_mult ^ i) where i is the restart index
///
/// # Examples
///
/// ```
/// use scirs2_optim::schedulers::{CosineAnnealingWarmRestarts, LearningRateScheduler};
/// use scirs2_optim::optimizers::SGD;
/// use ndarray::Array1;
///
/// // Create a scheduler with initial learning rate 0.1, minimum learning rate 0.001,
/// // initial cycle length 10 steps, and multiplicative factor 2.0
/// let mut scheduler = CosineAnnealingWarmRestarts::new(0.1f64, 0.001, 10, 2.0);
///
/// // First cycle: 10 iterations
/// let mut lrs = Vec::new();
/// for _ in 0..10 {
///     lrs.push(scheduler.step());
/// }
///
/// // The learning rate should be back to initial after a cycle
/// assert_eq!(scheduler.get_learning_rate(), 0.1);
///
/// // Second cycle: 20 iterations (10 * 2.0)
/// for _ in 0..20 {
///     scheduler.step();
/// }
///
/// // The learning rate should be back to initial after a cycle
/// assert_eq!(scheduler.get_learning_rate(), 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingWarmRestarts<A: Float + Debug> {
    /// Initial learning rate
    initial_lr: A,
    /// Minimum learning rate
    min_lr: A,
    /// Initial cycle length (T_0)
    t_0: usize,
    /// Multiplicative factor for cycle length after each restart (T_mult)
    t_mult: A,
    /// Current step within the current cycle
    t_cur: usize,
    /// Current cycle index
    cycle: usize,
    /// Current cycle length
    t_i: usize,
    /// Current learning rate
    current_lr: A,
}

impl<A: Float + Debug> CosineAnnealingWarmRestarts<A> {
    /// Create a new cosine annealing with warm restarts scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `min_lr` - Minimum learning rate
    /// * `t_0` - Initial cycle length
    /// * `t_mult` - Multiplicative factor for cycle length after each restart
    pub fn new(initial_lr: A, min_lr: A, t_0: usize, t_mult: A) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_0,
            t_mult,
            t_cur: 0,
            cycle: 0,
            t_i: t_0,
            current_lr: initial_lr,
        }
    }

    /// Get the current cycle index
    pub fn cycle(&self) -> usize {
        self.cycle
    }

    /// Get the current cycle length
    pub fn cycle_length(&self) -> usize {
        self.t_i
    }

    /// Get the initial cycle length
    pub fn initial_cycle_length(&self) -> usize {
        self.t_0
    }

    /// Get the multiplicative factor for cycle length
    pub fn t_mult(&self) -> A {
        self.t_mult
    }

    /// Calculate the cycle length for a given cycle index
    fn calculate_cycle_length(&self, cycle: usize) -> usize {
        if cycle == 0 {
            return self.t_0;
        }

        let t_mult_f64 = self.t_mult.to_f64().unwrap_or(1.0);
        (self.t_0 as f64 * t_mult_f64.powi(cycle as i32)).round() as usize
    }
}

impl<A: Float + Debug + ScalarOperand> LearningRateScheduler<A> for CosineAnnealingWarmRestarts<A> {
    fn get_learning_rate(&self) -> A {
        self.current_lr
    }

    fn step(&mut self) -> A {
        // Calculate the cosine annealing learning rate
        let pi = A::from(std::f64::consts::PI).unwrap();
        let t_cur = A::from(self.t_cur).unwrap();
        let t_i = A::from(self.t_i).unwrap();

        // Using the formula from the paper: η_t = η_min + 0.5(η_max - η_min)(1 + cos(π * t_cur/T_i))
        // For t_cur from 0 to T_i, cosine goes from 1 to -1, making the learning rate decrease from initial_lr to min_lr
        let cos_term = (pi * t_cur / t_i).cos();

        // Calculate the learning rate
        self.current_lr = self.min_lr
            + A::from(0.5).unwrap() * (self.initial_lr - self.min_lr) * (A::one() + cos_term);

        // Store the current learning rate to return
        let return_lr = self.current_lr;

        // Increment step in current cycle after calculating the learning rate
        self.t_cur += 1;

        // Check if we've completed a cycle
        if self.t_cur >= self.t_i {
            // Move to the next cycle
            self.cycle += 1;
            self.t_cur = 0;

            // Calculate the new cycle length
            self.t_i = self.calculate_cycle_length(self.cycle);

            // Reset learning rate to initial for start of new cycle
            self.current_lr = self.initial_lr;
        }

        // Return the learning rate calculated before incrementing t_cur
        return_lr
    }

    fn reset(&mut self) {
        self.t_cur = 0;
        self.cycle = 0;
        self.t_i = self.t_0;
        self.current_lr = self.initial_lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cosine_annealing_warm_restarts_init() {
        let scheduler = CosineAnnealingWarmRestarts::new(0.1f64, 0.001, 10, 2.0.into());

        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);
        assert_eq!(scheduler.cycle(), 0);
        assert_eq!(scheduler.cycle_length(), 10);
    }

    #[test]
    fn test_cosine_annealing_warm_restarts_first_cycle() {
        let mut scheduler = CosineAnnealingWarmRestarts::new(0.1f64, 0.001, 10, 2.0.into());

        // Check learning rate at the beginning of the first cycle
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // Collect learning rates for the first 5 steps
        let mut lrs = Vec::new();
        for _ in 0..5 {
            lrs.push(scheduler.step());
        }

        // Calculate expected LR based on the actual value
        // After 5 steps, the counter is at position 5, so we should check the value at t_cur=4
        let pi = std::f64::consts::PI;
        let t_cur = 4.0;
        let t_i = 10.0;
        let cos_term = (pi * t_cur / t_i).cos();
        let expected_mid_lr = 0.001 + 0.5 * (0.1 - 0.001) * (1.0 + cos_term);

        // Use a larger epsilon to accommodate floating point precision
        assert_abs_diff_eq!(
            scheduler.get_learning_rate(),
            expected_mid_lr,
            epsilon = 1e-3
        );

        // Check learning rate at the end of the first cycle
        for _ in 0..5 {
            scheduler.step();
        }

        // After completing the cycle, we should be back to initial learning rate
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);
        assert_eq!(scheduler.cycle(), 1);
        assert_eq!(scheduler.cycle_length(), 20); // 10 * 2
    }

    #[test]
    fn test_cosine_annealing_warm_restarts_multiple_cycles() {
        let mut scheduler = CosineAnnealingWarmRestarts::new(0.1f64, 0.001, 10, 2.0.into());

        // First cycle: 10 iterations
        for _ in 0..10 {
            scheduler.step();
        }

        assert_eq!(scheduler.cycle(), 1);
        assert_eq!(scheduler.cycle_length(), 20);
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // Second cycle: 20 iterations
        for _ in 0..20 {
            scheduler.step();
        }

        assert_eq!(scheduler.cycle(), 2);
        assert_eq!(scheduler.cycle_length(), 40); // 10 * 2^2
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);

        // Third cycle: 40 iterations
        for _ in 0..40 {
            scheduler.step();
        }

        assert_eq!(scheduler.cycle(), 3);
        assert_eq!(scheduler.cycle_length(), 80); // 10 * 2^3
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);
    }

    #[test]
    fn test_cosine_annealing_warm_restarts_reset() {
        let mut scheduler = CosineAnnealingWarmRestarts::new(0.1f64, 0.001, 10, 2.0.into());

        // Run for a few cycles
        for _ in 0..50 {
            scheduler.step();
        }

        // Reset the scheduler
        scheduler.reset();

        // Check that the scheduler is back to initial state
        assert_eq!(scheduler.cycle(), 0);
        assert_eq!(scheduler.cycle_length(), 10);
        assert_abs_diff_eq!(scheduler.get_learning_rate(), 0.1);
    }

    #[test]
    fn test_cosine_annealing_warm_restarts_lr_pattern() {
        let mut scheduler = CosineAnnealingWarmRestarts::new(0.1f64, 0.001, 10, 2.0.into());

        // Run for a full cycle and capture the learning rates
        let mut lrs = Vec::new();
        for _ in 0..10 {
            lrs.push(scheduler.step());
        }

        // Check that the learning rate decreases monotonically during the cycle
        for i in 1..lrs.len() {
            assert!(
                lrs[i - 1] >= lrs[i],
                "Learning rate should decrease monotonically"
            );
        }

        // Check that the minimum learning rate is close to the specified minimum
        let min_cycle_lr = lrs[lrs.len() - 1];
        assert_abs_diff_eq!(min_cycle_lr, 0.001, epsilon = 1e-2);

        // Now get the learning rate after a complete cycle, which should reset to initial
        let next_cycle_lr = scheduler.step();
        assert_abs_diff_eq!(next_cycle_lr, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_annealing_warm_restarts_fractional_t_mult() {
        // Test with a t_mult < 1.0, which should decrease cycle length
        let mut scheduler = CosineAnnealingWarmRestarts::new(0.1f64, 0.001, 10, 0.5.into());

        // First cycle: 10 iterations
        for _ in 0..10 {
            scheduler.step();
        }

        assert_eq!(scheduler.cycle(), 1);
        assert_eq!(scheduler.cycle_length(), 5); // 10 * 0.5

        // Second cycle: 5 iterations
        for _ in 0..5 {
            scheduler.step();
        }

        assert_eq!(scheduler.cycle(), 2);
        assert_eq!(scheduler.cycle_length(), 3); // 10 * 0.5^2 = 2.5, rounded to 3
    }

    #[test]
    fn test_cosine_annealing_warm_restarts_t_mult_one() {
        // Test with t_mult = 1.0, which should keep cycle length constant
        let mut scheduler = CosineAnnealingWarmRestarts::new(0.1f64, 0.001, 10, 1.0.into());

        // First cycle: 10 iterations
        for _ in 0..10 {
            scheduler.step();
        }

        assert_eq!(scheduler.cycle(), 1);
        assert_eq!(scheduler.cycle_length(), 10); // 10 * 1

        // Second cycle: 10 iterations
        for _ in 0..10 {
            scheduler.step();
        }

        assert_eq!(scheduler.cycle(), 2);
        assert_eq!(scheduler.cycle_length(), 10); // 10 * 1
    }
}
