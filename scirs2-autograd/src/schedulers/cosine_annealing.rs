//! Cosine annealing learning rate scheduler

use crate::schedulers::LRScheduler;
use crate::Float;
use std::f64::consts::PI;

/// Cosine annealing learning rate scheduler
///
/// Sets the learning rate using a cosine annealing schedule, where the learning rate
/// follows a cosine curve from the initial value down to a minimum value and optionally
/// back up again (if using restarts).
///
/// The basic cosine annealing formula is:
/// `lr = eta_min + (eta_max - eta_min) * (1 + cos(π * T_cur / T_max)) / 2`
///
/// Where:
/// - `eta_min` is the minimum learning rate
/// - `eta_max` is the maximum learning rate (initial)
/// - `T_cur` is the current step
/// - `T_max` is the maximum number of steps in one cycle
///
/// This scheduler is particularly effective for deep learning as it allows for
/// both large and small learning rates during training, potentially helping
/// to escape local minima and fine-tune the solution.
///
/// # Example
/// ```
/// use scirs2_autograd::schedulers::{CosineAnnealingLR, LRScheduler};
///
/// let scheduler = CosineAnnealingLR::new(0.1f32, 0.001f32, 100);
///
/// // Initial learning rate
/// assert!((scheduler.get_lr(0) - 0.1).abs() < 1e-6);
///
/// // At middle of cycle, learning rate is at midpoint
/// // cos(π/2) = 0, so lr = eta_min + (eta_max - eta_min) * 0.5
/// let mid_lr = scheduler.get_lr(50);
/// assert!((mid_lr - (0.001 + (0.1 - 0.001) * 0.5)).abs() < 1e-3);
///
/// // At end of cycle, learning rate returns to minimum
/// // cos(π) = -1, so lr = eta_min + (eta_max - eta_min) * 0 = eta_min
/// assert!((scheduler.get_lr(100) - 0.001).abs() < 1e-6);
/// ```
pub struct CosineAnnealingLR<F: Float> {
    /// Maximum learning rate (initial value)
    pub eta_max: F,
    /// Minimum learning rate
    pub eta_min: F,
    /// Maximum number of steps in one cycle
    pub t_max: usize,
    /// Enable warm restarts (SGDR)
    pub restart: bool,
    /// Factor to increase t_max after each restart (only used with restart=true)
    pub t_mult: usize,
    /// Current cycle number (for restarts)
    #[allow(dead_code)]
    current_cycle: usize,
    /// Current t_max for this cycle (for restarts)
    #[allow(dead_code)]
    current_t_max: usize,
}

impl<F: Float> CosineAnnealingLR<F> {
    /// Create a new cosine annealing learning rate scheduler
    ///
    /// # Arguments
    /// * `eta_max` - Maximum learning rate (initial value)
    /// * `eta_min` - Minimum learning rate
    /// * `t_max` - Maximum number of steps in one cycle
    ///
    /// # Panics
    /// Panics if `t_max` is 0 or if `eta_min` >= `eta_max`
    pub fn new(eta_max: F, eta_min: F, t_max: usize) -> Self {
        assert!(t_max > 0, "t_max must be greater than 0");
        assert!(eta_min < eta_max, "eta_min must be less than eta_max");

        Self {
            eta_max,
            eta_min,
            t_max,
            restart: false,
            t_mult: 1,
            current_cycle: 0,
            current_t_max: t_max,
        }
    }

    /// Create a cosine annealing scheduler with warm restarts (SGDR)
    ///
    /// Stochastic Gradient Descent with Warm Restarts allows the learning rate
    /// to restart at the maximum value periodically, which can help escape
    /// local minima and explore the loss landscape more effectively.
    ///
    /// # Arguments
    /// * `eta_max` - Maximum learning rate (initial value)
    /// * `eta_min` - Minimum learning rate
    /// * `t_0` - Number of steps in the first cycle
    /// * `t_mult` - Factor to increase cycle length after each restart
    ///
    /// # Panics
    /// Panics if `t_0` is 0, `t_mult` is 0, or if `eta_min` >= `eta_max`
    pub fn with_warm_restarts(eta_max: F, eta_min: F, t_0: usize, t_mult: usize) -> Self {
        assert!(t_0 > 0, "t_0 must be greater than 0");
        assert!(t_mult > 0, "t_mult must be greater than 0");
        assert!(eta_min < eta_max, "eta_min must be less than eta_max");

        Self {
            eta_max,
            eta_min,
            t_max: t_0,
            restart: true,
            t_mult,
            current_cycle: 0,
            current_t_max: t_0,
        }
    }

    /// Create a standard cosine annealing scheduler with common parameters
    ///
    /// Uses eta_min = eta_max / 100, providing a 100x learning rate range.
    ///
    /// # Arguments
    /// * `eta_max` - Maximum learning rate
    /// * `t_max` - Maximum number of steps in one cycle
    pub fn standard(eta_max: F, t_max: usize) -> Self {
        let eta_min = eta_max / F::from(100.0).unwrap();
        Self::new(eta_max, eta_min, t_max)
    }

    /// Create a cosine annealing scheduler for fine-tuning
    ///
    /// Uses a smaller learning rate range (10x) suitable for fine-tuning scenarios.
    ///
    /// # Arguments
    /// * `eta_max` - Maximum learning rate
    /// * `t_max` - Maximum number of steps in one cycle
    pub fn for_fine_tuning(eta_max: F, t_max: usize) -> Self {
        let eta_min = eta_max / F::from(10.0).unwrap();
        Self::new(eta_max, eta_min, t_max)
    }

    /// Calculate the current position within the cycle and which cycle we're in
    ///
    /// For restarts, this determines the current cycle and position within that cycle.
    /// For non-restart mode, this just returns the position within the single cycle.
    ///
    /// # Arguments
    /// * `step` - Current step number
    ///
    /// # Returns
    /// (cycle_number, step_in_cycle, cycle_length)
    fn get_cycle_info(&self, step: usize) -> (usize, usize, usize) {
        if !self.restart {
            // Simple case: no restarts, just cycle within t_max
            let step_in_cycle = step % (self.t_max + 1);
            (0, step_in_cycle, self.t_max)
        } else {
            // With restarts: determine which cycle we're in and position within that cycle
            let mut cycle = 0;
            let mut remaining_steps = step;
            let mut cycle_length = self.t_max;

            // Find which cycle we're currently in
            while remaining_steps >= cycle_length {
                remaining_steps -= cycle_length;
                cycle += 1;
                cycle_length *= self.t_mult;
            }

            (cycle, remaining_steps, cycle_length)
        }
    }

    /// Get the current learning rate using the cosine annealing formula
    ///
    /// # Arguments
    /// * `step_in_cycle` - Current step within the cycle
    /// * `cycle_length` - Length of the current cycle
    ///
    /// # Returns
    /// Learning rate for the current position in the cycle
    fn cosine_annealing(&self, step_in_cycle: usize, cycle_length: usize) -> F {
        if cycle_length == 0 {
            return self.eta_max;
        }

        // Cosine annealing formula
        let t_cur = F::from(step_in_cycle).unwrap();
        let t_max = F::from(cycle_length).unwrap();

        // cos(π * t_cur / t_max)
        let cos_arg = F::from(PI).unwrap() * t_cur / t_max;
        let cos_val = cos_arg.to_f64().unwrap().cos();
        let cos_f = F::from(cos_val).unwrap();

        // lr = eta_min + (eta_max - eta_min) * (1 + cos(π * t_cur / t_max)) / 2
        let lr_range = self.eta_max - self.eta_min;
        let factor = (F::one() + cos_f) / F::from(2.0).unwrap();

        self.eta_min + lr_range * factor
    }

    /// Check if the current step is a restart step
    ///
    /// # Arguments
    /// * `step` - Current step number
    ///
    /// # Returns
    /// `true` if this step is a restart step (only relevant for warm restarts)
    pub fn is_restart_step(&self, step: usize) -> bool {
        if !self.restart || step == 0 {
            return false;
        }

        let (cycle, step_in_cycle, _) = self.get_cycle_info(step);
        step_in_cycle == 0 && cycle > 0
    }

    /// Get the next restart step
    ///
    /// # Arguments
    /// * `current_step` - Current step number
    ///
    /// # Returns
    /// Step number when the next restart will occur, or None if no restarts
    pub fn next_restart_step(&self, current_step: usize) -> Option<usize> {
        if !self.restart {
            return None;
        }

        let (_cycle, step_in_cycle, cycle_length) = self.get_cycle_info(current_step);
        let steps_until_restart = cycle_length - step_in_cycle;

        Some(current_step + steps_until_restart)
    }

    /// Get the learning rate at the minimum point of the current cycle
    pub fn min_lr(&self) -> F {
        self.eta_min
    }

    /// Get the learning rate at the maximum point of the current cycle
    pub fn max_lr(&self) -> F {
        self.eta_max
    }

    /// Get the learning rate range (max - min)
    pub fn lr_range(&self) -> F {
        self.eta_max - self.eta_min
    }
}

impl<F: Float> LRScheduler<F> for CosineAnnealingLR<F> {
    fn get_lr(&self, step: usize) -> F {
        let (_, step_in_cycle, cycle_length) = self.get_cycle_info(step);
        self.cosine_annealing(step_in_cycle, cycle_length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_annealing_basic() {
        let scheduler = CosineAnnealingLR::new(1.0f32, 0.0, 100);

        // At start, should be max learning rate
        assert!((scheduler.get_lr(0) - 1.0).abs() < 1e-6);

        // At middle (step 50), should be minimum learning rate
        // cos(π * 50/100) = cos(π/2) = 0, so lr = 0 + (1-0) * (1+0)/2 = 0.5
        assert!((scheduler.get_lr(50) - 0.5).abs() < 1e-6);

        // At end, should be min learning rate (cos(π) = -1)
        // lr = 0 + (1-0) * (1+(-1))/2 = 0
        assert!((scheduler.get_lr(100) - 0.0).abs() < 1e-6);

        // Should cycle - middle of next cycle
        assert!((scheduler.get_lr(151) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_with_min() {
        let scheduler = CosineAnnealingLR::new(1.0f32, 0.1, 100);

        // At start, should be max learning rate
        assert!((scheduler.get_lr(0) - 1.0).abs() < 1e-6);

        // At middle (step 50), cos(π/2) = 0, so lr = 0.1 + (1.0-0.1) * (1+0)/2 = 0.1 + 0.45 = 0.55
        assert!((scheduler.get_lr(50) - 0.55).abs() < 1e-6);

        // At end, should be min learning rate (cos(π) = -1)
        // lr = 0.1 + (1.0-0.1) * (1+(-1))/2 = 0.1
        assert!((scheduler.get_lr(100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_warm_restarts() {
        let scheduler = CosineAnnealingLR::with_warm_restarts(1.0f32, 0.0, 10, 2);

        // First cycle: 0-9 (10 steps total)
        assert!((scheduler.get_lr(0) - 1.0).abs() < 1e-6);
        // At middle of first cycle: cos(π * 5/10) = cos(π/2) = 0, so lr = 0 + (1-0) * (1+0)/2 = 0.5
        assert!((scheduler.get_lr(5) - 0.5).abs() < 1e-6);
        // Step 10 is the start of cycle 1 (restart), so it should be max again
        assert!((scheduler.get_lr(10) - 1.0).abs() < 1e-6);

        // Second cycle should be longer (10 * 2 = 20): 10-29
        // Step 10 is already tested above as the start of cycle 1
        // Let's test step 15 (middle of second cycle)
        let lr_15 = scheduler.get_lr(15);
        // Step 15 is step_in_cycle 5 in a 20-step cycle: cos(π * 5/20) = cos(π/4) ≈ 0.707
        // lr = 0 + (1-0) * (1+0.707)/2 ≈ 0.854
        assert!((lr_15 - 0.8535534).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_presets() {
        let standard = CosineAnnealingLR::standard(1.0f32, 100);
        assert!((standard.eta_max - 1.0).abs() < 1e-6);
        assert!((standard.eta_min - 0.01).abs() < 1e-6);

        let fine_tune = CosineAnnealingLR::for_fine_tuning(0.1f32, 50);
        assert!((fine_tune.eta_max - 0.1).abs() < 1e-6);
        assert!((fine_tune.eta_min - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_cycle_info() {
        let scheduler = CosineAnnealingLR::with_warm_restarts(1.0f32, 0.0, 10, 2);

        // First cycle (steps 0-9)
        assert_eq!(scheduler.get_cycle_info(0), (0, 0, 10));
        assert_eq!(scheduler.get_cycle_info(5), (0, 5, 10));
        assert_eq!(scheduler.get_cycle_info(9), (0, 9, 10));

        // Second cycle (steps 10-29, length 20)
        assert_eq!(scheduler.get_cycle_info(10), (1, 0, 20));
        assert_eq!(scheduler.get_cycle_info(15), (1, 5, 20));
        assert_eq!(scheduler.get_cycle_info(25), (1, 15, 20));
    }

    #[test]
    fn test_cosine_annealing_restart_detection() {
        let scheduler = CosineAnnealingLR::with_warm_restarts(1.0f32, 0.0, 5, 1);

        assert!(!scheduler.is_restart_step(0)); // Start of first cycle
        assert!(!scheduler.is_restart_step(3)); // Middle of first cycle
        assert!(scheduler.is_restart_step(5)); // Start of second cycle
        assert!(scheduler.is_restart_step(10)); // Start of third cycle (t_mult=1, so same length)
    }

    #[test]
    fn test_cosine_annealing_lr_properties() {
        let scheduler = CosineAnnealingLR::new(1.0f32, 0.1, 100);

        assert_eq!(scheduler.min_lr(), 0.1);
        assert_eq!(scheduler.max_lr(), 1.0);
        assert!((scheduler.lr_range() - 0.9).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "t_max must be greater than 0")]
    fn test_cosine_annealing_zero_t_max() {
        CosineAnnealingLR::new(1.0f32, 0.1, 0);
    }

    #[test]
    #[should_panic(expected = "eta_min must be less than eta_max")]
    fn test_cosine_annealing_invalid_lr_range() {
        CosineAnnealingLR::new(0.1f32, 1.0, 100);
    }
}
