//! Custom scheduler framework
//!
//! This module provides a flexible framework for creating custom learning rate schedulers
//! using closures and function combinators.

use ndarray::ScalarOperand;
use num_traits::Float;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

use super::LearningRateScheduler;

/// A custom scheduler that uses a closure to compute the learning rate
pub struct CustomScheduler<A, F>
where
    A: Float + Debug + ScalarOperand,
    F: FnMut(usize) -> A,
{
    /// Function to compute the learning rate, wrapped in RefCell for interior mutability
    lr_func: Rc<RefCell<F>>,
    /// Current step count
    step_count: usize,
    /// Phantom data for type parameter
    _phantom: PhantomData<A>,
}

impl<A, F> CustomScheduler<A, F>
where
    A: Float + Debug + ScalarOperand,
    F: FnMut(usize) -> A,
{
    /// Create a new custom scheduler with the given initial learning rate and computation function
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - The initial learning rate (used only for documentation)
    /// * `lr_func` - A function that takes the current step count and returns the learning rate
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_optim::schedulers::{CustomScheduler, LearningRateScheduler};
    ///
    /// // Create a scheduler that reduces the learning rate by 10% every 10 steps
    /// let mut scheduler = CustomScheduler::new(0.1, |step| {
    ///     0.1 * 0.9_f64.powi((step / 10) as i32)
    /// });
    ///
    /// assert_eq!(scheduler.get_learning_rate(), 0.1);
    /// scheduler.step();
    /// ```
    pub fn new(_initial_lr: A, lr_func: F) -> Self {
        Self {
            lr_func: Rc::new(RefCell::new(lr_func)),
            step_count: 0,
            _phantom: PhantomData,
        }
    }

    /// Get the current step count
    pub fn get_step_count(&self) -> usize {
        self.step_count
    }
}

impl<A, F> LearningRateScheduler<A> for CustomScheduler<A, F>
where
    A: Float + Debug + ScalarOperand,
    F: FnMut(usize) -> A,
{
    fn get_learning_rate(&self) -> A {
        // Borrow the function mutably and call it
        let mut func = self.lr_func.borrow_mut();
        func(self.step_count)
    }

    fn step(&mut self) -> A {
        self.step_count += 1;
        self.get_learning_rate()
    }

    fn reset(&mut self) {
        self.step_count = 0;
    }
}

/// Scheduler combinator that allows combining multiple schedulers
pub struct CombinedScheduler<A, F1, F2, C>
where
    A: Float + Debug + ScalarOperand,
    F1: FnMut(usize) -> A,
    F2: FnMut(usize) -> A,
    C: FnMut(A, A) -> A,
{
    /// First scheduler
    scheduler1: CustomScheduler<A, F1>,
    /// Second scheduler
    scheduler2: CustomScheduler<A, F2>,
    /// Combinator function wrapped in RefCell for interior mutability
    combinator: Rc<RefCell<C>>,
}

impl<A, F1, F2, C> CombinedScheduler<A, F1, F2, C>
where
    A: Float + Debug + ScalarOperand,
    F1: FnMut(usize) -> A,
    F2: FnMut(usize) -> A,
    C: FnMut(A, A) -> A,
{
    /// Create a new combined scheduler
    ///
    /// # Arguments
    ///
    /// * `scheduler1` - The first scheduler
    /// * `scheduler2` - The second scheduler
    /// * `combinator` - A function that combines the learning rates from both schedulers
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_optim::schedulers::{CustomScheduler, CombinedScheduler, LearningRateScheduler};
    ///
    /// // Create a scheduler that uses both exponential decay and cosine annealing
    /// let exponential = CustomScheduler::new(0.1, |step| {
    ///     0.1 * 0.9_f64.powi((step / 10) as i32)
    /// });
    ///
    /// let cosine = CustomScheduler::new(0.1, |step| {
    ///     let total_steps = 100;
    ///     let min_lr = 0.001;
    ///     let progress = std::f64::consts::PI * (step as f64) / (total_steps as f64);
    ///     min_lr + (0.1 - min_lr) * (1.0 + progress.cos()) / 2.0
    /// });
    ///
    /// let mut scheduler = CombinedScheduler::new(
    ///     exponential,
    ///     cosine,
    ///     |lr1, lr2| (lr1 + lr2) / 2.0  // Average the learning rates
    /// );
    ///
    /// assert_eq!(scheduler.get_learning_rate(), 0.1);
    /// scheduler.step();
    /// ```
    pub fn new(
        scheduler1: CustomScheduler<A, F1>,
        scheduler2: CustomScheduler<A, F2>,
        combinator: C,
    ) -> Self {
        Self {
            scheduler1,
            scheduler2,
            combinator: Rc::new(RefCell::new(combinator)),
        }
    }
}

impl<A, F1, F2, C> LearningRateScheduler<A> for CombinedScheduler<A, F1, F2, C>
where
    A: Float + Debug + ScalarOperand,
    F1: FnMut(usize) -> A,
    F2: FnMut(usize) -> A,
    C: FnMut(A, A) -> A,
{
    fn get_learning_rate(&self) -> A {
        let lr1 = self.scheduler1.get_learning_rate();
        let lr2 = self.scheduler2.get_learning_rate();

        // Borrow the combinator function mutably and call it
        let mut combinator = self.combinator.borrow_mut();
        combinator(lr1, lr2)
    }

    fn step(&mut self) -> A {
        self.scheduler1.step();
        self.scheduler2.step();
        self.get_learning_rate()
    }

    fn reset(&mut self) {
        self.scheduler1.reset();
        self.scheduler2.reset();
    }
}

/// Builder for creating custom schedulers
pub struct SchedulerBuilder<A>
where
    A: Float + Debug + ScalarOperand,
{
    initial_lr: A,
}

impl<A> SchedulerBuilder<A>
where
    A: Float + Debug + ScalarOperand,
{
    /// Create a new scheduler builder with the given initial learning rate
    pub fn new(initial_lr: A) -> Self {
        Self { initial_lr }
    }

    /// Create a step decay scheduler
    ///
    /// # Arguments
    ///
    /// * `step_size` - The number of steps after which the learning rate is decayed
    /// * `gamma` - The decay factor
    pub fn step_decay(
        self,
        step_size: usize,
        gamma: A,
    ) -> CustomScheduler<A, impl FnMut(usize) -> A> {
        let initial_lr = self.initial_lr;
        CustomScheduler::new(initial_lr, move |step| {
            let decay_factor = gamma.powi((step / step_size) as i32);
            initial_lr * decay_factor
        })
    }

    /// Create an exponential decay scheduler
    ///
    /// # Arguments
    ///
    /// * `gamma` - The decay factor
    pub fn exponential_decay(self, gamma: A) -> CustomScheduler<A, impl FnMut(usize) -> A> {
        let initial_lr = self.initial_lr;
        CustomScheduler::new(initial_lr, move |step| initial_lr * gamma.powi(step as i32))
    }

    /// Create a linear decay scheduler
    ///
    /// # Arguments
    ///
    /// * `total_steps` - The total number of steps
    /// * `final_lr` - The final learning rate
    pub fn linear_decay(
        self,
        total_steps: usize,
        final_lr: A,
    ) -> CustomScheduler<A, impl FnMut(usize) -> A> {
        let initial_lr = self.initial_lr;
        let total_steps = A::from(total_steps).unwrap();
        CustomScheduler::new(initial_lr, move |step| {
            let step = A::from(step).unwrap();
            if step >= total_steps {
                final_lr
            } else {
                let progress = step / total_steps;
                initial_lr + progress * (final_lr - initial_lr)
            }
        })
    }

    /// Create a cosine annealing scheduler
    ///
    /// # Arguments
    ///
    /// * `total_steps` - The total number of steps
    /// * `min_lr` - The minimum learning rate
    pub fn cosine_annealing(
        self,
        total_steps: usize,
        min_lr: A,
    ) -> CustomScheduler<A, impl FnMut(usize) -> A> {
        let initial_lr = self.initial_lr;
        let total_steps = A::from(total_steps).unwrap();
        let pi = A::from(std::f64::consts::PI).unwrap();
        CustomScheduler::new(initial_lr, move |step| {
            let step = A::from(step).unwrap();
            if step >= total_steps {
                min_lr
            } else {
                let progress = pi * step / total_steps;
                min_lr + (initial_lr - min_lr) * (A::one() + progress.cos()) / (A::one() + A::one())
            }
        })
    }

    /// Create a cyclic learning rate scheduler
    ///
    /// # Arguments
    ///
    /// * `step_size` - The half cycle size
    /// * `max_lr` - The maximum learning rate
    /// * `mode` - The cycle mode (triangular, triangular2, or exp_range)
    pub fn cyclic_lr(
        self,
        step_size: usize,
        max_lr: A,
        mode: CyclicMode<A>,
    ) -> CustomScheduler<A, impl FnMut(usize) -> A> {
        let min_lr = self.initial_lr;
        let step_size = A::from(step_size).unwrap();
        let two = A::one() + A::one();

        // Move mode into the closure
        let mode_inner = mode;

        CustomScheduler::new(min_lr, move |step| {
            let step = A::from(step).unwrap();
            let cycle = (step / (two * step_size)).floor();
            let x = (step / step_size - two * cycle).abs();

            let scale = match mode_inner {
                CyclicMode::Triangular => A::one(),
                CyclicMode::Triangular2 => A::one() / (two.powi(cycle.to_i32().unwrap_or(0))),
                CyclicMode::ExpRange(gamma) => gamma.powi(step.to_i32().unwrap_or(0)),
            };

            min_lr + scale * (max_lr - min_lr) * (A::one() - x).max(A::zero())
        })
    }

    /// Create a custom scheduler with a user-defined function
    ///
    /// # Arguments
    ///
    /// * `func` - A function that takes the current step count and returns the learning rate
    pub fn custom<F>(self, func: F) -> CustomScheduler<A, F>
    where
        F: FnMut(usize) -> A,
    {
        CustomScheduler::new(self.initial_lr, func)
    }
}

/// Cyclic learning rate modes
#[derive(Debug, Clone, Copy)]
pub enum CyclicMode<A: Float> {
    /// Triangular mode
    Triangular,
    /// Triangular2 mode - Cycle amplitude is cut in half after each cycle
    Triangular2,
    /// ExpRange mode - Cycle amplitude is scaled by gamma^step
    ExpRange(A),
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_custom_scheduler() {
        let mut scheduler =
            CustomScheduler::new(0.1f64, |step| 0.1 * 0.9f64.powi((step / 10) as i32));

        assert_eq!(scheduler.get_learning_rate(), 0.1);
        assert_eq!(scheduler.step(), 0.1);
        assert_eq!(scheduler.step(), 0.1);

        // After 10 steps, we should see decay
        for _ in 0..8 {
            scheduler.step();
        }
        assert_relative_eq!(scheduler.get_learning_rate(), 0.09, epsilon = 1e-10);
    }

    #[test]
    fn test_combined_scheduler() {
        let scheduler1 = CustomScheduler::new(0.1f64, |step| 0.1 * 0.9f64.powi((step / 10) as i32));

        let scheduler2 = CustomScheduler::new(0.2f64, |step| 0.2 * 0.8f64.powi((step / 5) as i32));

        let mut combined =
            CombinedScheduler::new(scheduler1, scheduler2, |lr1, lr2| lr1 * 0.3 + lr2 * 0.7);

        assert_relative_eq!(
            combined.get_learning_rate(),
            0.1 * 0.3 + 0.2 * 0.7,
            epsilon = 1e-10
        );
        combined.step();
        assert_relative_eq!(
            combined.get_learning_rate(),
            0.1 * 0.3 + 0.2 * 0.7,
            epsilon = 1e-10
        );

        // After 5 steps, scheduler2 should decay
        for _ in 0..4 {
            combined.step();
        }
        assert_relative_eq!(
            combined.get_learning_rate(),
            0.1 * 0.3 + 0.2 * 0.8 * 0.7,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_scheduler_builder() {
        // Test step decay
        let mut step_scheduler = SchedulerBuilder::new(0.1f64).step_decay(10, 0.5);
        assert_eq!(step_scheduler.get_learning_rate(), 0.1);
        for _ in 0..10 {
            step_scheduler.step();
        }
        assert_relative_eq!(step_scheduler.get_learning_rate(), 0.05, epsilon = 1e-10);

        // Test exponential decay
        let mut exp_scheduler = SchedulerBuilder::new(0.1f64).exponential_decay(0.95);
        assert_eq!(exp_scheduler.get_learning_rate(), 0.1);
        exp_scheduler.step();
        assert_relative_eq!(
            exp_scheduler.get_learning_rate(),
            0.1 * 0.95,
            epsilon = 1e-10
        );

        // Test linear decay
        let mut linear_scheduler = SchedulerBuilder::new(0.1f64).linear_decay(100, 0.01);
        assert_eq!(linear_scheduler.get_learning_rate(), 0.1);
        linear_scheduler.step();
        assert_relative_eq!(
            linear_scheduler.get_learning_rate(),
            0.1 - 0.0009, // 0.1 + 1/100 * (0.01 - 0.1)
            epsilon = 1e-10
        );

        // Test cosine annealing
        let mut cosine_scheduler = SchedulerBuilder::new(0.1f64).cosine_annealing(100, 0.01);
        assert_eq!(cosine_scheduler.get_learning_rate(), 0.1);
        cosine_scheduler.step();
        // Check that the first step is less than the initial
        assert!(cosine_scheduler.get_learning_rate() < 0.1);
        // And greater than the minimum
        assert!(cosine_scheduler.get_learning_rate() > 0.01);
    }
}
