//! Learning rate scheduler integration for optimizers
//!
//! This module provides a wrapper around optimizers that integrates
//! them with learning rate schedulers from the callbacks module.

use crate::callbacks::LearningRateScheduler;
use crate::error::Result;
use crate::optimizers::Optimizer;
use ndarray::{Array, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

/// Wrapper around an optimizer that integrates with a learning rate scheduler
///
/// This wrapper delegates all optimizer operations to the underlying optimizer,
/// but updates the learning rate using the provided scheduler before each update.
pub struct LRSchedulerOptimizer<O, S, F>
where
    F: Float + Debug + ScalarOperand,
    O: Optimizer<F>,
    S: LearningRateScheduler<F>,
{
    /// The underlying optimizer
    optimizer: O,
    /// The learning rate scheduler
    scheduler: S,
    /// Current step count (used to calculate progress)
    step: usize,
    /// Total steps (used to calculate progress)
    total_steps: usize,
    /// Phantom data to keep the type parameter F
    _phantom: std::marker::PhantomData<F>,
}

impl<O, S, F> LRSchedulerOptimizer<O, S, F>
where
    F: Float + Debug + ScalarOperand,
    O: Optimizer<F>,
    S: LearningRateScheduler<F>,
{
    /// Create a new LRSchedulerOptimizer with the given optimizer and scheduler
    ///
    /// # Arguments
    /// * `optimizer` - The optimizer to wrap
    /// * `scheduler` - The learning rate scheduler to use
    /// * `total_steps` - The total number of steps for calculating progress
    ///
    /// # Returns
    /// A new LRSchedulerOptimizer
    pub fn new(optimizer: O, scheduler: S, total_steps: usize) -> Self {
        Self {
            optimizer,
            scheduler,
            step: 0,
            total_steps,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get a reference to the underlying optimizer
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Get a mutable reference to the underlying optimizer
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Get a reference to the scheduler
    pub fn scheduler(&self) -> &S {
        &self.scheduler
    }

    /// Get a mutable reference to the scheduler
    pub fn scheduler_mut(&mut self) -> &mut S {
        &mut self.scheduler
    }

    /// Reset the step counter and scheduler state
    pub fn reset(&mut self) {
        self.step = 0;
        self.scheduler.reset();
    }

    /// Set the current step manually
    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }

    /// Get the current step
    pub fn get_step(&self) -> usize {
        self.step
    }
}

impl<O, S, F> Optimizer<F> for LRSchedulerOptimizer<O, S, F>
where
    F: Float + Debug + ScalarOperand,
    O: Optimizer<F>,
    S: LearningRateScheduler<F>,
{
    fn update(
        &mut self,
        params: &mut [Array<F, ndarray::IxDyn>],
        grads: &[Array<F, ndarray::IxDyn>],
    ) -> Result<()> {
        // Calculate progress as a value between 0.0 and 1.0
        let progress = if self.total_steps == 0 {
            0.0
        } else {
            self.step as f64 / self.total_steps as f64
        };

        // Update learning rate based on progress
        let new_lr = self.scheduler.get_learning_rate(progress)?;
        self.optimizer.set_learning_rate(new_lr);

        // Delegate update to the underlying optimizer
        let result = self.optimizer.update(params, grads);

        // Increment step counter if update was successful
        if result.is_ok() {
            self.step += 1;
        }

        result
    }

    fn get_learning_rate(&self) -> F {
        self.optimizer.get_learning_rate()
    }

    fn set_learning_rate(&mut self, lr: F) {
        self.optimizer.set_learning_rate(lr);
    }
}

/// Helper function to create an optimizer with a step decay learning rate schedule
pub fn with_step_decay<O, F>(
    optimizer: O,
    initial_lr: F,
    factor: F,
    step_size: usize,
    min_lr: F,
    total_steps: usize,
) -> LRSchedulerOptimizer<O, crate::callbacks::StepDecay<F>, F>
where
    F: Float + Debug + ScalarOperand,
    O: Optimizer<F>,
{
    let scheduler = crate::callbacks::StepDecay::new(
        initial_lr,
        factor,
        step_size,
        crate::callbacks::ScheduleMethod::Epoch,
        min_lr,
    );

    LRSchedulerOptimizer::new(optimizer, scheduler, total_steps)
}

/// Helper function to create an optimizer with a cosine annealing learning rate schedule
pub fn with_cosine_annealing<O, F>(
    optimizer: O,
    max_lr: F,
    min_lr: F,
    cycle_epochs: usize,
    total_steps: usize,
) -> LRSchedulerOptimizer<O, crate::callbacks::CosineAnnealingLR<F>, F>
where
    F: Float + Debug + ScalarOperand,
    O: Optimizer<F>,
{
    let scheduler = crate::callbacks::CosineAnnealingLR::new(
        max_lr,
        min_lr,
        cycle_epochs,
        crate::callbacks::ScheduleMethod::Epoch,
        total_steps,
    );

    LRSchedulerOptimizer::new(optimizer, scheduler, total_steps)
}
