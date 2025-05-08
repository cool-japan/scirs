//! Learning rate scheduler callbacks implementation

use crate::callbacks::{Callback, CallbackContext, CallbackTiming};
use crate::error::Result;
use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Enumeration of scheduling methods for learning rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleMethod {
    /// Schedule on every epoch
    Epoch,
    /// Schedule on every batch
    Batch,
}

/// Step decay learning rate scheduler
///
/// Reduces the learning rate by a factor every n steps (epochs or batches).
pub struct StepDecay<F: Float + Debug + ScalarOperand> {
    /// Initial learning rate
    initial_lr: F,
    /// Factor to multiply learning rate by
    factor: F,
    /// Number of steps between reductions
    step_size: usize,
    /// Whether to schedule on epoch or batch
    method: ScheduleMethod,
    /// Current learning rate
    current_lr: F,
    /// Minimum learning rate
    min_lr: F,
}

impl<F: Float + Debug + ScalarOperand> StepDecay<F> {
    /// Create a new step decay scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `factor` - Factor to multiply learning rate by (should be less than 1.0)
    /// * `step_size` - Number of steps between reductions
    /// * `method` - Whether to schedule on epoch or batch
    /// * `min_lr` - Minimum learning rate
    pub fn new(
        initial_lr: F,
        factor: F,
        step_size: usize,
        method: ScheduleMethod,
        min_lr: F,
    ) -> Self {
        Self {
            initial_lr,
            factor,
            step_size,
            method,
            current_lr: initial_lr,
            min_lr,
        }
    }

    /// Get the initial learning rate
    pub fn get_initial_lr(&self) -> F {
        self.initial_lr
    }

    /// Get the current learning rate
    pub fn get_lr(&self) -> F {
        self.current_lr
    }

    /// Update the learning rate based on the current step
    pub fn update_lr(&mut self, step: usize) {
        if step > 0 && step % self.step_size == 0 {
            self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
        }
    }

    /// Reset the learning rate to the initial value
    pub fn reset_to_initial(&mut self) {
        self.current_lr = self.initial_lr;
    }
}

impl<F: Float + Debug + ScalarOperand> Callback<F> for StepDecay<F> {
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()> {
        match (timing, self.method) {
            (CallbackTiming::BeforeEpoch, ScheduleMethod::Epoch) => {
                self.update_lr(context.epoch);
                // In a real implementation, we'd update the optimizer's learning rate here
                // optimizer.set_learning_rate(self.current_lr);
                println!("Setting learning rate to: {:.6?}", self.current_lr);
            }
            (CallbackTiming::BeforeBatch, ScheduleMethod::Batch) => {
                let step = context.epoch * context.total_batches + context.batch;
                self.update_lr(step);
                // In a real implementation, we'd update the optimizer's learning rate here
                // optimizer.set_learning_rate(self.current_lr);
            }
            _ => {}
        }

        Ok(())
    }
}

/// Reduce on plateau learning rate scheduler
///
/// Reduces the learning rate when a metric has stopped improving.
pub struct ReduceOnPlateau<F: Float + Debug + ScalarOperand> {
    /// Factor to multiply learning rate by
    factor: F,
    /// Number of epochs with no improvement after which learning rate will be reduced
    patience: usize,
    /// Threshold for measuring the new optimum
    threshold: F,
    /// Minimum learning rate
    min_lr: F,
    /// Whether to monitor validation loss (true) or training loss (false)
    monitor_val_loss: bool,
    /// Best value of the monitored metric so far
    best_value: Option<F>,
    /// Current count of epochs with no improvement
    patience_counter: usize,
    /// Current learning rate
    current_lr: F,
    /// Initial learning rate
    initial_lr: F,
}

impl<F: Float + Debug + ScalarOperand> ReduceOnPlateau<F> {
    /// Create a new reduce on plateau scheduler
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `factor` - Factor to multiply learning rate by (should be less than 1.0)
    /// * `patience` - Number of epochs with no improvement after which learning rate will be reduced
    /// * `threshold` - Threshold for measuring the new optimum
    /// * `min_lr` - Minimum learning rate
    pub fn new(initial_lr: F, factor: F, patience: usize, threshold: F, min_lr: F) -> Self {
        Self {
            factor,
            patience,
            threshold,
            min_lr,
            monitor_val_loss: true,
            best_value: None,
            patience_counter: 0,
            current_lr: initial_lr,
            initial_lr,
        }
    }

    /// Configure to monitor training loss instead of validation loss
    pub fn monitor_train_loss(mut self) -> Self {
        self.monitor_val_loss = false;
        self
    }

    /// Get the initial learning rate
    pub fn get_initial_lr(&self) -> F {
        self.initial_lr
    }

    /// Get the current learning rate
    pub fn get_lr(&self) -> F {
        self.current_lr
    }

    /// Reset the learning rate to the initial value
    pub fn reset_to_initial(&mut self) {
        self.current_lr = self.initial_lr;
        self.best_value = None;
        self.patience_counter = 0;
    }
}

impl<F: Float + Debug + ScalarOperand> Callback<F> for ReduceOnPlateau<F> {
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()> {
        if timing == CallbackTiming::AfterEpoch {
            // Get the monitored value
            let current_value = if self.monitor_val_loss {
                context.val_loss
            } else {
                context.epoch_loss
            };

            // If we don't have a value to monitor, do nothing
            if let Some(current) = current_value {
                match self.best_value {
                    None => {
                        // First epoch, just store the value
                        self.best_value = Some(current);
                        self.patience_counter = 0;
                    }
                    Some(best) => {
                        // Check if there is improvement
                        if current < best - self.threshold {
                            // Reset counter and update best value
                            self.best_value = Some(current);
                            self.patience_counter = 0;
                        } else {
                            // Increment counter
                            self.patience_counter += 1;

                            // If patience is exceeded, reduce learning rate
                            if self.patience_counter >= self.patience {
                                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                                self.patience_counter = 0; // Reset counter

                                println!(
                                    "ReduceOnPlateau: No improvement detected, reducing learning rate to {:.6?}",
                                    self.current_lr
                                );

                                // In a real implementation, we'd update the optimizer's learning rate here
                                // optimizer.set_learning_rate(self.current_lr);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Cosine annealing learning rate scheduler
///
/// Cycles the learning rate between a maximum and minimum value using a cosine function.
pub struct CosineAnnealingLR<F: Float + Debug + ScalarOperand> {
    /// Maximum learning rate
    max_lr: F,
    /// Minimum learning rate
    min_lr: F,
    /// Number of epochs in a cycle
    cycle_epochs: usize,
    /// Current learning rate
    current_lr: F,
    /// Whether to schedule on epoch or batch
    method: ScheduleMethod,
    /// Total number of steps (epochs or batches)
    pub total_steps: usize,
    /// Phantom data for generic type
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand> CosineAnnealingLR<F> {
    /// Create a new cosine annealing scheduler
    ///
    /// # Arguments
    ///
    /// * `max_lr` - Maximum learning rate
    /// * `min_lr` - Minimum learning rate
    /// * `cycle_epochs` - Number of epochs in a cycle
    /// * `method` - Whether to schedule on epoch or batch
    /// * `total_steps` - Total number of steps (epochs or batches)
    pub fn new(
        max_lr: F,
        min_lr: F,
        cycle_epochs: usize,
        method: ScheduleMethod,
        total_steps: usize,
    ) -> Self {
        Self {
            max_lr,
            min_lr,
            cycle_epochs,
            current_lr: max_lr,
            method,
            total_steps,
            _phantom: PhantomData,
        }
    }

    /// Get the initial learning rate (max_lr for cosine annealing)
    pub fn get_initial_lr(&self) -> F {
        self.max_lr
    }

    /// Get the current learning rate
    pub fn get_lr(&self) -> F {
        self.current_lr
    }

    /// Calculate learning rate for the given step
    pub fn calculate_lr(&self, step: usize) -> F {
        let cycle = step % self.cycle_epochs;
        let percent = F::from(cycle).unwrap() / F::from(self.cycle_epochs).unwrap();

        // Use cosine function
        let cosine = (F::one() + (percent * F::from(std::f64::consts::PI).unwrap()).cos())
            / F::from(2.0).unwrap();
        self.min_lr + (self.max_lr - self.min_lr) * cosine
    }
}

impl<F: Float + Debug + ScalarOperand> Callback<F> for CosineAnnealingLR<F> {
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()> {
        match (timing, self.method) {
            (CallbackTiming::BeforeEpoch, ScheduleMethod::Epoch) => {
                let step = context.epoch;
                self.current_lr = self.calculate_lr(step);

                // In a real implementation, we'd update the optimizer's learning rate here
                // optimizer.set_learning_rate(self.current_lr);
                println!("Setting learning rate to: {:.6?}", self.current_lr);
            }
            (CallbackTiming::BeforeBatch, ScheduleMethod::Batch) => {
                let step = context.epoch * context.total_batches + context.batch;
                self.current_lr = self.calculate_lr(step);

                // In a real implementation, we'd update the optimizer's learning rate here
                // optimizer.set_learning_rate(self.current_lr);
            }
            _ => {}
        }

        Ok(())
    }
}
