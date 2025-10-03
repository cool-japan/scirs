//! Learning rate scheduler callbacks implementation

use crate::callbacks::{Callback, CallbackContext, CallbackTiming};
use crate::error::Result;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
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
impl<F: Float + Debug + ScalarOperand> StepDecay<F> {
    /// Create a new step decay scheduler
    ///
    /// # Arguments
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
    /// Get the current learning rate
    pub fn get_lr(&self) -> F {
        self.current_lr
    /// Update the learning rate based on the current step
    pub fn update_lr(&mut self, step: usize) {
        if step > 0 && step % self.step_size == 0 {
            self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
    /// Reset the learning rate to the initial value
    pub fn reset_to_initial(&mut self) {
        self.current_lr = self.initial_lr;
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
            _ => {}
        Ok(())
/// Reduce on plateau learning rate scheduler
/// Reduces the learning rate when a metric has stopped improving.
pub struct ReduceOnPlateau<F: Float + Debug + ScalarOperand> {
    /// Number of epochs with no improvement after which learning rate will be reduced
    patience: usize,
    /// Threshold for measuring the new optimum
    threshold: F,
    /// Whether to monitor validation loss (true) or training loss (false)
    monitor_val_loss: bool,
    /// Best value of the monitored metric so far
    best_value: Option<F>,
    /// Current count of epochs with no improvement
    patience_counter: usize,
impl<F: Float + Debug + ScalarOperand> ReduceOnPlateau<F> {
    /// Create a new reduce on plateau scheduler
    /// * `patience` - Number of epochs with no improvement after which learning rate will be reduced
    /// * `threshold` - Threshold for measuring the new optimum
    pub fn new(_initial_lr: F, factor: F, patience: usize, threshold: F, minlr: F) -> Self {
            patience,
            threshold,
            monitor_val_loss: true,
            best_value: None,
            patience_counter: 0,
    /// Configure to monitor training loss instead of validation loss
    pub fn monitor_train_loss(mut self) -> Self {
        self.monitor_val_loss = false;
        self
        self.best_value = None;
        self.patience_counter = 0;
impl<F: Float + Debug + ScalarOperand> Callback<F> for ReduceOnPlateau<F> {
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
/// Cosine annealing learning rate scheduler
/// Cycles the learning rate between a maximum and minimum value using a cosine function.
pub struct CosineAnnealingLR<F: Float + Debug + ScalarOperand> {
    /// Maximum learning rate
    max_lr: F,
    /// Number of epochs in a cycle
    cycle_epochs: usize,
    /// Total number of steps (epochs or batches)
    pub total_steps: usize,
    /// Phantom data for generic type
    _phantom: PhantomData<F>,
impl<F: Float + Debug + ScalarOperand> CosineAnnealingLR<F> {
    /// Create a new cosine annealing scheduler
    /// * `max_lr` - Maximum learning rate
    /// * `cycle_epochs` - Number of epochs in a cycle
    /// * `total_steps` - Total number of steps (epochs or batches)
        max_lr: F,
        cycle_epochs: usize,
        total_steps: usize,
            max_lr,
            cycle_epochs,
            current_lr: max_lr,
            total_steps_phantom: PhantomData,
    /// Get the initial learning rate (max_lr for cosine annealing)
        self.max_lr
    /// Calculate learning rate for the given step
    pub fn calculate_lr(&self, step: usize) -> F {
        let cycle = step % self.cycle_epochs;
        let percent = F::from(cycle).unwrap() / F::from(self.cycle_epochs).unwrap();
        // Use cosine function
        let cosine = (F::one() + (percent * F::from(std::f64::consts::PI).unwrap()).cos())
            / F::from(2.0).unwrap();
        self.min_lr + (self.max_lr - self.min_lr) * cosine
impl<F: Float + Debug + ScalarOperand> Callback<F> for CosineAnnealingLR<F> {
                let step = context.epoch;
                self.current_lr = self.calculate_lr(step);
