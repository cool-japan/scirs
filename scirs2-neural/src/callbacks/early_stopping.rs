//! Early stopping callback implementation

use crate::callbacks::{Callback, CallbackContext, CallbackTiming};
use crate::error::Result;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// Early stopping callback that stops training when
/// a monitored metric has stopped improving.
pub struct EarlyStopping<F: Float + Debug + ScalarOperand> {
    /// Number of epochs with no improvement after which training will be stopped
    patience: usize,
    /// Minimum change in the monitored quantity to qualify as an improvement
    min_delta: F,
    /// Current count of epochs with no improvement
    patience_counter: usize,
    /// Best value of the monitored metric so far
    best_value: Option<F>,
    /// Whether to restore the model weights from the epoch with the best value
    restore_best_weights: bool,
    /// Whether to monitor validation loss (true) or training loss (false)
    monitor_val_loss: bool,
    /// Whether to monitor if values are decreasing (lower is better) or increasing (higher is better)
    monitor_decrease: bool,
}
impl<F: Float + Debug + ScalarOperand> EarlyStopping<F> {
    /// Create a new early stopping callback that monitors validation loss (lower is better)
    ///
    /// # Arguments
    /// * `patience` - Number of epochs with no improvement after which training will be stopped
    /// * `min_delta` - Minimum change in the monitored quantity to qualify as an improvement
    /// * `restore_best_weights` - Whether to restore the model weights from the epoch with the best value
    pub fn new(_patience: usize, min_delta: F, restore_bestweights: bool) -> Self {
        Self {
            patience,
            min_delta,
            _patience_counter: 0,
            best_value: None,
            restore_best_weights,
            monitor_val_loss: true,
            monitor_decrease: true,
        }
    }
    /// Configure to monitor training loss instead of validation loss
    pub fn monitor_train_loss(mut self) -> Self {
        self.monitor_val_loss = false;
        self
    /// Configure to monitor if values are increasing (higher is better)
    /// Default is monitoring decreases (lower is better)
    pub fn monitor_increase(mut self) -> Self {
        self.monitor_decrease = false;
impl<F: Float + Debug + ScalarOperand> Callback<F> for EarlyStopping<F> {
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
                        let improved = if self.monitor_decrease {
                            // Lower is better
                            current < best - self.min_delta
                        } else {
                            // Higher is better
                            current > best + self.min_delta
                        };
                        if improved {
                            // Reset counter and update best value
                            self.best_value = Some(current);
                            self.patience_counter = 0;
                            // If we're restoring best weights, save them here
                            if self.restore_best_weights {
                                // In a real implementation, we'd save the model weights here
                                // self.best_weights = Some(model.get_weights());
                            }
                            // Increment counter
                            self.patience_counter += 1;
                            // If patience is exceeded, stop training
                            if self.patience_counter >= self.patience {
                                println!(
                                    "Early stopping triggered. No improvement after {} epochs.",
                                    self.patience
                                );
                                context.stop_training = true;
                        }
                }
            }
        } else if timing == CallbackTiming::AfterTraining {
            // Restore best weights if needed
            if self.restore_best_weights {
                // In a real implementation, we'd restore the model weights here
                // if let Some(weights) = &self.best_weights {
                //     model.set_weights(weights);
                // }
        Ok(())
