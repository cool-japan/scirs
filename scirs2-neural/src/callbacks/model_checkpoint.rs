//! Model checkpoint callback implementation

use crate::callbacks::{Callback, CallbackContext, CallbackTiming};
use crate::error::Result;
use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

/// Model checkpoint callback that saves the model after every epoch
/// and optionally only saves the best models based on a monitored metric.
pub struct ModelCheckpoint<F: Float + Debug + ScalarOperand> {
    /// Directory to save the model
    filepath: PathBuf,
    /// Whether to save only the best model based on the monitored metric
    save_best_only: bool,
    /// Whether to monitor validation loss (true) or training loss (false)
    monitor_val_loss: bool,
    /// Whether to monitor if values are decreasing (lower is better) or increasing (higher is better)
    monitor_decrease: bool,
    /// Best value of the monitored metric so far
    best_value: Option<F>,
}

impl<F: Float + Debug + ScalarOperand> ModelCheckpoint<F> {
    /// Create a new model checkpoint callback
    ///
    /// # Arguments
    ///
    /// * `filepath` - Directory or file path to save the model
    /// * `save_best_only` - Whether to save only the best model based on the monitored metric
    #[allow(dead_code)]
    pub fn new<P: AsRef<Path>>(filepath: P, save_best_only: bool) -> Self {
        Self {
            filepath: filepath.as_ref().to_path_buf(),
            save_best_only,
            monitor_val_loss: true,
            monitor_decrease: true,
            best_value: None,
        }
    }

    /// Configure to monitor training loss instead of validation loss
    #[allow(dead_code)]
    pub fn monitor_train_loss(mut self) -> Self {
        self.monitor_val_loss = false;
        self
    }

    /// Configure to monitor if values are increasing (higher is better)
    /// Default is monitoring decreases (lower is better)
    #[allow(dead_code)]
    pub fn monitor_increase(mut self) -> Self {
        self.monitor_decrease = false;
        self
    }
}

impl<F: Float + Debug + ScalarOperand> Callback<F> for ModelCheckpoint<F> {
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()> {
        if timing == CallbackTiming::AfterEpoch {
            let should_save = if self.save_best_only {
                // Get the monitored value
                let current_value = if self.monitor_val_loss {
                    context.val_loss
                } else {
                    context.epoch_loss
                };

                if let Some(current) = current_value {
                    match self.best_value {
                        None => {
                            // First epoch, save the model
                            self.best_value = Some(current);
                            true
                        }
                        Some(best) => {
                            // Check if there is improvement
                            let improved = if self.monitor_decrease {
                                // Lower is better
                                current < best
                            } else {
                                // Higher is better
                                current > best
                            };

                            if improved {
                                // Update best value and save the model
                                self.best_value = Some(current);
                                true
                            } else {
                                false
                            }
                        }
                    }
                } else {
                    // No value to monitor, don't save
                    false
                }
            } else {
                // Always save
                true
            };

            if should_save {
                let epoch = context.epoch;
                let epoch_display = epoch + 1; // Convert to 1-based for display
                let filepath = if self.filepath.is_dir() {
                    self.filepath
                        .join(format!("model_epoch_{}.pth", epoch_display))
                } else {
                    self.filepath.clone()
                };

                println!("Saving model to: {}", filepath.display());

                // In a real implementation, we'd save the model here
                // model.save(filepath);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // No imports needed for this test

    #[test]
    fn test_model_checkpoint_creation() {
        // Test creating with default values
        let checkpoint = ModelCheckpoint::<f32>::new("test_path", true);
        assert_eq!(checkpoint.filepath.to_str().unwrap(), "test_path");
        assert!(checkpoint.save_best_only);
        assert!(checkpoint.monitor_val_loss);
        assert!(checkpoint.monitor_decrease);
        assert!(checkpoint.best_value.is_none());

        // Test monitor_train_loss
        let checkpoint = ModelCheckpoint::<f32>::new("test_path", true).monitor_train_loss();
        assert!(!checkpoint.monitor_val_loss);

        // Test monitor_increase
        let checkpoint = ModelCheckpoint::<f32>::new("test_path", true).monitor_increase();
        assert!(!checkpoint.monitor_decrease);
    }
}
