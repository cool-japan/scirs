//! Gradient Clipping callback
//!
//! This module provides a callback for gradient clipping during training.

use super::{Callback, CallbackContext, CallbackTiming};
use crate::error::Result;
use crate::layers::Layer;

use ndarray::{Array, IxDyn, ScalarOperand};
use num_traits::Float;
use std::fmt::{Debug, Display};

/// Gradient clipping method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientClippingMethod {
    /// Clip by global norm (divides by global norm if it exceeds max_norm)
    ClipByGlobalNorm,
    /// Clip by value (clip each value to be within [-max_value, max_value])
    ClipByValue,
}

/// Gradient clipping callback
#[derive(Debug)]
pub struct GradientClipping<F: Float + Debug + ScalarOperand + Display> {
    /// Maximum norm for gradient clipping
    pub max_norm: F,
    /// Clipping method
    pub method: GradientClippingMethod,
    /// Whether to log clipping statistics
    pub log_stats: bool,
    /// Whether clipping was applied in the last step
    clipping_applied: bool,
    /// Clipping ratio in the last step (if global norm method is used)
    clipping_ratio: Option<F>,
}

impl<F: Float + Debug + ScalarOperand + Display> GradientClipping<F> {
    /// Create a new gradient clipping callback using global norm
    pub fn by_global_norm(max_norm: F, log_stats: bool) -> Self {
        Self {
            max_norm,
            method: GradientClippingMethod::ClipByGlobalNorm,
            log_stats,
            clipping_applied: false,
            clipping_ratio: None,
        }
    }

    /// Create a new gradient clipping callback using value clipping
    pub fn by_value(max_value: F, log_stats: bool) -> Self {
        Self {
            max_norm: max_value,
            method: GradientClippingMethod::ClipByValue,
            log_stats,
            clipping_applied: false,
            clipping_ratio: None,
        }
    }

    /// Clip gradients by global norm
    fn clip_by_global_norm<L: Layer<F>>(&mut self, model: &mut L) -> Result<()> {
        let gradients = model.gradients();

        // Compute global norm
        let mut global_norm_sq = F::zero();
        for grad in &gradients {
            for &val in grad.iter() {
                global_norm_sq = global_norm_sq + val * val;
            }
        }

        let global_norm = global_norm_sq.sqrt();

        // Clip if necessary
        if global_norm > self.max_norm {
            let scale = self.max_norm / global_norm;
            self.clipping_applied = true;
            self.clipping_ratio = Some(scale);

            let clipped_gradients: Vec<Array<F, IxDyn>> =
                gradients.iter().map(|grad| grad.clone() * scale).collect();

            // Apply clipped gradients
            model.set_gradients(&clipped_gradients)?;

            if self.log_stats {
                println!(
                    "Gradient clipping applied - global norm: {:.4}, scale: {:.4}",
                    global_norm, scale
                );
            }
        } else {
            self.clipping_applied = false;
            self.clipping_ratio = None;
        }

        Ok(())
    }

    /// Clip gradients by value
    fn clip_by_value<L: Layer<F>>(&mut self, model: &mut L) -> Result<()> {
        let gradients = model.gradients();

        // Check if any value exceeds the maximum
        let mut clipping_needed = false;
        for grad in &gradients {
            for &val in grad.iter() {
                if val.abs() > self.max_norm {
                    clipping_needed = true;
                    break;
                }
            }
            if clipping_needed {
                break;
            }
        }

        // Clip if necessary
        if clipping_needed {
            let clipped_gradients: Vec<Array<F, IxDyn>> = gradients
                .iter()
                .map(|grad| {
                    let mut clipped = grad.clone();
                    for val in clipped.iter_mut() {
                        if *val > self.max_norm {
                            *val = self.max_norm;
                        } else if *val < -self.max_norm {
                            *val = -self.max_norm;
                        }
                    }
                    clipped
                })
                .collect();

            // Apply clipped gradients
            model.set_gradients(&clipped_gradients)?;

            self.clipping_applied = true;

            if self.log_stats {
                println!(
                    "Gradient value clipping applied - max value: {:.4}",
                    self.max_norm
                );
            }
        } else {
            self.clipping_applied = false;
        }

        Ok(())
    }
}

impl<F: Float + Debug + ScalarOperand + Display> Callback<F> for GradientClipping<F> {
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()> {
        // The callback should be executed after each batch, before optimization
        if timing == CallbackTiming::AfterBatch {
            if let Some(batch_loss) = context.batch_loss {
                // We assume model and gradients are available in the context
                // In a real implementation, we would need to access these through the context

                // TODO: Implement proper model/gradient access through the context
                // For now, we just log that clipping would be applied
                if self.log_stats {
                    println!("Gradient clipping would be applied (callback execution successful)");
                }

                // In real implementation:
                // 1. Get model from context
                // 2. Apply clipping based on method
                // 3. Update metrics
            }
        }

        Ok(())
    }
}
