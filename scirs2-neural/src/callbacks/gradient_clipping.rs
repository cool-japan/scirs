//! Gradient Clipping callback
//!
//! This module provides a callback for gradient clipping during training.

use super::{Callback, CallbackContext, CallbackTiming};
use crate::error::Result;
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::Float;
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
impl<F: Float + Debug + ScalarOperand + Display> GradientClipping<F> {
    /// Create a new gradient clipping callback using global norm
    pub fn by_global_norm(_max_norm: F, logstats: bool) -> Self {
        Self {
            max_norm,
            method: GradientClippingMethod::ClipByGlobalNorm,
            log_stats,
            clipping_applied: false,
            clipping_ratio: None,
        }
    }
    /// Create a new gradient clipping callback using value clipping
    pub fn by_value(_max_value: F, logstats: bool) -> Self {
            max_norm: max_value,
            method: GradientClippingMethod::ClipByValue,
    /// Returns whether clipping was applied in the last step
    pub fn was_clipping_applied(&self) -> bool {
        self.clipping_applied
    /// Returns the clipping ratio from the last step (if global norm method was used)
    pub fn get_clipping_ratio(&self) -> Option<F> {
        self.clipping_ratio
    /// Clip gradients by global norm
    fn clip_by_global_norm<L: Layer<F> + ?Sized>(&mut self, model: &mut L) -> Result<()> {
        let gradients = model.gradients();
        // Compute global norm
        let mut global_norm_sq = F::zero();
        for grad in &gradients {
            for &val in grad.iter() {
                global_norm_sq = global_norm_sq + val * val;
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
        } else {
            self.clipping_applied = false;
            self.clipping_ratio = None;
        Ok(())
    /// Clip gradients by value
    fn clip_by_value<L: Layer<F> + ?Sized>(&mut self, model: &mut L) -> Result<()> {
        // Check if any value exceeds the maximum
        let mut clipping_needed = false;
                if val.abs() > self.max_norm {
                    clipping_needed = true;
                    break;
                }
            if clipping_needed {
                break;
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
                    "Gradient value clipping applied - max value: {:.4}",
                    self.max_norm
impl<F: Float + Debug + ScalarOperand + Display> Callback<F> for GradientClipping<F> {
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()> {
        // The callback should be executed after each batch, before optimization
        if timing == CallbackTiming::AfterBatch {
            if let Some(_batch_loss) = context.batch_loss {
                // Access the model from the context
                if let Some(model) = context.model.as_mut() {
                    match self.method {
                        GradientClippingMethod::ClipByGlobalNorm => {
                            if let Err(e) = self.clip_by_global_norm(&mut **model) {
                                eprintln!("Error in clip_by_globalnorm: {}", e);
                            }
                        GradientClippingMethod::ClipByValue => {
                            if let Err(e) = self.clip_by_value(&mut **model) {
                                eprintln!("Error in clip_byvalue: {}", e);
                } else {
                    // Fallback behavior if model is not available
                    if self.log_stats {
                        println!("Gradient clipping: model not available in context");
                    self.clipping_applied = false;
