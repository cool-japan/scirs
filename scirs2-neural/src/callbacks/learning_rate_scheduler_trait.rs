//! Learning Rate Scheduler trait
//!
//! This module defines the LearningRateScheduler trait used by the gradient accumulation
//! and other components that need to adjust learning rates.

use crate::error::Result;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// Trait for learning rate schedulers
pub trait LearningRateScheduler<F: Float + Debug + ScalarOperand> {
    /// Get the learning rate for the current progress level (0.0 to 1.0)
    fn get_learning_rate(&mut self, progress: f64) -> Result<F>;
    /// Reset the scheduler state
    fn reset(&mut self) {
        // Default implementation: do nothing
    }
}
// Implement LearningRateScheduler for the existing schedulers
impl<F: Float + Debug + ScalarOperand> LearningRateScheduler<F> for super::StepDecay<F> {
    fn get_learning_rate(&mut self, progress: f64) -> Result<F> {
        // Convert progress to step based on total steps
        let step = (progress * 100.0).floor() as usize; // Assuming 100 steps total
        self.update_lr(step);
        Ok(self.get_lr())
impl<F: Float + Debug + ScalarOperand> LearningRateScheduler<F> for super::CosineAnnealingLR<F> {
        let step = (progress * self.total_steps as f64).floor() as usize;
        let lr = self.calculate_lr(step);
        Ok(lr)
impl<F: Float + Debug + ScalarOperand> LearningRateScheduler<F> for super::ReduceOnPlateau<F> {
    fn get_learning_rate(&mut selfprogress: f64) -> Result<F> {
        // ReduceOnPlateau doesn't use progress - it relies on callback monitoring
        // So we just return the current learning rate
