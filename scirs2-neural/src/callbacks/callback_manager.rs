//! Callback manager implementation
//!
//! This module provides the CallbackManager for managing multiple callbacks
//! during the neural network training process.

use crate::error::Result;
use crate::models::History;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use super::{Callback, CallbackContext, CallbackTiming};
/// Manager for multiple callbacks
pub struct CallbackManager<F: Float + Debug + ScalarOperand> {
    /// List of registered callbacks
    callbacks: Vec<Box<dyn Callback<F> + Send + Sync>>,
}
impl<F: Float + Debug + ScalarOperand> Debug for CallbackManager<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackManager")
            .field("callbacks_count", &self.callbacks.len())
            .finish()
    }
impl<F: Float + Debug + ScalarOperand> CallbackManager<F> {
    /// Create a new callback manager
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    /// Add a callback to the manager
    pub fn add_callback(&mut self, callback: Box<dyn Callback<F> + Send + Sync>) {
        self.callbacks.push(callback);
    /// Called at the beginning of training
    pub fn on_train_begin(&mut self) -> Result<()> {
        let mut context = CallbackContext {
            epoch: 0,
            total_epochs: 0,
            batch: 0,
            total_batches: 0,
            batch_loss: None,
            epoch_loss: None,
            val_loss: None,
            metrics: Vec::new(),
            history: &History::default(),
            stop_training: false,
            model: None,
        };
        for callback in &mut self.callbacks {
            callback.on_event(CallbackTiming::BeforeTraining, &mut context)?;
        Ok(())
    /// Called at the end of training
    pub fn on_train_end(&mut self) -> Result<()> {
            callback.on_event(CallbackTiming::AfterTraining, &mut context)?;
    /// Called at the beginning of an epoch
    pub fn on_epoch_begin(&mut self, epoch: usize) -> Result<()> {
            epoch,
            callback.on_event(CallbackTiming::BeforeEpoch, &mut context)?;
    /// Called at the end of an epoch
    pub fn on_epoch_end(&mut self, epoch: usize, metrics: &HashMap<String, F>) -> Result<bool> {
        let mut _stop_training = false;
        // Extract just the values from metrics
        let metrics_vec = metrics.values().cloned().collect::<Vec<_>>();
            batch_loss: metrics.get("loss").copied(),
            epoch_loss: metrics.get("loss").copied(),
            val_loss: metrics.get("val_loss").copied(),
            metrics: metrics_vec,
            callback.on_event(CallbackTiming::AfterEpoch, &mut context)?;
        let stop_training = context.stop_training;
        Ok(stop_training)
    /// Called at the beginning of a batch
    pub fn on_batch_begin(&mut self, batch: usize) -> Result<()> {
            batch,
            callback.on_event(CallbackTiming::BeforeBatch, &mut context)?;
    /// Called at the end of a batch
    pub fn on_batch_end(&mut self, batch: usize, metrics: &HashMap<String, F>) -> Result<()> {
            callback.on_event(CallbackTiming::AfterBatch, &mut context)?;
impl<F: Float + Debug + ScalarOperand> Default for CallbackManager<F> {
    fn default() -> Self {
        Self::new()
