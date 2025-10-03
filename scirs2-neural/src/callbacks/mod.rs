//! Callback system for neural network training
//!
//! This module provides callbacks for customizing the training process,
//! such as early stopping, model checkpointing, and learning rate scheduling.

use crate::error::Result;
use crate::layers::Layer;
use crate::models::History;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::Float;
use std::fmt::Debug;
/// Enum for callback execution timing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackTiming {
    /// Execute before training begins
    BeforeTraining,
    /// Execute before each epoch
    BeforeEpoch,
    /// Execute before each batch
    BeforeBatch,
    /// Execute after each batch
    AfterBatch,
    /// Execute after each epoch
    AfterEpoch,
    /// Execute after training ends
    AfterTraining,
}
/// Struct containing state during training
pub struct CallbackContext<'a, F: Float + Debug + ScalarOperand> {
    /// Current epoch (0-based)
    pub epoch: usize,
    /// Total number of epochs
    pub total_epochs: usize,
    /// Current batch (0-based)
    pub batch: usize,
    /// Total number of batches in current epoch
    pub total_batches: usize,
    /// Loss for current batch
    pub batch_loss: Option<F>,
    /// Loss for current epoch
    pub epoch_loss: Option<F>,
    /// Validation loss for current epoch
    pub val_loss: Option<F>,
    /// Training metrics for current epoch
    pub metrics: Vec<F>,
    /// Training history so far
    pub history: &'a History<F>,
    /// Whether to stop training
    pub stop_training: bool,
    /// Optional reference to the model for gradient access
    pub model: Option<&'a mut dyn Layer<F>>,
/// Trait for training callbacks
pub trait Callback<F: Float + Debug + ScalarOperand> {
    /// Called during training at specific points
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()>;
mod callback_manager;
mod checkpoint;
mod early_stopping;
mod gradient_clipping;
mod learning_rate_scheduler;
mod learning_rate_scheduler_trait;
mod metrics;
mod model_checkpoint;
mod tensorboard;
mod visualization_callback;
/// Adapter to use a function as a callback
pub struct FunctionCallback<F: Float + Debug + ScalarOperand + Send + Sync> {
    /// The function to call
    func: Box<dyn Fn() -> Result<()> + Send + Sync>,
    /// Phantom data for F
    _phantom: std::marker::PhantomData<F>,
impl<F: Float + Debug + ScalarOperand + Send + Sync> FunctionCallback<F> {
    /// Create a new function callback
    pub fn new(func: Box<dyn Fn() -> Result<()> + Send + Sync>) -> Self {
        Self {
            _func_phantom: std::marker::PhantomData,
        }
    }
impl<F: Float + Debug + ScalarOperand + Send + Sync> Callback<F> for FunctionCallback<F> {
    fn on_event(
        &mut self_timing: CallbackTiming, _context: &mut CallbackContext<F>,
    ) -> Result<()> {
        (self.func)()
pub use callback_manager::CallbackManager;
pub use checkpoint::ModelCheckpoint;
pub use early_stopping::EarlyStopping;
pub use gradient_clipping::{GradientClipping, GradientClippingMethod};
pub use learning_rate_scheduler::{CosineAnnealingLR, ReduceOnPlateau, ScheduleMethod, StepDecay};
pub use learning_rate_scheduler_trait::LearningRateScheduler;
#[cfg(feature = "metrics_integration")]
pub use metrics::*;
pub use tensorboard::TensorBoardLogger;
pub use visualization_callback::VisualizationCallback;
