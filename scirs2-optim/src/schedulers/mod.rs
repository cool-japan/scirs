//! Learning rate schedulers for optimizers
//!
//! This module provides various learning rate schedulers that adjust the learning rate
//! of optimizers during training based on different strategies.

use ndarray::{Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;

use crate::optimizers::Optimizer;

/// Trait for learning rate schedulers
pub trait LearningRateScheduler<A: Float + Debug + ScalarOperand> {
    /// Get the learning rate at the current step
    fn get_learning_rate(&self) -> A;

    /// Update the scheduler state and return the new learning rate
    fn step(&mut self) -> A;

    /// Apply the scheduler to an optimizer
    fn apply_to<D: Dimension, O: Optimizer<A, D>>(&self, optimizer: &mut O)
    where
        Self: Sized,
    {
        optimizer.set_learning_rate(self.get_learning_rate());
    }

    /// Reset the scheduler state
    fn reset(&mut self);
}

mod constant;
mod cosine_annealing;
mod cosine_annealing_warm_restarts;
mod curriculum;
mod custom_scheduler;
mod cyclic_lr;
mod exponential_decay;
mod linear_decay;
mod linear_warmup_decay;
mod noise_injection;
mod one_cycle;
mod reduce_on_plateau;
mod step_decay;

// Re-export schedulers
pub use constant::ConstantScheduler;
pub use cosine_annealing::CosineAnnealing;
pub use cosine_annealing_warm_restarts::CosineAnnealingWarmRestarts;
pub use curriculum::{CurriculumScheduler, CurriculumStage, TransitionStrategy};
pub use custom_scheduler::{CombinedScheduler, CustomScheduler, SchedulerBuilder};
pub use cyclic_lr::{CyclicLR, CyclicMode};
pub use exponential_decay::ExponentialDecay;
pub use linear_decay::LinearDecay;
pub use linear_warmup_decay::{DecayStrategy, LinearWarmupDecay};
pub use noise_injection::{NoiseDistribution, NoiseInjectionScheduler};
pub use one_cycle::{AnnealStrategy, OneCycle};
pub use reduce_on_plateau::ReduceOnPlateau;
pub use step_decay::StepDecay;
