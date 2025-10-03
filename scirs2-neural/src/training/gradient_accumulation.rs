//! Gradient accumulation utilities

use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Configuration for gradient accumulation
#[derive(Debug, Clone)]
pub struct GradientAccumulationConfig {
    /// Number of batches to accumulate gradients over
    pub accumulation_steps: usize,
    /// Whether to normalize gradients by accumulation steps
    pub normalize: bool,
}

impl Default for GradientAccumulationConfig {
    fn default() -> Self {
        Self {
            accumulation_steps: 1,
            normalize: true,
        }
    }
}

/// Statistics for gradient tracking
#[derive(Debug, Clone)]
pub struct GradientStats<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive> {
    /// Average gradient norm
    pub avg_grad_norm: F,
    /// Maximum gradient norm
    pub max_grad_norm: F,
    /// Minimum gradient norm
    pub min_grad_norm: F,
}

/// Gradient accumulator for training
#[derive(Debug)]
pub struct GradientAccumulator<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive> {
    /// Configuration
    pub config: GradientAccumulationConfig,
    /// Current step in accumulation
    pub current_step: usize,
    /// Statistics
    pub stats: Option<GradientStats<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + FromPrimitive> GradientAccumulator<F> {
    /// Create a new gradient accumulator
    pub fn new(config: GradientAccumulationConfig) -> Self {
        Self {
            config,
            current_step: 0,
            stats: None,
        }
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.stats = None;
    }

    /// Check if we should apply accumulated gradients
    pub fn should_update(&self) -> bool {
        self.current_step >= self.config.accumulation_steps
    }
}
