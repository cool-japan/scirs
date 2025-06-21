//! Gradient accumulation for large batch training
//!
//! This module provides utilities for accumulating gradients across multiple
//! micro-batches to simulate larger batch sizes without increasing memory usage.

use crate::error::{OptimError, Result};
use ndarray::{Array, Dimension, ScalarOperand, Zip};
use num_traits::Float;
use std::fmt::Debug;

/// Type alias for adaptive step conditions
pub type AdaptiveStepCondition = Box<dyn Fn(usize) -> bool>;

/// Gradient accumulation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccumulationMode {
    /// Sum gradients (standard accumulation)
    Sum,
    /// Average gradients (normalize by number of accumulations)
    Average,
}

/// Gradient accumulator for micro-batch training
#[derive(Debug)]
pub struct GradientAccumulator<A: Float, D: Dimension> {
    /// Accumulated gradients
    accumulated_gradients: Vec<Array<A, D>>,
    /// Number of accumulation steps taken
    accumulation_count: usize,
    /// Target number of accumulations before update
    target_accumulations: usize,
    /// Accumulation mode
    mode: AccumulationMode,
    /// Whether accumulator has been initialized
    initialized: bool,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> GradientAccumulator<A, D> {
    /// Create a new gradient accumulator
    pub fn new(target_accumulations: usize, mode: AccumulationMode) -> Self {
        Self {
            accumulated_gradients: Vec::new(),
            accumulation_count: 0,
            target_accumulations,
            mode,
            initialized: false,
        }
    }

    /// Initialize accumulator with gradient shapes
    pub fn initialize(&mut self, gradients: &[Array<A, D>]) -> Result<()> {
        if self.initialized {
            return Err(OptimError::InvalidConfig(
                "Accumulator already initialized".to_string(),
            ));
        }

        self.accumulated_gradients = gradients
            .iter()
            .map(|g| Array::zeros(g.raw_dim()))
            .collect();

        self.initialized = true;
        Ok(())
    }

    /// Accumulate gradients from a micro-batch
    pub fn accumulate(&mut self, gradients: &[Array<A, D>]) -> Result<()> {
        if !self.initialized {
            self.initialize(gradients)?;
        }

        if gradients.len() != self.accumulated_gradients.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "Expected {} gradient arrays, got {}",
                self.accumulated_gradients.len(),
                gradients.len()
            )));
        }

        // Accumulate gradients
        for (acc_grad, micro_grad) in self.accumulated_gradients.iter_mut().zip(gradients.iter()) {
            if acc_grad.raw_dim() != micro_grad.raw_dim() {
                return Err(OptimError::DimensionMismatch(
                    "Gradient dimensions don't match".to_string(),
                ));
            }

            Zip::from(acc_grad).and(micro_grad).for_each(|acc, &micro| {
                *acc = *acc + micro;
            });
        }

        self.accumulation_count += 1;
        Ok(())
    }

    /// Check if accumulation is complete
    pub fn is_ready(&self) -> bool {
        self.accumulation_count >= self.target_accumulations
    }

    /// Get accumulated gradients and reset accumulator
    pub fn get_and_reset(&mut self) -> Result<Vec<Array<A, D>>> {
        if !self.is_ready() {
            return Err(OptimError::InvalidConfig(format!(
                "Accumulation not ready: {}/{} steps completed",
                self.accumulation_count, self.target_accumulations
            )));
        }

        let mut result = self.accumulated_gradients.clone();

        // Apply accumulation mode
        match self.mode {
            AccumulationMode::Sum => {
                // Gradients are already summed, nothing to do
            }
            AccumulationMode::Average => {
                let scale = A::one() / A::from(self.accumulation_count).unwrap();
                for grad in &mut result {
                    grad.mapv_inplace(|x| x * scale);
                }
            }
        }

        // Reset accumulator
        self.reset();

        Ok(result)
    }

    /// Reset accumulator state
    pub fn reset(&mut self) {
        for grad in &mut self.accumulated_gradients {
            grad.fill(A::zero());
        }
        self.accumulation_count = 0;
    }

    /// Get current accumulation count
    pub fn accumulation_count(&self) -> usize {
        self.accumulation_count
    }

    /// Get target accumulation count
    pub fn target_accumulations(&self) -> usize {
        self.target_accumulations
    }

    /// Set new target accumulation count
    pub fn set_target_accumulations(&mut self, target: usize) {
        self.target_accumulations = target;
    }

    /// Get accumulation mode
    pub fn mode(&self) -> AccumulationMode {
        self.mode
    }

    /// Set accumulation mode
    pub fn set_mode(&mut self, mode: AccumulationMode) {
        self.mode = mode;
    }

    /// Check if accumulator is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get current progress as a fraction (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        if self.target_accumulations == 0 {
            1.0
        } else {
            self.accumulation_count as f64 / self.target_accumulations as f64
        }
    }
}

/// Variable accumulation scheduler
pub struct VariableAccumulator<A: Float, D: Dimension> {
    /// Base accumulator
    accumulator: GradientAccumulator<A, D>,
    /// Variable accumulation steps based on conditions
    adaptive_steps: Vec<(AdaptiveStepCondition, usize)>,
    /// Current step count
    step_count: usize,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> VariableAccumulator<A, D> {
    /// Create a new variable accumulator
    pub fn new(initial_target: usize, mode: AccumulationMode) -> Self {
        Self {
            accumulator: GradientAccumulator::new(initial_target, mode),
            adaptive_steps: Vec::new(),
            step_count: 0,
        }
    }

    /// Add a condition-based accumulation rule
    pub fn add_adaptive_rule<F>(&mut self, condition: F, accumulation_steps: usize)
    where
        F: Fn(usize) -> bool + 'static,
    {
        self.adaptive_steps
            .push((Box::new(condition), accumulation_steps));
    }

    /// Update target accumulations based on current step
    fn update_target(&mut self) {
        for (condition, steps) in &self.adaptive_steps {
            if condition(self.step_count) {
                self.accumulator.set_target_accumulations(*steps);
                break;
            }
        }
    }

    /// Accumulate gradients with adaptive targeting
    pub fn accumulate(&mut self, gradients: &[Array<A, D>]) -> Result<()> {
        self.update_target();
        self.accumulator.accumulate(gradients)
    }

    /// Check if accumulation is ready
    pub fn is_ready(&self) -> bool {
        self.accumulator.is_ready()
    }

    /// Get accumulated gradients and advance step
    pub fn get_and_step(&mut self) -> Result<Vec<Array<A, D>>> {
        let result = self.accumulator.get_and_reset()?;
        self.step_count += 1;
        Ok(result)
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get underlying accumulator
    pub fn accumulator(&self) -> &GradientAccumulator<A, D> {
        &self.accumulator
    }

    /// Get mutable reference to underlying accumulator
    pub fn accumulator_mut(&mut self) -> &mut GradientAccumulator<A, D> {
        &mut self.accumulator
    }
}

/// Micro-batch trainer that uses gradient accumulation
#[derive(Debug)]
pub struct MicroBatchTrainer<A: Float, D: Dimension> {
    /// Gradient accumulator
    accumulator: GradientAccumulator<A, D>,
    /// Micro-batch size
    micro_batch_size: usize,
    /// Effective batch size (micro_batch_size * accumulation_steps)
    effective_batch_size: usize,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> MicroBatchTrainer<A, D> {
    /// Create a new micro-batch trainer
    pub fn new(
        micro_batch_size: usize,
        effective_batch_size: usize,
        mode: AccumulationMode,
    ) -> Result<Self> {
        if effective_batch_size < micro_batch_size {
            return Err(OptimError::InvalidConfig(
                "Effective batch size must be >= micro batch size".to_string(),
            ));
        }

        let accumulation_steps = effective_batch_size / micro_batch_size;
        let accumulator = GradientAccumulator::new(accumulation_steps, mode);

        Ok(Self {
            accumulator,
            micro_batch_size,
            effective_batch_size,
        })
    }

    /// Process a micro-batch and accumulate gradients
    pub fn process_micro_batch(&mut self, gradients: &[Array<A, D>]) -> Result<()> {
        self.accumulator.accumulate(gradients)
    }

    /// Check if ready for optimizer step
    pub fn ready_for_step(&self) -> bool {
        self.accumulator.is_ready()
    }

    /// Get accumulated gradients for optimizer step
    pub fn get_accumulated_gradients(&mut self) -> Result<Vec<Array<A, D>>> {
        self.accumulator.get_and_reset()
    }

    /// Get micro-batch size
    pub fn micro_batch_size(&self) -> usize {
        self.micro_batch_size
    }

    /// Get effective batch size
    pub fn effective_batch_size(&self) -> usize {
        self.effective_batch_size
    }

    /// Get accumulation progress
    pub fn progress(&self) -> f64 {
        self.accumulator.progress()
    }

    /// Set new effective batch size
    pub fn set_effective_batch_size(&mut self, effective_batch_size: usize) -> Result<()> {
        if effective_batch_size < self.micro_batch_size {
            return Err(OptimError::InvalidConfig(
                "Effective batch size must be >= micro batch size".to_string(),
            ));
        }

        self.effective_batch_size = effective_batch_size;
        let accumulation_steps = effective_batch_size / self.micro_batch_size;
        self.accumulator
            .set_target_accumulations(accumulation_steps);
        Ok(())
    }
}

/// Utility functions for gradient accumulation
pub mod utils {
    use super::*;

    /// Calculate optimal micro-batch size given memory constraints
    pub fn calculate_micro_batch_size(
        total_batch_size: usize,
        max_memory_mb: usize,
        param_count: usize,
        bytes_per_param: usize,
    ) -> usize {
        // Estimate memory usage per sample
        let memory_per_sample = param_count * bytes_per_param * 3; // params + grads + activations
        let max_samples = (max_memory_mb * 1_000_000) / memory_per_sample;

        // Choose micro-batch size that divides total batch size evenly
        let mut micro_batch_size = max_samples.min(total_batch_size);
        while total_batch_size % micro_batch_size != 0 && micro_batch_size > 1 {
            micro_batch_size -= 1;
        }

        micro_batch_size.max(1)
    }

    /// Calculate accumulation steps needed
    pub fn calculate_accumulation_steps(total_batch_size: usize, micro_batch_size: usize) -> usize {
        total_batch_size.div_ceil(micro_batch_size) // Ceiling division
    }

    /// Validate gradient accumulation configuration
    pub fn validate_config(
        micro_batch_size: usize,
        effective_batch_size: usize,
        accumulation_steps: usize,
    ) -> Result<()> {
        if micro_batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "Micro batch size must be > 0".to_string(),
            ));
        }

        if effective_batch_size == 0 {
            return Err(OptimError::InvalidConfig(
                "Effective batch size must be > 0".to_string(),
            ));
        }

        if accumulation_steps == 0 {
            return Err(OptimError::InvalidConfig(
                "Accumulation steps must be > 0".to_string(),
            ));
        }

        if effective_batch_size != micro_batch_size * accumulation_steps {
            return Err(OptimError::InvalidConfig(format!(
                "Effective batch size ({}) != micro batch size ({}) * accumulation steps ({})",
                effective_batch_size, micro_batch_size, accumulation_steps
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_gradient_accumulator_sum() {
        let mut accumulator = GradientAccumulator::new(3, AccumulationMode::Sum);

        // First micro-batch
        let grad1 = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        accumulator.accumulate(&grad1).unwrap();
        assert!(!accumulator.is_ready());

        // Second micro-batch
        let grad2 = vec![Array1::from_vec(vec![2.0, 3.0, 4.0])];
        accumulator.accumulate(&grad2).unwrap();
        assert!(!accumulator.is_ready());

        // Third micro-batch
        let grad3 = vec![Array1::from_vec(vec![1.0, 1.0, 1.0])];
        accumulator.accumulate(&grad3).unwrap();
        assert!(accumulator.is_ready());

        // Get accumulated gradients
        let result = accumulator.get_and_reset().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_slice().unwrap(), &[4.0, 6.0, 8.0]); // Sum of all gradients

        // Should be reset
        assert!(!accumulator.is_ready());
        assert_eq!(accumulator.accumulation_count(), 0);
    }

    #[test]
    fn test_gradient_accumulator_average() {
        let mut accumulator = GradientAccumulator::new(2, AccumulationMode::Average);

        let grad1 = vec![Array1::from_vec(vec![2.0, 4.0])];
        let grad2 = vec![Array1::from_vec(vec![4.0, 2.0])];

        accumulator.accumulate(&grad1).unwrap();
        accumulator.accumulate(&grad2).unwrap();

        let result = accumulator.get_and_reset().unwrap();
        assert_eq!(result[0].as_slice().unwrap(), &[3.0, 3.0]); // Average of gradients
    }

    #[test]
    fn test_variable_accumulator() {
        let mut var_accumulator = VariableAccumulator::new(2, AccumulationMode::Sum);

        // Add rule: if step > 5, use 4 accumulation steps
        var_accumulator.add_adaptive_rule(|step| step > 5, 4);

        // First few steps should use 2 accumulations
        let grad = vec![Array1::from_vec(vec![1.0])];
        var_accumulator.accumulate(&grad).unwrap();
        var_accumulator.accumulate(&grad).unwrap();
        assert!(var_accumulator.is_ready());

        let _result = var_accumulator.get_and_step().unwrap();

        // Simulate more steps to trigger adaptive rule
        for _ in 0..6 {
            var_accumulator.accumulate(&grad).unwrap();
            var_accumulator.accumulate(&grad).unwrap();
            if var_accumulator.is_ready() {
                var_accumulator.get_and_step().unwrap();
            }
        }

        // Now should require 4 accumulations
        assert_eq!(var_accumulator.accumulator().target_accumulations(), 4);
    }

    #[test]
    fn test_micro_batch_trainer() {
        let mut trainer = MicroBatchTrainer::new(
            2, // micro batch size
            6, // effective batch size
            AccumulationMode::Sum,
        )
        .unwrap();

        assert_eq!(trainer.micro_batch_size(), 2);
        assert_eq!(trainer.effective_batch_size(), 6);

        let grad = vec![Array1::from_vec(vec![1.0, 1.0])];

        // Process 3 micro-batches (to reach effective batch size of 6)
        trainer.process_micro_batch(&grad).unwrap();
        assert!(!trainer.ready_for_step());

        trainer.process_micro_batch(&grad).unwrap();
        assert!(!trainer.ready_for_step());

        trainer.process_micro_batch(&grad).unwrap();
        assert!(trainer.ready_for_step());

        let result = trainer.get_accumulated_gradients().unwrap();
        assert_eq!(result[0].as_slice().unwrap(), &[3.0, 3.0]); // Sum of 3 micro-batches
    }

    #[test]
    fn test_calculate_micro_batch_size() {
        let micro_batch = utils::calculate_micro_batch_size(
            128,  // total batch size
            100,  // max memory MB
            1000, // param count
            8,    // bytes per param (f64)
        );

        // Should return a size that divides 128 evenly
        assert!(128 % micro_batch == 0);
        assert!(micro_batch > 0);
    }

    #[test]
    fn test_accumulation_steps_calculation() {
        assert_eq!(utils::calculate_accumulation_steps(128, 32), 4);
        assert_eq!(utils::calculate_accumulation_steps(100, 32), 4); // Ceiling division
        assert_eq!(utils::calculate_accumulation_steps(96, 32), 3);
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        utils::validate_config(32, 128, 4).unwrap();

        // Invalid: micro batch size is 0
        assert!(utils::validate_config(0, 128, 4).is_err());

        // Invalid: sizes don't match
        assert!(utils::validate_config(32, 100, 4).is_err());
    }

    #[test]
    fn test_accumulator_progress() {
        let mut accumulator = GradientAccumulator::new(4, AccumulationMode::Sum);

        assert_relative_eq!(accumulator.progress(), 0.0);

        let grad = vec![Array1::from_vec(vec![1.0])];

        accumulator.accumulate(&grad).unwrap();
        assert_relative_eq!(accumulator.progress(), 0.25);

        accumulator.accumulate(&grad).unwrap();
        assert_relative_eq!(accumulator.progress(), 0.5);

        accumulator.accumulate(&grad).unwrap();
        assert_relative_eq!(accumulator.progress(), 0.75);

        accumulator.accumulate(&grad).unwrap();
        assert_relative_eq!(accumulator.progress(), 1.0);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut accumulator = GradientAccumulator::new(2, AccumulationMode::Sum);

        let grad1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        accumulator.accumulate(&grad1).unwrap();

        // Try to accumulate gradients with different dimensions
        let grad2 = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        assert!(accumulator.accumulate(&grad2).is_err());

        // Try to accumulate different number of arrays
        let grad3 = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0, 4.0]),
        ];
        assert!(accumulator.accumulate(&grad3).is_err());
    }

    #[test]
    fn test_get_before_ready_error() {
        let mut accumulator = GradientAccumulator::new(3, AccumulationMode::Sum);

        let grad = vec![Array1::from_vec(vec![1.0])];
        accumulator.accumulate(&grad).unwrap();

        // Try to get gradients before accumulation is complete
        assert!(accumulator.get_and_reset().is_err());
    }
}
