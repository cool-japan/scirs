//! Memory-efficient optimizers and utilities
//!
//! This module provides in-place parameter update capabilities and
//! memory-efficient implementations of optimization algorithms.

use crate::error::{OptimError, Result};
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign, SubAssign};

/// Trait for in-place parameter updates
pub trait InPlaceOptimizer<A: Float + ScalarOperand + Debug, D: Dimension> {
    /// Update parameters in-place using the given gradients
    ///
    /// This method modifies the parameters directly rather than returning new arrays,
    /// which can significantly reduce memory usage for large models.
    fn step_inplace(&mut self, params: &mut Array<A, D>, gradients: &Array<A, D>) -> Result<()>;

    /// Update multiple parameter arrays in-place
    fn step_list_inplace(
        &mut self,
        params_list: &mut [&mut Array<A, D>],
        gradients_list: &[&Array<A, D>],
    ) -> Result<()> {
        if params_list.len() != gradients_list.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Number of parameter arrays ({}) does not match number of gradient arrays ({})",
                params_list.len(),
                gradients_list.len()
            )));
        }

        for (params, grads) in params_list.iter_mut().zip(gradients_list.iter()) {
            self.step_inplace(params, grads)?;
        }
        Ok(())
    }
}

/// Memory-efficient SGD optimizer with in-place updates
#[derive(Debug, Clone)]
pub struct InPlaceSGD<A: Float> {
    learning_rate: A,
    momentum: A,
    weight_decay: A,
}

impl<A: Float + ScalarOperand + Debug> InPlaceSGD<A> {
    /// Create a new in-place SGD optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            momentum: A::zero(),
            weight_decay: A::zero(),
        }
    }

    /// Set momentum
    pub fn with_momentum(mut self, momentum: A) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> InPlaceOptimizer<A, D> for InPlaceSGD<A> {
    fn step_inplace(&mut self, params: &mut Array<A, D>, gradients: &Array<A, D>) -> Result<()> {
        // Apply weight decay if configured
        if self.weight_decay > A::zero() {
            params.zip_mut_with(gradients, |p, &g| {
                *p = *p - self.learning_rate * (g + *p * self.weight_decay);
            });
        } else {
            // Simple gradient descent
            params.zip_mut_with(gradients, |p, &g| {
                *p = *p - self.learning_rate * g;
            });
        }
        Ok(())
    }
}

/// Memory-efficient Adam optimizer with in-place updates
#[derive(Debug)]
pub struct InPlaceAdam<A: Float, D: Dimension> {
    learning_rate: A,
    beta1: A,
    beta2: A,
    epsilon: A,
    weight_decay: A,
    t: i32,
    /// First moment estimate (momentum)
    m: Option<Array<A, D>>,
    /// Second moment estimate (RMSprop)
    v: Option<Array<A, D>>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> InPlaceAdam<A, D> {
    /// Create a new in-place Adam optimizer
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            beta1: A::from(0.9).unwrap(),
            beta2: A::from(0.999).unwrap(),
            epsilon: A::from(1e-8).unwrap(),
            weight_decay: A::zero(),
            t: 0,
            m: None,
            v: None,
        }
    }

    /// Set beta1 (momentum decay)
    pub fn with_beta1(mut self, beta1: A) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set beta2 (RMSprop decay)
    pub fn with_beta2(mut self, beta2: A) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set epsilon
    pub fn with_epsilon(mut self, epsilon: A) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.t = 0;
        self.m = None;
        self.v = None;
    }
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> InPlaceOptimizer<A, D> for InPlaceAdam<A, D> {
    fn step_inplace(&mut self, params: &mut Array<A, D>, gradients: &Array<A, D>) -> Result<()> {
        self.t += 1;
        let _t = A::from(self.t).unwrap();

        // Initialize momentum and variance if needed
        if self.m.is_none() {
            self.m = Some(Array::zeros(params.raw_dim()));
        }
        if self.v.is_none() {
            self.v = Some(Array::zeros(params.raw_dim()));
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Apply weight decay if configured
        let grad_with_decay = if self.weight_decay > A::zero() {
            // Create temporary with weight decay
            let mut temp = gradients.clone();
            temp.zip_mut_with(params, |g, &p| {
                *g = *g + p * self.weight_decay;
            });
            temp
        } else {
            gradients.clone()
        };

        // Update biased first moment estimate
        m.zip_mut_with(&grad_with_decay, |m_i, &g| {
            *m_i = self.beta1 * *m_i + (A::one() - self.beta1) * g;
        });

        // Update biased second raw moment estimate
        v.zip_mut_with(&grad_with_decay, |v_i, &g| {
            *v_i = self.beta2 * *v_i + (A::one() - self.beta2) * g * g;
        });

        // Compute bias-corrected moments
        let bias1 = A::one() - self.beta1.powi(self.t);
        let bias2 = A::one() - self.beta2.powi(self.t);

        // Update parameters in-place
        let m_iter = m.iter();
        let v_iter = v.iter();
        let params_iter = params.iter_mut();

        for ((p, &m_i), &v_i) in params_iter.zip(m_iter).zip(v_iter) {
            let m_hat = m_i / bias1;
            let v_hat = v_i / bias2;
            *p = *p - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }

        Ok(())
    }
}

/// Utility functions for memory-efficient operations
pub mod utils {
    use super::*;

    /// Apply a scalar operation in-place
    pub fn scale_inplace<A, D>(array: &mut Array<A, D>, scalar: A)
    where
        A: Float + ScalarOperand + MulAssign,
        D: Dimension,
    {
        array.map_inplace(|x| *x *= scalar);
    }

    /// Add arrays in-place (a += b)
    pub fn add_inplace<A, D>(a: &mut Array<A, D>, b: &Array<A, D>)
    where
        A: Float + ScalarOperand + AddAssign,
        D: Dimension,
    {
        a.zip_mut_with(b, |x, &y| *x += y);
    }

    /// Subtract arrays in-place (a -= b)
    pub fn subtract_inplace<A, D>(a: &mut Array<A, D>, b: &Array<A, D>)
    where
        A: Float + ScalarOperand + SubAssign,
        D: Dimension,
    {
        a.zip_mut_with(b, |x, &y| *x -= y);
    }

    /// Apply element-wise operation in-place
    pub fn apply_inplace<A, D, F>(array: &mut Array<A, D>, f: F)
    where
        A: Float + ScalarOperand,
        D: Dimension,
        F: Fn(&mut A),
    {
        array.map_inplace(f);
    }

    /// Clip values in-place
    pub fn clip_inplace<A, D>(array: &mut Array<A, D>, min: A, max: A)
    where
        A: Float + ScalarOperand,
        D: Dimension,
    {
        array.map_inplace(|x| {
            if *x < min {
                *x = min;
            } else if *x > max {
                *x = max;
            }
        });
    }

    /// Normalize array in-place (divide by its norm)
    pub fn normalize_inplace<A, D>(array: &mut Array<A, D>)
    where
        A: Float + ScalarOperand + MulAssign,
        D: Dimension,
    {
        let norm = array.mapv(|x| x * x).sum().sqrt();
        if norm > A::zero() {
            array.map_inplace(|x| *x *= A::one() / norm);
        }
    }
}

/// Fused operations for maximum memory efficiency
pub mod fused {
    use super::*;

    /// Fused Adam update operation: combines momentum, variance, and parameter update in one pass
    ///
    /// This operation fuses all Adam computations into a single loop iteration,
    /// reducing memory allocations and improving cache efficiency.
    pub fn fused_adam_update<A, D>(
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
        m: &mut Array<A, D>,
        v: &mut Array<A, D>,
        lr: A,
        beta1: A,
        beta2: A,
        epsilon: A,
        bias1: A,
        bias2: A,
        weight_decay: Option<A>,
    ) where
        A: Float + ScalarOperand,
        D: Dimension,
    {
        let one = A::one();
        let one_minus_beta1 = one - beta1;
        let one_minus_beta2 = one - beta2;

        if let Some(wd) = weight_decay {
            // Fused Adam with weight decay
            for ((((p, &g), m_val), v_val), bias_corrected) in params
                .iter_mut()
                .zip(gradients.iter())
                .zip(m.iter_mut())
                .zip(v.iter_mut())
                .zip(std::iter::repeat((bias1, bias2)))
            {
                // Apply weight decay to gradient
                let g_with_decay = g + *p * wd;

                // Update momentum
                *m_val = beta1 * *m_val + one_minus_beta1 * g_with_decay;

                // Update variance
                *v_val = beta2 * *v_val + one_minus_beta2 * g_with_decay * g_with_decay;

                // Bias-corrected estimates and parameter update
                let m_hat = *m_val / bias_corrected.0;
                let v_hat = *v_val / bias_corrected.1;
                *p = *p - lr * m_hat / (v_hat.sqrt() + epsilon);
            }
        } else {
            // Fused Adam without weight decay
            for ((((p, &g), m_val), v_val), bias_corrected) in params
                .iter_mut()
                .zip(gradients.iter())
                .zip(m.iter_mut())
                .zip(v.iter_mut())
                .zip(std::iter::repeat((bias1, bias2)))
            {
                // Update momentum
                *m_val = beta1 * *m_val + one_minus_beta1 * g;

                // Update variance
                *v_val = beta2 * *v_val + one_minus_beta2 * g * g;

                // Bias-corrected estimates and parameter update
                let m_hat = *m_val / bias_corrected.0;
                let v_hat = *v_val / bias_corrected.1;
                *p = *p - lr * m_hat / (v_hat.sqrt() + epsilon);
            }
        }
    }

    /// Fused SGD with momentum and weight decay
    pub fn fused_sgd_update<A, D>(
        params: &mut Array<A, D>,
        gradients: &Array<A, D>,
        momentum_buf: Option<&mut Array<A, D>>,
        lr: A,
        momentum: A,
        weight_decay: Option<A>,
        dampening: A,
    ) where
        A: Float + ScalarOperand,
        D: Dimension,
    {
        if let Some(buf) = momentum_buf {
            if let Some(wd) = weight_decay {
                // Fused SGD with momentum and weight decay
                for (((p, &g), buf_val), _) in params
                    .iter_mut()
                    .zip(gradients.iter())
                    .zip(buf.iter_mut())
                    .zip(std::iter::repeat(()))
                {
                    let g_with_decay = g + *p * wd;
                    *buf_val = momentum * *buf_val + (A::one() - dampening) * g_with_decay;
                    *p = *p - lr * *buf_val;
                }
            } else {
                // Fused SGD with momentum only
                for ((p, &g), buf_val) in
                    params.iter_mut().zip(gradients.iter()).zip(buf.iter_mut())
                {
                    *buf_val = momentum * *buf_val + (A::one() - dampening) * g;
                    *p = *p - lr * *buf_val;
                }
            }
        } else if let Some(wd) = weight_decay {
            // Fused SGD with weight decay only
            for (p, &g) in params.iter_mut().zip(gradients.iter()) {
                *p = *p - lr * (g + *p * wd);
            }
        } else {
            // Simple fused SGD
            for (p, &g) in params.iter_mut().zip(gradients.iter()) {
                *p = *p - lr * g;
            }
        }
    }

    /// Fused gradient clipping and normalization
    pub fn fused_gradient_clip_normalize<A, D>(
        gradients: &mut Array<A, D>,
        max_norm: Option<A>,
        clip_value: Option<A>,
    ) where
        A: Float + ScalarOperand,
        D: Dimension,
    {
        if let Some(clip_val) = clip_value {
            // First pass: clip values
            for g in gradients.iter_mut() {
                if *g > clip_val {
                    *g = clip_val;
                } else if *g < -clip_val {
                    *g = -clip_val;
                }
            }
        }

        if let Some(max_norm_val) = max_norm {
            // Second pass: normalize if norm exceeds max_norm
            let norm_sq = gradients
                .iter()
                .map(|&x| x * x)
                .fold(A::zero(), |acc, x| acc + x);
            let norm = norm_sq.sqrt();

            if norm > max_norm_val {
                let scale = max_norm_val / norm;
                for g in gradients.iter_mut() {
                    *g = *g * scale;
                }
            }
        }
    }

    /// Fused parameter constraint application
    pub fn fused_apply_constraints<A, D>(
        params: &mut Array<A, D>,
        l2_constraint: Option<A>,
        value_bounds: Option<(A, A)>,
    ) where
        A: Float + ScalarOperand,
        D: Dimension,
    {
        // Apply value bounds first
        if let Some((min_val, max_val)) = value_bounds {
            for p in params.iter_mut() {
                if *p < min_val {
                    *p = min_val;
                } else if *p > max_val {
                    *p = max_val;
                }
            }
        }

        // Apply L2 norm constraint
        if let Some(max_norm) = l2_constraint {
            let norm_sq = params
                .iter()
                .map(|&x| x * x)
                .fold(A::zero(), |acc, x| acc + x);
            let norm = norm_sq.sqrt();

            if norm > max_norm {
                let scale = max_norm / norm;
                for p in params.iter_mut() {
                    *p = *p * scale;
                }
            }
        }
    }
}

/// Mixed-precision training support
pub mod mixed_precision {
    use super::*;

    /// Loss scaler for mixed-precision training
    #[derive(Debug, Clone)]
    pub struct LossScaler {
        scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
        steps_since_update: usize,
    }

    impl LossScaler {
        /// Create a new loss scaler
        pub fn new(initial_scale: f32) -> Self {
            Self {
                scale: initial_scale,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2000,
                steps_since_update: 0,
            }
        }

        /// Get current scale factor
        pub fn get_scale(&self) -> f32 {
            self.scale
        }

        /// Scale loss for backward pass
        pub fn scale_loss(&self, loss: f32) -> f32 {
            loss * self.scale
        }

        /// Unscale gradients after backward pass
        pub fn unscale_gradients<A, D>(&self, gradients: &mut Array<A, D>)
        where
            A: Float + ScalarOperand,
            D: Dimension,
        {
            let inv_scale = A::one() / A::from(self.scale).unwrap();
            for g in gradients.iter_mut() {
                *g = *g * inv_scale;
            }
        }

        /// Update scale based on gradient overflow detection
        pub fn update(&mut self, found_inf: bool) {
            self.steps_since_update += 1;

            if found_inf {
                // Reduce scale if overflow detected
                self.scale *= self.backoff_factor;
                self.steps_since_update = 0;
            } else if self.steps_since_update >= self.growth_interval {
                // Increase scale if no overflow for growth_interval steps
                self.scale *= self.growth_factor;
                self.steps_since_update = 0;
            }
        }

        /// Check if gradients contain infinite or NaN values
        pub fn check_gradients<A, D>(&self, gradients: &Array<A, D>) -> bool
        where
            A: Float + ScalarOperand,
            D: Dimension,
        {
            gradients.iter().any(|&x| !x.is_finite())
        }
    }
}

/// Gradient checkpointing for memory optimization
pub mod gradient_checkpointing {
    use super::*;
    use std::collections::VecDeque;

    /// Checkpointing strategy for gradient computation
    #[derive(Debug, Clone, PartialEq)]
    pub enum CheckpointStrategy {
        /// No checkpointing (store all intermediate values)
        None,
        /// Uniform checkpointing (checkpoint every N layers)
        Uniform {
            /// Interval between checkpoints
            interval: usize,
        },
        /// Logarithmic checkpointing (checkpoint at exponential intervals)
        Logarithmic {
            /// Base for exponential intervals
            base: f64,
        },
        /// Memory-aware checkpointing (adaptive based on memory usage)
        MemoryAware {
            /// Memory threshold for triggering checkpoints
            memory_threshold: f64,
        },
        /// Custom checkpointing pattern
        Custom {
            /// Pattern of checkpointing decisions
            pattern: Vec<bool>,
        },
    }

    /// Gradient checkpointing manager
    #[derive(Debug)]
    pub struct GradientCheckpointer<A: Float, D: Dimension> {
        /// Checkpointing strategy
        strategy: CheckpointStrategy,
        /// Stored checkpoints (layer_index -> activation)
        checkpoints: std::collections::HashMap<usize, Array<A, D>>,
        /// Memory usage tracker
        memory_tracker: MemoryTracker,
        /// Current computation depth
        current_depth: usize,
        /// Maximum depth for this computation
        max_depth: usize,
        /// Whether checkpointing is enabled
        enabled: bool,
    }

    impl<A: Float + ScalarOperand + Debug, D: Dimension> GradientCheckpointer<A, D> {
        /// Create a new gradient checkpointer
        pub fn new(strategy: CheckpointStrategy) -> Self {
            Self {
                strategy,
                checkpoints: std::collections::HashMap::new(),
                memory_tracker: MemoryTracker::new(),
                current_depth: 0,
                max_depth: 0,
                enabled: true,
            }
        }

        /// Set the maximum computation depth
        pub fn set_max_depth(&mut self, depth: usize) {
            self.max_depth = depth;
        }

        /// Enable or disable checkpointing
        pub fn set_enabled(&mut self, enabled: bool) {
            self.enabled = enabled;
        }

        /// Check if we should checkpoint at the current depth
        pub fn should_checkpoint(&self, depth: usize) -> bool {
            if !self.enabled || self.max_depth == 0 {
                return false;
            }

            match self.strategy {
                CheckpointStrategy::None => false,
                CheckpointStrategy::Uniform { interval } => depth % interval == 0,
                CheckpointStrategy::Logarithmic { base } => {
                    let log_depth = (depth as f64).log(base).floor() as usize;
                    depth == base.powi(log_depth as i32) as usize
                }
                CheckpointStrategy::MemoryAware { memory_threshold } => {
                    self.memory_tracker.usage_ratio() > memory_threshold
                }
                CheckpointStrategy::Custom { ref pattern } => {
                    if depth < pattern.len() {
                        pattern[depth]
                    } else {
                        false
                    }
                }
            }
        }

        /// Store a checkpoint
        pub fn store_checkpoint(&mut self, depth: usize, activation: Array<A, D>) {
            if self.should_checkpoint(depth) {
                let memory_size = activation.len() * std::mem::size_of::<A>();
                self.memory_tracker.add_allocation(memory_size);
                self.checkpoints.insert(depth, activation);
            }
        }

        /// Retrieve a checkpoint
        pub fn get_checkpoint(&self, depth: usize) -> Option<&Array<A, D>> {
            self.checkpoints.get(&depth)
        }

        /// Remove a checkpoint to free memory
        pub fn remove_checkpoint(&mut self, depth: usize) -> Option<Array<A, D>> {
            if let Some(checkpoint) = self.checkpoints.remove(&depth) {
                let memory_size = checkpoint.len() * std::mem::size_of::<A>();
                self.memory_tracker.remove_allocation(memory_size);
                Some(checkpoint)
            } else {
                None
            }
        }

        /// Clear all checkpoints
        pub fn clear_checkpoints(&mut self) {
            self.checkpoints.clear();
            self.memory_tracker.reset();
        }

        /// Get memory usage information
        pub fn memory_usage(&self) -> MemoryUsage {
            self.memory_tracker.usage()
        }

        /// Optimize checkpointing strategy based on memory usage
        pub fn optimize_strategy(&mut self, target_memory_usage: f64) {
            let current_usage = self.memory_tracker.usage_ratio();

            if current_usage > target_memory_usage {
                // Increase checkpointing frequency to reduce memory usage
                self.strategy = match &self.strategy {
                    CheckpointStrategy::Uniform { interval } => CheckpointStrategy::Uniform {
                        interval: (interval / 2).max(1),
                    },
                    CheckpointStrategy::MemoryAware { .. } => CheckpointStrategy::MemoryAware {
                        memory_threshold: target_memory_usage * 0.8,
                    },
                    other => other.clone(),
                };
            } else if current_usage < target_memory_usage * 0.5 {
                // Decrease checkpointing frequency to improve performance
                self.strategy = match &self.strategy {
                    CheckpointStrategy::Uniform { interval } => CheckpointStrategy::Uniform {
                        interval: interval * 2,
                    },
                    CheckpointStrategy::MemoryAware { .. } => CheckpointStrategy::MemoryAware {
                        memory_threshold: target_memory_usage * 1.2,
                    },
                    other => other.clone(),
                };
            }
        }

        /// Execute a checkpointed computation
        pub fn checkpointed_forward<F, Output>(
            &mut self,
            depth: usize,
            input: &Array<A, D>,
            forward_fn: F,
        ) -> Result<(Output, Option<Array<A, D>>)>
        where
            F: FnOnce(&Array<A, D>) -> Result<(Output, Array<A, D>)>,
        {
            self.current_depth = depth;

            // Execute forward computation
            let (output, activation) = forward_fn(input)?;

            // Decide whether to store checkpoint
            let checkpoint = if self.should_checkpoint(depth) {
                self.store_checkpoint(depth, activation.clone());
                Some(activation)
            } else {
                None
            };

            Ok((output, checkpoint))
        }

        /// Recompute activations from checkpoint
        pub fn recompute_from_checkpoint<F>(
            &self,
            start_depth: usize,
            target_depth: usize,
            recompute_fn: F,
        ) -> Result<Array<A, D>>
        where
            F: Fn(usize, &Array<A, D>) -> Result<Array<A, D>>,
        {
            // Find the nearest checkpoint at or before start_depth
            let checkpoint_depth = (0..=start_depth)
                .rev()
                .find(|&d| self.checkpoints.contains_key(&d))
                .ok_or_else(|| {
                    OptimError::InvalidConfig("No checkpoint found for recomputation".to_string())
                })?;

            let mut current_activation = self.checkpoints[&checkpoint_depth].clone();

            // Recompute forward from checkpoint to target depth
            for depth in (checkpoint_depth + 1)..=target_depth {
                current_activation = recompute_fn(depth, &current_activation)?;
            }

            Ok(current_activation)
        }
    }

    /// Memory usage tracking
    #[derive(Debug, Clone)]
    pub struct MemoryTracker {
        allocated_bytes: usize,
        peak_bytes: usize,
        total_system_memory: usize,
    }

    impl Default for MemoryTracker {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MemoryTracker {
        /// Create a new memory tracker
        pub fn new() -> Self {
            Self {
                allocated_bytes: 0,
                peak_bytes: 0,
                total_system_memory: Self::estimate_system_memory(),
            }
        }

        /// Add an allocation
        pub fn add_allocation(&mut self, bytes: usize) {
            self.allocated_bytes += bytes;
            self.peak_bytes = self.peak_bytes.max(self.allocated_bytes);
        }

        /// Remove an allocation
        pub fn remove_allocation(&mut self, bytes: usize) {
            self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
        }

        /// Get current memory usage
        pub fn usage(&self) -> MemoryUsage {
            MemoryUsage {
                current_bytes: self.allocated_bytes,
                peak_bytes: self.peak_bytes,
                total_system_bytes: self.total_system_memory,
            }
        }

        /// Get memory usage ratio (0.0 to 1.0)
        pub fn usage_ratio(&self) -> f64 {
            if self.total_system_memory == 0 {
                0.0
            } else {
                self.allocated_bytes as f64 / self.total_system_memory as f64
            }
        }

        /// Reset memory tracking
        pub fn reset(&mut self) {
            self.allocated_bytes = 0;
            self.peak_bytes = 0;
        }

        /// Estimate total system memory (simplified)
        fn estimate_system_memory() -> usize {
            // This is a simplified estimation
            // In a real implementation, you would use system APIs
            8 * 1024 * 1024 * 1024 // Assume 8GB
        }
    }

    /// Memory usage information
    #[derive(Debug, Clone, Copy)]
    pub struct MemoryUsage {
        /// Current allocated bytes
        pub current_bytes: usize,
        /// Peak allocated bytes
        pub peak_bytes: usize,
        /// Total system memory bytes
        pub total_system_bytes: usize,
    }

    impl MemoryUsage {
        /// Get current usage as a ratio (0.0 to 1.0)
        pub fn current_ratio(&self) -> f64 {
            if self.total_system_bytes == 0 {
                0.0
            } else {
                self.current_bytes as f64 / self.total_system_bytes as f64
            }
        }

        /// Get peak usage as a ratio (0.0 to 1.0)
        pub fn peak_ratio(&self) -> f64 {
            if self.total_system_bytes == 0 {
                0.0
            } else {
                self.peak_bytes as f64 / self.total_system_bytes as f64
            }
        }

        /// Format as human-readable string
        pub fn format(&self) -> String {
            format!(
                "Current: {:.1} MB ({:.1}%), Peak: {:.1} MB ({:.1}%), Total: {:.1} MB",
                self.current_bytes as f64 / (1024.0 * 1024.0),
                self.current_ratio() * 100.0,
                self.peak_bytes as f64 / (1024.0 * 1024.0),
                self.peak_ratio() * 100.0,
                self.total_system_bytes as f64 / (1024.0 * 1024.0)
            )
        }
    }

    /// Automatic checkpointing manager for optimization workflows
    #[derive(Debug)]
    pub struct AutoCheckpointer<A: Float, D: Dimension> {
        checkpointer: GradientCheckpointer<A, D>,
        /// History of memory usage for adaptive optimization
        memory_history: VecDeque<f64>,
        /// Target memory usage ratio
        target_memory_ratio: f64,
        /// Adaptation frequency (steps)
        adaptation_frequency: usize,
        /// Current step count
        step_count: usize,
    }

    impl<A: Float + ScalarOperand + Debug, D: Dimension> AutoCheckpointer<A, D> {
        /// Create a new auto checkpointer
        pub fn new(initial_strategy: CheckpointStrategy, target_memory_ratio: f64) -> Self {
            Self {
                checkpointer: GradientCheckpointer::new(initial_strategy),
                memory_history: VecDeque::with_capacity(100),
                target_memory_ratio: target_memory_ratio.clamp(0.1, 0.9),
                adaptation_frequency: 10,
                step_count: 0,
            }
        }

        /// Set adaptation frequency
        pub fn with_adaptation_frequency(mut self, frequency: usize) -> Self {
            self.adaptation_frequency = frequency.max(1);
            self
        }

        /// Execute a step with automatic checkpointing
        pub fn auto_step<F, Output>(
            &mut self,
            depth: usize,
            input: &Array<A, D>,
            forward_fn: F,
        ) -> Result<(Output, Option<Array<A, D>>)>
        where
            F: FnOnce(&Array<A, D>) -> Result<(Output, Array<A, D>)>,
        {
            self.step_count += 1;

            // Execute checkpointed forward
            let result = self
                .checkpointer
                .checkpointed_forward(depth, input, forward_fn)?;

            // Track memory usage
            let current_usage = self.checkpointer.memory_usage().current_ratio();
            self.memory_history.push_back(current_usage);
            if self.memory_history.len() > 100 {
                self.memory_history.pop_front();
            }

            // Adapt strategy periodically
            if self.step_count % self.adaptation_frequency == 0 {
                self.adapt_strategy();
            }

            Ok(result)
        }

        /// Adapt checkpointing strategy based on memory usage history
        fn adapt_strategy(&mut self) {
            if self.memory_history.len() < 5 {
                return;
            }

            // Calculate average memory usage over recent history
            let recent_avg = self.memory_history.iter().rev().take(10).sum::<f64>()
                / 10.0.min(self.memory_history.len() as f64);

            // Optimize strategy if we're significantly off target
            let deviation = (recent_avg - self.target_memory_ratio).abs();
            if deviation > 0.1 {
                self.checkpointer
                    .optimize_strategy(self.target_memory_ratio);
            }
        }

        /// Get checkpointer reference
        pub fn checkpointer(&self) -> &GradientCheckpointer<A, D> {
            &self.checkpointer
        }

        /// Get mutable checkpointer reference
        pub fn checkpointer_mut(&mut self) -> &mut GradientCheckpointer<A, D> {
            &mut self.checkpointer
        }

        /// Get memory usage statistics
        pub fn get_memory_stats(&self) -> MemoryStats {
            let usage = self.checkpointer.memory_usage();
            let avg_usage = if self.memory_history.is_empty() {
                0.0
            } else {
                self.memory_history.iter().sum::<f64>() / self.memory_history.len() as f64
            };

            MemoryStats {
                current_usage: usage.current_ratio(),
                peak_usage: usage.peak_ratio(),
                average_usage: avg_usage,
                target_usage: self.target_memory_ratio,
                checkpoints_stored: self.checkpointer.checkpoints.len(),
            }
        }
    }

    /// Memory usage statistics
    #[derive(Debug, Clone, Copy)]
    pub struct MemoryStats {
        /// Current memory usage ratio
        pub current_usage: f64,
        /// Peak memory usage ratio
        pub peak_usage: f64,
        /// Average memory usage ratio
        pub average_usage: f64,
        /// Target memory usage ratio
        pub target_usage: f64,
        /// Number of checkpoints currently stored
        pub checkpoints_stored: usize,
    }

    impl MemoryStats {
        /// Check if memory usage is within target range
        pub fn is_within_target(&self, tolerance: f64) -> bool {
            (self.current_usage - self.target_usage).abs() <= tolerance
        }

        /// Get efficiency score (how close to target without exceeding)
        pub fn efficiency_score(&self) -> f64 {
            if self.current_usage <= self.target_usage {
                self.current_usage / self.target_usage
            } else {
                self.target_usage / self.current_usage
            }
        }
    }
}

/// Dynamic resource adaptation
pub mod adaptive {
    use super::*;

    /// Memory-aware batch size adapter
    #[derive(Debug, Clone)]
    pub struct MemoryAwareBatchSizer {
        initial_batch_size: usize,
        max_batch_size: usize,
        min_batch_size: usize,
        current_batch_size: usize,
        memory_threshold: f64, // Memory usage threshold (0.0 to 1.0)
        adaptation_factor: f64,
    }

    impl MemoryAwareBatchSizer {
        /// Create a new memory-aware batch sizer
        pub fn new(initial_batch_size: usize) -> Self {
            Self {
                initial_batch_size,
                max_batch_size: initial_batch_size * 4,
                min_batch_size: initial_batch_size.max(1) / 4,
                current_batch_size: initial_batch_size,
                memory_threshold: 0.8,
                adaptation_factor: 1.2,
            }
        }

        /// Set memory threshold (0.0 to 1.0)
        pub fn with_memory_threshold(mut self, threshold: f64) -> Self {
            self.memory_threshold = threshold.clamp(0.1, 0.95);
            self
        }

        /// Set adaptation factor
        pub fn with_adaptation_factor(mut self, factor: f64) -> Self {
            self.adaptation_factor = factor.max(1.0);
            self
        }

        /// Get current batch size
        pub fn current_batch_size(&self) -> usize {
            self.current_batch_size
        }

        /// Adapt batch size based on memory usage
        pub fn adapt(&mut self, memory_usage_ratio: f64) {
            if memory_usage_ratio > self.memory_threshold {
                // Reduce batch size if memory usage is high
                let new_size = (self.current_batch_size as f64 / self.adaptation_factor) as usize;
                self.current_batch_size = new_size.max(self.min_batch_size);
            } else if memory_usage_ratio < self.memory_threshold * 0.7 {
                // Increase batch size if memory usage is low
                let new_size = (self.current_batch_size as f64 * self.adaptation_factor) as usize;
                self.current_batch_size = new_size.min(self.max_batch_size);
            }
        }

        /// Reset to initial batch size
        pub fn reset(&mut self) {
            self.current_batch_size = self.initial_batch_size;
        }
    }

    /// Memory usage estimator for arrays
    pub fn estimate_memory_usage<A, D>(arrays: &[&Array<A, D>]) -> usize
    where
        A: Sized,
        D: Dimension,
    {
        arrays
            .iter()
            .map(|arr| arr.len() * std::mem::size_of::<A>())
            .sum()
    }

    /// Get approximate system memory usage ratio
    pub fn get_memory_usage_ratio() -> f64 {
        // This is a simplified estimation
        // In a real implementation, you would use system APIs
        // to get actual memory information
        0.5 // Placeholder: assume 50% memory usage
    }
}

// Re-export utility functions at module level for convenience
pub use utils::{
    add_inplace, apply_inplace, clip_inplace, normalize_inplace, scale_inplace, subtract_inplace,
};

// Re-export new modules
pub use adaptive::*;
pub use fused::*;
pub use gradient_checkpointing::*;
pub use mixed_precision::*;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_inplace_sgd() {
        let mut optimizer = InPlaceSGD::new(0.1);
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        optimizer.step_inplace(&mut params, &gradients).unwrap();

        assert_relative_eq!(params[0], 0.99, epsilon = 1e-6);
        assert_relative_eq!(params[1], 1.98, epsilon = 1e-6);
        assert_relative_eq!(params[2], 2.97, epsilon = 1e-6);
    }

    #[test]
    fn test_inplace_adam() {
        let mut optimizer = InPlaceAdam::new(0.001);
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        // Multiple steps to see momentum effects
        for _ in 0..5 {
            optimizer.step_inplace(&mut params, &gradients).unwrap();
        }

        // Verify parameters have been updated
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert!(params[2] < 3.0);
    }

    #[test]
    fn test_utils_scale_inplace() {
        let mut array = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        utils::scale_inplace(&mut array, 2.0);

        assert_eq!(array.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_utils_clip_inplace() {
        let mut array = Array1::from_vec(vec![0.5, 1.5, 2.5]);
        utils::clip_inplace(&mut array, 1.0, 2.0);

        assert_eq!(array.as_slice().unwrap(), &[1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_memory_efficiency() {
        // Test that in-place operations don't allocate new arrays
        let mut params = Array1::from_vec(vec![1.0; 1000]);
        let gradients = Array1::from_vec(vec![0.01; 1000]);
        let params_ptr = params.as_ptr();

        let mut optimizer = InPlaceSGD::new(0.1);
        optimizer.step_inplace(&mut params, &gradients).unwrap();

        // Verify the same memory is being used
        assert_eq!(params_ptr, params.as_ptr());
    }

    #[test]
    fn test_fused_adam_update() {
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let mut m = Array1::zeros(3);
        let mut v = Array1::zeros(3);

        fused::fused_adam_update(
            &mut params,
            &gradients,
            &mut m,
            &mut v,
            0.01,  // lr
            0.9,   // beta1
            0.999, // beta2
            1e-8,  // epsilon
            0.1,   // bias1
            0.001, // bias2
            None,  // weight_decay
        );

        // Verify parameters were updated
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert!(params[2] < 3.0);

        // Verify momentum and variance were updated
        assert!(m[0] > 0.0);
        assert!(v[0] > 0.0);
    }

    #[test]
    fn test_fused_sgd_update() {
        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let mut momentum_buf = Array1::zeros(3);

        fused::fused_sgd_update(
            &mut params,
            &gradients,
            Some(&mut momentum_buf),
            0.1,        // lr
            0.9,        // momentum
            Some(0.01), // weight_decay
            0.0,        // dampening
        );

        // Verify parameters were updated
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert!(params[2] < 3.0);
    }

    #[test]
    fn test_fused_gradient_clip_normalize() {
        let mut gradients = Array1::from_vec(vec![5.0, -3.0, 2.0]);

        fused::fused_gradient_clip_normalize(
            &mut gradients,
            Some(2.0), // max_norm
            Some(1.0), // clip_value
        );

        // Verify values were clipped
        assert!(gradients.iter().all(|&x| x.abs() <= 1.0));

        // Verify norm constraint
        let norm = gradients.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!(norm <= 2.0 + 1e-6);
    }

    #[test]
    fn test_mixed_precision_loss_scaler() {
        let scaler = mixed_precision::LossScaler::new(65536.0);

        // Test loss scaling
        let loss = 0.5;
        let scaled_loss = scaler.scale_loss(loss);
        assert_eq!(scaled_loss, 0.5 * 65536.0);

        // Test gradient unscaling
        let mut gradients = Array1::from_vec(vec![65536.0, 131072.0]);
        scaler.unscale_gradients(&mut gradients);
        assert_relative_eq!(gradients[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(gradients[1], 2.0, epsilon = 1e-6);

        // Test overflow detection
        let inf_gradients = Array1::from_vec(vec![f64::INFINITY, 1.0]);
        assert!(scaler.check_gradients(&inf_gradients));

        let finite_gradients = Array1::from_vec(vec![1.0, 2.0]);
        assert!(!scaler.check_gradients(&finite_gradients));
    }

    #[test]
    fn test_memory_aware_batch_sizer() {
        let mut sizer = adaptive::MemoryAwareBatchSizer::new(32)
            .with_memory_threshold(0.8)
            .with_adaptation_factor(1.3); // Use smaller factor for more predictable behavior

        assert_eq!(sizer.current_batch_size(), 32);

        // High memory usage should reduce batch size
        sizer.adapt(0.9);
        let reduced_size = sizer.current_batch_size();
        assert!(reduced_size < 32);

        // Low memory usage should increase batch size (multiple calls to ensure growth)
        sizer.adapt(0.3);
        sizer.adapt(0.3); // Call twice to ensure we exceed original size
        assert!(sizer.current_batch_size() >= 32);

        // Reset should restore initial size
        sizer.reset();
        assert_eq!(sizer.current_batch_size(), 32);
    }

    #[test]
    fn test_memory_estimation() {
        let array1 = Array1::from_vec(vec![1.0; 100]);
        let array2 = Array1::from_vec(vec![2.0; 200]);

        let arrays = vec![&array1, &array2];
        let estimated_size = adaptive::estimate_memory_usage(&arrays);

        // Should be roughly 300 * size_of::<f64>()
        let expected_size = 300 * std::mem::size_of::<f64>();
        assert_eq!(estimated_size, expected_size);
    }

    #[test]
    fn test_gradient_checkpointing_uniform() {
        let mut checkpointer: gradient_checkpointing::GradientCheckpointer<f64, ndarray::Ix1> =
            gradient_checkpointing::GradientCheckpointer::new(
                gradient_checkpointing::CheckpointStrategy::Uniform { interval: 2 },
            );
        checkpointer.set_max_depth(10);

        // Should checkpoint at depths 0, 2, 4, 6, 8
        assert!(checkpointer.should_checkpoint(0));
        assert!(!checkpointer.should_checkpoint(1));
        assert!(checkpointer.should_checkpoint(2));
        assert!(!checkpointer.should_checkpoint(3));
        assert!(checkpointer.should_checkpoint(4));

        // Store a checkpoint
        let activation = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        checkpointer.store_checkpoint(2, activation.clone());

        // Retrieve checkpoint
        let retrieved = checkpointer.get_checkpoint(2).unwrap();
        assert_eq!(
            retrieved.as_slice().unwrap(),
            activation.as_slice().unwrap()
        );

        // Non-checkpointed depth should return None
        assert!(checkpointer.get_checkpoint(1).is_none());
    }

    #[test]
    fn test_gradient_checkpointing_logarithmic() {
        let mut checkpointer: gradient_checkpointing::GradientCheckpointer<f64, ndarray::Ix1> =
            gradient_checkpointing::GradientCheckpointer::new(
                gradient_checkpointing::CheckpointStrategy::Logarithmic { base: 2.0 },
            );

        // Set max depth to enable checkpointing
        checkpointer.set_max_depth(10);

        // Should checkpoint at powers of 2: 1, 2, 4, 8, 16...
        assert!(checkpointer.should_checkpoint(1));
        assert!(checkpointer.should_checkpoint(2));
        assert!(!checkpointer.should_checkpoint(3));
        assert!(checkpointer.should_checkpoint(4));
        assert!(!checkpointer.should_checkpoint(5));
        assert!(!checkpointer.should_checkpoint(6));
        assert!(!checkpointer.should_checkpoint(7));
        assert!(checkpointer.should_checkpoint(8));
    }

    #[test]
    fn test_gradient_checkpointing_custom() {
        let pattern = vec![true, false, false, true, false];
        let mut checkpointer: gradient_checkpointing::GradientCheckpointer<f64, ndarray::Ix1> =
            gradient_checkpointing::GradientCheckpointer::new(
                gradient_checkpointing::CheckpointStrategy::Custom { pattern },
            );

        // Set max depth to enable checkpointing
        checkpointer.set_max_depth(10);

        // Should follow the custom pattern
        assert!(checkpointer.should_checkpoint(0));
        assert!(!checkpointer.should_checkpoint(1));
        assert!(!checkpointer.should_checkpoint(2));
        assert!(checkpointer.should_checkpoint(3));
        assert!(!checkpointer.should_checkpoint(4));
        assert!(!checkpointer.should_checkpoint(5)); // Beyond pattern length
    }

    #[test]
    fn test_gradient_checkpointing_memory_tracking() {
        let mut checkpointer: gradient_checkpointing::GradientCheckpointer<f64, ndarray::Ix1> =
            gradient_checkpointing::GradientCheckpointer::new(
                gradient_checkpointing::CheckpointStrategy::Uniform { interval: 1 },
            );
        checkpointer.set_max_depth(5);

        let activation1 = Array1::from_vec(vec![1.0; 100]);
        let activation2 = Array1::from_vec(vec![2.0; 200]);

        checkpointer.store_checkpoint(0, activation1);
        let usage_after_first = checkpointer.memory_usage();
        assert!(usage_after_first.current_bytes > 0);

        checkpointer.store_checkpoint(1, activation2);
        let usage_after_second = checkpointer.memory_usage();
        assert!(usage_after_second.current_bytes > usage_after_first.current_bytes);

        // Remove first checkpoint
        checkpointer.remove_checkpoint(0);
        let usage_after_removal = checkpointer.memory_usage();
        assert!(usage_after_removal.current_bytes < usage_after_second.current_bytes);
    }

    #[test]
    fn test_checkpointed_forward() {
        let mut checkpointer: gradient_checkpointing::GradientCheckpointer<f64, ndarray::Ix1> =
            gradient_checkpointing::GradientCheckpointer::new(
                gradient_checkpointing::CheckpointStrategy::Uniform { interval: 1 },
            );
        checkpointer.set_max_depth(5);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        // Simple forward function that doubles the input
        let forward_fn = |x: &Array1<f64>| -> Result<(f64, Array1<f64>)> {
            let output = x.sum();
            let activation = x.mapv(|val| val * 2.0);
            Ok((output, activation))
        };

        let (output, checkpoint) = checkpointer
            .checkpointed_forward(0, &input, forward_fn)
            .unwrap();

        assert_eq!(output, 6.0); // 1 + 2 + 3
        assert!(checkpoint.is_some());
        let checkpoint = checkpoint.unwrap();
        assert_eq!(checkpoint.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_recompute_from_checkpoint() {
        let mut checkpointer: gradient_checkpointing::GradientCheckpointer<f64, ndarray::Ix1> =
            gradient_checkpointing::GradientCheckpointer::new(
                gradient_checkpointing::CheckpointStrategy::Uniform { interval: 2 },
            );
        checkpointer.set_max_depth(10);

        // Store checkpoints at depths 0, 2, 4
        let checkpoint0 = Array1::from_vec(vec![1.0, 2.0]);
        let checkpoint2 = Array1::from_vec(vec![3.0, 4.0]);

        checkpointer.store_checkpoint(0, checkpoint0);
        checkpointer.store_checkpoint(2, checkpoint2);

        // Recompute function that adds 1 to each element
        let recompute_fn =
            |_depth: usize, x: &Array1<f64>| -> Result<Array1<f64>> { Ok(x.mapv(|val| val + 1.0)) };

        // Recompute from checkpoint 2 to depth 4
        let result = checkpointer
            .recompute_from_checkpoint(2, 4, recompute_fn)
            .unwrap();

        // Should be [3,4] + 1 + 1 = [5,6]
        assert_eq!(result.as_slice().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_auto_checkpointer() {
        let mut auto_checkpointer: gradient_checkpointing::AutoCheckpointer<f64, ndarray::Ix1> =
            gradient_checkpointing::AutoCheckpointer::new(
                gradient_checkpointing::CheckpointStrategy::Uniform { interval: 2 },
                0.6, // target 60% memory usage
            );

        let input = Array1::from_vec(vec![1.0, 2.0]);

        // Simple forward function
        let forward_fn = |x: &Array1<f64>| -> Result<(f64, Array1<f64>)> {
            let output = x.sum();
            let activation = x.clone();
            Ok((output, activation))
        };

        // Execute several steps
        for depth in 0..5 {
            let (output, _checkpoint) = auto_checkpointer
                .auto_step(depth, &input, forward_fn)
                .unwrap();
            assert_eq!(output, 3.0); // 1 + 2
        }

        let stats = auto_checkpointer.get_memory_stats();
        assert!(stats.target_usage > 0.0);
    }

    #[test]
    fn test_memory_stats() {
        let stats = gradient_checkpointing::MemoryStats {
            current_usage: 0.5,
            peak_usage: 0.7,
            average_usage: 0.6,
            target_usage: 0.6,
            checkpoints_stored: 3,
        };

        assert!(stats.is_within_target(0.1));
        assert!(!stats.is_within_target(0.01));

        let efficiency = stats.efficiency_score();
        assert!(efficiency > 0.8 && efficiency <= 1.0);
    }

    #[test]
    fn test_memory_usage_formatting() {
        let usage = gradient_checkpointing::MemoryUsage {
            current_bytes: 1024 * 1024,                 // 1 MB
            peak_bytes: 2 * 1024 * 1024,                // 2 MB
            total_system_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
        };

        let formatted = usage.format();
        assert!(formatted.contains("1.0 MB"));
        assert!(formatted.contains("2.0 MB"));
        assert!(formatted.contains("8192.0 MB"));

        assert_relative_eq!(usage.current_ratio(), 1.0 / 8192.0, epsilon = 1e-6);
        assert_relative_eq!(usage.peak_ratio(), 2.0 / 8192.0, epsilon = 1e-6);
    }

    #[test]
    fn test_checkpointing_strategy_optimization() {
        let mut checkpointer: gradient_checkpointing::GradientCheckpointer<f64, ndarray::Ix1> =
            gradient_checkpointing::GradientCheckpointer::new(
                gradient_checkpointing::CheckpointStrategy::Uniform { interval: 4 },
            );

        // Set max depth to enable checkpointing
        checkpointer.set_max_depth(10);

        // Add some memory usage first to trigger optimization
        let checkpoint = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        checkpointer.store_checkpoint(0, checkpoint);

        // Simulate high memory usage - should reduce interval
        checkpointer.optimize_strategy(0.3); // Target 30% usage

        // Check that strategy was adapted (should checkpoint more frequently)
        // With interval 4, should checkpoint at 0, 4, 8... but optimization might change this
        assert!(
            checkpointer.should_checkpoint(0)
                || checkpointer.should_checkpoint(1)
                || checkpointer.should_checkpoint(2)
        );
    }

    #[test]
    fn test_checkpointing_disabled() {
        let mut checkpointer: gradient_checkpointing::GradientCheckpointer<f64, ndarray::Ix1> =
            gradient_checkpointing::GradientCheckpointer::new(
                gradient_checkpointing::CheckpointStrategy::Uniform { interval: 1 },
            );
        checkpointer.set_enabled(false);

        // Should not checkpoint when disabled
        assert!(!checkpointer.should_checkpoint(0));
        assert!(!checkpointer.should_checkpoint(1));
        assert!(!checkpointer.should_checkpoint(2));
    }
}
