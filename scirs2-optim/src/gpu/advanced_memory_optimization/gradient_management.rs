//! Gradient management for memory-efficient training
//!
//! This module provides gradient accumulation, zero redundancy optimization,
//! mixed precision training, and gradient synchronization capabilities.

use std::collections::HashMap;
use std::time::Instant;
use ndarray::{Array1, Array2};
use num_traits::Float;

use crate::error::{OptimError, Result};

/// Gradient accumulation for memory efficiency
#[derive(Debug)]
pub struct GradientAccumulator<T: Float> {
    /// Accumulated gradients by parameter name
    accumulated_gradients: HashMap<String, Array1<T>>,

    /// Current accumulation step
    current_step: usize,

    /// Target accumulation steps before update
    target_steps: usize,

    /// Gradient scaling for numerical stability
    gradient_scale: T,

    /// Enable gradient compression
    enable_compression: bool,

    /// Compression ratio achieved
    compression_ratio: f32,

    /// Memory saved through accumulation (bytes)
    memory_saved: usize,

    /// Total gradient norm for clipping
    total_grad_norm: T,

    /// Gradient clipping configuration
    gradient_clipping: GradientClipping<T>,
}

impl<T: Float + Default + Clone> GradientAccumulator<T> {
    /// Create a new gradient accumulator
    pub fn new(target_steps: usize) -> Self {
        Self {
            accumulated_gradients: HashMap::new(),
            current_step: 0,
            target_steps,
            gradient_scale: T::one(),
            enable_compression: false,
            compression_ratio: 1.0,
            memory_saved: 0,
            total_grad_norm: T::zero(),
            gradient_clipping: GradientClipping::default(),
        }
    }

    /// Configure gradient accumulation parameters
    pub fn configure(
        &mut self,
        target_steps: usize,
        gradient_scale: T,
        enable_compression: bool,
        gradient_clipping: GradientClipping<T>,
    ) {
        self.target_steps = target_steps;
        self.gradient_scale = gradient_scale;
        self.enable_compression = enable_compression;
        self.gradient_clipping = gradient_clipping;
    }

    /// Accumulate gradients for a parameter
    pub fn accumulate_gradients(&mut self, param_name: &str, gradients: Array1<T>) -> Result<()> {
        // Apply gradient scaling
        let scaled_gradients = &gradients * self.gradient_scale;

        // Accumulate or initialize
        match self.accumulated_gradients.get_mut(param_name) {
            Some(accumulated) => {
                if accumulated.len() != scaled_gradients.len() {
                    return Err(OptimError::DimensionMismatch(
                        format!("Gradient dimension mismatch for parameter {}", param_name)
                    ));
                }
                *accumulated = &*accumulated + &scaled_gradients;
            }
            None => {
                self.accumulated_gradients.insert(param_name.to_string(), scaled_gradients);
            }
        }

        Ok(())
    }

    /// Check if accumulation is ready for optimizer step
    pub fn is_ready_for_step(&self) -> bool {
        self.current_step >= self.target_steps
    }

    /// Get accumulated gradients and reset accumulator
    pub fn get_accumulated_gradients(&mut self) -> HashMap<String, Array1<T>> {
        // Apply gradient clipping if enabled
        if self.gradient_clipping.enabled {
            self.apply_gradient_clipping();
        }

        // Average gradients over accumulation steps
        let mut result = HashMap::new();
        for (name, accumulated) in self.accumulated_gradients.drain() {
            let scale = T::one() / num_traits::cast::cast(self.target_steps).unwrap_or_else(|| T::zero());
            result.insert(name, accumulated * scale);
        }

        // Reset state
        self.current_step = 0;
        self.total_grad_norm = T::zero();

        result
    }

    /// Increment accumulation step
    pub fn increment_step(&mut self) {
        self.current_step += 1;
    }

    /// Get current accumulation progress
    pub fn get_progress(&self) -> f32 {
        self.current_step as f32 / self.target_steps as f32
    }

    /// Get memory savings estimation
    pub fn get_memory_savings(&self) -> usize {
        // Estimate based on avoiding full gradient storage
        let gradient_memory = self.accumulated_gradients
            .values()
            .map(|g| g.len() * std::mem::size_of::<T>())
            .sum::<usize>();

        // Memory saved by not storing full gradients for each microbatch
        gradient_memory * (self.target_steps - 1)
    }

    /// Apply gradient clipping to accumulated gradients
    fn apply_gradient_clipping(&mut self) {
        if !self.gradient_clipping.enabled {
            return;
        }

        // Compute total gradient norm
        self.total_grad_norm = T::zero();
        for gradients in self.accumulated_gradients.values() {
            let grad_norm_sq = gradients.iter().map(|&g| g * g).fold(T::zero(), |acc, x| acc + x);
            self.total_grad_norm = self.total_grad_norm + grad_norm_sq;
        }
        self.total_grad_norm = self.total_grad_norm.sqrt();

        // Apply clipping if necessary
        let max_norm = num_traits::cast::cast(self.gradient_clipping.max_norm).unwrap_or_else(|| T::zero());
        if self.total_grad_norm > max_norm {
            let clip_factor = max_norm / self.total_grad_norm;
            for gradients in self.accumulated_gradients.values_mut() {
                *gradients = &*gradients * clip_factor;
            }
        }
    }
}

/// Gradient clipping configuration
#[derive(Debug, Clone)]
pub struct GradientClipping<T: Float> {
    /// Enable gradient clipping
    pub enabled: bool,

    /// Maximum gradient norm
    pub max_norm: f32,

    /// Clipping strategy
    pub strategy: ClippingStrategy,

    /// Adaptive clipping parameters
    pub adaptive_threshold: Option<T>,
}

impl<T: Float> Default for GradientClipping<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            max_norm: 1.0,
            strategy: ClippingStrategy::GlobalNorm,
            adaptive_threshold: None,
        }
    }
}

/// Gradient clipping strategies
#[derive(Debug, Clone, Copy)]
pub enum ClippingStrategy {
    /// Clip by global gradient norm
    GlobalNorm,

    /// Clip by per-parameter norm
    PerParameter,

    /// Adaptive clipping based on gradient statistics
    Adaptive,
}

/// Zero Redundancy Optimizer (ZeRO) state management
#[derive(Debug)]
pub struct ZeroRedundancyState<T: Float> {
    /// Parameter partitions across workers
    parameter_partitions: Vec<ParameterPartition<T>>,

    /// Gradient synchronization manager
    gradient_sync: GradientSynchronizer<T>,

    /// Parameter updates cache
    parameter_updates: HashMap<String, Array1<T>>,

    /// Communication backend identifier
    communication_backend: String,

    /// Current partition ownership
    owned_partitions: Vec<usize>,

    /// Memory savings from partitioning
    memory_savings: usize,
}

impl<T: Float + Default + Clone> ZeroRedundancyState<T> {
    /// Create new ZeRO state
    pub fn new() -> Self {
        Self {
            parameter_partitions: Vec::new(),
            gradient_sync: GradientSynchronizer::new(),
            parameter_updates: HashMap::new(),
            communication_backend: "default".to_string(),
            owned_partitions: Vec::new(),
            memory_savings: 0,
        }
    }

    /// Initialize parameter partitions
    pub fn initialize_partitions(
        &mut self,
        parameters: &HashMap<String, Array1<T>>,
        num_workers: usize,
        worker_rank: usize,
    ) -> Result<()> {
        if num_workers == 0 {
            return Err(OptimError::InvalidParameter("num_workers must be positive".to_string()));
        }

        self.parameter_partitions.clear();
        self.owned_partitions.clear();

        let mut partition_id = 0;
        for (name, param) in parameters {
            let param_size = param.len();
            let partition_size = (param_size + num_workers - 1) / num_workers; // Ceiling division

            for worker in 0..num_workers {
                let start_idx = worker * partition_size;
                let end_idx = (start_idx + partition_size).min(param_size);

                if start_idx < param_size {
                    let partition = ParameterPartition {
                        id: partition_id,
                        parameter_name: name.clone(),
                        worker_rank: worker,
                        start_index: start_idx,
                        end_index: end_idx,
                        data: param.slice(ndarray::s![start_idx..end_idx]).to_owned(),
                    };

                    if worker == worker_rank {
                        self.owned_partitions.push(partition_id);
                    }

                    self.parameter_partitions.push(partition);
                    partition_id += 1;
                }
            }
        }

        // Calculate memory savings
        let total_param_memory = parameters.values()
            .map(|p| p.len() * std::mem::size_of::<T>())
            .sum::<usize>();
        self.memory_savings = total_param_memory * (num_workers - 1) / num_workers;

        Ok(())
    }

    /// Get owned parameter partitions
    pub fn get_owned_partitions(&self) -> Vec<&ParameterPartition<T>> {
        self.owned_partitions
            .iter()
            .filter_map(|&id| self.parameter_partitions.get(id))
            .collect()
    }

    /// Synchronize gradients across workers
    pub fn synchronize_gradients(&mut self, gradients: &HashMap<String, Array1<T>>) -> Result<()> {
        for (param_name, gradient) in gradients {
            self.gradient_sync.add_gradient_for_sync(param_name.clone(), gradient.clone())?;
        }
        Ok(())
    }

    /// Get memory savings from ZeRO optimization
    pub fn get_memory_savings(&self) -> usize {
        self.memory_savings
    }
}

/// Parameter partition for ZeRO
#[derive(Debug, Clone)]
pub struct ParameterPartition<T: Float> {
    /// Unique partition ID
    pub id: usize,

    /// Name of the parameter this partition belongs to
    pub parameter_name: String,

    /// Worker rank that owns this partition
    pub worker_rank: usize,

    /// Start index in the original parameter
    pub start_index: usize,

    /// End index in the original parameter
    pub end_index: usize,

    /// Partition data
    pub data: Array1<T>,
}

/// Gradient synchronization manager
#[derive(Debug)]
pub struct GradientSynchronizer<T: Float> {
    /// Pending gradient reductions
    pending_reductions: HashMap<String, GradientReduction<T>>,

    /// Reduction strategy
    reduction_strategy: ReductionStrategy,

    /// Enable compression for communication
    compression_enabled: bool,

    /// Synchronization overhead tracking
    sync_overhead: std::time::Duration,
}

impl<T: Float + Default + Clone> GradientSynchronizer<T> {
    /// Create new gradient synchronizer
    pub fn new() -> Self {
        Self {
            pending_reductions: HashMap::new(),
            reduction_strategy: ReductionStrategy::AllReduce,
            compression_enabled: false,
            sync_overhead: std::time::Duration::ZERO,
        }
    }

    /// Add gradient for synchronization
    pub fn add_gradient_for_sync(&mut self, param_name: String, gradient: Array1<T>) -> Result<()> {
        let reduction = GradientReduction {
            gradient_buffer: gradient,
            operation: ReductionOperation::Sum,
            ranks: vec![0], // Would be populated based on actual distributed setup
            status: ReductionStatus::Pending,
            timestamp: Instant::now(),
        };

        self.pending_reductions.insert(param_name, reduction);
        Ok(())
    }

    /// Execute pending reductions
    pub fn execute_reductions(&mut self) -> Result<HashMap<String, Array1<T>>> {
        let start_time = Instant::now();
        let mut results = HashMap::new();

        for (param_name, reduction) in self.pending_reductions.drain() {
            // In a real implementation, this would perform actual distributed reduction
            results.insert(param_name, reduction.gradient_buffer);
        }

        self.sync_overhead = start_time.elapsed();
        Ok(results)
    }

    /// Get synchronization overhead
    pub fn get_sync_overhead(&self) -> std::time::Duration {
        self.sync_overhead
    }
}

/// Gradient reduction operation
#[derive(Debug, Clone)]
pub struct GradientReduction<T: Float> {
    /// Gradient buffer to be reduced
    pub gradient_buffer: Array1<T>,

    /// Type of reduction operation
    pub operation: ReductionOperation,

    /// Participating worker ranks
    pub ranks: Vec<usize>,

    /// Current status of the reduction
    pub status: ReductionStatus,

    /// Timestamp when reduction was initiated
    pub timestamp: Instant,
}

/// Reduction operation types
#[derive(Debug, Clone, Copy)]
pub enum ReductionOperation {
    /// Sum across all workers
    Sum,

    /// Average across all workers
    Average,

    /// Maximum across all workers
    Max,

    /// Minimum across all workers
    Min,
}

/// Reduction status
#[derive(Debug, Clone, Copy)]
pub enum ReductionStatus {
    /// Reduction is pending
    Pending,

    /// Reduction is in progress
    InProgress,

    /// Reduction completed successfully
    Completed,

    /// Reduction failed
    Failed,
}

/// Reduction strategy for distributed training
#[derive(Debug, Clone, Copy)]
pub enum ReductionStrategy {
    /// All-reduce operation
    AllReduce,

    /// Reduce-scatter followed by all-gather
    ReduceScatter,

    /// Hierarchical reduction
    Hierarchical,

    /// Ring-based reduction
    Ring,
}

/// Mixed precision training manager
#[derive(Debug)]
pub struct MixedPrecisionManager<T: Float> {
    /// Enable mixed precision training
    enabled: bool,

    /// FP16 parameters for forward pass
    fp16_parameters: HashMap<String, Array1<f32>>,

    /// FP32 master weights for gradient updates
    fp32_master_weights: HashMap<String, Array1<T>>,

    /// Loss scaling factor
    loss_scale: f32,

    /// Dynamic loss scaling parameters
    dynamic_scaling: DynamicLossScaling,

    /// Gradient overflow detection
    gradient_overflow: bool,

    /// Memory savings from mixed precision
    memory_savings: usize,
}

impl<T: Float + Default + Clone> MixedPrecisionManager<T> {
    /// Create new mixed precision manager
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            fp16_parameters: HashMap::new(),
            fp32_master_weights: HashMap::new(),
            loss_scale: 65536.0, // Common initial loss scale
            dynamic_scaling: DynamicLossScaling::default(),
            gradient_overflow: false,
            memory_savings: 0,
        }
    }

    /// Initialize mixed precision for parameters
    pub fn initialize_parameters(&mut self, parameters: &HashMap<String, Array1<T>>) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        for (name, param) in parameters {
            // Store FP32 master weights
            self.fp32_master_weights.insert(name.clone(), param.clone());

            // Create FP16 copy for forward pass
            let fp16_param = param.iter()
                .map(|&x| x.to_f32().unwrap_or(0.0))
                .collect::<Vec<f32>>();
            self.fp16_parameters.insert(name.clone(), Array1::from_vec(fp16_param));
        }

        // Calculate memory savings (approximate)
        let total_params = parameters.values().map(|p| p.len()).sum::<usize>();
        self.memory_savings = total_params * (std::mem::size_of::<T>() - std::mem::size_of::<f32>());

        Ok(())
    }

    /// Get FP16 parameters for forward pass
    pub fn get_fp16_parameters(&self) -> &HashMap<String, Array1<f32>> {
        &self.fp16_parameters
    }

    /// Update master weights with scaled gradients
    pub fn update_master_weights(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<bool> {
        if !self.enabled {
            return Ok(true);
        }

        // Check for gradient overflow
        self.gradient_overflow = self.detect_gradient_overflow(gradients);

        if self.gradient_overflow {
            // Skip update and adjust loss scale
            self.dynamic_scaling.on_overflow();
            self.loss_scale = self.dynamic_scaling.get_loss_scale();
            return Ok(false);
        }

        // Scale gradients down by loss scale
        let scale_factor = num_traits::cast::cast(1.0 / self.loss_scale).unwrap_or_else(|| T::zero());

        for (name, gradient) in gradients {
            if let Some(master_weight) = self.fp32_master_weights.get_mut(name) {
                let scaled_gradient = gradient * scale_factor;
                *master_weight = &*master_weight - &scaled_gradient; // Simple SGD step
            }
        }

        // Update loss scale
        self.dynamic_scaling.on_success();
        self.loss_scale = self.dynamic_scaling.get_loss_scale();

        // Update FP16 parameters from master weights
        self.sync_fp16_from_master()?;

        Ok(true)
    }

    /// Get current loss scale
    pub fn get_loss_scale(&self) -> f32 {
        self.loss_scale
    }

    /// Get memory savings from mixed precision
    pub fn get_memory_savings(&self) -> usize {
        self.memory_savings
    }

    fn detect_gradient_overflow(&self, gradients: &HashMap<String, Array1<T>>) -> bool {
        for gradient in gradients.values() {
            for &g in gradient.iter() {
                if !g.is_finite() {
                    return true;
                }
            }
        }
        false
    }

    fn sync_fp16_from_master(&mut self) -> Result<()> {
        for (name, master_weight) in &self.fp32_master_weights {
            if let Some(fp16_param) = self.fp16_parameters.get_mut(name) {
                for (i, &weight) in master_weight.iter().enumerate() {
                    fp16_param[i] = weight.to_f32().unwrap_or(0.0);
                }
            }
        }
        Ok(())
    }
}

/// Dynamic loss scaling for mixed precision training
#[derive(Debug, Clone)]
pub struct DynamicLossScaling {
    /// Current loss scale
    loss_scale: f32,

    /// Scale factor for increasing loss scale
    growth_factor: f32,

    /// Scale factor for decreasing loss scale
    shrink_factor: f32,

    /// Number of successful steps before increasing scale
    growth_interval: usize,

    /// Current successful step count
    successful_steps: usize,
}

impl Default for DynamicLossScaling {
    fn default() -> Self {
        Self {
            loss_scale: 65536.0,
            growth_factor: 2.0,
            shrink_factor: 0.5,
            growth_interval: 2000,
            successful_steps: 0,
        }
    }
}

impl DynamicLossScaling {
    /// Handle successful gradient update
    pub fn on_success(&mut self) {
        self.successful_steps += 1;
        if self.successful_steps >= self.growth_interval {
            self.loss_scale *= self.growth_factor;
            self.successful_steps = 0;
        }
    }

    /// Handle gradient overflow
    pub fn on_overflow(&mut self) {
        self.loss_scale *= self.shrink_factor;
        self.successful_steps = 0;
    }

    /// Get current loss scale
    pub fn get_loss_scale(&self) -> f32 {
        self.loss_scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_accumulator() {
        let mut accumulator = GradientAccumulator::<f32>::new(4);

        let grad1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        accumulator.accumulate_gradients("param1", grad1).unwrap();
        accumulator.increment_step();

        assert!(!accumulator.is_ready_for_step());
        assert_eq!(accumulator.get_progress(), 0.25);

        // Accumulate more steps
        for _ in 0..3 {
            let grad = Array1::from_vec(vec![1.0, 1.0, 1.0]);
            accumulator.accumulate_gradients("param1", grad).unwrap();
            accumulator.increment_step();
        }

        assert!(accumulator.is_ready_for_step());
    }

    #[test]
    fn test_zero_redundancy_state() {
        let mut zero_state = ZeroRedundancyState::<f32>::new();

        let mut parameters = HashMap::new();
        parameters.insert("param1".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]));

        zero_state.initialize_partitions(&parameters, 2, 0).unwrap();

        let owned = zero_state.get_owned_partitions();
        assert!(!owned.is_empty());
        assert!(zero_state.get_memory_savings() > 0);
    }

    #[test]
    fn test_mixed_precision_manager() {
        let mut mp_manager = MixedPrecisionManager::<f32>::new(true);

        let mut parameters = HashMap::new();
        parameters.insert("param1".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));

        mp_manager.initialize_parameters(&parameters).unwrap();

        assert!(mp_manager.get_fp16_parameters().contains_key("param1"));
        assert!(mp_manager.get_memory_savings() > 0);
        assert_eq!(mp_manager.get_loss_scale(), 65536.0);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut clipping = GradientClipping::<f32> {
            enabled: true,
            max_norm: 1.0,
            strategy: ClippingStrategy::GlobalNorm,
            adaptive_threshold: None,
        };

        let mut accumulator = GradientAccumulator::<f32>::new(1);
        accumulator.gradient_clipping = clipping;

        // Add a large gradient that should be clipped
        let large_grad = Array1::from_vec(vec![10.0, 10.0, 10.0]);
        accumulator.accumulate_gradients("param1", large_grad).unwrap();
        accumulator.increment_step();

        let result = accumulator.get_accumulated_gradients();
        let clipped_grad = &result["param1"];

        // Check that gradient norm is close to max_norm
        let norm: f32 = clipped_grad.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1);
    }
}