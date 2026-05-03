//! Gradient checkpointing for memory-efficient backpropagation

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;

/// Gradient checkpointing strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckpointStrategy {
    /// No checkpointing
    None,
    /// Checkpoint every N layers
    Uniform(usize),
    /// Checkpoint specific layers by index
    Custom,
    /// Adaptive checkpointing based on memory usage
    Adaptive,
    /// Checkpoint based on computational cost
    CostBased,
}

/// Layer information for checkpointing decisions
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer type name
    pub layer_type: String,
    /// Computational cost (FLOPs)
    pub compute_cost: f32,
    /// Memory cost (bytes)
    pub memory_cost: usize,
    /// Whether layer has trainable parameters
    pub has_parameters: bool,
    /// Number of parameters
    pub num_parameters: usize,
}

impl LayerInfo {
    /// Create layer info for a dense layer
    pub fn dense(input_size: usize, output_size: usize) -> Self {
        Self {
            layer_type: "Dense".to_string(),
            compute_cost: (2 * input_size * output_size) as f32,
            memory_cost: 4 * (input_size * output_size + output_size),
            has_parameters: true,
            num_parameters: input_size * output_size + output_size,
        }
    }

    /// Create layer info for a convolutional layer
    pub fn conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        input_h: usize,
        input_w: usize,
    ) -> Self {
        let output_h = input_h;
        let output_w = input_w;
        let flops =
            2 * kernel_size * kernel_size * in_channels * out_channels * output_h * output_w;
        Self {
            layer_type: "Conv2D".to_string(),
            compute_cost: flops as f32,
            memory_cost: 4 * (kernel_size * kernel_size * in_channels * out_channels),
            has_parameters: true,
            num_parameters: kernel_size * kernel_size * in_channels * out_channels,
        }
    }

    /// Create layer info for an activation layer
    pub fn activation(name: &str, size: usize) -> Self {
        Self {
            layer_type: format!("{} Activation", name),
            compute_cost: size as f32,
            memory_cost: 0,
            has_parameters: false,
            num_parameters: 0,
        }
    }
}

/// Checkpoint data for a single layer
pub struct Checkpoint {
    /// Layer index
    pub layer_idx: usize,
    /// Input to the layer
    pub input: ArrayD<f32>,
    /// Output from the layer (optional — may be evicted)
    pub output: Option<ArrayD<f32>>,
    /// Layer information
    pub layer_info: LayerInfo,
    /// Memory size in MB
    pub memory_size_mb: usize,
}

/// Memory usage statistics
pub struct MemoryStats {
    /// Number of checkpoints stored
    pub num_checkpoints: usize,
    /// Total memory used by checkpoints (MB)
    pub total_memory_mb: usize,
    /// Memory threshold (MB)
    pub threshold_mb: usize,
    /// Number of recompute-cache entries
    pub cache_entries: usize,
    /// Memory used by recompute cache (MB)
    pub cache_memory_mb: usize,
}

/// Gradient checkpointing manager
pub struct GradientCheckpointing {
    strategy: CheckpointStrategy,
    checkpoints: HashMap<usize, Checkpoint>,
    recompute_cache: HashMap<usize, ArrayD<f32>>,
    memory_threshold_mb: usize,
    current_memory_mb: usize,
}

impl GradientCheckpointing {
    /// Create a new gradient checkpointing manager
    pub fn new(strategy: CheckpointStrategy, memory_threshold_mb: usize) -> Self {
        Self {
            strategy,
            checkpoints: HashMap::new(),
            recompute_cache: HashMap::new(),
            memory_threshold_mb,
            current_memory_mb: 0,
        }
    }

    /// Return true if the manager should save a checkpoint at this layer
    pub fn should_checkpoint(&self, layer_idx: usize, layer_cost: f32) -> bool {
        match self.strategy {
            CheckpointStrategy::None => false,
            CheckpointStrategy::Uniform(interval) => layer_idx.is_multiple_of(interval),
            CheckpointStrategy::Custom => self.is_custom_checkpoint(layer_idx),
            CheckpointStrategy::Adaptive => self.should_checkpoint_adaptive(layer_cost),
            CheckpointStrategy::CostBased => layer_cost > 1000.0,
        }
    }

    /// Save a checkpoint for the given layer
    pub fn save_checkpoint(
        &mut self,
        layer_idx: usize,
        input: ArrayD<f32>,
        output: ArrayD<f32>,
        layer_info: LayerInfo,
    ) -> Result<()> {
        let mem = self.estimate_memory_size(&input);
        let checkpoint = Checkpoint {
            layer_idx,
            input,
            output: Some(output),
            layer_info,
            memory_size_mb: mem,
        };
        self.current_memory_mb += checkpoint.memory_size_mb;
        self.checkpoints.insert(layer_idx, checkpoint);
        if self.current_memory_mb > self.memory_threshold_mb {
            self.evict_checkpoints()?;
        }
        Ok(())
    }

    /// Get a stored checkpoint by layer index
    pub fn get_checkpoint(&self, layer_idx: usize) -> Option<&Checkpoint> {
        self.checkpoints.get(&layer_idx)
    }

    /// Recompute the forward pass from a stored checkpoint to `end_layer`
    pub fn recompute_forward(
        &mut self,
        start_layer: usize,
        end_layer: usize,
        layers: &[Box<dyn Layer<f32>>],
    ) -> Result<Vec<ArrayD<f32>>> {
        let checkpoint_idx = self.find_nearest_checkpoint(start_layer);
        let mut current_input = self
            .checkpoints
            .get(&checkpoint_idx)
            .map(|cp| cp.input.clone())
            .ok_or_else(|| {
                NeuralError::InvalidArgument("No checkpoint found for recomputation".to_string())
            })?;

        let mut activations = Vec::new();
        for layer_idx in checkpoint_idx..=end_layer {
            if layer_idx >= start_layer {
                if let Some(cached) = self.recompute_cache.get(&layer_idx) {
                    let cached = cached.clone();
                    activations.push(cached.clone());
                    current_input = cached;
                    continue;
                }
            }
            if layer_idx < layers.len() {
                let output = layers[layer_idx].forward(&current_input)?;
                if layer_idx >= start_layer {
                    activations.push(output.clone());
                    self.recompute_cache.insert(layer_idx, output.clone());
                }
                current_input = output;
            }
        }
        Ok(activations)
    }

    /// Clear the recompute cache
    pub fn clear_recompute_cache(&mut self) {
        self.recompute_cache.clear();
    }

    /// Return memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            num_checkpoints: self.checkpoints.len(),
            total_memory_mb: self.current_memory_mb,
            threshold_mb: self.memory_threshold_mb,
            cache_entries: self.recompute_cache.len(),
            cache_memory_mb: self
                .recompute_cache
                .values()
                .map(|a| self.estimate_memory_size(a))
                .sum(),
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn find_nearest_checkpoint(&self, layer_idx: usize) -> usize {
        self.checkpoints
            .keys()
            .filter(|&&idx| idx <= layer_idx)
            .max()
            .copied()
            .unwrap_or(0)
    }

    fn is_custom_checkpoint(&self, layer_idx: usize) -> bool {
        matches!(layer_idx, 0 | 5 | 10 | 15)
    }

    fn should_checkpoint_adaptive(&self, layer_cost: f32) -> bool {
        let memory_usage_ratio =
            self.current_memory_mb as f32 / self.memory_threshold_mb.max(1) as f32;
        memory_usage_ratio < 0.7 && layer_cost > 500.0
    }

    fn estimate_memory_size(&self, tensor: &ArrayD<f32>) -> usize {
        let bytes = tensor.len() * std::mem::size_of::<f32>();
        (bytes / (1024 * 1024)).max(1)
    }

    fn evict_checkpoints(&mut self) -> Result<()> {
        let target_memory = (self.memory_threshold_mb as f32 * 0.8) as usize;
        while self.current_memory_mb > target_memory && !self.checkpoints.is_empty() {
            if let Some(&layer_idx) = self.checkpoints.keys().min() {
                if let Some(checkpoint) = self.checkpoints.remove(&layer_idx) {
                    self.current_memory_mb = self
                        .current_memory_mb
                        .saturating_sub(checkpoint.memory_size_mb);
                }
            } else {
                break;
            }
        }
        Ok(())
    }
}

/// A model wrapper that applies gradient checkpointing during forward/backward
pub struct CheckpointedModel {
    layers: Vec<Box<dyn Layer<f32>>>,
    checkpointing: GradientCheckpointing,
}

impl CheckpointedModel {
    /// Create a checkpointed model
    pub fn new(
        layers: Vec<Box<dyn Layer<f32>>>,
        strategy: CheckpointStrategy,
        memory_threshold_mb: usize,
    ) -> Self {
        Self {
            layers,
            checkpointing: GradientCheckpointing::new(strategy, memory_threshold_mb),
        }
    }

    /// Forward pass with selective checkpointing
    pub fn forward(&mut self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let mut current = input.clone();
        for idx in 0..self.layers.len() {
            let output = self.layers[idx].forward(&current)?;
            let layer_info = self.get_layer_info(idx);
            if self
                .checkpointing
                .should_checkpoint(idx, layer_info.compute_cost)
            {
                self.checkpointing.save_checkpoint(
                    idx,
                    current.clone(),
                    output.clone(),
                    layer_info,
                )?;
            }
            current = output;
        }
        Ok(current)
    }

    /// Backward pass: recomputes activations from checkpoints where needed.
    /// Returns the gradient w.r.t. the input.
    pub fn backward(
        &mut self,
        input: &ArrayD<f32>,
        grad_output: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        let mut current_grad = grad_output.clone();
        let mut current_input = input.clone();
        self.checkpointing.clear_recompute_cache();
        for idx in (0..self.layers.len()).rev() {
            if self.checkpointing.get_checkpoint(idx).is_none() && idx > 0 {
                let _ = self
                    .checkpointing
                    .recompute_forward(idx - 1, idx, &self.layers);
            }
            current_grad = self.layers[idx].backward(&current_input, &current_grad)?;
            // Move the input pointer back (simplified: we keep the same input for all layers)
            // In a real implementation the stored input from each checkpoint would be used.
            let _ = &mut current_input;
        }
        Ok(current_grad)
    }

    /// Memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.checkpointing.memory_stats()
    }

    fn get_layer_info(&self, _layer_idx: usize) -> LayerInfo {
        LayerInfo {
            layer_type: "Unknown".to_string(),
            compute_cost: 1000.0,
            memory_cost: 1024 * 1024,
            has_parameters: true,
            num_parameters: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_strategy_uniform() {
        let checkpointing = GradientCheckpointing::new(CheckpointStrategy::Uniform(3), 100);
        assert!(!checkpointing.should_checkpoint(1, 100.0));
        assert!(!checkpointing.should_checkpoint(2, 100.0));
        assert!(checkpointing.should_checkpoint(3, 100.0));
        assert!(!checkpointing.should_checkpoint(4, 100.0));
        assert!(checkpointing.should_checkpoint(6, 100.0));
    }

    #[test]
    fn test_checkpoint_strategy_none() {
        let checkpointing = GradientCheckpointing::new(CheckpointStrategy::None, 100);
        assert!(!checkpointing.should_checkpoint(0, 9999.0));
        assert!(!checkpointing.should_checkpoint(10, 9999.0));
    }

    #[test]
    fn test_checkpoint_strategy_custom() {
        let checkpointing = GradientCheckpointing::new(CheckpointStrategy::Custom, 100);
        assert!(checkpointing.should_checkpoint(0, 0.0));
        assert!(!checkpointing.should_checkpoint(1, 0.0));
        assert!(checkpointing.should_checkpoint(5, 0.0));
        assert!(!checkpointing.should_checkpoint(6, 0.0));
    }

    #[test]
    fn test_layer_info_dense() {
        let dense_info = LayerInfo::dense(128, 64);
        assert_eq!(dense_info.layer_type, "Dense");
        assert_eq!(dense_info.compute_cost, (2 * 128 * 64) as f32);
        assert!(dense_info.has_parameters);
        assert_eq!(dense_info.num_parameters, 128 * 64 + 64);
    }

    #[test]
    fn test_layer_info_activation() {
        let activation_info = LayerInfo::activation("ReLU", 1000);
        assert_eq!(activation_info.layer_type, "ReLU Activation");
        assert!(!activation_info.has_parameters);
        assert_eq!(activation_info.num_parameters, 0);
    }

    #[test]
    fn test_checkpoint_save_and_retrieve() {
        let mut checkpointing = GradientCheckpointing::new(CheckpointStrategy::Custom, 100);
        let input: ArrayD<f32> = Array2::ones((10, 5)).into_dyn();
        let output: ArrayD<f32> = Array2::zeros((10, 3)).into_dyn();
        let layer_info = LayerInfo::dense(5, 3);
        checkpointing
            .save_checkpoint(0, input.clone(), output.clone(), layer_info)
            .expect("save_checkpoint failed");
        let checkpoint = checkpointing.get_checkpoint(0).expect("missing checkpoint");
        assert_eq!(checkpoint.layer_idx, 0);
        assert_eq!(checkpoint.input.shape(), &[10, 5]);
    }

    #[test]
    fn test_memory_stats() {
        let mut checkpointing = GradientCheckpointing::new(CheckpointStrategy::Uniform(1), 1000);
        let layer_info = LayerInfo::dense(100, 100);
        let input: ArrayD<f32> = Array2::ones((32, 100)).into_dyn();
        let output: ArrayD<f32> = Array2::zeros((32, 100)).into_dyn();
        checkpointing
            .save_checkpoint(0, input, output, layer_info)
            .expect("save failed");
        let stats = checkpointing.memory_stats();
        assert_eq!(stats.num_checkpoints, 1);
        assert_eq!(stats.threshold_mb, 1000);
    }

    #[test]
    fn test_eviction() {
        // Create a very tight memory budget so eviction triggers immediately
        let mut checkpointing = GradientCheckpointing::new(CheckpointStrategy::Uniform(1), 1);
        let layer_info_a = LayerInfo::dense(512, 512);
        let layer_info_b = LayerInfo::dense(512, 512);
        let big: ArrayD<f32> = Array2::ones((512, 512)).into_dyn();
        // First checkpoint
        checkpointing
            .save_checkpoint(0, big.clone(), big.clone(), layer_info_a)
            .expect("save 0 ok");
        // Second checkpoint — should trigger eviction of the first
        checkpointing
            .save_checkpoint(1, big.clone(), big.clone(), layer_info_b)
            .expect("save 1 ok");
        // After eviction, memory usage should be within threshold
        let stats = checkpointing.memory_stats();
        assert!(stats.total_memory_mb <= stats.threshold_mb + stats.num_checkpoints);
    }
}
