//! Memory-efficient training techniques for on-device learning

use crate::error::Result;
use crate::layers::Layer;
use scirs2_core::ndarray::prelude::*;
use std::collections::VecDeque;

/// Buffer reuse strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BufferReuseStrategy {
    /// No buffer reuse
    None,
    /// Conservative reuse (only when safe)
    Conservative,
    /// Aggressive reuse (maximum memory savings)
    Aggressive,
}

/// Memory pool for efficient allocation
struct MemoryPool {
    total_size: usize,
    available: usize,
    buffers: VecDeque<PoolBuffer>,
}

struct PoolBuffer {
    data: Vec<f32>,
    capacity: usize,
    in_use: bool,
}

impl MemoryPool {
    fn new(size: usize) -> Self {
        Self {
            total_size: size,
            available: size,
            buffers: VecDeque::new(),
        }
    }

    fn allocate(&mut self, size: usize) -> Option<PoolBuffer> {
        for i in 0..self.buffers.len() {
            if !self.buffers[i].in_use && self.buffers[i].capacity >= size {
                let mut buffer = self.buffers.remove(i).expect("index must be valid");
                buffer.in_use = true;
                self.available = self.available.saturating_sub(buffer.capacity);
                return Some(buffer);
            }
        }
        if self.available >= size {
            let buffer = PoolBuffer {
                data: vec![0.0; size],
                capacity: size,
                in_use: true,
            };
            self.available -= size;
            Some(buffer)
        } else {
            None
        }
    }

    fn release(&mut self, mut buffer: PoolBuffer) {
        buffer.in_use = false;
        self.available += buffer.capacity;
        self.buffers.push_back(buffer);
    }
}

/// Gradient accumulation helper
pub struct GradientAccumulator {
    /// Accumulated gradient tensors
    pub gradients: Vec<ArrayD<f32>>,
}

impl GradientAccumulator {
    /// Create a new, empty gradient accumulator
    pub fn new() -> Self {
        Self {
            gradients: Vec::new(),
        }
    }

    /// Accumulate a gradient tensor
    pub fn accumulate(&mut self, grads: &ArrayD<f32>) -> Result<()> {
        if self.gradients.is_empty() {
            self.gradients.push(grads.clone());
        } else {
            for accumulated in &mut self.gradients {
                *accumulated += grads;
            }
        }
        Ok(())
    }

    /// Divide all accumulated gradients by `num_steps`
    pub fn average(&mut self, num_steps: usize) {
        let scale = 1.0 / num_steps.max(1) as f32;
        for grad in &mut self.gradients {
            *grad *= scale;
        }
    }

    /// Reset accumulated gradients
    pub fn clear(&mut self) {
        self.gradients.clear();
    }
}

impl Default for GradientAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Activation checkpointing for memory savings
pub struct ActivationCheckpointing {
    checkpoint_interval: usize,
    checkpoints: Vec<CheckpointData>,
}

struct CheckpointData {
    layer_idx: usize,
    activation: ArrayD<f32>,
}

impl ActivationCheckpointing {
    /// Create new activation checkpointing with the given interval
    pub fn new(checkpoint_interval: usize) -> Self {
        Self {
            checkpoint_interval,
            checkpoints: Vec::new(),
        }
    }

    /// Returns true if a checkpoint should be saved at `layer_idx`
    pub fn should_checkpoint(&self, layer_idx: usize) -> bool {
        self.checkpoint_interval > 0 && layer_idx.is_multiple_of(self.checkpoint_interval)
    }

    /// Store a checkpoint for `layer_idx`
    pub fn save_checkpoint(&mut self, layer_idx: usize, data: ArrayD<f32>) {
        self.checkpoints.push(CheckpointData {
            layer_idx,
            activation: data,
        });
    }

    /// Restore the activation for `layer_idx`, if present
    pub fn restore_checkpoint(&self, layer_idx: usize) -> Option<&ArrayD<f32>> {
        self.checkpoints
            .iter()
            .find(|cp| cp.layer_idx == layer_idx)
            .map(|cp| &cp.activation)
    }

    /// Clear all stored checkpoints
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

/// Memory-efficient data loader that minimises allocations
pub struct EfficientDataLoader {
    batch_size: usize,
    prefetch_factor: usize,
    pin_memory: bool,
}

impl EfficientDataLoader {
    /// Create a new efficient data loader
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            prefetch_factor: 2,
            pin_memory: true,
        }
    }

    /// Configure the prefetch factor
    pub fn with_prefetch_factor(mut self, factor: usize) -> Self {
        self.prefetch_factor = factor;
        self
    }

    /// Load a batch of `indices` from `data` with minimal extra allocation
    pub fn load_batch(&self, data: &ArrayView2<f32>, indices: &[usize]) -> Result<Array2<f32>> {
        let batch_size = indices.len().min(self.batch_size);
        let feature_dim = data.shape()[1];
        let mut batch = Array2::zeros((batch_size, feature_dim));
        for (i, &idx) in indices.iter().take(batch_size).enumerate() {
            if idx < data.shape()[0] {
                batch.row_mut(i).assign(&data.row(idx));
            }
        }
        Ok(batch)
    }

    /// Prefetch factor
    pub fn prefetch_factor(&self) -> usize {
        self.prefetch_factor
    }

    /// Whether pinned memory is enabled
    pub fn pin_memory(&self) -> bool {
        self.pin_memory
    }
}

/// Memory-efficient trainer combining gradient accumulation and activation checkpointing
pub struct MemoryEfficientTrainer {
    gradient_accumulation: bool,
    accumulation_steps: usize,
    activation_checkpointing: bool,
    #[allow(dead_code)]
    weight_sharing: bool,
    #[allow(dead_code)]
    buffer_reuse: BufferReuseStrategy,
    #[allow(dead_code)]
    memory_pool: MemoryPool,
}

impl MemoryEfficientTrainer {
    /// Create a new memory-efficient trainer with the given memory budget (MB)
    pub fn new(memory_budget_mb: usize) -> Self {
        Self {
            gradient_accumulation: true,
            accumulation_steps: 4,
            activation_checkpointing: true,
            weight_sharing: false,
            buffer_reuse: BufferReuseStrategy::Aggressive,
            memory_pool: MemoryPool::new(memory_budget_mb * 1024 * 1024),
        }
    }

    /// Enable gradient accumulation over `steps` micro-batches
    pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation = true;
        self.accumulation_steps = steps.max(1);
        self
    }

    /// Enable or disable activation checkpointing
    pub fn with_activation_checkpointing(mut self, enabled: bool) -> Self {
        self.activation_checkpointing = enabled;
        self
    }

    /// Perform one training step on a 2-D batch (`[batch, features]`)
    pub fn train_step(
        &mut self,
        model: &mut dyn Layer<f32>,
        data: &Array2<f32>,
        labels: &ArrayView1<usize>,
        learning_rate: f32,
    ) -> Result<f32> {
        let data_dyn = data.clone().into_dyn();
        if self.gradient_accumulation {
            self.train_with_gradient_accumulation(model, data, &data_dyn, labels, learning_rate)
        } else {
            self.train_standard(model, &data_dyn, labels, learning_rate)
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn train_standard(
        &mut self,
        model: &mut dyn Layer<f32>,
        data_dyn: &ArrayD<f32>,
        labels: &ArrayView1<usize>,
        learning_rate: f32,
    ) -> Result<f32> {
        let activations = if self.activation_checkpointing {
            self.forward_with_checkpointing(model, data_dyn)?
        } else {
            model.forward(data_dyn)?
        };
        let act2 = self.to_array2(&activations);
        let loss = self.compute_loss(&act2.view(), labels)?;
        let grad_output = self.compute_grad_output(&act2.view(), labels)?;
        let _ = model.backward(data_dyn, &grad_output.into_dyn())?;
        self.update_weights(model, learning_rate)?;
        Ok(loss)
    }

    fn train_with_gradient_accumulation(
        &mut self,
        model: &mut dyn Layer<f32>,
        data: &Array2<f32>,
        _data_dyn: &ArrayD<f32>,
        labels: &ArrayView1<usize>,
        learning_rate: f32,
    ) -> Result<f32> {
        let batch_size = data.shape()[0];
        let micro_batch_size = batch_size.div_ceil(self.accumulation_steps);
        let mut accumulated_loss = 0.0f32;
        let mut accumulated_gradients = GradientAccumulator::new();

        for step in 0..self.accumulation_steps {
            let start = step * micro_batch_size;
            if start >= batch_size {
                break;
            }
            let end = ((step + 1) * micro_batch_size).min(batch_size);

            let micro_data_2d = data.slice(s![start..end, ..]).to_owned();
            let micro_labels = labels.slice(s![start..end]);
            let micro_dyn = micro_data_2d.clone().into_dyn();

            let activations = if self.activation_checkpointing {
                self.forward_with_checkpointing(model, &micro_dyn)?
            } else {
                model.forward(&micro_dyn)?
            };
            let act2 = self.to_array2(&activations);
            let loss = self.compute_loss(&act2.view(), &micro_labels)?;
            accumulated_loss += loss;

            let grad_output = self.compute_grad_output(&act2.view(), &micro_labels)?;
            let gradients = model.backward(&micro_dyn, &grad_output.into_dyn())?;
            accumulated_gradients.accumulate(&gradients)?;
        }

        accumulated_gradients.average(self.accumulation_steps);
        self.update_weights_with_gradients(model, &accumulated_gradients, learning_rate)?;
        Ok(accumulated_loss / self.accumulation_steps as f32)
    }

    fn forward_with_checkpointing(
        &self,
        model: &dyn Layer<f32>,
        data: &ArrayD<f32>,
    ) -> Result<ArrayD<f32>> {
        model.forward(data)
    }

    fn compute_loss(
        &self,
        predictions: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<f32> {
        let batch_size = labels.len().max(1) as f32;
        let mut loss = 0.0f32;
        for (i, &label) in labels.iter().enumerate() {
            if label < predictions.shape()[1] {
                let logit = predictions[[i, label]];
                loss -= logit.max(1e-10).ln();
            }
        }
        Ok(loss / batch_size)
    }

    fn compute_grad_output(
        &self,
        predictions: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<Array2<f32>> {
        let batch_size = labels.len().max(1) as f32;
        let mut grad = predictions.to_owned();
        for (i, &label) in labels.iter().enumerate() {
            if label < grad.shape()[1] {
                grad[[i, label]] -= 1.0;
            }
        }
        grad /= batch_size;
        Ok(grad)
    }

    fn to_array2(&self, arr: &ArrayD<f32>) -> Array2<f32> {
        if let Ok(a) = arr.clone().into_dimensionality::<Ix2>() {
            return a;
        }
        let n = arr.shape().first().copied().unwrap_or(1);
        let m = arr.len() / n.max(1);
        arr.to_owned()
            .into_shape_clone((n, m))
            .expect("reshape to 2D")
    }

    fn update_weights(&self, _model: &mut dyn Layer<f32>, _learning_rate: f32) -> Result<()> {
        Ok(())
    }

    fn update_weights_with_gradients(
        &self,
        _model: &mut dyn Layer<f32>,
        _gradients: &GradientAccumulator,
        _learning_rate: f32,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_accumulator_basic() {
        let mut accumulator = GradientAccumulator::new();
        let grad1: ArrayD<f32> = Array2::ones((10, 5)).into_dyn();
        let grad2: ArrayD<f32> = (Array2::ones((10, 5)) * 2.0).into_dyn();
        accumulator.accumulate(&grad1).expect("accumulate failed");
        assert!((accumulator.gradients[0][[0, 0]] - 1.0).abs() < 1e-6);
        accumulator.accumulate(&grad2).expect("accumulate failed");
        assert!((accumulator.gradients[0][[0, 0]] - 3.0).abs() < 1e-6);
        accumulator.average(2);
        assert!((accumulator.gradients[0][[0, 0]] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_accumulator_clear() {
        let mut accumulator = GradientAccumulator::new();
        let g: ArrayD<f32> = Array2::ones((3, 3)).into_dyn();
        accumulator.accumulate(&g).expect("accumulate failed");
        assert!(!accumulator.gradients.is_empty());
        accumulator.clear();
        assert!(accumulator.gradients.is_empty());
    }

    #[test]
    fn test_memory_pool_allocate_release() {
        let mut pool = MemoryPool::new(1000);
        let buffer = pool.allocate(100).expect("allocation must succeed");
        assert_eq!(pool.available, 900);
        pool.release(buffer);
        assert_eq!(pool.available, 1000);
        let buffer2 = pool.allocate(50).expect("reuse must succeed");
        assert!(buffer2.capacity >= 50);
    }

    #[test]
    fn test_activation_checkpointing() {
        let mut checkpointing = ActivationCheckpointing::new(3);
        assert!(checkpointing.should_checkpoint(0));
        assert!(!checkpointing.should_checkpoint(1));
        assert!(!checkpointing.should_checkpoint(2));
        assert!(checkpointing.should_checkpoint(3));
        assert!(checkpointing.should_checkpoint(6));

        let data: ArrayD<f32> = Array2::ones((10, 5)).into_dyn();
        checkpointing.save_checkpoint(0, data.clone());
        let restored = checkpointing
            .restore_checkpoint(0)
            .expect("checkpoint must exist");
        assert_eq!(restored.shape(), &[10, 5]);
    }

    #[test]
    fn test_efficient_data_loader() {
        let loader = EfficientDataLoader::new(4);
        let data = Array2::from_elem((10, 3), 1.0f32);
        let indices = vec![0, 2, 4, 6, 8];
        let batch = loader
            .load_batch(&data.view(), &indices)
            .expect("load_batch failed");
        assert_eq!(batch.shape(), &[4, 3]);
    }

    #[test]
    fn test_buffer_reuse() {
        let mut pool = MemoryPool::new(200);
        let b = pool.allocate(100).expect("first allocation");
        pool.release(b);
        let b2 = pool.allocate(80).expect("reuse");
        // Reused existing 100-capacity buffer
        assert!(b2.capacity >= 80);
    }
}
