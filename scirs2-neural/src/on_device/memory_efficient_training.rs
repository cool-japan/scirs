//! Memory-efficient training techniques for on-device learning

use crate::error::Result;
use crate::layers::Layer;
use scirs2_core::ndarray::prelude::*;
use std::collections::VecDeque;
use scirs2_core::ndarray::ArrayView1;
/// Memory-efficient trainer
pub struct MemoryEfficientTrainer {
    /// Enable gradient accumulation
    gradient_accumulation: bool,
    /// Number of accumulation steps
    accumulation_steps: usize,
    /// Enable activation checkpointing
    activation_checkpointing: bool,
    /// Enable weight sharing
    weight_sharing: bool,
    /// Buffer reuse strategy
    buffer_reuse: BufferReuseStrategy,
    /// Memory pool
    memory_pool: MemoryPool,
}
impl MemoryEfficientTrainer {
    /// Create a new memory-efficient trainer
    pub fn new(_memory_budgetmb: usize) -> Self {
        Self {
            gradient_accumulation: true,
            accumulation_steps: 4,
            activation_checkpointing: true,
            weight_sharing: false,
            buffer_reuse: BufferReuseStrategy::Aggressive,
            memory_pool: MemoryPool::new(_memory_budget_mb * 1024 * 1024),
        }
    }
    
    /// Configure gradient accumulation
    pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
        self.gradient_accumulation = true;
        self.accumulation_steps = steps;
        self
    pub fn with_activation_checkpointing(mut self, enabled: bool) -> Self {
        self.activation_checkpointing = enabled;
    /// Train a model with memory-efficient techniques
    pub fn train_step(
        &mut self,
        model: &mut dyn Layer<f32>,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
        learning_rate: f32,
    ) -> Result<f32> {
        let batch_size = data.shape()[0];
        
        if self.gradient_accumulation {
            self.train_with_gradient_accumulation(model, data, labels, learning_rate)
        } else {
            self.train_standard(model, data, labels, learning_rate)
    /// Standard training step
    fn train_standard(
        // Forward pass
        let activations = if self.activation_checkpointing {
            self.forward_with_checkpointing(model, data)?
            model.forward(data)?
        };
        // Compute loss (simplified)
        let loss = self.compute_loss(&activations, labels)?;
        // Backward pass
        let grad_output = self.compute_grad_output(&activations, labels)?;
        let _ = model.backward(&grad_output.view())?;
        // Update weights
        self.update_weights(model, learning_rate)?;
        Ok(loss)
    /// Training with gradient accumulation
    fn train_with_gradient_accumulation(
        let micro_batch_size = (batch_size + self.accumulation_steps - 1) / self.accumulation_steps;
        let mut accumulated_loss = 0.0;
        let mut accumulated_gradients = GradientAccumulator::new();
        for step in 0..self.accumulation_steps {
            let start = step * micro_batch_size;
            let end = ((step + 1) * micro_batch_size).min(batch_size);
            
            if start >= batch_size {
                break;
            }
            let micro_data = data.slice(s![start..end, ..]);
            let micro_labels = labels.slice(s![start..end]);
            // Forward pass on micro-batch
            let activations = if self.activation_checkpointing {
                self.forward_with_checkpointing(model, &micro_data)?
            } else {
                model.forward(&micro_data)?
            };
            // Compute loss
            let loss = self.compute_loss(&activations, &micro_labels)?;
            accumulated_loss += loss;
            // Backward pass
            let grad_output = self.compute_grad_output(&activations, &micro_labels)?;
            let gradients = model.backward(&grad_output.view())?;
            // Accumulate gradients
            accumulated_gradients.accumulate(&gradients)?;
        // Average gradients and update weights
        accumulated_gradients.average(self.accumulation_steps);
        self.update_weights_with_gradients(model, &accumulated_gradients, learning_rate)?;
        Ok(accumulated_loss / self.accumulation_steps as f32)
    /// Forward pass with activation checkpointing
    fn forward_with_checkpointing(
        model: &dyn Layer<f32>,
    ) -> Result<Array2<f32>> {
        // Simplified checkpointing - in practice would save intermediate activations
        model.forward(data)
    /// Compute loss (simplified)
    fn compute_loss(&self, predictions: &Array2<f32>, labels: &ArrayView1<usize>) -> Result<f32> {
        // Simplified cross-entropy loss
        let batch_size = labels.len() as f32;
        let mut loss = 0.0;
        for (i, &label) in labels.iter().enumerate() {
            if label < predictions.shape()[1] {
                loss -= predictions[[i, label]].ln();
        Ok(loss / batch_size)
    /// Compute gradient of loss w.r.t output
    fn compute_grad_output(
        &self,
        predictions: &Array2<f32>,
        let mut grad = predictions.clone();
            if label < grad.shape()[1] {
                grad[[i, label]] -= 1.0;
        grad /= batch_size;
        Ok(grad)
    /// Update model weights
    fn update_weights(&self, model: &mut dyn Layer<f32>, learningrate: f32) -> Result<()> {
        // Simplified weight update - in practice would use optimizer
        Ok(())
    /// Update weights with accumulated gradients
    fn update_weights_with_gradients(
        gradients: &GradientAccumulator,
    ) -> Result<()> {
        // Simplified weight update with gradients
/// Gradient accumulation helper
pub struct GradientAccumulator {
    gradients: Vec<Array2<f32>>,
impl GradientAccumulator {
    /// Create a new gradient accumulator
    pub fn new() -> Self {
            gradients: Vec::new(),
    /// Accumulate gradients
    pub fn accumulate(&mut self, grads: &Array2<f32>) -> Result<()> {
        if self.gradients.is_empty() {
            self.gradients.push(grads.clone());
            // Add to existing gradients
            for (accumulated, new) in self.gradients.iter_mut().zip(std::iter::once(grads)) {
                *accumulated += new;
    /// Average accumulated gradients
    pub fn average(&mut self, numsteps: usize) {
        let scale = 1.0 / num_steps as f32;
        for grad in &mut self.gradients {
            *grad *= scale;
    /// Clear accumulated gradients
    pub fn clear(&mut self) {
        self.gradients.clear();
/// Buffer reuse strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BufferReuseStrategy {
    /// No buffer reuse
    None,
    /// Conservative reuse (only when safe)
    Conservative,
    /// Aggressive reuse (maximum memory savings)
    Aggressive,
/// Memory pool for efficient allocation
struct MemoryPool {
    total_size: usize,
    available: usize,
    buffers: VecDeque<Buffer>,
impl MemoryPool {
    fn new(size: usize) -> Self {
            total_size: size,
            available: size,
            buffers: VecDeque::new(),
    fn allocate(&mut self, size: usize) -> Option<Buffer> {
        // Try to reuse existing buffer
        for (i, buffer) in self.buffers.iter().enumerate() {
            if !buffer.in_use && buffer._size >= _size {
                let mut buffer = self.buffers.remove(i).unwrap();
                buffer.in_use = true;
                self.available -= buffer._size;
                return Some(buffer);
        // Allocate new buffer if space available
        if self.available >= _size {
            let buffer = Buffer {
                data: vec![0.0; _size],
                size,
                in_use: true,
            self.available -= size;
            Some(buffer)
            None
    fn release(&mut self, mut buffer: Buffer) {
        buffer.in_use = false;
        self.available += buffer._size;
        self.buffers.push_back(buffer);
struct Buffer {
    data: Vec<f32>,
    size: usize,
    in_use: bool,
/// Activation checkpointing for memory savings
pub struct ActivationCheckpointing {
    /// Checkpoint interval (every N layers)
    checkpoint_interval: usize,
    /// Stored checkpoints
    checkpoints: Vec<CheckpointData>,
impl ActivationCheckpointing {
    /// Create new activation checkpointing
    pub fn new(_checkpointinterval: usize) -> Self {
            checkpoint_interval,
            checkpoints: Vec::new(),
    /// Should checkpoint at this layer
    pub fn should_checkpoint(&self, layeridx: usize) -> bool {
        layer_idx % self.checkpoint_interval == 0
    /// Save checkpoint
    pub fn save_checkpoint(&mut self, layeridx: usize, data: Array2<f32>) {
        self.checkpoints.push(CheckpointData {
            layer_idx,
            activation: data,
        });
    /// Restore from checkpoint
    pub fn restore_checkpoint(&self, layeridx: usize) -> Option<&Array2<f32>> {
        self.checkpoints
            .iter()
            .find(|cp| cp.layer_idx == layer_idx)
            .map(|cp| &cp.activation)
    /// Clear all checkpoints
        self.checkpoints.clear();
struct CheckpointData {
    layer_idx: usize,
    activation: Array2<f32>,
/// Memory-efficient data loading
pub struct EfficientDataLoader {
    batch_size: usize,
    prefetch_factor: usize,
    pin_memory: bool,
impl EfficientDataLoader {
    /// Create a new efficient data loader
    pub fn new(_batchsize: usize) -> Self {
            batch_size,
            prefetch_factor: 2,
            pin_memory: true,
    /// Load batch with minimal memory overhead
    pub fn load_batch(&self, data: &ArrayView2<f32>, indices: &[usize]) -> Result<Array2<f32>> {
        let batch_size = indices.len().min(self.batch_size);
        let feature_dim = data.shape()[1];
        let mut batch = Array2::zeros((batch_size, feature_dim));
        for (i, &idx) in indices.iter().take(batch_size).enumerate() {
            if idx < data.shape()[0] {
                batch.row_mut(i).assign(&data.row(idx));
        Ok(batch)
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gradient_accumulator() {
        let mut accumulator = GradientAccumulator::new();
        let grad1 = Array2::ones((10, 5));
        let grad2 = Array2::ones((10, 5)) * 2.0;
        accumulator.accumulate(&grad1).unwrap();
        accumulator.accumulate(&grad2).unwrap();
        accumulator.average(2);
        // After averaging, should have (1 + 2) / 2 = 1.5
        assert_eq!(accumulator.gradients[0][[0, 0]], 1.5);
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(1000);
        let buffer1 = pool.allocate(100).unwrap();
        assert_eq!(pool.available, 900);
        pool.release(buffer1);
        assert_eq!(pool.available, 1000);
        // Should reuse the released buffer
        let buffer2 = pool.allocate(50).unwrap();
        assert_eq!(buffer2.size, 100); // Reused the 100-size buffer
    fn test_activation_checkpointing() {
        let mut checkpointing = ActivationCheckpointing::new(3);
        assert!(checkpointing.should_checkpoint(0));
        assert!(!checkpointing.should_checkpoint(1));
        assert!(!checkpointing.should_checkpoint(2));
        assert!(checkpointing.should_checkpoint(3));
        let data = Array2::ones((10, 5));
        checkpointing.save_checkpoint(0, data.clone());
        let restored = checkpointing.restore_checkpoint(0).unwrap();
        assert_eq!(restored.shape(), &[10, 5]);
