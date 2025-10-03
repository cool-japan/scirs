//! Gradient checkpointing for memory-efficient backpropagation

use crate::error::Result;
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
    /// Checkpoint specific layers
    Custom,
    /// Adaptive checkpointing based on memory usage
    Adaptive,
    /// Checkpoint based on computational cost
    CostBased,
}
/// Gradient checkpointing manager
pub struct GradientCheckpointing {
    strategy: CheckpointStrategy,
    checkpoints: HashMap<usize, Checkpoint>,
    recompute_cache: HashMap<usize, Array2<f32>>,
    memory_threshold_mb: usize,
    current_memory_mb: usize,
impl GradientCheckpointing {
    /// Create a new gradient checkpointing manager
    pub fn new(_strategy: CheckpointStrategy, memory_thresholdmb: usize) -> Self {
        Self {
            strategy,
            checkpoints: HashMap::new(),
            recompute_cache: HashMap::new(),
            memory_threshold_mb,
            current_memory_mb: 0,
        }
    }
    
    /// Check if should checkpoint at layer
    pub fn should_checkpoint(&self, layer_idx: usize, layercost: f32) -> bool {
        match self.strategy {
            CheckpointStrategy::None => false,
            CheckpointStrategy::Uniform(interval) => layer_idx % interval == 0,
            CheckpointStrategy::Custom => self.is_custom_checkpoint(layer_idx),
            CheckpointStrategy::Adaptive => self.should_checkpoint_adaptive(layer_cost),
            CheckpointStrategy::CostBased => layer_cost > 1000.0, // Threshold for expensive ops
    /// Save checkpoint
    pub fn save_checkpoint(
        &mut self,
        layer_idx: usize,
        input: Array2<f32>,
        output: Array2<f32>,
        layer_info: LayerInfo,
    ) -> Result<()> {
        let checkpoint = Checkpoint {
            layer_idx,
            input,
            output: Some(output),
            layer_info,
            memory_size_mb: self.estimate_memory_size(&input),
        };
        
        self.current_memory_mb += checkpoint.memory_size_mb;
        self.checkpoints.insert(layer_idx, checkpoint);
        // Evict checkpoints if memory limit exceeded
        if self.current_memory_mb > self.memory_threshold_mb {
            self.evict_checkpoints()?;
        Ok(())
    /// Get checkpoint for layer
    pub fn get_checkpoint(&self, layeridx: usize) -> Option<&Checkpoint> {
        self.checkpoints.get(&layer_idx)
    /// Recompute forward pass from checkpoint
    pub fn recompute_forward(
        start_layer: usize,
        end_layer: usize,
        layers: &[Box<dyn Layer<f32>>],
    ) -> Result<Vec<Array2<f32>>> {
        // Find nearest checkpoint before start_layer
        let checkpoint_idx = self.find_nearest_checkpoint(start_layer);
        let mut activations = Vec::new();
        let mut current_input = if let Some(checkpoint) = self.checkpoints.get(&checkpoint_idx) {
            checkpoint.input.clone()
        } else {
            return Err(crate::error::NeuralError::InvalidArgument(
                "No checkpoint found for recomputation".to_string()
            ));
        // Recompute from checkpoint to end_layer
        for layer_idx in checkpoint_idx..=end_layer {
            if layer_idx >= start_layer {
                // Check recompute cache first
                if let Some(cached) = self.recompute_cache.get(&layer_idx) {
                    activations.push(cached.clone());
                    current_input = cached.clone();
                    continue;
                }
            }
            
            let output = layers[layer_idx].forward(&current_input.view())?;
                activations.push(output.clone());
                
                // Cache recomputed activation
                self.recompute_cache.insert(layer_idx, output.clone());
            current_input = output;
        Ok(activations)
    /// Clear recompute cache
    pub fn clear_recompute_cache(&mut self) {
        self.recompute_cache.clear();
    /// Find nearest checkpoint before layer
    fn find_nearest_checkpoint(&self, layeridx: usize) -> usize {
        self.checkpoints
            .keys()
            .filter(|&&idx| idx <= layer_idx)
            .max()
            .copied()
            .unwrap_or(0)
    /// Check if layer is custom checkpoint
    fn is_custom_checkpoint(&self, layeridx: usize) -> bool {
        // Custom logic for specific layers
        matches!(layer_idx, 0 | 5 | 10 | 15)
    /// Adaptive checkpointing decision
    fn should_checkpoint_adaptive(&self, layercost: f32) -> bool {
        // Checkpoint if memory usage is low and layer is expensive
        let memory_usage_ratio = self.current_memory_mb as f32 / self.memory_threshold_mb as f32;
        memory_usage_ratio < 0.7 && layer_cost > 500.0
    /// Estimate memory size in MB
    fn estimate_memory_size(&self, tensor: &Array2<f32>) -> usize {
        let bytes = tensor.len() * std::mem::size_of::<f32>();
        (bytes / (1024 * 1024)).max(1)
    /// Evict checkpoints when memory limit exceeded
    fn evict_checkpoints(&mut self) -> Result<()> {
        // Simple LRU-style eviction
        // In practice, would use more sophisticated strategy
        let target_memory = (self.memory_threshold_mb as f32 * 0.8) as usize;
        while self.current_memory_mb > target_memory && !self.checkpoints.is_empty() {
            // Find checkpoint with lowest importance score
            if let Some(&layer_idx) = self.checkpoints.keys().min() {
                if let Some(checkpoint) = self.checkpoints.remove(&layer_idx) {
                    self.current_memory_mb -= checkpoint.memory_size_mb;
            } else {
                break;
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            num_checkpoints: self.checkpoints.len(),
            total_memory_mb: self.current_memory_mb,
            threshold_mb: self.memory_threshold_mb,
            cache_entries: self.recompute_cache.len(),
            cache_memory_mb: self.recompute_cache.values()
                .map(|a| self.estimate_memory_size(a))
                .sum(),
/// Checkpoint data
pub struct Checkpoint {
    /// Layer index
    pub layer_idx: usize,
    /// Input to the layer
    pub input: Array2<f32>,
    /// Output from the layer (optional)
    pub output: Option<Array2<f32>>,
    /// Layer information
    pub layer_info: LayerInfo,
    /// Memory size in MB
    pub memory_size_mb: usize,
/// Layer information for checkpointing decisions
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer type
    pub layer_type: String,
    /// Computational cost (FLOPs)
    pub compute_cost: f32,
    /// Memory cost (bytes)
    pub memory_cost: usize,
    /// Whether layer has trainable parameters
    pub has_parameters: bool,
    /// Number of parameters
    pub num_parameters: usize,
impl LayerInfo {
    /// Create layer info for a dense layer
    pub fn dense(_input_size: usize, outputsize: usize) -> Self {
            layer_type: "Dense".to_string(),
            compute_cost: (2 * _input_size * output_size) as f32,
            memory_cost: 4 * (_input_size * output_size + output_size),
            has_parameters: true,
            num_parameters: _input_size * output_size + output_size,
    /// Create layer info for a convolutional layer
    pub fn conv2d(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        input_h: usize,
        input_w: usize,
    ) -> Self {
        let output_h = input_h; // Simplified, assuming same padding
        let output_w = input_w;
        let flops = 2 * kernel_size * kernel_size * in_channels * out_channels * output_h * output_w;
            layer_type: "Conv2D".to_string(),
            compute_cost: flops as f32,
            memory_cost: 4 * (kernel_size * kernel_size * in_channels * out_channels),
            num_parameters: kernel_size * kernel_size * in_channels * out_channels,
    /// Create layer info for an activation layer
    pub fn activation(name: &str, size: usize) -> Self {
            layer_type: format!("{} Activation", name),
            compute_cost: size as f32,
            memory_cost: 0,
            has_parameters: false,
            num_parameters: 0,
/// Memory usage statistics
pub struct MemoryStats {
    /// Number of checkpoints
    pub num_checkpoints: usize,
    /// Total memory used by checkpoints
    pub total_memory_mb: usize,
    /// Memory threshold
    pub threshold_mb: usize,
    /// Number of cache entries
    pub cache_entries: usize,
    /// Memory used by cache
    pub cache_memory_mb: usize,
/// Checkpointed model wrapper
pub struct CheckpointedModel {
    layers: Vec<Box<dyn Layer<f32>>>,
    checkpointing: GradientCheckpointing,
impl CheckpointedModel {
    /// Create a new checkpointed model
    pub fn new(
        layers: Vec<Box<dyn Layer<f32>>>,
        strategy: CheckpointStrategy,
        memory_threshold_mb: usize,
            layers,
            checkpointing: GradientCheckpointing::new(strategy, memory_threshold_mb),
    /// Forward pass with checkpointing
    pub fn forward(&mut self, input: &ArrayView2<f32>) -> Result<Array2<f32>> {
        let mut current = input.to_owned();
        for (idx, layer) in self.layers.iter().enumerate() {
            let output = layer.forward(&current.view())?;
            // Decide whether to checkpoint
            let layer_info = self.get_layer_info(idx);
            if self.checkpointing.should_checkpoint(idx, layer_info.compute_cost) {
                self.checkpointing.save_checkpoint(
                    idx,
                    current.clone(),
                    output.clone(),
                    layer_info,
                )?;
            current = output;
        Ok(current)
    /// Backward pass with recomputation
    pub fn backward(&self, gradoutput: &ArrayView2<f32>) -> Result<Array2<f32>> {
        let mut current_grad = grad_output.to_owned();
        // Clear recompute cache before backward pass
        self.checkpointing.clear_recompute_cache();
        for idx in (0..self.layers.len()).rev() {
            // Check if we need to recompute activations
            if !self.checkpointing.get_checkpoint(idx).is_some() && idx > 0 {
                // Recompute activations from nearest checkpoint
                let activations = self.checkpointing.recompute_forward(
                    idx - 1,
                    &self.layers,
                // Use recomputed activation for backward pass
                if let Some(activation) = activations.last() {
                    current_grad = self.layers[idx].backward(&current_grad.view())?;
                current_grad = self.layers[idx].backward(&current_grad.view())?;
        Ok(current_grad)
    /// Get layer information
    fn get_layer_info(&self, layeridx: usize) -> LayerInfo {
        // Simplified - would extract actual layer info
        LayerInfo {
            layer_type: "Unknown".to_string(),
            compute_cost: 1000.0,
            memory_cost: 1024 * 1024,
            num_parameters: 1000,
    /// Get memory statistics
        self.checkpointing.memory_stats()
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_checkpoint_strategy() {
        let uniform = CheckpointStrategy::Uniform(3);
        let checkpointing = GradientCheckpointing::new(uniform, 100);
        assert!(!checkpointing.should_checkpoint(1, 100.0));
        assert!(!checkpointing.should_checkpoint(2, 100.0));
        assert!(checkpointing.should_checkpoint(3, 100.0));
        assert!(!checkpointing.should_checkpoint(4, 100.0));
    fn test_layer_info() {
        let dense_info = LayerInfo::dense(128, 64);
        assert_eq!(dense_info.layer_type, "Dense");
        assert_eq!(dense_info.compute_cost, (2 * 128 * 64) as f32);
        assert!(dense_info.has_parameters);
        let activation_info = LayerInfo::activation("ReLU", 1000);
        assert_eq!(activation_info.layer_type, "ReLU Activation");
        assert!(!activation_info.has_parameters);
    fn test_checkpoint_save_and_retrieve() {
        let mut checkpointing = GradientCheckpointing::new(CheckpointStrategy::Custom, 100);
        let input = Array2::ones((10, 5));
        let output = Array2::zeros((10, 3));
        let layer_info = LayerInfo::dense(5, 3);
        checkpointing.save_checkpoint(0, input.clone(), output.clone(), layer_info).unwrap();
        let checkpoint = checkpointing.get_checkpoint(0).unwrap();
        assert_eq!(checkpoint.layer_idx, 0);
        assert_eq!(checkpoint.input.shape(), &[10, 5]);
