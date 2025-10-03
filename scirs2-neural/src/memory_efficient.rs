//! Memory-efficient implementations for neural networks
//!
//! This module provides memory optimization techniques including:
//! - Gradient checkpointing for reduced memory usage during training
//! - In-place operations to minimize memory allocations
//! - Memory pool management for efficient tensor allocation
//! - Memory-aware batch processing
//! - Lazy evaluation and computation graphs

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, ArrayD, ArrayView, IxDyn};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};
#[cfg(feature = "memory_management")]
use scirs2_core::memory_efficient::BufferPool;
#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{chunk_wise_op, ChunkingStrategy};
#[cfg(feature = "linalg")]
use scirs2_linalg;
// Note: Using BufferPool for memory management instead of MemoryManager
// Note: These imports may need to be adjusted based on available types in scirs2_core
// #[cfg(feature = "memory_management")]
// use scirs2_core::memory_management::{
//     AllocationStrategy, BufferPool, MemoryManager, MemoryMetrics,
// };
#[cfg(feature = "cache")]
use scirs2_core::cache::{CacheBuilder, TTLSizedCache};
use statrs::statistics::Statistics;
/// Memory usage tracking and reporting
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Current memory usage in bytes
    pub current_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Total allocations made
    pub total_allocations: usize,
}
impl MemoryUsage {
    /// Create a new memory usage tracker
    pub fn new() -> Self {
        Self {
            current_bytes: 0,
            peak_bytes: 0,
            active_allocations: 0,
            total_allocations: 0,
        }
    }
    /// Update memory usage statistics
    pub fn allocate(&mut self, bytes: usize) {
        self.current_bytes += bytes;
        self.peak_bytes = self.peak_bytes.max(self.current_bytes);
        self.active_allocations += 1;
        self.total_allocations += 1;
    /// Record memory deallocation
    pub fn deallocate(&mut self, bytes: usize) {
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
        self.active_allocations = self.active_allocations.saturating_sub(1);
    /// Get memory usage in MB
    pub fn current_mb(&self) -> f64 {
        self.current_bytes as f64 / (1024.0 * 1024.0)
    /// Get peak memory usage in MB
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
impl Default for MemoryUsage {
    fn default() -> Self {
        Self::new()
/// Memory pool for efficient tensor allocation and reuse
pub struct MemoryPool<F: Float + Debug> {
    /// Available tensors organized by size
    available_tensors: HashMap<Vec<usize>, VecDeque<ArrayD<F>>>,
    /// Memory usage tracking
    usage: Arc<Mutex<MemoryUsage>>,
    /// Maximum pool size in bytes
    max_pool_size: usize,
    /// Current pool size in bytes
    current_pool_size: usize,
impl<F: Float + Debug + Clone + 'static> MemoryPool<F> {
    /// Create a new memory pool
    pub fn new(_max_pool_sizemb: usize) -> Self {
            available_tensors: HashMap::new(),
            usage: Arc::new(Mutex::new(MemoryUsage::new())),
            max_pool_size: _max_pool_size_mb * 1024 * 1024,
            current_pool_size: 0,
    /// Allocate or reuse a tensor with the given shape
    pub fn allocate(&mut self, shape: &[usize]) -> ArrayD<F> {
        let shape_vec = shape.to_vec();
        // Try to reuse an existing tensor
        if let Some(tensors) = self.available_tensors.get_mut(&shape_vec) {
            if let Some(mut tensor) = tensors.pop_front() {
                // Zero out the tensor for reuse
                tensor.fill(F::zero());
                // Update memory usage
                if let Ok(mut usage) = self.usage.lock() {
                    let bytes = Self::calculate_bytes(&shape_vec);
                    usage.allocate(bytes);
                }
                return tensor;
            }
        // Create a new tensor if none available
        let tensor = Array::zeros(IxDyn(shape));
        // Update memory usage
        if let Ok(mut usage) = self.usage.lock() {
            let bytes = Self::calculate_bytes(&shape_vec);
            usage.allocate(bytes);
        tensor
    /// Return a tensor to the pool for reuse
    pub fn deallocate(&mut self, tensor: ArrayD<F>) {
        let shape = tensor.shape().to_vec();
        let bytes = Self::calculate_bytes(&shape);
        // Check if we have space in the pool
        if self.current_pool_size + bytes <= self.max_pool_size {
            self.available_tensors
                .entry(shape)
                .or_default()
                .push_back(tensor);
            self.current_pool_size += bytes;
            usage.deallocate(bytes);
    /// Get current memory usage
    pub fn get_usage(&self) -> MemoryUsage {
        self.usage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    /// Clear the memory pool
    pub fn clear(&mut self) {
        self.available_tensors.clear();
        self.current_pool_size = 0;
    /// Calculate memory usage for a tensor shape (assuming F is f32/f64)
    fn calculate_bytes(shape: &[usize]) -> usize {
        let elements: usize = shape.iter().product();
        elements * std::mem::size_of::<F>()
    /// Get pool statistics
    pub fn get_pool_stats(&self) -> PoolStatistics {
        let total_tensors: usize = self.available_tensors.values().map(|v| v.len()).sum();
        let uniqueshapes = self.available_tensors.len();
        PoolStatistics {
            total_cached_tensors: total_tensors,
            uniqueshapes,
            current_pool_size_mb: self.current_pool_size as f64 / (1024.0 * 1024.0),
            max_pool_size_mb: self.max_pool_size as f64 / (1024.0 * 1024.0),
/// Statistics about the memory pool
pub struct PoolStatistics {
    /// Number of tensors currently cached
    pub total_cached_tensors: usize,
    /// Number of unique tensor shapes in the pool
    pub uniqueshapes: usize,
    /// Current pool size in MB
    pub current_pool_size_mb: f64,
    /// Maximum pool size in MB
    pub max_pool_size_mb: f64,
/// Gradient checkpointing implementation for memory-efficient training
pub struct GradientCheckpointing<F: Float + Debug> {
    /// Checkpoint layers - only these will store activations
    checkpoint_layers: Vec<String>,
    /// Stored activations at checkpoint layers
    checkpoints: HashMap<String, ArrayD<F>>,
    /// Memory usage threshold for automatic checkpointing
    memory_threshold_mb: f64,
    /// Current memory usage tracker
    memory_usage: Arc<RwLock<MemoryUsage>>,
impl<F: Float + Debug + Clone + 'static + scirs2_core::ndarray::ScalarOperand> GradientCheckpointing<F> {
    /// Create a new gradient checkpointing manager
    pub fn new(_memory_thresholdmb: f64) -> Self {
            checkpoint_layers: Vec::new(),
            checkpoints: HashMap::new(),
            memory_threshold_mb,
            memory_usage: Arc::new(RwLock::new(MemoryUsage::new())),
    /// Add a layer as a checkpoint point
    pub fn add_checkpoint_layer(&mut self, layername: String) {
        self.checkpoint_layers.push(layer_name);
    /// Store activation at a checkpoint
    pub fn store_checkpoint(&mut self, layername: &str, activation: ArrayD<F>) -> Result<()> {
        if self.checkpoint_layers.contains(&layer_name.to_string()) {
            // Calculate memory usage
            let bytes = activation.len() * std::mem::size_of::<F>();
            if let Ok(mut usage) = self.memory_usage.write() {
                usage.allocate(bytes);
                // Check if we're exceeding memory threshold
                if usage.current_mb() > self.memory_threshold_mb {
                    return Err(NeuralError::ComputationError(format!(
                        "Memory threshold exceeded: {:.2}MB > {:.2}MB",
                        usage.current_mb(),
                        self.memory_threshold_mb
                    )));
            self.checkpoints.insert(layer_name.to_string(), activation);
        Ok(())
    /// Retrieve activation from checkpoint
    pub fn get_checkpoint(&self, layername: &str) -> Option<&ArrayD<F>> {
        self.checkpoints.get(layer_name)
    /// Clear checkpoints to free memory
    pub fn clear_checkpoints(&mut self) {
        let total_bytes: usize = self
            .checkpoints
            .values()
            .map(|arr| arr.len() * std::mem::size_of::<F>())
            .sum();
        self.checkpoints.clear();
        if let Ok(mut usage) = self.memory_usage.write() {
            usage.deallocate(total_bytes);
    pub fn get_memory_usage(&self) -> MemoryUsage {
        self.memory_usage
            .read()
            .map(|usage| usage.clone())
            .unwrap_or_default()
    /// Recompute forward pass from last checkpoint
    pub fn recompute_from_checkpoint<L>(
        &self,
        layers: &[L],
        start_layer: &str, _target_layer: &str, _input: &ArrayD<F>,
    ) -> Result<ArrayD<F>>
    where
        L: Layer<F>,
    {
        // Find the checkpoint closest to target_layer
        let checkpoint_activation = self.get_checkpoint(start_layer).ok_or_else(|| {
            NeuralError::ComputationError(format!("No checkpoint found for layer: {}", start_layer))
        })?;
        // Recompute forward pass from checkpoint to target
        let mut current_activation = checkpoint_activation.clone();
        // This is a simplified implementation
        // In practice, you'd need layer ordering and proper forward pass logic
        for layer in layers {
            current_activation = layer.forward(&current_activation)?;
        Ok(current_activation)
/// In-place operations manager for minimizing memory allocations
pub struct InPlaceOperations;
impl InPlaceOperations {
    /// In-place ReLU activation
    pub fn relu_inplace<F: Float + Debug>(array: &mut ArrayD<F>) {
        array.mapv_inplace(|x| x.max(F::zero()));
    /// In-place sigmoid activation
    pub fn sigmoid_inplace<F: Float + Debug>(array: &mut ArrayD<F>) {
        array.mapv_inplace(|x| F::one() / (F::one() + (-x).exp()));
    /// In-place tanh activation
    pub fn tanh_inplace<F: Float + Debug>(array: &mut ArrayD<F>) {
        array.mapv_inplace(|x| x.tanh());
    /// In-place addition
    pub fn add_inplace<F: Float + Debug>(target: &mut ArrayD<F>, source: &ArrayD<F>) -> Result<()> {
        if target.shape() != source.shape() {
            return Err(NeuralError::ShapeMismatch(
                "Arrays must have the same shape for in-place addition".to_string(),
            ));
        for (t, &s) in target.iter_mut().zip(source.iter()) {
            *t = *t + s;
    /// In-place scalar multiplication
    pub fn scale_inplace<F: Float + Debug>(array: &mut ArrayD<F>, factor: F) {
        array.mapv_inplace(|x| x * factor);
    /// In-place normalization (subtract mean, divide by std)
    pub fn normalize_inplace<F: Float + Debug + Clone + scirs2_core::numeric::FromPrimitive>(
        array: &mut ArrayD<F>,
    ) -> Result<()> {
        let mean = array.mean().unwrap_or(F::zero());
        let variance = _array
            .mapv(|x| (x - mean) * (x - mean))
            .mean()
            .unwrap_or(F::zero());
        let std_dev = variance.sqrt();
        if std_dev == F::zero() {
            return Ok(()); // Avoid division by zero
        array.mapv_inplace(|x| (x - mean) / std_dev);
    /// In-place dropout (sets elements to zero based on probability)
    pub fn dropout_inplace<F: Float + Debug>(
        dropout_rate: f64,
        training: bool,
        if !training {
            return Ok(());
        let keep_prob = 1.0 - dropout_rate;
        let scale_factor = F::from(1.0 / keep_prob).unwrap();
        for element in array.iter_mut() {
            if scirs2_core::random::random::<f64>() < dropout_rate {
                *element = F::zero();
            } else {
                *element = *element * scale_factor;
/// Memory-aware batch processor for handling large datasets
pub struct MemoryAwareBatchProcessor<F: Float + Debug> {
    /// Maximum batch size based on available memory
    max_batch_size: usize,
    /// Memory pool for tensor reuse
    memory_pool: MemoryPool<F>,
    /// Current memory usage threshold
impl<F: Float + Debug + Clone + 'static> MemoryAwareBatchProcessor<F> {
    /// Create a new memory-aware batch processor
    pub fn new(_max_memory_mb: usize, memory_threshold_mb: f64, pool_sizemb: usize) -> Self {
            max_batch_size: Self::calculate_max_batch_size(_max_memory_mb),
            memory_pool: MemoryPool::new(pool_size_mb),
    /// Process batches with automatic size adjustment based on memory usage
    pub fn process_batches<ProcessFn>(
        &mut self,
        input: &ArrayD<F>,
        mut process_fn: ProcessFn,
    ) -> Result<Vec<ArrayD<F>>>
        ProcessFn: FnMut(&ArrayView<F, IxDyn>) -> Result<ArrayD<F>>,
        let total_samples = input.shape()[0];
        let mut results = Vec::new();
        let mut start_idx = 0;
        while start_idx < total_samples {
            // Determine batch size based on current memory usage
            let current_usage = self.memory_pool.get_usage();
            let available_memory_mb = self.memory_threshold_mb - current_usage.current_mb();
            let batch_size = if available_memory_mb < 100.0 {
                // Low memory - use smaller batches
                (self.max_batch_size / 4).max(1)
            } else if available_memory_mb < 200.0 {
                // Medium memory - use half batch size
                self.max_batch_size / 2
                // Plenty of memory - use full batch size
                self.max_batch_size
            };
            let end_idx = (start_idx + batch_size).min(total_samples);
            let batch = input.slice(scirs2_core::ndarray::s![start_idx..end_idx, ..]).into_dyn();
            // Process the batch
            let result = process_fn(&batch)?;
            results.push(result);
            start_idx = end_idx;
            // Optional: Force garbage collection if memory usage is high
            if current_usage.current_mb() > self.memory_threshold_mb * 0.8 {
                self.memory_pool.clear();
        Ok(results)
    /// Calculate maximum batch size based on available memory
    fn calculate_max_batch_size(_max_memorymb: usize) -> usize {
        // Heuristic: assume each sample uses ~1KB on average
        let max_memory_bytes = _max_memory_mb * 1024 * 1024;
        let bytes_per_sample = 1024; // 1KB per sample estimate
        (max_memory_bytes / bytes_per_sample).max(1)
    /// Get current batch processor statistics
    pub fn get_stats(&self) -> BatchProcessorStats {
        let usage = self.memory_pool.get_usage();
        let pool_stats = self.memory_pool.get_pool_stats();
        BatchProcessorStats {
            max_batch_size: self.max_batch_size,
            current_memory_mb: usage.current_mb(),
            peak_memory_mb: usage.peak_mb(),
            memory_threshold_mb: self.memory_threshold_mb,
            pool_stats,
/// Statistics for the batch processor
pub struct BatchProcessorStats {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Current memory usage in MB
    pub current_memory_mb: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Memory threshold in MB
    pub memory_threshold_mb: f64,
    /// Memory pool statistics
    pub pool_stats: PoolStatistics,
/// Memory-efficient neural network layer that processes data in chunks
pub struct MemoryEfficientLayer {
    /// Weight matrix stored in memory-efficient format
    #[cfg(feature = "memory_efficient")]
    #[allow(dead_code)]
    weights: ArrayD<f32>,
    /// Bias vector
    bias: scirs2_core::ndarray::Array1<f32>,
    /// Chunk size for processing
    chunk_size: usize,
    // Memory management handled through BufferPool
    /// Buffer pool for temporary allocations
    #[cfg(feature = "memory_management")]
    buffer_pool: Arc<BufferPool>,
    /// Cache for activations (useful during training)
    #[cfg(feature = "cache")]
    activation_cache: TTLSizedCache<String, ArrayD<f32>>,
impl MemoryEfficientLayer {
    /// Create a new memory-efficient layer
    pub fn new(_input_size: usize, output_size: usize, chunksize: Option<usize>) -> Result<Self> {
        let _weightsshape = [input_size, output_size];
        let default_chunk_size = chunk_size.unwrap_or(1024);
        #[cfg(feature = "memory_efficient")]
        let weights = ArrayD::zeros(IxDyn(&_weightsshape));
        let bias = scirs2_core::ndarray::Array1::zeros(output_size);
        // Buffer pool handles memory management
        #[cfg(feature = "memory_management")]
        let buffer_pool = Arc::new(
            BufferPool::new(
                1000,                             // pool_size
                default_chunk_size * output_size, // buffer_size
                false,                            // numa_aware
                64,                               // alignment
            )
            .unwrap(),
        );
        #[cfg(feature = "cache")]
        let activation_cache = CacheBuilder::new()
            .with_size(100)
            .with_ttl(300)
            .build_sized_cache();
        Ok(Self {
            #[cfg(feature = "memory_efficient")]
            weights,
            bias,
            chunk_size: default_chunk_size,
            #[cfg(feature = "memory_management")]
            buffer_pool,
            #[cfg(feature = "cache")]
            activation_cache,
        })
    /// Forward pass with memory-efficient chunk processing
    pub fn forward(&self, input: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let inputshape = input.shape();
        let batch_size = inputshape[0];
        let _input_size = inputshape[1];
        let output_size = self.bias.len();
        // Create output array
        let mut output = Array::zeros((batch_size, output_size));
        // Process in chunks to minimize memory usage
        let chunks = batch_size.div_ceil(self.chunk_size);
        for chunk_idx in 0..chunks {
            let start_idx = chunk_idx * self.chunk_size;
            let end_idx = std::cmp::min(start_idx + self.chunk_size, batch_size);
            let _chunk_batch_size = end_idx - start_idx;
            // Extract input chunk
            let input_chunk = input.slice(scirs2_core::ndarray::s![start_idx..end_idx, ..]);
            // Compute matrix multiplication for this chunk
            let chunk_output = self.forward_chunk(&input_chunk.into_dyn())?;
            #[cfg(not(feature = "memory_efficient"))]
            let chunk_output = self.forward_chunk_fallback(&input_chunk.into_dyn())?;
            // Copy result to output array
            output
                .slice_mut(scirs2_core::ndarray::s![start_idx..end_idx, ..])
                .assign(&chunk_output);
        Ok(output.into_dyn())
    /// Memory-efficient forward pass for a single chunk
    fn forward_chunk(&self, inputchunk: &ArrayView<f32, IxDyn>) -> Result<scirs2_core::ndarray::Array2<f32>> {
        let chunkshape = input_chunk.shape();
        let chunk_batch_size = chunkshape[0];
        // Use chunk-wise operation for memory efficiency
        {
            let weights = &self.weights;
            let bias = &self.bias;
            let result = chunk_wise_op(
                &input_chunk.to_owned(),
                |chunk| {
                    // Perform matrix multiplication and add bias
                    let output = chunk.dot(weights) + bias;
                    Ok(output)
                },
                ChunkingStrategy::Fixed(1024), // Processing chunk size
            .map_err(|e| {
                NeuralError::ComputationError(format!("Chunk-wise operation failed: {:?}", e))
            })?;
            result
        #[cfg(not(feature = "memory_efficient"))]
            // Fallback - simple matrix multiplication
            input_chunk.to_owned()
        // Add bias
        let mut output = scirs2_core::ndarray::Array2::zeros((chunk_batch_size, output_size));
        for (mut row, bias_val) in output.rows_mut().into_iter().zip(self.bias.iter().cycle()) {
            for (out_val, result_val) in row.iter_mut().zip(result.iter()) {
                *out_val = result_val + bias_val;
        Ok(output)
    /// Fallback implementation when memory_efficient feature is not available
    #[cfg(not(feature = "memory_efficient"))]
    fn forward_chunk_fallback(
        input_chunk: &ArrayView<f32, IxDyn>,
    ) -> Result<scirs2_core::ndarray::Array2<f32>> {
        // Simple fallback using regular ndarray operations
        let input_2d = input_chunk
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>(), NeuralError::DimensionMismatch(format!("Failed to convert to 2D: {}", e))
        // For fallback, create a simple weight matrix
        let (_chunk_batch_size, input_size) = input_2d.dim();
        let weights_2d = scirs2_core::ndarray::Array2::<f32>::zeros((input_size, output_size));
        // Use ndarray's built-in dot operation for matrix multiplication
        let mut result = input_2d.dot(&weights_2d);
        for mut row in result.rows_mut() {
            for (out_val, bias_val) in row.iter_mut().zip(self.bias.iter()) {
                *out_val += bias_val;
        Ok(result)
    // Memory statistics are tracked through BufferPool instead
    /// Cache activation for reuse during training
    pub fn cache_activation(&mut self, key: String, activation: ArrayD<f32>) {
        self.activation_cache.insert(key, activation);
    /// Retrieve cached activation
    pub fn get_cached_activation(&mut self, key: &str) -> Option<ArrayD<f32>> {
        self.activation_cache.get(&key.to_string())
/// Processor for chunk-wise forward operations
#[allow(dead_code)]
struct ChunkForwardProcessor<'a> {
    weights: &'a ArrayD<f32>,
    bias: &'a scirs2_core::ndarray::Array1<f32>,
// ChunkForwardProcessor functionality is now handled directly in the chunk_wise_op closure
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::<f32>::new(10); // 10MB max
        // Allocate a tensor
        let tensor1 = pool.allocate(&[100, 100]);
        assert_eq!(tensor1.shape(), [100, 100]);
        // Return it to the pool
        pool.deallocate(tensor1);
        // Allocate again - should reuse
        let tensor2 = pool.allocate(&[100, 100]);
        assert_eq!(tensor2.shape(), [100, 100]);
        let stats = pool.get_pool_stats();
        assert_eq!(stats.uniqueshapes, 1);
    fn test_gradient_checkpointing() {
        let mut checkpointing = GradientCheckpointing::<f64>::new(100.0); // 100MB threshold
        checkpointing.add_checkpoint_layer("layer1".to_string());
        let activation = Array2::from_elem((10, 10), 1.0).into_dyn();
        checkpointing
            .store_checkpoint("layer1", activation)
            .unwrap();
        assert!(checkpointing.get_checkpoint("layer1").is_some());
        checkpointing.clear_checkpoints();
        assert!(checkpointing.get_checkpoint("layer1").is_none());
    fn test_in_place_operations() {
        let mut array = Array2::from_elem((3, 3), -1.0).into_dyn();
        // Test in-place ReLU
        InPlaceOperations::relu_inplace(&mut array);
        for &val in array.iter() {
            assert!(val >= 0.0);
        // Test in-place scaling
        InPlaceOperations::scale_inplace(&mut array, 2.0);
            assert_eq!(val, 0.0); // Was negative, became 0 after ReLU, then scaled
    fn test_memory_aware_batch_processor() {
        let mut processor = MemoryAwareBatchProcessor::<f32>::new(100, 50.0, 10);
        let input = Array2::from_elem((20, 5), 1.0).into_dyn();
        let results = processor
            .process_batches(&input, |batch| Ok(batch.to_owned()))
        assert!(!results.is_empty());
        let stats = processor.get_stats();
        assert!(stats.max_batch_size > 0);
    fn test_memory_usage_tracking() {
        let mut usage = MemoryUsage::new();
        usage.allocate(1024 * 1024); // 1MB
        assert_eq!(usage.current_mb(), 1.0);
        assert_eq!(usage.peak_mb(), 1.0);
        assert_eq!(usage.active_allocations, 1);
        usage.allocate(2 * 1024 * 1024); // 2MB more
        assert_eq!(usage.current_mb(), 3.0);
        assert_eq!(usage.peak_mb(), 3.0);
        assert_eq!(usage.active_allocations, 2);
        usage.deallocate(1024 * 1024); // Release 1MB
        assert_eq!(usage.current_mb(), 2.0);
        assert_eq!(usage.peak_mb(), 3.0); // Peak should remain
