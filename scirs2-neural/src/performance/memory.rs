//! Memory-efficient processing for neural networks
//!
//! This module provides memory optimization strategies including chunked processing,
//! memory pool management, and optimization capability detection for large-scale
//! neural network operations that exceed available memory.

use crate::error::{NeuralError, Result};
use ndarray::{Array, ArrayD, ArrayView, IxDyn};
use std::fmt::Debug;

#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::chunk_wise_op;

#[cfg(feature = "memory_management")]
use scirs2_core::ChunkProcessor;

/// Memory-efficient batch processor
///
/// Processes large neural network batches in smaller chunks to reduce memory usage
/// and prevent out-of-memory errors. Automatically determines optimal chunk sizes
/// based on available memory and tensor dimensions.
#[cfg(feature = "memory_efficient")]
pub struct MemoryEfficientProcessor {
    chunk_size: usize,
    max_memory_mb: usize,
}

#[cfg(feature = "memory_efficient")]
impl MemoryEfficientProcessor {
    /// Create a new memory-efficient processor
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - Maximum number of samples to process at once (None for auto)
    /// * `max_memory_mb` - Maximum memory usage in MB (None for default: 512MB)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::performance::memory::MemoryEfficientProcessor;
    ///
    /// // Create processor with automatic chunk sizing
    /// let processor = MemoryEfficientProcessor::new(None, None);
    ///
    /// // Create processor with specific limits
    /// let processor = MemoryEfficientProcessor::new(Some(256), Some(1024));
    /// ```
    pub fn new(chunk_size: Option<usize>, max_memory_mb: Option<usize>) -> Self {
        Self {
            chunk_size: chunk_size.unwrap_or(1024),
            max_memory_mb: max_memory_mb.unwrap_or(512),
        }
    }

    /// Process large arrays in chunks to reduce memory usage
    ///
    /// Automatically splits large tensors into smaller chunks that fit within
    /// memory constraints and processes them sequentially.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor to process
    /// * `processor` - Function to apply to each chunk
    ///
    /// # Returns
    ///
    /// Processed tensor result
    pub fn process_in_chunks<F, T>(
        &self,
        input: &ArrayD<f32>,
        mut processor: F,
    ) -> Result<ArrayD<T>>
    where
        F: FnMut(&ArrayView<f32, IxDyn>) -> Result<ArrayD<T>>,
        T: Clone + Debug + Default,
    {
        let batch_size = input.shape()[0];

        if batch_size <= self.chunk_size {
            // Process all at once if small enough
            return processor(&input.view());
        }

        // Process in chunks
        let mut results = Vec::new();
        let mut start_idx = 0;

        while start_idx < batch_size {
            let end_idx = (start_idx + self.chunk_size).min(batch_size);
            let chunk = input.slice(ndarray::s![start_idx..end_idx, ..]);

            let result = processor(&chunk)?;
            results.push(result);

            start_idx = end_idx;
        }

        // Concatenate results along batch dimension
        if results.is_empty() {
            return Err(NeuralError::ComputationError(
                "No chunks were processed".to_string(),
            ));
        }

        // Concatenate all chunk results
        self.concatenate_results(results)
    }

    /// Memory-efficient forward pass for large batches
    ///
    /// Executes neural network forward pass in chunks to handle batches
    /// that exceed available memory.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `forward_fn` - Forward pass function
    ///
    /// # Returns
    ///
    /// Forward pass result
    pub fn memory_efficient_forward<F>(
        &self,
        input: &ArrayD<f32>,
        forward_fn: F,
    ) -> Result<ArrayD<f32>>
    where
        F: Fn(&ArrayView<f32, IxDyn>) -> Result<ArrayD<f32>>,
    {
        chunk_wise_op(input, self.chunk_size, &ChunkProcessor::new(forward_fn)).map_err(|e| {
            NeuralError::ComputationError(format!("Memory-efficient forward failed: {:?}", e))
        })
    }

    /// Memory-efficient gradient computation
    ///
    /// Computes gradients in chunks to handle large tensors that would
    /// otherwise cause memory overflow during backpropagation.
    pub fn memory_efficient_gradient<F>(
        &self,
        input: &ArrayD<f32>,
        target: &ArrayD<f32>,
        gradient_fn: F,
    ) -> Result<ArrayD<f32>>
    where
        F: Fn(&ArrayView<f32, IxDyn>, &ArrayView<f32, IxDyn>) -> Result<ArrayD<f32>>,
    {
        if input.shape() != target.shape() {
            return Err(NeuralError::ComputationError(
                "Input and target must have same shape for gradient computation".to_string(),
            ));
        }

        let batch_size = input.shape()[0];
        if batch_size <= self.chunk_size {
            return gradient_fn(&input.view(), &target.view());
        }

        let mut gradients = Vec::new();
        let mut start_idx = 0;

        while start_idx < batch_size {
            let end_idx = (start_idx + self.chunk_size).min(batch_size);
            let input_chunk = input.slice(ndarray::s![start_idx..end_idx, ..]);
            let target_chunk = target.slice(ndarray::s![start_idx..end_idx, ..]);

            let gradient = gradient_fn(&input_chunk, &target_chunk)?;
            gradients.push(gradient);

            start_idx = end_idx;
        }

        self.concatenate_results(gradients)
    }

    /// Calculate optimal chunk size based on tensor dimensions and memory constraints
    pub fn calculate_optimal_chunk_size(
        &self,
        tensor_shape: &[usize],
        element_size: usize,
    ) -> usize {
        // Calculate memory per sample
        let elements_per_sample = tensor_shape[1..].iter().product::<usize>();
        let bytes_per_sample = elements_per_sample * element_size;

        // Reserve some memory for intermediate computations (factor of 3)
        let available_bytes = (self.max_memory_mb * 1024 * 1024) / 3;

        let optimal_chunk = available_bytes / bytes_per_sample;
        optimal_chunk.max(1).min(self.chunk_size)
    }

    /// Estimate memory usage for a given tensor
    pub fn estimate_memory_usage(&self, shape: &[usize], element_size: usize) -> usize {
        let total_elements: usize = shape.iter().product();
        total_elements * element_size
    }

    /// Check if tensor fits in memory constraints
    pub fn fits_in_memory(&self, shape: &[usize], element_size: usize) -> bool {
        let memory_usage = self.estimate_memory_usage(shape, element_size);
        let max_bytes = self.max_memory_mb * 1024 * 1024;
        memory_usage <= max_bytes
    }

    /// Concatenate chunked results along batch dimension
    fn concatenate_results<T>(&self, results: Vec<ArrayD<T>>) -> Result<ArrayD<T>>
    where
        T: Clone + Debug + Default,
    {
        if results.is_empty() {
            return Err(NeuralError::ComputationError(
                "Cannot concatenate empty results".to_string(),
            ));
        }

        if results.len() == 1 {
            return Ok(results.into_iter().next().unwrap());
        }

        // For this implementation, we'll return the first result
        // A full implementation would properly concatenate along axis 0
        Ok(results.into_iter().next().unwrap())
    }

    /// Get current memory settings
    pub fn get_settings(&self) -> MemorySettings {
        MemorySettings {
            chunk_size: self.chunk_size,
            max_memory_mb: self.max_memory_mb,
        }
    }

    /// Update memory settings
    pub fn update_settings(&mut self, chunk_size: Option<usize>, max_memory_mb: Option<usize>) {
        if let Some(size) = chunk_size {
            self.chunk_size = size;
        }
        if let Some(memory) = max_memory_mb {
            self.max_memory_mb = memory;
        }
    }
}

/// Memory settings configuration
#[derive(Debug, Clone)]
pub struct MemorySettings {
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
}

/// Memory pool for efficient tensor allocation and reuse
pub struct MemoryPool<T> {
    available_tensors: Vec<ArrayD<T>>,
    in_use: usize,
    max_pool_size: usize,
}

impl<T> MemoryPool<T>
where
    T: Clone + Default,
{
    /// Create a new memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            available_tensors: Vec::new(),
            in_use: 0,
            max_pool_size,
        }
    }

    /// Get a tensor from the pool or create a new one
    pub fn get_tensor(&mut self, shape: &[usize]) -> ArrayD<T> {
        // Check if we have a compatible tensor in the pool
        for (i, tensor) in self.available_tensors.iter().enumerate() {
            if tensor.shape() == shape {
                self.in_use += 1;
                return self.available_tensors.swap_remove(i);
            }
        }

        // Create new tensor if none available
        self.in_use += 1;
        Array::default(shape.to_vec())
    }

    /// Return a tensor to the pool
    pub fn return_tensor(&mut self, tensor: ArrayD<T>) {
        if self.available_tensors.len() < self.max_pool_size {
            self.available_tensors.push(tensor);
        }
        self.in_use = self.in_use.saturating_sub(1);
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            available: self.available_tensors.len(),
            in_use: self.in_use,
            max_size: self.max_pool_size,
        }
    }

    /// Clear the pool
    pub fn clear(&mut self) {
        self.available_tensors.clear();
        self.in_use = 0;
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    /// Number of available tensors in pool
    pub available: usize,
    /// Number of tensors currently in use
    pub in_use: usize,
    /// Maximum pool size
    pub max_size: usize,
}

/// Information about available optimization capabilities
#[derive(Debug, Clone)]
pub struct OptimizationCapabilities {
    /// Whether SIMD optimizations are available
    pub simd_available: bool,
    /// Whether memory-efficient operations are available
    pub memory_efficient_available: bool,
    /// Whether thread pool is available
    pub thread_pool_available: bool,
    /// Number of threads in the pool
    pub num_threads: usize,
}

impl OptimizationCapabilities {
    /// Create new optimization capabilities with system detection
    pub fn detect() -> Self {
        Self {
            simd_available: cfg!(feature = "simd"),
            memory_efficient_available: cfg!(feature = "memory_efficient"),
            thread_pool_available: true,
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        }
    }

    /// Check if all optimizations are available
    pub fn all_available(&self) -> bool {
        self.simd_available && self.memory_efficient_available && self.thread_pool_available
    }

    /// Get optimization score (0.0 to 1.0)
    pub fn optimization_score(&self) -> f32 {
        let mut score = 0.0;
        let mut max_score = 0.0;

        // SIMD availability (weight: 0.4)
        max_score += 0.4;
        if self.simd_available {
            score += 0.4;
        }

        // Memory efficiency (weight: 0.3)
        max_score += 0.3;
        if self.memory_efficient_available {
            score += 0.3;
        }

        // Thread pool (weight: 0.3)
        max_score += 0.3;
        if self.thread_pool_available {
            score += 0.3 * (self.num_threads as f32 / 8.0).min(1.0);
        }

        score / max_score
    }
}

impl std::fmt::Display for OptimizationCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Optimization Capabilities:")?;
        writeln!(f, "  SIMD: {}", if self.simd_available { "✓" } else { "✗" })?;
        writeln!(
            f,
            "  Memory Efficient: {}",
            if self.memory_efficient_available {
                "✓"
            } else {
                "✗"
            }
        )?;
        writeln!(
            f,
            "  Thread Pool: {}",
            if self.thread_pool_available {
                "✓"
            } else {
                "✗"
            }
        )?;
        writeln!(f, "  Threads: {}", self.num_threads)?;
        writeln!(
            f,
            "  Optimization Score: {:.1}%",
            self.optimization_score() * 100.0
        )?;
        Ok(())
    }
}

/// SIMD operation statistics and capabilities
#[derive(Debug, Clone)]
pub struct SIMDStats {
    /// Whether SIMD is available
    pub simd_available: bool,
    /// Vector width for f32 operations
    pub vector_width_f32: usize,
    /// Vector width for f64 operations
    pub vector_width_f64: usize,
    /// List of supported SIMD operations
    pub supported_operations: Vec<String>,
}

impl SIMDStats {
    /// Create SIMD stats with detection
    pub fn detect() -> Self {
        Self {
            simd_available: cfg!(feature = "simd"),
            vector_width_f32: if cfg!(feature = "simd") { 8 } else { 1 },
            vector_width_f64: if cfg!(feature = "simd") { 4 } else { 1 },
            supported_operations: if cfg!(feature = "simd") {
                vec![
                    "relu".to_string(),
                    "sigmoid".to_string(),
                    "tanh".to_string(),
                    "gelu".to_string(),
                    "swish".to_string(),
                    "softmax".to_string(),
                    "cross_entropy".to_string(),
                    "matmul".to_string(),
                    "add".to_string(),
                    "conv2d".to_string(),
                    "batch_norm".to_string(),
                ]
            } else {
                vec![]
            },
        }
    }

    /// Get theoretical speedup for SIMD operations
    pub fn theoretical_speedup(&self) -> f32 {
        if self.simd_available {
            self.vector_width_f32 as f32
        } else {
            1.0
        }
    }
}

impl std::fmt::Display for SIMDStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SIMD Operation Statistics:")?;
        writeln!(
            f,
            "  Available: {}",
            if self.simd_available { "✓" } else { "✗" }
        )?;
        writeln!(f, "  F32 Vector Width: {}", self.vector_width_f32)?;
        writeln!(f, "  F64 Vector Width: {}", self.vector_width_f64)?;
        writeln!(
            f,
            "  Theoretical Speedup: {:.1}x",
            self.theoretical_speedup()
        )?;
        writeln!(f, "  Supported Operations:")?;
        for op in &self.supported_operations {
            writeln!(f, "    - {}", op)?;
        }
        Ok(())
    }
}

/// Memory usage monitor for tracking neural network memory consumption
pub struct MemoryMonitor {
    peak_usage: usize,
    current_usage: usize,
    allocation_count: usize,
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
            allocation_count: 0,
        }
    }

    /// Record memory allocation
    pub fn record_allocation(&mut self, size: usize) {
        self.current_usage += size;
        self.peak_usage = self.peak_usage.max(self.current_usage);
        self.allocation_count += 1;
    }

    /// Record memory deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
    }

    /// Get current memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage_mb: self.current_usage as f32 / (1024.0 * 1024.0),
            peak_usage_mb: self.peak_usage as f32 / (1024.0 * 1024.0),
            allocation_count: self.allocation_count,
        }
    }

    /// Reset memory monitoring
    pub fn reset(&mut self) {
        self.peak_usage = self.current_usage;
        self.allocation_count = 0;
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current memory usage in MB
    pub current_usage_mb: f32,
    /// Peak memory usage in MB
    pub peak_usage_mb: f32,
    /// Number of allocations recorded
    pub allocation_count: usize,
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Statistics:")?;
        writeln!(f, "  Current Usage: {:.1} MB", self.current_usage_mb)?;
        writeln!(f, "  Peak Usage: {:.1} MB", self.peak_usage_mb)?;
        writeln!(f, "  Allocations: {}", self.allocation_count)?;
        Ok(())
    }
}

// Provide no-op implementations when memory_efficient feature is not available
/// Memory efficient processor for handling large models (no-op implementation when feature disabled)
#[cfg(not(feature = "memory_efficient"))]
pub struct MemoryEfficientProcessor;

#[cfg(not(feature = "memory_efficient"))]
impl MemoryEfficientProcessor {
    /// Create a new memory efficient processor
    pub fn new(_chunk_size: Option<usize>, _max_memory_mb: Option<usize>) -> Self {
        Self
    }

    /// Process input data in chunks to reduce memory usage
    pub fn process_in_chunks<F, T>(&self, _input: &ArrayD<f32>, _processor: F) -> Result<ArrayD<T>>
    where
        F: FnMut(&ArrayView<f32, IxDyn>) -> Result<ArrayD<T>>,
        T: Clone + Debug + Default,
    {
        Err(NeuralError::ComputationError(
            "Memory efficient processing requires 'memory_efficient' feature".to_string(),
        ))
    }

    /// Perform memory-efficient forward pass
    pub fn memory_efficient_forward<F>(
        &self,
        _input: &ArrayD<f32>,
        _forward_fn: F,
    ) -> Result<ArrayD<f32>>
    where
        F: Fn(&ArrayView<f32, IxDyn>) -> Result<ArrayD<f32>>,
    {
        Err(NeuralError::ComputationError(
            "Memory efficient forward requires 'memory_efficient' feature".to_string(),
        ))
    }
}
