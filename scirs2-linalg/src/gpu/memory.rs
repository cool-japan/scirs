//! GPU memory management utilities

use super::{GpuBuffer, GpuContext, GpuContextAlloc};
use crate::error::LinalgResult;

/// Memory allocation strategies for GPU operations
#[derive(Debug, Clone, Copy)]
pub enum MemoryStrategy {
    /// Allocate memory as needed
    OnDemand,
    /// Pre-allocate memory pools
    Pooled,
    /// Use unified/managed memory where available
    Unified,
    /// Use pinned host memory for faster transfers
    Pinned,
}

/// Memory pool for reusing GPU allocations
pub struct MemoryPool<T> {
    buffers: Vec<Box<dyn GpuBuffer<T>>>,
    max_poolsize: usize,
    total_allocated: usize,
}

impl<T: Clone + Send + Sync + Copy + 'static + std::fmt::Debug> MemoryPool<T> {
    /// Create a new memory pool
    pub fn new(max_poolsize: usize) -> Self {
        Self {
            buffers: Vec::new(),
            max_poolsize,
            total_allocated: 0,
        }
    }

    /// Get a buffer from the pool or allocate a new one
    pub fn get_buffer<C: GpuContextAlloc>(
        &mut self,
        context: &C,
        size: usize,
    ) -> LinalgResult<Box<dyn GpuBuffer<T>>> {
        // Try to find a buffer of suitable size in the pool
        for i in 0..self.buffers.len() {
            if self.buffers[i].len() >= size {
                return Ok(self.buffers.swap_remove(i));
            }
        }

        // No suitable buffer found, allocate a new one
        let buffer = context.allocate_buffer(size)?;
        self.total_allocated += size;
        Ok(buffer)
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer: Box<dyn GpuBuffer<T>>) {
        if self.buffers.len() < self.max_poolsize {
            self.buffers.push(buffer);
        }
        // If pool is full, buffer will be dropped
    }

    /// Clear all buffers from the pool
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.total_allocated = 0;
    }

    /// Get total allocated memory in elements
    pub fn total_allocated(&self) -> usize {
        self.total_allocated
    }

    /// Get number of buffers in pool
    pub fn poolsize(&self) -> usize {
        self.buffers.len()
    }
}

/// Memory transfer operations between host and device
pub trait MemoryTransfer<T> {
    /// Copy data from host to device asynchronously
    fn copy_host_to_device_async(
        &self,
        host_data: &[T],
        device_buffer: &mut dyn GpuBuffer<T>,
    ) -> LinalgResult<()>;

    /// Copy data from device to host asynchronously
    fn copy_device_to_host_async(
        &self,
        device_buffer: &dyn GpuBuffer<T>,
        host_data: &mut [T],
    ) -> LinalgResult<()>;

    /// Copy data between device buffers
    fn copy_device_to_device(
        &self,
        src_buffer: &dyn GpuBuffer<T>,
        dst_buffer: &mut dyn GpuBuffer<T>,
    ) -> LinalgResult<()>;
}

/// Memory bandwidth measurement and optimization
pub struct MemoryBandwidthProfiler {
    measurements: Vec<f64>, // GB/s measurements
}

impl Default for MemoryBandwidthProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBandwidthProfiler {
    /// Create a new bandwidth profiler
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    /// Measure memory bandwidth for a given transfer size
    pub fn measure_bandwidth<T, C: GpuContextAlloc>(
        &mut self,
        context: &C,
        transfersize: usize,
    ) -> LinalgResult<f64>
    where
        T: Clone + Send + Sync + Default + Copy + 'static + std::fmt::Debug,
    {
        let _start_time = std::time::Instant::now();

        // Allocate buffers
        let mut buffer1 = context.allocate_buffer::<T>(transfersize)?;
        let _buffer2 = context.allocate_buffer::<T>(transfersize)?;

        // Create test data
        let test_data: Vec<T> = (0..transfersize).map(|_| T::default()).collect();

        // Measure host-to-device transfer
        let h2d_start = std::time::Instant::now();
        buffer1.copy_from_host(&test_data)?;
        context.synchronize()?;
        let h2d_time = h2d_start.elapsed().as_secs_f64();

        // Measure device-to-host transfer
        let mut result_data = vec![T::default(); transfersize];
        let d2h_start = std::time::Instant::now();
        buffer1.copy_to_host(&mut result_data)?;
        context.synchronize()?;
        let d2h_time = d2h_start.elapsed().as_secs_f64();

        // Calculate bandwidth (bytes per second -> GB/s)
        let bytes_transferred = transfersize * std::mem::size_of::<T>();
        let total_time = h2d_time + d2h_time;
        let bandwidth_gb_s = (bytes_transferred as f64 * 2.0) / (total_time * 1e9);

        self.measurements.push(bandwidth_gb_s);
        Ok(bandwidth_gb_s)
    }

    /// Get average measured bandwidth
    pub fn average_bandwidth(&self) -> f64 {
        if self.measurements.is_empty() {
            0.0
        } else {
            self.measurements.iter().sum::<f64>() / self.measurements.len() as f64
        }
    }

    /// Get peak measured bandwidth
    pub fn peak_bandwidth(&self) -> f64 {
        self.measurements.iter().copied().fold(0.0, f64::max)
    }
}

/// Memory usage tracking and optimization suggestions
pub struct MemoryOptimizer {
    usage_history: Vec<usize>,
    peak_usage: usize,
}

impl Default for MemoryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new() -> Self {
        Self {
            usage_history: Vec::new(),
            peak_usage: 0,
        }
    }

    /// Record memory usage
    pub fn record_usage(&mut self, usagebytes: usize) {
        self.usage_history.push(usagebytes);
        self.peak_usage = self.peak_usage.max(usagebytes);
    }

    /// Get optimization suggestions based on usage patterns
    pub fn get_suggestions(&self, devicememory: usize) -> Vec<String> {
        let mut suggestions = Vec::new();

        if self.peak_usage > devicememory / 2 {
            suggestions
                .push("Consider using _memory pooling to reduce allocation overhead".to_string());
        }

        if self.usage_history.len() > 10 {
            let recent_usage: Vec<_> = self.usage_history.iter().rev().take(10).collect();
            let avg_recent = recent_usage.iter().copied().sum::<usize>() / recent_usage.len();

            if avg_recent < self.peak_usage / 4 {
                suggestions.push(
                    "Memory usage varies significantly - consider dynamic allocation".to_string(),
                );
            }
        }

        if self.peak_usage > devicememory * 3 / 4 {
            suggestions
                .push("High _memory usage detected - consider out-of-core algorithms".to_string());
        }

        suggestions
    }

    /// Calculate memory efficiency score (0-100)
    pub fn efficiency_score(&self, devicememory: usize) -> f64 {
        if self.usage_history.is_empty() {
            return 100.0;
        }

        let avg_usage = self.usage_history.iter().sum::<usize>() / self.usage_history.len();
        let utilization = avg_usage as f64 / devicememory as f64;

        // Score based on reasonable utilization (30-70% is good)
        if utilization < 0.3 {
            (utilization / 0.3) * 50.0
        } else if utilization <= 0.7 {
            100.0
        } else if utilization <= 1.0 {
            100.0 - ((utilization - 0.7) / 0.3) * 50.0
        } else {
            0.0 // Over 100% utilization
        }
    }
}

/// Check if a given operation fits in GPU memory
#[allow(dead_code)]
pub fn check_memory_requirements(
    context: &dyn GpuContext,
    matricessizes: &[(usize, usize)],
    elementsize: usize,
) -> LinalgResult<bool> {
    let total_elements: usize = matricessizes.iter().map(|(rows, cols)| rows * cols).sum();

    let total_bytes = total_elements * elementsize;
    let available_memory = context.available_memory()?;

    // Need some overhead for temporary variables
    Ok(total_bytes < available_memory / 2)
}

/// Suggest optimal memory strategy for given problem characteristics
#[allow(dead_code)]
pub fn suggest_memory_strategy(
    problemsize: usize,
    available_memory: usize,
    unified_memory_available: bool,
) -> MemoryStrategy {
    let memory_ratio = problemsize as f64 / available_memory as f64;

    if memory_ratio > 0.8 {
        // Large problem relative to _memory
        if unified_memory_available {
            MemoryStrategy::Unified
        } else {
            MemoryStrategy::OnDemand
        }
    } else if memory_ratio < 0.1 {
        // Small problem
        MemoryStrategy::Pinned
    } else {
        // Medium problem
        MemoryStrategy::Pooled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::backends::CpuFallbackBackend;
    use crate::gpu::GpuBackend;

    #[test]
    fn test_memory_pool_operations() {
        let mut pool = MemoryPool::<f32>::new(5);
        assert_eq!(pool.poolsize(), 0);
        assert_eq!(pool.total_allocated(), 0);

        // Can't test actual allocation without a real GPU context
        pool.clear();
        assert_eq!(pool.poolsize(), 0);
    }

    #[test]
    fn test_memory_optimizer() {
        let mut optimizer = MemoryOptimizer::new();

        // Record some usage patterns that will trigger suggestions
        optimizer.record_usage(3000);
        optimizer.record_usage(6000); // This will be > 5000 (10000/2)
        optimizer.record_usage(4500);

        let efficiency = optimizer.efficiency_score(10000);
        assert!((0.0..=100.0).contains(&efficiency));

        let suggestions = optimizer.get_suggestions(10000);
        // Should have suggestions since peak_usage (6000) > device_memory/2 (5000)
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_memory_strategy_suggestions() {
        // Large problem
        let strategy = suggest_memory_strategy(1000000, 1000000, true);
        assert!(matches!(strategy, MemoryStrategy::Unified));

        // Small problem
        let strategy = suggest_memory_strategy(10000, 1000000, false);
        assert!(matches!(strategy, MemoryStrategy::Pinned));

        // Medium problem
        let strategy = suggest_memory_strategy(300000, 1000000, false);
        assert!(matches!(strategy, MemoryStrategy::Pooled));
    }

    #[test]
    fn test_check_memory_requirements() {
        let backend = CpuFallbackBackend::new();
        let context = backend.create_context(0).unwrap();

        // Small matrices should fit
        let matrices = vec![(10, 10), (10, 10)];
        let fits = check_memory_requirements(context.as_ref(), &matrices, 8).unwrap();
        assert!(fits);

        // Very large matrices might not fit (depends on system memory)
        let matrices = vec![(100000, 100000)];
        let fits = check_memory_requirements(context.as_ref(), &matrices, 8).unwrap();
        // Result depends on available system memory, just check it doesn't panic
        let _ = fits;
    }
}
