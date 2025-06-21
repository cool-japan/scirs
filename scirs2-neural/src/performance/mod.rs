//! Performance optimization utilities for neural networks
//!
//! This module provides comprehensive performance optimizations for neural network operations
//! including SIMD acceleration, memory-efficient processing, and parallel execution capabilities.
//! The module is organized into three focused submodules:
//!
//! - [`simd`] - SIMD-accelerated operations for vectorized computations
//! - [`memory`] - Memory-efficient processing and optimization capabilities
//! - [`threading`] - Thread pool management, profiling, and distributed training
//!
//! # Quick Start
//!
//! ## SIMD Operations
//!
//! ```rust
//! use scirs2_neural::performance::simd::SIMDOperations;
//! use ndarray::Array;
//!
//! let input = Array::ones((1000, 512)).into_dyn();
//! let result = SIMDOperations::simd_relu_f32(&input.view());
//! ```
//!
//! ## Memory-Efficient Processing
//!
//! ```rust
//! use scirs2_neural::performance::memory::MemoryEfficientProcessor;
//!
//! let processor = MemoryEfficientProcessor::new(Some(256), Some(1024));
//! // Process large tensors in manageable chunks
//! ```
//!
//! ## Thread Pool Management
//!
//! ```rust
//! use scirs2_neural::performance::threading::ThreadPoolManager;
//! use ndarray::Array;
//!
//! let a = Array::ones((100, 200)).into_dyn();
//! let b = Array::ones((200, 150)).into_dyn();
//! let pool = ThreadPoolManager::new(Some(8)).unwrap();
//! let result = pool.parallel_matmul(&a, &b).unwrap();
//! assert_eq!(result.shape(), &[100, 150]);
//! ```
//!
//! ## Unified Performance Optimization
//!
//! ```rust
//! use scirs2_neural::performance::PerformanceOptimizer;
//! use ndarray::Array;
//!
//! let a = Array::ones((100, 200)).into_dyn();
//! let b = Array::ones((200, 150)).into_dyn();
//! let mut optimizer = PerformanceOptimizer::new(
//!     Some(256),  // chunk_size
//!     Some(1024), // max_memory_mb
//!     Some(8),    // num_threads
//!     true        // enable_profiling
//! ).unwrap();
//!
//! let result = optimizer.optimized_matmul(&a, &b).unwrap();
//! optimizer.profiler().print_summary();
//! ```

// Re-export all public modules
pub mod memory;
pub mod simd;
pub mod threading;

// Re-export commonly used types and functions
pub use simd::SIMDOperations;

pub use memory::{
    MemoryEfficientProcessor, MemoryMonitor, MemoryPool, MemoryPoolStats, MemorySettings,
    MemoryStats, OptimizationCapabilities, SIMDStats,
};

pub use threading::{
    distributed::{
        CommunicationBackend, DistributedConfig, DistributedManager, DistributedStats,
        DistributedStrategy, GradientSyncMethod, ProcessInfo,
    },
    PerformanceProfiler, ProfilingStats, ThreadPoolManager, ThreadPoolStats,
};

use crate::error::{NeuralError, Result};
use ndarray::ArrayD;
use std::sync::Arc;

/// Unified performance optimization manager
///
/// Combines all performance optimization techniques including SIMD, memory efficiency,
/// and parallel processing to provide optimal performance for neural network operations.
pub struct PerformanceOptimizer {
    #[cfg(feature = "simd")]
    #[allow(dead_code)]
    simd_ops: SIMDOperations,

    #[cfg(feature = "memory_efficient")]
    memory_processor: MemoryEfficientProcessor,

    thread_pool: Arc<ThreadPoolManager>,
    profiler: PerformanceProfiler,
    capabilities: OptimizationCapabilities,
}

impl PerformanceOptimizer {
    /// Create a new performance optimizer
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - Chunk size for memory-efficient processing
    /// * `max_memory_mb` - Maximum memory usage in MB
    /// * `num_threads` - Number of threads for parallel processing
    /// * `enable_profiling` - Whether to enable performance profiling
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_neural::performance::PerformanceOptimizer;
    ///
    /// let optimizer = PerformanceOptimizer::new(
    ///     Some(256),  // 256 samples per chunk
    ///     Some(1024), // 1GB memory limit
    ///     Some(8),    // 8 threads
    ///     true        // enable profiling
    /// ).unwrap();
    /// ```
    pub fn new(
        _chunk_size: Option<usize>,
        _max_memory_mb: Option<usize>,
        num_threads: Option<usize>,
        enable_profiling: bool,
    ) -> Result<Self> {
        let capabilities = OptimizationCapabilities::detect();

        Ok(Self {
            #[cfg(feature = "simd")]
            simd_ops: SIMDOperations,

            #[cfg(feature = "memory_efficient")]
            memory_processor: MemoryEfficientProcessor::new(chunk_size, max_memory_mb),

            thread_pool: Arc::new(ThreadPoolManager::new(num_threads)?),
            profiler: PerformanceProfiler::new(enable_profiling),
            capabilities,
        })
    }

    /// Get reference to thread pool
    pub fn thread_pool(&self) -> &Arc<ThreadPoolManager> {
        &self.thread_pool
    }

    /// Get mutable reference to profiler
    pub fn profiler_mut(&mut self) -> &mut PerformanceProfiler {
        &mut self.profiler
    }

    /// Get reference to profiler
    pub fn profiler(&self) -> &PerformanceProfiler {
        &self.profiler
    }

    /// Get optimization capabilities
    pub fn get_capabilities(&self) -> &OptimizationCapabilities {
        &self.capabilities
    }

    /// Optimized matrix multiplication using all available optimizations
    ///
    /// Automatically selects the best optimization strategy based on matrix size,
    /// available features, and system capabilities.
    pub fn optimized_matmul(&mut self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let timer = self.profiler.start_timer("optimized_matmul");

        let result = {
            #[cfg(feature = "simd")]
            {
                // Try SIMD first if available and matrices are suitable
                if self.is_suitable_for_simd(a, b) {
                    if let Ok(result) = SIMDOperations::simd_matmul_f32(&a.view(), &b.view()) {
                        result
                    } else {
                        // Fallback to parallel matmul
                        self.thread_pool.parallel_matmul(a, b)?
                    }
                } else {
                    // Use parallel matmul for large matrices
                    self.thread_pool.parallel_matmul(a, b)?
                }
            }
            #[cfg(not(feature = "simd"))]
            {
                // Use parallel matmul when SIMD is not available
                self.thread_pool.parallel_matmul(a, b)?
            }
        };

        self.profiler
            .end_timer("optimized_matmul".to_string(), timer);
        Ok(result)
    }

    /// Optimized convolution using all available optimizations
    pub fn optimized_conv2d(
        &mut self,
        input: &ArrayD<f32>,
        kernel: &ArrayD<f32>,
        bias: Option<&[f32]>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<f32>> {
        let timer = self.profiler.start_timer("optimized_conv2d");

        let result = {
            #[cfg(feature = "simd")]
            {
                // Try SIMD convolution if available
                if let Ok(result) = SIMDOperations::simd_conv2d_f32(
                    &input.view(),
                    &kernel.view(),
                    bias,
                    stride,
                    padding,
                ) {
                    result
                } else {
                    // Fallback to parallel convolution
                    self.thread_pool
                        .parallel_conv2d(input, kernel, bias, stride, padding)?
                }
            }
            #[cfg(not(feature = "simd"))]
            {
                // Use parallel convolution when SIMD is not available
                self.thread_pool
                    .parallel_conv2d(input, kernel, bias, stride, padding)?
            }
        };

        self.profiler
            .end_timer("optimized_conv2d".to_string(), timer);
        Ok(result)
    }

    /// Memory-efficient forward pass for large batches
    #[cfg(feature = "memory_efficient")]
    pub fn memory_efficient_forward<F>(
        &mut self,
        input: &ArrayD<f32>,
        forward_fn: F,
    ) -> Result<ArrayD<f32>>
    where
        F: Fn(&ndarray::ArrayView<f32, ndarray::IxDyn>) -> Result<ArrayD<f32>>,
    {
        let timer = self.profiler.start_timer("memory_efficient_forward");
        let result = self
            .memory_processor
            .memory_efficient_forward(input, forward_fn);
        self.profiler
            .end_timer("memory_efficient_forward".to_string(), timer);
        result
    }

    /// Process large tensors in memory-efficient chunks
    #[cfg(feature = "memory_efficient")]
    pub fn process_in_chunks<F, T>(
        &mut self,
        input: &ArrayD<f32>,
        processor: F,
    ) -> Result<ArrayD<T>>
    where
        F: FnMut(&ndarray::ArrayView<f32, ndarray::IxDyn>) -> Result<ArrayD<T>>,
        T: Clone + std::fmt::Debug + Default,
    {
        let timer = self.profiler.start_timer("process_in_chunks");
        let result = self.memory_processor.process_in_chunks(input, processor);
        self.profiler
            .end_timer("process_in_chunks".to_string(), timer);
        result
    }

    /// Get comprehensive performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        PerformanceStats {
            capabilities: self.capabilities.clone(),
            profiling_stats: self.profiler.get_stats(),
            thread_pool_stats: self.thread_pool.get_stats(),
            simd_stats: SIMDStats::detect(),
        }
    }

    /// Reset all performance tracking
    pub fn reset_stats(&mut self) {
        self.profiler.clear();
    }

    /// Helper to determine if matrices are suitable for SIMD operations
    #[allow(dead_code)]
    fn is_suitable_for_simd(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> bool {
        if a.ndim() != 2 || b.ndim() != 2 {
            return false;
        }

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];

        // SIMD is more effective for medium-sized matrices
        // Very small matrices have too much overhead, very large ones benefit more from parallelism
        m >= 32 && n >= 32 && k >= 32 && m <= 2048 && n <= 2048 && k <= 2048
    }

    /// Benchmark different optimization strategies
    pub fn benchmark_strategies(
        &mut self,
        a: &ArrayD<f32>,
        b: &ArrayD<f32>,
        iterations: usize,
    ) -> Result<BenchmarkResults> {
        let mut results = BenchmarkResults::default();

        // Benchmark SIMD matmul
        #[cfg(feature = "simd")]
        {
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = SIMDOperations::simd_matmul_f32(&a.view(), &b.view());
            }
            results.simd_time = Some(start.elapsed() / iterations as u32);
        }

        // Benchmark parallel matmul
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = self.thread_pool.parallel_matmul(a, b)?;
        }
        results.parallel_time = start.elapsed() / iterations as u32;

        // Benchmark serial matmul
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = self.serial_matmul(a, b)?;
        }
        results.serial_time = start.elapsed() / iterations as u32;

        Ok(results)
    }

    /// Serial matrix multiplication for benchmarking
    fn serial_matmul(&self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(NeuralError::ComputationError(
                "Serial matmul requires 2D arrays".to_string(),
            ));
        }

        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        let mut result = ndarray::Array::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for ki in 0..k {
                    sum += a[[i, ki]] * b[[ki, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result.into_dyn())
    }
}

/// Comprehensive performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// System optimization capabilities
    pub capabilities: OptimizationCapabilities,
    /// Profiling statistics
    pub profiling_stats: ProfilingStats,
    /// Thread pool statistics
    pub thread_pool_stats: ThreadPoolStats,
    /// SIMD statistics
    pub simd_stats: SIMDStats,
}

impl std::fmt::Display for PerformanceStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Statistics:")?;
        writeln!(f, "======================")?;
        writeln!(f)?;
        writeln!(f, "{}", self.capabilities)?;
        writeln!(f, "{}", self.simd_stats)?;
        writeln!(f, "Thread Pool:")?;
        writeln!(f, "  Threads: {}", self.thread_pool_stats.num_threads)?;
        writeln!(f, "  Active: {}", self.thread_pool_stats.active)?;
        writeln!(f)?;
        writeln!(f, "Profiling:")?;
        writeln!(f, "  Enabled: {}", self.profiling_stats.enabled)?;
        writeln!(f, "  Operations: {}", self.profiling_stats.total_operations)?;
        writeln!(f, "  Total Calls: {}", self.profiling_stats.total_calls)?;
        writeln!(
            f,
            "  Total Time: {:.3}ms",
            self.profiling_stats.total_time.as_secs_f64() * 1000.0
        )?;
        writeln!(f, "  Active Timers: {}", self.profiling_stats.active_timers)?;
        Ok(())
    }
}

/// Benchmark results for different optimization strategies
#[derive(Debug, Clone, Default)]
pub struct BenchmarkResults {
    /// Time taken by SIMD implementation
    pub simd_time: Option<std::time::Duration>,
    /// Time taken by parallel implementation
    pub parallel_time: std::time::Duration,
    /// Time taken by serial implementation
    pub serial_time: std::time::Duration,
}

impl BenchmarkResults {
    /// Get speedup of parallel vs serial
    pub fn parallel_speedup(&self) -> f64 {
        self.serial_time.as_secs_f64() / self.parallel_time.as_secs_f64()
    }

    /// Get speedup of SIMD vs serial
    pub fn simd_speedup(&self) -> Option<f64> {
        self.simd_time
            .map(|simd| self.serial_time.as_secs_f64() / simd.as_secs_f64())
    }

    /// Get the best performing strategy
    pub fn best_strategy(&self) -> &'static str {
        let parallel_faster = self.parallel_time < self.serial_time;

        if let Some(simd_time) = self.simd_time {
            if simd_time < self.parallel_time && simd_time < self.serial_time {
                "SIMD"
            } else if parallel_faster {
                "Parallel"
            } else {
                "Serial"
            }
        } else if parallel_faster {
            "Parallel"
        } else {
            "Serial"
        }
    }
}

impl std::fmt::Display for BenchmarkResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Benchmark Results:")?;
        writeln!(f, "==================")?;
        writeln!(
            f,
            "Serial Time: {:.3}ms",
            self.serial_time.as_secs_f64() * 1000.0
        )?;
        writeln!(
            f,
            "Parallel Time: {:.3}ms",
            self.parallel_time.as_secs_f64() * 1000.0
        )?;
        writeln!(f, "Parallel Speedup: {:.2}x", self.parallel_speedup())?;

        if let Some(simd_time) = self.simd_time {
            writeln!(f, "SIMD Time: {:.3}ms", simd_time.as_secs_f64() * 1000.0)?;
            writeln!(
                f,
                "SIMD Speedup: {:.2}x",
                self.simd_speedup().unwrap_or(0.0)
            )?;
        }

        writeln!(f, "Best Strategy: {}", self.best_strategy())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_optimizer_creation() {
        let optimizer = PerformanceOptimizer::new(Some(256), Some(1024), Some(4), true);
        assert!(optimizer.is_ok());

        let optimizer = optimizer.unwrap();
        assert_eq!(optimizer.thread_pool().num_threads(), 4);
        assert!(optimizer.profiler().get_stats().enabled);
    }

    #[test]
    fn test_thread_pool_manager() {
        let pool = ThreadPoolManager::new(Some(2));
        assert!(pool.is_ok());

        let pool = pool.unwrap();
        assert_eq!(pool.num_threads(), 2);
    }

    #[test]
    fn test_performance_profiler() {
        let mut profiler = PerformanceProfiler::new(true);
        let timer = profiler.start_timer("test_operation");
        std::thread::sleep(std::time::Duration::from_millis(1));
        profiler.end_timer("test_operation".to_string(), timer);

        let timings = profiler.get_timings();
        assert!(timings.contains_key("test_operation"));
        assert!(timings["test_operation"] > std::time::Duration::ZERO);
    }

    #[test]
    fn test_optimization_capabilities() {
        let caps = OptimizationCapabilities::detect();
        assert!(caps.optimization_score() >= 0.0 && caps.optimization_score() <= 1.0);
    }

    #[test]
    #[cfg(feature = "memory_efficient")]
    fn test_memory_efficient_processor() {
        let processor = MemoryEfficientProcessor::new(Some(10), Some(100));
        let input = Array::ones((20, 5)).into_dyn();

        let result = processor.process_in_chunks(&input, |chunk| Ok(chunk.to_owned()));

        assert!(result.is_ok());
    }

    #[test]
    fn test_benchmark_results() {
        let results = BenchmarkResults {
            simd_time: Some(std::time::Duration::from_millis(10)),
            parallel_time: std::time::Duration::from_millis(15),
            serial_time: std::time::Duration::from_millis(30),
        };

        assert!(results.parallel_speedup() > 1.0);
        assert!(results.simd_speedup().unwrap() > 1.0);
        assert_eq!(results.best_strategy(), "SIMD");
    }
}
