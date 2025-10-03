//! Memory profiling and optimization utilities for statistical operations
//!
//! This module provides tools for profiling memory usage and implementing
//! memory-efficient algorithms with adaptive strategies based on available memory.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use std::collections::VecDeque;
use std::sync::Arc;

/// Memory usage profiler for statistical operations
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Average memory usage in bytes
    pub avg_memory: usize,
    /// Number of allocations
    pub allocations: usize,
    /// Number of deallocations
    pub deallocations: usize,
    /// Memory efficiency score (0-1)
    pub efficiency_score: f64,
}

/// Memory-aware algorithm selector
pub struct MemoryAdaptiveAlgorithm {
    /// Available memory in bytes
    available_memory: usize,
    /// Preferred chunk size for streaming operations
    preferred_chunksize: usize,
    /// Whether to use in-place algorithms when possible
    #[allow(dead_code)]
    prefer_inplace: bool,
}

impl Default for MemoryAdaptiveAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryAdaptiveAlgorithm {
    /// Create a new memory-adaptive algorithm selector
    pub fn new() -> Self {
        // Estimate available memory (simplified - in production would use system calls)
        let available_memory = Self::estimate_available_memory();
        let preferred_chunksize = Self::calculate_optimal_chunksize(available_memory);

        Self {
            available_memory,
            preferred_chunksize,
            prefer_inplace: available_memory < 1_000_000_000, // Prefer in-place if < 1GB
        }
    }

    /// Estimate available system memory using platform-specific calls
    fn estimate_available_memory() -> usize {
        #[cfg(target_os = "linux")]
        {
            Self::get_available_memory_linux()
        }
        #[cfg(target_os = "windows")]
        {
            Self::get_available_memory_windows()
        }
        #[cfg(target_os = "macos")]
        {
            Self::get_available_memory_macos()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            // Fallback for other systems
            Self::get_available_memory_fallback()
        }
    }

    #[cfg(target_os = "linux")]
    fn get_available_memory_linux() -> usize {
        use std::fs;

        // Read /proc/meminfo for accurate memory information
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            let mut mem_available = None;
            let mut mem_free = None;
            let mut mem_total = None;

            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = value.parse::<usize>() {
                            mem_available = Some(kb * 1024); // Convert KB to bytes
                        }
                    }
                } else if line.starts_with("MemFree:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = value.parse::<usize>() {
                            mem_free = Some(kb * 1024);
                        }
                    }
                } else if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = value.parse::<usize>() {
                            mem_total = Some(kb * 1024);
                        }
                    }
                }
            }

            // Prefer MemAvailable (includes reclaimable memory), fallback to MemFree
            if let Some(available) = mem_available {
                return available;
            } else if let Some(free) = mem_free {
                return free;
            } else if let Some(total) = mem_total {
                // Conservative estimate: 50% of total if we can't get precise info
                return total / 2;
            }
        }

        // Fallback if /proc/meminfo is not readable
        Self::get_available_memory_fallback()
    }

    #[cfg(target_os = "windows")]
    fn get_available_memory_windows() -> usize {
        // On Windows, we would use GlobalMemoryStatusEx API
        // For cross-platform compatibility without external dependencies,
        // we'll use a conservative estimate based on typical Windows systems

        // This could be improved by using winapi crate:
        // use winapi::um::sysinfoapi::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // But to avoid dependencies, we use a reasonable estimate

        // Assume at least 4GB total, use 25% as available
        let conservative_total = 4_000_000_000; // 4GB
        conservative_total / 4
    }

    #[cfg(target_os = "macos")]
    fn get_available_memory_macos() -> usize {
        use std::process::Command;

        // Use vm_stat command to get memory information
        if let Ok(output) = Command::new("vm_stat").output() {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                let mut pagesize = 4096; // Default page size
                let mut free_pages = 0;
                let mut inactive_pages = 0;

                for line in stdout.lines() {
                    if line.starts_with("Mach Virtual Memory Statistics:") {
                        // Try to extract page size if available
                        if line.contains("page size of") {
                            if let Some(size_str) = line.split("page size of ").nth(1) {
                                if let Some(size_str) = size_str.split(" bytes").next() {
                                    if let Ok(size) = size_str.parse::<usize>() {
                                        pagesize = size;
                                    }
                                }
                            }
                        }
                    } else if line.starts_with("Pages free:") {
                        if let Some(count_str) = line.split(':').nth(1) {
                            if let Some(count_str) = count_str.trim().split('.').next() {
                                if let Ok(count) = count_str.parse::<usize>() {
                                    free_pages = count;
                                }
                            }
                        }
                    } else if line.starts_with("Pages inactive:") {
                        if let Some(count_str) = line.split(':').nth(1) {
                            if let Some(count_str) = count_str.trim().split('.').next() {
                                if let Ok(count) = count_str.parse::<usize>() {
                                    inactive_pages = count;
                                }
                            }
                        }
                    }
                }

                // Available memory is approximately free + inactive pages
                return (free_pages + inactive_pages) * pagesize;
            }
        }

        // Fallback if vm_stat fails
        Self::get_available_memory_fallback()
    }

    fn get_available_memory_fallback() -> usize {
        // Conservative fallback for unknown systems
        // Assume at least 2GB total memory, use 25% as available
        let conservative_total = 2_000_000_000; // 2GB
        conservative_total / 4 // 500MB
    }

    /// Calculate optimal chunk size based on available memory
    fn calculate_optimal_chunksize(_availablememory: usize) -> usize {
        // Aim for chunks that fit comfortably in L3 cache (typically 8-32MB)
        let l3_cache_estimate = 8_000_000; // 8MB
        let max_chunk = _availablememory / 10; // Use at most 10% of available _memory

        l3_cache_estimate.min(max_chunk).max(4096)
    }

    /// Check if an operation can be performed in available memory
    pub fn can_allocate(&self, bytes: usize) -> bool {
        bytes <= self.available_memory / 2 // Conservative: use at most half
    }

    /// Get recommended algorithm based on data size
    pub fn recommend_algorithm<F: Float>(&self, datasize: usize) -> AlgorithmChoice {
        let elementsize = std::mem::size_of::<F>();
        let total_bytes = datasize * elementsize;

        if total_bytes < 1_000_000 {
            // < 1MB
            AlgorithmChoice::Direct
        } else if self.can_allocate(total_bytes) {
            AlgorithmChoice::Optimized
        } else {
            AlgorithmChoice::Streaming(self.preferred_chunksize / elementsize)
        }
    }
}

#[derive(Debug, Clone)]
pub enum AlgorithmChoice {
    /// Use direct algorithm (small data)
    Direct,
    /// Use optimized algorithm (medium data)
    Optimized,
    /// Use streaming algorithm with given chunk size
    Streaming(usize),
}

/// Zero-copy view-based statistics
///
/// These functions operate on views to avoid unnecessary copying
pub mod zero_copy {
    use super::*;

    /// Compute statistics on overlapping windows without copying
    pub fn rolling_stats_zerocopy<F, D, S>(
        data: &ArrayBase<D, Ix1>,
        windowsize: usize,
        stat_fn: S,
    ) -> StatsResult<Array1<F>>
    where
        F: Float + NumCast,
        D: Data<Elem = F>,
        S: Fn(ArrayView1<F>) -> StatsResult<F>,
    {
        let n = data.len();
        if windowsize == 0 || windowsize > n {
            return Err(StatsError::invalid_argument("Invalid window size"));
        }

        let output_len = n - windowsize + 1;
        let mut results = Array1::zeros(output_len);

        // Use views to avoid copying
        for i in 0..output_len {
            let window = data.slice(s![i..i + windowsize]);
            results[i] = stat_fn(window)?;
        }

        Ok(results)
    }

    /// Compute pairwise operations using views
    pub fn pairwise_operation_zerocopy<F, D, Op>(
        data: &ArrayBase<D, Ix2>,
        operation: Op,
    ) -> StatsResult<Array2<F>>
    where
        F: Float + NumCast,
        D: Data<Elem = F>,
        Op: Fn(ArrayView1<F>, ArrayView1<F>) -> StatsResult<F>,
    {
        let n = data.nrows();
        let mut result = Array2::zeros((n, n));

        for i in 0..n {
            result[(i, i)] = F::one(); // Diagonal
            for j in (i + 1)..n {
                let row_i = data.row(i);
                let row_j = data.row(j);
                let value = operation(row_i, row_j)?;
                result[(i, j)] = value;
                result[(j, i)] = value; // Symmetric
            }
        }

        Ok(result)
    }
}

/// Memory-mapped statistical operations for very large datasets
pub mod memory_mapped {
    use super::*;

    /// Chunked mean calculation for memory-mapped data
    pub fn mmap_mean<'a, F: Float + NumCast + std::fmt::Display + std::iter::Sum<F> + 'a>(
        data_chunks: impl Iterator<Item = ArrayView1<'a, F>>,
        total_count: usize,
    ) -> StatsResult<F> {
        if total_count == 0 {
            return Err(StatsError::invalid_argument("Empty dataset"));
        }

        let mut total_sum = F::zero();
        let mut count_processed = 0;

        for chunk in data_chunks {
            let chunk_sum = chunk.sum();
            total_sum = total_sum + chunk_sum;
            count_processed += chunk.len();
        }

        if count_processed != total_count {
            return Err(StatsError::invalid_argument("Chunk _count mismatch"));
        }

        Ok(total_sum / F::from(total_count).unwrap())
    }

    /// Chunked variance calculation using Welford's algorithm
    pub fn mmap_variance<'a, F: Float + NumCast + std::fmt::Display + 'a>(
        data_chunks: impl Iterator<Item = ArrayView1<'a, F>>,
        total_count: usize,
        ddof: usize,
    ) -> StatsResult<(F, F)> {
        if total_count <= ddof {
            return Err(StatsError::invalid_argument("Insufficient data for ddof"));
        }

        let mut mean = F::zero();
        let mut m2 = F::zero();
        let mut _count = 0;

        for chunk in data_chunks {
            for &value in chunk.iter() {
                _count += 1;
                let delta = value - mean;
                mean = mean + delta / F::from(_count).unwrap();
                let delta2 = value - mean;
                m2 = m2 + delta * delta2;
            }
        }

        let variance = m2 / F::from(_count - ddof).unwrap();
        Ok((mean, variance))
    }
}

/// Ring buffer for streaming statistics with fixed memory usage
pub struct RingBufferStats<F: Float> {
    buffer: VecDeque<F>,
    capacity: usize,
    sum: F,
    sum_squares: F,
}

impl<F: Float + NumCast + std::fmt::Display> RingBufferStats<F> {
    /// Create a new ring buffer with fixed capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            sum: F::zero(),
            sum_squares: F::zero(),
        }
    }

    /// Add a new value, potentially evicting the oldest
    pub fn push(&mut self, value: F) {
        if self.buffer.len() >= self.capacity {
            if let Some(old_value) = self.buffer.pop_front() {
                self.sum = self.sum - old_value;
                self.sum_squares = self.sum_squares - old_value * old_value;
            }
        }

        self.buffer.push_back(value);
        self.sum = self.sum + value;
        self.sum_squares = self.sum_squares + value * value;
    }

    /// Get current mean
    pub fn mean(&self) -> F {
        if self.buffer.is_empty() {
            F::zero()
        } else {
            self.sum / F::from(self.buffer.len()).unwrap()
        }
    }

    /// Get current variance
    pub fn variance(&self, ddof: usize) -> Option<F> {
        let n = self.buffer.len();
        if n <= ddof {
            return None;
        }

        let mean = self.mean();
        let var = self.sum_squares / F::from(n).unwrap() - mean * mean;
        Some(var * F::from(n).unwrap() / F::from(n - ddof).unwrap())
    }

    /// Get current standard deviation
    pub fn std(&self, ddof: usize) -> Option<F> {
        self.variance(ddof).map(|v| v.sqrt())
    }
}

/// Lazy evaluation for statistical operations
pub struct LazyStatComputation<F: Float> {
    data_ref: Arc<Vec<F>>,
    operations: Vec<StatOperation>,
}

#[derive(Clone)]
enum StatOperation {
    Mean,
    Variance(usize), // ddof
    Quantile(f64),
    #[allow(dead_code)]
    StandardScaling,
}

impl<F: Float + NumCast + std::iter::Sum + std::fmt::Display> LazyStatComputation<F> {
    /// Create a new lazy computation
    pub fn new(data: Vec<F>) -> Self {
        Self {
            data_ref: Arc::new(data),
            operations: Vec::new(),
        }
    }

    /// Add mean computation
    pub fn mean(mut self) -> Self {
        self.operations.push(StatOperation::Mean);
        self
    }

    /// Add variance computation
    pub fn variance(mut self, ddof: usize) -> Self {
        self.operations.push(StatOperation::Variance(ddof));
        self
    }

    /// Add quantile computation
    pub fn quantile(mut self, q: f64) -> Self {
        self.operations.push(StatOperation::Quantile(q));
        self
    }

    /// Execute all operations efficiently
    pub fn compute(&self) -> StatsResult<Vec<F>> {
        let mut results = Vec::new();
        let data = &*self.data_ref;

        // Check which operations we need
        let need_mean = self
            .operations
            .iter()
            .any(|op| matches!(op, StatOperation::Mean | StatOperation::Variance(_)));
        let need_sorted = self
            .operations
            .iter()
            .any(|op| matches!(op, StatOperation::Quantile(_)));

        // Compute shared values
        let mean = if need_mean {
            Some(data.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(data.len()).unwrap())
        } else {
            None
        };

        let sorteddata = if need_sorted {
            let mut sorted = data.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Some(sorted)
        } else {
            None
        };

        // Execute operations
        for op in &self.operations {
            match op {
                StatOperation::Mean => {
                    results.push(mean.unwrap());
                }
                StatOperation::Variance(ddof) => {
                    let m = mean.unwrap();
                    let var = data
                        .iter()
                        .map(|&x| {
                            let diff = x - m;
                            diff * diff
                        })
                        .sum::<F>()
                        / F::from(data.len() - ddof).unwrap();
                    results.push(var);
                }
                StatOperation::Quantile(q) => {
                    let sorted = sorteddata.as_ref().unwrap();
                    let pos = *q * (sorted.len() - 1) as f64;
                    let idx = pos.floor() as usize;
                    let frac = pos - pos.floor();

                    let result = if frac == 0.0 {
                        sorted[idx]
                    } else {
                        let lower = sorted[idx];
                        let upper = sorted[idx + 1];
                        lower + F::from(frac).unwrap() * (upper - lower)
                    };
                    results.push(result);
                }
                StatOperation::StandardScaling => {
                    // This would require returning transformed data
                    // For now, just return a placeholder
                    results.push(F::one());
                }
            }
        }

        Ok(results)
    }
}

/// Memory usage tracker for profiling
pub struct MemoryTracker {
    current_usage: usize,
    peak_usage: usize,
    allocations: usize,
    deallocations: usize,
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
            current_usage: 0,
            peak_usage: 0,
            allocations: 0,
            deallocations: 0,
        }
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, bytes: usize) {
        self.current_usage += bytes;
        self.peak_usage = self.peak_usage.max(self.current_usage);
        self.allocations += 1;
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, bytes: usize) {
        self.current_usage = self.current_usage.saturating_sub(bytes);
        self.deallocations += 1;
    }

    /// Get memory profile
    pub fn get_profile(&self) -> MemoryProfile {
        let efficiency_score = if self.peak_usage > 0 {
            1.0 - (self.current_usage as f64 / self.peak_usage as f64)
        } else {
            1.0
        };

        MemoryProfile {
            peak_memory: self.peak_usage,
            avg_memory: (self.peak_usage + self.current_usage) / 2,
            allocations: self.allocations,
            deallocations: self.deallocations,
            efficiency_score,
        }
    }
}

/// Cache-friendly matrix operations
pub mod cache_friendly {
    use super::*;

    /// Tiled matrix multiplication for better cache usage
    pub fn tiled_matrix_operation<F, D1, D2, Op>(
        a: &ArrayBase<D1, Ix2>,
        b: &ArrayBase<D2, Ix2>,
        tilesize: usize,
        operation: Op,
    ) -> StatsResult<Array2<F>>
    where
        F: Float + NumCast,
        D1: Data<Elem = F>,
        D2: Data<Elem = F>,
        Op: Fn(ArrayView2<F>, ArrayView2<F>) -> StatsResult<Array2<F>>,
    {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            return Err(StatsError::dimension_mismatch(
                "Matrix dimensions incompatible",
            ));
        }

        let mut result = Array2::zeros((m, n));

        // Process in tiles for better cache locality
        for i in (0..m).step_by(tilesize) {
            for j in (0..n).step_by(tilesize) {
                for k in (0..k1).step_by(tilesize) {
                    let i_end = (i + tilesize).min(m);
                    let j_end = (j + tilesize).min(n);
                    let k_end = (k + tilesize).min(k1);

                    let a_tile = a.slice(s![i..i_end, k..k_end]);
                    let b_tile = b.slice(s![k..k_end, j..j_end]);

                    let tile_result = operation(a_tile, b_tile)?;

                    // Add to result
                    let mut result_tile = result.slice_mut(s![i..i_end, j..j_end]);
                    result_tile.zip_mut_with(&tile_result, |r, &t| *r = *r + t);
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_memory_adaptive_algorithm() {
        let adapter = MemoryAdaptiveAlgorithm::new();

        // Test algorithm selection
        match adapter.recommend_algorithm::<f64>(100) {
            AlgorithmChoice::Direct => (), // Small data
            _ => panic!("Expected Direct algorithm for small data"),
        }

        // Use a much larger data size that will definitely exceed available memory
        // Force streaming by using a size that requires more than available memory
        let hugedatasize = adapter.available_memory / 4; // This will definitely trigger streaming
        match adapter.recommend_algorithm::<f64>(hugedatasize) {
            AlgorithmChoice::Streaming(_) => (), // Large data
            other => panic!(
                "Expected Streaming algorithm for large data, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_ring_buffer_stats() {
        let mut buffer = RingBufferStats::<f64>::new(5);

        // Add values
        for i in 1..=5 {
            buffer.push(i as f64);
        }

        assert_relative_eq!(buffer.mean(), 3.0, epsilon = 1e-10);

        // Add more values (should evict oldest)
        buffer.push(6.0);
        assert_relative_eq!(buffer.mean(), 4.0, epsilon = 1e-10); // (2+3+4+5+6)/5
    }

    #[test]
    fn test_lazy_computation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let lazy = LazyStatComputation::new(data)
            .mean()
            .variance(1)
            .quantile(0.5);

        let results = lazy.compute().unwrap();
        assert_eq!(results.len(), 3);
        assert_relative_eq!(results[0], 3.0, epsilon = 1e-10); // mean
        assert_relative_eq!(results[1], 2.5, epsilon = 1e-10); // variance
        assert_relative_eq!(results[2], 3.0, epsilon = 1e-10); // median
    }

    #[test]
    fn test_zero_copy_rolling() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let results =
            zero_copy::rolling_stats_zerocopy(&data.view(), 3, |window| Ok(window.mean().unwrap()))
                .unwrap();

        assert_eq!(results.len(), 3);
        assert_relative_eq!(results[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(results[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(results[2], 4.0, epsilon = 1e-10);
    }
}
