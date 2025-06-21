//! Memory access pattern optimization utilities
//!
//! This module provides utilities for optimizing memory access patterns in
//! numerical computations. It includes cache-friendly algorithms, data layout
//! optimizations, and memory pooling strategies to improve performance.
//!
//! # Memory Layout Optimization
//!
//! Different memory layouts can significantly impact performance:
//! - Row-major vs column-major for matrices
//! - Structure of Arrays (SoA) vs Array of Structures (AoS)
//! - Cache-friendly blocking and tiling
//!
//! # Examples
//!
//! ```
//! use scirs2_integrate::memory::{MemoryPool, CacheFriendlyMatrix, BlockingStrategy, MatrixLayout};
//!
//! // Use memory pool for frequent allocations
//! let mut pool = MemoryPool::<f64>::new(1024);
//! let buffer = pool.allocate(100);
//!
//! // Cache-friendly matrix operations
//! let matrix = CacheFriendlyMatrix::<f64>::new(100, 100, MatrixLayout::RowMajor);
//! let blocking = BlockingStrategy::new(64); // 64x64 blocks
//! ```

use crate::common::IntegrateFloat;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use std::collections::VecDeque;

/// Memory pool for efficient allocation and reuse of buffers
#[derive(Debug)]
pub struct MemoryPool<F: IntegrateFloat> {
    /// Available buffers by size
    buffers: std::collections::HashMap<usize, VecDeque<Vec<F>>>,
    /// Maximum buffer size to pool
    max_buffer_size: usize,
    /// Total memory currently allocated
    total_allocated: usize,
    /// Maximum total memory allowed
    max_total_memory: usize,
}

impl<F: IntegrateFloat> MemoryPool<F> {
    /// Create a new memory pool
    pub fn new(max_total_memory: usize) -> Self {
        Self {
            buffers: std::collections::HashMap::new(),
            max_buffer_size: max_total_memory / 4, // Quarter of total for single buffer
            total_allocated: 0,
            max_total_memory,
        }
    }

    /// Allocate a buffer of specified size
    pub fn allocate(&mut self, size: usize) -> PooledBuffer<F> {
        if size <= self.max_buffer_size {
            // Try to reuse existing buffer
            if let Some(queue) = self.buffers.get_mut(&size) {
                if let Some(mut buffer) = queue.pop_front() {
                    buffer.clear();
                    buffer.resize(size, F::zero());
                    return PooledBuffer::new(buffer, Some(size));
                }
            }
        }

        // Allocate new buffer
        if self.total_allocated + size * std::mem::size_of::<F>() <= self.max_total_memory {
            let buffer = vec![F::zero(); size];
            self.total_allocated += size * std::mem::size_of::<F>();

            if size <= self.max_buffer_size {
                PooledBuffer::new(buffer, Some(size))
            } else {
                PooledBuffer::new(buffer, None)
            }
        } else {
            // Fallback to regular allocation if pool is full
            PooledBuffer::new(vec![F::zero(); size], None)
        }
    }

    /// Return a buffer to the pool
    pub fn deallocate(&mut self, buffer: Vec<F>) {
        let size = buffer.len();
        if size <= self.max_buffer_size && self.buffers.len() < 100 {
            self.buffers.entry(size).or_default().push_back(buffer);
        } else {
            self.total_allocated = self
                .total_allocated
                .saturating_sub(size * std::mem::size_of::<F>());
        }
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> MemoryUsage {
        let pooled_buffers: usize = self.buffers.values().map(|q| q.len()).sum();
        let pooled_memory: usize = self
            .buffers
            .iter()
            .map(|(size, queue)| size * queue.len() * std::mem::size_of::<F>())
            .sum();

        MemoryUsage {
            total_allocated: self.total_allocated,
            pooled_memory,
            pooled_buffers,
            max_total_memory: self.max_total_memory,
        }
    }

    /// Clear all pooled buffers
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.total_allocated = 0;
    }
}

/// RAII wrapper for pooled buffers
pub struct PooledBuffer<F: IntegrateFloat> {
    buffer: Option<Vec<F>>,
    #[allow(dead_code)]
    pool_size: Option<usize>,
}

impl<F: IntegrateFloat> PooledBuffer<F> {
    fn new(buffer: Vec<F>, pool_size: Option<usize>) -> Self {
        Self {
            buffer: Some(buffer),
            pool_size,
        }
    }

    /// Get reference to the buffer
    pub fn as_slice(&self) -> &[F] {
        self.buffer.as_ref().unwrap()
    }

    /// Get mutable reference to the buffer
    pub fn as_mut_slice(&mut self) -> &mut [F] {
        self.buffer.as_mut().unwrap()
    }

    /// Convert to owned Vec (consumes the buffer)
    pub fn into_vec(mut self) -> Vec<F> {
        self.buffer.take().unwrap()
    }

    /// Get length of buffer
    pub fn len(&self) -> usize {
        self.buffer.as_ref().unwrap().len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.as_ref().unwrap().is_empty()
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub total_allocated: usize,
    pub pooled_memory: usize,
    pub pooled_buffers: usize,
    pub max_total_memory: usize,
}

/// Cache-friendly matrix with optimized memory layout
#[derive(Debug, Clone)]
pub struct CacheFriendlyMatrix<F: IntegrateFloat> {
    data: Vec<F>,
    rows: usize,
    cols: usize,
    layout: MatrixLayout,
}

#[derive(Debug, Clone, Copy)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
    Blocked { block_size: usize },
}

impl<F: IntegrateFloat> CacheFriendlyMatrix<F> {
    /// Create new matrix with specified layout
    pub fn new(rows: usize, cols: usize, layout: MatrixLayout) -> Self {
        let data = vec![F::zero(); rows * cols];
        Self {
            data,
            rows,
            cols,
            layout,
        }
    }

    /// Create from existing Array2 with optimal layout
    pub fn from_array(array: &Array2<F>) -> Self {
        let (rows, cols) = array.dim();
        let mut matrix = Self::new(rows, cols, MatrixLayout::RowMajor);

        for i in 0..rows {
            for j in 0..cols {
                matrix.set(i, j, array[[i, j]]);
            }
        }

        matrix
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> F {
        let index = self.compute_index(row, col);
        self.data[index]
    }

    /// Set element at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: F) {
        let index = self.compute_index(row, col);
        self.data[index] = value;
    }

    /// Compute linear index based on layout
    fn compute_index(&self, row: usize, col: usize) -> usize {
        match self.layout {
            MatrixLayout::RowMajor => row * self.cols + col,
            MatrixLayout::ColumnMajor => col * self.rows + row,
            MatrixLayout::Blocked { block_size } => {
                let block_row = row / block_size;
                let block_col = col / block_size;
                let in_block_row = row % block_size;
                let in_block_col = col % block_size;

                let blocks_per_row = self.cols.div_ceil(block_size);
                let block_index = block_row * blocks_per_row + block_col;
                let in_block_index = in_block_row * block_size + in_block_col;

                block_index * block_size * block_size + in_block_index
            }
        }
    }

    /// Get matrix dimensions
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Cache-friendly matrix-vector multiplication
    pub fn matvec(&self, x: ArrayView1<F>) -> Array1<F> {
        let mut y = Array1::zeros(self.rows);

        match self.layout {
            MatrixLayout::RowMajor => {
                for i in 0..self.rows {
                    let mut sum = F::zero();
                    for j in 0..self.cols {
                        sum += self.get(i, j) * x[j];
                    }
                    y[i] = sum;
                }
            }
            MatrixLayout::ColumnMajor => {
                for j in 0..self.cols {
                    let x_j = x[j];
                    for i in 0..self.rows {
                        y[i] += self.get(i, j) * x_j;
                    }
                }
            }
            MatrixLayout::Blocked { block_size } => {
                self.blocked_matvec(x.view(), y.view_mut(), block_size);
            }
        }

        y
    }

    /// Blocked matrix-vector multiplication for better cache performance
    fn blocked_matvec(&self, x: ArrayView1<F>, mut y: ArrayViewMut1<F>, block_size: usize) {
        let rows_blocks = self.rows.div_ceil(block_size);
        let cols_blocks = self.cols.div_ceil(block_size);

        for i_block in 0..rows_blocks {
            for j_block in 0..cols_blocks {
                let i_start = i_block * block_size;
                let i_end = (i_start + block_size).min(self.rows);
                let j_start = j_block * block_size;
                let j_end = (j_start + block_size).min(self.cols);

                for i in i_start..i_end {
                    let mut sum = F::zero();
                    for j in j_start..j_end {
                        sum += self.get(i, j) * x[j];
                    }
                    y[i] += sum;
                }
            }
        }
    }
}

/// Blocking strategy for cache-friendly algorithms
#[derive(Debug, Clone)]
pub struct BlockingStrategy {
    /// Block size for L1 cache
    pub l1_block_size: usize,
    /// Block size for L2 cache
    pub l2_block_size: usize,
    /// Block size for L3 cache
    pub l3_block_size: usize,
}

impl BlockingStrategy {
    /// Create blocking strategy with specified L1 block size
    pub fn new(l1_block_size: usize) -> Self {
        Self {
            l1_block_size,
            l2_block_size: l1_block_size * 4,
            l3_block_size: l1_block_size * 16,
        }
    }

    /// Get optimal block size for given matrix dimension and cache level
    pub fn optimal_block_size(&self, matrix_size: usize, cache_level: CacheLevel) -> usize {
        let block_size = match cache_level {
            CacheLevel::L1 => self.l1_block_size,
            CacheLevel::L2 => self.l2_block_size,
            CacheLevel::L3 => self.l3_block_size,
        };

        // Adjust for small matrices
        block_size.min(matrix_size)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
}

/// Cache-aware algorithms for common operations
pub struct CacheAwareAlgorithms;

impl CacheAwareAlgorithms {
    /// Cache-friendly matrix transpose
    pub fn transpose<F: IntegrateFloat>(
        input: ArrayView2<F>,
        mut output: ArrayViewMut2<F>,
        block_size: usize,
    ) {
        let (rows, cols) = input.dim();

        for i_block in (0..rows).step_by(block_size) {
            for j_block in (0..cols).step_by(block_size) {
                let i_end = (i_block + block_size).min(rows);
                let j_end = (j_block + block_size).min(cols);

                for i in i_block..i_end {
                    for j in j_block..j_end {
                        output[[j, i]] = input[[i, j]];
                    }
                }
            }
        }
    }

    /// Cache-friendly vector operations with temporal locality
    pub fn vector_add_blocked<F: IntegrateFloat>(
        a: ArrayView1<F>,
        b: ArrayView1<F>,
        mut c: ArrayViewMut1<F>,
        block_size: usize,
    ) {
        let n = a.len();

        for start in (0..n).step_by(block_size) {
            let end = (start + block_size).min(n);

            for i in start..end {
                c[i] = a[i] + b[i];
            }
        }
    }

    /// Memory-efficient reduction with minimal cache misses
    pub fn reduction_blocked<F: IntegrateFloat>(data: ArrayView1<F>, block_size: usize) -> F {
        let n = data.len();
        let mut partial_sums = Vec::new();

        // Compute partial sums for each block
        for start in (0..n).step_by(block_size) {
            let end = (start + block_size).min(n);
            let mut sum = F::zero();

            for i in start..end {
                sum += data[i];
            }

            partial_sums.push(sum);
        }

        // Final reduction of partial sums
        partial_sums.into_iter().fold(F::zero(), |acc, x| acc + x)
    }
}

/// Data layout optimization utilities
pub struct DataLayoutOptimizer;

impl DataLayoutOptimizer {
    /// Convert Array of Structures (AoS) to Structure of Arrays (SoA)
    pub fn aos_to_soa<F: IntegrateFloat>(
        data: &[(F, F, F)], // Input: array of (x, y, z) tuples
    ) -> (Vec<F>, Vec<F>, Vec<F>) {
        let mut x_values = Vec::with_capacity(data.len());
        let mut y_values = Vec::with_capacity(data.len());
        let mut z_values = Vec::with_capacity(data.len());

        for &(x, y, z) in data {
            x_values.push(x);
            y_values.push(y);
            z_values.push(z);
        }

        (x_values, y_values, z_values)
    }

    /// Convert Structure of Arrays (SoA) to Array of Structures (AoS)
    pub fn soa_to_aos<F: IntegrateFloat>(
        x_values: &[F],
        y_values: &[F],
        z_values: &[F],
    ) -> Vec<(F, F, F)> {
        assert_eq!(x_values.len(), y_values.len());
        assert_eq!(y_values.len(), z_values.len());

        x_values
            .iter()
            .zip(y_values.iter())
            .zip(z_values.iter())
            .map(|((&x, &y), &z)| (x, y, z))
            .collect()
    }

    /// Reorder data for better spatial locality
    pub fn spatial_reorder<F: IntegrateFloat>(data: &mut [(F, F)], // Input: (value, key) pairs
    ) {
        // Sort by key to improve spatial locality
        data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    }
}

/// Memory prefetching utilities for performance optimization
pub struct MemoryPrefetch;

impl MemoryPrefetch {
    /// Software prefetch hint (platform-specific)
    #[inline]
    pub fn prefetch_read<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for other architectures
            let _ = ptr;
        }
    }

    /// Prefetch for write access
    #[inline]
    pub fn prefetch_write<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for other architectures
            let _ = ptr;
        }
    }

    /// Stream data access pattern (non-temporal)
    #[inline]
    pub fn prefetch_stream<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_NTA);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for other architectures
            let _ = ptr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::<f64>::new(1024 * 1024); // 1MB

        // Allocate buffer
        let buffer = pool.allocate(100);
        assert_eq!(buffer.len(), 100);

        // Check memory usage
        let usage = pool.memory_usage();
        assert!(usage.total_allocated > 0);

        // Return buffer to pool
        let vec_buffer = buffer.into_vec();
        pool.deallocate(vec_buffer);

        // Allocate again (should reuse)
        let buffer2 = pool.allocate(100);
        assert_eq!(buffer2.len(), 100);
    }

    #[test]
    fn test_cache_friendly_matrix() {
        let mut matrix = CacheFriendlyMatrix::new(3, 3, MatrixLayout::RowMajor);

        // Set some values
        matrix.set(0, 0, 1.0);
        matrix.set(1, 1, 2.0);
        matrix.set(2, 2, 3.0);

        // Check retrieval
        assert_abs_diff_eq!(matrix.get(0, 0), 1.0);
        assert_abs_diff_eq!(matrix.get(1, 1), 2.0);
        assert_abs_diff_eq!(matrix.get(2, 2), 3.0);

        // Test matrix-vector multiplication
        let x = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let y = matrix.matvec(x.view());

        assert_abs_diff_eq!(y[0], 1.0);
        assert_abs_diff_eq!(y[1], 2.0);
        assert_abs_diff_eq!(y[2], 3.0);
    }

    #[test]
    fn test_blocking_strategy() {
        let strategy = BlockingStrategy::new(64);

        assert_eq!(strategy.l1_block_size, 64);
        assert_eq!(strategy.l2_block_size, 256);
        assert_eq!(strategy.l3_block_size, 1024);

        // Test optimal block size selection
        let block_size = strategy.optimal_block_size(32, CacheLevel::L1);
        assert_eq!(block_size, 32); // Should be limited by matrix size

        let block_size = strategy.optimal_block_size(128, CacheLevel::L1);
        assert_eq!(block_size, 64); // Should use L1 block size
    }

    #[test]
    fn test_data_layout_optimizer() {
        // Test AoS to SoA conversion
        let aos_data = vec![(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)];
        let (x, y, z) = DataLayoutOptimizer::aos_to_soa(&aos_data);

        assert_eq!(x, vec![1.0, 4.0, 7.0]);
        assert_eq!(y, vec![2.0, 5.0, 8.0]);
        assert_eq!(z, vec![3.0, 6.0, 9.0]);

        // Test SoA to AoS conversion
        let reconstructed = DataLayoutOptimizer::soa_to_aos(&x, &y, &z);
        assert_eq!(reconstructed, aos_data);
    }

    #[test]
    fn test_cache_aware_algorithms() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Test blocked reduction
        let sum = CacheAwareAlgorithms::reduction_blocked(data.view(), 3);
        assert_abs_diff_eq!(sum, 36.0); // 1+2+...+8 = 36

        // Test blocked vector addition
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
        let mut c = Array1::zeros(4);

        CacheAwareAlgorithms::vector_add_blocked(a.view(), b.view(), c.view_mut(), 2);

        assert_abs_diff_eq!(c[0], 6.0);
        assert_abs_diff_eq!(c[1], 8.0);
        assert_abs_diff_eq!(c[2], 10.0);
        assert_abs_diff_eq!(c[3], 12.0);
    }
}
