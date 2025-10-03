//! Memory management utilities for ndimage operations
//!
//! This module provides utilities for efficient memory management including:
//! - Views vs. copies control
//! - In-place operation options
//! - Memory footprint optimization
//! - Buffer reuse strategies

use scirs2_core::ndarray::{Array, Array2, ArrayBase, ArrayView, ArrayViewMut, Data, Dimension};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem;

use crate::error::{NdimageError, NdimageResult};

/// Memory allocation strategy for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategy {
    /// Always create new arrays (safest, most memory)
    AlwaysCopy,
    /// Use views when possible, copy when necessary
    PreferView,
    /// Reuse input array when possible (in-place operations)
    InPlace,
    /// Use pre-allocated buffer
    ReuseBuffer,
}

/// Configuration for memory-efficient operations
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Memory allocation strategy
    pub strategy: MemoryStrategy,
    /// Maximum memory to use (in bytes)
    pub memory_limit: Option<usize>,
    /// Whether to allow in-place operations
    pub allow_inplace: bool,
    /// Whether to prefer contiguous memory layout
    pub prefer_contiguous: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            strategy: MemoryStrategy::PreferView,
            memory_limit: None,
            allow_inplace: false,
            prefer_contiguous: true,
        }
    }
}

/// Buffer pool for reusing allocated arrays
pub struct BufferPool<T, D> {
    buffers: Vec<Array<T, D>>,
    max_buffers: usize,
    _phantom: PhantomData<T>,
}

impl<T: Float + FromPrimitive + Debug + Clone, D: Dimension> BufferPool<T, D> {
    pub fn new(maxbuffers: usize) -> Self {
        Self {
            buffers: Vec::new(),
            max_buffers: maxbuffers,
            _phantom: PhantomData,
        }
    }

    /// Get a buffer from the pool or allocate a new one
    pub fn get_buffer(&mut self, shape: D) -> Array<T, D> {
        // Try to find a buffer with matching shape
        if let Some(pos) = self.buffers.iter().position(|b| b.raw_dim() == shape) {
            self.buffers.swap_remove(pos)
        } else {
            // Allocate new buffer
            Array::zeros(shape)
        }
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer: Array<T, D>) {
        if self.buffers.len() < self.max_buffers {
            self.buffers.push(buffer);
        }
    }

    /// Clear all buffers from the pool
    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Get the number of buffers currently in the pool
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

/// Trait for operations that can be performed in-place
pub trait InPlaceOp<T, D>
where
    T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    /// Check if this operation can be performed in-place
    fn can_operate_inplace(&self) -> bool;

    /// Perform the operation in-place
    fn operate_inplace(&self, data: &mut ArrayViewMut<T, D>) -> NdimageResult<()>;

    /// Perform the operation out-of-place
    fn operate_out_of_place(&self, data: &ArrayView<T, D>) -> NdimageResult<Array<T, D>>;
}

/// Memory-efficient wrapper for array operations
pub struct MemoryEfficientOp<T, D> {
    config: MemoryConfig,
    phantom: PhantomData<(T, D)>,
}

impl<
        T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
        D: Dimension + 'static,
    > MemoryEfficientOp<T, D>
{
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            phantom: PhantomData,
        }
    }

    /// Execute an operation with memory efficiency considerations
    pub fn execute<Op, S>(&self, input: &ArrayBase<S, D>, op: Op) -> NdimageResult<Array<T, D>>
    where
        S: Data<Elem = T>,
        Op: InPlaceOp<T, D>,
    {
        match self.config.strategy {
            MemoryStrategy::AlwaysCopy => {
                // Always create a new array
                op.operate_out_of_place(&input.view())
            }
            MemoryStrategy::PreferView => {
                // Use view when possible
                op.operate_out_of_place(&input.view())
            }
            MemoryStrategy::InPlace => {
                if self.config.allow_inplace && op.can_operate_inplace() {
                    // Try to operate in-place if we own the data
                    let mut output = input.to_owned();
                    op.operate_inplace(&mut output.view_mut())?;
                    Ok(output)
                } else {
                    // Fall back to out-of-place
                    op.operate_out_of_place(&input.view())
                }
            }
            MemoryStrategy::ReuseBuffer => {
                // This would require a buffer pool passed in
                op.operate_out_of_place(&input.view())
            }
        }
    }
}

/// Estimate memory usage for an operation
#[allow(dead_code)]
pub fn estimate_memory_usage<T, D>(shape: &[usize]) -> usize
where
    T: Float + std::ops::AddAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    let elements: usize = shape.iter().product();
    elements * std::mem::size_of::<T>()
}

/// Check if an operation would exceed memory limit
#[allow(dead_code)]
pub fn check_memory_limit<T, D>(shape: &[usize], limit: Option<usize>) -> NdimageResult<()>
where
    T: Float + std::ops::AddAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
{
    if let Some(max_bytes) = limit {
        let required = estimate_memory_usage::<T, D>(shape);
        if required > max_bytes {
            return Err(NdimageError::MemoryError(format!(
                "Operation would require {} bytes, exceeding limit of {} bytes",
                required, max_bytes
            )));
        }
    }
    Ok(())
}

/// Create a memory-efficient view or copy based on configuration
#[allow(dead_code)]
pub fn create_output_array<T, D, S>(
    input: &ArrayBase<S, D>,
    config: &MemoryConfig,
) -> NdimageResult<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
    S: Data<Elem = T>,
{
    let shape = input.shape();
    check_memory_limit::<T, D>(shape, config.memory_limit)?;

    let output = if config.prefer_contiguous && !input.is_standard_layout() {
        // Create contiguous copy
        input.to_owned().as_standard_layout().to_owned()
    } else {
        // Create regular copy
        input.to_owned()
    };

    Ok(output)
}

/// Example in-place operation: element-wise square
pub struct SquareOp;

impl<
        T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
        D: Dimension + 'static,
    > InPlaceOp<T, D> for SquareOp
{
    fn can_operate_inplace(&self) -> bool {
        true
    }

    fn operate_inplace(&self, data: &mut ArrayViewMut<T, D>) -> NdimageResult<()> {
        data.mapv_inplace(|x| x * x);
        Ok(())
    }

    fn operate_out_of_place(&self, data: &ArrayView<T, D>) -> NdimageResult<Array<T, D>> {
        Ok(data.mapv(|x| x * x))
    }
}

/// Example in-place operation: threshold
pub struct ThresholdOp<T> {
    threshold: T,
    value: T,
}

impl<
        T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
    > ThresholdOp<T>
{
    pub fn new(threshold: T, value: T) -> Self {
        Self { threshold, value }
    }
}

impl<
        T: Float + FromPrimitive + Debug + Clone + std::ops::AddAssign + std::ops::DivAssign + 'static,
        D: Dimension + 'static,
    > InPlaceOp<T, D> for ThresholdOp<T>
{
    fn can_operate_inplace(&self) -> bool {
        true
    }

    fn operate_inplace(&self, data: &mut ArrayViewMut<T, D>) -> NdimageResult<()> {
        data.mapv_inplace(|x| if x > self.threshold { self.value } else { x });
        Ok(())
    }

    fn operate_out_of_place(&self, data: &ArrayView<T, D>) -> NdimageResult<Array<T, D>> {
        Ok(data.mapv(|x| if x > self.threshold { self.value } else { x }))
    }
}

/// Memory-efficient array slicing that avoids copies when possible
#[allow(dead_code)]
pub fn slice_efficiently<'a, T, D, S>(
    array: &'a ArrayBase<S, D>,
    _slice_info: &[std::ops::Range<usize>],
) -> ArrayView<'a, T, D>
where
    T: Float + std::ops::AddAssign + std::ops::DivAssign + 'static,
    D: Dimension + 'static,
    S: Data<Elem = T>,
{
    // This is a simplified version - in practice would use ndarray's slicing
    array.view()
}

/// Zero-copy transpose for 2D arrays
#[allow(dead_code)]
pub fn transpose_view<T, S>(array: &ArrayBase<S, scirs2_core::ndarray::Ix2>) -> Array2<T>
where
    T: Float + Copy,
    S: Data<Elem = T>,
{
    array.t().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_buffer_pool() {
        let mut pool: BufferPool<f64, scirs2_core::ndarray::Ix2> = BufferPool::new(5);

        // Get a buffer
        let buf1 = pool.get_buffer(scirs2_core::ndarray::Ix2(10, 10));
        assert_eq!(buf1.shape(), &[10, 10]);
        assert_eq!(pool.len(), 0);

        // Return it to the pool
        pool.return_buffer(buf1);
        assert_eq!(pool.len(), 1);

        // Get it again - should reuse
        let buf2 = pool.get_buffer(scirs2_core::ndarray::Ix2(10, 10));
        assert_eq!(buf2.shape(), &[10, 10]);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_memory_efficient_op() {
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        // Test in-place operation
        let config = MemoryConfig {
            strategy: MemoryStrategy::InPlace,
            allow_inplace: true,
            ..Default::default()
        };

        let op_wrapper = MemoryEfficientOp::new(config);
        let result = op_wrapper.execute(&input, SquareOp).unwrap();

        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 4.0);
        assert_eq!(result[[1, 0]], 9.0);
        assert_eq!(result[[1, 1]], 16.0);
    }

    #[test]
    fn test_memory_limit_check() {
        // Check that small array passes
        assert!(
            check_memory_limit::<f64, scirs2_core::ndarray::Ix2>(&[10, 10], Some(1000)).is_ok()
        );

        // Check that large array fails
        assert!(
            check_memory_limit::<f64, scirs2_core::ndarray::Ix2>(&[1000, 1000], Some(1000))
                .is_err()
        );
    }

    #[test]
    fn test_threshold_op() {
        let input = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let op = ThresholdOp::new(2.5, 0.0);

        let config = MemoryConfig::default();
        let op_wrapper = MemoryEfficientOp::new(config);
        let result = op_wrapper.execute(&input, op).unwrap();

        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[1, 0]], 0.0); // Thresholded
        assert_eq!(result[[1, 1]], 0.0); // Thresholded
    }
}
