//! Memory optimization utilities for tensor operations
//!
//! This module provides memory optimization tools and utilities to reduce
//! memory allocations, improve memory locality, and provide memory profiling
//! capabilities for autograd operations.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use ndarray::IxDyn;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Global memory pool for reusing tensor storage
static MEMORY_POOL: LazyLock<Mutex<MemoryPool>> = LazyLock::new(|| Mutex::new(MemoryPool::new()));

/// Global memory usage tracker
static MEMORY_TRACKER: LazyLock<Mutex<MemoryTracker>> =
    LazyLock::new(|| Mutex::new(MemoryTracker::new()));

/// Memory pool for reusing tensor allocations
///
/// The memory pool maintains buffers of different sizes to reduce
/// the frequency of memory allocations during tensor operations.
/// This is particularly useful for operations that create many
/// temporary tensors.
struct MemoryPool {
    /// Pool of available buffers organized by size
    buffers: HashMap<usize, Vec<Vec<u8>>>,
    /// Maximum number of buffers to keep per size
    max_buffers_per_size: usize,
    /// Total memory currently pooled (bytes)
    total_pooled_memory: usize,
    /// Maximum total memory to pool (bytes)
    max_pool_memory: usize,
    /// Whether the pool is enabled
    enabled: bool,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            max_buffers_per_size: 16,
            total_pooled_memory: 0,
            max_pool_memory: 100 * 1024 * 1024, // 100MB default
            enabled: true,
        }
    }

    /// Get a buffer of the specified size, either from the pool or newly allocated
    fn get_buffer(&mut self, size: usize) -> Vec<u8> {
        if !self.enabled {
            return vec![0u8; size];
        }

        if let Some(buffers) = self.buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                self.total_pooled_memory = self.total_pooled_memory.saturating_sub(size);
                return buffer;
            }
        }

        // No buffer available, allocate new one
        vec![0u8; size]
    }

    /// Return a buffer to the pool for reuse
    fn return_buffer(&mut self, mut buffer: Vec<u8>) {
        if !self.enabled {
            return;
        }

        let size = buffer.len();

        // Clear the buffer for reuse
        buffer.fill(0);

        // Check if we should add this buffer to the pool
        let buffers = self.buffers.entry(size).or_default();

        if buffers.len() < self.max_buffers_per_size
            && self.total_pooled_memory + size <= self.max_pool_memory
        {
            buffers.push(buffer);
            self.total_pooled_memory += size;
        }
        // Otherwise, let the buffer be dropped
    }

    /// Configure the memory pool
    fn configure(&mut self, max_buffers_per_size: usize, max_pool_memory: usize) {
        self.max_buffers_per_size = max_buffers_per_size;
        self.max_pool_memory = max_pool_memory;
    }

    /// Enable or disable the memory pool
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear();
        }
    }

    /// Clear all buffers from the pool
    fn clear(&mut self) {
        self.buffers.clear();
        self.total_pooled_memory = 0;
    }

    /// Get memory pool statistics
    fn stats(&self) -> MemoryPoolStats {
        let total_buffers: usize = self.buffers.values().map(|v| v.len()).sum();
        MemoryPoolStats {
            total_buffers,
            total_pooled_memory: self.total_pooled_memory,
            buffer_sizes: self.buffers.keys().cloned().collect(),
            enabled: self.enabled,
        }
    }
}

/// Statistics about the memory pool
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    /// Total number of buffers in the pool
    pub total_buffers: usize,
    /// Total memory pooled in bytes
    pub total_pooled_memory: usize,
    /// Available buffer sizes
    pub buffer_sizes: Vec<usize>,
    /// Whether the pool is enabled
    pub enabled: bool,
}

/// Memory usage tracker for profiling tensor operations
struct MemoryTracker {
    /// Current memory usage by operation type
    current_usage: HashMap<String, usize>,
    /// Peak memory usage by operation type
    peak_usage: HashMap<String, usize>,
    /// Total allocations by operation type
    total_allocations: HashMap<String, usize>,
    /// Whether tracking is enabled
    enabled: bool,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            current_usage: HashMap::new(),
            peak_usage: HashMap::new(),
            total_allocations: HashMap::new(),
            enabled: false,
        }
    }

    fn record_allocation(&mut self, op_name: &str, size: usize) {
        if !self.enabled {
            return;
        }

        // Update current usage
        let current = self.current_usage.entry(op_name.to_string()).or_insert(0);
        *current += size;

        // Update peak usage
        let peak = self.peak_usage.entry(op_name.to_string()).or_insert(0);
        if *current > *peak {
            *peak = *current;
        }

        // Update total allocations
        let total = self
            .total_allocations
            .entry(op_name.to_string())
            .or_insert(0);
        *total += size;
    }

    #[allow(dead_code)]
    fn record_deallocation(&mut self, op_name: &str, size: usize) {
        if !self.enabled {
            return;
        }

        let current = self.current_usage.entry(op_name.to_string()).or_insert(0);
        *current = current.saturating_sub(size);
    }

    fn enable(&mut self) {
        self.enabled = true;
    }

    fn disable(&mut self) {
        self.enabled = false;
    }

    fn reset(&mut self) {
        self.current_usage.clear();
        self.peak_usage.clear();
        self.total_allocations.clear();
    }

    fn get_stats(&self) -> MemoryTrackerStats {
        MemoryTrackerStats {
            current_usage: self.current_usage.clone(),
            peak_usage: self.peak_usage.clone(),
            total_allocations: self.total_allocations.clone(),
            enabled: self.enabled,
        }
    }
}

/// Memory tracking statistics
#[derive(Debug, Clone)]
pub struct MemoryTrackerStats {
    /// Current memory usage by operation type
    pub current_usage: HashMap<String, usize>,
    /// Peak memory usage by operation type
    pub peak_usage: HashMap<String, usize>,
    /// Total allocations by operation type
    pub total_allocations: HashMap<String, usize>,
    /// Whether tracking is enabled
    pub enabled: bool,
}

/// Memory-efficient in-place operation that reuses tensor storage when possible
pub struct InPlaceOp<F: Float> {
    operation: InPlaceOperation,
    _phantom: std::marker::PhantomData<F>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
pub enum InPlaceOperation {
    /// In-place addition: x += y
    AddAssign,
    /// In-place subtraction: x -= y  
    SubAssign,
    /// In-place multiplication: x *= y
    MulAssign,
    /// In-place division: x /= y
    DivAssign,
    /// In-place negation: x = -x
    NegAssign,
    /// In-place absolute value: x = |x|
    AbsAssign,
    /// In-place scalar multiplication: x *= scalar
    ScalarMulAssign,
    /// In-place scalar addition: x += scalar
    ScalarAddAssign,
}

impl<F: Float> InPlaceOp<F> {
    pub fn new(operation: InPlaceOperation) -> Self {
        Self {
            operation,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float> Op<F> for InPlaceOp<F> {
    fn name(&self) -> &'static str {
        match self.operation {
            InPlaceOperation::AddAssign => "InPlaceAdd",
            InPlaceOperation::SubAssign => "InPlaceSub",
            InPlaceOperation::MulAssign => "InPlaceMul",
            InPlaceOperation::DivAssign => "InPlaceDiv",
            InPlaceOperation::NegAssign => "InPlaceNeg",
            InPlaceOperation::AbsAssign => "InPlaceAbs",
            InPlaceOperation::ScalarMulAssign => "InPlaceScalarMul",
            InPlaceOperation::ScalarAddAssign => "InPlaceScalarAdd",
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let mut target = ctx.input(0).to_owned();

        // Record memory allocation for tracking
        let op_name = self.name();
        let size = target.len() * std::mem::size_of::<F>();
        MEMORY_TRACKER
            .lock()
            .unwrap()
            .record_allocation(op_name, size);

        match self.operation {
            InPlaceOperation::AddAssign => {
                let rhs = ctx.input(1);
                target += &rhs;
            }
            InPlaceOperation::SubAssign => {
                let rhs = ctx.input(1);
                target -= &rhs;
            }
            InPlaceOperation::MulAssign => {
                let rhs = ctx.input(1);
                target *= &rhs;
            }
            InPlaceOperation::DivAssign => {
                let rhs = ctx.input(1);
                target /= &rhs;
            }
            InPlaceOperation::NegAssign => {
                target.mapv_inplace(|x| -x);
            }
            InPlaceOperation::AbsAssign => {
                target.mapv_inplace(|x| x.abs());
            }
            InPlaceOperation::ScalarMulAssign => {
                if ctx.inputs().len() > 1 {
                    let scalar_tensor = ctx.input(1);
                    if let Some(scalar_val) = scalar_tensor.iter().next() {
                        target.mapv_inplace(|x| x * (*scalar_val));
                    }
                }
            }
            InPlaceOperation::ScalarAddAssign => {
                if ctx.inputs().len() > 1 {
                    let scalar_tensor = ctx.input(1);
                    if let Some(scalar_val) = scalar_tensor.iter().next() {
                        target.mapv_inplace(|x| x + (*scalar_val));
                    }
                }
            }
        }

        ctx.append_output(target);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let _g = ctx.graph();

        match self.operation {
            InPlaceOperation::AddAssign => {
                // Both inputs get the same gradient
                ctx.append_input_grad(0, Some(*gy));
                if ctx.inputs().len() > 1 {
                    ctx.append_input_grad(1, Some(*gy));
                }
            }
            InPlaceOperation::SubAssign => {
                // Left input gets positive gradient, right gets negative
                ctx.append_input_grad(0, Some(*gy));
                if ctx.inputs().len() > 1 {
                    ctx.append_input_grad(1, Some(crate::tensor_ops::neg(gy)));
                }
            }
            InPlaceOperation::MulAssign => {
                // Product rule gradients
                if ctx.inputs().len() > 1 {
                    let right_input = ctx.input(1);
                    let left_input = ctx.input(0);
                    ctx.append_input_grad(0, Some((*gy) * right_input));
                    ctx.append_input_grad(1, Some((*gy) * left_input));
                } else {
                    ctx.append_input_grad(0, Some(*gy));
                }
            }
            InPlaceOperation::DivAssign => {
                // Division rule gradients
                if ctx.inputs().len() > 1 {
                    let right_input = ctx.input(1);
                    let left_input = ctx.input(0);
                    ctx.append_input_grad(0, Some((*gy) / right_input));
                    let neg_two = F::from(-2.0).unwrap();
                    let right_grad = crate::tensor_ops::neg(left_input)
                        * crate::tensor_ops::pow(right_input, neg_two)
                        * (*gy);
                    ctx.append_input_grad(1, Some(right_grad));
                } else {
                    ctx.append_input_grad(0, Some(*gy));
                }
            }
            InPlaceOperation::NegAssign => {
                // Gradient of negation
                ctx.append_input_grad(0, Some(crate::tensor_ops::neg(gy)));
            }
            InPlaceOperation::AbsAssign => {
                // Gradient of absolute value (sign of input)
                let input = ctx.input(0);
                let sign = crate::tensor_ops::sign(input);
                ctx.append_input_grad(0, Some((*gy) * sign));
            }
            InPlaceOperation::ScalarMulAssign | InPlaceOperation::ScalarAddAssign => {
                // For scalar operations, gradient flows through unchanged
                ctx.append_input_grad(0, Some(*gy));
                if ctx.inputs().len() > 1 {
                    // Scalar doesn't need gradient
                    ctx.append_input_grad(1, None);
                }
            }
        }
    }
}

/// Memory-efficient view operation that creates zero-copy views when possible
pub struct ViewOp {
    pub new_shape: Vec<usize>,
}

impl<F: Float> Op<F> for ViewOp {
    fn name(&self) -> &'static str {
        "MemoryEfficientView"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);

        // Try to create a view with the new shape
        let input_shape = input.shape();
        let input_size: usize = input_shape.iter().product();
        let new_size: usize = self.new_shape.iter().product();

        if input_size != new_size {
            return Err(OpError::IncompatibleShape(format!(
                "Cannot reshape array of size {} into shape {:?} (size {})",
                input_size, self.new_shape, new_size
            )));
        }

        // Create a reshaped view - this is zero-copy when possible
        let reshaped = input
            .view()
            .into_shape_with_order(IxDyn(&self.new_shape))
            .map_err(|_| OpError::IncompatibleShape("Cannot create view with new shape".into()))?;

        ctx.append_output(reshaped.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let input = ctx.input(0);

        // Gradient needs to be reshaped back to input shape
        let input_shape = crate::tensor_ops::shape(input);
        let reshaped_grad = crate::tensor_ops::reshape(gy, &input_shape);
        ctx.append_input_grad(0, Some(reshaped_grad));
    }
}

/// Public API functions for memory optimization
///
/// Get a buffer from the memory pool
pub fn get_pooled_buffer(size: usize) -> Vec<u8> {
    MEMORY_POOL.lock().unwrap().get_buffer(size)
}

/// Return a buffer to the memory pool  
pub fn return_pooled_buffer(buffer: Vec<u8>) {
    MEMORY_POOL.lock().unwrap().return_buffer(buffer);
}

/// Configure the memory pool
pub fn configure_memory_pool(max_buffers_per_size: usize, max_pool_memory: usize) {
    MEMORY_POOL
        .lock()
        .unwrap()
        .configure(max_buffers_per_size, max_pool_memory);
}

/// Enable or disable the memory pool
pub fn set_memory_pool_enabled(enabled: bool) {
    MEMORY_POOL.lock().unwrap().set_enabled(enabled);
}

/// Clear all buffers from the memory pool
pub fn clear_memory_pool() {
    MEMORY_POOL.lock().unwrap().clear();
}

/// Get memory pool statistics
pub fn get_memory_pool_stats() -> MemoryPoolStats {
    MEMORY_POOL.lock().unwrap().stats()
}

/// Enable memory usage tracking
pub fn enable_memory_tracking() {
    MEMORY_TRACKER.lock().unwrap().enable();
}

/// Disable memory usage tracking
pub fn disable_memory_tracking() {
    MEMORY_TRACKER.lock().unwrap().disable();
}

/// Reset memory tracking statistics
pub fn reset_memory_tracking() {
    MEMORY_TRACKER.lock().unwrap().reset();
}

/// Get memory tracking statistics
pub fn get_memory_tracking_stats() -> MemoryTrackerStats {
    MEMORY_TRACKER.lock().unwrap().get_stats()
}

/// Create a memory-efficient view of a tensor with a new shape
pub fn efficient_view<'g, F: Float>(tensor: &Tensor<'g, F>, new_shape: &[usize]) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(ViewOp {
            new_shape: new_shape.to_vec(),
        })
}

/// Perform in-place addition to reduce memory allocations
pub fn inplace_add<'g, F: Float>(lhs: &Tensor<'g, F>, rhs: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = lhs.graph();
    Tensor::builder(g)
        .append_input(lhs, false)
        .append_input(rhs, false)
        .build(InPlaceOp::new(InPlaceOperation::AddAssign))
}

/// Perform in-place subtraction to reduce memory allocations
pub fn inplace_sub<'g, F: Float>(lhs: &Tensor<'g, F>, rhs: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = lhs.graph();
    Tensor::builder(g)
        .append_input(lhs, false)
        .append_input(rhs, false)
        .build(InPlaceOp::new(InPlaceOperation::SubAssign))
}

/// Perform in-place multiplication to reduce memory allocations
pub fn inplace_mul<'g, F: Float>(lhs: &Tensor<'g, F>, rhs: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = lhs.graph();
    Tensor::builder(g)
        .append_input(lhs, false)
        .append_input(rhs, false)
        .build(InPlaceOp::new(InPlaceOperation::MulAssign))
}

/// Perform in-place division to reduce memory allocations
pub fn inplace_div<'g, F: Float>(lhs: &Tensor<'g, F>, rhs: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = lhs.graph();
    Tensor::builder(g)
        .append_input(lhs, false)
        .append_input(rhs, false)
        .build(InPlaceOp::new(InPlaceOperation::DivAssign))
}

/// Perform in-place negation to reduce memory allocations
pub fn inplace_neg<'g, F: Float>(tensor: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(InPlaceOp::new(InPlaceOperation::NegAssign))
}

/// Perform in-place absolute value to reduce memory allocations
pub fn inplace_abs<'g, F: Float>(tensor: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(InPlaceOp::new(InPlaceOperation::AbsAssign))
}

/// Perform in-place scalar multiplication to reduce memory allocations
pub fn inplace_scalar_mul<'g, F: Float>(
    tensor: &Tensor<'g, F>,
    scalar: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .append_input(scalar, false)
        .build(InPlaceOp::new(InPlaceOperation::ScalarMulAssign))
}

/// Memory-efficient tensor creation using the memory pool
pub fn efficient_zeros<'g, F: Float>(shape: &[usize], graph: &'g crate::Graph<F>) -> Tensor<'g, F> {
    // For now, use the standard zeros implementation
    // In a full implementation, this would use the memory pool
    crate::tensor_ops::zeros(
        &crate::tensor_ops::convert_to_tensor(
            ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[shape.len()]),
                shape
                    .iter()
                    .map(|&x| F::from(x).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            graph,
        ),
        graph,
    )
}

/// Memory-efficient tensor creation using the memory pool
pub fn efficient_ones<'g, F: Float>(shape: &[usize], graph: &'g crate::Graph<F>) -> Tensor<'g, F> {
    // For now, use the standard ones implementation
    // In a full implementation, this would use the memory pool
    crate::tensor_ops::ones(
        &crate::tensor_ops::convert_to_tensor(
            ndarray::Array::from_shape_vec(
                ndarray::IxDyn(&[shape.len()]),
                shape
                    .iter()
                    .map(|&x| F::from(x).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            graph,
        ),
        graph,
    )
}

/// A utility struct for memory optimization context
pub struct MemoryOptimizer;

impl MemoryOptimizer {
    /// Start a memory optimization session
    pub fn start_session() {
        enable_memory_tracking();
        set_memory_pool_enabled(true);
    }

    /// End a memory optimization session and return statistics
    pub fn end_session() -> (MemoryTrackerStats, MemoryPoolStats) {
        let tracking_stats = get_memory_tracking_stats();
        let pool_stats = get_memory_pool_stats();
        disable_memory_tracking();
        reset_memory_tracking();
        (tracking_stats, pool_stats)
    }

    /// Print memory optimization report
    pub fn print_report() {
        let tracking_stats = get_memory_tracking_stats();
        let pool_stats = get_memory_pool_stats();

        println!("=== Memory Optimization Report ===");
        println!(
            "Memory Pool: {} buffers, {} bytes pooled",
            pool_stats.total_buffers, pool_stats.total_pooled_memory
        );

        if !tracking_stats.current_usage.is_empty() {
            println!("Current Memory Usage by Operation:");
            for (op, usage) in &tracking_stats.current_usage {
                println!("  {}: {} bytes", op, usage);
            }
        }

        if !tracking_stats.peak_usage.is_empty() {
            println!("Peak Memory Usage by Operation:");
            for (op, usage) in &tracking_stats.peak_usage {
                println!("  {}: {} bytes", op, usage);
            }
        }

        if !tracking_stats.total_allocations.is_empty() {
            println!("Total Allocations by Operation:");
            for (op, total) in &tracking_stats.total_allocations {
                println!("  {}: {} bytes", op, total);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        let mut pool = MemoryPool::new();

        // Get a buffer
        let buffer = pool.get_buffer(1024);
        assert_eq!(buffer.len(), 1024);

        // Return it to the pool
        pool.return_buffer(buffer);

        // Get another buffer of the same size - should come from the pool
        let buffer2 = pool.get_buffer(1024);
        assert_eq!(buffer2.len(), 1024);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();
        tracker.enable();

        tracker.record_allocation("test_op", 1024);
        tracker.record_allocation("test_op", 512);

        let stats = tracker.get_stats();
        assert_eq!(stats.current_usage.get("test_op"), Some(&1536));
        assert_eq!(stats.peak_usage.get("test_op"), Some(&1536));
        assert_eq!(stats.total_allocations.get("test_op"), Some(&1536));

        tracker.record_deallocation("test_op", 512);
        let stats = tracker.get_stats();
        assert_eq!(stats.current_usage.get("test_op"), Some(&1024));
    }

    #[test]
    fn test_inplace_operations() {
        let add_op = InPlaceOp::<f32>::new(InPlaceOperation::AddAssign);
        assert_eq!(add_op.name(), "InPlaceAdd");

        let mul_op = InPlaceOp::<f32>::new(InPlaceOperation::MulAssign);
        assert_eq!(mul_op.name(), "InPlaceMul");

        let neg_op = InPlaceOp::<f32>::new(InPlaceOperation::NegAssign);
        assert_eq!(neg_op.name(), "InPlaceNeg");
    }

    #[test]
    fn test_view_op() {
        let view_op = ViewOp {
            new_shape: vec![2, 3],
        };
        assert_eq!(
            <ViewOp as crate::op::Op<f32>>::name(&view_op),
            "MemoryEfficientView"
        );
        assert_eq!(view_op.new_shape, vec![2, 3]);
    }

    #[test]
    fn test_public_api() {
        // Test that we can call the public API functions
        configure_memory_pool(8, 50 * 1024 * 1024);
        set_memory_pool_enabled(true);

        let buffer = get_pooled_buffer(1024);
        assert_eq!(buffer.len(), 1024);
        return_pooled_buffer(buffer);

        let stats = get_memory_pool_stats();
        assert!(stats.enabled);

        clear_memory_pool();
    }
}
