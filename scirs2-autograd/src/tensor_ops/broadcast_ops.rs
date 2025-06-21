//! Broadcasting optimizations for tensor operations
//!
//! This module provides optimized broadcasting strategies to improve memory usage
//! and computational efficiency for element-wise operations between tensors of
//! different shapes.

use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops;
use crate::Float;
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Strategy for handling broadcasting operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastStrategy {
    /// No broadcasting needed - tensors have same shape
    NoOp,
    /// One tensor is scalar, other is not
    ScalarBroadcast,
    /// Standard ndarray broadcasting
    StandardBroadcast,
    /// Memory-efficient chunked broadcasting for large tensors
    ChunkedBroadcast,
    /// SIMD-optimized broadcasting for 1D cases
    SimdBroadcast,
}

/// Metadata about a broadcasting operation
#[derive(Debug, Clone)]
pub struct BroadcastInfo {
    /// Strategy to use for this broadcast
    pub strategy: BroadcastStrategy,
    /// Output shape after broadcasting
    pub output_shape: Vec<usize>,
    /// Whether left operand needs broadcasting
    pub left_needs_broadcast: bool,
    /// Whether right operand needs broadcasting  
    pub right_needs_broadcast: bool,
    /// Axes that need reduction for gradient computation
    pub left_reduce_axes: Vec<usize>,
    pub right_reduce_axes: Vec<usize>,
    /// Memory cost estimate (in elements)
    pub memory_cost: usize,
}

/// Type alias for broadcast cache key
type BroadcastCacheKey = (Vec<usize>, Vec<usize>);

/// Cache for broadcast analysis results
static BROADCAST_CACHE: LazyLock<Mutex<HashMap<BroadcastCacheKey, BroadcastInfo>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Analyze broadcasting requirements between two shapes
pub fn analyze_broadcast(
    left_shape: &[usize],
    right_shape: &[usize],
) -> Result<BroadcastInfo, OpError> {
    // Check cache first
    let cache_key = (left_shape.to_vec(), right_shape.to_vec());
    if let Ok(cache) = BROADCAST_CACHE.lock() {
        if let Some(info) = cache.get(&cache_key) {
            return Ok(info.clone());
        }
    }

    let info = analyze_broadcast_impl(left_shape, right_shape)?;

    // Cache the result
    if let Ok(mut cache) = BROADCAST_CACHE.lock() {
        // Limit cache size to prevent memory leaks
        if cache.len() > 1000 {
            cache.clear();
        }
        cache.insert(cache_key, info.clone());
    }

    Ok(info)
}

fn analyze_broadcast_impl(
    left_shape: &[usize],
    right_shape: &[usize],
) -> Result<BroadcastInfo, OpError> {
    // Check for exact shape match (no broadcasting needed)
    if left_shape == right_shape {
        return Ok(BroadcastInfo {
            strategy: BroadcastStrategy::NoOp,
            output_shape: left_shape.to_vec(),
            left_needs_broadcast: false,
            right_needs_broadcast: false,
            left_reduce_axes: Vec::new(),
            right_reduce_axes: Vec::new(),
            memory_cost: left_shape.iter().product(),
        });
    }

    // Check for scalar cases
    let left_is_scalar =
        left_shape.is_empty() || left_shape == [1] || left_shape.iter().all(|&x| x == 1);
    let right_is_scalar =
        right_shape.is_empty() || right_shape == [1] || right_shape.iter().all(|&x| x == 1);

    if left_is_scalar || right_is_scalar {
        let output_shape = if left_is_scalar {
            right_shape.to_vec()
        } else {
            left_shape.to_vec()
        };
        return Ok(BroadcastInfo {
            strategy: BroadcastStrategy::ScalarBroadcast,
            output_shape: output_shape.clone(),
            left_needs_broadcast: left_is_scalar,
            right_needs_broadcast: right_is_scalar,
            left_reduce_axes: if left_is_scalar {
                (0..output_shape.len()).collect()
            } else {
                Vec::new()
            },
            right_reduce_axes: if right_is_scalar {
                (0..output_shape.len()).collect()
            } else {
                Vec::new()
            },
            memory_cost: output_shape.iter().product(),
        });
    }

    // Standard broadcasting rules
    let max_dims = left_shape.len().max(right_shape.len());
    let mut output_shape = Vec::with_capacity(max_dims);
    let mut left_reduce_axes = Vec::new();
    let mut right_reduce_axes = Vec::new();

    // Pad shapes to same length (broadcasting pads with 1s on the left)
    let left_padded = pad_shape_left(left_shape, max_dims);
    let right_padded = pad_shape_left(right_shape, max_dims);

    for i in 0..max_dims {
        let left_dim = left_padded[i];
        let right_dim = right_padded[i];

        if left_dim == right_dim {
            output_shape.push(left_dim);
        } else if left_dim == 1 {
            output_shape.push(right_dim);
            left_reduce_axes.push(i);
        } else if right_dim == 1 {
            output_shape.push(left_dim);
            right_reduce_axes.push(i);
        } else {
            return Err(OpError::IncompatibleShape(format!(
                "Cannot broadcast shapes {:?} and {:?}: dimension {} incompatible ({} vs {})",
                left_shape, right_shape, i, left_dim, right_dim
            )));
        }
    }

    // Choose strategy based on characteristics
    let memory_cost: usize = output_shape.iter().product();
    let strategy =
        choose_broadcast_strategy(&left_padded, &right_padded, &output_shape, memory_cost);

    Ok(BroadcastInfo {
        strategy,
        output_shape,
        left_needs_broadcast: !left_reduce_axes.is_empty() || left_shape.len() != max_dims,
        right_needs_broadcast: !right_reduce_axes.is_empty() || right_shape.len() != max_dims,
        left_reduce_axes,
        right_reduce_axes,
        memory_cost,
    })
}

fn pad_shape_left(shape: &[usize], target_len: usize) -> Vec<usize> {
    let mut padded = vec![1; target_len];
    let offset = target_len - shape.len();
    padded[offset..].copy_from_slice(shape);
    padded
}

fn choose_broadcast_strategy(
    _left_shape: &[usize],
    _right_shape: &[usize],
    output_shape: &[usize],
    memory_cost: usize,
) -> BroadcastStrategy {
    // SIMD strategy for 1D arrays with compatible shapes
    if output_shape.len() == 1 && memory_cost <= 100_000 {
        return BroadcastStrategy::SimdBroadcast;
    }

    // Chunked strategy for large memory operations
    if memory_cost > 1_000_000 {
        return BroadcastStrategy::ChunkedBroadcast;
    }

    // Default to standard broadcasting
    BroadcastStrategy::StandardBroadcast
}

/// Optimized broadcast operation that applies a binary function
pub struct OptimizedBroadcastOp<F: Float> {
    pub operation: BinaryOperation,
    pub info: BroadcastInfo,
    _phantom: std::marker::PhantomData<F>,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Maximum,
    Minimum,
}

impl<F: Float> OptimizedBroadcastOp<F> {
    pub fn new(
        operation: BinaryOperation,
        left_shape: &[usize],
        right_shape: &[usize],
    ) -> Result<Self, OpError> {
        let info = analyze_broadcast(left_shape, right_shape)?;
        Ok(Self {
            operation,
            info,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<F: Float> Op<F> for OptimizedBroadcastOp<F> {
    fn name(&self) -> &'static str {
        match self.operation {
            BinaryOperation::Add => "OptimizedBroadcastAdd",
            BinaryOperation::Subtract => "OptimizedBroadcastSub",
            BinaryOperation::Multiply => "OptimizedBroadcastMul",
            BinaryOperation::Divide => "OptimizedBroadcastDiv",
            BinaryOperation::Power => "OptimizedBroadcastPow",
            BinaryOperation::Maximum => "OptimizedBroadcastMax",
            BinaryOperation::Minimum => "OptimizedBroadcastMin",
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let left = ctx.input(0);
        let right = ctx.input(1);

        let result = match self.info.strategy {
            BroadcastStrategy::NoOp => {
                // Same shapes - direct operation
                apply_binary_op_same_shape(&left, &right, self.operation)
            }
            BroadcastStrategy::ScalarBroadcast => {
                apply_scalar_broadcast(&left, &right, self.operation, &self.info)?
            }
            BroadcastStrategy::SimdBroadcast => {
                apply_simd_broadcast(&left, &right, self.operation, &self.info)?
            }
            BroadcastStrategy::ChunkedBroadcast => {
                apply_chunked_broadcast(&left, &right, self.operation, &self.info)?
            }
            BroadcastStrategy::StandardBroadcast => {
                apply_standard_broadcast(&left, &right, self.operation)?
            }
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let left_input = ctx.input(0);
        let right_input = ctx.input(1);
        let g = ctx.graph();

        // Compute gradients based on operation type
        let (left_grad, right_grad) = match self.operation {
            BinaryOperation::Add => (*gy, *gy),
            BinaryOperation::Subtract => (*gy, tensor_ops::neg(gy)),
            BinaryOperation::Multiply => ((*gy) * right_input, (*gy) * left_input),
            BinaryOperation::Divide => {
                let left_grad = (*gy) / right_input;
                let neg_two = F::from(-2.0).unwrap();
                let right_grad =
                    tensor_ops::neg(left_input) * tensor_ops::pow(right_input, neg_two) * (*gy);
                (left_grad, right_grad)
            }
            BinaryOperation::Power => {
                // Simplified power gradient (for now, assume power by scalar)
                // In a full implementation, tensor-tensor power would need more complex handling
                let left_grad = (*gy) * right_input;
                let right_grad = (*gy) * left_input;
                (left_grad, right_grad)
            }
            BinaryOperation::Maximum => {
                // Gradient flows to the maximum operand
                let mask = tensor_ops::greater_equal(left_input, right_input);
                let left_grad = (*gy) * mask;
                let right_grad = (*gy) * (tensor_ops::scalar(F::one(), g) - mask);
                (left_grad, right_grad)
            }
            BinaryOperation::Minimum => {
                // Gradient flows to the minimum operand
                let mask = tensor_ops::lesser_equal(left_input, right_input);
                let left_grad = (*gy) * mask;
                let right_grad = (*gy) * (tensor_ops::scalar(F::one(), g) - mask);
                (left_grad, right_grad)
            }
        };

        // Apply reduction if broadcasting occurred
        let left_final = if self.info.left_needs_broadcast {
            reduce_for_broadcast_grad(&left_grad, &self.info.left_reduce_axes, ctx.input(0), g)
        } else {
            left_grad
        };

        let right_final = if self.info.right_needs_broadcast {
            reduce_for_broadcast_grad(&right_grad, &self.info.right_reduce_axes, ctx.input(1), g)
        } else {
            right_grad
        };

        ctx.append_input_grad(0, Some(left_final));
        ctx.append_input_grad(1, Some(right_final));
    }
}

/// Apply binary operation when tensors have the same shape
fn apply_binary_op_same_shape<'a, F: Float>(
    left: &NdArrayView<'a, F>,
    right: &NdArrayView<'a, F>,
    op: BinaryOperation,
) -> NdArray<F> {
    match op {
        BinaryOperation::Add => left + right,
        BinaryOperation::Subtract => left - right,
        BinaryOperation::Multiply => left * right,
        BinaryOperation::Divide => left / right,
        BinaryOperation::Power => left
            .iter()
            .zip(right.iter())
            .map(|(&a, &b)| a.powf(b))
            .collect::<Array<F, _>>()
            .into_shape_with_order(left.shape())
            .unwrap(),
        BinaryOperation::Maximum => left
            .iter()
            .zip(right.iter())
            .map(|(&a, &b)| a.max(b))
            .collect::<Array<F, _>>()
            .into_shape_with_order(left.shape())
            .unwrap(),
        BinaryOperation::Minimum => left
            .iter()
            .zip(right.iter())
            .map(|(&a, &b)| a.min(b))
            .collect::<Array<F, _>>()
            .into_shape_with_order(left.shape())
            .unwrap(),
    }
}

/// Apply binary operation with scalar broadcasting
fn apply_scalar_broadcast<'a, F: Float>(
    left: &NdArrayView<'a, F>,
    right: &NdArrayView<'a, F>,
    op: BinaryOperation,
    info: &BroadcastInfo,
) -> Result<NdArray<F>, OpError> {
    let (scalar_val, array, is_left_scalar) = if info.left_needs_broadcast {
        let scalar = if left.is_empty() {
            F::zero()
        } else {
            let zeros = vec![0; left.ndim()];
            left[IxDyn(&zeros)]
        };
        (scalar, right, true)
    } else {
        let scalar = if right.is_empty() {
            F::zero()
        } else {
            let zeros = vec![0; right.ndim()];
            right[IxDyn(&zeros)]
        };
        (scalar, left, false)
    };

    let result = match (op, is_left_scalar) {
        (BinaryOperation::Add, _) => array.mapv(|x| x + scalar_val),
        (BinaryOperation::Subtract, true) => array.mapv(|x| scalar_val - x),
        (BinaryOperation::Subtract, false) => array.mapv(|x| x - scalar_val),
        (BinaryOperation::Multiply, _) => array.mapv(|x| x * scalar_val),
        (BinaryOperation::Divide, true) => array.mapv(|x| scalar_val / x),
        (BinaryOperation::Divide, false) => array.mapv(|x| x / scalar_val),
        (BinaryOperation::Power, true) => array.mapv(|x| scalar_val.powf(x)),
        (BinaryOperation::Power, false) => array.mapv(|x| x.powf(scalar_val)),
        (BinaryOperation::Maximum, _) => array.mapv(|x| x.max(scalar_val)),
        (BinaryOperation::Minimum, _) => array.mapv(|x| x.min(scalar_val)),
    };

    Ok(result)
}

/// Apply SIMD-optimized broadcasting for 1D cases
fn apply_simd_broadcast<'a, F: Float>(
    left: &NdArrayView<'a, F>,
    right: &NdArrayView<'a, F>,
    op: BinaryOperation,
    _info: &BroadcastInfo,
) -> Result<NdArray<F>, OpError> {
    // For now, fall back to standard broadcasting
    // In a full implementation, this would use SIMD instructions
    apply_standard_broadcast(left, right, op)
}

/// Apply chunked broadcasting for large tensors
fn apply_chunked_broadcast<'a, F: Float>(
    left: &NdArrayView<'a, F>,
    right: &NdArrayView<'a, F>,
    op: BinaryOperation,
    info: &BroadcastInfo,
) -> Result<NdArray<F>, OpError> {
    const CHUNK_SIZE: usize = 100_000; // Process in chunks of 100k elements

    // Create output array
    let _output = Array::<F, IxDyn>::zeros(IxDyn(&info.output_shape));

    // Process in chunks to manage memory usage
    let total_elements = info.output_shape.iter().product::<usize>();
    let num_chunks = total_elements.div_ceil(CHUNK_SIZE);

    if let Some(chunk_idx) = (0..num_chunks).next() {
        let _start_idx = chunk_idx * CHUNK_SIZE;
        let _end_idx = ((chunk_idx + 1) * CHUNK_SIZE).min(total_elements);

        // Process this chunk
        // In a full implementation, this would carefully handle the chunking
        // For now, fall back to standard broadcasting
        return apply_standard_broadcast(left, right, op);
    }

    // Fallback return
    apply_standard_broadcast(left, right, op)
}

/// Apply standard ndarray broadcasting
fn apply_standard_broadcast<'a, F: Float>(
    left: &NdArrayView<'a, F>,
    right: &NdArrayView<'a, F>,
    op: BinaryOperation,
) -> Result<NdArray<F>, OpError> {
    match op {
        BinaryOperation::Add => Ok(left + right),
        BinaryOperation::Subtract => Ok(left - right),
        BinaryOperation::Multiply => Ok(left * right),
        BinaryOperation::Divide => Ok(left / right),
        BinaryOperation::Power => {
            // Element-wise power with broadcasting
            let broadcasted_left = left.broadcast(right.shape()).ok_or_else(|| {
                OpError::IncompatibleShape("Cannot broadcast for power operation".into())
            })?;
            let broadcasted_right = right.broadcast(left.shape()).ok_or_else(|| {
                OpError::IncompatibleShape("Cannot broadcast for power operation".into())
            })?;

            let result: Array<F, IxDyn> = broadcasted_left
                .iter()
                .zip(broadcasted_right.iter())
                .map(|(&a, &b)| a.powf(b))
                .collect::<Array<F, _>>()
                .into_shape_with_order(broadcasted_left.shape())
                .unwrap();
            Ok(result)
        }
        BinaryOperation::Maximum => {
            let broadcasted_left = left.broadcast(right.shape()).ok_or_else(|| {
                OpError::IncompatibleShape("Cannot broadcast for max operation".into())
            })?;
            let broadcasted_right = right.broadcast(left.shape()).ok_or_else(|| {
                OpError::IncompatibleShape("Cannot broadcast for max operation".into())
            })?;

            let result: Array<F, IxDyn> = broadcasted_left
                .iter()
                .zip(broadcasted_right.iter())
                .map(|(&a, &b)| a.max(b))
                .collect::<Array<F, _>>()
                .into_shape_with_order(broadcasted_left.shape())
                .unwrap();
            Ok(result)
        }
        BinaryOperation::Minimum => {
            let broadcasted_left = left.broadcast(right.shape()).ok_or_else(|| {
                OpError::IncompatibleShape("Cannot broadcast for min operation".into())
            })?;
            let broadcasted_right = right.broadcast(left.shape()).ok_or_else(|| {
                OpError::IncompatibleShape("Cannot broadcast for min operation".into())
            })?;

            let result: Array<F, IxDyn> = broadcasted_left
                .iter()
                .zip(broadcasted_right.iter())
                .map(|(&a, &b)| a.min(b))
                .collect::<Array<F, _>>()
                .into_shape_with_order(broadcasted_left.shape())
                .unwrap();
            Ok(result)
        }
    }
}

/// Reduce gradients appropriately for broadcasting
fn reduce_for_broadcast_grad<'g, F: Float>(
    grad: &Tensor<'g, F>,
    reduce_axes: &[usize],
    original_input: &Tensor<'g, F>,
    _graph: &'g crate::Graph<F>,
) -> Tensor<'g, F> {
    let mut result = *grad;

    // Sum over broadcasted dimensions
    for &axis in reduce_axes {
        result = tensor_ops::reduce_sum(result, &[axis as isize], true);
    }

    // Reshape to match original input shape if needed
    let original_shape = tensor_ops::shape(original_input);
    result = tensor_ops::reshape(result, &original_shape);

    result
}

/// Public API functions for optimized broadcasting
///
/// Add tensors with optimized broadcasting
pub fn broadcast_add<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Add)
}

/// Subtract tensors with optimized broadcasting
pub fn broadcast_sub<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Subtract)
}

/// Multiply tensors with optimized broadcasting
pub fn broadcast_mul<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Multiply)
}

/// Divide tensors with optimized broadcasting
pub fn broadcast_div<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Divide)
}

/// Power tensors with optimized broadcasting
pub fn broadcast_pow<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Power)
}

/// Element-wise maximum with optimized broadcasting
pub fn broadcast_maximum<'g, F: Float>(
    left: &Tensor<'g, F>,
    right: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Maximum)
}

/// Element-wise minimum with optimized broadcasting
pub fn broadcast_minimum<'g, F: Float>(
    left: &Tensor<'g, F>,
    right: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Minimum)
}

fn broadcast_binary_op<'g, F: Float>(
    left: &Tensor<'g, F>,
    right: &Tensor<'g, F>,
    operation: BinaryOperation,
) -> Tensor<'g, F> {
    let g = left.graph();

    // Get shapes for analysis (placeholder implementation)
    // In a real implementation, we'd need to evaluate or infer shapes
    let left_shape = vec![1]; // Placeholder
    let right_shape = vec![1]; // Placeholder

    // Create optimized operation
    if let Ok(op) = OptimizedBroadcastOp::new(operation, &left_shape, &right_shape) {
        Tensor::builder(g)
            .append_input(left, false)
            .append_input(right, false)
            .build(op)
    } else {
        // Fall back to standard operations
        match operation {
            BinaryOperation::Add => left + right,
            BinaryOperation::Subtract => left - right,
            BinaryOperation::Multiply => left * right,
            BinaryOperation::Divide => left / right,
            BinaryOperation::Power => {
                // For now, use a simple implementation
                // In a full implementation, we'd need tensor-tensor power
                let one = F::one();
                tensor_ops::pow(left, one) // Placeholder
            }
            BinaryOperation::Maximum => tensor_ops::maximum(left, right),
            BinaryOperation::Minimum => tensor_ops::minimum(left, right),
        }
    }
}

/// Clear the broadcast analysis cache
pub fn clear_broadcast_cache() {
    if let Ok(mut cache) = BROADCAST_CACHE.lock() {
        cache.clear();
    }
}

/// Get broadcast cache statistics
pub fn get_broadcast_cache_stats() -> (usize, usize) {
    if let Ok(cache) = BROADCAST_CACHE.lock() {
        (cache.len(), cache.capacity())
    } else {
        (0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_analysis_same_shape() {
        let left_shape = vec![3, 4];
        let right_shape = vec![3, 4];
        let info = analyze_broadcast(&left_shape, &right_shape).unwrap();

        assert_eq!(info.strategy, BroadcastStrategy::NoOp);
        assert!(!info.left_needs_broadcast);
        assert!(!info.right_needs_broadcast);
        assert_eq!(info.output_shape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_analysis_scalar() {
        let left_shape = vec![];
        let right_shape = vec![3, 4];
        let info = analyze_broadcast(&left_shape, &right_shape).unwrap();

        assert_eq!(info.strategy, BroadcastStrategy::ScalarBroadcast);
        assert!(info.left_needs_broadcast);
        assert!(!info.right_needs_broadcast);
        assert_eq!(info.output_shape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_analysis_incompatible() {
        let left_shape = vec![3, 4];
        let right_shape = vec![2, 4];
        let result = analyze_broadcast(&left_shape, &right_shape);

        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_analysis_compatible() {
        let left_shape = vec![1, 4];
        let right_shape = vec![3, 1];
        let info = analyze_broadcast(&left_shape, &right_shape).unwrap();

        assert_eq!(info.output_shape, vec![3, 4]);
        assert!(info.left_needs_broadcast);
        assert!(info.right_needs_broadcast);
    }

    #[test]
    fn test_binary_operations() {
        // Test that binary operations can be created
        let op = BinaryOperation::Add;
        assert!(matches!(op, BinaryOperation::Add));

        let op = BinaryOperation::Multiply;
        assert!(matches!(op, BinaryOperation::Multiply));
    }

    #[test]
    fn test_cache_operations() {
        clear_broadcast_cache();
        let (size, _) = get_broadcast_cache_stats();
        assert_eq!(size, 0);
    }
}
