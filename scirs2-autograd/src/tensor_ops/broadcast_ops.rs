//! Broadcasting optimizations for tensor operations
//!
//! This module provides optimized broadcasting strategies to improve memory usage
//! and computational efficiency for element-wise operations between tensors of
//! different shapes.

use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops;
use crate::tensor_ops::binary_ops;
use crate::Float;
use scirs2_core::ndarray::{Array, IxDyn, Zip};
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
    pub outputshape: Vec<usize>,
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
#[allow(dead_code)]
pub fn analyze_broadcast(
    leftshape: &[usize],
    rightshape: &[usize],
) -> Result<BroadcastInfo, OpError> {
    // Check cache first
    let cache_key = (leftshape.to_vec(), rightshape.to_vec());
    if let Ok(cache) = BROADCAST_CACHE.lock() {
        if let Some(info) = cache.get(&cache_key) {
            return Ok(info.clone());
        }
    }

    let info = analyze_broadcast_impl(leftshape, rightshape)?;

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

#[allow(dead_code)]
fn analyze_broadcast_impl(
    leftshape: &[usize],
    rightshape: &[usize],
) -> Result<BroadcastInfo, OpError> {
    // Check for exact shape match (no broadcasting needed)
    if leftshape == rightshape {
        return Ok(BroadcastInfo {
            strategy: BroadcastStrategy::NoOp,
            outputshape: leftshape.to_vec(),
            left_needs_broadcast: false,
            right_needs_broadcast: false,
            left_reduce_axes: Vec::new(),
            right_reduce_axes: Vec::new(),
            memory_cost: leftshape.iter().product(),
        });
    }

    // Check for scalar cases
    let left_is_scalar =
        leftshape.is_empty() || leftshape == [1] || leftshape.iter().all(|&x| x == 1);
    let right_is_scalar =
        rightshape.is_empty() || rightshape == [1] || rightshape.iter().all(|&x| x == 1);

    if left_is_scalar || right_is_scalar {
        let outputshape = if left_is_scalar {
            rightshape.to_vec()
        } else {
            leftshape.to_vec()
        };
        return Ok(BroadcastInfo {
            strategy: BroadcastStrategy::ScalarBroadcast,
            outputshape: outputshape.clone(),
            left_needs_broadcast: left_is_scalar,
            right_needs_broadcast: right_is_scalar,
            left_reduce_axes: if left_is_scalar {
                (0..outputshape.len()).collect()
            } else {
                Vec::new()
            },
            right_reduce_axes: if right_is_scalar {
                (0..outputshape.len()).collect()
            } else {
                Vec::new()
            },
            memory_cost: outputshape.iter().product(),
        });
    }

    // Standard broadcasting rules
    let max_dims = leftshape.len().max(rightshape.len());
    let mut outputshape = Vec::with_capacity(max_dims);
    let mut left_reduce_axes = Vec::new();
    let mut right_reduce_axes = Vec::new();

    // Pad shapes to same length (broadcasting pads with 1s on the left)
    let left_padded = padshape_left(leftshape, max_dims);
    let right_padded = padshape_left(rightshape, max_dims);

    for i in 0..max_dims {
        let left_dim = left_padded[i];
        let right_dim = right_padded[i];

        if left_dim == right_dim {
            outputshape.push(left_dim);
        } else if left_dim == 1 {
            outputshape.push(right_dim);
            left_reduce_axes.push(i);
        } else if right_dim == 1 {
            outputshape.push(left_dim);
            right_reduce_axes.push(i);
        } else {
            return Err(OpError::IncompatibleShape(format!(
                "Cannot broadcast shapes {leftshape:?} and {rightshape:?}: dimension {i} incompatible ({left_dim} vs {right_dim})"
            )));
        }
    }

    // Choose strategy based on characteristics
    let memory_cost: usize = outputshape.iter().product();
    let strategy =
        choose_broadcast_strategy(&left_padded, &right_padded, &outputshape, memory_cost);

    Ok(BroadcastInfo {
        strategy,
        outputshape,
        left_needs_broadcast: !left_reduce_axes.is_empty() || leftshape.len() != max_dims,
        right_needs_broadcast: !right_reduce_axes.is_empty() || rightshape.len() != max_dims,
        left_reduce_axes,
        right_reduce_axes,
        memory_cost,
    })
}

#[allow(dead_code)]
fn padshape_left(shape: &[usize], targetlen: usize) -> Vec<usize> {
    let mut padded = vec![1; targetlen];
    let offset = targetlen - shape.len();
    padded[offset..].copy_from_slice(shape);
    padded
}

#[allow(dead_code)]
fn choose_broadcast_strategy(
    _leftshape: &[usize],
    _rightshape: &[usize],
    outputshape: &[usize],
    memory_cost: usize,
) -> BroadcastStrategy {
    // SIMD strategy for 1D arrays with compatible shapes
    if outputshape.len() == 1 && memory_cost <= 100_000 {
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
    phantom: std::marker::PhantomData<F>,
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
        leftshape: &[usize],
        rightshape: &[usize],
    ) -> Result<Self, OpError> {
        let info = analyze_broadcast(leftshape, rightshape)?;
        Ok(Self {
            operation,
            info,
            phantom: std::marker::PhantomData,
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
                apply_binary_op_sameshape(&left, &right, self.operation)?
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

        // Compute gradients based on operation type. These may still be shaped like the
        // (possibly-broadcast) output/`gy` rather than like `left_input`/`right_input`
        // themselves -- the reduction back to each operand's own shape happens
        // unconditionally below.
        let (left_grad, right_grad) = match self.operation {
            BinaryOperation::Add => (*gy, *gy),
            BinaryOperation::Subtract => (*gy, tensor_ops::neg(gy)),
            BinaryOperation::Multiply => ((*gy) * right_input, (*gy) * left_input),
            BinaryOperation::Divide => {
                let left_grad = (*gy) / right_input;
                let neg_two = F::from(-2.0).expect("Failed to convert constant to float");
                let right_grad =
                    tensor_ops::neg(left_input) * tensor_ops::pow(right_input, neg_two) * (*gy);
                (left_grad, right_grad)
            }
            BinaryOperation::Power => {
                // General elementwise tensor^tensor power (both operands may vary; this is
                // NOT `tensor_ops::pow`, which only raises to a fixed Rust-level scalar
                // exponent). For `z = x^y`:
                //   dz/dx = y * x^(y-1) = y * z / x
                //   dz/dy = x^y * ln(x) = z * ln(x)
                // Reusing the already-computed output `z` avoids needing a dedicated
                // tensor-tensor power primitive just for this derivative.
                let output = *ctx.output();
                let left_grad = (*gy) * right_input * (output / left_input);
                let right_grad = (*gy) * output * tensor_ops::ln(left_input);
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

        // Reduce each contribution back down to its own operand's actual shape.
        //
        // This MUST be computed dynamically -- mirroring `AddOp`/`SubOp`/`MulOp`/`DivOp`
        // in `binary_ops.rs`, which build a runtime `shape(x)` node and reduce against it
        // via `MaybeReduceSum` -- rather than trusting `self.info.{left,right}_reduce_axes`.
        // `broadcast_binary_op` (below) builds `self.info` from hardcoded placeholder
        // shapes, so those fields never reflect the operands actually bound to this node;
        // using them here silently returned a gradient shaped like `gy`/the output instead
        // of like the operand, which later fails (or corrupts an accumulation) wherever
        // that operand's gradient is consumed at its own shape.
        let left_final = binary_ops::maybe_reduce(&tensor_ops::shape(left_input), &left_grad, g);
        let right_final = binary_ops::maybe_reduce(&tensor_ops::shape(right_input), &right_grad, g);

        ctx.append_input_grad(0, Some(left_final));
        ctx.append_input_grad(1, Some(right_final));
    }
}

/// Apply binary operation when tensors have the same shape
#[allow(dead_code)]
fn apply_binary_op_sameshape<'a, F: Float>(
    left: &NdArrayView<'a, F>,
    right: &NdArrayView<'a, F>,
    op: BinaryOperation,
) -> Result<NdArray<F>, OpError> {
    match op {
        BinaryOperation::Add => Ok(left + right),
        BinaryOperation::Subtract => Ok(left - right),
        BinaryOperation::Multiply => Ok(left * right),
        BinaryOperation::Divide => Ok(left / right),
        // `+`/`-`/`*`//` broadcast correctly via `ndarray`'s own operator overloads even
        // when `left`/`right` differ in shape (see `ndarray::impl_ops`), but there is no
        // built-in broadcasting operator for an arbitrary closure, so `powf`/`max`/`min`
        // need their own broadcasting helper.
        BinaryOperation::Power => broadcast_elementwise(left, right, |a, b| a.powf(b)),
        BinaryOperation::Maximum => broadcast_elementwise(left, right, |a, b| a.max(b)),
        BinaryOperation::Minimum => broadcast_elementwise(left, right, |a, b| a.min(b)),
    }
}

/// Applies `f` element-wise to `left` and `right`, broadcasting either operand up to
/// their common shape first (matching `numpy`/`ndarray`'s broadcasting rules).
///
/// [`BinaryOperation::Power`], [`BinaryOperation::Maximum`] and [`BinaryOperation::Minimum`]
/// used to zip `left.iter()` directly with `right.iter()`. That silently truncates to the
/// shorter of the two *flat* iterators whenever the shapes differ -- exactly the case this
/// module exists for -- and then panics (`Array::from_shape_vec(...).expect(...)`)
/// reshaping the truncated result back into `left`'s full shape. Reusing
/// [`analyze_broadcast`] keeps the notion of "the broadcast output shape" identical to the
/// one this module's gradient-shape metadata is computed from.
fn broadcast_elementwise<F: Float>(
    left: &NdArrayView<F>,
    right: &NdArrayView<F>,
    f: impl Fn(F, F) -> F,
) -> Result<NdArray<F>, OpError> {
    if left.shape() == right.shape() {
        return Ok(Zip::from(left).and(right).map_collect(|&a, &b| f(a, b)));
    }
    let outputshape = analyze_broadcast(left.shape(), right.shape())?.outputshape;
    let left_b = left.broadcast(outputshape.as_slice()).ok_or_else(|| {
        OpError::IncompatibleShape(format!(
            "Cannot broadcast shape {:?} to {outputshape:?}",
            left.shape()
        ))
    })?;
    let right_b = right.broadcast(outputshape.as_slice()).ok_or_else(|| {
        OpError::IncompatibleShape(format!(
            "Cannot broadcast shape {:?} to {outputshape:?}",
            right.shape()
        ))
    })?;
    Ok(Zip::from(&left_b)
        .and(&right_b)
        .map_collect(|&a, &b| f(a, b)))
}

/// Apply binary operation with scalar broadcasting
#[allow(dead_code)]
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
#[allow(dead_code)]
fn apply_simd_broadcast<'a, F: Float>(
    left: &NdArrayView<'a, F>,
    right: &NdArrayView<'a, F>,
    op: BinaryOperation,
    info: &BroadcastInfo,
) -> Result<NdArray<F>, OpError> {
    // For now, fall back to standard broadcasting
    // In a full implementation, this would use SIMD instructions
    apply_standard_broadcast(left, right, op)
}

/// Apply chunked broadcasting for large tensors
#[allow(dead_code)]
fn apply_chunked_broadcast<'a, F: Float>(
    left: &NdArrayView<'a, F>,
    right: &NdArrayView<'a, F>,
    op: BinaryOperation,
    info: &BroadcastInfo,
) -> Result<NdArray<F>, OpError> {
    const CHUNK_SIZE: usize = 100_000; // Process in chunks of 100k elements

    // Create output array
    let _output = Array::<F, IxDyn>::zeros(IxDyn(&info.outputshape));

    // Process in chunks to manage memory usage
    let total_elements = info.outputshape.iter().product::<usize>();
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
#[allow(dead_code)]
fn apply_standard_broadcast<'a, F: Float>(
    left: &NdArrayView<'a, F>,
    right: &NdArrayView<'a, F>,
    op: BinaryOperation,
) -> Result<NdArray<F>, OpError> {
    // NOTE: this used to broadcast `left` to `right`'s shape AND `right` to `left`'s shape
    // unconditionally (`left.broadcast(right.shape())?` followed by
    // `right.broadcast(left.shape())?`). `ArrayView::broadcast` only ever *grows* a shape
    // (adds/repeats dimensions), so for the very asymmetric-shape case this module exists
    // for (e.g. left=[3,4], right=[4]) the first call -- broadcasting the *larger* [3,4]
    // down to [4] -- always failed, short-circuiting the whole function via `?` before the
    // correct direction was ever tried. Delegating to `apply_binary_op_sameshape` computes
    // the true broadcast output shape once (via `analyze_broadcast`) and only grows each
    // operand up to it.
    apply_binary_op_sameshape(left, right, op)
}

/// Public API functions for optimized broadcasting
///
/// Add tensors with optimized broadcasting
#[allow(dead_code)]
pub fn broadcast_add<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Add)
}

/// Subtract tensors with optimized broadcasting
#[allow(dead_code)]
pub fn broadcast_sub<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Subtract)
}

/// Multiply tensors with optimized broadcasting
#[allow(dead_code)]
pub fn broadcast_mul<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Multiply)
}

/// Divide tensors with optimized broadcasting
#[allow(dead_code)]
pub fn broadcast_div<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Divide)
}

/// Power tensors with optimized broadcasting
#[allow(dead_code)]
pub fn broadcast_pow<'g, F: Float>(left: &Tensor<'g, F>, right: &Tensor<'g, F>) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Power)
}

/// Element-wise maximum with optimized broadcasting
#[allow(dead_code)]
pub fn broadcast_maximum<'g, F: Float>(
    left: &Tensor<'g, F>,
    right: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Maximum)
}

/// Element-wise minimum with optimized broadcasting
#[allow(dead_code)]
pub fn broadcast_minimum<'g, F: Float>(
    left: &Tensor<'g, F>,
    right: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    broadcast_binary_op(left, right, BinaryOperation::Minimum)
}

#[allow(dead_code)]
fn broadcast_binary_op<'g, F: Float>(
    left: &Tensor<'g, F>,
    right: &Tensor<'g, F>,
    operation: BinaryOperation,
) -> Tensor<'g, F> {
    let g = left.graph();

    // NOTE: these are deliberately NOT `left.shape()`/`right.shape()`. `Tensor::shape()`
    // only reports a "known shape" hint that is set by *some* ops at construction time and
    // is `vec![]` (empty) whenever it isn't -- and an empty shape is indistinguishable here
    // from a genuine scalar. Feeding a wrong/stale hint into `analyze_broadcast` could
    // select `BroadcastStrategy::ScalarBroadcast`, whose `compute()` path
    // (`apply_scalar_broadcast`) then extracts a *single* element and applies it to the
    // whole other operand -- silently discarding real data if the tensor was not actually
    // a scalar. Using a fixed, always-equal placeholder instead forces
    // `analyze_broadcast_impl` down its `leftshape == rightshape` branch every time, i.e.
    // `BroadcastStrategy::NoOp` unconditionally, whose `compute()` path
    // (`apply_binary_op_sameshape`) delegates to plain `ndarray` operator overloads —
    // which broadcast correctly on their own (see `ndarray::impl_ops`) regardless of the
    // operands' actual shapes. So `compute()` is correct today independent of this
    // placeholder. `grad()` (above) does NOT read `self.info` for its shape-reduction
    // decisions for the same reason -- see its own comment -- so nothing downstream
    // depends on this analysis being accurate either.
    let leftshape = vec![1];
    let rightshape = vec![1];

    // Create optimized operation
    if let Ok(op) = OptimizedBroadcastOp::new(operation, &leftshape, &rightshape) {
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
#[allow(dead_code)]
pub fn clear_broadcast_cache() {
    if let Ok(mut cache) = BROADCAST_CACHE.lock() {
        cache.clear();
    }
}

/// Get broadcast cache statistics
#[allow(dead_code)]
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
    fn test_broadcast_analysis_sameshape() {
        let leftshape = vec![3, 4];
        let rightshape = vec![3, 4];
        let info = analyze_broadcast(&leftshape, &rightshape).expect("Operation failed");

        assert_eq!(info.strategy, BroadcastStrategy::NoOp);
        assert!(!info.left_needs_broadcast);
        assert!(!info.right_needs_broadcast);
        assert_eq!(info.outputshape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_analysis_scalar() {
        let leftshape = vec![];
        let rightshape = vec![3, 4];
        let info = analyze_broadcast(&leftshape, &rightshape).expect("Operation failed");

        assert_eq!(info.strategy, BroadcastStrategy::ScalarBroadcast);
        assert!(info.left_needs_broadcast);
        assert!(!info.right_needs_broadcast);
        assert_eq!(info.outputshape, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_analysis_incompatible() {
        let leftshape = vec![3, 4];
        let rightshape = vec![2, 4];
        let result = analyze_broadcast(&leftshape, &rightshape);

        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_analysis_compatible() {
        let leftshape = vec![1, 4];
        let rightshape = vec![3, 1];
        let info = analyze_broadcast(&leftshape, &rightshape).expect("Operation failed");

        assert_eq!(info.outputshape, vec![3, 4]);
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
