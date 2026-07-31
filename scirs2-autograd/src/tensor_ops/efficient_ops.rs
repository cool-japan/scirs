//! Efficient tensor operations with optimized reshape and slice implementations
//!
//! This module provides memory-efficient and performance-optimized implementations
//! of common tensor operations, particularly focusing on reshape and slice operations
//! that minimize memory allocations and maximize cache efficiency.

use crate::ndarray_ext::NdArrayView;
use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use scirs2_core::ndarray::{Array, IxDyn, SliceInfo, SliceInfoElem};
use std::sync::{LazyLock, Mutex};

/// Cache for storing reshape operation metadata
static RESHAPE_CACHE: LazyLock<Mutex<ReshapeCache>> =
    LazyLock::new(|| Mutex::new(ReshapeCache::new()));

/// Cache for reshape operations to avoid recomputing stride information
struct ReshapeCache {
    /// Cached reshape transformations: (fromshape, toshape) -> is_contiguous
    cache: std::collections::HashMap<(Vec<usize>, Vec<usize>), bool>,
    /// Maximum cache size
    max_size: usize,
}

impl ReshapeCache {
    fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size: 1000,
        }
    }

    fn get(&mut self, fromshape: &[usize], toshape: &[usize]) -> Option<bool> {
        self.cache
            .get(&(fromshape.to_vec(), toshape.to_vec()))
            .copied()
    }

    fn insert(&mut self, fromshape: &[usize], toshape: &[usize], iscontiguous: bool) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: clear half the cache
            let keys_to_remove: Vec<_> =
                self.cache.keys().take(self.max_size / 2).cloned().collect();
            for key in keys_to_remove {
                self.cache.remove(&key);
            }
        }
        self.cache
            .insert((fromshape.to_vec(), toshape.to_vec()), iscontiguous);
    }

    fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Efficient reshape operation that minimizes memory allocations
///
/// This operation optimizes reshape by:
/// 1. Using zero-copy views when the reshape is compatible with the current memory layout
/// 2. Caching reshape metadata to avoid repeated computations
/// 3. Using efficient memory layouts for better cache performance
pub struct EfficientReshapeOp {
    pub newshape: Vec<usize>,
    pub allow_zero_copy: bool,
}

impl<F: Float> Op<F> for EfficientReshapeOp {
    fn name(&self) -> &'static str {
        "EfficientReshape"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let inputshape = input.shape();

        // Validate reshape compatibility
        let input_size: usize = inputshape.iter().product();
        let new_size: usize = self.newshape.iter().product();

        if input_size != new_size {
            return Err(OpError::IncompatibleShape(format!(
                "Cannot reshape array of size {} into shape {:?} (size {})",
                input_size, self.newshape, new_size
            )));
        }

        // Check cache for reshape compatibility
        let mut cache = RESHAPE_CACHE.lock().expect("Operation failed");
        let _is_contiguous = cache.get(inputshape, &self.newshape);

        let result = if self.allow_zero_copy {
            // Try zero-copy reshape first
            match input.view().into_shape_with_order(IxDyn(&self.newshape)) {
                Ok(reshaped_view) => {
                    // Successful zero-copy reshape
                    cache.insert(inputshape, &self.newshape, true);
                    reshaped_view.to_owned()
                }
                Err(_) => {
                    // Zero-copy failed, fall back to copying
                    cache.insert(inputshape, &self.newshape, false);
                    let vec: Vec<F> = input.iter().cloned().collect();
                    Array::from_shape_vec(IxDyn(&self.newshape), vec).expect("Operation failed")
                }
            }
        } else {
            // Always copy (useful when zero-copy behavior is not desired)
            let vec: Vec<F> = input.iter().cloned().collect();
            Array::from_shape_vec(IxDyn(&self.newshape), vec).expect("Operation failed")
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let input = ctx.input(0);

        // Gradient needs to be reshaped back to input shape
        let inputshape = crate::tensor_ops::shape(input);
        let reshaped_grad = efficient_reshape(gy, &inputshape);
        ctx.append_input_grad(0, Some(reshaped_grad));
    }
}

/// Reshape whose target shape is supplied as a tensor (input 1) instead of a constant.
///
/// Used by the backward pass, which must restore the *runtime* shape of the forward
/// input.  Input 1 is a shape vector, i.e. the output of `tensor_ops::shape`.
pub struct EfficientReshapeDynamicOp {
    pub allow_zero_copy: bool,
}

impl<F: Float> Op<F> for EfficientReshapeDynamicOp {
    fn name(&self) -> &'static str {
        "EfficientReshapeDynamic"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let target = ctx.input(1);

        let newshape: Vec<usize> = target
            .iter()
            .map(|v| v.to_usize().unwrap_or(0))
            .collect::<Vec<_>>();

        let input_size: usize = input.shape().iter().product();
        let new_size: usize = newshape.iter().product();
        if input_size != new_size {
            return Err(OpError::IncompatibleShape(format!(
                "Cannot reshape array of size {input_size} into shape {newshape:?} (size {new_size})"
            )));
        }

        let result = if self.allow_zero_copy {
            match input.view().into_shape_with_order(IxDyn(&newshape)) {
                Ok(reshaped_view) => reshaped_view.to_owned(),
                Err(_) => {
                    let vec: Vec<F> = input.iter().cloned().collect();
                    Array::from_shape_vec(IxDyn(&newshape), vec).map_err(|e| {
                        OpError::IncompatibleShape(format!("EfficientReshapeDynamic: {e}"))
                    })?
                }
            }
        } else {
            let vec: Vec<F> = input.iter().cloned().collect();
            Array::from_shape_vec(IxDyn(&newshape), vec)
                .map_err(|e| OpError::IncompatibleShape(format!("EfficientReshapeDynamic: {e}")))?
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let input = ctx.input(0);
        let inputshape = crate::tensor_ops::shape(input);
        ctx.append_input_grad(0, Some(efficient_reshape(gy, &inputshape)));
        // Input 1 is a shape vector: non-differentiable.
        ctx.append_input_grad(1, None);
    }
}

/// Efficient slice operation that optimizes memory access patterns
///
/// This operation provides optimizations for:
/// 1. Contiguous slices that can use memcpy-style operations
/// 2. Strided slices with optimized access patterns
/// 3. Multi-dimensional slices with cache-friendly traversal
pub struct EfficientSliceOp {
    pub slices: Vec<SliceRange>,
}

#[derive(Debug, Clone)]
pub struct SliceRange {
    pub start: Option<isize>,
    pub end: Option<isize>,
    pub step: Option<isize>,
}

impl SliceRange {
    pub fn new(start: Option<isize>, end: Option<isize>, step: Option<isize>) -> Self {
        Self { start, end, step }
    }

    pub fn full() -> Self {
        Self {
            start: None,
            end: None,
            step: None,
        }
    }

    pub fn single(index: isize) -> Self {
        Self {
            start: Some(index),
            end: Some(index + 1),
            step: Some(1),
        }
    }
}

/// Resolve `SliceRange`s (possibly-negative/omitted bounds) against a
/// concrete input shape into ndarray's `SliceInfoElem`, padding any
/// dimensions beyond `slices.len()` with a full slice. Shared between the
/// forward slice and its backward scatter so both index the same elements.
fn resolve_slice_info_elems(slices: &[SliceRange], inputshape: &[usize]) -> Vec<SliceInfoElem> {
    let mut slice_elements = Vec::with_capacity(inputshape.len());

    for (i, slice_range) in slices.iter().enumerate() {
        let dim_size = inputshape[i] as isize;

        let start = slice_range.start.unwrap_or(0);
        let end = slice_range.end.unwrap_or(dim_size);
        let step = slice_range.step.unwrap_or(1);

        // Normalize negative indices
        let norm_start = if start < 0 { dim_size + start } else { start };
        let norm_end = if end < 0 { dim_size + end } else { end };

        // Clamp to valid range
        let clamped_start = norm_start.max(0).min(dim_size);
        let clamped_end = norm_end.max(0).min(dim_size);

        slice_elements.push(SliceInfoElem::Slice {
            start: clamped_start,
            end: Some(clamped_end),
            step,
        });
    }

    // Add full slices for remaining dimensions
    for _ in slices.len()..inputshape.len() {
        slice_elements.push(SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        });
    }

    slice_elements
}

impl<F: Float> Op<F> for EfficientSliceOp {
    fn name(&self) -> &'static str {
        "EfficientSlice"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let inputshape = input.shape();

        if self.slices.len() > inputshape.len() {
            return Err(OpError::IncompatibleShape(
                "More slice dimensions than tensor dimensions".into(),
            ));
        }

        let slice_elements = resolve_slice_info_elems(&self.slices, inputshape);

        // Create slice info - using the raw slice info creation
        let slice_info = unsafe {
            SliceInfo::<Vec<SliceInfoElem>, IxDyn, IxDyn>::new(slice_elements)
                .expect("Operation failed")
        };

        // Apply the slice
        let sliced = input.slice(slice_info);
        ctx.append_output(sliced.to_owned());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let x_tensor = ctx.input(0);
        let g = ctx.graph();

        let backward_op = EfficientSliceBackwardOp {
            slices: self.slices.clone(),
        };
        let xshape = crate::tensor_ops::shape(x_tensor);
        let gx = Tensor::builder(g)
            .append_input(x_tensor, false)
            .append_input(gy, false)
            .setshape(&xshape)
            .build(backward_op);

        ctx.append_input_grad(0, Some(gx));
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// Backward op for [`EfficientSliceOp`]: scatters the incoming gradient back
/// into a zero tensor at the same positions the forward slice read from.
/// Mirrors `array_ops::SliceGrad`, but resolves its `SliceRange`s against the
/// input's actual runtime shape at `compute()` time (via
/// [`resolve_slice_info_elems`]) rather than requiring pre-resolved
/// `SliceInfoElem`s, since `EfficientSliceOp`'s ranges may contain
/// unresolved (negative/omitted) bounds.
pub struct EfficientSliceBackwardOp {
    pub slices: Vec<SliceRange>,
}

impl<F: Float> Op<F> for EfficientSliceBackwardOp {
    fn name(&self) -> &'static str {
        "EfficientSliceBackward"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);
        let inputshape = x.shape().to_vec();
        let gy = ctx.input(1);

        let slice_elements = resolve_slice_info_elems(&self.slices, &inputshape);

        let mut gx = Array::<F, IxDyn>::zeros(IxDyn(&inputshape));
        unsafe {
            gx.slice_mut(
                SliceInfo::<Vec<SliceInfoElem>, IxDyn, IxDyn>::new(slice_elements)
                    .expect("Operation failed")
                    .as_ref(),
            )
            .zip_mut_with(&gy, |a, &g| *a = g);
        }

        ctx.append_output(gx);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Second-order gradients through this scatter are not needed.
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// Efficient concatenation operation optimized for memory layout
pub struct EfficientConcatOp {
    pub axis: usize,
    pub num_inputs: usize,
}

impl<F: Float> Op<F> for EfficientConcatOp {
    fn name(&self) -> &'static str {
        "EfficientConcat"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let inputs = ctx.inputs();

        if inputs.len() != self.num_inputs {
            return Err(OpError::Other(format!(
                "Expected {} inputs, got {}",
                self.num_inputs,
                inputs.len()
            )));
        }

        if inputs.is_empty() {
            return Err(OpError::Other("Cannot concatenate empty list".into()));
        }

        // Check that all inputs have compatible shapes
        let firstshape = inputs[0].shape();
        let axis = self.axis;

        if axis >= firstshape.len() {
            return Err(OpError::IncompatibleShape(
                "Axis out of bounds for concatenation".into(),
            ));
        }

        for input in inputs.iter().skip(1) {
            let shape = input.shape();
            if shape.len() != firstshape.len() {
                return Err(OpError::IncompatibleShape(
                    "All inputs must have the same number of dimensions".into(),
                ));
            }

            for (i, (&dim1, &dim2)) in firstshape.iter().zip(shape.iter()).enumerate() {
                if i != axis && dim1 != dim2 {
                    return Err(OpError::IncompatibleShape(format!(
                        "All inputs must have the same size except in axis {axis}"
                    )));
                }
            }
        }

        // Calculate output shape
        let mut outputshape = firstshape.to_vec();
        let concat_size: usize = inputs.iter().map(|input| input.shape()[axis]).sum();
        outputshape[axis] = concat_size;

        // Create output array
        let mut output = Array::<F, IxDyn>::zeros(IxDyn(&outputshape));

        // Copy data from each input
        let mut current_offset = 0;
        for input in inputs {
            let input_size = input.shape()[axis];

            // Create slice for where to place this input
            let mut slice_start = vec![0; outputshape.len()];
            let mut slice_end = outputshape.clone();

            slice_start[axis] = current_offset;
            slice_end[axis] = current_offset + input_size;

            // Use a simple approach for copying - in a full implementation,
            // this would be optimized for contiguous memory operations
            copy_slice_data(&mut output, &input, &slice_start, &slice_end, axis);

            current_offset += input_size;
        }

        ctx.append_output(output);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let _gy = ctx.output_grad();

        // Get inputs count before borrowing mutably
        let num_inputs = ctx.inputs().len();
        let graph = ctx.graph();

        // Split the gradient back to each input
        for i in 0..num_inputs {
            let input = ctx.input(i);
            let inputshape = crate::tensor_ops::shape(input);

            // Create a slice of the gradient for this input
            // For now, use a simplified approach
            let grad_slice = crate::tensor_ops::zeros(&inputshape, graph);

            ctx.append_input_grad(i, Some(grad_slice));
        }
    }
}

/// Helper function to copy slice data efficiently
#[allow(dead_code)]
fn copy_slice_data<F: Float>(
    output: &mut Array<F, IxDyn>,
    input: &NdArrayView<F>,
    _slice_start: &[usize],
    _slice_end: &[usize],
    _axis: usize,
) {
    // This is a simplified implementation
    // In practice, this would use optimized memory copying operations

    // For now, just copy element by element
    // This would be replaced with vectorized operations in a full implementation
    let input_iter = input.iter();

    // Skip to the correct position and copy elements
    // This is highly simplified - real implementation would handle multi-dimensional slicing properly
    for (&input_val, output_val) in input_iter.zip(output.iter_mut()) {
        *output_val = input_val;
    }
}

// Public API functions

/// Efficient reshape operation with optional zero-copy optimization
#[allow(dead_code)]
pub fn efficient_reshape<'g, F: Float>(
    tensor: &Tensor<'g, F>,
    newshape: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = tensor.graph();

    // The target shape is only known at run time, so it is passed as a second *input*
    // and read by `EfficientReshapeDynamicOp::compute`.  This function used to ignore
    // `newshape` entirely and hard-code `vec![1]`, which silently reshaped every tensor
    // to a single element (and failed outright for anything with more than one).
    Tensor::builder(g)
        .append_input(tensor, false)
        .append_input(newshape, false)
        .build(EfficientReshapeDynamicOp {
            allow_zero_copy: true,
        })
}

/// Efficient reshape with explicit shape values
#[allow(dead_code)]
pub fn efficient_reshape_withshape<'g, F: Float>(
    tensor: &Tensor<'g, F>,
    newshape: &[usize],
) -> Tensor<'g, F> {
    let g = tensor.graph();

    Tensor::builder(g)
        .append_input(tensor, false)
        .build(EfficientReshapeOp {
            newshape: newshape.to_vec(),
            allow_zero_copy: true,
        })
}

/// Efficient slice operation
#[allow(dead_code)]
pub fn efficient_slice<'g, F: Float>(
    tensor: &Tensor<'g, F>,
    slices: &[SliceRange],
) -> Tensor<'g, F> {
    let g = tensor.graph();

    Tensor::builder(g)
        .append_input(tensor, false)
        .build(EfficientSliceOp {
            slices: slices.to_vec(),
        })
}

/// Efficient concatenation of multiple tensors
#[allow(dead_code)]
pub fn efficient_concat<'g, F: Float>(tensors: &[&Tensor<'g, F>], axis: usize) -> Tensor<'g, F> {
    if tensors.is_empty() {
        panic!("Cannot concatenate empty tensor list");
    }

    let g = tensors[0].graph();
    let mut builder = Tensor::builder(g);

    for tensor in tensors {
        builder = builder.append_input(*tensor, false);
    }

    builder.build(EfficientConcatOp {
        axis,
        num_inputs: tensors.len(),
    })
}

/// Efficient transpose operation with cache-friendly memory access
#[allow(dead_code)]
pub fn efficient_transpose<'g, F: Float>(
    tensor: &Tensor<'g, F>,
    _axes: Option<&[usize]>,
) -> Tensor<'g, F> {
    // For now, use the standard transpose operation
    // In a full implementation, this would be optimized for cache efficiency
    // We'll simplify this to just use the default transpose
    let g = tensor.graph();
    let axes_tensor = crate::tensor_ops::convert_to_tensor(
        scirs2_core::ndarray::Array::from_shape_vec(
            scirs2_core::ndarray::IxDyn(&[2]),
            vec![
                F::from(1).expect("Failed to convert constant to float"),
                F::from(0).expect("Failed to convert constant to float"),
            ],
        )
        .expect("Failed to reshape"),
        g,
    );
    crate::tensor_ops::transpose(tensor, &axes_tensor)
}

/// Clear the reshape operation cache
#[allow(dead_code)]
pub fn clear_reshape_cache() {
    RESHAPE_CACHE.lock().expect("Operation failed").clear();
}

/// Get statistics about the reshape cache
#[allow(dead_code)]
pub fn get_reshape_cache_stats() -> (usize, usize) {
    let cache = RESHAPE_CACHE.lock().expect("Operation failed");
    (cache.cache.len(), cache.max_size)
}

/// Utility struct for managing efficient operations
pub struct EfficientOpsManager;

impl EfficientOpsManager {
    /// Configure efficient operations for optimal performance
    pub fn configure_for_performance() {
        // Clear caches to start fresh
        clear_reshape_cache();

        // Enable memory optimization features
        crate::tensor_ops::set_memory_pool_enabled(true);
        crate::tensor_ops::enable_memory_tracking();
    }

    /// Configure efficient operations for minimal memory usage
    pub fn configure_for_memory() {
        // Clear caches to reduce memory usage
        clear_reshape_cache();

        // Configure memory pool for memory efficiency
        crate::tensor_ops::configure_memory_pool(4, 10 * 1024 * 1024); // Smaller limits
        crate::tensor_ops::set_memory_pool_enabled(true);
    }

    /// Get performance statistics
    pub fn get_performance_stats() -> EfficientOpsStats {
        let (cache_size, cache_max) = get_reshape_cache_stats();
        let memory_stats = crate::tensor_ops::get_memory_tracking_stats();
        let pool_stats = crate::tensor_ops::get_memory_pool_stats();

        EfficientOpsStats {
            reshape_cache_size: cache_size,
            reshape_cache_max: cache_max,
            memory_tracking_enabled: memory_stats.enabled,
            memory_pool_enabled: pool_stats.enabled,
            total_memory_allocations: memory_stats.total_allocations.values().sum(),
            pooled_memory: pool_stats.total_pooled_memory,
        }
    }
}

/// Statistics for efficient operations
#[derive(Debug, Clone)]
pub struct EfficientOpsStats {
    pub reshape_cache_size: usize,
    pub reshape_cache_max: usize,
    pub memory_tracking_enabled: bool,
    pub memory_pool_enabled: bool,
    pub total_memory_allocations: usize,
    pub pooled_memory: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_range_creation() {
        let slice = SliceRange::new(Some(0), Some(10), Some(1));
        assert_eq!(slice.start, Some(0));
        assert_eq!(slice.end, Some(10));
        assert_eq!(slice.step, Some(1));

        let full_slice = SliceRange::full();
        assert_eq!(full_slice.start, None);
        assert_eq!(full_slice.end, None);
        assert_eq!(full_slice.step, None);

        let single_slice = SliceRange::single(5);
        assert_eq!(single_slice.start, Some(5));
        assert_eq!(single_slice.end, Some(6));
        assert_eq!(single_slice.step, Some(1));
    }

    #[test]
    fn test_efficient_reshape_op_creation() {
        let op = EfficientReshapeOp {
            newshape: vec![2, 3, 4],
            allow_zero_copy: true,
        };
        assert_eq!(
            <EfficientReshapeOp as crate::op::Op<f32>>::name(&op),
            "EfficientReshape"
        );
        assert_eq!(op.newshape, vec![2, 3, 4]);
        assert!(op.allow_zero_copy);
    }

    #[test]
    fn test_efficient_slice_op_creation() {
        let slices = vec![
            SliceRange::new(Some(0), Some(5), Some(1)),
            SliceRange::full(),
        ];
        let op = EfficientSliceOp {
            slices: slices.clone(),
        };
        assert_eq!(
            <EfficientSliceOp as crate::op::Op<f32>>::name(&op),
            "EfficientSlice"
        );
        assert_eq!(op.slices.len(), 2);
    }

    #[test]
    fn test_efficient_concat_op_creation() {
        let op = EfficientConcatOp {
            axis: 1,
            num_inputs: 3,
        };
        assert_eq!(
            <EfficientConcatOp as crate::op::Op<f32>>::name(&op),
            "EfficientConcat"
        );
        assert_eq!(op.axis, 1);
        assert_eq!(op.num_inputs, 3);
    }

    #[test]
    fn test_reshape_cache() {
        let mut cache = ReshapeCache::new();

        let fromshape = vec![2, 3];
        let toshape = vec![6];

        assert_eq!(cache.get(&fromshape, &toshape), None);

        cache.insert(&fromshape, &toshape, true);
        assert_eq!(cache.get(&fromshape, &toshape), Some(true));

        cache.clear();
        assert_eq!(cache.get(&fromshape, &toshape), None);
    }

    #[test]
    fn test_efficient_ops_manager() {
        EfficientOpsManager::configure_for_performance();
        let stats = EfficientOpsManager::get_performance_stats();

        // Just verify we can call these functions without error
        assert!(stats.reshape_cache_max > 0);
    }

    /// Regression test for `EfficientReshapeDynamicOp::compute` (driven by the
    /// public `efficient_reshape` function): the forward pass used to hard-code
    /// `shape_values = vec![1]` and ignore the requested target shape entirely,
    /// so this either failed outright on a size mismatch or silently collapsed
    /// every tensor to a single element. Non-constant, distinct values are used
    /// so a collapsed/garbled result is caught, not just a shape or panic
    /// check.
    #[test]
    fn efficient_reshape_forward_uses_the_real_requested_shape() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let original = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
            let x = crate::tensor_ops::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2, 3]), original.clone()).expect("Operation failed"),
                ctx,
            );

            let targetshape = crate::tensor_ops::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![3.0f64, 2.0]).expect("Operation failed"),
                ctx,
            );
            let reshaped = efficient_reshape(&x, &targetshape);
            let reshaped_val = reshaped.eval(ctx).expect("Operation failed");
            assert_eq!(reshaped_val.shape(), &[3, 2]);
            // Row-major reinterpretation must preserve the flat element order.
            assert_eq!(reshaped_val.iter().copied().collect::<Vec<_>>(), original);

            // Reshape back to the original shape: a full round trip.
            let backshape = crate::tensor_ops::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![2.0f64, 3.0]).expect("Operation failed"),
                ctx,
            );
            let back = efficient_reshape(&reshaped, &backshape);
            let back_val = back.eval(ctx).expect("Operation failed");
            assert_eq!(back_val.shape(), &[2, 3]);
            assert_eq!(back_val.iter().copied().collect::<Vec<_>>(), original);
        });
    }

    /// The gradient of `efficient_reshape` reshapes `gy` back to the input's
    /// shape, so it silently depends on the same forward-shape bug: this
    /// exercises the full round trip through the crate's public gradient API.
    /// `x` is built with `variable` (not `convert_to_tensor`, which marks its
    /// output non-differentiable and would make any gradient -- correct or
    /// not -- silently collapse to zero; see `tests/gradient_fd_harness.rs`'s
    /// header comment).
    #[test]
    fn efficient_reshape_backward_restores_inputshape_and_gradient() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = crate::tensor_ops::variable(
                Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
                    .expect("Operation failed"),
                ctx,
            );
            let targetshape = crate::tensor_ops::convert_to_tensor(
                Array::from_shape_vec(IxDyn(&[2]), vec![3.0f64, 2.0]).expect("Operation failed"),
                ctx,
            );
            let y = efficient_reshape(&x, &targetshape);

            let gx = crate::tensor_ops::grad(&[y], &[x])[0];
            let gx_val = gx.eval(ctx).expect("Operation failed");
            assert_eq!(gx_val.shape(), &[2, 3]);
            assert_eq!(gx_val.iter().copied().collect::<Vec<_>>(), vec![1.0f64; 6]);
        });
    }
}
