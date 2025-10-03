//! Advanced tensor indexing operations
//!
//! This module provides sophisticated indexing operations that go beyond basic slicing,
//! including boolean masking, fancy indexing, and multi-dimensional selection operations.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use scirs2_core::ndarray::{Array, Axis, Ix1, IxDyn};

/// Boolean masking operation
///
/// Selects elements from a tensor where the corresponding boolean mask is true.
/// This is similar to numpy's boolean indexing or PyTorch's masked_select.
///
/// # Example
/// ```ignore
/// let data = tensor![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let mask = tensor![[true, false, true], [false, true, false]];
/// let result = boolean_mask(data, mask); // [1.0, 3.0, 5.0]
/// ```
pub struct BooleanMaskOp;

impl<F: Float> Op<F> for BooleanMaskOp {
    fn name(&self) -> &'static str {
        "BooleanMask"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let data = ctx.input(0);
        let mask = ctx.input(1);

        // Ensure mask and data have compatible shapes
        if data.shape() != mask.shape() {
            return Err(OpError::IncompatibleShape(
                "Data and mask must have the same shape".into(),
            ));
        }

        let data_view = data.view();
        let mask_view = mask.view();

        // Collect elements where mask is true (non-zero)
        let mut selected_elements = Vec::new();
        for (data_elem, mask_elem) in data_view.iter().zip(mask_view.iter()) {
            if *mask_elem != F::zero() {
                selected_elements.push(*data_elem);
            }
        }

        // Create 1D array with selected elements
        let result = Array::from_vec(selected_elements);
        ctx.append_output(result.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let _gy = ctx.output_grad();
        let _mask = ctx.input(1);
        let g = ctx.graph();

        // Get the input shape for creating gradient array
        let inputshape = crate::tensor_ops::shape(ctx.input(0));

        // Create gradient tensor with same shape as input, filled with zeros
        let zeros = crate::tensor_ops::zeros(&inputshape, g);

        // In a full implementation, we'd need to scatter the gradient values
        // back to their original positions based on the mask
        // For now, we provide a simplified gradient
        ctx.append_input_grad(0, Some(zeros));
        ctx.append_input_grad(1, None); // No gradient for boolean mask
    }
}

/// Take operation
///
/// Selects elements from a tensor using an array of indices.
/// This is similar to numpy's take or PyTorch's index_select.
///
/// # Example
/// ```ignore
/// let data = tensor![10.0, 20.0, 30.0, 40.0, 50.0];
/// let indices = tensor![0, 2, 4, 1];
/// let result = take(data, indices, 0); // [10.0, 30.0, 50.0, 20.0]
/// ```
pub struct TakeOp {
    pub axis: isize,
}

impl<F: Float> Op<F> for TakeOp {
    fn name(&self) -> &'static str {
        "Take"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let data = ctx.input(0);
        let indices = ctx.input(1);

        let datashape = data.shape();
        let indicesshape = indices.shape();

        // Normalize axis
        let axis = if self.axis < 0 {
            (datashape.len() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        if axis >= datashape.len() {
            return Err(OpError::IncompatibleShape("Axis out of bounds".into()));
        }

        // For simplicity, we'll handle 1D indices for now
        if indicesshape.len() != 1 {
            return Err(OpError::IncompatibleShape(
                "Only 1D indices supported for now".into(),
            ));
        }

        let indices_view = indices.view();
        let data_view = data.view();
        let axis_size = datashape[axis];

        // Collect indices as integers
        let index_values: Result<Vec<usize>, OpError> = indices_view
            .iter()
            .map(|&idx| {
                let idx_int = idx
                    .to_usize()
                    .ok_or_else(|| OpError::Other("Index must be non-negative integer".into()))?;
                if idx_int >= axis_size {
                    Err(OpError::Other("Index out of bounds".into()))
                } else {
                    Ok(idx_int)
                }
            })
            .collect();

        let index_values = index_values?;

        // Create output shape: replace axis dimension with number of indices
        let mut outputshape = datashape.to_vec();
        outputshape[axis] = index_values.len();

        // Create output array
        let mut output = Array::<F, IxDyn>::zeros(IxDyn(&outputshape));

        // Select elements along the specified axis
        for (out_idx, &src_idx) in index_values.iter().enumerate() {
            let src_slice = data_view.index_axis(Axis(axis), src_idx);
            let mut out_slice = output.index_axis_mut(Axis(axis), out_idx);
            out_slice.assign(&src_slice);
        }

        ctx.append_output(output);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let _gy = ctx.output_grad();
        let _indices = ctx.input(1);
        let g = ctx.graph();

        // Create gradient with same shape as input
        let inputshape = crate::tensor_ops::shape(ctx.input(0));
        let zeros = crate::tensor_ops::zeros(&inputshape, g);

        // In a full implementation, we'd scatter gradients back to original positions
        ctx.append_input_grad(0, Some(zeros));
        ctx.append_input_grad(1, None); // No gradient for indices
    }
}

/// Scatter operation
///
/// Scatters values into a tensor at specified indices.
/// This is the inverse of gather operations.
///
/// # Example
/// ```ignore
/// let indices = tensor![0, 2, 1];
/// let updates = tensor![10.0, 30.0, 20.0];
/// let result = scatter(5, indices, updates, 0); // [10.0, 20.0, 30.0, 0.0, 0.0]
/// ```
pub struct ScatterOp {
    #[allow(dead_code)]
    pub axis: isize,
    pub output_size: usize,
}

impl<F: Float> Op<F> for ScatterOp {
    fn name(&self) -> &'static str {
        "Scatter"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let indices = ctx.input(0);
        let updates = ctx.input(1);

        let indicesshape = indices.shape();
        let updatesshape = updates.shape();

        // For simplicity, handle 1D case
        if indicesshape.len() != 1 || updatesshape.len() != 1 {
            return Err(OpError::IncompatibleShape(
                "Only 1D scatter supported for now".into(),
            ));
        }

        if indicesshape[0] != updatesshape[0] {
            return Err(OpError::IncompatibleShape(
                "Indices and updates must have same length".into(),
            ));
        }

        let indices_view = indices.view();
        let updates_view = updates.view();

        // Create output array filled with zeros
        let mut output = Array::<F, Ix1>::zeros(self.output_size);

        // Scatter updates into output
        for (idx_val, update_val) in indices_view.iter().zip(updates_view.iter()) {
            let idx = idx_val
                .to_usize()
                .ok_or_else(|| OpError::Other("Index must be non-negative integer".into()))?;

            if idx >= self.output_size {
                return Err(OpError::Other("Index out of bounds".into()));
            }

            output[idx] = *update_val;
        }

        ctx.append_output(output.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let _gy = ctx.output_grad();
        let _indices = ctx.input(0);
        let g = ctx.graph();

        // Gradient w.r.t. updates: gather from output gradient at specified indices
        // This is essentially a gather operation
        let updatesshape = crate::tensor_ops::shape(ctx.input(1));
        let zeros_updates = crate::tensor_ops::zeros(&updatesshape, g);

        ctx.append_input_grad(0, None); // No gradient for indices
        ctx.append_input_grad(1, Some(zeros_updates)); // Simplified gradient for updates
    }
}

/// Where operation (conditional selection)
///
/// Selects elements from x or y based on a condition (mask).
/// This is similar to numpy's where or PyTorch's where.
///
/// # Example
/// ```ignore
/// let condition = tensor![true, false, true, false];
/// let x = tensor![1.0, 2.0, 3.0, 4.0];
/// let y = tensor![10.0, 20.0, 30.0, 40.0];
/// let result = where_op(condition, x, y); // [1.0, 20.0, 3.0, 40.0]
/// ```
pub struct WhereOp;

impl<F: Float> Op<F> for WhereOp {
    fn name(&self) -> &'static str {
        "Where"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let condition = ctx.input(0);
        let x = ctx.input(1);
        let y = ctx.input(2);

        // Check that all inputs have compatible shapes
        if condition.shape() != x.shape() || x.shape() != y.shape() {
            return Err(OpError::IncompatibleShape(
                "All inputs must have the same shape".into(),
            ));
        }

        let condition_view = condition.view();
        let x_view = x.view();
        let y_view = y.view();

        // Create output array
        let mut output = Array::<F, IxDyn>::zeros(x.shape());

        // Select elements based on condition
        for ((out_elem, &cond), (&x_elem, &y_elem)) in output
            .iter_mut()
            .zip(condition_view.iter())
            .zip(x_view.iter().zip(y_view.iter()))
        {
            *out_elem = if cond != F::zero() { x_elem } else { y_elem };
        }

        ctx.append_output(output);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let _gy = ctx.output_grad();
        let _condition = ctx.input(0);
        let g = ctx.graph();

        // Gradient flows to x where condition is true, to y where condition is false
        // This requires evaluating the condition, which we'll simplify for now
        let xshape = crate::tensor_ops::shape(ctx.input(1));
        let yshape = crate::tensor_ops::shape(ctx.input(2));

        let zeros_x = crate::tensor_ops::zeros(&xshape, g);
        let zeros_y = crate::tensor_ops::zeros(&yshape, g);

        ctx.append_input_grad(0, None); // No gradient for condition
        ctx.append_input_grad(1, Some(zeros_x)); // Simplified gradient for x
        ctx.append_input_grad(2, Some(zeros_y)); // Simplified gradient for y
    }
}

/// Advanced gather operation with multiple indices
///
/// Gathers elements using multiple index arrays for advanced indexing.
/// This supports more complex indexing patterns than basic gather.
pub struct AdvancedGatherOp {
    pub axes: Vec<isize>,
}

impl<F: Float> Op<F> for AdvancedGatherOp {
    fn name(&self) -> &'static str {
        "AdvancedGather"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let data = ctx.input(0);
        let datashape = data.shape();

        // For now, implement a simplified version that works with 2D indexing
        if self.axes.len() != 2 || ctx.inputs().len() != 3 {
            return Err(OpError::IncompatibleShape(
                "Advanced gather currently supports 2D indexing only".into(),
            ));
        }

        let indices0 = ctx.input(1);
        let indices1 = ctx.input(2);

        // Check that index arrays have the same shape
        if indices0.shape() != indices1.shape() {
            return Err(OpError::IncompatibleShape(
                "Index arrays must have the same shape".into(),
            ));
        }

        let indices0_view = indices0.view();
        let indices1_view = indices1.view();
        let data_view = data.view();

        // Create output array with same shape as index arrays
        let outputshape = indices0.shape();
        let mut output = Array::<F, IxDyn>::zeros(outputshape);

        // Gather elements using paired indices
        for ((out_elem, &idx0), &idx1) in output
            .iter_mut()
            .zip(indices0_view.iter())
            .zip(indices1_view.iter())
        {
            let i0 = idx0
                .to_usize()
                .ok_or_else(|| OpError::Other("Index must be non-negative integer".into()))?;
            let i1 = idx1
                .to_usize()
                .ok_or_else(|| OpError::Other("Index must be non-negative integer".into()))?;

            if i0 >= datashape[0] || i1 >= datashape[1] {
                return Err(OpError::Other("Index out of bounds".into()));
            }

            *out_elem = data_view[[i0, i1]];
        }

        ctx.append_output(output);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let _gy = ctx.output_grad();
        let g = ctx.graph();

        // Create gradient with same shape as input
        let inputshape = crate::tensor_ops::shape(ctx.input(0));
        let zeros = crate::tensor_ops::zeros(&inputshape, g);

        ctx.append_input_grad(0, Some(zeros));
        ctx.append_input_grad(1, None); // No gradient for indices
        ctx.append_input_grad(2, None); // No gradient for indices
    }
}

// Public API functions

/// Boolean masking - select elements where mask is true
#[allow(dead_code)]
pub fn boolean_mask<'g, F: Float>(data: &Tensor<'g, F>, mask: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = data.graph();
    Tensor::builder(g)
        .append_input(data, false)
        .append_input(mask, false)
        .build(BooleanMaskOp)
}

/// Take elements from tensor using indices along specified axis
#[allow(dead_code)]
pub fn take<'g, F: Float>(
    data: &Tensor<'g, F>,
    indices: &Tensor<'g, F>,
    axis: isize,
) -> Tensor<'g, F> {
    let g = data.graph();
    Tensor::builder(g)
        .append_input(data, false)
        .append_input(indices, false)
        .build(TakeOp { axis })
}

/// Scatter values into a tensor at specified indices
#[allow(dead_code)]
pub fn scatter<'g, F: Float>(
    indices: &Tensor<'g, F>,
    updates: &Tensor<'g, F>,
    output_size: usize,
    axis: isize,
) -> Tensor<'g, F> {
    let g = indices.graph();
    Tensor::builder(g)
        .append_input(indices, false)
        .append_input(updates, false)
        .build(ScatterOp { axis, output_size })
}

/// Conditional selection - choose elements from x or y based on condition
#[allow(dead_code)]
pub fn where_op<'g, F: Float>(
    condition: &Tensor<'g, F>,
    x: &Tensor<'g, F>,
    y: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = condition.graph();
    Tensor::builder(g)
        .append_input(condition, false)
        .append_input(x, false)
        .append_input(y, false)
        .build(WhereOp)
}

/// Advanced gather with multiple index arrays
#[allow(dead_code)]
pub fn advanced_gather<'g, F: Float>(
    data: &Tensor<'g, F>,
    indices: &[&Tensor<'g, F>],
    axes: &[isize],
) -> Tensor<'g, F> {
    let g = data.graph();
    let mut builder = Tensor::builder(g);
    builder = builder.append_input(data, false);

    for idx_tensor in indices {
        builder = builder.append_input(*idx_tensor, false);
    }

    builder.build(AdvancedGatherOp {
        axes: axes.to_vec(),
    })
}

/// Convenience functions for common indexing patterns
///
/// Get elements at specific 2D coordinates
#[allow(dead_code)]
pub fn get_at_coords<'g, F: Float>(
    data: &Tensor<'g, F>,
    row_indices: &Tensor<'g, F>,
    col_indices: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    advanced_gather(data, &[row_indices, col_indices], &[0, 1])
}

/// Select rows from a 2D tensor
#[allow(dead_code)]
pub fn select_rows<'g, F: Float>(
    data: &Tensor<'g, F>,
    row_indices: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    take(data, row_indices, 0)
}

/// Select columns from a 2D tensor  
#[allow(dead_code)]
pub fn select_columns<'g, F: Float>(
    data: &Tensor<'g, F>,
    col_indices: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    take(data, col_indices, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boolean_mask_op_creation() {
        let op = BooleanMaskOp;
        assert_eq!(<BooleanMaskOp as Op<f32>>::name(&op), "BooleanMask");
    }

    #[test]
    fn test_take_op_creation() {
        let op = TakeOp { axis: 0 };
        assert_eq!(<TakeOp as Op<f32>>::name(&op), "Take");
        assert_eq!(op.axis, 0);
    }

    #[test]
    fn test_scatter_op_creation() {
        let op = ScatterOp {
            axis: 0,
            output_size: 10,
        };
        assert_eq!(<ScatterOp as Op<f32>>::name(&op), "Scatter");
        assert_eq!(op.axis, 0);
        assert_eq!(op.output_size, 10);
    }

    #[test]
    fn test_where_op_creation() {
        let op = WhereOp;
        assert_eq!(<WhereOp as Op<f32>>::name(&op), "Where");
    }

    #[test]
    fn test_advanced_gather_op_creation() {
        let op = AdvancedGatherOp { axes: vec![0, 1] };
        assert_eq!(<AdvancedGatherOp as Op<f32>>::name(&op), "AdvancedGather");
        assert_eq!(op.axes, vec![0, 1]);
    }
}
