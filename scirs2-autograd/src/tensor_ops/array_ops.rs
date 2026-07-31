use crate::ndarray;
use crate::ndarray_ext;
#[cfg(feature = "blas")]
use crate::ndarray_ext::NdArrayViewMut;
use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::tensor_ops::*;
use crate::Float;
use scirs2_core::ndarray::SliceInfoElem;
use std::iter::FromIterator;

pub struct ExpandDims;

pub struct Squeeze;

pub struct Slice {
    pub indices: Vec<SliceInfoElem>,
}

pub struct SliceGrad {
    pub indices: Vec<SliceInfoElem>,
}

pub struct Split {
    pub axis: isize,
    pub start_index: isize,
    pub end_index: isize,
}

pub struct SplitGrad {
    pub axis: isize,
    pub start_index: isize,
    pub end_index: isize,
}

pub struct Tile {
    pub axis: isize,
    pub num: usize,
}

pub struct Concat {
    pub axis: isize,
}

pub struct ConcatGrad {
    pub axis: isize,
    pub index: usize,
}

pub struct Clip<T: Float> {
    pub min: T,
    pub max: T,
}

pub struct ClipGrad<T: Float> {
    pub min: T,
    pub max: T,
}

pub struct AddN;

pub struct Gather {
    pub axis: isize,
    pub should_normalize_negative_indices: bool,
}

pub struct GatherGrad {
    pub axis: isize,
}

pub struct IndexOp {
    pub index: isize,
}

pub struct IndexOpGrad {
    pub index: isize,
}

pub struct SetDiff1D;

pub struct Shape;

pub struct Rank;

pub struct Size;

pub struct Reshape;

pub struct InferBinOpShape;

pub struct Assign;

impl<T: Float> op::Op<T> for Assign {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let input1 = ctx.input(1).to_owned();
        ctx.input_mut(0).assign(&input1);
        ctx.append_output(scirs2_core::ndarray::Array::zeros(vec![]).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for InferBinOpShape {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ashape_float = ctx.input(0);
        let bshape_float = ctx.input(1);

        // Check for negative values (e.g. -1 from reshape/flatten).
        // These indicate "infer this dimension" and cannot be converted
        // to usize.  If either operand has a negative dimension, we
        // take the other operand's value for that dimension.  If both
        // are negative, we propagate the negative value as float.
        let a_has_neg = ashape_float.iter().any(|x| *x < T::zero());
        let b_has_neg = bshape_float.iter().any(|x| *x < T::zero());

        // Helper: convert to usize, treating negatives as 0 for
        // is_scalarshape check purposes only.
        let to_usize_safe = |x: &T| -> usize { x.to_usize().unwrap_or(0) };

        let ashape: Vec<usize> = ashape_float.iter().map(to_usize_safe).collect();
        let bshape: Vec<usize> = bshape_float.iter().map(to_usize_safe).collect();

        let a_is_scalar = ndarray_ext::is_scalarshape(ashape.as_slice());
        let b_is_scalar = ndarray_ext::is_scalarshape(bshape.as_slice());

        if !a_is_scalar && !b_is_scalar {
            // NumPy-style broadcasting: right-align the two shapes and take the larger
            // extent per axis.  Requiring equal ranks (which this used to do) rejected
            // the most common broadcast of all -- a bias vector added to a batch matrix,
            // `[3, 4] + [4]` -- and the resulting `OpError` then propagated as a *missing
            // input* to whatever consumed the shape.
            let a_rank = ashape_float.len();
            let b_rank = bshape_float.len();
            let rank = a_rank.max(b_rank);
            let a_pad = rank - a_rank;
            let b_pad = rank - b_rank;

            // Element-wise max, but handle negative sentinel values:
            // - If both are non-negative, take max
            // - If one is negative, take the other (it's the known dimension)
            // - If both are negative, propagate the negative value
            // Axes that only one operand has are simply that operand's extent.
            let mut max: Vec<T> = Vec::with_capacity(rank);
            for axis in 0..rank {
                let a = if axis >= a_pad {
                    Some(ashape_float[axis - a_pad])
                } else {
                    None
                };
                let b = if axis >= b_pad {
                    Some(bshape_float[axis - b_pad])
                } else {
                    None
                };
                let dim = match (a, b) {
                    (Some(a), Some(b)) => {
                        let a_neg = a < T::zero();
                        let b_neg = b < T::zero();
                        if !a_neg && !b_neg {
                            if a > b {
                                a
                            } else {
                                b
                            }
                        } else if a_neg && !b_neg {
                            b
                        } else {
                            // b unknown (or both unknown): use / propagate a
                            a
                        }
                    }
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => T::zero(),
                };
                max.push(dim);
            }
            ctx.append_output(
                NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[rank]), max)
                    .map_err(|e| op::OpError::IncompatibleShape(format!("InferBinOpShape: {e}")))?,
            )
        } else if !a_is_scalar {
            ctx.append_output(ashape_float.to_owned());
        } else {
            ctx.append_output(bshape_float.to_owned());
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for Shape {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = &ctx.input(0);
        let shape_vec = ndarray_ext::shape_of_view(x);
        let shape_t: Vec<T> = shape_vec
            .iter()
            .map(|&s| T::from(s).expect("Operation failed"))
            .collect();
        let ret = NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[shape_vec.len()]), shape_t)
            .expect("Failed to create shape vector");
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Rank {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let ret = NdArray::from_elem(
            scirs2_core::ndarray::IxDyn(&[]),
            T::from(x.ndim()).expect("Operation failed"),
        );
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Size {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let ret = NdArray::from_elem(
            scirs2_core::ndarray::IxDyn(&[]),
            T::from(x.len()).expect("Operation failed"),
        );
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Reshape {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = &ctx.input(0);
        let shape_arr = &ctx.input(1);
        let target = shape_arr
            .iter()
            .map(|&dim_size| {
                if dim_size != -T::one() {
                    dim_size.to_usize().expect("Operation failed")
                } else {
                    let product: T = shape_arr.iter().fold(T::one(), |acc, &x| acc * x);
                    x.len() / product.neg().to_usize().expect("Operation failed")
                }
            })
            .collect::<Vec<_>>();
        // If x is *not* a c-contiguous, just copying it for now
        // due to current state of ndarray: https://github.com/rust-ndarray/ndarray/issues/390
        if x.is_standard_layout() {
            if let Ok(a) = x
                .clone()
                .into_shape_with_order(scirs2_core::ndarray::IxDyn(target.as_slice()))
            {
                ctx.append_output(a.to_owned());
            } else {
                let copy = ndarray_ext::deep_copy(x);
                if let Ok(a) =
                    copy.into_shape_with_order(scirs2_core::ndarray::IxDyn(target.as_slice()))
                {
                    ctx.append_output(a);
                } else {
                    return Err(op::OpError::IncompatibleShape(format!(
                        "reshape failed: {:?} vs {:?}",
                        x.shape(),
                        target
                    )));
                }
            }
        } else if let Ok(a) = ndarray_ext::deep_copy(x)
            .into_shape_with_order(scirs2_core::ndarray::IxDyn(target.as_slice()))
        {
            ctx.append_output(a)
        } else {
            return Err(op::OpError::IncompatibleShape(format!(
                "reshape failed: {:?} vs {:?}",
                x.shape(),
                target
            )));
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let gx = Tensor::builder(ctx.graph())
            .append_input(gy, false)
            .append_input(shape(x), false)
            .build(Reshape);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for SetDiff1D {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x0 = ctx.input(0);
        let x1 = &ctx.input(1);

        let set_a: crate::FxHashSet<isize> = crate::FxHashSet::from_iter(
            x0.as_slice()
                .expect("Failed to get slice")
                .iter()
                .map(|&a| a.to_isize().expect("Operation failed")),
        );

        let set_b: crate::FxHashSet<isize> = crate::FxHashSet::from_iter(
            x1.as_slice()
                .expect("Failed to get slice")
                .iter()
                .map(|&a| a.to_isize().expect("Operation failed")),
        );

        let diff = set_a.difference(&set_b);

        let mut vec = diff.collect::<Vec<&isize>>();
        vec.sort();
        let vec = vec
            .into_iter()
            .map(|&a| T::from(a).expect("Operation failed"))
            .collect::<Vec<T>>();
        let len = vec.len();
        // safe unwrap
        let ret = NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[len]), vec)
            .expect("Operation failed");
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for IndexOp {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let i = if self.index < 0 {
            ((x.len() as isize) + self.index) as usize
        } else {
            self.index as usize
        };
        // unwrap is safe
        let flat_x = x
            .view()
            .into_shape_with_order(x.len())
            .expect("Operation failed");
        if let Some(ret) = flat_x.get(i) {
            ctx.append_output(scirs2_core::ndarray::arr0(*ret).into_dyn());
            Ok(())
        } else {
            Err(op::OpError::OutOfBounds(format!(
                "access_elem: tried to access index {} in tensor of length {} (shape: {:?})",
                i,
                x.len(),
                x.shape(),
            )))
        }
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let op = IndexOpGrad { index: self.index };
        let x = ctx.input(0);
        let gy = ctx.output_grad();
        let gx = Tensor::builder(ctx.graph())
            .setshape(&shape(x))
            .append_input(x, false)
            .append_input(gy, false)
            .build(op);
        ctx.append_input_grad(0, Some(gx));
    }
}

impl<T: Float> op::Op<T> for IndexOpGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let gy = &ctx.input(1);
        let mut result = NdArray::zeros(x.shape());
        let i = if self.index < 0 {
            ((x.len() as isize) + self.index) as usize
        } else {
            self.index as usize
        };
        // unwrap is safe
        let len = result.len();
        if let Some(a) = result
            .view_mut()
            .into_shape_with_order(len)
            .expect("Failed to reshape")
            .get_mut(i)
        {
            *a = gy[scirs2_core::ndarray::IxDyn(&[])];
        } else {
            return Err(op::OpError::OutOfBounds(format!(
                "access_elem: tried to access index {} in tensor of length {} (shape: {:?})",
                i,
                x.len(),
                x.shape(),
            )));
        }
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Gather {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let param = &ctx.input(1);
        let indices = &ctx.input(0);
        let indicesshape = indices.shape();
        let paramshape = param.shape();
        let axis = ndarray_ext::normalize_negative_axis(self.axis, param.ndim());

        let outputshape: Vec<usize> = {
            let former: &[usize] = &paramshape[..axis];
            let latter: &[usize] = &paramshape[axis + 1..];
            // doing former + indices.shape() + latter
            former
                .iter()
                .chain(indicesshape)
                .chain(latter)
                .cloned()
                .collect()
        };

        let flat_indices = if self.should_normalize_negative_indices {
            ndarray_ext::normalize_negative_axes(indices, paramshape[axis])
        } else {
            indices
                .map(|a| a.to_usize().expect("Invalid index value"))
                .iter()
                .cloned()
                .collect::<Vec<_>>()
        };
        let selected = ndarray_ext::select(
            param,
            scirs2_core::ndarray::Axis(axis),
            flat_indices.as_slice(),
        );
        let ret = selected
            .into_shape_with_order(outputshape.as_slice())
            .expect("Failed to reshape output");
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let x = ctx.input(0);
        let x1 = ctx.input(1);
        let gy = ctx.output_grad();
        let gx = Tensor::builder(ctx.graph())
            .append_input(x, false)
            .append_input(x1, false)
            .append_input(gy, false)
            .setshape(&shape(x))
            .build(GatherGrad { axis: self.axis });
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, Some(gx));
    }
}

impl<T: Float> op::Op<T> for GatherGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let indices = ctx.input(0);
        let param = &ctx.input(1);
        let paramshape = param.shape();
        let gy = &ctx.input(2);
        let axis = if self.axis == -1 {
            param.ndim()
        } else {
            self.axis as usize
        };

        // get read-only view of gy and reshape it
        let gy = {
            let former = &paramshape[..axis];
            let latter = &paramshape[axis + 1..];
            let shape: Vec<usize> = former
                .iter()
                .chain(&[indices.len()])
                .chain(latter)
                .cloned()
                .collect();
            gy.view()
                .into_shape_with_order(shape)
                .expect("Operation failed")
        };

        let mut gx = NdArray::zeros(param.shape());

        for (gy_sub, &i) in gy.axis_iter(scirs2_core::ndarray::Axis(axis)).zip(indices) {
            let i = i.to_isize().expect("Operation failed");
            // get gx's sub view
            let gx_sliced = unsafe {
                gx.slice_mut(
                    scirs2_core::ndarray::SliceInfo::<
                        _,
                        scirs2_core::ndarray::IxDyn,
                        scirs2_core::ndarray::IxDyn,
                    >::new(
                        (0..param.ndim())
                            .map(|dim| {
                                if dim == axis {
                                    SliceInfoElem::Slice {
                                        start: i,
                                        end: Some(i + 1),
                                        step: 1,
                                    }
                                } else {
                                    SliceInfoElem::Slice {
                                        start: 0,
                                        end: None,
                                        step: 1,
                                    }
                                }
                            })
                            .collect::<Vec<_>>(),
                    )
                    .expect("Failed to create slice")
                    .as_ref(),
                )
            };

            // squeeze
            let mut gx_sliced = gx_sliced.index_axis_move(scirs2_core::ndarray::Axis(axis), 0);
            // assign gy to sliced view
            gx_sliced.zip_mut_with(&gy_sub, |gx, &gy| {
                *gx += gy;
            });
        }

        ctx.append_output(gx);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}

#[cfg(feature = "blas")]
pub(crate) fn inplace_add_impl<F: Float>(mut a: NdArrayViewMut<F>, b: &NdArrayView<F>) {
    // Simplified fallback - just use element-wise addition
    a += b;
}

impl<T: Float> op::Op<T> for AddN {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let inputs_len = ctx.inputs().len();
        if inputs_len == 0 {
            // Previously `unreachable!()`: reachable in practice whenever a
            // caller builds `AddN` with no inputs (e.g. an op with zero
            // backprop consumers gets its gradients accumulated via AddN).
            // Report it as a normal `OpError` instead of panicking.
            return Err(op::OpError::IncompatibleShape(
                "AddN: requires at least one input, got 0".to_string(),
            ));
        }

        // All inputs must share exactly one shape: AddN exists to accumulate
        // several gradients flowing into the *same* tensor, so a shape
        // mismatch here means an upstream op produced a wrongly-shaped
        // gradient (see e.g. the einsum/tensordot backward gaps). ndarray's
        // `+`/`+=` operators panic on mismatched shapes rather than
        // returning a `Result`, which used to surface as an uncatchable
        // panic (or a `ShapeError`-flavoured abort) instead of a normal
        // `OpError` -- validate shapes explicitly first so this becomes a
        // recoverable error.
        let firstshape = ctx.input(0).shape().to_vec();
        for i in 1..inputs_len {
            let ishape = ctx.input(i).shape().to_vec();
            if ishape != firstshape {
                return Err(op::OpError::IncompatibleShape(format!(
                    "AddN: mismatched shapes -- input 0 has shape {firstshape:?}, \
                     input {i} has shape {ishape:?}"
                )));
            }
        }

        if inputs_len == 1 {
            let ret = ctx.input(0);
            ctx.append_output(ret.to_owned());
        } else {
            let mut base = ctx.input(0).to_owned();
            for i in 1..inputs_len {
                base += &ctx.input(i);
            }
            ctx.append_output(base);
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let gy = ctx.output_grad().to_owned();
        for i in 0..ctx.inputs().len() {
            ctx.append_input_grad(i, Some(gy));
        }
    }
}

impl<T: Float> op::Op<T> for Clip<T> {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(move |a| a.min(self.max).max(self.min));
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let gy = ctx.output_grad();
        let x0 = ctx.input(0);
        let gx = Tensor::builder(ctx.graph())
            .setshape(&shape(gy))
            .append_input(x0, false)
            .append_input(gy, false)
            .build(ClipGrad {
                min: self.min,
                max: self.max,
            });
        ctx.append_input_grad(0, Some(gx));
    }
}

impl<T: Float> op::Op<T> for ClipGrad<T> {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let mut ret = ctx.input(0).mapv(move |x| {
            // x > min && x < max
            T::from((((x > self.min) as i32) as f32) * (((x < self.max) as i32) as f32))
                .expect("Operation failed")
        });
        ret *= &ctx.input(1);
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for Concat {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let mut views = Vec::with_capacity(ctx.inputs().len());
        for i in 0..ctx.inputs().len() {
            views.push(ctx.input(i));
        }

        let axis = if self.axis < 0 {
            (ctx.input(0).ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        match scirs2_core::ndarray::concatenate(scirs2_core::ndarray::Axis(axis), views.as_slice())
        {
            Ok(y) => {
                ctx.append_output(y);
                Ok(())
            }
            Err(e) => Err(op::OpError::NdArrayError("concat".to_string(), e)),
        }
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        // ConcatGrad's inputs are laid out as [gy, x1, x2, x3, ...].
        let num_inputs = ctx.inputs().len();
        let output_grad = ctx.output_grad();
        let graph = ctx.graph();

        // Clone all inputs to avoid borrow issues
        let inputs: Vec<&Tensor<T>> = (0..num_inputs).map(|i| ctx.input(i)).collect();

        for i in 0..num_inputs {
            // The shape hint must be the shape of *this* input; concat accepts inputs
            // with different extents along `axis`, so reusing input 0's shape made
            // `shape(gx_i)` lie for every i > 0.
            let mut builder = Tensor::builder(graph)
                .setshape(&shape(inputs[i]))
                .append_input(output_grad, false);

            for input in &inputs {
                builder = builder.append_input(input, false);
            }

            let gx = builder.build(ConcatGrad {
                index: i,
                axis: self.axis,
            });
            ctx.append_input_grad(i, Some(gx));
        }
    }
}

impl<T: Float> op::Op<T> for ConcatGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let gy = ctx.input(0);

        let axis = if self.axis < 0 {
            (ctx.input(0).ndim() as isize + self.axis) as usize
        } else {
            self.axis as usize
        };

        // Inputs are laid out as `[gy, x0, x1, ..., xn]`, so the tensor concatenated at
        // position `j` is `ctx.input(j + 1)`.  The region of `gy` belonging to
        // `self.index` therefore starts after all of x0..x_{index-1} and ends
        // `region_len` later -- both the start offset (which skipped x_{index-1}) and the
        // end bound (which used the length as an absolute index) used to be wrong, so
        // every input except the first read back the *first* input's slice of `gy`.
        let mut start_idx = 0_usize;
        for i in 0..self.index {
            start_idx += ctx.input(i + 1).shape()[axis];
        }
        let region_len = ctx.input(self.index + 1).shape()[axis];
        let end_idx = (start_idx + region_len) as isize;
        let indices = (0..gy.ndim())
            .map(move |_axis| {
                if _axis == axis {
                    // partial region
                    SliceInfoElem::Slice {
                        start: start_idx as isize,
                        end: Some(end_idx),
                        step: 1,
                    }
                } else {
                    // full slice
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
            })
            .collect::<Vec<_>>();

        // Clone the *view*
        unsafe {
            match scirs2_core::ndarray::SliceInfo::<
                _,
                scirs2_core::ndarray::IxDyn,
                scirs2_core::ndarray::IxDyn,
            >::new(indices)
            {
                Ok(ok) => {
                    // do slice
                    let ret = gy.clone().slice_move(ok.as_ref());
                    ctx.append_output(ret.to_owned());
                    Ok(())
                }
                Err(e) => Err(op::OpError::NdArrayError("ConcatGrad: ".to_string(), e)),
            }
        }
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let inputs = ctx.inputs();
        for i in 0..inputs.len() {
            ctx.append_input_grad(i, None);
        }
    }
}

impl<T: Float> op::Op<T> for Tile {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let axis = ndarray_ext::normalize_negative_axis(self.axis, x.ndim());
        let views = vec![x.clone(); self.num];
        match scirs2_core::ndarray::concatenate(scirs2_core::ndarray::Axis(axis), views.as_slice())
        {
            Ok(ret) => {
                ctx.append_output(ret);
                Ok(())
            }
            Err(e) => Err(op::OpError::NdArrayError("tile: ".to_string(), e)),
        }
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(reduce_sum(ctx.output_grad(), &[self.axis], true)));
    }
}

impl<T: Float> op::Op<T> for Split {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = &ctx.input(0);
        let axis = ndarray_ext::normalize_negative_axis(self.axis, x.ndim());
        let mut ret = x.clone();
        let indices = make_indices_for_split(x, self.start_index, self.end_index, axis);
        ret.slice_collapse(indices.as_slice());
        ctx.append_output(ret.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let op = SplitGrad {
            axis: self.axis,
            start_index: self.start_index,
            end_index: self.end_index,
        };
        let x = ctx.input(0);
        let gy = ctx.output_grad();
        let gx = Tensor::builder(ctx.graph())
            .append_input(x, false)
            .append_input(gy, false)
            .setshape(&shape(x))
            .build(op);
        ctx.append_input_grad(0, Some(gx));
    }
}

impl<T: Float> op::Op<T> for SplitGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let mut gx = NdArray::zeros(x.shape());

        let axis = ndarray_ext::normalize_negative_axis(self.axis, x.ndim());
        let indices = make_indices_for_split(&x, self.start_index, self.end_index, axis);

        unsafe {
            gx.slice_mut(
                scirs2_core::ndarray::SliceInfo::<
                    _,
                    scirs2_core::ndarray::IxDyn,
                    scirs2_core::ndarray::IxDyn,
                >::new(indices)
                .expect("Failed to create indices")
                .as_ref(),
            )
            .zip_mut_with(&ctx.input(1), |a, &g| *a = g);
        }
        ctx.append_output(gx);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

#[inline]
#[allow(dead_code)]
fn make_indices_for_split<T: Float>(
    x: &NdArrayView<T>,
    start_index: isize,
    end_index: isize,
    axis: usize,
) -> Vec<SliceInfoElem> {
    let ndim = x.ndim();
    assert!(ndim > axis, "Wrong split axis");
    (0..ndim)
        .map(|i| {
            if i == axis {
                SliceInfoElem::Slice {
                    start: start_index,
                    end: Some(end_index),
                    step: 1,
                }
            } else {
                // full slice
                SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }
            }
        })
        .collect::<Vec<_>>()
}

impl<T: Float> op::Op<T> for Slice {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let mut y = ctx.input(0);
        y.slice_collapse(self.indices.as_slice());
        ctx.append_output(y.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let op = SliceGrad {
            indices: self.indices.clone(),
        };
        let x = ctx.input(0);
        let gy = ctx.output_grad();
        let gx = Tensor::builder(ctx.graph())
            .append_input(x, false)
            .append_input(gy, false)
            .setshape(&shape(x))
            .build(op);
        ctx.append_input_grad(0, Some(gx));
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

impl<T: Float> op::Op<T> for SliceGrad {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x = ctx.input(0);
        let mut gx = NdArray::zeros(x.shape());
        // sliced view
        unsafe {
            gx.slice_mut(
                scirs2_core::ndarray::SliceInfo::<
                    &[SliceInfoElem],
                    scirs2_core::ndarray::IxDyn,
                    scirs2_core::ndarray::IxDyn,
                >::new(&self.indices)
                .expect("Failed to create indices")
                .as_ref(),
            )
            .zip_mut_with(&ctx.input(1), |a, &g| *a = g);
        }
        ctx.append_output(gx);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        // is this ok?
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}
impl<T: Float> op::Op<T> for Squeeze {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let mut x = ctx.input(0).clone();
        let mut axes = ctx
            .input(1)
            .iter()
            .map(|a| a.to_isize().expect("Operation failed"))
            .collect::<Vec<_>>();
        axes.sort();
        for (adjust, &i) in axes.iter().enumerate() {
            let axis = if i < 0 {
                (x.ndim() as isize + i) as usize
            } else {
                i as usize
            };
            let axis = axis - adjust;
            assert_eq!(1, x.shape()[axis], "Can't squeeze a dim whose size != 1");
            // axis making ok
            x = x.index_axis_move(scirs2_core::ndarray::Axis(axis), 0);
        }
        ctx.append_output(x.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(expand_dims(ctx.output_grad(), ctx.input(1))));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ExpandDims {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0);
        let mut axes = ctx
            .input(1)
            .iter()
            .map(|a| a.to_isize().expect("Operation failed"))
            .collect::<Vec<_>>();
        axes.sort();
        let mut outputshape = ret.shape().to_vec();
        for &i in axes.iter() {
            let axis = if i < 0 {
                (ret.ndim() as isize + i) as usize
            } else {
                i as usize
            };
            outputshape.insert(axis, 1);
        }
        ctx.append_output(
            ret.into_shape_with_order(outputshape)
                .expect("Operation failed")
                .to_owned(),
        );
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(squeeze(ctx.output_grad(), ctx.input(1))));
        ctx.append_input_grad(1, None);
    }
}

#[cfg(test)]
mod addn_tests {
    // Regression tests for `AddN::compute` (see the impl above): it used to
    // reach `unreachable!()` for zero inputs and let ndarray's `+`/`+=`
    // operators panic on any shape mismatch instead of returning an
    // `OpError`. Both would have aborted the whole test process rather than
    // failing gracefully, so these tests would have crashed the test binary
    // before the fix and now correctly observe an `Err`.
    use crate::tensor::Tensor;
    use crate::tensor_ops::{add_n, concat, convert_to_tensor, grad, variable};

    #[test]
    fn add_n_with_zero_inputs_reports_error_not_unreachable_panic() {
        // `add_n(&[])` cannot be built through the public API (it asserts
        // `len != 0`), but nothing stops a raw `Tensor::builder(..).build(AddN)`
        // with no appended inputs, which used to hit `unreachable!()`.
        crate::run(|ctx: &mut crate::Context<f64>| {
            let zero_input_sum: Tensor<f64> = Tensor::builder(ctx).build(super::AddN);
            let result = zero_input_sum.eval(ctx);
            assert!(
                result.is_err(),
                "AddN with zero inputs must report an OpError, not panic via unreachable!()"
            );
        });
    }

    #[test]
    fn add_n_reports_shape_mismatch_as_error_not_panic() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let a = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[3]),
                    vec![1.0f64, 2.0, 3.0],
                )
                .expect("Operation failed"),
                ctx,
            );
            let b = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[2]),
                    vec![4.0f64, 5.0],
                )
                .expect("Operation failed"),
                ctx,
            );

            let sum = add_n(&[a, b]);
            let result = sum.eval(ctx);
            assert!(
                result.is_err(),
                "AddN must report a shape mismatch as an OpError, not panic or silently succeed"
            );
        });
    }

    #[test]
    fn add_n_still_sums_matching_shapes_correctly() {
        // Guards against an overzealous fix that rejects legitimate,
        // same-shape accumulation.
        crate::run(|ctx: &mut crate::Context<f64>| {
            let a = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[3]),
                    vec![1.0f64, 2.0, 3.0],
                )
                .expect("Operation failed"),
                ctx,
            );
            let b = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[3]),
                    vec![10.0f64, 20.0, 30.0],
                )
                .expect("Operation failed"),
                ctx,
            );
            let c = convert_to_tensor(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[3]),
                    vec![100.0f64, 200.0, 300.0],
                )
                .expect("Operation failed"),
                ctx,
            );

            let sum = add_n(&[a, b, c]).eval(ctx).expect("Operation failed");
            assert_eq!(
                sum.iter().copied().collect::<Vec<_>>(),
                vec![111.0f64, 222.0, 333.0]
            );
        });
    }

    /// `concat(&[x, x], 0)` feeds `x` into the same `Concat` op twice, so its
    /// backward pass produces two separate gradient contributions for `x`
    /// that must be summed via `AddN`. Before the gradient-dispatch routing
    /// fix landed (see `gradient.rs`) this combination panicked inside
    /// `AddN::compute`. Non-constant data: a uniform (all-ones/all-equal)
    /// `x` could not distinguish "correctly doubled gradient" from a
    /// fabricated or mis-routed one. Built with `variable` (not
    /// `convert_to_tensor`, which marks its output non-differentiable and
    /// would make any gradient -- correct or not -- silently collapse to
    /// zero; see `tests/gradient_fd_harness.rs`'s header comment).
    #[test]
    fn concat_self_backward_no_longer_panics_and_doubles_gradient() {
        crate::run(|ctx: &mut crate::Context<f64>| {
            let x = variable(
                scirs2_core::ndarray::Array::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[3]),
                    vec![1.0f64, 2.0, 3.0],
                )
                .expect("Operation failed"),
                ctx,
            );

            let y = concat(&[x, x], 0);
            let yshape = y.eval(ctx).expect("Operation failed").shape().to_vec();
            assert_eq!(yshape, vec![6]);

            let gx = grad(&[y], &[x])[0];
            let gx_val = gx.eval(ctx).expect("Operation failed");
            // Each element of `x` feeds two positions of `y`, so d(sum(y))/dx
            // is 2 everywhere, regardless of x's own (non-constant) values.
            assert_eq!(gx_val.iter().copied().collect::<Vec<_>>(), vec![2.0f64; 3]);
        });
    }
}
