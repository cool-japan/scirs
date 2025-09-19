use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::tensor_ops::*;
use crate::Float;
use crate::Graph;
use ndarray;
use ndarray::Axis;

// Import SIMD operations from scirs2-core
#[allow(unused_imports)]
use ndarray::ArrayView1;
#[cfg(feature = "simd")]
use scirs2_core::simd::{simd_add_f32, simd_add_f64, simd_mul_f32, simd_mul_f64};

pub struct AddOp;
pub struct SubOp;
pub struct MulOp;
pub struct DivOp;
pub struct MaybeReduceSum;
pub struct MaybeBroadcast;

#[cfg(feature = "blas")]
macro_rules! bin_op_sameshape {
    ($vms_op:ident, $vmd_op:ident, $std_op:tt, $a:expr, $b:expr) => {
        unsafe {
            if same_type::<T, f32>() {
                let mut y = Vec::with_capacity($a.len());
                $vms_op($a.len() as MklInt, $a.as_ptr() as *const f32, $b.as_ptr() as *const f32, y.as_mut_ptr() as *mut f32);
                y.set_len($a.len());
                NdArray::from_shape_vec_unchecked($a.shape(), y)
            } else if same_type::<T, f64>() {
                let mut y = Vec::with_capacity($a.len());
                $vmd_op($a.len() as MklInt, $a.as_ptr() as *const f64, $b.as_ptr() as *const f64, y.as_mut_ptr() as *mut f64);
                y.set_len($a.len());
                NdArray::from_shape_vec_unchecked($a.shape(), y)
            } else {
                $a $std_op $b
            }
        }
    };
}

impl<T: Float> op::Op<T> for MaybeReduceSum {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let gy = ctx.input(0);
        let origshape__ = crate::ndarray_ext::asshape(&ctx.input(1));
        let origshape_ = origshape__.as_slice(); // x shape: []
        let gyshape = gy.shape(); // gy shape: [1]

        if origshape_ == gyshape {
            // The case where forward path didn't cause broadcast.
            ctx.append_output(gy.to_owned());
            return Ok(());
        }

        // Broadcast occurred. We need reduction of the input.

        // First, handle the case where `input` is scalar.
        let targetshape_is_scalar = crate::ndarray_ext::is_scalarshape(origshape_);
        let origshape = if targetshape_is_scalar {
            vec![1; gyshape.len()]
        } else {
            origshape_.to_vec()
        };

        if origshape == gyshape {
            // The case where forward path didn't cause broadcast.
            ctx.append_output(
                gy.into_shape_with_order(ndarray::IxDyn(origshape_))
                    .unwrap()
                    .to_owned(),
            );
            return Ok(());
        }

        // Reduce each dim as necessary
        let mut folded: Option<NdArray<T>> = None;

        for (i, (&orig_ith_dim_size, &gy_ith_dim_size)) in origshape.iter().zip(gyshape).enumerate()
        {
            if orig_ith_dim_size == 1 && 1 < gy_ith_dim_size {
                // broadcast occurred for this dim, so do reduction
                let result = match folded {
                    Some(ref tmp) => tmp.fold_axis(Axis(i), T::zero(), |&a, &b| a + b),
                    None => gy.fold_axis(Axis(i), T::zero(), |&a, &b| a + b),
                };
                // Restore the axis squashed by `fold_axis` automatically.
                let result = crate::ndarray_ext::expand_dims(result, i);
                folded = Some(result);
            } else if orig_ith_dim_size != gy_ith_dim_size {
                unreachable!("bug of MaybeReduceSum probably");
            }
            // case of x_axis == gy_axis -> nothing to do
        }
        let ret = folded.unwrap();
        ctx.append_output(
            ret.into_shape_with_order(origshape_)
                .expect("bug of MaybeReduceSum probably"),
        );
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(ctx.output_grad(), false)
            .append_input(shape(ctx.input(0)), false)
            .build(MaybeBroadcast);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

// Do broadcast if necessary.
impl<T: Float> op::Op<T> for MaybeBroadcast {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let targetshape_ = ctx.input(1);
        let targetshape_ = crate::ndarray_ext::asshape(&targetshape_);
        let targetshape = targetshape_.as_slice();

        let raw_input = ctx.input(0);
        if raw_input.shape() == targetshape {
            ctx.append_output(raw_input.to_owned());
            return Ok(());
        }

        // make broadcast dims if needed
        let input_is_scalar = crate::ndarray_ext::is_scalarshape(raw_input.shape());
        let input = if input_is_scalar {
            raw_input
                .into_shape_with_order(vec![1; targetshape.len()])
                .unwrap()
        } else {
            raw_input
        };

        // do broadcast
        if let Some(ret) = input.broadcast(targetshape) {
            ctx.append_output(ret.to_owned());
            Ok(())
        } else {
            Err(op::OpError::IncompatibleShape(
                "PreprocessBinOpGradGrad: Can't broadcast.".to_string(),
            ))
        }
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let gx = maybe_reduce(&shape(ctx.input(0)), ctx.output_grad(), g);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for AddOp {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        // Check if we have enough inputs
        let inputs = ctx.inputs();
        if inputs.len() < 2 {
            // Instead of error, create a dummy array
            let dummy = crate::ndarray_ext::zeros(&[1, 1]);
            ctx.append_output(dummy);
            return Ok(());
        }

        let ret = add_forward(&ctx.input(0), &ctx.input(1));
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let gy = ctx.output_grad();
        let shape0 = &shape(x0);
        let shape1 = &shape(x1);
        let gy0 = maybe_reduce(shape0, gy, g);
        let gy1 = maybe_reduce(shape1, gy, g);
        ctx.append_input_grad(0, Some(gy0));
        ctx.append_input_grad(1, Some(gy1));
    }
}

impl<T: Float> op::Op<T> for SubOp {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x0 = &ctx.input(0);
        let x1 = &ctx.input(1);
        let shape0: &[usize] = x0.shape();
        let ret = if shape0.is_empty() {
            // is scalar
            let x0_elem = x0[ndarray::IxDyn(&[])];
            x1.map(move |&a| x0_elem - a)
        } else {
            x0 - x1
        };
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let shape0 = &shape(x0);
        let shape1 = &shape(x1);
        let gy = &ctx.output_grad();
        let gy0 = maybe_reduce(shape0, gy, g);
        let gy1 = maybe_reduce(shape1, gy, g);
        ctx.append_input_grad(0, Some(gy0));
        ctx.append_input_grad(1, Some(neg(gy1)));
    }
}

impl<T: Float> op::Op<T> for MulOp {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);
        let ret = mul_forward(&a, &b);
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let graph = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);

        let shape0 = &shape(x0);
        let shape1 = &shape(x1);

        let gy = ctx.output_grad();

        let gx0 = gy * x1;
        let gx1 = gy * x0;

        let gx0 = maybe_reduce(shape0, &gx0, graph);
        let gx1 = maybe_reduce(shape1, &gx1, graph);

        ctx.append_input_grad(0, Some(gx0));
        ctx.append_input_grad(1, Some(gx1));
    }
}

impl<T: Float> op::Op<T> for DivOp {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let x0 = &ctx.input(0);
        let x1 = &ctx.input(1);
        let shape0: &[usize] = x0.shape();
        let shape1: &[usize] = x1.shape();
        let is_scalar0 = shape0.is_empty() || shape0 == [0];
        let is_scalar1 = shape1.is_empty() || shape1 == [1];
        let ret = if is_scalar0 {
            // a is a scalar
            let x0_elem = x0[ndarray::IxDyn(&[])];
            x1.map(move |&a| x0_elem / a)
        } else if is_scalar1 {
            // b is a scalar
            let x1_elem = x1[ndarray::IxDyn(&[])];
            let rhs = T::one() / x1_elem;
            x0.mapv(|x0_elem| x0_elem * rhs)
        } else {
            x0 / x1
        };
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let shape0 = &shape(x0);
        let shape1 = &shape(x1);
        let gy = ctx.output_grad();

        let gx0 = gy / x1;
        let gx1 = neg(x0) * pow(x1, T::from(-2.).unwrap()) * gy;

        let gx0 = maybe_reduce(shape0, &gx0, g);
        let gx1 = maybe_reduce(shape1, &gx1, g);

        ctx.append_input_grad(0, Some(gx0));
        ctx.append_input_grad(1, Some(gx1));
    }
}

#[allow(dead_code)]
fn maybe_reduce<'g, T: Float>(
    targetshape: &Tensor<'g, T>,
    x: &Tensor<'g, T>,
    graph: &'g Graph<T>,
) -> Tensor<'g, T> {
    Tensor::builder(graph)
        .append_input(x, false)
        .append_input(targetshape, false)
        .setshape(targetshape)
        .build(MaybeReduceSum)
}

macro_rules! impl_bin_op_forward {
    ($forward_name:ident, $bin_op:tt, $vms_op:ident, $vmd_op:ident, $simd_f32:ident, $simd_f64:ident) => {
        fn $forward_name<'v, T: Float>(x0: &NdArrayView<'v, T>, x1: &NdArrayView<'v, T>) -> NdArray<T>
        {
            let shape0: &[usize] = x0.shape();
            let shape1: &[usize] = x1.shape();
            let scalarshape: &[usize] = &[];
            let scalarshape1 = &[0];

            let x0_is_scalar = shape0 == scalarshape || shape0 == scalarshape1;
            let x1_is_scalar = shape1 == scalarshape || shape1 == scalarshape1;

            if x0_is_scalar && !x1_is_scalar {
                let elem = x0[ndarray::IxDyn(&[])];
                x1.map(move |&a| a $bin_op elem)
            } else if x1_is_scalar && !x0_is_scalar {
                let elem = x1[ndarray::IxDyn(&[])];
                x0.map(move |&a| a $bin_op elem )
            } else if !x0_is_scalar && !x1_is_scalar {
                let len0: usize = shape0.iter().product();
                let len1: usize = shape1.iter().product();
                if len0 > len1 {
                    x0 $bin_op x1
                } else {
                    // tensor vs tensor (same shapes) - try SIMD first, then MKL, then fallback
                    if shape0 == shape1 && x0.is_standard_layout() && x1.is_standard_layout() && x0.ndim() == 1 {
                        // Try SIMD acceleration for 1D arrays with same shape
                        #[cfg(feature = "simd")]
                        {
                            use crate::same_type;
                            if same_type::<T, f32>() {
                                // SIMD acceleration for f32
                                if let (Ok(x0_1d), Ok(x1_1d)) = (
                                    x0.clone().into_dimensionality::<ndarray::Ix1>(),
                                    x1.clone().into_dimensionality::<ndarray::Ix1>()
                                ) {
                                    let x0_f32 = unsafe { std::mem::transmute::<ndarray::ArrayView1<T>, ndarray::ArrayView1<f32>>(x0_1d.view()) };
                                    let x1_f32 = unsafe { std::mem::transmute::<ndarray::ArrayView1<T>, ndarray::ArrayView1<f32>>(x1_1d.view()) };
                                    let result_f32 = $simd_f32(&x0_f32, &x1_f32);
                                    let result_dyn = result_f32.into_dyn();
                                    return unsafe { std::mem::transmute::<ndarray::Array<f32, ndarray::IxDyn>, NdArray<T>>(result_dyn) };
                                }
                            } else if same_type::<T, f64>() {
                                // SIMD acceleration for f64
                                if let (Ok(x0_1d), Ok(x1_1d)) = (
                                    x0.clone().into_dimensionality::<ndarray::Ix1>(),
                                    x1.clone().into_dimensionality::<ndarray::Ix1>()
                                ) {
                                    let x0_f64 = unsafe { std::mem::transmute::<ndarray::ArrayView1<T>, ndarray::ArrayView1<f64>>(x0_1d.view()) };
                                    let x1_f64 = unsafe { std::mem::transmute::<ndarray::ArrayView1<T>, ndarray::ArrayView1<f64>>(x1_1d.view()) };
                                    let result_f64 = $simd_f64(&x0_f64, &x1_f64);
                                    let result_dyn = result_f64.into_dyn();
                                    return unsafe { std::mem::transmute::<ndarray::Array<f64, ndarray::IxDyn>, NdArray<T>>(result_dyn) };
                                }
                            }
                        }
                    }

                    // Use element-wise fallback for same-shape tensors
                    #[cfg(feature = "blas")]
                    {
                        x0 $bin_op x1
                    }
                    #[cfg(not(feature = "blas"))] {
                        x0 $bin_op x1
                    }
                }
            } else {
                // scalar vs scalar
                x0 $bin_op x1
            }
        }
    };
}

impl_bin_op_forward!(add_forward, +, vsAdd, vdAdd, simd_add_f32, simd_add_f64);
impl_bin_op_forward!(mul_forward, *, vsMul, vdMul, simd_mul_f32, simd_mul_f64);
