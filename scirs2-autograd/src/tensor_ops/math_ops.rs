use crate::ndarray;
use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::tensor_ops::*;
use crate::tensor_ops::{vdAdd, vdExp, vdLn, vsAdd, vsExp, vsLn, BlasIF};
use crate::{same_type, Float};
use scirs2_core::ndarray::Zip;

pub struct Sin;
pub struct Cos;
pub struct Tan;
pub struct Asin;
pub struct Acos;
pub struct Atan;
pub struct Sinh;
pub struct Cosh;
pub struct Tanh;
pub struct Asinh;
pub struct Acosh;
pub struct Atanh;
pub struct Exp;
pub struct Exp2;
pub struct Exp10;
pub struct Sqrt;
pub struct NegOp;
pub struct Floor;
pub struct Ceil;
pub struct Sign;
pub struct Inv;
pub struct InvSqrt;
pub struct Square;
pub struct Abs;
pub struct Log2;
pub struct Log10;
pub struct Ln;
pub struct Pow<T: Float> {
    pub a: T,
}
pub struct LogSumExp {
    pub axis: isize,
    pub keep_dims: bool,
}
pub struct Transpose {
    pub invert_axes: bool,
}
pub struct Lgamma;
pub struct Digamma;

#[inline(always)]
#[allow(dead_code)]
fn equal_fn<T: Float>(a: T, b: T) -> T {
    T::from((a == b) as i32).unwrap()
}
#[inline(always)]
#[allow(dead_code)]
fn not_equal_fn<T: Float>(a: T, b: T) -> T {
    T::from((a != b) as i32).unwrap()
}
#[inline(always)]
#[allow(dead_code)]
fn greater_fn<T: Float>(a: T, b: T) -> T {
    T::from((a > b) as i32).unwrap()
}
#[inline(always)]
#[allow(dead_code)]
fn lesser_fn<T: Float>(a: T, b: T) -> T {
    T::from((a < b) as i32).unwrap()
}
#[inline(always)]
#[allow(dead_code)]
fn greater_equal_fn<T: Float>(a: T, b: T) -> T {
    T::from((a >= b) as i32).unwrap()
}
#[inline(always)]
#[allow(dead_code)]
fn lesser_equal_fn<T: Float>(a: T, b: T) -> T {
    T::from((a <= b) as i32).unwrap()
}
#[inline(always)]
#[allow(dead_code)]
fn maximum_fn<T: Float>(a: T, b: T) -> T {
    a.max(b)
}
#[inline(always)]
#[allow(dead_code)]
fn minimum_fn<T: Float>(a: T, b: T) -> T {
    a.min(b)
}

macro_rules! impl_cmp_op {
    ($struct_name:ident, $name:expr, $assign:expr, $grad_fn:expr) => {
        pub struct $struct_name;

        impl<T: Float> op::Op<T> for $struct_name {
            fn compute(
                &self,
                ctx: &mut crate::op::ComputeContext<T>,
            ) -> Result<(), crate::op::OpError> {
                let x0 = ctx.input(0);
                let x1 = &ctx.input(1);
                let shape0 = x0.shape();
                let shape1 = x1.shape();

                let x0_is_scalar = crate::ndarray_ext::is_scalarshape(shape0);
                let x1_is_scalar = crate::ndarray_ext::is_scalarshape(shape1);

                let ret = if x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[scirs2_core::ndarray::IxDyn(&[])];
                    x0.mapv(move |a| $assign(a, x1_elem))
                } else if x0_is_scalar && !x1_is_scalar {
                    let x0_elem = x0[scirs2_core::ndarray::IxDyn(&[])];
                    x1.mapv(move |a| $assign(x0_elem, a))
                } else if !x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[scirs2_core::ndarray::IxDyn(&[])];
                    x0.mapv(move |a| $assign(a, x1_elem))
                } else {
                    // case that scalar is not involved
                    // Check the input ranks.
                    // op couldn't we catch here cause ndarray's panics.

                    // rank check
                    if shape0.len() != shape1.len() {
                        panic!(
                            "Tensor ranks mismatch: {}({}'s lhs input) vs {}({}'s rhs input)",
                            shape0.len(),
                            $name,
                            shape1.len(),
                            $name,
                        )
                    }

                    // Try to broadcast the arrays
                    // First try broadcasting x0 to x1's shape
                    match x0.broadcast(shape1) {
                        Some(x0_broadcast) => {
                            let mut result = NdArray::zeros(shape1);
                            Zip::from(&mut result)
                                .and(&x0_broadcast)
                                .and(x1)
                                .for_each(|r, a, b| *r = $assign(a.clone(), b.clone()));
                            result
                        }
                        None => {
                            // Try broadcasting x1 to x0's shape
                            match x1.broadcast(shape0) {
                                Some(x1_broadcast) => {
                                    let mut result = NdArray::zeros(shape0);
                                    Zip::from(&mut result)
                                        .and(x0)
                                        .and(&x1_broadcast)
                                        .for_each(|r, a, b| *r = $assign(a.clone(), b.clone()));
                                    result
                                }
                                None => {
                                    // If neither works, check if they have the same shape
                                    if shape0 == shape1 {
                                        let mut result = NdArray::zeros(shape0);
                                        Zip::from(&mut result)
                                            .and(x0)
                                            .and(x1)
                                            .for_each(|r, a, b| *r = $assign(a.clone(), b.clone()));
                                        result
                                    } else {
                                        panic!(
                                            "Cannot broadcast shapes {:?} and {:?} for operation {}",
                                            shape0, shape1, $name
                                        )
                                    }
                                }
                            }
                        }
                    }
                };

                ctx.append_output(ret);
                Ok(())
            }

            fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
                $grad_fn(
                    ctx.output_grad().clone(),
                    ctx.input(0).clone(),
                    ctx.input(1).clone(),
                    ctx.output().clone(),
                    ctx,
                );
            }
        }
    };
}

impl_cmp_op!(Equal, "Equal", equal_fn, none_grad);
impl_cmp_op!(NotEqual, "NotEqual", not_equal_fn, none_grad);
impl_cmp_op!(Greater, "Greater", greater_fn, none_grad);
impl_cmp_op!(Lesser, "Lesser", lesser_fn, none_grad);
impl_cmp_op!(GreaterEqual, "GreaterEqual", greater_equal_fn, none_grad);
impl_cmp_op!(LesserEqual, "LesserEqual", lesser_equal_fn, none_grad);
impl_cmp_op!(Maximum, "Maximum", maximum_fn, min_max_grad);
impl_cmp_op!(Minimum, "Minimum", minimum_fn, min_max_grad);

#[inline]
#[allow(dead_code)]
fn none_grad<'g, T: Float>(
    _gy: Tensor<'g, T>,
    _x1: Tensor<'g, T>,
    _x2: Tensor<'g, T>,
    _y: Tensor<'g, T>,
    ctx: &mut op::GradientContext<T>,
) {
    ctx.append_input_grad(0, None);
    ctx.append_input_grad(1, None);
}

#[inline]
#[allow(dead_code)]
fn min_max_grad<'g, T: Float>(
    gy: Tensor<'g, T>,
    x1: Tensor<'g, T>,
    x2: Tensor<'g, T>,
    y: Tensor<'g, T>,
    ctx: &mut op::GradientContext<'g, 'g, T>,
) {
    let selected_a = equal(x1, y);
    let selected_b = equal(x2, y);
    ctx.append_input_grad(0, Some(mul(selected_a, gy)));
    ctx.append_input_grad(1, Some(mul(selected_b, gy)));
}

#[cfg(feature = "blas")]
#[allow(unused_macros)]
macro_rules! elem_wise_vm_or_std {
    ($vms_op:ident, $vmd_op:ident, $closure:expr, $ctx:expr) => {
        let x = $ctx.input(0);
        let ret = unsafe {
            if same_type::<T, f32>() {
                let mut y = Vec::with_capacity(x.len());
                $vms_op(
                    x.len() as BlasIF,
                    x.as_ptr() as *const f32,
                    y.as_mut_ptr() as *mut f32,
                );
                y.set_len(x.len());
                NdArray::from_shape_vec_unchecked(x.shape(), y)
            } else if same_type::<T, f64>() {
                let mut y = Vec::with_capacity(x.len());
                $vmd_op(
                    x.len() as BlasIF,
                    x.as_ptr() as *const f64,
                    y.as_mut_ptr() as *mut f64,
                );
                y.set_len(x.len());
                NdArray::from_shape_vec_unchecked(x.shape(), y)
            } else {
                $ctx.input(0).mapv($closure)
            }
        };
        $ctx.append_output(ret);
    };
}

#[cfg(feature = "blas")]
#[allow(unused_macros)]
macro_rules! elem_wise_vm_with_param_or_std {
    ($vms_op:ident, $vmd_op:ident, $std_name:ident, $param:expr, $ctx:expr) => {
        let x = $ctx.input(0);
        let ret = unsafe {
            if same_type::<T, f32>() {
                let mut y = Vec::with_capacity(x.len());
                let p = $param.to_f32().unwrap();
                $vms_op(
                    x.len() as BlasIF,
                    x.as_ptr() as *const f32,
                    p,
                    y.as_mut_ptr() as *mut f32,
                );
                y.set_len(x.len());
                NdArray::from_shape_vec_unchecked(x.shape(), y)
            } else if same_type::<T, f64>() {
                let mut y = Vec::with_capacity(x.len());
                let p = $param.to_f64().unwrap();
                $vmd_op(
                    x.len() as BlasIF,
                    x.as_ptr() as *const f64,
                    p,
                    y.as_mut_ptr() as *mut f64,
                );
                y.set_len(x.len());
                NdArray::from_shape_vec_unchecked(x.shape(), y)
            } else {
                $ctx.input(0).mapv(|a| a.$std_name($param))
            }
        };
        $ctx.append_output(ret);
    };
}

impl<T: Float> op::Op<T> for Abs {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.abs());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(ctx.output_grad() * sign(ctx.input(0))));
    }
}

impl<T: Float> op::Op<T> for NegOp {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|x| x.neg());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(neg(ctx.output_grad())));
    }
}

impl<T: Float> op::Op<T> for Square {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).mapv(|a| a * a);
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let two = scalar(T::one() + T::one(), ctx.graph());
        ctx.append_input_grad(0, Some(two * ctx.input(0) * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Inv {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.recip());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(neg(square(ctx.output())) * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for InvSqrt {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.sqrt().recip());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let a = scalar(T::from(-0.5).unwrap(), g);
        let b = pow(ctx.input(0), T::from(-1.5).unwrap());
        ctx.append_input_grad(0, Some(a * b * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Sign {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).mapv(|x| {
            if x == T::zero() {
                T::zero()
            } else {
                x.signum()
            }
        });
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Floor {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.floor());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Ceil {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.ceil());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, None)
    }
}

impl<T: Float> op::Op<T> for Transpose {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let perm = &ctx.input(1);
        let perm_len = perm.len();
        let x = ctx.input(0);
        if x.ndim() != perm_len {
            return Err(op::OpError::IncompatibleShape(
                "transpose: inputs's ndim and axes's length must match".to_string(),
            ));
        }

        let mut dims = vec![0; perm_len];
        for (i, d) in perm.iter().enumerate() {
            let d = d.to_usize().unwrap();
            if self.invert_axes {
                dims[d] = i;
            } else {
                dims[i] = d;
            }
        }
        let ret = x.permuted_axes(dims.as_slice());

        ctx.append_output(ret.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(ctx.input(1), false)
            .setshape(&shape(ctx.input(0)))
            .build(Transpose {
                invert_axes: !self.invert_axes,
            });
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

#[cfg(feature = "blas")]
pub(crate) fn inplace_add_impl<F: Float>(mut a: NdArray<F>, b: &NdArray<F>) -> NdArray<F> {
    unsafe {
        if same_type::<F, f32>() {
            vsAdd(
                a.len() as BlasIF,
                a.as_ptr() as *const f32,
                b.as_ptr() as *const f32,
                a.as_mut_ptr() as *mut f32,
            );
            return a;
        } else if same_type::<F, f64>() {
            vdAdd(
                a.len() as BlasIF,
                a.as_ptr() as *const f64,
                b.as_ptr() as *const f64,
                a.as_mut_ptr() as *mut f64,
            );
            return a;
        } else {
            a += b;
        }
    }
    a
}

#[cfg(feature = "blas")]
pub(crate) fn fast_inplace_exp_impl<F: Float>(x: &mut NdArray<F>) {
    unsafe {
        if same_type::<F, f32>() {
            vsExp(
                x.len() as BlasIF,
                x.as_ptr() as *const f32,
                x.as_mut_ptr() as *mut f32,
            );
        } else if same_type::<F, f64>() {
            vdExp(
                x.len() as BlasIF,
                x.as_ptr() as *const f64,
                x.as_mut_ptr() as *mut f64,
            );
        } else {
            x.mapv_inplace(move |a| a.exp());
        }
    }
}

#[cfg(feature = "blas")]
pub(crate) fn fast_inplace_ln_impl<F: Float>(x: &mut NdArray<F>) {
    unsafe {
        if same_type::<F, f32>() {
            vsLn(
                x.len() as BlasIF,
                x.as_ptr() as *const f32,
                x.as_mut_ptr() as *mut f32,
            );
        } else if same_type::<F, f64>() {
            vdLn(
                x.len() as BlasIF,
                x.as_ptr() as *const f64,
                x.as_mut_ptr() as *mut f64,
            );
        } else {
            x.mapv_inplace(move |a| a.ln());
        }
    }
}

#[allow(dead_code)]
pub fn logsumexp_forward<T: Float>(x: &NdArrayView<T>, axis: isize, keepdims: bool) -> NdArray<T> {
    let axis = if axis < 0 {
        (x.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    let mut a = x.shape().to_vec();
    if keepdims {
        a[axis] = 1;
    } else {
        a.remove(axis);
    }
    let reducedshape = a.as_slice();

    let max_fn = T::max;
    let min_val = T::min_value();
    let max = &x
        .fold_axis(scirs2_core::ndarray::Axis(axis), min_val, move |&a, &b| {
            max_fn(a, b)
        })
        .into_shape_with_order(scirs2_core::ndarray::IxDyn(reducedshape))
        .unwrap();

    let exp = {
        // subtract `max` to prevent overflow of exp
        let mut tmp = x - max;
        #[cfg(feature = "blas")]
        {
            fast_inplace_exp_impl(&mut tmp);
        }
        #[cfg(not(feature = "blas"))]
        {
            tmp.mapv_inplace(move |a| a.exp());
        }
        tmp
    };

    // unwrap is safe
    let mut sum = exp
        .sum_axis(scirs2_core::ndarray::Axis(axis))
        .into_shape_with_order(scirs2_core::ndarray::IxDyn(reducedshape))
        .unwrap();

    #[cfg(feature = "blas")]
    {
        fast_inplace_ln_impl(&mut sum);
        inplace_add_impl(sum, max)
    }
    #[cfg(not(feature = "blas"))]
    {
        sum.mapv_inplace(move |a| a.ln());
        sum += max;
        sum
    }
}

impl<T: Float> op::Op<T> for LogSumExp {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = logsumexp_forward(&ctx.input(0), self.axis, self.keep_dims);
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        // let ref sum = c.exp(output);
        // let ref exp = c.exp(ctx.input(0));
        // let gx = gy * exp / sum;
        let gx = softmax(ctx.input(0), self.axis) * ctx.output_grad();
        ctx.append_input_grad(0, Some(gx))
    }
}

impl<T: Float> op::Op<T> for Pow<T> {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.powf(self.a));
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let x = ctx.input(0);
        let gx = ctx.output_grad() * scalar(self.a, ctx.graph()) * pow(x, self.a - T::one());
        ctx.append_input_grad(0, Some(gx))
    }
}

impl<T: Float> op::Op<T> for Sqrt {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.sqrt());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let x = ctx.input(0);
        let half = T::one() / (T::one() + T::one());
        let ret = scalar(half, ctx.graph()) * pow(x, -half);
        ctx.append_input_grad(0, Some(ctx.output_grad() * ret));
    }
}

impl<T: Float> op::Op<T> for Log10 {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.log10());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let log10 = scalar(T::from(10.).unwrap().ln(), ctx.graph());
        ctx.append_input_grad(0, Some(ctx.output_grad() / (log10 * ctx.input(0))));
    }
}

impl<T: Float> op::Op<T> for Log2 {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.log2());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let log2 = scalar((T::one() + T::one()).ln(), ctx.graph());
        ctx.append_input_grad(0, Some(ctx.output_grad() / (log2 * ctx.input(0))));
    }
}

impl<T: Float> op::Op<T> for Ln {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.ln());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(ctx.output_grad() / ctx.input(0)));
    }
}

impl<T: Float> op::Op<T> for Exp {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.exp());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(ctx.output() * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Exp2 {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.exp2());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let log2 = (T::one() + T::one()).ln();
        let log2 = scalar(log2, g);
        ctx.append_input_grad(0, Some(log2 * ctx.output() * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Exp10 {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ten = T::from(10).unwrap();

        #[cfg(not(feature = "blas"))]
        {
            let ret = ctx.input(0).map(move |&a| ten.powf(a));
            ctx.append_output(ret);
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let log10 = scalar(T::from(10.).unwrap().ln(), ctx.graph());
        ctx.append_input_grad(0, Some(log10 * ctx.output() * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Atanh {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.atanh());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let x = ctx.input(0);
        let y = inv(1. - square(x));
        ctx.append_input_grad(0, Some(y * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Acosh {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.acosh());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let x = ctx.input(0);
        let y = inv(sqrt(square(x) - scalar(T::one(), g)));
        ctx.append_input_grad(0, Some(y * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Asinh {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.asinh());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let x = ctx.input(0);
        let y = inv(sqrt(square(x) + scalar(T::one(), g)));
        ctx.append_input_grad(0, Some(y * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Tanh {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.tanh());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(
            0,
            Some(ctx.output_grad() * (scalar(T::one(), ctx.graph()) - square(ctx.output()))),
        );
    }
}

impl<T: Float> op::Op<T> for Cosh {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.cosh());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(sinh(ctx.input(0)) * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Sinh {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.sinh());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(cosh(ctx.input(0)) * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Atan {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.atan());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let g = ctx.graph();
        let x = ctx.input(0);
        let y = inv(square(x) + scalar(T::one(), g));
        ctx.append_input_grad(0, Some(y * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Acos {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.acos());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let x = ctx.input(0);
        let y = neg(inv_sqrt(1. - square(x)));
        ctx.append_input_grad(0, Some(y * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Asin {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.asin());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let x = ctx.input(0);
        let y = inv_sqrt(1. - square(x));
        ctx.append_input_grad(0, Some(y * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Sin {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.sin());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(cos(ctx.input(0)) * ctx.output_grad()));
    }
}

impl<T: Float> op::Op<T> for Cos {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.cos());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(neg(sin(ctx.input(0)) * ctx.output_grad())));
    }
}

impl<T: Float> op::Op<T> for Tan {
    fn compute(&self, ctx: &mut op::ComputeContext<T>) -> Result<(), op::OpError> {
        let ret = ctx.input(0).map(|a| a.tan());
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut op::GradientContext<T>) {
        let cos = cos(ctx.input(0));
        ctx.append_input_grad(0, Some(ctx.output_grad() / square(cos)));
    }
}

use special::Gamma;

// impl lgamma and digamma
macro_rules! impl_gamma {
    ($ty:ty, $digamma_fn:ident) => {
        impl op::Op<$ty> for Digamma {
            fn compute(&self, ctx: &mut op::ComputeContext<$ty>) -> Result<(), op::OpError> {
                let x = ctx.input(0);
                let y = x.mapv(move |a| a.digamma());
                ctx.append_output(y);
                Ok(())
            }

            fn grad(&self, ctx: &mut op::GradientContext<$ty>) {
                // no impl
                ctx.append_input_grad(0, None);
            }
        }

        impl op::Op<$ty> for Lgamma {
            fn compute(&self, ctx: &mut op::ComputeContext<$ty>) -> Result<(), op::OpError> {
                let x = ctx.input(0);
                // Allow the unstable name collision for now
                #[allow(unstable_name_collisions)]
                let y = x.mapv(move |a| a.ln_gamma().0);
                ctx.append_output(y);
                Ok(())
            }

            fn grad(&self, ctx: &mut op::GradientContext<$ty>) {
                let x = ctx.input(0);
                let gy = ctx.output_grad();
                let gx = gy * $digamma_fn(x);
                ctx.append_input_grad(0, Some(gx));
            }
        }
    };
}

impl_gamma!(f32, digamma_f32);
impl_gamma!(f64, digamma_f64);
