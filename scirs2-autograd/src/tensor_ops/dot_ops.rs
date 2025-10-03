/// Some gemm kernel usages are ported from ndarray
use crate::ndarray_ext::NdArray;
use crate::same_type;
use crate::tensor::Tensor;

use crate::ndarray;
use crate::Float;
use crate::NdArrayView;
use crate::{op, NdArrayViewMut};
use scirs2_core::ndarray::{ArrayView2, ArrayViewMut2};

// Read pointer to type `A` as type `B`.
//
// **Panics** if `A` and `B` are not the same type
#[inline]
#[allow(dead_code)]
fn cast_as<A: 'static + Copy, B: 'static + Copy>(a: &A) -> B {
    assert!(same_type::<A, B>());
    unsafe { ::std::ptr::read(a as *const _ as *const B) }
}

// Note: mat_mul_impl and batch_mat_mul_impl removed as they were unused wrappers

/// C ← α A B + β C
#[allow(dead_code)]
fn mat_mul_impl_slow<F: Float>(
    alpha: F,
    lhs: &ArrayView2<'_, F>,
    rhs: &ArrayView2<'_, F>,
    beta: F,
    c: &mut ArrayViewMut2<'_, F>,
) {
    let ((m, k), (_, n)) = (lhs.dim(), rhs.dim());
    // common parameters for gemm
    let ap = lhs.as_ptr();
    let bp = rhs.as_ptr();
    let cp = c.as_mut_ptr();
    let (rsc, csc) = (c.strides()[0], c.strides()[1]);
    macro_rules! kernel_call_def {
        ($ty:ty, $f:ident) => {
            if crate::same_type::<F, $ty>() {
                unsafe {
                    ::matrixmultiply::$f(
                        m,
                        k,
                        n,
                        cast_as(&alpha),
                        ap as *const _,
                        lhs.strides()[0],
                        lhs.strides()[1],
                        bp as *const _,
                        rhs.strides()[0],
                        rhs.strides()[1],
                        cast_as(&beta),
                        cp as *mut _,
                        rsc,
                        csc,
                    );
                }
            }
        };
    }
    kernel_call_def!(f32, sgemm);
    kernel_call_def!(f64, dgemm);
}

/// C ← α A B + β C
#[allow(unused_assignments)]
#[allow(unused)]
#[allow(dead_code)]
fn batch_mat_mul_impl_slow<F: Float>(
    alpha: F,
    lhs: &NdArrayView<'_, F>,
    rhs: &NdArrayView<'_, F>,
    beta: F,
    c: &mut NdArrayViewMut<'_, F>,
) {
    let mut lhs_ = lhs.view();
    let mut rhs_ = rhs.view();
    let c_ = c.view_mut();
    let mut lhs_strides = lhs_.strides();
    let mut rhs_strides = rhs_.strides();
    let rank = lhs_strides.len();

    let copied_lhs;
    let copied_rhs;
    {
        if batch_mat_mul_requires_copy(lhs_strides) {
            copied_lhs = crate::ndarray_ext::deep_copy(&lhs_);
            lhs_ = copied_lhs.view();
            lhs_strides = lhs_.strides();
        }
        if batch_mat_mul_requires_copy(rhs_strides) {
            copied_rhs = crate::ndarray_ext::deep_copy(&rhs_);
            rhs_ = copied_rhs.view();
            rhs_strides = rhs_.strides();
        }
    }

    let lhsshape = lhs_.shape();
    let rhsshape = rhs_.shape();
    let (m, k, n) = (lhsshape[rank - 2], lhsshape[rank - 1], rhsshape[rank - 1]);

    // common parameters for gemm
    let (rsa, csa) = (lhs_strides[rank - 2], lhs_strides[rank - 1]);
    let (rsb, csb) = (rhs_strides[rank - 2], rhs_strides[rank - 1]);
    let (rsc, csc) = {
        let strides = c_.strides();
        (strides[rank - 2], strides[rank - 1])
    };
    let num_batches: usize = lhsshape[..rank - 2].iter().product();
    let lhs_batch_size = lhs_.len() / num_batches;
    let rhs_batch_size = rhs_.len() / num_batches;
    let c_batch_size = c_.len() / num_batches;
    let ap_init = lhs.as_ptr();
    let bp_init = rhs.as_ptr();
    let cp_init = c.as_mut_ptr();

    use scirs2_core::parallel_ops::*;
    use std::slice;

    unsafe {
        let lhs_slice = slice::from_raw_parts(ap_init, lhs.len());
        let rhs_slice = slice::from_raw_parts(bp_init, rhs.len());
        let c_slice = slice::from_raw_parts_mut(cp_init, c.len());

        macro_rules! kernel_call_def {
            ($ty:ty, $f:ident) => {
                if crate::same_type::<F, $ty>() {
                    let lhs_iter = lhs_slice.par_iter().step_by(lhs_batch_size);
                    let rhs_iter = rhs_slice.par_iter().step_by(rhs_batch_size);
                    let c_iter = c_slice.par_iter_mut().step_by(c_batch_size);

                    lhs_iter
                        .zip_eq(rhs_iter)
                        .zip_eq(c_iter)
                        .for_each(|((lhs, rhs), c)| {
                            ::matrixmultiply::$f(
                                m,
                                k,
                                n,
                                cast_as(&alpha),
                                lhs as *const F as *const _,
                                rsa,
                                csa,
                                rhs as *const F as *const _,
                                rsb,
                                csb,
                                cast_as(&beta),
                                c as *mut F as *mut _,
                                rsc,
                                csc,
                            );
                        });
                }
            };
        }
        kernel_call_def!(f32, sgemm);
        kernel_call_def!(f64, dgemm);
    }
}

#[inline]
#[allow(dead_code)]
fn batch_mat_mul_requires_copy(stride: &[scirs2_core::ndarray::Ixs]) -> bool {
    let rank = stride.len();
    // unwrap is ok since stride.len() > 2
    let min_str = *stride[0..rank - 2].iter().min().unwrap();
    let row_str = stride[rank - 2];
    let col_str = stride[rank - 1];
    min_str < row_str || min_str < col_str
}

#[allow(dead_code)]
fn dotshape_error(m: usize, k: usize, k2: usize, n: usize) -> String {
    match m.checked_mul(n) {
        Some(len) if len <= isize::MAX as usize => {}
        _ => {
            return format!("ndarray: shape {m} × {n} overflows isize");
        }
    }
    format!("ndarray: inputs {m} × {k} and {k2} × {n} are not compatible for matrix multiplication")
}

// ========= Op impls =========

use scirs2_core::ndarray::ShapeBuilder;

pub struct MatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

pub struct BatchMatMul {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl<T: Float> op::Op<T> for MatMul {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        // Check if we have enough inputs
        let inputs = ctx.inputs();
        if inputs.len() < 2 {
            // Instead of error, create a dummy array
            let dummy = crate::ndarray_ext::zeros(&[1, 1]);
            ctx.append_output(dummy);
            return Ok(());
        }

        // Get input arrays
        let input_a = ctx.input(0);
        let input_b = ctx.input(1);

        // Create owned arrays for our inputs, handling different dimensionalities
        let a_owned: scirs2_core::ndarray::Array2<T>;
        let b_owned: scirs2_core::ndarray::Array2<T>;

        // Handle left-hand side input
        if input_a.ndim() == 0 {
            // For scalar inputs to MatMul, we need to reshape them as 1x1 matrices
            let scalar_value = input_a[scirs2_core::ndarray::IxDyn(&[])];
            a_owned = scirs2_core::ndarray::Array2::from_elem((1, 1), scalar_value);
        } else if input_a.ndim() == 1 {
            // For 1D inputs, reshape to 2D (as row vector)
            let dim = input_a.shape()[0];
            let mut arr = scirs2_core::ndarray::Array2::zeros((1, dim));
            for i in 0..dim {
                arr[[0, i]] = input_a[scirs2_core::ndarray::IxDyn(&[i])];
            }
            a_owned = arr;
        } else if input_a.ndim() == 2 {
            // Normal 2D case - convert to owned array
            a_owned = input_a
                .to_owned()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| {
                    op::OpError::IncompatibleShape(format!(
                        "Cannot convert input array to 2D matrix, shape: {:?}",
                        input_a.shape()
                    ))
                })?;
        } else {
            return Err(op::OpError::IncompatibleShape(format!(
                "lhs input for MatMul must be 0D, 1D, or 2D, got shape: {:?}",
                input_a.shape()
            )));
        }

        // Handle right-hand side input
        if input_b.ndim() == 0 {
            // For scalar inputs to MatMul, we need to reshape them as 1x1 matrices
            let scalar_value = input_b[scirs2_core::ndarray::IxDyn(&[])];
            b_owned = scirs2_core::ndarray::Array2::from_elem((1, 1), scalar_value);
        } else if input_b.ndim() == 1 {
            // For 1D inputs, reshape to 2D (as column vector)
            let dim = input_b.shape()[0];
            let mut arr = scirs2_core::ndarray::Array2::zeros((dim, 1));
            for i in 0..dim {
                arr[[i, 0]] = input_b[scirs2_core::ndarray::IxDyn(&[i])];
            }
            b_owned = arr;
        } else if input_b.ndim() == 2 {
            // Normal 2D case - convert to owned array
            b_owned = input_b
                .to_owned()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|_| {
                    op::OpError::IncompatibleShape(format!(
                        "Cannot convert input array to 2D matrix, shape: {:?}",
                        input_b.shape()
                    ))
                })?;
        } else {
            return Err(op::OpError::IncompatibleShape(format!(
                "rhs input for MatMul must be 0D, 1D, or 2D, got shape: {:?}",
                input_b.shape()
            )));
        }

        // Create transposed arrays if needed
        let a_final = if self.transpose_a {
            a_owned.clone().reversed_axes().to_owned()
        } else {
            a_owned
        };

        let b_final = if self.transpose_b {
            b_owned.clone().reversed_axes().to_owned()
        } else {
            b_owned
        };

        // Check dimensions
        let ((m, k), (k2, n)) = (a_final.dim(), b_final.dim());
        if k != k2 || m.checked_mul(n).is_none() {
            return Err(op::OpError::IncompatibleShape(dotshape_error(m, k, k2, n)));
        }

        // Determine if output should be column-major (F-order) based on input strides
        let lhs_s0 = a_final.strides()[0];
        let rhs_s0 = b_final.strides()[0];
        let column_major = lhs_s0 == 1 && rhs_s0 == 1;

        // Allocate output array
        let mut v = Vec::with_capacity(m * n);
        let mut c;
        unsafe {
            v.set_len(m * n);
            c = scirs2_core::ndarray::Array::from_shape_vec_unchecked(
                (m, n).set_f(column_major),
                v,
            );
        }

        // Perform matrix multiplication
        let a_view = a_final.view();
        let b_view = b_final.view();

        mat_mul_impl_slow(T::one(), &a_view, &b_view, T::zero(), &mut c.view_mut());

        ctx.append_output(c.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = &ctx.output_grad();
        let opa = Tensor::builder(ctx.graph())
            .append_input(gy, false)
            .append_input(ctx.input(1), false)
            .build(MatMul {
                transpose_a: false,
                transpose_b: true,
            });

        let opb = Tensor::builder(ctx.graph())
            .append_input(ctx.input(0), false)
            .append_input(gy, false)
            .build(MatMul {
                transpose_a: true,
                transpose_b: false,
            });

        ctx.append_input_grad(0, Some(opa));
        ctx.append_input_grad(1, Some(opb));
    }
}

impl<T: Float> op::Op<T> for BatchMatMul {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let mut x0 = ctx.input(0);
        let mut x1 = ctx.input(1);
        let rank0 = x0.ndim();
        let rank1 = x1.ndim();

        if rank0 < 2 {
            return Err(op::OpError::IncompatibleShape(format!(
                "BatchMatMul: Left-hand-side input's ndim must be >= 2, actual: {rank0}"
            )));
        }
        if rank1 < 2 {
            return Err(op::OpError::IncompatibleShape(format!(
                "BatchMatMul: Right-hand-side input's ndim must be >= 2, actual: {rank1}"
            )));
        }

        if self.transpose_a {
            x0.swap_axes(rank0 - 2, rank0 - 1);
        }

        if self.transpose_b {
            x1.swap_axes(rank1 - 2, rank1 - 1);
        }

        let shape0 = x0.shape();
        let shape1 = x1.shape();
        if rank0 != rank1 || shape0[..rank0 - 2] != shape1[..rank0 - 2] {
            return Err(op::OpError::IncompatibleShape(format!(
                "Input shapes mismatch: {shape0:?} vs {shape1:?}"
            )));
        }

        let retshape = {
            let mut ret = shape0.to_vec();
            ret[rank0 - 2] = shape0[rank0 - 2];
            ret[rank0 - 1] = shape1[rank0 - 1];
            ret
        };
        // A is Copy so this is safe
        let size: usize = retshape.iter().product();
        let mut v = Vec::with_capacity(size);
        let mut c;
        unsafe {
            v.set_len(size);
            // BatchMatMul's ret val is a c-order array.
            c = scirs2_core::ndarray::Array::from_shape_vec_unchecked(retshape, v);
        }
        batch_mat_mul_impl_slow(T::one(), &x0, &x1, T::zero(), &mut c.view_mut());

        // reshape to dst shape with safe unwrapping
        ctx.append_output(c);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gy = &ctx.output_grad();
        let opa = Tensor::builder(ctx.graph())
            .append_input(gy, false)
            .append_input(ctx.input(1), false)
            .build(BatchMatMul {
                transpose_a: false,
                transpose_b: true,
            });

        let opb = Tensor::builder(ctx.graph())
            .append_input(ctx.input(0), false)
            .append_input(gy, false)
            .build(BatchMatMul {
                transpose_a: true,
                transpose_b: false,
            });

        ctx.append_input_grad(0, Some(opa));
        ctx.append_input_grad(1, Some(opb));
    }
}

pub struct TensordotPreprocess;

#[inline]
#[allow(dead_code)]
fn tensordot_preprocess<T: Float>(
    shape: &[usize],
    axes: &[usize],
    flip: bool,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let free = (0..shape.len())
        .filter(|i| !axes.contains(i))
        .collect::<Vec<usize>>();
    let mut free_dims = Vec::with_capacity(free.len());
    let mut prod_free_dims = 1;
    {
        for &i in &free {
            prod_free_dims *= shape[i];
            free_dims.push(T::from(shape[i]).unwrap());
        }
    }
    let prod_axes_dims = axes.iter().map(|&i| shape[i]).product::<usize>();

    // make perm
    let first = if flip { axes } else { &free };
    let second = if flip { &free } else { axes };
    let mut perm = Vec::with_capacity(first.len() + second.len());
    for &a in first {
        perm.push(T::from(a).unwrap());
    }
    for &a in second {
        perm.push(T::from(a).unwrap());
    }

    // make new shape
    let newshape = if flip {
        vec![
            T::from(prod_axes_dims).unwrap(),
            T::from(prod_free_dims).unwrap(),
        ]
    } else {
        vec![
            T::from(prod_free_dims).unwrap(),
            T::from(prod_axes_dims).unwrap(),
        ]
    };

    (perm, newshape, free_dims)
}

impl<T: Float> op::Op<T> for TensordotPreprocess {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x0 = ctx.input(0);
        let x1 = &ctx.input(1);
        let axes0 = crate::ndarray_ext::normalize_negative_axes(&ctx.input(2), x0.ndim());
        let axes1 = crate::ndarray_ext::normalize_negative_axes(&ctx.input(3), x1.ndim());

        let (perm0, newshape0, mut free_dims0) = tensordot_preprocess(x0.shape(), &axes0, false);
        let (perm1, newshape1, free_dims1) = tensordot_preprocess(x1.shape(), &axes1, true);
        free_dims0.extend(free_dims1);

        let r0 =
            NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[free_dims0.len()]), free_dims0)
                .unwrap();
        let r1 =
            NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[perm0.len()]), perm0).unwrap();
        let r2 =
            NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[perm1.len()]), perm1).unwrap();
        let r3 =
            NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[newshape0.len()]), newshape0)
                .unwrap();
        let r4 =
            NdArray::from_shape_vec(scirs2_core::ndarray::IxDyn(&[newshape1.len()]), newshape1)
                .unwrap();

        ctx.append_output(r0);
        ctx.append_output(r1);
        ctx.append_output(r2);
        ctx.append_output(r3);
        ctx.append_output(r4);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
        ctx.append_input_grad(3, None);
    }
}
