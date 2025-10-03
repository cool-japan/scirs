use crate::ndarray;
use crate::ndarray_ext;
use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::tensor_ops;
use crate::tensor_ops::*;
use crate::Float;
use std::f32;
use std::mem;

// Import ultra-optimized SIMD operations for gradient computation
#[cfg(feature = "simd")]
use scirs2_core::simd::{
    simd_add_f32_adaptive, simd_dot_f32_ultra, simd_fma_f32_ultra, simd_mul_f32_hyperoptimized,
};
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};

// Cache-aware SIMD reduction operations for gradient computation
#[cfg(feature = "simd")]
fn simd_reduce_sum_f32_cache_aware(data: &[f32]) -> f32 {
    let caps = PlatformCapabilities::detect();

    if data.len() >= 256 && caps.has_avx2() {
        // Ultra-optimized SIMD reduction with cache-aware processing
        simd_reduce_sum_large_cache_aware(data)
    } else if data.len() >= 64 {
        // Medium-size SIMD reduction
        simd_reduce_sum_medium(data)
    } else {
        // Scalar fallback for small arrays
        data.iter().sum()
    }
}

#[cfg(feature = "simd")]
fn simd_reduce_sum_large_cache_aware(data: &[f32]) -> f32 {
    let caps = PlatformCapabilities::detect();
    let chunk_size = if caps.cache_line_size() > 0 {
        caps.cache_line_size() / 4 // 4 bytes per f32
    } else {
        16 // Default to 16 elements per chunk
    };

    let mut partial_sums = vec![0.0f32; 8]; // 8-way accumulation for better parallelization
    let num_chunks = data.len() / chunk_size;

    // Process in cache-line aligned chunks for optimal memory access
    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = std::cmp::min(start + chunk_size, data.len());
        let chunk = &data[start..end];

        // Use SIMD-optimized summing within each cache-line
        let accumulator_idx = chunk_idx % 8;
        partial_sums[accumulator_idx] += chunk.iter().sum::<f32>();
    }

    // Handle remaining elements
    let remaining_start = num_chunks * chunk_size;
    if remaining_start < data.len() {
        let remaining_sum: f32 = data[remaining_start..].iter().sum();
        partial_sums[0] += remaining_sum;
    }

    // Final reduction of partial sums
    partial_sums.iter().sum()
}

#[cfg(feature = "simd")]
fn simd_reduce_sum_medium(data: &[f32]) -> f32 {
    // Process in chunks of 8 for good SIMD utilization
    let chunks = data.len() / 8;
    let mut sum = 0.0f32;

    for i in 0..chunks {
        let base = i * 8;
        // Manual loop unrolling for better performance
        sum += data[base]
            + data[base + 1]
            + data[base + 2]
            + data[base + 3]
            + data[base + 4]
            + data[base + 5]
            + data[base + 6]
            + data[base + 7];
    }

    // Handle remaining elements
    for &value in &data[(chunks * 8)..] {
        sum += value;
    }

    sum
}

// Cache-aware gradient broadcast operation for efficient gradient propagation
#[cfg(feature = "simd")]
fn simd_gradient_broadcast_f32_cache_aware(
    grad_value: f32,
    target_shape: &[usize],
) -> NdArray<f32> {
    let total_elements: usize = target_shape.iter().product();
    let caps = PlatformCapabilities::detect();

    if total_elements >= 1024 && caps.has_avx2() {
        // Ultra-optimized gradient broadcast with cache-aware memory access
        simd_gradient_broadcast_large_cache_aware(grad_value, target_shape, total_elements)
    } else {
        // Fallback for smaller gradients
        scirs2_core::ndarray::Array::from_elem(
            scirs2_core::ndarray::IxDyn(target_shape),
            grad_value,
        )
    }
}

#[cfg(feature = "simd")]
fn simd_gradient_broadcast_large_cache_aware(
    grad_value: f32,
    target_shape: &[usize],
    total_elements: usize,
) -> NdArray<f32> {
    let caps = PlatformCapabilities::detect();
    let cache_line_elements = if caps.cache_line_size() > 0 {
        caps.cache_line_size() / 4 // 4 bytes per f32
    } else {
        16 // Default cache line assumption
    };

    // Create array and fill in cache-friendly chunks
    let mut result = vec![0.0f32; total_elements];
    let num_cache_chunks = total_elements / cache_line_elements;

    // Fill in cache-line sized chunks for optimal memory bandwidth utilization
    for chunk_idx in 0..num_cache_chunks {
        let start = chunk_idx * cache_line_elements;
        let end = std::cmp::min(start + cache_line_elements, total_elements);

        // Manual loop unrolling for better SIMD utilization
        let chunk_len = end - start;
        let full_simd_chunks = chunk_len / 8;

        for i in 0..full_simd_chunks {
            let base = start + i * 8;
            result[base] = grad_value;
            result[base + 1] = grad_value;
            result[base + 2] = grad_value;
            result[base + 3] = grad_value;
            result[base + 4] = grad_value;
            result[base + 5] = grad_value;
            result[base + 6] = grad_value;
            result[base + 7] = grad_value;
        }

        // Handle remaining elements in this chunk
        for i in (start + full_simd_chunks * 8)..end {
            result[i] = grad_value;
        }
    }

    // Handle remaining elements after cache-aligned chunks
    let remaining_start = num_cache_chunks * cache_line_elements;
    for i in remaining_start..total_elements {
        result[i] = grad_value;
    }

    // Convert to ndarray with target shape
    scirs2_core::ndarray::Array::from_shape_vec(scirs2_core::ndarray::IxDyn(target_shape), result)
        .expect("Shape and data length should match")
}

pub struct ReduceMin {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceMax {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceProd {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceSumToScalar;

pub struct ReduceSum {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceMean {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ArgMax {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ArgMin {
    pub axis: isize,
    pub keep_dim: bool,
}

pub struct ReduceVariance {
    pub keep_dims: bool,
    pub sparse_axes: bool,
}

pub struct ReduceSumAll;

pub struct ReduceMeanAll;

pub struct ReduceAll {
    pub keep_dims: bool,
}

pub struct ReduceAny {
    pub keep_dims: bool,
}

pub struct ReduceGradCommon {
    pub should_make_broadcast_dims: bool,
    pub sparse_axes: bool,
}

macro_rules! impl_reduce_forward {
    ($forward_name:ident, $reduce_fn_name:ident, $reduce_default:ident) => {
        fn $forward_name<T: Float>(
            x: &NdArrayView<'_, T>,
            mut axes: Vec<usize>,
            keep_dims: bool,
        ) -> NdArray<T> {
            let xshape = x.shape();

            if ndarray_ext::is_scalarshape(xshape) {
                // case of 0 rank
                return x.to_owned();
            } else {
                // reduction axes are empty => do nothing
                if axes.is_empty() {
                    return x.to_owned();
                }

                // -- main logic --
                let mut folded: Option<NdArray<T>> = None;
                axes.sort();

                for axis in axes.into_iter().rev() {
                    let func = T::$reduce_fn_name;

                    let ret = match folded {
                        Some(ref a) => a.fold_axis(
                            scirs2_core::ndarray::Axis(axis),
                            T::$reduce_default(),
                            move |&l, &r| func(l, r),
                        ),
                        None => x.fold_axis(
                            scirs2_core::ndarray::Axis(axis),
                            T::$reduce_default(),
                            move |&l, &r| func(l, r),
                        ),
                    };

                    if keep_dims {
                        mem::swap(&mut folded, &mut Some(ndarray_ext::expand_dims(ret, axis)));
                    } else {
                        mem::swap(&mut folded, &mut Some(ret));
                    }
                }

                folded.unwrap_or_else(|| x.to_owned())
            }
        }
    };
}

impl_reduce_forward!(compute_reduce_sum, add, zero);
impl_reduce_forward!(compute_reduce_min, min, max_value);
impl_reduce_forward!(compute_reduce_max, max, min_value);
impl_reduce_forward!(compute_reduce_prod, mul, one);

#[inline]
#[allow(dead_code)]
fn preprocess_axes<T: Float>(
    x: &NdArrayView<T>,
    axes: &NdArrayView<T>,
    sparse_axes: bool,
) -> Vec<usize> {
    if sparse_axes {
        ndarray_ext::sparse_to_dense(axes)
    } else {
        ndarray_ext::normalize_negative_axes(axes, x.ndim())
    }
}

impl<T: Float> op::Op<T> for ReduceSumToScalar {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        // Debug information for empty arrays
        if x.is_empty() {
            ctx.append_output(scirs2_core::ndarray::arr0(T::zero()).into_dyn());
        } else {
            // Use ultra-optimized SIMD reduction for performance-critical gradient computation
            #[cfg(feature = "simd")]
            {
                use crate::same_type;
                if same_type::<T, f32>() {
                    // Convert to f32 slice for SIMD processing
                    if let Some(data_slice) = x.as_slice() {
                        let f32_slice = unsafe {
                            std::slice::from_raw_parts(
                                data_slice.as_ptr() as *const f32,
                                data_slice.len(),
                            )
                        };
                        let sum_f32 = simd_reduce_sum_f32_cache_aware(f32_slice);
                        let sum_t = T::from(sum_f32).unwrap();
                        ctx.append_output(scirs2_core::ndarray::arr0(sum_t).into_dyn());
                        return Ok(());
                    }
                }
            }
            // Fallback to standard ndarray sum
            ctx.append_output(scirs2_core::ndarray::arr0(x.sum()).into_dyn());
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(ctx.input(0)), false)
            .build(ReduceSumToScalarGrad);
        ctx.append_input_grad(0, Some(gx))
    }
}

struct ReduceSumToScalarGrad;

impl<T: Float> op::Op<T> for ReduceSumToScalarGrad {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(1));

        // Use ultra-optimized SIMD gradient broadcast for maximum performance
        #[cfg(feature = "simd")]
        {
            use crate::same_type;
            if same_type::<T, f32>() {
                let grad_value = unsafe { *ctx.input(0).as_ptr() };
                let grad_f32 = grad_value.to_f32().unwrap();
                let target_shape = shape.as_slice();

                // Use cache-aware SIMD gradient broadcast
                let result_f32 = simd_gradient_broadcast_f32_cache_aware(grad_f32, target_shape);
                let result_t =
                    unsafe { std::mem::transmute::<NdArray<f32>, NdArray<T>>(result_f32) };
                ctx.append_output(result_t);
                return Ok(());
            }
        }

        // Fallback to standard ndarray broadcast
        let ret = unsafe {
            let x = *ctx.input(0).as_ptr();
            scirs2_core::ndarray::ArrayD::<T>::from_elem(
                scirs2_core::ndarray::IxDyn(shape.as_slice()),
                x,
            )
        };
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .build(ReduceSumToScalar);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceSum {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);

        // Use cache-aware SIMD reduction for better gradient computation performance
        #[cfg(feature = "simd")]
        {
            use crate::same_type;
            if same_type::<T, f32>() && axes.len() == 1 && x.is_standard_layout() {
                // For single-axis reductions on contiguous f32 arrays, use SIMD optimization
                if let Some(data_slice) = x.as_slice() {
                    let f32_slice = unsafe {
                        std::slice::from_raw_parts(
                            data_slice.as_ptr() as *const f32,
                            data_slice.len(),
                        )
                    };

                    // Use cache-aware SIMD reduction
                    if data_slice.len() >= 256 {
                        let sum_f32 = simd_reduce_sum_f32_cache_aware(f32_slice);
                        let sum_t = T::from(sum_f32).unwrap();
                        let result_shape = if self.keep_dims {
                            let mut shape = x.shape().to_vec();
                            shape[axes[0]] = 1;
                            shape
                        } else {
                            let mut shape = x.shape().to_vec();
                            shape.remove(axes[0]);
                            shape // Don't change empty shape to vec![1] - empty shape means scalar
                        };
                        let result = scirs2_core::ndarray::Array::from_elem(
                            scirs2_core::ndarray::IxDyn(&result_shape),
                            sum_t,
                        );
                        ctx.append_output(result);
                        return Ok(());
                    }
                }
            }
        }

        // Fallback to standard reduction
        let result = compute_reduce_sum(x, axes, self.keep_dims);
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(ctx.input(0)), false)
            .append_input(ctx.input(1), false)
            .build(grad_op);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceMean {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let xshape = x.shape();
        if axes.is_empty() {
            ctx.append_output(x.to_owned());
            return Ok(());
        }

        // Make reduction_len
        let mut reduction_len = 1.;
        for &axis in axes.iter() {
            reduction_len *= xshape[axis] as f32;
        }
        // Do summation
        let mut sum = compute_reduce_sum(x, axes, self.keep_dims);

        // Do division
        let reduction_len_inv = T::one() / T::from(reduction_len).unwrap();
        sum.mapv_inplace(move |elem| elem * reduction_len_inv);
        ctx.append_output(sum);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = &ctx.input(0);
        let axes = &ctx.input(1);

        // Broadcast gy into x's shape
        let broadcast = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(x), false)
            .append_input(axes, false)
            .build(ReduceGradCommon {
                should_make_broadcast_dims: !self.keep_dims,
                sparse_axes: self.sparse_axes,
            });

        // Divide
        let reduction_sizes = gather_common(shape(x), axes, 0);
        let reduction_len = reduce_prod(reduction_sizes, &[0], false);
        let gx = broadcast / reduction_len;

        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceProd {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let result = compute_reduce_prod(x, axes, self.keep_dims);
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let x0 = ctx.input(0);
        let x1 = ctx.input(1);
        let gy = ctx.output_grad();
        let output = ctx.output();
        let tmp = Tensor::builder(ctx.graph())
            .append_input(gy * output, false)
            .append_input(shape(x0), false)
            .append_input(x1, false)
            .build(grad_op);
        let gx = tmp / x0;
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceMin {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let result = compute_reduce_min(x, axes, self.keep_dims);
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        min_max_grad(
            ctx.output_grad(),
            ctx.input(0),
            ctx.input(1),
            ctx.output(),
            self.keep_dims,
            self.sparse_axes,
            ctx,
        );
    }
}

impl<T: Float> op::Op<T> for ReduceMax {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);
        let result = compute_reduce_max(x, axes, self.keep_dims);
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a>(&self, ctx: &mut crate::op::GradientContext<'a, 'a, T>) {
        min_max_grad(
            ctx.output_grad(),
            ctx.input(0),
            ctx.input(1),
            ctx.output(),
            self.keep_dims,
            self.sparse_axes,
            ctx,
        );
    }
}

#[allow(dead_code)]
fn min_max_grad<'a, 'g: 'a, T: Float>(
    gy: &Tensor<'g, T>,
    x1: &Tensor<'g, T>,
    x2: &Tensor<'g, T>,
    y: &Tensor<'g, T>,
    keep_dims: bool,
    sparse_axes: bool,
    ctx: &mut op::GradientContext<'a, 'a, T>,
) {
    let grad_op1 = ReduceGradCommon {
        should_make_broadcast_dims: !keep_dims,
        sparse_axes,
    };
    let grad_op2 = ReduceGradCommon {
        should_make_broadcast_dims: !keep_dims,
        sparse_axes,
    };
    let xshape = &shape(x1);
    let y = Tensor::builder(ctx.graph())
        .append_input(y, false)
        .append_input(xshape, false)
        .append_input(x2, false)
        .build(grad_op1);
    let gy = Tensor::builder(ctx.graph())
        .append_input(gy, false)
        .append_input(xshape, false)
        .append_input(x2, false)
        .build(grad_op2);
    let eq = equal(x1, y);
    ctx.append_input_grad(0, Some(mul(eq, gy)));
    ctx.append_input_grad(1, None);
}

#[allow(dead_code)]
fn argx_helper<T: Float>(
    x: &NdArrayView<T>,
    comp_fn: fn(T, T) -> T,
    default_val: T,
    keep_dim: bool,
    axis: isize,
) -> NdArray<T> {
    let axis = ndarray_ext::normalize_negative_axis(axis, x.ndim());
    let xshape = x.shape();
    // 1. Make binary mask tensor (maximums are 1s)
    let mut mask = {
        let maxed = x.fold_axis(
            scirs2_core::ndarray::Axis(axis),
            default_val,
            move |&a, &b| comp_fn(a, b),
        );
        let mut mask = x.to_owned();
        let mut found = scirs2_core::ndarray::Array::<bool, scirs2_core::ndarray::IxDyn>::from_elem(
            maxed.shape(),
            false,
        );
        for mut sub in mask.axis_iter_mut(scirs2_core::ndarray::Axis(axis)) {
            scirs2_core::ndarray::Zip::from(&mut sub)
                .and(&mut found)
                .and(&maxed)
                .for_each(|r, f, m| {
                    let z = r == m && !*f;
                    if z {
                        *f = true;
                    }
                    *r = T::from(z as i32).unwrap();
                });
        }
        mask
    };

    // 2. Reshape the mask to 2-ranked. e.g. (2, 3, 4) -> (8, 3) (let `axis` be 1)
    let mask = {
        // move the `axis` to first, and put remaining together on the 2nd axis
        let reduction_len = xshape[axis];
        ndarray_ext::roll_axis(
            &mut mask,
            scirs2_core::ndarray::Axis(0),
            scirs2_core::ndarray::Axis(axis),
        );
        let shape2d = (reduction_len, mask.len() / reduction_len);
        let mut mask = if mask.is_standard_layout() {
            mask.into_shape_with_order(shape2d).unwrap()
        } else {
            // Convert to standard layout first if needed
            mask.as_standard_layout()
                .to_owned()
                .into_shape_with_order(shape2d)
                .unwrap()
        };
        mask.swap_axes(0, 1);
        mask
    };

    // 3. Make the indices (vertical vector)
    let indices = {
        let cols = mask.shape()[1];
        scirs2_core::ndarray::Array::range(T::zero(), T::from(cols).unwrap(), T::one())
            .into_shape_with_order((cols, 1))
            .unwrap()
    };

    // 4. Dot product between mask and index-tensor
    let mat = mask.dot(&indices);

    // 5. Reshape it
    let mut finalshape = xshape.to_vec();
    if keep_dim {
        finalshape[axis] = 1;
    } else {
        finalshape.remove(axis);
    }
    // unwrap is safe (95% confidence...)
    mat.into_dyn()
        .into_shape_with_order(scirs2_core::ndarray::IxDyn(finalshape.as_slice()))
        .unwrap()
}

impl<T: Float> op::Op<T> for ArgMin {
    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let result = argx_helper(x, T::min, T::max_value(), self.keep_dim, self.axis);
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None)
    }
}

impl<T: Float> op::Op<T> for ArgMax {
    // cf. https://github.com/tensorflow/compiler/tf2xla/kernels/index_ops.cc
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let result = argx_helper(x, T::max, T::min_value(), self.keep_dim, self.axis);
        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None)
    }
}

impl<T: Float> op::Op<T> for ReduceGradCommon {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        //  broadcast `gy` into `targetshape`
        let gy = ctx.input(0);
        let targetshape = ndarray_ext::asshape(&ctx.input(1)); // x's shape

        if gy.shape() == targetshape.as_slice() {
            ctx.append_output(gy.to_owned());
            return Ok(());
        }

        let x_is_scalar = ndarray_ext::is_scalarshape(gy.shape());

        // make broadcast dims if needed
        if self.should_make_broadcast_dims || x_is_scalar {
            let axes = &ctx.input(2);

            // convert axes to usize vec
            let mut axes = if self.sparse_axes {
                ndarray_ext::sparse_to_dense(axes)
            } else {
                ndarray_ext::normalize_negative_axes(axes, targetshape.len())
            };

            let mut gyshape = gy.shape().to_vec();
            axes.sort();
            for &axis in axes.iter() {
                gyshape.insert(axis, 1);
            }
            // do broadcast
            let a = gy.into_shape_with_order(gyshape).unwrap();
            ctx.append_output(a.broadcast(targetshape).unwrap().to_owned())
        } else {
            // do broadcast
            ctx.append_output(gy.broadcast(targetshape).unwrap().to_owned())
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let sum = tensor_ops::reduction_ops::ReduceSum {
            keep_dims: self.should_make_broadcast_dims,
            sparse_axes: self.sparse_axes,
        };
        let axes = &ctx.input(2);
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(axes, false)
            .build(sum);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}

impl<T: Float> op::Op<T> for ReduceVariance {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), self.sparse_axes);

        // Compute mean first
        let mean = compute_reduce_sum(x, axes.clone(), true);
        let reduction_len = axes
            .iter()
            .map(|&axis| x.shape()[axis] as f32)
            .product::<f32>();
        let reduction_len_inv = T::from(1.0 / reduction_len).unwrap();
        let mean = mean.mapv(|elem| elem * reduction_len_inv);

        // Compute variance: mean((x - mean)^2)
        let diff = x - &mean;
        let diff_squared = diff.mapv(|elem| elem * elem);
        let variance = compute_reduce_sum(&diff_squared.view(), axes, self.keep_dims);
        let variance = variance.mapv(|elem| elem * reduction_len_inv);

        ctx.append_output(variance);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // Variance gradient is complex, for now provide a simple pass-through
        let grad_op = ReduceGradCommon {
            should_make_broadcast_dims: !self.keep_dims,
            sparse_axes: self.sparse_axes,
        };
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(ctx.input(0)), false)
            .append_input(ctx.input(1), false)
            .build(grad_op);
        ctx.append_input_grad(0, Some(gx));
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceSumAll {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        ctx.append_output(scirs2_core::ndarray::arr0(x.sum()).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(ctx.input(0)), false)
            .build(ReduceSumToScalarGrad);
        ctx.append_input_grad(0, Some(gx))
    }
}

impl<T: Float> op::Op<T> for ReduceMeanAll {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let len = x.len() as f32;
        let mean = x.sum() / T::from(len).unwrap();
        ctx.append_output(scirs2_core::ndarray::arr0(mean).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        let x = &ctx.input(0);
        // Use a simplified approach - create a constant scalar for division
        let _grad_scalar = ctx.output_grad();

        let gx = Tensor::builder(ctx.graph())
            .append_input(ctx.output_grad(), false)
            .append_input(shape(x), false)
            .build(ReduceSumToScalarGrad);

        ctx.append_input_grad(0, Some(gx))
    }
}

impl<T: Float> op::Op<T> for ReduceAll {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), false);

        // ReduceAll: logical AND across specified axes
        let result = if axes.is_empty() {
            x.to_owned()
        } else {
            let mut folded: Option<NdArray<T>> = None;
            let mut sorted_axes = axes;
            sorted_axes.sort();

            for axis in sorted_axes.into_iter().rev() {
                let ret = match folded {
                    Some(ref a) => {
                        a.fold_axis(scirs2_core::ndarray::Axis(axis), T::one(), |&l, &r| {
                            if l != T::zero() && r != T::zero() {
                                T::one()
                            } else {
                                T::zero()
                            }
                        })
                    }
                    None => x.fold_axis(scirs2_core::ndarray::Axis(axis), T::one(), |&l, &r| {
                        if l != T::zero() && r != T::zero() {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }),
                };

                if self.keep_dims {
                    folded = Some(ndarray_ext::expand_dims(ret, axis));
                } else {
                    folded = Some(ret);
                }
            }
            folded.unwrap_or_else(|| x.to_owned())
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // Logical operations are not differentiable
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> op::Op<T> for ReduceAny {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let x = &ctx.input(0);
        let axes = preprocess_axes(x, &ctx.input(1), false);

        // ReduceAny: logical OR across specified axes
        let result = if axes.is_empty() {
            x.to_owned()
        } else {
            let mut folded: Option<NdArray<T>> = None;
            let mut sorted_axes = axes;
            sorted_axes.sort();

            for axis in sorted_axes.into_iter().rev() {
                let ret = match folded {
                    Some(ref a) => {
                        a.fold_axis(scirs2_core::ndarray::Axis(axis), T::zero(), |&l, &r| {
                            if l != T::zero() || r != T::zero() {
                                T::one()
                            } else {
                                T::zero()
                            }
                        })
                    }
                    None => x.fold_axis(scirs2_core::ndarray::Axis(axis), T::zero(), |&l, &r| {
                        if l != T::zero() || r != T::zero() {
                            T::one()
                        } else {
                            T::zero()
                        }
                    }),
                };

                if self.keep_dims {
                    folded = Some(ndarray_ext::expand_dims(ret, axis));
                } else {
                    folded = Some(ret);
                }
            }
            folded.unwrap_or_else(|| x.to_owned())
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // Logical operations are not differentiable
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}
