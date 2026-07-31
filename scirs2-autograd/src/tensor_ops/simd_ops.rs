//! SIMD-accelerated tensor operations for autograd
//!
//! This module provides high-performance SIMD implementations of common tensor
//! operations used in neural network training and inference. It leverages the
//! `scirs2_core::simd` infrastructure for portable SIMD across x86_64 (AVX2/SSE)
//! and aarch64 (NEON) architectures.
//!
//! ## Operations
//!
//! ### Element-wise arithmetic (forward + backward)
//! - [`simd_elementwise_add`] / [`simd_elementwise_sub`]
//! - [`simd_elementwise_mul`] / [`simd_elementwise_div`]
//!
//! ### Gradient accumulation (critical for backprop)
//! - [`simd_gradient_accumulate`]
//! - [`simd_scaled_gradient_accumulate`]
//!
//! ### Broadcasting operations
//! - [`simd_broadcast_add`] / [`simd_broadcast_mul`]
//!
//! ### Activation functions (forward + backward)
//! - [`simd_activation_relu`] / `simd_activation_relu_backward`
//! - [`simd_activation_sigmoid`] / `simd_activation_sigmoid_backward`
//! - [`simd_activation_tanh`] / `simd_activation_tanh_backward`
//!
//! ### Dot product / reduction
//! - [`simd_dot_product`] / [`simd_reduction_sum`]
//!
//! ## Feature gating
//!
//! All operations in this module are gated behind the `simd` feature flag.
//! When the feature is not enabled, scalar fallback implementations are used.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;

// ============================================================================
// Internal SIMD dispatch helpers
// ============================================================================

/// Apply a SIMD-accelerated element-wise binary operation on flat f32 slices.
/// Falls back to scalar when the `simd` feature is disabled or slices are non-contiguous.
#[cfg(feature = "simd")]
fn dispatch_binary_f32(a: &[f32], b: &[f32], op: SimdBinaryKind) -> Vec<f32> {
    use scirs2_core::ndarray::{Array1, ArrayView1};

    let a_arr = ArrayView1::from(a);
    let b_arr = ArrayView1::from(b);

    let result: Array1<f32> = match op {
        SimdBinaryKind::Add => scirs2_core::simd::simd_add_f32(&a_arr, &b_arr),
        SimdBinaryKind::Sub => scirs2_core::simd::simd_sub_f32(&a_arr, &b_arr),
        SimdBinaryKind::Mul => scirs2_core::simd::simd_mul_f32(&a_arr, &b_arr),
        SimdBinaryKind::Div => scirs2_core::simd::simd_div_f32(&a_arr, &b_arr),
    };
    result.to_vec()
}

/// Apply a SIMD-accelerated element-wise binary operation on flat f64 slices.
#[cfg(feature = "simd")]
fn dispatch_binary_f64(a: &[f64], b: &[f64], op: SimdBinaryKind) -> Vec<f64> {
    use scirs2_core::ndarray::{Array1, ArrayView1};

    let a_arr = ArrayView1::from(a);
    let b_arr = ArrayView1::from(b);

    let result: Array1<f64> = match op {
        SimdBinaryKind::Add => scirs2_core::simd::simd_add_f64(&a_arr, &b_arr),
        SimdBinaryKind::Sub => scirs2_core::simd::simd_sub_f64(&a_arr, &b_arr),
        SimdBinaryKind::Mul => scirs2_core::simd::simd_mul_f64(&a_arr, &b_arr),
        SimdBinaryKind::Div => scirs2_core::simd::simd_div_f64(&a_arr, &b_arr),
    };
    result.to_vec()
}

#[cfg(feature = "simd")]
#[derive(Debug, Clone, Copy)]
enum SimdBinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

// ============================================================================
// Element-wise Arithmetic Ops (Op trait implementations)
// ============================================================================

/// SIMD-accelerated element-wise addition operator.
///
/// Forward:  `y = a + b`
/// Backward: `da = dy`, `db = dy`
pub struct SimdElementwiseAdd;

impl<F: Float> Op<F> for SimdElementwiseAdd {
    fn name(&self) -> &'static str {
        "SimdElementwiseAdd"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        #[cfg(feature = "simd")]
        {
            if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
                if a_slice.len() == b_slice.len() {
                    if let Some(result) =
                        try_simd_binary_op::<F>(a_slice, b_slice, SimdBinaryKind::Add)
                    {
                        let shape = a.shape().to_vec();
                        let arr = scirs2_core::ndarray::Array::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(&shape),
                            result,
                        )
                        .map_err(|e| OpError::NdArrayError("SimdElementwiseAdd shape".into(), e))?;
                        ctx.append_output(arr);
                        return Ok(());
                    }
                }
            }
        }

        // Scalar fallback
        let result = &a.to_owned() + &b.to_owned();
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        ctx.append_input_grad(0, Some(*gy));
        ctx.append_input_grad(1, Some(*gy));
    }
}

/// SIMD-accelerated element-wise subtraction operator.
///
/// Forward:  `y = a - b`
/// Backward: `da = dy`, `db = -dy`
pub struct SimdElementwiseSub;

impl<F: Float> Op<F> for SimdElementwiseSub {
    fn name(&self) -> &'static str {
        "SimdElementwiseSub"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        #[cfg(feature = "simd")]
        {
            if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
                if a_slice.len() == b_slice.len() {
                    if let Some(result) =
                        try_simd_binary_op::<F>(a_slice, b_slice, SimdBinaryKind::Sub)
                    {
                        let shape = a.shape().to_vec();
                        let arr = scirs2_core::ndarray::Array::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(&shape),
                            result,
                        )
                        .map_err(|e| OpError::NdArrayError("SimdElementwiseSub shape".into(), e))?;
                        ctx.append_output(arr);
                        return Ok(());
                    }
                }
            }
        }

        let result = &a.to_owned() - &b.to_owned();
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        ctx.append_input_grad(0, Some(*gy));
        ctx.append_input_grad(1, Some(crate::tensor_ops::neg(*gy)));
    }
}

/// SIMD-accelerated element-wise multiplication operator.
///
/// Forward:  `y = a * b`
/// Backward: `da = dy * b`, `db = dy * a`
pub struct SimdElementwiseMul;

impl<F: Float> Op<F> for SimdElementwiseMul {
    fn name(&self) -> &'static str {
        "SimdElementwiseMul"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        #[cfg(feature = "simd")]
        {
            if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
                if a_slice.len() == b_slice.len() {
                    if let Some(result) =
                        try_simd_binary_op::<F>(a_slice, b_slice, SimdBinaryKind::Mul)
                    {
                        let shape = a.shape().to_vec();
                        let arr = scirs2_core::ndarray::Array::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(&shape),
                            result,
                        )
                        .map_err(|e| OpError::NdArrayError("SimdElementwiseMul shape".into(), e))?;
                        ctx.append_output(arr);
                        return Ok(());
                    }
                }
            }
        }

        let result = &a.to_owned() * &b.to_owned();
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        let a = ctx.input(0);
        let b = ctx.input(1);
        // da = dy * b
        ctx.append_input_grad(0, Some(*gy * b));
        // db = dy * a
        ctx.append_input_grad(1, Some(*gy * a));
    }
}

/// SIMD-accelerated element-wise division operator.
///
/// Forward:  `y = a / b`
/// Backward: `da = dy / b`, `db = -dy * a / b^2`
pub struct SimdElementwiseDiv;

impl<F: Float> Op<F> for SimdElementwiseDiv {
    fn name(&self) -> &'static str {
        "SimdElementwiseDiv"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        #[cfg(feature = "simd")]
        {
            if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
                if a_slice.len() == b_slice.len() {
                    if let Some(result) =
                        try_simd_binary_op::<F>(a_slice, b_slice, SimdBinaryKind::Div)
                    {
                        let shape = a.shape().to_vec();
                        let arr = scirs2_core::ndarray::Array::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(&shape),
                            result,
                        )
                        .map_err(|e| OpError::NdArrayError("SimdElementwiseDiv shape".into(), e))?;
                        ctx.append_output(arr);
                        return Ok(());
                    }
                }
            }
        }

        let result = &a.to_owned() / &b.to_owned();
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        let a = ctx.input(0);
        let b = ctx.input(1);
        let g = ctx.graph();

        // da = dy / b
        ctx.append_input_grad(0, Some(*gy / b));

        // db = -dy * a / b^2
        let neg_one = crate::tensor_ops::scalar(-F::one(), g);
        let b_sq = b * b;
        ctx.append_input_grad(1, Some(neg_one * *gy * a / b_sq));
    }
}

// ============================================================================
// Gradient Accumulation Ops
// ============================================================================

/// SIMD-accelerated gradient accumulation operator.
///
/// Accumulates gradient `g` into existing gradient buffer `acc`:
/// Forward: `y = acc + g`
///
/// This is the critical inner-loop operation during backpropagation.
/// Using SIMD here gives substantial training speedups.
pub struct SimdGradientAccumulate;

impl<F: Float> Op<F> for SimdGradientAccumulate {
    fn name(&self) -> &'static str {
        "SimdGradientAccumulate"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let acc = ctx.input(0);
        let grad = ctx.input(1);

        #[cfg(feature = "simd")]
        {
            if let (Some(acc_slice), Some(grad_slice)) = (acc.as_slice(), grad.as_slice()) {
                if acc_slice.len() == grad_slice.len() {
                    if let Some(result) =
                        try_simd_binary_op::<F>(acc_slice, grad_slice, SimdBinaryKind::Add)
                    {
                        let shape = acc.shape().to_vec();
                        let arr = scirs2_core::ndarray::Array::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(&shape),
                            result,
                        )
                        .map_err(|e| {
                            OpError::NdArrayError("SimdGradientAccumulate shape".into(), e)
                        })?;
                        ctx.append_output(arr);
                        return Ok(());
                    }
                }
            }
        }

        let result = &acc.to_owned() + &grad.to_owned();
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        // Both inputs receive the upstream gradient as-is
        ctx.append_input_grad(0, Some(*gy));
        ctx.append_input_grad(1, Some(*gy));
    }
}

/// SIMD-accelerated scaled gradient accumulation: `acc + scale * grad`
///
/// Combines scaling and accumulation into a single fused operation,
/// which is common in optimizers (e.g., momentum SGD: `v = mu * v + lr * grad`).
pub struct SimdScaledGradientAccumulate<F: Float> {
    /// The scale factor applied to the gradient before accumulation
    pub scale: F,
}

impl<F: Float> Op<F> for SimdScaledGradientAccumulate<F> {
    fn name(&self) -> &'static str {
        "SimdScaledGradientAccumulate"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let acc = ctx.input(0);
        let grad = ctx.input(1);

        // scale * grad + acc  (FMA pattern)
        #[cfg(feature = "simd")]
        {
            if let (Some(acc_slice), Some(grad_slice)) = (acc.as_slice(), grad.as_slice()) {
                if acc_slice.len() == grad_slice.len() {
                    if let Some(result) = try_simd_fma::<F>(grad_slice, self.scale, acc_slice) {
                        let shape = acc.shape().to_vec();
                        let arr = scirs2_core::ndarray::Array::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(&shape),
                            result,
                        )
                        .map_err(|e| {
                            OpError::NdArrayError("SimdScaledGradAccum shape".into(), e)
                        })?;
                        ctx.append_output(arr);
                        return Ok(());
                    }
                }
            }
        }

        // Scalar fallback: acc + scale * grad
        let scaled = grad.mapv(|v| v * self.scale);
        let result = &acc.to_owned() + &scaled;
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        let g = ctx.graph();
        // d(acc + scale * grad)/d(acc) = 1
        ctx.append_input_grad(0, Some(*gy));
        // d(acc + scale * grad)/d(grad) = scale
        let scale_tensor = crate::tensor_ops::scalar(self.scale, g);
        ctx.append_input_grad(1, Some(*gy * scale_tensor));
    }
}

// ============================================================================
// Broadcasting Operations
// ============================================================================

/// SIMD-accelerated broadcast addition.
///
/// Adds a bias vector (1D) to each row of a 2D tensor:
/// `y[i, :] = x[i, :] + bias[:]`
///
/// This is the standard pattern for bias addition in dense layers.
pub struct SimdBroadcastAdd;

impl<F: Float> Op<F> for SimdBroadcastAdd {
    fn name(&self) -> &'static str {
        "SimdBroadcastAdd"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);
        let bias = ctx.input(1);

        let x_shape = x.shape().to_vec();
        let bias_shape = bias.shape().to_vec();

        // Handle the common case: x is [batch, features], bias is [features]
        if x_shape.len() == 2 && bias_shape.len() == 1 && x_shape[1] == bias_shape[0] {
            let rows = x_shape[0];
            let cols = x_shape[1];

            #[cfg(feature = "simd")]
            {
                if let (Some(x_slice), Some(bias_slice)) = (x.as_slice(), bias.as_slice()) {
                    if let Some(mut result_vec) =
                        try_simd_broadcast_add_2d::<F>(x_slice, bias_slice, rows, cols)
                    {
                        let arr = scirs2_core::ndarray::Array::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(&x_shape),
                            result_vec,
                        )
                        .map_err(|e| OpError::NdArrayError("SimdBroadcastAdd shape".into(), e))?;
                        ctx.append_output(arr);
                        return Ok(());
                    }
                }
            }
        }

        // General fallback using ndarray broadcasting
        let result = &x.to_owned() + &bias.to_owned();
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        // dx = dy (same shape)
        ctx.append_input_grad(0, Some(*gy));
        // dbias = sum over batch dimension
        let dbias = crate::tensor_ops::reduce_sum(*gy, &[0], false);
        ctx.append_input_grad(1, Some(dbias));
    }
}

/// SIMD-accelerated broadcast multiplication.
///
/// Multiplies each row of a 2D tensor by a 1D scale vector:
/// `y[i, :] = x[i, :] * scale[:]`
pub struct SimdBroadcastMul;

impl<F: Float> Op<F> for SimdBroadcastMul {
    fn name(&self) -> &'static str {
        "SimdBroadcastMul"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);
        let scale = ctx.input(1);

        let x_shape = x.shape().to_vec();
        let scale_shape = scale.shape().to_vec();

        if x_shape.len() == 2 && scale_shape.len() == 1 && x_shape[1] == scale_shape[0] {
            let rows = x_shape[0];
            let cols = x_shape[1];

            #[cfg(feature = "simd")]
            {
                if let (Some(x_slice), Some(scale_slice)) = (x.as_slice(), scale.as_slice()) {
                    if let Some(result_vec) =
                        try_simd_broadcast_mul_2d::<F>(x_slice, scale_slice, rows, cols)
                    {
                        let arr = scirs2_core::ndarray::Array::from_shape_vec(
                            scirs2_core::ndarray::IxDyn(&x_shape),
                            result_vec,
                        )
                        .map_err(|e| OpError::NdArrayError("SimdBroadcastMul shape".into(), e))?;
                        ctx.append_output(arr);
                        return Ok(());
                    }
                }
            }
        }

        let result = &x.to_owned() * &scale.to_owned();
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let scale = ctx.input(1);

        // dx = dy * scale (broadcast)
        ctx.append_input_grad(0, Some(*gy * scale));
        // dscale = sum_over_batch(dy * x)
        let dscale = crate::tensor_ops::reduce_sum(*gy * x, &[0], false);
        ctx.append_input_grad(1, Some(dscale));
    }
}

// ============================================================================
// Activation Functions (Forward + Backward)
// ============================================================================

/// SIMD-accelerated ReLU activation operator.
///
/// Forward:  `y = max(0, x)`
/// Backward: `dx = dy * (x > 0)`
pub struct SimdReLU;

impl<F: Float> Op<F> for SimdReLU {
    fn name(&self) -> &'static str {
        "SimdReLU"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);

        #[cfg(feature = "simd")]
        {
            if let Some(x_slice) = x.as_slice() {
                if let Some(result) = try_simd_relu::<F>(x_slice) {
                    let shape = x.shape().to_vec();
                    let arr = scirs2_core::ndarray::Array::from_shape_vec(
                        scirs2_core::ndarray::IxDyn(&shape),
                        result,
                    )
                    .map_err(|e| OpError::NdArrayError("SimdReLU shape".into(), e))?;
                    ctx.append_output(arr);
                    return Ok(());
                }
            }
        }

        // Scalar fallback
        let result = x.mapv(|v| if v > F::zero() { v } else { F::zero() });
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let g = ctx.graph();

        // ReLU backward: dy * (x > 0)
        let zero = crate::tensor_ops::scalar(F::zero(), g);
        let mask = crate::tensor_ops::greater(x, zero);
        ctx.append_input_grad(0, Some(*gy * mask));
    }
}

/// SIMD-accelerated Sigmoid activation operator.
///
/// Forward:  `y = 1 / (1 + exp(-x))`
/// Backward: `dx = dy * y * (1 - y)`
pub struct SimdSigmoid;

impl<F: Float> Op<F> for SimdSigmoid {
    fn name(&self) -> &'static str {
        "SimdSigmoid"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);

        #[cfg(feature = "simd")]
        {
            if let Some(x_slice) = x.as_slice() {
                if let Some(result) = try_simd_sigmoid::<F>(x_slice) {
                    let shape = x.shape().to_vec();
                    let arr = scirs2_core::ndarray::Array::from_shape_vec(
                        scirs2_core::ndarray::IxDyn(&shape),
                        result,
                    )
                    .map_err(|e| OpError::NdArrayError("SimdSigmoid shape".into(), e))?;
                    ctx.append_output(arr);
                    return Ok(());
                }
            }
        }

        // Scalar fallback: sigmoid(x) = 0.5 * (tanh(0.5*x) + 1)
        let half = F::from(0.5).ok_or_else(|| OpError::ConversionError {
            context: "SimdSigmoid half constant".into(),
            from_type: "f64".into(),
            to_type: std::any::type_name::<F>().into(),
        })?;
        let result = x.mapv(move |v| ((v * half).tanh() * half) + half);
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        let y = ctx.output();
        let g = ctx.graph();

        // sigmoid backward: dy * y * (1 - y)
        let one = crate::tensor_ops::scalar(F::one(), g);
        let one_minus_y = one - y;
        ctx.append_input_grad(0, Some(*gy * y * one_minus_y));
    }
}

/// SIMD-accelerated Tanh activation operator.
///
/// Forward:  `y = tanh(x)`
/// Backward: `dx = dy * (1 - y^2)`
pub struct SimdTanh;

impl<F: Float> Op<F> for SimdTanh {
    fn name(&self) -> &'static str {
        "SimdTanh"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);

        #[cfg(feature = "simd")]
        {
            if let Some(x_slice) = x.as_slice() {
                if let Some(result) = try_simd_tanh::<F>(x_slice) {
                    let shape = x.shape().to_vec();
                    let arr = scirs2_core::ndarray::Array::from_shape_vec(
                        scirs2_core::ndarray::IxDyn(&shape),
                        result,
                    )
                    .map_err(|e| OpError::NdArrayError("SimdTanh shape".into(), e))?;
                    ctx.append_output(arr);
                    return Ok(());
                }
            }
        }

        let result = x.mapv(|v| v.tanh());
        ctx.append_output(result);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        let y = ctx.output();
        let g = ctx.graph();

        // tanh backward: dy * (1 - y^2)
        let one = crate::tensor_ops::scalar(F::one(), g);
        let y_sq = y * y;
        ctx.append_input_grad(0, Some(*gy * (one - y_sq)));
    }
}

// ============================================================================
// Dot Product / Reduction Operations
// ============================================================================

/// SIMD-accelerated dot product operator.
///
/// Computes the inner product of two 1-D tensors.
/// Forward:  `y = sum(a * b)`
/// Backward: `da = dy * b`, `db = dy * a`
pub struct SimdDotProduct;

impl<F: Float> Op<F> for SimdDotProduct {
    fn name(&self) -> &'static str {
        "SimdDotProduct"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a = ctx.input(0);
        let b = ctx.input(1);

        if a.ndim() != 1 || b.ndim() != 1 {
            return Err(OpError::IncompatibleShape(
                "SimdDotProduct requires 1-D inputs".into(),
            ));
        }

        if a.len() != b.len() {
            return Err(OpError::IncompatibleShape(format!(
                "SimdDotProduct: length mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        #[cfg(feature = "simd")]
        {
            if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
                if let Some(dot_val) = try_simd_dot::<F>(a_slice, b_slice) {
                    let arr = scirs2_core::ndarray::arr0(dot_val).into_dyn();
                    ctx.append_output(arr);
                    return Ok(());
                }
            }
        }

        // Scalar fallback
        let mut sum = F::zero();
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            sum += ai * bi;
        }
        let arr = scirs2_core::ndarray::arr0(sum).into_dyn();
        ctx.append_output(arr);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        let a = ctx.input(0);
        let b = ctx.input(1);

        // da = dy * b, db = dy * a
        ctx.append_input_grad(0, Some(*gy * b));
        ctx.append_input_grad(1, Some(*gy * a));
    }
}

/// SIMD-accelerated sum reduction operator.
///
/// Computes the sum of all elements in a 1-D tensor.
/// Forward:  `y = sum(x)`
/// Backward: `dx = ones_like(x) * dy`
pub struct SimdReductionSum;

impl<F: Float> Op<F> for SimdReductionSum {
    fn name(&self) -> &'static str {
        "SimdReductionSum"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);

        #[cfg(feature = "simd")]
        {
            if let Some(x_slice) = x.as_slice() {
                if let Some(sum_val) = try_simd_sum::<F>(x_slice) {
                    let arr = scirs2_core::ndarray::arr0(sum_val).into_dyn();
                    ctx.append_output(arr);
                    return Ok(());
                }
            }
        }

        // Scalar fallback
        let sum = x.iter().fold(F::zero(), |acc, &v| acc + v);
        let arr = scirs2_core::ndarray::arr0(sum).into_dyn();
        ctx.append_output(arr);
        Ok(())
    }

    fn grad<'a, 'g>(&self, ctx: &mut GradientContext<'a, 'g, F>) {
        let gy = ctx.output_grad();
        let x = ctx.input(0);
        let g = ctx.graph();

        // dx = broadcast(dy) to the shape of x  => ones_like(x) * dy
        let ones_shape = crate::tensor_ops::shape(x);
        let ones_val = crate::tensor_ops::ones(&ones_shape, g);
        ctx.append_input_grad(0, Some(ones_val * *gy));
    }
}

// ============================================================================
// Type-dispatched SIMD helpers (compile-time type routing)
// ============================================================================

/// Attempt to dispatch a binary SIMD operation for the concrete float type.
/// Returns `None` if the type is not f32/f64 (meaning fallback is needed).
#[cfg(feature = "simd")]
fn try_simd_binary_op<F: Float>(a: &[F], b: &[F], kind: SimdBinaryKind) -> Option<Vec<F>> {
    use crate::same_type;

    if same_type::<F, f32>() {
        // SAFETY: we checked the type is f32
        let a_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
        let b_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
        let result = dispatch_binary_f32(a_f32, b_f32, kind);
        // SAFETY: F == f32
        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else if same_type::<F, f64>() {
        let a_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let b_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f64, b.len()) };
        let result = dispatch_binary_f64(a_f64, b_f64, kind);
        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else {
        None
    }
}

/// Attempt to dispatch SIMD FMA: `a * scale + c`
#[cfg(feature = "simd")]
fn try_simd_fma<F: Float>(a: &[F], scale: F, c: &[F]) -> Option<Vec<F>> {
    use crate::same_type;

    if same_type::<F, f32>() {
        let a_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
        let c_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, c.len()) };
        let scale_f32: f32 = unsafe { *(&scale as *const F as *const f32) };

        // Create scale array for FMA: result = a * scale_arr + c
        let scale_arr = scirs2_core::ndarray::Array1::from_elem(a.len(), scale_f32);
        let a_view = scirs2_core::ndarray::ArrayView1::from(a_f32);
        let scale_view = scale_arr.view();
        let c_view = scirs2_core::ndarray::ArrayView1::from(c_f32);

        let result = scirs2_core::simd::simd_fma_f32_ultra(&a_view, &scale_view, &c_view);
        let result_vec = result.to_vec();

        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result_vec);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else if same_type::<F, f64>() {
        // No f64 FMA in core, fall back to manual scale + add
        let a_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let c_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f64, c.len()) };
        let scale_f64: f64 = unsafe { *(&scale as *const F as *const f64) };

        let a_view = scirs2_core::ndarray::ArrayView1::from(a_f64);
        let scale_arr = scirs2_core::simd::simd_scalar_mul_f64(&a_view, scale_f64);
        let scale_view = scale_arr.view();
        let c_view = scirs2_core::ndarray::ArrayView1::from(c_f64);
        let result = scirs2_core::simd::simd_add_f64(&scale_view, &c_view);
        let result_vec = result.to_vec();

        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result_vec);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else {
        None
    }
}

/// SIMD broadcast add: add bias to each row of a 2D tensor
#[cfg(feature = "simd")]
fn try_simd_broadcast_add_2d<F: Float>(
    x: &[F],
    bias: &[F],
    rows: usize,
    cols: usize,
) -> Option<Vec<F>> {
    use crate::same_type;

    if same_type::<F, f32>() {
        let x_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let bias_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f32, bias.len()) };

        let bias_view = scirs2_core::ndarray::ArrayView1::from(bias_f32);
        let mut result: Vec<f32> = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let row_start = row * cols;
            let row_end = row_start + cols;
            let row_slice = &x_f32[row_start..row_end];
            let row_view = scirs2_core::ndarray::ArrayView1::from(row_slice);
            let row_result = scirs2_core::simd::simd_add_f32(&row_view, &bias_view);
            result.extend(row_result.iter().copied());
        }

        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else if same_type::<F, f64>() {
        let x_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let bias_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(bias.as_ptr() as *const f64, bias.len()) };

        let bias_view = scirs2_core::ndarray::ArrayView1::from(bias_f64);
        let mut result: Vec<f64> = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let row_start = row * cols;
            let row_end = row_start + cols;
            let row_slice = &x_f64[row_start..row_end];
            let row_view = scirs2_core::ndarray::ArrayView1::from(row_slice);
            let row_result = scirs2_core::simd::simd_add_f64(&row_view, &bias_view);
            result.extend(row_result.iter().copied());
        }

        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else {
        None
    }
}

/// SIMD broadcast mul: multiply each row of a 2D tensor by a scale vector
#[cfg(feature = "simd")]
fn try_simd_broadcast_mul_2d<F: Float>(
    x: &[F],
    scale: &[F],
    rows: usize,
    cols: usize,
) -> Option<Vec<F>> {
    use crate::same_type;

    if same_type::<F, f32>() {
        let x_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let scale_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(scale.as_ptr() as *const f32, scale.len()) };

        let scale_view = scirs2_core::ndarray::ArrayView1::from(scale_f32);
        let mut result: Vec<f32> = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let row_start = row * cols;
            let row_end = row_start + cols;
            let row_slice = &x_f32[row_start..row_end];
            let row_view = scirs2_core::ndarray::ArrayView1::from(row_slice);
            let row_result = scirs2_core::simd::simd_mul_f32(&row_view, &scale_view);
            result.extend(row_result.iter().copied());
        }

        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else if same_type::<F, f64>() {
        let x_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let scale_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(scale.as_ptr() as *const f64, scale.len()) };

        let scale_view = scirs2_core::ndarray::ArrayView1::from(scale_f64);
        let mut result: Vec<f64> = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let row_start = row * cols;
            let row_end = row_start + cols;
            let row_slice = &x_f64[row_start..row_end];
            let row_view = scirs2_core::ndarray::ArrayView1::from(row_slice);
            let row_result = scirs2_core::simd::simd_mul_f64(&row_view, &scale_view);
            result.extend(row_result.iter().copied());
        }

        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else {
        None
    }
}

/// SIMD-accelerated ReLU for f32/f64
#[cfg(feature = "simd")]
fn try_simd_relu<F: Float>(x: &[F]) -> Option<Vec<F>> {
    use crate::same_type;

    if same_type::<F, f32>() {
        let x_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let x_view = scirs2_core::ndarray::ArrayView1::from(x_f32);
        let result = scirs2_core::simd::simd_relu_f32(&x_view);
        let result_vec = result.to_vec();
        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result_vec);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else if same_type::<F, f64>() {
        let x_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let x_view = scirs2_core::ndarray::ArrayView1::from(x_f64);
        let result = scirs2_core::simd::simd_relu_f64(&x_view);
        let result_vec = result.to_vec();
        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result_vec);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else {
        None
    }
}

/// SIMD-accelerated sigmoid for f32/f64
#[cfg(feature = "simd")]
fn try_simd_sigmoid<F: Float>(x: &[F]) -> Option<Vec<F>> {
    use crate::same_type;

    if same_type::<F, f32>() {
        let x_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let x_view = scirs2_core::ndarray::ArrayView1::from(x_f32);
        let result = scirs2_core::simd::simd_sigmoid_f32(&x_view);
        let result_vec = result.to_vec();
        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result_vec);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else if same_type::<F, f64>() {
        let x_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let x_view = scirs2_core::ndarray::ArrayView1::from(x_f64);
        let result = scirs2_core::simd::simd_sigmoid_f64(&x_view);
        let result_vec = result.to_vec();
        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result_vec);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else {
        None
    }
}

/// SIMD-accelerated tanh for f32/f64
#[cfg(feature = "simd")]
fn try_simd_tanh<F: Float>(x: &[F]) -> Option<Vec<F>> {
    use crate::same_type;

    if same_type::<F, f32>() {
        let x_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let x_view = scirs2_core::ndarray::ArrayView1::from(x_f32);
        let result = scirs2_core::simd::simd_tanh_f32(&x_view);
        let result_vec = result.to_vec();
        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result_vec);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else if same_type::<F, f64>() {
        let x_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let x_view = scirs2_core::ndarray::ArrayView1::from(x_f64);
        let result = scirs2_core::simd::simd_tanh_f64(&x_view);
        let result_vec = result.to_vec();
        let result_f: Vec<F> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(result_vec);
            Vec::from_raw_parts(v.as_mut_ptr() as *mut F, v.len(), v.capacity())
        };
        Some(result_f)
    } else {
        None
    }
}

/// SIMD-accelerated dot product for f32/f64
#[cfg(feature = "simd")]
fn try_simd_dot<F: Float>(a: &[F], b: &[F]) -> Option<F> {
    use crate::same_type;

    if same_type::<F, f32>() {
        let a_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
        let b_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
        let a_view = scirs2_core::ndarray::ArrayView1::from(a_f32);
        let b_view = scirs2_core::ndarray::ArrayView1::from(b_f32);
        let result_f32 = scirs2_core::simd::simd_dot_f32(&a_view, &b_view);
        // SAFETY: F == f32
        let result: F = unsafe { *(&result_f32 as *const f32 as *const F) };
        Some(result)
    } else if same_type::<F, f64>() {
        let a_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
        let b_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f64, b.len()) };
        let a_view = scirs2_core::ndarray::ArrayView1::from(a_f64);
        let b_view = scirs2_core::ndarray::ArrayView1::from(b_f64);
        let result_f64 = scirs2_core::simd::simd_dot_f64(&a_view, &b_view);
        let result: F = unsafe { *(&result_f64 as *const f64 as *const F) };
        Some(result)
    } else {
        None
    }
}

/// SIMD-accelerated sum for f32/f64
#[cfg(feature = "simd")]
fn try_simd_sum<F: Float>(x: &[F]) -> Option<F> {
    use crate::same_type;

    if same_type::<F, f32>() {
        let x_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let x_view = scirs2_core::ndarray::ArrayView1::from(x_f32);
        let result_f32 = scirs2_core::simd::simd_sum_f32(&x_view);
        let result: F = unsafe { *(&result_f32 as *const f32 as *const F) };
        Some(result)
    } else if same_type::<F, f64>() {
        let x_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let x_view = scirs2_core::ndarray::ArrayView1::from(x_f64);
        let result_f64 = scirs2_core::simd::simd_sum_f64(&x_view);
        let result: F = unsafe { *(&result_f64 as *const f64 as *const F) };
        Some(result)
    } else {
        None
    }
}

// ============================================================================
// Public API: Tensor-level functions
// ============================================================================

/// SIMD-accelerated element-wise addition of two tensors.
///
/// Uses hardware SIMD instructions (AVX2/NEON) when the `simd` feature is enabled
/// and the tensors are contiguous in memory. Falls back to scalar otherwise.
///
/// # Arguments
/// * `a` - Left operand tensor
/// * `b` - Right operand tensor
///
/// # Returns
/// A new tensor containing the element-wise sum.
pub fn simd_elementwise_add<'g, F: Float>(a: &Tensor<'g, F>, b: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(SimdElementwiseAdd)
}

/// SIMD-accelerated element-wise subtraction.
pub fn simd_elementwise_sub<'g, F: Float>(a: &Tensor<'g, F>, b: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(SimdElementwiseSub)
}

/// SIMD-accelerated element-wise multiplication.
pub fn simd_elementwise_mul<'g, F: Float>(a: &Tensor<'g, F>, b: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(SimdElementwiseMul)
}

/// SIMD-accelerated element-wise division.
pub fn simd_elementwise_div<'g, F: Float>(a: &Tensor<'g, F>, b: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(SimdElementwiseDiv)
}

/// SIMD-accelerated gradient accumulation.
///
/// This is the critical inner-loop operation during backpropagation.
/// Accumulates `gradient` into the existing `accumulator`.
///
/// # Arguments
/// * `accumulator` - Existing gradient accumulator
/// * `gradient` - New gradient to accumulate
pub fn simd_gradient_accumulate<'g, F: Float>(
    accumulator: &Tensor<'g, F>,
    gradient: &Tensor<'g, F>,
) -> Tensor<'g, F> {
    let g = accumulator.graph();
    Tensor::builder(g)
        .append_input(accumulator, false)
        .append_input(gradient, false)
        .build(SimdGradientAccumulate)
}

/// SIMD-accelerated scaled gradient accumulation: `acc + scale * grad`
///
/// Fused operation common in optimizers (momentum SGD, Adam, etc.)
pub fn simd_scaled_gradient_accumulate<'g, F: Float>(
    accumulator: &Tensor<'g, F>,
    gradient: &Tensor<'g, F>,
    scale: F,
) -> Tensor<'g, F> {
    let g = accumulator.graph();
    Tensor::builder(g)
        .append_input(accumulator, false)
        .append_input(gradient, false)
        .build(SimdScaledGradientAccumulate { scale })
}

/// SIMD-accelerated broadcast addition (bias addition pattern).
///
/// Adds a 1-D bias to each row of a 2-D tensor.
pub fn simd_broadcast_add<'g, F: Float>(x: &Tensor<'g, F>, bias: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .append_input(bias, false)
        .build(SimdBroadcastAdd)
}

/// SIMD-accelerated broadcast multiplication.
///
/// Multiplies each row of a 2-D tensor by a 1-D scale vector.
pub fn simd_broadcast_mul<'g, F: Float>(x: &Tensor<'g, F>, scale: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .append_input(scale, false)
        .build(SimdBroadcastMul)
}

/// SIMD-accelerated ReLU activation.
pub fn simd_activation_relu<'g, F: Float>(x: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = x.graph();
    Tensor::builder(g).append_input(x, false).build(SimdReLU)
}

/// SIMD-accelerated sigmoid activation.
pub fn simd_activation_sigmoid<'g, F: Float>(x: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = x.graph();
    Tensor::builder(g).append_input(x, false).build(SimdSigmoid)
}

/// SIMD-accelerated tanh activation.
pub fn simd_activation_tanh<'g, F: Float>(x: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = x.graph();
    Tensor::builder(g).append_input(x, false).build(SimdTanh)
}

/// SIMD-accelerated dot product of two 1-D tensors.
pub fn simd_dot_product<'g, F: Float>(a: &Tensor<'g, F>, b: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a, false)
        .append_input(b, false)
        .build(SimdDotProduct)
}

/// SIMD-accelerated sum reduction of a 1-D tensor.
pub fn simd_reduction_sum<'g, F: Float>(x: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .build(SimdReductionSum)
}

// ============================================================================
// Configuration
// ============================================================================

/// Performance configuration for SIMD operations.
///
/// Controls minimum array sizes for engaging SIMD vs scalar paths,
/// and other tuning parameters.
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Minimum number of elements before SIMD is used (default: 16)
    pub min_simd_length: usize,
    /// Whether to prefer FMA (fused multiply-add) when available (default: true)
    pub prefer_fma: bool,
    /// Whether to use adaptive algorithm selection (default: true)
    pub adaptive_dispatch: bool,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            min_simd_length: 16,
            prefer_fma: true,
            adaptive_dispatch: true,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate as ag;
    use scirs2_core::ndarray::{array, Array1, ArrayView1};

    /// Helper to compare two float slices within epsilon
    fn assert_approx_eq_f32(actual: &[f32], expected: &[f32], epsilon: f32) {
        assert_eq!(actual.len(), expected.len(), "Length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < epsilon,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i,
                a,
                e,
                (a - e).abs()
            );
        }
    }

    fn assert_approx_eq_f64(actual: &[f64], expected: &[f64], epsilon: f64) {
        assert_eq!(actual.len(), expected.len(), "Length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < epsilon,
                "Mismatch at index {}: actual={}, expected={}, diff={}",
                i,
                a,
                e,
                (a - e).abs()
            );
        }
    }

    // -------------------------------------------------------
    // Element-wise arithmetic: correctness (SIMD vs scalar)
    // -------------------------------------------------------

    #[test]
    fn test_simd_elementwise_add_f32() {
        ag::run::<f32, _, _>(|ctx| {
            let a_arr = array![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ];
            let b_arr = array![
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
            ];
            let expected: Vec<f32> = a_arr
                .iter()
                .zip(b_arr.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            let a = ag::tensor_ops::convert_to_tensor(a_arr.clone(), ctx);
            let b = ag::tensor_ops::convert_to_tensor(b_arr.clone(), ctx);
            let y = simd_elementwise_add(&a, &b);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f32(result_slice, &expected, 1e-6);
                }
            }
        });
    }

    #[test]
    fn test_simd_elementwise_sub_f64() {
        ag::run::<f64, _, _>(|ctx| {
            let a_arr = array![10.0f64, 20.0, 30.0, 40.0];
            let b_arr = array![1.0f64, 2.0, 3.0, 4.0];
            let expected: Vec<f64> = a_arr
                .iter()
                .zip(b_arr.iter())
                .map(|(&a, &b)| a - b)
                .collect();

            let a = ag::tensor_ops::convert_to_tensor(a_arr, ctx);
            let b = ag::tensor_ops::convert_to_tensor(b_arr, ctx);
            let y = simd_elementwise_sub(&a, &b);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f64(result_slice, &expected, 1e-12);
                }
            }
        });
    }

    #[test]
    fn test_simd_elementwise_mul_f32() {
        ag::run::<f32, _, _>(|ctx| {
            let a_arr = array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b_arr = array![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            let expected: Vec<f32> = a_arr
                .iter()
                .zip(b_arr.iter())
                .map(|(&a, &b)| a * b)
                .collect();

            let a = ag::tensor_ops::convert_to_tensor(a_arr, ctx);
            let b = ag::tensor_ops::convert_to_tensor(b_arr, ctx);
            let y = simd_elementwise_mul(&a, &b);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f32(result_slice, &expected, 1e-6);
                }
            }
        });
    }

    #[test]
    fn test_simd_elementwise_div_f64() {
        ag::run::<f64, _, _>(|ctx| {
            let a_arr = array![10.0f64, 20.0, 30.0, 40.0];
            let b_arr = array![2.0f64, 4.0, 5.0, 8.0];
            let expected: Vec<f64> = a_arr
                .iter()
                .zip(b_arr.iter())
                .map(|(&a, &b)| a / b)
                .collect();

            let a = ag::tensor_ops::convert_to_tensor(a_arr, ctx);
            let b = ag::tensor_ops::convert_to_tensor(b_arr, ctx);
            let y = simd_elementwise_div(&a, &b);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f64(result_slice, &expected, 1e-12);
                }
            }
        });
    }

    // -------------------------------------------------------
    // Gradient accumulation
    // -------------------------------------------------------

    #[test]
    fn test_simd_gradient_accumulate_f32() {
        ag::run::<f32, _, _>(|ctx| {
            let acc_arr = array![1.0f32, 2.0, 3.0, 4.0];
            let grad_arr = array![0.1f32, 0.2, 0.3, 0.4];
            let expected = vec![1.1, 2.2, 3.3, 4.4];

            let acc = ag::tensor_ops::convert_to_tensor(acc_arr, ctx);
            let grad = ag::tensor_ops::convert_to_tensor(grad_arr, ctx);
            let y = simd_gradient_accumulate(&acc, &grad);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f32(result_slice, &expected, 1e-6);
                }
            }
        });
    }

    #[test]
    fn test_simd_scaled_gradient_accumulate_f32() {
        ag::run::<f32, _, _>(|ctx| {
            let acc_arr = array![1.0f32, 2.0, 3.0, 4.0];
            let grad_arr = array![10.0f32, 20.0, 30.0, 40.0];
            let scale = 0.1f32;
            // expected: acc + scale * grad = [2.0, 4.0, 6.0, 8.0]
            let expected = vec![2.0, 4.0, 6.0, 8.0];

            let acc = ag::tensor_ops::convert_to_tensor(acc_arr, ctx);
            let grad = ag::tensor_ops::convert_to_tensor(grad_arr, ctx);
            let y = simd_scaled_gradient_accumulate(&acc, &grad, scale);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f32(result_slice, &expected, 1e-5);
                }
            }
        });
    }

    // -------------------------------------------------------
    // Activation functions
    // -------------------------------------------------------

    #[test]
    fn test_simd_relu_f32() {
        ag::run::<f32, _, _>(|ctx| {
            let x_arr = array![-3.0f32, -1.0, 0.0, 1.0, 3.0, -0.5, 2.0, -2.0];
            let expected = vec![0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0];

            let x = ag::tensor_ops::convert_to_tensor(x_arr, ctx);
            let y = simd_activation_relu(&x);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f32(result_slice, &expected, 1e-6);
                }
            }
        });
    }

    #[test]
    fn test_simd_sigmoid_f32() {
        ag::run::<f32, _, _>(|ctx| {
            let x_arr = array![0.0f32, 1.0, -1.0, 5.0, -5.0, 0.5, -0.5, 2.0];
            let expected: Vec<f32> = x_arr.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();

            let x = ag::tensor_ops::convert_to_tensor(x_arr, ctx);
            let y = simd_activation_sigmoid(&x);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f32(result_slice, &expected, 1e-4);
                }
            }
        });
    }

    #[test]
    fn test_simd_tanh_f64() {
        ag::run::<f64, _, _>(|ctx| {
            let x_arr = array![0.0f64, 1.0, -1.0, 2.0, -2.0, 0.5];
            let expected: Vec<f64> = x_arr.iter().map(|&v| v.tanh()).collect();

            let x = ag::tensor_ops::convert_to_tensor(x_arr, ctx);
            let y = simd_activation_tanh(&x);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f64(result_slice, &expected, 1e-10);
                }
            }
        });
    }

    // -------------------------------------------------------
    // Dot product / reduction
    // -------------------------------------------------------

    #[test]
    fn test_simd_dot_product_f32() {
        ag::run::<f32, _, _>(|ctx| {
            let a_arr = array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b_arr = array![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            let expected: f32 = a_arr.iter().zip(b_arr.iter()).map(|(&a, &b)| a * b).sum();

            let a = ag::tensor_ops::convert_to_tensor(a_arr, ctx);
            let b = ag::tensor_ops::convert_to_tensor(b_arr, ctx);
            let y = simd_dot_product(&a, &b);

            if let Ok(result) = y.eval(ctx) {
                let val = result.iter().next().copied().unwrap_or(0.0);
                assert!(
                    (val - expected).abs() < 1e-3,
                    "dot product: got {}, expected {}",
                    val,
                    expected
                );
            }
        });
    }

    #[test]
    fn test_simd_reduction_sum_f64() {
        ag::run::<f64, _, _>(|ctx| {
            let x_arr = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let expected: f64 = x_arr.iter().sum();

            let x = ag::tensor_ops::convert_to_tensor(x_arr, ctx);
            let y = simd_reduction_sum(&x);

            if let Ok(result) = y.eval(ctx) {
                let val = result.iter().next().copied().unwrap_or(0.0);
                assert!(
                    (val - expected).abs() < 1e-10,
                    "sum: got {}, expected {}",
                    val,
                    expected
                );
            }
        });
    }

    // -------------------------------------------------------
    // Edge cases
    // -------------------------------------------------------

    #[test]
    fn test_simd_empty_array() {
        ag::run::<f32, _, _>(|ctx| {
            let empty = scirs2_core::ndarray::Array1::<f32>::zeros(0);
            let a = ag::tensor_ops::convert_to_tensor(empty.clone(), ctx);
            let b = ag::tensor_ops::convert_to_tensor(empty, ctx);
            let y = simd_elementwise_add(&a, &b);
            if let Ok(result) = y.eval(ctx) {
                assert_eq!(result.len(), 0);
            }
        });
    }

    #[test]
    fn test_simd_single_element() {
        ag::run::<f64, _, _>(|ctx| {
            let a_arr = array![42.0f64];
            let b_arr = array![8.0f64];
            let a = ag::tensor_ops::convert_to_tensor(a_arr, ctx);
            let b = ag::tensor_ops::convert_to_tensor(b_arr, ctx);
            let y = simd_elementwise_mul(&a, &b);
            if let Ok(result) = y.eval(ctx) {
                if let Some(slice) = result.as_slice() {
                    assert_approx_eq_f64(slice, &[336.0], 1e-12);
                }
            }
        });
    }

    #[test]
    fn test_simd_relu_all_negative() {
        ag::run::<f32, _, _>(|ctx| {
            let x_arr = array![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
            let expected = vec![0.0f32; 8];

            let x = ag::tensor_ops::convert_to_tensor(x_arr, ctx);
            let y = simd_activation_relu(&x);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f32(result_slice, &expected, 1e-6);
                }
            }
        });
    }

    #[test]
    fn test_simd_sigmoid_extreme_values() {
        ag::run::<f32, _, _>(|ctx| {
            let x_arr = array![-100.0f32, 100.0, 0.0, -50.0, 50.0, -10.0, 10.0, 0.0];
            // sigmoid(-100) ~ 0, sigmoid(100) ~ 1, sigmoid(0) = 0.5
            let x = ag::tensor_ops::convert_to_tensor(x_arr, ctx);
            let y = simd_activation_sigmoid(&x);

            if let Ok(result) = y.eval(ctx) {
                if let Some(slice) = result.as_slice() {
                    // Very negative => ~0
                    assert!(
                        slice[0] < 1e-6,
                        "sigmoid(-100) should be near 0, got {}",
                        slice[0]
                    );
                    // Very positive => ~1
                    assert!(
                        (slice[1] - 1.0).abs() < 1e-6,
                        "sigmoid(100) should be near 1, got {}",
                        slice[1]
                    );
                    // Zero => 0.5
                    assert!(
                        (slice[2] - 0.5).abs() < 1e-4,
                        "sigmoid(0) should be 0.5, got {}",
                        slice[2]
                    );
                }
            }
        });
    }

    #[test]
    fn test_simd_large_array_add() {
        // Test with a large array to exercise the full SIMD path (>32 elements)
        ag::run::<f32, _, _>(|ctx| {
            let n = 1024;
            let a_vec: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let b_vec: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
            let expected: Vec<f32> = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            let a_arr = Array1::from_vec(a_vec);
            let b_arr = Array1::from_vec(b_vec);

            let a = ag::tensor_ops::convert_to_tensor(a_arr, ctx);
            let b = ag::tensor_ops::convert_to_tensor(b_arr, ctx);
            let y = simd_elementwise_add(&a, &b);

            if let Ok(result) = y.eval(ctx) {
                if let Some(result_slice) = result.as_slice() {
                    assert_approx_eq_f32(result_slice, &expected, 1e-4);
                }
            }
        });
    }

    // -------------------------------------------------------
    // Gradient correctness tests
    // -------------------------------------------------------

    #[test]
    fn test_simd_add_gradient() {
        ag::run::<f64, _, _>(|ctx| {
            let x = ctx.placeholder("x", &[4]);
            let y = ctx.placeholder("y", &[4]);
            let z = simd_elementwise_add(&x, &y);
            let sum_z = ag::tensor_ops::sum_all(z);

            let grads = ag::tensor_ops::grad(&[sum_z], &[x, y]);

            let x_val = array![1.0f64, 2.0, 3.0, 4.0];
            let y_val = array![5.0f64, 6.0, 7.0, 8.0];

            let results = ctx
                .evaluator()
                .push(&grads[0])
                .push(&grads[1])
                .feed(x, x_val.view().into_dyn())
                .feed(y, y_val.view().into_dyn())
                .run();

            // d(sum(x+y))/dx = [1,1,1,1], d(sum(x+y))/dy = [1,1,1,1]
            if let Some(Ok(dx)) = results.first() {
                if let Some(dx_slice) = dx.as_slice() {
                    assert_approx_eq_f64(dx_slice, &[1.0, 1.0, 1.0, 1.0], 1e-10);
                }
            }
            if let Some(Ok(dy)) = results.get(1) {
                if let Some(dy_slice) = dy.as_slice() {
                    assert_approx_eq_f64(dy_slice, &[1.0, 1.0, 1.0, 1.0], 1e-10);
                }
            }
        });
    }

    #[test]
    fn test_simd_mul_gradient() {
        ag::run::<f64, _, _>(|ctx| {
            let x = ctx.placeholder("x", &[4]);
            let y = ctx.placeholder("y", &[4]);
            let z = simd_elementwise_mul(&x, &y);
            let sum_z = ag::tensor_ops::sum_all(z);

            let grads = ag::tensor_ops::grad(&[sum_z], &[x, y]);

            let x_val = array![1.0f64, 2.0, 3.0, 4.0];
            let y_val = array![5.0f64, 6.0, 7.0, 8.0];

            let results = ctx
                .evaluator()
                .push(&grads[0])
                .push(&grads[1])
                .feed(x, x_val.view().into_dyn())
                .feed(y, y_val.view().into_dyn())
                .run();

            // d(sum(x*y))/dx = y, d(sum(x*y))/dy = x
            if let Some(Ok(dx)) = results.first() {
                if let Some(dx_slice) = dx.as_slice() {
                    assert_approx_eq_f64(dx_slice, &[5.0, 6.0, 7.0, 8.0], 1e-10);
                }
            }
            if let Some(Ok(dy)) = results.get(1) {
                if let Some(dy_slice) = dy.as_slice() {
                    assert_approx_eq_f64(dy_slice, &[1.0, 2.0, 3.0, 4.0], 1e-10);
                }
            }
        });
    }
}
