use crate::ndarray_ext::NdArray;
use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::{Context, Float};
use scirs2_core::ndarray::{Array1, Array2, Ix1, Ix2};

/// Identity matrix operation
pub struct EyeOp {
    pub size: usize,
}

impl<F: Float> Op<F> for EyeOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let mut arr = Array2::<F>::zeros((self.size, self.size));
        for i in 0..self.size {
            arr[[i, i]] = F::one();
        }
        ctx.append_output(arr.into_dyn());
        Ok(())
    }

    /// `eye(n)` is a generator: its value depends only on `self.size`, and the node has
    /// **no inputs**.  There is nothing to append and nothing to propagate.
    fn grad<'a, 'g>(&self, _ctx: &mut GradientContext<'a, 'g, F>) {}
}

/// Trace operation
pub struct TraceOp;

impl<F: Float> Op<F> for TraceOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Trace requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to reshape".into()))?;

        // Compute the trace by summing diagonal elements
        let mut trace = F::zero();
        for i in 0..shape[0] {
            // Extract diagonal values
            let diag_val = input_2d[[i, i]];
            trace += diag_val;
        }

        // Create a proper scalar output
        ctx.append_output(scirs2_core::ndarray::arr0(trace).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Evaluate input to get dimensions
        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let shape = input_array.shape();
        let n = shape[0];

        // Create a diagonal matrix with gradient value
        let mut grad = NdArray::<F>::zeros(shape);
        if let Ok(mut grad_2d) = grad.view_mut().into_dimensionality::<Ix2>() {
            // Get scalar gradient value
            let gy_array = match gy.eval(g) {
                Ok(arr) => arr,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    return;
                }
            };

            let scalar_grad = gy_array[[]];

            for i in 0..n {
                grad_2d[[i, i]] = scalar_grad;
            }
        }

        ctx.append_input_grad(
            0,
            Some(crate::tensor_ops::convert_to_tensor(grad, ctx.graph())),
        );
    }
}

/// Backward op for [`TraceOp`].
///
/// `trace` maps an `n × n` matrix `A` to the scalar `Σ_i A_ii`, so its
/// reverse-mode VJP w.r.t. `A` for an upstream scalar cotangent `gy` is the
/// `n × n` matrix `gy · I_n` (the diagonal carries `gy`, off-diagonals are
/// zero).  This is delivered through a dedicated backward op rather than the
/// `Op::grad` trait method because the string-dispatch gradient path in
/// `gradient.rs::compute_grad_for_input` does not invoke `Op::grad`; returning
/// the bare scalar `gy` there (the previous behaviour) produced a 0-dimensional
/// gradient that downstream matrix backward ops could not consume.
///
/// Inputs: `(x_input, gy)` — `x_input` provides the matrix dimension `n` (its
/// values are unused, so no spurious dependency on the forward graph is
/// introduced for higher-order differentiation); `gy` is the upstream scalar.
pub(crate) struct TraceBackwardOp;

impl<F: Float> Op<F> for TraceBackwardOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let x = ctx.input(0);
        let gy = ctx.input(1);

        let x_shape = x.shape();
        if x_shape.len() != 2 || x_shape[0] != x_shape[1] {
            return Err(OpError::IncompatibleShape(
                "TraceBackward: input must be a square matrix".into(),
            ));
        }
        let n = x_shape[0];

        // Extract the scalar cotangent (trace produces a 0-d output, but be
        // permissive about a single-element representation).
        let gy_scalar = if gy.ndim() == 0 {
            gy[scirs2_core::ndarray::IxDyn(&[])]
        } else if gy.len() == 1 {
            match gy.iter().next() {
                Some(&v) => v,
                None => {
                    return Err(OpError::IncompatibleShape(
                        "TraceBackward: empty cotangent".into(),
                    ))
                }
            }
        } else {
            return Err(OpError::IncompatibleShape(
                "TraceBackward: cotangent of trace must be a scalar".into(),
            ));
        };

        let mut grad = Array2::<F>::zeros((n, n));
        for i in 0..n {
            grad[[i, i]] = gy_scalar;
        }
        ctx.append_output(grad.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Second-order gradient unsupported (mirrors the other matrix backward
        // ops); first-order trace gradients are exact.
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

/// Diagonal matrix creation
pub struct DiagOp;

impl<F: Float> Op<F> for DiagOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);

        // Check if input is a vector
        let shape = input.shape();
        if shape.len() != 1 {
            return Err(OpError::IncompatibleShape(
                "Diag op requires a 1D vector input".into(),
            ));
        }

        let n = shape[0];

        // Get the input data as a 1D array
        let input_1d = input
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 1D vector".into()))?;

        // Create a diagonal matrix
        let mut output = Array2::<F>::zeros((n, n));
        for i in 0..n {
            output[[i, i]] = input_1d[i];
        }

        ctx.append_output(output.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let g = ctx.graph();

        // Get gradient array via evaluation
        let gy_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let shape = gy_array.shape();

        if shape.len() == 2 && shape[0] == shape[1] {
            let n = shape[0];
            let mut grad = Array1::<F>::zeros(n).into_dyn();

            // Get 2D view of gradient array
            if let Ok(gy_2d) = gy_array.view().into_dimensionality::<Ix2>() {
                for i in 0..n {
                    grad[[i]] = gy_2d[[i, i]];
                }
            }

            ctx.append_input_grad(
                0,
                Some(crate::tensor_ops::convert_to_tensor(grad, ctx.graph())),
            );
        } else {
            // If shape is not compatible, return None gradient
            ctx.append_input_grad(0, None);
        }
    }
}

/// Extract diagonal operation
pub struct ExtractDiagOp;

impl<F: Float> Op<F> for ExtractDiagOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "Extract diag requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to reshape".into()))?;

        let n = shape[0];
        let mut diag = Array1::<F>::zeros(n).into_dyn();

        for i in 0..n {
            diag[[i]] = input_2d[[i, i]];
        }

        ctx.append_output(diag);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let input = ctx.input(0);
        let g = ctx.graph();

        // Get input array via evaluation to get its shape
        let input_array = match input.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Get gradient array via evaluation
        let gy_array = match gy.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        let shape = input_array.shape();

        // Create a zero matrix and fill diagonal with gradient values
        let mut grad = NdArray::<F>::zeros(shape);
        if let Ok(mut grad_2d) = grad.view_mut().into_dimensionality::<Ix2>() {
            let n = gy_array.len();

            // Get 1D view of gradient array if possible
            let gy_1d = match gy_array.view().into_dimensionality::<Ix1>() {
                Ok(view) => view,
                Err(_) => {
                    ctx.append_input_grad(0, None);
                    return;
                }
            };

            for i in 0..n {
                grad_2d[[i, i]] = gy_1d[i];
            }
        }

        ctx.append_input_grad(
            0,
            Some(crate::tensor_ops::convert_to_tensor(grad, ctx.graph())),
        );
    }
}

// Public functions

/// Create an identity matrix
#[allow(dead_code)]
pub fn eye<'g, F: Float>(n: usize, ctx: &'g Context<F>) -> Tensor<'g, F> {
    Tensor::builder(ctx).build(EyeOp { size: n })
}

/// Compute the trace of a matrix
#[allow(dead_code)]
pub fn trace<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(TraceOp)
}

/// Create a diagonal matrix from a vector
#[allow(dead_code)]
pub fn diag<'g, F: Float>(v: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = v.graph();
    Tensor::builder(g).append_input(v, false).build(DiagOp)
}

/// Extract diagonal elements from a matrix
#[allow(dead_code)]
pub fn extract_diag<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(matrix, false)
        .build(ExtractDiagOp)
}
