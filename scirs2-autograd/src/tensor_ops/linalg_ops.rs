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

    fn grad(&self, _ctx: &mut GradientContext<F>) {
        // Identity matrix is constant, no gradient
    }
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
        // Trace gradient requires eager eval which is not available during graph construction.
        // TODO: implement symbolically as gy * eye(n)
        ctx.append_input_grad(0, None);
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
        // DiagOp gradient requires eager eval which is not available during graph construction.
        // TODO: implement symbolically as extract_diag(gy)
        ctx.append_input_grad(0, None);
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
        // ExtractDiagOp gradient requires eager eval which is not available during graph construction.
        // TODO: implement symbolically as diag(gy)
        ctx.append_input_grad(0, None);
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
