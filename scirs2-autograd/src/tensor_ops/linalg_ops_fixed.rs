use crate::ndarray_ext::NdArray;
use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::{Context, Float};
use scirs2_core::ndarray::{Array2, Ix2};

/// Identity matrix operation
pub struct EyeOp {
    size: usize,
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

    fn grad(selfctx: &mut GradientContext<F>) {
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
            return Err(OpError::InvalidShape("Trace requires square matrix".into()));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::InvalidShape("Failed to reshape".into()))?;

        let mut trace = F::zero();
        for i in 0..shape[0] {
            trace = trace + input_2d[[i, i]];
        }

        ctx.append_output(scirs2_core::ndarray::arr0(trace).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let shape = ctx.input(0).shape().clone();
        
        let mut grad = NdArray::<F>::zeros(&shape);
        if let Ok(grad_2d) = grad.view_mut().into_dimensionality::<Ix2>() {
            let n = shape[0];
            for i in 0..n {
                grad_2d[[i, i]] = gy[[]] * F::one();
            }
        }
        
        ctx.append_input_grad(0, Some(crate::tensor_ops::convert_to_tensor(grad, ctx.graph())));
    }
}

/// Diagonal matrix creation
pub struct DiagOp;

impl<F: Float> Op<F> for DiagOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let n = input.len();
        
        let mut output = Array2::<F>::zeros((n, n));
        for i in 0..n {
            output[[i, i]] = input[[i]];
        }
        
        ctx.append_output(output.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let shape = gy.shape();
        
        if shape.len() == 2 && shape[0] == shape[1] {
            let n = shape[0];
            let mut grad = NdArray::<F>::zeros(n);
            
            if let Ok(gy_2d) = gy.as_array().view().into_dimensionality::<Ix2>() {
                for i in 0..n {
                    grad[[i]] = gy_2d[[i, i]];
                }
            }
            
            ctx.append_input_grad(0, Some(crate::tensor_ops::convert_to_tensor(grad, ctx.graph())));
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
            return Err(OpError::InvalidShape("Extract diag requires square matrix".into()));
        }
        
        let input_2d = input.view().into_dimensionality::<Ix2>()
            .map_err(|_| OpError::InvalidShape("Failed to reshape".into()))?;
        
        let n = shape[0];
        let mut diag = NdArray::<F>::zeros(n);
        
        for i in 0..n {
            diag[[i]] = input_2d[[i, i]];
        }
        
        ctx.append_output(diag);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let shape = ctx.input(0).shape().clone();
        
        let mut grad = NdArray::<F>::zeros(&shape);
        if let Ok(grad_2d) = grad.view_mut().into_dimensionality::<Ix2>() {
            let n = gy.len();
            for i in 0..n {
                grad_2d[[i, i]] = gy[[i]];
            }
        }
        
        ctx.append_input_grad(0, Some(crate::tensor_ops::convert_to_tensor(grad, ctx.graph())));
    }
}

// Public functions

/// Create an identity matrix
#[allow(dead_code)]
pub fn eye<'g, F: Float>(n: usize, ctx: &'g Context<F>) -> Tensor<'g, F> {
    Tensor::builder(ctx)
        .build(EyeOp { size: n })
}

/// Compute the trace of a matrix
#[allow(dead_code)]
pub fn trace<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(_matrix, false)
        .build(TraceOp)
}

/// Create a diagonal matrix from a vector
#[allow(dead_code)]
pub fn diag<'g, F: Float>(v: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = v.graph();
    Tensor::builder(g)
        .append_input(v, false)
        .build(DiagOp)
}

/// Extract diagonal elements from a matrix
#[allow(dead_code)]
pub fn extract_diag<'g, F: Float>(matrix: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = matrix.graph();
    Tensor::builder(g)
        .append_input(_matrix, false)
        .build(ExtractDiagOp)
}
