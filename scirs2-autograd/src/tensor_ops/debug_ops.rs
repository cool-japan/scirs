// Debug operators for testing gradient computation
use crate::ndarray;
use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;

/// A simple operator that passes through the input unchanged but has a very simple gradient
pub struct DebugIdentityWithGradient;

impl<F: Float> Op<F> for DebugIdentityWithGradient {
    fn name(&self) -> &'static str {
        "DebugIdentityWithGradient"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Just pass the input through directly
        let input = ctx.input(0);
        ctx.append_output(input.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        println!("DEBUG: DebugIdentityWithGradient::grad is called");

        // Get the output gradient
        let grad_output = ctx.output_grad();
        println!("DEBUG: Output gradient tensor id: {}", grad_output.id);

        // Pass it straight through as the input gradient
        ctx.append_input_grad(0, Some(grad_output));
        println!("DEBUG: Input gradient appended");
    }
}

/// A simple operator that returns a scalar filled with 1.0 and has a simple gradient of all 1s
pub struct DebugScalarOne;

impl<F: Float> Op<F> for DebugScalarOne {
    fn name(&self) -> &'static str {
        "DebugScalarOne"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Return a scalar with value 1.0
        ctx.append_output(scirs2_core::ndarray::arr0(F::one()).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Gradient requires eager eval which is unavailable during graph construction
        ctx.append_input_grad(0, None);
    }
}

// Public API function
#[allow(dead_code)]
pub fn debug_identity_with_gradient<'g, F: Float>(tensor: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(DebugIdentityWithGradient)
}

// Public API function
#[allow(dead_code)]
pub fn debug_scalar_one<'g, F: Float>(tensor: &Tensor<'g, F>) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(DebugScalarOne)
}
