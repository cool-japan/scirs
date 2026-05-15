use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;

/// Scalar multiplication operation.
///
/// `as_any()` is implemented so that `gradient.rs` can downcast the op and
/// retrieve the scalar for symbolic gradient propagation.
pub struct ScalarMulOp<F: Float> {
    pub scalar: F,
}

impl<F: Float> Op<F> for ScalarMulOp<F> {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let input_array = input.view();
        let output = input_array.mapv(|x| x * self.scalar);
        ctx.append_output(output);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Propagate symbolically so higher-order gradients work correctly.
        // Calling .eval() here would collapse the tape and break second/third
        // derivative chains.
        //
        // NOTE: gradient.rs uses op-name-based dispatch and downcasts via
        // as_any() to retrieve the scalar.  This method exists as a fallback.
        let grad_output = ctx.output_grad();
        let grad_input = crate::tensor_ops::scalar_mul(grad_output, self.scalar);
        ctx.append_input_grad(0, Some(grad_input));
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

#[allow(dead_code)]
pub fn scalar_mul<'g, F: Float>(tensor: &Tensor<'g, F>, scalar: F) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .setshape(&crate::tensor_ops::shape(tensor))
        .build(ScalarMulOp { scalar })
}
