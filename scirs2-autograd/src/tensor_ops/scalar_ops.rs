use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;

/// Scalar multiplication operation
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
        // Gradient requires eager eval which is unavailable during graph construction
        ctx.append_input_grad(0, None);
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
