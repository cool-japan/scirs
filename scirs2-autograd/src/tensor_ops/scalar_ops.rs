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
        let grad_output = ctx.output_grad();
        let g = ctx.graph();

        // Evaluate gradient tensor to array
        let grad_output_array = match grad_output.eval(g) {
            Ok(arr) => arr,
            Err(_) => {
                ctx.append_input_grad(0, None);
                return;
            }
        };

        // Compute gradient (scalar multiplication)
        let grad_input = grad_output_array.mapv(|x| x * self.scalar);

        // Convert to tensor and append
        let grad_tensor = crate::tensor_ops::convert_to_tensor(grad_input, g);
        ctx.append_input_grad(0, Some(grad_tensor));
    }
}

pub fn scalar_mul<'g, F: Float>(tensor: &Tensor<'g, F>, scalar: F) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .set_shape(&crate::tensor_ops::shape(tensor))
        .build(ScalarMulOp { scalar })
}
