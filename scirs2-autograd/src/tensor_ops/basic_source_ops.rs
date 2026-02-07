use crate::ndarray_ext::NdArray;
use crate::op::OpError;
use crate::Float;

// Structure for Variable op which provides access to a variable in the graph
pub struct Variable;

impl<T: Float> crate::op::Op<T> for Variable {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), OpError> {
        // For a Variable op, the operation should just pass through any inputs
        // directly to the outputs, or return an empty array if no inputs exist
        if ctx.inputs().is_empty() {
            // If there are no inputs (this might be a variable initialization),
            // create an empty output as a placeholder
            ctx.append_output(NdArray::zeros(vec![]));
        } else {
            // Otherwise, propagate the first input to the output
            let input = ctx.input(0).to_owned();
            ctx.append_output(input);
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // Just pass through any gradients
        let gy = ctx.output_grad();
        ctx.append_input_grad(0, Some(gy));
    }
}

// Structure for Const op which represents a constant value in the graph
pub struct Const;

impl<T: Float> crate::op::Op<T> for Const {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), OpError> {
        // Similar implementation to Variable for now
        if ctx.inputs().is_empty() {
            ctx.append_output(NdArray::zeros(vec![]));
        } else {
            let input = ctx.input(0).to_owned();
            ctx.append_output(input);
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // Constants have zero gradient
        ctx.append_input_grad(0, None);
    }
}

// Structure for Placeholder op which represents placeholders in the graph
pub struct Placeholder;

impl<T: Float> crate::op::Op<T> for Placeholder {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), OpError> {
        // Similar implementation to Variable
        if ctx.inputs().is_empty() {
            ctx.append_output(NdArray::zeros(vec![]));
        } else {
            let input = ctx.input(0).to_owned();
            ctx.append_output(input);
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        // Pass through gradients like Variable
        let gy = ctx.output_grad();
        ctx.append_input_grad(0, Some(gy));
    }
}
