use crate::ndarray;
use crate::ndarray_ext;
use crate::ndarray_ext::NdArray;
use crate::op;
use crate::Float;

pub struct Zeros;
pub struct Ones;
pub struct ConvertToTensor<T: Float> {
    pub arr: NdArray<T>,
}
pub struct Scalar<T: Float> {
    pub val: T,
}

impl<T: Float> op::Op<T> for Scalar<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        ctx.append_output(scirs2_core::ndarray::arr0(self.val).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Zeros {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = &ctx.input(0);
        let ret = if let Some(a) = shape.as_slice() {
            ndarray_ext::zeros(
                a.iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        } else {
            ndarray_ext::zeros(
                shape
                    .iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        };
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Ones {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = &ctx.input(0);
        let ret = if let Some(a) = shape.as_slice() {
            ndarray_ext::ones(
                a.iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        } else {
            ndarray_ext::ones(
                shape
                    .iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        };
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for ConvertToTensor<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        // Save the original array shape for debugging
        let shape = self.arr.shape();
        println!(
            "ConvertToTensor: Input array shape: {:?}, ndim: {}",
            shape,
            self.arr.ndim()
        );

        // Always preserve the original array's shape exactly
        let output = self.arr.clone();

        // Check the created array
        println!(
            "DEBUG: Created tensor with shape: {:?}, ndim: {}",
            output.shape(),
            output.ndim()
        );

        // Additional debugging info
        if output.ndim() == 0 {
            println!("DEBUG: Scalar tensor value: {}", output[[]]);
        } else if output.ndim() == 1 && !output.is_empty() {
            println!("DEBUG: Vector tensor first element: {}", output[[0]]);
        } else if output.ndim() == 2 && output.shape()[0] > 0 && output.shape()[1] > 0 {
            println!("DEBUG: Matrix tensor first element: {}", output[[0, 0]]);
        } else {
            println!("DEBUG: Tensor cannot display elements (empty or higher dimension)");
        }

        // Append the output, preserving the exact shape
        ctx.append_output(output);
        Ok(())
    }

    fn grad<'a>(&self, _ctx: &mut crate::op::GradientContext<'a, 'a, T>) {}
}
