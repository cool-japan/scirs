use crate::ndarray_ext::{self, ArrayRng};
use crate::op;
use crate::Float;

pub struct StandardNormal<T: Float> {
    pub arr_rng: ArrayRng<T>,
}

impl<T: Float> StandardNormal<T> {
    pub fn new(arr_rng: ArrayRng<T>) -> Self {
        Self { arr_rng }
    }
}

pub struct StandardUniform<T: Float> {
    pub arr_rng: ArrayRng<T>,
}

impl<T: Float> StandardUniform<T> {
    pub fn new(arr_rng: ArrayRng<T>) -> Self {
        Self { arr_rng }
    }
}

pub struct RandomUniform<T: Float> {
    pub arr_rng: ArrayRng<T>,
    pub max: f64,
    pub min: f64,
}

impl<T: Float> RandomUniform<T> {
    pub fn new(arr_rng: ArrayRng<T>, min: f64, max: f64) -> Self {
        Self { arr_rng, max, min }
    }
}

pub struct RandomNormal<T: Float> {
    pub arr_rng: ArrayRng<T>,
    pub mean: f64,
    pub stddev: f64,
}

impl<T: Float> RandomNormal<T> {
    pub fn new(arr_rng: ArrayRng<T>, mean: f64, stddev: f64) -> Self {
        Self {
            arr_rng,
            mean,
            stddev,
        }
    }
}

pub struct Bernoulli<T: Float> {
    pub arr_rng: ArrayRng<T>,
    pub p: f64,
}

impl<T: Float> Bernoulli<T> {
    pub fn new(arr_rng: ArrayRng<T>, p: f64) -> Self {
        Self { arr_rng, p }
    }
}

pub struct Exponential<T: Float> {
    pub arr_rng: ArrayRng<T>,
    pub lambda: f64,
}

impl<T: Float> Exponential<T> {
    pub fn new(arr_rng: ArrayRng<T>, lambda: f64) -> Self {
        Self { arr_rng, lambda }
    }
}

pub struct LogNormal<T: Float> {
    pub arr_rng: ArrayRng<T>,
    pub mean: f64,
    pub stddev: f64,
}

impl<T: Float> LogNormal<T> {
    pub fn new(arr_rng: ArrayRng<T>, mean: f64, stddev: f64) -> Self {
        Self {
            arr_rng,
            mean,
            stddev,
        }
    }
}

pub struct Gamma<T: Float> {
    pub arr_rng: ArrayRng<T>,
    pub shape_param: f64,
    pub scale: f64,
}

impl<T: Float> Gamma<T> {
    pub fn new(arr_rng: ArrayRng<T>, shape_param: f64, scale: f64) -> Self {
        Self {
            arr_rng,
            shape_param,
            scale,
        }
    }
}

impl<T: Float> op::Op<T> for RandomNormal<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(0));
        // Clone the ArrayRng to get a mutable version
        let mut arr_rng = self.arr_rng.clone();
        ctx.append_output(arr_rng.normal(shape.as_slice(), self.mean, self.stddev));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for RandomUniform<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(0));
        // Clone the ArrayRng to get a mutable version
        let mut arr_rng = self.arr_rng.clone();
        ctx.append_output(arr_rng.uniform(shape.as_slice(), self.min, self.max));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for StandardNormal<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(0));
        // Clone the ArrayRng to get a mutable version
        let mut arr_rng = self.arr_rng.clone();
        ctx.append_output(arr_rng.standard_normal(shape.as_slice()));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for StandardUniform<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(0));
        // Clone the ArrayRng to get a mutable version
        let mut arr_rng = self.arr_rng.clone();
        ctx.append_output(arr_rng.standard_uniform(shape.as_slice()));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Bernoulli<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(0));
        // Clone the ArrayRng to get a mutable version
        let mut arr_rng = self.arr_rng.clone();
        ctx.append_output(arr_rng.bernoulli(shape.as_slice(), self.p));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Exponential<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(0));
        // Clone the ArrayRng to get a mutable version
        let mut arr_rng = self.arr_rng.clone();
        ctx.append_output(arr_rng.exponential(shape.as_slice(), self.lambda));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for LogNormal<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(0));
        // Clone the ArrayRng to get a mutable version
        let mut arr_rng = self.arr_rng.clone();
        ctx.append_output(arr_rng.log_normal(shape.as_slice(), self.mean, self.stddev));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

impl<T: Float> op::Op<T> for Gamma<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let shape = ndarray_ext::asshape(&ctx.input(0));
        // Clone the ArrayRng to get a mutable version
        let mut arr_rng = self.arr_rng.clone();
        ctx.append_output(arr_rng.gamma(shape.as_slice(), self.shape_param, self.scale));
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
    }
}

use crate::tensor_ops::*;

pub struct Dropout<F: Float> {
    pub arr_rng: ArrayRng<F>,
    pub dropout_ratio: F,
    pub train: bool,
}

impl<F: Float> op::Op<F> for Dropout<F> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> Result<(), crate::op::OpError> {
        let x = ctx.input(0);

        // Clone the ArrayRng to get a mutable version
        let mut arr_rng = self.arr_rng.clone();

        if self.train {
            let mask = arr_rng.bernoulli(
                x.shape(),
                (F::one() - self.dropout_ratio)
                    .to_f64()
                    .expect("Operation failed"),
            );

            // Calculate sum of mask elements
            let sum = mask.fold(F::zero(), |acc, &val| acc + val);

            // Create a new array with element-wise multiplication
            let result = x.mapv(|v| v * sum / (F::from(mask.len()).expect("Operation failed")));
            ctx.append_output(result);
            ctx.append_output(mask);
        } else {
            let coef = F::one() - self.dropout_ratio;
            ctx.append_output(x.mapv(move |x| x * coef));
        }
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        let gy = ctx.output_grad();
        let mask = nth_tensor(ctx.output(), 1);
        ctx.append_input_grad(0, Some(gy * mask));
    }
}
