use crate::Float;

// mutable op
pub(crate) struct SGDOp<F> {
    pub(crate) alpha: F,
}

// mutable op
pub(crate) struct MomentumSGDOp<T: Float> {
    pub(crate) lr: T,
    pub(crate) momentum: T,
}

impl<F: Float> crate::op::Op<F> for SGDOp<F> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> Result<(), crate::op::OpError> {
        // Clone the update to avoid borrow checking issues
        let update = ctx.input(1).to_owned();
        let mut var = ctx.input_mut(0);
        var.zip_mut_with(&update, move |l, &r| *l -= self.alpha * r);
        ctx.append_output(scirs2_core::ndarray::Array::zeros(vec![]).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

impl<T: Float> crate::op::Op<T> for MomentumSGDOp<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        // Clone the gradient to avoid borrow checking issues
        let grad = ctx.input(1).to_owned();

        // First update the velocity v
        let mut v = ctx.input_mut(2);
        v.zip_mut_with(&grad, move |v, &g| *v = *v * self.momentum - self.lr * g);

        // Clone v to avoid double mutable borrow
        let v_clone = v.to_owned();

        // Then update parameters
        let mut param = ctx.input_mut(0);
        param.zip_mut_with(&v_clone, move |p, &v| *p += v);

        ctx.append_output(scirs2_core::ndarray::Array::zeros(vec![]).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}
