use crate::Float;

pub(crate) struct AdaGradOp<F: Float> {
    pub(crate) lr: F,
}

impl<F: Float> crate::op::Op<F> for AdaGradOp<F> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<F>) -> Result<(), crate::op::OpError> {
        // Clone the gradient to avoid borrow issues
        let grad = ctx.input(1).to_owned();

        // Update the accumulated squared gradients
        let mut h = ctx.input_mut(2);
        h.zip_mut_with(&grad, |h, &g| *h += g * g);

        // Clone h to avoid borrow issues
        let h_clone = h.to_owned();

        // Update parameters
        let mut param = ctx.input_mut(0);
        let eps = F::from(1e-7).unwrap();
        scirs2_core::ndarray::Zip::from(&mut param)
            .and(&grad)
            .and(&h_clone)
            .for_each(move |p, &g, &h| *p -= self.lr * g / (h.sqrt() + eps));

        ctx.append_output(scirs2_core::ndarray::Array::zeros(vec![]).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
        ctx.append_input_grad(2, None);
    }
}
