use crate::op;
use crate::Float;
use std::marker::PhantomData;

pub(crate) struct HookOp<T: Float, H: crate::hooks::Hook<T>> {
    phantom: PhantomData<T>,
    pub hook: H,
}

impl<T: Float, H: crate::hooks::Hook<T>> HookOp<T, H> {
    #[inline]
    pub fn new(hook: H) -> Self {
        HookOp {
            phantom: PhantomData,
            hook,
        }
    }
}

impl<T: Float, H: crate::hooks::Hook<T> + 'static> op::Op<T> for HookOp<T, H> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let ret = ctx.input(0);
        self.hook.call(&ret);
        ctx.append_output(ret.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(0, Some(*ctx.output_grad()));
    }
}
