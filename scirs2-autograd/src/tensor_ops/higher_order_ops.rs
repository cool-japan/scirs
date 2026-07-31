use crate::Float;
use crate::{op, NdArray, NdArrayView};
use std::marker::PhantomData;

pub(crate) struct MapOp<T: Float> {
    pub(crate) phantom: PhantomData<T>,
    pub(crate) f: fn(NdArrayView<T>) -> NdArray<T>,
}

impl<F: Float> op::Op<F> for MapOp<F> {
    fn compute(&self, ctx: &mut op::ComputeContext<F>) -> Result<(), op::OpError> {
        let f = self.f;
        let x = ctx.input(0);
        ctx.append_output(f(x));
        Ok(())
    }
    /// `map` applies a caller-supplied `fn(NdArrayView<T>) -> NdArray<T>` element block by
    /// element block. That function is opaque: the op holds no derivative for it and
    /// cannot obtain one, so there is no rule to implement here.
    ///
    /// Leaving the body empty would make the backward pass substitute a zero gradient —
    /// a definite, wrong claim that `d map(f, x) / dx = 0`.  Instead the gradient is a
    /// node that refuses to evaluate and says why, pointing at
    /// [`crate::custom_gradient::custom_unary_op`], which takes a backward closure
    /// alongside the forward one.
    fn grad<'a, 'g>(&self, ctx: &mut op::GradientContext<'a, 'g, F>) {
        crate::tensor_ops::matrix_calculus::append_unsupported_grad(
            ctx,
            "map(): the mapped function is an opaque `fn(NdArrayView) -> NdArray` with no \
             known derivative, so this op has no gradient. Use \
             `custom_unary_op(name, forward, backward, ..)` to supply one."
                .into(),
        );
    }
}
