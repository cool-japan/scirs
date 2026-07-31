//! LU decomposition operations — LUOp and LUExtractOp.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::tensor_ops::decomposition_backward::lu_backward;
use crate::Float;
use scirs2_core::ndarray::{Array2, Ix2};

/// LU Decomposition Operation
pub struct LUOp;

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for LUOp {
    fn name(&self) -> &'static str {
        "LU"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 {
            return Err(OpError::IncompatibleShape(format!(
                "LU decomposition requires 2D matrix, got shape {shape:?}"
            )));
        }

        let n = shape[0];
        let m = shape[1];

        if n != m {
            return Err(OpError::IncompatibleShape(
                "LU decomposition requires square matrix".into(),
            ));
        }

        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Compute LU decomposition with partial pivoting
        let mut l = Array2::<F>::eye(n);
        let mut u = input_2d.to_owned();
        let mut p = Array2::<F>::eye(n); // Permutation matrix

        // Gaussian elimination with partial pivoting
        for k in 0..n - 1 {
            // Find pivot
            let mut max_val = u[[k, k]].abs();
            let mut max_row = k;

            for i in (k + 1)..n {
                if u[[i, k]].abs() > max_val {
                    max_val = u[[i, k]].abs();
                    max_row = i;
                }
            }

            // Swap rows if needed
            if max_row != k {
                // Swap rows in U
                for j in 0..m {
                    let temp = u[[k, j]];
                    u[[k, j]] = u[[max_row, j]];
                    u[[max_row, j]] = temp;
                }

                // Swap rows in P
                for j in 0..n {
                    let temp = p[[k, j]];
                    p[[k, j]] = p[[max_row, j]];
                    p[[max_row, j]] = temp;
                }

                // Swap rows in L (only the computed part)
                for j in 0..k {
                    let temp = l[[k, j]];
                    l[[k, j]] = l[[max_row, j]];
                    l[[max_row, j]] = temp;
                }
            }

            // Elimination
            if u[[k, k]].abs() > F::epsilon() {
                for i in (k + 1)..n {
                    l[[i, k]] = u[[i, k]] / u[[k, k]];
                    for j in k..m {
                        u[[i, j]] = u[[i, j]] - l[[i, k]] * u[[k, j]];
                    }
                }
            }
        }

        // Zero out lower triangular part of U
        for i in 0..n {
            for j in 0..i {
                u[[i, j]] = F::zero();
            }
        }

        // Append outputs: P, L, U
        ctx.append_output(p.into_dyn());
        ctx.append_output(l.into_dyn());
        ctx.append_output(u.into_dyn());

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // LU decomposition gradient is complex, using simplified version
        ctx.append_input_grad(0, None);
    }
}

/// LU component extraction operators
pub struct LUExtractOp {
    pub component: usize, // 0 for P, 1 for L, 2 for U
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for LUExtractOp {
    fn name(&self) -> &'static str {
        match self.component {
            0 => "LUExtractP",
            1 => "LUExtractL",
            2 => "LUExtractU",
            _ => "LUExtractUnknown",
        }
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        let shape = input.shape();

        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(OpError::IncompatibleShape(
                "LU requires square matrix".into(),
            ));
        }

        let n = shape[0];
        let input_2d = input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("Failed to convert to 2D array".into()))?;

        // Re-run the LU computation (no pivoting, so P = I)
        let mut l = Array2::<F>::eye(n);
        let mut u = input_2d.to_owned();
        let p = Array2::<F>::eye(n);

        // Simplified LU without pivoting for extraction
        for k in 0..n - 1 {
            if u[[k, k]].abs() > F::epsilon() {
                for i in (k + 1)..n {
                    l[[i, k]] = u[[i, k]] / u[[k, k]];
                    for j in k..n {
                        u[[i, j]] = u[[i, j]] - l[[i, k]] * u[[k, j]];
                    }
                }
            }
        }

        // Zero out lower triangular part of U
        for i in 0..n {
            for j in 0..i {
                u[[i, j]] = F::zero();
            }
        }

        match self.component {
            0 => ctx.append_output(p.into_dyn()),
            1 => ctx.append_output(l.into_dyn()),
            2 => ctx.append_output(u.into_dyn()),
            _ => return Err(OpError::IncompatibleShape("Invalid component index".into())),
        }

        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Murray (2016) LU backward, exploiting linearity:
        //   LUExtractOp{1} (L component): grad_a = lu_backward(p, l, u, gL, 0)
        //   LUExtractOp{2} (U component): grad_a = lu_backward(p, l, u, 0, gU)
        //   LUExtractOp{0} (P component): no gradient through permutation
        //
        // When autograd sums contributions from all three ExtractOps the total
        // gradient w.r.t. A equals lu_backward(p, l, u, gL, gU).

        // P carries no smooth gradient.
        if self.component == 0 {
            ctx.append_input_grad(0, None);
            return;
        }

        // NOTE: this must NOT eagerly `.eval()` `input`/`gy` here (a previous version
        // did). `Op::grad` only ever has a bare `&Graph` (`ctx.graph()`), never the
        // `Context`/`VariableEnvironment` that resolves `Variable` nodes, so evaluating a
        // tensor that traces back to a `Variable` from here fails honestly instead of
        // fabricating a value -- which silently turned into a *shape* bug: the failed
        // eval fell through to `ctx.append_input_grad(0, None)`, and `tensor_ops::grad`'s
        // "no gradient accumulated" fallback then invented a zero gradient from
        // `Tensor::shape()`'s (empty, since it is never set for this node) hint, i.e. a
        // 0-d "gradient" in place of an n×n matrix. Building a lazy `LUExtractGradOp`
        // instead defers the re-derivation + `lu_backward` call to normal (non-eager)
        // graph evaluation, exactly like every other backward op in this crate.
        let input = *ctx.input(0);
        let gy = *ctx.output_grad();
        let g = ctx.graph();
        let gx = Tensor::builder(g)
            .append_input(input, false)
            .append_input(gy, false)
            .build(LUExtractGradOp {
                component: self.component,
            });
        ctx.append_input_grad(0, Some(gx));
    }
}

/// Lazy backward for [`LUExtractOp`] (`component` 1 or 2): re-derives `(P, L, U)` from `A`
/// and applies Murray's LU backward rule. See `LUExtractOp::grad` for why this must be
/// computed lazily rather than eagerly inside `Op::grad`.
///
/// Inputs are `(A, dComponent)` where `dComponent` is the upstream cotangent of whichever
/// of L/U this instance's `component` names. Output is `dA`.
struct LUExtractGradOp {
    component: usize,
}

impl<F: Float + scirs2_core::ndarray::ScalarOperand> Op<F> for LUExtractGradOp {
    fn name(&self) -> &'static str {
        "LUExtractGrad"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input_2d = ctx
            .input(0)
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("LUExtractGrad: A must be 2D".into()))?
            .to_owned();
        let grad_2d = ctx
            .input(1)
            .into_dimensionality::<Ix2>()
            .map_err(|_| {
                OpError::IncompatibleShape("LUExtractGrad: upstream gradient must be 2D".into())
            })?
            .to_owned();

        let n = input_2d.nrows();

        // Re-run LU (no pivoting) to recover p, l, u.
        let p = Array2::<F>::eye(n);
        let mut l = Array2::<F>::eye(n);
        let mut u = input_2d.clone();
        for k in 0..n - 1 {
            if u[[k, k]].abs() > F::epsilon() {
                for i in (k + 1)..n {
                    l[[i, k]] = u[[i, k]] / u[[k, k]];
                    for j in k..n {
                        u[[i, j]] = u[[i, j]] - l[[i, k]] * u[[k, j]];
                    }
                }
            }
        }
        for i in 0..n {
            for j in 0..i {
                u[[i, j]] = F::zero();
            }
        }

        let zero = Array2::<F>::zeros((n, n));
        let (grad_l, grad_u) = match self.component {
            1 => (grad_2d, zero),
            2 => (zero, grad_2d),
            _ => {
                return Err(OpError::IncompatibleShape(
                    "LUExtractGrad: invalid component".into(),
                ))
            }
        };

        let grad_a = lu_backward(&p, &l, &u, &grad_l, &grad_u);
        ctx.append_output(grad_a.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        crate::tensor_ops::matrix_calculus::append_unsupported_grad(
            ctx,
            "LU: second-order differentiation is not implemented.".into(),
        );
    }
}

/// Compute LU decomposition of a square matrix
///
/// Returns (P, L, U) where PA = LU
/// - P is the permutation matrix
/// - L is lower triangular with ones on diagonal
/// - U is upper triangular
#[allow(dead_code)]
pub fn lu<'g, F: Float + scirs2_core::ndarray::ScalarOperand>(
    matrix: &Tensor<'g, F>,
) -> (Tensor<'g, F>, Tensor<'g, F>, Tensor<'g, F>) {
    let g = matrix.graph();

    let p = Tensor::builder(g)
        .append_input(matrix, false)
        .build(LUExtractOp { component: 0 });

    let l = Tensor::builder(g)
        .append_input(matrix, false)
        .build(LUExtractOp { component: 1 });

    let u = Tensor::builder(g)
        .append_input(matrix, false)
        .build(LUExtractOp { component: 2 });

    (p, l, u)
}
