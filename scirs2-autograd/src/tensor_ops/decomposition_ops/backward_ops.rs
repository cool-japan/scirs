//! Backward-pass ops for gradient.rs dispatch.
//!
//! These ops are built by gradient.rs::compute_grad_for_input when it
//! encounters "Cholesky", "LUExtractL"/"LUExtractU", or "QRExtractQ"/"QRExtractR"
//! op names.  Each takes (original_input, upstream_gradient) as inputs and
//! computes the Murray/Townsend backward on demand.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::Float;
use scirs2_core::ndarray::{Array2, Ix2};

// ─────────────────────────────────────────────────────────────────────────────

/// Gradient op for Cholesky: computes dA from dL via Murray (2016).
/// Inputs: [original_input A, upstream_gradient dL]
pub(crate) struct CholeskyBackwardOp;

impl<F: Float> Op<F> for CholeskyBackwardOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        // Input 0: original matrix A (for shape only — L is recomputed here)
        // Input 1: upstream gradient dL
        let a_input = ctx.input(0);
        let dl_input = ctx.input(1);

        let a_2d = a_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("CholeskyBackward: A must be 2D".into()))?;
        let dl_2d = dl_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("CholeskyBackward: dL must be 2D".into()))?;

        let n = a_2d.nrows();

        // Re-run Cholesky to recover L (same algorithm as CholeskyOp::compute).
        let mut l = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    let mut sum = F::zero();
                    for k in 0..j {
                        sum += l[[j, k]] * l[[j, k]];
                    }
                    let diag_sq = a_2d[[j, j]] - sum;
                    if diag_sq <= F::zero() {
                        return Err(OpError::Other(
                            "CholeskyBackward: matrix not positive definite".into(),
                        ));
                    }
                    l[[j, j]] = diag_sq.sqrt();
                } else {
                    let mut sum = F::zero();
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }
                    let diag = l[[j, j]];
                    if diag == F::zero() {
                        l[[i, j]] = F::zero();
                    } else {
                        l[[i, j]] = (a_2d[[i, j]] - sum) / diag;
                    }
                }
            }
        }

        let grad_a =
            crate::tensor_ops::decomposition_backward::cholesky_backward(&l, &dl_2d.to_owned());
        ctx.append_output(grad_a.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Higher-order gradient not supported — pass None.
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Gradient op for LUExtract (L or U component): computes dA via Murray (2016).
/// `component`: 1 = L upstream gradient is live, 2 = U upstream gradient is live.
/// Inputs: [original_input A, upstream_gradient dComponent]
pub(crate) struct LUExtractBackwardOp {
    pub(crate) component: usize, // 1 for L, 2 for U
}

impl<F: Float> Op<F> for LUExtractBackwardOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_input = ctx.input(0);
        let dg_input = ctx.input(1);

        let a_2d = a_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("LUExtractBackward: A must be 2D".into()))?
            .to_owned();
        let dg_2d = dg_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("LUExtractBackward: dG must be 2D".into()))?
            .to_owned();

        let n = a_2d.nrows();

        // Re-run LU without pivoting to recover L, U.
        let p = Array2::<F>::eye(n);
        let mut l = Array2::<F>::eye(n);
        let mut u = a_2d.clone();
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
            1 => (dg_2d, zero),
            2 => (zero, dg_2d),
            _ => {
                return Err(OpError::IncompatibleShape(
                    "LUExtractBackward: invalid component".into(),
                ))
            }
        };

        let grad_a =
            crate::tensor_ops::decomposition_backward::lu_backward(&p, &l, &u, &grad_l, &grad_u);
        ctx.append_output(grad_a.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Gradient op for QRExtract (Q or R component): computes dA via Townsend/Murray.
/// `component`: 0 = Q upstream gradient is live, 1 = R upstream gradient is live.
/// Inputs: [original_input A, upstream_gradient dComponent]
pub(crate) struct QRExtractBackwardOp {
    pub(crate) component: usize, // 0 for Q, 1 for R
}

impl<F: Float> Op<F> for QRExtractBackwardOp {
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_input = ctx.input(0);
        let dg_input = ctx.input(1);

        let a_2d = a_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("QRExtractBackward: A must be 2D".into()))?
            .to_owned();
        let dg_2d = dg_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("QRExtractBackward: dG must be 2D".into()))?
            .to_owned();

        let m = a_2d.nrows();
        let n = a_2d.ncols();

        // Re-compute full QR (Q is m×m, R is m×n) for backward.
        let mut q_full = Array2::<F>::zeros((m, m));
        let mut r_full = Array2::<F>::zeros((m, n));

        // Augment: pad A with standard basis columns when m > n.
        let mut a_aug = Array2::<F>::zeros((m, m));
        for i in 0..m {
            for j in 0..n {
                a_aug[[i, j]] = a_2d[[i, j]];
            }
            if i >= n {
                a_aug[[i, i]] = F::one();
            }
        }
        for j in 0..m {
            for i in 0..m {
                q_full[[i, j]] = a_aug[[i, j]];
            }
            for i in 0..j {
                let mut dot = F::zero();
                for row in 0..m {
                    dot += q_full[[row, i]] * q_full[[row, j]];
                }
                for row in 0..m {
                    q_full[[row, j]] = q_full[[row, j]] - dot * q_full[[row, i]];
                }
            }
            let mut norm = F::zero();
            for row in 0..m {
                norm += q_full[[row, j]] * q_full[[row, j]];
            }
            norm = norm.sqrt();
            if norm > F::epsilon() {
                for row in 0..m {
                    q_full[[row, j]] /= norm;
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                let mut dot = F::zero();
                for row in 0..m {
                    dot += q_full[[row, i]] * a_2d[[row, j]];
                }
                r_full[[i, j]] = dot;
            }
        }

        // Embed thin upstream gradient into full shape.
        // component=0: dG is m×k (thin Q), embed in m×m zero matrix.
        // component=1: dG is k×n (thin R), embed in m×n zero matrix (top k rows).
        let grad_q_full;
        let grad_r_full;
        match self.component {
            0 => {
                let k = dg_2d.ncols();
                let mut gq = Array2::<F>::zeros((m, m));
                for i in 0..m {
                    for j in 0..k {
                        gq[[i, j]] = dg_2d[[i, j]];
                    }
                }
                grad_q_full = gq;
                grad_r_full = Array2::<F>::zeros((m, n));
            }
            1 => {
                let k = dg_2d.nrows();
                let mut gr = Array2::<F>::zeros((m, n));
                for i in 0..k {
                    for j in 0..n {
                        gr[[i, j]] = dg_2d[[i, j]];
                    }
                }
                grad_q_full = Array2::<F>::zeros((m, m));
                grad_r_full = gr;
            }
            _ => {
                return Err(OpError::IncompatibleShape(
                    "QRExtractBackward: invalid component".into(),
                ))
            }
        }

        let grad_a = crate::tensor_ops::decomposition_backward::qr_backward(
            &q_full,
            &r_full,
            &grad_q_full,
            &grad_r_full,
        );
        ctx.append_output(grad_a.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}
