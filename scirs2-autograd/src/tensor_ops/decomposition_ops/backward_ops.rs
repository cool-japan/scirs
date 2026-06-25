//! Backward-pass ops for gradient.rs dispatch.
//!
//! These ops are built by gradient.rs::compute_grad_for_input when it
//! encounters "Cholesky", "LUExtractL"/"LUExtractU", or "QRExtractQ"/"QRExtractR"
//! op names.  Each takes (original_input, upstream_gradient) as inputs and
//! computes the Murray/Townsend backward on demand.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::Float;
use scirs2_core::ndarray::{Array2, Ix1, Ix2};

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

// ─────────────────────────────────────────────────────────────────────────────
// SVD backward op
// ─────────────────────────────────────────────────────────────────────────────

/// f64 one-sided Jacobi reduced SVD: `A = U·diag(s)·Vᵀ`.
///
/// Returns `(U m×k, s len-k descending, Vt k×n)` with `k = min(m,n)`.
/// Self-contained (no `FromPrimitive`/`ScalarOperand` bound) so it can be called
/// from the `Float`-only gradient dispatch path.
fn svd_jacobi_f64(a: &Array2<f64>) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    let mut w = a.clone(); // working copy; columns become U·diag(s)
    let mut v = Array2::<f64>::eye(n);

    let max_sweeps = 60;
    let eps = 1e-15;
    for _ in 0..max_sweeps {
        let mut converged = true;
        for i in 0..n {
            for j in (i + 1)..n {
                // 2×2 sub-Gram entries on columns i, j.
                let mut alpha = 0.0; // ‖w_i‖²
                let mut beta = 0.0; // ‖w_j‖²
                let mut gamma = 0.0; // w_iᵀ w_j
                for r in 0..m {
                    alpha += w[[r, i]] * w[[r, i]];
                    beta += w[[r, j]] * w[[r, j]];
                    gamma += w[[r, i]] * w[[r, j]];
                }
                if gamma.abs() <= eps * (alpha * beta).sqrt() {
                    continue;
                }
                converged = false;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                // Apply Jacobi rotation to columns i, j of w and v.
                for r in 0..m {
                    let wi = w[[r, i]];
                    let wj = w[[r, j]];
                    w[[r, i]] = c * wi - s * wj;
                    w[[r, j]] = s * wi + c * wj;
                }
                for r in 0..n {
                    let vi = v[[r, i]];
                    let vj = v[[r, j]];
                    v[[r, i]] = c * vi - s * vj;
                    v[[r, j]] = s * vi + c * vj;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Singular values = column norms of w; U columns = normalized w columns.
    let mut sigma = vec![0.0_f64; n];
    for jcol in 0..n {
        let mut nrm = 0.0;
        for r in 0..m {
            nrm += w[[r, jcol]] * w[[r, jcol]];
        }
        sigma[jcol] = nrm.sqrt();
    }
    // Sort columns by descending singular value.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&ia, &ib| sigma[ib].total_cmp(&sigma[ia]));

    let mut u = Array2::<f64>::zeros((m, k));
    let mut s_out = vec![0.0_f64; k];
    let mut vt = Array2::<f64>::zeros((k, n));
    for (newc, &oldc) in idx.iter().take(k).enumerate() {
        let sv = sigma[oldc];
        s_out[newc] = sv;
        if sv > 1e-300 {
            for r in 0..m {
                u[[r, newc]] = w[[r, oldc]] / sv;
            }
        }
        for r in 0..n {
            vt[[newc, r]] = v[[r, oldc]];
        }
    }
    (u, s_out, vt)
}

/// Backward op for SVD component extraction.
///
/// `component`: 0 = U, 1 = singular values (S), 2 = Vᵀ.
/// Inputs: `[original_input A, upstream_gradient dComponent]`.
///
/// Recomputes the reduced SVD of `A` in f64, places the upstream cotangent in
/// the corresponding slot, and applies the exact reduced-SVD VJP
/// ([`crate::tensor_ops::decomposition_backward::svd_backward`]).
///
/// Returns an honest `OpError` when the singular values are (near-)degenerate,
/// because the `dU`/`dV` part of the VJP is then ill-defined — emitting a loud
/// error instead of a fabricated gradient.
pub(crate) struct SVDBackwardOp {
    pub(crate) component: usize,
}

impl<F: Float> Op<F> for SVDBackwardOp {
    fn name(&self) -> &'static str {
        "SVDBackward"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let a_input = ctx.input(0);
        let dg_input = ctx.input(1);

        let a_2d = a_input
            .view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| OpError::IncompatibleShape("SVDBackward: A must be 2D".into()))?;

        let (m, n) = (a_2d.nrows(), a_2d.ncols());
        let k = m.min(n);

        // Convert A to f64 and recompute the reduced SVD.
        let mut a_f64 = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                a_f64[[i, j]] = a_2d[[i, j]].to_f64().unwrap_or(0.0);
            }
        }
        let (u, s, vt) = svd_jacobi_f64(&a_f64);

        // Degenerate singular values ⇒ dU/dV VJP undefined: raise honest error
        // unless only the singular-value gradient (component 1) is requested,
        // which remains well-defined under repeated singular values.
        if self.component != 1
            && crate::tensor_ops::decomposition_backward::svd_has_repeated_singular_values(&s)
        {
            return Err(OpError::Other(
                "SVD backward: repeated/degenerate singular values make the U/V \
                 gradient mathematically undefined; refusing to fabricate a gradient"
                    .into(),
            ));
        }

        // Build per-component cotangents (only the requested slot is nonzero).
        let mut grad_u = Array2::<f64>::zeros((m, k));
        let mut grad_s = vec![0.0_f64; k];
        let mut grad_vt = Array2::<f64>::zeros((k, n));

        match self.component {
            0 => {
                let dg = dg_input
                    .view()
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| OpError::IncompatibleShape("SVDBackward: dU must be 2D".into()))?;
                for i in 0..m.min(dg.nrows()) {
                    for j in 0..k.min(dg.ncols()) {
                        grad_u[[i, j]] = dg[[i, j]].to_f64().unwrap_or(0.0);
                    }
                }
            }
            1 => {
                // Singular-value gradient: dg is a length-k vector.
                let dg = dg_input
                    .view()
                    .into_dimensionality::<Ix1>()
                    .map_err(|_| OpError::IncompatibleShape("SVDBackward: dS must be 1D".into()))?;
                for j in 0..k.min(dg.len()) {
                    grad_s[j] = dg[j].to_f64().unwrap_or(0.0);
                }
            }
            2 => {
                let dg = dg_input.view().into_dimensionality::<Ix2>().map_err(|_| {
                    OpError::IncompatibleShape("SVDBackward: dVt must be 2D".into())
                })?;
                for i in 0..k.min(dg.nrows()) {
                    for j in 0..n.min(dg.ncols()) {
                        grad_vt[[i, j]] = dg[[i, j]].to_f64().unwrap_or(0.0);
                    }
                }
            }
            _ => {
                return Err(OpError::IncompatibleShape(
                    "SVDBackward: invalid component".into(),
                ))
            }
        }

        let da_f64 = crate::tensor_ops::decomposition_backward::svd_backward(
            &u, &s, &vt, &grad_u, &grad_s, &grad_vt,
        );

        // Convert dA back to F.
        let mut da = Array2::<F>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                da[[i, j]] = F::from(da_f64[[i, j]]).unwrap_or_else(F::zero);
            }
        }
        ctx.append_output(da.into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Second-order gradient unsupported.
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, None);
    }
}
