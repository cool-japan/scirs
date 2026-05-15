//! `diffgeom::covariant_derivative` — covariant derivative of a tensor.
//!
//! For a `(p, q)` tensor `T`, the covariant derivative `∇_μ T` is a `(p, q+1)` tensor:
//!
//! `(∇_μ T)^{a1..ap}_{b1..bq μ} =
//!    ∂_μ T^{a1..ap}_{b1..bq}
//!    + Σ_s Σ_c Γ^{a_s}_{μ,c} T^{a1..c..ap}_{b1..bq}   (upper index correction)
//!      − Σ_s Σ_c Γ^c_{μ,b_s} T^{a1..ap}_{b1..c..bq}     (lower index correction)`
//!
//! The new covariant index is placed last in the index list.

use crate::eml::grad::grad;
use crate::eml::op::LoweredOp;
use ndarray::IxDyn;

use super::tensor::Tensor;

/// Build a raw left-fold sum.
fn raw_sum_owned(terms: Vec<LoweredOp>) -> LoweredOp {
    terms
        .into_iter()
        .reduce(|acc, x| LoweredOp::Add(Box::new(acc), Box::new(x)))
        .unwrap_or(LoweredOp::Const(0.0))
}

/// Compute the covariant derivative of tensor `t` with respect to `coords[wrt_idx]`.
///
/// - `t`: the tensor to differentiate (any mixed valence `(p, q)`)
/// - `christoffel_tensor`: the `(1, 2)` Christoffel symbol tensor `Γᵏᵢⱼ`
/// - `coords`: variable indices for each coordinate direction
/// - `wrt_idx`: index into `coords` for the differentiation direction μ
///
/// Returns a `(p, q+1)` tensor where the new lower index (from differentiation) is placed last.
pub fn covariant_derivative(
    t: &Tensor,
    christoffel_tensor: &Tensor,
    coords: &[usize],
    wrt_idx: usize,
) -> Tensor {
    let dim = t.dim;
    let p = t.rank_up;
    let q = t.rank_down;
    let mu_var = coords[wrt_idx];

    // Result has rank (p, q+1): same upper indices, one more lower index
    let mut result = Tensor::zeros(p, q + 1, dim);

    // Iterate over all output multi-indices
    // Output shape: [dim; p + q + 1] with new lower index last
    let out_rank = p + q + 1;
    let total_out = if out_rank == 0 {
        1
    } else {
        dim.pow(out_rank as u32)
    };

    for flat_out in 0..total_out {
        // Decode flat_out into out_idx of length p + q + 1
        let mut out_idx = vec![0usize; out_rank];
        let mut tmp = flat_out;
        for pos in (0..out_rank).rev() {
            out_idx[pos] = tmp % dim;
            tmp /= dim;
        }

        // The last index of out_idx is the new differentiation index (mu position in output)
        // but since we fixed wrt_idx, the new index just tracks which output slot we're filling.
        // Actually the output tensor has wrt_idx baked in — the mu slot in output is the last
        // lower index (index p + q in the total index list, 0-based). We've set wrt_idx,
        // so we only compute the slice where out_idx[p + q] == wrt_idx.
        if out_idx[p + q] != wrt_idx {
            // This output position corresponds to a different covariant derivative direction;
            // skip (caller can call for each wrt_idx and assemble a full (p, q+1) tensor).
            // Since we're called for one specific wrt_idx, we just skip others.
            continue;
        }

        // The "base" multi-index into t is out_idx[0..p+q]
        let t_idx = &out_idx[..p + q];

        // 1. Partial derivative term: ∂_μ T^{a1..ap}_{b1..bq}
        let t_component = t.components[IxDyn(t_idx)].clone();
        let partial_term = grad(&t_component, mu_var);

        let mut correction_terms = Vec::new();

        // 2. Upper index corrections: +Σ_s Σ_c Γ^{a_s}_{μ,c} T^{a1..c..ap}_{b1..bq}
        for s in 0..p {
            for c in 0..dim {
                // Christoffel: Γ^{a_s}_{wrt_idx, c} = gamma[a_s][wrt_idx][c]
                let a_s = t_idx[s];
                let gamma_val = christoffel_tensor.get(&[a_s, wrt_idx, c]).clone();

                // Build the modified t_idx: replace position s with c
                let mut modified_idx: Vec<usize> = t_idx.to_vec();
                modified_idx[s] = c;
                let t_modified = t.components[IxDyn(&modified_idx)].clone();

                correction_terms.push(LoweredOp::Mul(Box::new(gamma_val), Box::new(t_modified)));
            }
        }

        // 3. Lower index corrections: −Σ_s Σ_c Γ^c_{μ,b_s} T^{a1..ap}_{b1..c..bq}
        for s in 0..q {
            for c in 0..dim {
                // b_s is the (p + s)-th index in t_idx
                let b_s = t_idx[p + s];
                // Christoffel: Γ^c_{wrt_idx, b_s} = gamma[c][wrt_idx][b_s]
                let gamma_val = christoffel_tensor.get(&[c, wrt_idx, b_s]).clone();

                // Build modified t_idx: replace lower position s with c
                let mut modified_idx: Vec<usize> = t_idx.to_vec();
                modified_idx[p + s] = c;
                let t_modified = t.components[IxDyn(&modified_idx)].clone();

                // Minus sign for lower corrections
                correction_terms.push(LoweredOp::Neg(Box::new(LoweredOp::Mul(
                    Box::new(gamma_val),
                    Box::new(t_modified),
                ))));
            }
        }

        let correction = raw_sum_owned(correction_terms);
        let total = LoweredOp::Add(Box::new(partial_term), Box::new(correction));
        result.components[IxDyn(&out_idx)] = total;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffgeom::christoffel::christoffel;
    use crate::diffgeom::metric::Metric;
    use crate::eml::eval::{eval_real, EvalCtx};
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn flat_covariant_derivative_of_const_tensor_is_zero() {
        // For a flat metric, covariant derivative = partial derivative.
        // For a constant tensor, all partials are 0.
        let mut g = ArrayD::from_elem(IxDyn(&[2, 2]), LoweredOp::Const(0.0));
        g[IxDyn(&[0, 0])] = LoweredOp::Const(1.0);
        g[IxDyn(&[1, 1])] = LoweredOp::Const(1.0);
        let metric = Metric::new(g.clone(), vec![0, 1]).expect("flat");
        let gamma = christoffel(&metric);

        // Create a (0,1) vector with constant components
        let mut v_tensor = Tensor::zeros(0, 1, 2);
        v_tensor.set(&[0], LoweredOp::Const(3.0));
        v_tensor.set(&[1], LoweredOp::Const(5.0));

        // Covariant derivative w.r.t. direction 0
        let dv = covariant_derivative(&v_tensor, &gamma, &[0, 1], 0);
        let ctx = EvalCtx::new(&[1.0, 1.0]);

        // ∇_0 v_0 = ∂_0(3) + Γ^c_{00} v_c = 0 + 0 = 0
        let v00 = eval_real(dv.get(&[0, 0]), &ctx).expect("eval");
        assert!(v00.abs() < 1e-10, "∇_0 v_0 = {v00} (expected 0)");
    }
}
