//! `diffgeom::ricci` — Ricci curvature tensor.
//!
//! Computes the Ricci tensor from Christoffel symbols via the formula:
//!
//! `Rᵢⱼ = ∂ₖΓᵏᵢⱼ − ∂ⱼΓᵏᵢₖ + Γᵏₖₗ·Γˡᵢⱼ − Γᵏᵢₗ·Γˡⱼₖ`
//!
//! where repeated indices k and l are summed. This is the standard trace of
//! the Riemann curvature tensor: `Rᵢⱼ = Rᵏᵢₖⱼ`.
//!
//! The result is a `(0, 2)` tensor.
//!
//! **Performance note**: Intermediate expressions are NOT canonicalized between
//! terms to keep the symbolic tree manageable on 4D metrics (Schwarzschild etc.).
//! Each final Rᵢⱼ entry is simplified via canonicalize at the end.

use crate::eml::grad::grad;
use crate::eml::op::LoweredOp;
use ndarray::IxDyn;

use super::tensor::Tensor;

/// Build a simple sum tree over a vec of terms (left-fold, no canonicalization).
///
/// Used internally to avoid the overhead of balanced-tree canonicalization
/// on intermediate Ricci summands.
fn raw_sum(terms: Vec<LoweredOp>) -> LoweredOp {
    terms
        .into_iter()
        .reduce(|acc, x| LoweredOp::Add(Box::new(acc), Box::new(x)))
        .unwrap_or(LoweredOp::Const(0.0))
}

/// Compute the Ricci tensor from the Christoffel symbol tensor.
///
/// Returns a `(0, 2)` tensor `R` where `R.get(&[i, j]) = Rᵢⱼ`.
///
/// Formula:
/// `Rᵢⱼ = Σₖ ∂ₖΓᵏᵢⱼ − Σₖ ∂ⱼΓᵏᵢₖ + Σₖ Σₗ Γᵏₖₗ·Γˡᵢⱼ − Σₖ Σₗ Γᵏᵢₗ·Γˡⱼₖ`
pub fn ricci_tensor(christoffel_tensor: &Tensor, coords: &[usize]) -> Tensor {
    let dim = coords.len();
    let mut r = Tensor::zeros(0, 2, dim);

    for i in 0..dim {
        for j in 0..dim {
            let mut term1_parts = Vec::with_capacity(dim);
            let mut term2_parts = Vec::with_capacity(dim);
            let mut term3_parts = Vec::with_capacity(dim * dim);
            let mut term4_parts = Vec::with_capacity(dim * dim);

            for k in 0..dim {
                // Term1: ∂_{coords[k]} Γᵏᵢⱼ
                let gamma_kij = christoffel_tensor.get(&[k, i, j]).clone();
                let wrt_k = coords[k];
                let d_k_gamma_kij = grad(&gamma_kij, wrt_k);
                term1_parts.push(d_k_gamma_kij);

                // Term2: ∂_{coords[j]} Γᵏᵢₖ
                let gamma_kik = christoffel_tensor.get(&[k, i, k]).clone();
                let wrt_j = coords[j];
                let d_j_gamma_kik = grad(&gamma_kik, wrt_j);
                term2_parts.push(d_j_gamma_kik);

                // Terms 3 and 4: sum over l
                for l in 0..dim {
                    // Term3: Γᵏₖₗ · Γˡᵢⱼ
                    let gamma_kkl = christoffel_tensor.get(&[k, k, l]).clone();
                    let gamma_lij = christoffel_tensor.get(&[l, i, j]).clone();
                    term3_parts.push(LoweredOp::Mul(Box::new(gamma_kkl), Box::new(gamma_lij)));

                    // Term4: Γᵏᵢₗ · Γˡⱼₖ
                    let gamma_kil = christoffel_tensor.get(&[k, i, l]).clone();
                    let gamma_ljk = christoffel_tensor.get(&[l, j, k]).clone();
                    term4_parts.push(LoweredOp::Mul(Box::new(gamma_kil), Box::new(gamma_ljk)));
                }
            }

            let t1 = raw_sum(term1_parts);
            let t2 = raw_sum(term2_parts);
            let t3 = raw_sum(term3_parts);
            let t4 = raw_sum(term4_parts);

            // Rᵢⱼ = T1 - T2 + T3 - T4
            let r_ij = LoweredOp::Sub(
                Box::new(LoweredOp::Add(
                    Box::new(LoweredOp::Sub(Box::new(t1), Box::new(t2))),
                    Box::new(t3),
                )),
                Box::new(t4),
            );
            // Store WITHOUT canonicalization for performance;
            // callers evaluate numerically or canonicalize themselves
            r.components[IxDyn(&[i, j])] = r_ij;
        }
    }

    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffgeom::christoffel::christoffel;
    use crate::diffgeom::metric::Metric;
    use crate::eml::eval::{eval_real, EvalCtx};
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn flat_ricci_is_zero() {
        let mut g = ArrayD::from_elem(IxDyn(&[2, 2]), LoweredOp::Const(0.0));
        g[IxDyn(&[0, 0])] = LoweredOp::Const(1.0);
        g[IxDyn(&[1, 1])] = LoweredOp::Const(1.0);
        let metric = Metric::new(g, vec![0, 1]).expect("flat");
        let gamma = christoffel(&metric);
        let r = ricci_tensor(&gamma, &[0, 1]);
        let ctx = EvalCtx::new(&[1.0, 1.0]);
        for i in 0..2 {
            for j in 0..2 {
                let v = eval_real(r.get(&[i, j]), &ctx).expect("eval");
                assert!(
                    v.abs() < 1e-10,
                    "R[{i},{j}] = {v} for flat metric (expected 0)"
                );
            }
        }
    }
}
