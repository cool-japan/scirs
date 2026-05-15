//! `diffgeom::einstein` — Einstein tensor.
//!
//! Computes `Gᵢⱼ = Rᵢⱼ − ½ g_{ij} R`
//! where `R = g^{ij} Rᵢⱼ` is the Ricci scalar.
//!
//! The result is a `(0, 2)` tensor.

use crate::eml::op::LoweredOp;
use ndarray::IxDyn;

use super::metric::Metric;
use super::tensor::Tensor;

/// Compute the Einstein tensor `G = Ric - ½ g R`.
///
/// Returns a `(0, 2)` tensor `G` where `G.get(&[i, j]) = Gᵢⱼ`.
pub fn einstein_tensor(metric: &Metric, ricci: &Tensor) -> Tensor {
    let dim = metric.coords.len();
    let mut g_tensor = Tensor::zeros(0, 2, dim);

    // Compute Ricci scalar: R = Σᵢⱼ g^{ij} R_{ij}
    let mut ricci_scalar_terms = Vec::with_capacity(dim * dim);
    for i in 0..dim {
        for j in 0..dim {
            let g_inv_ij = metric.g_inv.get(&[i, j]).clone();
            let r_ij = ricci.get(&[i, j]).clone();
            ricci_scalar_terms.push(LoweredOp::Mul(Box::new(g_inv_ij), Box::new(r_ij)));
        }
    }
    let ricci_scalar = ricci_scalar_terms
        .into_iter()
        .reduce(|acc, x| LoweredOp::Add(Box::new(acc), Box::new(x)))
        .unwrap_or(LoweredOp::Const(0.0));

    // Compute Gᵢⱼ = Rᵢⱼ - ½ g_{ij} R
    for i in 0..dim {
        for j in 0..dim {
            let r_ij = ricci.get(&[i, j]).clone();
            let g_ij = metric.g.get(&[i, j]).clone();
            let half_g_r = LoweredOp::Mul(
                Box::new(LoweredOp::Mul(
                    Box::new(LoweredOp::Const(0.5)),
                    Box::new(g_ij),
                )),
                Box::new(ricci_scalar.clone()),
            );
            let g_ij_val = LoweredOp::Sub(Box::new(r_ij), Box::new(half_g_r));
            g_tensor.components[IxDyn(&[i, j])] = g_ij_val;
        }
    }

    g_tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffgeom::christoffel::christoffel;
    use crate::diffgeom::metric::Metric;
    use crate::diffgeom::ricci::ricci_tensor;
    use crate::eml::eval::{eval_real, EvalCtx};
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn flat_einstein_is_zero() {
        let mut g = ArrayD::from_elem(IxDyn(&[2, 2]), LoweredOp::Const(0.0));
        g[IxDyn(&[0, 0])] = LoweredOp::Const(1.0);
        g[IxDyn(&[1, 1])] = LoweredOp::Const(1.0);
        let metric = Metric::new(g, vec![0, 1]).expect("flat");
        let gamma = christoffel(&metric);
        let r = ricci_tensor(&gamma, &[0, 1]);
        let g_tensor = einstein_tensor(&metric, &r);
        let ctx = EvalCtx::new(&[1.0, 1.0]);
        for i in 0..2 {
            for j in 0..2 {
                let v = eval_real(g_tensor.get(&[i, j]), &ctx).expect("eval");
                assert!(v.abs() < 1e-10, "G[{i},{j}] = {v} for flat (expected 0)");
            }
        }
    }
}
