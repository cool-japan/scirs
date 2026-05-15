//! `diffgeom::christoffel` — symbolic Christoffel symbols of the second kind.
//!
//! Computes `Γᵏᵢⱼ = ½ g^{kl}(∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})`
//! using symbolic differentiation via [`mod@crate::eml::grad`].
//!
//! The result is a `(1,2)` tensor: `christoffel[k][i][j] = Γᵏᵢⱼ`.
//!
//! Note: Christoffels are symmetric in the lower indices: `Γᵏᵢⱼ = Γᵏⱼᵢ`.

use crate::cas::canonicalize;
use crate::eml::grad::grad;
use crate::eml::op::LoweredOp;
use ndarray::IxDyn;

use super::contraction::sum_over_axis;
use super::metric::Metric;
use super::tensor::Tensor;

/// Compute the Christoffel symbols of the second kind for the given metric.
///
/// Returns a `(1, 2)` tensor `Γ` where `Γ.get(&[k, i, j]) = Γᵏᵢⱼ`.
///
/// Formula: `Γᵏᵢⱼ = ½ Σ_l g^{kl} (∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})`
#[allow(clippy::needless_range_loop)]
pub fn christoffel(metric: &Metric) -> Tensor {
    let dim = metric.coords.len();
    let mut gamma = Tensor::zeros(1, 2, dim);

    // Precompute partial[k][i][j] = ∂_{coords[k]} g_{ij}
    // partial[k] is indexed as partial[k][i][j]
    let mut partial = vec![vec![vec![LoweredOp::Const(0.0); dim]; dim]; dim];
    for k in 0..dim {
        let wrt = metric.coords[k];
        for i in 0..dim {
            for j in 0..dim {
                let g_ij = metric.g.get(&[i, j]).clone();
                partial[k][i][j] = grad(&g_ij, wrt);
            }
        }
    }

    // Compute Γᵏᵢⱼ for each (k, i, j)
    for k in 0..dim {
        for i in 0..dim {
            for j in 0..dim {
                // Sum over l: g^{kl} * (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
                let mut terms = Vec::with_capacity(dim);
                for l in 0..dim {
                    let g_kl = metric.g_inv.get(&[k, l]).clone();
                    // ∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij}
                    let bracket = LoweredOp::Sub(
                        Box::new(LoweredOp::Add(
                            Box::new(partial[i][j][l].clone()),
                            Box::new(partial[j][i][l].clone()),
                        )),
                        Box::new(partial[l][i][j].clone()),
                    );
                    let term = LoweredOp::Mul(Box::new(g_kl), Box::new(bracket));
                    terms.push(term);
                }
                let total = sum_over_axis(&terms);
                // Multiply by 1/2
                let gamma_kij = LoweredOp::Mul(Box::new(LoweredOp::Const(0.5)), Box::new(total));
                // Canonicalize each Christoffel symbol
                let gamma_kij_canon = canonicalize(&gamma_kij).into_op();
                gamma.set(&[k, i, j], gamma_kij_canon);
            }
        }
    }

    gamma
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};
    use ndarray::{ArrayD, IxDyn};

    fn flat_euclidean_2d() -> Metric {
        let mut g = ArrayD::from_elem(IxDyn(&[2, 2]), LoweredOp::Const(0.0));
        g[IxDyn(&[0, 0])] = LoweredOp::Const(1.0);
        g[IxDyn(&[1, 1])] = LoweredOp::Const(1.0);
        Metric::new(g, vec![0, 1]).expect("flat metric")
    }

    #[test]
    fn flat_euclidean_christoffels_all_zero() {
        let metric = flat_euclidean_2d();
        let gamma = christoffel(&metric);
        let ctx = EvalCtx::new(&[0.0, 0.0]);
        // All 8 components should be 0
        for k in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    let v = eval_real(gamma.get(&[k, i, j]), &ctx).expect("eval");
                    assert!(v.abs() < 1e-10, "Γ[{k},{i},{j}] = {v} (expected 0)");
                }
            }
        }
    }
}
