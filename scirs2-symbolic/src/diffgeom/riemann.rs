//! `diffgeom::riemann` — full Riemann curvature tensor R^μ_{νρσ}.
//!
//! The Riemann tensor is the fundamental object of Riemannian geometry.
//! It encodes all information about the curvature of a manifold.
//!
//! ## Formula
//!
//! ```text
//! R^μ_{νρσ} = ∂_ρ Γ^μ_{σν} - ∂_σ Γ^μ_{ρν} + Σ_λ Γ^μ_{ρλ} Γ^λ_{σν}
//!                                              - Σ_λ Γ^μ_{σλ} Γ^λ_{ρν}
//! ```
//!
//! ## Storage convention
//!
//! The result is a `(1, 3)` tensor where:
//! - `riemann.get(&[mu, nu, rho, sigma]) = R^μ_{νρσ}`
//! - Upper index μ first, then lower indices ν, ρ, σ
//!
//! ## Anti-symmetry
//!
//! `R^μ_{νρσ} = -R^μ_{νσρ}` (anti-symmetric in last two lower indices)
//!
//! ## Consistency with Ricci tensor
//!
//! `R_{νσ} = Σ_k R^k_{νkσ}` — the Ricci tensor is the trace of the Riemann tensor
//! on indices 0 and 2 (first upper and second lower).

use crate::eml::grad::grad;
use crate::eml::op::LoweredOp;
use ndarray::IxDyn;

use super::tensor::Tensor;

/// Build a raw left-fold sum of symbolic expressions (no canonicalization).
///
/// Avoids the overhead of canonicalization on intermediate summands when
/// assembling large tensors like Riemann (256 components for dim=4).
fn raw_sum(terms: Vec<LoweredOp>) -> LoweredOp {
    terms
        .into_iter()
        .reduce(|acc, x| LoweredOp::Add(Box::new(acc), Box::new(x)))
        .unwrap_or(LoweredOp::Const(0.0))
}

/// Compute the full Riemann curvature tensor `R^μ_{νρσ}` from Christoffel symbols.
///
/// Returns a `(1, 3)` tensor where `result.get(&[mu, nu, rho, sigma]) = R^μ_{νρσ}`.
///
/// ## Formula
///
/// ```text
/// R^μ_{νρσ} = ∂_ρ Γ^μ_{σν} - ∂_σ Γ^μ_{ρν} + Σ_λ Γ^μ_{ρλ} Γ^λ_{σν}
///                                              - Σ_λ Γ^μ_{σλ} Γ^λ_{ρν}
/// ```
///
/// ## Performance
///
/// Intermediate expressions are NOT canonicalized to keep the symbolic tree
/// manageable. For dim=4 there are 256 components, each a sum of O(dim) terms.
/// Call [`fn@crate::cas::canonicalize`] on individual components if needed.
///
/// ## Anti-symmetry
///
/// The anti-symmetry `R^μ_{νρσ} = -R^μ_{νσρ}` is enforced by computing all
/// 256 entries independently (which they must satisfy) — callers can verify
/// this numerically via `riemann_antisymmetry_last_two_indices`.
#[allow(clippy::needless_range_loop)]
pub fn riemann_tensor(christoffel_tensor: &Tensor, coords: &[usize]) -> Tensor {
    let dim = coords.len();
    // Result is (1, 3): R^μ_{νρσ} stored at [mu, nu, rho, sigma]
    let mut riemann = Tensor::zeros(1, 3, dim);

    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    // Term 1: ∂_ρ Γ^μ_{σν}
                    let gamma_mu_sigma_nu = christoffel_tensor.get(&[mu, sigma, nu]).clone();
                    let wrt_rho = coords[rho];
                    let t1 = grad(&gamma_mu_sigma_nu, wrt_rho);

                    // Term 2: -∂_σ Γ^μ_{ρν}
                    let gamma_mu_rho_nu = christoffel_tensor.get(&[mu, rho, nu]).clone();
                    let wrt_sigma = coords[sigma];
                    let d_sigma_gamma = grad(&gamma_mu_rho_nu, wrt_sigma);
                    let t2 = LoweredOp::Neg(Box::new(d_sigma_gamma));

                    // Term 3: +Σ_λ Γ^μ_{ρλ} Γ^λ_{σν}
                    let mut t3_parts = Vec::with_capacity(dim);
                    for lam in 0..dim {
                        let gamma_mu_rho_lam = christoffel_tensor.get(&[mu, rho, lam]).clone();
                        let gamma_lam_sigma_nu = christoffel_tensor.get(&[lam, sigma, nu]).clone();
                        t3_parts.push(LoweredOp::Mul(
                            Box::new(gamma_mu_rho_lam),
                            Box::new(gamma_lam_sigma_nu),
                        ));
                    }
                    let t3 = raw_sum(t3_parts);

                    // Term 4: -Σ_λ Γ^μ_{σλ} Γ^λ_{ρν}
                    let mut t4_parts = Vec::with_capacity(dim);
                    for lam in 0..dim {
                        let gamma_mu_sigma_lam = christoffel_tensor.get(&[mu, sigma, lam]).clone();
                        let gamma_lam_rho_nu = christoffel_tensor.get(&[lam, rho, nu]).clone();
                        t4_parts.push(LoweredOp::Mul(
                            Box::new(gamma_mu_sigma_lam),
                            Box::new(gamma_lam_rho_nu),
                        ));
                    }
                    let t4 = LoweredOp::Neg(Box::new(raw_sum(t4_parts)));

                    // R^μ_{νρσ} = t1 + t2 + t3 + t4
                    let r_component = LoweredOp::Add(
                        Box::new(LoweredOp::Add(
                            Box::new(LoweredOp::Add(Box::new(t1), Box::new(t2))),
                            Box::new(t3),
                        )),
                        Box::new(t4),
                    );
                    riemann.components[IxDyn(&[mu, nu, rho, sigma])] = r_component;
                }
            }
        }
    }

    riemann
}

/// Compute the Ricci tensor from the Riemann tensor by tracing on indices 0 and 2.
///
/// `R_{νσ} = Σ_k R^k_{νkσ}` — i.e., sum `riemann.get(&[k, nu, k, sigma])` over k.
///
/// This is a consistency-check function: the result should match `ricci_tensor`
/// computed directly from Christoffel symbols.
///
/// Returns a `(0, 2)` tensor where `result.get(&[nu, sigma]) = R_{νσ}`.
#[allow(clippy::needless_range_loop)]
pub fn ricci_from_riemann(riemann: &Tensor) -> Tensor {
    let dim = riemann.dim;
    let mut ricci = Tensor::zeros(0, 2, dim);

    for nu in 0..dim {
        for sigma in 0..dim {
            // R_{νσ} = Σ_k R^k_{νkσ}
            let mut terms = Vec::with_capacity(dim);
            for k in 0..dim {
                terms.push(riemann.get(&[k, nu, k, sigma]).clone());
            }
            ricci.components[IxDyn(&[nu, sigma])] = raw_sum(terms);
        }
    }

    ricci
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffgeom::christoffel::christoffel;
    use crate::diffgeom::metric::Metric;
    use crate::diffgeom::ricci::ricci_tensor;
    use crate::eml::eval::{eval_real, EvalCtx};
    use ndarray::{ArrayD, IxDyn};

    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    fn zero_g(dim: usize) -> ArrayD<LoweredOp> {
        ArrayD::from_elem(IxDyn(&[dim, dim]), c(0.0))
    }

    fn eval(op: &LoweredOp, vals: &[f64]) -> f64 {
        eval_real(op, &EvalCtx::new(vals)).unwrap_or(f64::NAN)
    }

    #[test]
    fn flat_riemann_is_zero() {
        let mut g = zero_g(2);
        g[IxDyn(&[0, 0])] = c(1.0);
        g[IxDyn(&[1, 1])] = c(1.0);
        let metric = Metric::new(g, vec![0, 1]).expect("flat");
        let gamma = christoffel(&metric);
        let r = riemann_tensor(&gamma, &[0, 1]);
        let ctx = EvalCtx::new(&[1.0, 1.0]);
        for mu in 0..2 {
            for nu in 0..2 {
                for rho in 0..2 {
                    for sigma in 0..2 {
                        let v = eval_real(r.get(&[mu, nu, rho, sigma]), &ctx).expect("eval");
                        assert!(
                            v.abs() < 1e-10,
                            "R^{mu}_{nu}{rho}{sigma} = {v} (flat, expected 0)"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ricci_from_riemann_agrees_with_direct_ricci() {
        // S² metric: g = [[1,0],[0,sin²θ]], Var(0)=θ, Var(1)=φ
        let mut g = zero_g(2);
        g[IxDyn(&[0, 0])] = c(1.0);
        g[IxDyn(&[1, 1])] =
            LoweredOp::Pow(Box::new(LoweredOp::Sin(Box::new(var(0)))), Box::new(c(2.0)));
        let metric = Metric::new(g, vec![0, 1]).expect("S2");
        let gamma = christoffel(&metric);

        let riemann = riemann_tensor(&gamma, &[0, 1]);
        let ricci_via_riemann = ricci_from_riemann(&riemann);
        let ricci_direct = ricci_tensor(&gamma, &[0, 1]);

        let vals = [std::f64::consts::PI / 3.0, 1.0];
        for i in 0..2 {
            for j in 0..2 {
                let via_r = eval(ricci_via_riemann.get(&[i, j]), &vals);
                let direct = eval(ricci_direct.get(&[i, j]), &vals);
                assert!(
                    (via_r - direct).abs() < 1e-7,
                    "R[{i},{j}]: ricci_from_riemann={via_r}, direct={direct}"
                );
            }
        }
    }
}
