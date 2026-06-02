//! `diffgeom::weyl` — Weyl conformal tensor C_{μνρσ}.
//!
//! The Weyl tensor is the traceless part of the Riemann tensor. It encodes
//! conformal curvature (the part of curvature not captured by the Ricci tensor).
//!
//! ## Formula (general dimension n)
//!
//! For n ≥ 4:
//!
//! ```text
//! C_{μνρσ} = R_{μνρσ}
//!            - 1/(n-2) * (g_{μρ}R_{νσ} - g_{μσ}R_{νρ} - g_{νρ}R_{μσ} + g_{νσ}R_{μρ})
//!            + 1/((n-1)(n-2)) * R * (g_{μρ}g_{νσ} - g_{μσ}g_{νρ})
//! ```
//!
//! For n < 4, the Weyl tensor is identically zero.
//!
//! ## Key properties
//!
//! - Trace-free: `Σ_μ g^{μρ} C_{μνρσ} = 0`
//! - Anti-symmetric: `C_{μνρσ} = -C_{νμρσ}` and `C_{μνρσ} = -C_{μνσρ}`
//! - Conformally flat in n=2,3 (identically zero)
//!
//! ## Storage convention
//!
//! Returns a `(0, 4)` tensor where `weyl.get(&[mu, nu, rho, sigma]) = C_{μνρσ}`.

use crate::eml::op::LoweredOp;
use ndarray::IxDyn;

use super::metric::Metric;
use super::tensor::Tensor;

/// Build a raw left-fold sum (no canonicalization).
fn raw_sum(terms: Vec<LoweredOp>) -> LoweredOp {
    terms
        .into_iter()
        .reduce(|acc, x| LoweredOp::Add(Box::new(acc), Box::new(x)))
        .unwrap_or(LoweredOp::Const(0.0))
}

/// Compute the Weyl conformal tensor `C_{μνρσ}` from pre-computed geometric objects.
///
/// ## Arguments
///
/// - `metric`: the Riemannian/pseudo-Riemannian metric (provides g_{ij} and g^{ij})
/// - `riemann_up`: the `(1, 3)` Riemann tensor `R^λ_{νρσ}` (first index contravariant)
/// - `ricci`: the `(0, 2)` Ricci tensor `R_{ij}`
/// - `ricci_scalar`: the scalar curvature `R = g^{ij} R_{ij}` as a symbolic expression
///
/// ## Returns
///
/// A `(0, 4)` tensor where `result.get(&[mu, nu, rho, sigma]) = C_{μνρσ}`.
///
/// ## Dimension handling
///
/// For `dim < 4`, the Weyl tensor is identically zero and a zero tensor is returned
/// immediately (the formula has a `1/(n-2)` singularity for `n=2,3`).
///
/// ## Formula
///
/// For n = dim ≥ 4:
///
/// ```text
/// C_{μνρσ} = R_{μνρσ}
///            - 1/(n-2) * (g_{μρ}R_{νσ} - g_{μσ}R_{νρ} - g_{νρ}R_{μσ} + g_{νσ}R_{μρ})
///            + 1/((n-1)(n-2)) * R * (g_{μρ}g_{νσ} - g_{μσ}g_{νρ})
/// ```
///
/// where `R_{μνρσ} = Σ_λ g_{μλ} R^λ_{νρσ}` is the fully covariant Riemann tensor.
#[allow(clippy::needless_range_loop)]
pub fn weyl_tensor(
    metric: &Metric,
    riemann_up: &Tensor,
    ricci: &Tensor,
    ricci_scalar: &LoweredOp,
) -> Tensor {
    let dim = metric.coords.len();

    // Weyl is identically zero for dim < 4.
    // The formula has 1/(n-2) which is singular for n=2,3.
    if dim < 4 {
        return Tensor::zeros(0, 4, dim);
    }

    let n = dim as f64;
    let inv_n_minus_2 = 1.0 / (n - 2.0);
    let inv_n_minus_1_times_n_minus_2 = 1.0 / ((n - 1.0) * (n - 2.0));

    // Step 1: lower the first index of riemann_up to get fully covariant R_{μνρσ}.
    // R_{μνρσ} = Σ_λ g_{μλ} R^λ_{νρσ}
    // riemann_up stores R^λ_{νρσ} at [λ, ν, ρ, σ].
    // We compute a (0,4) tensor r_cov stored at [mu, nu, rho, sigma].
    let mut r_cov = Tensor::zeros(0, 4, dim);
    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    let mut terms = Vec::with_capacity(dim);
                    for lam in 0..dim {
                        let g_mu_lam = metric.g.get(&[mu, lam]).clone();
                        let r_lam_nu_rho_sigma = riemann_up.get(&[lam, nu, rho, sigma]).clone();
                        terms.push(LoweredOp::Mul(
                            Box::new(g_mu_lam),
                            Box::new(r_lam_nu_rho_sigma),
                        ));
                    }
                    r_cov.components[IxDyn(&[mu, nu, rho, sigma])] = raw_sum(terms);
                }
            }
        }
    }

    // Step 2: compute C_{μνρσ} for each (μ, ν, ρ, σ).
    let mut weyl = Tensor::zeros(0, 4, dim);
    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    // Term A: R_{μνρσ} (fully covariant Riemann)
                    let term_a = r_cov.get(&[mu, nu, rho, sigma]).clone();

                    // Term B: -1/(n-2) * (g_{μρ}R_{νσ} - g_{μσ}R_{νρ} - g_{νρ}R_{μσ} + g_{νσ}R_{μρ})
                    let g_mu_rho = metric.g.get(&[mu, rho]).clone();
                    let g_mu_sigma = metric.g.get(&[mu, sigma]).clone();
                    let g_nu_rho = metric.g.get(&[nu, rho]).clone();
                    let g_nu_sigma = metric.g.get(&[nu, sigma]).clone();

                    let r_nu_sigma = ricci.get(&[nu, sigma]).clone();
                    let r_nu_rho = ricci.get(&[nu, rho]).clone();
                    let r_mu_sigma = ricci.get(&[mu, sigma]).clone();
                    let r_mu_rho = ricci.get(&[mu, rho]).clone();

                    // bracket = g_{μρ}R_{νσ} - g_{μσ}R_{νρ} - g_{νρ}R_{μσ} + g_{νσ}R_{μρ}
                    let p1 = LoweredOp::Mul(Box::new(g_mu_rho), Box::new(r_nu_sigma));
                    let p2 = LoweredOp::Neg(Box::new(LoweredOp::Mul(
                        Box::new(g_mu_sigma),
                        Box::new(r_nu_rho),
                    )));
                    let p3 = LoweredOp::Neg(Box::new(LoweredOp::Mul(
                        Box::new(g_nu_rho),
                        Box::new(r_mu_sigma),
                    )));
                    let p4 = LoweredOp::Mul(Box::new(g_nu_sigma), Box::new(r_mu_rho));
                    let bracket_b = raw_sum(vec![p1, p2, p3, p4]);

                    let term_b = LoweredOp::Neg(Box::new(LoweredOp::Mul(
                        Box::new(LoweredOp::Const(inv_n_minus_2)),
                        Box::new(bracket_b),
                    )));

                    // Term C: 1/((n-1)(n-2)) * R * (g_{μρ}g_{νσ} - g_{μσ}g_{νρ})
                    let g_mu_rho_c = metric.g.get(&[mu, rho]).clone();
                    let g_nu_sigma_c = metric.g.get(&[nu, sigma]).clone();
                    let g_mu_sigma_c = metric.g.get(&[mu, sigma]).clone();
                    let g_nu_rho_c = metric.g.get(&[nu, rho]).clone();

                    let q1 = LoweredOp::Mul(Box::new(g_mu_rho_c), Box::new(g_nu_sigma_c));
                    let q2 = LoweredOp::Neg(Box::new(LoweredOp::Mul(
                        Box::new(g_mu_sigma_c),
                        Box::new(g_nu_rho_c),
                    )));
                    let bracket_c = LoweredOp::Add(Box::new(q1), Box::new(q2));

                    let term_c = LoweredOp::Mul(
                        Box::new(LoweredOp::Mul(
                            Box::new(LoweredOp::Const(inv_n_minus_1_times_n_minus_2)),
                            Box::new(ricci_scalar.clone()),
                        )),
                        Box::new(bracket_c),
                    );

                    // C_{μνρσ} = A + B + C
                    let c_component = LoweredOp::Add(
                        Box::new(LoweredOp::Add(Box::new(term_a), Box::new(term_b))),
                        Box::new(term_c),
                    );
                    weyl.components[IxDyn(&[mu, nu, rho, sigma])] = c_component;
                }
            }
        }
    }

    weyl
}

/// Compute the Ricci scalar from metric and Ricci tensor.
///
/// `R = Σᵢⱼ g^{ij} R_{ij}`
///
/// This is a convenience helper for callers that need to pass the scalar to `weyl_tensor`.
pub fn compute_ricci_scalar(metric: &Metric, ricci: &Tensor) -> LoweredOp {
    let dim = metric.coords.len();
    let mut terms = Vec::with_capacity(dim * dim);
    for i in 0..dim {
        for j in 0..dim {
            let g_inv_ij = metric.g_inv.get(&[i, j]).clone();
            let r_ij = ricci.get(&[i, j]).clone();
            terms.push(LoweredOp::Mul(Box::new(g_inv_ij), Box::new(r_ij)));
        }
    }
    raw_sum(terms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffgeom::christoffel::christoffel;
    use crate::diffgeom::metric::Metric;
    use crate::diffgeom::ricci::ricci_tensor;
    use crate::diffgeom::riemann::riemann_tensor;
    use crate::eml::eval::{eval_real, EvalCtx};
    use ndarray::{ArrayD, IxDyn};

    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    fn zero_g(dim: usize) -> ArrayD<LoweredOp> {
        ArrayD::from_elem(IxDyn(&[dim, dim]), c(0.0))
    }

    fn eval_at(op: &LoweredOp, vals: &[f64]) -> f64 {
        eval_real(op, &EvalCtx::new(vals)).unwrap_or(f64::NAN)
    }

    #[test]
    fn weyl_flat_minkowski_is_zero() {
        // Flat Minkowski metric — Weyl must vanish
        let mut g = zero_g(4);
        g[IxDyn(&[0, 0])] = c(-1.0);
        g[IxDyn(&[1, 1])] = c(1.0);
        g[IxDyn(&[2, 2])] = c(1.0);
        g[IxDyn(&[3, 3])] = c(1.0);
        let metric = Metric::new(g, vec![0, 1, 2, 3]).expect("Minkowski");
        let gamma = christoffel(&metric);
        let riemann = riemann_tensor(&gamma, &[0, 1, 2, 3]);
        let ricci = ricci_tensor(&gamma, &[0, 1, 2, 3]);
        let r_scalar = compute_ricci_scalar(&metric, &ricci);
        let weyl = weyl_tensor(&metric, &riemann, &ricci, &r_scalar);

        let vals = [0.0_f64; 4];
        let ctx = EvalCtx::new(&vals);
        for mu in 0..4 {
            for nu in 0..4 {
                for rho in 0..4 {
                    for sigma in 0..4 {
                        let v = eval_real(weyl.get(&[mu, nu, rho, sigma]), &ctx).expect("eval");
                        assert!(
                            v.abs() < 1e-9,
                            "C[{mu},{nu},{rho},{sigma}] = {v} (flat, expected 0)"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn weyl_2d_is_zero_short_circuit() {
        // For dim=2, Weyl is identically zero (short-circuit path)
        let mut g = zero_g(2);
        g[IxDyn(&[0, 0])] = c(1.0);
        g[IxDyn(&[1, 1])] = c(1.0);
        let metric = Metric::new(g, vec![0, 1]).expect("flat 2D");
        let gamma = christoffel(&metric);
        let riemann = riemann_tensor(&gamma, &[0, 1]);
        let ricci = ricci_tensor(&gamma, &[0, 1]);
        let r_scalar = compute_ricci_scalar(&metric, &ricci);
        let weyl = weyl_tensor(&metric, &riemann, &ricci, &r_scalar);

        assert_eq!(weyl.dim, 2);
        // All 16 components should be Const(0.0)
        let ctx = EvalCtx::new(&[1.0, 1.0]);
        for mu in 0..2 {
            for nu in 0..2 {
                for rho in 0..2 {
                    for sigma in 0..2 {
                        let v = eval_real(weyl.get(&[mu, nu, rho, sigma]), &ctx).expect("eval");
                        assert!(
                            v.abs() < 1e-10,
                            "C[{mu},{nu},{rho},{sigma}] = {v} for dim=2 (expected 0)"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn compute_ricci_scalar_flat_is_zero() {
        let mut g = zero_g(2);
        g[IxDyn(&[0, 0])] = c(1.0);
        g[IxDyn(&[1, 1])] = c(1.0);
        let metric = Metric::new(g, vec![0, 1]).expect("flat");
        let gamma = christoffel(&metric);
        let ricci = ricci_tensor(&gamma, &[0, 1]);
        let scalar = compute_ricci_scalar(&metric, &ricci);
        let ctx = EvalCtx::new(&[1.0, 1.0]);
        let v = eval_real(&scalar, &ctx).expect("eval");
        assert!(v.abs() < 1e-10, "Ricci scalar for flat = {v} (expected 0)");
    }
}
