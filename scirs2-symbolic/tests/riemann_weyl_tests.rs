//! Integration tests for `scirs2_symbolic::diffgeom` Riemann and Weyl tensors.
//!
//! Covers:
//! 1. Schwarzschild vacuum: Riemann → Ricci via `ricci_from_riemann` agrees with direct `ricci_tensor`
//! 2. Flat Minkowski: all 256 Riemann components vanish
//! 3. Anti-symmetry `R^μ_{νρσ} = -R^μ_{νσρ}` (last two lower indices) for S²
//! 4. Anti-symmetry `R_{μνρσ} = -R_{νμρσ}` (first pair) for S²
//! 5. First Bianchi identity `R^μ_{νρσ} + R^μ_{ρσν} + R^μ_{σνρ} = 0` for S²
//! 6. Weyl tensor for 2D (conformally flat) is identically zero
//! 7. Weyl tensor is trace-free: `Σ_μ g^{μρ} C_{μνρσ} = 0` for Schwarzschild
//! 8. Weyl anti-symmetry: `C_{μνρσ} = -C_{νμρσ}` for Schwarzschild
//! 9. Weyl anti-symmetry: `C_{μνρσ} = -C_{μνσρ}` for Schwarzschild
//! 10. Kretschmann scalar `K = R_{μνρσ} R^{μνρσ}` for S³ metric

use ndarray::{ArrayD, IxDyn};
use scirs2_symbolic::diffgeom::{
    christoffel, compute_ricci_scalar, ricci_from_riemann, ricci_tensor, riemann_tensor,
    weyl_tensor, Metric,
};
use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

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

/// Build Schwarzschild metric: Var(0)=r, Var(1)=θ, Var(2)=φ, Var(3)=t, Var(10)=rs
fn schwarzschild_metric() -> Metric {
    let r = || var(0);
    let theta = || var(1);
    let rs = || var(10);

    // f = 1 - rs/r
    let f = || {
        LoweredOp::Sub(
            Box::new(c(1.0)),
            Box::new(LoweredOp::Div(Box::new(rs()), Box::new(r()))),
        )
    };

    let mut g = zero_g(4);
    // g_rr = 1/f at [0,0]
    g[IxDyn(&[0, 0])] = LoweredOp::Div(Box::new(c(1.0)), Box::new(f()));
    // g_θθ = r² at [1,1]
    g[IxDyn(&[1, 1])] = LoweredOp::Pow(Box::new(r()), Box::new(c(2.0)));
    // g_φφ = r² sin²θ at [2,2]
    g[IxDyn(&[2, 2])] = LoweredOp::Mul(
        Box::new(LoweredOp::Pow(Box::new(r()), Box::new(c(2.0)))),
        Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Sin(Box::new(theta()))),
            Box::new(c(2.0)),
        )),
    );
    // g_tt = -(1 - rs/r) at [3,3]
    g[IxDyn(&[3, 3])] = LoweredOp::Neg(Box::new(f()));

    // coords: r=0, θ=1, φ=2, t=3
    Metric::new(g, vec![0, 1, 2, 3]).expect("Schwarzschild metric")
}

/// Evaluation point for Schwarzschild: rs=2, r=10, θ=π/2, φ=0, t=0
fn schw_vals() -> Vec<f64> {
    let mut vals = vec![0.0_f64; 11];
    vals[0] = 10.0; // r
    vals[1] = std::f64::consts::PI / 2.0; // θ
    vals[2] = 0.0; // φ
    vals[3] = 0.0; // t
    vals[10] = 2.0; // rs
    vals
}

/// Build S² metric: g = [[1,0],[0,sin²θ]], Var(0)=θ, Var(1)=φ
fn sphere_s2_metric() -> Metric {
    let mut g = zero_g(2);
    g[IxDyn(&[0, 0])] = c(1.0);
    g[IxDyn(&[1, 1])] =
        LoweredOp::Pow(Box::new(LoweredOp::Sin(Box::new(var(0)))), Box::new(c(2.0)));
    Metric::new(g, vec![0, 1]).expect("S2 metric")
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Schwarzschild Riemann → ricci_from_riemann agrees with ricci_tensor
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn riemann_tensor_schwarzschild_vacuum_ricci_agrees() {
    let metric = schwarzschild_metric();
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1, 2, 3]);
    let ricci_via_riemann = ricci_from_riemann(&riemann);
    let ricci_direct = ricci_tensor(&gamma, &[0, 1, 2, 3]);

    let vals = schw_vals();
    for i in 0..4 {
        for j in 0..4 {
            let via = eval(ricci_via_riemann.get(&[i, j]), &vals);
            let direct = eval(ricci_direct.get(&[i, j]), &vals);
            assert!(
                via.is_finite() && direct.is_finite(),
                "R[{i},{j}]: via={via}, direct={direct} — at least one is NaN/Inf"
            );
            assert!(
                (via - direct).abs() < 1e-8,
                "R[{i},{j}]: ricci_from_riemann={via}, ricci_tensor={direct} — diff > 1e-8"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Flat Minkowski Riemann is all zero (256 components)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn riemann_tensor_flat_minkowski_is_zero() {
    let mut g = zero_g(4);
    g[IxDyn(&[0, 0])] = c(-1.0);
    g[IxDyn(&[1, 1])] = c(1.0);
    g[IxDyn(&[2, 2])] = c(1.0);
    g[IxDyn(&[3, 3])] = c(1.0);
    let metric = Metric::new(g, vec![0, 1, 2, 3]).expect("Minkowski");
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1, 2, 3]);

    let vals = [1.0_f64, 1.0, 1.0, 1.0];
    let ctx = EvalCtx::new(&vals);
    for mu in 0..4 {
        for nu in 0..4 {
            for rho in 0..4 {
                for sigma in 0..4 {
                    let v = eval_real(riemann.get(&[mu, nu, rho, sigma]), &ctx)
                        .expect("eval Minkowski Riemann");
                    assert!(
                        v.abs() < 1e-10,
                        "R^{mu}_{nu}{rho}{sigma} = {v} (flat Minkowski, expected 0)"
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Anti-symmetry R^μ_{νρσ} = -R^μ_{νσρ} (last two lower indices) for S²
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn riemann_antisymmetry_last_two_indices() {
    let metric = sphere_s2_metric();
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1]);

    let vals = [std::f64::consts::PI / 3.0, 1.0];
    let dim = 2;
    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    let r_mnrs = eval(riemann.get(&[mu, nu, rho, sigma]), &vals);
                    let r_mnsr = eval(riemann.get(&[mu, nu, sigma, rho]), &vals);
                    let sum = r_mnrs + r_mnsr;
                    assert!(
                        sum.abs() < 1e-8,
                        "R^{mu}_{{{nu}{rho}{sigma}}} + R^{mu}_{{{nu}{sigma}{rho}}} = {sum} (expected 0, anti-symmetry)"
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Anti-symmetry R_{μνρσ} = -R_{νμρσ} (first pair) for S²
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn riemann_antisymmetry_first_pair() {
    let metric = sphere_s2_metric();
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1]);

    let vals = [std::f64::consts::PI / 3.0, 1.0];
    let dim = 2;

    // Lower the first index: R_{μνρσ} = Σ_λ g_{μλ} R^λ_{νρσ}
    let eval_r_cov = |mu: usize, nu: usize, rho: usize, sigma: usize| -> f64 {
        let mut sum = 0.0;
        for lam in 0..dim {
            let g_mu_lam = eval(metric.g.get(&[mu, lam]), &vals);
            let r_lam_nu_rho_sigma = eval(riemann.get(&[lam, nu, rho, sigma]), &vals);
            sum += g_mu_lam * r_lam_nu_rho_sigma;
        }
        sum
    };

    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    let r_mnrs = eval_r_cov(mu, nu, rho, sigma);
                    let r_nmrs = eval_r_cov(nu, mu, rho, sigma);
                    let sum = r_mnrs + r_nmrs;
                    assert!(
                        sum.abs() < 1e-8,
                        "R_{{{mu}{nu}{rho}{sigma}}} + R_{{{nu}{mu}{rho}{sigma}}} = {sum} (expected 0)"
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: First Bianchi identity R^μ_{νρσ} + R^μ_{ρσν} + R^μ_{σνρ} = 0 for S²
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn riemann_first_bianchi_identity() {
    let metric = sphere_s2_metric();
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1]);

    let vals = [std::f64::consts::PI / 3.0, 1.0];
    let dim = 2;
    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    let r1 = eval(riemann.get(&[mu, nu, rho, sigma]), &vals);
                    let r2 = eval(riemann.get(&[mu, rho, sigma, nu]), &vals);
                    let r3 = eval(riemann.get(&[mu, sigma, nu, rho]), &vals);
                    let bianchi_sum = r1 + r2 + r3;
                    assert!(
                        bianchi_sum.abs() < 1e-7,
                        "Bianchi: R^{mu}_{{{nu}{rho}{sigma}}} + R^{mu}_{{{rho}{sigma}{nu}}} + R^{mu}_{{{sigma}{nu}{rho}}} = {bianchi_sum} (expected 0)"
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Weyl tensor for conformally flat 2D metric is zero
//   Any 2D Riemannian manifold is conformally flat; Weyl ≡ 0 for dim < 4
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn weyl_tensor_conformally_flat_is_zero() {
    // Use S² (2D metric) — any 2D manifold has Weyl ≡ 0
    let metric = sphere_s2_metric();
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1]);
    let ricci = ricci_tensor(&gamma, &[0, 1]);
    let r_scalar = compute_ricci_scalar(&metric, &ricci);
    let weyl = weyl_tensor(&metric, &riemann, &ricci, &r_scalar);

    assert_eq!(weyl.dim, 2);
    assert_eq!(weyl.rank_up, 0);
    assert_eq!(weyl.rank_down, 4);

    let vals = [std::f64::consts::PI / 3.0, 1.0];
    let ctx = EvalCtx::new(&vals);
    let dim = 2;
    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    let v = eval_real(weyl.get(&[mu, nu, rho, sigma]), &ctx).expect("eval 2D Weyl");
                    assert!(
                        v.abs() < 1e-10,
                        "C[{mu},{nu},{rho},{sigma}] = {v} for 2D (expected 0, Weyl ≡ 0)"
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Weyl tensor trace vanishes — Σ_{μρ} g^{μρ} C_{μνρσ} = 0
//   For Schwarzschild at sample point
//
// The trace-free condition in full Einstein notation:
//   g^{μρ} C_{μνρσ} = 0  means summing over BOTH μ and ρ simultaneously.
//   This collapses the 4-index contraction to R_{νσ} - R_{νσ} - ... = 0.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn weyl_tensor_trace_vanishes() {
    let metric = schwarzschild_metric();
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1, 2, 3]);
    let ricci = ricci_tensor(&gamma, &[0, 1, 2, 3]);
    let r_scalar = compute_ricci_scalar(&metric, &ricci);
    let weyl = weyl_tensor(&metric, &riemann, &ricci, &r_scalar);

    let vals = schw_vals();
    let dim = 4;

    // The trace-free condition: Σ_{μ,ρ} g^{μρ} C_{μνρσ} = 0 for all free ν, σ
    // We sum over BOTH μ and ρ (both are contracted with g^{μρ}).
    // Vacuum Schwarzschild: Ricci = 0, so Weyl = Riemann.
    // The trace Σ_{μρ} g^{μρ} R_{μνρσ} = g^{μρ} Σ_λ g_{μλ} R^λ_{νρσ}
    //   = δ^ρ_λ R^λ_{νρσ} = R^ρ_{νρσ} = R_{νσ} ≈ 0 (vacuum).
    for nu in 0..dim {
        for sigma in 0..dim {
            let mut trace_sum = 0.0;
            for mu in 0..dim {
                for rho in 0..dim {
                    let g_inv_mu_rho = eval(metric.g_inv.get(&[mu, rho]), &vals);
                    let c_mu_nu_rho_sigma = eval(weyl.get(&[mu, nu, rho, sigma]), &vals);
                    trace_sum += g_inv_mu_rho * c_mu_nu_rho_sigma;
                }
            }
            assert!(
                trace_sum.is_finite() && trace_sum.abs() < 1e-5,
                "Weyl trace Σ_{{μρ}} g^{{μρ}} C_{{μ{nu}ρ{sigma}}} = {trace_sum} (expected 0)"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: Weyl anti-symmetry C_{μνρσ} = -C_{νμρσ} for Schwarzschild
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn weyl_antisymmetry_pair1() {
    let metric = schwarzschild_metric();
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1, 2, 3]);
    let ricci = ricci_tensor(&gamma, &[0, 1, 2, 3]);
    let r_scalar = compute_ricci_scalar(&metric, &ricci);
    let weyl = weyl_tensor(&metric, &riemann, &ricci, &r_scalar);

    let vals = schw_vals();
    let dim = 4;
    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    let c_mn = eval(weyl.get(&[mu, nu, rho, sigma]), &vals);
                    let c_nm = eval(weyl.get(&[nu, mu, rho, sigma]), &vals);
                    let sum = c_mn + c_nm;
                    assert!(
                        sum.is_finite() && sum.abs() < 1e-5,
                        "C[{mu},{nu},{rho},{sigma}] + C[{nu},{mu},{rho},{sigma}] = {sum} (expected 0, anti-symmetry)"
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: Weyl anti-symmetry C_{μνρσ} = -C_{μνσρ} for Schwarzschild
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn weyl_antisymmetry_pair2() {
    let metric = schwarzschild_metric();
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1, 2, 3]);
    let ricci = ricci_tensor(&gamma, &[0, 1, 2, 3]);
    let r_scalar = compute_ricci_scalar(&metric, &ricci);
    let weyl = weyl_tensor(&metric, &riemann, &ricci, &r_scalar);

    let vals = schw_vals();
    let dim = 4;
    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    let c_mnrs = eval(weyl.get(&[mu, nu, rho, sigma]), &vals);
                    let c_mnsr = eval(weyl.get(&[mu, nu, sigma, rho]), &vals);
                    let sum = c_mnrs + c_mnsr;
                    assert!(
                        sum.is_finite() && sum.abs() < 1e-5,
                        "C[{mu},{nu},{rho},{sigma}] + C[{mu},{nu},{sigma},{rho}] = {sum} (expected 0)"
                    );
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: Kretschmann scalar K = R_{μνρσ} R^{μνρσ} for S³ metric
//
// S³ metric (round 3-sphere of radius κ=1):
//   g_{11} = 1/(1-r²), g_{22} = r², g_{33} = r² sin²θ
//   Var(0)=r, Var(1)=θ, Var(2)=φ
//
// For a maximally symmetric space of curvature k=1/κ², we expect:
//   K = R_{μνρσ} R^{μνρσ} = 2n(n-1) k²
// For n=3 and k=1: K = 2*3*2*1 = 12
//
// We compute K numerically and check it equals the expected value.
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::needless_range_loop)]
#[test]
fn riemann_dim3_sphere_kretschmann_scalar() {
    // S³ metric in "geographic" coordinates:
    // g_rr = 1/(1-r²), g_θθ = r², g_φφ = r² sin²θ
    // This is the standard round 3-sphere metric of radius 1
    // Var(0)=r, Var(1)=θ, Var(2)=φ
    let r = || var(0);
    let theta = || var(1);

    let mut g = zero_g(3);
    // g_rr = 1/(1-r²)
    g[IxDyn(&[0, 0])] = LoweredOp::Div(
        Box::new(c(1.0)),
        Box::new(LoweredOp::Sub(
            Box::new(c(1.0)),
            Box::new(LoweredOp::Pow(Box::new(r()), Box::new(c(2.0)))),
        )),
    );
    // g_θθ = r²
    g[IxDyn(&[1, 1])] = LoweredOp::Pow(Box::new(r()), Box::new(c(2.0)));
    // g_φφ = r² sin²θ
    g[IxDyn(&[2, 2])] = LoweredOp::Mul(
        Box::new(LoweredOp::Pow(Box::new(r()), Box::new(c(2.0)))),
        Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Sin(Box::new(theta()))),
            Box::new(c(2.0)),
        )),
    );

    let metric = Metric::new(g, vec![0, 1, 2]).expect("S3 metric");
    let gamma = christoffel(&metric);
    let riemann = riemann_tensor(&gamma, &[0, 1, 2]);

    // Evaluation point: r=0.5, θ=π/3, φ=0.5 (well inside S³, avoids coordinate singularity at r=1)
    let vals = [0.5_f64, std::f64::consts::PI / 3.0, 0.5];
    let dim = 3;

    // Compute fully covariant R_{μνρσ} = Σ_λ g_{μλ} R^λ_{νρσ} numerically
    let mut r_cov = [[[[0.0f64; 3]; 3]; 3]; 3];
    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    let mut sum = 0.0;
                    for lam in 0..dim {
                        let g_ml = eval(metric.g.get(&[mu, lam]), &vals);
                        let r_l = eval(riemann.get(&[lam, nu, rho, sigma]), &vals);
                        sum += g_ml * r_l;
                    }
                    r_cov[mu][nu][rho][sigma] = sum;
                }
            }
        }
    }

    // Compute R^{μνρσ} = g^{μα} g^{νβ} g^{ργ} g^{σδ} R_{αβγδ} numerically
    // We'll compute K = R_{μνρσ} R^{μνρσ} = Σ_{all} R_{μνρσ} * g^{μα} g^{νβ} g^{ργ} g^{σδ} R_{αβγδ}
    // Equivalently: K = Σ_{all} R_{cov}[mu,nu,rho,sigma] * R_up[mu,nu,rho,sigma]
    // where R_up[mu,nu,rho,sigma] = Σ_α Σ_β Σ_γ Σ_δ g^{μα} g^{νβ} g^{ργ} g^{σδ} R_{cov}[α,β,γ,δ]

    // Precompute g^{ij} numerically
    let mut g_inv_num = [[0.0f64; 3]; 3];
    for i in 0..dim {
        for j in 0..dim {
            g_inv_num[i][j] = eval(metric.g_inv.get(&[i, j]), &vals);
        }
    }

    // Compute K = R_{αβγδ} R^{αβγδ}
    // R^{μνρσ} = Σ_{α,β,γ,δ} g^{μα} g^{νβ} g^{ργ} g^{σδ} R_{αβγδ}
    let mut kretschmann = 0.0f64;
    for mu in 0..dim {
        for nu in 0..dim {
            for rho in 0..dim {
                for sigma in 0..dim {
                    // R^{μνρσ}
                    let mut r_up_mnrs = 0.0;
                    for alpha in 0..dim {
                        for beta in 0..dim {
                            for gamma in 0..dim {
                                for delta in 0..dim {
                                    r_up_mnrs += g_inv_num[mu][alpha]
                                        * g_inv_num[nu][beta]
                                        * g_inv_num[rho][gamma]
                                        * g_inv_num[sigma][delta]
                                        * r_cov[alpha][beta][gamma][delta];
                                }
                            }
                        }
                    }
                    kretschmann += r_cov[mu][nu][rho][sigma] * r_up_mnrs;
                }
            }
        }
    }

    // For S³ of radius 1 (k=1) with n=3 dimensions:
    // K = 2 * n * (n-1) * k² = 2 * 3 * 2 * 1 = 12
    let expected_k = 12.0;
    assert!(
        kretschmann.is_finite(),
        "Kretschmann scalar is not finite: {kretschmann}"
    );
    assert!(
        (kretschmann - expected_k).abs() < 1e-4,
        "Kretschmann scalar K = {kretschmann} (expected {expected_k} for S³ of radius 1)"
    );
}
