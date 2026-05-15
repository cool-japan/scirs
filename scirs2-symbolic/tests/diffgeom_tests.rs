//! Integration tests for `scirs2_symbolic::diffgeom`.
//!
//! Covers: flat metrics, polar metrics, sphere S², 4D Lorentzian inverse,
//! metric compatibility, and the Schwarzschild vacuum solution.

use ndarray::{ArrayD, IxDyn};
use scirs2_symbolic::diffgeom::{
    christoffel, covariant_derivative, einstein_tensor, ricci_tensor, Metric,
};
use scirs2_symbolic::eml::{eval_real, EvalCtx, LoweredOp};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}

fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}

/// Build a dim×dim zero ArrayD<LoweredOp>.
fn zero_g(dim: usize) -> ArrayD<LoweredOp> {
    ArrayD::from_elem(IxDyn(&[dim, dim]), c(0.0))
}

/// Evaluate a LoweredOp at a variable slice (returns NaN on error for test clarity).
fn eval(op: &LoweredOp, vals: &[f64]) -> f64 {
    eval_real(op, &EvalCtx::new(vals)).unwrap_or(f64::NAN)
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: 2D flat Euclidean — all Christoffels = 0
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_flat_euclidean_christoffels_all_zero() {
    let mut g = zero_g(2);
    g[IxDyn(&[0, 0])] = c(1.0);
    g[IxDyn(&[1, 1])] = c(1.0);
    let metric = Metric::new(g, vec![0, 1]).expect("flat 2D metric");
    let gamma = christoffel(&metric);
    // Evaluate at (x=1, y=1)
    let ctx = EvalCtx::new(&[1.0, 1.0]);
    for k in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let v = eval_real(gamma.get(&[k, i, j]), &ctx).expect("eval");
                assert!(
                    v.abs() < 1e-10,
                    "Γ[{k},{i},{j}] = {v} (expected 0 for flat metric)"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: 2D polar metric g = [[1,0],[0,r²]] — specific Christoffels
//   Var(0)=r, Var(1)=θ
//   Γ^r_{θθ} = -r, Γ^θ_{rθ} = Γ^θ_{θr} = 1/r, all others = 0
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_polar_metric_christoffels() {
    // g_rr=1, g_θθ=r², off-diagonal=0; Var(0)=r, Var(1)=θ
    let mut g = zero_g(2);
    g[IxDyn(&[0, 0])] = c(1.0);
    g[IxDyn(&[1, 1])] = LoweredOp::Pow(Box::new(var(0)), Box::new(c(2.0)));
    let metric = Metric::new(g, vec![0, 1]).expect("polar metric");
    let gamma = christoffel(&metric);

    // Evaluate at r=3, θ=0.5
    let r_val = 3.0_f64;
    let polar_vals = [r_val, 0.5_f64];
    let ctx = EvalCtx::new(&polar_vals);

    // Γ^r_{θθ} = gamma[0][1][1] = -r
    let g_r_tt = eval(gamma.get(&[0, 1, 1]), &[r_val, 0.5]);
    assert!(
        (g_r_tt - (-r_val)).abs() < 1e-8,
        "Γ^r_θθ = {g_r_tt} (expected {})",
        -r_val
    );

    // Gamma^theta_{r theta} = gamma[1][0][1] = 1/r
    let g_th_rt = eval(gamma.get(&[1, 0, 1]), &[r_val, 0.5]);
    assert!(
        (g_th_rt - 1.0 / r_val).abs() < 1e-8,
        "Gamma^th_rt = {} (expected {})",
        g_th_rt,
        1.0 / r_val
    );

    // Gamma^theta_{theta r} = gamma[1][1][0] = 1/r (symmetry)
    let g_th_tr = eval(gamma.get(&[1, 1, 0]), &[r_val, 0.5]);
    assert!(
        (g_th_tr - 1.0 / r_val).abs() < 1e-8,
        "Gamma^th_tr = {} (expected {})",
        g_th_tr,
        1.0 / r_val
    );

    // Gamma^r_{rr} = 0
    let g_r_rr = eval(gamma.get(&[0, 0, 0]), &[r_val, 0.5]);
    assert!(g_r_rr.abs() < 1e-8, "Gamma^r_rr = {} (expected 0)", g_r_rr);

    // Gamma^theta_{theta theta} = 0
    let _ = ctx;
    let g_th_tt = eval(gamma.get(&[1, 1, 1]), &[r_val, 0.5]);
    assert!(
        g_th_tt.abs() < 1e-8,
        "Gamma^th_tt = {} (expected 0)",
        g_th_tt
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: 4×4 Lorentzian (Minkowski) metric inverse
//   g = diag(-1, 1, 1, 1) → g_inv = diag(-1, 1, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_minkowski_4d_inverse() {
    let mut g = zero_g(4);
    g[IxDyn(&[0, 0])] = c(-1.0);
    g[IxDyn(&[1, 1])] = c(1.0);
    g[IxDyn(&[2, 2])] = c(1.0);
    g[IxDyn(&[3, 3])] = c(1.0);
    let metric = Metric::new(g, vec![0, 1, 2, 3]).expect("Minkowski metric");
    let ctx = EvalCtx::new(&[]);

    let v00 = eval_real(metric.g_inv.get(&[0, 0]), &ctx).expect("g_inv^00");
    let v11 = eval_real(metric.g_inv.get(&[1, 1]), &ctx).expect("g_inv^11");
    let v01 = eval_real(metric.g_inv.get(&[0, 1]), &ctx).expect("g_inv^01");

    assert!((v00 - (-1.0)).abs() < 1e-10, "g_inv^00 = {v00}");
    assert!((v11 - 1.0).abs() < 1e-10, "g_inv^11 = {v11}");
    assert!(v01.abs() < 1e-10, "g_inv^01 = {v01}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: g^{ij} g_{jk} = δ^i_k (2D flat — metric completeness)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metric_completeness_2d() {
    let mut g = zero_g(2);
    // Non-trivial diagonal metric: g = diag(4, 9)
    g[IxDyn(&[0, 0])] = c(4.0);
    g[IxDyn(&[1, 1])] = c(9.0);
    let metric = Metric::new(g, vec![0, 1]).expect("diag metric");
    let ctx = EvalCtx::new(&[]);

    // Compute g^{ik} g_{kj} for all i, j
    let dim = 2;
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0;
            for k in 0..dim {
                let g_inv_ik = eval_real(metric.g_inv.get(&[i, k]), &ctx).expect("g_inv");
                let g_kj = eval_real(metric.g.get(&[k, j]), &ctx).expect("g");
                sum += g_inv_ik * g_kj;
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-9,
                "g^{{i{i}k}} g_{{k{j}}} = {sum} (expected {expected})"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: 4×4 inverse (numerical) — diag(2,3,5,7) → diag(1/2,1/3,1/5,1/7)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_4x4_diagonal_inverse() {
    let diag_vals = [2.0, 3.0, 5.0, 7.0];
    let mut g = zero_g(4);
    for (i, &val) in diag_vals.iter().enumerate() {
        g[IxDyn(&[i, i])] = c(val);
    }
    let metric = Metric::new(g, vec![0, 1, 2, 3]).expect("diag 4D metric");
    let ctx = EvalCtx::new(&[]);
    for (i, &dv) in diag_vals.iter().enumerate() {
        let v = eval_real(metric.g_inv.get(&[i, i]), &ctx).expect("g_inv diag");
        let expected = 1.0 / dv;
        assert!(
            (v - expected).abs() < 1e-10,
            "g_inv[{i},{i}] = {v} (expected {expected})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Flat Ricci tensor is zero (2D)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_flat_ricci_is_zero() {
    let mut g = zero_g(2);
    g[IxDyn(&[0, 0])] = c(1.0);
    g[IxDyn(&[1, 1])] = c(1.0);
    let metric = Metric::new(g, vec![0, 1]).expect("flat");
    let gamma = christoffel(&metric);
    let r = ricci_tensor(&gamma, &[0, 1]);
    let ctx = EvalCtx::new(&[1.0, 1.0]);
    for i in 0..2 {
        for j in 0..2 {
            let v = eval(r.get(&[i, j]), &[1.0, 1.0]);
            let _ = ctx;
            assert!(v.abs() < 1e-10, "R[{i},{j}] = {v} (expected 0)");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Flat Einstein tensor is zero (2D)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_flat_einstein_is_zero() {
    let mut g = zero_g(2);
    g[IxDyn(&[0, 0])] = c(1.0);
    g[IxDyn(&[1, 1])] = c(1.0);
    let metric = Metric::new(g, vec![0, 1]).expect("flat");
    let gamma = christoffel(&metric);
    let r = ricci_tensor(&gamma, &[0, 1]);
    let g_tensor = einstein_tensor(&metric, &r);
    let ctx = EvalCtx::new(&[1.0, 1.0]);
    for i in 0..2 {
        for j in 0..2 {
            let v = eval_real(g_tensor.get(&[i, j]), &ctx).expect("eval");
            assert!(v.abs() < 1e-10, "G[{i},{j}] = {v} (expected 0)");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: Covariant derivative of constant in flat space = 0
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_covariant_derivative_const_flat() {
    let mut g = zero_g(2);
    g[IxDyn(&[0, 0])] = c(1.0);
    g[IxDyn(&[1, 1])] = c(1.0);
    let metric = Metric::new(g, vec![0, 1]).expect("flat");
    let gamma = christoffel(&metric);

    use scirs2_symbolic::diffgeom::tensor::Tensor;
    let mut v = Tensor::zeros(0, 1, 2);
    v.set(&[0], c(3.0));
    v.set(&[1], c(5.0));

    let dv = covariant_derivative(&v, &gamma, &[0, 1], 0);
    let ctx = EvalCtx::new(&[1.0, 1.0]);
    let v_00 = eval_real(dv.get(&[0, 0]), &ctx).expect("eval");
    assert!(v_00.abs() < 1e-10, "∇_0 v_0 = {v_00} (expected 0)");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: Polar metric Ricci (2D) — sanity, should be zero (flat R²)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_polar_metric_ricci_is_zero() {
    // Polar coordinates on flat R² — Ricci tensor must be zero
    let mut g = zero_g(2);
    g[IxDyn(&[0, 0])] = c(1.0);
    g[IxDyn(&[1, 1])] = LoweredOp::Pow(Box::new(var(0)), Box::new(c(2.0)));
    let metric = Metric::new(g, vec![0, 1]).expect("polar");
    let gamma = christoffel(&metric);
    let r = ricci_tensor(&gamma, &[0, 1]);

    // Evaluate at r=2, θ=0.5
    for i in 0..2 {
        for j in 0..2 {
            let v = eval(r.get(&[i, j]), &[2.0, 0.5]);
            assert!(
                v.abs() < 1e-8,
                "R_polar[{i},{j}] = {v} (flat R², must be 0)"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10: Sphere S² Ricci tensor check
//   g = [[1,0],[0,sin²θ]], Var(0)=θ, Var(1)=φ
//   R_θθ = 1, R_φφ = sin²θ (numerically), scalar R = 2
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_sphere_s2_ricci() {
    // g_θθ = 1, g_φφ = sin²θ
    let mut g = zero_g(2);
    g[IxDyn(&[0, 0])] = c(1.0);
    g[IxDyn(&[1, 1])] =
        LoweredOp::Pow(Box::new(LoweredOp::Sin(Box::new(var(0)))), Box::new(c(2.0)));
    let metric = Metric::new(g, vec![0, 1]).expect("S2 metric");
    let gamma = christoffel(&metric);
    let r_tensor = ricci_tensor(&gamma, &[0, 1]);

    // Evaluate at θ = π/3 (60°), φ = 1.0
    let theta = std::f64::consts::PI / 3.0;
    let phi = 1.0;
    let vals = [theta, phi];

    // R_θθ should be 1
    let r_tt = eval(r_tensor.get(&[0, 0]), &vals);
    assert!((r_tt - 1.0).abs() < 1e-7, "R_θθ = {r_tt} (expected 1.0)");

    // R_φφ should be sin²θ
    let expected_rpp = theta.sin().powi(2);
    let r_pp = eval(r_tensor.get(&[1, 1]), &vals);
    assert!(
        (r_pp - expected_rpp).abs() < 1e-7,
        "R_φφ = {r_pp} (expected sin²θ = {expected_rpp})"
    );

    // Off-diagonal R_θφ should be 0
    let r_tp = eval(r_tensor.get(&[0, 1]), &vals);
    assert!(r_tp.abs() < 1e-7, "R_θφ = {r_tp} (expected 0)");

    // Scalar curvature R = g^{ij} R_{ij}
    // g^{θθ} = 1, g^{φφ} = 1/sin²θ
    // R = 1 * R_θθ + (1/sin²θ) * R_φφ = 1*1 + (1/sin²θ)*sin²θ = 1 + 1 = 2
    let g_inv_tt = eval(metric.g_inv.get(&[0, 0]), &vals);
    let g_inv_pp = eval(metric.g_inv.get(&[1, 1]), &vals);
    let r_scalar = g_inv_tt * r_tt + g_inv_pp * r_pp;
    assert!(
        (r_scalar - 2.0).abs() < 1e-6,
        "S2 scalar curvature R = {r_scalar} (expected 2)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11: Schwarzschild vacuum — Rᵢⱼ = 0 (two evaluation points)
//
// g_tt = -(1 - rs/r)  = Var(3) [time component index in output]
// g_rr = 1/(1 - rs/r)
// g_θθ = r²
// g_φφ = r² sin²θ
//
// Variable layout: Var(0)=r, Var(1)=θ, Var(2)=φ, Var(3)=t_coord, Var(10)=rs
//
// We use coords=[0,1,2,3] (r,θ,φ,t as coordinate indices).
// ─────────────────────────────────────────────────────────────────────────────

fn schwarzschild_metric() -> Metric {
    // rs = Var(10), r = Var(0), θ = Var(1)
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
    // g_tt = -(1 - rs/r) at position [3,3] (t is last coordinate)
    g[IxDyn(&[3, 3])] = LoweredOp::Neg(Box::new(f()));
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

    // coords: r=0, θ=1, φ=2, t=3
    Metric::new(g, vec![0, 1, 2, 3]).expect("Schwarzschild metric")
}

#[test]
fn test_schwarzschild_ricci_vacuum_point1() {
    let metric = schwarzschild_metric();
    let gamma = christoffel(&metric);
    let r_tensor = ricci_tensor(&gamma, &[0, 1, 2, 3]);

    // Point 1: rs=2, r=10, θ=π/2, φ=0, t=0
    // Var layout: [0]=r=10, [1]=θ=π/2, [2]=φ=0, [3]=t=0, ..., [10]=rs=2
    let mut vals = vec![0.0_f64; 11];
    vals[0] = 10.0; // r
    vals[1] = std::f64::consts::PI / 2.0; // θ
    vals[2] = 0.0; // φ
    vals[3] = 0.0; // t
    vals[10] = 2.0; // rs

    for i in 0..4 {
        for j in 0..4 {
            let v = eval(r_tensor.get(&[i, j]), &vals);
            assert!(
                v.is_finite() && v.abs() < 1e-6,
                "Schwarzschild R[{i},{j}] = {v} at point1 (expected 0, vacuum)"
            );
        }
    }
}

#[test]
fn test_schwarzschild_ricci_vacuum_point2() {
    let metric = schwarzschild_metric();
    let gamma = christoffel(&metric);
    let r_tensor = ricci_tensor(&gamma, &[0, 1, 2, 3]);

    // Point 2: rs=2, r=5, θ=π/4, φ=π
    let mut vals = vec![0.0_f64; 11];
    vals[0] = 5.0;
    vals[1] = std::f64::consts::PI / 4.0;
    vals[2] = std::f64::consts::PI;
    vals[3] = 0.0;
    vals[10] = 2.0;

    for i in 0..4 {
        for j in 0..4 {
            let v = eval(r_tensor.get(&[i, j]), &vals);
            assert!(
                v.is_finite() && v.abs() < 1e-6,
                "Schwarzschild R[{i},{j}] = {v} at point2 (expected 0, vacuum)"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12: Schwarzschild Einstein tensor is zero (vacuum)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_schwarzschild_einstein_vacuum() {
    let metric = schwarzschild_metric();
    let gamma = christoffel(&metric);
    let r_tensor = ricci_tensor(&gamma, &[0, 1, 2, 3]);
    let g_tensor = einstein_tensor(&metric, &r_tensor);

    let mut vals = vec![0.0_f64; 11];
    vals[0] = 10.0;
    vals[1] = std::f64::consts::PI / 2.0;
    vals[2] = 0.0;
    vals[3] = 0.0;
    vals[10] = 2.0;

    for i in 0..4 {
        for j in 0..4 {
            let v = eval(g_tensor.get(&[i, j]), &vals);
            assert!(
                v.is_finite() && v.abs() < 1e-5,
                "Schwarzschild G[{i},{j}] = {v} at sample point (expected 0)"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 13: Schwarzschild scalar curvature R = 0
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_schwarzschild_scalar_curvature_zero() {
    let metric = schwarzschild_metric();
    let gamma = christoffel(&metric);
    let r_tensor = ricci_tensor(&gamma, &[0, 1, 2, 3]);

    let mut vals = vec![0.0_f64; 11];
    vals[0] = 10.0;
    vals[1] = std::f64::consts::PI / 2.0;
    vals[2] = 0.0;
    vals[3] = 0.0;
    vals[10] = 2.0;

    // R = g^{ij} R_{ij}
    let mut ricci_scalar = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            let g_inv_ij = eval(metric.g_inv.get(&[i, j]), &vals);
            let r_ij = eval(r_tensor.get(&[i, j]), &vals);
            ricci_scalar += g_inv_ij * r_ij;
        }
    }
    assert!(
        ricci_scalar.is_finite() && ricci_scalar.abs() < 1e-5,
        "Schwarzschild scalar curvature R = {ricci_scalar} (expected 0)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 14: 3×3 symbolic metric inverse round-trip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_3d_metric_inverse_roundtrip() {
    // Diagonal 3D metric: g = diag(r², r²sin²θ, 1) in spherical (angular part)
    // Var(0)=r, Var(1)=θ
    let mut g = zero_g(3);
    g[IxDyn(&[0, 0])] = LoweredOp::Pow(Box::new(var(0)), Box::new(c(2.0)));
    g[IxDyn(&[1, 1])] = LoweredOp::Mul(
        Box::new(LoweredOp::Pow(Box::new(var(0)), Box::new(c(2.0)))),
        Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Sin(Box::new(var(1)))),
            Box::new(c(2.0)),
        )),
    );
    g[IxDyn(&[2, 2])] = c(1.0);

    let metric = Metric::new(g, vec![0, 1, 2]).expect("3D metric");

    // Verify g^{ii} g_{ii} ≈ 1 for diagonal at (r=3, θ=π/4, φ=0)
    let vals = [3.0_f64, std::f64::consts::PI / 4.0, 0.0];
    let ctx = EvalCtx::new(&vals);
    for i in 0..3 {
        let g_inv_ii = eval_real(metric.g_inv.get(&[i, i]), &ctx).expect("g_inv");
        let g_ii = eval_real(metric.g.get(&[i, i]), &ctx).expect("g");
        let product = g_inv_ii * g_ii;
        assert!(
            (product - 1.0).abs() < 1e-9,
            "g^{{ii}} g_{{ii}} = {product} for i={i} (expected 1)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 15: Metric compatibility — ∇_μ g_νρ = 0
//
// The Levi-Civita connection satisfies metric compatibility by construction.
// This test verifies ∇_μ g_{νρ} = 0 explicitly using covariant_derivative
// on the polar metric g = [[1,0],[0,r²]], evaluated at (r=3, θ=0.5).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_metric_compatibility_polar() {
    use scirs2_symbolic::diffgeom::tensor::Tensor;

    // Build polar metric g = diag(1, r²)
    let mut g_data = zero_g(2);
    g_data[IxDyn(&[0, 0])] = c(1.0);
    g_data[IxDyn(&[1, 1])] = LoweredOp::Pow(Box::new(var(0)), Box::new(c(2.0)));
    let metric = Metric::new(g_data.clone(), vec![0, 1]).expect("polar metric");
    let gamma = christoffel(&metric);

    // Package metric components into a (0,2) Tensor for covariant_derivative
    let g_tensor = Tensor::from_components(0, 2, 2, g_data);

    // Evaluate at r=3, θ=0.5
    let r_val = 3.0_f64;
    let theta_val = 0.5_f64;
    let vals = [r_val, theta_val];

    // ∇_μ g_{νρ} must be zero for all μ, ν, ρ ∈ {0,1}
    // covariant_derivative of a (0,2) tensor w.r.t. coord index μ gives a (0,3) tensor
    for mu in 0..2_usize {
        let nabla_mu_g = covariant_derivative(&g_tensor, &gamma, &[0, 1], mu);
        // nabla_mu_g has shape (0,3) → indices [nu, rho, mu_output]
        // The wrt index occupies the last (2nd) down-index slot
        for nu in 0..2_usize {
            for rho in 0..2_usize {
                // index order: [nu, rho] for (0,2) output plus mu slot
                // covariant_derivative appends the new covariant slot last
                let v = eval(nabla_mu_g.get(&[nu, rho, mu]), &vals);
                assert!(
                    v.is_finite() && v.abs() < 1e-8,
                    "nabla_{mu} g_[{nu},{rho}] = {v} (expected 0, metric compatibility)"
                );
            }
        }
    }
}
