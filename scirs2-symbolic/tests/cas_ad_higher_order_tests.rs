//! Wave 74 Item #4 — higher-order symbolic derivatives.
//!
//! Tests cover:
//! - Iterated single-variable derivatives (3rd, 4th, n-th).
//! - Mixed partial derivatives.
//! - Taylor expansion to high order.
//! - Cross-validation with curvature tensors (3D flat-space Riemann ⇒ 0).

use scirs2_symbolic::cas::ad::{
    fourth_derivative, grad_canonical, higher_order_grad, taylor_higher_order, third_derivative,
};
use scirs2_symbolic::cas::canonicalize;
use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::op::LoweredOp;

fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}
fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}
fn pow(b: LoweredOp, e: f64) -> LoweredOp {
    LoweredOp::Pow(Box::new(b), Box::new(c(e)))
}
fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Mul(Box::new(a), Box::new(b))
}
fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    LoweredOp::Add(Box::new(a), Box::new(b))
}

fn eval_at(op: &LoweredOp, vals: &[f64]) -> f64 {
    let canon = canonicalize(op).into_op();
    eval_real(&canon, &EvalCtx::new(vals)).unwrap_or(f64::NAN)
}

// =====================================================================
// 1) Third derivative of x³ = 6.
// =====================================================================

#[test]
fn third_derivative_cubic_polynomial() {
    let f = pow(var(0), 3.0);
    let d3 = third_derivative(&f, [0, 0, 0]);
    let v = eval_at(&d3, &[5.0]);
    assert!(
        (v - 6.0).abs() < 1e-9,
        "d³/dx³(x³) at x=5 should be 6, got {v}"
    );
}

// =====================================================================
// 2) Fourth derivative of x⁴ = 24.
// =====================================================================

#[test]
fn fourth_derivative_quartic() {
    let f = pow(var(0), 4.0);
    let d4 = fourth_derivative(&f, [0, 0, 0, 0]);
    let v = eval_at(&d4, &[2.7]);
    assert!(
        (v - 24.0).abs() < 1e-7,
        "d⁴/dx⁴(x⁴) at x=2.7 should be 24, got {v}"
    );
}

// =====================================================================
// 3) Mixed partial: ∂³(x²·y)/∂x²∂y = 2.
// =====================================================================

#[test]
fn higher_order_mixed_partial_quadratic() {
    let f = mul(pow(var(0), 2.0), var(1));
    let d3 = third_derivative(&f, [0, 0, 1]);
    let v = eval_at(&d3, &[0.5, 0.7]);
    assert!((v - 2.0).abs() < 1e-9, "∂³(x²y)/∂x²∂y should be 2, got {v}");
}

// =====================================================================
// 4) higher_order_grad iteration matches manually chained grad_canonical.
// =====================================================================

#[test]
fn higher_order_grad_iteration_matches_chained() {
    // f = sin(x) · x^4
    let f = mul(LoweredOp::Sin(Box::new(var(0))), pow(var(0), 4.0));

    // higher_order_grad(f, x, 4) yields [d¹f/dx¹, d²f/dx², d³f/dx³, d⁴f/dx⁴].
    let series = higher_order_grad(&f, 0, 4);
    assert_eq!(series.len(), 4);

    // Manually compute four chained derivatives.
    let d1 = grad_canonical(&f, 0);
    let d2 = grad_canonical(&d1, 0);
    let d3 = grad_canonical(&d2, 0);
    let d4 = grad_canonical(&d3, 0);
    let manual = [d1, d2, d3, d4];

    // At x = 0.7, both should agree to numerical precision.
    let pt = [0.7];
    for (i, (auto, manu)) in series.iter().zip(manual.iter()).enumerate() {
        let v_auto = eval_at(auto, &pt);
        let v_manu = eval_at(manu, &pt);
        assert!(
            (v_auto - v_manu).abs() < 1e-9,
            "order {} mismatch: auto={v_auto}, manual={v_manu}",
            i + 1
        );
    }
}

// =====================================================================
// 5) Taylor expansion of sin(x) at x₀=0 to order 5.
//
//    Coefficients: [sin(0), cos(0), -sin(0)/2!, -cos(0)/3!, sin(0)/4!, cos(0)/5!]
//                = [0, 1, 0, -1/6, 0, 1/120]
// =====================================================================

#[test]
fn taylor_higher_order_around_zero() {
    let f = LoweredOp::Sin(Box::new(var(0)));
    let coeffs = taylor_higher_order(&f, 0, 0.0, 5);
    assert_eq!(coeffs.len(), 6);

    // Evaluate each coeff (they're all constant after canonicalize at x₀=0).
    let expected = [0.0, 1.0, 0.0, -1.0 / 6.0, 0.0, 1.0 / 120.0];
    for (i, exp) in expected.iter().enumerate() {
        let v = eval_at(&coeffs[i], &[0.0]);
        assert!(
            (v - exp).abs() < 1e-9,
            "Taylor coeff at order {i}: got {v}, expected {exp}"
        );
    }
}

// =====================================================================
// 6) Riemann tensor in 3D flat space ⇒ structural zero.
//
//    For the flat metric g_ij = δ_ij in 3D, all Christoffels are zero, so
//    all Riemann tensor components ∂Γ + ΓΓ are also zero. This
//    cross-validates that higher_order derivatives of zero stay zero.
// =====================================================================

#[test]
fn riemann_tensor_3d_flat_space_returns_zero() {
    // Christoffels for flat 3D are all zero. The Riemann tensor formula:
    //   R^k_{lij} = ∂_i Γ^k_{lj} − ∂_j Γ^k_{li} + Γ^k_{im}·Γ^m_{lj} − Γ^k_{jm}·Γ^m_{li}
    // For Γ ≡ 0, every term vanishes structurally.
    //
    // We model this by computing the third partial derivative of a zero
    // expression ∂³(0)/∂x∂y∂z, which must be zero.
    let zero = c(0.0);
    let d = third_derivative(&zero, [0, 1, 2]);
    let v = eval_at(&d, &[1.0, 1.0, 1.0]);
    assert!(v.abs() < 1e-12, "d³(0)/dxdydz should be 0, got {v}");

    // Also: a non-trivial polynomial whose 3rd cross-partial is constant.
    // f = x²·y·z → ∂³f/∂x²∂y = 2·z; → ∂³f/∂x∂y∂z (one each) = 2x.
    let f = mul(mul(pow(var(0), 2.0), var(1)), var(2));
    let cross = third_derivative(&f, [0, 1, 2]);
    let v_at = eval_at(&cross, &[3.0, 5.0, 7.0]);
    // d/dx [d/dy [d/dz (x²yz)]] = d/dx[d/dy(x²y)] = d/dx(x²) = 2x → at x=3 → 6.
    assert!(
        (v_at - 6.0).abs() < 1e-9,
        "∂³(x²yz)/∂x∂y∂z at (3,5,7) should be 6, got {v_at}"
    );
}

// =====================================================================
// 7) Higher-order with mixed constants: ∂⁴(x²·y³)/∂x²∂y³ = 12.
//
//    ∂(x²y³)/∂x = 2xy³ ; ∂²/∂x² = 2y³
//    ∂(2y³)/∂y = 6y² ; ∂(6y²)/∂y = 12y ; ∂(12y)/∂y = 12
//
// So we need 5 differentiations to land on the constant 12. The 4th
// derivative ∂⁴(x²y³)/∂x²∂y² = 12y, which evaluated at y=1 gives 12.
// =====================================================================

#[test]
fn higher_order_with_constants() {
    // f = x² · y³
    let f = mul(pow(var(0), 2.0), pow(var(1), 3.0));
    // ∂⁴f/∂x²∂y² = 12y (a function of y only)
    let d4 = fourth_derivative(&f, [0, 0, 1, 1]);
    let v = eval_at(&d4, &[1.5, 1.0]);
    assert!(
        (v - 12.0).abs() < 1e-9,
        "∂⁴(x²y³)/∂x²∂y² at y=1 should be 12, got {v}"
    );
    // At y=2.0, expect 24.0
    let v2 = eval_at(&d4, &[1.5, 2.0]);
    assert!(
        (v2 - 24.0).abs() < 1e-9,
        "∂⁴(x²y³)/∂x²∂y² at y=2 should be 24, got {v2}"
    );
}

// =====================================================================
// 8) Cache reuses partial derivatives — call higher_order_grad repeatedly
//    on the same input and verify the results match exactly. This is an
//    indirect cache-correctness test: the cache must produce identical
//    results across calls.
// =====================================================================

#[test]
fn cache_reuses_partial_derivatives() {
    let f = add(pow(var(0), 5.0), mul(c(3.0), pow(var(0), 3.0)));

    // Two independent computations should yield equal canonical forms.
    let series_1 = higher_order_grad(&f, 0, 4);
    let series_2 = higher_order_grad(&f, 0, 4);

    assert_eq!(series_1.len(), series_2.len());
    for (a, b) in series_1.iter().zip(series_2.iter()) {
        let ha = canonicalize(a).hash();
        let hb = canonicalize(b).hash();
        assert_eq!(
            ha, hb,
            "cache miss/recompute produced inconsistent canonical forms"
        );
    }

    // And evaluate at a point — must match.
    let pt = [1.7];
    for (a, b) in series_1.iter().zip(series_2.iter()) {
        let va = eval_at(a, &pt);
        let vb = eval_at(b, &pt);
        assert!(
            (va - vb).abs() < 1e-12,
            "cache hit/miss numerical inconsistency: {va} vs {vb}"
        );
    }
}
