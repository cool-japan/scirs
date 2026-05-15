//! Integration: symbolic gradient end-to-end.
//!
//! Phase 1 fast-follow item C — exercises `grad`, `jacobian`, `hessian` on
//! representative formulas; checks correctness against analytic derivatives
//! and central-difference numerics; verifies the constant-exponent `Pow`
//! fast path (avoids `ln(neg)` for plain integer powers); and stress-tests
//! the iterative work-stack against deep expressions.

use scirs2_symbolic::eml::eval::{eval_real, EvalCtx};
use scirs2_symbolic::eml::{grad, hessian, jacobian, LoweredOp};

const TOL: f64 = 1e-10;

#[test]
fn grad_quadratic() {
    // f(x) = x², df/dx = 2x. At x = 3 → 6.
    let f = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(0)));
    let g = grad(&f, 0);
    let r = eval_real(&g, &EvalCtx::new(&[3.0])).expect("eval");
    assert!((r - 6.0).abs() < TOL);
}

#[test]
fn grad_chain_rule_sin_x_squared() {
    // f(x) = sin(x²), df/dx = cos(x²) · 2x. At x = 0.7.
    let x = LoweredOp::Var(0);
    let x_sq = LoweredOp::Mul(Box::new(x.clone()), Box::new(x.clone()));
    let f = LoweredOp::Sin(Box::new(x_sq));
    let g = grad(&f, 0);

    let xv = 0.7;
    let r = eval_real(&g, &EvalCtx::new(&[xv])).expect("eval");
    let expected = (xv * xv).cos() * 2.0 * xv;
    assert!(
        (r - expected).abs() < TOL,
        "got {}, expected {}",
        r,
        expected
    );
}

#[test]
fn jacobian_2x2_polynomial() {
    // f(x, y) = xy + x. ∂f/∂x = y + 1, ∂f/∂y = x.
    let f = LoweredOp::Add(
        Box::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        )),
        Box::new(LoweredOp::Var(0)),
    );
    let j = jacobian(&f, 2);
    assert_eq!(j.len(), 2);
    let r0 = eval_real(&j[0], &EvalCtx::new(&[2.0, 3.0])).expect("eval ∂f/∂x");
    let r1 = eval_real(&j[1], &EvalCtx::new(&[2.0, 3.0])).expect("eval ∂f/∂y");
    assert!(
        (r0 - 4.0).abs() < TOL,
        "∂f/∂x at (2,3) = {}, expected 4",
        r0
    );
    assert!(
        (r1 - 2.0).abs() < TOL,
        "∂f/∂y at (2,3) = {}, expected 2",
        r1
    );
}

#[test]
fn hessian_quadratic_2d() {
    // f(x, y) = x²·y + x·y².
    // ∂²f/∂x² = 2y, ∂²f/∂y² = 2x.
    // At (2, 3): ∂²f/∂x² = 6, ∂²f/∂y² = 4.
    let x = LoweredOp::Var(0);
    let y = LoweredOp::Var(1);
    let xx = LoweredOp::Mul(Box::new(x.clone()), Box::new(x.clone()));
    let yy = LoweredOp::Mul(Box::new(y.clone()), Box::new(y.clone()));
    let f = LoweredOp::Add(
        Box::new(LoweredOp::Mul(Box::new(xx), Box::new(y.clone()))),
        Box::new(LoweredOp::Mul(Box::new(x.clone()), Box::new(yy))),
    );
    let h = hessian(&f, 2);

    let h00 = eval_real(&h[0][0], &EvalCtx::new(&[2.0, 3.0])).expect("eval H[0][0]");
    let h11 = eval_real(&h[1][1], &EvalCtx::new(&[2.0, 3.0])).expect("eval H[1][1]");
    assert!((h00 - 6.0).abs() < TOL, "H[0][0] = {}, expected 6", h00);
    assert!((h11 - 4.0).abs() < TOL, "H[1][1] = {}, expected 4", h11);
}

#[test]
fn central_difference_parity_sin() {
    // d/dx sin(x) = cos(x). Compare symbolic vs O(h²) central difference.
    let f = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
    let g = grad(&f, 0);

    let h = 1e-6;
    for xv in [0.5, 1.0, 1.5, -0.5, -1.0] {
        let f_plus = eval_real(&f, &EvalCtx::new(&[xv + h])).expect("eval +");
        let f_minus = eval_real(&f, &EvalCtx::new(&[xv - h])).expect("eval -");
        let central = (f_plus - f_minus) / (2.0 * h);
        let symbolic = eval_real(&g, &EvalCtx::new(&[xv])).expect("eval g");
        assert!(
            (central - symbolic).abs() < 1e-5,
            "sin'({}): central={}, symbolic={}",
            xv,
            central,
            symbolic
        );
    }
}

#[test]
fn central_difference_parity_exp() {
    // d/dx exp(x) = exp(x). Compare via central difference.
    let f = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
    let g = grad(&f, 0);
    let h = 1e-6;
    for xv in [-1.0, 0.0, 0.5, 1.0] {
        let f_plus = eval_real(&f, &EvalCtx::new(&[xv + h])).expect("eval +");
        let f_minus = eval_real(&f, &EvalCtx::new(&[xv - h])).expect("eval -");
        let central = (f_plus - f_minus) / (2.0 * h);
        let symbolic = eval_real(&g, &EvalCtx::new(&[xv])).expect("eval g");
        // Use either absolute or relative tolerance — exp grows fast.
        let abs_err = (central - symbolic).abs();
        let rel_err = if symbolic.abs() > 0.0 {
            abs_err / symbolic.abs()
        } else {
            abs_err
        };
        assert!(
            abs_err < 1e-5 || rel_err < 1e-5,
            "exp'({}): central={}, symbolic={}",
            xv,
            central,
            symbolic
        );
    }
}

#[test]
fn pow_const_fast_path_no_ln_neg() {
    // f(x) = (x - 1)². df/dx = 2·(x - 1).
    // The constant-exponent fast path emits 2·(x-1)^1·1 directly, NOT
    // exp(2·ln(x-1))·(...) which would NaN at x=0 because ln(-1) on the
    // real path is a domain error. At x = 0 the result must be -2.
    let inner = LoweredOp::Sub(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
    let f = LoweredOp::Pow(Box::new(inner), Box::new(LoweredOp::Const(2.0)));
    let g = grad(&f, 0);

    let r = eval_real(&g, &EvalCtx::new(&[0.0])).expect("eval — must not enter ln(-1) branch");
    assert!((r - (-2.0)).abs() < TOL, "got {}, expected -2", r);
}

#[test]
fn deep_grad_no_overflow() {
    // Build x + x + x + ... + x (1001 copies). df/dx = 1001.
    // grad must traverse this iteratively — recursion would overflow.
    let mut f = LoweredOp::Var(0);
    for _ in 0..1000 {
        f = LoweredOp::Add(Box::new(f), Box::new(LoweredOp::Var(0)));
    }
    let g = grad(&f, 0);
    let r = eval_real(&g, &EvalCtx::new(&[1.0])).expect("eval");
    assert!((r - 1001.0).abs() < TOL, "got {}, expected 1001", r);
}

#[test]
fn grad_constant_is_zero() {
    // d/dx (5) = 0.
    let f = LoweredOp::Const(5.0);
    let g = grad(&f, 0);
    let r = eval_real(&g, &EvalCtx::new(&[0.0])).expect("eval");
    assert!(r.abs() < TOL);
}

#[test]
fn grad_partial_orthogonality() {
    // f(x, y) = x. ∂f/∂y = 0 (Var(1) does not appear in f).
    let f = LoweredOp::Var(0);
    let g = grad(&f, 1);
    let r = eval_real(&g, &EvalCtx::new(&[2.0, 3.0])).expect("eval");
    assert!(r.abs() < TOL, "∂x/∂y = {}, expected 0", r);
}
