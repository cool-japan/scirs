//! Integration: eval_real and eval_complex on full pipelines.
//!
//! Phase 1 fast-follow item C — exercises the iterative stack-machine
//! evaluator over both the primitive-op path and the canonical-encoding
//! path (where `Canonical::sin` produces a 543-node-deep tree that would
//! blow the OS stack under recursive evaluation).

use num_complex::Complex64;
use scirs2_symbolic::eml::eval::{eval_complex, eval_real, EvalCtx};
use scirs2_symbolic::eml::{lower, Canonical, EmlTree, LoweredOp};

const TOL: f64 = 1e-10;

#[test]
fn eval_basic_arithmetic() {
    // f(x) = x² + 1 at x = 3 → 10
    let op = LoweredOp::Add(
        Box::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(0)),
        )),
        Box::new(LoweredOp::Const(1.0)),
    );
    let r = eval_real(&op, &EvalCtx::new(&[3.0])).expect("eval");
    assert!((r - 10.0).abs() < TOL);
}

#[test]
fn deep_sin_no_overflow() {
    // Canonical::sin builds a ~543-node-deep tree; composing two of them
    // yields ~1086 nodes. Recursive evaluation would overflow the OS stack.
    // Real eval may fail because canonical sin uses Euler's formula and
    // contains `ln(-1)` in its lowered tree — route via the complex path
    // (which has no domain restrictions on `ln`).
    let x = EmlTree::var(0);
    let inner = Canonical::sin(&x);
    let outer = Canonical::sin(&inner);
    let lowered = lower(&outer);

    let xv = Complex64::new(0.3, 0.0);
    let r = eval_complex(&lowered, &[xv]).expect("eval_complex");
    let expected = (0.3_f64).sin().sin();
    // 1e-8 tolerance — accumulated float error across the ~1086-node tree.
    assert!(
        (r.re - expected).abs() < 1e-8,
        "sin(sin(0.3)): got re={}, expected {}",
        r.re,
        expected
    );
}

#[test]
fn multi_var_eval() {
    // f(x, y, z) = x*y + z at (2, 3, 5) → 11
    let op = LoweredOp::Add(
        Box::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Var(1)),
        )),
        Box::new(LoweredOp::Var(2)),
    );
    let r = eval_real(&op, &EvalCtx::new(&[2.0, 3.0, 5.0])).expect("eval");
    assert!((r - 11.0).abs() < TOL);
}

#[test]
fn complex_via_real_when_possible() {
    // For real-valued inputs, the complex path's real part must agree with
    // the real path; the imaginary part must be zero.
    let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
    let real_r = eval_real(&op, &EvalCtx::new(&[0.5])).expect("real");
    let complex_r = eval_complex(&op, &[Complex64::new(0.5, 0.0)]).expect("complex");
    assert!((real_r - complex_r.re).abs() < TOL);
    assert!(complex_r.im.abs() < TOL);
}

#[test]
fn unbound_variable_returns_err() {
    use scirs2_symbolic::error::EmlError;
    // `Var(5)` against a 2-binding context must surface UnboundVariableIndex.
    let op = LoweredOp::Var(5);
    let ctx = EvalCtx::new(&[1.0, 2.0]);
    let result = eval_real(&op, &ctx);
    assert!(
        matches!(
            result,
            Err(EmlError::UnboundVariableIndex { idx: 5, len: 2 })
        ),
        "expected UnboundVariableIndex {{ idx: 5, len: 2 }}, got {:?}",
        result
    );
}

#[test]
fn ln_negative_returns_err() {
    use scirs2_symbolic::error::EmlError;
    // ln(-1) is a domain error on the real path.
    let op = LoweredOp::Ln(Box::new(LoweredOp::Const(-1.0)));
    let result = eval_real(&op, &EvalCtx::new(&[]));
    assert!(
        matches!(result, Err(EmlError::EvalDomain(_))),
        "expected EvalDomain, got {:?}",
        result
    );
}

#[test]
fn ln_negative_via_complex_works() {
    // ln(-1) = i·π on the principal branch.
    let op = LoweredOp::Ln(Box::new(LoweredOp::Const(-1.0)));
    let r = eval_complex(&op, &[]).expect("complex ln(-1) = i*pi");
    let expected = Complex64::new(0.0, std::f64::consts::PI);
    assert!((r - expected).norm() < TOL);
}

#[test]
fn pythagorean_identity() {
    // sin²(x) + cos²(x) = 1 (using primitive ops, not the canonical encoding,
    // which would route through `ln(-1)` and require the complex path).
    let x = LoweredOp::Var(0);
    let sin_x = LoweredOp::Sin(Box::new(x.clone()));
    let cos_x = LoweredOp::Cos(Box::new(x.clone()));
    let sin_sq = LoweredOp::Mul(Box::new(sin_x.clone()), Box::new(sin_x));
    let cos_sq = LoweredOp::Mul(Box::new(cos_x.clone()), Box::new(cos_x));
    let identity = LoweredOp::Add(Box::new(sin_sq), Box::new(cos_sq));

    // Use 3.15 (and a few other points) instead of 3.14 — avoids clippy's
    // `approx_constant` lint while still spanning negative, zero, and
    // multi-radian inputs.
    for xv in [-1.0, 0.0, 0.5, 1.5, 3.15] {
        let r = eval_real(&identity, &EvalCtx::new(&[xv])).expect("eval");
        assert!(
            (r - 1.0).abs() < TOL,
            "sin²({}) + cos²({}) = {}, expected 1",
            xv,
            xv,
            r
        );
    }
}

#[test]
fn division_by_zero_returns_err() {
    use scirs2_symbolic::error::EmlError;
    // 1 / 0 must error rather than producing inf.
    let op = LoweredOp::Div(
        Box::new(LoweredOp::Const(1.0)),
        Box::new(LoweredOp::Const(0.0)),
    );
    let result = eval_real(&op, &EvalCtx::new(&[]));
    assert!(
        matches!(result, Err(EmlError::DivisionByZero)),
        "expected DivisionByZero, got {:?}",
        result
    );
}
