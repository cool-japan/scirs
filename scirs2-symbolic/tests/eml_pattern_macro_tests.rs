//! Integration tests for the `eml_pattern!` / `eml_template!` proc-macro DSL.
//!
//! Enabled only when the `macros` feature is active.

use scirs2_symbolic::cas::pattern::{
    instantiate, match_pattern, BinaryKind, Bindings, Pattern, UnaryKind,
};
use scirs2_symbolic::{eml_pattern, eml_template, LoweredOp};

// ---------------------------------------------------------------------------
// Basic pattern matching
// ---------------------------------------------------------------------------

#[test]
fn test_eml_pattern_simple_add() {
    let pat = eml_pattern!(add(?0, const(0.0)));
    let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(0.0)));
    let mut bindings = Bindings::new();
    assert!(match_pattern(&pat, &op, &mut bindings));
    assert!(bindings.contains_key(&0));
}

#[test]
fn test_eml_pattern_const_integer_literal() {
    // const(0) — integer literal — must also produce PatConst(0.0 f64)
    let pat = eml_pattern!(add(?0, const(0)));
    let op = LoweredOp::Add(Box::new(LoweredOp::Var(1)), Box::new(LoweredOp::Const(0.0)));
    let mut bindings = Bindings::new();
    assert!(match_pattern(&pat, &op, &mut bindings));
}

#[test]
fn test_eml_pattern_unary() {
    let pat = eml_pattern!(sin(?0));
    let op = LoweredOp::Sin(Box::new(LoweredOp::Var(1)));
    let mut bindings = Bindings::new();
    assert!(match_pattern(&pat, &op, &mut bindings));
    assert_eq!(bindings[&0], LoweredOp::Var(1));
}

#[test]
fn test_eml_pattern_nested() {
    let pat = eml_pattern!(mul(?0, add(?1, const(1.0))));
    let op = LoweredOp::Mul(
        Box::new(LoweredOp::Var(0)),
        Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Var(1)),
            Box::new(LoweredOp::Const(1.0)),
        )),
    );
    let mut bindings = Bindings::new();
    assert!(match_pattern(&pat, &op, &mut bindings));
    assert_eq!(bindings[&0], LoweredOp::Var(0));
    assert_eq!(bindings[&1], LoweredOp::Var(1));
}

#[test]
fn test_eml_pattern_no_match() {
    let pat = eml_pattern!(exp(?0));
    let op = LoweredOp::Ln(Box::new(LoweredOp::Var(0)));
    let mut bindings = Bindings::new();
    assert!(!match_pattern(&pat, &op, &mut bindings));
}

// ---------------------------------------------------------------------------
// Template (rhs) / instantiate
// ---------------------------------------------------------------------------

#[test]
fn test_eml_template_instantiate() {
    // Pattern: ?0 + ?1; Template: ?1 + ?0 (commutativity rewrite)
    let lhs = eml_pattern!(add(?0, ?1));
    let rhs = eml_template!(add(?1, ?0));
    let op = LoweredOp::Add(Box::new(LoweredOp::Var(5)), Box::new(LoweredOp::Const(3.0)));
    let mut bindings = Bindings::new();
    assert!(match_pattern(&lhs, &op, &mut bindings));
    let result = instantiate(&rhs, &bindings).expect("instantiate should succeed");
    // Result should be Add(Const(3.0), Var(5)) — the commuted form
    match result {
        LoweredOp::Add(left, right) => {
            assert_eq!(*left, LoweredOp::Const(3.0));
            assert_eq!(*right, LoweredOp::Var(5));
        }
        other => panic!("expected Add, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// PatConstInt
// ---------------------------------------------------------------------------

#[test]
fn test_eml_pattern_int_literal() {
    let pat = eml_pattern!(int(2));
    let mut b_ok = Bindings::new();
    assert!(match_pattern(&pat, &LoweredOp::Const(2.0), &mut b_ok));
    let mut b_fail = Bindings::new();
    assert!(!match_pattern(&pat, &LoweredOp::Const(3.0), &mut b_fail));
}

// ---------------------------------------------------------------------------
// PatGroundVar
// ---------------------------------------------------------------------------

#[test]
fn test_eml_pattern_ground_var() {
    let pat = eml_pattern!(var(0));
    let mut b_ok = Bindings::new();
    assert!(match_pattern(&pat, &LoweredOp::Var(0), &mut b_ok));
    let mut b_fail = Bindings::new();
    assert!(!match_pattern(&pat, &LoweredOp::Var(1), &mut b_fail));
}

// ---------------------------------------------------------------------------
// PatVar wildcard
// ---------------------------------------------------------------------------

#[test]
fn test_eml_pattern_wildcard_binds_anything() {
    let pat = eml_pattern!(?0);
    let mut b = Bindings::new();
    let op = LoweredOp::Mul(
        Box::new(LoweredOp::Var(99)),
        Box::new(LoweredOp::Const(7.5)),
    );
    assert!(match_pattern(&pat, &op, &mut b));
    assert_eq!(b[&0], op);
}

// ---------------------------------------------------------------------------
// All unary operators exercise
// ---------------------------------------------------------------------------

#[test]
fn test_eml_pattern_all_unary_ops() {
    let pairs: &[(Pattern, LoweredOp)] = &[
        (
            eml_pattern!(neg(?0)),
            LoweredOp::Neg(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(exp(?0)),
            LoweredOp::Exp(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(ln(?0)),
            LoweredOp::Ln(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(sin(?0)),
            LoweredOp::Sin(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(cos(?0)),
            LoweredOp::Cos(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(tan(?0)),
            LoweredOp::Tan(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(sinh(?0)),
            LoweredOp::Sinh(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(cosh(?0)),
            LoweredOp::Cosh(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(tanh(?0)),
            LoweredOp::Tanh(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(arcsin(?0)),
            LoweredOp::Arcsin(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(arccos(?0)),
            LoweredOp::Arccos(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(arctan(?0)),
            LoweredOp::Arctan(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(arcsinh(?0)),
            LoweredOp::Arcsinh(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(arccosh(?0)),
            LoweredOp::Arccosh(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(arctanh(?0)),
            LoweredOp::Arctanh(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(sqrt(?0)),
            LoweredOp::Sqrt(Box::new(LoweredOp::Var(0))),
        ),
        (
            eml_pattern!(abs(?0)),
            LoweredOp::Abs(Box::new(LoweredOp::Var(0))),
        ),
    ];
    for (pat, op) in pairs {
        let mut b = Bindings::new();
        assert!(
            match_pattern(pat, op, &mut b),
            "failed to match {pat:?} against {op:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// All binary operators exercise
// ---------------------------------------------------------------------------

#[test]
fn test_eml_pattern_all_binary_ops() {
    let pairs: &[(Pattern, LoweredOp)] = &[
        (
            eml_pattern!(add(?0, ?1)),
            LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
        ),
        (
            eml_pattern!(sub(?0, ?1)),
            LoweredOp::Sub(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
        ),
        (
            eml_pattern!(mul(?0, ?1)),
            LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
        ),
        (
            eml_pattern!(div(?0, ?1)),
            LoweredOp::Div(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
        ),
        (
            eml_pattern!(pow(?0, ?1)),
            LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
        ),
    ];
    for (pat, op) in pairs {
        let mut b = Bindings::new();
        assert!(
            match_pattern(pat, op, &mut b),
            "failed to match {pat:?} against {op:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Consistency: same wildcard used twice
// ---------------------------------------------------------------------------

#[test]
fn test_eml_pattern_repeated_wildcard_consistent() {
    let pat = eml_pattern!(add(?0, ?0));
    let op_same = LoweredOp::Add(Box::new(LoweredOp::Var(1)), Box::new(LoweredOp::Var(1)));
    let op_diff = LoweredOp::Add(Box::new(LoweredOp::Var(1)), Box::new(LoweredOp::Var(2)));
    let mut b1 = Bindings::new();
    assert!(match_pattern(&pat, &op_same, &mut b1));
    let mut b2 = Bindings::new();
    assert!(!match_pattern(&pat, &op_diff, &mut b2));
}

// ---------------------------------------------------------------------------
// eml_template as a pattern (same code path)
// ---------------------------------------------------------------------------

#[test]
fn test_eml_template_as_pattern() {
    // eml_template! should produce identical code to eml_pattern!
    let t = eml_template!(mul(exp(?0), exp(?1)));
    let op = LoweredOp::Mul(
        Box::new(LoweredOp::Exp(Box::new(LoweredOp::Var(0)))),
        Box::new(LoweredOp::Exp(Box::new(LoweredOp::Var(1)))),
    );
    let mut b = Bindings::new();
    assert!(match_pattern(&t, &op, &mut b));
}
