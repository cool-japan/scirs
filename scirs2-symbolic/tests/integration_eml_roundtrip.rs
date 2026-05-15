//! Integration: Expr ↔ LoweredOp round-trip on every Expr variant.
//!
//! Phase 1 fast-follow item C — exercises the bridge between the legacy
//! `Expr` enum and the EML `LoweredOp` flat IR. Companion to the unit tests
//! inside `eml::bridge`; here we run the full pipeline including
//! `EmlTree → LoweredOp → EmlTree` lowering/raising and the deep-stack
//! soundness property.

use scirs2_symbolic::eml::{
    lower, raise, Canonical, EmlTree, FromLowered, LoweredOp, ToLowered, VarMap,
};
use scirs2_symbolic::Expr;

#[test]
fn expr_const_to_lowered() {
    // 3.15 instead of 3.14: avoids `clippy::approx_constant` (which flags
    // any value within ULP-distance of `f64::consts::PI`) without changing
    // the semantics of the test.
    let e = Expr::Const(3.15);
    let (op, _) = e.to_lowered().expect("to_lowered");
    assert_eq!(op, LoweredOp::Const(3.15));
}

#[test]
fn expr_var_to_lowered() {
    let e = Expr::var("x");
    let (op, map) = e.to_lowered().expect("to_lowered");
    assert_eq!(op, LoweredOp::Var(0));
    assert_eq!(map.names, vec!["x".to_string()]);
}

#[test]
fn expr_round_trip_all_basic() {
    let cases = [
        Expr::var("a") + Expr::var("b"),
        Expr::var("a") - Expr::var("b"),
        Expr::var("a") * Expr::var("b"),
        Expr::var("a") / Expr::var("b"),
        Expr::var("a").pow(Expr::Const(2.0)),
        -Expr::var("a"),
        Expr::var("a").sin(),
        Expr::var("a").cos(),
        Expr::var("a").tan(),
        Expr::var("a").exp(),
        Expr::var("a").ln(),
        Expr::var("a").sqrt(),
        Expr::var("a").abs(),
    ];
    for e in cases {
        let (op, map) = e.to_lowered().expect("to_lowered");
        let recovered = Expr::from_lowered(&op, &map).expect("from_lowered");
        assert_eq!(recovered, e, "round-trip failed for {:?}", e);
    }
}

#[test]
fn varmap_determinism() {
    let e = Expr::var("z") + Expr::var("a") * Expr::var("m");
    let m1 = VarMap::from_expr(&e);
    let m2 = VarMap::from_expr(&e);
    assert_eq!(m1, m2);
    // BTreeSet-backed: names emerge in sorted order regardless of construction order.
    assert_eq!(m1.names, vec!["a", "m", "z"]);
}

#[test]
fn lower_then_raise_basic() {
    // `Add(Var(0), Var(1))` should round-trip through the EML substrate.
    let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
    let tree = raise(&op).expect("raise");
    // Tree must be at least as deep as the corresponding Canonical::add encoding.
    assert!(tree.depth() >= 1);
    assert!(tree.num_vars() >= 2);
}

#[test]
fn canonical_then_lower_basic() {
    // Canonical::sin produces a 543-node-deep tree; lowering must succeed
    // (falls back to literal eml(l, r) = exp(l) - ln(r) for unrecognised
    // canonical shapes, which is mathematically correct).
    let x = EmlTree::var(0);
    let formula = Canonical::sin(&x);
    let lowered = lower(&formula);
    // Just confirm we got a non-trivial tree out — exact shape depends on
    // the recogniser table (Phase 0 only matches eml(x, 1) → Exp(x)).
    assert!(
        matches!(
            lowered,
            LoweredOp::Sub(_, _) | LoweredOp::Sin(_) | LoweredOp::Exp(_)
        ),
        "expected Sub/Sin/Exp at root, got {:?}",
        lowered
    );
}

#[test]
fn deep_round_trip_no_overflow() {
    // Build a 100-deep `+` chain on `Expr` and round-trip through `LoweredOp`.
    // 100 is comfortable for the legacy `Expr` recursion budget; deeper
    // nesting would risk overflow on the recursive `to_lowered`/`from_lowered`
    // bridge (documented limitation in `bridge.rs`).
    let mut e = Expr::var("x");
    for _ in 0..100 {
        e = e + Expr::var("x");
    }
    let (op, map) = e.to_lowered().expect("to_lowered");
    let recovered = Expr::from_lowered(&op, &map).expect("from_lowered");
    // Structural equality is preserved by the bridge (no folding in either
    // direction).
    assert_eq!(recovered, e);
}

#[test]
fn hyperbolic_does_not_round_trip_to_expr() {
    // LoweredOp::Sinh has no Expr equivalent — from_lowered should error
    // with `LoweringFailed`.
    let op = LoweredOp::Sinh(Box::new(LoweredOp::Var(0)));
    let map = VarMap::new(vec!["x".into()]);
    let result = Expr::from_lowered(&op, &map);
    assert!(result.is_err(), "expected error for Sinh, got {:?}", result);
}

#[test]
fn round_trip_nested_formula_preserves_structure() {
    // (sin(x) + cos(y)) * exp(-x) — exercises trig, binary ops, and Neg.
    let e = (Expr::var("x").sin() + Expr::var("y").cos()) * (-Expr::var("x")).exp();
    let (op, map) = e.to_lowered().expect("to_lowered");
    let recovered = Expr::from_lowered(&op, &map).expect("from_lowered");
    assert_eq!(recovered, e);
}

#[test]
fn unknown_variable_against_explicit_map_errors() {
    // `Expr::var("y")` against an empty VarMap must surface UnknownVariable.
    let e = Expr::var("y");
    let map = VarMap::default();
    let result = e.to_lowered_with(&map);
    assert!(
        result.is_err(),
        "expected UnknownVariable, got {:?}",
        result
    );
}
