//! Canonical-form rewrite rules beyond [`crate::eml::simplify::simplify_op`].
//!
//! These are the structural rewrites that complete [`mod@crate::cas::canonicalize`]:
//! - **log identities** — `ln(a*b) → ln(a) + ln(b)`, `ln(a/b) → ln(a) - ln(b)`,
//!   `ln(exp(x)) → x`
//! - **exp identities** — `exp(a)·exp(b) → exp(a+b)`, `exp(ln(x)) → x`
//! - **power identities** — `a^m · a^n → a^(m+n)` (structural-hash-equal bases),
//!   `(a^m)^n → a^(m*n)`
//!
//! # Pipeline position
//!
//! `apply_canonical_rules` runs **after** `simplify_op` (constant folding,
//! identity rules, hash-based commutative ordering) and **before** another
//! `simplify_op` pass (which folds the resulting `Add(Const, Const)` etc.).
//! The full canonical pipeline lives in [`crate::cas::canonicalize::canonicalize`].
//!
//! # Soundness
//!
//! Like [`crate::eml::simplify::simplify_op`], these rules are
//! **structural** — they do not track domains. `ln(a*b) → ln(a)+ln(b)` is
//! a valid algebraic identity but loses domain when `a*b > 0` while
//! `a < 0, b < 0`. Domain tracking is a future Phase 2 concern; the
//! `canonicalize` decidability boundary excludes branch-cut-sensitive
//! expressions for that reason.
//!
//! # No recursion
//!
//! Like the rest of the EML stack, every traversal here is iterative
//! (work-stack pattern). A 543-node-deep `Canonical::sin(x)` tree must
//! not blow the OS stack.

#![warn(missing_docs)]

use crate::eml::op::LoweredOp;

/// Maximum fixed-point iterations for [`apply_canonical_rules`].
///
/// Set generously so realistic inputs always converge. Combined with the
/// 32-iteration outer fixed-point loop in [`mod@crate::cas::canonicalize`] and
/// the 64-iteration inner [`crate::eml::simplify::MAX_SIMPLIFY_ITER`],
/// the total rewrite budget is bounded.
pub const MAX_RULE_ITER: usize = 16;

/// Apply the full canonical-rules rewrite set to fixed point.
///
/// Iterative — repeatedly applies `apply_rules_once` until the
/// [`LoweredOp::structural_hash`] stops changing or [`MAX_RULE_ITER`] is
/// reached. Idempotent on outputs that are already at the fixed point.
pub fn apply_canonical_rules(op: &LoweredOp) -> LoweredOp {
    let mut current = op.clone();
    // Sentinel hash 0 — overwhelmingly unlikely to be a real structural hash
    // (would require ahash collision on the empty tape).
    let mut prev_hash: u128 = 0;
    for _ in 0..MAX_RULE_ITER {
        let next = apply_rules_once(&current);
        let h = next.structural_hash();
        if h == prev_hash {
            return next;
        }
        prev_hash = h;
        current = next;
    }
    current
}

/// Single bottom-up rewrite pass over the tree.
///
/// Iterative (no recursion). Walks the tree in post-order, popping
/// already-rewritten children from a value stack and invoking the per-op
/// rule applier on the parent. Returns the rewritten root.
fn apply_rules_once(op: &LoweredOp) -> LoweredOp {
    let mut work: Vec<(&LoweredOp, bool)> = vec![(op, false)];
    let mut stack: Vec<LoweredOp> = Vec::with_capacity(16);

    while let Some((node, visited)) = work.pop() {
        if visited {
            let rewritten = match node {
                LoweredOp::Const(c) => LoweredOp::Const(*c),
                LoweredOp::Var(i) => LoweredOp::Var(*i),
                LoweredOp::Add(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child on stack");
                    rule_add(a, b)
                }
                LoweredOp::Sub(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child on stack");
                    rule_sub(a, b)
                }
                LoweredOp::Mul(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child on stack");
                    rule_mul(a, b)
                }
                LoweredOp::Div(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child on stack");
                    LoweredOp::Div(Box::new(a), Box::new(b))
                }
                LoweredOp::Pow(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child on stack");
                    rule_pow(a, b)
                }
                LoweredOp::Neg(_) => {
                    let c = stack.pop().expect("post-order invariant: child on stack");
                    LoweredOp::Neg(Box::new(c))
                }
                LoweredOp::Exp(_) => {
                    let c = stack.pop().expect("post-order invariant: child on stack");
                    rule_exp(c)
                }
                LoweredOp::Ln(_) => {
                    let c = stack.pop().expect("post-order invariant: child on stack");
                    rule_ln(c)
                }
                LoweredOp::Sin(_)
                | LoweredOp::Cos(_)
                | LoweredOp::Tan(_)
                | LoweredOp::Sinh(_)
                | LoweredOp::Cosh(_)
                | LoweredOp::Tanh(_)
                | LoweredOp::Arcsin(_)
                | LoweredOp::Arccos(_)
                | LoweredOp::Arctan(_)
                | LoweredOp::Arcsinh(_)
                | LoweredOp::Arccosh(_)
                | LoweredOp::Arctanh(_)
                | LoweredOp::Sqrt(_)
                | LoweredOp::Abs(_) => {
                    let c = stack.pop().expect("post-order invariant: child on stack");
                    wrap_unary(node, c)
                }
            };
            stack.push(rewritten);
        } else {
            match node {
                LoweredOp::Const(_) | LoweredOp::Var(_) => {
                    work.push((node, true));
                }
                LoweredOp::Add(a, b)
                | LoweredOp::Sub(a, b)
                | LoweredOp::Mul(a, b)
                | LoweredOp::Div(a, b)
                | LoweredOp::Pow(a, b) => {
                    work.push((node, true));
                    // Push right first so left pops first.
                    work.push((b, false));
                    work.push((a, false));
                }
                LoweredOp::Neg(c)
                | LoweredOp::Exp(c)
                | LoweredOp::Ln(c)
                | LoweredOp::Sin(c)
                | LoweredOp::Cos(c)
                | LoweredOp::Tan(c)
                | LoweredOp::Sinh(c)
                | LoweredOp::Cosh(c)
                | LoweredOp::Tanh(c)
                | LoweredOp::Arcsin(c)
                | LoweredOp::Arccos(c)
                | LoweredOp::Arctan(c)
                | LoweredOp::Arcsinh(c)
                | LoweredOp::Arccosh(c)
                | LoweredOp::Arctanh(c)
                | LoweredOp::Sqrt(c)
                | LoweredOp::Abs(c) => {
                    work.push((node, true));
                    work.push((c, false));
                }
            }
        }
    }

    stack
        .pop()
        .expect("post-order invariant: final result on stack")
}

// =====================================================================
// Per-op canonical-rule appliers.
//
// These run **in addition to** the rules in `simplify_op`. They focus on
// algebraic *expansions* (ln(ab) → ln(a)+ln(b)) and *combinations*
// (exp(a)·exp(b) → exp(a+b), a^m · a^n → a^(m+n)) that are out of
// `simplify_op`'s scope (which focuses on local folds and identity rules).
// =====================================================================

fn rule_add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    // ln(x) + ln(y) → ln(x*y) is the inverse direction of rule_ln; we keep
    // the *expanded* form here (consistent with downstream sort+CSE) so the
    // canonical form for log products has logs on the leaves rather than
    // inside the log.
    LoweredOp::Add(Box::new(a), Box::new(b))
}

/// Sub canonical-rules.
///
/// Pulls inner Sub out of a containing Sub so that
/// `a − (b − c) → (a + c) − b`. This is required for trig closure on
/// expressions like `cos²(x) − (1 − sin²(x))`: after this rewrite the
/// surface form is `Add(cos², sin²) − 1 → 1 − 1 → 0`, which lets the
/// existing Pythagorean identity fire on the Add side.
///
/// Direction-pinning: we always rewrite the inner-Sub form into an Add+Sub
/// shape; the outer simplify pass collapses constants. The reverse direction
/// is never produced by simplify, so this rule is non-oscillating.
fn rule_sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    // a − (b − c) → (a + c) − b
    if let LoweredOp::Sub(b_inner, c_inner) = &b {
        return LoweredOp::Sub(
            Box::new(LoweredOp::Add(Box::new(a), c_inner.clone())),
            b_inner.clone(),
        );
    }
    LoweredOp::Sub(Box::new(a), Box::new(b))
}

fn rule_mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    // exp(a) * exp(b) → exp(a + b)
    if let (LoweredOp::Exp(ea), LoweredOp::Exp(eb)) = (&a, &b) {
        return LoweredOp::Exp(Box::new(LoweredOp::Add(ea.clone(), eb.clone())));
    }
    // a^m * a^n → a^(m+n) — only if structural-hash-equal bases.
    // Hash equality is necessary and sufficient under our canonical
    // ordering (simplify_op already canonicalised both sides).
    if let (LoweredOp::Pow(b1, e1), LoweredOp::Pow(b2, e2)) = (&a, &b) {
        if b1.structural_hash() == b2.structural_hash() {
            return LoweredOp::Pow(b1.clone(), Box::new(LoweredOp::Add(e1.clone(), e2.clone())));
        }
    }
    // a * a^n → a^(n+1) — promote the bare base to Pow form.
    if let LoweredOp::Pow(b1, e1) = &b {
        if b1.structural_hash() == a.structural_hash() {
            return LoweredOp::Pow(
                b1.clone(),
                Box::new(LoweredOp::Add(e1.clone(), Box::new(LoweredOp::Const(1.0)))),
            );
        }
    }
    if let LoweredOp::Pow(b1, e1) = &a {
        if b1.structural_hash() == b.structural_hash() {
            return LoweredOp::Pow(
                b1.clone(),
                Box::new(LoweredOp::Add(e1.clone(), Box::new(LoweredOp::Const(1.0)))),
            );
        }
    }
    LoweredOp::Mul(Box::new(a), Box::new(b))
}

fn rule_pow(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    // (a^m)^n → a^(m*n)
    if let LoweredOp::Pow(inner_base, inner_expo) = a {
        return LoweredOp::Pow(
            inner_base,
            Box::new(LoweredOp::Mul(inner_expo, Box::new(b))),
        );
    }
    LoweredOp::Pow(Box::new(a), Box::new(b))
}

fn rule_exp(c: LoweredOp) -> LoweredOp {
    // exp(ln(x)) → x  (algebraic; domain tracking deferred)
    if let LoweredOp::Ln(inner) = c {
        return *inner;
    }
    // exp(0) → 1 — already handled by simplify_op via Const-fold; we
    // include it here for completeness so the rule set is self-contained
    // even when invoked outside the canonicalize pipeline.
    if let LoweredOp::Const(v) = c {
        if v == 0.0 {
            return LoweredOp::Const(1.0);
        }
        return LoweredOp::Exp(Box::new(LoweredOp::Const(v)));
    }
    LoweredOp::Exp(Box::new(c))
}

fn rule_ln(c: LoweredOp) -> LoweredOp {
    // ln(exp(x)) → x
    if let LoweredOp::Exp(inner) = c {
        return *inner;
    }
    // ln(a * b) → ln(a) + ln(b)
    if let LoweredOp::Mul(a, b) = c {
        return LoweredOp::Add(Box::new(LoweredOp::Ln(a)), Box::new(LoweredOp::Ln(b)));
    }
    // ln(a / b) → ln(a) - ln(b)
    if let LoweredOp::Div(a, b) = c {
        return LoweredOp::Sub(Box::new(LoweredOp::Ln(a)), Box::new(LoweredOp::Ln(b)));
    }
    // ln(a^n) → n * ln(a) — power-rule expansion.
    if let LoweredOp::Pow(base, expo) = c {
        return LoweredOp::Mul(expo, Box::new(LoweredOp::Ln(base)));
    }
    LoweredOp::Ln(Box::new(c))
}

/// Wrap an unary op without rewriting (used when no canonical rule applies).
///
/// `node` provides the variant tag (its original child is ignored — the
/// already-rewritten `c` is the new child).
fn wrap_unary(node: &LoweredOp, c: LoweredOp) -> LoweredOp {
    match node {
        LoweredOp::Sin(_) => LoweredOp::Sin(Box::new(c)),
        LoweredOp::Cos(_) => LoweredOp::Cos(Box::new(c)),
        LoweredOp::Tan(_) => LoweredOp::Tan(Box::new(c)),
        LoweredOp::Sinh(_) => LoweredOp::Sinh(Box::new(c)),
        LoweredOp::Cosh(_) => LoweredOp::Cosh(Box::new(c)),
        LoweredOp::Tanh(_) => LoweredOp::Tanh(Box::new(c)),
        LoweredOp::Arcsin(_) => LoweredOp::Arcsin(Box::new(c)),
        LoweredOp::Arccos(_) => LoweredOp::Arccos(Box::new(c)),
        LoweredOp::Arctan(_) => LoweredOp::Arctan(Box::new(c)),
        LoweredOp::Arcsinh(_) => LoweredOp::Arcsinh(Box::new(c)),
        LoweredOp::Arccosh(_) => LoweredOp::Arccosh(Box::new(c)),
        LoweredOp::Arctanh(_) => LoweredOp::Arctanh(Box::new(c)),
        LoweredOp::Sqrt(_) => LoweredOp::Sqrt(Box::new(c)),
        LoweredOp::Abs(_) => LoweredOp::Abs(Box::new(c)),
        // Defensive default — should be unreachable since `apply_rules_once`
        // routes all unary variants here. Returning the parent clone is
        // semantics-preserving in the unlikely future-variant case.
        _ => node.clone(),
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    #[test]
    fn ln_of_product_expands() {
        // ln(x * y) → ln(x) + ln(y)
        let op = LoweredOp::Ln(Box::new(LoweredOp::Mul(Box::new(var(0)), Box::new(var(1)))));
        let result = apply_canonical_rules(&op);
        assert!(
            matches!(result, LoweredOp::Add(_, _)),
            "expected Add, got {:?}",
            result
        );
    }

    #[test]
    fn ln_of_quotient_expands() {
        // ln(x / y) → ln(x) - ln(y)
        let op = LoweredOp::Ln(Box::new(LoweredOp::Div(Box::new(var(0)), Box::new(var(1)))));
        let result = apply_canonical_rules(&op);
        assert!(
            matches!(result, LoweredOp::Sub(_, _)),
            "expected Sub, got {:?}",
            result
        );
    }

    #[test]
    fn ln_of_power_expands() {
        // ln(x^3) → 3 * ln(x)
        let op = LoweredOp::Ln(Box::new(LoweredOp::Pow(Box::new(var(0)), Box::new(c(3.0)))));
        let result = apply_canonical_rules(&op);
        assert!(
            matches!(result, LoweredOp::Mul(_, _)),
            "expected Mul, got {:?}",
            result
        );
    }

    #[test]
    fn exp_of_ln_cancels() {
        // exp(ln(x)) → x
        let op = LoweredOp::Exp(Box::new(LoweredOp::Ln(Box::new(var(0)))));
        let result = apply_canonical_rules(&op);
        assert!(
            !matches!(result, LoweredOp::Exp(_)),
            "exp/ln should cancel, got {:?}",
            result
        );
    }

    #[test]
    fn ln_of_exp_cancels() {
        // ln(exp(x)) → x
        let op = LoweredOp::Ln(Box::new(LoweredOp::Exp(Box::new(var(0)))));
        let result = apply_canonical_rules(&op);
        assert_eq!(result, var(0));
    }

    #[test]
    fn power_of_power_combines() {
        // (x^2)^3 → x^(2*3)
        let op = LoweredOp::Pow(
            Box::new(LoweredOp::Pow(Box::new(var(0)), Box::new(c(2.0)))),
            Box::new(c(3.0)),
        );
        let result = apply_canonical_rules(&op);
        // Expect Pow(Var(0), Mul(2, 3))
        if let LoweredOp::Pow(base, expo) = result {
            assert_eq!(*base, var(0));
            assert!(
                matches!(*expo, LoweredOp::Mul(_, _)),
                "expected Mul exponent, got {:?}",
                expo
            );
        } else {
            panic!("expected Pow result");
        }
    }

    #[test]
    fn exp_times_exp_combines() {
        // exp(x) * exp(y) → exp(x + y)
        let op = LoweredOp::Mul(
            Box::new(LoweredOp::Exp(Box::new(var(0)))),
            Box::new(LoweredOp::Exp(Box::new(var(1)))),
        );
        let result = apply_canonical_rules(&op);
        if let LoweredOp::Exp(inner) = result {
            assert!(
                matches!(*inner, LoweredOp::Add(_, _)),
                "expected Add inside Exp, got {:?}",
                inner
            );
        } else {
            panic!("expected Exp result, got {:?}", result);
        }
    }

    #[test]
    fn power_law_combine_same_base() {
        // x^2 * x^3 → x^(2+3)
        let op = LoweredOp::Mul(
            Box::new(LoweredOp::Pow(Box::new(var(0)), Box::new(c(2.0)))),
            Box::new(LoweredOp::Pow(Box::new(var(0)), Box::new(c(3.0)))),
        );
        let result = apply_canonical_rules(&op);
        if let LoweredOp::Pow(base, _) = result {
            assert_eq!(*base, var(0));
        } else {
            panic!("expected Pow result, got {:?}", result);
        }
    }

    #[test]
    fn idempotent_on_already_canonical() {
        let op = var(0);
        let r1 = apply_canonical_rules(&op);
        let r2 = apply_canonical_rules(&r1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn deep_chain_no_overflow() {
        // 1000-deep Add chain — must not blow the stack.
        let mut op = var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(c(0.0)));
        }
        let _ = apply_canonical_rules(&op);
    }
}
