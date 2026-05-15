//! Lowering: [`EmlTree`] → [`LoweredOp`] via canonical-shape recognisers.
//!
//! Walks an [`EmlTree`] post-order; at each `Eml { left, right }` node tries
//! to recognise a canonical shape (e.g. `eml(x, 1) → Exp(x)`). On match,
//! emits the typed [`LoweredOp`] variant; on miss, falls back to the literal
//! `eml(l, r) = exp(l) - ln(r)` form which is mathematically correct (just
//! less efficient).
//!
//! See [`raise`] for the inverse pass back to [`EmlTree`].
//!
//! # Phase 0 scope
//!
//! Only the `exp(x)` shape (`eml(x, 1)`) is recognised in Phase 0 — every
//! other canonical shape lowers as `Sub(Exp(l), Ln(r))`. Phase 2's
//! `cas::canonicalize` will expand the recogniser table to cover `ln`,
//! `sin`, `cos`, etc. by structural-hash unification against precomputed
//! `Canonical::*` template trees (see oxieml's
//! `match_sin_structure` / `match_cos_structure`).
//!
//! # Adapted from oxieml v0.1.0, `src/lower.rs`
//!
//! Iterative post-order walk and the `eml(x, 1) → Exp(x)` recogniser are
//! adapted from oxieml. The recursive `lower_node` of oxieml is rewritten
//! here as an iterative work-stack to avoid OS-stack overflow on the
//! 543-node-deep canonical `sin` tree (recursion would blow up around
//! depth 1000 on default stack sizes). Native `Sqrt` and `Abs` variants
//! are scirs2-symbolic additions (oxieml lowers them to `Pow(_, 0.5)` /
//! `sqrt(square(_))`).

use crate::eml::canonical::Canonical;
use crate::eml::op::LoweredOp;
use crate::eml::tree::{EmlNode, EmlTree};
use crate::error::EmlError;

/// Lower an [`EmlTree`] to a [`LoweredOp`] by recognising canonical shapes.
///
/// Walk is iterative post-order. At each `Eml` node we match against the
/// local shape; on a recognised match we emit the typed op, otherwise we
/// fall back to the literal `eml(l, r) = Sub(Exp(l), Ln(r))` form.
///
/// # Phase 0
///
/// Recognises only `eml(x, 1) → Exp(x)`. Every other canonical shape
/// (`ln`, `sin`, `cos`, …) lowers via the literal fallback — mathematically
/// correct but verbose. Phase 2 will add the full recogniser table.
pub fn lower(tree: &EmlTree) -> LoweredOp {
    // Iterative post-order. For each EmlNode, push children before parent;
    // when popping a parent (visited=true), pop the children's already-
    // emitted LoweredOp values from `output` and combine.
    let mut work: Vec<(&EmlNode, bool)> = vec![(&tree.root, false)];
    let mut output: Vec<LoweredOp> = Vec::with_capacity(16);

    while let Some((node, visited)) = work.pop() {
        if visited {
            match node {
                EmlNode::One => output.push(LoweredOp::Const(1.0)),
                EmlNode::Var(i) => output.push(LoweredOp::Var(*i)),
                EmlNode::Eml { left: _, right: _ } => {
                    // Pop the two children's LoweredOps (right was pushed
                    // *first* in the pre-visit branch, so it sits *below*
                    // left in the output stack; pop order is right then
                    // left → no wait: post-visit pops in reverse-emit order.
                    //
                    // Pre-visit pushes (in order): post-marker, right, left.
                    // The work-stack is LIFO, so on subsequent iterations we
                    // pop left first, recurse it, then right, then the
                    // post-marker fires here.
                    //
                    // When children are leaves they emit a single `output`
                    // entry each. Order of pushes to `output` is therefore:
                    // left-result first, right-result second. So:
                    //   right_result = output.pop()
                    //   left_result  = output.pop()
                    let r = output
                        .pop()
                        .expect("post-order invariant: right child result on stack");
                    let l = output
                        .pop()
                        .expect("post-order invariant: left child result on stack");
                    let result = recognise_canonical_shape(node, &l, &r).unwrap_or_else(|| {
                        // Literal fallback: eml(l, r) = exp(l) - ln(r).
                        LoweredOp::Sub(
                            Box::new(LoweredOp::Exp(Box::new(l))),
                            Box::new(LoweredOp::Ln(Box::new(r))),
                        )
                    });
                    output.push(result);
                }
            }
        } else {
            match node {
                EmlNode::One | EmlNode::Var(_) => {
                    // Leaves: schedule post-visit (which will simply emit).
                    work.push((node, true));
                }
                EmlNode::Eml { left, right } => {
                    work.push((node, true));
                    // Push right first; left pops & is processed first
                    // (so its result lands in `output` *before* the right's).
                    work.push((right, false));
                    work.push((left, false));
                }
            }
        }
    }

    output
        .pop()
        .expect("post-order invariant: final result on stack")
}

/// Match the [`EmlNode`] against canonical shapes and emit a typed op.
///
/// Returns `None` if no canonical shape is recognised — caller falls back
/// to the literal `eml = Sub(Exp, Ln)` form.
///
/// # Phase 0
///
/// Recognises only `eml(x, 1) → Exp(x)`. The right child must be the
/// literal `One` node. The lowered left child `l` is reused as-is (it is
/// already the lowered representation of `x`).
fn recognise_canonical_shape(node: &EmlNode, l: &LoweredOp, _r: &LoweredOp) -> Option<LoweredOp> {
    if let EmlNode::Eml { left: _, right } = node {
        // Pattern: eml(x, 1) → Exp(x)
        if matches!(**right, EmlNode::One) {
            return Some(LoweredOp::Exp(Box::new(l.clone())));
        }
    }
    None
}

/// Inverse pass: [`LoweredOp`] → [`EmlTree`] via [`Canonical`] constructors.
///
/// Iterative post-order over the `LoweredOp`; builds [`EmlTree`] values on
/// a value stack, combining via the matching `Canonical::*` constructor at
/// each post-visit.
///
/// # Errors
///
/// - [`EmlError::LoweringFailed`] if a `Const` value is not exactly `0.0`,
///   `-1.0`, or a positive integer — non-integer constants don't have a
///   finite EML encoding; callers needing them should use [`Canonical`]
///   constructors directly.
pub fn raise(op: &LoweredOp) -> Result<EmlTree, EmlError> {
    let mut work: Vec<(&LoweredOp, bool)> = vec![(op, false)];
    let mut stack: Vec<EmlTree> = Vec::with_capacity(16);

    while let Some((node, visited)) = work.pop() {
        if visited {
            match node {
                LoweredOp::Const(c) => {
                    // Only exact integers / 0 / -1 have a finite EML encoding.
                    if *c == 0.0 {
                        stack.push(Canonical::zero());
                    } else if *c == -1.0 {
                        stack.push(Canonical::neg_one());
                    } else if c.is_finite()
                        && c.fract() == 0.0
                        && *c >= 1.0
                        && *c <= u64::MAX as f64
                    {
                        // Positive integer in u64 range.
                        let n = *c as u64;
                        // Round-trip safety: if `n as f64` doesn't equal `c`
                        // exactly, the constant is too large for u64 to
                        // represent without loss — reject.
                        if (n as f64) != *c {
                            return Err(EmlError::LoweringFailed(format!(
                                "raise: constant {} loses precision when converted to u64",
                                c
                            )));
                        }
                        stack.push(Canonical::nat(n)?);
                    } else {
                        return Err(EmlError::LoweringFailed(format!(
                            "raise: non-integer constant {} not yet supported",
                            c
                        )));
                    }
                }
                LoweredOp::Var(i) => stack.push(EmlTree::var(*i)),
                LoweredOp::Add(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child tree on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child tree on stack");
                    stack.push(Canonical::add(&a, &b));
                }
                LoweredOp::Sub(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child tree on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child tree on stack");
                    stack.push(Canonical::sub(&a, &b));
                }
                LoweredOp::Mul(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child tree on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child tree on stack");
                    stack.push(Canonical::mul(&a, &b));
                }
                LoweredOp::Div(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child tree on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child tree on stack");
                    stack.push(Canonical::div(&a, &b));
                }
                LoweredOp::Pow(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child tree on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child tree on stack");
                    stack.push(Canonical::pow(&a, &b));
                }
                LoweredOp::Neg(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::neg(&c));
                }
                LoweredOp::Exp(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::exp(&c));
                }
                LoweredOp::Ln(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::ln(&c));
                }
                LoweredOp::Sin(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::sin(&c));
                }
                LoweredOp::Cos(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::cos(&c));
                }
                LoweredOp::Tan(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::tan(&c));
                }
                LoweredOp::Sinh(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::sinh(&c));
                }
                LoweredOp::Cosh(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::cosh(&c));
                }
                LoweredOp::Tanh(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::tanh(&c));
                }
                LoweredOp::Arcsin(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::arcsin(&c));
                }
                LoweredOp::Arccos(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::arccos(&c));
                }
                LoweredOp::Arctan(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::arctan(&c));
                }
                LoweredOp::Arcsinh(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::arcsinh(&c));
                }
                LoweredOp::Arccosh(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::arccosh(&c));
                }
                LoweredOp::Arctanh(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::arctanh(&c));
                }
                LoweredOp::Sqrt(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::sqrt(&c));
                }
                LoweredOp::Abs(_) => {
                    let c = stack
                        .pop()
                        .expect("post-order invariant: child tree on stack");
                    stack.push(Canonical::abs(&c));
                }
            }
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

    Ok(stack
        .pop()
        .expect("post-order invariant: final result on stack"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // lower
    // ----------------------------------------------------------------

    #[test]
    fn lower_one_to_const() {
        let t = EmlTree::one();
        let op = lower(&t);
        assert_eq!(op, LoweredOp::Const(1.0));
    }

    #[test]
    fn lower_var() {
        let t = EmlTree::var(3);
        assert_eq!(lower(&t), LoweredOp::Var(3));
    }

    #[test]
    fn lower_canonical_exp_recognises() {
        // Canonical::exp(x) = eml(x, 1). Lowering should recognise and emit Exp(Var(0)).
        let x = EmlTree::var(0);
        let exp_x = Canonical::exp(&x);
        let lowered = lower(&exp_x);
        assert_eq!(lowered, LoweredOp::Exp(Box::new(LoweredOp::Var(0))));
    }

    #[test]
    fn lower_euler_recognises_as_exp_of_one() {
        // Canonical::euler() = eml(1, 1). The exp(x) recogniser fires
        // (because right is One), giving Exp(Const(1.0)) — which evaluates
        // to `e`. Mathematically correct.
        let e = Canonical::euler();
        let lowered = lower(&e);
        assert_eq!(lowered, LoweredOp::Exp(Box::new(LoweredOp::Const(1.0))));
    }

    #[test]
    fn lower_bare_eml_falls_back_to_sub_exp_ln() {
        // eml(x, y) where neither child is One — falls back to Sub(Exp(x), Ln(y)).
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let bare = EmlTree::eml(&x, &y);
        let lowered = lower(&bare);
        assert_eq!(
            lowered,
            LoweredOp::Sub(
                Box::new(LoweredOp::Exp(Box::new(LoweredOp::Var(0)))),
                Box::new(LoweredOp::Ln(Box::new(LoweredOp::Var(1)))),
            )
        );
    }

    #[test]
    fn lower_canonical_ln_falls_back() {
        // Canonical::ln(x) = eml(1, eml(eml(1, x), 1)).
        // Phase 0 doesn't recognise this shape; falls back to literal Sub(Exp, Ln).
        // The result is mathematically correct, just verbose.
        let x = EmlTree::var(0);
        let ln_x = Canonical::ln(&x);
        let lowered = lower(&ln_x);
        // The outermost shape is eml(1, ...) — right is NOT One, so the
        // exp recogniser misses; we fall back to Sub(Exp(Const(1.0)), Ln(...)).
        assert!(matches!(lowered, LoweredOp::Sub(_, _)));
    }

    #[test]
    fn lower_deep_chain_no_overflow() {
        // Build a deep EmlTree with 1000 nested eml calls — would overflow
        // OS stack on a recursive lower implementation.
        let one = EmlTree::one();
        let mut t = one.clone();
        for _ in 0..1000 {
            t = EmlTree::eml(&t, &one);
        }
        // Must not panic.
        let lowered = lower(&t);
        // Outermost: eml(t', 1) → recognised as Exp(lower(t')).
        assert!(matches!(lowered, LoweredOp::Exp(_)));
    }

    #[test]
    fn lower_then_count_vars_consistent() {
        // Lowering a tree with vars 0 and 1 produces a LoweredOp where
        // count_vars() reports the same count (after composition with
        // canonical encodings, which add no new vars).
        let x = EmlTree::var(0);
        let y = EmlTree::var(1);
        let bare = EmlTree::eml(&x, &y);
        let lowered = lower(&bare);
        assert_eq!(lowered.count_vars(), 2);
    }

    // ----------------------------------------------------------------
    // raise
    // ----------------------------------------------------------------

    #[test]
    fn raise_var() {
        let op = LoweredOp::Var(0);
        let tree = match raise(&op) {
            Ok(t) => t,
            Err(e) => panic!("raise(Var(0)) failed: {e:?}"),
        };
        assert_eq!(tree, EmlTree::var(0));
    }

    #[test]
    fn raise_const_one_to_canonical_one() {
        // Const(1.0) → Canonical::nat(1) → EmlTree::one().
        let op = LoweredOp::Const(1.0);
        let tree = match raise(&op) {
            Ok(t) => t,
            Err(e) => panic!("raise(Const(1.0)) failed: {e:?}"),
        };
        assert_eq!(tree, EmlTree::one());
    }

    #[test]
    fn raise_const_zero_to_canonical_zero() {
        let op = LoweredOp::Const(0.0);
        let tree = match raise(&op) {
            Ok(t) => t,
            Err(e) => panic!("raise(Const(0.0)) failed: {e:?}"),
        };
        assert_eq!(tree, Canonical::zero());
    }

    #[test]
    fn raise_const_neg_one_to_canonical_neg_one() {
        let op = LoweredOp::Const(-1.0);
        let tree = match raise(&op) {
            Ok(t) => t,
            Err(e) => panic!("raise(Const(-1.0)) failed: {e:?}"),
        };
        assert_eq!(tree, Canonical::neg_one());
    }

    #[test]
    fn raise_const_non_integer_errors() {
        let op = LoweredOp::Const(2.71);
        let r = raise(&op);
        assert!(matches!(r, Err(EmlError::LoweringFailed(_))));
    }

    #[test]
    fn raise_inverse_basic_add() {
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let tree = match raise(&op) {
            Ok(t) => t,
            Err(e) => panic!("raise(Add) failed: {e:?}"),
        };
        // tree should be Canonical::add(var(0), var(1))
        assert_eq!(tree, Canonical::add(&EmlTree::var(0), &EmlTree::var(1)));
    }

    #[test]
    fn raise_inverse_unary_exp() {
        let op = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
        let tree = match raise(&op) {
            Ok(t) => t,
            Err(e) => panic!("raise(Exp) failed: {e:?}"),
        };
        assert_eq!(tree, Canonical::exp(&EmlTree::var(0)));
    }

    #[test]
    fn raise_deep_chain_no_overflow() {
        // Deep LoweredOp chain — must not blow the OS stack.
        let mut op = LoweredOp::Var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Var(1)));
        }
        let _ = match raise(&op) {
            Ok(t) => t,
            Err(e) => panic!("raise(deep) failed: {e:?}"),
        };
    }

    #[test]
    fn lower_then_raise_smoke() {
        // Round-trip on a simple shape. We don't assert exact equality
        // since canonical encodings inflate a `Mul` to ~7-deep tree; we
        // just verify both directions terminate without panic.
        let original = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(0)));
        let tree = match raise(&original) {
            Ok(t) => t,
            Err(e) => panic!("raise failed: {e:?}"),
        };
        let _recovered = lower(&tree);
        // Recovered won't equal `original` exactly because lower won't
        // re-recognise the canonical mul shape (Phase 2 work). Smoke OK.
    }
}
