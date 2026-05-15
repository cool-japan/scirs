//! Algebraic simplification of [`LoweredOp`] via fixed-point rewriting.
//!
//! Applies a curated set of local algebraic rules — constant folding,
//! identity elimination, annihilation, inverse cancellation, sign
//! canonicalisation, and hash-based commutative ordering — until a
//! fixed point is reached or [`MAX_SIMPLIFY_ITER`] iterations elapse.
//!
//! # Adapted from oxieml v0.1.0, `src/lower_simplify.rs` (lines 127-169)
//!
//! The local rule recognisers (`sin/cos → tan`-style canonical patterns,
//! const folding, identity / annihilation / inverse cancellation) are
//! adapted from oxieml's `LoweredOp::simplify`. The recursive shape of
//! the oxieml original is rewritten here as an iterative bottom-up
//! work-stack walk (mirroring the `to_oxi_ops` post-order traversal in
//! [`crate::eml::op`]) to avoid OS-stack overflow on deep inputs (the
//! 543-node-deep `Canonical::sin` tree alone would risk overflow on
//! plausible compositions). The `f64.is_finite()` guard on every
//! constant-folding arm is a scirs2-symbolic divergence: oxieml folds
//! unconditionally and propagates `NaN`/`Inf`, but for a CAS we prefer
//! to keep the operation symbolic when the numeric result would be
//! non-finite, deferring the divergence-handling decision to the user.
//!
//! # Soundness over `f64`
//!
//! Every fold is gated on `is_finite()`. Inverse cancellations like
//! `exp(ln(x)) → x` and `ln(exp(x)) → x` are applied unconditionally
//! and *do not preserve domain* (e.g. `exp(ln(-1))` is `NaN` while `-1`
//! is finite — but a CAS user expects symbolic cancellation). Domain
//! tracking is a Phase 2 concern (`cas::canonicalize`).

#![warn(missing_docs)]

use crate::eml::op::LoweredOp;

/// Maximum fixed-point iterations before [`simplify_op_full`] gives up.
///
/// Set to a generous bound so realistic inputs always converge. If the
/// loop ever exhausts this budget on a real workload, the rule set is
/// non-confluent — file a bug.
pub const MAX_SIMPLIFY_ITER: usize = 64;

/// Simplify a [`LoweredOp`] to fixed point (default budget).
///
/// Convenience wrapper for [`simplify_op_full`] with [`MAX_SIMPLIFY_ITER`].
pub fn simplify_op(op: &LoweredOp) -> LoweredOp {
    simplify_op_full(op, MAX_SIMPLIFY_ITER)
}

/// Simplify a [`LoweredOp`] to fixed point, capped at `max_iter` rewrites.
///
/// Repeatedly applies `rewrite_once` until the [`LoweredOp::structural_hash`]
/// stops changing (fixed point) or the iteration budget is exhausted. On
/// budget exhaustion, emits a `tracing::warn!` and returns the current
/// (partially simplified) tree — the result is still mathematically
/// equivalent to the input, just possibly missing optimisations.
pub fn simplify_op_full(op: &LoweredOp, max_iter: usize) -> LoweredOp {
    let mut current = op.clone();
    // Sentinel: 0 is overwhelmingly unlikely to be a real structural hash
    // (would require the SHA-style two-seed ahash to collide on the empty
    // tape, which has measure zero). One unnecessary rewrite if the input
    // is already canonical — acceptable.
    let mut prev_hash: u128 = 0;
    for i in 0..max_iter {
        let next = rewrite_once(&current);
        let h = next.structural_hash();
        if h == prev_hash {
            return next;
        }
        prev_hash = h;
        current = next;
        if i + 1 == max_iter {
            tracing::warn!(
                "simplify_op_full: max_iter ({}) reached without fixed point on op of size ~{} — possible non-confluent rules",
                max_iter,
                current_size_estimate(&current)
            );
        }
    }
    current
}

/// Iterative O(N) node-count estimate for diagnostics. Kept private; not
/// part of the public surface because the exact count is implementation-
/// defined (we count every constructor including leaves).
fn current_size_estimate(op: &LoweredOp) -> usize {
    let mut n: usize = 0;
    let mut work: Vec<&LoweredOp> = vec![op];
    while let Some(node) = work.pop() {
        n += 1;
        match node {
            LoweredOp::Const(_) | LoweredOp::Var(_) => {}
            LoweredOp::Add(a, b)
            | LoweredOp::Sub(a, b)
            | LoweredOp::Mul(a, b)
            | LoweredOp::Div(a, b)
            | LoweredOp::Pow(a, b) => {
                work.push(a);
                work.push(b);
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
                work.push(c);
            }
        }
    }
    n
}

/// Single bottom-up rewrite pass over the tree.
///
/// Iterative (no recursion). Walks the tree in post-order, popping
/// already-rewritten children from a value stack and invoking the
/// per-op rule applier on the parent. Returns the rewritten root.
fn rewrite_once(op: &LoweredOp) -> LoweredOp {
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
                    apply_add_rules(a, b)
                }
                LoweredOp::Sub(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child on stack");
                    apply_sub_rules(a, b)
                }
                LoweredOp::Mul(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child on stack");
                    apply_mul_rules(a, b)
                }
                LoweredOp::Div(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child on stack");
                    apply_div_rules(a, b)
                }
                LoweredOp::Pow(_, _) => {
                    let b = stack
                        .pop()
                        .expect("post-order invariant: right child on stack");
                    let a = stack
                        .pop()
                        .expect("post-order invariant: left child on stack");
                    apply_pow_rules(a, b)
                }
                LoweredOp::Neg(_) => {
                    let c = stack.pop().expect("post-order invariant: child on stack");
                    apply_neg_rules(c)
                }
                LoweredOp::Exp(_) => {
                    let c = stack.pop().expect("post-order invariant: child on stack");
                    apply_exp_rules(c)
                }
                LoweredOp::Ln(_) => {
                    let c = stack.pop().expect("post-order invariant: child on stack");
                    apply_ln_rules(c)
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
                    apply_unary_rules(node, c)
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
                    // Push right first so left pops first → left result lands
                    // in `stack` before right.
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
// Per-op rule appliers
//
// Each `apply_*_rules` takes already-simplified children (by value, so
// it can move them) and returns the rewritten parent. Arms are ordered
// so the most-reductive rule wins (constant fold → identity →
// annihilation → inverse cancel → sign canonical → wrap with hash sort).
// =====================================================================

/// Pack-and-fold a candidate constant: only commit if the value is finite.
///
/// Non-finite results stay symbolic (the parent `else` branch wraps them).
fn fold_finite(v: f64) -> Option<LoweredOp> {
    if v.is_finite() {
        Some(LoweredOp::Const(v))
    } else {
        None
    }
}

fn apply_add_rules(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    // Constant fold
    if let (LoweredOp::Const(x), LoweredOp::Const(y)) = (&a, &b) {
        if let Some(c) = fold_finite(x + y) {
            return c;
        }
    }
    // Identity: x + 0 → x
    if let LoweredOp::Const(y) = &b {
        if *y == 0.0 {
            return a;
        }
    }
    if let LoweredOp::Const(x) = &a {
        if *x == 0.0 {
            return b;
        }
    }
    // x + (-y) → x - y
    if let LoweredOp::Neg(inner) = &b {
        return LoweredOp::Sub(Box::new(a), inner.clone());
    }
    // (-x) + y → y - x
    if let LoweredOp::Neg(inner) = &a {
        return LoweredOp::Sub(Box::new(b), inner.clone());
    }
    // Wrap with hash-based commutative ordering (Add is commutative).
    let ah = a.structural_hash();
    let bh = b.structural_hash();
    if ah <= bh {
        LoweredOp::Add(Box::new(a), Box::new(b))
    } else {
        LoweredOp::Add(Box::new(b), Box::new(a))
    }
}

fn apply_sub_rules(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    // Constant fold
    if let (LoweredOp::Const(x), LoweredOp::Const(y)) = (&a, &b) {
        if let Some(c) = fold_finite(x - y) {
            return c;
        }
    }
    // x - 0 → x
    if let LoweredOp::Const(y) = &b {
        if *y == 0.0 {
            return a;
        }
    }
    // 0 - x → -x
    if let LoweredOp::Const(x) = &a {
        if *x == 0.0 {
            return apply_neg_rules(b);
        }
    }
    // a - (-b) → a + b
    if let LoweredOp::Neg(inner) = &b {
        return apply_add_rules(a, *inner.clone());
    }
    // x - x → 0  (structural identity via hash)
    if a.structural_hash() == b.structural_hash() {
        return LoweredOp::Const(0.0);
    }
    LoweredOp::Sub(Box::new(a), Box::new(b))
}

fn apply_mul_rules(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    // Constant fold
    if let (LoweredOp::Const(x), LoweredOp::Const(y)) = (&a, &b) {
        if let Some(c) = fold_finite(x * y) {
            return c;
        }
    }
    // Annihilation: x * 0 → 0, 0 * x → 0
    if let LoweredOp::Const(x) = &a {
        if *x == 0.0 {
            return LoweredOp::Const(0.0);
        }
    }
    if let LoweredOp::Const(y) = &b {
        if *y == 0.0 {
            return LoweredOp::Const(0.0);
        }
    }
    // Identity: 1 * x → x, x * 1 → x
    if let LoweredOp::Const(x) = &a {
        if *x == 1.0 {
            return b;
        }
    }
    if let LoweredOp::Const(y) = &b {
        if *y == 1.0 {
            return a;
        }
    }
    // -1 * x → -x, x * -1 → -x
    if let LoweredOp::Const(x) = &a {
        if *x == -1.0 {
            return apply_neg_rules(b);
        }
    }
    if let LoweredOp::Const(y) = &b {
        if *y == -1.0 {
            return apply_neg_rules(a);
        }
    }
    // Wrap with hash-based commutative ordering (Mul is commutative).
    let ah = a.structural_hash();
    let bh = b.structural_hash();
    if ah <= bh {
        LoweredOp::Mul(Box::new(a), Box::new(b))
    } else {
        LoweredOp::Mul(Box::new(b), Box::new(a))
    }
}

fn apply_div_rules(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    // Constant fold (finite-guarded — protects against division by zero
    // producing Inf/NaN; we keep the symbolic form in that case).
    if let (LoweredOp::Const(x), LoweredOp::Const(y)) = (&a, &b) {
        if let Some(c) = fold_finite(x / y) {
            return c;
        }
    }
    // x / 1 → x
    if let LoweredOp::Const(y) = &b {
        if *y == 1.0 {
            return a;
        }
    }
    // 0 / x → 0  (only if x is structurally non-zero — a constant zero
    // is caught by the constant-fold arm above and rejected for being
    // non-finite, so we won't fold 0/0 here. For symbolic x we trust the
    // user not to introduce a removable singularity.)
    if let LoweredOp::Const(x) = &a {
        if *x == 0.0 {
            // Only safe if b isn't itself a constant zero (caught above) —
            // for symbolic divisors, return 0.
            if !matches!(&b, LoweredOp::Const(c) if *c == 0.0) {
                return LoweredOp::Const(0.0);
            }
        }
    }
    // sin(x) / cos(x) → tan(x)  (canonical pattern)
    if let (LoweredOp::Sin(sa), LoweredOp::Cos(ca)) = (&a, &b) {
        if sa.structural_hash() == ca.structural_hash() {
            return LoweredOp::Tan(sa.clone());
        }
    }
    // sinh(x) / cosh(x) → tanh(x)
    if let (LoweredOp::Sinh(sa), LoweredOp::Cosh(ca)) = (&a, &b) {
        if sa.structural_hash() == ca.structural_hash() {
            return LoweredOp::Tanh(sa.clone());
        }
    }
    LoweredOp::Div(Box::new(a), Box::new(b))
}

fn apply_pow_rules(a: LoweredOp, b: LoweredOp) -> LoweredOp {
    // x^0 → 1  (even if x is symbolic; we accept the mathematically-iffy
    // 0^0 = 1 convention as IEEE pow does)
    if let LoweredOp::Const(y) = &b {
        if *y == 0.0 {
            return LoweredOp::Const(1.0);
        }
        // x^1 → x
        if *y == 1.0 {
            return a;
        }
    }
    // Constant fold (finite-guarded)
    if let (LoweredOp::Const(x), LoweredOp::Const(y)) = (&a, &b) {
        if let Some(c) = fold_finite(x.powf(*y)) {
            return c;
        }
    }
    LoweredOp::Pow(Box::new(a), Box::new(b))
}

fn apply_neg_rules(a: LoweredOp) -> LoweredOp {
    // Constant fold
    if let LoweredOp::Const(c) = &a {
        if let Some(folded) = fold_finite(-c) {
            return folded;
        }
    }
    // -(-x) → x
    if let LoweredOp::Neg(inner) = &a {
        return *inner.clone();
    }
    // -(a - b) → b - a
    if let LoweredOp::Sub(lhs, rhs) = &a {
        return apply_sub_rules(*rhs.clone(), *lhs.clone());
    }
    LoweredOp::Neg(Box::new(a))
}

fn apply_exp_rules(a: LoweredOp) -> LoweredOp {
    // exp(0) → 1
    if let LoweredOp::Const(c) = &a {
        if *c == 0.0 {
            return LoweredOp::Const(1.0);
        }
        if let Some(folded) = fold_finite(c.exp()) {
            return folded;
        }
    }
    // exp(ln(x)) → x  (unconditional; domain is the user's responsibility)
    if let LoweredOp::Ln(inner) = &a {
        return *inner.clone();
    }
    LoweredOp::Exp(Box::new(a))
}

fn apply_ln_rules(a: LoweredOp) -> LoweredOp {
    // ln(1) → 0
    if let LoweredOp::Const(c) = &a {
        if *c == 1.0 {
            return LoweredOp::Const(0.0);
        }
        // Only fold for c > 0 (NaN is rejected by fold_finite)
        if *c > 0.0 {
            if let Some(folded) = fold_finite(c.ln()) {
                return folded;
            }
        }
    }
    // ln(exp(x)) → x
    if let LoweredOp::Exp(inner) = &a {
        return *inner.clone();
    }
    LoweredOp::Ln(Box::new(a))
}

/// Dispatch unary-rule application based on the *original* parent node's
/// variant (`node` is the pre-rewrite parent — we use its tag to know
/// which unary kind to wrap or fold). The already-rewritten child `c`
/// has been popped from the value stack.
fn apply_unary_rules(node: &LoweredOp, c: LoweredOp) -> LoweredOp {
    match node {
        LoweredOp::Sin(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.sin()) {
                    return folded;
                }
            }
            // sin(arcsin(x)) → x
            if let LoweredOp::Arcsin(inner) = &c {
                return *inner.clone();
            }
            LoweredOp::Sin(Box::new(c))
        }
        LoweredOp::Cos(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.cos()) {
                    return folded;
                }
            }
            // cos(arccos(x)) → x
            if let LoweredOp::Arccos(inner) = &c {
                return *inner.clone();
            }
            LoweredOp::Cos(Box::new(c))
        }
        LoweredOp::Tan(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.tan()) {
                    return folded;
                }
            }
            // tan(arctan(x)) → x
            if let LoweredOp::Arctan(inner) = &c {
                return *inner.clone();
            }
            LoweredOp::Tan(Box::new(c))
        }
        LoweredOp::Sinh(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.sinh()) {
                    return folded;
                }
            }
            // sinh(arcsinh(x)) → x
            if let LoweredOp::Arcsinh(inner) = &c {
                return *inner.clone();
            }
            LoweredOp::Sinh(Box::new(c))
        }
        LoweredOp::Cosh(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.cosh()) {
                    return folded;
                }
            }
            // cosh(arccosh(x)) → x
            if let LoweredOp::Arccosh(inner) = &c {
                return *inner.clone();
            }
            LoweredOp::Cosh(Box::new(c))
        }
        LoweredOp::Tanh(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.tanh()) {
                    return folded;
                }
            }
            // tanh(arctanh(x)) → x
            if let LoweredOp::Arctanh(inner) = &c {
                return *inner.clone();
            }
            LoweredOp::Tanh(Box::new(c))
        }
        LoweredOp::Arcsin(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.asin()) {
                    return folded;
                }
            }
            LoweredOp::Arcsin(Box::new(c))
        }
        LoweredOp::Arccos(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.acos()) {
                    return folded;
                }
            }
            LoweredOp::Arccos(Box::new(c))
        }
        LoweredOp::Arctan(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.atan()) {
                    return folded;
                }
            }
            LoweredOp::Arctan(Box::new(c))
        }
        LoweredOp::Arcsinh(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.asinh()) {
                    return folded;
                }
            }
            LoweredOp::Arcsinh(Box::new(c))
        }
        LoweredOp::Arccosh(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.acosh()) {
                    return folded;
                }
            }
            LoweredOp::Arccosh(Box::new(c))
        }
        LoweredOp::Arctanh(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.atanh()) {
                    return folded;
                }
            }
            LoweredOp::Arctanh(Box::new(c))
        }
        LoweredOp::Sqrt(_) => {
            if let LoweredOp::Const(x) = &c {
                // Only fold for x >= 0 — fold_finite will reject NaN anyway.
                if *x >= 0.0 {
                    if let Some(folded) = fold_finite(x.sqrt()) {
                        return folded;
                    }
                }
            }
            // sqrt(x²) → |x|  (signed-correct: sqrt is non-negative)
            if let LoweredOp::Pow(base, expo) = &c {
                if let LoweredOp::Const(e) = expo.as_ref() {
                    if *e == 2.0 {
                        return apply_unary_rules(
                            &LoweredOp::Abs(Box::new(LoweredOp::Const(0.0))), // tag for dispatch
                            *base.clone(),
                        );
                    }
                }
            }
            LoweredOp::Sqrt(Box::new(c))
        }
        LoweredOp::Abs(_) => {
            if let LoweredOp::Const(x) = &c {
                if let Some(folded) = fold_finite(x.abs()) {
                    return folded;
                }
            }
            // |-x| → |x|
            if let LoweredOp::Neg(inner) = &c {
                return apply_unary_rules(
                    &LoweredOp::Abs(Box::new(LoweredOp::Const(0.0))),
                    *inner.clone(),
                );
            }
            // ||x|| → |x|  (idempotent)
            if matches!(&c, LoweredOp::Abs(_)) {
                return c;
            }
            LoweredOp::Abs(Box::new(c))
        }
        // Defensive default — should be unreachable since rewrite_once
        // routes every variant. We don't `unreachable!()` because that
        // could panic on a future variant added without updating this
        // arm; instead we just return the child wrapped in the original
        // variant via clone (preserves semantics).
        _ => node.clone(),
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ----------------------------------------------------------------
    // Constant folding
    // ----------------------------------------------------------------

    #[test]
    fn fold_const_add() {
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Const(2.0)),
            Box::new(LoweredOp::Const(3.0)),
        );
        assert_eq!(simplify_op(&op), LoweredOp::Const(5.0));
    }

    #[test]
    fn fold_const_mul() {
        let op = LoweredOp::Mul(
            Box::new(LoweredOp::Const(2.0)),
            Box::new(LoweredOp::Const(3.0)),
        );
        assert_eq!(simplify_op(&op), LoweredOp::Const(6.0));
    }

    #[test]
    fn fold_const_div_by_zero_stays_symbolic() {
        // 1/0 = Inf — non-finite, so we keep the symbolic Div.
        let op = LoweredOp::Div(
            Box::new(LoweredOp::Const(1.0)),
            Box::new(LoweredOp::Const(0.0)),
        );
        let s = simplify_op(&op);
        // Should not have folded to Const(Inf).
        assert!(matches!(s, LoweredOp::Div(_, _)));
    }

    // ----------------------------------------------------------------
    // Identity elimination
    // ----------------------------------------------------------------

    #[test]
    fn x_plus_0() {
        let x = LoweredOp::Var(0);
        let op = LoweredOp::Add(Box::new(x.clone()), Box::new(LoweredOp::Const(0.0)));
        assert_eq!(simplify_op(&op), x);
    }

    #[test]
    fn zero_plus_x() {
        let x = LoweredOp::Var(0);
        let op = LoweredOp::Add(Box::new(LoweredOp::Const(0.0)), Box::new(x.clone()));
        assert_eq!(simplify_op(&op), x);
    }

    #[test]
    fn x_times_1() {
        let x = LoweredOp::Var(0);
        let op = LoweredOp::Mul(Box::new(x.clone()), Box::new(LoweredOp::Const(1.0)));
        assert_eq!(simplify_op(&op), x);
    }

    #[test]
    fn x_pow_0() {
        let op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(0.0)));
        assert_eq!(simplify_op(&op), LoweredOp::Const(1.0));
    }

    #[test]
    fn x_pow_1() {
        let op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        assert_eq!(simplify_op(&op), LoweredOp::Var(0));
    }

    // ----------------------------------------------------------------
    // Annihilation
    // ----------------------------------------------------------------

    #[test]
    fn x_times_0() {
        let op = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(0.0)));
        assert_eq!(simplify_op(&op), LoweredOp::Const(0.0));
    }

    // ----------------------------------------------------------------
    // Inverse cancellation
    // ----------------------------------------------------------------

    #[test]
    fn ln_exp_cancels() {
        let op = LoweredOp::Ln(Box::new(LoweredOp::Exp(Box::new(LoweredOp::Var(0)))));
        assert_eq!(simplify_op(&op), LoweredOp::Var(0));
    }

    #[test]
    fn exp_ln_cancels() {
        let op = LoweredOp::Exp(Box::new(LoweredOp::Ln(Box::new(LoweredOp::Var(0)))));
        assert_eq!(simplify_op(&op), LoweredOp::Var(0));
    }

    #[test]
    fn double_neg_cancels() {
        let op = LoweredOp::Neg(Box::new(LoweredOp::Neg(Box::new(LoweredOp::Var(0)))));
        assert_eq!(simplify_op(&op), LoweredOp::Var(0));
    }

    #[test]
    fn sin_arcsin_cancels() {
        let op = LoweredOp::Sin(Box::new(LoweredOp::Arcsin(Box::new(LoweredOp::Var(0)))));
        assert_eq!(simplify_op(&op), LoweredOp::Var(0));
    }

    #[test]
    fn cos_arccos_cancels() {
        let op = LoweredOp::Cos(Box::new(LoweredOp::Arccos(Box::new(LoweredOp::Var(0)))));
        assert_eq!(simplify_op(&op), LoweredOp::Var(0));
    }

    #[test]
    fn tan_arctan_cancels() {
        let op = LoweredOp::Tan(Box::new(LoweredOp::Arctan(Box::new(LoweredOp::Var(0)))));
        assert_eq!(simplify_op(&op), LoweredOp::Var(0));
    }

    #[test]
    fn sqrt_x_squared_to_abs() {
        // sqrt(x²) → |x|
        let op = LoweredOp::Sqrt(Box::new(LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(2.0)),
        )));
        assert_eq!(
            simplify_op(&op),
            LoweredOp::Abs(Box::new(LoweredOp::Var(0)))
        );
    }

    #[test]
    fn x_minus_x_is_zero() {
        let x = LoweredOp::Var(0);
        let op = LoweredOp::Sub(Box::new(x.clone()), Box::new(x));
        assert_eq!(simplify_op(&op), LoweredOp::Const(0.0));
    }

    // ----------------------------------------------------------------
    // Sign canonicalisation
    // ----------------------------------------------------------------

    #[test]
    fn neg_one_times_x_to_neg_x() {
        let op = LoweredOp::Mul(
            Box::new(LoweredOp::Const(-1.0)),
            Box::new(LoweredOp::Var(0)),
        );
        assert_eq!(
            simplify_op(&op),
            LoweredOp::Neg(Box::new(LoweredOp::Var(0)))
        );
    }

    #[test]
    fn zero_minus_x_to_neg_x() {
        let op = LoweredOp::Sub(Box::new(LoweredOp::Const(0.0)), Box::new(LoweredOp::Var(0)));
        assert_eq!(
            simplify_op(&op),
            LoweredOp::Neg(Box::new(LoweredOp::Var(0)))
        );
    }

    #[test]
    fn a_minus_neg_b_to_a_plus_b() {
        // a - (-b) → a + b. With hash-canonical Add ordering, the result
        // becomes Add(min(a,b), max(a,b)). With Var(0) and Var(1), the
        // structural-hash ordering is deterministic — we just check the
        // op is now Add (no more Neg under Sub).
        let op = LoweredOp::Sub(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Neg(Box::new(LoweredOp::Var(1)))),
        );
        let s = simplify_op(&op);
        assert!(matches!(s, LoweredOp::Add(_, _)));
    }

    // ----------------------------------------------------------------
    // Hash-based commutative ordering
    // ----------------------------------------------------------------

    #[test]
    fn add_commutative_canonical() {
        // Add(Var(0), Var(1)) and Add(Var(1), Var(0)) should simplify to
        // the same canonical form.
        let a = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let b = LoweredOp::Add(Box::new(LoweredOp::Var(1)), Box::new(LoweredOp::Var(0)));
        let sa = simplify_op(&a);
        let sb = simplify_op(&b);
        assert_eq!(sa, sb);
    }

    #[test]
    fn mul_commutative_canonical() {
        let a = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let b = LoweredOp::Mul(Box::new(LoweredOp::Var(1)), Box::new(LoweredOp::Var(0)));
        let sa = simplify_op(&a);
        let sb = simplify_op(&b);
        assert_eq!(sa, sb);
    }

    // ----------------------------------------------------------------
    // Idempotence + edge cases
    // ----------------------------------------------------------------

    #[test]
    fn idempotent() {
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(1.0)),
            )),
            Box::new(LoweredOp::Const(0.0)),
        );
        let s1 = simplify_op(&op);
        let s2 = simplify_op(&s1);
        assert_eq!(s1, s2);
    }

    #[test]
    fn deep_left_chain_no_overflow() {
        let mut op = LoweredOp::Var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Const(0.0)));
        }
        let simplified = simplify_op(&op);
        assert_eq!(simplified, LoweredOp::Var(0));
    }

    #[test]
    fn deep_right_chain_no_overflow() {
        let mut op = LoweredOp::Var(0);
        for _ in 0..1000 {
            op = LoweredOp::Mul(Box::new(LoweredOp::Const(1.0)), Box::new(op));
        }
        let simplified = simplify_op(&op);
        assert_eq!(simplified, LoweredOp::Var(0));
    }

    // ----------------------------------------------------------------
    // Idempotence proptest
    // ----------------------------------------------------------------

    /// Generate a small random `LoweredOp` tree (depth ≤ 4) for proptest.
    fn arb_op() -> impl Strategy<Value = LoweredOp> {
        let leaf = prop_oneof![
            (-5.0f64..5.0f64).prop_map(LoweredOp::Const),
            (0usize..3usize).prop_map(LoweredOp::Var),
        ];
        leaf.prop_recursive(
            4,  // depth
            16, // max nodes
            8,  // expected items per collection
            |inner| {
                prop_oneof![
                    (inner.clone(), inner.clone())
                        .prop_map(|(a, b)| LoweredOp::Add(Box::new(a), Box::new(b))),
                    (inner.clone(), inner.clone())
                        .prop_map(|(a, b)| LoweredOp::Sub(Box::new(a), Box::new(b))),
                    (inner.clone(), inner.clone())
                        .prop_map(|(a, b)| LoweredOp::Mul(Box::new(a), Box::new(b))),
                    inner.clone().prop_map(|c| LoweredOp::Neg(Box::new(c))),
                    inner.clone().prop_map(|c| LoweredOp::Sin(Box::new(c))),
                    inner.clone().prop_map(|c| LoweredOp::Cos(Box::new(c))),
                    inner.prop_map(|c| LoweredOp::Exp(Box::new(c))),
                ]
            },
        )
    }

    proptest! {
        #[test]
        fn simplify_is_idempotent(op in arb_op()) {
            let s1 = simplify_op(&op);
            let s2 = simplify_op(&s1);
            // Compare via structural hash — equal hashes ⇔ equal trees
            // under our canonical ordering. (Direct `==` would also work
            // here because PartialEq is structural, but hashes are the
            // intended canonical equality for downstream cache keys.)
            prop_assert_eq!(s1.structural_hash(), s2.structural_hash());
        }
    }
}
