//! Pattern matching and instantiation for symbolic rewriting.
//!
//! A [`Pattern`] is like a [`LoweredOp`] but with wildcard variables (`PatVar(n)`)
//! that can bind to any subexpression. [`match_pattern`] tries to match a pattern
//! against a concrete `LoweredOp` tree, filling a [`Bindings`] map. [`instantiate`]
//! replaces wildcards with their bound expressions.
//!
//! # Wildcard consistency
//!
//! If the same wildcard `?n` appears twice in a pattern, both occurrences must
//! match structurally-identical subexpressions (compared by structural hash).
//! This guarantees, e.g., that `sin²(?0) + cos²(?0)` only matches when the
//! same argument appears in both.
//!
//! # No recursion
//!
//! Like the rest of the EML stack, every traversal here is iterative
//! (work-stack pattern). A deeply nested `LoweredOp` or `Pattern` tree must
//! not blow the OS stack.

#![warn(missing_docs)]

use crate::eml::LoweredOp;
use hashbrown::HashMap;

/// Binding map: wildcard index → matched subexpression.
pub type Bindings = HashMap<u32, LoweredOp>;

/// A pattern for matching [`LoweredOp`] trees.
///
/// `PatVar(n)` is a wildcard that captures any subexpression and binds it to `n`.
/// `PatConstInt(n)` captures a `LoweredOp::Const` whose value equals the integer `n`.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Wildcard: matches any subexpression, binds to the given index.
    PatVar(u32),
    /// Matches `LoweredOp::Const(v)` where `(v - pv).abs() <= f64::EPSILON * pv.abs().max(1.0)`.
    PatConst(f64),
    /// Matches `LoweredOp::Const(v)` where `v` is an integer equal to `n`.
    PatConstInt(u32),
    /// Matches `LoweredOp::Var(i)` exactly.
    PatGroundVar(usize),
    /// Matches a unary op with the given [`UnaryKind`] and child pattern.
    PatOp1(UnaryKind, Box<Pattern>),
    /// Matches a binary op with the given [`BinaryKind`] and two child patterns.
    PatOp2(BinaryKind, Box<Pattern>, Box<Pattern>),
}

/// Unary operator kinds mirroring the unary variants of [`LoweredOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryKind {
    /// Negation: `-x`
    Neg,
    /// Natural exponential: `e^x`
    Exp,
    /// Natural logarithm: `ln(x)`
    Ln,
    /// Sine
    Sin,
    /// Cosine
    Cos,
    /// Tangent
    Tan,
    /// Hyperbolic sine
    Sinh,
    /// Hyperbolic cosine
    Cosh,
    /// Hyperbolic tangent
    Tanh,
    /// Arcsine
    Arcsin,
    /// Arccosine
    Arccos,
    /// Arctangent
    Arctan,
    /// Hyperbolic arcsine
    Arcsinh,
    /// Hyperbolic arccosine
    Arccosh,
    /// Hyperbolic arctangent
    Arctanh,
    /// Square root
    Sqrt,
    /// Absolute value
    Abs,
}

/// Binary operator kinds mirroring the binary variants of [`LoweredOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryKind {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Exponentiation: `base^exp`
    Pow,
}

/// Error returned by [`instantiate`].
#[derive(Debug, Clone)]
pub enum PatternError {
    /// A `PatVar(n)` in the pattern was not present in `bindings`.
    MissingBinding(u32),
}

impl std::fmt::Display for PatternError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternError::MissingBinding(n) => write!(f, "missing binding for wildcard ?{n}"),
        }
    }
}

impl std::error::Error for PatternError {}

/// Try to match `pat` against `op`, filling `out` with wildcard bindings.
///
/// Returns `false` if the shape doesn't match or a wildcard is bound
/// to two structurally-different subexpressions (consistency check via
/// [`LoweredOp::structural_hash`]).
///
/// Uses an iterative work-stack — no recursion.
///
/// # Example
///
/// ```rust
/// use scirs2_symbolic::cas::pattern::{match_pattern, Pattern, BinaryKind, Bindings};
/// use scirs2_symbolic::eml::LoweredOp;
///
/// let op = LoweredOp::Add(
///     Box::new(LoweredOp::Var(0)),
///     Box::new(LoweredOp::Const(1.0)),
/// );
/// let pat = Pattern::PatOp2(
///     BinaryKind::Add,
///     Box::new(Pattern::PatVar(0)),
///     Box::new(Pattern::PatConst(1.0)),
/// );
/// let mut bindings = Bindings::new();
/// assert!(match_pattern(&pat, &op, &mut bindings));
/// assert_eq!(bindings[&0], LoweredOp::Var(0));
/// ```
pub fn match_pattern(pat: &Pattern, op: &LoweredOp, out: &mut Bindings) -> bool {
    // Stack of (pattern, concrete_op) pairs to process.
    let mut stack: Vec<(Pattern, LoweredOp)> = vec![(pat.clone(), op.clone())];

    while let Some((p, o)) = stack.pop() {
        match (p, o) {
            (Pattern::PatVar(n), o) => {
                // Check consistency: if already bound, new op must have same structural hash.
                if let Some(prev) = out.get(&n) {
                    if prev.structural_hash() != o.structural_hash() {
                        return false;
                    }
                } else {
                    out.insert(n, o);
                }
            }

            (Pattern::PatConst(pv), LoweredOp::Const(cv)) => {
                if (pv - cv).abs() > f64::EPSILON * pv.abs().max(1.0) {
                    return false;
                }
            }

            (Pattern::PatConstInt(n), LoweredOp::Const(cv)) => {
                let expected = n as f64;
                if cv.fract() != 0.0 || (cv - expected).abs() > f64::EPSILON {
                    return false;
                }
            }

            (Pattern::PatGroundVar(pi), LoweredOp::Var(oi)) => {
                if pi != oi {
                    return false;
                }
            }

            (Pattern::PatOp1(pk, pc), o) => {
                match (pk, o) {
                    (UnaryKind::Neg, LoweredOp::Neg(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Exp, LoweredOp::Exp(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Ln, LoweredOp::Ln(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Sin, LoweredOp::Sin(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Cos, LoweredOp::Cos(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Tan, LoweredOp::Tan(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Sinh, LoweredOp::Sinh(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Cosh, LoweredOp::Cosh(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Tanh, LoweredOp::Tanh(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Arcsin, LoweredOp::Arcsin(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Arccos, LoweredOp::Arccos(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Arctan, LoweredOp::Arctan(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Arcsinh, LoweredOp::Arcsinh(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Arccosh, LoweredOp::Arccosh(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Arctanh, LoweredOp::Arctanh(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Sqrt, LoweredOp::Sqrt(c)) => stack.push((*pc, *c)),
                    (UnaryKind::Abs, LoweredOp::Abs(c)) => stack.push((*pc, *c)),
                    // UnaryKind does not match the concrete op's variant — shape mismatch.
                    _ => return false,
                }
            }

            (Pattern::PatOp2(pk, pl, pr), o) => {
                match (pk, o) {
                    (BinaryKind::Add, LoweredOp::Add(l, r)) => {
                        stack.push((*pl, *l));
                        stack.push((*pr, *r));
                    }
                    (BinaryKind::Sub, LoweredOp::Sub(l, r)) => {
                        stack.push((*pl, *l));
                        stack.push((*pr, *r));
                    }
                    (BinaryKind::Mul, LoweredOp::Mul(l, r)) => {
                        stack.push((*pl, *l));
                        stack.push((*pr, *r));
                    }
                    (BinaryKind::Div, LoweredOp::Div(l, r)) => {
                        stack.push((*pl, *l));
                        stack.push((*pr, *r));
                    }
                    (BinaryKind::Pow, LoweredOp::Pow(l, r)) => {
                        stack.push((*pl, *l));
                        stack.push((*pr, *r));
                    }
                    // BinaryKind does not match the concrete op's variant — shape mismatch.
                    _ => return false,
                }
            }

            // Shape mismatch between pattern leaf and concrete op variant.
            _ => return false,
        }
    }
    true
}

/// Substitute all wildcards in `pat` with their bindings from `bindings`.
///
/// Uses an iterative post-order frame stack — no recursion.
///
/// Returns `Err(PatternError::MissingBinding(n))` if any `PatVar(n)` is
/// not present in `bindings`.
///
/// # Example
///
/// ```rust
/// use scirs2_symbolic::cas::pattern::{
///     instantiate, match_pattern, Pattern, BinaryKind, Bindings,
/// };
/// use scirs2_symbolic::eml::LoweredOp;
///
/// // Pattern: ?0 + ?1
/// let pat = Pattern::PatOp2(
///     BinaryKind::Add,
///     Box::new(Pattern::PatVar(0)),
///     Box::new(Pattern::PatVar(1)),
/// );
/// let mut bindings = Bindings::new();
/// bindings.insert(0, LoweredOp::Var(0));
/// bindings.insert(1, LoweredOp::Const(2.0));
///
/// let result = instantiate(&pat, &bindings).unwrap();
/// assert_eq!(
///     result,
///     LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(2.0))),
/// );
/// ```
pub fn instantiate(pat: &Pattern, bindings: &Bindings) -> Result<LoweredOp, PatternError> {
    // Post-order frame stack. Each frame is either:
    //   Process(Pattern)   — recurse into children, then build
    //   BuildOp1(kind)     — pop one result from op_stack, wrap it
    //   BuildOp2(kind)     — pop two results (right then left), combine
    //
    // Frame stack is LIFO. For `PatOp2(kind, left, right)` we push:
    //   BuildOp2(kind) → Process(right) → Process(left)   [in this order]
    // so Process(left) pops first → left_result on op_stack first,
    // then Process(right) → right_result on op_stack second,
    // then BuildOp2 pops right then left (matching constructor order).
    enum Frame {
        Process(Pattern),
        BuildOp1(UnaryKind),
        BuildOp2(BinaryKind),
    }

    let mut frames: Vec<Frame> = vec![Frame::Process(pat.clone())];
    let mut op_stack: Vec<LoweredOp> = Vec::new();

    while let Some(frame) = frames.pop() {
        match frame {
            Frame::Process(p) => match p {
                Pattern::PatVar(n) => {
                    let op = bindings
                        .get(&n)
                        .ok_or(PatternError::MissingBinding(n))?
                        .clone();
                    op_stack.push(op);
                }
                Pattern::PatConst(v) => op_stack.push(LoweredOp::Const(v)),
                Pattern::PatConstInt(n) => op_stack.push(LoweredOp::Const(n as f64)),
                Pattern::PatGroundVar(i) => op_stack.push(LoweredOp::Var(i)),
                Pattern::PatOp1(kind, child) => {
                    frames.push(Frame::BuildOp1(kind));
                    frames.push(Frame::Process(*child));
                }
                Pattern::PatOp2(kind, left, right) => {
                    // Push build frame first (processed last), then right, then left
                    // so left is processed first, placing its result on op_stack before right's.
                    frames.push(Frame::BuildOp2(kind));
                    frames.push(Frame::Process(*right));
                    frames.push(Frame::Process(*left));
                }
            },

            Frame::BuildOp1(kind) => {
                let child = op_stack
                    .pop()
                    .expect("post-order invariant: child on op_stack for BuildOp1");
                let result = build_op1(kind, child);
                op_stack.push(result);
            }

            Frame::BuildOp2(kind) => {
                // op_stack has [..., left_result, right_result]
                // left was processed (pushed) first, right second.
                let right = op_stack
                    .pop()
                    .expect("post-order invariant: right child on op_stack for BuildOp2");
                let left = op_stack
                    .pop()
                    .expect("post-order invariant: left child on op_stack for BuildOp2");
                let result = build_op2(kind, left, right);
                op_stack.push(result);
            }
        }
    }

    op_stack
        .pop()
        .expect("post-order invariant: final result on op_stack after instantiate")
        .pipe_ok()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Wrap `child` with the given `UnaryKind`.
fn build_op1(kind: UnaryKind, child: LoweredOp) -> LoweredOp {
    let b = Box::new(child);
    match kind {
        UnaryKind::Neg => LoweredOp::Neg(b),
        UnaryKind::Exp => LoweredOp::Exp(b),
        UnaryKind::Ln => LoweredOp::Ln(b),
        UnaryKind::Sin => LoweredOp::Sin(b),
        UnaryKind::Cos => LoweredOp::Cos(b),
        UnaryKind::Tan => LoweredOp::Tan(b),
        UnaryKind::Sinh => LoweredOp::Sinh(b),
        UnaryKind::Cosh => LoweredOp::Cosh(b),
        UnaryKind::Tanh => LoweredOp::Tanh(b),
        UnaryKind::Arcsin => LoweredOp::Arcsin(b),
        UnaryKind::Arccos => LoweredOp::Arccos(b),
        UnaryKind::Arctan => LoweredOp::Arctan(b),
        UnaryKind::Arcsinh => LoweredOp::Arcsinh(b),
        UnaryKind::Arccosh => LoweredOp::Arccosh(b),
        UnaryKind::Arctanh => LoweredOp::Arctanh(b),
        UnaryKind::Sqrt => LoweredOp::Sqrt(b),
        UnaryKind::Abs => LoweredOp::Abs(b),
    }
}

/// Combine `left` and `right` with the given `BinaryKind`.
fn build_op2(kind: BinaryKind, left: LoweredOp, right: LoweredOp) -> LoweredOp {
    let (bl, br) = (Box::new(left), Box::new(right));
    match kind {
        BinaryKind::Add => LoweredOp::Add(bl, br),
        BinaryKind::Sub => LoweredOp::Sub(bl, br),
        BinaryKind::Mul => LoweredOp::Mul(bl, br),
        BinaryKind::Div => LoweredOp::Div(bl, br),
        BinaryKind::Pow => LoweredOp::Pow(bl, br),
    }
}

/// Trivial identity adapter: wraps a value in `Ok`.
///
/// Defined to avoid the awkward `Ok(op_stack.pop().expect(...))` double-wrapping.
trait PipeOk: Sized {
    fn pipe_ok<E>(self) -> Result<Self, E>;
}

impl PipeOk for LoweredOp {
    fn pipe_ok<E>(self) -> Result<Self, E> {
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::LoweredOp;

    // -----------------------------------------------------------------------
    // match_pattern tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_wildcard_match_binds() {
        // PatVar(0) matches any concrete op and stores it.
        let op = LoweredOp::Var(3);
        let pat = Pattern::PatVar(0);
        let mut bindings = Bindings::new();
        assert!(match_pattern(&pat, &op, &mut bindings));
        assert_eq!(bindings[&0], LoweredOp::Var(3));
    }

    #[test]
    fn test_repeated_wildcard_consistency_same() {
        // PatOp2(Add, PatVar(0), PatVar(0)) should match Add(Var(1), Var(1))
        // because both occurrences of ?0 bind to Var(1) consistently.
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(1)), Box::new(LoweredOp::Var(1)));
        let pat = Pattern::PatOp2(
            BinaryKind::Add,
            Box::new(Pattern::PatVar(0)),
            Box::new(Pattern::PatVar(0)),
        );
        let mut bindings = Bindings::new();
        assert!(match_pattern(&pat, &op, &mut bindings));
        assert_eq!(bindings[&0], LoweredOp::Var(1));
    }

    #[test]
    fn test_repeated_wildcard_rejects_mismatch() {
        // PatOp2(Add, PatVar(0), PatVar(0)) must NOT match Add(Var(1), Var(2))
        // because the two occurrences of ?0 would bind to different subexpressions.
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(1)), Box::new(LoweredOp::Var(2)));
        let pat = Pattern::PatOp2(
            BinaryKind::Add,
            Box::new(Pattern::PatVar(0)),
            Box::new(Pattern::PatVar(0)),
        );
        let mut bindings = Bindings::new();
        assert!(!match_pattern(&pat, &op, &mut bindings));
    }

    #[test]
    fn test_const_literal_match() {
        // PatConst(1.23456) matches Const(1.23456), not Const(7.89).
        let op_ok = LoweredOp::Const(1.23456);
        let op_fail = LoweredOp::Const(7.89);
        let pat = Pattern::PatConst(1.23456);
        let mut b1 = Bindings::new();
        let mut b2 = Bindings::new();
        assert!(match_pattern(&pat, &op_ok, &mut b1));
        assert!(!match_pattern(&pat, &op_fail, &mut b2));
    }

    #[test]
    fn test_const_int_match() {
        // PatConstInt(2) matches Const(2.0) but not Const(2.5) or Const(3.0).
        let pat = Pattern::PatConstInt(2);
        let op_ok = LoweredOp::Const(2.0);
        let op_frac = LoweredOp::Const(2.5);
        let op_wrong = LoweredOp::Const(3.0);
        let mut b = Bindings::new();
        assert!(match_pattern(&pat, &op_ok, &mut b));
        let mut b2 = Bindings::new();
        assert!(!match_pattern(&pat, &op_frac, &mut b2));
        let mut b3 = Bindings::new();
        assert!(!match_pattern(&pat, &op_wrong, &mut b3));
    }

    #[test]
    fn test_ground_var_match() {
        // PatGroundVar(0) matches Var(0) but not Var(1).
        let pat = Pattern::PatGroundVar(0);
        let mut b1 = Bindings::new();
        assert!(match_pattern(&pat, &LoweredOp::Var(0), &mut b1));
        let mut b2 = Bindings::new();
        assert!(!match_pattern(&pat, &LoweredOp::Var(1), &mut b2));
    }

    #[test]
    fn test_op_shape_match() {
        // PatOp2(Mul, PatVar(0), PatVar(1)) matches Mul(Var(0), Const(2.0)).
        let op = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(2.0)));
        let pat = Pattern::PatOp2(
            BinaryKind::Mul,
            Box::new(Pattern::PatVar(0)),
            Box::new(Pattern::PatVar(1)),
        );
        let mut bindings = Bindings::new();
        assert!(match_pattern(&pat, &op, &mut bindings));
        assert_eq!(bindings[&0], LoweredOp::Var(0));
        assert_eq!(bindings[&1], LoweredOp::Const(2.0));
    }

    #[test]
    fn test_shape_mismatch_returns_false() {
        // PatOp2(Add, ...) does NOT match Mul(...).
        let op = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let pat = Pattern::PatOp2(
            BinaryKind::Add,
            Box::new(Pattern::PatVar(0)),
            Box::new(Pattern::PatVar(1)),
        );
        let mut bindings = Bindings::new();
        assert!(!match_pattern(&pat, &op, &mut bindings));
    }

    #[test]
    fn test_unary_op_match() {
        // PatOp1(Sin, PatVar(0)) matches Sin(Var(2)).
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(2)));
        let pat = Pattern::PatOp1(UnaryKind::Sin, Box::new(Pattern::PatVar(0)));
        let mut bindings = Bindings::new();
        assert!(match_pattern(&pat, &op, &mut bindings));
        assert_eq!(bindings[&0], LoweredOp::Var(2));
    }

    #[test]
    fn test_unary_kind_mismatch() {
        // PatOp1(Cos, PatVar(0)) does NOT match Sin(Var(0)).
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let pat = Pattern::PatOp1(UnaryKind::Cos, Box::new(Pattern::PatVar(0)));
        let mut bindings = Bindings::new();
        assert!(!match_pattern(&pat, &op, &mut bindings));
    }

    // -----------------------------------------------------------------------
    // instantiate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_instantiate_wildcard() {
        // instantiate PatVar(0) with binding {0 → Var(5)} → Var(5).
        let pat = Pattern::PatVar(0);
        let mut bindings = Bindings::new();
        bindings.insert(0, LoweredOp::Var(5));
        let result = instantiate(&pat, &bindings).unwrap();
        assert_eq!(result, LoweredOp::Var(5));
    }

    #[test]
    fn test_instantiate_missing_binding_returns_err() {
        let pat = Pattern::PatVar(99);
        let bindings = Bindings::new();
        let result = instantiate(&pat, &bindings);
        assert!(matches!(result, Err(PatternError::MissingBinding(99))));
    }

    #[test]
    fn test_instantiate_round_trip() {
        // Build a concrete op, match it against a pattern, instantiate — should recover.
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Sin(Box::new(LoweredOp::Var(0)))),
            Box::new(LoweredOp::Const(1.0)),
        );
        let pat = Pattern::PatOp2(
            BinaryKind::Add,
            Box::new(Pattern::PatOp1(
                UnaryKind::Sin,
                Box::new(Pattern::PatVar(0)),
            )),
            Box::new(Pattern::PatConst(1.0)),
        );
        let mut bindings = Bindings::new();
        assert!(match_pattern(&pat, &op, &mut bindings));
        let result = instantiate(&pat, &bindings).unwrap();
        assert_eq!(result, op);
    }

    #[test]
    fn test_instantiate_const_variants() {
        // PatConst and PatConstInt instantiate without bindings.
        let pat_c = Pattern::PatConst(99.1234567890);
        let pat_i = Pattern::PatConstInt(3);
        let bindings = Bindings::new();
        assert_eq!(
            instantiate(&pat_c, &bindings).unwrap(),
            LoweredOp::Const(99.1234567890)
        );
        assert_eq!(
            instantiate(&pat_i, &bindings).unwrap(),
            LoweredOp::Const(3.0)
        );
    }

    // -----------------------------------------------------------------------
    // No-overflow / iterative depth tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_deep_concrete_op_no_overflow() {
        // Build a 1000-deep Add(_, Const(0.0)) chain.
        // match_pattern with outermost PatOp2(Add, PatVar(0), PatConst(0.0))
        // should bind ?0 to the 999-deep inner subtree — no stack overflow.
        let mut op = LoweredOp::Var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Const(0.0)));
        }
        let pat = Pattern::PatOp2(
            BinaryKind::Add,
            Box::new(Pattern::PatVar(0)),
            Box::new(Pattern::PatConst(0.0)),
        );
        let mut bindings = Bindings::new();
        assert!(match_pattern(&pat, &op, &mut bindings));
        assert!(bindings.contains_key(&0));
    }

    #[test]
    fn test_deep_pattern_instantiate_no_overflow() {
        // Build a 500-deep Pattern chain: PatOp2(Add, PatOp2(Add, ..., PatConst(0.0)), PatConst(0.0))
        // bound to a matching LoweredOp. instantiate must not overflow the OS stack.
        let mut pat = Pattern::PatVar(0);
        let mut op = LoweredOp::Var(0);
        for _ in 0..500 {
            pat = Pattern::PatOp2(
                BinaryKind::Add,
                Box::new(pat),
                Box::new(Pattern::PatConst(0.0)),
            );
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Const(0.0)));
        }
        // Bind wildcard 0 to Var(0).
        let mut bindings = Bindings::new();
        bindings.insert(0, LoweredOp::Var(0));
        // instantiate should succeed and produce the same tree.
        let result = instantiate(&pat, &bindings).unwrap();
        assert_eq!(result.structural_hash(), op.structural_hash());
    }

    #[test]
    fn test_all_unary_kinds_roundtrip() {
        // Each UnaryKind must correctly match and instantiate.
        let kinds = [
            (UnaryKind::Neg, LoweredOp::Neg(Box::new(LoweredOp::Var(0)))),
            (UnaryKind::Exp, LoweredOp::Exp(Box::new(LoweredOp::Var(0)))),
            (UnaryKind::Ln, LoweredOp::Ln(Box::new(LoweredOp::Var(0)))),
            (UnaryKind::Sin, LoweredOp::Sin(Box::new(LoweredOp::Var(0)))),
            (UnaryKind::Cos, LoweredOp::Cos(Box::new(LoweredOp::Var(0)))),
            (UnaryKind::Tan, LoweredOp::Tan(Box::new(LoweredOp::Var(0)))),
            (
                UnaryKind::Sinh,
                LoweredOp::Sinh(Box::new(LoweredOp::Var(0))),
            ),
            (
                UnaryKind::Cosh,
                LoweredOp::Cosh(Box::new(LoweredOp::Var(0))),
            ),
            (
                UnaryKind::Tanh,
                LoweredOp::Tanh(Box::new(LoweredOp::Var(0))),
            ),
            (
                UnaryKind::Arcsin,
                LoweredOp::Arcsin(Box::new(LoweredOp::Var(0))),
            ),
            (
                UnaryKind::Arccos,
                LoweredOp::Arccos(Box::new(LoweredOp::Var(0))),
            ),
            (
                UnaryKind::Arctan,
                LoweredOp::Arctan(Box::new(LoweredOp::Var(0))),
            ),
            (
                UnaryKind::Arcsinh,
                LoweredOp::Arcsinh(Box::new(LoweredOp::Var(0))),
            ),
            (
                UnaryKind::Arccosh,
                LoweredOp::Arccosh(Box::new(LoweredOp::Var(0))),
            ),
            (
                UnaryKind::Arctanh,
                LoweredOp::Arctanh(Box::new(LoweredOp::Var(0))),
            ),
            (
                UnaryKind::Sqrt,
                LoweredOp::Sqrt(Box::new(LoweredOp::Var(0))),
            ),
            (UnaryKind::Abs, LoweredOp::Abs(Box::new(LoweredOp::Var(0)))),
        ];
        for (kind, concrete) in kinds {
            let pat = Pattern::PatOp1(kind, Box::new(Pattern::PatVar(0)));
            let mut bindings = Bindings::new();
            assert!(
                match_pattern(&pat, &concrete, &mut bindings),
                "failed to match {kind:?}"
            );
            let result = instantiate(&pat, &bindings).unwrap();
            assert_eq!(
                result.structural_hash(),
                concrete.structural_hash(),
                "instantiate mismatch for {kind:?}"
            );
        }
    }

    #[test]
    fn test_all_binary_kinds_roundtrip() {
        let kinds = [
            (
                BinaryKind::Add,
                LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
            ),
            (
                BinaryKind::Sub,
                LoweredOp::Sub(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
            ),
            (
                BinaryKind::Mul,
                LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
            ),
            (
                BinaryKind::Div,
                LoweredOp::Div(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
            ),
            (
                BinaryKind::Pow,
                LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1))),
            ),
        ];
        for (kind, concrete) in kinds {
            let pat = Pattern::PatOp2(
                kind,
                Box::new(Pattern::PatVar(0)),
                Box::new(Pattern::PatVar(1)),
            );
            let mut bindings = Bindings::new();
            assert!(
                match_pattern(&pat, &concrete, &mut bindings),
                "failed to match {kind:?}"
            );
            let result = instantiate(&pat, &bindings).unwrap();
            assert_eq!(
                result.structural_hash(),
                concrete.structural_hash(),
                "instantiate mismatch for {kind:?}"
            );
        }
    }
}
