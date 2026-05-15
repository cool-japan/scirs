//! Property-based tests for `cas` rewrite correctness using proptest.
//!
//! Three properties are checked:
//!
//! 1. **Idempotence** — `canonicalize(canonicalize(t)) == canonicalize(t)` for all trees.
//! 2. **Identity-db hash stability** — applying `apply_identity_db` (even when no rule
//!    fires) does not change the canonical hash.
//! 3. **Simplify semantics preservation** — `simplify_op` does not change the evaluated
//!    value at a fixed real point for finite, well-defined expressions.
//!
//! 1024 cases per property; proptest persists failing seeds in
//! `proptest-regressions/cas_rewrite_proptest.txt` for CI reproducibility.

use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;

use scirs2_symbolic::cas::{apply_identity_db, canonicalize, IdentityDb};
use scirs2_symbolic::eml::{eval_real, simplify_op, EvalCtx, LoweredOp};

// ---------------------------------------------------------------------------
// Arbitrary `LoweredOp` strategy
//
// Restrictions to avoid spurious eval failures:
//  - No `Div` or `Ln` — they can produce NaN / Inf during eval.
//  - `Var(i)` limited to i < 3 — evaluation uses a 3-element binding array.
//  - Constants in (0.01, 10.1) — positive to keep `Sin` / `Cos` domain-safe
//    and avoid `sqrt` domain violation if that op were ever added.
//  - Integer exponents in 1..=3 for `Pow` — avoids fractional-power NaN.
// ---------------------------------------------------------------------------

/// Build a proptest strategy for random `LoweredOp` trees up to the given depth.
///
/// Returns a `BoxedStrategy<LoweredOp>` so both the `depth == 0` leaf case and
/// the recursive case unify to the same concrete type.
fn arb_lowered_op(depth: u32) -> BoxedStrategy<LoweredOp> {
    if depth == 0 {
        // Only leaves.
        prop_oneof![
            // Positive constant, well away from 0 to avoid domain violations.
            (1u32..101u32).prop_map(|v| LoweredOp::Const(v as f64 * 0.1 + 0.01)),
            // Variable index strictly < 3 (evaluation point has 3 slots).
            (0usize..3usize).prop_map(LoweredOp::Var),
        ]
        .boxed()
    } else {
        prop_oneof![
            // Leaves (weight 3) — bias toward leaves to keep average depth reasonable.
            3 => arb_lowered_op(0),
            // Binary ops (weight 5 total).
            1 => (arb_lowered_op(depth - 1), arb_lowered_op(depth - 1))
                .prop_map(|(a, b)| LoweredOp::Add(Box::new(a), Box::new(b))),
            1 => (arb_lowered_op(depth - 1), arb_lowered_op(depth - 1))
                .prop_map(|(a, b)| LoweredOp::Mul(Box::new(a), Box::new(b))),
            1 => (arb_lowered_op(depth - 1), arb_lowered_op(depth - 1))
                .prop_map(|(a, b)| LoweredOp::Sub(Box::new(a), Box::new(b))),
            // Unary ops (weight 2 total).
            1 => arb_lowered_op(depth - 1).prop_map(|a| LoweredOp::Neg(Box::new(a))),
            1 => arb_lowered_op(depth - 1).prop_map(|a| LoweredOp::Sin(Box::new(a))),
            // Pow with small positive integer exponent — prevents NaN cascade on
            // negative bases that would arise from fractional exponents.
            1 => (arb_lowered_op(depth - 1), (1i64..=3i64))
                .prop_map(|(a, n)| LoweredOp::Pow(Box::new(a), Box::new(LoweredOp::Const(n as f64)))),
        ]
        .boxed()
    }
}

// ---------------------------------------------------------------------------
// Fixed evaluation point (3 variables, all positive and well-separated).
// ---------------------------------------------------------------------------

const EVAL_POINT: [f64; 3] = [1.5, 0.5, 2.0];

// ---------------------------------------------------------------------------
// Property 1: canonicalize is idempotent
//
// For every generated tree `t`:
//   canonicalize(canonicalize(t).op()).hash() == canonicalize(t).hash()
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn prop_canonicalize_idempotent(op in arb_lowered_op(4)) {
        let c1 = canonicalize(&op);
        let c2 = canonicalize(c1.op());
        prop_assert_eq!(
            c1.hash(),
            c2.hash(),
            "canonicalize(canonicalize(op)) != canonicalize(op) for op: {:?}",
            op
        );
    }
}

// ---------------------------------------------------------------------------
// Property 2: identity_db does not change the canonical hash
//
// `apply_identity_db` is a semantics-preserving rewrite step.  Whether or
// not any rule fires, the canonical hash of the result must equal the
// canonical hash of the input.  (This is an "identity_db is hash-stable"
// invariant; random sampling rarely produces exact rule-LHS shapes, so
// the property also confirms that the no-op path does not corrupt the hash.)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn prop_identity_db_preserves_canonical(op in arb_lowered_op(3)) {
        let db = IdentityDb::standard();
        let rewritten = apply_identity_db(&db, &op);
        let hash_original = canonicalize(&op).hash();
        let hash_rewritten = canonicalize(&rewritten).hash();
        prop_assert_eq!(
            hash_original,
            hash_rewritten,
            "identity_db changed canonical hash:\n  original  {:?}\n  rewritten {:?}",
            op,
            rewritten
        );
    }
}

// ---------------------------------------------------------------------------
// Property 3: simplify_op preserves the evaluated value at a fixed real point
//
// For any tree that evaluates to a finite value before and after simplification,
// the two values must agree to within a relative tolerance of 1e-9.
// NaN / Inf / domain errors are silently skipped — they are numerical failures,
// not simplification bugs.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn prop_simplify_preserves_eval(op in arb_lowered_op(3)) {
        let ctx = EvalCtx::new(&EVAL_POINT);
        let simplified = simplify_op(&op);
        let v1 = eval_real(&op, &ctx);
        let v2 = eval_real(&simplified, &ctx);
        match (v1, v2) {
            (Ok(a), Ok(b)) if a.is_finite() && b.is_finite() => {
                // Use a relative + absolute tolerance to handle values near zero.
                let tol = 1e-9 * (1.0 + a.abs() + b.abs());
                prop_assert!(
                    (a - b).abs() <= tol,
                    "simplify changed value: {} -> {} (diff {}, tol {}) for op {:?}",
                    a,
                    b,
                    (a - b).abs(),
                    tol,
                    op
                );
            }
            // Eval failed (domain error, unbound var, etc.) or produced non-finite
            // value on at least one side — skip, this is not a simplify bug.
            _ => {}
        }
    }
}
