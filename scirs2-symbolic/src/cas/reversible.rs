//! Reversible CAS trace — record rewrite steps during canonicalization and
//! optionally replay them in reverse to recover the original expression.
//!
//! # Design: Option B (batch-pass tracing)
//!
//! The `canonical_rules.rs` module exposes one monolithic function,
//! `apply_canonical_rules`, which performs multiple named sub-rules inside a
//! single bottom-up pass. Similarly, constant folding, commutativity sorting,
//! and identity rules (`add_zero`, `mul_by_one`, etc.) all live inside
//! `simplify_op` in `eml/simplify.rs`, not in `canonical_rules.rs`.
//!
//! Because there is no per-rule granularity exposed at the public API boundary,
//! this module records one [`RewriteStep`] per outer fixed-point iteration of
//! the pipeline (batch-pass strategy). Every batch step is marked
//! `is_reversible = false`, meaning [`RewriteTrace::reverse`] returns `Some`
//! **only** when the trace is empty (i.e. the expression was already at the
//! canonical fixed point).
//!
//! # Future work (v0.4.5+)
//!
//! Per-rule granularity can be added by refactoring `apply_canonical_rules` to
//! return a list of `(new_op, rule_id, is_reversible)` triples, and similarly
//! exposing rule-level callbacks from `simplify_op`. At that point, Option A
//! (full per-rule recording) becomes practical and `reverse()` can reconstruct
//! the original expression for purely invertible chains (e.g. `exp(ln(x)) → x`
//! is invertible: `x → exp(ln(x))`).
//!
//! # Pipeline fidelity
//!
//! [`canonicalize_traced`] mirrors the exact four-step fixed-point pipeline
//! from [`crate::cas::canonicalize::canonicalize`]:
//! 1. `simplify_op` — constant folding, identity rules, commutative ordering.
//! 2. `apply_identity_db` — trig / log / hyperbolic identity database.
//! 3. `apply_canonical_rules` — log/exp expansion, power identities.
//! 4. `simplify_op` again — folds new constants introduced by step 3.
//!
//! The `final_op` in the returned [`RewriteTrace`] is therefore byte-equal to
//! `canonicalize(op).into_op()`.

use crate::cas::canonical_rules::apply_canonical_rules;
use crate::cas::canonicalize::MAX_CANONICALIZE_ITER;
use crate::cas::identity_db::{apply_identity_db, IdentityDb};
use crate::eml::op::LoweredOp;
use crate::eml::simplify::simplify_op;
use once_cell::sync::Lazy;

/// Lazily-initialized standard identity database, identical to the one used
/// in [`crate::cas::canonicalize::canonicalize`].
static TRACED_IDENTITY_DB: Lazy<IdentityDb> = Lazy::new(IdentityDb::standard);

// =========================================================================
// Public types
// =========================================================================

/// A single recorded rewrite step.
///
/// In the current batch-pass implementation (Option B), each step corresponds
/// to one outer iteration of the four-stage canonicalization pipeline. The
/// `rule_id` is always `"batch_pass"` and `is_reversible` is always `false`.
#[derive(Clone, Debug)]
pub struct RewriteStep {
    /// Human-readable identifier of the rule or batch that fired.
    ///
    /// Currently always `"batch_pass"` (one entry per outer fixed-point
    /// iteration). A future per-rule implementation will produce named
    /// identifiers like `"exp_ln_cancel"` or `"ln_mul_expand"`.
    pub rule_id: &'static str,
    /// Structural hash of the expression **before** this step.
    pub input_hash: u128,
    /// Structural hash of the expression **after** this step.
    pub output_hash: u128,
    /// Whether a well-defined inverse transformation exists for this step.
    ///
    /// Always `false` in the batch-pass implementation because a single
    /// pass can apply dozens of rules of mixed reversibility. Future per-rule
    /// tracking will set this to `true` for invertible rules such as
    /// `exp_ln_cancel` and `ln_exp_cancel`.
    pub is_reversible: bool,
}

/// Ordered record of all rewrite steps applied during canonicalization.
///
/// Produced by [`canonicalize_traced`]. The `steps` vector is in forward
/// order: first applied to last.
#[derive(Clone, Debug)]
pub struct RewriteTrace {
    /// The expression before any rewrites were applied.
    pub initial: LoweredOp,
    /// The expression after all rewrites (identical to
    /// `canonicalize(&initial).into_op()`).
    pub final_op: LoweredOp,
    /// Rewrite steps in forward (application) order.
    ///
    /// Empty when the input was already at the canonical fixed point.
    pub steps: Vec<RewriteStep>,
}

impl RewriteTrace {
    /// Returns `true` iff every step in the trace has a defined inverse.
    ///
    /// With the current batch-pass implementation this returns `true` if and
    /// only if `steps` is empty (no rewrites were needed).
    pub fn is_fully_reversible(&self) -> bool {
        self.steps.iter().all(|s| s.is_reversible)
    }

    /// Replay the trace in reverse to recover (an approximation of) the
    /// original expression.
    ///
    /// Returns `Some(original)` when [`Self::is_fully_reversible`] is `true`.
    /// In the current implementation this is only the case for an empty trace,
    /// where `original == initial`. Returns `None` whenever any step is
    /// irreversible or involves mixed rule applications whose inverse is
    /// undefined.
    ///
    /// # Note on conservatism
    ///
    /// The batch-pass design is intentionally conservative: a single
    /// fixed-point iteration may have applied both reversible rules (e.g.
    /// `exp(ln(x)) → x`) and irreversible rules (e.g. `add_zero`, constant
    /// folding) with no way to disentangle them post-hoc. Returning `None` in
    /// that case is correct — an incorrect approximate reconstruction would be
    /// worse than no reconstruction.
    pub fn reverse(&self) -> Option<LoweredOp> {
        if !self.is_fully_reversible() {
            return None;
        }
        // is_fully_reversible() is true ⟺ every step has is_reversible = true.
        // In the current implementation that only happens when steps is empty,
        // meaning the input was already canonical. The exact original is
        // recovered trivially.
        Some(self.initial.clone())
    }
}

// =========================================================================
// Public function
// =========================================================================

/// Run `cas::canonicalize` while recording each rewrite step.
///
/// Mirrors the four-stage fixed-point pipeline in
/// [`crate::cas::canonicalize::canonicalize`] exactly:
///
/// 1. `simplify_op`
/// 2. `apply_identity_db`
/// 3. `apply_canonical_rules`
/// 4. `simplify_op`
///
/// Each outer iteration where the structural hash changes emits one
/// [`RewriteStep`] with `rule_id = "batch_pass"`.
///
/// # Returns
///
/// A `(LoweredOp, RewriteTrace)` pair where the `LoweredOp` is identical to
/// `canonicalize(op).into_op()`.
///
/// # Termination
///
/// The outer loop is bounded by [`MAX_CANONICALIZE_ITER`] (32), matching the
/// budget of `canonicalize`. Budget exhaustion is noted in the trace but does
/// not panic.
pub fn canonicalize_traced(op: &LoweredOp) -> (LoweredOp, RewriteTrace) {
    let initial = op.clone();
    let mut steps: Vec<RewriteStep> = Vec::new();

    let initial_hash = op.structural_hash();

    // Phase 0: initial simplify_op pass (constant folding, identity rules,
    // commutative ordering). This runs BEFORE the main fixed-point loop and
    // can itself apply irreversible rules (e.g. `add_zero`, `mul_by_one`,
    // constant folding). We record it as a separate batch step if it changed
    // the expression, so that callers can detect "this was irreversible".
    let mut current = simplify_op(op);
    let after_initial_hash = current.structural_hash();
    if after_initial_hash != initial_hash {
        steps.push(RewriteStep {
            rule_id: "batch_pass",
            input_hash: initial_hash,
            output_hash: after_initial_hash,
            // simplify_op applies irreversible rules — mark conservatively.
            is_reversible: false,
        });
    }

    // Seed the convergence check from the post-initial-simplify hash so the
    // first loop iteration correctly detects "the four-stage pipeline adds
    // nothing more" for inputs already at the canonical fixed point.
    let mut prev_hash: u128 = after_initial_hash;

    for _ in 0..MAX_CANONICALIZE_ITER {
        let input_hash = current.structural_hash();

        // Step 1: identity-database rewrites (trig/log/hyperbolic identities).
        let after_id = apply_identity_db(&TRACED_IDENTITY_DB, &current);
        // Step 2: canonical-rules (log/exp expansion, power identities).
        let after_rules = apply_canonical_rules(&after_id);
        // Step 3: re-simplify (folds constants, re-runs commutative ordering).
        let resimplified = simplify_op(&after_rules);

        let h = resimplified.structural_hash();
        if h == prev_hash {
            // Fixed point reached — the full four-stage pipeline did not
            // change the expression on this iteration. Return without recording
            // a further step.
            return (
                resimplified.clone(),
                RewriteTrace {
                    initial,
                    final_op: resimplified,
                    steps,
                },
            );
        }

        // The pipeline changed the expression. Record one batch-pass step.
        steps.push(RewriteStep {
            rule_id: "batch_pass",
            input_hash,
            output_hash: h,
            // False: a single batch pass applies an unknown mix of reversible
            // and irreversible rules (simplify_op, identity_db, canonical_rules).
            is_reversible: false,
        });

        prev_hash = h;
        current = resimplified;
    }

    // Budget exhausted — return best-effort (most recently resimplified) form.
    let final_op = current;
    (
        final_op.clone(),
        RewriteTrace {
            initial,
            final_op,
            steps,
        },
    )
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::op::LoweredOp;

    // ---- helpers ----

    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn cst(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn exp(x: LoweredOp) -> LoweredOp {
        LoweredOp::Exp(Box::new(x))
    }
    fn ln(x: LoweredOp) -> LoweredOp {
        LoweredOp::Ln(Box::new(x))
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Box::new(a), Box::new(b))
    }
    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Box::new(a), Box::new(b))
    }

    // ---- tests ----

    /// Test 1: `x + 0` simplifies to `x`; trace is NOT fully reversible
    /// (the `add_zero` identity is irreversible — the `Const(0.0)` node is lost).
    #[test]
    fn add_zero_not_reversible() {
        let (final_op, trace) = canonicalize_traced(&add(var(0), cst(0.0)));
        assert_eq!(final_op, var(0), "x + 0 should simplify to x");
        assert!(
            !trace.is_fully_reversible(),
            "add_zero is an irreversible rule — trace must not be fully reversible"
        );
        assert_eq!(
            trace.reverse(),
            None,
            "reverse() must return None for irreversible traces"
        );
    }

    /// Test 2: `exp(ln(x))` cancels to `x` via the canonical rules.
    /// Under Option B (batch-pass), the step is recorded as irreversible
    /// because the batch also ran simplify_op (which may have applied other
    /// rules). `reverse()` returns `None`.
    #[test]
    fn exp_ln_cancel_batch_b() {
        let (final_op, trace) = canonicalize_traced(&exp(ln(var(0))));
        assert_eq!(final_op, var(0), "exp(ln(x)) should cancel to x");
        // Option B: batch steps are always irreversible.
        assert!(
            !trace.is_fully_reversible(),
            "batch steps are always irreversible under Option B"
        );
        assert_eq!(
            trace.reverse(),
            None,
            "reverse() must return None for a non-empty, non-reversible trace"
        );
    }

    /// Test 3: Identity — a variable is already canonical; the trace is empty
    /// and `reverse()` recovers the original.
    #[test]
    fn identity_empty_trace() {
        let (final_op, trace) = canonicalize_traced(&var(0));
        assert_eq!(final_op, var(0), "Var(0) is already canonical");
        assert!(
            trace.steps.is_empty(),
            "no steps should be recorded for an already-canonical expression"
        );
        assert!(
            trace.is_fully_reversible(),
            "empty trace is trivially fully reversible"
        );
        assert_eq!(
            trace.reverse(),
            Some(var(0)),
            "reverse() of empty trace should return the original"
        );
    }

    /// Test 4: Constant folding `3.0 + 4.0 → 7.0` is irreversible.
    #[test]
    fn const_fold_not_reversible() {
        let (final_op, trace) = canonicalize_traced(&add(cst(3.0), cst(4.0)));
        assert_eq!(
            final_op,
            LoweredOp::Const(7.0),
            "3.0 + 4.0 should constant-fold to 7.0"
        );
        assert!(
            !trace.is_fully_reversible(),
            "constant folding is irreversible"
        );
        assert_eq!(
            trace.reverse(),
            None,
            "reverse() must return None for constant-folded trace"
        );
    }

    /// Test 5: `x * 1.0` simplifies to `x`; trace is irreversible (mul_by_one).
    #[test]
    fn mul_by_one_not_reversible() {
        let (final_op, trace) = canonicalize_traced(&mul(var(0), cst(1.0)));
        assert_eq!(final_op, var(0), "x * 1.0 should simplify to x");
        assert!(
            !trace.is_fully_reversible(),
            "mul_by_one is irreversible — Const(1.0) node is lost"
        );
        assert_eq!(trace.reverse(), None, "reverse() must return None");
    }

    /// Test 6: Round-trip for empty trace — `reverse().unwrap() == initial`.
    #[test]
    fn round_trip_empty_trace() {
        let op = var(1);
        let (_, trace) = canonicalize_traced(&op);
        assert!(trace.steps.is_empty(), "Var(1) should produce no steps");
        let recovered = trace.reverse().expect("empty trace must round-trip");
        assert_eq!(
            recovered, op,
            "recovered expression must equal the original"
        );
    }

    /// Test 7: `exp(ln(x))` produces at least one step in the trace because
    /// the expression changes between the initial and the final form.
    #[test]
    fn exp_ln_step_recorded() {
        let (_, trace) = canonicalize_traced(&exp(ln(var(0))));
        assert!(
            !trace.steps.is_empty(),
            "exp(ln(x)) should record at least one batch-pass step"
        );
        // Each recorded step carries the expected rule_id.
        for step in &trace.steps {
            assert_eq!(step.rule_id, "batch_pass");
            assert!(!step.is_reversible);
        }
    }

    /// Test 8: `is_fully_reversible()` and `reverse()` are consistent:
    /// whenever `is_fully_reversible()` returns `true`, `reverse()` returns
    /// `Some(...)`.
    #[test]
    fn fully_reversible_implies_reverse_some() {
        // Case A: empty trace (already canonical) — consistent.
        let (_, trace_a) = canonicalize_traced(&var(0));
        assert!(trace_a.is_fully_reversible());
        assert!(
            trace_a.reverse().is_some(),
            "is_fully_reversible() true ⇒ reverse() must be Some"
        );

        // Case B: non-trivial rewrite — consistent in the other direction.
        let (_, trace_b) = canonicalize_traced(&add(var(0), cst(0.0)));
        assert!(!trace_b.is_fully_reversible());
        assert!(
            trace_b.reverse().is_none(),
            "is_fully_reversible() false ⇒ reverse() must be None"
        );
    }
}
