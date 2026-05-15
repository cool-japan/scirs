//! SMT-certified rewrite engine for symbolic expressions.
//!
//! This module provides [`rewrite_certified`] and [`rewrite_certified_fixpoint`]:
//! pattern-based rewrite rules whose soundness is verified by the OxiZ SMT
//! solver before application. Only rules that the solver can prove equal
//! (i.e., `check_equal` returns `true`) are applied to the expression.
//!
//! # Pre-canonicalization
//!
//! Both the LHS and RHS instantiations are run through
//! [`crate::cas::canonicalize::canonicalize`] before any SMT check. This is
//! mandatory to work around OxiZ 0.2.1's incomplete commutativity:
//! `mk_distinct(x+1, 1+x)` incorrectly returns `Sat`. By canonicalizing first,
//! structurally equivalent forms are collapsed to the same hash before the
//! solver sees them, so the structural-hash fast path succeeds without invoking
//! the incomplete NLSAT procedure.
//!
//! # Handling of `SmtError::Unknown`
//!
//! When OxiZ returns `Unknown` (incomplete decision), the rule is treated as
//! **not certified** and the next rule is tried. The Unknown result is never
//! propagated as an error from `rewrite_certified` — it is a normal outcome
//! for the incomplete QF_NRA decision procedure. Only structurally incorrect
//! usage (e.g., encoding non-finite constants) is propagated as `Err`.
//!
//! # Feature gate
//!
//! The entire module is gated on the `smt` feature.
//!
//! # No recursion
//!
//! All algorithms here are iterative. Neither `rewrite_certified` nor
//! `rewrite_certified_fixpoint` recurse.
//!
//! [`crate::cas::canonicalize::canonicalize`]: crate::cas::canonicalize::canonicalize

#![cfg(feature = "smt")]

use crate::cas::canonicalize::canonicalize;
use crate::cas::pattern::{instantiate, match_pattern, Bindings, Pattern, PatternError};
use crate::cas::smt::{EmlSmtSolver, SmtError};
use crate::eml::op::LoweredOp;

/// A rewrite rule `lhs → rhs` that must be SMT-certified before application.
///
/// The `lhs` pattern is matched against the input expression. If it matches,
/// `rhs` is instantiated with the same bindings. The rule is only applied if
/// the SMT solver can prove `lhs_instantiated == rhs_instantiated` (after
/// pre-canonicalization of both sides).
///
/// Both `lhs` and `rhs` must use the same set of wildcard indices — any
/// wildcard in `rhs` that was not bound by matching `lhs` will cause
/// [`instantiate`] to fail with `PatternError::MissingBinding`.
pub struct CertifiedRule {
    /// Pattern for the left-hand side. Matched against the input expression.
    pub lhs: Pattern,
    /// Pattern for the right-hand side. Instantiated with bindings from `lhs`.
    pub rhs: Pattern,
    /// Human-readable name for this rule (used for diagnostics and test assertions).
    pub name: &'static str,
}

/// Errors that can occur during certified rewriting.
#[derive(Debug)]
pub enum CertifiedRewriteError {
    /// An SMT encoding error that indicates misuse (non-finite constant,
    /// unsupported operator). Note: `SmtError::Unknown` is NOT propagated —
    /// it is silently treated as "not certified" (see module docs).
    SmtError(SmtError),
    /// The RHS pattern could not be instantiated (wildcard not bound by LHS).
    InstantiationError(PatternError),
    /// The fixpoint loop exceeded the iteration budget without converging.
    FixpointExceeded {
        /// The maximum number of iterations that was configured.
        max: u32,
    },
}

impl std::fmt::Display for CertifiedRewriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CertifiedRewriteError::SmtError(e) => write!(f, "SMT error: {e}"),
            CertifiedRewriteError::InstantiationError(e) => {
                write!(f, "instantiation error: {e}")
            }
            CertifiedRewriteError::FixpointExceeded { max } => {
                write!(f, "certified rewrite fixpoint exceeded {max} iterations")
            }
        }
    }
}

impl std::error::Error for CertifiedRewriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CertifiedRewriteError::SmtError(e) => Some(e),
            CertifiedRewriteError::InstantiationError(e) => Some(e),
            CertifiedRewriteError::FixpointExceeded { .. } => None,
        }
    }
}

/// Maximum fixpoint iterations in [`rewrite_certified_fixpoint`].
///
/// 8 iterations are generous for any realistic rule set. If this is reached
/// without convergence, [`CertifiedRewriteError::FixpointExceeded`] is returned.
pub const MAX_CERT_ITER: u32 = 8;

/// Determine whether an `SmtError` should be propagated or treated as
/// "not certified — continue trying rules".
///
/// `Unknown` (OxiZ gave up) and `Unsupported` (operator outside QF_NRA) are
/// both treated as "not certified": the rule is skipped, not an error. Only
/// `NonFiniteConstant` and `SolverError` indicate a structural problem that
/// should surface as an error.
fn is_fatal(e: &SmtError) -> bool {
    matches!(e, SmtError::NonFiniteConstant(_) | SmtError::SolverError(_))
}

/// Try to apply the first matching rule from `rules` to `op` that the SMT
/// solver certifies as sound.
///
/// For each rule in order:
/// 1. Match `rule.lhs` against `op`. Skip if no match.
/// 2. Instantiate `rule.rhs` with the bindings. Propagate `PatternError` if
///    a required wildcard is missing.
/// 3. Pre-canonicalize both `lhs_inst` and `rhs_inst` (mandatory — see module
///    docs for the OxiZ 0.2.1 incompleteness workaround).
/// 4. Structural-hash fast path: if the canonical hashes match, accept the
///    rule immediately (no SMT query needed).
/// 5. SMT path: call `solver.check_equal` with RAII push/pop. If the solver
///    returns `true`, return `(rhs_inst, Some(rule.name))`. If it returns
///    `false` or `Err(Unknown)`/`Err(Unsupported)`, skip the rule. Fatal SMT
///    errors are propagated.
///
/// # Returns
///
/// - `Ok((new_op, Some(rule_name)))` — a rule was applied and certified.
/// - `Ok((original_op, None))` — no rule matched or could be certified.
///
/// # Errors
///
/// - [`CertifiedRewriteError::InstantiationError`] — `rule.rhs` referenced a
///   wildcard not bound by `rule.lhs`.
/// - [`CertifiedRewriteError::SmtError`] — a fatal SMT encoding error (non-finite
///   constant, internal solver error).
pub fn rewrite_certified(
    op: LoweredOp,
    rules: &[CertifiedRule],
    solver: &mut EmlSmtSolver,
) -> Result<(LoweredOp, Option<&'static str>), CertifiedRewriteError> {
    for rule in rules {
        let mut bindings = Bindings::new();
        if !match_pattern(&rule.lhs, &op, &mut bindings) {
            continue;
        }

        // Instantiate both sides with the matched bindings.
        let lhs_inst =
            instantiate(&rule.lhs, &bindings).map_err(CertifiedRewriteError::InstantiationError)?;
        let rhs_inst =
            instantiate(&rule.rhs, &bindings).map_err(CertifiedRewriteError::InstantiationError)?;

        // Pre-canonicalize both sides (MANDATORY: mitigates OxiZ 0.2.1
        // commutativity incompleteness — see module-level documentation).
        let lhs_can = canonicalize(&lhs_inst);
        let rhs_can = canonicalize(&rhs_inst);

        // Structural-hash fast path: if canonical hashes match, the rule is
        // sound by construction (no SMT query needed).
        if lhs_can.hash() == rhs_can.hash() {
            return Ok((rhs_inst, Some(rule.name)));
        }

        // SMT path with RAII push/pop discipline.
        // push() and pop() are infallible (return `()`).
        solver.push();
        let smt_result = solver.check_equal(lhs_can.op(), rhs_can.op());
        // pop() is ALWAYS called, even if check_equal returned an error.
        solver.pop();

        match smt_result {
            Ok(true) => {
                // Solver proved the rule is sound — apply it.
                return Ok((rhs_inst, Some(rule.name)));
            }
            Ok(false) => {
                // Solver found a counterexample — rule is not universally valid,
                // skip it.
                continue;
            }
            Err(e) if is_fatal(&e) => {
                // Fatal SMT error — propagate it.
                return Err(CertifiedRewriteError::SmtError(e));
            }
            Err(_) => {
                // Non-fatal (Unknown or Unsupported) — treat as "not certified", continue.
                continue;
            }
        }
    }

    // No rule matched or could be certified.
    Ok((op, None))
}

/// Apply certified rewrites to fixed point, bounded by [`MAX_CERT_ITER`].
///
/// Repeatedly calls [`rewrite_certified`] until either:
/// - No rule fires (the expression has stabilised), or
/// - The structural hash of the result matches the previous iteration's hash
///   (idempotent rule — no structural change).
///
/// If neither condition is reached within `MAX_CERT_ITER` iterations,
/// returns [`CertifiedRewriteError::FixpointExceeded`].
///
/// # Errors
///
/// - [`CertifiedRewriteError::FixpointExceeded`] — iteration budget exhausted.
/// - Any error from [`rewrite_certified`].
pub fn rewrite_certified_fixpoint(
    op: LoweredOp,
    rules: &[CertifiedRule],
    solver: &mut EmlSmtSolver,
) -> Result<LoweredOp, CertifiedRewriteError> {
    let mut current = op;

    for _ in 0..MAX_CERT_ITER {
        let prev_hash = current.structural_hash();
        let (next, applied) = rewrite_certified(current, rules, solver)?;

        // Check structural progress BEFORE deciding whether to loop.
        let next_hash = next.structural_hash();

        if applied.is_none() {
            // No rule fired — fixpoint reached.
            return Ok(next);
        }

        if next_hash == prev_hash {
            // A rule fired but produced the same structure (idempotent rule
            // like `Var(0) → Var(0)`). Structural fixpoint: stop here.
            return Ok(next);
        }

        current = next;
    }

    Err(CertifiedRewriteError::FixpointExceeded { max: MAX_CERT_ITER })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cas::pattern::{BinaryKind, UnaryKind};

    // Convenience constructors for test expressions.
    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    // ----------------------------------------------------------------
    // Helper: build the `x * 1 → x` rule (PatVar(0) * PatConst(1) → PatVar(0)).
    // ----------------------------------------------------------------
    fn rule_mul_one() -> CertifiedRule {
        CertifiedRule {
            lhs: Pattern::PatOp2(
                BinaryKind::Mul,
                Box::new(Pattern::PatVar(0)),
                Box::new(Pattern::PatConst(1.0)),
            ),
            rhs: Pattern::PatVar(0),
            name: "x*1→x",
        }
    }

    // ----------------------------------------------------------------
    // Helper: build the `0 + x → x` rule (PatConst(0) + PatVar(0) → PatVar(0)).
    // ----------------------------------------------------------------
    fn rule_add_zero_left() -> CertifiedRule {
        CertifiedRule {
            lhs: Pattern::PatOp2(
                BinaryKind::Add,
                Box::new(Pattern::PatConst(0.0)),
                Box::new(Pattern::PatVar(0)),
            ),
            rhs: Pattern::PatVar(0),
            name: "0+x→x",
        }
    }

    // ----------------------------------------------------------------
    // Test 1: sound arithmetic rule `x * 1 → x` is certified and applied.
    // ----------------------------------------------------------------
    #[test]
    fn test_mul_one_certified_and_applied() {
        let mut solver = EmlSmtSolver::new();
        let rules = [rule_mul_one()];

        // Expr: Var(0) * Const(1.0) — should become Var(0).
        let op = LoweredOp::Mul(Box::new(var(0)), Box::new(c(1.0)));
        let (result, applied) = rewrite_certified(op, &rules, &mut solver)
            .expect("rewrite_certified must not error on a valid rule");

        assert!(applied.is_some(), "rule should have been applied");
        assert_eq!(applied.unwrap(), "x*1→x");
        assert_eq!(result, var(0), "result should be Var(0)");
    }

    // ----------------------------------------------------------------
    // Test 2: sound additive identity `0 + x → x` is certified and applied.
    // ----------------------------------------------------------------
    #[test]
    fn test_add_zero_certified_and_applied() {
        let mut solver = EmlSmtSolver::new();
        let rules = [rule_add_zero_left()];

        // Expr: Const(0.0) + Var(0) — should become Var(0).
        let op = LoweredOp::Add(Box::new(c(0.0)), Box::new(var(0)));
        let (result, applied) =
            rewrite_certified(op, &rules, &mut solver).expect("rewrite_certified must not error");

        assert!(applied.is_some(), "rule should have been applied");
        assert_eq!(applied.unwrap(), "0+x→x");
        assert_eq!(result, var(0), "result should be Var(0)");
    }

    // ----------------------------------------------------------------
    // Test 3: poison-pill `x * x → x` is rejected (x=2 gives 4 ≠ 2).
    // ----------------------------------------------------------------
    #[test]
    fn test_poison_mul_x_x_rejected() {
        let mut solver = EmlSmtSolver::new();

        let poison = CertifiedRule {
            // Pattern: PatVar(0) * PatVar(0)
            lhs: Pattern::PatOp2(
                BinaryKind::Mul,
                Box::new(Pattern::PatVar(0)),
                Box::new(Pattern::PatVar(0)),
            ),
            // RHS: PatVar(0) — would collapse x² → x (UNSOUND)
            rhs: Pattern::PatVar(0),
            name: "x²→x (POISON)",
        };

        // Expr: Var(0) * Var(0)
        let op = LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)));
        let (_, applied) = rewrite_certified(op, &[poison], &mut solver)
            .expect("rewrite_certified must not error");

        assert!(
            applied.is_none(),
            "poison rule must NOT be applied (solver should reject x²→x)"
        );
    }

    // ----------------------------------------------------------------
    // Test 4: poison-pill `sin(x) → x` is rejected.
    // ----------------------------------------------------------------
    #[test]
    fn test_poison_sin_x_rejected() {
        let mut solver = EmlSmtSolver::new();

        let poison = CertifiedRule {
            // Pattern: Sin(PatVar(0))
            lhs: Pattern::PatOp1(UnaryKind::Sin, Box::new(Pattern::PatVar(0))),
            // RHS: PatVar(0) — would collapse sin(x) → x (UNSOUND)
            rhs: Pattern::PatVar(0),
            name: "sin(x)→x (POISON)",
        };

        // Expr: Sin(Var(0))
        let op = LoweredOp::Sin(Box::new(var(0)));
        let (_, applied) = rewrite_certified(op, &[poison], &mut solver)
            .expect("rewrite_certified must not error");

        assert!(
            applied.is_none(),
            "poison rule must NOT be applied (solver should reject sin(x)→x)"
        );
    }

    // ----------------------------------------------------------------
    // Test 5: no-match passthrough — if no rule's LHS matches the op,
    //         rewrite_certified returns (original_op, None).
    // ----------------------------------------------------------------
    #[test]
    fn test_no_match_returns_original() {
        let mut solver = EmlSmtSolver::new();
        let rules = [rule_mul_one()]; // Only matches Mul(_, Const(1.0))

        // Expr: Var(0) + Var(1) — no rule matches this shape.
        let op = LoweredOp::Add(Box::new(var(0)), Box::new(var(1)));
        let original_hash = op.structural_hash();

        let (result, applied) =
            rewrite_certified(op, &rules, &mut solver).expect("rewrite_certified must not error");

        assert!(applied.is_none(), "no rule should have matched");
        assert_eq!(
            result.structural_hash(),
            original_hash,
            "original op must be returned unchanged"
        );
    }

    // ----------------------------------------------------------------
    // Test 6: fixpoint terminates on idempotent / self-mapping rules.
    //         A rule `Var(0) → Var(0)` via hash fast-path hits terminates
    //         immediately (idempotent structural change → early exit).
    // ----------------------------------------------------------------
    #[test]
    fn test_fixpoint_terminates_on_idempotent_rule() {
        let mut solver = EmlSmtSolver::new();

        // Rule: PatVar(0) → PatVar(0)  — always applies to any expression,
        // but produces the same structure. The fixpoint loop must detect this
        // and terminate without hitting MAX_CERT_ITER.
        let tautology = CertifiedRule {
            lhs: Pattern::PatVar(0),
            rhs: Pattern::PatVar(0),
            name: "x→x (idempotent)",
        };

        let op = var(0);
        let result = rewrite_certified_fixpoint(op, &[tautology], &mut solver);

        // Must NOT return FixpointExceeded — idempotent rule should terminate
        // on structural-hash equality check after first application.
        assert!(
            result.is_ok(),
            "fixpoint must terminate on idempotent rule, got: {result:?}"
        );
        assert_eq!(
            result.unwrap(),
            var(0),
            "result of idempotent rule must be the original expression"
        );
    }

    // ----------------------------------------------------------------
    // Test 7: fixpoint applies a rule sequentially to convergence.
    //         Rule `0 + x → x` applied to `0 + Var(0)` at root converges
    //         in one step to `Var(0)`.
    // ----------------------------------------------------------------
    #[test]
    fn test_fixpoint_sequential_application() {
        let mut solver = EmlSmtSolver::new();

        // Only the root node is rewritten by rewrite_certified; we test
        // that fixpoint converges correctly at root level.
        let rules = [rule_add_zero_left()];

        // Input: Const(0.0) + Var(0)  →  Var(0) in one fixpoint step.
        let op = LoweredOp::Add(Box::new(c(0.0)), Box::new(var(0)));
        let result = rewrite_certified_fixpoint(op, &rules, &mut solver)
            .expect("fixpoint must not error on a valid rule");

        assert_eq!(result, var(0), "fixpoint must converge to Var(0)");
    }
}
