//! `cas::canonicalize` — the world-first EML-IR-native CAS canonical form.
//!
//! Hash equality on [`Canonical`] values implies mathematical equality on
//! a documented decidable subset:
//!
//! - **Polynomial subring** — trivially decidable. Any expression in
//!   variables `x_0, x_1, ...` involving only `+`, `-`, `*`, integer-power
//!   `^n` is canonicalised to a hash-unique form.
//! - **Analytic identities** — decidable-with-exceptions for `sin`, `cos`,
//!   `exp`, `ln`, `sqrt`. The exceptions are branch-cut-sensitive arguments
//!   (`ln(-1)`, `sqrt(-1)` — these are not normalised) and transcendental
//!   identities outside the rule set.
//!
//! # Pipeline
//!
//! 1. **Lower** input `LoweredOp` to a working tree (already lowered by
//!    construction — accept `&LoweredOp` directly).
//! 2. **Simplify** via [`crate::eml::simplify::simplify_op`]
//!    (constant folding, identity rules, inverse cancellation,
//!    hash-based commutative ordering).
//! 3. **Apply canonical-form rewrites** via
//!    [`crate::cas::canonical_rules::apply_canonical_rules`] — additional
//!    rewrites beyond `simplify_op` (log/exp expansion, power identities,
//!    `(a^m)^n → a^(m*n)`, etc.).
//! 4. **Re-simplify** so the post-rule constants fold and the commutative
//!    sort runs again.
//! 5. **Sort commutative subexpressions** by structural hash — handled by
//!    `simplify_op`'s `apply_add_rules`/`apply_mul_rules` so `x+y` and
//!    `y+x` produce identical canonical trees.
//! 6. **DAG common-subexpression elimination** via the EML hash-cons pool
//!    (already done at `EmlNode` creation; `LoweredOp` is structural
//!    equality through `structural_hash`).
//! 7. **Wrap result in [`Canonical`]** newtype with the cached u128 hash.
//!
//! The whole pipeline runs to fixed point (bounded by
//! [`MAX_CANONICALIZE_ITER`]).
//!
//! # Decidability boundary
//!
//! `canonicalize(a).hash() == canonicalize(b).hash()` implies `a ≡ b` on:
//!
//! - Polynomials in any number of variables, with arithmetic ops and
//!   integer powers.
//! - Trig identities of the form `sin(x)/cos(x) → tan(x)` and inverse
//!   cancellations like `sin(arcsin(x)) → x`.
//! - Log/exp identities: `ln(a*b) → ln(a) + ln(b)`, `ln(a^n) → n * ln(a)`,
//!   `exp(a)·exp(b) → exp(a+b)`, `ln(exp(x)) → x`, `exp(ln(x)) → x`.
//!
//! It does **not** decide:
//!
//! - Arbitrary transcendental equalities (Liouville closure).
//! - Special-function identities (Bessel, Gamma, hypergeometric, ...).
//! - Branch-cut-sensitive expressions (e.g. `sqrt(x)^2 ≠ x` for `x < 0` in ℂ).
//! - Trig sum identities like `sin²(x) + cos²(x) → 1` (deferred to a future
//!   trig-rewrite pass; the structural shape is too easily inflated by
//!   `simplify_op`'s commutative ordering to be reliably matched here).
//!
//! # Idempotence
//!
//! `canonicalize(canonicalize(x).op()) == canonicalize(x)` is enforced by
//! the fixed-point loop. Successive applications converge in O(1) extra
//! work because `apply_canonical_rules` and `simplify_op` are themselves
//! idempotent on canonical inputs.
//!
//! # No recursion
//!
//! All traversals are iterative. A 543-node-deep `Canonical::sin(x)` tree
//! must not blow the OS stack — both `simplify_op` and
//! `apply_canonical_rules` use the work-stack pattern.

#![warn(missing_docs)]

use crate::cas::canonical_rules::apply_canonical_rules;
use crate::cas::identity_db::{apply_identity_db, IdentityDb};
use crate::eml::op::LoweredOp;
use crate::eml::simplify::simplify_op;
use once_cell::sync::Lazy;

/// Lazily-initialized standard identity database used inside the
/// [`canonicalize`] fixed-point loop.
///
/// Constructed once; shared across all calls to `canonicalize`.
static CANON_IDENTITY_DB: Lazy<IdentityDb> = Lazy::new(IdentityDb::standard);

/// Maximum outer fixed-point iterations for [`canonicalize`].
///
/// Each outer iteration runs `simplify_op → apply_canonical_rules →
/// simplify_op`, then checks the structural hash for convergence. 32 is
/// generous: realistic inputs converge in 1-3 iterations.
pub const MAX_CANONICALIZE_ITER: usize = 32;

/// A canonicalized formula. Hash equality on [`Canonical`] values implies
/// mathematical equality on the documented decidable subset (see module docs).
///
/// The newtype wraps a [`LoweredOp`] together with its precomputed
/// structural hash so [`Canonical::hash`] is O(1).
///
/// # PartialEq / Eq / Hash
///
/// `PartialEq` and `Eq` compare both the cached hash and the underlying
/// `LoweredOp`. The standard-library `Hash` impl mixes the cached u128 hash
/// into the user's hasher (so `Canonical` is suitable as a `HashMap` key);
/// the **canonical** structural hash is exposed via [`Canonical::hash`].
#[derive(Clone, Debug)]
pub struct Canonical {
    op: LoweredOp,
    cached_hash: u128,
}

impl Canonical {
    /// Get a reference to the underlying [`LoweredOp`].
    pub fn op(&self) -> &LoweredOp {
        &self.op
    }

    /// Move out the underlying [`LoweredOp`].
    pub fn into_op(self) -> LoweredOp {
        self.op
    }

    /// O(1) cached structural u128 hash.
    ///
    /// This is the canonical-equality fingerprint: two [`Canonical`] values
    /// with the same `hash()` are mathematically equal on the decidable
    /// subset documented in the module-level docs.
    pub fn hash(&self) -> u128 {
        self.cached_hash
    }
}

impl PartialEq for Canonical {
    fn eq(&self, other: &Self) -> bool {
        // Hash check first (cheap), then full structural equality on the
        // op (defends against the astronomically-unlikely hash collision).
        self.cached_hash == other.cached_hash && self.op == other.op
    }
}

impl Eq for Canonical {}

impl std::hash::Hash for Canonical {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Mix the cached u128 hash into the user's hasher.
        self.cached_hash.hash(state);
    }
}

/// Canonicalize a [`LoweredOp`]. Hash equality of two results implies
/// mathematical equality on the decidable subset (see module docs).
///
/// Idempotent: `canonicalize(canonicalize(x).op()) == canonicalize(x)`.
///
/// # Pipeline
///
/// 1. `simplify_op` (Phase 0 baseline rewrites).
/// 2. `apply_canonical_rules` (log/exp expansion, power identities).
/// 3. `simplify_op` again (settles commutative ordering after rewrites).
/// 4. Loop to fixed point (bounded by [`MAX_CANONICALIZE_ITER`]).
/// 5. Wrap in [`Canonical`] with the cached structural hash.
///
/// # Convergence
///
/// In practice 1-3 outer iterations suffice. The bound exists only to
/// guarantee termination on pathological non-confluent rule interactions
/// (none currently known, but the budget protects against future rule
/// additions accidentally introducing one).
pub fn canonicalize(op: &LoweredOp) -> Canonical {
    let mut current = simplify_op(op);
    // Hash trail for oscillation detection: if we observe the same hash twice
    // (immediate fixed point) we exit. If we observe a hash we've seen before
    // but separated by other hashes (cycle), we also exit — this catches
    // 2-cycles and longer cycles that arise from non-confluent rule
    // interactions. The trail is bounded at MAX_CANONICALIZE_ITER entries so
    // memory is constant.
    let mut hash_trail: Vec<u128> = Vec::with_capacity(MAX_CANONICALIZE_ITER + 1);
    hash_trail.push(0); // sentinel

    for _ in 0..MAX_CANONICALIZE_ITER {
        // Step 2a: identity database rewrites (trig, hyperbolic, log identities).
        let after_id = apply_identity_db(&CANON_IDENTITY_DB, &current);
        // Step 3: canonical-rules rewrite (log/exp expansion, power identities).
        let after_rules = apply_canonical_rules(&after_id);
        // Step 4-5: simplify again (folds new constants from the rules,
        // re-runs commutative ordering on the new shape).
        let resimplified = simplify_op(&after_rules);

        let h = resimplified.structural_hash();
        // Fixed-point: same as previous iteration.
        if hash_trail.last().copied() == Some(h) {
            return Canonical {
                op: resimplified,
                cached_hash: h,
            };
        }
        // Cycle: the hash appears earlier in the trail. Bail out — emitting
        // the current form is still mathematically equivalent.
        if hash_trail.contains(&h) {
            tracing::warn!(
                "canonicalize: oscillation detected at hash {h}; trail length {}",
                hash_trail.len()
            );
            return Canonical {
                op: resimplified,
                cached_hash: h,
            };
        }
        hash_trail.push(h);
        current = resimplified;
    }

    // Budget exhausted — return the current (best-effort) form. Still
    // mathematically equivalent, just not guaranteed at the fixed point.
    tracing::warn!(
        "canonicalize: MAX_CANONICALIZE_ITER ({}) exhausted on op of size estimate {}",
        MAX_CANONICALIZE_ITER,
        crate::eml::simplify::MAX_SIMPLIFY_ITER
    );
    let h = current.structural_hash();
    Canonical {
        op: current,
        cached_hash: h,
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

    // ----------------------------------------------------------------
    // Idempotence
    // ----------------------------------------------------------------

    #[test]
    fn idempotent() {
        // canonicalize(canonicalize(x).op()) == canonicalize(x)
        let op = LoweredOp::Mul(Box::new(var(0)), Box::new(c(1.0)));
        let c1 = canonicalize(&op);
        let c2 = canonicalize(c1.op());
        assert_eq!(c1.hash(), c2.hash());
    }

    #[test]
    fn idempotent_complex() {
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Mul(Box::new(var(0)), Box::new(c(1.0)))),
            Box::new(LoweredOp::Sub(Box::new(c(2.0)), Box::new(c(2.0)))),
        );
        let c1 = canonicalize(&op);
        let c2 = canonicalize(c1.op());
        let c3 = canonicalize(c2.op());
        assert_eq!(c1.hash(), c2.hash());
        assert_eq!(c2.hash(), c3.hash());
    }

    // ----------------------------------------------------------------
    // Identity elimination → structural equality
    // ----------------------------------------------------------------

    #[test]
    fn x_plus_zero_canonical_x() {
        let a = LoweredOp::Add(Box::new(var(0)), Box::new(c(0.0)));
        let b = var(0);
        assert_eq!(canonicalize(&a).hash(), canonicalize(&b).hash());
    }

    #[test]
    fn x_times_one_canonical_x() {
        let a = LoweredOp::Mul(Box::new(var(0)), Box::new(c(1.0)));
        let b = var(0);
        assert_eq!(canonicalize(&a).hash(), canonicalize(&b).hash());
    }

    // ----------------------------------------------------------------
    // Commutativity
    // ----------------------------------------------------------------

    #[test]
    fn x_plus_y_eq_y_plus_x() {
        let a = LoweredOp::Add(Box::new(var(0)), Box::new(var(1)));
        let b = LoweredOp::Add(Box::new(var(1)), Box::new(var(0)));
        assert_eq!(
            canonicalize(&a).hash(),
            canonicalize(&b).hash(),
            "x+y should canonicalize to same form as y+x"
        );
    }

    #[test]
    fn x_times_y_eq_y_times_x() {
        let a = LoweredOp::Mul(Box::new(var(0)), Box::new(var(1)));
        let b = LoweredOp::Mul(Box::new(var(1)), Box::new(var(0)));
        assert_eq!(canonicalize(&a).hash(), canonicalize(&b).hash());
    }

    // ----------------------------------------------------------------
    // Inverse cancellation
    // ----------------------------------------------------------------

    #[test]
    fn ln_exp_cancels() {
        let a = LoweredOp::Ln(Box::new(LoweredOp::Exp(Box::new(var(0)))));
        let b = var(0);
        assert_eq!(canonicalize(&a).hash(), canonicalize(&b).hash());
    }

    #[test]
    fn exp_ln_cancels() {
        let a = LoweredOp::Exp(Box::new(LoweredOp::Ln(Box::new(var(0)))));
        let b = var(0);
        assert_eq!(canonicalize(&a).hash(), canonicalize(&b).hash());
    }

    // ----------------------------------------------------------------
    // Polynomial subring
    // ----------------------------------------------------------------

    #[test]
    fn polynomial_x_squared_minus_zero_eq_x_squared() {
        // x² - 0 vs x²
        let a = LoweredOp::Sub(
            Box::new(LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)))),
            Box::new(c(0.0)),
        );
        let b = LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)));
        assert_eq!(canonicalize(&a).hash(), canonicalize(&b).hash());
    }

    // ----------------------------------------------------------------
    // Negative tests — known unequal
    // ----------------------------------------------------------------

    #[test]
    fn known_unequal() {
        // x + 1 should NOT canonicalize the same as x - 1.
        let a = LoweredOp::Add(Box::new(var(0)), Box::new(c(1.0)));
        let b = LoweredOp::Sub(Box::new(var(0)), Box::new(c(1.0)));
        assert_ne!(canonicalize(&a).hash(), canonicalize(&b).hash());
    }

    #[test]
    fn known_unequal_vars() {
        // x and y should NOT canonicalize the same.
        assert_ne!(canonicalize(&var(0)).hash(), canonicalize(&var(1)).hash());
    }

    // ----------------------------------------------------------------
    // Robustness — no stack overflow
    // ----------------------------------------------------------------

    #[test]
    fn deep_chain_no_overflow() {
        let mut op = var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(c(0.0)));
        }
        let canon = canonicalize(&op);
        assert_eq!(canon.hash(), canonicalize(&var(0)).hash());
    }

    #[test]
    fn deep_chain_no_overflow_mul() {
        let mut op = var(0);
        for _ in 0..1000 {
            op = LoweredOp::Mul(Box::new(c(1.0)), Box::new(op));
        }
        let canon = canonicalize(&op);
        assert_eq!(canon.hash(), canonicalize(&var(0)).hash());
    }

    // ----------------------------------------------------------------
    // PartialEq + Hash trait integration
    // ----------------------------------------------------------------

    #[test]
    fn canonical_eq_implies_via_partialeq() {
        let a = LoweredOp::Add(Box::new(var(0)), Box::new(c(0.0)));
        let b = var(0);
        let ca = canonicalize(&a);
        let cb = canonicalize(&b);
        assert_eq!(ca, cb);
    }

    #[test]
    fn canonical_hash_consistent_with_eq() {
        // Trait-Hash and the canonical-hash u128 fingerprint are
        // independent, but two `Canonical` values that are PartialEq must
        // also produce the same standard-library hash.
        use std::collections::HashSet;
        let a = canonicalize(&LoweredOp::Add(Box::new(var(0)), Box::new(c(0.0))));
        let b = canonicalize(&var(0));
        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b), "HashSet membership via PartialEq + Hash");
    }

    #[test]
    fn canonical_into_op() {
        let a = canonicalize(&var(0));
        let op = a.into_op();
        assert_eq!(op, var(0));
    }

    // ----------------------------------------------------------------
    // Log expansion via canonical rules
    // ----------------------------------------------------------------

    #[test]
    fn ln_of_product_canonicalises_to_sum_of_logs() {
        // ln(x*y) and ln(x) + ln(y) should both canonicalize to the same form.
        let a = LoweredOp::Ln(Box::new(LoweredOp::Mul(Box::new(var(0)), Box::new(var(1)))));
        let b = LoweredOp::Add(
            Box::new(LoweredOp::Ln(Box::new(var(0)))),
            Box::new(LoweredOp::Ln(Box::new(var(1)))),
        );
        assert_eq!(canonicalize(&a).hash(), canonicalize(&b).hash());
    }

    #[test]
    fn exp_product_canonicalises_to_exp_sum() {
        // exp(x) * exp(y) and exp(x + y) should canonicalize to same form.
        let a = LoweredOp::Mul(
            Box::new(LoweredOp::Exp(Box::new(var(0)))),
            Box::new(LoweredOp::Exp(Box::new(var(1)))),
        );
        let b = LoweredOp::Exp(Box::new(LoweredOp::Add(Box::new(var(0)), Box::new(var(1)))));
        assert_eq!(canonicalize(&a).hash(), canonicalize(&b).hash());
    }
}
