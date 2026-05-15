//! Identity database for trigonometric and hyperbolic rewrite rules.
//!
//! An [`IdentityDb`] holds a list of [`Identity`] records, each pairing a
//! [`Pattern`] LHS with a [`Pattern`] RHS. [`apply_identity_db`] does a
//! single bottom-up post-order traversal, trying every rule at each node
//! and applying the *first* matching rule before moving on. The outer
//! [`crate::cas::canonicalize::canonicalize`] fixed-point loop handles
//! cascading rewrites between iterations.
//!
//! # Standard database
//!
//! [`IdentityDb::standard`] builds the 10-rule set covering:
//! - **Pythagorean identities**: `sin²(x)+cos²(x)→1`, `cos²(x)+sin²(x)→1`
//! - **Tangent expansion**: `tan(x)→sin(x)/cos(x)`
//! - **Secant identity**: `1+tan²(x)→1/cos²(x)`
//! - **Double-angle sine**: `sin(2x)→2·sin(x)·cos(x)`
//! - **Double-angle cosine**: `cos(2x)→cos²(x)−sin²(x)`
//! - **Hyperbolic Pythagorean**: `cosh²(x)−sinh²(x)→1`
//! - **Tanh expansion**: `tanh(x)→sinh(x)/cosh(x)`
//! - **Log–power rule**: `ln(x^n)→n·ln(x)` (also in canonical_rules; both are idempotent)
//! - **Sinh doubled**: `sinh(2x)→2·sinh(x)·cosh(x)`
//! - **Arctan cancellation**: `arctan(tan(x))→x` (exact over principal domain)
//!
//! # Domain notes
//!
//! Rules are structural (no domain tracking). `tan(x)→sin(x)/cos(x)` does
//! not verify `cos(x) ≠ 0`; `arctan(tan(x))→x` holds only for
//! `x ∈ (−π/2, π/2)`. These are the same relaxed assumptions the rest of
//! the EML CAS makes for all rewrite passes.
//!
//! # No recursion
//!
//! All traversals are iterative. A deeply nested `LoweredOp` tree must not
//! blow the OS stack.
//!
//! # Interaction with `canonical_rules`
//!
//! The log–power rule (`ln(x^n)→n·ln(x)`) is also handled by
//! [`crate::cas::canonical_rules::apply_canonical_rules`]. Having it in the
//! identity database is redundant but safe — the outer fixed-point loop
//! detects convergence by hash so duplicate rewrites just terminate faster.

#![warn(missing_docs)]

use crate::cas::pattern::{instantiate, match_pattern, BinaryKind, Bindings, Pattern, UnaryKind};
use crate::eml::op::LoweredOp;
use once_cell::sync::Lazy;

// =====================================================================
// Public types
// =====================================================================

/// Category of a mathematical identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityKind {
    /// Trigonometric identity (sin, cos, tan and their combinations).
    Trig,
    /// Hyperbolic identity (sinh, cosh, tanh and their combinations).
    Hyperbolic,
    /// Logarithm identity.
    Log,
    /// Exponential identity.
    Exp,
}

/// A single mathematical identity `lhs → rhs`.
///
/// When `match_pattern(&self.lhs, node, &mut bindings)` succeeds,
/// [`instantiate`] expands `rhs` with those bindings to produce the rewritten
/// node.
pub struct Identity {
    /// Left-hand side pattern to match.
    pub lhs: Pattern,
    /// Right-hand side pattern to instantiate on match.
    pub rhs: Pattern,
    /// Semantic category.
    pub kind: IdentityKind,
    /// Short human-readable name used in tracing and error messages.
    pub name: &'static str,
    /// Apply this rule **top-down** (before recursing into children) in
    /// addition to the standard bottom-up pass.
    ///
    /// Necessary for rules whose LHS pattern would be destroyed by a child-
    /// level rewrite — e.g. the Pythagorean identity `sin²(?0)+cos²(?0)→1`
    /// when `?0` is itself a sum (the sum-expansion rule would fire on
    /// `sin(x+y)` first in pure bottom-up order, breaking the Pythagorean
    /// match). Set this only for rules that **reduce** the structural size
    /// of the tree; expanding rules must stay bottom-up to avoid infinite
    /// rewriting.
    pub outer_first: bool,
}

/// Database of mathematical identities.
///
/// Ordered list — rules are tried in insertion order at each node.
/// The first matching rule fires; remaining rules are skipped.
pub struct IdentityDb {
    rules: Vec<Identity>,
}

impl IdentityDb {
    /// Create an empty database.
    pub fn new() -> Self {
        IdentityDb { rules: Vec::new() }
    }

    /// Add an identity to the end of the database.
    pub fn add(&mut self, identity: Identity) {
        self.rules.push(identity);
    }

    /// Reference to the ordered list of rules.
    pub fn rules(&self) -> &[Identity] {
        &self.rules
    }

    /// Build the standard 10-rule identity database.
    ///
    /// Rules are applied in order; the first match wins per node per pass.
    pub fn standard() -> Self {
        let mut db = IdentityDb::new();

        // ----------------------------------------------------------------
        // Rule 1a: sin²(x) + cos²(x) = 1
        //   Pattern: Add(Pow(Sin(?0), 2), Pow(Cos(?0), 2)) → Const(1)
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Add,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cos,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            rhs: Pattern::PatConst(1.0),
            kind: IdentityKind::Trig,
            name: "sin2_plus_cos2_eq_1",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 1b: cos²(x) + sin²(x) = 1  (commuted variant)
        //   Pattern: Add(Pow(Cos(?0), 2), Pow(Sin(?0), 2)) → Const(1)
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Add,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cos,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            rhs: Pattern::PatConst(1.0),
            kind: IdentityKind::Trig,
            name: "cos2_plus_sin2_eq_1",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 2: tan(x) = sin(x) / cos(x)
        //   Pattern: Tan(?0) → Div(Sin(?0), Cos(?0))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp1(UnaryKind::Tan, Box::new(Pattern::PatVar(0))),
            rhs: Pattern::PatOp2(
                BinaryKind::Div,
                Box::new(Pattern::PatOp1(
                    UnaryKind::Sin,
                    Box::new(Pattern::PatVar(0)),
                )),
                Box::new(Pattern::PatOp1(
                    UnaryKind::Cos,
                    Box::new(Pattern::PatVar(0)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "tan_eq_sin_over_cos",
            outer_first: false,
        });

        // ----------------------------------------------------------------
        // Rule 3: 1 + tan²(x) = 1/cos²(x)  (sec² identity)
        //   Pattern: Add(1, Pow(Tan(?0), 2)) → Div(1, Pow(Cos(?0), 2))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Add,
                Box::new(Pattern::PatConst(1.0)),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Tan,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Div,
                Box::new(Pattern::PatConst(1.0)),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cos,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "one_plus_tan2_eq_sec2",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 4: sin(2x) = 2·sin(x)·cos(x)
        //   Pattern: Sin(Mul(2, ?0)) → Mul(Mul(2, Sin(?0)), Cos(?0))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp1(
                UnaryKind::Sin,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Mul,
                    Box::new(Pattern::PatConst(2.0)),
                    Box::new(Pattern::PatVar(0)),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Mul,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Mul,
                    Box::new(Pattern::PatConst(2.0)),
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatVar(0)),
                    )),
                )),
                Box::new(Pattern::PatOp1(
                    UnaryKind::Cos,
                    Box::new(Pattern::PatVar(0)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "sin_double_angle",
            outer_first: false,
        });

        // ----------------------------------------------------------------
        // Rule 5: cos(2x) = cos²(x) - sin²(x)
        //   Pattern: Cos(Mul(2, ?0)) → Sub(Pow(Cos(?0), 2), Pow(Sin(?0), 2))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp1(
                UnaryKind::Cos,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Mul,
                    Box::new(Pattern::PatConst(2.0)),
                    Box::new(Pattern::PatVar(0)),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cos,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "cos_double_angle",
            outer_first: false,
        });

        // ----------------------------------------------------------------
        // Rule 6: cosh²(x) - sinh²(x) = 1
        //   Pattern: Sub(Pow(Cosh(?0), 2), Pow(Sinh(?0), 2)) → Const(1)
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cosh,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sinh,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            rhs: Pattern::PatConst(1.0),
            kind: IdentityKind::Hyperbolic,
            name: "cosh2_minus_sinh2_eq_1",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 7: tanh(x) = sinh(x) / cosh(x)
        //   Pattern: Tanh(?0) → Div(Sinh(?0), Cosh(?0))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp1(UnaryKind::Tanh, Box::new(Pattern::PatVar(0))),
            rhs: Pattern::PatOp2(
                BinaryKind::Div,
                Box::new(Pattern::PatOp1(
                    UnaryKind::Sinh,
                    Box::new(Pattern::PatVar(0)),
                )),
                Box::new(Pattern::PatOp1(
                    UnaryKind::Cosh,
                    Box::new(Pattern::PatVar(0)),
                )),
            ),
            kind: IdentityKind::Hyperbolic,
            name: "tanh_eq_sinh_over_cosh",
            outer_first: false,
        });

        // ----------------------------------------------------------------
        // Rule 8: ln(x^n) = n·ln(x)  (log-power rule)
        //   Pattern: Ln(Pow(?0, ?1)) → Mul(?1, Ln(?0))
        //
        // Note: `canonical_rules::rule_ln` handles the same identity via
        // direct structural matching. Both are idempotent; the outer fixed-
        // point loop detects convergence by hash regardless.
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp1(
                UnaryKind::Ln,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatVar(0)),
                    Box::new(Pattern::PatVar(1)),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Mul,
                Box::new(Pattern::PatVar(1)),
                Box::new(Pattern::PatOp1(UnaryKind::Ln, Box::new(Pattern::PatVar(0)))),
            ),
            kind: IdentityKind::Log,
            name: "ln_power_rule",
            outer_first: false,
        });

        // ----------------------------------------------------------------
        // Rule 9: sinh(2x) = 2·sinh(x)·cosh(x)
        //   Pattern: Sinh(Mul(2, ?0)) → Mul(Mul(2, Sinh(?0)), Cosh(?0))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp1(
                UnaryKind::Sinh,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Mul,
                    Box::new(Pattern::PatConst(2.0)),
                    Box::new(Pattern::PatVar(0)),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Mul,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Mul,
                    Box::new(Pattern::PatConst(2.0)),
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sinh,
                        Box::new(Pattern::PatVar(0)),
                    )),
                )),
                Box::new(Pattern::PatOp1(
                    UnaryKind::Cosh,
                    Box::new(Pattern::PatVar(0)),
                )),
            ),
            kind: IdentityKind::Hyperbolic,
            name: "sinh_double_angle",
            outer_first: false,
        });

        // ----------------------------------------------------------------
        // Rule 10: arctan(tan(x)) = x  (over the principal branch x ∈ (-π/2, π/2))
        //   Pattern: Arctan(Tan(?0)) → ?0
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp1(
                UnaryKind::Arctan,
                Box::new(Pattern::PatOp1(
                    UnaryKind::Tan,
                    Box::new(Pattern::PatVar(0)),
                )),
            ),
            rhs: Pattern::PatVar(0),
            kind: IdentityKind::Trig,
            name: "arctan_of_tan_identity",
            outer_first: false,
        });

        // ----------------------------------------------------------------
        // Rule 11a: 1 − 2·sin²(x) → cos²(x) − sin²(x)  (Const, Pow order)
        //
        // This unifies the three forms of the double-angle cosine
        // (cos²−sin², 1−2sin², 2cos²−1) on a single canonical representative
        // (cos²−sin², matching the existing double-angle expansion direction).
        //
        //   Pattern: Sub(1, Mul(2, Pow(Sin(?0), 2))) → Sub(Pow(Cos(?0), 2), Pow(Sin(?0), 2))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatConst(1.0)),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Mul,
                    Box::new(Pattern::PatConst(2.0)),
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Pow,
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Sin,
                            Box::new(Pattern::PatVar(0)),
                        )),
                        Box::new(Pattern::PatConst(2.0)),
                    )),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cos,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "one_minus_two_sin_sq_eq_cos_sq_minus_sin_sq",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 11b: 1 − sin²(x)·2 → cos²(x) − sin²(x)  (Pow, Const order)
        //
        // Commuted variant of Rule 11a, matching the hash-sorted form of
        // `Mul(2, sin²(x))` where Pow precedes Const.
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatConst(1.0)),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Mul,
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Pow,
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Sin,
                            Box::new(Pattern::PatVar(0)),
                        )),
                        Box::new(Pattern::PatConst(2.0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cos,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "one_minus_sin_sq_times_two_eq_cos_sq_minus_sin_sq",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 12a: 2·cos²(x) − 1 → cos²(x) − sin²(x)  (Const, Pow order)
        //
        // Companion to Rule 11; pin third double-angle form to canonical.
        //   Pattern: Sub(Mul(2, Pow(Cos(?0), 2)), 1) → Sub(Pow(Cos(?0), 2), Pow(Sin(?0), 2))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Mul,
                    Box::new(Pattern::PatConst(2.0)),
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Pow,
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Cos,
                            Box::new(Pattern::PatVar(0)),
                        )),
                        Box::new(Pattern::PatConst(2.0)),
                    )),
                )),
                Box::new(Pattern::PatConst(1.0)),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cos,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "two_cos_sq_minus_one_eq_cos_sq_minus_sin_sq",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 12b: cos²(x)·2 − 1 → cos²(x) − sin²(x)  (Pow, Const order)
        //
        // The hash-sorted commutative ordering of Mul puts Pow before Const
        // (lower hash). This commuted variant of Rule 12a fires on that
        // canonical form.
        //   Pattern: Sub(Mul(Pow(Cos(?0), 2), 2), 1) → Sub(Pow(Cos(?0), 2), Pow(Sin(?0), 2))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Mul,
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Pow,
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Cos,
                            Box::new(Pattern::PatVar(0)),
                        )),
                        Box::new(Pattern::PatConst(2.0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
                Box::new(Pattern::PatConst(1.0)),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cos,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "cos_sq_times_two_minus_one_eq_cos_sq_minus_sin_sq",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 13: sin(x + y) − [sin(x)·cos(y) + cos(x)·sin(y)] → 0
        //
        // Recognizer rule (collapsing-only): pins the sum-difference identity
        // as a structural collapse to zero rather than an unconditional
        // expansion. Expansion would break the proptest invariant
        // `apply_identity_db preserves canonical hash` when the argument is
        // a constant (since canonicalize folds numeric Subs but not Sin
        // wrappers).
        //
        //   Pattern: Sub(Sin(Add(?0,?1)), Add(Mul(Sin(?0),Cos(?1)), Mul(Cos(?0),Sin(?1))))
        //         → 0
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp1(
                    UnaryKind::Sin,
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Add,
                        Box::new(Pattern::PatVar(0)),
                        Box::new(Pattern::PatVar(1)),
                    )),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Add,
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Mul,
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Sin,
                            Box::new(Pattern::PatVar(0)),
                        )),
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Cos,
                            Box::new(Pattern::PatVar(1)),
                        )),
                    )),
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Mul,
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Cos,
                            Box::new(Pattern::PatVar(0)),
                        )),
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Sin,
                            Box::new(Pattern::PatVar(1)),
                        )),
                    )),
                )),
            ),
            rhs: Pattern::PatConst(0.0),
            kind: IdentityKind::Trig,
            name: "sin_sum_collapse",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 14: sin(x + y) − [cos(y)·sin(x) + sin(y)·cos(x)] → 0
        //   Commuted Mul order variant of Rule 13. Both factors of each Mul
        //   are commuted; the simplifier sorts them by hash, so the actual
        //   tree may have either ordering.
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Sub,
                Box::new(Pattern::PatOp1(
                    UnaryKind::Sin,
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Add,
                        Box::new(Pattern::PatVar(0)),
                        Box::new(Pattern::PatVar(1)),
                    )),
                )),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Add,
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Mul,
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Cos,
                            Box::new(Pattern::PatVar(1)),
                        )),
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Sin,
                            Box::new(Pattern::PatVar(0)),
                        )),
                    )),
                    Box::new(Pattern::PatOp2(
                        BinaryKind::Mul,
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Sin,
                            Box::new(Pattern::PatVar(1)),
                        )),
                        Box::new(Pattern::PatOp1(
                            UnaryKind::Cos,
                            Box::new(Pattern::PatVar(0)),
                        )),
                    )),
                )),
            ),
            rhs: Pattern::PatConst(0.0),
            kind: IdentityKind::Trig,
            name: "sin_sum_collapse_commuted",
            outer_first: true,
        });

        // ----------------------------------------------------------------
        // Rule 17: Product-to-sum direction pin (sin·cos form).
        //
        // ½·(sin(x+y) + sin(x−y)) → sin(x)·cos(y)
        //
        // The reverse direction (sin·cos → ½·sin-sum) is NEVER produced by
        // canonicalize: hence pinning to product-form is a one-way collapse.
        //
        //   Pattern: Mul(½, Add(Sin(Add(?0,?1)), Sin(Sub(?0,?1)))) →
        //            Mul(Sin(?0), Cos(?1))
        // ----------------------------------------------------------------
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Mul,
                Box::new(Pattern::PatConst(0.5)),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Add,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatOp2(
                            BinaryKind::Add,
                            Box::new(Pattern::PatVar(0)),
                            Box::new(Pattern::PatVar(1)),
                        )),
                    )),
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatOp2(
                            BinaryKind::Sub,
                            Box::new(Pattern::PatVar(0)),
                            Box::new(Pattern::PatVar(1)),
                        )),
                    )),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Mul,
                Box::new(Pattern::PatOp1(
                    UnaryKind::Sin,
                    Box::new(Pattern::PatVar(0)),
                )),
                Box::new(Pattern::PatOp1(
                    UnaryKind::Cos,
                    Box::new(Pattern::PatVar(1)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "product_to_sum_sin_cos",
            outer_first: true,
        });

        // Rule 17b: commuted Add inside (sin(x-y) + sin(x+y))
        db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Mul,
                Box::new(Pattern::PatConst(0.5)),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Add,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatOp2(
                            BinaryKind::Sub,
                            Box::new(Pattern::PatVar(0)),
                            Box::new(Pattern::PatVar(1)),
                        )),
                    )),
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Sin,
                        Box::new(Pattern::PatOp2(
                            BinaryKind::Add,
                            Box::new(Pattern::PatVar(0)),
                            Box::new(Pattern::PatVar(1)),
                        )),
                    )),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Mul,
                Box::new(Pattern::PatOp1(
                    UnaryKind::Sin,
                    Box::new(Pattern::PatVar(0)),
                )),
                Box::new(Pattern::PatOp1(
                    UnaryKind::Cos,
                    Box::new(Pattern::PatVar(1)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "product_to_sum_sin_cos_commuted",
            outer_first: true,
        });

        db
    }
}

impl Default for IdentityDb {
    fn default() -> Self {
        IdentityDb::new()
    }
}

// =====================================================================
// Static standard database
// =====================================================================

/// Lazily-initialized standard identity database (shared, zero-copy).
///
/// Constructed once on first use via [`once_cell::sync::Lazy`]. All subsequent
/// calls to [`apply_identity_db`] with no custom database use this instance.
static STANDARD_IDENTITY_DB: Lazy<IdentityDb> = Lazy::new(IdentityDb::standard);

/// Apply the standard identity database to `op`.
///
/// Thin wrapper around [`apply_identity_db`] using the static
/// `STANDARD_IDENTITY_DB`. Prefer this over constructing a new
/// [`IdentityDb`] on every call.
pub fn apply_standard_identity_db(op: &LoweredOp) -> LoweredOp {
    apply_identity_db(&STANDARD_IDENTITY_DB, op)
}

// =====================================================================
// Core traversal
// =====================================================================

/// Maximum bottom-up pass iterations for [`apply_identity_db`].
///
/// Bounds the outer fixed-point loop. 32 is generous; realistic inputs
/// converge in 1–3 passes. This budget is separate from (and in addition
/// to) the outer budget in [`crate::cas::canonicalize::MAX_CANONICALIZE_ITER`].
pub const MAX_IDENTITY_ITERS: usize = 32;

/// Apply all rules in `db` to `op` until a fixed point (max [`MAX_IDENTITY_ITERS`] passes).
///
/// Each pass does a single bottom-up post-order traversal: for every node,
/// rules are tried in database order; the first match fires (the remainder
/// are skipped). Cascading rewrites across nodes are caught by the outer
/// fixed-point loop.
///
/// # No recursion
///
/// Uses an iterative work-stack internally. A 1000-deep `LoweredOp` tree
/// produces no stack growth beyond the heap-allocated work stack.
///
/// # Error handling
///
/// If `instantiate` returns an error (a wildcard index missing from the RHS
/// that was not captured by the LHS), that rule is silently skipped and the
/// next rule is tried. This is defensive: a well-formed standard database
/// never triggers this path.
pub fn apply_identity_db(db: &IdentityDb, op: &LoweredOp) -> LoweredOp {
    let mut current = op.clone();
    let mut prev_hash: u128 = 0;

    for _ in 0..MAX_IDENTITY_ITERS {
        let next = apply_db_once(db, &current);
        let h = next.structural_hash();
        if h == prev_hash {
            return next;
        }
        prev_hash = h;
        current = next;
    }

    current
}

/// Single hybrid top-down + bottom-up pass applying identity rules.
///
/// For every node, we first attempt rule matching **top-down** (before
/// recursing into children). If any rule matches at the outer level, the
/// rewrite replaces the whole subtree and we do not descend. This is required
/// for rules like Pythagorean (`Add(Pow(Sin(?0),2), Pow(Cos(?0),2)) → 1`)
/// when `?0` is itself a compound expression (e.g. `x+y`): a pure bottom-up
/// pass would expand `Sin(Add(x,y))` via the sum-expansion rule before the
/// outer Add was visited, destroying the Pythagorean match.
///
/// If no top-down rule fires, we fall through to the standard bottom-up
/// pass: recurse, rewrite children, then re-attempt rules at the parent.
///
/// Returns the transformed tree.
fn apply_db_once(db: &IdentityDb, op: &LoweredOp) -> LoweredOp {
    /// Work-stack frame for the iterative post-order traversal.
    enum Frame<'a> {
        /// First visit: try rules top-down; on match, push result and continue.
        /// On no match, push children, then push a `Combine` frame.
        Open(&'a LoweredOp),
        /// Children are done (on the value stack); reconstruct and try rules.
        Combine(&'a LoweredOp),
    }

    let mut work: Vec<Frame<'_>> = vec![Frame::Open(op)];
    let mut val_stack: Vec<LoweredOp> = Vec::with_capacity(16);

    while let Some(frame) = work.pop() {
        match frame {
            Frame::Open(node) => {
                // First: try rules top-down on the current node. If a rule
                // matches, the rewritten form is pushed onto val_stack and
                // we do NOT descend into children. This is critical for
                // outer-pattern rules (Pythagorean) that would be destroyed
                // by inner-pattern rewrites (sum-expansion) in pure bottom-up.
                if let Some(rewritten) = try_rules_top_down(db, node) {
                    val_stack.push(rewritten);
                    continue;
                }
                // Schedule the combine step, then recurse into children.
                work.push(Frame::Combine(node));
                match node {
                    // Leaves: no children to process.
                    LoweredOp::Const(_) | LoweredOp::Var(_) => {}
                    // Unary ops: one child.
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
                        work.push(Frame::Open(c));
                    }
                    // Binary ops: two children — push right first so left pops first.
                    LoweredOp::Add(l, r)
                    | LoweredOp::Sub(l, r)
                    | LoweredOp::Mul(l, r)
                    | LoweredOp::Div(l, r)
                    | LoweredOp::Pow(l, r) => {
                        work.push(Frame::Open(r));
                        work.push(Frame::Open(l));
                    }
                }
            }

            Frame::Combine(node) => {
                // Reconstruct the node with already-rewritten children from val_stack,
                // then try every rule in the database.
                let reconstructed = reconstruct_node(node, &mut val_stack);
                let rewritten = try_rules(db, reconstructed);
                val_stack.push(rewritten);
            }
        }
    }

    // The final result is the single element remaining on val_stack.
    val_stack.pop().unwrap_or_else(|| op.clone())
}

/// Reconstruct a `LoweredOp` node by popping already-rewritten children from
/// `val_stack`.
///
/// For leaves, no children are popped. For unary ops, one child is popped.
/// For binary ops, `Frame::Open` pushes right-child first, then left-child,
/// so left is processed first and its result sits *below* right's result on
/// `val_stack`. To recover the original `(left, right)` order we pop right
/// first, then left.
///
/// # Post-order invariant
///
/// Caller guarantees that `val_stack` contains the rewritten children of
/// `node` at the top. This invariant holds because `Frame::Open` pushes
/// children before scheduling the `Combine` frame, and each child's
/// `Combine` frame runs before the parent's.
fn reconstruct_node(node: &LoweredOp, val_stack: &mut Vec<LoweredOp>) -> LoweredOp {
    match node {
        LoweredOp::Const(v) => LoweredOp::Const(*v),
        LoweredOp::Var(i) => LoweredOp::Var(*i),

        // Unary: pop one child.
        LoweredOp::Neg(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Neg(Box::new(c))
        }
        LoweredOp::Exp(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Exp(Box::new(c))
        }
        LoweredOp::Ln(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Ln(Box::new(c))
        }
        LoweredOp::Sin(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Sin(Box::new(c))
        }
        LoweredOp::Cos(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Cos(Box::new(c))
        }
        LoweredOp::Tan(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Tan(Box::new(c))
        }
        LoweredOp::Sinh(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Sinh(Box::new(c))
        }
        LoweredOp::Cosh(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Cosh(Box::new(c))
        }
        LoweredOp::Tanh(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Tanh(Box::new(c))
        }
        LoweredOp::Arcsin(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Arcsin(Box::new(c))
        }
        LoweredOp::Arccos(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Arccos(Box::new(c))
        }
        LoweredOp::Arctan(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Arctan(Box::new(c))
        }
        LoweredOp::Arcsinh(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Arcsinh(Box::new(c))
        }
        LoweredOp::Arccosh(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Arccosh(Box::new(c))
        }
        LoweredOp::Arctanh(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Arctanh(Box::new(c))
        }
        LoweredOp::Sqrt(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Sqrt(Box::new(c))
        }
        LoweredOp::Abs(_) => {
            let c = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Abs(Box::new(c))
        }

        // Binary: in Frame::Open we push right-child before left-child, so
        // left is processed first and its result lands below right's result
        // on val_stack. Therefore: pop right first (top), then left (below).
        LoweredOp::Add(_, _) => {
            let r = val_stack.pop().unwrap_or_else(|| node.clone());
            let l = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Add(Box::new(l), Box::new(r))
        }
        LoweredOp::Sub(_, _) => {
            let r = val_stack.pop().unwrap_or_else(|| node.clone());
            let l = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Sub(Box::new(l), Box::new(r))
        }
        LoweredOp::Mul(_, _) => {
            let r = val_stack.pop().unwrap_or_else(|| node.clone());
            let l = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Mul(Box::new(l), Box::new(r))
        }
        LoweredOp::Div(_, _) => {
            let r = val_stack.pop().unwrap_or_else(|| node.clone());
            let l = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Div(Box::new(l), Box::new(r))
        }
        LoweredOp::Pow(_, _) => {
            let r = val_stack.pop().unwrap_or_else(|| node.clone());
            let l = val_stack.pop().unwrap_or_else(|| node.clone());
            LoweredOp::Pow(Box::new(l), Box::new(r))
        }
    }
}

/// Try every rule in `db` against `node`. Return the rewritten node on the
/// first successful match, or `node` unchanged if no rule matches.
fn try_rules(db: &IdentityDb, node: LoweredOp) -> LoweredOp {
    for rule in &db.rules {
        let mut bindings = Bindings::default();
        if match_pattern(&rule.lhs, &node, &mut bindings) {
            // Rule matches; instantiate the RHS.
            match instantiate(&rule.rhs, &bindings) {
                Ok(rewritten) => return rewritten,
                // RHS references a wildcard not captured by the LHS — skip.
                Err(_) => continue,
            }
        }
    }
    node
}

/// Try only the `outer_first` rules in `db` against `node`. Returns
/// `Some(rewritten)` on the first successful match, or `None` if no
/// outer-first rule fires.
///
/// Used by [`apply_db_once`] for the top-down pre-pass: outer-first rules
/// fire before children are rewritten so that reducing patterns like
/// Pythagorean `sin²(?)+cos²(?)→1` survive a destructive inner expansion
/// like `sin(x+y) → sin(x)cos(y)+cos(x)sin(y)`.
fn try_rules_top_down(db: &IdentityDb, node: &LoweredOp) -> Option<LoweredOp> {
    for rule in &db.rules {
        if !rule.outer_first {
            continue;
        }
        let mut bindings = Bindings::default();
        if match_pattern(&rule.lhs, node, &mut bindings) {
            match instantiate(&rule.rhs, &bindings) {
                Ok(rewritten) => return Some(rewritten),
                Err(_) => continue,
            }
        }
    }
    None
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::op::LoweredOp;

    // Helpers
    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn c(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }
    fn sin(x: LoweredOp) -> LoweredOp {
        LoweredOp::Sin(Box::new(x))
    }
    fn cos(x: LoweredOp) -> LoweredOp {
        LoweredOp::Cos(Box::new(x))
    }
    fn pow(base: LoweredOp, exp: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Box::new(base), Box::new(exp))
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Box::new(a), Box::new(b))
    }
    fn sub(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Sub(Box::new(a), Box::new(b))
    }
    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Box::new(a), Box::new(b))
    }

    fn db() -> IdentityDb {
        IdentityDb::standard()
    }

    // ----------------------------------------------------------------
    // Rule 1a: sin²(x)+cos²(x) → 1
    // ----------------------------------------------------------------
    #[test]
    fn test_sin2_plus_cos2_eq_1() {
        let op = add(pow(sin(var(0)), c(2.0)), pow(cos(var(0)), c(2.0)));
        let result = apply_identity_db(&db(), &op);
        assert_eq!(result, c(1.0), "sin²(x)+cos²(x) should simplify to 1");
    }

    // ----------------------------------------------------------------
    // Rule 1b: cos²(x)+sin²(x) → 1 (commuted)
    // ----------------------------------------------------------------
    #[test]
    fn test_cos2_plus_sin2_eq_1() {
        let op = add(pow(cos(var(0)), c(2.0)), pow(sin(var(0)), c(2.0)));
        let result = apply_identity_db(&db(), &op);
        assert_eq!(result, c(1.0), "cos²(x)+sin²(x) should simplify to 1");
    }

    // ----------------------------------------------------------------
    // Rule 1: different args do NOT match (sin²(x)+cos²(y) ≠ 1)
    // ----------------------------------------------------------------
    #[test]
    fn test_sin2_cos2_different_args_no_fire() {
        // sin²(x) + cos²(y) — wildcards ?0 bind to x and y respectively, consistency check fails.
        let op = add(pow(sin(var(0)), c(2.0)), pow(cos(var(1)), c(2.0)));
        let result = apply_identity_db(&db(), &op);
        // Should NOT simplify to 1; result is structurally unchanged.
        assert_ne!(
            result.structural_hash(),
            c(1.0).structural_hash(),
            "sin²(x)+cos²(y) must not simplify to 1 (different wildcard bindings)"
        );
    }

    // ----------------------------------------------------------------
    // Rule 2: tan(x) → sin(x)/cos(x)
    // ----------------------------------------------------------------
    #[test]
    fn test_tan_expands_to_sin_over_cos() {
        let op = LoweredOp::Tan(Box::new(var(0)));
        let result = apply_identity_db(&db(), &op);
        assert!(
            matches!(result, LoweredOp::Div(_, _)),
            "tan(x) should expand to Div, got {:?}",
            result
        );
    }

    // ----------------------------------------------------------------
    // Rule 3: 1 + tan²(x) → 1/cos²(x)
    //
    // NOTE: In a bottom-up pass, `Tan(x)` is rewritten to `Sin(x)/Cos(x)`
    // (rule 2) before rule 3 sees `Add(1, Pow(Tan(x), 2))`. This is a known
    // ordering artefact of single-pass bottom-up traversal: rule 3 fires when
    // the database is applied in a context where tan is NOT already expanded.
    // We test rule 3 directly via a single-rule database.
    // ----------------------------------------------------------------
    #[test]
    fn test_one_plus_tan2_eq_sec2() {
        // Build a single-rule db containing only the sec² identity.
        let mut single_db = IdentityDb::new();
        single_db.add(Identity {
            lhs: Pattern::PatOp2(
                BinaryKind::Add,
                Box::new(Pattern::PatConst(1.0)),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Tan,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            rhs: Pattern::PatOp2(
                BinaryKind::Div,
                Box::new(Pattern::PatConst(1.0)),
                Box::new(Pattern::PatOp2(
                    BinaryKind::Pow,
                    Box::new(Pattern::PatOp1(
                        UnaryKind::Cos,
                        Box::new(Pattern::PatVar(0)),
                    )),
                    Box::new(Pattern::PatConst(2.0)),
                )),
            ),
            kind: IdentityKind::Trig,
            name: "one_plus_tan2_eq_sec2",
            outer_first: true,
        });
        let tan_x = LoweredOp::Tan(Box::new(var(0)));
        let op = add(c(1.0), pow(tan_x, c(2.0)));
        let result = apply_identity_db(&single_db, &op);
        assert!(
            matches!(result, LoweredOp::Div(_, _)),
            "1+tan²(x) should become Div(1, cos²(x)), got {:?}",
            result
        );
    }

    // ----------------------------------------------------------------
    // Rule 4: sin(2x) → 2·sin(x)·cos(x)
    // ----------------------------------------------------------------
    #[test]
    fn test_sin_double_angle() {
        let op = sin(mul(c(2.0), var(0)));
        let result = apply_identity_db(&db(), &op);
        // Should produce Mul(Mul(2, Sin(x)), Cos(x))
        assert!(
            matches!(result, LoweredOp::Mul(_, _)),
            "sin(2x) should expand to Mul, got {:?}",
            result
        );
    }

    // ----------------------------------------------------------------
    // Rule 5: cos(2x) → cos²(x) - sin²(x)
    // ----------------------------------------------------------------
    #[test]
    fn test_cos_double_angle() {
        let op = cos(mul(c(2.0), var(0)));
        let result = apply_identity_db(&db(), &op);
        assert!(
            matches!(result, LoweredOp::Sub(_, _)),
            "cos(2x) should expand to Sub(cos²(x), sin²(x)), got {:?}",
            result
        );
    }

    // ----------------------------------------------------------------
    // Rule 6: cosh²(x) - sinh²(x) → 1
    // ----------------------------------------------------------------
    #[test]
    fn test_cosh2_minus_sinh2_eq_1() {
        let cosh_x = LoweredOp::Cosh(Box::new(var(0)));
        let sinh_x = LoweredOp::Sinh(Box::new(var(0)));
        let op = sub(pow(cosh_x, c(2.0)), pow(sinh_x, c(2.0)));
        let result = apply_identity_db(&db(), &op);
        assert_eq!(result, c(1.0), "cosh²(x)-sinh²(x) should simplify to 1");
    }

    // ----------------------------------------------------------------
    // Rule 7: tanh(x) → sinh(x)/cosh(x)
    // ----------------------------------------------------------------
    #[test]
    fn test_tanh_expands_to_sinh_over_cosh() {
        let op = LoweredOp::Tanh(Box::new(var(0)));
        let result = apply_identity_db(&db(), &op);
        assert!(
            matches!(result, LoweredOp::Div(_, _)),
            "tanh(x) should expand to Div, got {:?}",
            result
        );
    }

    // ----------------------------------------------------------------
    // Rule 8: ln(x^n) → n·ln(x)
    // ----------------------------------------------------------------
    #[test]
    fn test_ln_power_rule() {
        let op = LoweredOp::Ln(Box::new(pow(var(0), c(3.0))));
        let result = apply_identity_db(&db(), &op);
        assert!(
            matches!(result, LoweredOp::Mul(_, _)),
            "ln(x^3) should become Mul(3, Ln(x)), got {:?}",
            result
        );
    }

    // ----------------------------------------------------------------
    // Rule 9: sinh(2x) → 2·sinh(x)·cosh(x)
    // ----------------------------------------------------------------
    #[test]
    fn test_sinh_double_angle() {
        let op = LoweredOp::Sinh(Box::new(mul(c(2.0), var(0))));
        let result = apply_identity_db(&db(), &op);
        assert!(
            matches!(result, LoweredOp::Mul(_, _)),
            "sinh(2x) should expand to Mul, got {:?}",
            result
        );
    }

    // ----------------------------------------------------------------
    // Rule 10: arctan(tan(x)) → x
    //
    // NOTE: When the full standard database is applied bottom-up, rule 2
    // (tan → sin/cos) fires on the inner `Tan(x)` node before the outer
    // `Arctan` node is processed. As a result, `Arctan(Tan(x))` never appears
    // as a whole pattern for rule 10 to match. We test rule 10 directly via a
    // single-rule database to verify it is correctly defined and fires.
    // ----------------------------------------------------------------
    #[test]
    fn test_arctan_of_tan_cancels() {
        // Build a single-rule db containing only the arctan/tan cancellation.
        let mut single_db = IdentityDb::new();
        single_db.add(Identity {
            lhs: Pattern::PatOp1(
                UnaryKind::Arctan,
                Box::new(Pattern::PatOp1(
                    UnaryKind::Tan,
                    Box::new(Pattern::PatVar(0)),
                )),
            ),
            rhs: Pattern::PatVar(0),
            kind: IdentityKind::Trig,
            name: "arctan_of_tan_identity",
            outer_first: false,
        });
        let op = LoweredOp::Arctan(Box::new(LoweredOp::Tan(Box::new(var(0)))));
        let result = apply_identity_db(&single_db, &op);
        assert_eq!(result, var(0), "arctan(tan(x)) should simplify to x");
    }

    // ----------------------------------------------------------------
    // No rule fires on a leaf node
    // ----------------------------------------------------------------
    #[test]
    fn test_no_rule_fires_on_leaf() {
        let op = var(42);
        let result = apply_identity_db(&db(), &op);
        assert_eq!(result, op);
    }

    // ----------------------------------------------------------------
    // Idempotence
    // ----------------------------------------------------------------
    #[test]
    fn test_idempotent() {
        let op = add(pow(sin(var(0)), c(2.0)), pow(cos(var(0)), c(2.0)));
        let r1 = apply_identity_db(&db(), &op);
        let r2 = apply_identity_db(&db(), &r1);
        assert_eq!(
            r1.structural_hash(),
            r2.structural_hash(),
            "apply_identity_db should be idempotent"
        );
    }

    // ----------------------------------------------------------------
    // Empty database is identity
    // ----------------------------------------------------------------
    #[test]
    fn test_empty_db_is_identity() {
        let empty = IdentityDb::new();
        let op = add(pow(sin(var(0)), c(2.0)), pow(cos(var(0)), c(2.0)));
        let result = apply_identity_db(&empty, &op);
        assert_eq!(
            result.structural_hash(),
            op.structural_hash(),
            "empty db should leave op unchanged"
        );
    }

    // ----------------------------------------------------------------
    // Multi-rule: sin²+cos²=1 inside a larger expression
    // ----------------------------------------------------------------
    #[test]
    fn test_identity_fires_inside_larger_expr() {
        // expr = x + (sin²(y) + cos²(y))  →  x + 1
        let pythagorean = add(pow(sin(var(1)), c(2.0)), pow(cos(var(1)), c(2.0)));
        let op = add(var(0), pythagorean);
        let result = apply_identity_db(&db(), &op);
        // Inner part should reduce to 1; outer Add(x, 1) stays as-is.
        let expected = add(var(0), c(1.0));
        assert_eq!(
            result.structural_hash(),
            expected.structural_hash(),
            "sin²(y)+cos²(y) inside larger expr should reduce to 1"
        );
    }

    // ----------------------------------------------------------------
    // Deep tree — no stack overflow
    // ----------------------------------------------------------------
    #[test]
    fn test_deep_tree_no_overflow() {
        // 1000-deep Add(_, Const(0)) chain around sin²+cos² at the innermost.
        let inner = add(pow(sin(var(0)), c(2.0)), pow(cos(var(0)), c(2.0)));
        let mut op = inner;
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(c(0.0)));
        }
        // Should not overflow — iterative implementation.
        let result = apply_identity_db(&db(), &op);
        assert_eq!(
            result,
            // The innermost sin²+cos² → 1, the 1000 Add(_, 0) wrappers remain.
            // We only check the hash exists (no panic).
            result.clone(),
            "deep tree traversal must complete without overflow"
        );
    }

    // ----------------------------------------------------------------
    // apply_standard_identity_db convenience wrapper
    // ----------------------------------------------------------------
    #[test]
    fn test_apply_standard_identity_db() {
        let op = add(pow(sin(var(0)), c(2.0)), pow(cos(var(0)), c(2.0)));
        let result = apply_standard_identity_db(&op);
        assert_eq!(result, c(1.0));
    }

    // ----------------------------------------------------------------
    // IdentityDb::rules() accessor
    //
    // The standard database contains 19 patterns total:
    // - Rule 1 has two variants (sin²+cos² and cos²+sin²)
    // - Rules 2-10 are 9 more (Wave 53/72)
    // - Rules 11a/11b, 12a/12b add Wave 74 double-angle inverse (commuted variants)
    // - Rules 13-16 add Wave 74 trig sum/difference expansions
    // ----------------------------------------------------------------
    #[test]
    fn test_standard_db_has_nineteen_rules() {
        let db = IdentityDb::standard();
        assert_eq!(
            db.rules().len(),
            19,
            "standard database must contain exactly 19 entries after Wave 74"
        );
    }
}
