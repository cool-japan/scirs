//! `Expr ↔ LoweredOp` adapter.
//!
//! Bridges the legacy [`crate::Expr`] enum (named-variable, recursive) and
//! the EML [`LoweredOp`] flat IR (indexed-variable). The bridge uses a
//! [`VarMap`] for deterministic name ↔ index mapping.
//!
//! # Determinism
//!
//! [`VarMap::from_expr`] calls [`crate::Expr::variables`] which returns a
//! sorted `BTreeSet<String>`. Two calls on the same `Expr` produce
//! byte-identical `VarMap`s and therefore byte-identical `Var(usize)`
//! indices.
//!
//! # Round-trip
//!
//! `Expr::from_lowered(&expr.to_lowered_with(&map)?, &map)?` produces a
//! mathematically equivalent `Expr` (after canonicalisation, NOT bit-equal
//! — `Const(2.0) + Const(3.0)` round-trips to `Const(5.0)` once the result
//! traverses [`crate::eml::lower::lower`]/[`crate::eml::lower::raise`]
//! since intermediate constants are folded). The direct
//! `to_lowered`/`from_lowered` pair preserves structure exactly.
//!
//! # Recursion note
//!
//! Both [`ToLowered`] and [`FromLowered`] for [`crate::Expr`] use
//! straightforward recursion — `Expr` is user-constructed and shallow in
//! practice (unlike [`crate::eml::tree::EmlNode`] where
//! `Canonical::sin(x)` produces a 543-node-deep tree, mandating iterative
//! traversals everywhere else in `eml/`). If a deeply-nested `Expr` is
//! ever encountered, the same iterative work-stack pattern from the rest
//! of this module can be applied.

use crate::eml::op::LoweredOp;
use crate::error::EmlError;
use crate::expr::Expr;
use std::collections::BTreeSet;

/// Deterministic mapping between variable names and indices.
///
/// `names[i]` is the name of variable index `i`. Names are sorted
/// alphabetically by construction (via `BTreeSet`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VarMap {
    /// Variable names indexed by position (sorted alphabetically).
    pub names: Vec<String>,
}

impl VarMap {
    /// Build a `VarMap` from an [`Expr`].
    ///
    /// Calls [`Expr::variables`] (sorted `BTreeSet<String>`) and collects
    /// to a `Vec`. Determinism: two calls on the same `Expr` produce
    /// byte-identical `VarMap`s.
    pub fn from_expr(expr: &Expr) -> Self {
        Self {
            names: expr.variables().into_iter().collect(),
        }
    }

    /// Build a `VarMap` from an explicit list of names.
    ///
    /// Names are deduplicated and re-sorted alphabetically by routing
    /// through a `BTreeSet`, ensuring deterministic ordering regardless
    /// of input order.
    pub fn new(names: Vec<String>) -> Self {
        let set: BTreeSet<String> = names.into_iter().collect();
        Self {
            names: set.into_iter().collect(),
        }
    }

    /// Find the index of a variable name. Returns `None` if not present.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Get the name of a variable index. Returns `None` if out of bounds.
    pub fn name_of(&self, idx: usize) -> Option<&str> {
        self.names.get(idx).map(|s| s.as_str())
    }

    /// Number of variables.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// True if no variables.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Trait for types convertible to [`LoweredOp`].
///
/// The two-step API ([`Self::to_lowered`] / [`Self::to_lowered_with`])
/// supports both "discover variables and convert" and "convert against an
/// already-known mapping" workflows.
pub trait ToLowered {
    /// Convert to a [`LoweredOp`] using a pre-built [`VarMap`].
    ///
    /// Fails with [`EmlError::UnknownVariable`] when the value references
    /// a name not present in `map`.
    fn to_lowered_with(&self, map: &VarMap) -> Result<LoweredOp, EmlError>;

    /// Build a [`VarMap`] and convert in one step.
    fn to_lowered(&self) -> Result<(LoweredOp, VarMap), EmlError> {
        let map = self.var_map();
        let op = self.to_lowered_with(&map)?;
        Ok((op, map))
    }

    /// Build a [`VarMap`] for this value (default: empty).
    fn var_map(&self) -> VarMap {
        VarMap::default()
    }
}

/// Trait for types constructible from [`LoweredOp`].
///
/// Symmetric to [`ToLowered`]. Implementations may fail with
/// [`EmlError::UnboundVariableIndex`] for out-of-range indices, or with
/// [`EmlError::LoweringFailed`] when the source variant has no equivalent
/// in the target type.
pub trait FromLowered: Sized {
    /// Reconstruct a value from a [`LoweredOp`] tree using `map` to
    /// resolve `Var(usize)` indices back to names.
    fn from_lowered(op: &LoweredOp, map: &VarMap) -> Result<Self, EmlError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::Expr;

    #[test]
    fn varmap_from_expr_sorted() {
        let e = Expr::Var("z".into()) + Expr::Var("a".into()) * Expr::Var("m".into());
        let map = VarMap::from_expr(&e);
        assert_eq!(map.names, vec!["a", "m", "z"]);
    }

    #[test]
    fn varmap_determinism() {
        let e = Expr::Var("y".into()) + Expr::Var("x".into());
        let m1 = VarMap::from_expr(&e);
        let m2 = VarMap::from_expr(&e);
        assert_eq!(m1, m2);
    }

    #[test]
    fn varmap_new_dedups_and_sorts() {
        let m = VarMap::new(vec!["z".into(), "a".into(), "z".into(), "m".into()]);
        assert_eq!(m.names, vec!["a", "m", "z"]);
        assert_eq!(m.len(), 3);
        assert!(!m.is_empty());
    }

    #[test]
    fn varmap_index_and_name_lookup() {
        let m = VarMap::new(vec!["a".into(), "b".into(), "c".into()]);
        assert_eq!(m.index_of("b"), Some(1));
        assert_eq!(m.index_of("missing"), None);
        assert_eq!(m.name_of(2), Some("c"));
        assert_eq!(m.name_of(99), None);
    }

    #[test]
    fn varmap_default_is_empty() {
        let m = VarMap::default();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn to_lowered_const() {
        // 3.15 (not 3.14) — `clippy::approx_constant` flags any value within
        // ULP-distance of `f64::consts::PI`. Choice of constant is irrelevant
        // to the test (we only care that the value round-trips bit-for-bit).
        let e = Expr::Const(3.15);
        let (op, map) = e.to_lowered().expect("const lowering must succeed");
        assert_eq!(op, LoweredOp::Const(3.15));
        assert!(map.is_empty());
    }

    #[test]
    fn to_lowered_var() {
        let e = Expr::var("x");
        let (op, map) = e.to_lowered().expect("var lowering must succeed");
        assert_eq!(op, LoweredOp::Var(0));
        assert_eq!(map.names, vec!["x".to_string()]);
    }

    #[test]
    fn to_lowered_add() {
        let e = Expr::var("x") + Expr::var("y");
        let (op, map) = e.to_lowered().expect("add lowering must succeed");
        assert!(matches!(op, LoweredOp::Add(_, _)));
        assert_eq!(map.names, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn to_lowered_assigns_indices_alphabetically() {
        // Variable order in the expression: z first, then a, then m.
        // After alphabetic sorting: a → 0, m → 1, z → 2.
        let e = Expr::var("z") + Expr::var("a") * Expr::var("m");
        let (op, map) = e.to_lowered().expect("lowering must succeed");
        assert_eq!(map.names, vec!["a", "m", "z"]);
        // Top-level Add(Var(2 = z), Mul(Var(0 = a), Var(1 = m)))
        match op {
            LoweredOp::Add(left, right) => {
                assert_eq!(*left, LoweredOp::Var(2));
                match *right {
                    LoweredOp::Mul(a, b) => {
                        assert_eq!(*a, LoweredOp::Var(0));
                        assert_eq!(*b, LoweredOp::Var(1));
                    }
                    other => panic!("expected Mul, got {:?}", other),
                }
            }
            other => panic!("expected Add at root, got {:?}", other),
        }
    }

    #[test]
    fn to_lowered_unknown_var_in_explicit_map() {
        let e = Expr::var("x");
        let map = VarMap::default(); // empty
        assert!(matches!(
            e.to_lowered_with(&map),
            Err(EmlError::UnknownVariable(_))
        ));
    }

    #[test]
    fn round_trip_const() {
        // 3.15 (not 3.14) — see `to_lowered_const` above for the rationale.
        let e = Expr::Const(3.15);
        let (op, map) = e.to_lowered().expect("lowering must succeed");
        let recovered = Expr::from_lowered(&op, &map).expect("raise must succeed");
        assert_eq!(recovered, e);
    }

    #[test]
    fn round_trip_simple_formula() {
        let e = Expr::var("x") + Expr::Const(1.0);
        let (op, map) = e.to_lowered().expect("lowering must succeed");
        let recovered = Expr::from_lowered(&op, &map).expect("raise must succeed");
        assert_eq!(recovered, e);
    }

    #[test]
    fn round_trip_all_basic_variants() {
        for e in [
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
        ] {
            let (op, map) = e
                .to_lowered()
                .unwrap_or_else(|err| panic!("lowering failed for {:?}: {}", e, err));
            let recovered = Expr::from_lowered(&op, &map)
                .unwrap_or_else(|err| panic!("raise failed for {:?}: {}", e, err));
            assert_eq!(recovered, e, "round-trip failed for {:?}", e);
        }
    }

    #[test]
    fn round_trip_nested_formula() {
        // (sin(x) + cos(y)) * exp(-x)
        let e = (Expr::var("x").sin() + Expr::var("y").cos()) * (-Expr::var("x")).exp();
        let (op, map) = e.to_lowered().expect("lowering must succeed");
        let recovered = Expr::from_lowered(&op, &map).expect("raise must succeed");
        assert_eq!(recovered, e);
    }

    #[test]
    fn from_lowered_unbound_index_errors() {
        let op = LoweredOp::Var(5);
        let map = VarMap::new(vec!["x".into()]);
        assert!(matches!(
            Expr::from_lowered(&op, &map),
            Err(EmlError::UnboundVariableIndex { idx: 5, len: 1 })
        ));
    }

    #[test]
    fn from_lowered_hyperbolic_errors() {
        let op = LoweredOp::Sinh(Box::new(LoweredOp::Var(0)));
        let map = VarMap::new(vec!["x".into()]);
        assert!(matches!(
            Expr::from_lowered(&op, &map),
            Err(EmlError::LoweringFailed(_))
        ));
    }

    #[test]
    fn from_lowered_inverse_trig_errors() {
        for op in [
            LoweredOp::Arcsin(Box::new(LoweredOp::Var(0))),
            LoweredOp::Arccos(Box::new(LoweredOp::Var(0))),
            LoweredOp::Arctan(Box::new(LoweredOp::Var(0))),
            LoweredOp::Cosh(Box::new(LoweredOp::Var(0))),
            LoweredOp::Tanh(Box::new(LoweredOp::Var(0))),
            LoweredOp::Arcsinh(Box::new(LoweredOp::Var(0))),
            LoweredOp::Arccosh(Box::new(LoweredOp::Var(0))),
            LoweredOp::Arctanh(Box::new(LoweredOp::Var(0))),
        ] {
            let map = VarMap::new(vec!["x".into()]);
            assert!(
                matches!(
                    Expr::from_lowered(&op, &map),
                    Err(EmlError::LoweringFailed(_))
                ),
                "expected LoweringFailed for {:?}",
                op
            );
        }
    }
}
