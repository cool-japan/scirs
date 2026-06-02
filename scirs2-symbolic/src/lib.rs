//! # scirs2-symbolic — Symbolic Mathematics for SciRS2
//!
//! This crate provides a symbolic expression tree, automatic symbolic differentiation,
//! algebraic simplification, and numeric evaluation of symbolic expressions.
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_symbolic::{Expr, diff, simplify, eval};
//! use std::collections::HashMap;
//!
//! // Build the expression  f(x) = x² + 3x
//! let x = Expr::var("x");
//! let f = x.clone().pow(Expr::from(2.0)) + Expr::from(3.0) * x.clone();
//!
//! // Differentiate symbolically:  f'(x) = 2x + 3
//! let df = simplify(&diff(&f, "x"));
//!
//! // Evaluate at x = 2:  2*2 + 3 = 7
//! let mut vars = HashMap::new();
//! vars.insert("x", 2.0_f64);
//! let result = eval(&df, &vars).unwrap();
//! assert!((result - 7.0).abs() < 1e-10);
//! ```
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`expr`] | [`Expr`] enum and arithmetic operator overloads |
//! | [`mod@diff`] | Symbolic differentiation ([`fn@diff`], [`diff_n`]) |
//! | [`mod@simplify`] | Constant folding + identity rules ([`fn@simplify`], [`simplify_full`]) |
//! | [`mod@eval`] | Numeric evaluation ([`fn@eval`]) |
//! | [`display`] | `Display` impl for infix notation |
//! | [`error`] | [`SymbolicError`] variants |
//!
//! ## Design Notes
//!
//! - **Pure Rust, no external dependencies** (beyond `thiserror` for error derives).
//! - **No `unwrap()`** in production code — all fallible paths return `Result`.
//! - Expression trees are **immutable** `Clone`-able values; there is no shared mutable
//!   state, making them safe to use across threads.
//!
//! ## LaTeX Export
//!
//! Any [`eml::LoweredOp`] can be rendered as a LaTeX math string:
//!
//! ```rust
//! use scirs2_symbolic::eml::{to_latex, LoweredOp};
//!
//! let op = LoweredOp::Div(
//!     Box::new(LoweredOp::Const(1.0)),
//!     Box::new(LoweredOp::Var(0)),
//! );
//! assert_eq!(to_latex(&op), "\\frac{1}{x_{0}}");
//! ```

pub mod attention;
pub mod autograd_bridge;
pub mod cas;
#[cfg(feature = "jit")]
pub mod compile;
pub mod diffgeom;
pub mod neural_priors;

#[cfg(feature = "macros")]
pub use scirs2_symbolic_macros::{eml_pattern, eml_template};
pub mod diff;
pub mod display;
pub mod eml;
pub mod error;
pub mod eval;
pub mod expr;
pub mod regression;
pub mod simplify;
pub mod units;

pub use autograd_bridge::{BinaryKind, SymbolicTape, TapeNode, ToTape, UnaryKind};
pub use diff::{diff, diff_n};
pub use eml::{
    eval_interval, Canonical, EmlNode, EmlTree, FromLowered, Interval, LoweredOp, OxiOp, ToLowered,
    VarMap,
};
pub use error::{EmlError, SymbolicError};
pub use eval::eval;
pub use expr::Expr;
pub use neural_priors::{
    discover_series_prior, eval_series_prior, series_prior_regularization, SeriesPrior,
    SymRegConfig,
};
pub use regression::{
    discover, discover_multi, discover_multi_best, discover_ode, discover_ode_best,
    with_constraints, BuildingBlock, ConstrainedConfig, Constraint, DiscoveredFormula, Fitness,
    OdeConfig, SrConfig,
};
pub use simplify::{simplify, simplify_full};
pub use units::{Dimension, SiBase, UnitAware, UnitError};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_interval_reexport_works() {
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let result = eval_interval(&op, &[Interval::new(1.0, 2.0)]);
        assert!(result.lo <= 2.0);
        assert!(result.hi >= 3.0);
    }
}
