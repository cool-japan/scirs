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
//! | [`diff`] | Symbolic differentiation ([`diff`], [`diff_n`]) |
//! | [`simplify`] | Constant folding + identity rules ([`simplify`], [`simplify_full`]) |
//! | [`eval`] | Numeric evaluation ([`eval`]) |
//! | [`display`] | `Display` impl for infix notation |
//! | [`error`] | [`SymbolicError`] variants |
//!
//! ## Design Notes
//!
//! - **Pure Rust, no external dependencies** (beyond `thiserror` for error derives).
//! - **No `unwrap()`** in production code — all fallible paths return `Result`.
//! - Expression trees are **immutable** `Clone`-able values; there is no shared mutable
//!   state, making them safe to use across threads.

pub mod diff;
pub mod display;
pub mod error;
pub mod eval;
pub mod expr;
pub mod simplify;

pub use diff::{diff, diff_n};
pub use error::SymbolicError;
pub use eval::eval;
pub use expr::Expr;
pub use simplify::{simplify, simplify_full};
