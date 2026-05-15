//! SI-unit-aware dimensional analysis for symbolic mathematics.
//!
//! Provides [`Dimension`] (a 7-tuple of SI base unit exponents) and
//! [`UnitAware`] (a `LoweredOp` paired with a [`Dimension`]) for
//! dimensional consistency checking in formula construction and
//! symbolic regression.
//!
//! # SI Base Units
//!
//! | Symbol | Quantity | Index |
//! |--------|----------|-------|
//! | m | length | 0 |
//! | kg | mass | 1 |
//! | s | time | 2 |
//! | A | current | 3 |
//! | K | temperature | 4 |
//! | mol | amount of substance | 5 |
//! | cd | luminous intensity | 6 |
//!
//! # Use in Symbolic Regression
//!
//! Given features with known dimensions and a target with a known
//! dimension, candidate formulas constructed via [`UnitAware`]
//! that produce a mismatched [`Dimension`] can be pruned without
//! ever evaluating them.  This is the building block for
//! dimensionally-constrained SR pruning.
//!
//! # Limitations
//!
//! Exponents are stored as `i32`; fractional dimensions (e.g. `[m^(1/2)]`)
//! are not representable.  [`UnitAware::sqrt`] therefore requires that
//! all base-unit exponents be even.  A future revision may move to
//! rational exponents.
//!
//! # Adapted from oxieml v0.1.0, `src/units.rs`
//!
//! The 7-tuple shape and the dimensional-arithmetic rules
//! (product/quotient/integer-power, transcendental dimensionless
//! constraint, sqrt even-exponent rule) follow the reference
//! implementation.  The [`UnitAware`] integration with `LoweredOp`
//! is original to scirs2-symbolic.

#![warn(missing_docs)]

pub mod aware;
pub mod dimension;
pub mod infer;

pub use aware::{UnitAware, UnitError};
pub use dimension::{Dimension, SiBase};
pub use infer::infer_dimension;
