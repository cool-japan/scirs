//! Symbolic cross-crate facades for scirs2-neural.
//!
//! This module provides integration between the neural network layer
//! infrastructure and `scirs2_symbolic`:
//!
//! * [`weight_init`] — initialise linear layer weights `(W, b)` by
//!   least-squares fitting a symbolic formula over a sample grid.
//! * [`formula_extract`] — discover compact symbolic formulas that
//!   approximate the input-output behaviour of any callable (including
//!   trained neural network layers).
//! * [`rope_attention`] — symbolic closed-form RoPE attention logit;
//!   proves that the dot-product of RoPE-rotated query and key depends
//!   only on the relative position `(m − n)`.
//!
//! All items require the `symbolic` crate feature.

pub mod formula_extract;
pub mod rope_attention;
pub mod weight_init;

pub use formula_extract::{
    extract_formula_from_callable, FormulaExtractionConfig, FormulaExtractionError,
};
pub use rope_attention::{
    build_vars, rope_attention_logit, RopeAttentionError, RopeAttentionSymbolic, RopeVarMap,
};
pub use weight_init::{init_weights_from_formula, InitFromFormulaError};
