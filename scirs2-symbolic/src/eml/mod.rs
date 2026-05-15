//! # EML — Elementary Mathematical Library uniform binary tree
//!
//! Implementation of the EML construction (Odrzywolek 2026,
//! [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)) where every
//! elementary function reduces to a single binary operator
//! `eml(x, y) = exp(x) - ln(y)` plus the constant `1`.
//!
//! ## Modules
//!
//! - [`tree`] — `EmlNode` + `EmlTree`, the uniform binary tree IR
//! - [`canonical`] — `Canonical` namespace of elementary-function constructors
//! - [`op`] — `LoweredOp` + `OxiOp`, the flat operator IR and stack-machine tape
//! - [`mod@lower`] — lowering `EmlTree → LoweredOp` and the inverse `raise`
//! - [`eval`] — iterative stack-machine evaluator for `LoweredOp` (real + complex)
//! - [`interval`] — outward-rounded interval arithmetic with sin/cos critical-point splits
//! - [`simplify`] — fixed-point algebraic simplification for `LoweredOp`
//! - [`mod@grad`] — symbolic gradient on `LoweredOp` (chain/product/quotient
//!   rules, constant-exponent `Pow` fast path, native `Sqrt`/`Abs` rules);
//!   also exposes `grad_all`, `jacobian`, `hessian`
//! - [`parser`] — text → [`EmlTree`] (with `eml(...)` / `E(...)` notation) + `to_compact_string`
//! - [`display`] — `Display` impls for [`EmlTree`] / [`EmlNode`] / [`LoweredOp`] + LaTeX export
//! - [`bridge`] — `Expr ↔ LoweredOp` adapter via `ToLowered`/`FromLowered`
//!   traits with deterministic `VarMap`
//! - `hash` (crate-private) — shared structural-hash machinery (u128 via two-seed `ahash`)
//!
//! ## Soundness
//!
//! All traversals use iterative work-stack patterns over a heap-allocated
//! `Vec`; no recursive functions over `EmlNode` ship in production code.
//! `Canonical::sin(x)` produces a 543-node-deep tree, so any recursive
//! traversal would blow the OS stack on plausible user inputs.
//!
//! ## Hash-Cons
//!
//! Identical subtrees are deduplicated via a thread-local
//! `HashMap<u128, Weak<EmlNode>>` pool keyed on the structural hash.
//! Set the env var `SCIRS2_SYMBOLIC_NO_HASHCONS=1` to disable hash-cons
//! (debugging escape hatch).

#![warn(missing_docs)]

pub mod bridge;
pub mod canonical;
pub mod display;
pub mod eval;
pub mod grad;
pub(crate) mod hash;
pub mod interval;
pub mod lower;
pub mod op;
pub mod parser;
pub mod simplify;
pub mod tree;

pub use bridge::{FromLowered, ToLowered, VarMap};
pub use canonical::Canonical;
pub use display::to_latex;
pub use eval::{eval_complex, eval_ops_complex, eval_ops_real, eval_real, EvalCtx};
pub use grad::{grad, grad_all, hessian, jacobian};
pub use interval::{eval_interval, Interval};
pub use lower::{lower, raise};
pub use op::{LoweredOp, OxiOp};
pub use parser::{parse, to_compact_string};
pub use simplify::{simplify_op, simplify_op_full};
pub use tree::{EmlNode, EmlTree, PostOrderIter};
