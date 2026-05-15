//! JIT compilation of [`crate::eml::op::LoweredOp`] to native machine code.
//!
//! Provides [`to_jit`] which compiles a `LoweredOp` formula to a native
//! function via Cranelift. Repeated compilation of the same formula
//! (keyed by structural u128 hash) hits the [`JitCache`].
//!
//! # Performance
//!
//! Typical speedup over the iterative interpreter ([`crate::eml::eval::eval_real`])
//! on a deeply nested transcendental formula: ~50–100×. Compilation cost
//! amortises after roughly 100 evaluations of the same formula.
//!
//! # Feature gate
//!
//! All items in this module are gated behind the crate-local `jit` cargo
//! feature; default builds (and `--no-default-features`) do not pull in
//! the Cranelift dependency stack.
//!
//! # Example
//!
//! ```no_run
//! use scirs2_symbolic::eml::op::LoweredOp;
//! use scirs2_symbolic::compile::to_jit;
//!
//! // f(x) = x² + 2x + 1
//! let x = LoweredOp::Var(0);
//! let f = LoweredOp::Add(
//!     Box::new(LoweredOp::Add(
//!         Box::new(LoweredOp::Mul(Box::new(x.clone()), Box::new(x.clone()))),
//!         Box::new(LoweredOp::Mul(Box::new(LoweredOp::Const(2.0)), Box::new(x))),
//!     )),
//!     Box::new(LoweredOp::Const(1.0)),
//! );
//! let func = to_jit(&f).expect("compile");
//! assert!((func.eval(&[3.0]) - 16.0).abs() < 1e-12);
//! ```

#![cfg(feature = "jit")]

pub mod cache;
pub mod jit;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use cache::JitCache;
pub use jit::{to_jit, JitError, JitFunction};

#[cfg(feature = "gpu")]
pub use gpu::{to_gpu, to_jit_auto, GpuError, GpuKernel, JitDispatch};
