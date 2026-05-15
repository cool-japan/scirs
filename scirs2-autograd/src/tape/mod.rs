//! Tape-based automatic differentiation
//!
//! This module re-exports the original `GradientTape` from `crate::tape` (now
//! renamed to avoid conflicts) and adds the enhanced tape implementations:
//! [`ReverseTape`], [`ForwardTape`], [`MixedMode`], [`TapeCheckpoint`], and
//! [`TapeOptimizer`].

pub mod enhanced;

#[cfg(feature = "symbolic")]
pub mod dispatch;
#[cfg(feature = "symbolic")]
pub mod eml_tape;

pub use enhanced::{
    ForwardTape, MixedMode, ReverseTape, TapeCheckpoint, TapeOp, TapeOptimizationReport,
    TapeOptimizer,
};
