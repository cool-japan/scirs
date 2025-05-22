//! Time series decomposition methods
//!
//! This module provides implementations for decomposing time series into trend,
//! seasonal, and residual components.

pub mod common;
pub mod exponential;
pub mod seasonal;
pub mod ssa;
pub mod stl;
pub mod str;
pub mod tbats;

// Re-export common types and functions
pub use common::*;

// Re-export decomposition methods
pub use exponential::*;
pub use seasonal::*;
pub use ssa::*;
pub use stl::*;
pub use str::*;
pub use tbats::*;
