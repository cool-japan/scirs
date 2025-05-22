//! Time series decomposition methods (Compatibility layer)
//!
//! This module provides backwards compatibility for the older, monolithic decomposition module.
//! For new code, prefer using the modular structure in the `decomposition` module.

// Re-export all public items from the new modular decomposition structure
pub use crate::decomposition::common::*;
pub use crate::decomposition::exponential::*;
pub use crate::decomposition::seasonal::*;
pub use crate::decomposition::ssa::*;
pub use crate::decomposition::stl::*;
pub use crate::decomposition::str::*;
pub use crate::decomposition::tbats::*;
