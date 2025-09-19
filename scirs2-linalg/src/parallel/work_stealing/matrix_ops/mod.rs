//! Matrix-specific work-stealing algorithms
//!
//! This module provides work-stealing implementations for various matrix operations
//! including GEMM, decompositions, and utility functions.

pub mod decomposition;
pub mod gemm;
pub mod utilities;

// Re-export all public functions for backward compatibility
pub use decomposition::*;
pub use gemm::*;
pub use utilities::*;
