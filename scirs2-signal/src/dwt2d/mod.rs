//! Two-dimensional Discrete Wavelet Transform (DWT2D)
//!
//! This module provides implementations of the 2D Discrete Wavelet Transform,
//! useful for image processing, compression, and multi-resolution analysis
//! of 2D signals like images.

// Declare modules
pub mod types;
pub mod simd;

// Re-export main types for backward compatibility
pub use types::{Dwt2dConfig, Dwt2dResult, MemoryPool, ThresholdMethod};

// Re-export SIMD functions
pub use simd::{simd_calculate_energy, simd_threshold_coefficients};

// Import the original implementation (temporarily keeping it as a separate module)
mod legacy;
pub use legacy::*;