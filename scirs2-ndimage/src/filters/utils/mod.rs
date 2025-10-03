//! Utility Functions for Filter Operations
//!
//! This module provides comprehensive utilities for filtering operations including:
//! - Kernel generation (Gaussian kernels, separable filtering)
//! - Convolution operations (memory-efficient implementations)
//! - Array padding (comprehensive border handling modes)
//! - Window functions (sliding window operations)
//!
//! All functions are organized into focused submodules while maintaining
//! full backward compatibility through comprehensive re-exports.

pub mod convolution;
pub mod kernel;
pub mod padding;
pub mod window;

// Re-export all public functions for backward compatibility

// Kernel generation functions
pub use kernel::{calculate_kernel_size, generate_gaussian_kernel, separable_gaussian_blur};

// Convolution functions
pub use convolution::memory_efficient_convolution;

// Padding functions
pub use padding::{get_window, pad_array};

// Window functions
pub use window::apply_window_function;

// Re-export helper function from kernel module
// (This was previously defined in the main utils.rs file)
use scirs2_core::ndarray::{Array, Dimension};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::BorderMode;
use crate::error::{NdimageError, NdimageResult};
use crate::utils::safe_f64_to_float;

/// Helper function for safe usize conversion
#[allow(dead_code)]
pub fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}
