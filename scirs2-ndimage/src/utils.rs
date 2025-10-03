//! Common utility functions for ndimage processing
//!
//! This module provides shared utility functions used across the ndimage crate.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::numeric::{Float, FromPrimitive};

/// Helper function for safe conversion of hardcoded constants from f64 to generic float type
#[allow(dead_code)]
pub fn safe_f64_to_float<T: Float + FromPrimitive>(value: f64) -> NdimageResult<T> {
    T::from_f64(value).ok_or_else(|| {
        NdimageError::ComputationError(format!(
            "Failed to convert constant {} to float type",
            value
        ))
    })
}

/// Helper function for safe float to f64 conversion
#[allow(dead_code)]
pub fn safe_float_to_f64<T: Float>(value: T) -> NdimageResult<f64> {
    value
        .to_f64()
        .ok_or_else(|| NdimageError::ComputationError("Failed to convert float to f64".to_string()))
}

/// Helper function for safe usize conversion
#[allow(dead_code)]
pub fn safe_usize_to_float<T: Float + FromPrimitive>(value: usize) -> NdimageResult<T> {
    T::from_usize(value).ok_or_else(|| {
        NdimageError::ComputationError(format!("Failed to convert usize {} to float type", value))
    })
}

/// Helper function for safe float to usize conversion
#[allow(dead_code)]
pub fn safe_float_to_usize<T: Float>(value: T) -> NdimageResult<usize> {
    value.to_usize().ok_or_else(|| {
        NdimageError::ComputationError("Failed to convert float to usize".to_string())
    })
}
