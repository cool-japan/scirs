//! Common types and utilities shared across GPU backends
//!
//! This module contains shared functionality that is used by multiple
//! GPU backend implementations including error handling, trait definitions,
//! and common data structures.

// Re-export traits and types that all backends need
pub use crate::error::{LinalgError, LinalgResult};
pub use crate::gpu::{
    GpuBackend, GpuBuffer, GpuContext, GpuContextAlloc, GpuDeviceInfo, GpuDeviceType,
};
pub use std::collections::HashMap;

/// External dependencies commonly used by CPU fallback functionality
#[cfg(any(
    feature = "cpu-fallback",
    not(any(
        feature = "cuda",
        feature = "opencl",
        feature = "metal",
        feature = "rocm"
    ))
))]
pub use num_cpus;

/// Common result type for backend operations
pub type BackendResult<T> = Result<T, LinalgError>;

/// Helper function to validate device ID against available devices
pub fn validate_device_id(device_id: usize, available_devices: usize) -> LinalgResult<()> {
    if device_id >= available_devices {
        return Err(LinalgError::ComputationError(format!(
            "Invalid device ID: {} (available devices: {})",
            device_id, available_devices
        )));
    }
    Ok(())
}

/// Convert bytes to human-readable format for memory reporting
pub fn format_memory_size(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Common error messages for backend implementations
pub mod error_messages {
    pub const BACKEND_NOT_INITIALIZED: &str = "Backend not initialized";
    pub const NO_DEVICES_FOUND: &str = "No compatible devices found";
    pub const DEVICE_ALLOCATION_FAILED: &str = "Device memory allocation failed";
    pub const DEVICE_COPY_FAILED: &str = "Device memory copy failed";
    pub const SYNCHRONIZATION_FAILED: &str = "Device synchronization failed";
    pub const CONTEXT_CREATION_FAILED: &str = "Failed to create device context";
    pub const BUFFER_SIZE_MISMATCH: &str = "Buffer size mismatch";
}

/// Helper macro for creating backend errors with context
#[macro_export]
macro_rules! backend_error {
    ($msg:expr) => {
        LinalgError::ComputationError($msg.to_string())
    };
    ($fmt:expr, $($arg:tt)*) => {
        LinalgError::ComputationError(format!($fmt, $($arg)*))
    };
}

/// Helper macro for checking result codes and converting to LinalgError
#[macro_export]
macro_rules! check_result {
    ($result:expr, $success_value:expr, $error_msg:expr) => {
        if $result != $success_value {
            return Err(LinalgError::ComputationError(format!(
                "{}: error code {}",
                $error_msg, $result
            )));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_device_id() {
        assert!(validate_device_id(0, 1).is_ok());
        assert!(validate_device_id(0, 2).is_ok());
        assert!(validate_device_id(1, 2).is_ok());
        assert!(validate_device_id(2, 2).is_err());
        assert!(validate_device_id(10, 1).is_err());
    }

    #[test]
    fn test_format_memory_size() {
        assert_eq!(format_memory_size(1024), "1.00 KB");
        assert_eq!(format_memory_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_memory_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_memory_size(2 * 1024 * 1024 * 1024), "2.00 GB");
    }
}
