//! Cross-Module Integration Framework for SciRS2 v0.2.0
//!
//! This module provides comprehensive cross-module integration capabilities
//! including zero-copy data sharing, unified configuration, type conversion
//! traits, and common interface definitions for ecosystem-wide interoperability.
//!
//! # Features
//!
//! - **Zero-Copy Operations**: Efficient data sharing between modules without memory copies
//! - **Unified Configuration**: Ecosystem-wide settings management with validation
//! - **Type Conversion Traits**: Consistent patterns for converting between module types
//! - **Interface Traits**: Common traits for module interoperability
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::integration::prelude::*;
//!
//! // Use unified configuration
//! let config = EcosystemConfig::builder()
//!     .with_precision(Precision::Double)
//!     .with_parallel(true)
//!     .build();
//!
//! // Zero-copy array sharing
//! let data = vec![1.0f64, 2.0, 3.0, 4.0];
//! let view = SharedArrayView::from_slice(&data);
//! ```

pub mod config;
pub mod conversion;
pub mod traits;
pub mod zero_copy;

// Mobile FFI bindings (v0.2.0). The bindings are backed by the ARM NEON
// kernels in `crate::simd::neon`, which only exist with the `simd` feature on
// aarch64/arm, so the module is gated on all three (x86_64 Android emulators
// and default-feature builds simply do not get the FFI symbols).
#[cfg(all(
    feature = "simd",
    any(target_os = "ios", target_os = "android"),
    any(target_arch = "aarch64", target_arch = "arm")
))]
pub mod mobile_ffi;

// Re-export primary types
pub use config::{
    DiagnosticsConfig, EcosystemConfig, EcosystemConfigBuilder, LogLevel, MemoryConfig,
    ModuleConfig, NumericConfig, ParallelConfig, Precision, PrecisionConfig,
};
pub use conversion::{
    ArrayConvert, ConversionError, ConversionOptions, ConversionResult, CrossModuleConvert,
    DataFlowConverter, LosslessConvert, LossyConvert, TypeAdapter,
};
pub use traits::{
    Composable, Configurable, CrossModuleOperator, DataConsumer, DataProvider, Diagnosable,
    Identifiable, ModuleCapability, ModuleInterface, ResourceAware, Serializable,
    VersionedInterface,
};
pub use zero_copy::{
    Alignment, ArrayBridge, BorrowedArray, BufferMut, BufferRef, ContiguousMemory, MemoryLayout,
    OwnedArray, SharedArrayView, SharedArrayViewMut, TypedBuffer, ZeroCopyBuffer, ZeroCopySlice,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::config::{
        DiagnosticsConfig, EcosystemConfig, EcosystemConfigBuilder, LogLevel, MemoryConfig,
        ModuleConfig, NumericConfig, ParallelConfig, Precision, PrecisionConfig,
    };
    pub use super::conversion::{
        ArrayConvert, ConversionError, ConversionOptions, ConversionResult, CrossModuleConvert,
        DataFlowConverter, LosslessConvert, LossyConvert, TypeAdapter,
    };
    pub use super::traits::{
        Composable, Configurable, CrossModuleOperator, DataConsumer, DataProvider, Diagnosable,
        Identifiable, ModuleCapability, ModuleInterface, ResourceAware, Serializable,
        VersionedInterface,
    };
    pub use super::zero_copy::{
        Alignment, ArrayBridge, BorrowedArray, BufferMut, BufferRef, ContiguousMemory,
        MemoryLayout, OwnedArray, SharedArrayView, SharedArrayViewMut, TypedBuffer, ZeroCopyBuffer,
        ZeroCopySlice,
    };
}

/// Integration error types
#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationError {
    /// Configuration error
    ConfigError(String),
    /// Type conversion error
    ConversionError(String),
    /// Memory layout incompatibility
    LayoutError(String),
    /// Module compatibility error
    CompatibilityError(String),
    /// Resource exhaustion
    ResourceError(String),
    /// Invalid operation
    InvalidOperation(String),
}

impl std::fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntegrationError::ConfigError(msg) => write!(f, "Configuration error: {msg}"),
            IntegrationError::ConversionError(msg) => write!(f, "Conversion error: {msg}"),
            IntegrationError::LayoutError(msg) => write!(f, "Layout error: {msg}"),
            IntegrationError::CompatibilityError(msg) => write!(f, "Compatibility error: {msg}"),
            IntegrationError::ResourceError(msg) => write!(f, "Resource error: {msg}"),
            IntegrationError::InvalidOperation(msg) => write!(f, "Invalid operation: {msg}"),
        }
    }
}

impl std::error::Error for IntegrationError {}

/// Result type for integration operations
pub type IntegrationResult<T> = Result<T, IntegrationError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_error_display() {
        let err = IntegrationError::ConfigError("test error".to_string());
        assert_eq!(err.to_string(), "Configuration error: test error");
    }
}
