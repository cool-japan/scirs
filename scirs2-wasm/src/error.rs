//! Error types for WASM bindings
//!
//! This module provides structured, JS-friendly error types for all SciRS2-WASM
//! operations. Each error variant carries a numeric error code alongside a human-readable
//! message so JavaScript callers can branch on `err.code` without parsing strings.

use thiserror::Error;
use wasm_bindgen::prelude::*;

/// Numeric error codes surfaced to JavaScript.
///
/// These codes are stable across releases; add new values at the end only.
pub mod codes {
    /// Operation succeeded (never used in an error, provided for completeness).
    pub const OK: u32 = 0;
    /// Generic invalid input (e.g., NaN or out-of-range value).
    pub const INVALID_INPUT: u32 = 1001;
    /// Array/tensor dimension mismatch.
    pub const DIMENSION_MISMATCH: u32 = 1002;
    /// Numerical computation failed (singular matrix, divergence, etc.).
    pub const COMPUTATION_FAILED: u32 = 1003;
    /// Requested feature or algorithm is not yet implemented.
    pub const UNIMPLEMENTED: u32 = 1004;
    /// Array shape mismatch.
    pub const SHAPE_MISMATCH: u32 = 1005;
    /// Invalid array dimensions.
    pub const INVALID_DIMENSIONS: u32 = 1006;
    /// Array index is out of bounds.
    pub const INDEX_OUT_OF_BOUNDS: u32 = 1007;
    /// General invalid parameter.
    pub const INVALID_PARAMETER: u32 = 1008;
    /// Computation error (general).
    pub const COMPUTATION_ERROR: u32 = 1009;
    /// Serialization or deserialization error.
    pub const SERIALIZATION_ERROR: u32 = 1010;
    /// Error propagated from scirs2-core.
    pub const CORE_ERROR: u32 = 1011;
}

/// Error types for SciRS2-WASM operations.
///
/// Every variant maps to a stable numeric `code` (see [`codes`]) so JavaScript
/// callers can write `if (err.code === 1002) { /* dimension mismatch */ }`.
#[derive(Error, Debug)]
pub enum WasmError {
    /// Invalid input value with an associated error code and descriptive message.
    ///
    /// Use this when the caller passes `NaN`, `Infinity`, or a value outside
    /// the valid domain for the requested operation.
    #[error("[{code}] Invalid input: {message}")]
    InvalidInput {
        /// Numeric error code; typically [`codes::INVALID_INPUT`].
        code: u32,
        /// Human-readable description of what was invalid.
        message: String,
    },

    /// Dimension mismatch between two arrays or tensors.
    ///
    /// Carries the expected and actual dimension vectors for precise diagnostics.
    #[error("[1002] Dimension mismatch: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        /// Expected dimension sizes (e.g., `[3, 4]` for a 3×4 matrix).
        expected: Vec<usize>,
        /// Actual dimension sizes received from the caller.
        actual: Vec<usize>,
    },

    /// A numerical computation failed (e.g., singular matrix, non-convergence).
    ///
    /// `code` allows callers to distinguish sub-categories without string matching.
    #[error("[{code}] Computation failed: {details}")]
    ComputationFailed {
        /// Numeric sub-code; typically [`codes::COMPUTATION_FAILED`].
        code: u32,
        /// Detailed explanation of the failure.
        details: String,
    },

    /// A requested algorithm or feature is not yet implemented.
    #[error("[1004] Unimplemented: {feature}")]
    Unimplemented {
        /// Name of the unimplemented feature.
        feature: String,
    },

    /// Array shape mismatch
    #[error("[1005] Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },

    /// Invalid dimensions
    #[error("[1006] Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Index out of bounds
    #[error("[1007] Index out of bounds: {0}")]
    IndexOutOfBounds(String),

    /// Invalid parameter
    #[error("[1008] Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Computation error
    #[error("[1009] Computation error: {0}")]
    ComputationError(String),

    /// Serialization error
    #[error("[1010] Serialization error: {0}")]
    SerializationError(String),

    /// Core error from scirs2-core
    #[error("[1011] Core error: {0}")]
    CoreError(String),
}

impl WasmError {
    /// Return the numeric error code for this error variant.
    ///
    /// JavaScript callers can read this directly: `err.code`.
    pub fn error_code(&self) -> u32 {
        match self {
            WasmError::InvalidInput { code, .. } => *code,
            WasmError::DimensionMismatch { .. } => codes::DIMENSION_MISMATCH,
            WasmError::ComputationFailed { code, .. } => *code,
            WasmError::Unimplemented { .. } => codes::UNIMPLEMENTED,
            WasmError::ShapeMismatch { .. } => codes::SHAPE_MISMATCH,
            WasmError::InvalidDimensions(_) => codes::INVALID_DIMENSIONS,
            WasmError::IndexOutOfBounds(_) => codes::INDEX_OUT_OF_BOUNDS,
            WasmError::InvalidParameter(_) => codes::INVALID_PARAMETER,
            WasmError::ComputationError(_) => codes::COMPUTATION_ERROR,
            WasmError::SerializationError(_) => codes::SERIALIZATION_ERROR,
            WasmError::CoreError(_) => codes::CORE_ERROR,
        }
    }

    /// Serialize the error to a `JsValue` object with `code` and `message` fields.
    ///
    /// This is the preferred way to surface errors to JavaScript:
    /// ```javascript
    /// try { /* ... */ } catch (e) { console.log(e.code, e.message); }
    /// ```
    pub fn to_js_value(&self) -> JsValue {
        let code = self.error_code();
        let message = self.to_string();
        let obj = serde_json::json!({ "code": code, "message": message });
        // Serialize to a JSON string; if that fails, fall back to a plain string.
        match serde_json::to_string(&obj) {
            Ok(s) => JsValue::from_str(&s),
            Err(_) => JsValue::from_str(&message),
        }
    }

    /// Construct an [`InvalidInput`](WasmError::InvalidInput) error with the
    /// default code [`codes::INVALID_INPUT`].
    pub fn invalid_input(message: impl Into<String>) -> Self {
        WasmError::InvalidInput {
            code: codes::INVALID_INPUT,
            message: message.into(),
        }
    }

    /// Construct a [`ComputationFailed`](WasmError::ComputationFailed) error
    /// with the default code [`codes::COMPUTATION_FAILED`].
    pub fn computation_failed(details: impl Into<String>) -> Self {
        WasmError::ComputationFailed {
            code: codes::COMPUTATION_FAILED,
            details: details.into(),
        }
    }

    /// Construct an [`Unimplemented`](WasmError::Unimplemented) error.
    pub fn unimplemented(feature: impl Into<String>) -> Self {
        WasmError::Unimplemented {
            feature: feature.into(),
        }
    }
}

impl From<WasmError> for JsValue {
    fn from(err: WasmError) -> Self {
        // On wasm32 targets, create a proper JS string error.
        // On native targets (used for testing), JsValue::from_str() panics
        // because the JS runtime is unavailable. Use JsValue::NULL as a
        // safe placeholder so error-path tests can verify Result::is_err()
        // without triggering SIGABRT.
        #[cfg(target_arch = "wasm32")]
        {
            JsValue::from_str(&err.to_string())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = err;
            JsValue::NULL
        }
    }
}

impl From<scirs2_core::error::CoreError> for WasmError {
    fn from(err: scirs2_core::error::CoreError) -> Self {
        WasmError::CoreError(err.to_string())
    }
}

impl From<serde_json::Error> for WasmError {
    fn from(err: serde_json::Error) -> Self {
        WasmError::SerializationError(err.to_string())
    }
}

impl From<serde_wasm_bindgen::Error> for WasmError {
    fn from(err: serde_wasm_bindgen::Error) -> Self {
        WasmError::SerializationError(err.to_string())
    }
}

/// Result type for WASM operations
pub type WasmResult<T> = Result<T, WasmError>;
