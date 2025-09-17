//! Error types for the SciRS2 special functions module
//!
//! This module provides comprehensive error handling for special function computations,
//! with detailed error types that help users understand what went wrong and how to fix it.
//!
//! ## Error Categories
//!
//! The error types are organized into several categories:
//!
//! - **Mathematical Errors**: Domain errors, convergence failures, overflow
//! - **Input Validation**: Value errors, parameter validation failures  
//! - **Implementation Status**: Not implemented functionality
//! - **System Errors**: GPU unavailability, IO failures
//! - **Core Integration**: Errors from scirs2-core dependencies
//!
//! ## Usage Examples
//!
//! ```rust,ignore
//! use scirs2_special::{gamma, SpecialError, SpecialResult};
//!
//! // Function that can return different error types
//! fn safe_gamma(x: f64) -> SpecialResult<f64> {
//!     if x <= 0.0 && x.fract() == 0.0 {
//!         return Err(SpecialError::DomainError(
//!             format!("Gamma function undefined for non-positive integer: {x}")
//!         ));
//!     }
//!     
//!     let result = gamma(x);
//!     if !result.is_finite() {
//!         return Err(SpecialError::OverflowError(
//!             format!("Gamma({x}) resulted in overflow")
//!         ));
//!     }
//!     
//!     Ok(result)
//! }
//! ```

use scirs2_core::error::CoreError;
use thiserror::Error;

/// Error types for special function computations
///
/// This enum provides detailed error information for different failure modes
/// that can occur during special function computations. Each variant includes
/// contextual information to help users understand and resolve issues.
#[derive(Error, Debug)]
pub enum SpecialError {
    /// Generic computation error for unexpected failures
    ///
    /// Used when a computation fails for reasons not covered by more specific error types.
    /// Often indicates internal algorithmic issues or unexpected edge cases.
    ///
    /// # Examples
    /// - Numerical integration failure
    /// - Memory allocation failure during computation  
    /// - Internal algorithm assertion failure
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Mathematical domain error - input outside the valid mathematical domain
    ///
    /// Thrown when function arguments are outside the mathematical domain where
    /// the function is defined. This follows mathematical conventions strictly.
    ///
    /// # Examples
    /// - `gamma(-1)` - gamma function undefined at negative integers
    /// - `sqrt(-1)` for real-valued square root
    /// - `log(0)` or `log(negative)` for real logarithm
    /// - `asin(2)` - arcsine undefined outside [-1, 1]
    #[error("Domain error: {0}")]
    DomainError(String),

    /// Input validation error for invalid parameter values
    ///
    /// Used for parameter validation failures where inputs don't meet function
    /// requirements (but may be mathematically valid elsewhere).
    ///
    /// # Examples  
    /// - Negative array dimensions
    /// - Invalid tolerance parameters (e.g., negative tolerance)
    /// - Array shape mismatches
    /// - Invalid enumeration values
    #[error("Value error: {0}")]
    ValueError(String),

    /// Feature or function not yet implemented
    ///
    /// Indicates that a requested feature exists in the API but hasn't been
    /// implemented yet. This is used during development phases.
    ///
    /// # Examples
    /// - Complex number support for certain functions
    /// - Specific algorithm variants
    /// - Platform-specific optimizations
    #[error("Not implemented: {0}")]
    NotImplementedError(String),

    /// Iterative algorithm convergence failure
    ///
    /// Thrown when iterative algorithms fail to converge within specified
    /// tolerances or iteration limits. Often indicates numerical instability
    /// or inappropriate algorithm parameters.
    ///
    /// # Examples
    /// - Newton-Raphson method divergence
    /// - Series expansion truncation errors
    /// - Root-finding algorithm failure
    /// - Integration quadrature non-convergence
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Numerical overflow - result too large to represent
    ///
    /// Occurs when mathematical operations produce results that exceed the
    /// representable range of floating-point numbers (typically > 1.8e308 for f64).
    ///
    /// # Examples
    /// - `gamma(1000)` - extremely large gamma function values
    /// - `exp(1000)` - exponential overflow  
    /// - Factorial of large numbers
    /// - Product operations with many large terms
    #[error("Overflow error: {0}")]
    OverflowError(String),

    /// GPU acceleration unavailable, computation falls back to CPU
    ///
    /// Indicates that GPU computation was requested but is not available.
    /// This is often a non-fatal error where computation continues on CPU,
    /// but can be used to inform users about performance implications.
    ///
    /// # Examples
    /// - No GPU hardware available
    /// - GPU drivers not installed
    /// - CUDA/OpenCL runtime unavailable
    /// - GPU memory exhaustion
    #[error("GPU not available: {0}")]
    GpuNotAvailable(String),

    /// Error propagated from scirs2-core dependency
    ///
    /// Wraps errors from the core SciRS2 infrastructure. These typically
    /// relate to low-level operations like SIMD, parallel processing, or
    /// hardware acceleration.
    #[error("Core error: {0}")]
    CoreError(#[from] CoreError),
}

/// Convenient Result type alias for special function operations
///
/// This type alias simplifies function signatures throughout the special functions module.
/// All special function computations that can fail return this type, allowing for
/// consistent error handling patterns.
///
/// # Usage Examples
///
/// ```rust,ignore
/// use scirs2_special::{SpecialResult, SpecialError};
///
/// fn safe_computation(x: f64) -> SpecialResult<f64> {
///     if x < 0.0 {
///         Err(SpecialError::DomainError("x must be non-negative".to_string()))
///     } else {
///         Ok(x.sqrt())
///     }
/// }
///
/// // Pattern matching for error handling
/// match safe_computation(-1.0) {
///     Ok(result) => println!("Result: {}", result),
///     Err(SpecialError::DomainError(msg)) => eprintln!("Domain error: {msg}"),
///     Err(e) => eprintln!("Other error: {e}"),
/// }
/// ```
pub type SpecialResult<T> = Result<T, SpecialError>;

// Automatic error conversions for common error types

/// Convert from standard library float parsing errors
///
/// Automatically converts `ParseFloatError` into `SpecialError::ValueError`
/// for seamless error propagation when parsing numerical input.
impl From<std::num::ParseFloatError> for SpecialError {
    fn from(err: std::num::ParseFloatError) -> Self {
        SpecialError::ValueError(format!("Failed to parse float: {err}"))
    }
}

/// Convert from standard library IO errors  
///
/// Automatically converts `std::io::Error` into `SpecialError::ComputationError`
/// for handling file I/O failures during computation or data loading.
impl From<std::io::Error> for SpecialError {
    fn from(err: std::io::Error) -> Self {
        SpecialError::ComputationError(format!("IO error: {err}"))
    }
}
