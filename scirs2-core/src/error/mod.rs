//! Advanced error handling and recovery system for ``SciRS2``
//!
//! This module provides a comprehensive error handling framework including:
//! - Core error types and context management
//! - Advanced recovery strategies with retry mechanisms
//! - Circuit breaker patterns for fault tolerance
//! - Async error handling with timeout and progress tracking
//! - Comprehensive error diagnostics and pattern analysis
//! - Environment-aware error analysis and suggestions
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_core::error::{CoreError, CoreResult, RecoverableError};
//! use scirs2_core::error_context;
//!
//! fn example_operation() -> CoreResult<i32> {
//!     // Basic error with context
//!     if false {
//!         return Err(CoreError::ComputationError(
//!             error_context!("Operation failed")
//!         ));
//!     }
//!     
//!     Ok(42)
//! }
//!
//! // Enhanced error with recovery suggestions
//! let error = CoreError::MemoryError(error_context!("Out of memory"));
//! let recoverable = RecoverableError::new(error);
//! println!("{}", recoverable.recovery_report());
//! ```
//!
//! ## Advanced Features
//!
//! ### Retry Mechanisms
//!
//! ```rust
//! use scirs2_core::error::recovery::{RetryExecutor, RecoveryStrategy};
//! use std::time::Duration;
//!
//! let executor = RetryExecutor::new(RecoveryStrategy::ExponentialBackoff {
//!     max_attempts: 3,
//!     initial_delay: Duration::from_millis(100),
//!     max_delay: Duration::from_secs(5),
//!     multiplier: 2.0,
//! });
//!
//! let result = executor.execute(|| {
//!     // Your operation here
//!     Ok(42)
//! });
//! ```
//!
//! ### Circuit Breaker Pattern
//!
//! ```rust
//! use scirs2_core::error::recovery::CircuitBreaker;
//! use std::time::Duration;
//!
//! let circuit_breaker = CircuitBreaker::new(
//!     5, // failure threshold
//!     Duration::from_secs(30), // timeout
//!     Duration::from_secs(60), // recovery timeout
//! );
//!
//! let result = circuit_breaker.execute(|| {
//!     // Your operation here
//!     Ok(42)
//! });
//! ```
//!
//! ### Async Error Handling
//!
//! ```rust
//! use scirs2_core::error::recovery::{RecoveryStrategy, RetryExecutor};
//! use std::time::Duration;
//!
//! // Synchronous retry example
//! let executor = RetryExecutor::new(
//!     RecoveryStrategy::LinearBackoff {
//!         max_attempts: 3,
//!         delay: Duration::from_millis(100),
//!     }
//! );
//!
//! let mut counter = 0;
//! let retry_result = executor.execute(|| {
//!     counter += 1;
//!     if counter < 3 {
//!         Err(scirs2_core::error::CoreError::ComputationError(
//!             scirs2_core::error::ErrorContext::new("Temporary failure".to_string())
//!         ))
//!     } else {
//!         Ok::<i32, scirs2_core::error::CoreError>(123)
//!     }
//! });
//! assert!(retry_result.is_ok());
//! ```
//!
//! ### Error Diagnostics
//!
//! ```rust
//! use scirs2_core::error::diagnostics::diagnose_error;
//! use scirs2_core::error::{CoreError, ErrorContext, ErrorLocation};
//!
//! let error = CoreError::ConvergenceError(
//!     ErrorContext::new("Failed to converge")
//!         .with_location(ErrorLocation::new(file!(), line!()))
//! );
//! let diagnostics = diagnose_error(&error);
//!
//! println!("{}", diagnostics); // Comprehensive diagnostic report
//! ```

// Core error types and functionality
#[allow(clippy::module_inception)]
mod error;

// Recovery strategies and mechanisms
pub mod recovery;
pub use recovery::{
    hints, ErrorAggregator, ErrorSeverity, RecoverableError, RecoveryHint, RecoveryStrategy,
};

// Async error handling
#[cfg(feature = "async")]
pub mod async_handling;
#[cfg(feature = "async")]
pub use async_handling::{
    execute_with_error_aggregation, retry_with_exponential_backoff, with_timeout,
    AsyncCircuitBreaker, AsyncErrorAggregator, AsyncProgressTracker, AsyncRetryExecutor,
    TimeoutWrapper, TrackedAsyncOperation,
};

// Advanced error diagnostics
pub mod diagnostics;
pub use diagnostics::{
    diagnose_error, diagnose_error_with_context, EnvironmentInfo, ErrorDiagnosticReport,
    ErrorDiagnostics, ErrorOccurrence, ErrorPattern, PerformanceImpact,
};

// Circuit breaker and error recovery for production systems
pub mod circuit_breaker;
pub use circuit_breaker::{
    get_circuit_breaker, list_circuit_breakers, CircuitBreaker, CircuitBreakerConfig,
    CircuitBreakerStatus, CircuitState, FallbackStrategy, ResilientExecutor, RetryExecutor,
    RetryPolicy,
};

// Convenience re-exports for common patterns
pub use error::{
    chain_error, check_dimensions, check_domain, check_value, convert_error, validate, CoreError,
    CoreResult, ErrorContext, ErrorLocation,
};

/// Alpha 6 Enhanced Diagnostic Functions
///
/// Analyze an error with comprehensive diagnostics including Alpha 6 features
///
/// # Errors
///
/// This function does not return errors but analyzes the provided error.
#[must_use]
pub fn diagnose_error_advanced(
    error: &CoreError,
    context: Option<&str>,
    domain: Option<&str>,
) -> ErrorDiagnosticReport {
    let diagnostics = ErrorDiagnostics::global();
    let mut report = diagnostics.analyze_error(error);

    // Add predictive analysis if context is provided
    if let Some(ctx) = context {
        report.predictions = diagnostics.predict_potential_errors(ctx);
    }

    // Add domain-specific recovery strategies if domain is provided
    if let Some(dom) = domain {
        report.domain_strategies = diagnostics.suggest_domain_recovery(error, dom);
    }

    report
}

/// Record an error for pattern analysis
pub fn record_error_occurrence(error: &CoreError, context: String) {
    let diagnostics = ErrorDiagnostics::global();
    diagnostics.record_error(error, context);
}

/// Get predictive error analysis for a given context
#[must_use]
pub fn predict_errors_for_context(context: &str) -> Vec<String> {
    let diagnostics = ErrorDiagnostics::global();
    diagnostics.predict_potential_errors(context)
}

/// Get domain-specific recovery strategies for an error
#[must_use]
pub fn get_domain_recovery_strategies(error: &CoreError, domain: &str) -> Vec<String> {
    let diagnostics = ErrorDiagnostics::global();
    diagnostics.suggest_domain_recovery(error, domain)
}

pub mod prelude {
    //! Prelude module for convenient imports
    //! Commonly used error handling types and functions

    pub use super::{
        diagnose_error, CircuitBreaker, CoreError, CoreResult, EnvironmentInfo, ErrorAggregator,
        ErrorContext, ErrorDiagnosticReport, ErrorLocation, ErrorSeverity, RecoverableError,
        RecoveryStrategy, RetryExecutor,
    };

    #[cfg(feature = "async")]
    pub use super::{
        retry_with_exponential_backoff, with_timeout, AsyncCircuitBreaker, AsyncProgressTracker,
        AsyncRetryExecutor,
    };

    // Convenience macros
    pub use crate::{computation_error, dimension_error, domain_error, error_context, value_error};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_context;
    use std::time::Duration;

    #[test]
    fn test_error_integration() {
        let error = CoreError::DomainError(error_context!("Test error"));
        let recoverable = RecoverableError::new(error);

        assert!(!recoverable.retryable);
        assert_eq!(recoverable.severity, ErrorSeverity::Error);
        assert!(!recoverable.hints.is_empty());
    }

    #[test]
    fn test_retry_executor() {
        let policy = RetryPolicy {
            max_attempts: 2,
            base_delay: Duration::from_millis(1),
            ..Default::default()
        };
        let executor = RetryExecutor::new(policy);

        let attempts = std::cell::RefCell::new(0);
        let result = executor.execute(|| {
            let mut count = attempts.borrow_mut();
            *count += 1;
            if *count == 1 {
                Err(CoreError::ComputationError(error_context!(
                    "Temporary failure"
                )))
            } else {
                Ok(42)
            }
        });

        assert_eq!(result.unwrap(), 42);
        assert_eq!(*attempts.borrow(), 2);
    }

    #[test]
    fn test_error_diagnostics() {
        let error = CoreError::MemoryError(error_context!("Out of memory"));
        let report = diagnose_error(&error);

        assert!(matches!(report.error, CoreError::MemoryError(_)));
        assert_eq!(report.performance_impact, PerformanceImpact::High);
        assert!(!report.contextual_suggestions.is_empty());
    }

    #[test]
    fn test_circuit_breaker() {
        use crate::error::circuit_breaker::{CircuitBreakerConfig, CircuitState};
        use std::time::Duration;

        // Configure circuit breaker with low threshold for testing
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            failure_window: Duration::from_secs(60),
            recovery_timeout: Duration::from_secs(30),
            success_threshold: 1,
            max_half_open_requests: 1,
            minimum_request_threshold: 1,
        };
        let breaker = CircuitBreaker::with_config("test_breaker".to_string(), config);

        // First failure should work and trigger circuit opening
        let result: std::result::Result<(), _> =
            breaker.execute(|| Err(CoreError::ComputationError(error_context!("Test failure"))));
        assert!(result.is_err());

        // Second call should be blocked by circuit breaker
        let result = breaker.execute(|| Ok(42));
        assert!(result.is_err());

        let status = breaker.status().unwrap();
        assert_eq!(status.failure_count, 1);
        assert_eq!(status.state, CircuitState::Open);
    }

    #[test]
    fn test_error_aggregator() {
        let mut aggregator = ErrorAggregator::new();

        aggregator.add_simple_error(CoreError::ValueError(error_context!("Error 1")));
        aggregator.add_simple_error(CoreError::DomainError(error_context!("Error 2")));

        assert_eq!(aggregator.error_count(), 2);
        assert!(aggregator.has_errors());

        let summary = aggregator.summary();
        assert!(summary.contains("Collected 2 error(s)"));
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_error_handling() {
        use super::async_handling::*;

        // Test timeout
        let result = with_timeout(
            async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok::<i32, CoreError>(42)
            },
            Duration::from_millis(50),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CoreError::TimeoutError(_)));

        // Test async retry
        let executor = AsyncRetryExecutor::new(RecoveryStrategy::LinearBackoff {
            max_attempts: 2,
            delay: Duration::from_millis(1),
        });

        let mut attempts = 0;
        let result = executor
            .execute(|| {
                attempts += 1;
                async move {
                    if attempts == 1 {
                        Err(CoreError::ComputationError(error_context!("Async failure")))
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempts, 2);
    }
}
