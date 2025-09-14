//! Unified Error Handling System
//!
//! This module provides a comprehensive, unified interface to all error handling
//! capabilities in the scirs2-stats library, combining diagnostics, monitoring,
//! standardization, and recovery suggestions into a single cohesive system.

use crate::error::{StatsError, StatsResult};
use crate::error_diagnostics::{
    generate_global_health_report, get_global_statistics, global_monitor, record_global_error,
    ErrorMonitor, HealthReport,
};
use crate::error_handling_v2::{EnhancedError, ErrorBuilder, ErrorCode};
use crate::error_standardization::{ErrorMessages, StandardizedErrorReporter};
use scirs2_core::validation::{check_finite, check_positive};
use std::sync::Once;
use std::time::Instant;

/// Unified error handling facade providing comprehensive error management
pub struct UnifiedErrorHandler {
    #[allow(dead_code)]
    monitor: &'static ErrorMonitor,
    start_time: Instant,
}

impl UnifiedErrorHandler {
    /// Create a new unified error handler
    pub fn new() -> Self {
        Self {
            monitor: global_monitor(),
            start_time: Instant::now(),
        }
    }

    /// Create and record a comprehensive error with automatic diagnostics
    pub fn create_error(
        &self,
        code: ErrorCode,
        operation: impl Into<String>,
        message: impl Into<String>,
    ) -> EnhancedError {
        let operation_str = operation.into();

        // Record the error for monitoring
        record_global_error(code, &operation_str);

        // Create enhanced error with context and suggestions
        let base_error = StatsError::computation(message);
        let enhanced = ErrorBuilder::new(code, operation_str).build(base_error);

        enhanced
    }

    /// Create error with parameter validation and automatic suggestions
    pub fn create_validation_error(
        &self,
        code: ErrorCode,
        operation: impl Into<String>,
        parameter_name: &str,
        parameter_value: impl std::fmt::Display,
        validation_message: impl Into<String>,
    ) -> EnhancedError {
        let operation_str = operation.into();

        // Record the error for monitoring
        record_global_error(code, &operation_str);

        // Create enhanced error with parameter context
        let base_error = StatsError::invalid_argument(validation_message);
        let enhanced = ErrorBuilder::new(code, operation_str)
            .parameter(parameter_name, parameter_value)
            .build(base_error);

        enhanced
    }

    /// Validate array and create appropriate error if validation fails
    pub fn validate_array_or_error<T>(
        &self,
        data: &[T],
        name: &str,
        operation: &str,
    ) -> StatsResult<()>
    where
        T: PartialOrd + Copy,
    {
        // Check if slice is empty
        if data.is_empty() {
            let code = ErrorCode::E2004;
            record_global_error(code, operation);
            return Err(ErrorMessages::empty_array(name));
        }
        Ok(())
    }

    /// Validate finite array and create appropriate error if validation fails
    pub fn validate_finite_array_or_error(
        &self,
        data: &[f64],
        name: &str,
        operation: &str,
    ) -> StatsResult<()> {
        // Check if slice is empty
        if data.is_empty() {
            let code = ErrorCode::E2004;
            record_global_error(code, operation);
            return Err(ErrorMessages::empty_array(name));
        }

        // Check for finite values using core validation
        for &value in data {
            if let Err(_) = check_finite(value, name) {
                let code = if value.is_nan() {
                    ErrorCode::E3005
                } else if value.is_infinite() {
                    ErrorCode::E3006
                } else {
                    ErrorCode::E1001
                };
                record_global_error(code, operation);
                return Err(ErrorMessages::nan_detected(operation));
            }
        }
        Ok(())
    }

    /// Validate probability and create appropriate error if validation fails
    pub fn validate_probability_or_error(
        &self,
        value: f64,
        name: &str,
        operation: &str,
    ) -> StatsResult<()> {
        // Use scirs2-core validation
        if let Err(_) = scirs2_core::validation::check_probability(value, name) {
            let code = if value.is_nan() {
                ErrorCode::E3005
            } else if value < 0.0 || value > 1.0 {
                ErrorCode::E1003
            } else {
                ErrorCode::E1001
            };

            record_global_error(code, operation);
            return Err(ErrorMessages::invalid_probability(name, value));
        }
        Ok(())
    }

    /// Validate positive value and create appropriate error if validation fails
    pub fn validate_positive_or_error(
        &self,
        value: f64,
        name: &str,
        operation: &str,
    ) -> StatsResult<()> {
        // Use scirs2-core validation
        if let Err(_) = check_positive(value, name) {
            let code = if value.is_nan() {
                ErrorCode::E3005
            } else if value.is_infinite() {
                ErrorCode::E3006
            } else if value <= 0.0 {
                ErrorCode::E1002
            } else {
                ErrorCode::E1001
            };

            record_global_error(code, operation);
            return Err(ErrorMessages::non_positive_value(name, value));
        }
        Ok(())
    }

    /// Generate a comprehensive error report with diagnostics
    pub fn generate_comprehensive_report(
        &self,
        error: &StatsError,
        context: Option<&str>,
    ) -> String {
        let mut report = StandardizedErrorReporter::generate_report(error, context);

        // Add system health information
        let health_report = generate_global_health_report();
        if health_report.health_score < 80 {
            report.push_str("\n🚨 SYSTEM HEALTH ALERT:\n");
            report.push_str(&format!(
                "Overall health score: {}/100\n",
                health_report.health_score
            ));

            if !health_report.critical_issues.is_empty() {
                report.push_str("Critical issues detected:\n");
                for issue in &health_report.critical_issues {
                    report.push_str(&format!(
                        "  • {} (Severity: {})\n",
                        issue.title, issue.severity
                    ));
                }
            }
        }

        // Add monitoring statistics
        let stats = get_global_statistics();
        if stats.total_errors > 10 {
            report.push_str(&format!(
                "\n📊 Error Statistics: {} total errors, {:.4} errors/sec\n",
                stats.total_errors, stats.error_rate
            ));
        }

        report
    }

    /// Get current system health status
    pub fn get_health_status(&self) -> HealthReport {
        generate_global_health_report()
    }

    /// Check if system requires immediate attention
    pub fn requires_immediate_attention(&self) -> bool {
        let health_report = generate_global_health_report();
        health_report.requires_immediate_action()
    }

    /// Get uptime since unified error handler creation
    pub fn uptime(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Print health summary to console
    pub fn print_health_summary(&self) {
        let health_report = generate_global_health_report();
        println!("{}", health_report.to_formatted_string());
    }
}

impl Default for UnifiedErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Global unified error handler instance
static GLOBAL_HANDLER: Once = Once::new();
static mut GLOBAL_HANDLER_INSTANCE: Option<UnifiedErrorHandler> = None;

/// Get the global unified error handler instance
#[allow(dead_code)]
#[allow(static_mut_refs)]
pub fn global_error_handler() -> &'static UnifiedErrorHandler {
    unsafe {
        GLOBAL_HANDLER.call_once(|| {
            GLOBAL_HANDLER_INSTANCE = Some(UnifiedErrorHandler::new());
        });
        GLOBAL_HANDLER_INSTANCE.as_ref().unwrap()
    }
}

/// Convenience macro for creating standardized errors with automatic monitoring
#[macro_export]
macro_rules! stats_error_unified {
    ($code:expr, $op:expr, $msg:expr) => {
        $crate::unified_error_handling::global_error_handler().create_error($code, $op, $msg)
    };

    ($code:expr, $op:expr, $msg:expr, $param:expr => $value:expr) => {
        $crate::unified_error_handling::global_error_handler()
            .create_validation_error($code, $op, $param, $value, $msg)
    };
}

/// Convenience macro for validation with automatic error creation
#[macro_export]
macro_rules! validate_or_error {
    (array: $data:expr, $name:expr, $op:expr) => {
        $crate::unified_error_handling::global_error_handler()
            .validate_array_or_error($data, $name, $op)?
    };

    (finite: $data:expr, $name:expr, $op:expr) => {
        $crate::unified_error_handling::global_error_handler()
            .validate_finite_array_or_error($data, $name, $op)?
    };

    (probability: $value:expr, $name:expr, $op:expr) => {
        $crate::unified_error_handling::global_error_handler()
            .validate_probability_or_error($value, $name, $op)?
    };

    (positive: $value:expr, $name:expr, $op:expr) => {
        $crate::unified_error_handling::global_error_handler()
            .validate_positive_or_error($value, $name, $op)?
    };
}

/// Helper function to create standardized error messages
#[allow(dead_code)]
pub fn create_standardized_error(
    error_type: &str,
    parameter: &str,
    value: &str,
    operation: &str,
) -> StatsError {
    match error_type {
        "dimension_mismatch" => ErrorMessages::dimension_mismatch(parameter, value),
        "empty_array" => ErrorMessages::empty_array(parameter),
        "non_positive" => {
            ErrorMessages::non_positive_value(parameter, value.parse().unwrap_or(0.0))
        }
        "invalid_probability" => {
            ErrorMessages::invalid_probability(parameter, value.parse().unwrap_or(-1.0))
        }
        "nan_detected" => ErrorMessages::nan_detected(operation),
        "infinite_detected" => ErrorMessages::infinite_value_detected(operation),
        "convergence_failure" => {
            ErrorMessages::convergence_failure(operation, value.parse().unwrap_or(100))
        }
        _ => StatsError::invalid_argument(format!("Unknown error type: {}", error_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_handling_v2::ErrorCode;

    #[test]
    #[ignore = "timeout"]
    fn test_unified_error_handler() {
        let handler = UnifiedErrorHandler::new();

        // Test error creation
        let error = handler.create_error(ErrorCode::E3005, "test_operation", "Test error message");

        assert_eq!(error.code, ErrorCode::E3005);
        assert_eq!(error.context.operation, "test_operation");
    }

    #[test]
    fn test_validation_errors() {
        let handler = UnifiedErrorHandler::new();

        // Test empty array validation
        let empty_data: &[f64] = &[];
        let result = handler.validate_array_or_error(empty_data, "test_array", "test_op");
        assert!(result.is_err());

        // Test NaN validation
        let nandata = &[1.0, f64::NAN, 3.0];
        let result = handler.validate_finite_array_or_error(nandata, "test_array", "test_op");
        assert!(result.is_err());

        // Test invalid probability
        let result = handler.validate_probability_or_error(-0.5, "probability", "test_op");
        assert!(result.is_err());

        // Test non-positive value
        let result = handler.validate_positive_or_error(-1.0, "positive_param", "test_op");
        assert!(result.is_err());
    }

    #[test]
    fn test_global_handler() {
        let handler1 = global_error_handler();
        let handler2 = global_error_handler();

        // Should be the same instance
        assert_eq!(handler1 as *const _, handler2 as *const _);
    }

    #[test]
    fn test_macros() {
        // Test error creation macro
        let _error = stats_error_unified!(ErrorCode::E1001, "test_operation", "Test message");

        // Test validation macro would need valid arrays to test properly
        let validdata = &[1.0, 2.0, 3.0];
        let result: Result<(), StatsError> = (|| {
            validate_or_error!(finite: validdata, "testdata", "test_op");
            Ok(())
        })();
        assert!(result.is_ok());
    }
}
