//! Error types and validation results
//!
//! This module provides error types, validation results, and related structures
//! for reporting validation outcomes and issues.

use std::collections::HashMap;
use std::time::Duration;

use super::config::{ErrorSeverity, ValidationErrorType};
use crate::error::{CoreError, ErrorContext, ErrorLocation};

use serde::{Deserialize, Serialize};

/// Validation error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error type
    pub errortype: ValidationErrorType,
    /// Field path where error occurred
    pub fieldpath: String,
    /// Error message
    pub message: String,
    /// Expected value/type
    pub expected: Option<String>,
    /// Actual value found
    pub actual: Option<String>,
    /// Constraint that was violated
    pub constraint: Option<String>,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Additional context
    pub context: HashMap<String, String>,
}

impl ValidationError {
    /// Create a new validation error
    pub fn new(errortype: ValidationErrorType, fieldpath: &str, message: &str) -> Self {
        Self {
            errortype,
            fieldpath: fieldpath.to_string(),
            message: message.to_string(),
            expected: None,
            actual: None,
            constraint: None,
            severity: ErrorSeverity::Error,
            context: HashMap::new(),
        }
    }

    /// Set expected value
    pub fn with_expected(mut self, expected: &str) -> Self {
        self.expected = Some(expected.to_string());
        self
    }

    /// Set actual value
    pub fn with_actual(mut self, actual: &str) -> Self {
        self.actual = Some(actual.to_string());
        self
    }

    /// Set constraint
    pub fn with_constraint(mut self, constraint: &str) -> Self {
        self.constraint = Some(constraint.to_string());
        self
    }

    /// Set severity
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Add context information
    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }

    /// Get formatted error message
    pub fn formatted_message(&self) -> String {
        let mut message = format!("{}, {}", self.fieldpath, self.message);

        if let Some(expected) = &self.expected {
            message.push_str(&format!(" (expected: {expected})"));
        }

        if let Some(actual) = &self.actual {
            message.push_str(&format!(" (actual: {actual})"));
        }

        if let Some(constraint) = &self.constraint {
            message.push_str(&format!(" (constraint: {constraint})"));
        }

        message
    }
}

/// Convert ValidationError to CoreError
impl From<ValidationError> for CoreError {
    fn from(err: ValidationError) -> Self {
        // Choose the appropriate CoreError variant based on ValidationErrorType
        match err.errortype {
            ValidationErrorType::MissingRequiredField => CoreError::ValidationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::TypeMismatch => CoreError::TypeError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::ConstraintViolation => CoreError::ValueError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::OutOfRange => CoreError::DomainError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::InvalidFormat => CoreError::ValidationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::InvalidArraySize => CoreError::ValidationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::DuplicateValues => CoreError::ValidationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::IntegrityFailure => CoreError::ValidationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::CustomRuleFailure => CoreError::ValidationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::SchemaError => CoreError::ValidationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::ShapeError => CoreError::ShapeError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::InvalidNumeric => CoreError::ValueError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::StatisticalViolation => CoreError::ValidationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::Performance => CoreError::ComputationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::IntegrityError => CoreError::ValidationError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
            ValidationErrorType::TypeConversion => CoreError::TypeError(
                ErrorContext::new(err.formatted_message())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ),
        }
    }
}

/// Validation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationStats {
    /// Number of fields validated
    pub fields_validated: usize,
    /// Number of constraints checked
    pub constraints_checked: usize,
    /// Number of elements processed (for arrays)
    pub elements_processed: usize,
    /// Cache hit rate (if caching enabled)
    pub cache_hit_rate: f64,
    /// Memory usage during validation
    pub memory_used: Option<usize>,
}

impl ValidationStats {
    /// Create new validation statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Add field validation
    pub fn add_field_validation(&mut self) {
        self.fields_validated += 1;
    }

    /// Add constraint check
    pub fn add_constraint_check(&mut self) {
        self.constraints_checked += 1;
    }

    /// Add multiple constraint checks
    pub fn add_constraint_checks(&mut self, count: usize) {
        self.constraints_checked += count;
    }

    /// Add element processing
    pub fn add_elements_processed(&mut self, count: usize) {
        self.elements_processed += count;
    }

    /// Set cache hit rate
    pub fn set_cache_hit_rate(&mut self, rate: f64) {
        self.cache_hit_rate = rate;
    }

    /// Set memory usage
    pub fn set_memory_used(&mut self, bytes: usize) {
        self.memory_used = Some(bytes);
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationError>,
    /// Validation statistics
    pub stats: ValidationStats,
    /// Processing time
    pub duration: Duration,
}

impl ValidationResult {
    /// Create a new validation result
    pub fn new() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            stats: ValidationStats::new(),
            duration: Duration::from_secs(0),
        }
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.valid && self.errors.is_empty()
    }

    /// Check if there are warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get all errors
    pub fn errors(&self) -> &[ValidationError] {
        &self.errors
    }

    /// Get all warnings
    pub fn warnings(&self) -> &[ValidationError] {
        &self.warnings
    }

    /// Add an error
    pub fn adderror(&mut self, error: ValidationError) {
        self.valid = false;
        self.errors.push(error);
    }

    /// Add multiple errors
    pub fn adderrors(&mut self, mut errors: Vec<ValidationError>) {
        if !errors.is_empty() {
            self.valid = false;
            self.errors.append(&mut errors);
        }
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: ValidationError) {
        self.warnings.push(warning);
    }

    /// Add multiple warnings
    pub fn add_warnings(&mut self, mut warnings: Vec<ValidationError>) {
        self.warnings.append(&mut warnings);
    }

    /// Set processing duration
    pub fn set_duration(&mut self, duration: Duration) {
        self.duration = duration;
    }

    /// Get error count by severity
    pub fn error_count_by_severity(&self, severity: ErrorSeverity) -> usize {
        self.errors
            .iter()
            .filter(|e| e.severity == severity)
            .count()
    }

    /// Get warning count by severity
    pub fn warning_count_by_severity(&self, severity: ErrorSeverity) -> usize {
        self.warnings
            .iter()
            .filter(|w| w.severity == severity)
            .count()
    }

    /// Get errors by field path
    pub fn errors_for_field(&self, fieldpath: &str) -> Vec<&ValidationError> {
        self.errors
            .iter()
            .filter(|e| e.fieldpath == fieldpath)
            .collect()
    }

    /// Get warnings by field path
    pub fn warnings_for_field(&self, fieldpath: &str) -> Vec<&ValidationError> {
        self.warnings
            .iter()
            .filter(|w| w.fieldpath == fieldpath)
            .collect()
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        if self.is_valid() && !self.has_warnings() {
            "Validation passed successfully".to_string()
        } else if self.is_valid() && self.has_warnings() {
            format!("Validation passed with {} warning(s)", self.warnings.len())
        } else {
            format!(
                "Validation failed with {} error(s) and {} warning(s)",
                self.errors.len(),
                self.warnings.len()
            )
        }
    }

    /// Get detailed report
    pub fn detailed_report(&self) -> String {
        let mut report = self.summary();

        if !self.errors.is_empty() {
            report.push_str("\n\nErrors:");
            for (i, error) in self.errors.iter().enumerate() {
                report.push_str(&format!("{}. {}", i + 1, error.formatted_message()));
            }
        }

        if !self.warnings.is_empty() {
            report.push_str("\n\nWarnings:");
            for (i, warning) in self.warnings.iter().enumerate() {
                report.push_str(&format!("{}. {}", i + 1, warning.formatted_message()));
            }
        }

        report.push_str("\n\nStatistics:");
        report.push_str(&format!(
            "\n  Fields validated: {}",
            self.stats.fields_validated
        ));
        report.push_str(&format!(
            "\n  Constraints checked: {}",
            self.stats.constraints_checked
        ));
        report.push_str(&format!(
            "\n  Elements processed: {}",
            self.stats.elements_processed
        ));
        report.push_str(&format!("\n  Duration: {:?}", self.duration));

        if self.stats.cache_hit_rate > 0.0 {
            report.push_str(&format!(
                "\n  Cache hit rate: {:.2}%",
                self.stats.cache_hit_rate * 100.0
            ));
        }

        if let Some(memory) = self.stats.memory_used {
            report.push_str(&format!("\n  Memory used: {} bytes", memory));
        }

        report
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validationerror() {
        let error = ValidationError::new(
            ValidationErrorType::TypeMismatch,
            "test_field",
            "Type mismatch error",
        )
        .with_expected("string")
        .with_actual("integer")
        .with_constraint("type_check")
        .with_severity(ErrorSeverity::Error)
        .with_context("line", "42");

        assert_eq!(error.errortype, ValidationErrorType::TypeMismatch);
        assert_eq!(error.fieldpath, "test_field");
        assert_eq!(error.message, "Type mismatch error");
        assert_eq!(error.expected, Some("string".to_string()));
        assert_eq!(error.actual, Some("integer".to_string()));
        assert_eq!(error.constraint, Some("type_check".to_string()));
        assert_eq!(error.severity, ErrorSeverity::Error);
        assert_eq!(error.context.get("line"), Some(&"42".to_string()));

        let formatted = error.formatted_message();
        assert!(formatted.contains("test_field"));
        assert!(formatted.contains("Type mismatch error"));
        assert!(formatted.contains("expected: string"));
        assert!(formatted.contains("actual: integer"));
    }

    #[test]
    fn test_validation_stats() {
        let mut stats = ValidationStats::new();

        stats.add_field_validation();
        stats.add_constraint_checks(5);
        stats.add_elements_processed(100);
        stats.set_cache_hit_rate(0.85);
        stats.set_memory_used(1024);

        assert_eq!(stats.fields_validated, 1);
        assert_eq!(stats.constraints_checked, 5);
        assert_eq!(stats.elements_processed, 100);
        assert_eq!(stats.cache_hit_rate, 0.85);
        assert_eq!(stats.memory_used, Some(1024));
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();

        // Test initial state
        assert!(result.is_valid());
        assert!(!result.has_warnings());
        assert_eq!(result.errors().len(), 0);
        assert_eq!(result.warnings().len(), 0);

        // Add error
        let error =
            ValidationError::new(ValidationErrorType::TypeMismatch, "field1", "Error message");
        result.adderror(error);

        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 1);

        // Add warning
        let warning = ValidationError::new(
            ValidationErrorType::Performance,
            "field2",
            "Warning message",
        )
        .with_severity(ErrorSeverity::Warning);
        result.add_warning(warning);

        assert!(result.has_warnings());
        assert_eq!(result.warnings().len(), 1);

        // Test field filtering
        let field1errors = result.errors_for_field("field1");
        assert_eq!(field1errors.len(), 1);

        let field2_warnings = result.warnings_for_field("field2");
        assert_eq!(field2_warnings.len(), 1);

        // Test summary
        let summary = result.summary();
        assert!(summary.contains("failed"));
        assert!(summary.contains("1 error"));
        assert!(summary.contains("1 warning"));

        // Test detailed report
        let report = result.detailed_report();
        assert!(report.contains("Errors:"));
        assert!(report.contains("Warnings:"));
        assert!(report.contains("Statistics:"));
    }

    #[test]
    fn testerror_severity_counting() {
        let mut result = ValidationResult::new();

        let criticalerror = ValidationError::new(
            ValidationErrorType::IntegrityFailure,
            "field1",
            "Critical error",
        )
        .with_severity(ErrorSeverity::Critical);

        let warning = ValidationError::new(ValidationErrorType::Performance, "field2", "Warning")
            .with_severity(ErrorSeverity::Warning);

        result.adderror(criticalerror);
        result.add_warning(warning);

        assert_eq!(result.error_count_by_severity(ErrorSeverity::Critical), 1);
        assert_eq!(result.error_count_by_severity(ErrorSeverity::Error), 0);
        assert_eq!(result.warning_count_by_severity(ErrorSeverity::Warning), 1);
    }

    #[test]
    fn test_successful_validation_result() {
        let result = ValidationResult::new();

        assert!(result.is_valid());
        assert!(!result.has_warnings());

        let summary = result.summary();
        assert_eq!(summary, "Validation passed successfully");
    }
}
