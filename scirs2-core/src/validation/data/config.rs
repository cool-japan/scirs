//! Configuration types for data validation
//!
//! This module provides configuration structures and types for controlling
//! the behavior of the data validation system.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Data validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable strict mode (fail fast on first error)
    pub strict_mode: bool,
    /// Maximum validation depth for nested structures
    pub max_depth: usize,
    /// Enable validation caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Enable parallel validation for arrays
    pub enable_parallel_validation: bool,
    /// Custom validation rules
    pub custom_rules: HashMap<String, String>,
    /// Enable detailed error reporting
    pub detailederrors: bool,
    /// Performance mode (reduced checks for speed)
    pub performance_mode: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            max_depth: 100,
            enable_caching: true,
            cache_size_limit: 1000,
            enable_parallel_validation: false, // Can be expensive
            custom_rules: HashMap::new(),
            detailederrors: true,
            performance_mode: false,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Warning - data may still be usable
    Warning,
    /// Error - data should not be used
    Error,
    /// Critical - data is corrupted or dangerous
    Critical,
}

/// Types of data quality issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityIssueType {
    /// Missing or null values
    MissingData,
    /// Invalid numeric values (NaN, infinity)
    InvalidNumeric,
    /// Out-of-range values
    OutOfRange,
    /// Inconsistent format
    FormatInconsistency,
    /// Statistical outliers
    Outlier,
    /// Duplicate entries
    Duplicate,
    /// Type mismatch
    TypeMismatch,
    /// Constraint violation
    ConstraintViolation,
    /// Performance issue
    Performance,
}

/// Enhanced validation error type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationErrorType {
    /// Required field missing
    MissingRequiredField,
    /// Type mismatch
    TypeMismatch,
    /// Constraint violation
    ConstraintViolation,
    /// Invalid format
    InvalidFormat,
    /// Out of range value
    OutOfRange,
    /// Invalid array size
    InvalidArraySize,
    /// Duplicate values where unique required
    DuplicateValues,
    /// Data integrity failure
    IntegrityFailure,
    /// Custom validation rule failure
    CustomRuleFailure,
    /// Schema validation error
    SchemaError,
    /// Shape validation error
    ShapeError,
    /// Numeric quality error (NaN, infinity)
    InvalidNumeric,
    /// Statistical constraint violation
    StatisticalViolation,
    /// Performance issue
    Performance,
    /// Data integrity error
    IntegrityError,
    /// Type conversion error
    TypeConversion,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ValidationConfig::default();
        assert!(!config.strict_mode);
        assert_eq!(config.max_depth, 100);
        assert!(config.enable_caching);
        assert_eq!(config.cache_size_limit, 1000);
        assert!(!config.enable_parallel_validation);
        assert!(config.detailederrors);
        assert!(!config.performance_mode);
    }

    #[test]
    fn testerror_severity_ordering() {
        assert!(ErrorSeverity::Warning < ErrorSeverity::Error);
        assert!(ErrorSeverity::Error < ErrorSeverity::Critical);
    }

    #[test]
    fn test_quality_issue_types() {
        let issue_type = QualityIssueType::MissingData;
        assert_eq!(issue_type, QualityIssueType::MissingData);
    }

    #[test]
    fn test_validationerrortypes() {
        let errortype = ValidationErrorType::TypeMismatch;
        assert_eq!(errortype, ValidationErrorType::TypeMismatch);
    }
}
