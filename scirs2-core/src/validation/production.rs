//! # Production-Level Input Validation and Sanitization
//!
//! This module provides comprehensive validation for all public APIs with security hardening,
//! performance optimization, and robustness guarantees for production environments.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

#[cfg(feature = "regex")]
use regex;

/// Validation severity levels for different environments
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationLevel {
    /// Minimal validation - development only
    Development,
    /// Standard validation - testing environment
    Testing,
    /// Strict validation - staging environment
    Staging,
    /// Maximum validation - production environment
    Production,
}

/// Validation context with environment and performance tracking
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// Current validation level
    pub level: ValidationLevel,
    /// Maximum validation time allowed
    pub timeout: Duration,
    /// Whether to collect validation metrics
    pub collect_metrics: bool,
    /// Custom validation rules
    pub custom_rules: HashMap<String, String>,
}

impl Default for ValidationContext {
    fn default() -> Self {
        Self {
            level: ValidationLevel::Production,
            timeout: Duration::from_millis(100), // 100ms timeout for validation
            collect_metrics: true,
            custom_rules: HashMap::new(),
        }
    }
}

/// Validation result with detailed information
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors if any
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Performance metrics
    pub metrics: ValidationMetrics,
}

/// Detailed validation error information
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error code for programmatic handling
    pub code: String,
    /// Human-readable error message
    pub message: String,
    /// Field or parameter that failed validation
    pub field: Option<String>,
    /// Suggested fix or workaround
    pub suggestion: Option<String>,
    /// Severity level
    pub severity: ValidationSeverity,
}

/// Validation error severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    /// Information only
    Info,
    /// Warning - operation can continue
    Warning,
    /// Error - operation should not continue
    Error,
    /// Critical - system integrity at risk
    Critical,
}

/// Validation performance metrics
#[derive(Debug, Clone)]
pub struct ValidationMetrics {
    /// Time taken for validation
    pub duration: Duration,
    /// Number of rules checked
    pub rules_checked: usize,
    /// Number of values validated
    pub values_validated: usize,
    /// Memory used during validation
    pub memory_used: usize,
    /// Whether timeout was hit
    pub timed_out: bool,
}

impl Default for ValidationMetrics {
    fn default() -> Self {
        Self {
            duration: Duration::ZERO,
            rules_checked: 0,
            values_validated: 0,
            memory_used: 0,
            timed_out: false,
        }
    }
}

/// Production-level validator with comprehensive security and performance features
pub struct ProductionValidator {
    /// Validation context
    context: ValidationContext,
    /// Validation cache for performance
    cache: HashMap<String, ValidationResult>,
    /// Metrics collector
    metrics_collector: Option<Box<dyn ValidationMetricsCollector + Send + Sync>>,
}

/// Trait for collecting validation metrics
pub trait ValidationMetricsCollector {
    /// Record validation metrics
    fn record_validation(&mut self, result: &ValidationResult);

    /// Get aggregated metrics
    fn get_metrics(&self) -> ValidationSummary;

    /// Clear collected metrics
    fn clear(&mut self);
}

/// Summary of validation metrics
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total validations performed
    pub total_validations: usize,
    /// Total validation time
    pub total_duration: Duration,
    /// Average validation time
    pub average_duration: Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Most common error codes
    pub common_errors: HashMap<String, usize>,
}

impl ProductionValidator {
    /// Create a new production validator with default settings
    pub fn new() -> Self {
        Self {
            context: ValidationContext::default(),
            cache: HashMap::new(),
            metrics_collector: None,
        }
    }

    /// Create a validator with custom context
    pub fn with_context(context: ValidationContext) -> Self {
        Self {
            context,
            cache: HashMap::new(),
            metrics_collector: None,
        }
    }

    /// Set metrics collector
    pub fn with_metrics_collector(
        mut self,
        collector: Box<dyn ValidationMetricsCollector + Send + Sync>,
    ) -> Self {
        self.metrics_collector = Some(collector);
        self
    }

    /// Validate numeric value with comprehensive checks
    pub fn validate_numeric<T>(
        &mut self,
        value: T,
        constraints: &NumericConstraints<T>,
    ) -> ValidationResult
    where
        T: PartialOrd + Copy + fmt::Debug + fmt::Display,
    {
        let start_time = Instant::now();
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: ValidationMetrics::default(),
        };

        // Check timeout
        if start_time.elapsed() > self.context.timeout {
            result.metrics.timed_out = true;
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "VALIDATION_TIMEOUT".to_string(),
                message: "Validation timed out".to_string(),
                field: None,
                suggestion: Some("Increase validation timeout or simplify constraints".to_string()),
                severity: ValidationSeverity::Error,
            });
            return result;
        }

        let mut rules_checked = 0;

        // Range validation
        if let Some(min) = constraints.min {
            rules_checked += 1;
            if value < min {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "VALUE_TOO_SMALL".to_string(),
                    message: format!("Value {} is less than minimum {}", value, min),
                    field: constraints.field_name.clone(),
                    suggestion: Some(format!("Use a value >= {}", min)),
                    severity: ValidationSeverity::Error,
                });
            }
        }

        if let Some(max) = constraints.max {
            rules_checked += 1;
            if value > max {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "VALUE_TOO_LARGE".to_string(),
                    message: format!("Value {} is greater than maximum {}", value, max),
                    field: constraints.field_name.clone(),
                    suggestion: Some(format!("Use a value <= {}", max)),
                    severity: ValidationSeverity::Error,
                });
            }
        }

        // Custom validation rules
        for rule in &constraints.custom_rules {
            rules_checked += 1;
            if let Err(error) = rule.validate(value) {
                match constraints.allow_custom_rule_failures {
                    true => result.warnings.push(error),
                    false => {
                        result.is_valid = false;
                        result.errors.push(ValidationError {
                            code: "CUSTOM_RULE_FAILED".to_string(),
                            message: error,
                            field: constraints.field_name.clone(),
                            suggestion: Some("Check custom validation rules".to_string()),
                            severity: ValidationSeverity::Error,
                        });
                    }
                }
            }
        }

        // Performance warnings
        if rules_checked > 10 {
            result
                .warnings
                .push("High number of validation rules may impact performance".to_string());
        }

        result.metrics = ValidationMetrics {
            duration: start_time.elapsed(),
            rules_checked,
            values_validated: 1,
            memory_used: std::mem::size_of::<T>(),
            timed_out: false,
        };

        // Record metrics
        if let Some(collector) = &mut self.metrics_collector {
            collector.record_validation(&result);
        }

        result
    }

    /// Validate array/collection with size and content constraints
    pub fn validate_collection<T, I>(
        &mut self,
        collection: I,
        constraints: &CollectionConstraints<T>,
    ) -> ValidationResult
    where
        T: fmt::Debug,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let start_time = Instant::now();
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: ValidationMetrics::default(),
        };

        let iter = collection.into_iter();
        let size = iter.len();
        let mut rules_checked = 0;
        let mut values_validated = 0;

        // Size validation
        if let Some(min_size) = constraints.min_size {
            rules_checked += 1;
            if size < min_size {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "COLLECTION_TOO_SMALL".to_string(),
                    message: format!("Collection size {} is less than minimum {}", size, min_size),
                    field: constraints.field_name.clone(),
                    suggestion: Some(format!("Provide at least {} elements", min_size)),
                    severity: ValidationSeverity::Error,
                });
            }
        }

        if let Some(max_size) = constraints.max_size {
            rules_checked += 1;
            if size > max_size {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "COLLECTION_TOO_LARGE".to_string(),
                    message: format!("Collection size {} exceeds maximum {}", size, max_size),
                    field: constraints.field_name.clone(),
                    suggestion: Some(format!("Limit collection to {} elements", max_size)),
                    severity: ValidationSeverity::Error,
                });
            }
        }

        // Content validation (with timeout protection)
        if let Some(validator) = &constraints.element_validator {
            for (index, element) in iter.enumerate() {
                // Check timeout
                if start_time.elapsed() > self.context.timeout {
                    result.metrics.timed_out = true;
                    result.warnings.push(format!(
                        "Validation timed out after checking {} elements",
                        index
                    ));
                    break;
                }

                values_validated += 1;
                rules_checked += 1;

                if let Err(error) = validator.validate(&element) {
                    result.is_valid = false;
                    result.errors.push(ValidationError {
                        code: "ELEMENT_VALIDATION_FAILED".to_string(),
                        message: format!("Element at index {} failed validation: {}", index, error),
                        field: constraints.field_name.clone(),
                        suggestion: Some("Check element constraints".to_string()),
                        severity: ValidationSeverity::Error,
                    });
                }
            }
        }

        result.metrics = ValidationMetrics {
            duration: start_time.elapsed(),
            rules_checked,
            values_validated,
            memory_used: size * std::mem::size_of::<T>(),
            timed_out: start_time.elapsed() > self.context.timeout,
        };

        // Record metrics
        if let Some(collector) = &mut self.metrics_collector {
            collector.record_validation(&result);
        }

        result
    }

    /// Validate string with comprehensive security checks
    pub fn validate_string(
        &mut self,
        value: &str,
        constraints: &StringConstraints,
    ) -> ValidationResult {
        let start_time = Instant::now();
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: ValidationMetrics::default(),
        };

        let mut rules_checked = 0;

        // Length validation
        if let Some(min_length) = constraints.min_length {
            rules_checked += 1;
            if value.len() < min_length {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "STRING_TOO_SHORT".to_string(),
                    message: format!(
                        "String length {} is less than minimum {}",
                        value.len(),
                        min_length
                    ),
                    field: constraints.field_name.clone(),
                    suggestion: Some(format!("Provide at least {} characters", min_length)),
                    severity: ValidationSeverity::Error,
                });
            }
        }

        if let Some(max_length) = constraints.max_length {
            rules_checked += 1;
            if value.len() > max_length {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "STRING_TOO_LONG".to_string(),
                    message: format!(
                        "String length {} exceeds maximum {}",
                        value.len(),
                        max_length
                    ),
                    field: constraints.field_name.clone(),
                    suggestion: Some(format!("Limit string to {} characters", max_length)),
                    severity: ValidationSeverity::Error,
                });
            }
        }

        // Security validation
        if constraints.check_injection_attacks {
            rules_checked += 1;
            if self.contains_injection_patterns(value) {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "POTENTIAL_INJECTION_ATTACK".to_string(),
                    message: "String contains potential injection attack patterns".to_string(),
                    field: constraints.field_name.clone(),
                    suggestion: Some("Sanitize input or use parameterized queries".to_string()),
                    severity: ValidationSeverity::Critical,
                });
            }
        }

        // Character set validation
        if let Some(allowed_chars) = &constraints.allowed_characters {
            rules_checked += 1;
            for ch in value.chars() {
                if !allowed_chars.contains(&ch) {
                    result.is_valid = false;
                    result.errors.push(ValidationError {
                        code: "INVALID_CHARACTER".to_string(),
                        message: format!("Character '{}' is not allowed", ch),
                        field: constraints.field_name.clone(),
                        suggestion: Some("Use only allowed characters".to_string()),
                        severity: ValidationSeverity::Error,
                    });
                    break; // Only report first invalid character
                }
            }
        }

        // Pattern validation
        #[cfg(feature = "regex")]
        if let Some(pattern) = &constraints.pattern {
            rules_checked += 1;
            if !pattern.is_match(value) {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "PATTERN_MISMATCH".to_string(),
                    message: "String does not match required pattern".to_string(),
                    field: constraints.field_name.clone(),
                    suggestion: Some("Check string format requirements".to_string()),
                    severity: ValidationSeverity::Error,
                });
            }
        }

        result.metrics = ValidationMetrics {
            duration: start_time.elapsed(),
            rules_checked,
            values_validated: 1,
            memory_used: value.len(),
            timed_out: false,
        };

        // Record metrics
        if let Some(collector) = &mut self.metrics_collector {
            collector.record_validation(&result);
        }

        result
    }

    /// Check for common injection attack patterns
    fn contains_injection_patterns(&self, value: &str) -> bool {
        let dangerous_patterns = [
            // SQL injection patterns
            "'; DROP TABLE",
            "' OR '1'='1",
            "UNION SELECT",
            "'; --",
            // Script injection patterns
            "<script",
            "javascript:",
            "onload=",
            "onerror=",
            // Command injection patterns
            "; rm -rf",
            "| rm -rf",
            "&& rm -rf",
            "$(rm -rf)",
            // Path traversal patterns
            "../",
            "..\\",
            "%2e%2e%2f",
            "%2e%2e\\",
        ];

        let value_lower = value.to_lowercase();
        dangerous_patterns
            .iter()
            .any(|pattern| value_lower.contains(&pattern.to_lowercase()))
    }

    /// Clear validation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get validation statistics
    pub fn get_metrics(&self) -> Option<ValidationSummary> {
        self.metrics_collector
            .as_ref()
            .map(|collector| collector.get_metrics())
    }
}

impl Default for ProductionValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Constraints for numeric validation
#[derive(Debug)]
pub struct NumericConstraints<T> {
    /// Minimum allowed value
    pub min: Option<T>,
    /// Maximum allowed value
    pub max: Option<T>,
    /// Field name for error reporting
    pub field_name: Option<String>,
    /// Custom validation rules
    pub custom_rules: Vec<Box<dyn NumericRule<T>>>,
    /// Whether to allow custom rule failures as warnings
    pub allow_custom_rule_failures: bool,
}

impl<T: Clone> Clone for NumericConstraints<T> {
    fn clone(&self) -> Self {
        Self {
            min: self.min.clone(),
            max: self.max.clone(),
            field_name: self.field_name.clone(),
            custom_rules: Vec::new(), // Cannot clone trait objects, start with empty vec
            allow_custom_rule_failures: self.allow_custom_rule_failures,
        }
    }
}

/// Trait for custom numeric validation rules
pub trait NumericRule<T>: std::fmt::Debug {
    /// Validate the value
    fn validate(&self, value: T) -> Result<(), String>;
}

/// Constraints for collection validation
#[derive(Debug)]
pub struct CollectionConstraints<T> {
    /// Minimum collection size
    pub min_size: Option<usize>,
    /// Maximum collection size
    pub max_size: Option<usize>,
    /// Element validator
    pub element_validator: Option<Box<dyn ElementValidator<T>>>,
    /// Field name for error reporting
    pub field_name: Option<String>,
}

/// Trait for validating collection elements
pub trait ElementValidator<T>: std::fmt::Debug {
    /// Validate an element
    fn validate(&self, element: &T) -> Result<(), String>;
}

/// Constraints for string validation
#[derive(Debug)]
pub struct StringConstraints {
    /// Minimum string length
    pub min_length: Option<usize>,
    /// Maximum string length
    pub max_length: Option<usize>,
    /// Allowed character set
    pub allowed_characters: Option<std::collections::HashSet<char>>,
    /// Regular expression pattern
    #[cfg(feature = "regex")]
    pub pattern: Option<regex::Regex>,
    /// Whether to check for injection attacks
    pub check_injection_attacks: bool,
    /// Field name for error reporting
    pub field_name: Option<String>,
}

/// Default metrics collector implementation
pub struct DefaultMetricsCollector {
    /// Collected validation results
    results: Vec<ValidationResult>,
}

impl DefaultMetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }
}

impl ValidationMetricsCollector for DefaultMetricsCollector {
    fn record_validation(&mut self, result: &ValidationResult) {
        self.results.push(result.clone());
    }

    fn get_metrics(&self) -> ValidationSummary {
        let total_validations = self.results.len();

        if total_validations == 0 {
            return ValidationSummary {
                total_validations: 0,
                total_duration: Duration::ZERO,
                average_duration: Duration::ZERO,
                success_rate: 0.0,
                common_errors: HashMap::new(),
            };
        }

        let total_duration: Duration = self.results.iter().map(|r| r.metrics.duration).sum();

        let average_duration = total_duration / total_validations as u32;

        let successful_validations = self.results.iter().filter(|r| r.is_valid).count();

        let success_rate = successful_validations as f64 / total_validations as f64;

        let mut common_errors = HashMap::new();
        for result in &self.results {
            for error in &result.errors {
                *common_errors.entry(error.code.clone()).or_insert(0) += 1;
            }
        }

        ValidationSummary {
            total_validations,
            total_duration,
            average_duration,
            success_rate,
            common_errors,
        }
    }

    fn clear(&mut self) {
        self.results.clear();
    }
}

impl Default for DefaultMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common validation scenarios
/// Validate a positive number
pub fn validate_positive<T>(value: T, field_name: Option<String>) -> CoreResult<T>
where
    T: PartialOrd + Copy + fmt::Debug + fmt::Display + Default,
{
    let mut validator = ProductionValidator::new();
    let constraints = NumericConstraints {
        min: Some(T::default()), // Zero or equivalent
        max: None,
        field_name,
        custom_rules: Vec::new(),
        allow_custom_rule_failures: false,
    };

    let result = validator.validate_numeric(value, &constraints);
    if result.is_valid {
        Ok(value)
    } else {
        Err(CoreError::ValidationError(ErrorContext::new(format!(
            "Validation failed: {:?}",
            result.errors
        ))))
    }
}

/// Validate array dimensions
pub fn validate_array_dimensions(dims: &[usize], max_dimensions: usize) -> CoreResult<()> {
    if dims.is_empty() {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Array must have at least one dimension",
        )));
    }

    if dims.len() > max_dimensions {
        return Err(CoreError::ValidationError(ErrorContext::new(format!(
            "Array has {} dimensions, maximum allowed is {}",
            dims.len(),
            max_dimensions
        ))));
    }

    for (i, &dim) in dims.iter().enumerate() {
        if dim == 0 {
            return Err(CoreError::ValidationError(ErrorContext::new(format!(
                "Dimension {} has size 0, which is not allowed",
                i
            ))));
        }

        // Prevent integer overflow in total size calculation
        if dim > usize::MAX / 1024 {
            return Err(CoreError::ValidationError(ErrorContext::new(format!(
                "Dimension {} size {} is too large",
                i, dim
            ))));
        }
    }

    // Check total size doesn't overflow
    let total_size = dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            CoreError::ValidationError(ErrorContext::new("Array total size would overflow"))
        })
    })?;

    // Reasonable maximum total size (1GB of f64 values)
    const MAX_TOTAL_SIZE: usize = 1024 * 1024 * 1024 / 8;
    if total_size > MAX_TOTAL_SIZE {
        return Err(CoreError::ValidationError(ErrorContext::new(format!(
            "Array total size {} exceeds maximum allowed size {}",
            total_size, MAX_TOTAL_SIZE
        ))));
    }

    Ok(())
}

/// Validate file path for security
pub fn validate_file_path(path: &str) -> CoreResult<()> {
    // Check for path traversal attempts
    if path.contains("..") {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Path traversal detected in file path",
        )));
    }

    // Check for null bytes
    if path.contains('\0') {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "Null byte detected in file path",
        )));
    }

    // Check length
    if path.len() > 4096 {
        return Err(CoreError::ValidationError(ErrorContext::new(
            "File path too long",
        )));
    }

    // Check for dangerous patterns
    let dangerous_patterns = ["/dev/", "/proc/", "/sys/", "//", "\\\\"];
    for pattern in &dangerous_patterns {
        if path.contains(pattern) {
            return Err(CoreError::ValidationError(ErrorContext::new(format!(
                "Dangerous pattern '{}' detected in file path",
                pattern
            ))));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_numeric_validation() {
        let mut validator = ProductionValidator::new();
        let constraints = NumericConstraints {
            min: Some(0.0),
            max: Some(100.0),
            field_name: Some("test_value".to_string()),
            custom_rules: Vec::new(),
            allow_custom_rule_failures: false,
        };

        // Valid value
        let result = validator.validate_numeric(50.0, &constraints);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());

        // Too small
        let result = validator.validate_numeric(-10.0, &constraints);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.code == "VALUE_TOO_SMALL"));

        // Too large
        let result = validator.validate_numeric(150.0, &constraints);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.code == "VALUE_TOO_LARGE"));
    }

    #[test]
    fn test_string_validation() {
        let mut validator = ProductionValidator::new();
        let mut allowed_chars = HashSet::new();
        allowed_chars.extend("abcdefghijklmnopqrstuvwxyz0123456789".chars());

        let constraints = StringConstraints {
            min_length: Some(3),
            max_length: Some(20),
            allowed_characters: Some(allowed_chars),
            #[cfg(feature = "regex")]
            pattern: None,
            check_injection_attacks: true,
            field_name: Some("username".to_string()),
        };

        // Valid string
        let result = validator.validate_string("validuser123", &constraints);
        assert!(result.is_valid);

        // Too short
        let result = validator.validate_string("ab", &constraints);
        assert!(!result.is_valid);

        // Invalid character
        let result = validator.validate_string("user@name", &constraints);
        assert!(!result.is_valid);

        // Injection attack
        let result = validator.validate_string("'; DROP TABLE users; --", &constraints);
        assert!(!result.is_valid);
        assert!(result
            .errors
            .iter()
            .any(|e| e.code == "POTENTIAL_INJECTION_ATTACK"));
    }

    #[test]
    fn test_array_dimensions_validation() {
        // Valid dimensions
        assert!(validate_array_dimensions(&[10, 20, 30], 5).is_ok());

        // Empty dimensions
        assert!(validate_array_dimensions(&[], 5).is_err());

        // Zero dimension
        assert!(validate_array_dimensions(&[10, 0, 30], 5).is_err());

        // Too many dimensions
        assert!(validate_array_dimensions(&[1, 2, 3, 4, 5, 6], 5).is_err());
    }

    #[test]
    fn test_file_path_validation() {
        // Valid path
        assert!(validate_file_path("/home/user/data.txt").is_ok());

        // Path traversal
        assert!(validate_file_path("../../../etc/passwd").is_err());

        // Null byte
        assert!(validate_file_path("/home/user\0/data.txt").is_err());

        // Dangerous pattern
        assert!(validate_file_path("/dev/null").is_err());
    }

    #[test]
    fn test_metrics_collection() {
        let mut validator = ProductionValidator::new()
            .with_metrics_collector(Box::new(DefaultMetricsCollector::new()));

        let constraints = NumericConstraints {
            min: Some(0),
            max: Some(100),
            field_name: None,
            custom_rules: Vec::new(),
            allow_custom_rule_failures: false,
        };

        // Perform several validations
        validator.validate_numeric(50, &constraints);
        validator.validate_numeric(150, &constraints); // This will fail
        validator.validate_numeric(25, &constraints);

        let metrics = validator.get_metrics().unwrap();
        assert_eq!(metrics.total_validations, 3);
        assert_eq!(metrics.success_rate, 2.0 / 3.0);
        assert!(metrics.common_errors.contains_key("VALUE_TOO_LARGE"));
    }
}
