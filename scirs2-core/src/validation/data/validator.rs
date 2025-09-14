//! Main validator implementation
//!
//! This module provides the core `Validator` struct that orchestrates all validation
//! operations and manages caching, custom rules, and configuration.
//!
//! ## Features
//!
//! - **Schema-based validation**: Validate data against predefined schemas
//! - **Custom validation rules**: Extend validation with custom business logic
//! - **Caching**: Improve performance with result caching
//! - **Array validation**: Specialized validation for scientific arrays
//! - **Quality analysis**: Generate comprehensive data quality reports
//!
//! ## Examples
//!
//! ### Basic validation
//!
//! ```rust
//! use scirs2_core::validation::data::{Validator, ValidationConfig, ValidationSchema, DataType};
//!
//! let config = ValidationConfig::default();
//! let validator = Validator::new(config)?;
//!
//! let schema = ValidationSchema::new()
//!     .name(user_schema)
//!     .require_field(name, DataType::String)
//!     .require_field("age", DataType::Integer);
//!
//! #
//! # {
//! let data = serde_json::json!({
//!     name: "John Doe",
//!     "age": 30
//! });
//!
//! let result = validator.validate(&data, &schema)?;
//! assert!(result.is_valid());
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Array validation
//!
//! ```rust
//! use scirs2_core::validation::data::{Validator, ValidationConfig, ArrayValidationConstraints};
//! use ndarray::Array2;
//!
//! let config = ValidationConfig::default();
//! let validator = Validator::new(config.clone())?;
//!
//! let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//!
//! let constraints = ArrayValidationConstraints::new()
//!     .withshape(vec![3, 2])
//!     .check_numeric_quality();
//!
//! let result = validator.validate_ndarray(&data, &constraints, &config)?;
//! assert!(result.is_valid());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{CoreError, ErrorContext};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use super::array_validation::ArrayValidator;
use super::config::{ErrorSeverity, ValidationConfig, ValidationErrorType};
use super::constraints::{ArrayValidationConstraints, Constraint};
use super::errors::{ValidationError, ValidationResult, ValidationStats};
use super::quality::{DataQualityReport, QualityAnalyzer};
use super::schema::{DataType, ValidationSchema};

// Core dependencies for array/matrix validation
use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt;

use serde_json::Value as JsonValue;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Cache entry for validation results
#[derive(Debug, Clone)]
struct CacheEntry {
    result: ValidationResult,
    timestamp: Instant,
    hit_count: usize,
}

/// Trait for custom validation rules
pub trait ValidationRule {
    /// Validate a value

    fn validate(&self, value: &JsonValue, fieldpath: &str) -> Result<(), String>;

    /// Get rule name
    fn name(&self) -> &str;

    /// Get rule description
    fn description(&self) -> &str;
}

/// Main data validator with comprehensive validation capabilities
pub struct Validator {
    /// Validation configuration
    config: ValidationConfig,
    /// Validation result cache
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Custom validation rules
    custom_rules: HashMap<String, Box<dyn ValidationRule + Send + Sync>>,
    /// Array validator
    array_validator: ArrayValidator,
    /// Quality analyzer
    quality_analyzer: QualityAnalyzer,
}

impl Validator {
    /// Create a new validator with configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Validation configuration settings
    ///
    /// # Returns
    ///
    /// A new `Validator` instance or an error if initialization fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::validation::data::{Validator, ValidationConfig};
    ///
    /// let config = ValidationConfig::default()
    ///     .with_max_depth(10)
    ///     .with_strict_mode(true);
    ///
    /// let validator = Validator::new(config)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(config: ValidationConfig) -> Result<Self, CoreError> {
        Ok(Self {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            custom_rules: HashMap::new(),
            array_validator: ArrayValidator::new(),
            quality_analyzer: QualityAnalyzer::new(),
        })
    }

    /// Validate JSON data against a schema
    ///
    /// This method performs comprehensive validation of JSON data against a predefined schema,
    /// including type checking, constraint validation, and custom rules.
    ///
    /// # Arguments
    ///
    /// * `data` - The JSON data to validate
    /// * `schema` - The validation schema to apply
    ///
    /// # Returns
    ///
    /// A `ValidationResult` containing the validation outcome and any errors/warnings
    ///
    /// # Example
    ///
    /// ```rust
    /// #
    /// # {
    /// use scirs2_core::validation::data::{Validator, ValidationSchema, DataType, Constraint, ValidationConfig};
    ///
    /// let validator = Validator::new(ValidationConfig::default())?;
    ///
    /// let schema = ValidationSchema::new()
    ///     .require_field("email", DataType::String)
    ///     .add_constraint("email", Constraint::Pattern("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$".to_string()));
    ///
    /// let data = serde_json::json!({
    ///     "email": "user@example.com"
    /// });
    ///
    /// let result = validator.validate(&data, &schema)?;
    /// assert!(result.is_valid());
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```

    pub fn validate(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
    ) -> Result<ValidationResult, CoreError> {
        let start_time = Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut stats = ValidationStats::default();

        // Check cache if enabled
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(data, schema)?;
            if let Some(mut cached_result) = self.get_cached_result(&cache_key)? {
                // Update cache hit rate
                let cache_hit_rate = self.calculate_cache_hit_rate()?;
                cached_result.stats.set_cache_hit_rate(cache_hit_rate);
                return Ok(cached_result);
            }
        }

        // Validate each field in the schema
        self.validate_fields(data, schema, &mut errors, &mut warnings, &mut stats, 0)?;

        // Apply global constraints
        self.validate_global_constraints(data, schema, &mut errors, &mut warnings, &mut stats)?;

        // Check for additional fields if not allowed
        if !schema.allow_additional_fields {
            self.check_additional_fields(data, schema, &mut errors, &mut warnings)?;
        }

        let valid = errors.is_empty()
            && !warnings
                .iter()
                .any(|w| w.severity == ErrorSeverity::Critical);
        let duration = start_time.elapsed();

        let mut result = ValidationResult {
            valid,
            errors,
            warnings,
            stats,
            duration,
        };

        // Cache result if enabled
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(data, schema)?;
            self.cache_result(&cache_key, result.clone())?;
        }

        // Update cache hit rate
        if self.config.enable_caching {
            let cache_hit_rate = self.calculate_cache_hit_rate()?;
            result.stats.set_cache_hit_rate(cache_hit_rate);
        }

        Ok(result)
    }

    /// Validate ndarray with comprehensive checks
    ///
    /// Performs validation on scientific arrays including shape validation, numeric quality
    /// checks, statistical constraints, and performance characteristics.
    ///
    /// # Arguments
    ///
    /// * `array` - The ndarray to validate
    /// * `constraints` - Validation constraints to apply
    /// * `config` - Validation configuration
    ///
    /// # Type Parameters
    ///
    /// * `S` - Storage type (must implement `Data`)
    /// * `D` - Dimension type
    /// * `S::Elem` - Element type (must be a floating-point type)
    ///
    /// # Returns
    ///
    /// A `ValidationResult` with detailed validation information
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::validation::data::{Validator, ValidationConfig, ArrayValidationConstraints};
    /// use ndarray::Array2;
    ///
    /// let validator = Validator::new(ValidationConfig::default())?;
    ///
    /// let data = Array2::from_shape_vec((3, 3), vec![
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0
    /// ])?;
    ///
    /// let constraints = ArrayValidationConstraints::new()
    ///     .withshape(vec![3, 3])
    ///     .check_numeric_quality();
    ///
    /// let result = validator.validate_ndarray(&data, &constraints, &ValidationConfig::default())?;
    /// assert!(result.is_valid());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn validate_ndarray<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        constraints: &ArrayValidationConstraints,
        config: &ValidationConfig,
    ) -> Result<ValidationResult, CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + Send + Sync + ScalarOperand + FromPrimitive,
    {
        self.array_validator
            .validate_ndarray(array, constraints, config)
    }

    /// Generate comprehensive data quality report
    ///
    /// Analyzes an array and generates a detailed quality report including completeness,
    /// accuracy, consistency, and statistical properties.
    ///
    /// # Arguments
    ///
    /// * `array` - The array to analyze
    /// * `fieldname` - Name of the field for reporting
    ///
    /// # Returns
    ///
    /// A `DataQualityReport` with quality metrics and recommendations
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::validation::data::{Validator, ValidationConfig};
    /// use ndarray::Array1;
    ///
    /// let validator = Validator::new(ValidationConfig::default())?;
    ///
    /// let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let report = validator.generate_quality_report(&data, "measurements")?;
    ///
    /// println!("Quality score: {}", report.quality_score);
    /// println!("Completeness: {}", report.metrics.completeness);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn generate_quality_report<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        fieldname: &str,
    ) -> Result<DataQualityReport, CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + ScalarOperand + Send + Sync + FromPrimitive,
    {
        self.quality_analyzer
            .generate_quality_report(array, fieldname)
    }

    /// Add a custom validation rule
    ///
    /// Registers a custom validation rule that can be referenced in schemas.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for the rule
    /// * `rule` - The validation rule implementation
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_core::validation::data::{Validator, ValidationConfig, ValidationRule};
    ///
    /// struct EmailRule;
    ///
    /// impl ValidationRule for EmailRule {
    ///     
    ///     fn path( &str) -> Result<(), String> {
    ///         if let Some(email) = value.as_str() {
    ///             if email.contains('@') {
    ///                 Ok(())
    ///             } else {
    ///                 Err(format!("{field_path}"))
    ///             }
    ///         } else {
    ///             Err(format!("{field_path}"))
    ///         }
    ///     }
    ///
    ///     fn name(&self) -> &str { "email" }
    ///     fn description(&self) -> &str { "Validates email format" }
    /// }
    ///
    /// let mut validator = Validator::new(ValidationConfig::default())?;
    /// validator.add_custom_rule(email.to_string(), Box::new(EmailRule));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_custom_rule(&mut self, name: String, rule: Box<dyn ValidationRule + Send + Sync>) {
        self.custom_rules.insert(name, rule);
    }

    /// Clear validation cache
    pub fn clear_cache(&self) -> Result<(), CoreError> {
        let mut cache = self.cache.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
                "Failed to acquire cache write lock".to_string(),
            ))
        })?;
        cache.clear();
        Ok(())
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> Result<(usize, f64), CoreError> {
        let cache = self.cache.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
                "Failed to acquire cache read lock".to_string(),
            ))
        })?;

        let size = cache.len();
        let hit_rate = self.calculate_cache_hit_rate()?;

        Ok((size, hit_rate))
    }

    /// Validate individual fields

    fn validate_fields(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
        stats: &mut ValidationStats,
        depth: usize,
    ) -> Result<(), CoreError> {
        if depth > self.config.max_depth {
            errors.push(ValidationError {
                errortype: ValidationErrorType::SchemaError,
                field_path: root.to_string(),
                message: "Maximum validation _depth exceeded".to_string(),
                expected: None,
                actual: None,
                constraint: None,
                severity: ErrorSeverity::Error,
                context: HashMap::new(),
            });
            return Ok(());
        }

        let data_obj = match data {
            JsonValue::Object(obj) => obj,
            _ => {
                errors.push(ValidationError {
                    errortype: ValidationErrorType::TypeMismatch,
                    field_path: root.to_string(),
                    message: "Expected object".to_string(),
                    expected: Some(object.to_string()),
                    actual: Some(self.get_value_type_name(data)),
                    constraint: None,
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
                return Ok(());
            }
        };

        for (fieldname, field_def) in &schema.fields {
            stats.add_field_validation();

            let field_path = if depth == 0 {
                fieldname.clone()
            } else {
                format!("{fieldname}")
            };

            if let Some(field_value) = data_obj.get(fieldname) {
                // Field exists, validate type and constraints
                self.validate_field_type(field_value, &field_def.datatype, &field_path, errors)?;
                self.validate_field_constraints(
                    field_value,
                    &field_def.constraints,
                    &field_path,
                    errors,
                    warnings,
                    stats,
                )?;

                // Validate custom rules
                for rule_name in &field_def.validation_rules {
                    if let Some(rule) = self.custom_rules.get(rule_name) {
                        if let Err(ruleerror) = rule.validate(field_value, &field_path) {
                            errors.push(ValidationError {
                                errortype: ValidationErrorType::CustomRuleFailure,
                                field_path: field_path.clone(),
                                message: ruleerror,
                                expected: None,
                                actual: None,
                                constraint: Some(rule_name.clone()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        }
                    }
                }
            } else if field_def.required {
                // Required field is missing
                errors.push(ValidationError {
                    errortype: ValidationErrorType::MissingRequiredField,
                    field_path,
                    message: format!("Required field '{}' is missing", fieldname),
                    expected: Some(format!("{:?}", field_def.datatype)),
                    actual: Some(missing.to_string()),
                    constraint: Some(required.to_string()),
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
            }
        }

        Ok(())
    }

    /// Validate field type

    fn validate_field_type(
        &self,
        value: &JsonValue,
        expected_type: &DataType,
        field_path: &str,
        errors: &mut Vec<ValidationError>,
    ) -> Result<(), CoreError> {
        let type_matches = match expected_type {
            DataType::Boolean => value.is_boolean(),
            DataType::Integer => value.is_i64(),
            DataType::Float32 | DataType::Float64 => value.is_f64() || value.is_i64(),
            DataType::String => value.is_string(),
            DataType::Array(_) => value.is_array(),
            DataType::Object => value.is_object(),
            DataType::Null => value.is_null(),
            _ => true, // Other types not yet implemented
        };

        if !type_matches {
            errors.push(ValidationError {
                errortype: ValidationErrorType::TypeMismatch,
                field_path: field_path.to_string(),
                message: format!(
                    "Type mismatch: expected {:?}, got {}",
                    expected_type,
                    self.get_value_type_name(value)
                ),
                expected: Some(format!("{expected_type:?}")),
                actual: Some(self.get_value_type_name(value)),
                constraint: Some("type".to_string()),
                severity: ErrorSeverity::Error,
                context: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Validate field constraints

    #[allow(clippy::only_used_in_recursion)]
    fn validate_field_constraints(
        &self,
        value: &JsonValue,
        constraints: &[Constraint],
        field_path: &str,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
        stats: &mut ValidationStats,
    ) -> Result<(), CoreError> {
        for constraint in constraints {
            stats.add_constraint_check();

            match constraint {
                Constraint::Range { min, max } => {
                    if let Some(num) = value.as_f64() {
                        if num < *min || num > *max {
                            errors.push(ValidationError {
                                errortype: ValidationErrorType::OutOfRange,
                                field_path: field_path.to_string(),
                                message: format!(
                                    "Value {} is out of range [{}, {}]",
                                    num, min, max
                                ),
                                expected: Some(format!("[{}, {}]", min, max)),
                                actual: Some(num.to_string()),
                                constraint: Some(range.to_string()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        }
                    }
                }
                Constraint::Length { min, max } => {
                    if let Some(s) = value.as_str() {
                        let len = s.len();
                        if len < *min || len > *max {
                            errors.push(ValidationError {
                                errortype: ValidationErrorType::ConstraintViolation,
                                field_path: field_path.to_string(),
                                message: format!(
                                    "String length {} is out of range [{}, {}]",
                                    len, min, max
                                ),
                                expected: Some(format!("length in [{}, {}]", min, max)),
                                actual: Some(len.to_string()),
                                constraint: Some(length.to_string()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        }
                    }
                }
                Constraint::NotNull => {
                    if value.is_null() {
                        errors.push(ValidationError {
                            errortype: ValidationErrorType::ConstraintViolation,
                            field_path: field_path.to_string(),
                            message: "Value cannot be null".to_string(),
                            expected: Some("non-null value".to_string()),
                            actual: Some(null.to_string()),
                            constraint: Some(not_null.to_string()),
                            severity: ErrorSeverity::Error,
                            context: HashMap::new(),
                        });
                    }
                }
                Constraint::Unique => {
                    if let Some(arr) = value.as_array() {
                        let mut seen = HashSet::new();
                        for item in arr {
                            let item_str = item.to_string();
                            if !seen.insert(item_str.clone()) {
                                errors.push(ValidationError {
                                    errortype: ValidationErrorType::DuplicateValues,
                                    field_path: field_path.to_string(),
                                    message: format!("{item_str}"),
                                    expected: Some("unique values".to_string()),
                                    actual: Some("duplicate found".to_string()),
                                    constraint: Some(unique.to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            }
                        }
                    }
                }
                Constraint::Pattern(pattern) => {
                    if let Some(s) = value.as_str() {
                        #[cfg(feature = "regex")]
                        {
                            if let Ok(re) = regex::Regex::new(pattern) {
                                if !re.is_match(s) {
                                    errors.push(ValidationError {
                                        errortype: ValidationErrorType::InvalidFormat,
                                        field_path: field_path.to_string(),
                                        message: format!(
                                            "Value '{}' does not match pattern '{}'",
                                            s, pattern
                                        ),
                                        expected: Some(format!("{pattern}")),
                                        actual: Some(s.to_string()),
                                        constraint: Some(pattern.to_string()),
                                        severity: ErrorSeverity::Error,
                                        context: HashMap::new(),
                                    });
                                }
                            }
                        }
                        #[cfg(not(feature = "regex"))]
                        {
                            warnings.push(ValidationError {
                                errortype: ValidationErrorType::SchemaError,
                                field_path: field_path.to_string(),
                                message: "Pattern validation requires 'regex' feature".to_string(),
                                expected: None,
                                actual: None,
                                constraint: Some(pattern.to_string()),
                                severity: ErrorSeverity::Warning,
                                context: HashMap::new(),
                            });
                        }
                    }
                }
                Constraint::AllowedValues(allowed) => {
                    let value_str = match value {
                        JsonValue::String(s) => s.clone(),
                        _ => value.to_string(),
                    };
                    if !allowed.contains(&value_str) {
                        errors.push(ValidationError {
                            errortype: ValidationErrorType::ConstraintViolation,
                            field_path: field_path.to_string(),
                            message: format!(
                                "Value '{}' is not in allowed values: {:?}",
                                value_str, allowed
                            ),
                            expected: Some(format!("{allowed:?}")),
                            actual: Some(value_str),
                            constraint: Some(allowed_values.to_string()),
                            severity: ErrorSeverity::Error,
                            context: HashMap::new(),
                        });
                    }
                }
                Constraint::Precision { decimal_places } => {
                    if let Some(num) = value.as_f64() {
                        let num_str = num.to_string();
                        if let Some(dot_pos) = num_str.find('.') {
                            let actual_precision = num_str.len() - dot_pos - 1;
                            if actual_precision > *decimal_places {
                                errors.push(ValidationError {
                                    errortype: ValidationErrorType::ConstraintViolation,
                                    field_path: field_path.to_string(),
                                    message: format!(
                                        "Value {} has {} decimal places, expected at most {}",
                                        num, actual_precision, decimal_places
                                    ),
                                    expected: Some(format!(
                                        "max {} decimal places",
                                        decimal_places
                                    )),
                                    actual: Some(format!("{} decimal places", actual_precision)),
                                    constraint: Some(precision.to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            }
                        }
                    }
                }
                Constraint::ArraySize { min, max } => {
                    if let Some(arr) = value.as_array() {
                        let size = arr.len();
                        if size < *min || size > *max {
                            errors.push(ValidationError {
                                errortype: ValidationErrorType::ConstraintViolation,
                                field_path: field_path.to_string(),
                                message: format!(
                                    "Array size {} is out of range [{}, {}]",
                                    size, min, max
                                ),
                                expected: Some(format!("size in [{}, {}]", min, max)),
                                actual: Some(size.to_string()),
                                constraint: Some(array_size.to_string()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        }
                    }
                }
                Constraint::ArrayElements(element_constraint) => {
                    if let Some(arr) = value.as_array() {
                        for (idx, element) in arr.iter().enumerate() {
                            let element_path = format!("{}[{}]", field_path, idx);
                            self.validate_field_constraints(
                                element,
                                &[(**element_constraint).clone()],
                                &element_path,
                                errors,
                                warnings,
                                stats,
                            )?;
                        }
                    }
                }
                Constraint::Custom(_rule_name) => {
                    // Custom constraint validation is handled separately in validate_fields
                    // This is just a placeholder for consistency
                }
                Constraint::Statistical(stats_constraints) => {
                    // Validate statistical properties of numeric arrays
                    if let Some(arr) = value.as_array() {
                        let mut numeric_values: Vec<f64> = Vec::new();

                        // Extract numeric values from array
                        for (idx, val) in arr.iter().enumerate() {
                            if let Some(num) = val.as_f64() {
                                numeric_values.push(num);
                            } else if let Some(num) = val.as_i64() {
                                numeric_values.push(num as f64);
                            } else {
                                errors.push(ValidationError {
                                    errortype: ValidationErrorType::TypeMismatch,
                                    field_path: format!("{}[{}]", field_path, idx),
                                    message: format!("{val}"),
                                    expected: Some(number.to_string()),
                                    actual: Some(val.to_string()),
                                    constraint: Some(statistical.to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                                continue;
                            }
                        }

                        if numeric_values.is_empty() {
                            errors.push(ValidationError {
                                errortype: ValidationErrorType::ConstraintViolation,
                                field_path: field_path.to_string(),
                                message: "Statistical validation requires numeric values"
                                    .to_string(),
                                expected: Some("numeric array".to_string()),
                                actual: Some("empty or non-numeric array".to_string()),
                                constraint: Some(statistical.to_string()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        } else {
                            // Calculate statistics
                            let count = numeric_values.len() as f64;
                            let mean = numeric_values.iter().sum::<f64>() / count;

                            // Calculate standard deviation
                            let variance = numeric_values
                                .iter()
                                .map(|x| (x - mean).powi(2))
                                .sum::<f64>()
                                / count;
                            let std_dev = variance.sqrt();

                            // Check mean constraints
                            if let Some(min_mean) = stats_constraints.min_mean {
                                if mean < min_mean {
                                    errors.push(ValidationError {
                                        errortype: ValidationErrorType::ConstraintViolation,
                                        field_path: field_path.to_string(),
                                        message: format!(
                                            "Mean {:.4} is less than minimum {:.4}",
                                            mean, min_mean
                                        ),
                                        expected: Some(format!(":.4{min_mean}")),
                                        actual: Some(format!(":.4{mean}")),
                                        constraint: Some("statistical.min_mean".to_string()),
                                        severity: ErrorSeverity::Error,
                                        context: HashMap::new(),
                                    });
                                }
                            }

                            if let Some(max_mean) = stats_constraints.max_mean {
                                if mean > max_mean {
                                    errors.push(ValidationError {
                                        errortype: ValidationErrorType::ConstraintViolation,
                                        field_path: field_path.to_string(),
                                        message: format!(
                                            "Mean {:.4} exceeds maximum {:.4}",
                                            mean, max_mean
                                        ),
                                        expected: Some(format!(":.4{max_mean}")),
                                        actual: Some(format!(":.4{mean}")),
                                        constraint: Some("statistical.max_mean".to_string()),
                                        severity: ErrorSeverity::Error,
                                        context: HashMap::new(),
                                    });
                                }
                            }

                            // Check standard deviation constraints
                            if let Some(min_std) = stats_constraints.min_std {
                                if std_dev < min_std {
                                    errors.push(ValidationError {
                                        errortype: ValidationErrorType::ConstraintViolation,
                                        field_path: field_path.to_string(),
                                        message: format!(
                                            "Standard deviation {:.4} is less than minimum {:.4}",
                                            std_dev, min_std
                                        ),
                                        expected: Some(format!(":.4{min_std}")),
                                        actual: Some(format!(":.4{std_dev}")),
                                        constraint: Some("statistical.min_std".to_string()),
                                        severity: ErrorSeverity::Error,
                                        context: HashMap::new(),
                                    });
                                }
                            }

                            if let Some(max_std) = stats_constraints.max_std {
                                if std_dev > max_std {
                                    errors.push(ValidationError {
                                        errortype: ValidationErrorType::ConstraintViolation,
                                        field_path: field_path.to_string(),
                                        message: format!(
                                            "Standard deviation {:.4} exceeds maximum {:.4}",
                                            std_dev, max_std
                                        ),
                                        expected: Some(format!(":.4{max_std}")),
                                        actual: Some(format!(":.4{std_dev}")),
                                        constraint: Some("statistical.max_std".to_string()),
                                        severity: ErrorSeverity::Error,
                                        context: HashMap::new(),
                                    });
                                }
                            }

                            // Check distribution (if specified)
                            if let Some(expected_dist) = &stats_constraints.expected_distribution {
                                // For now, just add a warning - full distribution testing would require more complex analysis
                                warnings.push(ValidationError {
                                    errortype: ValidationErrorType::SchemaError,
                                    field_path: field_path.to_string(),
                                    message: format!(
                                        "Distribution testing for '{}' not yet implemented",
                                        expected_dist
                                    ),
                                    expected: None,
                                    actual: None,
                                    constraint: Some("statistical.distribution".to_string()),
                                    severity: ErrorSeverity::Warning,
                                    context: HashMap::new(),
                                });
                            }
                        }
                    } else {
                        errors.push(ValidationError {
                            errortype: ValidationErrorType::TypeMismatch,
                            field_path: field_path.to_string(),
                            message: "Statistical constraints require an array of numeric values"
                                .to_string(),
                            expected: Some("numeric array".to_string()),
                            actual: Some(format!("{value}")),
                            constraint: Some(statistical.to_string()),
                            severity: ErrorSeverity::Error,
                            context: HashMap::new(),
                        });
                    }
                }
                Constraint::Temporal(time_constraints) => {
                    // Validate temporal data (array of timestamps)
                    if let Some(arr) = value.as_array() {
                        let mut timestamps: Vec<i64> = Vec::new();

                        // Extract timestamps from array
                        for (idx, val) in arr.iter().enumerate() {
                            if let Some(ts) = val.as_i64() {
                                timestamps.push(ts);
                            } else if let Some(ts) = val.as_f64() {
                                timestamps.push(ts as i64);
                            } else {
                                errors.push(ValidationError {
                                    errortype: ValidationErrorType::TypeMismatch,
                                    field_path: format!("{}[{}]", field_path, idx),
                                    message: format!("{val}"),
                                    expected: Some("timestamp (integer or float)".to_string()),
                                    actual: Some(val.to_string()),
                                    constraint: Some(temporal.to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                                continue;
                            }
                        }

                        if timestamps.len() < 2 {
                            errors.push(ValidationError {
                                errortype: ValidationErrorType::ConstraintViolation,
                                field_path: field_path.to_string(),
                                message: "Temporal validation requires at least 2 timestamps"
                                    .to_string(),
                                expected: Some("at least 2 timestamps".to_string()),
                                actual: Some(format!("{} timestamps", timestamps.len())),
                                constraint: Some(temporal.to_string()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        } else {
                            // Check for monotonic ordering if required
                            if time_constraints.require_monotonic {
                                let mut _is_monotonic = true;
                                for i in 1..timestamps.len() {
                                    if timestamps[0] < timestamps[0.saturating_sub(1)] {
                                        _is_monotonic = false;
                                        errors.push(ValidationError {
                                            errortype: ValidationErrorType::ConstraintViolation,
                                            field_path: field_path.to_string(),
                                            message: format!(
                                                "Timestamps not monotonic: {} comes after {}",
                                                timestamps[0],
                                                timestamps[0.saturating_sub(1)]
                                            ),
                                            expected: Some(
                                                "monotonic increasing timestamps".to_string(),
                                            ),
                                            actual: Some("non-monotonic timestamps".to_string()),
                                            constraint: Some("temporal.monotonic".to_string()),
                                            severity: ErrorSeverity::Error,
                                            context: HashMap::new(),
                                        });
                                        break;
                                    }
                                }
                            }

                            // Check for duplicates if not allowed
                            if !time_constraints.allow_duplicates {
                                let mut seen = std::collections::HashSet::new();
                                for &ts in &timestamps {
                                    if !seen.insert(ts) {
                                        errors.push(ValidationError {
                                            errortype: ValidationErrorType::ConstraintViolation,
                                            field_path: field_path.to_string(),
                                            message: format!("{ts}"),
                                            expected: Some("unique timestamps".to_string()),
                                            actual: Some("duplicate timestamps".to_string()),
                                            constraint: Some("temporal.unique".to_string()),
                                            severity: ErrorSeverity::Error,
                                            context: HashMap::new(),
                                        });
                                        break;
                                    }
                                }
                            }

                            // Check interval constraints
                            for i in 1..timestamps.len() {
                                let interval_ms =
                                    (timestamps[0] - timestamps[0.saturating_sub(1)]).abs();
                                let interval = std::time::Duration::from_millis(interval_ms as u64);

                                if let Some(min_interval) = &time_constraints.min_interval {
                                    if interval < *min_interval {
                                        errors.push(ValidationError {
                                            errortype: ValidationErrorType::ConstraintViolation,
                                            field_path: field_path.to_string(),
                                            message: format!(
                                                "Interval {:?} is less than minimum {:?}",
                                                interval, min_interval
                                            ),
                                            expected: Some(format!(
                                                "min interval {:?}",
                                                min_interval
                                            )),
                                            actual: Some(format!("{:?}", interval)),
                                            constraint: Some("temporal.min_interval".to_string()),
                                            severity: ErrorSeverity::Error,
                                            context: HashMap::new(),
                                        });
                                        break;
                                    }
                                }

                                if let Some(max_interval) = &time_constraints.max_interval {
                                    if interval > *max_interval {
                                        errors.push(ValidationError {
                                            errortype: ValidationErrorType::ConstraintViolation,
                                            field_path: field_path.to_string(),
                                            message: format!(
                                                "Interval {:?} exceeds maximum {:?}",
                                                interval, max_interval
                                            ),
                                            expected: Some(format!(
                                                "max interval {:?}",
                                                max_interval
                                            )),
                                            actual: Some(format!("{:?}", interval)),
                                            constraint: Some("temporal.max_interval".to_string()),
                                            severity: ErrorSeverity::Error,
                                            context: HashMap::new(),
                                        });
                                        break;
                                    }
                                }
                            }
                        }
                    } else {
                        errors.push(ValidationError {
                            errortype: ValidationErrorType::TypeMismatch,
                            field_path: field_path.to_string(),
                            message: "Temporal constraints require an array of timestamps"
                                .to_string(),
                            expected: Some("array of timestamps".to_string()),
                            actual: Some(format!("{value}")),
                            constraint: Some(temporal.to_string()),
                            severity: ErrorSeverity::Error,
                            context: HashMap::new(),
                        });
                    }
                }
                Constraint::Shape(shape_constraints) => {
                    // Validate array/matrix shape properties
                    if let Some(arr) = value.as_array() {
                        // For JSON arrays, we can only validate 1D arrays directly
                        // Multi-dimensional arrays would need to be nested arrays
                        let mut shape = vec![arr.len()];

                        // Check if it's a nested array (2D)
                        let mut is_2d = true;
                        let mut inner_sizes = Vec::new();
                        for elem in arr {
                            if let Some(inner_arr) = elem.as_array() {
                                inner_sizes.push(inner_arr.len());
                            } else {
                                is_2d = false;
                                break;
                            }
                        }

                        if is_2d && !inner_sizes.is_empty() {
                            // Check if all inner arrays have the same size
                            let first_size = inner_sizes[0];
                            if inner_sizes.iter().all(|&s| s == first_size) {
                                shape = vec![arr.len(), first_size];
                            } else {
                                errors.push(ValidationError {
                                    errortype: ValidationErrorType::ShapeError,
                                    field_path: field_path.to_string(),
                                    message: "Jagged arrays are not supported - all rows must have the same length".to_string(),
                                    expected: Some("rectangular array".to_string()),
                                    actual: Some("jagged array".to_string()),
                                    constraint: Some(shape.to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                                return Ok(());
                            }
                        }

                        // Validate dimensions
                        if !shape_constraints.dimensions.is_empty() {
                            let expected_dims = &shape_constraints.dimensions;
                            if shape.len() != expected_dims.len() {
                                errors.push(ValidationError {
                                    errortype: ValidationErrorType::ShapeError,
                                    field_path: field_path.to_string(),
                                    message: format!(
                                        "Array has {} dimensions, expected {}",
                                        shape.len(),
                                        expected_dims.len()
                                    ),
                                    expected: Some(format!("{} dimensions", expected_dims.len())),
                                    actual: Some(format!("{} dimensions", shape.len())),
                                    constraint: Some("shape.dimensions".to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            } else {
                                // Check each dimension
                                for (0, (actual_dim, expected_dim)) in
                                    shape.iter().zip(expected_dims.iter()).enumerate()
                                {
                                    if let Some(expected) = expected_dim {
                                        if actual_dim != expected {
                                            errors.push(ValidationError {
                                                errortype: ValidationErrorType::ShapeError,
                                                field_path: field_path.to_string(),
                                                message: format!(
                                                    "Dimension {} has size {}, expected {}",
                                                    0, actual_dim, expected
                                                ),
                                                expected: Some(format!(
                                                    "dimension {} = {}",
                                                    0, expected
                                                )),
                                                actual: Some(format!(
                                                    "dimension {} = {}",
                                                    0, actual_dim
                                                )),
                                                constraint: Some(format!("shape.dimension[{}]", i)),
                                                severity: ErrorSeverity::Error,
                                                context: HashMap::new(),
                                            });
                                        }
                                    }
                                }
                            }
                        }

                        // Check total element count
                        let total_elements: usize = shape.iter().product();

                        if let Some(min_elements) = shape_constraints.min_elements {
                            if total_elements < min_elements {
                                errors.push(ValidationError {
                                    errortype: ValidationErrorType::ShapeError,
                                    field_path: field_path.to_string(),
                                    message: format!(
                                        "Array has {} elements, minimum required is {}",
                                        total_elements, min_elements
                                    ),
                                    expected: Some(format!(">= {} elements", min_elements)),
                                    actual: Some(format!("{} elements", total_elements)),
                                    constraint: Some("shape.min_elements".to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            }
                        }

                        if let Some(max_elements) = shape_constraints.max_elements {
                            if total_elements > max_elements {
                                errors.push(ValidationError {
                                    errortype: ValidationErrorType::ShapeError,
                                    field_path: field_path.to_string(),
                                    message: format!(
                                        "Array has {} elements, maximum allowed is {}",
                                        total_elements, max_elements
                                    ),
                                    expected: Some(format!("<= {} elements", max_elements)),
                                    actual: Some(format!("{} elements", total_elements)),
                                    constraint: Some("shape.max_elements".to_string()),
                                    severity: ErrorSeverity::Error,
                                    context: HashMap::new(),
                                });
                            }
                        }

                        // Check if square matrix is required (only for 2D arrays)
                        if shape_constraints.require_square
                            && shape.len() == 2
                            && shape[0] != shape[1]
                        {
                            errors.push(ValidationError {
                                errortype: ValidationErrorType::ShapeError,
                                field_path: field_path.to_string(),
                                message: format!(
                                    "Matrix must be square, but has shape {}x{}",
                                    shape[0], shape[1]
                                ),
                                expected: Some("square matrix".to_string()),
                                actual: Some(format!("{}x{} matrix", shape[0], shape[1])),
                                constraint: Some("shape.square".to_string()),
                                severity: ErrorSeverity::Error,
                                context: HashMap::new(),
                            });
                        }
                    } else {
                        errors.push(ValidationError {
                            errortype: ValidationErrorType::TypeMismatch,
                            field_path: field_path.to_string(),
                            message: "Shape constraints require an array".to_string(),
                            expected: Some(array.to_string()),
                            actual: Some(format!("{value}")),
                            constraint: Some(shape.to_string()),
                            severity: ErrorSeverity::Error,
                            context: HashMap::new(),
                        });
                    }
                }
                Constraint::And(constraints) => {
                    // All constraints must pass
                    for constraint in constraints {
                        self.validate_field_constraints(
                            value,
                            &[constraint.clone()],
                            field_path,
                            errors_warnings,
                            stats,
                        )?;
                    }
                }
                Constraint::Or(constraints) => {
                    // At least one constraint must pass
                    let mut temperrors = Vec::new();
                    let mut any_passed = false;

                    for constraint in constraints {
                        let mut constrainterrors = Vec::new();
                        self.validate_field_constraints(
                            value,
                            &[constraint.clone()],
                            field_path,
                            &mut constrainterrors_warnings,
                            stats,
                        )?;

                        if constrainterrors.is_empty() {
                            any_passed = true;
                            break;
                        } else {
                            temperrors.extend(constrainterrors);
                        }
                    }

                    if !any_passed {
                        errors.push(ValidationError {
                            errortype: ValidationErrorType::ConstraintViolation,
                            field_path: field_path.to_string(),
                            message: format!(
                                "None of the OR constraints passed: {} errors",
                                temperrors.len()
                            ),
                            expected: Some("at least one constraint to pass".to_string()),
                            actual: Some("all constraints failed".to_string()),
                            constraint: Some(or.to_string()),
                            severity: ErrorSeverity::Error,
                            context: HashMap::new(),
                        });
                    }
                }
                Constraint::Not(constraint) => {
                    // Constraint must not pass
                    let mut temperrors = Vec::new();
                    self.validate_field_constraints(
                        value,
                        &[*constraint.clone()],
                        field_path,
                        &mut temperrors_warnings,
                        stats,
                    )?;

                    if temperrors.is_empty() {
                        errors.push(ValidationError {
                            errortype: ValidationErrorType::ConstraintViolation,
                            field_path: field_path.to_string(),
                            message: "NOT constraint failed: inner constraint passed".to_string(),
                            expected: Some("constraint to fail".to_string()),
                            actual: Some("constraint passed".to_string()),
                            constraint: Some(not.to_string()),
                            severity: ErrorSeverity::Error,
                            context: HashMap::new(),
                        });
                    }
                }
                Constraint::If {
                    condition,
                    then_constraint,
                    else_constraint,
                } => {
                    // Conditional constraint
                    let mut conditionerrors = Vec::new();
                    self.validate_field_constraints(
                        value,
                        &[*condition.clone()],
                        field_path,
                        &mut conditionerrors_warnings,
                        stats,
                    )?;

                    if conditionerrors.is_empty() {
                        // Condition passed, apply then_constraint
                        self.validate_field_constraints(
                            value,
                            &[*then_constraint.clone()],
                            field_path,
                            errors_warnings,
                            stats,
                        )?;
                    } else if let Some(else_constraint) = else_constraint {
                        // Condition failed, apply else_constraint
                        self.validate_field_constraints(
                            value,
                            &[*else_constraint.clone()],
                            field_path,
                            errors_warnings,
                            stats,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Validate global constraints

    #[allow(clippy::ptr_arg)]
    fn validate_global_constraints(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
        stats: &mut ValidationStats,
    ) -> Result<(), CoreError> {
        // Global constraints would be implemented here
        Ok(())
    }

    /// Check for additional fields

    #[allow(clippy::ptr_arg)]
    fn check_additional_fields(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
    ) -> Result<(), CoreError> {
        if let JsonValue::Object(obj) = data {
            for key in obj.keys() {
                if !schema.fields.contains_key(key) {
                    errors.push(ValidationError {
                        errortype: ValidationErrorType::SchemaError,
                        field_path: key.clone(),
                        message: format!("Additional field '{}' not allowed", key),
                        expected: None,
                        actual: Some(key.clone()),
                        constraint: None,
                        severity: ErrorSeverity::Warning,
                        context: HashMap::new(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Get the type name for a JSON value

    fn get_value_type_name(&self, value: &JsonValue) -> String {
        match value {
            JsonValue::Null => "null".to_string(),
            JsonValue::Bool(_) => "boolean".to_string(),
            JsonValue::Number(n) => {
                if n.is_i64() {
                    "integer".to_string()
                } else {
                    "number".to_string()
                }
            }
            JsonValue::String(_) => "string".to_string(),
            JsonValue::Array(_) => "array".to_string(),
            JsonValue::Object(_) => "object".to_string(),
        }
    }

    /// Generate cache key for validation result

    fn generate_cache_key(
        &self,
        data: &JsonValue,
        schema: &ValidationSchema,
    ) -> Result<String, CoreError> {
        let mut hasher = DefaultHasher::new();
        data.to_string().hash(&mut hasher);
        schema.name.hash(&mut hasher);
        schema.version.hash(&mut hasher);

        Ok(format!("{:x}", hasher.finish()))
    }

    /// Get cached validation result
    fn get_cached_result(&self, cachekey: &str) -> Result<Option<ValidationResult>, CoreError> {
        let cache = self.cache.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
                "Failed to acquire cache read lock".to_string(),
            ))
        })?;

        if let Some(entry) = cache.get(cache_key) {
            // Check if cache entry is still valid (for now, always valid)
            return Ok(Some(entry.result.clone()));
        }

        Ok(None)
    }

    /// Cache validation result
    fn cache_result(&self, cachekey: &str, result: ValidationResult) -> Result<(), CoreError> {
        let mut cache = self.cache.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
                "Failed to acquire cache write lock".to_string(),
            ))
        })?;

        // Remove oldest entries if cache is full
        if cache.len() >= self.config.cache_size_limit {
            if let Some((oldest_key, _)) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.timestamp)
                .map(|(k, v)| (k.clone(), v.clone()))
            {
                cache.remove(&oldest_key);
            }
        }

        let entry = CacheEntry {
            result,
            timestamp: Instant::now(),
            hit_count: 0,
        };

        cache.insert(cache_key.to_string(), entry);
        Ok(())
    }

    /// Calculate cache hit rate
    fn calculate_cache_hit_rate(&self) -> Result<f64, CoreError> {
        let cache = self.cache.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new(
                "Failed to acquire cache read lock".to_string(),
            ))
        })?;

        if cache.is_empty() {
            return Ok(0.0);
        }

        let total_hits: usize = cache.values().map(|entry| entry.hit_count).sum();
        let total_entries = cache.len();

        Ok(total_hits as f64 / total_entries as f64)
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new(ValidationConfig::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_validator_creation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        // Test basic properties
        assert!(!validator.config.strict_mode);
        assert_eq!(validator.config.max_depth, 100);
    }

    #[test]
    fn test_array_validation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config.clone()).unwrap();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let constraints = ArrayValidationConstraints::new()
            .withshape(vec![5])
            .with_fieldname("test_array")
            .check_numeric_quality();

        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_quality_report_generation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let report = validator
            .generate_quality_report(&array, "test_field")
            .unwrap();

        assert!(report.quality_score > 0.9); // Should be high quality
        assert_eq!(report.metrics.completeness, 1.0); // No missing values
    }

    #[test]
    fn test_cache_management() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        // Test cache clearing
        assert!(validator.clear_cache().is_ok());

        // Test cache stats
        let (size, hit_rate) = validator.get_cache_stats().unwrap();
        assert_eq!(size, 0); // Should be empty after clearing
        assert_eq!(hit_rate, 0.0); // No hits yet
    }

    #[test]
    fn test_json_validation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        let schema = ValidationSchema::new()
            .name("test_schema")
            .require_field("name", DataType::String)
            .require_field("age", DataType::Integer);

        let valid_data = serde_json::json!({
            "name": "John Doe",
            "age": 30
        });

        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({
            "name": "John Doe"
            // Missing required "age" field
        });

        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 1);
    }

    #[test]
    fn test_allowed_values_constraint() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        let schema = ValidationSchema::new()
            .name("test_schema")
            .optional_field("status", DataType::String)
            .add_constraint(
                "status",
                Constraint::AllowedValues(vec![
                    "active".to_string(),
                    "inactive".to_string(),
                    "pending".to_string(),
                ]),
            );

        let valid_data = serde_json::json!({
            "status": "active"
        });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({
            "status": "deleted"
        });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_precision_constraint() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        let schema = ValidationSchema::new()
            .name("test_schema")
            .optional_field("price", DataType::Float64)
            .add_constraint("price", Constraint::Precision { decimalplaces: 2 });

        let valid_data = serde_json::json!({
            "price": 19.99
        });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({
            "price": 19.999
        });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_array_size_constraint() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        let schema = ValidationSchema::new()
            .name("test_schema")
            .optional_field("tags", DataType::Array(Box::new(DataType::String)))
            .add_constraint("tags", Constraint::ArraySize { min: 1, max: 5 });

        let valid_data = serde_json::json!({
            "tags": ["rust", "programming", "science"]
        });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({
            "tags": ["too", "many", "tags", "here", "six", "seven"]
        });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_array_elements_constraint() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        let schema = ValidationSchema::new()
            .name("test_schema")
            .optional_field("scores", DataType::Array(Box::new(DataType::Float64)))
            .add_constraint(
                "scores",
                Constraint::ArrayElements(Box::new(Constraint::Range {
                    min: 0.0,
                    max: 100.0,
                })),
            );

        let valid_data = serde_json::json!({
            "scores": [85.5, 92.0, 78.3, 95.0]
        });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({
            "scores": [85.5, 92.0, 105.0, 95.0]
        });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());
    }

    #[test]

    fn test_composite_constraint_validation() {
        let validator = Validator::new(ValidationConfig::default()).unwrap();

        // Test AND constraint
        let schema = ValidationSchema::new()
            .require_field("age", DataType::Float64)
            .add_constraint(
                "age",
                Constraint::And(vec![
                    Constraint::Range {
                        min: 0.0,
                        max: 150.0,
                    },
                    Constraint::NotNull,
                ]),
            );

        let valid_data = serde_json::json!({
            "age": 25.0
        });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({
            "age": -5.0
        });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());

        // Test OR constraint
        let schema = ValidationSchema::new()
            .require_field("status", DataType::String)
            .add_constraint(
                "status",
                Constraint::Or(vec![
                    Constraint::Pattern("^active$".to_string()),
                    Constraint::Pattern("^inactive$".to_string()),
                ]),
            );

        let valid_data = serde_json::json!({
            "status": "active"
        });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({
            "status": "pending"
        });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());

        // Test NOT constraint
        let schema = ValidationSchema::new()
            .require_field("password", DataType::String)
            .add_constraint(
                "password",
                Constraint::Not(Box::new(Constraint::Pattern("password".to_string()))),
            );

        let valid_data = serde_json::json!({
            "password": "s3cr3t"
        });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({
            "password": "password123"
        });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());

        // Test IF-THEN constraint
        let schema = ValidationSchema::new()
            .optional_field("premium", DataType::Boolean)
            .optional_field("limit", DataType::Float64)
            .add_constraint(
                "limit",
                Constraint::If {
                    condition: Box::new(Constraint::NotNull),
                    then_constraint: Box::new(Constraint::Range {
                        min: 0.0,
                        max: 1000000.0,
                    }),
                    else_constraint: None,
                },
            );

        let valid_data = serde_json::json!({
            "premium": true,
            "limit": 50000.0
        });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());
    }

    #[test]

    fn test_edge_case_validations() {
        let validator = Validator::new(ValidationConfig::default()).unwrap();

        // Test empty AND constraint
        let schema = ValidationSchema::new()
            .require_field(value, DataType::Float64)
            .add_constraint(value, Constraint::And(vec![]));

        let data = serde_json::json!({ value: 42.0 });
        let result = validator.validate(&data, &schema).unwrap();
        assert!(result.is_valid()); // Empty AND should pass

        // Test empty OR constraint
        let schema = ValidationSchema::new()
            .require_field(value, DataType::Float64)
            .add_constraint(value, Constraint::Or(vec![]));

        let result = validator.validate(&data, &schema).unwrap();
        assert!(result.is_valid()); // Empty OR currently passes, but could be considered invalid

        // Test nested AND/OR combinations
        let complex_constraint = Constraint::And(vec![
            Constraint::Or(vec![
                Constraint::Range {
                    min: 0.0,
                    max: 50.0,
                },
                Constraint::Range {
                    min: 100.0,
                    max: 150.0,
                },
            ]),
            Constraint::Not(Box::new(Constraint::Range {
                min: 25.0,
                max: 30.0,
            })),
        ]);

        let schema = ValidationSchema::new()
            .require_field("score", DataType::Float64)
            .add_constraint("score", complex_constraint);

        // Value 20 should pass: in range 0-50 AND not in range 25-30
        let valid_data = serde_json::json!({ "score": 20.0 });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        // Value 27 should fail: in range 0-50 BUT also in range 25-30
        let invalid_data = serde_json::json!({ "score": 27.0 });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());

        // Test IF-THEN-ELSE with field dependencies
        // Note: This test shows a limitation - we need a way to reference other fields
        // For now, we'll test a simpler case where the condition is on the same field
        let schema = ValidationSchema::new()
            .optional_field(value, DataType::Float64)
            .add_constraint(
                value,
                Constraint::If {
                    condition: Box::new(Constraint::Range {
                        min: 1000.0,
                        max: f64::INFINITY,
                    }),
                    then_constraint: Box::new(Constraint::Range {
                        min: 1000.0,
                        max: 10000.0,
                    }),
                    else_constraint: Some(Box::new(Constraint::Range {
                        min: 0.0,
                        max: 100.0,
                    })),
                },
            );

        // High value (>= 1000) must be in range 1000-10000
        let valid_high = serde_json::json!({
            value: 5000.0
        });
        let result = validator.validate(&valid_high, &schema).unwrap();
        assert!(result.is_valid());

        // Low value (< 1000) must be in range 0-100
        let valid_low = serde_json::json!({
            value: 50.0
        });
        let result = validator.validate(&valid_low, &schema).unwrap();
        assert!(result.is_valid());

        // High value out of allowed range (should fail)
        let invalid_high = serde_json::json!({
            value: 15000.0
        });
        let result = validator.validate(&invalid_high, &schema).unwrap();
        assert!(!result.is_valid());

        // Low value out of allowed range (should fail)
        let invalid_low = serde_json::json!({
            value: 150.0
        });
        let result = validator.validate(&invalid_low, &schema).unwrap();
        assert!(!result.is_valid());

        // Test multiple NOT constraints
        let schema = ValidationSchema::new()
            .require_field("code", DataType::String)
            .add_constraint(
                "code",
                Constraint::And(vec![
                    Constraint::Not(Box::new(Constraint::Pattern(test.to_string()))),
                    Constraint::Not(Box::new(Constraint::Pattern(debug.to_string()))),
                    Constraint::Length { min: 3, max: 10 },
                ]),
            );

        let valid_data = serde_json::json!({ "code": "prod123" });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({ "code": "test123" });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());
    }

    #[test]

    fn test_constrainterror_messages() {
        let validator = Validator::new(ValidationConfig::default()).unwrap();

        // Test that OR constraint provides meaningful error
        let schema = ValidationSchema::new()
            .require_field("format", DataType::String)
            .add_constraint(
                "format",
                Constraint::Or(vec![
                    Constraint::Pattern("^[A-Z]{3}$".to_string()),
                    Constraint::Pattern("^[0-9]{6}$".to_string()),
                ]),
            );

        let invalid_data = serde_json::json!({ "format": "abc123" });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());

        let errors = result.errors();
        assert!(!errors.is_empty());
        assert!(errors[0]
            .message
            .contains("None of the OR constraints passed"));

        // Test nested constraint error propagation
        let schema = ValidationSchema::new()
            .require_field("data", DataType::Array(Box::new(DataType::Float64)))
            .add_constraint(
                "data",
                Constraint::And(vec![
                    Constraint::ArraySize { min: 2, max: 10 },
                    Constraint::ArrayElements(Box::new(Constraint::Range {
                        min: 0.0,
                        max: 100.0,
                    })),
                ]),
            );

        let invalid_data = serde_json::json!({
            "data": [10.0, 20.0, 150.0] // 150 is out of range
        });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());

        let errors = result.errors();
        assert!(errors.iter().any(|e| e.field_path.contains("[2]")));
    }

    #[test]

    fn test_performance_edge_cases() {
        let validator = Validator::new(ValidationConfig::default()).unwrap();

        // Test deeply nested constraints
        let mut constraint = Constraint::Range {
            min: 0.0,
            max: 100.0,
        };
        for _ in 0..10 {
            constraint = Constraint::And(vec![constraint.clone(), Constraint::NotNull]);
        }

        let schema = ValidationSchema::new()
            .require_field(value, DataType::Float64)
            .add_constraint(value, constraint);

        let data = serde_json::json!({ value: 50.0 });
        let result = validator.validate(&data, &schema).unwrap();
        assert!(result.is_valid());

        // Test large OR constraint
        let many_patterns: Vec<Constraint> = (0..100)
            .map(|i| Constraint::Pattern(format!("pattern{}", i)))
            .collect();

        let schema = ValidationSchema::new()
            .require_field("text", DataType::String)
            .add_constraint("text", Constraint::Or(many_patterns));

        let valid_data = serde_json::json!({ "text": "pattern42" });
        let result = validator.validate(&valid_data, &schema).unwrap();
        assert!(result.is_valid());

        let invalid_data = serde_json::json!({ "text": "no-match" });
        let result = validator.validate(&invalid_data, &schema).unwrap();
        assert!(!result.is_valid());
    }
}
