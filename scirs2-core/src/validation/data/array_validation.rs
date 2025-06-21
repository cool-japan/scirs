//! Array validation functionality for ndarray types
//!
//! This module provides comprehensive validation for ndarray arrays,
//! including shape, numeric quality, statistical properties, and performance checks.

use crate::error::CoreError;
use std::collections::HashMap;
use std::fmt;

// Core dependencies for array/matrix validation
use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::config::{ErrorSeverity, ValidationErrorType};
use super::constraints::{ArrayValidationConstraints, StatisticalConstraints};
use super::errors::{ValidationError, ValidationResult, ValidationStats};

/// Array validator with comprehensive validation capabilities
pub struct ArrayValidator;

impl ArrayValidator {
    /// Create new array validator
    pub fn new() -> Self {
        Self
    }

    /// Validate ndarray with comprehensive checks
    pub fn validate_ndarray<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        constraints: &ArrayValidationConstraints,
        _config: &super::config::ValidationConfig,
    ) -> Result<ValidationResult, CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + Send + Sync + ScalarOperand + FromPrimitive,
    {
        let start_time = std::time::Instant::now();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut stats = ValidationStats::default();

        // Basic shape validation
        if let Some(expected_shape) = &constraints.expected_shape {
            if !self.validate_array_shape(array, expected_shape)? {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::ShapeError,
                    field_path: constraints
                        .field_name
                        .clone()
                        .unwrap_or("array".to_string()),
                    message: format!(
                        "Array shape {:?} does not match expected {:?}",
                        array.shape(),
                        expected_shape
                    ),
                    expected: Some(format!("{:?}", expected_shape)),
                    actual: Some(format!("{:?}", array.shape())),
                    constraint: Some("shape".to_string()),
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
            }
        }

        // Numeric quality validation
        if constraints.check_numeric_quality {
            self.validate_numeric_quality(array, &mut errors, &mut warnings, &mut stats)?;
        }

        // Statistical validation
        if let Some(stat_constraints) = &constraints.statistical_constraints {
            self.validate_statistical_properties(
                array,
                stat_constraints,
                &mut errors,
                &mut warnings,
            )?;
        }

        // Performance validation for large arrays
        if constraints.check_performance {
            self.validate_array_performance(array, &mut warnings)?;
        }

        // Element-wise validation if validator provided
        if let Some(ref validator) = constraints.element_validator {
            self.validate_elements(array, validator, &mut errors, &mut warnings)?;
        }

        let valid = errors.is_empty()
            && !warnings
                .iter()
                .any(|w| w.severity == ErrorSeverity::Critical);
        let duration = start_time.elapsed();

        Ok(ValidationResult {
            valid,
            errors,
            warnings,
            stats,
            duration,
        })
    }

    /// Validate array shape against constraints
    fn validate_array_shape<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        expected_shape: &[usize],
    ) -> Result<bool, CoreError>
    where
        S: Data,
        D: Dimension,
    {
        let actual_shape = array.shape();
        Ok(actual_shape == expected_shape)
    }

    /// Validate numeric quality (NaN, infinity, precision)
    #[allow(clippy::ptr_arg)]
    fn validate_numeric_quality<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        _errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
        stats: &mut ValidationStats,
    ) -> Result<(), CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + Send + Sync,
    {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let total_count = array.len();

        #[cfg(feature = "parallel")]
        let check_parallel = array.len() > 10000;

        #[cfg(feature = "parallel")]
        if check_parallel {
            if let Some(slice) = array.as_slice() {
                let results: Vec<_> = slice
                    .par_iter()
                    .map(|&value| {
                        let is_nan = value.is_nan();
                        let is_inf = value.is_infinite();
                        (is_nan, is_inf)
                    })
                    .collect();

                for (is_nan, is_inf) in results {
                    if is_nan {
                        nan_count += 1;
                    }
                    if is_inf {
                        inf_count += 1;
                    }
                }
            } else {
                // Fallback for non-contiguous arrays
                for value in array.iter() {
                    if value.is_nan() {
                        nan_count += 1;
                    }
                    if value.is_infinite() {
                        inf_count += 1;
                    }
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for value in array.iter() {
                if value.is_nan() {
                    nan_count += 1;
                }
                if value.is_infinite() {
                    inf_count += 1;
                }
            }
        }

        #[cfg(feature = "parallel")]
        if !check_parallel {
            for value in array.iter() {
                if value.is_nan() {
                    nan_count += 1;
                }
                if value.is_infinite() {
                    inf_count += 1;
                }
            }
        }

        stats.fields_validated += 1;
        stats.constraints_checked += 2; // NaN and infinity checks
        stats.elements_processed += total_count;

        if nan_count > 0 {
            warnings.push(ValidationError {
                error_type: ValidationErrorType::InvalidNumeric,
                field_path: "array".to_string(),
                message: format!(
                    "Found {} NaN values out of {} total",
                    nan_count, total_count
                ),
                expected: Some("finite values".to_string()),
                actual: Some(format!("{} NaN values", nan_count)),
                constraint: Some("numeric_quality".to_string()),
                severity: ErrorSeverity::Warning,
                context: HashMap::new(),
            });
        }

        if inf_count > 0 {
            warnings.push(ValidationError {
                error_type: ValidationErrorType::InvalidNumeric,
                field_path: "array".to_string(),
                message: format!(
                    "Found {} infinite values out of {} total",
                    inf_count, total_count
                ),
                expected: Some("finite values".to_string()),
                actual: Some(format!("{} infinite values", inf_count)),
                constraint: Some("numeric_quality".to_string()),
                severity: ErrorSeverity::Warning,
                context: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Validate statistical properties of numeric arrays
    fn validate_statistical_properties<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        constraints: &StatisticalConstraints,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationError>,
    ) -> Result<(), CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + ScalarOperand + FromPrimitive,
    {
        if array.is_empty() {
            return Ok(());
        }

        // Calculate basic statistics
        let mean = array.mean().unwrap_or(S::Elem::zero());
        let std_dev = array.std(num_traits::cast(1.0).unwrap());

        // Validate mean constraints
        if let Some(min_mean) = constraints.min_mean {
            let min_mean_typed: S::Elem = num_traits::cast(min_mean).unwrap_or(S::Elem::zero());
            if mean < min_mean_typed {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    field_path: "array.mean".to_string(),
                    message: format!("Array mean {:?} is below minimum {:?}", mean, min_mean),
                    expected: Some(format!("mean >= {}", min_mean)),
                    actual: Some(format!("{:?}", mean)),
                    constraint: Some("statistical.min_mean".to_string()),
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
            }
        }

        if let Some(max_mean) = constraints.max_mean {
            let max_mean_typed: S::Elem = num_traits::cast(max_mean).unwrap_or(S::Elem::zero());
            if mean > max_mean_typed {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    field_path: "array.mean".to_string(),
                    message: format!("Array mean {:?} exceeds maximum {:?}", mean, max_mean),
                    expected: Some(format!("mean <= {}", max_mean)),
                    actual: Some(format!("{:?}", mean)),
                    constraint: Some("statistical.max_mean".to_string()),
                    severity: ErrorSeverity::Error,
                    context: HashMap::new(),
                });
            }
        }

        // Validate standard deviation constraints
        if let Some(min_std) = constraints.min_std {
            let min_std_typed: S::Elem = num_traits::cast(min_std).unwrap_or(S::Elem::zero());
            if std_dev < min_std_typed {
                warnings.push(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    field_path: "array.std".to_string(),
                    message: format!(
                        "Array standard deviation {:?} is below minimum {:?}",
                        std_dev, min_std
                    ),
                    expected: Some(format!("std >= {}", min_std)),
                    actual: Some(format!("{:?}", std_dev)),
                    constraint: Some("statistical.min_std".to_string()),
                    severity: ErrorSeverity::Warning,
                    context: HashMap::new(),
                });
            }
        }

        if let Some(max_std) = constraints.max_std {
            let max_std_typed: S::Elem = num_traits::cast(max_std).unwrap_or(S::Elem::zero());
            if std_dev > max_std_typed {
                warnings.push(ValidationError {
                    error_type: ValidationErrorType::ConstraintViolation,
                    field_path: "array.std".to_string(),
                    message: format!(
                        "Array standard deviation {:?} exceeds maximum {:?}",
                        std_dev, max_std
                    ),
                    expected: Some(format!("std <= {}", max_std)),
                    actual: Some(format!("{:?}", std_dev)),
                    constraint: Some("statistical.max_std".to_string()),
                    severity: ErrorSeverity::Warning,
                    context: HashMap::new(),
                });
            }
        }

        Ok(())
    }

    /// Validate performance characteristics for large arrays
    fn validate_array_performance<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        warnings: &mut Vec<ValidationError>,
    ) -> Result<(), CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: fmt::Debug,
    {
        let element_count = array.len();
        let element_size = std::mem::size_of::<S::Elem>();
        let total_size = element_count * element_size;

        // Warn about very large arrays
        const LARGE_ARRAY_THRESHOLD: usize = 100_000_000; // 100M elements
        if element_count > LARGE_ARRAY_THRESHOLD {
            warnings.push(ValidationError {
                error_type: ValidationErrorType::Performance,
                field_path: "array.size".to_string(),
                message: format!(
                    "Large array detected: {} elements ({} bytes). Consider chunking for better performance.",
                    element_count, total_size
                ),
                expected: Some(format!("<= {} elements", LARGE_ARRAY_THRESHOLD)),
                actual: Some(format!("{} elements", element_count)),
                constraint: Some("performance.max_elements".to_string()),
                severity: ErrorSeverity::Warning,
                context: HashMap::new(),
            });
        }

        // Warn about memory usage
        const LARGE_MEMORY_THRESHOLD: usize = 1_000_000_000; // 1GB
        if total_size > LARGE_MEMORY_THRESHOLD {
            warnings.push(ValidationError {
                error_type: ValidationErrorType::Performance,
                field_path: "array.memory".to_string(),
                message: format!(
                    "High memory usage: {} bytes. Consider memory-efficient operations.",
                    total_size
                ),
                expected: Some(format!("<= {} bytes", LARGE_MEMORY_THRESHOLD)),
                actual: Some(format!("{} bytes", total_size)),
                constraint: Some("performance.max_memory".to_string()),
                severity: ErrorSeverity::Warning,
                context: HashMap::new(),
            });
        }

        Ok(())
    }

    /// Validate individual elements using custom validator function
    #[allow(clippy::ptr_arg)]
    fn validate_elements<S, D>(
        &self,
        array: &ArrayBase<S, D>,
        validator: &super::constraints::ElementValidatorFn<f64>,
        errors: &mut Vec<ValidationError>,
        _warnings: &mut Vec<ValidationError>,
    ) -> Result<(), CoreError>
    where
        S: Data,
        D: Dimension,
        S::Elem: Float + fmt::Debug + num_traits::cast::ToPrimitive,
    {
        let mut invalid_count = 0;

        for (index, element) in array.iter().enumerate() {
            if let Some(value) = element.to_f64() {
                if !validator(&value) {
                    invalid_count += 1;
                    if invalid_count <= 10 {
                        // Limit error reports to first 10
                        errors.push(ValidationError {
                            error_type: ValidationErrorType::CustomRuleFailure,
                            field_path: format!("array[{}]", index),
                            message: format!("Element {:?} failed custom validation", element),
                            expected: Some("valid element".to_string()),
                            actual: Some(format!("{:?}", element)),
                            constraint: Some("custom_element_validator".to_string()),
                            severity: ErrorSeverity::Error,
                            context: HashMap::new(),
                        });
                    }
                }
            }
        }

        // Report summary if too many errors
        if invalid_count > 10 {
            errors.push(ValidationError {
                error_type: ValidationErrorType::CustomRuleFailure,
                field_path: "array".to_string(),
                message: format!(
                    "Total of {} elements failed custom validation (showing first 10)",
                    invalid_count
                ),
                expected: Some("all elements to pass validation".to_string()),
                actual: Some(format!("{} failed elements", invalid_count)),
                constraint: Some("custom_element_validator".to_string()),
                severity: ErrorSeverity::Error,
                context: HashMap::new(),
            });
        }

        Ok(())
    }
}

impl Default for ArrayValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_array_validator() {
        let validator = ArrayValidator::new();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = super::super::config::ValidationConfig::default();

        let constraints = ArrayValidationConstraints::new()
            .with_shape(vec![5])
            .with_field_name("test_array")
            .check_numeric_quality();

        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_shape_validation() {
        let validator = ArrayValidator::new();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let config = super::super::config::ValidationConfig::default();

        // Test correct shape
        let constraints = ArrayValidationConstraints::new().with_shape(vec![3]);
        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());

        // Test incorrect shape
        let constraints = ArrayValidationConstraints::new().with_shape(vec![5]);
        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 1);
        assert_eq!(
            result.errors()[0].error_type,
            ValidationErrorType::ShapeError
        );
    }

    #[test]
    fn test_numeric_quality_validation() {
        let validator = ArrayValidator::new();
        let config = super::super::config::ValidationConfig::default();

        // Array with NaN values
        let array = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        let constraints = ArrayValidationConstraints::new().check_numeric_quality();

        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid()); // NaN is a warning, not an error
        assert!(result.has_warnings());
        assert_eq!(result.warnings().len(), 1);
        assert_eq!(
            result.warnings()[0].error_type,
            ValidationErrorType::InvalidNumeric
        );
    }

    #[test]
    fn test_statistical_constraints() {
        let validator = ArrayValidator::new();
        let config = super::super::config::ValidationConfig::default();

        // Array with mean around 3.0
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let constraints = ArrayValidationConstraints::new()
            .with_statistical_constraints(StatisticalConstraints::new().with_mean_range(2.0, 4.0));

        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());

        // Test with constraints that should fail
        let failing_constraints = ArrayValidationConstraints::new().with_statistical_constraints(
            StatisticalConstraints::new().with_mean_range(5.0, 6.0), // Mean is 3.0, so this should fail
        );

        let result = validator
            .validate_ndarray(&array, &failing_constraints, &config)
            .unwrap();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_performance_validation() {
        let validator = ArrayValidator::new();
        let config = super::super::config::ValidationConfig::default();

        // Small array - should not trigger performance warnings
        let small_array = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let constraints = ArrayValidationConstraints::new().check_performance();

        let result = validator
            .validate_ndarray(&small_array, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());
        assert!(result.warnings().is_empty());
    }

    #[test]
    fn test_element_validation() {
        let validator = ArrayValidator::new();
        let config = super::super::config::ValidationConfig::default();
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Custom validator that only allows values <= 3.0
        let mut constraints = ArrayValidationConstraints::new();
        constraints.element_validator = Some(Box::new(|&x| x <= 3.0));

        let result = validator
            .validate_ndarray(&array, &constraints, &config)
            .unwrap();
        assert!(!result.is_valid()); // Should fail because 4.0 and 5.0 > 3.0
        assert!(!result.errors().is_empty());
    }
}
