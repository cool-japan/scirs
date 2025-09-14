//! Comprehensive Data Validation System
//!
//! Production-grade data validation system for `SciRS2` Core providing schema
//! validation, constraint enforcement, and data integrity checks for scientific
//! computing applications in regulated environments.
//!
//! ## Features
//!
//! - JSON Schema validation with scientific extensions
//! - Constraint-based validation (range, format, pattern)
//! - Composite constraints with logical operators (AND, OR, NOT, IF-THEN)
//! - Data integrity verification with checksums
//! - Type safety validation for numeric data
//! - Custom validation rules and plugins
//! - Performance-optimized validation pipelines
//! - Integration with ndarray for array validation
//! - Support for complex nested data structures
//! - Validation caching for repeated validations
//! - Detailed error reporting with context
//! - ConstraintBuilder for fluent constraint composition
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::validation::data::{Validator, ValidationSchema, ValidationConfig, DataType, Constraint};
//! use ndarray::Array2;
//!
//! // Create a validation schema
//! let schema = ValidationSchema::new()
//!     .require_field(name, DataType::String)
//!     .require_field("age", DataType::Integer)
//!     .add_constraint("age", Constraint::Range { min: 0.0, max: 150.0 })
//!     .require_field("data", DataType::Array(Box::new(DataType::Float64)));
//!
//! let config = ValidationConfig::default();
//! let validator = Validator::new(config)?;
//!
//! // For JSON validation (when serde feature is enabled)
//!
//! {
//!     let data = serde_json::json!({
//!         name: "Test Dataset",
//!         "age": 25,
//!         "data": [[1.0, 2.0], [3.0, 4.0]]
//!     });
//!
//!     let result = validator.validate(&data, &schema)?;
//!     if result.is_valid() {
//!         println!("Data is valid!");
//!     } else {
//!         println!("Validation errors: {:#?}", result.errors());
//!     }
//! }
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Using Composite Constraints
//!
//! The validation system now supports composite constraints using logical operators:
//!
//! ```rust
//! use scirs2_core::validation::data::{Constraint, ConstraintBuilder, ValidationSchema};
//!
//! // Create complex constraints using the builder
//! let age_constraint = ConstraintBuilder::new()
//!     .range(18.0, 65.0)
//!     .not_null()
//!     .and();
//!
//! // Use logical operators for conditional validation
//! let email_or_phone = Constraint::Or(vec![
//!     Constraint::Pattern(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$".to_string()),
//!     Constraint::Pattern(r"^\+?[1-9]\d{1,14}$".to_string()),
//! ]);
//!
//! // Conditional constraints: if age > 18, require consent field
//! let consent_constraint = Constraint::if_then(
//!     Constraint::Range { min: 18.0, max: f64::INFINITY },
//!     Constraint::NotNull,
//!     None
//! );
//!
//! let schema = ValidationSchema::new()
//!     .require_field("age", DataType::Integer)
//!     .add_constraint("age", age_constraint)
//!     .require_field("contact", DataType::String)
//!     .add_constraint("contact", email_or_phone);
//! ```
//!
//! ## Performance Features
//!
//! The validation system includes several performance optimizations:
//!
//! - **Validation Caching**: Results are cached for repeated validations with configurable TTL
//! - **Parallel Validation**: Array elements can be validated in parallel when enabled
//! - **Early Exit**: Validation stops at first error when configured for fail-fast mode
//! - **Lazy Evaluation**: Composite constraints evaluate only as needed
//! - **Memory Efficiency**: Streaming validation for large datasets
//!
//! ```rust
//! use scirs2_core::validation::data::ValidationConfig;
//!
//! let mut config = ValidationConfig::default();
//! config.strict_mode = true; // Fail fast on first error
//! config.enable_caching = true; // Enable result caching
//! config.cache_size_limit = 1000; // Cache up to 1000 results
//! config.enable_parallel_validation = true; // Parallel array validation
//! config.performance_mode = true; // Optimize for speed
//! ```

// Core modules
pub mod array_validation;
pub mod config;
pub mod constraints;
pub mod errors;
pub mod quality;
pub mod schema;
pub mod validator;

// Re-export main types and functions for backward compatibility

// Configuration and types
pub use config::{ErrorSeverity, QualityIssueType, ValidationConfig, ValidationErrorType};

// Schema and constraints
pub use schema::{DataType, FieldDefinition, ValidationSchema};

pub use constraints::{
    ArrayValidationConstraints, Constraint, ConstraintBuilder, ElementValidatorFn,
    ShapeConstraints, SparseFormat, StatisticalConstraints, TimeConstraints,
};

// Errors and results
pub use errors::{ValidationError, ValidationResult, ValidationStats};

// Quality assessment
pub use quality::{
    DataQualityReport, QualityAnalyzer, QualityIssue, QualityMetrics, StatisticalSummary,
};

// Array validation
pub use array_validation::ArrayValidator;

// Main validator
pub use validator::{ValidationRule, Validator};

// Type aliases for convenience
pub type Array1<T> = ndarray::Array1<T>;
pub type Array2<T> = ndarray::Array2<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_module_integration() {
        // Test that all major functionality is accessible through the module
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        // Test array validation
        let constraints = ArrayValidationConstraints::new()
            .withshape(vec![6, 2])
            .with_fieldname(test_data)
            .check_numeric_quality();

        let config = ValidationConfig::default();
        let validator = Validator::new(config.clone()).unwrap();

        let result = validator
            .validate_ndarray(&data, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());

        // Test quality report generation
        let report = validator
            .generate_quality_report(&data, "test_data")
            .unwrap();
        assert!(report.quality_score > 0.9);
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that the old API still works after refactoring
        use crate::validation::data::*;

        let config = ValidationConfig::default();
        let validator = Validator::new(config.clone()).unwrap();

        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        // These should all work exactly as they did before refactoring
        let constraints = ArrayValidationConstraints::new()
            .withshape(vec![4, 2])
            .check_numeric_quality();

        let result = validator
            .validate_ndarray(&data, &constraints, &config)
            .unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_json_validation_integration() {
        // Test JSON validation functionality
        let schema = ValidationSchema::new()
            .name("test_schema")
            .require_field(name, DataType::String)
            .require_field("age", DataType::Integer)
            .add_constraint(
                "age",
                Constraint::Range {
                    min: 0.0,
                    max: 150.0,
                },
            );

        let config = ValidationConfig::default();
        let validator = Validator::new(config).unwrap();

        let data = serde_json::json!({
            name: "Test User",
            "age": 25
        });

        let result = validator.validate(&data, &schema).unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_constraint_types() {
        // Test various constraint types
        let range_constraint = Constraint::Range {
            min: 0.0,
            max: 100.0,
        };
        let length_constraint = Constraint::Length { min: 1, max: 50 };
        let not_null_constraint = Constraint::NotNull;
        let unique_constraint = Constraint::Unique;

        // Test that constraints can be created and used
        match range_constraint {
            Constraint::Range { min, max } => {
                assert_eq!(min, 0.0);
                assert_eq!(max, 100.0);
            }
            _ => panic!("Expected Range constraint"),
        }

        match length_constraint {
            Constraint::Length { min, max } => {
                assert_eq!(min, 1);
                assert_eq!(max, 50);
            }
            _ => panic!("Expected Length constraint"),
        }
    }

    #[test]
    fn test_datatypes() {
        // Test data type definitions
        let string_type = DataType::String;
        let integer_type = DataType::Integer;
        let array_type = DataType::Array(Box::new(DataType::Float64));
        let matrix_type = DataType::Matrix(Box::new(DataType::Float32));

        assert_eq!(string_type, DataType::String);
        assert_eq!(integer_type, DataType::Integer);

        match array_type {
            DataType::Array(inner) => assert_eq!(*inner, DataType::Float64),
            _ => panic!("Expected Array type"),
        }

        match matrix_type {
            DataType::Matrix(inner) => assert_eq!(*inner, DataType::Float32),
            _ => panic!("Expected Matrix type"),
        }
    }

    #[test]
    fn test_statistical_constraints() {
        // Test statistical constraints
        let constraints = StatisticalConstraints::new()
            .with_mean_range(0.0, 10.0)
            .with_std_range(1.0, 5.0)
            .with_distribution(normal);

        assert_eq!(constraints.min_mean, Some(0.0));
        assert_eq!(constraints.max_mean, Some(10.0));
        assert_eq!(constraints.min_std, Some(1.0));
        assert_eq!(constraints.max_std, Some(5.0));
        assert_eq!(constraints.expected_distribution, Some(normal.to_string()));
    }

    #[test]
    fn test_validationerror_creation() {
        // Test validation error creation and formatting
        let error = ValidationError::new(
            ValidationErrorType::TypeMismatch,
            "test_field",
            "Type mismatch error",
        )
        .with_expected(string)
        .with_actual(integer)
        .with_constraint(type_check)
        .with_severity(ErrorSeverity::Error);

        assert_eq!(error.errortype, ValidationErrorType::TypeMismatch);
        assert_eq!(error.field_path, "test_field");
        assert_eq!(error.message, "Type mismatch error");

        let formatted = error.formatted_message();
        assert!(formatted.contains(test_field));
        assert!(formatted.contains("Type mismatch error"));
    }

    #[test]
    fn test_schema_builder() {
        // Test schema builder pattern
        let schema = ValidationSchema::new()
            .name("test_schema")
            .version("1.0.0")
            .require_field(name, DataType::String)
            .optional_field("description", DataType::String)
            .add_constraint(name, Constraint::Length { min: 1, max: 100 })
            .allow_additional()
            .with_metadata("author", "test");

        assert_eq!(schema.name, "test_schema");
        assert_eq!(schema.version, "1.0.0");
        assert_eq!(schema.fields.len(), 2);
        assert!(schema.allow_additional_fields);
        assert_eq!(schema.metadata.get(author), Some(&test.to_string()));
    }
}
