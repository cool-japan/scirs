//! Comprehensive examples for the data validation system
//!
//! This example demonstrates all the validation features including:
//! - Basic data type validation
//! - Constraint validation (temporal, statistical, shape)
//! - Custom validation rules
//! - Schema composition
//! - Array validation
//! - Performance optimizations

#[cfg(feature = "data_validation")]
use scirs2_core::validation::data::*;
#[cfg(feature = "data_validation")]
use serde_json::Value as JsonValue;
#[cfg(feature = "data_validation")]
use std::time::Duration;

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comprehensive Data Validation Examples ===\n");

    // Example 1: Basic data type validation
    basic_validation_example()?;

    // Example 2: Temporal constraint validation
    temporal_validation_example()?;

    // Example 3: Statistical constraint validation
    statistical_validation_example()?;

    // Example 4: Shape constraint validation
    shape_validation_example()?;

    // Example 5: Complex schema composition
    complex_schema_example()?;

    // Example 6: Array validation with ndarray
    array_validation_example()?;

    // Example 7: Custom validation rules
    custom_validation_example()?;

    Ok(())
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn basic_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Data Type Validation");
    println!("-----------------------------");

    let config = ValidationConfig::default();
    let validator = Validator::new(config)?;

    // Create a schema for user data
    let schema = ValidationSchema::new()
        .name("user_profile")
        .require_field("username", DataType::String)
        .require_field("age", DataType::Integer)
        .require_field("email", DataType::String)
        .optional_field("bio", DataType::String)
        .require_field("scores", DataType::Array(Box::new(DataType::Float64)));

    // Valid data

    {
        use serde_json::json;

        let valid_data = json!({
            "username": "john_doe",
            "age": 30,
            "email": "john@example.com",
            "bio": "Software developer",
            "scores": [85.5, 92.0, 78.5, 94.5]
        });

        let result = validator.validate(&valid_data, &schema)?;
        println!(
            "Valid data validation: {}",
            if result.is_valid() {
                "PASSED"
            } else {
                "FAILED"
            }
        );

        // Invalid data (missing required field)
        let invalid_data = json!({
            "username": "jane_doe",
            "email": "jane@example.com"
            // Missing required fields: age, scores
        });

        let result = validator.validate(&invalid_data, &schema)?;
        println!(
            "Invalid data validation: {}",
            if !result.is_valid() {
                "CORRECTLY FAILED"
            } else {
                "INCORRECTLY PASSED"
            }
        );
        if !result.is_valid() {
            for error in result.errors() {
                println!("  - Error: {}", error.message);
            }
        }
    }

    println!();
    Ok(())
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn temporal_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Temporal Constraint Validation");
    println!("---------------------------------");

    let config = ValidationConfig::default();
    let validator = Validator::new(config)?;

    // Create temporal constraints for time series data
    let time_constraints = TimeConstraints::new()
        .with_min_interval(Duration::from_secs(1))
        .with_max_interval(Duration::from_secs(300)) // 5 minutes
        .require_monotonic()
        .disallow_duplicates();

    let schema = ValidationSchema::new()
        .name("sensor_data")
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .require_field("values", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    {
        use serde_json::json;

        // Valid time series data
        let valid_data = json!({
            "timestamps": [1000, 2000, 3500, 7000, 10000], // Monotonic with valid intervals
            "values": [23.5, 24.1, 23.8, 24.5, 25.0]
        });

        let result = validator.validate(&valid_data, &schema)?;
        println!(
            "Valid temporal data: {}",
            if result.is_valid() {
                "PASSED"
            } else {
                "FAILED"
            }
        );

        // Invalid time series data (non-monotonic)
        let invalid_data = json!({
            "timestamps": [1000, 3000, 2000, 4000, 5000], // 2000 breaks monotonic order
            "values": [23.5, 24.1, 23.8, 24.5, 25.0]
        });

        let result = validator.validate(&invalid_data, &schema)?;
        println!(
            "Non-monotonic timestamps: {}",
            if !result.is_valid() {
                "CORRECTLY FAILED"
            } else {
                "INCORRECTLY PASSED"
            }
        );
    }

    println!();
    Ok(())
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn statistical_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Statistical Constraint Validation");
    println!("-----------------------------------");

    let config = ValidationConfig::default();
    let validator = Validator::new(config)?;

    // Create statistical constraints for measurement data
    let stats_constraints = StatisticalConstraints::new()
        .with_mean_range(20.0, 30.0)  // Expected temperature range
        .with_std_range(0.5, 5.0)     // Expected variability
        .with_distribution("normal"); // Expected distribution

    let schema = ValidationSchema::new()
        .name("temperature_readings")
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    {
        use serde_json::json;

        // Valid measurements (mean ~25, std ~2)
        let valid_data = json!({
            "measurements": [23.0, 24.5, 25.0, 26.5, 27.0, 24.0, 25.5, 26.0, 23.5, 25.0]
        });

        let result = validator.validate(&valid_data, &schema)?;
        println!(
            "Valid statistical data: {}",
            if result.is_valid() {
                "PASSED"
            } else {
                "FAILED"
            }
        );

        // Invalid measurements (mean too high)
        let invalid_data = json!({
            "measurements": [35.0, 36.5, 37.0, 38.5, 39.0, 36.0, 37.5, 38.0, 35.5, 37.0]
        });

        let result = validator.validate(&invalid_data, &schema)?;
        println!(
            "Out-of-range mean: {}",
            if !result.is_valid() {
                "CORRECTLY FAILED"
            } else {
                "INCORRECTLY PASSED"
            }
        );
    }

    println!();
    Ok(())
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn shape_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Shape Constraint Validation");
    println!("------------------------------");

    let config = ValidationConfig::default();
    let validator = Validator::new(config)?;

    // Create shape constraints for matrix data
    let shape_constraints = ShapeConstraints::new()
        .with_dimensions(vec![Some(10), Some(10)])  // Expect 10x10 matrix
        .require_square();

    let schema = ValidationSchema::new()
        .name("correlation_matrix")
        .require_field(
            "matrix",
            DataType::Array(Box::new(DataType::Array(Box::new(DataType::Float64)))),
        )
        .add_constraint("matrix", Constraint::Shape(shape_constraints));

    {
        use serde_json::json;

        // Valid 3x3 matrix (smaller for example)
        let shape_3x3 = ShapeConstraints::new()
            .with_dimensions(vec![Some(3), Some(3)])
            .require_square();

        let schema_3x3 = ValidationSchema::new()
            .name("smallmatrix")
            .require_field(
                "matrix",
                DataType::Array(Box::new(DataType::Array(Box::new(DataType::Float64)))),
            )
            .add_constraint("matrix", Constraint::Shape(shape_3x3));

        let valid_data = json!({
            "matrix": [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.7],
                [0.3, 0.7, 1.0]
            ]
        });

        let result = validator.validate(&valid_data, &schema_3x3)?;
        println!(
            "Valid square matrix: {}",
            if result.is_valid() {
                "PASSED"
            } else {
                "FAILED"
            }
        );

        // Invalid matrix (not square)
        let invalid_data = json!({
            "matrix": [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.7]
                // Only 2x3, not square
            ]
        });

        let result = validator.validate(&invalid_data, &schema_3x3)?;
        println!(
            "Non-square matrix: {}",
            if !result.is_valid() {
                "CORRECTLY FAILED"
            } else {
                "INCORRECTLY PASSED"
            }
        );
    }

    println!();
    Ok(())
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn complex_schema_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Complex Schema Composition");
    println!("-----------------------------");

    let config = ValidationConfig::default();
    let validator = Validator::new(config)?;

    // Create a complex schema for scientific experiment data
    let time_constraints = TimeConstraints::new()
        .with_min_interval(Duration::from_millis(100))
        .require_monotonic();

    let measurement_constraints = StatisticalConstraints::new()
        .with_mean_range(0.0, 100.0)
        .with_std_range(0.1, 50.0);

    let schema = ValidationSchema::new()
        .name("experiment_data")
        .require_field("experiment_id", DataType::String)
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .require_field("metadata", DataType::Object)
        .add_constraint("timestamps", Constraint::Temporal(time_constraints))
        .add_constraint(
            "measurements",
            Constraint::Statistical(measurement_constraints),
        )
        .add_constraint(
            "measurements",
            Constraint::Range {
                min: -100.0,
                max: 200.0,
            },
        );

    {
        use serde_json::json;

        let experiment_data = json!({
            "experiment_id": "EXP-2025-001",
            "timestamps": [0.0, 100.0, 200.0, 300.0, 400.0],
            "measurements": [23.5, 24.1, 25.3, 24.8, 25.5],
            "metadata": {
                "instrument": "Thermometer-X100",
                "location": "Lab A",
                "operator": "Dr. Smith"
            }
        });

        let result = validator.validate(&experiment_data, &schema)?;
        println!(
            "Complex experiment data validation: {}",
            if result.is_valid() {
                "PASSED"
            } else {
                "FAILED"
            }
        );
    }

    println!();
    Ok(())
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn array_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Array Validation with ndarray");
    println!("--------------------------------");

    use ndarray::{Array1, Array2};

    let config = ValidationConfig::default();
    let validator = Validator::new(config.clone())?;

    // Validate a 1D array
    let data_1d = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let constraints_1d = ArrayValidationConstraints::new()
        .withshape(vec![5])
        .check_numeric_quality();

    let result = validator.validate_ndarray(&data_1d, &constraints_1d, &config)?;
    println!(
        "1D array validation: {}",
        if result.is_valid() {
            "PASSED"
        } else {
            "FAILED"
        }
    );

    // Validate a 2D array
    let data_2d =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])?;

    let constraints_2d = ArrayValidationConstraints::new()
        .withshape(vec![3, 3])
        .check_numeric_quality();

    let result = validator.validate_ndarray(&data_2d, &constraints_2d, &config)?;
    println!(
        "2D array validation: {}",
        if result.is_valid() {
            "PASSED"
        } else {
            "FAILED"
        }
    );

    // Test with invalid data (contains NaN)
    let invalid_data = Array1::from_vec(vec![1.0, 2.0, f64::NAN, 4.0, 5.0]);
    let result = validator.validate_ndarray(&invalid_data, &constraints_1d, &config)?;
    println!(
        "Array with NaN: {}",
        if !result.is_valid() {
            "CORRECTLY FAILED"
        } else {
            "INCORRECTLY PASSED"
        }
    );
    if result.is_valid() {
        println!("  - Expected validation to fail for NaN, but it passed");
        println!("  - Note: NaN checking might need to be explicitly enabled in constraints");
    }

    println!();
    Ok(())
}

#[cfg(feature = "data_validation")]
#[allow(dead_code)]
fn custom_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Custom Validation Rules");
    println!("--------------------------");

    // Create a custom validation rule
    struct EmailValidator;

    impl ValidationRule for EmailValidator {
        fn validate(&self, value: &JsonValue, _fieldpath: &str) -> Result<(), String> {
            if let Some(email) = value.as_str() {
                if email.contains('@') && email.contains('.') {
                    Ok(())
                } else {
                    Err("Invalid email format".to_string())
                }
            } else {
                Err("Expected string value for email".to_string())
            }
        }

        fn name(&self) -> &str {
            "email_format"
        }

        fn description(&self) -> &str {
            "Validates email format (must contain @ and .)"
        }
    }

    let config = ValidationConfig::default();
    let mut validator = Validator::new(config)?;

    // Add the custom rule
    validator.add_custom_rule("email_format".to_string(), Box::new(EmailValidator));

    // Create a schema that uses the custom rule
    // Note: Custom rules need to be added through field definition, not constraints
    let mut schema = ValidationSchema::new().name("contact_info");

    // Add the email field with custom validation rule
    let email_field = FieldDefinition::new(DataType::String)
        .required()
        .with_validation_rule("email_format");

    // Add the field to the schema
    schema.fields.insert("email".to_string(), email_field);

    {
        use serde_json::json;

        // Valid email
        let valid_data = json!({
            "email": "user@example.com"
        });

        let result = validator.validate(&valid_data, &schema)?;
        println!(
            "Valid email: {}",
            if result.is_valid() {
                "PASSED"
            } else {
                "FAILED"
            }
        );

        // Invalid email
        let invalid_data = json!({
            "email": "invalid-email"
        });

        let result = validator.validate(&invalid_data, &schema)?;
        println!(
            "Invalid email: {}",
            if !result.is_valid() {
                "CORRECTLY FAILED"
            } else {
                "INCORRECTLY PASSED"
            }
        );
        if result.is_valid() {
            println!("  - Expected validation to fail, but it passed");
        } else {
            for error in result.errors() {
                println!("  - Error: {}", error.message);
            }
        }
    }

    println!();
    Ok(())
}

#[cfg(not(feature = "data_validation"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'data_validation' feature to be enabled.");
    println!("Run with: cargo run --example validation_comprehensive --features data_validation");
}
