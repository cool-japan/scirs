//! Tests for statistical constraint validation

#[cfg(all(feature = "data_validation", feature = "serde"))]
use scirs2_core::validation::data::{
    Constraint, DataType, StatisticalConstraints, ValidationConfig, ValidationSchema, Validator,
};

#[cfg(all(feature = "data_validation", feature = "serde"))]
use serde_json::json;

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_mean() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create statistical constraints for mean
    let stats_constraints = StatisticalConstraints::new().with_mean_range(45.0, 55.0);

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test valid data (mean = 50)
    let data = json!({
        "measurements": [45.0, 50.0, 55.0, 48.0, 52.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(result.is_valid(), "Valid mean should pass validation");
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_mean_too_low() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create statistical constraints for mean
    let stats_constraints = StatisticalConstraints::new().with_mean_range(45.0, 55.0);

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test data with mean too low (mean = 20)
    let data = json!({
        "measurements": [10.0, 20.0, 30.0, 15.0, 25.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(!result.is_valid(), "Low mean should fail validation");

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("statistical.min_mean".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_mean_too_high() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create statistical constraints for mean
    let stats_constraints = StatisticalConstraints::new().with_mean_range(45.0, 55.0);

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test data with mean too high (mean = 80)
    let data = json!({
        "measurements": [70.0, 80.0, 90.0, 75.0, 85.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(!result.is_valid(), "High mean should fail validation");

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("statistical.max_mean".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_std_dev() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create statistical constraints for standard deviation
    let stats_constraints = StatisticalConstraints::new().with_std_range(5.0, 15.0);

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test valid data with moderate variance
    let data = json!({
        "measurements": [45.0, 50.0, 55.0, 40.0, 60.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        result.is_valid(),
        "Valid standard deviation should pass validation"
    );
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_std_too_low() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create statistical constraints for standard deviation
    let stats_constraints = StatisticalConstraints::new().with_std_range(5.0, 15.0);

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test data with very low variance (almost constant)
    let data = json!({
        "measurements": [50.0, 50.1, 49.9, 50.0, 50.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Low standard deviation should fail validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("statistical.min_std".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_std_too_high() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create statistical constraints for standard deviation
    let stats_constraints = StatisticalConstraints::new().with_std_range(5.0, 15.0);

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test data with very high variance
    let data = json!({
        "measurements": [10.0, 90.0, 5.0, 95.0, 50.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "High standard deviation should fail validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("statistical.max_std".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_non_numeric() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create statistical constraints
    let stats_constraints = StatisticalConstraints::new().with_mean_range(45.0, 55.0);

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test data with non-numeric values
    let data = json!({
        "measurements": [45.0, "not a number", 55.0, true, null]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Non-numeric values should fail validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.message.contains("Expected numeric value")));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_empty_array() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create statistical constraints
    let stats_constraints = StatisticalConstraints::new().with_mean_range(45.0, 55.0);

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test empty array
    let data = json!({
        "measurements": []
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Empty array should fail statistical validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.message.contains("requires numeric values")));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_integer_values() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create statistical constraints
    let stats_constraints = StatisticalConstraints::new().with_mean_range(45.0, 55.0);

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test integer values (should be converted to floats)
    let data = json!({
        "measurements": [45, 50, 55, 48, 52]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        result.is_valid(),
        "Integer values should be accepted and converted"
    );
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_statistical_constraint_complex() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create complex statistical constraints
    let stats_constraints = StatisticalConstraints::new()
        .with_mean_range(48.0, 52.0)
        .with_std_range(8.0, 12.0)
        .with_distribution("normal");

    let schema = ValidationSchema::new()
        .require_field("measurements", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("measurements", Constraint::Statistical(stats_constraints));

    // Test valid data that meets all constraints
    // Mean = 50, Std = ~10
    let data = json!({
        "measurements": [35.0, 45.0, 50.0, 55.0, 65.0]
    });

    let result = validator.validate(&data, &schema).unwrap();

    // Should be valid for mean and std, but get warning about distribution
    if !result.is_valid() {
        println!("Validation errors: {:?}", result.errors());
        // Calculate actual mean and std for debugging
        let values = [40.0, 45.0, 50.0, 55.0, 60.0];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();
        println!("Actual mean: {}, std: {}", mean, std);
    }
    assert!(
        result.is_valid(),
        "Data meeting mean and std constraints should pass"
    );

    let warnings = result.warnings();
    assert!(warnings
        .iter()
        .any(|w| w.message.contains("Distribution testing")));
}
