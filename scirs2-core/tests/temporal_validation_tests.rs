//! Tests for temporal constraint validation

#[cfg(all(feature = "data_validation", feature = "serde"))]
use scirs2_core::validation::data::{
    Constraint, DataType, TimeConstraints, ValidationConfig, ValidationSchema, Validator,
};

#[cfg(all(feature = "data_validation", feature = "serde"))]
use std::time::Duration;

#[cfg(all(feature = "data_validation", feature = "serde"))]
use serde_json::json;

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_temporal_constraint_basic() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create temporal constraints
    let time_constraints = TimeConstraints::new()
        .with_min_interval(Duration::from_secs(1))
        .with_max_interval(Duration::from_secs(60));

    let schema = ValidationSchema::new()
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    // Test valid timestamps (intervals: 10s, 15s, 55s)
    let data = json!({
        "timestamps": [1000, 11000, 26000, 81000]
    });

    let result = validator.validate(&data, &schema).unwrap();
    if !result.is_valid() {
        println!("Validation errors: {:?}", result.errors());
    }
    assert!(result.is_valid(), "Valid timestamps should pass validation");
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_temporal_constraint_monotonic() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create temporal constraints requiring monotonic timestamps
    let time_constraints = TimeConstraints::new().require_monotonic();

    let schema = ValidationSchema::new()
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    // Test non-monotonic timestamps
    let data = json!({
        "timestamps": [1000, 1010, 1005, 1020]  // 1005 breaks monotonic order
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Non-monotonic timestamps should fail validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("temporal.monotonic".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_temporal_constraint_no_duplicates() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create temporal constraints disallowing duplicates
    let time_constraints = TimeConstraints::new().disallow_duplicates();

    let schema = ValidationSchema::new()
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    // Test duplicate timestamps
    let data = json!({
        "timestamps": [1000, 1010, 1010, 1020]  // 1010 is duplicated
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Duplicate timestamps should fail validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("temporal.unique".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_temporal_constraint_interval_too_small() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create temporal constraints with minimum interval
    let time_constraints = TimeConstraints::new().with_min_interval(Duration::from_millis(100));

    let schema = ValidationSchema::new()
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    // Test intervals that are too small
    let data = json!({
        "timestamps": [1000, 1050, 1051, 1200]  // 1051-1050 = 1ms < 100ms
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(!result.is_valid(), "Small intervals should fail validation");

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("temporal.min_interval".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_temporal_constraint_interval_too_large() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create temporal constraints with maximum interval
    let time_constraints = TimeConstraints::new().with_max_interval(Duration::from_millis(1000));

    let schema = ValidationSchema::new()
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    // Test intervals that are too large
    let data = json!({
        "timestamps": [1000, 1500, 3000, 3100]  // 3000-1500 = 1500ms > 1000ms
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(!result.is_valid(), "Large intervals should fail validation");

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("temporal.max_interval".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_temporal_constraint_float_timestamps() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create temporal constraints
    let time_constraints = TimeConstraints::new().with_min_interval(Duration::from_secs(1));

    let schema = ValidationSchema::new()
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    // Test float timestamps (common for seconds with decimal)
    let data = json!({
        "timestamps": [1000.5, 2000.0, 3000.7, 5000.2]
    });

    let result = validator.validate(&data, &schema).unwrap();
    if !result.is_valid() {
        println!("Validation errors: {:?}", result.errors());
    }
    assert!(result.is_valid(), "Float timestamps should be accepted");
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_temporal_constraint_non_array() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create temporal constraints
    let time_constraints = TimeConstraints::new();

    let schema = ValidationSchema::new()
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    // Test non-array value
    let data = json!({
        "timestamps": "not an array"
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Non-array value should fail temporal validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.message.contains("require an array")));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_temporal_constraint_too_few_timestamps() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create temporal constraints
    let time_constraints = TimeConstraints::new().require_monotonic();

    let schema = ValidationSchema::new()
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    // Test with only one timestamp
    let data = json!({
        "timestamps": [1000]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Single timestamp should fail temporal validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.message.contains("at least 2 timestamps")));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_temporal_constraint_complex() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create complex temporal constraints
    let time_constraints = TimeConstraints::new()
        .with_min_interval(Duration::from_millis(100))
        .with_max_interval(Duration::from_secs(10))
        .require_monotonic()
        .disallow_duplicates();

    let schema = ValidationSchema::new()
        .require_field("timestamps", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("timestamps", Constraint::Temporal(time_constraints));

    // Test valid complex case
    let data = json!({
        "timestamps": [1000, 1200, 1500, 2000, 2800]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        result.is_valid(),
        "Valid complex timestamps should pass validation"
    );
}
