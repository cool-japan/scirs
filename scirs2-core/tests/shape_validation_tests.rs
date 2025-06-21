//! Tests for shape constraint validation

#[cfg(all(feature = "data_validation", feature = "serde"))]
use scirs2_core::validation::data::{
    Constraint, DataType, ShapeConstraints, ValidationConfig, ValidationSchema, Validator,
};

#[cfg(all(feature = "data_validation", feature = "serde"))]
use serde_json::json;

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_1d_array() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints for 1D array
    let shape_constraints = ShapeConstraints::new().with_dimensions(vec![Some(5)]);

    let schema = ValidationSchema::new()
        .require_field("vector", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("vector", Constraint::Shape(shape_constraints));

    // Test valid 1D array
    let data = json!({
        "vector": [1.0, 2.0, 3.0, 4.0, 5.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        result.is_valid(),
        "Valid 1D array shape should pass validation"
    );
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_1d_wrong_size() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints for 1D array
    let shape_constraints = ShapeConstraints::new().with_dimensions(vec![Some(5)]);

    let schema = ValidationSchema::new()
        .require_field("vector", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("vector", Constraint::Shape(shape_constraints));

    // Test wrong size
    let data = json!({
        "vector": [1.0, 2.0, 3.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Wrong array size should fail validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("shape.dimension[0]".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_2d_array() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints for 2D array (3x4)
    let shape_constraints = ShapeConstraints::new().with_dimensions(vec![Some(3), Some(4)]);

    let schema = ValidationSchema::new()
        .require_field(
            "matrix",
            DataType::Array(Box::new(DataType::Array(Box::new(DataType::Float64)))),
        )
        .add_constraint("matrix", Constraint::Shape(shape_constraints));

    // Test valid 2D array
    let data = json!({
        "matrix": [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        result.is_valid(),
        "Valid 2D array shape should pass validation"
    );
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_jagged_array() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints for 2D array
    let shape_constraints = ShapeConstraints::new().with_dimensions(vec![Some(3), Some(4)]);

    let schema = ValidationSchema::new()
        .require_field(
            "matrix",
            DataType::Array(Box::new(DataType::Array(Box::new(DataType::Float64)))),
        )
        .add_constraint("matrix", Constraint::Shape(shape_constraints));

    // Test jagged array (rows have different lengths)
    let data = json!({
        "matrix": [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0]
        ]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(!result.is_valid(), "Jagged array should fail validation");

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.message.contains("Jagged arrays are not supported")));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_square_matrix() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints for square matrix
    let shape_constraints = ShapeConstraints::new()
        .with_dimensions(vec![Some(3), Some(3)])
        .require_square();

    let schema = ValidationSchema::new()
        .require_field(
            "matrix",
            DataType::Array(Box::new(DataType::Array(Box::new(DataType::Float64)))),
        )
        .add_constraint("matrix", Constraint::Shape(shape_constraints));

    // Test valid square matrix
    let data = json!({
        "matrix": [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        result.is_valid(),
        "Valid square matrix should pass validation"
    );
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_non_square_fails() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints requiring square matrix
    let shape_constraints = ShapeConstraints::new().require_square();

    let schema = ValidationSchema::new()
        .require_field(
            "matrix",
            DataType::Array(Box::new(DataType::Array(Box::new(DataType::Float64)))),
        )
        .add_constraint("matrix", Constraint::Shape(shape_constraints));

    // Test non-square matrix
    let data = json!({
        "matrix": [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Non-square matrix should fail when square is required"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("shape.square".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_element_count() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints with element count limits
    let shape_constraints = ShapeConstraints::new().with_element_range(10, 20);

    let schema = ValidationSchema::new()
        .require_field("array", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("array", Constraint::Shape(shape_constraints));

    // Test valid element count (15 elements)
    let data = json!({
        "array": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        result.is_valid(),
        "Valid element count should pass validation"
    );
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_too_few_elements() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints with minimum element count
    let shape_constraints = ShapeConstraints::new().with_element_range(10, 20);

    let schema = ValidationSchema::new()
        .require_field("array", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("array", Constraint::Shape(shape_constraints));

    // Test too few elements
    let data = json!({
        "array": [1.0, 2.0, 3.0, 4.0, 5.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Too few elements should fail validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("shape.min_elements".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_too_many_elements() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints with maximum element count
    let shape_constraints = ShapeConstraints::new().with_element_range(5, 10);

    let schema = ValidationSchema::new()
        .require_field("array", DataType::Array(Box::new(DataType::Float64)))
        .add_constraint("array", Constraint::Shape(shape_constraints));

    // Test too many elements
    let data = json!({
        "array": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        !result.is_valid(),
        "Too many elements should fail validation"
    );

    let errors = result.errors();
    assert!(errors
        .iter()
        .any(|e| e.constraint == Some("shape.max_elements".to_string())));
}

#[cfg(all(feature = "data_validation", feature = "serde"))]
#[test]
fn test_shape_constraint_flexible_dimensions() {
    let config = ValidationConfig::default();
    let validator = Validator::new(config).unwrap();

    // Create shape constraints with flexible dimensions (None means any size)
    let shape_constraints = ShapeConstraints::new().with_dimensions(vec![None, Some(3)]);

    let schema = ValidationSchema::new()
        .require_field(
            "matrix",
            DataType::Array(Box::new(DataType::Array(Box::new(DataType::Float64)))),
        )
        .add_constraint("matrix", Constraint::Shape(shape_constraints));

    // Test matrix with any number of rows but exactly 3 columns
    let data = json!({
        "matrix": [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ]
    });

    let result = validator.validate(&data, &schema).unwrap();
    assert!(
        result.is_valid(),
        "Matrix with flexible row count should pass validation"
    );
}
