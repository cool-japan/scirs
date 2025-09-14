//! Schema definition and types for data validation
//!
//! This module provides schema structures for defining the expected structure
//! and constraints of data to be validated.

use super::constraints::Constraint;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Data types supported by the validation system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// Boolean value
    Boolean,
    /// Integer number
    Integer,
    /// Floating point number
    Float32,
    /// Double precision floating point
    Float64,
    /// UTF-8 string
    String,
    /// Array of elements
    Array(Box<DataType>),
    /// Object with fields
    Object,
    /// Null value
    Null,
    /// Binary data
    Binary,
    /// Date/time value
    DateTime,
    /// UUID value
    Uuid,
    /// Geographic coordinate
    GeoCoordinate,
    /// Complex number
    Complex,
    /// Matrix (2D array)
    Matrix(Box<DataType>),
    /// Tensor (N-dimensional array)
    Tensor {
        element_type: Box<DataType>,
        dimensions: Option<Vec<usize>>,
    },
    /// Time series data
    TimeSeries(Box<DataType>),
    /// Sparse matrix
    SparseMatrix {
        element_type: Box<DataType>,
        format: super::constraints::SparseFormat,
    },
}

/// Field definition in a validation schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field data type
    pub datatype: DataType,
    /// Whether field is required
    pub required: bool,
    /// Constraints applied to this field
    pub constraints: Vec<Constraint>,
    /// Field description
    pub description: Option<String>,
    /// Default value if not provided
    pub defaultvalue: Option<String>,
    /// Validation rule references
    pub validation_rules: Vec<String>,
}

impl FieldDefinition {
    /// Create a new field definition
    pub fn new(datatype: DataType) -> Self {
        Self {
            datatype,
            required: false,
            constraints: Vec::new(),
            description: None,
            defaultvalue: None,
            validation_rules: Vec::new(),
        }
    }

    /// Mark field as required
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Set default value
    pub fn with_default(mut self, default: &str) -> Self {
        self.defaultvalue = Some(default.to_string());
        self
    }

    /// Add validation rule
    pub fn with_validation_rule(mut self, rule: &str) -> Self {
        self.validation_rules.push(rule.to_string());
        self
    }
}

/// Validation schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSchema {
    /// Schema name
    pub name: String,
    /// Schema version
    pub version: String,
    /// Field definitions
    pub fields: HashMap<String, FieldDefinition>,
    /// Global constraints
    pub global_constraints: Vec<Constraint>,
    /// Schema metadata
    pub metadata: HashMap<String, String>,
    /// Allow additional fields not in schema
    pub allow_additional_fields: bool,
}

impl ValidationSchema {
    /// Create a new validation schema
    pub fn new() -> Self {
        Self {
            name: "unnamed".to_string(),
            version: "1.0.0".to_string(),
            fields: HashMap::new(),
            global_constraints: Vec::new(),
            metadata: HashMap::new(),
            allow_additional_fields: false,
        }
    }

    /// Set schema name
    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set schema version
    pub fn version(mut self, version: &str) -> Self {
        self.version = version.to_string();
        self
    }

    /// Add a required field
    pub fn require_field(mut self, name: &str, datatype: DataType) -> Self {
        let field = FieldDefinition::new(datatype).required();
        self.fields.insert(name.to_string(), field);
        self
    }

    /// Add an optional field
    pub fn optional_field(mut self, name: &str, datatype: DataType) -> Self {
        let field = FieldDefinition::new(datatype);
        self.fields.insert(name.to_string(), field);
        self
    }

    /// Add a field with custom definition
    pub fn add_field(mut self, name: &str, field: FieldDefinition) -> Self {
        self.fields.insert(name.to_string(), field);
        self
    }

    /// Add a constraint to a field
    pub fn add_constraint(mut self, fieldname: &str, constraint: Constraint) -> Self {
        if let Some(field) = self.fields.get_mut(fieldname) {
            field.constraints.push(constraint);
        }
        self
    }

    /// Add a global constraint
    pub fn add_global_constraint(mut self, constraint: Constraint) -> Self {
        self.global_constraints.push(constraint);
        self
    }

    /// Allow additional fields
    pub fn allow_additional(mut self) -> Self {
        self.allow_additional_fields = true;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get field definition
    pub fn get_field(&self, name: &str) -> Option<&FieldDefinition> {
        self.fields.get(name)
    }

    /// Check if field is required
    pub fn is_field_required(&self, name: &str) -> bool {
        self.fields.get(name).is_some_and(|f| f.required)
    }

    /// Get all required field names
    pub fn get_required_fields(&self) -> Vec<&String> {
        self.fields
            .iter()
            .filter(|(_, field)| field.required)
            .map(|(name, _)| name)
            .collect()
    }

    /// Validate schema consistency
    pub fn validate_schema(&self) -> Result<(), String> {
        // Check for empty schema name
        if self.name.is_empty() {
            return Err("Schema name cannot be empty".to_string());
        }

        // Check for empty version
        if self.version.is_empty() {
            return Err("Schema version cannot be empty".to_string());
        }

        // Check for circular references in array/matrix types
        for (fieldname, field) in &self.fields {
            self.check_circular_references(&field.datatype, fieldname)?;
        }

        Ok(())
    }

    /// Check for circular references in data types
    #[allow(clippy::only_used_in_recursion)]
    fn check_circular_references(
        &self,
        datatype: &DataType,
        fieldname: &str,
    ) -> Result<(), String> {
        match datatype {
            DataType::Array(inner) => self.check_circular_references(inner, fieldname),
            DataType::Matrix(inner) => self.check_circular_references(inner, fieldname),
            DataType::Tensor { element_type, .. } => {
                self.check_circular_references(element_type, fieldname)
            }
            DataType::TimeSeries(inner) => self.check_circular_references(inner, fieldname),
            DataType::SparseMatrix { element_type, .. } => {
                self.check_circular_references(element_type, fieldname)
            }
            _ => Ok(()),
        }
    }
}

impl Default for ValidationSchema {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::constraints::{Constraint, SparseFormat};
    use super::*;

    #[test]
    fn test_field_definition() {
        let field = FieldDefinition::new(DataType::String)
            .required()
            .with_description("Test field")
            .with_default("defaultvalue")
            .with_constraint(Constraint::NotNull)
            .with_validation_rule("custom_rule");

        assert_eq!(field.datatype, DataType::String);
        assert!(field.required);
        assert_eq!(field.description, Some("Test field".to_string()));
        assert_eq!(field.defaultvalue, Some("defaultvalue".to_string()));
        assert_eq!(field.constraints.len(), 1);
        assert_eq!(field.validation_rules.len(), 1);
    }

    #[test]
    fn test_validation_schema() {
        let schema = ValidationSchema::new()
            .name("test_schema")
            .version("1.0.0")
            .require_field("name", DataType::String)
            .optional_field("age", DataType::Integer)
            .add_constraint(
                "age",
                Constraint::Range {
                    min: 0.0,
                    max: 150.0,
                },
            )
            .allow_additional()
            .with_metadata("author", "test_author");

        assert_eq!(schema.name, "test_schema");
        assert_eq!(schema.version, "1.0.0");
        assert_eq!(schema.fields.len(), 2);
        assert!(schema.allow_additional_fields);
        assert_eq!(
            schema.metadata.get("author"),
            Some(&"test_author".to_string())
        );

        // Test field access
        assert!(schema.is_field_required("name"));
        assert!(!schema.is_field_required("age"));

        let required_fields = schema.get_required_fields();
        assert_eq!(required_fields.len(), 1);
        assert!(required_fields.contains(&&"name".to_string()));
    }

    #[test]
    fn test_complex_datatypes() {
        // Test array type
        let array_type = DataType::Array(Box::new(DataType::Float64));
        match array_type {
            DataType::Array(inner) => assert_eq!(*inner, DataType::Float64),
            _ => panic!("Expected Array type"),
        }

        // Test tensor type
        let tensor_type = DataType::Tensor {
            element_type: Box::new(DataType::Float32),
            dimensions: Some(vec![10, 20, 30]),
        };
        match tensor_type {
            DataType::Tensor {
                element_type,
                dimensions,
            } => {
                assert_eq!(*element_type, DataType::Float32);
                assert_eq!(dimensions, Some(vec![10, 20, 30]));
            }
            _ => panic!("Expected Tensor type"),
        }

        // Test sparse matrix type
        let sparse_type = DataType::SparseMatrix {
            element_type: Box::new(DataType::Float64),
            format: SparseFormat::CSR,
        };
        match sparse_type {
            DataType::SparseMatrix {
                element_type,
                format,
            } => {
                assert_eq!(*element_type, DataType::Float64);
                assert_eq!(format, SparseFormat::CSR);
            }
            _ => panic!("Expected SparseMatrix type"),
        }
    }

    #[test]
    fn test_schema_validation() {
        let valid_schema = ValidationSchema::new()
            .name("valid_schema")
            .version("1.0.0");

        assert!(valid_schema.validate_schema().is_ok());

        let invalid_schema = ValidationSchema::new()
            .name("")  // Empty name should be invalid
            .version("1.0.0");

        assert!(invalid_schema.validate_schema().is_err());
    }

    #[test]
    fn test_datatype_equality() {
        assert_eq!(DataType::String, DataType::String);
        assert_eq!(DataType::Float64, DataType::Float64);
        assert_ne!(DataType::Float32, DataType::Float64);

        let array1 = DataType::Array(Box::new(DataType::Integer));
        let array2 = DataType::Array(Box::new(DataType::Integer));
        let array3 = DataType::Array(Box::new(DataType::Float64));

        assert_eq!(array1, array2);
        assert_ne!(array1, array3);
    }
}
