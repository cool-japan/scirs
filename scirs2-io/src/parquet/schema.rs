//! Parquet schema handling and inference

use crate::error::{IoError, Result};
use arrow::datatypes::{DataType, Field, Schema};
use scirs2_core::ndarray::{ArrayBase, Data as NdData, Dimension};
use std::sync::Arc;

/// Parquet schema wrapper
#[derive(Debug, Clone)]
pub struct ParquetSchema {
    /// Arrow schema
    pub(crate) arrow_schema: Arc<Schema>,
}

impl ParquetSchema {
    /// Create a new ParquetSchema from Arrow schema
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            arrow_schema: schema,
        }
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        self.arrow_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    /// Get number of columns
    pub fn num_columns(&self) -> usize {
        self.arrow_schema.fields().len()
    }

    /// Get field by name
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.arrow_schema.field_with_name(name).ok()
    }

    /// Get Arrow schema reference
    pub fn arrow_schema(&self) -> &Arc<Schema> {
        &self.arrow_schema
    }
}

/// Infer Arrow schema from ndarray type
///
/// This creates a simple schema with a single column named "value"
pub fn infer_arrow_schema<S, D, T>(array: &ArrayBase<S, D>) -> Result<Arc<Schema>>
where
    S: NdData<Elem = T>,
    D: Dimension,
    T: InferArrowType,
{
    let data_type = T::arrow_data_type();
    let field = Field::new("value", data_type, false);
    Ok(Arc::new(Schema::new(vec![field])))
}

/// Trait for types that can be converted to Arrow DataType
pub trait InferArrowType {
    /// Get the Arrow data type for this Rust type
    fn arrow_data_type() -> DataType;
}

impl InferArrowType for f64 {
    fn arrow_data_type() -> DataType {
        DataType::Float64
    }
}

impl InferArrowType for f32 {
    fn arrow_data_type() -> DataType {
        DataType::Float32
    }
}

impl InferArrowType for i64 {
    fn arrow_data_type() -> DataType {
        DataType::Int64
    }
}

impl InferArrowType for i32 {
    fn arrow_data_type() -> DataType {
        DataType::Int32
    }
}

impl InferArrowType for i16 {
    fn arrow_data_type() -> DataType {
        DataType::Int16
    }
}

impl InferArrowType for i8 {
    fn arrow_data_type() -> DataType {
        DataType::Int8
    }
}

impl InferArrowType for u64 {
    fn arrow_data_type() -> DataType {
        DataType::UInt64
    }
}

impl InferArrowType for u32 {
    fn arrow_data_type() -> DataType {
        DataType::UInt32
    }
}

impl InferArrowType for u16 {
    fn arrow_data_type() -> DataType {
        DataType::UInt16
    }
}

impl InferArrowType for u8 {
    fn arrow_data_type() -> DataType {
        DataType::UInt8
    }
}

impl InferArrowType for bool {
    fn arrow_data_type() -> DataType {
        DataType::Boolean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_infer_schema_f64() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let schema = infer_arrow_schema(&arr).unwrap();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "value");
        assert_eq!(*schema.field(0).data_type(), DataType::Float64);
    }

    #[test]
    fn test_infer_schema_i32() {
        let arr = Array1::from_vec(vec![1i32, 2, 3]);
        let schema = infer_arrow_schema(&arr).unwrap();
        assert_eq!(*schema.field(0).data_type(), DataType::Int32);
    }

    #[test]
    fn test_parquet_schema() {
        let field = Field::new("test_column", DataType::Float64, false);
        let arrow_schema = Arc::new(Schema::new(vec![field]));
        let schema = ParquetSchema::new(arrow_schema);

        assert_eq!(schema.num_columns(), 1);
        assert_eq!(schema.column_names(), vec!["test_column"]);
        assert!(schema.field("test_column").is_some());
        assert!(schema.field("nonexistent").is_none());
    }
}
