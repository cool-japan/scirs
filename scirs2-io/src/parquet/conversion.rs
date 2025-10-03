//! Conversion utilities between Arrow and ndarray

use crate::error::{IoError, Result};
use arrow::array::{
    ArrayRef, AsArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, PrimitiveArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scirs2_core::ndarray::{Array1, ArrayBase, Data as NdData, Dimension};
use std::sync::Arc;

/// Convert ndarray to Arrow RecordBatch
pub fn ndarray_to_arrow<S, D, T>(array: &ArrayBase<S, D>, column_name: &str) -> Result<RecordBatch>
where
    S: NdData<Elem = T>,
    D: Dimension,
    T: ToArrowArray + Clone,
{
    // Flatten the array
    let flat_data: Vec<T> = array.iter().cloned().collect();

    // Create Arrow array
    let arrow_array = T::to_arrow_array(&flat_data)?;

    // Create schema
    let data_type = T::arrow_data_type();
    let field = Field::new(column_name, data_type, false);
    let schema = Arc::new(Schema::new(vec![field]));

    // Create RecordBatch
    RecordBatch::try_new(schema, vec![arrow_array])
        .map_err(|e| IoError::ParquetError(format!("Failed to create RecordBatch: {}", e)))
}

/// Convert Arrow RecordBatch to ndarray (single column)
pub fn arrow_to_ndarray<T>(batch: &RecordBatch, column_index: usize) -> Result<Array1<T>>
where
    T: FromArrowArray,
{
    if column_index >= batch.num_columns() {
        return Err(IoError::ParquetError(format!(
            "Column index {} out of bounds (num_columns={})",
            column_index,
            batch.num_columns()
        )));
    }

    let column = batch.column(column_index);
    T::from_arrow_array(column)
}

/// Trait for types that can be converted to Arrow arrays
pub trait ToArrowArray: Sized {
    /// Convert a slice of data to an Arrow array
    fn to_arrow_array(data: &[Self]) -> Result<ArrayRef>;
    /// Get the Arrow data type for this type
    fn arrow_data_type() -> DataType;
}

/// Trait for types that can be extracted from Arrow arrays
pub trait FromArrowArray: Sized {
    /// Extract data from an Arrow array into an ndarray
    fn from_arrow_array(array: &ArrayRef) -> Result<Array1<Self>>;
}

// Macro to implement ToArrowArray and FromArrowArray for primitive types
macro_rules! impl_arrow_conversion {
    ($rust_type:ty, $arrow_type:ty, $data_type:expr, $array_type:ty) => {
        impl ToArrowArray for $rust_type {
            fn to_arrow_array(data: &[Self]) -> Result<ArrayRef> {
                Ok(Arc::new(<$array_type>::from(Vec::from(data))))
            }

            fn arrow_data_type() -> DataType {
                $data_type
            }
        }

        impl FromArrowArray for $rust_type {
            fn from_arrow_array(array: &ArrayRef) -> Result<Array1<Self>> {
                let typed_array = array.as_primitive_opt::<$arrow_type>().ok_or_else(|| {
                    IoError::ParquetError(format!(
                        "Expected {} array, got {:?}",
                        stringify!($rust_type),
                        array.data_type()
                    ))
                })?;

                let values: Vec<$rust_type> = typed_array.values().iter().copied().collect();
                Ok(Array1::from_vec(values))
            }
        }
    };
}

// Implement for all primitive numeric types
impl_arrow_conversion!(
    f64,
    arrow::datatypes::Float64Type,
    DataType::Float64,
    Float64Array
);
impl_arrow_conversion!(
    f32,
    arrow::datatypes::Float32Type,
    DataType::Float32,
    Float32Array
);
impl_arrow_conversion!(
    i64,
    arrow::datatypes::Int64Type,
    DataType::Int64,
    Int64Array
);
impl_arrow_conversion!(
    i32,
    arrow::datatypes::Int32Type,
    DataType::Int32,
    Int32Array
);
impl_arrow_conversion!(
    i16,
    arrow::datatypes::Int16Type,
    DataType::Int16,
    Int16Array
);
impl_arrow_conversion!(i8, arrow::datatypes::Int8Type, DataType::Int8, Int8Array);
impl_arrow_conversion!(
    u64,
    arrow::datatypes::UInt64Type,
    DataType::UInt64,
    UInt64Array
);
impl_arrow_conversion!(
    u32,
    arrow::datatypes::UInt32Type,
    DataType::UInt32,
    UInt32Array
);
impl_arrow_conversion!(
    u16,
    arrow::datatypes::UInt16Type,
    DataType::UInt16,
    UInt16Array
);
impl_arrow_conversion!(u8, arrow::datatypes::UInt8Type, DataType::UInt8, UInt8Array);

// Boolean implementation (not a primitive type in Arrow)
impl ToArrowArray for bool {
    fn to_arrow_array(data: &[Self]) -> Result<ArrayRef> {
        Ok(Arc::new(BooleanArray::from(Vec::from(data))))
    }

    fn arrow_data_type() -> DataType {
        DataType::Boolean
    }
}

impl FromArrowArray for bool {
    fn from_arrow_array(array: &ArrayRef) -> Result<Array1<Self>> {
        let bool_array = array.as_boolean_opt().ok_or_else(|| {
            IoError::ParquetError(format!(
                "Expected Boolean array, got {:?}",
                array.data_type()
            ))
        })?;

        let values: Vec<bool> = (0..bool_array.len()).map(|i| bool_array.value(i)).collect();
        Ok(Array1::from_vec(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_to_arrow_f64() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let batch = ndarray_to_arrow(&arr, "test").unwrap();

        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), "test");
    }

    #[test]
    fn test_arrow_to_ndarray_f64() {
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let batch = ndarray_to_arrow(&arr, "test").unwrap();
        let recovered: Array1<f64> = arrow_to_ndarray(&batch, 0).unwrap();

        assert_eq!(arr, recovered);
    }

    #[test]
    fn test_roundtrip_i32() {
        let arr = Array1::from_vec(vec![10i32, 20, 30, 40]);
        let batch = ndarray_to_arrow(&arr, "integers").unwrap();
        let recovered: Array1<i32> = arrow_to_ndarray(&batch, 0).unwrap();

        assert_eq!(arr, recovered);
    }

    #[test]
    fn test_roundtrip_bool() {
        let arr = Array1::from_vec(vec![true, false, true, false]);
        let batch = ndarray_to_arrow(&arr, "booleans").unwrap();
        let recovered: Array1<bool> = arrow_to_ndarray(&batch, 0).unwrap();

        assert_eq!(arr, recovered);
    }
}
