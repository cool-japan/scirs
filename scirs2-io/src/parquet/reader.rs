//! Parquet file reading

use crate::error::{IoError, Result};
use crate::parquet::conversion::arrow_to_ndarray;
use crate::parquet::schema::ParquetSchema;
use arrow::array::RecordBatchReader;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Container for Parquet data
#[derive(Debug)]
pub struct ParquetData {
    /// Column data
    columns: HashMap<String, ColumnData>,
    /// Schema
    schema: ParquetSchema,
    /// Number of rows
    num_rows: usize,
}

/// Column data holder
#[derive(Debug, Clone)]
enum ColumnData {
    Float64(Array1<f64>),
    Float32(Array1<f32>),
    Int64(Array1<i64>),
    Int32(Array1<i32>),
    Int16(Array1<i16>),
    Int8(Array1<i8>),
    UInt64(Array1<u64>),
    UInt32(Array1<u32>),
    UInt16(Array1<u16>),
    UInt8(Array1<u8>),
    Boolean(Array1<bool>),
}

impl ParquetData {
    /// Create new ParquetData from RecordBatches
    pub(crate) fn from_batches(batches: Vec<RecordBatch>) -> Result<Self> {
        if batches.is_empty() {
            return Err(IoError::ParquetError("No data in Parquet file".to_string()));
        }

        let schema = ParquetSchema::new(batches[0].schema());
        let mut columns: HashMap<String, ColumnData> = HashMap::new();
        let mut num_rows = 0;

        // Process each batch
        for batch in &batches {
            num_rows += batch.num_rows();

            for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                let column_name = field.name().clone();
                let column_array = batch.column(col_idx);

                let column_data = match field.data_type() {
                    arrow::datatypes::DataType::Float64 => {
                        ColumnData::Float64(arrow_to_ndarray::<f64>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::Float32 => {
                        ColumnData::Float32(arrow_to_ndarray::<f32>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::Int64 => {
                        ColumnData::Int64(arrow_to_ndarray::<i64>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::Int32 => {
                        ColumnData::Int32(arrow_to_ndarray::<i32>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::Int16 => {
                        ColumnData::Int16(arrow_to_ndarray::<i16>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::Int8 => {
                        ColumnData::Int8(arrow_to_ndarray::<i8>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::UInt64 => {
                        ColumnData::UInt64(arrow_to_ndarray::<u64>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::UInt32 => {
                        ColumnData::UInt32(arrow_to_ndarray::<u32>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::UInt16 => {
                        ColumnData::UInt16(arrow_to_ndarray::<u16>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::UInt8 => {
                        ColumnData::UInt8(arrow_to_ndarray::<u8>(batch, col_idx)?)
                    }
                    arrow::datatypes::DataType::Boolean => {
                        ColumnData::Boolean(arrow_to_ndarray::<bool>(batch, col_idx)?)
                    }
                    other => {
                        return Err(IoError::ParquetError(format!(
                            "Unsupported data type: {:?}",
                            other
                        )))
                    }
                };

                // If column already exists, concatenate data
                if let Some(existing) = columns.get_mut(&column_name) {
                    *existing = concatenate_column_data(existing, &column_data)?;
                } else {
                    columns.insert(column_name, column_data);
                }
            }
        }

        Ok(Self {
            columns,
            schema,
            num_rows,
        })
    }

    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    /// Get number of rows
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get number of columns
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get schema
    pub fn schema(&self) -> &ParquetSchema {
        &self.schema
    }

    /// Get column as f64 array
    pub fn get_column_f64(&self, name: &str) -> Result<&Array1<f64>> {
        match self.columns.get(name) {
            Some(ColumnData::Float64(arr)) => Ok(arr),
            Some(_) => Err(IoError::ParquetError(format!(
                "Column '{}' is not Float64",
                name
            ))),
            None => Err(IoError::ParquetError(format!(
                "Column '{}' not found",
                name
            ))),
        }
    }

    /// Get column as f32 array
    pub fn get_column_f32(&self, name: &str) -> Result<&Array1<f32>> {
        match self.columns.get(name) {
            Some(ColumnData::Float32(arr)) => Ok(arr),
            Some(_) => Err(IoError::ParquetError(format!(
                "Column '{}' is not Float32",
                name
            ))),
            None => Err(IoError::ParquetError(format!(
                "Column '{}' not found",
                name
            ))),
        }
    }

    /// Get column as i64 array
    pub fn get_column_i64(&self, name: &str) -> Result<&Array1<i64>> {
        match self.columns.get(name) {
            Some(ColumnData::Int64(arr)) => Ok(arr),
            Some(_) => Err(IoError::ParquetError(format!(
                "Column '{}' is not Int64",
                name
            ))),
            None => Err(IoError::ParquetError(format!(
                "Column '{}' not found",
                name
            ))),
        }
    }

    /// Get column as i32 array
    pub fn get_column_i32(&self, name: &str) -> Result<&Array1<i32>> {
        match self.columns.get(name) {
            Some(ColumnData::Int32(arr)) => Ok(arr),
            Some(_) => Err(IoError::ParquetError(format!(
                "Column '{}' is not Int32",
                name
            ))),
            None => Err(IoError::ParquetError(format!(
                "Column '{}' not found",
                name
            ))),
        }
    }
}

/// Concatenate two ColumnData instances
fn concatenate_column_data(a: &ColumnData, b: &ColumnData) -> Result<ColumnData> {
    match (a, b) {
        (ColumnData::Float64(a), ColumnData::Float64(b)) => {
            Ok(ColumnData::Float64(concatenate_arrays(a, b)))
        }
        (ColumnData::Float32(a), ColumnData::Float32(b)) => {
            Ok(ColumnData::Float32(concatenate_arrays(a, b)))
        }
        (ColumnData::Int64(a), ColumnData::Int64(b)) => {
            Ok(ColumnData::Int64(concatenate_arrays(a, b)))
        }
        (ColumnData::Int32(a), ColumnData::Int32(b)) => {
            Ok(ColumnData::Int32(concatenate_arrays(a, b)))
        }
        (ColumnData::Int16(a), ColumnData::Int16(b)) => {
            Ok(ColumnData::Int16(concatenate_arrays(a, b)))
        }
        (ColumnData::Int8(a), ColumnData::Int8(b)) => {
            Ok(ColumnData::Int8(concatenate_arrays(a, b)))
        }
        (ColumnData::UInt64(a), ColumnData::UInt64(b)) => {
            Ok(ColumnData::UInt64(concatenate_arrays(a, b)))
        }
        (ColumnData::UInt32(a), ColumnData::UInt32(b)) => {
            Ok(ColumnData::UInt32(concatenate_arrays(a, b)))
        }
        (ColumnData::UInt16(a), ColumnData::UInt16(b)) => {
            Ok(ColumnData::UInt16(concatenate_arrays(a, b)))
        }
        (ColumnData::UInt8(a), ColumnData::UInt8(b)) => {
            Ok(ColumnData::UInt8(concatenate_arrays(a, b)))
        }
        (ColumnData::Boolean(a), ColumnData::Boolean(b)) => {
            Ok(ColumnData::Boolean(concatenate_arrays(a, b)))
        }
        _ => Err(IoError::ParquetError(
            "Cannot concatenate columns of different types".to_string(),
        )),
    }
}

/// Concatenate two Array1 instances
fn concatenate_arrays<T: Clone>(a: &Array1<T>, b: &Array1<T>) -> Array1<T> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    result.extend_from_slice(a.as_slice().unwrap());
    result.extend_from_slice(b.as_slice().unwrap());
    Array1::from_vec(result)
}

/// Parquet file reader
pub struct ParquetReader {
    reader_builder: ParquetRecordBatchReaderBuilder<File>,
}

impl ParquetReader {
    /// Open a Parquet file for reading
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            IoError::FileError(format!(
                "Failed to open file '{}': {}",
                path.as_ref().display(),
                e
            ))
        })?;

        let reader_builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            IoError::ParquetError(format!("Failed to create Parquet reader: {}", e))
        })?;

        Ok(Self { reader_builder })
    }

    /// Read all data from the file
    pub fn read_all(self) -> Result<ParquetData> {
        let reader = self
            .reader_builder
            .build()
            .map_err(|e| IoError::ParquetError(format!("Failed to build reader: {}", e)))?;

        let mut batches = Vec::new();
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| IoError::ParquetError(format!("Failed to read batch: {}", e)))?;
            batches.push(batch);
        }

        ParquetData::from_batches(batches)
    }

    /// Read specific columns
    pub fn read_columns(mut self, column_names: &[&str]) -> Result<ParquetData> {
        let schema = self.reader_builder.schema();
        let mut column_indices = Vec::new();

        for name in column_names {
            let index = schema
                .fields()
                .iter()
                .position(|f| f.name() == name)
                .ok_or_else(|| IoError::ParquetError(format!("Column '{}' not found", name)))?;
            column_indices.push(index);
        }

        let projection =
            ProjectionMask::roots(self.reader_builder.parquet_schema(), column_indices);

        self.reader_builder = self.reader_builder.with_projection(projection);

        self.read_all()
    }
}

/// Read a Parquet file into ParquetData
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<ParquetData> {
    ParquetReader::from_path(path)?.read_all()
}

/// Read specific columns from a Parquet file
pub fn read_parquet_columns<P: AsRef<Path>>(path: P, columns: &[&str]) -> Result<ParquetData> {
    ParquetReader::from_path(path)?.read_columns(columns)
}

/// Iterator for chunked Parquet reading
///
/// Allows memory-efficient processing of large Parquet files by reading
/// data in configurable chunks (batches of rows).
pub struct ParquetChunkIterator {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    projection: Option<Vec<String>>,
}

impl ParquetChunkIterator {
    /// Create a new chunk iterator from a reader
    pub(crate) fn new(
        reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
        projection: Option<Vec<String>>,
    ) -> Self {
        Self { reader, projection }
    }

    /// Get the schema of the data being read
    pub fn schema(&self) -> ParquetSchema {
        ParquetSchema::new(self.reader.schema())
    }
}

impl Iterator for ParquetChunkIterator {
    type Item = Result<ParquetData>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.next() {
            Some(Ok(batch)) => {
                // Convert the batch to ParquetData
                match ParquetData::from_batches(vec![batch]) {
                    Ok(data) => Some(Ok(data)),
                    Err(e) => Some(Err(e)),
                }
            }
            Some(Err(e)) => Some(Err(IoError::ParquetError(format!(
                "Failed to read batch: {}",
                e
            )))),
            None => None,
        }
    }
}

/// Read a Parquet file in chunks for memory-efficient processing
///
/// This function returns an iterator that yields chunks of data, allowing
/// you to process large files without loading them entirely into memory.
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
/// * `batch_size` - Number of rows per chunk
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::parquet::read_parquet_chunked;
///
/// let chunks = read_parquet_chunked("large_file.parquet", 10000)?;
/// for chunk_result in chunks {
///     let chunk = chunk_result?;
///     println!("Processing chunk with {} rows", chunk.num_rows());
///     // Process chunk data...
/// }
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn read_parquet_chunked<P: AsRef<Path>>(
    path: P,
    batch_size: usize,
) -> Result<ParquetChunkIterator> {
    let file = File::open(path.as_ref()).map_err(|e| {
        IoError::FileError(format!(
            "Failed to open file '{}': {}",
            path.as_ref().display(),
            e
        ))
    })?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| IoError::ParquetError(format!("Failed to create Parquet reader: {}", e)))?;

    let reader = builder
        .with_batch_size(batch_size)
        .build()
        .map_err(|e| IoError::ParquetError(format!("Failed to build reader: {}", e)))?;

    Ok(ParquetChunkIterator::new(reader, None))
}

/// Read specific columns from a Parquet file in chunks
///
/// Like `read_parquet_chunked`, but only reads specified columns.
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
/// * `columns` - Column names to read
/// * `batch_size` - Number of rows per chunk
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::parquet::read_parquet_chunked_columns;
///
/// let chunks = read_parquet_chunked_columns(
///     "data.parquet",
///     &["temperature", "pressure"],
///     5000
/// )?;
///
/// for chunk_result in chunks {
///     let chunk = chunk_result?;
///     // Process only temperature and pressure columns
/// }
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn read_parquet_chunked_columns<P: AsRef<Path>>(
    path: P,
    columns: &[&str],
    batch_size: usize,
) -> Result<ParquetChunkIterator> {
    let file = File::open(path.as_ref()).map_err(|e| {
        IoError::FileError(format!(
            "Failed to open file '{}': {}",
            path.as_ref().display(),
            e
        ))
    })?;

    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| IoError::ParquetError(format!("Failed to create Parquet reader: {}", e)))?;

    // Set up column projection
    let schema = builder.schema();
    let mut column_indices = Vec::new();

    for name in columns {
        let index = schema
            .fields()
            .iter()
            .position(|f| f.name() == name)
            .ok_or_else(|| IoError::ParquetError(format!("Column '{}' not found", name)))?;
        column_indices.push(index);
    }

    let projection = ProjectionMask::roots(builder.parquet_schema(), column_indices);
    builder = builder.with_projection(projection);

    let reader = builder
        .with_batch_size(batch_size)
        .build()
        .map_err(|e| IoError::ParquetError(format!("Failed to build reader: {}", e)))?;

    let column_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
    Ok(ParquetChunkIterator::new(reader, Some(column_names)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concatenate_arrays() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);
        let c = concatenate_arrays(&a, &b);

        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1.0);
        assert_eq!(c[3], 4.0);
    }
}
