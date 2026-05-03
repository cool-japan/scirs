//! Native Parquet dataset reader
//!
//! Provides `ParquetDataset` — reads a Parquet file into a typed dataset with
//! per-column data accessible as `ColumnData` variants. Requires the
//! `parquet_io` feature which activates the `parquet` and `arrow` crates.
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "parquet_io")]
//! use scirs2_datasets::parquet_reader::ParquetDataset;
//!
//! # #[cfg(feature = "parquet_io")]
//! # fn example() -> Result<(), scirs2_datasets::error::DatasetsError> {
//! let dataset = ParquetDataset::from_file("data.parquet")?;
//! println!("Rows: {}, Cols: {}", dataset.n_rows(), dataset.n_cols());
//! for name in dataset.column_names() {
//!     println!("  column: {}", name);
//! }
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "parquet_io")]
use crate::error::{DatasetsError, Result};
#[cfg(feature = "parquet_io")]
use arrow::array::RecordBatchReader;
#[cfg(feature = "parquet_io")]
use indexmap::IndexMap;
#[cfg(feature = "parquet_io")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
#[cfg(feature = "parquet_io")]
use scirs2_core::ndarray::Array2;
#[cfg(feature = "parquet_io")]
use std::fs::File;
#[cfg(feature = "parquet_io")]
use std::path::Path;

/// A single column's data from a Parquet file.
///
/// Nullable values are wrapped in `Option`; `None` represents a null.
#[cfg(feature = "parquet_io")]
#[derive(Debug, Clone)]
pub enum ColumnData {
    /// 32-bit signed integer column
    Int32(Vec<Option<i32>>),
    /// 64-bit signed integer column
    Int64(Vec<Option<i64>>),
    /// 32-bit IEEE-754 float column
    Float32(Vec<Option<f32>>),
    /// 64-bit IEEE-754 float column
    Float64(Vec<Option<f64>>),
    /// Boolean column
    Boolean(Vec<Option<bool>>),
    /// UTF-8 string column
    Utf8(Vec<Option<String>>),
}

#[cfg(feature = "parquet_io")]
impl ColumnData {
    /// Number of rows (including nulls) in this column.
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int32(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float32(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::Boolean(v) => v.len(),
            ColumnData::Utf8(v) => v.len(),
        }
    }

    /// Returns `true` if this column contains no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if this column holds numeric data (Int32/Int64/Float32/Float64).
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            ColumnData::Int32(_)
                | ColumnData::Int64(_)
                | ColumnData::Float32(_)
                | ColumnData::Float64(_)
        )
    }

    /// Cast each non-null value to f64; nulls become `f64::NAN`.
    pub fn to_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            ColumnData::Int32(v) => {
                Some(v.iter().map(|x| x.map_or(f64::NAN, |n| n as f64)).collect())
            }
            ColumnData::Int64(v) => {
                Some(v.iter().map(|x| x.map_or(f64::NAN, |n| n as f64)).collect())
            }
            ColumnData::Float32(v) => {
                Some(v.iter().map(|x| x.map_or(f64::NAN, |n| n as f64)).collect())
            }
            ColumnData::Float64(v) => Some(v.iter().map(|x| x.unwrap_or(f64::NAN)).collect()),
            ColumnData::Boolean(_) | ColumnData::Utf8(_) => None,
        }
    }
}

/// A dataset loaded from a Parquet file.
///
/// Columns are stored in an `IndexMap` so insertion order (i.e., file column
/// order) is preserved.  Column names are case-sensitive.
#[cfg(feature = "parquet_io")]
pub struct ParquetDataset {
    /// Per-column data indexed by column name
    pub columns: IndexMap<String, ColumnData>,
    /// Total number of rows across all columns
    pub n_rows: usize,
}

#[cfg(feature = "parquet_io")]
impl ParquetDataset {
    /// Read a Parquet file from the filesystem.
    ///
    /// # Errors
    ///
    /// Returns `DatasetsError` if the file cannot be opened, is not valid
    /// Parquet, or contains column types that are not supported (types other
    /// than Int32/Int64/Float32/Float64/Boolean/Utf8 are skipped with a
    /// warning rather than causing a hard error).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(DatasetsError::IoError)?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| DatasetsError::InvalidFormat(format!("Parquet open error: {e}")))?;

        let reader = builder.build().map_err(|e| {
            DatasetsError::InvalidFormat(format!("Parquet reader build error: {e}"))
        })?;

        Self::from_record_batch_reader(reader)
    }

    /// Internal constructor — consumes a `RecordBatchReader` and accumulates
    /// column data across all batches.
    fn from_record_batch_reader(mut reader: impl RecordBatchReader) -> Result<Self> {
        use arrow::array::{
            Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
        };
        use arrow::datatypes::DataType as ArrowDataType;

        let schema = reader.schema();
        let field_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

        // Pre-allocate per-column accumulators as Option<Vec<Option<...>>>
        // We start each accumulator as None; on the first batch we decide the
        // ColumnData variant. Columns with unsupported types get None and are
        // skipped.
        let num_cols = field_names.len();
        let mut accumulators: Vec<Option<ColumnAccumulator>> =
            (0..num_cols).map(|_| None).collect();
        let mut total_rows: usize = 0;

        for batch_result in reader.by_ref() {
            let batch = batch_result.map_err(|e| {
                DatasetsError::InvalidFormat(format!("Parquet read batch error: {e}"))
            })?;

            total_rows = total_rows.saturating_add(batch.num_rows());

            for (col_idx, field) in batch.schema().fields().iter().enumerate() {
                let array = batch.column(col_idx);

                let col_acc =
                    accumulators[col_idx].get_or_insert_with(|| match field.data_type() {
                        ArrowDataType::Int32 => ColumnAccumulator::Int32(Vec::new()),
                        ArrowDataType::Int64 => ColumnAccumulator::Int64(Vec::new()),
                        ArrowDataType::Float32 => ColumnAccumulator::Float32(Vec::new()),
                        ArrowDataType::Float64 => ColumnAccumulator::Float64(Vec::new()),
                        ArrowDataType::Boolean => ColumnAccumulator::Boolean(Vec::new()),
                        ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 => {
                            ColumnAccumulator::Utf8(Vec::new())
                        }
                        _ => ColumnAccumulator::Unsupported,
                    });

                match col_acc {
                    ColumnAccumulator::Int32(buf) => {
                        let typed =
                            array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                                DatasetsError::InvalidFormat(format!(
                                    "Column '{}' type mismatch",
                                    field.name()
                                ))
                            })?;
                        for i in 0..typed.len() {
                            buf.push(if typed.is_null(i) {
                                None
                            } else {
                                Some(typed.value(i))
                            });
                        }
                    }
                    ColumnAccumulator::Int64(buf) => {
                        let typed =
                            array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                                DatasetsError::InvalidFormat(format!(
                                    "Column '{}' type mismatch",
                                    field.name()
                                ))
                            })?;
                        for i in 0..typed.len() {
                            buf.push(if typed.is_null(i) {
                                None
                            } else {
                                Some(typed.value(i))
                            });
                        }
                    }
                    ColumnAccumulator::Float32(buf) => {
                        let typed =
                            array
                                .as_any()
                                .downcast_ref::<Float32Array>()
                                .ok_or_else(|| {
                                    DatasetsError::InvalidFormat(format!(
                                        "Column '{}' type mismatch",
                                        field.name()
                                    ))
                                })?;
                        for i in 0..typed.len() {
                            buf.push(if typed.is_null(i) {
                                None
                            } else {
                                Some(typed.value(i))
                            });
                        }
                    }
                    ColumnAccumulator::Float64(buf) => {
                        let typed =
                            array
                                .as_any()
                                .downcast_ref::<Float64Array>()
                                .ok_or_else(|| {
                                    DatasetsError::InvalidFormat(format!(
                                        "Column '{}' type mismatch",
                                        field.name()
                                    ))
                                })?;
                        for i in 0..typed.len() {
                            buf.push(if typed.is_null(i) {
                                None
                            } else {
                                Some(typed.value(i))
                            });
                        }
                    }
                    ColumnAccumulator::Boolean(buf) => {
                        let typed =
                            array
                                .as_any()
                                .downcast_ref::<BooleanArray>()
                                .ok_or_else(|| {
                                    DatasetsError::InvalidFormat(format!(
                                        "Column '{}' type mismatch",
                                        field.name()
                                    ))
                                })?;
                        for i in 0..typed.len() {
                            buf.push(if typed.is_null(i) {
                                None
                            } else {
                                Some(typed.value(i))
                            });
                        }
                    }
                    ColumnAccumulator::Utf8(buf) => {
                        let typed =
                            array
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .ok_or_else(|| {
                                    DatasetsError::InvalidFormat(format!(
                                        "Column '{}' type mismatch",
                                        field.name()
                                    ))
                                })?;
                        for i in 0..typed.len() {
                            buf.push(if typed.is_null(i) {
                                None
                            } else {
                                Some(typed.value(i).to_owned())
                            });
                        }
                    }
                    ColumnAccumulator::Unsupported => {
                        // Skip silently — column will be absent from dataset
                    }
                }
            }
        }

        // Build final IndexMap
        let mut columns: IndexMap<String, ColumnData> = IndexMap::with_capacity(num_cols);
        for (col_idx, name) in field_names.iter().enumerate() {
            match accumulators[col_idx].take() {
                Some(ColumnAccumulator::Int32(v)) => {
                    columns.insert(name.clone(), ColumnData::Int32(v));
                }
                Some(ColumnAccumulator::Int64(v)) => {
                    columns.insert(name.clone(), ColumnData::Int64(v));
                }
                Some(ColumnAccumulator::Float32(v)) => {
                    columns.insert(name.clone(), ColumnData::Float32(v));
                }
                Some(ColumnAccumulator::Float64(v)) => {
                    columns.insert(name.clone(), ColumnData::Float64(v));
                }
                Some(ColumnAccumulator::Boolean(v)) => {
                    columns.insert(name.clone(), ColumnData::Boolean(v));
                }
                Some(ColumnAccumulator::Utf8(v)) => {
                    columns.insert(name.clone(), ColumnData::Utf8(v));
                }
                Some(ColumnAccumulator::Unsupported) | None => {
                    // Omit unsupported columns
                }
            }
        }

        Ok(Self {
            columns,
            n_rows: total_rows,
        })
    }

    /// Look up a column by name.
    pub fn column(&self, name: &str) -> Option<&ColumnData> {
        self.columns.get(name)
    }

    /// Return column names in file order.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of supported columns in the dataset.
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// Convert all numeric columns to a dense `Array2<f64>` (column-major).
    ///
    /// String and Boolean columns are skipped. Null values become `f64::NAN`.
    /// Column order matches `column_names()`.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no numeric columns, or if column lengths
    /// are inconsistent.
    pub fn to_float_matrix(&self) -> Result<Array2<f64>> {
        let numeric_cols: Vec<(&str, Vec<f64>)> = self
            .columns
            .iter()
            .filter_map(|(name, col)| col.to_f64_vec().map(|v| (name.as_str(), v)))
            .collect();

        if numeric_cols.is_empty() {
            return Err(DatasetsError::InvalidFormat(
                "No numeric columns found in ParquetDataset".to_string(),
            ));
        }

        let n_rows = self.n_rows;
        let n_cols = numeric_cols.len();

        // Verify all numeric columns have the expected length
        for (name, col) in &numeric_cols {
            if col.len() != n_rows {
                return Err(DatasetsError::InvalidFormat(format!(
                    "Column '{}' has {} rows, expected {}",
                    name,
                    col.len(),
                    n_rows
                )));
            }
        }

        let mut matrix = Array2::<f64>::zeros((n_rows, n_cols));
        for (j, (_, col)) in numeric_cols.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                matrix[[i, j]] = v;
            }
        }

        Ok(matrix)
    }
}

/// Internal accumulator used while reading batches.
#[cfg(feature = "parquet_io")]
#[derive(Debug)]
enum ColumnAccumulator {
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Float32(Vec<Option<f32>>),
    Float64(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
    Utf8(Vec<Option<String>>),
    Unsupported,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "parquet_io")]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::io::Write;
    use std::sync::Arc;

    /// Write a minimal Parquet file to a temp path and return the path.
    fn write_test_parquet(
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("test.parquet");
        let file = std::fs::File::create(&path).expect("create file");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("create parquet writer");
        for batch in batches {
            writer.write(&batch).expect("write batch");
        }
        writer.close().expect("close writer");
        (dir, path)
    }

    #[test]
    fn test_parquet_read_numeric_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", ArrowDataType::Int32, false),
            Field::new("y", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Float64Array::from(vec![1.1, 2.2, 3.3])),
            ],
        )
        .expect("record batch");

        let (_dir, path) = write_test_parquet(schema, vec![batch]);
        let ds = ParquetDataset::from_file(&path).expect("from_file");

        assert_eq!(ds.n_rows(), 3);
        assert_eq!(ds.n_cols(), 2);
        assert!(ds.column("x").is_some());
        assert!(ds.column("y").is_some());

        if let Some(ColumnData::Int32(vals)) = ds.column("x") {
            assert_eq!(vals[0], Some(1));
            assert_eq!(vals[2], Some(3));
        } else {
            panic!("Expected Int32 column");
        }

        if let Some(ColumnData::Float64(vals)) = ds.column("y") {
            assert!((vals[1].expect("non-null") - 2.2).abs() < 1e-10);
        } else {
            panic!("Expected Float64 column");
        }
    }

    #[test]
    fn test_parquet_read_string_column() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "name",
            ArrowDataType::Utf8,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![
                Some("alice"),
                None,
                Some("bob"),
            ]))],
        )
        .expect("record batch");

        let (_dir, path) = write_test_parquet(schema, vec![batch]);
        let ds = ParquetDataset::from_file(&path).expect("from_file");

        assert_eq!(ds.n_rows(), 3);
        if let Some(ColumnData::Utf8(vals)) = ds.column("name") {
            assert_eq!(vals[0], Some("alice".to_owned()));
            assert_eq!(vals[1], None);
            assert_eq!(vals[2], Some("bob".to_owned()));
        } else {
            panic!("Expected Utf8 column");
        }
    }

    #[test]
    fn test_parquet_column_names_order() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("z", ArrowDataType::Int32, false),
            Field::new("a", ArrowDataType::Float64, false),
            Field::new("m", ArrowDataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(arrow::array::Int64Array::from(vec![0i64])),
            ],
        )
        .expect("record batch");

        let (_dir, path) = write_test_parquet(schema, vec![batch]);
        let ds = ParquetDataset::from_file(&path).expect("from_file");

        // Insertion order must be preserved
        assert_eq!(ds.column_names(), vec!["z", "a", "m"]);
    }

    #[test]
    fn test_parquet_to_float_matrix() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", ArrowDataType::Float64, false),
            Field::new("b", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 2.0])),
                Arc::new(Float64Array::from(vec![3.0, 4.0])),
            ],
        )
        .expect("record batch");

        let (_dir, path) = write_test_parquet(schema, vec![batch]);
        let ds = ParquetDataset::from_file(&path).expect("from_file");
        let mat = ds.to_float_matrix().expect("to_float_matrix");

        assert_eq!(mat.shape(), &[2, 2]);
        assert!((mat[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((mat[[0, 1]] - 3.0).abs() < 1e-10);
        assert!((mat[[1, 0]] - 2.0).abs() < 1e-10);
        assert!((mat[[1, 1]] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_parquet_nullable_values() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            ArrowDataType::Float64,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(vec![
                Some(1.0),
                None,
                Some(3.0),
            ]))],
        )
        .expect("record batch");

        let (_dir, path) = write_test_parquet(schema, vec![batch]);
        let ds = ParquetDataset::from_file(&path).expect("from_file");

        if let Some(ColumnData::Float64(vals)) = ds.column("v") {
            assert_eq!(vals[0], Some(1.0));
            assert_eq!(vals[1], None);
            assert_eq!(vals[2], Some(3.0));
        } else {
            panic!("Expected Float64 column");
        }
    }

    #[test]
    fn test_parquet_to_float_matrix_no_numeric_fails() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "name",
            ArrowDataType::Utf8,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(vec!["x"]))])
                .expect("record batch");

        let (_dir, path) = write_test_parquet(schema, vec![batch]);
        let ds = ParquetDataset::from_file(&path).expect("from_file");
        assert!(ds.to_float_matrix().is_err());
    }

    #[test]
    fn test_parquet_multiple_batches() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "v",
            ArrowDataType::Int32,
            false,
        )]));
        let batch1 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1, 2]))])
                .expect("batch1");
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![3, 4, 5]))],
        )
        .expect("batch2");

        let (_dir, path) = write_test_parquet(schema, vec![batch1, batch2]);
        let ds = ParquetDataset::from_file(&path).expect("from_file");

        assert_eq!(ds.n_rows(), 5);
        if let Some(ColumnData::Int32(vals)) = ds.column("v") {
            assert_eq!(vals.len(), 5);
            assert_eq!(vals[4], Some(5));
        } else {
            panic!("Expected Int32 column");
        }
    }
}
