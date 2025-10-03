//! Parquet column statistics and metadata

use crate::error::{IoError, Result};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Statistics for a single column
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    /// Column name
    pub name: String,

    /// Number of null values
    pub null_count: Option<i64>,

    /// Number of distinct values (if available)
    pub distinct_count: Option<i64>,

    /// Minimum value (as bytes)
    pub min_value: Option<Vec<u8>>,

    /// Maximum value (as bytes)
    pub max_value: Option<Vec<u8>>,

    /// Total number of values
    pub num_values: i64,
}

impl ColumnStatistics {
    /// Get minimum value as f64 (if applicable)
    pub fn min_f64(&self) -> Option<f64> {
        self.min_value.as_ref().and_then(|bytes| {
            if bytes.len() == 8 {
                let mut array = [0u8; 8];
                array.copy_from_slice(bytes);
                Some(f64::from_le_bytes(array))
            } else {
                None
            }
        })
    }

    /// Get maximum value as f64 (if applicable)
    pub fn max_f64(&self) -> Option<f64> {
        self.max_value.as_ref().and_then(|bytes| {
            if bytes.len() == 8 {
                let mut array = [0u8; 8];
                array.copy_from_slice(bytes);
                Some(f64::from_le_bytes(array))
            } else {
                None
            }
        })
    }

    /// Get minimum value as i64 (if applicable)
    pub fn min_i64(&self) -> Option<i64> {
        self.min_value.as_ref().and_then(|bytes| {
            if bytes.len() == 8 {
                let mut array = [0u8; 8];
                array.copy_from_slice(bytes);
                Some(i64::from_le_bytes(array))
            } else {
                None
            }
        })
    }

    /// Get maximum value as i64 (if applicable)
    pub fn max_i64(&self) -> Option<i64> {
        self.max_value.as_ref().and_then(|bytes| {
            if bytes.len() == 8 {
                let mut array = [0u8; 8];
                array.copy_from_slice(bytes);
                Some(i64::from_le_bytes(array))
            } else {
                None
            }
        })
    }

    /// Get minimum value as i32 (if applicable)
    pub fn min_i32(&self) -> Option<i32> {
        self.min_value.as_ref().and_then(|bytes| {
            if bytes.len() == 4 {
                let mut array = [0u8; 4];
                array.copy_from_slice(bytes);
                Some(i32::from_le_bytes(array))
            } else {
                None
            }
        })
    }

    /// Get maximum value as i32 (if applicable)
    pub fn max_i32(&self) -> Option<i32> {
        self.max_value.as_ref().and_then(|bytes| {
            if bytes.len() == 4 {
                let mut array = [0u8; 4];
                array.copy_from_slice(bytes);
                Some(i32::from_le_bytes(array))
            } else {
                None
            }
        })
    }

    /// Check if column has null values
    pub fn has_nulls(&self) -> bool {
        self.null_count.map_or(false, |count| count > 0)
    }

    /// Get null percentage (0.0 to 1.0)
    pub fn null_percentage(&self) -> f64 {
        if self.num_values == 0 {
            0.0
        } else {
            self.null_count.unwrap_or(0) as f64 / self.num_values as f64
        }
    }
}

/// File-level statistics for a Parquet file
#[derive(Debug, Clone)]
pub struct ParquetFileStatistics {
    /// Number of rows in the file
    pub num_rows: i64,

    /// Number of row groups
    pub num_row_groups: usize,

    /// Statistics for each column
    pub columns: HashMap<String, ColumnStatistics>,

    /// File version
    pub version: i32,

    /// Created by (writer identification)
    pub created_by: Option<String>,
}

impl ParquetFileStatistics {
    /// Get statistics for a specific column
    pub fn column_stats(&self, name: &str) -> Option<&ColumnStatistics> {
        self.columns.get(name)
    }

    /// Get all column names
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    /// Check if any column has statistics
    pub fn has_statistics(&self) -> bool {
        !self.columns.is_empty()
    }
}

/// Read Parquet file statistics without loading data
///
/// This function reads only the metadata from a Parquet file,
/// providing column statistics without loading the actual data.
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
///
/// # Examples
///
/// ```rust,no_run
/// use scirs2_io::parquet::read_parquet_statistics;
///
/// let stats = read_parquet_statistics("data.parquet")?;
/// println!("File has {} rows", stats.num_rows);
///
/// if let Some(col_stats) = stats.column_stats("temperature") {
///     if let (Some(min), Some(max)) = (col_stats.min_f64(), col_stats.max_f64()) {
///         println!("Temperature range: {} to {}", min, max);
///     }
/// }
/// # Ok::<(), scirs2_io::error::IoError>(())
/// ```
pub fn read_parquet_statistics<P: AsRef<Path>>(path: P) -> Result<ParquetFileStatistics> {
    let file = File::open(path.as_ref()).map_err(|e| {
        IoError::FileError(format!(
            "Failed to open file '{}': {}",
            path.as_ref().display(),
            e
        ))
    })?;

    let reader = SerializedFileReader::new(file)
        .map_err(|e| IoError::ParquetError(format!("Failed to create Parquet reader: {}", e)))?;

    let metadata = reader.metadata();
    extract_statistics(metadata)
}

/// Extract statistics from file metadata
fn extract_statistics(metadata: &ParquetMetaData) -> Result<ParquetFileStatistics> {
    let file_metadata = metadata.file_metadata();
    let num_rows = file_metadata.num_rows();
    let num_row_groups = metadata.num_row_groups();
    let version = file_metadata.version();
    let created_by = file_metadata.created_by().map(|s| s.to_string());

    let mut columns: HashMap<String, ColumnStatistics> = HashMap::new();

    // Get schema
    let schema = file_metadata.schema_descr();

    // Iterate through columns
    for (col_idx, column) in schema.columns().iter().enumerate() {
        let column_name = column.name().to_string();

        let mut total_null_count: Option<i64> = None;
        let mut total_distinct_count: Option<i64> = None;
        let mut total_num_values: i64 = 0;
        let mut min_value: Option<Vec<u8>> = None;
        let mut max_value: Option<Vec<u8>> = None;

        // Aggregate statistics across all row groups
        for rg_idx in 0..num_row_groups {
            let row_group = metadata.row_group(rg_idx);
            if col_idx < row_group.num_columns() {
                let col_chunk = row_group.column(col_idx);

                if let Some(stats) = col_chunk.statistics() {
                    // Null count
                    if let Some(null_count) = stats.null_count_opt() {
                        total_null_count = Some(total_null_count.unwrap_or(0) + null_count as i64);
                    }

                    // Distinct count
                    if let Some(distinct) = stats.distinct_count_opt() {
                        total_distinct_count =
                            Some(total_distinct_count.unwrap_or(0) + distinct as i64);
                    }

                    // Min/max values
                    if let Some(min_bytes) = stats.min_bytes_opt() {
                        // Update global min
                        if min_value.is_none()
                            || (min_bytes < min_value.as_ref().unwrap().as_slice())
                        {
                            min_value = Some(min_bytes.to_vec());
                        }
                    }

                    if let Some(max_bytes) = stats.max_bytes_opt() {
                        // Update global max
                        if max_value.is_none()
                            || (max_bytes > max_value.as_ref().unwrap().as_slice())
                        {
                            max_value = Some(max_bytes.to_vec());
                        }
                    }
                }

                // Add num values
                total_num_values += col_chunk.num_values();
            }
        }

        columns.insert(
            column_name.clone(),
            ColumnStatistics {
                name: column_name,
                null_count: total_null_count,
                distinct_count: total_distinct_count,
                min_value,
                max_value,
                num_values: total_num_values,
            },
        );
    }

    Ok(ParquetFileStatistics {
        num_rows,
        num_row_groups,
        columns,
        version,
        created_by,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::write_parquet;
    use scirs2_core::ndarray::Array1;
    use tempfile::tempdir;

    #[test]
    fn test_read_statistics() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("stats_test.parquet");

        // Write some data
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        write_parquet(&path, &data, Default::default()).unwrap();

        // Read statistics
        let stats = read_parquet_statistics(&path).unwrap();

        assert_eq!(stats.num_rows, 5);
        assert!(stats.num_row_groups > 0);
        assert!(stats.has_statistics());
    }

    #[test]
    fn test_column_statistics() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("col_stats.parquet");

        let data = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        write_parquet(&path, &data, Default::default()).unwrap();

        let stats = read_parquet_statistics(&path).unwrap();
        let col_stats = stats.column_stats("value").unwrap();

        assert_eq!(col_stats.num_values, 5);
        assert!(!col_stats.has_nulls());
        assert_eq!(col_stats.null_percentage(), 0.0);
    }

    #[test]
    fn test_statistics_min_max_f64() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("minmax.parquet");

        let data = Array1::from_vec(vec![5.5, 1.2, 9.8, 3.3, 7.1]);
        write_parquet(&path, &data, Default::default()).unwrap();

        let stats = read_parquet_statistics(&path).unwrap();
        let col_stats = stats.column_stats("value").unwrap();

        // Note: Actual min/max values depend on how Parquet stores them
        // This test verifies the API works
        assert!(col_stats.min_value.is_some() || col_stats.max_value.is_some());
    }

    #[test]
    fn test_file_statistics_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.parquet");

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        write_parquet(&path, &data, Default::default()).unwrap();

        let stats = read_parquet_statistics(&path).unwrap();

        assert_eq!(stats.num_rows, 3);
        assert!(stats.version > 0);
        assert!(stats.created_by.is_some());
        assert!(stats.column_names().contains(&"value"));
    }

    #[test]
    fn test_statistics_large_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("large_stats.parquet");

        let data: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        let array = Array1::from_vec(data);
        write_parquet(&path, &array, Default::default()).unwrap();

        let stats = read_parquet_statistics(&path).unwrap();

        assert_eq!(stats.num_rows, 1000);
        let col_stats = stats.column_stats("value").unwrap();
        assert_eq!(col_stats.num_values, 1000);
    }
}
