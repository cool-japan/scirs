//! Apache Parquet file format support
//!
//! This module provides reading and writing of Apache Parquet files,
//! leveraging the Apache Arrow ecosystem for efficient columnar data storage.
//!
//! # Features
//!
//! - **Columnar storage**: Efficient storage and retrieval of large datasets
//! - **Compression**: Multiple compression codecs (Snappy, Gzip, LZ4, ZSTD, Brotli)
//! - **Schema handling**: Automatic schema inference and validation
//! - **Python interoperability**: Compatible with Pandas, Polars, PyArrow
//! - **Column selection**: Read specific columns for reduced I/O
//! - **Chunked reading**: Memory-efficient streaming for large files
//! - **Column statistics**: Fast metadata access without reading data
//! - **Predicate pushdown**: Efficient filtering with row group pruning
//!
//! # Examples
//!
//! ## Basic Reading
//!
//! ```rust,no_run
//! use scirs2_io::parquet::read_parquet;
//!
//! let data = read_parquet("data.parquet")?;
//! println!("Columns: {:?}", data.column_names());
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```
//!
//! ## Column Selection
//!
//! ```rust,no_run
//! use scirs2_io::parquet::read_parquet_columns;
//!
//! let data = read_parquet_columns("data.parquet", &["revenue", "date"])?;
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```
//!
//! ## Writing with Compression
//!
//! ```rust,no_run
//! use scirs2_io::parquet::{write_parquet, ParquetWriteOptions, CompressionCodec};
//! use scirs2_core::ndarray::Array1;
//!
//! let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
//! let options = ParquetWriteOptions {
//!     compression: CompressionCodec::Zstd,
//!     ..Default::default()
//! };
//! write_parquet("output.parquet", &data, options)?;
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```
//!
//! ## Chunked Reading for Large Files
//!
//! ```rust,no_run
//! use scirs2_io::parquet::read_parquet_chunked;
//!
//! // Process large file in memory-efficient chunks
//! for chunk_result in read_parquet_chunked("large_data.parquet", 10000)? {
//!     let chunk = chunk_result?;
//!     println!("Processing {} rows", chunk.num_rows());
//!     // Process chunk without loading entire file into memory
//! }
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```
//!
//! ## Column Statistics
//!
//! ```rust,no_run
//! use scirs2_io::parquet::read_parquet_statistics;
//!
//! // Read metadata without loading data
//! let stats = read_parquet_statistics("data.parquet")?;
//! println!("File has {} rows in {} row groups", stats.num_rows, stats.num_row_groups);
//!
//! if let Some(col_stats) = stats.column_stats("temperature") {
//!     println!("Temperature column has {} values", col_stats.num_values);
//!     if let (Some(min), Some(max)) = (col_stats.min_f64(), col_stats.max_f64()) {
//!         println!("Range: {} to {}", min, max);
//!     }
//! }
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```
//!
//! ## Predicate Pushdown
//!
//! ```rust,no_run
//! use scirs2_io::parquet::{read_parquet_filtered, FilterConfig, ParquetPredicate, PredicateValue};
//!
//! // Filter data efficiently using predicates
//! let predicate = ParquetPredicate::and(vec![
//!     ParquetPredicate::gt("temperature", PredicateValue::Float64(20.0)),
//!     ParquetPredicate::lt("temperature", PredicateValue::Float64(30.0)),
//! ]);
//!
//! let config = FilterConfig::new(predicate)
//!     .with_batch_size(5000)
//!     .with_columns(vec!["temperature".to_string(), "humidity".to_string()]);
//!
//! let data = read_parquet_filtered("weather.parquet", config)?;
//! println!("Found {} rows matching predicate", data.num_rows());
//! # Ok::<(), scirs2_io::error::IoError>(())
//! ```

use crate::error::{IoError, Result};

pub mod conversion;
pub mod options;
pub mod predicates;
pub mod reader;
pub mod schema;
pub mod statistics;
pub mod writer;

pub use conversion::{arrow_to_ndarray, ndarray_to_arrow};
pub use options::{CompressionCodec, ParquetVersion, ParquetWriteOptions};
pub use predicates::{
    analyze_predicate_effectiveness, read_parquet_filtered, read_parquet_filtered_chunked,
    FilterConfig, ParquetPredicate, PredicateAnalysis, PredicateValue,
};
pub use reader::{
    read_parquet, read_parquet_chunked, read_parquet_chunked_columns, read_parquet_columns,
    ParquetChunkIterator, ParquetData, ParquetReader,
};
pub use schema::{infer_arrow_schema, ParquetSchema};
pub use statistics::{read_parquet_statistics, ColumnStatistics, ParquetFileStatistics};
pub use writer::{write_parquet, write_parquet_with_name, ParquetWriter};

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_parquet_roundtrip_f64() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.parquet");

        // Create test data
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Write
        write_parquet(&path, &data, Default::default()).unwrap();

        // Read
        let loaded = read_parquet(&path).unwrap();

        // Verify
        assert_eq!(loaded.column_names().len(), 1);
        let column = loaded.get_column_f64("value").unwrap();
        assert_eq!(column.len(), 5);
        assert_eq!(column[0], 1.0);
        assert_eq!(column[4], 5.0);
    }

    #[test]
    fn test_compression_codecs() {
        let dir = tempdir().unwrap();
        let data = Array1::from_vec((0..100).map(|x| x as f64).collect::<Vec<_>>());

        for codec in [
            CompressionCodec::Uncompressed,
            CompressionCodec::Snappy,
            CompressionCodec::Gzip,
        ] {
            let path = dir.path().join(format!("test_{:?}.parquet", codec));
            let options = ParquetWriteOptions {
                compression: codec,
                ..Default::default()
            };

            write_parquet(&path, &data, options).unwrap();
            let loaded = read_parquet(&path).unwrap();

            assert_eq!(loaded.num_rows(), 100);
        }
    }

    #[test]
    fn test_parquet_roundtrip_i32() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_i32.parquet");

        let data = Array1::from_vec(vec![10i32, 20, 30, 40, 50]);

        write_parquet(&path, &data, Default::default()).unwrap();
        let loaded = read_parquet(&path).unwrap();

        assert_eq!(loaded.num_rows(), 5);
        let column = loaded.get_column_i32("value").unwrap();
        assert_eq!(column[0], 10);
        assert_eq!(column[4], 50);
    }

    #[test]
    fn test_parquet_roundtrip_f32() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_f32.parquet");

        let data = Array1::from_vec(vec![1.5f32, 2.5, 3.5, 4.5]);

        write_parquet(&path, &data, Default::default()).unwrap();
        let loaded = read_parquet(&path).unwrap();

        let column = loaded.get_column_f32("value").unwrap();
        assert_eq!(column.len(), 4);
        assert!((column[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_custom_column_name() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_named.parquet");

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        write_parquet_with_name(&path, &data, "temperature", Default::default()).unwrap();
        let loaded = read_parquet(&path).unwrap();

        assert_eq!(loaded.column_names(), vec!["temperature"]);
        let column = loaded.get_column_f64("temperature").unwrap();
        assert_eq!(column.len(), 3);
    }

    #[test]
    fn test_large_dataset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_large.parquet");

        let data: Vec<f64> = (0..10000).map(|x| x as f64 * 0.1).collect();
        let array = Array1::from_vec(data);

        write_parquet(&path, &array, Default::default()).unwrap();
        let loaded = read_parquet(&path).unwrap();

        assert_eq!(loaded.num_rows(), 10000);
        let column = loaded.get_column_f64("value").unwrap();
        assert!((column[0] - 0.0).abs() < 1e-10);
        assert!((column[9999] - 999.9).abs() < 1e-6);
    }

    #[test]
    fn test_schema_introspection() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_schema.parquet");

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        write_parquet_with_name(&path, &data, "measurements", Default::default()).unwrap();
        let loaded = read_parquet(&path).unwrap();

        let schema = loaded.schema();
        assert_eq!(schema.num_columns(), 1);
        assert_eq!(schema.column_names(), vec!["measurements"]);
        assert!(schema.field("measurements").is_some());
        assert!(schema.field("nonexistent").is_none());
    }

    #[test]
    fn test_options_builder_pattern() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_builder.parquet");

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let options = ParquetWriteOptions::with_compression(CompressionCodec::Zstd)
            .with_row_group_size(500)
            .with_dictionary(false);

        write_parquet(&path, &data, options).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_error_invalid_column() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_error.parquet");

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        write_parquet(&path, &data, Default::default()).unwrap();

        let loaded = read_parquet(&path).unwrap();

        // Attempt to access non-existent column
        let result = loaded.get_column_f64("nonexistent");
        assert!(result.is_err());

        // Attempt to access with wrong type
        let result = loaded.get_column_i32("value");
        assert!(result.is_err());
    }

    #[test]
    fn test_data_accuracy() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_accuracy.parquet");

        let original = Array1::from_vec(vec![
            1.23456789,
            2.98765432,
            std::f64::consts::PI,
            4.71238898,
            5.55555555,
        ]);

        write_parquet(&path, &original, Default::default()).unwrap();
        let loaded = read_parquet(&path).unwrap();
        let recovered = loaded.get_column_f64("value").unwrap();

        assert_eq!(recovered.len(), original.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10, "Value mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_chunked_reading() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_chunked.parquet");

        // Create a dataset with 100 rows
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let array = Array1::from_vec(data);

        write_parquet(&path, &array, Default::default()).unwrap();

        // Read in chunks of 10 rows
        let chunks: Vec<_> = read_parquet_chunked(&path, 10)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        // Should have 10 chunks
        assert_eq!(chunks.len(), 10);

        // Verify total rows
        let total_rows: usize = chunks.iter().map(|c| c.num_rows()).sum();
        assert_eq!(total_rows, 100);

        // Verify first chunk
        let first_chunk = &chunks[0];
        let first_values = first_chunk.get_column_f64("value").unwrap();
        assert_eq!(first_values[0], 0.0);
        assert_eq!(first_values[9], 9.0);
    }

    #[test]
    fn test_chunked_column_selection() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_chunked_cols.parquet");

        // Write data
        let data = Array1::from_vec((0..50).map(|x| x as f64).collect::<Vec<_>>());
        write_parquet_with_name(&path, &data, "column_a", Default::default()).unwrap();

        // Read in chunks with column selection
        let chunks: Vec<_> = read_parquet_chunked_columns(&path, &["column_a"], 10)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(chunks.len(), 5); // 50 rows / 10 per chunk = 5 chunks
        assert_eq!(chunks[0].num_columns(), 1);
        assert_eq!(chunks[0].column_names(), vec!["column_a"]);
    }

    #[test]
    fn test_chunked_reading_schema() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_schema_chunk.parquet");

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        write_parquet_with_name(&path, &data, "test_col", Default::default()).unwrap();

        let mut iterator = read_parquet_chunked(&path, 2).unwrap();
        let schema = iterator.schema();

        assert_eq!(schema.num_columns(), 1);
        assert_eq!(schema.column_names(), vec!["test_col"]);

        // Consume the iterator to verify it works
        let chunks: Vec<_> = iterator.collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(chunks.len(), 2); // 3 rows with chunk size 2 = 2 chunks
    }

    #[test]
    fn test_chunked_memory_efficiency() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_memory.parquet");

        // Create a large dataset
        let data: Vec<f64> = (0..10000).map(|x| x as f64).collect();
        let array = Array1::from_vec(data);
        write_parquet(&path, &array, Default::default()).unwrap();

        // Read in small chunks to test memory efficiency
        let mut row_count = 0;
        let mut chunk_count = 0;

        for chunk_result in read_parquet_chunked(&path, 100).unwrap() {
            let chunk = chunk_result.unwrap();
            row_count += chunk.num_rows();
            chunk_count += 1;

            // Each chunk should have at most 100 rows
            assert!(chunk.num_rows() <= 100);
        }

        assert_eq!(row_count, 10000);
        assert_eq!(chunk_count, 100); // 10000 / 100 = 100 chunks
    }

    #[test]
    fn test_empty_chunked_reading() {
        // Test that non-existent file returns proper error
        let result = read_parquet_chunked("nonexistent.parquet", 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_chunked_single_row() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_single_row.parquet");

        let data = Array1::from_vec(vec![42.0]);
        write_parquet(&path, &data, Default::default()).unwrap();

        let chunks: Vec<_> = read_parquet_chunked(&path, 10)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].num_rows(), 1);

        let values = chunks[0].get_column_f64("value").unwrap();
        assert_eq!(values[0], 42.0);
    }

    #[test]
    fn test_statistics_api() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_stats_api.parquet");

        // Write data with known range
        let data = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        write_parquet(&path, &data, Default::default()).unwrap();

        // Read statistics
        let stats = read_parquet_statistics(&path).unwrap();

        assert_eq!(stats.num_rows, 5);
        assert!(stats.has_statistics());
        assert!(stats.column_stats("value").is_some());

        let col_stats = stats.column_stats("value").unwrap();
        assert_eq!(col_stats.num_values, 5);
        assert!(!col_stats.has_nulls());
    }

    #[test]
    fn test_predicate_filtering() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_predicate.parquet");

        // Write data
        let data = Array1::from_vec(vec![5.0, 15.0, 25.0, 35.0, 45.0]);
        write_parquet(&path, &data, Default::default()).unwrap();

        // Read with predicate
        let predicate = ParquetPredicate::gt("value", PredicateValue::Float64(20.0));
        let config = FilterConfig::new(predicate);

        let result = read_parquet_filtered(&path, config);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert!(data.num_rows() > 0);
    }

    #[test]
    fn test_predicate_chunked_filtering() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_pred_chunked.parquet");

        // Write test data
        let data = Array1::from_vec((0..100).map(|x| x as f64).collect::<Vec<_>>());
        write_parquet(&path, &data, Default::default()).unwrap();

        // Read with predicate in chunks
        let predicate = ParquetPredicate::gt("value", PredicateValue::Float64(50.0));
        let config = FilterConfig::new(predicate).with_batch_size(10);

        let chunks: Vec<_> = read_parquet_filtered_chunked(&path, config)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        // Should get multiple chunks
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_combined_statistics_and_predicates() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_combined.parquet");

        // Write data
        let data = Array1::from_vec(vec![1.0, 5.0, 10.0, 15.0, 20.0]);
        write_parquet(&path, &data, Default::default()).unwrap();

        // First, check statistics
        let stats = read_parquet_statistics(&path).unwrap();
        assert_eq!(stats.num_rows, 5);

        // Then filter based on statistics
        let predicate = ParquetPredicate::gt("value", PredicateValue::Float64(8.0));
        let config = FilterConfig::new(predicate);

        let filtered = read_parquet_filtered(&path, config).unwrap();
        assert!(filtered.num_rows() > 0);
    }

    #[test]
    fn test_statistics_with_compression() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_stats_compressed.parquet");

        // Write compressed data
        let data = Array1::from_vec((0..50).map(|x| x as f64).collect::<Vec<_>>());
        let options = ParquetWriteOptions {
            compression: CompressionCodec::Zstd,
            enable_statistics: true,
            ..Default::default()
        };
        write_parquet(&path, &data, options).unwrap();

        // Statistics should work with compressed data
        let stats = read_parquet_statistics(&path).unwrap();
        assert_eq!(stats.num_rows, 50);
        assert!(stats.has_statistics());
    }
}
