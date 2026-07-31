//! Pure Rust columnar storage format
//!
//! Provides a simplified Parquet-like columnar storage format with:
//! - Column-oriented storage for efficient analytical queries
//! - Run-length encoding (RLE) for repeated values
//! - Dictionary encoding for categorical string data
//! - Delta encoding for sorted numeric columns
//! - Support for f64, i64, String, and bool column types
//!
//! # File Format
//!
//! The `.scircol` binary format stores data column-by-column with
//! automatic encoding selection per column.
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_io::columnar::{write_columnar, read_columnar, Column, ColumnarTable};
//!
//! // Create table with multiple column types
//! let table = ColumnarTable::from_columns(vec![
//!     Column::float64("temperature", vec![20.5, 21.0, 19.8, 22.1]),
//!     Column::int64("sensor_id", vec![1, 2, 1, 3]),
//!     Column::string("location", vec![
//!         "lab_a".into(), "lab_b".into(), "lab_a".into(), "lab_c".into(),
//!     ]),
//!     Column::boolean("active", vec![true, true, false, true]),
//! ]).expect("Column creation failed");
//!
//! // Write to file
//! write_columnar("sensors.scircol", &table).expect("Write failed");
//!
//! // Read back
//! let loaded = read_columnar("sensors.scircol").expect("Read failed");
//! assert_eq!(loaded.num_rows(), 4);
//! assert_eq!(loaded.num_columns(), 4);
//! ```

pub mod delta;
pub mod dictionary;
pub mod encoding;
pub mod fsst;
pub mod reader;
pub mod rle;
/// Column statistics, row groups, and predicate pushdown support
pub mod statistics;
pub mod types;
pub mod writer;

pub use reader::read_columnar;
pub use statistics::{
    filter_table, read_columnar_with_columns, select_columns, split_into_row_groups, ColumnStats,
    Predicate, RowGroup, RowGroupConfig, TableStats,
};
pub use types::{Column, ColumnData, ColumnTypeTag, ColumnarTable, EncodingType};
pub use writer::{write_columnar, write_columnar_with_options, ColumnarWriteOptions};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_columnar_roundtrip_all_types() {
        let dir = std::env::temp_dir().join("scirs2_columnar_test_roundtrip");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("all_types.scircol");

        let table = ColumnarTable::from_columns(vec![
            Column::float64("f64_col", vec![1.0, 2.5, std::f64::consts::PI, -0.5, 100.0]),
            Column::int64("i64_col", vec![10, 20, 30, 40, 50]),
            Column::string(
                "str_col",
                vec![
                    "hello".into(),
                    "world".into(),
                    "foo".into(),
                    "bar".into(),
                    "baz".into(),
                ],
            ),
            Column::boolean("bool_col", vec![true, false, true, true, false]),
        ])
        .expect("Failed to create table");

        write_columnar(&path, &table).expect("Failed to write");
        let loaded = read_columnar(&path).expect("Failed to read");

        assert_eq!(loaded.num_rows(), 5);
        assert_eq!(loaded.num_columns(), 4);

        let f64_data = loaded.get_f64("f64_col").expect("Failed to get f64");
        assert!((f64_data[2] - std::f64::consts::PI).abs() < 1e-10);

        let i64_data = loaded.get_i64("i64_col").expect("Failed to get i64");
        assert_eq!(i64_data[3], 40);

        let str_data = loaded.get_str("str_col").expect("Failed to get str");
        assert_eq!(str_data[1], "world");

        let bool_data = loaded.get_bool("bool_col").expect("Failed to get bool");
        assert!(!bool_data[1]);
        assert!(bool_data[0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_columnar_rle_encoding() {
        let dir = std::env::temp_dir().join("scirs2_columnar_test_rle");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("rle.scircol");

        // Create data with lots of runs
        let values: Vec<f64> = [1.0f64; 100]
            .into_iter()
            .chain([2.0f64; 100])
            .chain([3.0f64; 100])
            .collect();

        let table = ColumnarTable::from_columns(vec![Column::float64("runs", values.clone())])
            .expect("Failed to create table");

        // Auto-detect should choose RLE
        write_columnar(&path, &table).expect("Failed to write");
        let loaded = read_columnar(&path).expect("Failed to read");

        let data = loaded.get_f64("runs").expect("Failed to get f64");
        assert_eq!(data.len(), 300);
        assert_eq!(data[0], 1.0);
        assert_eq!(data[100], 2.0);
        assert_eq!(data[200], 3.0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_columnar_dictionary_encoding() {
        let dir = std::env::temp_dir().join("scirs2_columnar_test_dict");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("dict.scircol");

        // Create categorical data (few unique values, many repeats)
        let categories = ["red", "green", "blue"];
        let mut values = Vec::new();
        for i in 0..300 {
            values.push(categories[i % 3].to_string());
        }

        let table = ColumnarTable::from_columns(vec![Column::string("color", values.clone())])
            .expect("Failed to create table");

        write_columnar(&path, &table).expect("Failed to write");
        let loaded = read_columnar(&path).expect("Failed to read");

        let data = loaded.get_str("color").expect("Failed to get str");
        assert_eq!(data.len(), 300);
        assert_eq!(data[0], "red");
        assert_eq!(data[1], "green");
        assert_eq!(data[2], "blue");
        assert_eq!(data[3], "red");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_columnar_delta_encoding() {
        let dir = std::env::temp_dir().join("scirs2_columnar_test_delta");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("delta.scircol");

        // Sorted integer data (should use delta encoding)
        let sorted_ints: Vec<i64> = (0..1000).collect();

        let table = ColumnarTable::from_columns(vec![Column::int64("sorted", sorted_ints.clone())])
            .expect("Failed to create table");

        write_columnar(&path, &table).expect("Failed to write");
        let loaded = read_columnar(&path).expect("Failed to read");

        let data = loaded.get_i64("sorted").expect("Failed to get i64");
        assert_eq!(data.len(), 1000);
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(val, i as i64);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_columnar_forced_plain_encoding() {
        let dir = std::env::temp_dir().join("scirs2_columnar_test_plain");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("plain.scircol");

        let table = ColumnarTable::from_columns(vec![Column::float64(
            "values",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )])
        .expect("Failed to create table");

        let options = ColumnarWriteOptions {
            encoding: Some(EncodingType::Plain),
        };
        write_columnar_with_options(&path, &table, options).expect("Failed to write");
        let loaded = read_columnar(&path).expect("Failed to read");

        let data = loaded.get_f64("values").expect("Failed to get f64");
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_columnar_empty_table() {
        let dir = std::env::temp_dir().join("scirs2_columnar_test_empty");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("empty.scircol");

        let table = ColumnarTable::new();
        write_columnar(&path, &table).expect("Failed to write");
        let loaded = read_columnar(&path).expect("Failed to read");

        assert_eq!(loaded.num_rows(), 0);
        assert_eq!(loaded.num_columns(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_columnar_mismatched_lengths() {
        let result = ColumnarTable::from_columns(vec![
            Column::float64("a", vec![1.0, 2.0]),
            Column::int64("b", vec![1, 2, 3]),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_columnar_duplicate_column_names() {
        let result = ColumnarTable::from_columns(vec![
            Column::float64("x", vec![1.0]),
            Column::float64("x", vec![2.0]),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_columnar_column_access() {
        let table = ColumnarTable::from_columns(vec![
            Column::float64("a", vec![1.0]),
            Column::int64("b", vec![2]),
        ])
        .expect("Failed to create table");

        assert_eq!(table.column_names(), vec!["a", "b"]);
        assert!(table.column("a").is_ok());
        assert!(table.column("nonexistent").is_err());
        assert!(table.column_by_index(0).is_ok());
        assert!(table.column_by_index(99).is_err());
    }

    #[test]
    fn test_columnar_type_mismatch_access() {
        let table = ColumnarTable::from_columns(vec![Column::float64("a", vec![1.0])])
            .expect("Failed to create table");

        assert!(table.get_f64("a").is_ok());
        assert!(table.get_i64("a").is_err());
        assert!(table.get_str("a").is_err());
        assert!(table.get_bool("a").is_err());
    }

    #[test]
    fn test_columnar_bool_packing() {
        let dir = std::env::temp_dir().join("scirs2_columnar_test_bool");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bools.scircol");

        // Test with non-byte-aligned count (13 bools => 2 bytes)
        let bools = vec![
            true, false, true, true, false, true, false, false, true, true, false, true, true,
        ];

        let table = ColumnarTable::from_columns(vec![Column::boolean("flags", bools.clone())])
            .expect("Failed to create table");

        let options = ColumnarWriteOptions {
            encoding: Some(EncodingType::Plain),
        };
        write_columnar_with_options(&path, &table, options).expect("Failed to write");
        let loaded = read_columnar(&path).expect("Failed to read");

        let data = loaded.get_bool("flags").expect("Failed to get bool");
        assert_eq!(data, &bools);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_null_mask_survives_roundtrip() {
        let dir = std::env::temp_dir().join("scirs2_columnar_test_nulls");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("nulls.scircol");

        // Non-constant data; the placeholder values at null positions (999.0 /
        // -999) are deliberately implausible so a bug that mis-restores the mask
        // (or drops it) is caught by the stats, not just by re-reading raw values.
        let nullable_temp = Column::float64_with_nulls(
            "temp",
            vec![20.5, 999.0, 18.0, 999.0],
            vec![false, true, false, true],
        )
        .expect("Failed to create column");
        let plain_id = Column::int64("id", vec![1, 2, 3, 4]);

        let table = ColumnarTable::from_columns(vec![nullable_temp, plain_id])
            .expect("Failed to create table");

        write_columnar(&path, &table).expect("Failed to write");
        let loaded = read_columnar(&path).expect("Failed to read");

        let temp_col = loaded.column("temp").expect("Failed to get column");
        assert_eq!(temp_col.null_count(), 2);
        assert!(temp_col.is_null(1));
        assert!(temp_col.is_null(3));
        assert!(!temp_col.is_null(0));
        assert!(!temp_col.is_null(2));

        let stats = ColumnStats::from_column(temp_col);
        assert_eq!(stats.null_count, 2);
        assert!((stats.min.expect("no min") - 18.0).abs() < 1e-10);
        assert!((stats.max.expect("no max") - 20.5).abs() < 1e-10);
        assert!((stats.sum.expect("no sum") - (20.5 + 18.0)).abs() < 1e-10);

        // The plain (non-nullable) column must round trip with no null mask.
        let id_col = loaded.column("id").expect("Failed to get column");
        assert_eq!(id_col.null_count(), 0);
        assert!(id_col.null_mask.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
