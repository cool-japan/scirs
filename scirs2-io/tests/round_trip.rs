//! Round-trip testing for all supported file formats
//!
//! This module provides comprehensive round-trip tests that write data to files
//! and then read it back to ensure data integrity and format compliance.
//!
//! Round-trip tests are critical for:
//! - Verifying data preservation across write/read cycles
//! - Ensuring format compliance
//! - Detecting precision loss and data corruption
//! - Validating metadata preservation

use ndarray::{array, Array1, Array2};
#[cfg(feature = "hdf5")]
use scirs2_io::hdf5::{self, AttributeValue, CompressionOptions, DatasetOptions};
use scirs2_io::{
    csv,
    matrix_market::{
        self, MMDataType, MMFormat, MMHeader, MMSparseMatrix, MMSymmetry, ParallelConfig,
        SparseEntry,
    },
    serialize, validation,
};
#[cfg(feature = "hdf5")]
use std::collections::HashMap;
use tempfile::tempdir;

#[test]
fn test_csv_round_trip_basic() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.csv");

    // Original data
    let original_data = vec![
        vec!["name".to_string(), "age".to_string(), "score".to_string()],
        vec!["Alice".to_string(), "25".to_string(), "95.5".to_string()],
        vec!["Bob".to_string(), "30".to_string(), "87.2".to_string()],
        vec!["Charlie".to_string(), "35".to_string(), "92.8".to_string()],
    ];

    // Write data - Convert Vec<Vec<String>> to Array2<String>
    let rows = original_data.len();
    let cols = if rows > 0 { original_data[0].len() } else { 0 };
    let flat_data: Vec<String> = original_data.clone().into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((rows, cols), flat_data).unwrap();

    // Extract headers
    let headers = if array_data.nrows() > 0 {
        Some(array_data.row(0).to_vec())
    } else {
        None
    };
    let data_only = if array_data.nrows() > 1 {
        array_data.slice(ndarray::s![1.., ..]).to_owned()
    } else {
        Array2::from_shape_vec((0, cols), Vec::new()).unwrap()
    };

    let write_result = csv::write_csv(&file_path, &data_only, headers.as_ref(), None);
    assert!(
        write_result.is_ok(),
        "Failed to write CSV: {:?}",
        write_result
    );

    // Read data back
    let read_result = csv::read_csv(&file_path, None);
    assert!(read_result.is_ok(), "Failed to read CSV: {:?}", read_result);

    let (read_headers, read_array) = read_result.unwrap();

    // Verify data integrity
    let total_rows = read_array.nrows() + if read_headers.is_empty() { 0 } else { 1 };
    assert_eq!(total_rows, original_data.len());

    // Check headers if present
    if !read_headers.is_empty() {
        for (j, header) in read_headers.iter().enumerate() {
            assert_eq!(header, &original_data[0][j]);
        }
    }

    // Check data rows
    for i in 0..read_array.nrows() {
        let original_row_idx = i + if read_headers.is_empty() { 0 } else { 1 };
        for j in 0..read_array.ncols() {
            assert_eq!(&read_array[[i, j]], &original_data[original_row_idx][j]);
        }
    }
}

#[test]
fn test_csv_round_trip_with_options() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test_semicolon.csv");

    // Original data with semicolon delimiter
    let original_data = vec![
        vec![
            "id".to_string(),
            "value".to_string(),
            "description".to_string(),
        ],
        vec![
            "1".to_string(),
            "100.5".to_string(),
            "First item".to_string(),
        ],
        vec![
            "2".to_string(),
            "200.3".to_string(),
            "Second item".to_string(),
        ],
    ];

    // CSV options with semicolon delimiter
    let mut options = csv::CsvReaderConfig::default();
    options.delimiter = ';';
    options.has_header = true;

    // Write and read with custom options
    // Convert Vec<Vec<String>> to Array2<String>
    let rows = original_data.len();
    let cols = if rows > 0 { original_data[0].len() } else { 0 };
    let flat_data: Vec<String> = original_data.clone().into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((rows, cols), flat_data).unwrap();

    // Extract headers
    let headers = if array_data.nrows() > 0 {
        Some(array_data.row(0).to_vec())
    } else {
        None
    };
    let data_only = if array_data.nrows() > 1 {
        array_data.slice(ndarray::s![1.., ..]).to_owned()
    } else {
        Array2::from_shape_vec((0, cols), Vec::new()).unwrap()
    };

    // Create write options that match read options
    let mut write_options = csv::CsvWriterConfig::default();
    write_options.delimiter = ';';
    csv::write_csv(
        &file_path,
        &data_only,
        headers.as_ref(),
        Some(write_options),
    )
    .unwrap();
    let (_headers, read_array) = csv::read_csv(&file_path, Some(options)).unwrap();

    // Verify round-trip
    assert_eq!(read_array.nrows(), data_only.nrows());
    assert_eq!(read_array.ncols(), data_only.ncols());
}

#[test]
fn test_matrix_market_sparse_round_trip() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.mtx");

    // Create original sparse matrix
    let header = MMHeader {
        object: "matrix".to_string(),
        format: MMFormat::Coordinate,
        data_type: MMDataType::Real,
        symmetry: MMSymmetry::General,
        comments: vec!["Round-trip test matrix".to_string()],
    };

    let entries = vec![
        SparseEntry {
            row: 0,
            col: 0,
            value: 1.5,
        },
        SparseEntry {
            row: 0,
            col: 2,
            value: -2.3,
        },
        SparseEntry {
            row: 1,
            col: 1,
            value: 4.7,
        },
        SparseEntry {
            row: 2,
            col: 0,
            value: 0.8,
        },
        SparseEntry {
            row: 2,
            col: 2,
            value: -1.1,
        },
    ];

    let original_matrix = MMSparseMatrix {
        header,
        rows: 3,
        cols: 3,
        nnz: 5,
        entries,
    };

    // Write matrix
    let write_result = matrix_market::write_sparse_matrix(&file_path, &original_matrix);
    assert!(
        write_result.is_ok(),
        "Failed to write Matrix Market file: {:?}",
        write_result
    );

    // Read matrix back
    let read_result = matrix_market::read_sparse_matrix(&file_path);
    assert!(
        read_result.is_ok(),
        "Failed to read Matrix Market file: {:?}",
        read_result
    );

    let read_matrix = read_result.unwrap();

    // Verify matrix properties
    assert_eq!(read_matrix.rows, original_matrix.rows);
    assert_eq!(read_matrix.cols, original_matrix.cols);
    assert_eq!(read_matrix.nnz, original_matrix.nnz);
    assert_eq!(read_matrix.header.format, original_matrix.header.format);
    assert_eq!(
        read_matrix.header.data_type,
        original_matrix.header.data_type
    );
    assert_eq!(read_matrix.header.symmetry, original_matrix.header.symmetry);

    // Verify entries (order might be different, so sort both)
    let mut original_entries = original_matrix.entries.clone();
    let mut read_entries = read_matrix.entries.clone();

    original_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));
    read_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));

    assert_eq!(read_entries.len(), original_entries.len());
    for (original, read) in original_entries.iter().zip(read_entries.iter()) {
        assert_eq!(read.row, original.row);
        assert_eq!(read.col, original.col);
        assert!(
            (read.value - original.value).abs() < 1e-10,
            "Value mismatch: {} vs {}",
            read.value,
            original.value
        );
    }
}

#[test]
fn test_matrix_market_parallel_round_trip() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test_parallel.mtx");

    // Create a larger matrix for parallel processing
    let header = MMHeader {
        object: "matrix".to_string(),
        format: MMFormat::Coordinate,
        data_type: MMDataType::Real,
        symmetry: MMSymmetry::General,
        comments: vec!["Parallel round-trip test".to_string()],
    };

    // Generate test entries
    let mut entries = Vec::new();
    for i in 0..1000 {
        entries.push(SparseEntry {
            row: i % 100,
            col: (i * 7) % 100,
            value: (i as f64) * 0.1,
        });
    }

    let original_matrix = MMSparseMatrix {
        header,
        rows: 100,
        cols: 100,
        nnz: entries.len(),
        entries,
    };

    let config = ParallelConfig::default();

    // Write matrix in parallel
    let write_result =
        matrix_market::write_sparse_matrix_parallel(&file_path, &original_matrix, config.clone());
    assert!(
        write_result.is_ok(),
        "Failed to parallel write Matrix Market file: {:?}",
        write_result
    );

    // Read matrix back in parallel
    let read_result = matrix_market::read_sparse_matrix_parallel(&file_path, config);
    assert!(
        read_result.is_ok(),
        "Failed to parallel read Matrix Market file: {:?}",
        read_result
    );

    let (read_matrix, _stats) = read_result.unwrap();

    // Verify matrix integrity
    assert_eq!(read_matrix.rows, original_matrix.rows);
    assert_eq!(read_matrix.cols, original_matrix.cols);
    assert_eq!(read_matrix.nnz, original_matrix.nnz);
    assert_eq!(read_matrix.entries.len(), original_matrix.entries.len());

    // Sort and compare entries
    let mut original_entries = original_matrix.entries.clone();
    let mut read_entries = read_matrix.entries.clone();

    original_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));
    read_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));

    for (original, read) in original_entries.iter().zip(read_entries.iter()) {
        assert_eq!(read.row, original.row);
        assert_eq!(read.col, original.col);
        assert!((read.value - original.value).abs() < 1e-10);
    }
}

#[test]
#[cfg(feature = "hdf5")]
fn test_hdf5_round_trip_basic() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.h5");

    // Original data
    let data1 = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let data2 = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let mut datasets = HashMap::new();
    datasets.insert("array1d".to_string(), data1.clone().into_dyn());
    datasets.insert("array2d".to_string(), data2.clone().into_dyn());

    // Write HDF5 file
    let write_result = hdf5::write_hdf5(&file_path, datasets);
    assert!(
        write_result.is_ok(),
        "Failed to write HDF5 file: {:?}",
        write_result
    );

    // Read data back
    let read_result = hdf5::read_hdf5(&file_path);
    assert!(
        read_result.is_ok(),
        "Failed to read HDF5 file: {:?}",
        read_result
    );

    let root_group = read_result.unwrap();

    // Verify datasets exist
    assert!(root_group.has_dataset("array1d"));
    assert!(root_group.has_dataset("array2d"));

    // Verify 1D array
    let dataset1d = root_group.get_dataset("array1d").unwrap();
    assert_eq!(dataset1d.shape, vec![5]);
    if let Some(data_vec) = dataset1d.as_float_vec() {
        assert_eq!(data_vec.len(), 5);
        for (i, &val) in data_vec.iter().enumerate() {
            assert!((val - data1[i]).abs() < 1e-10);
        }
    } else {
        panic!("Failed to get float data from 1D dataset");
    }

    // Verify 2D array
    let dataset2d = root_group.get_dataset("array2d").unwrap();
    assert_eq!(dataset2d.shape, vec![2, 3]);
    if let Some(data_vec) = dataset2d.as_float_vec() {
        assert_eq!(data_vec.len(), 6);
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for (i, &val) in data_vec.iter().enumerate() {
            assert!((val - expected[i]).abs() < 1e-10);
        }
    } else {
        panic!("Failed to get float data from 2D dataset");
    }
}

#[test]
#[cfg(feature = "hdf5")]
fn test_hdf5_round_trip_with_groups_and_attributes() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("structured.h5");

    // Create structured HDF5 file
    let result = hdf5::create_hdf5_with_structure(&file_path, |file| {
        let root = file.root_mut();

        // Set root attributes
        root.set_attribute("file_version", AttributeValue::String("1.0".to_string()));
        root.set_attribute(
            "created_by",
            AttributeValue::String("round_trip_test".to_string()),
        );

        // Create experiment group
        let experiment = root.create_group("experiment");
        experiment.set_attribute("experiment_id", AttributeValue::Integer(12345));
        experiment.set_attribute("temperature", AttributeValue::Float(25.5));

        // Create measurements subgroup
        let measurements = experiment.create_group("measurements");
        measurements.set_attribute("sensor_type", AttributeValue::String("thermal".to_string()));

        // Add datasets
        let temperature_data = Array1::from(vec![25.1, 25.3, 25.2, 25.4, 25.0]);
        let pressure_data = array![[1013.2, 1013.1], [1013.3, 1013.0]];

        file.create_dataset_from_array(
            "experiment/measurements/temperature",
            &temperature_data,
            None,
        )?;
        file.create_dataset_from_array("experiment/measurements/pressure", &pressure_data, None)?;

        Ok(())
    });

    assert!(
        result.is_ok(),
        "Failed to create structured HDF5 file: {:?}",
        result
    );

    // Read the file back
    let read_result = hdf5::read_hdf5(&file_path);
    assert!(
        read_result.is_ok(),
        "Failed to read structured HDF5 file: {:?}",
        read_result
    );

    let root_group = read_result.unwrap();

    // Verify root attributes
    assert!(root_group.has_attribute("file_version"));
    assert!(root_group.has_attribute("created_by"));

    if let Some(AttributeValue::String(version)) = root_group.get_attribute("file_version") {
        assert_eq!(version, "1.0");
    } else {
        panic!("file_version attribute not found or wrong type");
    }

    // Verify group structure
    assert!(root_group.has_group("experiment"));
    let experiment_group = root_group.get_group("experiment").unwrap();

    assert!(experiment_group.has_attribute("experiment_id"));
    assert!(experiment_group.has_attribute("temperature"));

    if let Some(AttributeValue::Integer(id)) = experiment_group.get_attribute("experiment_id") {
        assert_eq!(*id, 12345);
    } else {
        panic!("experiment_id attribute not found or wrong type");
    }

    // Verify nested group
    assert!(experiment_group.has_group("measurements"));
    let measurements_group = experiment_group.get_group("measurements").unwrap();

    assert!(measurements_group.has_attribute("sensor_type"));
    if let Some(AttributeValue::String(sensor)) = measurements_group.get_attribute("sensor_type") {
        assert_eq!(sensor, "thermal");
    } else {
        panic!("sensor_type attribute not found or wrong type");
    }

    // Verify datasets
    assert!(measurements_group.has_dataset("temperature"));
    assert!(measurements_group.has_dataset("pressure"));

    let temp_dataset = measurements_group.get_dataset("temperature").unwrap();
    assert_eq!(temp_dataset.shape, vec![5]);

    let pressure_dataset = measurements_group.get_dataset("pressure").unwrap();
    assert_eq!(pressure_dataset.shape, vec![2, 2]);
}

#[test]
fn test_serialize_round_trip_json() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.json");

    // Original array
    let original_array = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    // Write array as JSON
    let write_result = serialize::write_array_json(&file_path, &original_array.clone().into_dyn());
    assert!(
        write_result.is_ok(),
        "Failed to write JSON: {:?}",
        write_result
    );

    // Read array back
    let read_result = serialize::read_array_json(&file_path);
    assert!(
        read_result.is_ok(),
        "Failed to read JSON: {:?}",
        read_result
    );

    let read_array = read_result.unwrap();

    // Verify array integrity
    assert_eq!(read_array.shape(), original_array.shape());
    for (original, read) in original_array.iter().zip(read_array.iter()) {
        let original: f64 = *original;
        let read: f64 = *read;
        let diff: f64 = (original - read).abs();
        assert!(diff < 1e-10);
    }
}

#[test]
fn test_serialize_round_trip_messagepack() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.msgpack");

    // Original array
    let original_array = Array1::from(vec![1.5, -2.3, 4.7, 0.0, -1.1]);

    // Write array as MessagePack
    let write_result =
        serialize::write_array_messagepack(&file_path, &original_array.clone().into_dyn());
    assert!(
        write_result.is_ok(),
        "Failed to write MessagePack: {:?}",
        write_result
    );

    // Read array back
    let read_result = serialize::read_array_messagepack(&file_path);
    assert!(
        read_result.is_ok(),
        "Failed to read MessagePack: {:?}",
        read_result
    );

    let read_array = read_result.unwrap();

    // Verify array integrity
    assert_eq!(read_array.shape(), original_array.shape());
    for (original, read) in original_array.iter().zip(read_array.iter()) {
        let original: f64 = *original;
        let read: f64 = *read;
        let diff: f64 = (original - read).abs();
        assert!(diff < 1e-10);
    }
}

#[test]
fn test_serialize_round_trip_binary() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.bin");

    // Original array
    let original_array = Array2::from_shape_fn((10, 10), |(i, j)| (i * j) as f64);

    // Write array as binary
    let write_result =
        serialize::write_array_binary(&file_path, &original_array.clone().into_dyn());
    assert!(
        write_result.is_ok(),
        "Failed to write binary: {:?}",
        write_result
    );

    // Read array back
    let read_result = serialize::read_array_binary(&file_path);
    assert!(
        read_result.is_ok(),
        "Failed to read binary: {:?}",
        read_result
    );

    let read_array = read_result.unwrap();

    // Verify array integrity
    assert_eq!(read_array.shape(), original_array.shape());
    for (original, read) in original_array.iter().zip(read_array.iter()) {
        let original: f64 = *original;
        let read: f64 = *read;
        let diff: f64 = (original - read).abs();
        assert!(diff < 1e-10);
    }
}

#[test]
fn test_validation_round_trip() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("data.csv");

    // Create test data
    let test_data = vec![
        vec!["name".to_string(), "value".to_string()],
        vec!["test1".to_string(), "123.45".to_string()],
        vec!["test2".to_string(), "678.90".to_string()],
    ];

    // Write data - Convert Vec<Vec<String>> to Array2<String>
    let rows = test_data.len();
    let cols = if rows > 0 { test_data[0].len() } else { 0 };
    let flat_data: Vec<String> = test_data.clone().into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((rows, cols), flat_data).unwrap();

    // Extract headers and data
    let headers = if array_data.nrows() > 0 {
        Some(array_data.row(0).to_vec())
    } else {
        None
    };
    let data_only = if array_data.nrows() > 1 {
        array_data.slice(ndarray::s![1.., ..]).to_owned()
    } else {
        Array2::from_shape_vec((0, cols), Vec::new()).unwrap()
    };

    csv::write_csv(&file_path, &data_only, headers.as_ref(), None).unwrap();

    // Calculate checksums
    let original_crc32 = validation::calculate_crc32(&file_path).unwrap();
    let original_sha256 = validation::calculate_sha256(&file_path).unwrap();

    // Read and write again (round-trip)
    let (headers, read_array) = csv::read_csv(&file_path, None).unwrap();
    let round_trip_path = dir.path().join("data_round_trip.csv");
    csv::write_csv(&round_trip_path, &read_array, Some(&headers), None).unwrap();

    // Calculate checksums for round-trip file
    let round_trip_crc32 = validation::calculate_crc32(&round_trip_path).unwrap();
    let round_trip_sha256 = validation::calculate_sha256(&round_trip_path).unwrap();

    // Verify checksums match (indicating identical file content)
    assert_eq!(original_crc32, round_trip_crc32);
    assert_eq!(original_sha256, round_trip_sha256);

    // Verify file checksums match
    let original_recalc = validation::calculate_sha256(&file_path).unwrap();
    let round_trip_recalc = validation::calculate_sha256(&round_trip_path).unwrap();
    assert_eq!(original_sha256, original_recalc);
    assert_eq!(round_trip_sha256, round_trip_recalc);
}

#[test]
#[cfg(feature = "hdf5")]
fn test_compression_round_trip() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("compressed.h5");

    // Create test data
    let large_array = Array2::from_shape_fn((100, 100), |(i, j)| (i + j) as f64);

    // Create compression options
    let mut compression = CompressionOptions::default();
    compression.gzip = Some(6);
    compression.shuffle = true;

    let options = DatasetOptions {
        chunk_size: Some(vec![10, 10]),
        compression,
        fill_value: Some(0.0),
        fletcher32: true,
    };

    // Write with compression
    let mut file = hdf5::HDF5File::create(&file_path).unwrap();
    let result = file.create_dataset_from_array("compressed_data", &large_array, Some(options));
    assert!(
        result.is_ok(),
        "Failed to create compressed dataset: {:?}",
        result
    );
    file.close().unwrap();

    // Read back from a new file handle
    let read_file = hdf5::HDF5File::open(&file_path, hdf5::FileMode::ReadOnly).unwrap();
    let read_result = read_file.read_dataset("compressed_data");
    assert!(
        read_result.is_ok(),
        "Failed to read compressed dataset: {:?}",
        read_result
    );

    let read_array = read_result.unwrap();

    // Verify data integrity
    assert_eq!(read_array.shape(), large_array.shape());
    for (original, read) in large_array.iter().zip(read_array.iter()) {
        let original: f64 = *original;
        let read: f64 = *read;
        let diff: f64 = (original - read).abs();
        assert!(diff < 1e-10);
    }
}

#[test]
fn test_large_data_round_trip() {
    let dir = tempdir().unwrap();
    let matrix_file = dir.path().join("large_matrix.mtx");

    // Create a moderately large sparse matrix
    let header = MMHeader {
        object: "matrix".to_string(),
        format: MMFormat::Coordinate,
        data_type: MMDataType::Real,
        symmetry: MMSymmetry::General,
        comments: vec!["Large matrix round-trip test".to_string()],
    };

    let mut entries = Vec::new();
    for i in 0..10000 {
        entries.push(SparseEntry {
            row: i % 1000,
            col: (i * 37) % 1000, // Use prime number for better distribution
            value: (i as f64) * 0.001,
        });
    }

    let original_matrix = MMSparseMatrix {
        header,
        rows: 1000,
        cols: 1000,
        nnz: entries.len(),
        entries,
    };

    // Use parallel I/O for large matrix
    let config = ParallelConfig::default();

    // Write matrix
    let write_stats =
        matrix_market::write_sparse_matrix_parallel(&matrix_file, &original_matrix, config.clone())
            .unwrap();
    assert!(write_stats.entries_processed > 0);
    assert!(write_stats.io_time_ms > 0.0);

    // Read matrix back
    let (read_matrix, read_stats) =
        matrix_market::read_sparse_matrix_parallel(&matrix_file, config).unwrap();
    assert!(read_stats.entries_processed > 0);
    assert!(read_stats.io_time_ms > 0.0);

    // Verify matrix integrity
    assert_eq!(read_matrix.rows, original_matrix.rows);
    assert_eq!(read_matrix.cols, original_matrix.cols);
    assert_eq!(read_matrix.nnz, original_matrix.nnz);
    assert_eq!(read_matrix.entries.len(), original_matrix.entries.len());

    // Verify all entries match
    let mut original_entries = original_matrix.entries.clone();
    let mut read_entries = read_matrix.entries.clone();

    original_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));
    read_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));

    for (original, read) in original_entries.iter().zip(read_entries.iter()) {
        assert_eq!(read.row, original.row, "Row mismatch");
        assert_eq!(read.col, original.col, "Column mismatch");
        assert!(
            (read.value - original.value).abs() < 1e-12,
            "Value mismatch: {} vs {}",
            read.value,
            original.value
        );
    }

    // Verify performance characteristics
    assert!(
        write_stats.throughput_eps > 0.0,
        "Write throughput should be positive"
    );
    assert!(
        read_stats.throughput_eps > 0.0,
        "Read throughput should be positive"
    );
}

/// Test precision preservation across different numeric types
#[test]
fn test_precision_round_trip() {
    let dir = tempdir().unwrap();

    // Test various precision scenarios
    let precision_values = vec![
        1e-15,                // Very small values
        1e15,                 // Very large values
        std::f64::consts::PI, // Irrational numbers
        std::f64::consts::E,
        1.0 / 3.0,        // Repeating decimals
        1.23456789012345, // High precision decimals
    ];

    for (i, &value) in precision_values.iter().enumerate() {
        let json_file = dir.path().join(format!("precision_{}.json", i));
        let binary_file = dir.path().join(format!("precision_{}.bin", i));

        let original_array = Array1::from(vec![value, -value, value * 2.0, value / 2.0]).into_dyn();

        // Test JSON round-trip
        serialize::write_array_json(&json_file, &original_array).unwrap();
        let json_read = serialize::read_array_json(&json_file).unwrap();

        // Test binary round-trip
        serialize::write_array_binary(&binary_file, &original_array).unwrap();
        let binary_read: ndarray::Array<f64, ndarray::IxDyn> =
            serialize::read_array_binary(&binary_file).unwrap();

        // Verify precision preservation
        for j in 0..original_array.len() {
            // Binary should preserve exact precision
            assert_eq!(
                binary_read[j], original_array[j],
                "Binary precision loss for value {}",
                value
            );

            // JSON might have some floating-point representation limits
            let json_val: f64 = json_read[j];
            let orig_val: f64 = original_array[j];
            let json_error: f64 = (json_val - orig_val).abs();
            assert!(
                json_error < 1e-14,
                "JSON precision error too large: {} for value {}",
                json_error,
                value
            );
        }
    }
}
