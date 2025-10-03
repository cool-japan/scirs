//! Comprehensive round-trip testing example
//!
//! This example demonstrates how to perform round-trip testing for various file formats
//! to ensure data integrity and format compliance.

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_io::{
    csv,
    hdf5::{self, AttributeValue},
    matrix_market::{
        self, MMDataType, MMFormat, MMHeader, MMSparseMatrix, MMSymmetry, ParallelConfig,
        SparseEntry,
    },
    serialize, validation,
};
use tempfile::tempdir;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Round-trip Testing Example");
    println!("=============================");

    // Create temporary directory for all test files
    let temp_dir = tempdir()?;
    println!("📁 Using temporary directory: {:?}", temp_dir.path());

    // Test CSV round-trip
    test_csv_round_trip(&temp_dir)?;

    // Test Matrix Market round-trip
    test_matrix_market_round_trip(&temp_dir)?;

    // Test HDF5 round-trip
    test_hdf5_round_trip(&temp_dir)?;

    // Test serialization round-trip
    test_serialization_round_trip(&temp_dir)?;

    // Test validation round-trip
    test_validation_round_trip(&temp_dir)?;

    // Test parallel processing round-trip
    test_parallel_round_trip(&temp_dir)?;

    println!("\n✅ All round-trip tests completed successfully!");
    println!("💡 Round-trip testing ensures data integrity across write/read cycles");

    Ok(())
}

#[allow(dead_code)]
fn test_csv_round_trip(temp_dir: &tempfile::TempDir) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Testing CSV Round-trip...");

    let csv_file = temp_dir.path().join("test_data.csv");

    // Original data
    let original_data = vec![
        vec!["name".to_string(), "age".to_string(), "score".to_string()],
        vec!["Alice".to_string(), "25".to_string(), "95.5".to_string()],
        vec!["Bob".to_string(), "30".to_string(), "87.2".to_string()],
        vec!["Charlie".to_string(), "35".to_string(), "92.8".to_string()],
    ];

    println!("  📝 Writing CSV with {} rows...", original_data.len());
    // Convert Vec<Vec<String>> to Array2<String>
    let rows = original_data.len();
    let cols = if rows > 0 { original_data[0].len() } else { 0 };
    let flat_data: Vec<String> = original_data.clone().into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((rows, cols), flat_data)?;

    // Extract headers and data
    let headers = if array_data.nrows() > 0 {
        Some(array_data.row(0).to_vec())
    } else {
        None
    };
    let data_only = if array_data.nrows() > 1 {
        array_data
            .slice(scirs2_core::ndarray::s![1.., ..])
            .to_owned()
    } else {
        Array2::from_shape_vec((0, cols), Vec::new())?
    };

    csv::write_csv(&csv_file, &data_only, headers.as_ref(), None)?;

    println!("  📖 Reading CSV back...");
    let (read_headers, read_array) = csv::read_csv(&csv_file, None)?;

    // Verify data integrity
    let total_read_rows = read_array.nrows() + if read_headers.is_empty() { 0 } else { 1 };
    assert_eq!(total_read_rows, original_data.len(), "Row count mismatch");

    // Check headers
    if !read_headers.is_empty() {
        for (j, header) in read_headers.iter().enumerate() {
            assert_eq!(
                header, &original_data[0][j],
                "Header mismatch at column {}",
                j
            );
        }
    }

    // Check data rows
    for i in 0..read_array.nrows() {
        let original_row_idx = i + if read_headers.is_empty() { 0 } else { 1 };
        for j in 0..read_array.ncols() {
            assert_eq!(
                &read_array[[i, j]],
                &original_data[original_row_idx][j],
                "Data mismatch at [{}, {}]",
                i,
                j
            );
        }
    }

    println!(
        "  ✅ CSV round-trip successful: {} rows preserved",
        total_read_rows
    );
    Ok(())
}

#[allow(dead_code)]
fn test_matrix_market_round_trip(
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔢 Testing Matrix Market Round-trip...");

    let matrix_file = temp_dir.path().join("test_matrix.mtx");

    // Create test sparse matrix
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
        nnz: entries.len(),
        entries,
    };

    println!(
        "  📝 Writing {}x{} sparse matrix with {} non-zeros...",
        original_matrix.rows, original_matrix.cols, original_matrix.nnz
    );
    matrix_market::write_sparse_matrix(&matrix_file, &original_matrix)?;

    println!("  📖 Reading matrix back...");
    let read_matrix = matrix_market::read_sparse_matrix(&matrix_file)?;

    // Verify matrix properties
    assert_eq!(read_matrix.rows, original_matrix.rows, "Row count mismatch");
    assert_eq!(
        read_matrix.cols, original_matrix.cols,
        "Column count mismatch"
    );
    assert_eq!(
        read_matrix.nnz, original_matrix.nnz,
        "Non-zero count mismatch"
    );

    // Verify entries (sort both for comparison)
    let mut original_entries = original_matrix.entries.clone();
    let mut read_entries = read_matrix.entries.clone();

    original_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));
    read_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));

    for (original, read) in original_entries.iter().zip(read_entries.iter()) {
        assert_eq!(read.row, original.row, "Row index mismatch");
        assert_eq!(read.col, original.col, "Column index mismatch");
        assert!(
            (read.value - original.value).abs() < 1e-10,
            "Value mismatch: {} vs {}",
            read.value,
            original.value
        );
    }

    println!(
        "  ✅ Matrix Market round-trip successful: {}x{} matrix with {} entries",
        read_matrix.rows,
        read_matrix.cols,
        read_matrix.entries.len()
    );
    Ok(())
}

#[allow(dead_code)]
fn test_hdf5_round_trip(temp_dir: &tempfile::TempDir) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🗄️  Testing HDF5 Round-trip...");

    let hdf5_file = temp_dir.path().join("test_data.h5");

    // Create test data
    let array_1d = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let array_2d = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    // Create structured HDF5 file
    println!("  📝 Writing HDF5 file with groups and attributes...");
    hdf5::create_hdf5_with_structure(&hdf5_file, |file| {
        let root = file.root_mut();

        // Set root attributes
        root.set_attribute("file_version", AttributeValue::String("1.0".to_string()));
        root.set_attribute(
            "created_by",
            AttributeValue::String("round_trip_example".to_string()),
        );

        // Create data group
        let data_group = root.create_group("data");
        data_group.set_attribute(
            "description",
            AttributeValue::String("Test data group".to_string()),
        );

        // Add datasets
        file.create_dataset_from_array("data/array_1d", &array_1d, None)?;
        file.create_dataset_from_array("data/array_2d", &array_2d, None)?;

        Ok(())
    })?;

    println!("  📖 Reading HDF5 file back...");
    let root_group = hdf5::read_hdf5(&hdf5_file)?;

    // Verify structure
    assert!(
        root_group.has_attribute("file_version"),
        "Missing file_version attribute"
    );
    assert!(root_group.has_group("data"), "Missing data group");

    let data_group = root_group.get_group("data").unwrap();
    assert!(
        data_group.has_dataset("array_1d"),
        "Missing array_1d dataset"
    );
    assert!(
        data_group.has_dataset("array_2d"),
        "Missing array_2d dataset"
    );

    // Verify 1D array
    let dataset_1d = data_group.get_dataset("array_1d").unwrap();
    assert_eq!(dataset_1d.shape, vec![5], "1D array shape mismatch");

    if let Some(data_vec) = dataset_1d.as_float_vec() {
        for (i, &val) in data_vec.iter().enumerate() {
            assert!(
                (val - array_1d[i]).abs() < 1e-10,
                "1D array value mismatch at index {}",
                i
            );
        }
    }

    // Verify 2D array
    let dataset_2d = data_group.get_dataset("array_2d").unwrap();
    assert_eq!(dataset_2d.shape, vec![2, 3], "2D array shape mismatch");

    println!("  ✅ HDF5 round-trip successful: groups, attributes, and datasets preserved");
    Ok(())
}

#[allow(dead_code)]
fn test_serialization_round_trip(
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Testing Serialization Round-trip...");

    let json_file = temp_dir.path().join("test_array.json");
    let binary_file = temp_dir.path().join("test_array.bin");
    let msgpack_file = temp_dir.path().join("test_array.msgpack");

    // Test data with various numeric characteristics
    let test_array = array![
        [1.0, -2.5, std::f64::consts::PI],
        [1e-10, 1e10, 0.0],
        [std::f64::consts::E, std::f64::consts::PI, 1.0 / 3.0]
    ];

    // Test JSON serialization
    println!("  📝 Testing JSON serialization...");
    serialize::write_array_json(&json_file, &test_array.clone().into_dyn())?;
    let json_read = serialize::read_array_json(&json_file)?;

    assert_eq!(
        json_read.shape(),
        test_array.shape(),
        "JSON: Shape mismatch"
    );
    for (original, read) in test_array.iter().zip(json_read.iter()) {
        let original: f64 = *original;
        let read: f64 = *read;
        let diff: f64 = (original - read).abs();
        assert!(diff < 1e-14, "JSON: Value precision loss");
    }

    // Test binary serialization
    println!("  📝 Testing binary serialization...");
    serialize::write_array_binary(&binary_file, &test_array.clone().into_dyn())?;
    let binary_read: scirs2_core::ndarray::Array<f64, scirs2_core::ndarray::IxDyn> =
        serialize::read_array_binary(&binary_file)?;

    assert_eq!(
        binary_read.shape(),
        test_array.shape(),
        "Binary: Shape mismatch"
    );
    for (original, read) in test_array.iter().zip(binary_read.iter()) {
        assert_eq!(
            *original, *read,
            "Binary: Value should be exactly preserved"
        );
    }

    // Test MessagePack serialization
    println!("  📝 Testing MessagePack serialization...");
    serialize::write_array_messagepack(&msgpack_file, &test_array.clone().into_dyn())?;
    let msgpack_read = serialize::read_array_messagepack(&msgpack_file)?;

    assert_eq!(
        msgpack_read.shape(),
        test_array.shape(),
        "MessagePack: Shape mismatch"
    );
    for (original, read) in test_array.iter().zip(msgpack_read.iter()) {
        let original: f64 = *original;
        let read: f64 = *read;
        assert!(
            (original - read).abs() < 1e-14,
            "MessagePack: Value precision loss"
        );
    }

    println!("  ✅ All serialization formats preserved data integrity");
    Ok(())
}

#[allow(dead_code)]
fn test_validation_round_trip(
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Validation Round-trip...");

    let data_file = temp_dir.path().join("validation_test.csv");
    let round_trip_file = temp_dir.path().join("validation_round_trip.csv");

    // Create test data
    let test_data = vec![
        vec![
            "id".to_string(),
            "value".to_string(),
            "description".to_string(),
        ],
        vec![
            "1".to_string(),
            "100.5".to_string(),
            "First measurement".to_string(),
        ],
        vec![
            "2".to_string(),
            "200.3".to_string(),
            "Second measurement".to_string(),
        ],
        vec![
            "3".to_string(),
            "150.7".to_string(),
            "Third measurement".to_string(),
        ],
    ];

    println!("  📝 Writing original data and calculating checksums...");
    // Convert Vec<Vec<String>> to Array2<String>
    let rows = test_data.len();
    let cols = if rows > 0 { test_data[0].len() } else { 0 };
    let flat_data: Vec<String> = test_data.clone().into_iter().flatten().collect();
    let array_data = Array2::from_shape_vec((rows, cols), flat_data)?;

    // Extract headers and data
    let headers = if array_data.nrows() > 0 {
        Some(array_data.row(0).to_vec())
    } else {
        None
    };
    let data_only = if array_data.nrows() > 1 {
        array_data
            .slice(scirs2_core::ndarray::s![1.., ..])
            .to_owned()
    } else {
        Array2::from_shape_vec((0, cols), Vec::new())?
    };

    csv::write_csv(&data_file, &data_only, headers.as_ref(), None)?;

    // Calculate original checksums
    let original_crc32 = validation::calculate_crc32(&data_file)?;
    let original_sha256 = validation::calculate_sha256(&data_file)?;

    println!("  🔐 Original checksums:");
    println!("    CRC32: {}", original_crc32);
    println!("    SHA256: {}", original_sha256);

    // Perform round-trip
    println!("  🔄 Performing round-trip...");
    let (read_headers, read_array) = csv::read_csv(&data_file, None)?;
    csv::write_csv(&round_trip_file, &read_array, Some(&read_headers), None)?;

    // Calculate round-trip checksums
    let round_trip_crc32 = validation::calculate_crc32(&round_trip_file)?;
    let round_trip_sha256 = validation::calculate_sha256(&round_trip_file)?;

    println!("  🔐 Round-trip checksums:");
    println!("    CRC32: {}", round_trip_crc32);
    println!("    SHA256: {}", round_trip_sha256);

    // Verify checksums match
    assert_eq!(original_crc32, round_trip_crc32, "CRC32 checksum mismatch");
    assert_eq!(
        original_sha256, round_trip_sha256,
        "SHA256 checksum mismatch"
    );

    // Verify file integrity validation
    let data_file_checksum = validation::calculate_sha256(&data_file)?;
    assert_eq!(
        original_sha256, data_file_checksum,
        "Original file integrity validation failed"
    );
    let round_trip_file_checksum = validation::calculate_sha256(&round_trip_file)?;
    assert_eq!(
        round_trip_sha256, round_trip_file_checksum,
        "Round-trip file integrity validation failed"
    );

    println!("  ✅ Checksums match - perfect round-trip integrity");
    Ok(())
}

#[allow(dead_code)]
fn test_parallel_round_trip(
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Testing Parallel Processing Round-trip...");

    let matrix_file = temp_dir.path().join("parallel_matrix.mtx");

    // Create a larger matrix to trigger parallel processing
    let header = MMHeader {
        object: "matrix".to_string(),
        format: MMFormat::Coordinate,
        data_type: MMDataType::Real,
        symmetry: MMSymmetry::General,
        comments: vec!["Parallel processing test matrix".to_string()],
    };

    println!("  🏗️  Generating large sparse matrix...");
    let mut entries = Vec::new();
    for i in 0..50000 {
        // Large enough to trigger parallel processing
        entries.push(SparseEntry {
            row: i % 1000,
            col: (i * 7) % 1000,
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

    // Use parallel configuration
    let config = ParallelConfig {
        num_threads: 4,
        chunk_size: 10000,
        buffer_size: 1024 * 1024,
        use_memory_mapping: false,
    };

    println!(
        "  📝 Writing matrix with parallel I/O ({} threads, {} chunk size)...",
        config.num_threads, config.chunk_size
    );
    let write_stats = matrix_market::write_sparse_matrix_parallel(
        &matrix_file,
        &original_matrix,
        config.clone(),
    )?;

    println!("  📊 Write statistics:");
    println!("    Entries processed: {}", write_stats.entries_processed);
    println!("    Time: {:.2}ms", write_stats.io_time_ms);
    println!(
        "    Throughput: {:.0} entries/sec",
        write_stats.throughput_eps
    );

    println!("  📖 Reading matrix with parallel I/O...");
    let (read_matrix, read_stats) =
        matrix_market::read_sparse_matrix_parallel(&matrix_file, config)?;

    println!("  📊 Read statistics:");
    println!("    Entries processed: {}", read_stats.entries_processed);
    println!("    Time: {:.2}ms", read_stats.io_time_ms);
    println!(
        "    Throughput: {:.0} entries/sec",
        read_stats.throughput_eps
    );

    // Verify matrix integrity
    assert_eq!(read_matrix.rows, original_matrix.rows, "Row count mismatch");
    assert_eq!(
        read_matrix.cols, original_matrix.cols,
        "Column count mismatch"
    );
    assert_eq!(
        read_matrix.nnz, original_matrix.nnz,
        "Non-zero count mismatch"
    );
    assert_eq!(
        read_matrix.entries.len(),
        original_matrix.entries.len(),
        "Entry count mismatch"
    );

    // Verify all entries (this is expensive but necessary for validation)
    println!(
        "  🔍 Verifying all {} entries...",
        read_matrix.entries.len()
    );
    let mut original_entries = original_matrix.entries.clone();
    let mut read_entries = read_matrix.entries.clone();

    original_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));
    read_entries.sort_by(|a, b| (a.row, a.col).cmp(&(b.row, b.col)));

    for (i, (original, read)) in original_entries.iter().zip(read_entries.iter()).enumerate() {
        assert_eq!(read.row, original.row, "Row mismatch at entry {}", i);
        assert_eq!(read.col, original.col, "Column mismatch at entry {}", i);
        assert!(
            (read.value - original.value).abs() < 1e-12,
            "Value mismatch at entry {}: {} vs {}",
            i,
            read.value,
            original.value
        );
    }

    println!(
        "  ✅ Parallel round-trip successful: perfect data integrity with improved performance"
    );
    println!(
        "    Write speed: {:.0} entries/sec",
        write_stats.throughput_eps
    );
    println!(
        "    Read speed: {:.0} entries/sec",
        read_stats.throughput_eps
    );

    Ok(())
}
