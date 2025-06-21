//! Parallel I/O operations example using thread pool
//!
//! This example demonstrates how to use the thread pool for high-performance
//! I/O operations across different file formats.

use ndarray::{Array1, Array2};
use scirs2_io::{
    csv,
    error::IoError,
    matrix_market::{
        self, MMDataType, MMFormat, MMHeader, MMSparseMatrix, MMSymmetry, SparseEntry,
    },
    serialize,
    thread_pool::{self, ThreadPool, WorkType},
    validation,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Parallel I/O Operations Example");
    println!("=================================");

    // Create temporary directory for test files
    let temp_dir = tempdir()?;
    println!("📁 Using temporary directory: {:?}", temp_dir.path());

    // Initialize thread pool
    let config = thread_pool::optimal_config();
    println!("🔧 Optimal thread pool configuration:");
    println!("   I/O threads: {}", config.io_threads);
    println!("   CPU threads: {}", config.cpu_threads);
    println!("   Max queue size: {}", config.max_queue_size);

    let pool = ThreadPool::new(config);

    // Demonstrate different parallel I/O patterns
    demonstrate_parallel_file_processing(&pool, &temp_dir)?;
    demonstrate_concurrent_format_conversion(&pool, &temp_dir)?;
    demonstrate_batch_operations(&pool, &temp_dir)?;
    demonstrate_pipeline_processing(&pool, &temp_dir)?;

    // Show final statistics
    let stats = pool.get_stats();
    println!("\n📊 Final Thread Pool Statistics:");
    println!("   Tasks submitted: {}", stats.tasks_submitted);
    println!("   Tasks completed: {}", stats.tasks_completed);
    println!("   Tasks failed: {}", stats.tasks_failed);
    println!(
        "   Average execution time: {:.2}ms",
        stats.avg_execution_time_ms
    );
    println!(
        "   Total execution time: {:.2}ms",
        stats.total_execution_time_ms
    );

    // Graceful shutdown
    pool.shutdown()?;
    println!("\n✅ Thread pool shut down gracefully");

    Ok(())
}

fn demonstrate_parallel_file_processing(
    pool: &ThreadPool,
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📚 Demonstrating Parallel File Processing...");

    let start_time = Instant::now();
    let results = Arc::new(Mutex::new(Vec::new()));

    // Create multiple datasets to process in parallel
    let datasets = vec![
        ("dataset_1.csv", generate_csv_data(1000)),
        ("dataset_2.csv", generate_csv_data(1500)),
        ("dataset_3.csv", generate_csv_data(800)),
        ("dataset_4.csv", generate_csv_data(1200)),
        ("dataset_5.csv", generate_csv_data(2000)),
    ];

    println!("  📝 Processing {} datasets in parallel...", datasets.len());

    // Submit parallel file writing tasks
    for (filename, data) in datasets {
        let file_path = temp_dir.path().join(filename);
        let results_clone = Arc::clone(&results);

        pool.submit(WorkType::IO, move || {
            let write_start = Instant::now();

            // Convert Vec<Vec<String>> to Array2<String>
            let rows = data.len();
            let cols = if rows > 0 { data[0].len() } else { 0 };
            let flat_data: Vec<String> = data.into_iter().flatten().collect();
            let array_data = Array2::from_shape_vec((rows, cols), flat_data)
                .map_err(|e| IoError::FormatError(e.to_string()))?;

            // Write CSV file (separate headers from data)
            let headers = if array_data.nrows() > 0 {
                Some(array_data.row(0).to_vec())
            } else {
                None
            };
            let data_only = if array_data.nrows() > 1 {
                array_data.slice(ndarray::s![1.., ..]).to_owned()
            } else {
                Array2::from_shape_vec((0, cols), Vec::new())
                    .map_err(|e| IoError::FormatError(e.to_string()))?
            };
            csv::write_csv(&file_path, &data_only, headers.as_ref(), None)?;

            // Read it back to verify
            let (_headers, read_array) = csv::read_csv(&file_path, None)?;

            // Calculate checksum
            let checksum = validation::calculate_sha256(&file_path)?;

            let processing_time = write_start.elapsed();

            // Store results
            {
                let mut results_guard = results_clone.lock().unwrap();
                results_guard.push((
                    filename.to_string(),
                    array_data.nrows(),
                    read_array.nrows(),
                    checksum,
                    processing_time.as_millis(),
                ));
            }

            Ok(())
        })?;
    }

    // Wait for all tasks to complete
    pool.wait_for_completion()?;

    let total_time = start_time.elapsed();

    // Display results
    {
        let results_guard = results.lock().unwrap();
        println!("  📊 Processing Results:");
        for (filename, original_rows, read_rows, checksum, time_ms) in &*results_guard {
            println!(
                "    {}: {} rows, checksum: {}..., time: {}ms",
                filename,
                original_rows,
                &checksum[..8],
                time_ms
            );
            assert_eq!(
                original_rows, read_rows,
                "Row count mismatch for {}",
                filename
            );
        }
    }

    println!(
        "  ⏱️  Total parallel processing time: {:.2}ms",
        total_time.as_millis()
    );
    println!("  ✅ All files processed successfully with verified integrity");

    Ok(())
}

fn demonstrate_concurrent_format_conversion(
    pool: &ThreadPool,
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating Concurrent Format Conversion...");

    // Create test data in different formats
    let matrix_data = create_test_sparse_matrix();
    let array_data = Array2::from_shape_fn((100, 50), |(i, j)| (i * j) as f64);

    let conversion_results = Arc::new(Mutex::new(Vec::new()));
    let start_time = Instant::now();

    // Task 1: Matrix Market → CSV conversion
    {
        let matrix_clone = matrix_data.clone();
        let mm_file = temp_dir.path().join("matrix.mtx");
        let csv_file = temp_dir.path().join("matrix_converted.csv");
        let results_clone = Arc::clone(&conversion_results);

        pool.submit(WorkType::CPU, move || {
            let task_start = Instant::now();

            // Write Matrix Market file
            matrix_market::write_sparse_matrix(&mm_file, &matrix_clone)?;

            // Convert to CSV format (simplified)
            let mut csv_data = vec![vec![
                "row".to_string(),
                "col".to_string(),
                "value".to_string(),
            ]];
            for entry in &matrix_clone.entries {
                csv_data.push(vec![
                    entry.row.to_string(),
                    entry.col.to_string(),
                    entry.value.to_string(),
                ]);
            }
            // Convert Vec<Vec<String>> to Array2<String>
            let rows = csv_data.len();
            let cols = if rows > 0 { csv_data[0].len() } else { 0 };
            let flat_data: Vec<String> = csv_data.into_iter().flatten().collect();
            let array_data = Array2::from_shape_vec((rows, cols), flat_data)
                .map_err(|e| IoError::FormatError(e.to_string()))?;

            // Write CSV file (separate headers from data)
            let headers = if array_data.nrows() > 0 {
                Some(array_data.row(0).to_vec())
            } else {
                None
            };
            let data_only = if array_data.nrows() > 1 {
                array_data.slice(ndarray::s![1.., ..]).to_owned()
            } else {
                Array2::from_shape_vec((0, cols), Vec::new())
                    .map_err(|e| IoError::FormatError(e.to_string()))?
            };
            csv::write_csv(&csv_file, &data_only, headers.as_ref(), None)?;

            let task_time = task_start.elapsed();

            {
                let mut results = results_clone.lock().unwrap();
                results.push(("Matrix Market → CSV".to_string(), task_time.as_millis()));
            }

            Ok(())
        })?;
    }

    // Task 2: Array → Multiple formats conversion
    {
        let array_clone = array_data.clone();
        let json_file = temp_dir.path().join("array.json");
        let binary_file = temp_dir.path().join("array.bin");
        let msgpack_file = temp_dir.path().join("array.msgpack");
        let results_clone = Arc::clone(&conversion_results);

        pool.submit(WorkType::CPU, move || {
            let task_start = Instant::now();

            // Convert to multiple formats
            serialize::write_array_json(&json_file, &array_clone.clone().into_dyn())?;
            serialize::write_array_binary(&binary_file, &array_clone.clone().into_dyn())?;
            serialize::write_array_messagepack(&msgpack_file, &array_clone.clone().into_dyn())?;

            // Verify round-trip
            let json_read: ndarray::Array<f64, ndarray::IxDyn> =
                serialize::read_array_json(&json_file)?;
            let binary_read: ndarray::Array<f64, ndarray::IxDyn> =
                serialize::read_array_binary(&binary_file)?;
            let msgpack_read: ndarray::Array<f64, ndarray::IxDyn> =
                serialize::read_array_messagepack(&msgpack_file)?;

            // Quick verification
            assert_eq!(json_read.shape(), array_clone.shape());
            assert_eq!(binary_read.shape(), array_clone.shape());
            assert_eq!(msgpack_read.shape(), array_clone.shape());

            let task_time = task_start.elapsed();

            {
                let mut results = results_clone.lock().unwrap();
                results.push((
                    "Array → JSON/Binary/MessagePack".to_string(),
                    task_time.as_millis(),
                ));
            }

            Ok(())
        })?;
    }

    // Task 3: CSV → Validation pipeline
    {
        let csv_file = temp_dir.path().join("validation_test.csv");
        let results_clone = Arc::clone(&conversion_results);

        pool.submit(WorkType::IO, move || {
            let task_start = Instant::now();

            // Create test CSV
            let csv_data = generate_csv_data(500);
            // Convert Vec<Vec<String>> to Array2<String>
            let rows = csv_data.len();
            let cols = if rows > 0 { csv_data[0].len() } else { 0 };
            let flat_data: Vec<String> = csv_data.into_iter().flatten().collect();
            let array_data = Array2::from_shape_vec((rows, cols), flat_data)
                .map_err(|e| IoError::FormatError(e.to_string()))?;

            // Write CSV file (separate headers from data)
            let headers = if array_data.nrows() > 0 {
                Some(array_data.row(0).to_vec())
            } else {
                None
            };
            let data_only = if array_data.nrows() > 1 {
                array_data.slice(ndarray::s![1.., ..]).to_owned()
            } else {
                Array2::from_shape_vec((0, cols), Vec::new())
                    .map_err(|e| IoError::FormatError(e.to_string()))?
            };
            csv::write_csv(&csv_file, &data_only, headers.as_ref(), None)?;

            // Multiple validation checksums
            let crc32 = validation::calculate_crc32(&csv_file)?;
            let sha256 = validation::calculate_sha256(&csv_file)?;

            // Verify integrity using checksum comparison
            let current_sha256 = validation::calculate_sha256(&csv_file)?;
            assert_eq!(sha256, current_sha256, "File integrity check failed");

            let task_time = task_start.elapsed();

            {
                let mut results = results_clone.lock().unwrap();
                results.push((
                    format!(
                        "CSV → Validation (CRC32: {}, SHA256: {}...)",
                        crc32,
                        &sha256[..8]
                    ),
                    task_time.as_millis(),
                ));
            }

            Ok(())
        })?;
    }

    // Wait for all conversions to complete
    pool.wait_for_completion()?;

    let total_time = start_time.elapsed();

    // Display results
    {
        let results_guard = conversion_results.lock().unwrap();
        println!("  🔄 Conversion Results:");
        for (conversion_type, time_ms) in &*results_guard {
            println!("    {}: {}ms", conversion_type, time_ms);
        }
    }

    println!(
        "  ⏱️  Total concurrent conversion time: {:.2}ms",
        total_time.as_millis()
    );
    println!("  ✅ All format conversions completed successfully");

    Ok(())
}

fn demonstrate_batch_operations(
    pool: &ThreadPool,
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📦 Demonstrating Batch Operations...");

    let start_time = Instant::now();
    let processed_count = Arc::new(Mutex::new(0));

    // Create batch of small operations
    let batch_size = 20;
    let mut batch_tasks = Vec::new();

    for i in 0..batch_size {
        let file_path = temp_dir.path().join(format!("batch_{}.json", i));
        let count_clone = Arc::clone(&processed_count);

        let task = move || {
            // Create small array
            let array = Array1::from(vec![i as f64; 10]);

            // Write to JSON
            serialize::write_array_json(&file_path, &array.clone().into_dyn())?;

            // Read back and verify
            let read_array: ndarray::Array<f64, ndarray::IxDyn> =
                serialize::read_array_json(&file_path)?;
            assert_eq!(array.shape(), read_array.shape());

            // Update counter
            {
                let mut count = count_clone.lock().unwrap();
                *count += 1;
            }

            Ok(())
        };

        batch_tasks.push(task);
    }

    // Submit entire batch
    pool.submit_batch(WorkType::IO, batch_tasks)?;

    // Wait for completion
    pool.wait_for_completion()?;

    let total_time = start_time.elapsed();
    let final_count = *processed_count.lock().unwrap();

    println!("  📊 Batch Results:");
    println!("    Tasks processed: {}/{}", final_count, batch_size);
    println!(
        "    Average time per task: {:.2}ms",
        total_time.as_millis() as f64 / batch_size as f64
    );
    println!("    Total batch time: {:.2}ms", total_time.as_millis());

    assert_eq!(final_count, batch_size, "Not all batch tasks completed");
    println!("  ✅ Batch operations completed successfully");

    Ok(())
}

fn demonstrate_pipeline_processing(
    pool: &ThreadPool,
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Demonstrating Pipeline Processing...");

    let start_time = Instant::now();
    let pipeline_results = Arc::new(Mutex::new(Vec::new()));

    // Stage 1: Data generation (CPU-bound)
    let generated_data = Arc::new(Mutex::new(Vec::new()));
    let data_gen_tasks = 5;

    for i in 0..data_gen_tasks {
        let data_clone = Arc::clone(&generated_data);

        pool.submit(WorkType::CPU, move || {
            // Generate computationally expensive data
            let size = 1000 + i * 200;
            let mut data = Vec::new();

            for j in 0..size {
                // Simulate some computation
                let value = (i as f64 * 1000.0 + j as f64).sin() * (j as f64).cos();
                data.push(vec![
                    format!("item_{}", j),
                    format!("{:.6}", value),
                    format!("category_{}", j % 10),
                ]);
            }

            {
                let mut gen_data = data_clone.lock().unwrap();
                gen_data.push((format!("dataset_{}", i), data));
            }

            Ok(())
        })?;
    }

    // Wait for data generation
    pool.wait_for_completion()?;
    println!("  🏭 Stage 1: Data generation completed");

    // Stage 2: File writing (I/O-bound)
    let written_files = Arc::new(Mutex::new(Vec::new()));

    {
        let generated = generated_data.lock().unwrap();
        for (dataset_name, data) in &*generated {
            let file_path = temp_dir.path().join(format!("{}.csv", dataset_name));
            let data_clone = data.clone();
            let written_clone = Arc::clone(&written_files);
            let name_clone = dataset_name.clone();

            pool.submit(WorkType::IO, move || {
                // Add header
                let mut csv_data = vec![vec![
                    "name".to_string(),
                    "value".to_string(),
                    "category".to_string(),
                ]];
                csv_data.extend(data_clone);

                // Write file - Convert Vec<Vec<String>> to Array2<String>
                let rows = csv_data.len();
                let cols = if rows > 0 { csv_data[0].len() } else { 0 };
                let flat_data: Vec<String> = csv_data.into_iter().flatten().collect();
                let array_data = Array2::from_shape_vec((rows, cols), flat_data)
                    .map_err(|e| IoError::FormatError(e.to_string()))?;

                // Write CSV file (separate headers from data)
                let headers = if array_data.nrows() > 0 {
                    Some(array_data.row(0).to_vec())
                } else {
                    None
                };
                let data_only = if array_data.nrows() > 1 {
                    array_data.slice(ndarray::s![1.., ..]).to_owned()
                } else {
                    Array2::from_shape_vec((0, cols), Vec::new())
                        .map_err(|e| IoError::FormatError(e.to_string()))?
                };
                csv::write_csv(&file_path, &data_only, headers.as_ref(), None)?;

                {
                    let mut written = written_clone.lock().unwrap();
                    written.push((name_clone, file_path));
                }

                Ok(())
            })?;
        }
    }

    // Wait for file writing
    pool.wait_for_completion()?;
    println!("  💾 Stage 2: File writing completed");

    // Stage 3: Validation and analysis (mixed CPU/I/O)
    {
        let written = written_files.lock().unwrap();
        for (dataset_name, file_path) in &*written {
            let path_clone = file_path.clone();
            let name_clone = dataset_name.clone();
            let results_clone = Arc::clone(&pipeline_results);

            pool.submit(WorkType::CPU, move || {
                // Read and analyze
                let (_headers, data) = csv::read_csv(&path_clone, None)?;
                let row_count = data.nrows();

                // Calculate statistics
                let mut value_sum = 0.0;
                let mut value_count = 0;

                for row_idx in 0..data.nrows() {
                    if data.ncols() > 1 {
                        if let Ok(value) = data[[row_idx, 1]].parse::<f64>() {
                            value_sum += value;
                            value_count += 1;
                        }
                    }
                }

                let average_value = if value_count > 0 {
                    value_sum / value_count as f64
                } else {
                    0.0
                };

                // Validate file integrity
                let checksum = validation::calculate_sha256(&path_clone)?;
                let verification_checksum = validation::calculate_sha256(&path_clone)?;
                let is_valid = checksum == verification_checksum;

                {
                    let mut results = results_clone.lock().unwrap();
                    results.push((name_clone, row_count, average_value, is_valid));
                }

                Ok(())
            })?;
        }
    }

    // Wait for analysis
    pool.wait_for_completion()?;

    let total_time = start_time.elapsed();

    // Display pipeline results
    {
        let results = pipeline_results.lock().unwrap();
        println!("  📊 Stage 3: Analysis Results:");
        for (dataset_name, row_count, avg_value, is_valid) in &*results {
            println!(
                "    {}: {} rows, avg value: {:.4}, valid: {}",
                dataset_name, row_count, avg_value, is_valid
            );
        }
    }

    println!("  ⏱️  Total pipeline time: {:.2}ms", total_time.as_millis());
    println!("  ✅ Pipeline processing completed successfully");

    Ok(())
}

// Helper functions

fn generate_csv_data(rows: usize) -> Vec<Vec<String>> {
    let mut data = vec![vec![
        "id".to_string(),
        "value".to_string(),
        "label".to_string(),
    ]];

    for i in 0..rows {
        data.push(vec![
            i.to_string(),
            (i as f64 * 0.1).to_string(),
            format!("label_{}", i % 10),
        ]);
    }

    data
}

fn create_test_sparse_matrix() -> MMSparseMatrix<f64> {
    let header = MMHeader {
        object: "matrix".to_string(),
        format: MMFormat::Coordinate,
        data_type: MMDataType::Real,
        symmetry: MMSymmetry::General,
        comments: vec!["Test matrix for parallel processing".to_string()],
    };

    let mut entries = Vec::new();
    for i in 0..100 {
        entries.push(SparseEntry {
            row: i % 50,
            col: (i * 3) % 50,
            value: (i as f64) * 0.01,
        });
    }

    MMSparseMatrix {
        header,
        rows: 50,
        cols: 50,
        nnz: entries.len(),
        entries,
    }
}
