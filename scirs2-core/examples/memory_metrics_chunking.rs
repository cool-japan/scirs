//! # Memory Metrics with TrackedChunkProcessor
//!
//! This example demonstrates how to use the TrackedChunkProcessor to track
//! memory usage during chunk-based processing of large arrays.

#[cfg(not(feature = "memory_management"))]
fn main() {
    println!("This example requires the 'memory_management' feature to be enabled.");
    println!("Run with: cargo run --example memory_metrics_chunking --features memory_management");
}

#[cfg(feature = "memory_management")]
use ndarray::Array2;
#[cfg(feature = "memory_management")]
use scirs2_core::memory::metrics::{
    format_bytes, format_memory_report, generate_memory_report, reset_memory_metrics,
    TrackedChunkProcessor2D,
};
#[cfg(feature = "memory_management")]
use std::time::Instant;

// Structure to share data across chunks
#[cfg(feature = "memory_management")]
struct SharedData {
    total_sum: f64,
    total_count: usize,
    min_value: f64,
    max_value: f64,
}

#[cfg(feature = "memory_management")]
fn main() {
    println!("Memory Metrics with TrackedChunkProcessor Example");
    println!("=================================================\n");

    // Reset metrics to start fresh
    reset_memory_metrics();

    // Create a large 2D array (50MB)
    let size = 2500;
    println!("Creating a {}x{} array of f64 values...", size, size);
    let start = Instant::now();
    let large_array = Array2::<f64>::zeros((size, size));
    let creation_time = start.elapsed();
    println!("Array created in {:?}", creation_time);

    // Calculate memory size
    let total_size = large_array.len() * std::mem::size_of::<f64>();
    println!("Total array size: {}", format_bytes(total_size));

    // Process array with standard chunking (no tracking)
    println!("\nPerforming standard chunk processing (no memory tracking)");
    basic_chunking(&large_array);

    // Process array with tracked chunking
    println!("\nPerforming tracked chunk processing");
    tracked_chunking(&large_array);

    // Print memory report
    println!("\nFinal Memory Report:");
    println!("{}", format_memory_report());
}

// Function to process array with standard chunking
#[cfg(feature = "memory_management")]
fn basic_chunking(array: &Array2<f64>) {
    let start = Instant::now();
    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..array.nrows() {
        for j in 0..array.ncols() {
            sum += array[[i, j]];
            count += 1;
        }
    }

    let average = sum / count as f64;
    let elapsed = start.elapsed();
    println!("Standard processing complete in {:?}", elapsed);
    println!("Average value: {}", average);
}

// Function to process array with tracked chunking
#[cfg(feature = "memory_management")]
fn tracked_chunking(array: &Array2<f64>) {
    let start = Instant::now();

    // Define chunk size (500x500 chunks = ~2MB per chunk)
    let chunk_size = (500, 500);
    println!("Using chunk size: {}x{}", chunk_size.0, chunk_size.1);

    // Create a tracked chunk processor
    let mut processor = TrackedChunkProcessor2D::new(array, chunk_size, "ArrayChunking");

    // Variables to track global statistics
    let mut shared_data = SharedData {
        total_sum: 0.0,
        total_count: 0,
        min_value: f64::MAX,
        max_value: f64::MIN,
    };

    // Process the array in chunks
    processor.process_chunks(|chunk, coords| {
        println!("  Processing chunk at {:?}", coords);

        // Calculate statistics for this chunk
        let mut chunk_sum = 0.0;
        let mut chunk_count = 0;
        let mut chunk_min = f64::MAX;
        let mut chunk_max = f64::MIN;

        for r in 0..chunk.nrows() {
            for c in 0..chunk.ncols() {
                let value = chunk[[r, c]];
                chunk_sum += value;
                chunk_count += 1;

                if value < chunk_min {
                    chunk_min = value;
                }

                if value > chunk_max {
                    chunk_max = value;
                }
            }
        }

        // Update global statistics
        shared_data.total_sum += chunk_sum;
        shared_data.total_count += chunk_count;

        if chunk_min < shared_data.min_value {
            shared_data.min_value = chunk_min;
        }

        if chunk_max > shared_data.max_value {
            shared_data.max_value = chunk_max;
        }

        // Memory usage report for this chunk
        let report = generate_memory_report();
        println!(
            "    Current memory: {}",
            format_bytes(report.total_current_usage)
        );
        println!("    Peak memory: {}", format_bytes(report.total_peak_usage));
    });

    // Calculate global statistics
    let average = if shared_data.total_count > 0 {
        shared_data.total_sum / shared_data.total_count as f64
    } else {
        0.0
    };
    let elapsed = start.elapsed();

    println!("\nTracked processing complete in {:?}", elapsed);
    println!("Average value: {}", average);
    println!("Min value: {}", shared_data.min_value);
    println!("Max value: {}", shared_data.max_value);
}
