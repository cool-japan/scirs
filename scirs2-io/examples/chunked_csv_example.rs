use scirs2_core::ndarray::Array2;
use scirs2_io::csv::{read_csv_chunked, CsvReaderConfig, CsvWriterConfig};
use std::error::Error;
use std::time::{Duration, Instant};

/// This example demonstrates memory-efficient processing of large CSV files
/// using the chunked reading functionality.
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Chunked CSV Processing Example ===\n");

    // Create a large dataset for demonstration
    let rows = 10000;
    let cols = 5;
    println!(
        "Creating a large dataset ({} rows x {} columns)...",
        rows, cols
    );

    let start = Instant::now();
    let mut large_data = Array2::<String>::from_elem((rows, cols), String::new());

    for i in 0..rows {
        for j in 0..cols {
            large_data[[i, j]] = format!("Value_{}_{}", i, j);
        }
    }

    let headers = (0..cols)
        .map(|i| format!("Column{}", i))
        .collect::<Vec<_>>();

    // Write large dataset to a CSV file
    println!("Writing large dataset to CSV file...");
    let config = CsvWriterConfig {
        always_quote: true, // Ensure all fields are quoted for better test coverage
        ..Default::default()
    };

    scirs2_io::csv::write_csv(
        "scirs2-io/examples/large_dataset.csv",
        &large_data,
        Some(&headers),
        Some(config),
    )?;
    println!("Dataset created in {:.2?}", start.elapsed());

    // Process the large file in different chunk sizes and measure performance
    let chunk_sizes = [100, 500, 1000, 2000];

    for &chunk_size in &chunk_sizes {
        println!("\nProcessing with chunk size: {}", chunk_size);

        let start_time = Instant::now();
        let mut total_rows = 0;
        let mut total_chunks = 0;
        let reader_config = CsvReaderConfig::default();

        read_csv_chunked(
            "scirs2-io/examples/large_dataset.csv",
            Some(reader_config),
            chunk_size,
            |_, chunk| {
                total_chunks += 1;
                total_rows += chunk.shape()[0];

                // Simulate some processing time on each chunk
                std::thread::sleep(Duration::from_millis(5));

                true // continue processing
            },
        )?;

        let elapsed = start_time.elapsed();
        println!("Processed {} rows in {} chunks", total_rows, total_chunks);
        println!("Time taken: {:.2?}", elapsed);
        println!(
            "Average time per chunk: {:.2?}",
            elapsed / total_chunks as u32
        );
    }

    // Advanced example: performing operations on chunks
    println!("\nPerforming statistical analysis on chunks...");

    let mut total_rows = 0;
    let mut column_lengths = vec![0; cols];

    read_csv_chunked(
        "scirs2-io/examples/large_dataset.csv",
        None,
        1000,
        |_headers, chunk| {
            println!("Processing chunk with {} rows", chunk.shape()[0]);
            total_rows += chunk.shape()[0];

            // Calculate average string length in each column
            for j in 0..chunk.shape()[1] {
                for i in 0..chunk.shape()[0] {
                    column_lengths[j] += chunk[[i, j]].len();
                }
            }

            true // continue processing
        },
    )?;

    // Calculate and display average string length for each column
    println!("\nAverage string length by column:");
    for (i, &total_length) in column_lengths.iter().enumerate() {
        let avg_length = total_length as f64 / total_rows as f64;
        println!("Column {}: {:.2} characters", i, avg_length);
    }

    println!("\nChunked CSV example completed successfully!");
    Ok(())
}
