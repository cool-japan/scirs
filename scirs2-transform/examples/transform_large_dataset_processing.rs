//! Example demonstrating out-of-core and streaming processing for large datasets
//!
//! This example shows how to:
//! 1. Process datasets too large to fit in memory
//! 2. Handle streaming data in real-time
//! 3. Use windowed transformations

use scirs2_core::ndarray::{array, Array2};
use scirs2_transform::{
    normalize::NormalizationMethod,
    out_of_core::{csv_chunks, OutOfCoreNormalizer, OutOfCoreTransformer},
    streaming::{
        StreamingMinMaxScaler, StreamingQuantileTracker, StreamingStandardScaler,
        StreamingTransformer, WindowedStreamingTransformer,
    },
};
use std::fs::File;
use std::io::Write;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Large Dataset Processing Examples ===\n");

    // Example 1: Out-of-core normalization
    example_out_of_core_processing()?;

    // Example 2: Streaming transformations
    example_streaming_transformations()?;

    // Example 3: Windowed streaming
    example_windowed_streaming()?;

    // Example 4: Streaming quantile tracking
    example_streaming_quantiles()?;

    Ok(())
}

/// Demonstrate out-of-core processing on a large CSV file
#[allow(dead_code)]
fn example_out_of_core_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Out-of-Core Processing Example");
    println!("---------------------------------");

    // Create a sample large CSV file
    let csv_path = "/tmp/large_dataset.csv";
    create_sample_csv(csv_path, 10000, 50)?;
    println!("Created sample CSV with 10,000 rows and 50 columns");

    // Create out-of-core normalizer
    let mut normalizer = OutOfCoreNormalizer::new(NormalizationMethod::ZScore);

    // Fit on chunks
    println!("Fitting normalizer on data chunks...");
    let chunks = csv_chunks(csv_path, 1000, false)?;
    normalizer.fit_chunks(chunks)?;

    // Transform data in chunks
    println!("Transforming data in chunks...");
    let chunks = csv_chunks(csv_path, 1000, false)?;
    let writer = normalizer.transform_chunks(chunks)?;

    let output_path = writer.finalize()?;
    println!("Transformed data saved to: {}", output_path);

    // Clean up
    std::fs::remove_file(csv_path)?;
    std::fs::remove_file(output_path)?;

    println!();
    Ok(())
}

/// Demonstrate streaming transformations
#[allow(dead_code)]
fn example_streaming_transformations() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Streaming Transformations Example");
    println!("-----------------------------------");

    let n_features = 5;
    let mut scaler = StreamingStandardScaler::new(n_features, true, true);

    // Simulate streaming data
    println!("Processing streaming batches:");
    for i in 0..5 {
        // Generate a batch of data
        let batch = Array2::from_shape_fn((100, n_features), |(_, j)| {
            (i as f64 + j as f64) * 10.0 + scirs2_core::random::random::<f64>() * 5.0
        });

        // Update scaler
        scaler.partial_fit(&batch)?;

        println!(
            "  Batch {}: {} samples processed",
            i + 1,
            scaler.n_samples_seen()
        );

        // Transform some test data
        let test_data = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
        let transformed = scaler.transform(&test_data)?;
        println!("  Test data transformed: {:?}", transformed.row(0));
    }

    println!("\nFinal statistics:");
    println!("  Total samples seen: {}", scaler.n_samples_seen());

    println!();
    Ok(())
}

/// Demonstrate windowed streaming transformations
#[allow(dead_code)]
fn example_windowed_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Windowed Streaming Example");
    println!("-----------------------------");

    let n_features = 3;
    let window_size = 500;

    // Create windowed min-max scaler
    let base_scaler = StreamingMinMaxScaler::new(n_features, (0.0, 1.0));
    let mut windowed_scaler = WindowedStreamingTransformer::new(base_scaler, window_size);

    println!(
        "Processing data with sliding window of {} samples:",
        window_size
    );

    // Simulate concept drift
    for epoch in 0..3 {
        println!("\nEpoch {}: Data distribution shift", epoch + 1);

        for batch_idx in 0..5 {
            // Generate data with shifting distribution
            let offset = epoch as f64 * 50.0;
            let batch = Array2::from_shape_fn((100, n_features), |(_, j)| {
                offset + j as f64 * 10.0 + scirs2_core::random::random::<f64>() * 20.0
            });

            // Update windowed scaler
            windowed_scaler.update(&batch)?;

            // Transform test data
            let test_data = array![[offset + 5.0, offset + 15.0, offset + 25.0]];
            let transformed = windowed_scaler.transform(&test_data)?;

            if batch_idx == 0 {
                println!("  Transformed test data: {:?}", transformed.row(0));
            }
        }
    }

    println!();
    Ok(())
}

/// Demonstrate streaming quantile tracking
#[allow(dead_code)]
fn example_streaming_quantiles() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Streaming Quantile Tracking Example");
    println!("-------------------------------------");

    let n_features = 2;
    let quantiles = vec![0.25, 0.5, 0.75]; // Q1, median, Q3

    let mut tracker = StreamingQuantileTracker::new(n_features, quantiles.clone())?;

    println!("Tracking quantiles: {:?}", quantiles);

    // Process streaming data
    let n_batches = 10;
    for i in 0..n_batches {
        // Generate batch with known distribution
        let batch = Array2::from_shape_fn((1000, n_features), |(_, j)| {
            // Feature 0: Normal distribution
            // Feature 1: Uniform distribution
            if j == 0 {
                // Box-Muller transform for normal distribution
                let u1 = scirs2_core::random::random::<f64>();
                let u2 = scirs2_core::random::random::<f64>();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 10.0 + 50.0
            } else {
                scirs2_core::random::random::<f64>() * 100.0
            }
        });

        tracker.update(&batch)?;

        if i == n_batches - 1 {
            let estimated_quantiles = tracker.get_quantiles();
            println!("\nEstimated quantiles after {} samples:", (i + 1) * 1000);
            println!("Feature 0 (Normal, μ=50, σ=10):");
            println!("  Q1 (25%): {:.2}", estimated_quantiles[[0, 0]]);
            println!("  Median:   {:.2}", estimated_quantiles[[0, 1]]);
            println!("  Q3 (75%): {:.2}", estimated_quantiles[[0, 2]]);

            println!("\nFeature 1 (Uniform [0, 100]):");
            println!("  Q1 (25%): {:.2}", estimated_quantiles[[1, 0]]);
            println!("  Median:   {:.2}", estimated_quantiles[[1, 1]]);
            println!("  Q3 (75%): {:.2}", estimated_quantiles[[1, 2]]);

            // Compare with theoretical values
            println!("\nTheoretical values:");
            println!("Normal: Q1=43.26, Median=50.00, Q3=56.74");
            println!("Uniform: Q1=25.00, Median=50.00, Q3=75.00");
        }
    }

    println!();
    Ok(())
}

/// Helper function to create a sample CSV file
#[allow(dead_code)]
fn create_sample_csv(
    path: &str,
    n_rows: usize,
    n_cols: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;

    for i in 0..n_rows {
        let row: Vec<String> = (0..n_cols)
            .map(|j| {
                format!(
                    "{:.2}",
                    (i + j) as f64 * 0.1 + scirs2_core::random::random::<f64>()
                )
            })
            .collect();
        writeln!(file, "{}", row.join(","))?;
    }

    Ok(())
}
