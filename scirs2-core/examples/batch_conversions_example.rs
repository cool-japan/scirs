//! # Batch Type Conversions Example
//!
//! This example demonstrates the high-performance batch type conversion capabilities
//! in SciRS2 Core, including SIMD acceleration and parallel processing.

use scirs2_core::batch_conversions::{utils::*, BatchConversionConfig, BatchConverter};
use scirs2_core::error::CoreResult;
use scirs2_core::Complex64;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> CoreResult<()> {
    println!("=== SciRS2 Core Batch Type Conversions Example ===\n");

    // Basic batch conversion
    demo_basic_conversion()?;

    // Error handling in batch conversions
    demoerror_handling()?;

    // SIMD-accelerated conversions
    demo_simd_conversions()?;

    // Parallel processing for large datasets
    demo_parallel_conversions()?;

    // Complex number conversions
    demo_complex_conversions()?;

    // Configuration and optimization
    demo_configuration_options()?;

    // Performance benchmarking
    demo_performance_benchmarks()?;

    // Integration with ndarray
    #[cfg(feature = "array")]
    demo_ndarray_integration()?;

    println!("\n=== Batch Type Conversions Example Complete ===");
    Ok(())
}

/// Demonstrate basic batch type conversions
#[allow(dead_code)]
fn demo_basic_conversion() -> CoreResult<()> {
    println!("1. Basic Batch Type Conversions");
    println!("================================");

    // Create test data
    let f64_data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.123).collect();

    // Create a batch converter with default settings
    let converter = BatchConverter::with_default_config();

    // Convert f64 to f32
    let f32_result: Vec<f32> = converter.convert_slice(&f64_data)?;
    println!("Converted {} f64 values to f32", f32_result.len());
    println!(
        "Sample: [{:.3}, {:.3}, {:.3}, ...]",
        f32_result[0], f32_result[1], f32_result[2]
    );

    // Convert with clamping for out-of-range values
    let large_data: Vec<f64> = vec![1e20, 2.5, -1e20, 100.0];
    let clamped_result: Vec<f32> = converter.convert_slice_clamped(&large_data);
    println!("Clamped conversion: {:?}", clamped_result);

    // Use utility functions for common conversions
    let i32_data: Vec<i32> = vec![1, 2, 3, 4, 5];
    let f32_from_i32 = i32_to_f32_batch(&i32_data);
    println!("i32 to f32: {:?}", f32_from_i32);

    println!();
    Ok(())
}

/// Demonstrate error handling in batch conversions
#[allow(dead_code)]
fn demoerror_handling() -> CoreResult<()> {
    println!("2. Error Handling in Batch Conversions");
    println!("======================================");

    let converter = BatchConverter::with_default_config();

    // Data with problematic values
    let problematic_data: Vec<f64> = vec![
        1.0,           // Valid
        f64::NAN,      // Invalid (NaN)
        3.4e38,        // Valid for f32
        3.4e39,        // Too large for f32
        f64::INFINITY, // Invalid (Infinity)
        2.5,           // Valid
    ];

    // Convert with error reporting
    let (converted, errors) = converter.convert_slice_witherrors::<f64, f32>(&problematic_data);

    println!("Input data: {} elements", problematic_data.len());
    println!("Successfully converted: {} elements", converted.len());
    println!("Errors encountered: {} elements", errors.len());

    for error in &errors {
        println!("  Error at index {}: {}", error.index, error.error);
    }

    println!("Converted values: {:?}", converted);

    println!();
    Ok(())
}

/// Demonstrate SIMD-accelerated conversions
#[allow(dead_code)]
fn demo_simd_conversions() -> CoreResult<()> {
    println!("3. SIMD-Accelerated Conversions");
    println!("===============================");

    // Create a large dataset for SIMD demonstration
    let large_data: Vec<f64> = (0..10000).map(|i| i as f64 * 0.001).collect();

    // Configure for SIMD-only (no parallel)
    let simd_config = BatchConversionConfig::default()
        .with_simd(true)
        .with_parallel(false);
    let simd_converter = BatchConverter::new(simd_config);

    // Time SIMD conversion
    let start = Instant::now();
    let simd_result: Vec<f32> = simd_converter.convert_slice(&large_data)?;
    let simd_time = start.elapsed();

    println!(
        "SIMD conversion of {} elements completed in {:?}",
        simd_result.len(),
        simd_time
    );

    // Configure for sequential (no SIMD, no parallel)
    let sequential_config = BatchConversionConfig::default()
        .with_simd(false)
        .with_parallel(false);
    let sequential_converter = BatchConverter::new(sequential_config);

    // Time sequential conversion
    let start = Instant::now();
    let sequential_result: Vec<f32> = sequential_converter.convert_slice(&large_data)?;
    let sequential_time = start.elapsed();

    println!(
        "Sequential conversion of {} elements completed in {:?}",
        sequential_result.len(),
        sequential_time
    );

    if simd_time < sequential_time {
        let speedup = sequential_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        println!("SIMD speedup: {:.2}x faster", speedup);
    }

    // Verify results are identical
    let differences = simd_result
        .iter()
        .zip(sequential_result.iter())
        .filter(|(&a, &b)| (a - b).abs() > f32::EPSILON)
        .count();
    println!("Differences between SIMD and sequential: {}", differences);

    println!();
    Ok(())
}

/// Demonstrate parallel processing for large datasets
#[allow(dead_code)]
fn demo_parallel_conversions() -> CoreResult<()> {
    println!("4. Parallel Processing for Large Datasets");
    println!("=========================================");

    // Create a very large dataset
    let huge_data: Vec<f64> = (0..100000).map(|i| i as f64 * 0.0001).collect();

    // Configure for parallel processing
    let parallel_config = BatchConversionConfig::default()
        .with_parallel(true)
        .with_chunk_size(1000)
        .with_parallel_threshold(10000);
    let parallel_converter = BatchConverter::new(parallel_config);

    // Time parallel conversion
    let start = Instant::now();
    let parallel_result: Vec<f32> = parallel_converter.convert_slice(&huge_data)?;
    let parallel_time = start.elapsed();

    println!(
        "Parallel conversion of {} elements completed in {:?}",
        parallel_result.len(),
        parallel_time
    );

    // Configure for sequential processing
    let sequential_config = BatchConversionConfig::default()
        .with_simd(false)
        .with_parallel(false);
    let sequential_converter = BatchConverter::new(sequential_config);

    // Time sequential conversion
    let start = Instant::now();
    let sequential_result: Vec<f32> = sequential_converter.convert_slice(&huge_data)?;
    let sequential_time = start.elapsed();

    println!(
        "Sequential conversion of {} elements completed in {:?}",
        sequential_result.len(),
        sequential_time
    );

    if parallel_time < sequential_time {
        let speedup = sequential_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
        println!("Parallel speedup: {:.2}x faster", speedup);
    }

    println!();
    Ok(())
}

/// Demonstrate complex number conversions
#[allow(dead_code)]
fn demo_complex_conversions() -> CoreResult<()> {
    println!("5. Complex Number Conversions");
    println!("=============================");

    let converter = BatchConverter::with_default_config();

    // Create complex data
    let complex_data: Vec<Complex64> = (0..1000)
        .map(|i| Complex64::new(i as f64 * 0.1, (i as f64 * 0.1).sin()))
        .collect();

    println!(
        "Converting {} complex numbers from f64 to f32",
        complex_data.len()
    );

    // Convert complex numbers
    let start = Instant::now();
    let result: Vec<scirs2_core::Complex32> = converter.convert_complex_slice(&complex_data)?;
    let conversion_time = start.elapsed();

    println!("Complex conversion completed in {:?}", conversion_time);
    println!("Sample results:");
    for (i, (orig, conv)) in complex_data.iter().zip(result.iter()).enumerate().take(3) {
        println!(
            "  {}: {:.3}+{:.3}i -> {:.3}+{:.3}i",
            i, orig.re, orig.im, conv.re, conv.im
        );
    }

    println!();
    Ok(())
}

/// Demonstrate configuration options
#[allow(dead_code)]
fn demo_configuration_options() -> CoreResult<()> {
    println!("6. Configuration Options");
    println!("=======================");

    // Custom configuration
    let custom_config = BatchConversionConfig::default()
        .with_simd(true)
        .with_parallel(true)
        .with_chunk_size(512)
        .with_parallel_threshold(5000);

    println!("Custom configuration:");
    println!("  SIMD enabled: {}", custom_config.use_simd);
    println!("  Parallel enabled: {}", custom_config.use_parallel);
    println!("  Chunk size: {}", custom_config.parallel_chunk_size);
    println!("  Parallel threshold: {}", custom_config.parallel_threshold);

    let converter = BatchConverter::new(custom_config);

    // Test with medium-sized data
    let data: Vec<f64> = (0..7500).map(|i| i as f64).collect();
    let result: Vec<f32> = converter.convert_slice(&data)?;

    println!(
        "Converted {} elements with custom configuration",
        result.len()
    );

    println!();
    Ok(())
}

/// Demonstrate performance benchmarking
#[allow(dead_code)]
fn demo_performance_benchmarks() -> CoreResult<()> {
    println!("7. Performance Benchmarking");
    println!("===========================");

    // Test data of various sizes
    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        println!("Benchmarking conversions for {} elements:", size);

        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();

        // Benchmark different methods
        let results = benchmark_conversion_methods::<f64, f32>(&data);

        for (method, duration) in results {
            println!("  {}: {:?}", method, duration);
        }

        println!();
    }

    Ok(())
}

/// Demonstrate integration with ndarray
#[cfg(feature = "array")]
#[allow(dead_code)]
fn demo_ndarray_integration() -> CoreResult<()> {
    println!("8. Integration with ndarray");
    println!("===========================");

    use scirs2_core::ndarray::Array2;
    let converter = BatchConverter::with_default_config();

    // Create a 2D array
    let array_f64 = Array2::<f64>::from_shape_fn((100, 50), |(i, j)| (i * j) as f64 * 0.01);

    println!(
        "Converting {}x{} ndarray from f64 to f32",
        array_f64.nrows(),
        array_f64.ncols()
    );

    // Convert the entire array
    let start = Instant::now();
    let array_f32: scirs2_core::ndarray::Array2<f32> = converter.convert_array(&array_f64)?;
    let conversion_time = start.elapsed();

    println!("Array conversion completed in {:?}", conversion_time);
    println!("Original array shape: {:?}", array_f64.shape());
    println!("Converted array shape: {:?}", array_f32.shape());

    // Sample values
    println!("Sample conversions:");
    for i in 0..3 {
        for j in 0..3 {
            println!(
                "  [{}, {}]: {:.3} -> {:.3}",
                i,
                j,
                array_f64[[i, j]],
                array_f32[[i, j]]
            );
        }
    }

    println!();
    Ok(())
}

#[cfg(not(feature = "array"))]
#[allow(dead_code)]
fn demo_ndarray_integration() -> CoreResult<()> {
    println!("8. Integration with ndarray");
    println!("===========================");
    println!("ndarray integration requires the 'array' feature to be enabled.");
    println!();
    Ok(())
}
