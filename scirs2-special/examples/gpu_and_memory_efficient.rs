//! Example demonstrating GPU acceleration and memory-efficient processing
//!
//! This example shows how to use the GPU-accelerated and memory-efficient
//! implementations of special functions for large arrays.

use scirs2_core::ndarray::Array1;
use scirs2_special::error::SpecialResult;

#[cfg(feature = "gpu")]
use scirs2_special::gpu_ops::{erf_gpu, gamma_gpu, j0_gpu};

use scirs2_special::memory_efficient::{erf_chunked, gamma_chunked, j0_chunked, ChunkedConfig};

#[allow(dead_code)]
fn main() -> SpecialResult<()> {
    println!("Special Functions: GPU Acceleration and Memory-Efficient Processing Demo");
    println!("=====================================================================\n");

    // Create test arrays of different sizes
    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        println!("Array size: {}", size);

        // Create input array
        let input = Array1::linspace(0.1, 10.0, size);

        // Test memory-efficient chunked processing
        test_chunked_processing(&input)?;

        // Test GPU acceleration (if available)
        #[cfg(feature = "gpu")]
        test_gpu_acceleration(&input)?;

        println!();
    }

    // Demonstrate custom chunking configuration
    demonstrate_custom_chunking()?;

    Ok(())
}

/// Test memory-efficient chunked processing
#[allow(dead_code)]
fn test_chunked_processing(input: &Array1<f64>) -> SpecialResult<()> {
    use std::time::Instant;

    println!("\n  Memory-Efficient Chunked Processing:");

    // Gamma function
    let start = Instant::now();
    let _gamma_result = gamma_chunked(input, None)?;
    let gamma_time = start.elapsed();
    println!("    - Gamma function: {:?}", gamma_time);

    // Bessel J0 function
    let start = Instant::now();
    let _j0_result = j0_chunked(input, None)?;
    let j0_time = start.elapsed();
    println!("    - Bessel J0 function: {:?}", j0_time);

    // Error function
    let start = Instant::now();
    let _erf_result = erf_chunked(input, None)?;
    let erf_time = start.elapsed();
    println!("    - Error function: {:?}", erf_time);

    Ok(())
}

/// Test GPU acceleration (only compiled with gpu feature)
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn test_gpu_acceleration(input: &Array1<f64>) -> SpecialResult<()> {
    use std::time::Instant;

    println!("\n  GPU Acceleration (if available):");

    let mut output = Array1::zeros(input.len());

    // Gamma function
    let start = Instant::now();
    match gamma_gpu(&input.view(), &mut output.view_mut()) {
        Ok(_) => {
            let gpu_time = start.elapsed();
            println!("    - Gamma function (GPU): {:?}", gpu_time);
        }
        Err(e) => {
            println!("    - Gamma function (GPU): Not available - {}", e);
        }
    }

    // Bessel J0 function
    let start = Instant::now();
    match j0_gpu(&input.view(), &mut output.view_mut()) {
        Ok(_) => {
            let gpu_time = start.elapsed();
            println!("    - Bessel J0 function (GPU): {:?}", gpu_time);
        }
        Err(e) => {
            println!("    - Bessel J0 function (GPU): Not available - {}", e);
        }
    }

    // Error function
    let start = Instant::now();
    match erf_gpu(&input.view(), &mut output.view_mut()) {
        Ok(_) => {
            let gpu_time = start.elapsed();
            println!("    - Error function (GPU): {:?}", gpu_time);
        }
        Err(e) => {
            println!("    - Error function (GPU): Not available - {}", e);
        }
    }

    Ok(())
}

/// Demonstrate custom chunking configuration
#[allow(dead_code)]
fn demonstrate_custom_chunking() -> SpecialResult<()> {
    println!("\nCustom Chunking Configuration:");
    println!("==============================");

    // Create a large array
    let large_array = Array1::linspace(0.1, 100.0, 5_000_000);

    // Default configuration
    println!("\nDefault chunking:");
    let start = std::time::Instant::now();
    let _result = gamma_chunked(&large_array, None)?;
    println!("  Time: {:?}", start.elapsed());

    // Small chunks (more overhead, less memory)
    let small_chunk_config = ChunkedConfig {
        max_chunk_bytes: 1024 * 1024, // 1MB chunks
        parallel_chunks: true,
        min_arraysize: 1000,
        prefetch: false,
    };
    println!("\nSmall chunks (1MB):");
    let start = std::time::Instant::now();
    let _result = gamma_chunked(&large_array, Some(small_chunk_config))?;
    println!("  Time: {:?}", start.elapsed());

    // Large chunks (less overhead, more memory)
    let large_chunk_config = ChunkedConfig {
        max_chunk_bytes: 256 * 1024 * 1024, // 256MB chunks
        parallel_chunks: true,
        min_arraysize: 1000,
        prefetch: true,
    };
    println!("\nLarge chunks (256MB):");
    let start = std::time::Instant::now();
    let _result = gamma_chunked(&large_array, Some(large_chunk_config))?;
    println!("  Time: {:?}", start.elapsed());

    // Sequential processing
    let sequential_config = ChunkedConfig {
        max_chunk_bytes: 64 * 1024 * 1024, // 64MB chunks
        parallel_chunks: false,            // Sequential
        min_arraysize: 1000,
        prefetch: false,
    };
    println!("\nSequential processing:");
    let start = std::time::Instant::now();
    let _result = gamma_chunked(&large_array, Some(sequential_config))?;
    println!("  Time: {:?}", start.elapsed());

    Ok(())
}

/// Compare accuracy between different methods
#[allow(dead_code)]
fn compare_accuracy() -> SpecialResult<()> {
    println!("\nAccuracy Comparison:");
    println!("===================");

    let test_values: Vec<f64> = vec![0.5, 1.0, 2.5, 5.0, 10.0];

    for &x in &test_values {
        let single_value = scirs2_special::gamma(x);

        // Test chunked processing
        let input = Array1::from_elem(1, x);
        let chunked_result = gamma_chunked(&input, None)?;

        println!("  x = {:.1}:", x);
        println!("    Direct:  {:.10}", single_value);
        println!("    Chunked: {:.10}", chunked_result[0]);
        println!(
            "    Diff:    {:.2e}",
            (single_value - chunked_result[0]).abs()
        );
    }

    Ok(())
}
