//! Example of hardware acceleration features
//!
//! This example demonstrates how to use SIMD vectorization and other hardware
//! acceleration features for improved metrics computation performance.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_metrics::error::Result;
use scirs2_metrics::optimization::hardware::*;
use statrs::statistics::Statistics;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Hardware Acceleration Example");
    println!("===========================");

    // Example 1: Detect Hardware Capabilities
    println!("\n1. Hardware Capabilities Detection");
    println!("---------------------------------");

    hardware_detection_example();

    // Example 2: SIMD Distance Computations
    println!("\n2. SIMD Distance Computations");
    println!("----------------------------");

    simd_distance_example()?;

    // Example 3: SIMD Statistical Computations
    println!("\n3. SIMD Statistical Computations");
    println!("-------------------------------");

    simd_statistics_example()?;

    // Example 4: Hardware-Accelerated Matrix Operations
    println!("\n4. Hardware-Accelerated Matrix Operations");
    println!("----------------------------------------");

    hardware_matrix_example()?;

    // Example 5: Performance Benchmarking
    println!("\n5. Performance Benchmarking");
    println!("-------------------------");

    performance_benchmark_example()?;

    // Example 6: Configuration and Optimization
    println!("\n6. Configuration and Optimization");
    println!("-------------------------------");

    configuration_example()?;

    println!("\nHardware acceleration example completed successfully!");
    Ok(())
}

/// Example of hardware capabilities detection
#[allow(dead_code)]
fn hardware_detection_example() {
    let capabilities = HardwareCapabilities::detect();

    println!("Detected Hardware Capabilities:");
    println!("  SSE: {}", capabilities.has_sse);
    println!("  SSE2: {}", capabilities.has_sse2);
    println!("  SSE3: {}", capabilities.has_sse3);
    println!("  SSSE3: {}", capabilities.has_ssse3);
    println!("  SSE4.1: {}", capabilities.has_sse41);
    println!("  SSE4.2: {}", capabilities.has_sse42);
    println!("  AVX: {}", capabilities.has_avx);
    println!("  AVX2: {}", capabilities.has_avx2);
    println!("  AVX-512F: {}", capabilities.has_avx512f);
    println!("  FMA: {}", capabilities.has_fma);
    println!("  GPU Available: {}", capabilities.has_gpu);

    let optimal_width = capabilities.optimal_vector_width();
    println!("  Optimal Vector Width: {optimal_width:?}");
    println!("  SIMD Available: {}", capabilities.simd_available());

    if capabilities.simd_available() {
        println!("\n✓ SIMD acceleration is available on this system");

        if capabilities.has_avx2 {
            println!("✓ AVX2 support detected - expect significant performance improvements");
        } else if capabilities.has_sse2 {
            println!("✓ SSE2 support detected - moderate performance improvements expected");
        }
    } else {
        println!(
            "\n⚠ SIMD acceleration is not available - falling back to standard implementations"
        );
    }
}

/// Example of SIMD distance computations
#[allow(dead_code)]
fn simd_distance_example() -> Result<()> {
    let simd_metrics = SimdDistanceMetrics::new();

    // Create test vectors
    let vector_a = Array1::from_vec((0..1000).map(|i| i as f64 * 0.01).collect());
    let vector_b = Array1::from_vec((0..1000).map(|i| (i as f64 * 0.01) + 1.0).collect());

    println!(
        "Testing SIMD distance computations with vectors of length: {}",
        vector_a.len()
    );

    // Test Euclidean distance
    let euclidean_dist = simd_metrics.euclidean_distance_simd(&vector_a, &vector_b)?;
    println!("  Euclidean distance: {euclidean_dist:.6}");

    // Test Manhattan distance
    let manhattan_dist = simd_metrics.manhattan_distance_simd(&vector_a, &vector_b)?;
    println!("  Manhattan distance: {manhattan_dist:.6}");

    // Test cosine distance
    let cosine_dist = simd_metrics.cosine_distance_simd(&vector_a, &vector_b)?;
    println!("  Cosine distance: {cosine_dist:.6}");

    // Test dot product
    let dot_product = simd_metrics.dot_product_simd(&vector_a, &vector_b)?;
    println!("  Dot product: {dot_product:.6}");

    // Test Euclidean norm
    let norm_a = simd_metrics.euclidean_norm_simd(&vector_a)?;
    let norm_b = simd_metrics.euclidean_norm_simd(&vector_b)?;
    println!("  Norm of vector A: {norm_a:.6}");
    println!("  Norm of vector B: {norm_b:.6}");

    // Test with different vector configurations
    println!("\nTesting with different vector configurations:");
    test_distance_accuracy(&simd_metrics)?;

    Ok(())
}

/// Test accuracy of SIMD distance computations
#[allow(dead_code)]
fn test_distance_accuracy(_simdmetrics: &SimdDistanceMetrics) -> Result<()> {
    // Test with small vectors (should use standard implementation)
    let small_a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let small_b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

    let small_euclidean = _simdmetrics.euclidean_distance_simd(&small_a, &small_b)?;
    let expected_small = ((3.0_f64).powi(2) + (3.0_f64).powi(2) + (3.0_f64).powi(2)).sqrt();
    println!("  Small vector Euclidean: {small_euclidean:.6} (expected: {expected_small:.6})");

    // Test with large vectors (should use SIMD implementation)
    let large_size = 10000;
    let large_a = Array1::from_vec((0..large_size).map(|i| (i % 100) as f64).collect());
    let large_b = Array1::from_vec((0..large_size).map(|i| ((i + 50) % 100) as f64).collect());

    let large_euclidean = _simdmetrics.euclidean_distance_simd(&large_a, &large_b)?;
    println!("  Large vector Euclidean distance: {large_euclidean:.6}");

    // Verify accuracy by comparing with standard ndarray operations
    let diff = &large_a - &large_b;
    let standard_euclidean = diff.dot(&diff).sqrt();
    let accuracy_diff = (large_euclidean - standard_euclidean).abs();
    println!("  Accuracy difference vs standard: {accuracy_diff:.2e}");

    if accuracy_diff < 1e-10 {
        println!("  ✓ SIMD implementation maintains high accuracy");
    } else {
        println!("  ⚠ SIMD implementation has accuracy deviation");
    }

    Ok(())
}

/// Example of SIMD statistical computations
#[allow(dead_code)]
fn simd_statistics_example() -> Result<()> {
    let simd_stats = SimdStatistics::new();

    // Create test data
    let data = Array1::from_vec(
        (0..10000)
            .map(|i| (i as f64).sin() * 100.0 + 50.0)
            .collect(),
    );

    println!(
        "Testing SIMD statistical computations with data of length: {}",
        data.len()
    );

    // Test mean computation
    let mean = simd_stats.mean_simd(&data)?;
    println!("  Mean: {mean:.6}");

    // Test variance computation
    let variance = simd_stats.variance_simd(&data)?;
    println!("  Variance: {variance:.6}");

    // Test standard deviation computation
    let std_dev = simd_stats.std_simd(&data)?;
    println!("  Standard deviation: {std_dev:.6}");

    // Test sum computation
    let sum = simd_stats.sum_simd(&data)?;
    println!("  Sum: {sum:.6}");

    // Verify accuracy against standard implementations
    println!("\nAccuracy verification:");
    let std_mean = data.clone().mean();
    let std_sum = data.sum();

    println!("  Mean difference: {:.2e}", (mean - std_mean).abs());
    println!("  Sum difference: {:.2e}", (sum - std_sum).abs());

    // Test with different data distributions
    println!("\nTesting with different data distributions:");
    test_statistical_distributions(&simd_stats)?;

    Ok(())
}

/// Test statistical computations with different distributions
#[allow(dead_code)]
fn test_statistical_distributions(_simdstats: &SimdStatistics) -> Result<()> {
    // Normal distribution approximation
    let normal_data = Array1::from_vec(
        (0..5000)
            .map(|i| {
                let x = i as f64 / 1000.0;
                (-0.5 * x * x).exp() * 100.0 // Gaussian-like
            })
            .collect(),
    );

    let normal_mean = _simdstats.mean_simd(&normal_data)?;
    let normal_std = _simdstats.std_simd(&normal_data)?;
    println!("  Normal-like distribution - Mean: {normal_mean:.4}, Std: {normal_std:.4}");

    // Uniform distribution
    let uniform_data = Array1::from_vec((0..5000).map(|i| (i % 100) as f64).collect());

    let uniform_mean = _simdstats.mean_simd(&uniform_data)?;
    let uniform_std = _simdstats.std_simd(&uniform_data)?;
    println!("  Uniform distribution - Mean: {uniform_mean:.4}, Std: {uniform_std:.4}");

    // Exponential-like distribution
    let exp_data = Array1::from_vec(
        (0..5000)
            .map(|i| {
                let x = i as f64 / 1000.0;
                (-x).exp() * 1000.0
            })
            .collect(),
    );

    let exp_mean = _simdstats.mean_simd(&exp_data)?;
    let exp_std = _simdstats.std_simd(&exp_data)?;
    println!("  Exponential-like distribution - Mean: {exp_mean:.4}, Std: {exp_std:.4}");

    Ok(())
}

/// Example of hardware-accelerated matrix operations
#[allow(dead_code)]
fn hardware_matrix_example() -> Result<()> {
    let matrix_ops = HardwareAcceleratedMatrix::new();

    println!("Testing hardware-accelerated matrix operations:");

    // Test matrix-vector multiplication
    let matrix =
        Array2::from_shape_fn((1000, 500), |(i, j)| (i as f64 * 0.01) + (j as f64 * 0.001));
    let vector = Array1::from_shape_fn(500, |i| (i as f64).sin());

    println!(
        "  Matrix shape: {:?}, Vector length: {}",
        matrix.dim(),
        vector.len()
    );

    let result = matrix_ops.matvec_accelerated(&matrix, &vector)?;
    println!("  Matrix-vector result length: {}", result.len());
    println!(
        "  First few result values: {:?}",
        &result.slice(scirs2_core::ndarray::s![0..5]).to_vec()
    );

    // Test pairwise distance computation
    let data_points = Array2::from_shape_fn((100, 50), |(i, j)| {
        (i as f64 * 0.1) + (j as f64 * 0.05).cos()
    });

    println!("\nTesting pairwise distance computation:");
    println!("  Data shape: {:?}", data_points.dim());

    // Test different distance metrics
    let metrics = ["euclidean", "manhattan", "cosine"];
    for metric in &metrics {
        let distances = matrix_ops.pairwise_distances_accelerated(&data_points, metric)?;
        let avg_distance = distances.sum() / (distances.len() - data_points.nrows()) as f64; // Exclude diagonal
        println!("  Average {metric} distance: {avg_distance:.6}");
    }

    // Test correlation matrix computation
    let correlation_data = Array2::from_shape_fn((500, 20), |(i, j)| {
        (i as f64 * 0.01).sin() + (j as f64 * 0.1).cos() + (i + j) as f64 * 0.001
    });

    println!("\nTesting correlation matrix computation:");
    let correlation_matrix = matrix_ops.correlation_matrix_accelerated(&correlation_data)?;
    println!("  Correlation matrix shape: {:?}", correlation_matrix.dim());

    // Check some properties of the correlation matrix
    let diagonal_sum = (0..correlation_matrix.nrows())
        .map(|i| correlation_matrix[[i, i]])
        .sum::<f64>();
    let expected_diagonal_sum = correlation_matrix.nrows() as f64;
    println!("  Diagonal sum: {diagonal_sum:.6} (expected: {expected_diagonal_sum:.6})");

    Ok(())
}

/// Performance benchmarking example
#[allow(dead_code)]
fn performance_benchmark_example() -> Result<()> {
    println!("Performance benchmarking of SIMD vs standard implementations:");

    let test_sizes = vec![1000, 5000, 10000, 50000];

    for &size in &test_sizes {
        println!("\nBenchmarking with data size: {size}");
        benchmark_distance_performance(size)?;
        benchmark_statistics_performance(size)?;
    }

    Ok(())
}

/// Benchmark distance computation performance
#[allow(dead_code)]
fn benchmark_distance_performance(size: usize) -> Result<()> {
    let simd_metrics = SimdDistanceMetrics::new();

    // Create test data
    let vector_a = Array1::from_vec((0..size).map(|i| (i as f64).sin()).collect());
    let vector_b = Array1::from_vec((0..size).map(|i| (i as f64).cos()).collect());

    // Benchmark SIMD implementation
    let start_simd = Instant::now();
    let _simd_result = simd_metrics.euclidean_distance_simd(&vector_a, &vector_b)?;
    let simd_duration = start_simd.elapsed();

    // Benchmark standard implementation
    let start_std = Instant::now();
    let diff = &vector_a - &vector_b;
    let _std_result = diff.dot(&diff).sqrt();
    let std_duration = start_std.elapsed();

    println!("  Distance computation:");
    println!("    SIMD: {simd_duration:?}");
    println!("    Standard: {std_duration:?}");

    if simd_duration < std_duration {
        let speedup = std_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64;
        println!("    Speedup: {speedup:.2}x");
    } else {
        println!("    No speedup (possibly too small for SIMD benefit)");
    }

    Ok(())
}

/// Benchmark statistical computation performance
#[allow(dead_code)]
fn benchmark_statistics_performance(size: usize) -> Result<()> {
    let simd_stats = SimdStatistics::new();

    // Create test data
    let data = Array1::from_vec(
        (0..size)
            .map(|i| (i as f64 * 0.001).sin() * 100.0)
            .collect(),
    );

    // Benchmark SIMD mean computation
    let start_simd = Instant::now();
    let _simd_mean = simd_stats.mean_simd(&data)?;
    let simd_duration = start_simd.elapsed();

    // Benchmark standard mean computation
    let start_std = Instant::now();
    let _std_mean = data.clone().mean();
    let std_duration = start_std.elapsed();

    println!("  Statistical computation (mean):");
    println!("    SIMD: {simd_duration:?}");
    println!("    Standard: {std_duration:?}");

    if simd_duration < std_duration {
        let speedup = std_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64;
        println!("    Speedup: {speedup:.2}x");
    }

    Ok(())
}

/// Configuration and optimization example
#[allow(dead_code)]
fn configuration_example() -> Result<()> {
    println!("Testing different hardware acceleration configurations:");

    // Test with SIMD disabled
    let config_no_simd = HardwareAccelConfig::new()
        .with_simd_enabled(false)
        .with_min_data_size(1000);

    let metrics_no_simd = SimdDistanceMetrics::with_config(config_no_simd);

    // Test with SIMD enabled and different vector widths
    let config_simd_128 = HardwareAccelConfig::new()
        .with_simd_enabled(true)
        .with_vector_width(VectorWidth::V128)
        .with_min_data_size(500);

    let metrics_simd_128 = SimdDistanceMetrics::with_config(config_simd_128);

    let config_simd_256 = HardwareAccelConfig::new()
        .with_simd_enabled(true)
        .with_vector_width(VectorWidth::V256)
        .with_min_data_size(500);

    let metrics_simd_256 = SimdDistanceMetrics::with_config(config_simd_256);

    let config_simd_auto = HardwareAccelConfig::new()
        .with_simd_enabled(true)
        .with_vector_width(VectorWidth::Auto)
        .with_min_data_size(100);

    let metrics_simd_auto = SimdDistanceMetrics::with_config(config_simd_auto);

    // Test data
    let test_data_a = Array1::from_vec((0..5000).map(|i| (i as f64 * 0.001).sin()).collect());
    let test_data_b = Array1::from_vec((0..5000).map(|i| (i as f64 * 0.001).cos()).collect());

    // Compare results from different configurations
    let result_no_simd = metrics_no_simd.euclidean_distance_simd(&test_data_a, &test_data_b)?;
    let result_simd_128 = metrics_simd_128.euclidean_distance_simd(&test_data_a, &test_data_b)?;
    let result_simd_256 = metrics_simd_256.euclidean_distance_simd(&test_data_a, &test_data_b)?;
    let result_simd_auto = metrics_simd_auto.euclidean_distance_simd(&test_data_a, &test_data_b)?;

    println!("  Results from different configurations:");
    println!("    No SIMD: {result_no_simd:.10}");
    println!("    SIMD 128-bit: {result_simd_128:.10}");
    println!("    SIMD 256-bit: {result_simd_256:.10}");
    println!("    SIMD Auto: {result_simd_auto:.10}");

    // Check consistency
    let max_diff = [
        (result_no_simd - result_simd_128).abs(),
        (result_no_simd - result_simd_256).abs(),
        (result_no_simd - result_simd_auto).abs(),
    ]
    .iter()
    .fold(0.0_f64, |a, &b| a.max(b));

    println!("    Maximum difference: {max_diff:.2e}");

    if max_diff < 1e-10 {
        println!("    ✓ All configurations produce consistent results");
    } else {
        println!("    ⚠ Configuration results show significant differences");
    }

    // Performance recommendations
    println!("\nPerformance recommendations:");
    let capabilities = HardwareCapabilities::detect();

    if capabilities.has_avx2 {
        println!("  • Use VectorWidth::V256 or VectorWidth::Auto for best performance");
        println!("  • Set min_data_size to 1000 or higher for large datasets");
    } else if capabilities.has_sse2 {
        println!("  • Use VectorWidth::V128 or VectorWidth::Auto");
        println!("  • Set min_data_size to 500 or higher");
    } else {
        println!("  • SIMD not available - use standard implementations");
        println!("  • Consider upgrading hardware for better performance");
    }

    println!("  • Enable GPU acceleration when working with very large datasets (future feature)");
    println!("  • Use chunked processing for memory-constrained environments");

    Ok(())
}
