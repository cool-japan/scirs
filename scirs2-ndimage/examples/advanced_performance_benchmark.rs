//! Advanced-performance benchmark showcasing advanced SIMD optimizations
//!
//! This example demonstrates the performance benefits of the advanced SIMD
//! extensions compared to standard implementations. It includes:
//! - Wavelet pyramid decomposition performance
//! - Local Binary Pattern texture analysis speed
//! - Advanced edge detection throughput
//! - Memory efficiency comparisons
//! - Scalability analysis across different image sizes

use ndarray::{Array2, ArrayView2};
use scirs2_ndimage::{
    error::NdimageResult,
    profiling::{
        disable_profiling, enable_memory_profiling, enable_profiling, get_memory_report,
        get_performance_report,
    },
};
use std::time::{Duration, Instant};

// Import both standard and advanced SIMD functions for comparison
#[cfg(feature = "simd")]
use scirs2_ndimage::filters::{
    advanced_simd_advanced_edge_detection, advanced_simd_multi_scale_lbp,
    advanced_simd_wavelet_pyramid, laplace, sobel, WaveletType,
};

#[cfg(feature = "simd")]
use scirs2_ndimage::features::{canny, sobel_edges};

/// Benchmark configuration
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct BenchmarkConfig {
    /// Number of iterations for each benchmark
    iterations: usize,
    /// Test image sizes
    image_sizes: Vec<(usize, usize)>,
    /// Whether to enable detailed memory profiling
    enable_memory_profiling: bool,
    /// Whether to show detailed results
    verbose: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 5,
            image_sizes: vec![
                (128, 128),
                (256, 256),
                (512, 512),
                (1024, 1024),
                (2048, 2048),
            ],
            enable_memory_profiling: true,
            verbose: true,
        }
    }
}

/// Benchmark results for a single operation
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct BenchmarkResult {
    operation_name: String,
    image_size: (usize, usize),
    mean_duration: Duration,
    std_deviation: Duration,
    min_duration: Duration,
    max_duration: Duration,
    memory_used: Option<u64>,
    throughput_mpix_per_sec: f64,
}

#[allow(dead_code)]
impl BenchmarkResult {
    fn new(operation_name: String, image_size: (usize, usize), durations: &[Duration]) -> Self {
        let mean_duration = Duration::from_nanos(
            (durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128) as u64,
        );

        let mean_nanos = mean_duration.as_nanos() as f64;
        let variance = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;

        let std_deviation = Duration::from_nanos(variance.sqrt() as u64);
        let min_duration = *durations.iter().min().unwrap();
        let max_duration = *durations.iter().max().unwrap();

        // Calculate throughput in megapixels per second
        let total_pixels = (image_size.0 * image_size.1) as f64;
        let megapixels = total_pixels / 1_000_000.0;
        let seconds = mean_duration.as_secs_f64();
        let throughput_mpix_per_sec = if seconds > 0.0 {
            megapixels / seconds
        } else {
            0.0
        };

        Self {
            operation_name,
            image_size,
            mean_duration,
            std_deviation,
            min_duration,
            max_duration,
            memory_used: None,
            throughput_mpix_per_sec,
        }
    }

    fn display(&self) {
        println!(
            "  {} ({}x{}):",
            self.operation_name, self.image_size.0, self.image_size.1
        );
        println!(
            "    Mean: {:?} ± {:?}",
            self.mean_duration, self.std_deviation
        );
        println!(
            "    Range: {:?} - {:?}",
            self.min_duration, self.max_duration
        );
        println!("    Throughput: {:.2} MPix/s", self.throughput_mpix_per_sec);
        if let Some(memory) = self.memory_used {
            println!("    Memory: {:.2} MB", memory as f64 / 1_000_000.0);
        }
    }
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn main() -> NdimageResult<()> {
    println!("=== Advanced-Performance Benchmark for scirs2-ndimage ===\n");

    let config = BenchmarkConfig::default();

    if config.enable_memory_profiling {
        enable_memory_profiling();
    }
    enable_profiling();

    let mut all_results = Vec::new();

    for &image_size in &config.image_sizes {
        println!(
            "Benchmarking with image size: {}x{}",
            image_size.0, image_size.1
        );

        // Create test image with interesting patterns
        let testimage = create_benchmarkimage(image_size.0, image_size.1);

        // 1. Wavelet pyramid benchmarks
        #[cfg(feature = "simd")]
        {
            let wavelet_results = benchmark_wavelet_pyramid(&testimage, &config)?;
            all_results.extend(wavelet_results);
        }

        // 2. Local Binary Pattern benchmarks
        #[cfg(feature = "simd")]
        {
            let lbp_results = benchmark_lbp(&testimage, &config)?;
            all_results.extend(lbp_results);
        }

        // 3. Edge detection benchmarks
        #[cfg(feature = "simd")]
        {
            let edge_results = benchmark_edge_detection(&testimage, &config)?;
            all_results.extend(edge_results);
        }

        println!();
    }

    // Display comprehensive results
    println!("=== Comprehensive Performance Results ===\n");
    display_benchmark_summary(&all_results);

    // Display memory usage report
    if config.enable_memory_profiling {
        println!("\n=== Memory Usage Report ===");
        let memory_report = get_memory_report();
        memory_report.display();
    }

    // Display profiling report
    println!("\n=== Detailed Profiling Report ===");
    let perf_report = get_performance_report();
    println!(
        "Total operations: {}",
        perf_report.operation_breakdown.len()
    );

    disable_profiling();

    Ok(())
}

#[allow(dead_code)]
fn create_benchmarkimage(height: usize, width: usize) -> Array2<f64> {
    Array2::from_shape_fn((height, width), |(i, j)| {
        let x = i as f64 / height as f64;
        let y = j as f64 / width as f64;

        // Create a complex pattern with multiple frequency components
        let pattern1 =
            (x * std::f64::consts::PI * 8.0).sin() * (y * std::f64::consts::PI * 8.0).cos();
        let pattern2 = (x * std::f64::consts::PI * 16.0 + y * std::f64::consts::PI * 16.0).sin();
        let pattern3 = ((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt();

        // Add noise and edges
        let noise = if (i * j) % 17 == 0 { 0.3 } else { 0.0 };
        let edges = if i % 64 < 2 || j % 64 < 2 { 1.0 } else { 0.0 };

        0.5 * pattern1 + 0.3 * pattern2 + 0.2 * pattern3 + noise + edges
    })
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn benchmark_wavelet_pyramid(
    image: &Array2<f64>,
    config: &BenchmarkConfig,
) -> NdimageResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    // Benchmark advanced-SIMD wavelet pyramid with different wavelet types
    for (wavelet_name, wavelet_type) in [
        ("Advanced-SIMD Haar Wavelet", WaveletType::Haar),
        ("Advanced-SIMD Daubechies-4", WaveletType::Daubechies4),
        ("Advanced-SIMD Biorthogonal", WaveletType::Biorthogonal),
    ] {
        let mut durations = Vec::new();

        for _ in 0..config.iterations {
            let start = Instant::now();
            let _pyramid = advanced_simd_wavelet_pyramid(
                image.view(),
                3, // levels
                wavelet_type.clone(),
            )?;
            durations.push(start.elapsed());
        }

        let result = BenchmarkResult::new(wavelet_name.to_string(), image.dim(), &durations);

        if config.verbose {
            result.display();
        }
        results.push(result);
    }

    Ok(results)
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn benchmark_lbp(
    image: &Array2<f64>,
    config: &BenchmarkConfig,
) -> NdimageResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    // Benchmark multi-scale LBP with different configurations
    let lbp_configs = [
        ("LBP Single-scale 8", vec![1], vec![8]),
        ("LBP Multi-scale 3x8", vec![1, 2, 3], vec![8, 8, 8]),
        ("LBP Multi-scale 3x16", vec![1, 2, 3], vec![16, 16, 16]),
        ("LBP High-resolution", vec![1, 2, 3, 4], vec![8, 16, 24, 32]),
    ];

    for (config_name, radii, sample_points) in lbp_configs {
        let mut durations = Vec::new();

        for _ in 0..config.iterations {
            let start = Instant::now();
            let _lbp_result = advanced_simd_multi_scale_lbp(image.view(), &radii, &sample_points)?;
            durations.push(start.elapsed());
        }

        let result = BenchmarkResult::new(
            format!("Advanced-SIMD {}", config_name),
            image.dim(),
            &durations,
        );

        if config.verbose {
            result.display();
        }
        results.push(result);
    }

    Ok(results)
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn benchmark_edge_detection(
    image: &Array2<f64>,
    config: &BenchmarkConfig,
) -> NdimageResult<Vec<BenchmarkResult>> {
    let mut results = Vec::new();

    // Benchmark advanced edge detection
    let mut durations = Vec::new();

    for _ in 0..config.iterations {
        let start = Instant::now();
        let _edges = advanced_simd_advanced_edge_detection(
            image.view(),
            1.0, // sigma
            0.1, // low threshold factor
            0.3, // high threshold factor
        )?;
        durations.push(start.elapsed());
    }

    let result = BenchmarkResult::new(
        "Advanced-SIMD Advanced Edge Detection".to_string(),
        image.dim(),
        &durations,
    );

    if config.verbose {
        result.display();
    }
    results.push(result);

    // Compare with standard edge detection methods
    let edge_methods = [
        ("Standard Sobel", benchmark_standard_sobel),
        ("Standard Canny", benchmark_standard_canny),
    ];

    for (method_name, benchmark_fn) in edge_methods {
        let mut durations = Vec::new();

        for _ in 0..config.iterations {
            let start = Instant::now();
            let _result = benchmark_fn(image)?;
            durations.push(start.elapsed());
        }

        let result = BenchmarkResult::new(method_name.to_string(), image.dim(), &durations);

        if config.verbose {
            result.display();
        }
        results.push(result);
    }

    Ok(results)
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn benchmark_standard_sobel(image: &Array2<f64>) -> NdimageResult<Array2<f64>> {
    sobel(&image.view(), None, None, None)
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn benchmark_standard_canny(image: &Array2<f64>) -> NdimageResult<Array2<f64>> {
    canny(
        image.view(),
        1.0,  // sigma
        0.1,  // low_threshold
        0.3,  // high_threshold
        None, // mask
    )
}

#[allow(dead_code)]
fn display_benchmark_summary(results: &[BenchmarkResult]) {
    // Group _results by operation type
    use std::collections::HashMap;
    let mut grouped: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();

    for result in results {
        let operation_type = result
            .operation_name
            .split_whitespace()
            .next()
            .unwrap_or("Unknown");
        grouped
            .entry(operation_type.to_string())
            .or_default()
            .push(result);
    }

    for (operation_type, group_results) in grouped {
        println!("--- {} Operations ---", operation_type);

        // Find best performing implementation for each image size
        let mut size_groups: HashMap<(usize, usize), Vec<&BenchmarkResult>> = HashMap::new();
        for result in group_results {
            size_groups
                .entry(result.image_size)
                .or_default()
                .push(result);
        }

        for (size, size_results) in size_groups {
            println!("  {}x{} pixels:", size.0, size.1);

            // Sort by throughput (descending)
            let mut sorted_results = size_results;
            sorted_results.sort_by(|a, b| {
                b.throughput_mpix_per_sec
                    .partial_cmp(&a.throughput_mpix_per_sec)
                    .unwrap()
            });

            for (i, result) in sorted_results.iter().enumerate() {
                let rank = if i == 0 {
                    "🥇"
                } else if i == 1 {
                    "🥈"
                } else if i == 2 {
                    "🥉"
                } else {
                    "  "
                };
                println!(
                    "    {} {}: {:.2} MPix/s ({:?})",
                    rank,
                    result.operation_name,
                    result.throughput_mpix_per_sec,
                    result.mean_duration
                );
            }

            // Calculate speedup of best vs worst
            if sorted_results.len() > 1 {
                let speedup = sorted_results[0].throughput_mpix_per_sec
                    / sorted_results[sorted_results.len() - 1].throughput_mpix_per_sec;
                println!("    Speedup: {:.2}x", speedup);
            }
        }
        println!();
    }

    // Overall statistics
    println!("--- Overall Statistics ---");
    let total_operations = results.len();
    let avg_throughput: f64 = results
        .iter()
        .map(|r| r.throughput_mpix_per_sec)
        .sum::<f64>()
        / total_operations as f64;
    let max_throughput = results
        .iter()
        .map(|r| r.throughput_mpix_per_sec)
        .fold(0.0, f64::max);

    println!("Total benchmarks: {}", total_operations);
    println!("Average throughput: {:.2} MPix/s", avg_throughput);
    println!("Peak throughput: {:.2} MPix/s", max_throughput);
}

#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
fn main() -> NdimageResult<()> {
    println!("Advanced-performance benchmark requires SIMD features");
    println!(
        "Please compile with: cargo run --example advanced_performance_benchmark --features simd"
    );
    Ok(())
}
