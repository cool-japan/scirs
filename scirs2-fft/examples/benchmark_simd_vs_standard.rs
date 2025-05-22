//! Benchmark comparing SIMD-optimized FFT with standard implementation
//!
//! This example provides a detailed performance comparison between the
//! SIMD-optimized FFT implementations and the standard implementation.

use num_complex::Complex64;
use scirs2_fft::{fft, fft2, fft2_simd, fft_simd, fftn, fftn_simd, simd_support_available};
use std::f64::consts::PI;
use std::time::{Duration, Instant};

fn main() {
    println!("FFT Implementation Benchmark: SIMD vs Standard");
    println!("=============================================");

    // Check for SIMD support
    let simd_available = simd_support_available();
    println!("SIMD support available: {}", simd_available);

    if !simd_available {
        println!("SIMD support not detected. Skipping SIMD benchmarks.");
        return;
    }

    let mut benchmark_results = Vec::new();

    println!("\nRunning 1D FFT benchmarks...");
    run_1d_benchmarks(&mut benchmark_results);

    println!("\nRunning 2D FFT benchmarks...");
    run_2d_benchmarks(&mut benchmark_results);

    println!("\nRunning N-dimensional FFT benchmarks...");
    run_nd_benchmarks(&mut benchmark_results);

    println!("\nSummary Table");
    println!("=============");
    println!("Implementation | Size       | Execution Time | Relative Speed");
    println!("-------------- | ---------- | -------------- | --------------");
    print_summary(&benchmark_results);
}

// Data structures to store benchmark results
struct BenchmarkResult {
    operation: String,
    size: String,
    standard_time: Duration,
    simd_time: Duration,
}

// Add a benchmark result to the collection
fn record_benchmark(
    results: &mut Vec<BenchmarkResult>,
    operation: &str,
    size: &str,
    standard_time: Duration,
    simd_time: Duration,
) {
    results.push(BenchmarkResult {
        operation: operation.to_string(),
        size: size.to_string(),
        standard_time,
        simd_time,
    });
}

// Print the benchmark summary
fn print_summary(benchmark_results: &[BenchmarkResult]) {
    for result in benchmark_results {
        let speedup = result.standard_time.as_secs_f64() / result.simd_time.as_secs_f64();
        println!(
            "{:<14} | {:<10} | {:<14.6} | {:<14.2}x",
            result.operation,
            result.size,
            result.simd_time.as_secs_f64() * 1000.0, // Convert to ms
            speedup
        );
    }
}

// Generate a test signal
fn generate_test_signal(size: usize) -> Vec<f64> {
    let mut signal = Vec::with_capacity(size);
    for i in 0..size {
        let t = i as f64 / size as f64;
        let value = (2.0 * PI * 4.0 * t).sin() + 0.5 * (2.0 * PI * 16.0 * t).sin();
        signal.push(value);
    }
    signal
}

// Generate a complex test signal
fn generate_complex_signal(size: usize) -> Vec<Complex64> {
    let mut signal = Vec::with_capacity(size);
    for i in 0..size {
        let t = i as f64 / size as f64;
        let re = (2.0 * PI * 4.0 * t).sin();
        let im = (2.0 * PI * 4.0 * t).cos() * 0.5;
        signal.push(Complex64::new(re, im));
    }
    signal
}

// Run a benchmark for comparing standard and SIMD implementations
fn run_benchmark<F, G>(
    results: &mut Vec<BenchmarkResult>,
    name: &str,
    size_desc: &str,
    iterations: usize,
    standard_fn: F,
    simd_fn: G,
) where
    F: Fn() -> Vec<Complex64>,
    G: Fn() -> Vec<Complex64>,
{
    // Warm-up runs
    for _ in 0..5 {
        let _ = standard_fn();
        let _ = simd_fn();
    }

    // Benchmark standard implementation
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = standard_fn();
    }
    let standard_time = start.elapsed() / iterations as u32;

    // Benchmark SIMD implementation
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = simd_fn();
    }
    let simd_time = start.elapsed() / iterations as u32;

    // Calculate speedup
    let speedup = standard_time.as_secs_f64() / simd_time.as_secs_f64();

    // Print results
    println!("  {}: {} runs", name, iterations);
    println!(
        "    Standard implementation: {:.6} ms",
        standard_time.as_secs_f64() * 1000.0
    );
    println!(
        "    SIMD implementation:     {:.6} ms",
        simd_time.as_secs_f64() * 1000.0
    );
    println!("    Speedup:                 {:.2}x", speedup);

    // Record benchmark
    record_benchmark(results, name, size_desc, standard_time, simd_time);
}

// Run benchmarks for 1D FFT
fn run_1d_benchmarks(results: &mut Vec<BenchmarkResult>) {
    for &size in &[1024, 4096, 16384, 65536] {
        let signal = generate_test_signal(size);
        let iterations = if size < 10000 { 100 } else { 20 };

        let standard_fn = || fft(&signal, None).unwrap();
        let simd_fn = || fft_simd(&signal, None, None).unwrap();

        run_benchmark(
            results,
            "1D FFT",
            &format!("{}p", size),
            iterations,
            standard_fn,
            simd_fn,
        );
    }

    // Also test with complex input
    for &size in &[1024, 4096, 16384] {
        let signal = generate_complex_signal(size);
        let iterations = if size < 10000 { 100 } else { 20 };

        let standard_fn = || fft(&signal, None).unwrap();
        let simd_fn = || fft_simd(&signal, None, None).unwrap();

        run_benchmark(
            results,
            "1D FFT (complex)",
            &format!("{}p", size),
            iterations,
            standard_fn,
            simd_fn,
        );
    }
}

// Run benchmarks for 2D FFT
fn run_2d_benchmarks(results: &mut Vec<BenchmarkResult>) {
    for &size in &[32, 64, 128, 256] {
        let total_elements = size * size;
        let signal = generate_test_signal(total_elements);
        let iterations = if size < 100 { 50 } else { 10 };

        let standard_fn = || {
            // Create a properly dimensioned Array2
            let array = ndarray::Array::from_shape_vec((size, size), signal.clone()).unwrap();
            let result = fft2(&array, None, None, None).unwrap();
            result.into_raw_vec_and_offset().0
        };

        let simd_fn = || fft2_simd(&signal, [size, size], None, None).unwrap();

        run_benchmark(
            results,
            "2D FFT",
            &format!("{}x{}", size, size),
            iterations,
            standard_fn,
            simd_fn,
        );
    }
}

// Run benchmarks for N-dimensional FFT
fn run_nd_benchmarks(results: &mut Vec<BenchmarkResult>) {
    // 3D FFT benchmark
    {
        let width = 16;
        let height = 16;
        let depth = 16;
        let shape = [width, height, depth];
        let total_elements = width * height * depth;
        let signal = generate_test_signal(total_elements);
        let iterations = 20;

        let standard_fn = || {
            // Create a properly dimensioned ArrayD
            let array =
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), signal.clone()).unwrap();
            let shape_vec = None;
            let axes: Option<Vec<usize>> = None;
            let result = fftn(&array, shape_vec, axes, None, None, None).unwrap();
            result.into_raw_vec_and_offset().0
        };

        let simd_fn = || fftn_simd(&signal, &shape, None, None).unwrap();

        run_benchmark(
            results,
            "3D FFT",
            &format!("{}x{}x{}", width, height, depth),
            iterations,
            standard_fn,
            simd_fn,
        );
    }

    // 4D FFT benchmark
    {
        let dim1 = 8;
        let dim2 = 8;
        let dim3 = 8;
        let dim4 = 8;
        let shape = [dim1, dim2, dim3, dim4];
        let total_elements = dim1 * dim2 * dim3 * dim4;
        let signal = generate_test_signal(total_elements);
        let iterations = 10;

        let standard_fn = || {
            // Create a properly dimensioned ArrayD
            let array =
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), signal.clone()).unwrap();
            let shape_vec = None;
            let axes: Option<Vec<usize>> = None;
            let result = fftn(&array, shape_vec, axes, None, None, None).unwrap();
            result.into_raw_vec_and_offset().0
        };

        let simd_fn = || fftn_simd(&signal, &shape, None, None).unwrap();

        run_benchmark(
            results,
            "4D FFT",
            &format!("{}x{}x{}x{}", dim1, dim2, dim3, dim4),
            iterations,
            standard_fn,
            simd_fn,
        );
    }
}
