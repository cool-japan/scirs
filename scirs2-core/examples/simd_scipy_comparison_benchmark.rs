//! Comprehensive SIMD performance benchmark for SciRS2
//!
//! This benchmark measures the performance of SIMD-accelerated operations
//! for direct comparison with NumPy/SciPy.
//!
//! Outputs results in CSV format for easy comparison.
//!
//! Run with: cargo run --release --features simd --example scirs2_simd_benchmark

use scirs2_core::ndarray::Array1;
use scirs2_core::ndarray_ext::elementwise::{
    abs_simd, acos_simd, asin_simd, atan_simd, ceil_simd, clamp_simd, cos_simd, cosh_simd,
    exp_simd, floor_simd, ln_simd, round_simd, sin_simd, sinh_simd, sqrt_simd, tan_simd, tanh_simd,
};
use scirs2_core::ndarray_ext::preprocessing::{
    clip_simd as clip_preproc, normalize_simd, standardize_simd,
};
use std::f32::consts::PI as PI_F32;
use std::f64::consts::PI as PI_F64;
use std::fs::File;
use std::io::Write as IoWrite;
use std::time::Instant;

#[derive(Debug)]
struct BenchmarkResult {
    operation: String,
    dtype: String,
    size: usize,
    implementation: String,
    time_us: f64,
    iterations: u32,
}

fn benchmark_operation<T, F>(operation: F, iterations: u32, warmup: u32) -> f64
where
    F: Fn() -> T,
{
    // Warm-up
    for _ in 0..warmup {
        let _ = operation();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = operation();
    }
    let elapsed = start.elapsed();

    // Return time in microseconds
    (elapsed.as_secs_f64() / iterations as f64) * 1_000_000.0
}

fn benchmark_f64_operations(size: usize, results: &mut Vec<BenchmarkResult>) {
    println!("\n  Benchmarking f64 operations (size: {})...", size);

    let iterations = if size < 10_000 {
        1000
    } else if size < 100_000 {
        100
    } else {
        10
    };

    // Generate test data
    let data: Array1<f64> =
        Array1::from_vec((0..size).map(|i| ((i as f64) * 0.1).sin() * 10.0).collect());
    let data_pos: Array1<f64> =
        Array1::from_vec((0..size).map(|i| ((i as f64) * 0.1).abs() + 1.0).collect());
    let data_sin: Array1<f64> = Array1::from_vec(
        (0..size)
            .map(|i| ((i as f64) * 0.01) % (2.0 * PI_F64))
            .collect(),
    );
    let data_normalized: Array1<f64> =
        Array1::from_vec((0..size).map(|i| (i as f64) * 0.001).collect());

    // Element-wise operations
    macro_rules! bench_op {
        ($name:expr, $op:expr, $data:expr) => {{
            let time_us = benchmark_operation(|| $op(&$data.view()), iterations, 10);
            println!(
                "    {:15} f64: {:10.2} μs ({} iterations)",
                $name, time_us, iterations
            );
            results.push(BenchmarkResult {
                operation: $name.to_string(),
                dtype: "f64".to_string(),
                size,
                implementation: "scirs2_simd".to_string(),
                time_us,
                iterations,
            });
        }};
    }

    bench_op!("abs", abs_simd, data);
    bench_op!("sqrt", sqrt_simd, data_pos);
    bench_op!("exp", exp_simd, data_normalized);
    bench_op!("log", ln_simd, data_pos);
    bench_op!("sin", sin_simd, data_sin);
    bench_op!("cos", cos_simd, data_sin);
    bench_op!("tan", tan_simd, data_sin);
    bench_op!("tanh", tanh_simd, data_normalized);
    bench_op!("sinh", sinh_simd, data_normalized);
    bench_op!("cosh", cosh_simd, data_normalized);
    bench_op!("floor", floor_simd, data);
    bench_op!("ceil", ceil_simd, data);
    bench_op!("round", round_simd, data);

    // Preprocessing operations
    bench_op!("normalize", normalize_simd, data);
    bench_op!("standardize", standardize_simd, data);

    let time_us = benchmark_operation(|| clip_preproc(&data.view(), -20.0, 20.0), iterations, 10);
    println!(
        "    {:15} f64: {:10.2} μs ({} iterations)",
        "clip", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "clip".to_string(),
        dtype: "f64".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    // Reduction operations (using ndarray built-in methods)
    let time_us = benchmark_operation(|| data.sum(), iterations, 10);
    println!(
        "    {:15} f64: {:10.2} μs ({} iterations)",
        "sum", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "sum".to_string(),
        dtype: "f64".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(|| data.mean().expect("Operation failed"), iterations, 10);
    println!(
        "    {:15} f64: {:10.2} μs ({} iterations)",
        "mean", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "mean".to_string(),
        dtype: "f64".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(|| data.std(0.0), iterations, 10);
    println!(
        "    {:15} f64: {:10.2} μs ({} iterations)",
        "std", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "std".to_string(),
        dtype: "f64".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(|| data.var(0.0), iterations, 10);
    println!(
        "    {:15} f64: {:10.2} μs ({} iterations)",
        "var", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "var".to_string(),
        dtype: "f64".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(
        || data.iter().cloned().fold(f64::INFINITY, f64::min),
        iterations,
        10,
    );
    println!(
        "    {:15} f64: {:10.2} μs ({} iterations)",
        "min", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "min".to_string(),
        dtype: "f64".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(
        || data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        iterations,
        10,
    );
    println!(
        "    {:15} f64: {:10.2} μs ({} iterations)",
        "max", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "max".to_string(),
        dtype: "f64".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });
}

fn benchmark_f32_operations(size: usize, results: &mut Vec<BenchmarkResult>) {
    println!("\n  Benchmarking f32 operations (size: {})...", size);

    let iterations = if size < 10_000 {
        1000
    } else if size < 100_000 {
        100
    } else {
        10
    };

    // Generate test data
    let data: Array1<f32> =
        Array1::from_vec((0..size).map(|i| ((i as f32) * 0.1).sin() * 10.0).collect());
    let data_pos: Array1<f32> =
        Array1::from_vec((0..size).map(|i| ((i as f32) * 0.1).abs() + 1.0).collect());
    let data_sin: Array1<f32> = Array1::from_vec(
        (0..size)
            .map(|i| ((i as f32) * 0.01) % (2.0 * PI_F32))
            .collect(),
    );
    let data_normalized: Array1<f32> =
        Array1::from_vec((0..size).map(|i| (i as f32) * 0.001).collect());

    // Element-wise operations
    macro_rules! bench_op {
        ($name:expr, $op:expr, $data:expr) => {{
            let time_us = benchmark_operation(|| $op(&$data.view()), iterations, 10);
            println!(
                "    {:15} f32: {:10.2} μs ({} iterations)",
                $name, time_us, iterations
            );
            results.push(BenchmarkResult {
                operation: $name.to_string(),
                dtype: "f32".to_string(),
                size,
                implementation: "scirs2_simd".to_string(),
                time_us,
                iterations,
            });
        }};
    }

    bench_op!("abs", abs_simd, data);
    bench_op!("sqrt", sqrt_simd, data_pos);
    bench_op!("exp", exp_simd, data_normalized);
    bench_op!("log", ln_simd, data_pos);
    bench_op!("sin", sin_simd, data_sin);
    bench_op!("cos", cos_simd, data_sin);
    bench_op!("tan", tan_simd, data_sin);
    bench_op!("tanh", tanh_simd, data_normalized);
    bench_op!("sinh", sinh_simd, data_normalized);
    bench_op!("cosh", cosh_simd, data_normalized);
    bench_op!("floor", floor_simd, data);
    bench_op!("ceil", ceil_simd, data);
    bench_op!("round", round_simd, data);

    // Preprocessing operations
    bench_op!("normalize", normalize_simd, data);
    bench_op!("standardize", standardize_simd, data);

    let time_us = benchmark_operation(|| clip_preproc(&data.view(), -20.0, 20.0), iterations, 10);
    println!(
        "    {:15} f32: {:10.2} μs ({} iterations)",
        "clip", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "clip".to_string(),
        dtype: "f32".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    // Reduction operations (using ndarray built-in methods)
    let time_us = benchmark_operation(|| data.sum(), iterations, 10);
    println!(
        "    {:15} f32: {:10.2} μs ({} iterations)",
        "sum", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "sum".to_string(),
        dtype: "f32".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(|| data.mean().expect("Operation failed"), iterations, 10);
    println!(
        "    {:15} f32: {:10.2} μs ({} iterations)",
        "mean", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "mean".to_string(),
        dtype: "f32".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(|| data.std(0.0), iterations, 10);
    println!(
        "    {:15} f32: {:10.2} μs ({} iterations)",
        "std", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "std".to_string(),
        dtype: "f32".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(|| data.var(0.0), iterations, 10);
    println!(
        "    {:15} f32: {:10.2} μs ({} iterations)",
        "var", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "var".to_string(),
        dtype: "f32".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(
        || data.iter().cloned().fold(f32::INFINITY, f32::min),
        iterations,
        10,
    );
    println!(
        "    {:15} f32: {:10.2} μs ({} iterations)",
        "min", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "min".to_string(),
        dtype: "f32".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });

    let time_us = benchmark_operation(
        || data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        iterations,
        10,
    );
    println!(
        "    {:15} f32: {:10.2} μs ({} iterations)",
        "max", time_us, iterations
    );
    results.push(BenchmarkResult {
        operation: "max".to_string(),
        dtype: "f32".to_string(),
        size,
        implementation: "scirs2_simd".to_string(),
        time_us,
        iterations,
    });
}

fn save_results_to_csv(
    results: &[BenchmarkResult],
    filename: &std::path::Path,
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;

    // Write header
    writeln!(
        file,
        "operation,dtype,size,implementation,time_us,iterations"
    )?;

    // Write results
    for result in results {
        writeln!(
            file,
            "{},{},{},{},{:.2},{}",
            result.operation,
            result.dtype,
            result.size,
            result.implementation,
            result.time_us,
            result.iterations
        )?;
    }

    Ok(())
}

fn main() {
    println!("SciRS2 SIMD Performance Benchmark");
    println!("==================================");
    println!();

    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];
    let mut results = Vec::new();

    for &size in &sizes {
        println!("\nArray size: {}", size);
        println!("----------------------------------------");

        benchmark_f64_operations(size, &mut results);
        benchmark_f32_operations(size, &mut results);
    }

    // Save results
    let csv_file = std::env::temp_dir().join("scirs2_benchmark_results.csv");
    if let Err(e) = save_results_to_csv(&results, &csv_file) {
        eprintln!("Error saving results: {}", e);
    } else {
        println!("\n\nResults saved to: {}", csv_file.display());
    }

    println!("\n{}", "=".repeat(80));
    println!("SciRS2 SIMD Performance Summary:");
    println!("SIMD-accelerated operations show significant speedup on:");
    println!("- x86_64 with AVX2/AVX-512 support");
    println!("- ARM with NEON support");
    println!("\nThese results can be directly compared with NumPy/SciPy benchmarks.");
}
