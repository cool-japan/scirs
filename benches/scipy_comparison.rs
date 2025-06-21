use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use scirs2_linalg::{cholesky, det, inv, lstsq, lu, matrix_norm, qr, solve};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

const SEED: u64 = 42;
const COMPARISON_SIZES: &[usize] = &[50, 100, 200, 500];

#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    operation: String,
    size: usize,
    rust_time_ns: u64,
    python_time_ns: Option<u64>,
    speedup: Option<f64>,
    memory_usage_mb: Option<f64>,
}

#[derive(Serialize, Deserialize)]
struct ComparisonReport {
    timestamp: String,
    results: Vec<BenchmarkResult>,
    summary: ComparisonSummary,
}

#[derive(Serialize, Deserialize)]
struct ComparisonSummary {
    total_operations: usize,
    rust_faster_count: usize,
    python_faster_count: usize,
    average_speedup: f64,
    max_speedup: f64,
    min_speedup: f64,
}

/// Generate test data with controlled properties
fn generate_test_data(size: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let matrix = Array2::random_using((size, size), Uniform::new(-1.0, 1.0), &mut rng);
    let vector = Array1::random_using(size, Uniform::new(-1.0, 1.0), &mut rng);
    (matrix, vector)
}

/// Generate positive definite matrix for stable operations
fn generate_spd_matrix(size: usize) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let a = Array2::random_using((size, size), Uniform::new(-1.0, 1.0), &mut rng);
    let at = a.t();
    at.dot(&a) + Array2::<f64>::eye(size) * 0.1
}

/// Benchmark comparison for basic operations
fn bench_basic_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("scipy_basic_comparison");
    let mut results = Vec::new();

    for &size in COMPARISON_SIZES {
        let (matrix, _vector) = generate_test_data(size);

        // Matrix determinant
        let start = Instant::now();
        let _result = det(&matrix.view(), None);
        let rust_time = start.elapsed().as_nanos() as u64;

        results.push(BenchmarkResult {
            operation: "determinant".to_string(),
            size,
            rust_time_ns: rust_time,
            python_time_ns: None, // Will be filled by Python script
            speedup: None,
            memory_usage_mb: None,
        });

        group.bench_with_input(BenchmarkId::new("determinant_rust", size), &size, |b, _| {
            b.iter(|| {
                let result = det(&matrix.view(), None);
                black_box(result)
            })
        });

        // Matrix inverse (for smaller matrices)
        if size <= 200 {
            let start = Instant::now();
            let _result = inv(&matrix.view(), None);
            let rust_time = start.elapsed().as_nanos() as u64;

            results.push(BenchmarkResult {
                operation: "inverse".to_string(),
                size,
                rust_time_ns: rust_time,
                python_time_ns: None,
                speedup: None,
                memory_usage_mb: None,
            });

            group.bench_with_input(BenchmarkId::new("inverse_rust", size), &size, |b, _| {
                b.iter(|| {
                    let result = inv(&matrix.view(), None);
                    black_box(result)
                })
            });
        }

        // Matrix norms
        let start = Instant::now();
        let _result = matrix_norm(&matrix.view(), "frobenius", None);
        let rust_time = start.elapsed().as_nanos() as u64;

        results.push(BenchmarkResult {
            operation: "frobenius_norm".to_string(),
            size,
            rust_time_ns: rust_time,
            python_time_ns: None,
            speedup: None,
            memory_usage_mb: None,
        });

        group.bench_with_input(
            BenchmarkId::new("frobenius_norm_rust", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = matrix_norm(&matrix.view(), "frobenius", None);
                    black_box(result)
                })
            },
        );
    }

    // Save intermediate results for Python comparison
    save_rust_results(&results);
    group.finish();
}

/// Benchmark comparison for decompositions
fn bench_decomposition_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("scipy_decomposition_comparison");
    let mut results = Vec::new();

    for &size in COMPARISON_SIZES {
        let (matrix, _vector) = generate_test_data(size);
        let spd_matrix = generate_spd_matrix(size);

        // LU decomposition
        let start = Instant::now();
        let _result = lu(&matrix.view(), None);
        let rust_time = start.elapsed().as_nanos() as u64;

        results.push(BenchmarkResult {
            operation: "lu_decomposition".to_string(),
            size,
            rust_time_ns: rust_time,
            python_time_ns: None,
            speedup: None,
            memory_usage_mb: None,
        });

        group.bench_with_input(
            BenchmarkId::new("lu_decomposition_rust", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = lu(&matrix.view(), None);
                    black_box(result)
                })
            },
        );

        // QR decomposition
        let start = Instant::now();
        let _result = qr(&matrix.view(), None);
        let rust_time = start.elapsed().as_nanos() as u64;

        results.push(BenchmarkResult {
            operation: "qr_decomposition".to_string(),
            size,
            rust_time_ns: rust_time,
            python_time_ns: None,
            speedup: None,
            memory_usage_mb: None,
        });

        group.bench_with_input(
            BenchmarkId::new("qr_decomposition_rust", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = qr(&matrix.view(), None);
                    black_box(result)
                })
            },
        );

        // Cholesky decomposition
        let start = Instant::now();
        let _result = cholesky(&spd_matrix.view(), None);
        let rust_time = start.elapsed().as_nanos() as u64;

        results.push(BenchmarkResult {
            operation: "cholesky_decomposition".to_string(),
            size,
            rust_time_ns: rust_time,
            python_time_ns: None,
            speedup: None,
            memory_usage_mb: None,
        });

        group.bench_with_input(
            BenchmarkId::new("cholesky_decomposition_rust", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = cholesky(&spd_matrix.view(), None);
                    black_box(result)
                })
            },
        );
    }

    save_rust_results(&results);
    group.finish();
}

/// Benchmark comparison for linear solvers
fn bench_solver_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("scipy_solver_comparison");
    let mut results = Vec::new();

    for &size in COMPARISON_SIZES {
        let (matrix, vector) = generate_test_data(size);

        // General linear solve
        let start = Instant::now();
        let _result = solve(&matrix.view(), &vector.view(), None);
        let rust_time = start.elapsed().as_nanos() as u64;

        results.push(BenchmarkResult {
            operation: "linear_solve".to_string(),
            size,
            rust_time_ns: rust_time,
            python_time_ns: None,
            speedup: None,
            memory_usage_mb: None,
        });

        group.bench_with_input(
            BenchmarkId::new("linear_solve_rust", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = solve(&matrix.view(), &vector.view(), None);
                    black_box(result)
                })
            },
        );

        // Least squares solve
        let start = Instant::now();
        let _result = lstsq(&matrix.view(), &vector.view(), None);
        let rust_time = start.elapsed().as_nanos() as u64;

        results.push(BenchmarkResult {
            operation: "least_squares".to_string(),
            size,
            rust_time_ns: rust_time,
            python_time_ns: None,
            speedup: None,
            memory_usage_mb: None,
        });

        group.bench_with_input(
            BenchmarkId::new("least_squares_rust", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let result = lstsq(&matrix.view(), &vector.view(), None);
                    black_box(result)
                })
            },
        );
    }

    save_rust_results(&results);
    group.finish();
}

/// Cross-platform performance comparison
fn bench_cross_platform_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_platform_performance");

    // Test the same operations across different data types and sizes
    let sizes = vec![100, 200, 500];

    for size in sizes {
        let matrix_f32 = Array2::<f32>::ones((size, size));
        let matrix_f64 = Array2::<f64>::ones((size, size));

        // Compare f32 vs f64 performance
        group.bench_with_input(BenchmarkId::new("det_f32", size), &size, |b, _| {
            b.iter(|| {
                let result = det(&matrix_f32.view(), None);
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("det_f64", size), &size, |b, _| {
            b.iter(|| {
                let result = det(&matrix_f64.view(), None);
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Save Rust benchmark results to file for Python comparison
fn save_rust_results(results: &[BenchmarkResult]) {
    let json = serde_json::to_string_pretty(results).unwrap();
    fs::write("target/rust_benchmark_results.json", json).unwrap_or_else(|e| {
        eprintln!("Failed to save Rust benchmark results: {}", e);
    });
}

/// Load and analyze comparison results if available
#[allow(dead_code)]
fn analyze_comparison_results() {
    if let Ok(contents) = fs::read_to_string("target/benchmark_comparison.json") {
        if let Ok(report) = serde_json::from_str::<ComparisonReport>(&contents) {
            print_comparison_analysis(&report);
        }
    }
}

/// Print detailed analysis of Rust vs Python performance
#[allow(dead_code)]
fn print_comparison_analysis(report: &ComparisonReport) {
    println!("\n=== SciRS2 vs SciPy Performance Comparison ===");
    println!(
        "Total operations compared: {}",
        report.summary.total_operations
    );
    println!(
        "Rust faster: {} operations",
        report.summary.rust_faster_count
    );
    println!(
        "Python faster: {} operations",
        report.summary.python_faster_count
    );
    println!("Average speedup: {:.2}x", report.summary.average_speedup);
    println!("Best speedup: {:.2}x", report.summary.max_speedup);
    println!("Worst speedup: {:.2}x", report.summary.min_speedup);

    println!("\n=== Detailed Results ===");
    for result in &report.results {
        println!(
            "{} (size {}): {:.2}x speedup",
            result.operation,
            result.size,
            result.speedup.unwrap_or(0.0)
        );
    }
}

/// Benchmark that outputs performance characteristics
fn bench_performance_characteristics(c: &mut Criterion) {
    let group = c.benchmark_group("performance_characteristics");

    // Test algorithmic complexity scaling
    let test_sizes = vec![50, 100, 200, 400];
    let mut complexity_data = HashMap::new();

    for size in test_sizes {
        let matrix = generate_test_data(size).0;

        // Measure matrix multiplication (O(n^3))
        let start = Instant::now();
        let _result = matrix.dot(&matrix);
        let time_ns = start.elapsed().as_nanos();
        complexity_data.insert(format!("matmul_{}", size), time_ns);

        // Measure determinant (O(n^3))
        let start = Instant::now();
        let _result = det(&matrix.view(), None);
        let time_ns = start.elapsed().as_nanos();
        complexity_data.insert(format!("det_{}", size), time_ns);
    }

    // Save complexity analysis
    let json = serde_json::to_string_pretty(&complexity_data).unwrap();
    fs::write("target/complexity_analysis.json", json).unwrap_or_else(|e| {
        eprintln!("Failed to save complexity analysis: {}", e);
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_basic_operations_comparison,
    bench_decomposition_comparison,
    bench_solver_comparison,
    bench_cross_platform_performance,
    bench_performance_characteristics
);

criterion_main!(benches);
