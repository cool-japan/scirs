//! SciRS2 v0.2.0 Python Comparison Benchmarks
//!
//! This benchmark suite compares SciRS2 performance against Python equivalents:
//! - SciPy: scientific computing functions
//! - NumPy: array operations
//! - PyTorch: tensor operations and neural networks
//!
//! Methodology:
//! 1. Run Rust benchmarks and save results to JSON
//! 2. Python script reads results and runs equivalent operations
//! 3. Generate comparison report with speedup ratios
//!
//! Performance Targets:
//! - Array operations: 2-10x faster than NumPy
//! - Linear algebra: competitive with SciPy (using OxiBLAS)
//! - FFT: competitive with SciPy/NumPy
//! - Special functions: competitive or faster than SciPy

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use scirs2_core::ndarray::{Array1, Array2, RandomExt};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::time::Duration;

/// Helper to create a Uniform distribution without unwrap
fn uniform_f64(low: f64, high: f64) -> Uniform<f64> {
    Uniform::new(low, high).expect("Failed to create Uniform distribution")
}

// =============================================================================
// Result Serialization
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    category: String,
    operation: String,
    size: usize,
    mean_time_ns: f64,
    median_time_ns: f64,
    std_dev_ns: f64,
    throughput_per_sec: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkSuite {
    timestamp: String,
    platform: String,
    results: Vec<BenchmarkResult>,
}

static mut BENCHMARK_RESULTS: Option<Vec<BenchmarkResult>> = None;

fn init_results() {
    unsafe {
        BENCHMARK_RESULTS = Some(Vec::new());
    }
}

fn add_result(result: BenchmarkResult) {
    unsafe {
        if let Some(ref mut results) = BENCHMARK_RESULTS {
            results.push(result);
        }
    }
}

fn save_results(filename: &str) {
    unsafe {
        if let Some(ref results) = BENCHMARK_RESULTS {
            let suite = BenchmarkSuite {
                timestamp: {
                    let now = std::time::SystemTime::now();
                    let duration = now
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default();
                    format!("{}", duration.as_secs())
                },
                platform: std::env::consts::OS.to_string(),
                results: results.clone(),
            };

            let json = serde_json::to_string_pretty(&suite).expect("Failed to serialize results");

            let mut file = File::create(filename).expect("Failed to create results file");

            file.write_all(json.as_bytes())
                .expect("Failed to write results");

            println!(
                "\n✓ Saved {} benchmark results to {}",
                results.len(),
                filename
            );
        }
    }
}

// =============================================================================
// Array Operations
// =============================================================================

/// Benchmark array creation and basic operations
fn bench_array_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_comparison/array_ops");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [1000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Array creation (zeros)
        group.bench_with_input(BenchmarkId::new("zeros", size), &size, |b, &s| {
            b.iter(|| {
                let arr = Array1::<f64>::zeros(s);
                black_box(arr)
            })
        });

        // Array creation (arange)
        group.bench_with_input(BenchmarkId::new("arange", size), &size, |b, &s| {
            b.iter(|| {
                let arr = Array1::<f64>::from_shape_fn(s, |i| i as f64);
                black_box(arr)
            })
        });

        // Elementwise operations
        let a = Array1::<f64>::from_shape_fn(size, |i| i as f64);
        let b = Array1::<f64>::from_shape_fn(size, |i| (i + 1) as f64);

        group.bench_with_input(BenchmarkId::new("add", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = &a + &b;
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("multiply", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = &a * &b;
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("sum", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = a.sum();
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("mean", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = a.mean();
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Linear Algebra Operations
// =============================================================================

/// Benchmark linear algebra operations for Python comparison
fn bench_linalg_operations(c: &mut Criterion) {
    use scirs2_linalg::{det, inv, lu, qr, solve};

    let mut group = c.benchmark_group("python_comparison/linalg");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [50, 100, 200, 500];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = uniform_f64(-1.0, 1.0);

        let matrix = Array2::random_using((size, size), dist, &mut rng);
        let vector = Array1::random_using(size, dist, &mut rng);

        group.throughput(Throughput::Elements((size * size) as u64));

        // Matrix determinant
        group.bench_with_input(BenchmarkId::new("det", size), &size, |b, _| {
            b.iter(|| {
                let result = det(&matrix.view(), None);
                black_box(result)
            })
        });

        // Matrix inverse
        if size <= 200 {
            group.bench_with_input(BenchmarkId::new("inv", size), &size, |b, _| {
                b.iter(|| {
                    let result = inv(&matrix.view(), None);
                    black_box(result)
                })
            });
        }

        // LU decomposition
        group.bench_with_input(BenchmarkId::new("lu", size), &size, |b, _| {
            b.iter(|| {
                let result = lu(&matrix.view(), None);
                black_box(result)
            })
        });

        // QR decomposition
        group.bench_with_input(BenchmarkId::new("qr", size), &size, |b, _| {
            b.iter(|| {
                let result = qr(&matrix.view(), None);
                black_box(result)
            })
        });

        // Linear solve
        group.bench_with_input(BenchmarkId::new("solve", size), &size, |b, _| {
            b.iter(|| {
                let result = solve(&matrix.view(), &vector.view(), None);
                black_box(result)
            })
        });

        // Matrix multiplication
        let b_matrix = Array2::random_using((size, size), dist, &mut rng);
        group.bench_with_input(BenchmarkId::new("matmul", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = matrix.dot(&b_matrix);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// FFT Operations
// =============================================================================

/// Benchmark FFT operations for Python comparison
fn bench_fft_operations(c: &mut Criterion) {
    use scirs2_fft::{fft, ifft};

    let mut group = c.benchmark_group("python_comparison/fft");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [128, 512, 2048, 8192, 32768];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = uniform_f64(-1.0, 1.0);

        let signal = Array1::random_using(size, dist, &mut rng);
        let signal_slice: Vec<f64> = signal.iter().copied().collect();

        let complexity = (size as f64 * (size as f64).log2()) as u64;
        group.throughput(Throughput::Elements(complexity));

        // Forward FFT
        group.bench_with_input(BenchmarkId::new("fft", size), &size, |b, _| {
            b.iter(|| {
                let result = fft(&signal_slice, None);
                black_box(result)
            })
        });

        // Inverse FFT
        let freq = fft(&signal_slice, None).expect("FFT failed");
        group.bench_with_input(BenchmarkId::new("ifft", size), &size, |b, _| {
            b.iter(|| {
                let result = ifft(&freq, None);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Statistical Operations
// =============================================================================

/// Benchmark statistical operations for Python comparison
fn bench_stats_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("python_comparison/stats");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = uniform_f64(-1.0, 1.0);

        let data = Array1::random_using(size, dist, &mut rng);

        group.throughput(Throughput::Elements(size as u64));

        // Mean
        group.bench_with_input(BenchmarkId::new("mean", size), &size, |b, _| {
            b.iter(|| {
                let result = data.mean();
                black_box(result)
            })
        });

        // Standard deviation
        group.bench_with_input(BenchmarkId::new("std", size), &size, |b, _| {
            b.iter(|| {
                let result = data.std(0.0);
                black_box(result)
            })
        });

        // Variance
        group.bench_with_input(BenchmarkId::new("var", size), &size, |b, _| {
            b.iter(|| {
                let result = data.var(0.0);
                black_box(result)
            })
        });

        // Median (for smaller sizes)
        if size <= 100_000 {
            group.bench_with_input(BenchmarkId::new("median", size), &size, |b, _| {
                b.iter(|| {
                    let mut sorted_vec: Vec<f64> = data.iter().copied().collect();
                    sorted_vec.sort_by(|a, b_val| {
                        a.partial_cmp(b_val).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let mid = sorted_vec.len() / 2;
                    let result = if sorted_vec.len().is_multiple_of(2) {
                        (sorted_vec[mid - 1] + sorted_vec[mid]) / 2.0
                    } else {
                        sorted_vec[mid]
                    };
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

// =============================================================================
// Special Functions
// =============================================================================

/// Benchmark special mathematical functions for Python comparison
fn bench_special_functions(c: &mut Criterion) {
    use scirs2_special::{erf, erfc, gamma, j0, j1};

    let mut group = c.benchmark_group("python_comparison/special");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let n_points = 10_000;
    let x_values: Vec<f64> = (0..n_points).map(|i| (i as f64) * 0.01).collect();

    group.throughput(Throughput::Elements(n_points as u64));

    // Bessel J0
    group.bench_function("bessel_j0", |b| {
        b.iter(|| {
            let sum: f64 = x_values.iter().map(|&x| j0(x)).sum();
            black_box(sum)
        })
    });

    // Bessel J1
    group.bench_function("bessel_j1", |b| {
        b.iter(|| {
            let sum: f64 = x_values.iter().map(|&x| j1(x)).sum();
            black_box(sum)
        })
    });

    // Gamma function
    group.bench_function("gamma", |b| {
        b.iter(|| {
            let sum: f64 = x_values.iter().map(|&x| gamma(x + 1.0)).sum();
            black_box(sum)
        })
    });

    // Error function
    group.bench_function("erf", |b| {
        b.iter(|| {
            let sum: f64 = x_values.iter().map(|&x| erf(x)).sum();
            black_box(sum)
        })
    });

    // Complementary error function
    group.bench_function("erfc", |b| {
        b.iter(|| {
            let sum: f64 = x_values.iter().map(|&x| erfc(x)).sum();
            black_box(sum)
        })
    });

    group.finish();
}

// =============================================================================
// Integration Operations
// =============================================================================

/// Benchmark numerical integration for Python comparison
fn bench_integration_operations(c: &mut Criterion) {
    use scirs2_integrate::{quad, simpson, trapezoid};

    let mut group = c.benchmark_group("python_comparison/integration");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let f = |x: f64| x.sin();

    // Trapezoid rule
    let sample_counts = [100, 1000, 10_000];
    for &n in &sample_counts {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("trapz", n), &n, |b, &n_val| {
            b.iter(|| {
                let result = trapezoid(f, 0.0, std::f64::consts::PI, n_val);
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("simps", n), &n, |b, &n_val| {
            b.iter(|| {
                let result = simpson(f, 0.0, std::f64::consts::PI, n_val);
                black_box(result)
            })
        });
    }

    // Adaptive quadrature
    group.bench_function("quad", |b| {
        b.iter(|| {
            let result = quad(f, 0.0, std::f64::consts::PI, None);
            black_box(result)
        })
    });

    group.finish();
}

// =============================================================================
// Benchmarking Suite Orchestration
// =============================================================================

fn benchmark_startup(_: &mut Criterion) {
    init_results();
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     SciRS2 vs Python Performance Comparison Suite         ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  Results will be saved for Python comparison              ║");
    println!("║  Run: python benches/v020_python_comparison.py            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

fn benchmark_teardown(_: &mut Criterion) {
    let output_path = std::env::temp_dir().join("scirs2_v020_python_comparison_rust.json");
    save_results(&output_path.to_string_lossy());
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  Next: Run Python comparison script                       ║");
    println!("║  $ python benches/v020_python_comparison.py               ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(startup, benchmark_startup);

criterion_group!(array_benchmarks, bench_array_operations,);

criterion_group!(linalg_benchmarks, bench_linalg_operations,);

criterion_group!(fft_benchmarks, bench_fft_operations,);

criterion_group!(stats_benchmarks, bench_stats_operations,);

criterion_group!(special_benchmarks, bench_special_functions,);

criterion_group!(integration_benchmarks, bench_integration_operations,);

criterion_group!(teardown, benchmark_teardown);

criterion_main!(
    startup,
    array_benchmarks,
    linalg_benchmarks,
    fft_benchmarks,
    stats_benchmarks,
    special_benchmarks,
    integration_benchmarks,
    teardown,
);
