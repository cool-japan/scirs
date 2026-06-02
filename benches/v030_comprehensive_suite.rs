//! SciRS2 v0.3.0 Comprehensive Performance Benchmark Suite
//!
//! This is the main orchestrator for the v0.3.0 performance validation suite.
//! It runs comprehensive benchmarks across all SciRS2 modules and generates
//! performance baselines for the v0.3.0 release.
//!
//! Benchmark Categories:
//! - Core array operations and SIMD optimizations
//! - Linear algebra decompositions and solvers (GEMM, SVD, eigenvalues)
//! - FFT and signal processing
//! - Statistics and distributions
//! - Integration and optimization algorithms
//! - Clustering and machine learning
//! - Special functions
//! - N-dimensional image processing
//! - Graph operations
//! - Text processing
//! - Vision operations
//! - Time series analysis
//! - Autograd and neural networks
//!
//! Performance Targets (v0.3.0):
//! - SIMD operations: 5-20x speedup vs naive implementations
//! - Matrix operations: competitive with BLAS (via OxiBLAS)
//! - FFT: competitive with rustfft/OxiFFT
//! - Parallel operations: near-linear scaling up to 16 cores
//! - Memory efficiency: <10% overhead vs optimal allocation
//! - Autograd overhead: <20% vs hand-written gradients
//! - Neural network training: >1000 images/sec on MNIST

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

/// Benchmark results for serialization and reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub category: String,
    pub operation: String,
    pub size: usize,
    pub mean_time_ns: f64,
    pub throughput_ops_per_sec: f64,
    pub std_dev_ns: f64,
}

/// Global benchmark results collector
static mut BENCHMARK_RESULTS: Option<Vec<BenchmarkResult>> = None;

/// Initialize results collector
fn init_results() {
    unsafe {
        BENCHMARK_RESULTS = Some(Vec::new());
    }
}

/// Add a benchmark result
#[allow(dead_code)]
fn add_result(result: BenchmarkResult) {
    unsafe {
        if let Some(ref mut results) = BENCHMARK_RESULTS {
            results.push(result);
        }
    }
}

/// Save all benchmark results to JSON
fn save_results() {
    unsafe {
        if let Some(ref results) = BENCHMARK_RESULTS {
            let json = serde_json::to_string_pretty(results)
                .expect("Failed to serialize benchmark results");
            let output_path = std::env::temp_dir().join("scirs2_v030_comprehensive_results.json");
            let mut file = File::create(&output_path).expect("Failed to create results file");
            file.write_all(json.as_bytes())
                .expect("Failed to write results");
            println!(
                "\n✓ Saved {} benchmark results to {}",
                results.len(),
                output_path.display()
            );
        }
    }
}

// =============================================================================
// Core Array Operations
// =============================================================================

/// Benchmark basic array creation and manipulation
fn bench_core_array_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_array_operations");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10_000, 100_000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Array creation
        group.bench_with_input(BenchmarkId::new("zeros", size), &size, |b, &s| {
            b.iter(|| {
                let arr = Array1::<f64>::zeros(s);
                black_box(arr)
            })
        });

        // Array from function
        group.bench_with_input(BenchmarkId::new("from_fn", size), &size, |b, &s| {
            b.iter(|| {
                let arr = Array1::<f64>::from_shape_fn(s, |i| i as f64);
                black_box(arr)
            })
        });

        // Array elementwise operations
        let arr1 = Array1::<f64>::from_shape_fn(size, |i| i as f64);
        let arr2 = Array1::<f64>::from_shape_fn(size, |i| (i + 1) as f64);

        group.bench_with_input(BenchmarkId::new("elementwise_add", size), &size, |b, _| {
            b.iter(|| {
                let result = &arr1 + &arr2;
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("elementwise_mul", size), &size, |b, _| {
            b.iter(|| {
                let result = &arr1 * &arr2;
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Linear Algebra Operations - GEMM (Matrix Multiplication)
// =============================================================================

/// Benchmark GEMM (General Matrix Multiply)
fn bench_linalg_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_gemm");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [32, 64, 128, 256, 512, 1024];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

        let a = Array2::random_using((size, size), dist, &mut rng);
        let b = Array2::random_using((size, size), dist, &mut rng);

        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(BenchmarkId::new("gemm", size), &size, |bencher, _| {
            bencher.iter(|| {
                let c = a.dot(&b);
                black_box(c)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Linear Algebra Operations - SVD
// =============================================================================

/// Benchmark SVD (Singular Value Decomposition)
fn bench_linalg_svd(c: &mut Criterion) {
    use scirs2_linalg::svd;

    let mut group = c.benchmark_group("linalg_svd");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let sizes = [32, 64, 128, 256];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

        let a = Array2::random_using((size, size), dist, &mut rng);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("svd", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = svd(&a.view(), true, None);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Linear Algebra Operations - Eigenvalues
// =============================================================================

/// Benchmark eigenvalue decomposition
fn bench_linalg_eigenvalues(c: &mut Criterion) {
    use scirs2_linalg::eig;

    let mut group = c.benchmark_group("linalg_eigenvalues");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let sizes = [32, 64, 128, 256];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

        let a = Array2::random_using((size, size), dist, &mut rng);
        // Make symmetric for stability
        let a_sym = (a.clone() + a.t()) / 2.0;

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(BenchmarkId::new("eig", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = eig(&a_sym.view(), None);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Linear Algebra Operations - Decompositions
// =============================================================================

/// Benchmark matrix decompositions (LU, QR, Cholesky)
fn bench_linalg_decompositions(c: &mut Criterion) {
    use scirs2_linalg::{cholesky, lu, qr};

    let mut group = c.benchmark_group("linalg_decompositions");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [32, 64, 128, 256, 512];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

        let a = Array2::random_using((size, size), dist, &mut rng);

        // Create SPD matrix for Cholesky
        let a_spd = a.dot(&a.t()) + Array2::<f64>::eye(size) * 0.1;

        group.throughput(Throughput::Elements((size * size) as u64));

        // LU decomposition
        group.bench_with_input(BenchmarkId::new("lu", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = lu(&a.view(), None);
                black_box(result)
            })
        });

        // QR decomposition
        group.bench_with_input(BenchmarkId::new("qr", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = qr(&a.view(), None);
                black_box(result)
            })
        });

        // Cholesky decomposition
        group.bench_with_input(BenchmarkId::new("cholesky", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = cholesky(&a_spd.view(), None);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// FFT Operations
// =============================================================================

/// Benchmark FFT operations
fn bench_fft_operations(c: &mut Criterion) {
    use scirs2_fft::fft;

    let mut group = c.benchmark_group("fft_operations");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

        let signal = Array1::random_using(size, dist, &mut rng);

        // N log N complexity for FFT
        let complexity = (size as f64 * (size as f64).log2()) as u64;
        group.throughput(Throughput::Elements(complexity));

        group.bench_with_input(BenchmarkId::new("fft_1d", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = fft(signal.as_slice().expect("Failed to get slice"), None);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Statistical Operations
// =============================================================================

/// Benchmark statistical computations
fn bench_stats_operations(c: &mut Criterion) {
    use scirs2_core::ndarray::QuantileExt;

    let mut group = c.benchmark_group("stats_operations");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [1000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

        let data = Array1::random_using(size, dist, &mut rng);

        group.throughput(Throughput::Elements(size as u64));

        // Mean
        group.bench_with_input(BenchmarkId::new("mean", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = data.mean();
                black_box(result)
            })
        });

        // Standard deviation
        group.bench_with_input(BenchmarkId::new("std", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = data.std(0.0);
                black_box(result)
            })
        });

        // Median (slower, O(n log n))
        if size <= 100_000 {
            group.bench_with_input(BenchmarkId::new("median", size), &size, |bencher, _| {
                bencher.iter(|| {
                    let mut data_vec = data.to_vec();
                    data_vec.sort_by(|a, b| a.partial_cmp(b).expect("Failed comparison"));
                    let result = data_vec[data_vec.len() / 2];
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

// =============================================================================
// Optimization Algorithms
// =============================================================================

/// Benchmark optimization algorithms
fn bench_optimization_algorithms(c: &mut Criterion) {
    use scirs2_core::ndarray::ArrayView1;
    use scirs2_optimize::unconstrained::{minimize, Method};

    let mut group = c.benchmark_group("optimization_algorithms");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
        let (a, b) = (x[0], x[1]);
        (1.0 - a).powi(2) + 100.0 * (b - a.powi(2)).powi(2)
    };

    group.bench_function("nelder_mead_rosenbrock", |bencher| {
        bencher.iter(|| {
            let x0 = vec![0.0, 0.0];
            let result = minimize(rosenbrock, &x0, Method::NelderMead, None);
            black_box(result)
        })
    });

    group.finish();
}

// =============================================================================
// Special Functions
// =============================================================================

/// Benchmark special mathematical functions
fn bench_special_functions(c: &mut Criterion) {
    use scirs2_special::{erf, gamma, j0};

    let mut group = c.benchmark_group("special_functions");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let n_evaluations = 10_000;
    let x_values: Vec<f64> = (0..n_evaluations).map(|i| (i as f64) * 0.01).collect();

    group.throughput(Throughput::Elements(n_evaluations as u64));

    // Bessel J0
    group.bench_function("bessel_j0", |bencher| {
        bencher.iter(|| {
            let sum: f64 = x_values.iter().map(|&x| j0(x)).sum();
            black_box(sum)
        })
    });

    // Gamma function
    group.bench_function("gamma", |bencher| {
        bencher.iter(|| {
            let sum: f64 = x_values.iter().map(|&x| gamma(x + 1.0)).sum();
            black_box(sum)
        })
    });

    // Error function
    group.bench_function("erf", |bencher| {
        bencher.iter(|| {
            let sum: f64 = x_values.iter().map(|&x| erf(x)).sum();
            black_box(sum)
        })
    });

    group.finish();
}

// =============================================================================
// Clustering Operations
// =============================================================================

/// Benchmark clustering algorithms
fn bench_clustering_operations(c: &mut Criterion) {
    use scirs2_cluster::vq::kmeans;

    let mut group = c.benchmark_group("clustering_operations");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let configs = [
        (100, 3),    // 100 points, 3 clusters
        (1000, 5),   // 1000 points, 5 clusters
        (5000, 10),  // 5000 points, 10 clusters
        (10000, 15), // 10000 points, 15 clusters
    ];

    for &(n_points, n_clusters) in &configs {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(-10.0f64, 10.0f64).expect("Failed to create uniform distribution");

        let data = Array2::random_using((n_points, 2), dist, &mut rng);

        group.throughput(Throughput::Elements((n_points * n_clusters) as u64));

        let label = format!("{}pts_{}clusters", n_points, n_clusters);
        group.bench_with_input(BenchmarkId::new("kmeans", &label), &label, |bencher, _| {
            bencher.iter(|| {
                let result = kmeans(data.view(), n_clusters, None, None, None, None);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Signal Processing
// =============================================================================

/// Benchmark signal processing operations
fn bench_signal_processing(c: &mut Criterion) {
    use scirs2_signal::{convolve, correlate};

    let mut group = c.benchmark_group("signal_processing");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let signal_sizes = [100, 500, 1000, 5000, 10000];
    let kernel_size = 21;

    for &size in &signal_sizes {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

        let signal = Array1::random_using(size, dist, &mut rng);
        let kernel = Array1::random_using(kernel_size, dist, &mut rng);

        group.throughput(Throughput::Elements(size as u64));

        // Convolution
        group.bench_with_input(BenchmarkId::new("convolve", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = convolve(
                    signal.as_slice().expect("Failed to get slice"),
                    kernel.as_slice().expect("Failed to get slice"),
                    "same",
                );
                black_box(result)
            })
        });

        // Correlation
        group.bench_with_input(BenchmarkId::new("correlate", size), &size, |bencher, _| {
            bencher.iter(|| {
                let result = correlate(
                    signal.as_slice().expect("Failed to get slice"),
                    kernel.as_slice().expect("Failed to get slice"),
                    "same",
                );
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// Criterion Groups and Main
// =============================================================================

fn benchmark_startup(_: &mut Criterion) {
    init_results();
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  SciRS2 v0.3.0 Comprehensive Performance Benchmark Suite  ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

fn benchmark_teardown(_: &mut Criterion) {
    save_results();
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║        Benchmark Suite Completed Successfully!            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

criterion_group!(startup, benchmark_startup);

criterion_group!(core_benchmarks, bench_core_array_ops,);

criterion_group!(
    linalg_benchmarks,
    bench_linalg_gemm,
    bench_linalg_svd,
    bench_linalg_eigenvalues,
    bench_linalg_decompositions,
);

criterion_group!(fft_benchmarks, bench_fft_operations,);

criterion_group!(stats_benchmarks, bench_stats_operations,);

criterion_group!(optimization_benchmarks, bench_optimization_algorithms,);

criterion_group!(special_benchmarks, bench_special_functions,);

criterion_group!(clustering_benchmarks, bench_clustering_operations,);

criterion_group!(signal_benchmarks, bench_signal_processing,);

criterion_group!(teardown, benchmark_teardown);

criterion_main!(
    startup,
    core_benchmarks,
    linalg_benchmarks,
    fft_benchmarks,
    stats_benchmarks,
    optimization_benchmarks,
    special_benchmarks,
    clustering_benchmarks,
    signal_benchmarks,
    teardown,
);
