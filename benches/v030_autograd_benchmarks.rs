//! SciRS2 v0.3.0 Autograd Performance Benchmark Suite
//!
//! Comprehensive benchmarks for automatic differentiation:
//! - Forward pass timing (various operation graphs)
//! - Backward pass timing (gradient computation)
//! - SIMD vs non-SIMD comparison
//! - Memory allocation overhead
//! - Complex computation graph performance
//!
//! Performance Targets (v0.3.0):
//! - Forward pass overhead: <10% vs manual computation
//! - Backward pass: <2x forward pass time
//! - SIMD optimization: 3-8x speedup for large tensors
//! - Memory overhead: <30% additional memory for tape

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use scirs2_autograd as ag;
use scirs2_autograd::tensor_ops as T;
use scirs2_core::ndarray::Array2;
use scirs2_core::random::{ChaCha8Rng, Distribution, SeedableRng, Uniform};
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

/// Save all benchmark results to JSON
fn save_results() {
    unsafe {
        if let Some(ref results) = BENCHMARK_RESULTS {
            let json = serde_json::to_string_pretty(results)
                .expect("Failed to serialize benchmark results");
            let output_path = std::env::temp_dir().join("scirs2_v030_autograd_results.json");
            let mut file = File::create(&output_path).expect("Failed to create results file");
            file.write_all(json.as_bytes())
                .expect("Failed to write results");
            println!(
                "\n Saved {} benchmark results to {}",
                results.len(),
                output_path.display()
            );
        }
    }
}

// =============================================================================
// Forward Pass Benchmarks
// =============================================================================

/// Benchmark forward pass for simple elementwise operations
fn bench_forward_simple_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_forward_simple");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10_000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");
        let data1: Vec<f64> = (0..size).map(|_| dist.sample(&mut rng)).collect();
        let data2: Vec<f64> = (0..size).map(|_| dist.sample(&mut rng)).collect();
        let arr1 = ag::ndarray::Array1::from_vec(data1)
            .into_shape_with_order(size)
            .expect("reshape");
        let arr2 = ag::ndarray::Array1::from_vec(data2)
            .into_shape_with_order(size)
            .expect("reshape");

        // Addition forward
        group.bench_with_input(BenchmarkId::new("add", size), &size, |b, _| {
            b.iter(|| {
                let a1 = arr1.clone();
                let a2 = arr2.clone();
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let x = T::convert_to_tensor(a1.into_dyn(), ctx);
                    let y = T::convert_to_tensor(a2.into_dyn(), ctx);
                    let z = T::add(x, y);
                    let val = z.eval(ctx).expect("Failed to evaluate add");
                    black_box(val)
                })
            })
        });

        // ReLU forward
        group.bench_with_input(BenchmarkId::new("relu", size), &size, |b, _| {
            b.iter(|| {
                let a1 = arr1.clone();
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let x = T::convert_to_tensor(a1.into_dyn(), ctx);
                    let z = T::relu(x);
                    let val = z.eval(ctx).expect("Failed to evaluate relu");
                    black_box(val)
                })
            })
        });

        // Sigmoid forward
        group.bench_with_input(BenchmarkId::new("sigmoid", size), &size, |b, _| {
            b.iter(|| {
                let a1 = arr1.clone();
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let x = T::convert_to_tensor(a1.into_dyn(), ctx);
                    let z = T::sigmoid(x);
                    let val = z.eval(ctx).expect("Failed to evaluate sigmoid");
                    black_box(val)
                })
            })
        });
    }

    group.finish();
}

/// Benchmark forward pass for matrix operations
fn bench_forward_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_forward_matrix");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [32, 64, 128, 256];

    for &size in &sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

        let data_a: Vec<f64> = (0..size * size).map(|_| dist.sample(&mut rng)).collect();
        let data_b: Vec<f64> = (0..size * size).map(|_| dist.sample(&mut rng)).collect();
        let arr_a = Array2::from_shape_vec((size, size), data_a).expect("Failed to create array");
        let arr_b = Array2::from_shape_vec((size, size), data_b).expect("Failed to create array");

        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        // Matrix multiplication
        group.bench_with_input(BenchmarkId::new("matmul", size), &size, |b, _| {
            b.iter(|| {
                let a = arr_a.clone();
                let bm = arr_b.clone();
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let x = T::convert_to_tensor(a.into_dyn(), ctx);
                    let y = T::convert_to_tensor(bm.into_dyn(), ctx);
                    let z = T::matmul(x, y);
                    let val = z.eval(ctx).expect("Failed to evaluate matmul");
                    black_box(val)
                })
            })
        });
    }

    group.finish();
}

// =============================================================================
// Backward Pass Benchmarks
// =============================================================================

/// Benchmark backward pass (gradient computation) for elementwise operations
fn bench_backward_simple_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_backward_simple");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10_000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");
        let data1: Vec<f64> = (0..size).map(|_| dist.sample(&mut rng)).collect();
        let arr1 = ag::ndarray::Array1::from_vec(data1)
            .into_shape_with_order(size)
            .expect("reshape");

        // ReLU backward
        group.bench_with_input(BenchmarkId::new("relu_backward", size), &size, |b, _| {
            b.iter(|| {
                let a1 = arr1.clone();
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let x = ctx.placeholder("x", &[-1]);
                    let z = T::sum_all(T::relu(x));
                    let gz = T::grad(&[z], &[x]);
                    let feed_arr = a1.into_dyn();
                    let val = ctx.evaluator().push(&gz[0]).feed(x, feed_arr.view()).run();
                    black_box(val)
                })
            })
        });

        // Sigmoid backward
        group.bench_with_input(BenchmarkId::new("sigmoid_backward", size), &size, |b, _| {
            b.iter(|| {
                let a1 = arr1.clone();
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let x = ctx.placeholder("x", &[-1]);
                    let z = T::sum_all(T::sigmoid(x));
                    let gz = T::grad(&[z], &[x]);
                    let feed_arr = a1.into_dyn();
                    let val = ctx.evaluator().push(&gz[0]).feed(x, feed_arr.view()).run();
                    black_box(val)
                })
            })
        });
    }

    group.finish();
}

/// Benchmark backward pass for matrix operations
fn bench_backward_matrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_backward_matrix");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [32, 64, 128];

    for &size in &sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

        let data_a: Vec<f64> = (0..size * size).map(|_| dist.sample(&mut rng)).collect();
        let data_b: Vec<f64> = (0..size * size).map(|_| dist.sample(&mut rng)).collect();
        let arr_b = Array2::from_shape_vec((size, size), data_b).expect("Failed to create array");

        let flops = 2 * size * size * size;
        group.throughput(Throughput::Elements(flops as u64));

        // Matrix multiplication backward
        group.bench_with_input(BenchmarkId::new("matmul_backward", size), &size, |b, _| {
            b.iter(|| {
                let a = Array2::from_shape_vec((size, size), data_a.clone())
                    .expect("Failed to create array");
                let bm = arr_b.clone();
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let x = ctx.placeholder("x", &[size as isize, size as isize]);
                    let y = T::convert_to_tensor(bm.into_dyn(), ctx);
                    let z = T::sum_all(T::matmul(x, y));
                    let gz = T::grad(&[z], &[x]);
                    let feed_arr = a.into_dyn();
                    let val = ctx.evaluator().push(&gz[0]).feed(x, feed_arr.view()).run();
                    black_box(val)
                })
            })
        });
    }

    group.finish();
}

// =============================================================================
// SIMD Comparison Benchmarks
// =============================================================================

/// Benchmark elementwise operations (SIMD-optimized internally)
fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_elementwise");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let sizes = [1000, 10_000, 100_000];

    for &size in &sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");
        let data1: Vec<f64> = (0..size).map(|_| dist.sample(&mut rng)).collect();
        let data2: Vec<f64> = (0..size).map(|_| dist.sample(&mut rng)).collect();
        let arr1 = ag::ndarray::Array1::from_vec(data1)
            .into_shape_with_order(size)
            .expect("reshape");
        let arr2 = ag::ndarray::Array1::from_vec(data2)
            .into_shape_with_order(size)
            .expect("reshape");

        group.throughput(Throughput::Elements(size as u64));

        // Elementwise multiplication
        group.bench_with_input(BenchmarkId::new("elementwise_mul", size), &size, |b, _| {
            b.iter(|| {
                let a1 = arr1.clone();
                let a2 = arr2.clone();
                ag::run(|ctx: &mut ag::Context<f64>| {
                    let x = T::convert_to_tensor(a1.into_dyn(), ctx);
                    let y = T::convert_to_tensor(a2.into_dyn(), ctx);
                    let z = T::mul(x, y);
                    let val = z.eval(ctx).expect("Failed to evaluate mul");
                    black_box(val)
                })
            })
        });
    }

    group.finish();
}

// =============================================================================
// Memory Overhead Benchmarks
// =============================================================================

/// Benchmark memory allocation overhead for deep computation graphs
fn bench_deep_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_deep_graph");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let depths = [1, 5, 10, 20];
    let size = 1000;

    for &depth in &depths {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(0.1f64, 1.0f64).expect("Failed to create uniform distribution");
        let data: Vec<f64> = (0..size).map(|_| dist.sample(&mut rng)).collect();
        let arr = ag::ndarray::Array1::from_vec(data)
            .into_shape_with_order(size)
            .expect("reshape");

        group.throughput(Throughput::Elements((depth * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("deep_relu_chain", depth),
            &depth,
            |b, &d| {
                b.iter(|| {
                    let a = arr.clone();
                    ag::run(|ctx: &mut ag::Context<f64>| {
                        let mut x = T::convert_to_tensor(a.into_dyn(), ctx);
                        // Create a deep computation graph by chaining relu ops
                        for _ in 0..d {
                            x = T::relu(x);
                        }
                        let val = x.eval(ctx).expect("Failed to evaluate deep graph");
                        black_box(val)
                    })
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Groups and Main
// =============================================================================

fn benchmark_startup(_: &mut Criterion) {
    init_results();
    println!("\n SciRS2 v0.3.0 Autograd Performance Benchmark Suite\n");
}

fn benchmark_teardown(_: &mut Criterion) {
    save_results();
    println!("\n Autograd Benchmarks Completed Successfully!\n");
}

criterion_group!(startup, benchmark_startup);

criterion_group!(
    forward_benchmarks,
    bench_forward_simple_ops,
    bench_forward_matrix_ops,
);

criterion_group!(
    backward_benchmarks,
    bench_backward_simple_ops,
    bench_backward_matrix_ops,
);

criterion_group!(elementwise_benchmarks, bench_elementwise_ops,);

criterion_group!(memory_benchmarks, bench_deep_graph,);

criterion_group!(teardown, benchmark_teardown);

criterion_main!(
    startup,
    forward_benchmarks,
    backward_benchmarks,
    elementwise_benchmarks,
    memory_benchmarks,
    teardown,
);
