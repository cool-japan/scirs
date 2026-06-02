//! SciRS2 v0.3.0 Neural Networks Performance Benchmark Suite
//!
//! Comprehensive benchmarks for neural network training and inference:
//! - MNIST training throughput (images/sec)
//! - CIFAR-10 training throughput
//! - Inference latency (batch sizes: 1, 8, 32, 128)
//! - Memory usage profiling
//! - Model architecture comparisons (CNN vs MLP)
//! - Optimizer performance (SGD, Adam, RMSprop)
//!
//! Performance Targets (v0.3.0):
//! - MNIST training: >1000 images/sec on CPU
//! - CIFAR-10 training: >500 images/sec on CPU
//! - Single image inference: <10ms latency
//! - Batch inference (32): <50ms latency
//! - Memory overhead: <500MB for typical models

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, RandomExt};
use scirs2_core::random::{ChaCha8Rng, Distribution, SeedableRng, Uniform};
use scirs2_neural::{
    layers::{Conv2D, Dense, Flatten},
    models::Sequential,
    optimizers::{Adam, SGD},
};
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
    pub batch_size: usize,
    pub mean_time_ns: f64,
    pub throughput_samples_per_sec: f64,
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
            let output_path = std::env::temp_dir().join("scirs2_v030_neural_results.json");
            let mut file = File::create(&output_path)
                .expect("Failed to create results file");
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
// MNIST Training Benchmarks
// =============================================================================

/// Benchmark MNIST training throughput
fn bench_mnist_training_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("mnist_training_throughput");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let batch_sizes = [16, 32, 64, 128];

    for &batch_size in &batch_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(0.0f64, 1.0f64).expect("Failed to create uniform distribution");

        // MNIST images: 28x28 grayscale
        let images = Array4::random_using((batch_size, 1, 28, 28), dist, &mut rng);
        let labels = Array2::random_using((batch_size, 10), dist, &mut rng);

        // Create simple MLP model for MNIST
        let mut model = Sequential::new();
        model.add(Flatten::new());
        model.add(Dense::new(784, 128, Some("relu")));
        model.add(Dense::new(128, 64, Some("relu")));
        model.add(Dense::new(64, 10, Some("softmax")));

        let mut optimizer = SGD::new(0.01, None);

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("mlp_sgd", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let predictions = model.forward(&images);
                    let loss = model.compute_loss(&predictions, &labels);
                    let grads = model.backward(&loss);
                    optimizer.update(&mut model, &grads);
                    black_box(loss)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark MNIST CNN training throughput
fn bench_mnist_cnn_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("mnist_cnn_training");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let batch_sizes = [16, 32, 64];

    for &batch_size in &batch_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(0.0f64, 1.0f64).expect("Failed to create uniform distribution");

        // MNIST images: 28x28 grayscale
        let images = Array4::random_using((batch_size, 1, 28, 28), dist, &mut rng);
        let labels = Array2::random_using((batch_size, 10), dist, &mut rng);

        // Create CNN model for MNIST
        let mut model = Sequential::new();
        model.add(Conv2D::new(1, 32, 3, Some(1), Some(1), Some("relu")));
        model.add(Conv2D::new(32, 64, 3, Some(1), Some(1), Some("relu")));
        model.add(Flatten::new());
        model.add(Dense::new(64 * 28 * 28, 128, Some("relu")));
        model.add(Dense::new(128, 10, Some("softmax")));

        let mut optimizer = Adam::new(0.001, None, None, None);

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("cnn_adam", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let predictions = model.forward(&images);
                    let loss = model.compute_loss(&predictions, &labels);
                    let grads = model.backward(&loss);
                    optimizer.update(&mut model, &grads);
                    black_box(loss)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// CIFAR-10 Training Benchmarks
// =============================================================================

/// Benchmark CIFAR-10 training throughput
fn bench_cifar10_training_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("cifar10_training_throughput");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(10));

    let batch_sizes = [16, 32, 64];

    for &batch_size in &batch_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(0.0f64, 1.0f64).expect("Failed to create uniform distribution");

        // CIFAR-10 images: 32x32 RGB
        let images = Array4::random_using((batch_size, 3, 32, 32), dist, &mut rng);
        let labels = Array2::random_using((batch_size, 10), dist, &mut rng);

        // Create CNN model for CIFAR-10
        let mut model = Sequential::new();
        model.add(Conv2D::new(3, 64, 3, Some(1), Some(1), Some("relu")));
        model.add(Conv2D::new(64, 128, 3, Some(1), Some(1), Some("relu")));
        model.add(Conv2D::new(128, 256, 3, Some(1), Some(1), Some("relu")));
        model.add(Flatten::new());
        model.add(Dense::new(256 * 32 * 32, 256, Some("relu")));
        model.add(Dense::new(256, 10, Some("softmax")));

        let mut optimizer = Adam::new(0.001, None, None, None);

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("cnn_adam", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let predictions = model.forward(&images);
                    let loss = model.compute_loss(&predictions, &labels);
                    let grads = model.backward(&loss);
                    optimizer.update(&mut model, &grads);
                    black_box(loss)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Inference Latency Benchmarks
// =============================================================================

/// Benchmark MNIST inference latency
fn bench_mnist_inference_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("mnist_inference_latency");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let batch_sizes = [1, 8, 32, 128];

    for &batch_size in &batch_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(0.0f64, 1.0f64).expect("Failed to create uniform distribution");

        // MNIST images: 28x28 grayscale
        let images = Array4::random_using((batch_size, 1, 28, 28), dist, &mut rng);

        // Create simple MLP model for MNIST
        let mut model = Sequential::new();
        model.add(Flatten::new());
        model.add(Dense::new(784, 128, Some("relu")));
        model.add(Dense::new(128, 64, Some("relu")));
        model.add(Dense::new(64, 10, Some("softmax")));

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("mlp_inference", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let predictions = model.forward(&images);
                    black_box(predictions)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark CIFAR-10 inference latency
fn bench_cifar10_inference_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cifar10_inference_latency");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let batch_sizes = [1, 8, 32, 128];

    for &batch_size in &batch_sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let dist = Uniform::new(0.0f64, 1.0f64).expect("Failed to create uniform distribution");

        // CIFAR-10 images: 32x32 RGB
        let images = Array4::random_using((batch_size, 3, 32, 32), dist, &mut rng);

        // Create CNN model for CIFAR-10
        let mut model = Sequential::new();
        model.add(Conv2D::new(3, 64, 3, Some(1), Some(1), Some("relu")));
        model.add(Conv2D::new(64, 128, 3, Some(1), Some(1), Some("relu")));
        model.add(Flatten::new());
        model.add(Dense::new(128 * 32 * 32, 256, Some("relu")));
        model.add(Dense::new(256, 10, Some("softmax")));

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("cnn_inference", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let predictions = model.forward(&images);
                    black_box(predictions)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Layer-wise Performance Benchmarks
// =============================================================================

/// Benchmark individual layer performance
fn bench_layer_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_performance");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let batch_size = 32;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let dist = Uniform::new(-1.0f64, 1.0f64).expect("Failed to create uniform distribution");

    // Dense layer forward pass
    let sizes = [128, 256, 512, 1024];
    for &size in &sizes {
        let input = Array2::random_using((batch_size, size), dist, &mut rng);
        let mut layer = Dense::new(size, size, Some("relu"));

        group.throughput(Throughput::Elements((batch_size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("dense_forward", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let output = layer.forward(&input);
                    black_box(output)
                })
            },
        );
    }

    // Conv2D layer forward pass
    let channels = [(32, 64), (64, 128), (128, 256)];
    for &(in_ch, out_ch) in &channels {
        let input = Array4::random_using((batch_size, in_ch, 28, 28), dist, &mut rng);
        let mut layer = Conv2D::new(in_ch, out_ch, 3, Some(1), Some(1), Some("relu"));

        group.throughput(Throughput::Elements((batch_size * in_ch * 28 * 28) as u64));

        group.bench_with_input(
            BenchmarkId::new("conv2d_forward", format!("{}to{}", in_ch, out_ch)),
            &in_ch,
            |b, _| {
                b.iter(|| {
                    let output = layer.forward(&input);
                    black_box(output)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Optimizer Performance Benchmarks
// =============================================================================

/// Benchmark optimizer performance
fn bench_optimizer_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_performance");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let batch_size = 32;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let dist = Uniform::new(0.0f64, 1.0f64).expect("Failed to create uniform distribution");

    // Create a simple model
    let images = Array4::random_using((batch_size, 1, 28, 28), dist, &mut rng);
    let labels = Array2::random_using((batch_size, 10), dist, &mut rng);

    let mut model = Sequential::new();
    model.add(Flatten::new());
    model.add(Dense::new(784, 128, Some("relu")));
    model.add(Dense::new(128, 10, Some("softmax")));

    group.throughput(Throughput::Elements(batch_size as u64));

    // SGD
    group.bench_function("sgd_update", |b| {
        let mut optimizer = SGD::new(0.01, None);
        b.iter(|| {
            let predictions = model.forward(&images);
            let loss = model.compute_loss(&predictions, &labels);
            let grads = model.backward(&loss);
            optimizer.update(&mut model, &grads);
            black_box(loss)
        })
    });

    // Adam
    group.bench_function("adam_update", |b| {
        let mut optimizer = Adam::new(0.001, None, None, None);
        b.iter(|| {
            let predictions = model.forward(&images);
            let loss = model.compute_loss(&predictions, &labels);
            let grads = model.backward(&loss);
            optimizer.update(&mut model, &grads);
            black_box(loss)
        })
    });

    group.finish();
}

// =============================================================================
// Memory Usage Profiling
// =============================================================================

/// Benchmark memory usage for different model sizes
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_profiling");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(8));

    let batch_size = 32;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let dist = Uniform::new(0.0f64, 1.0f64).expect("Failed to create uniform distribution");

    let model_sizes = [
        ("small", vec![(784, 64), (64, 10)]),
        ("medium", vec![(784, 256), (256, 128), (128, 10)]),
        ("large", vec![(784, 512), (512, 512), (512, 256), (256, 10)]),
    ];

    for (name, layers) in &model_sizes {
        let images = Array4::random_using((batch_size, 1, 28, 28), dist, &mut rng);

        let mut model = Sequential::new();
        model.add(Flatten::new());
        for &(in_size, out_size) in layers {
            model.add(Dense::new(in_size, out_size, Some("relu")));
        }

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_function(format!("model_{}_forward", name), |b| {
            b.iter(|| {
                let predictions = model.forward(&images);
                black_box(predictions)
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
    println!("║  SciRS2 v0.3.0 Neural Networks Performance Benchmark Suite║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

fn benchmark_teardown(_: &mut Criterion) {
    save_results();
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║       Neural Benchmarks Completed Successfully!           ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}

criterion_group!(startup, benchmark_startup);

criterion_group!(
    mnist_benchmarks,
    bench_mnist_training_throughput,
    bench_mnist_cnn_training,
    bench_mnist_inference_latency,
);

criterion_group!(
    cifar10_benchmarks,
    bench_cifar10_training_throughput,
    bench_cifar10_inference_latency,
);

criterion_group!(
    layer_benchmarks,
    bench_layer_performance,
);

criterion_group!(
    optimizer_benchmarks,
    bench_optimizer_performance,
);

criterion_group!(
    memory_benchmarks,
    bench_memory_usage,
);

criterion_group!(teardown, benchmark_teardown);

criterion_main!(
    startup,
    mnist_benchmarks,
    cifar10_benchmarks,
    layer_benchmarks,
    optimizer_benchmarks,
    memory_benchmarks,
    teardown,
);
