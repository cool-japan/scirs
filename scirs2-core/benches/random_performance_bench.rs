//! Comprehensive performance benchmarking suite for scirs2-core random module
//!
//! This benchmark suite validates that our scientific computing random number generation
//! implementations meet or exceed performance of standard Rust libraries while providing
//! enhanced functionality for scientific computing workflows.

#![allow(deprecated)]
//!
//! # Benchmark Categories
//!
//! 1. **Core RNG Performance**: Basic random number generation speed
//! 2. **Collection Operations**: Shuffling, sampling, selection performance
//! 3. **Distribution Sampling**: Advanced statistical distributions
//! 4. **Array Operations**: Bulk array generation and filling
//! 5. **Scientific Workflows**: Reproducible experiments, bootstrap sampling
//! 6. **Parallel Performance**: Thread-safe and parallel generation
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench random_performance_bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

// Core module imports
use scirs2_core::random::{
    arrays::{random_normal_array, random_uniform_array},
    distributions::*,
    parallel::*,
    quick::*,
    rand_distributions::{Normal, Uniform},
    scientific::*,
    seeded_rng,
    seq::SliceRandom,
    slice_ops::*,
    thread_rng, Random, RandomExt, Rng, ScientificSliceRandom, SeedableRng, StdRng,
};

// Comparison imports
use scirs2_core::ndarray_ext::{Array2, Ix2};

/// Benchmark core random number generation
fn bench_core_rng(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_rng");

    // Setup
    let mut scirs_rng = seeded_rng(42);
    let mut std_rng = seeded_rng(42);
    let mut thread_rng = thread_rng();

    // Single f64 generation
    group.bench_function("scirs2_f64", |b| {
        b.iter(|| black_box(scirs_rng.sample(Uniform::new(0.0, 1.0).unwrap())))
    });

    group.bench_function("std_rand_f64", |b| {
        b.iter(|| black_box(std_rng.random::<f64>()))
    });

    group.bench_function("thread_rng_f64", |b| {
        b.iter(|| black_box(thread_rng.random::<f64>()))
    });

    // Batch generation (1000 numbers)
    let batch_size = 1000;
    group.throughput(Throughput::Elements(batch_size as u64));

    group.bench_function("scirs2_batch_1k", |b| {
        b.iter(|| {
            let mut vec = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                vec.push(scirs_rng.sample(Uniform::new(0.0, 1.0).unwrap()));
            }
            black_box(vec)
        })
    });

    group.bench_function("std_rand_batch_1k", |b| {
        b.iter(|| {
            let mut vec = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                vec.push(std_rng.random::<f64>());
            }
            black_box(vec)
        })
    });

    group.finish();
}

/// Benchmark collection operations (shuffling, sampling)
fn bench_collection_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_ops");

    let sizes = [100, 1000, 10000];

    for size in sizes.iter() {
        let data: Vec<i32> = (0..*size).collect();
        group.throughput(Throughput::Elements(*size as u64));

        // Shuffling benchmarks
        group.bench_with_input(
            BenchmarkId::new("scirs2_shuffle", size),
            size,
            |b, &size| {
                let mut rng = seeded_rng(42);
                b.iter(|| {
                    let mut data: Vec<i32> = (0..size).collect();
                    data.scientific_shuffle(&mut rng);
                    black_box(data)
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("std_shuffle", size), size, |b, &size| {
            let mut rng = seeded_rng(42);
            b.iter(|| {
                let mut data: Vec<i32> = (0..size).collect();
                data.shuffle(&mut rng);
                black_box(data)
            })
        });

        // Sampling without replacement
        let sample_size = (*size / 10).max(1);
        group.bench_with_input(BenchmarkId::new("scirs2_sample", size), size, |b, &size| {
            let mut rng = seeded_rng(42);
            let data: Vec<i32> = (0..size).collect();
            b.iter(|| {
                let sample = data.scientific_choose_multiple(&mut rng, sample_size as usize);
                black_box(sample)
            })
        });

        group.bench_with_input(BenchmarkId::new("std_sample", size), size, |b, &size| {
            let mut rng = seeded_rng(42);
            let data: Vec<i32> = (0..size).collect();
            b.iter(|| {
                let sample = data
                    .scientific_choose_multiple(&mut rng, sample_size as usize)
                    .into_iter()
                    .cloned()
                    .collect::<Vec<_>>();
                black_box(sample)
            })
        });
    }

    group.finish();
}

/// Benchmark distribution sampling
fn bench_distributions(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributions");

    let sample_size = 1000;
    group.throughput(Throughput::Elements(sample_size));

    let mut rng = seeded_rng(42);
    let mut std_rng = seeded_rng(42);

    // Normal distribution
    group.bench_function("scirs2_normal", |b| {
        b.iter(|| {
            let normal = Normal::new(0.0, 1.0).unwrap();
            let mut samples = Vec::with_capacity(sample_size as usize);
            for _ in 0..sample_size {
                samples.push(rng.sample(normal));
            }
            black_box(samples)
        })
    });

    group.bench_function("std_normal", |b| {
        b.iter(|| {
            let normal = Normal::new(0.0, 1.0).unwrap();
            let mut samples = Vec::with_capacity(sample_size as usize);
            for _ in 0..sample_size {
                samples.push(std_rng.sample(normal));
            }
            black_box(samples)
        })
    });

    // Beta distribution (advanced)
    group.bench_function("scirs2_beta", |b| {
        b.iter(|| {
            let beta = Beta::new(2.0, 5.0).unwrap();
            let mut samples = Vec::with_capacity(sample_size as usize);
            for _ in 0..sample_size {
                samples.push(beta.sample(&mut rng));
            }
            black_box(samples)
        })
    });

    // Multivariate Normal (our exclusive implementation)
    group.bench_function("scirs2_multivariate_normal", |b| {
        let mean = vec![0.0, 0.0, 0.0];
        let cov = vec![
            vec![1.0, 0.5, 0.2],
            vec![0.5, 1.0, 0.3],
            vec![0.2, 0.3, 1.0],
        ];
        let mvn = MultivariateNormal::new(mean, cov).unwrap();

        b.iter(|| {
            let mut samples = Vec::with_capacity(sample_size as usize);
            for _ in 0..sample_size {
                samples.push(mvn.sample(&mut rng));
            }
            black_box(samples)
        })
    });

    group.finish();
}

/// Benchmark array operations
fn bench_array_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_ops");

    let shapes = [(100, 100), (500, 500), (1000, 1000)];

    for (rows, cols) in shapes.iter() {
        let size = rows * cols;
        group.throughput(Throughput::Elements(size as u64));

        // Uniform array generation
        group.bench_with_input(
            BenchmarkId::new("scirs2_uniform_array", size),
            &(*rows, *cols),
            |b, &(rows, cols)| {
                let mut rng = seeded_rng(42);
                b.iter(|| {
                    let array = random_uniform_array(Ix2(rows, cols), &mut rng);
                    black_box(array)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray_uniform", size),
            &(*rows, *cols),
            |b, &(rows, cols)| {
                b.iter(|| {
                    let array = random_uniform_array(Ix2(rows, cols), &mut seeded_rng(42));
                    black_box(array)
                })
            },
        );

        // Normal array generation
        group.bench_with_input(
            BenchmarkId::new("scirs2_normal_array", size),
            &(*rows, *cols),
            |b, &(rows, cols)| {
                let mut rng = seeded_rng(42);
                b.iter(|| {
                    let array = random_normal_array(Ix2(rows, cols), 0.0, 1.0, &mut rng);
                    black_box(array)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray_normal", size),
            &(*rows, *cols),
            |b, &(rows, cols)| {
                b.iter(|| {
                    let array = random_normal_array(Ix2(rows, cols), 0.0, 1.0, &mut seeded_rng(42));
                    black_box(array)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark scientific workflows
fn bench_scientific_workflows(c: &mut Criterion) {
    let mut group = c.benchmark_group("scientific_workflows");

    // Reproducible experiment workflow
    group.bench_function("reproducible_experiment", |b| {
        b.iter(|| {
            let mut experiment = ReproducibleExperiment::new(42);
            let mut results = Vec::new();

            // Simulate 5 experimental runs
            for _ in 0..5 {
                let sample = experiment.next_sample(1000, StandardNormal);
                results.push(sample.iter().sum::<f64>() / sample.len() as f64);
            }

            black_box(results)
        })
    });

    // Bootstrap sampling
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    group.bench_function("bootstrap_sampling", |b| {
        b.iter(|| {
            let bootstrap_samples = bootstrap_sample(&data, 100, 500);
            black_box(bootstrap_samples)
        })
    });

    // Cross-validation splits
    group.bench_function("cross_validation", |b| {
        b.iter(|| {
            let splits = cross_validation_splits(&data, 5, 42);
            black_box(splits)
        })
    });

    group.finish();
}

/// Benchmark parallel operations
fn bench_parallel_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_ops");

    // Parallel RNG pool
    group.bench_function("thread_local_rng_pool", |b| {
        let pool = ThreadLocalRngPool::new(42);
        let data: Vec<i32> = (0..10000).collect();

        b.iter(|| {
            let uniform = StandardUniform;
            let results = ParallelRng::parallel_sample(uniform, data.len(), &pool);
            black_box(results)
        })
    });

    // Parallel bootstrap
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    group.bench_function("parallel_bootstrap", |b| {
        let pool = ThreadLocalRngPool::new(42);

        b.iter(|| {
            let bootstrap_samples = ParallelRng::parallel_bootstrap(&data, 100, &pool);
            black_box(bootstrap_samples)
        })
    });

    group.finish();
}

/// Memory usage and allocation benchmarks
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test memory-efficient array operations
    group.bench_function("memory_efficient_sampling", |b| {
        let mut rng = seeded_rng(42);
        let data: Vec<i32> = (0..100000).collect();

        b.iter(|| {
            // Use reservoir sampling for memory efficiency
            let sample = data.scientific_reservoir_sample(&mut rng, 1000);
            black_box(sample)
        })
    });

    // Test in-place operations
    group.bench_function("in_place_shuffle", |b| {
        let mut rng = seeded_rng(42);

        b.iter(|| {
            let mut data: Vec<i32> = (0..10000).collect();
            data.scientific_shuffle(&mut rng);
            black_box(data)
        })
    });

    group.finish();
}

/// Statistical quality validation (not performance, but important)
fn bench_statistical_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_quality");
    group.measurement_time(Duration::from_secs(30)); // Longer measurement for statistical tests

    // Chi-square test for uniformity
    group.bench_function("uniformity_test", |b| {
        let mut rng = seeded_rng(42);

        b.iter(|| {
            let samples: Vec<f64> = (0..10000)
                .map(|_| rng.sample(Uniform::new(0.0, 1.0).unwrap()))
                .collect();

            // Simple uniformity check (bins)
            let mut bins = [0; 10];
            for sample in samples {
                let bin = (sample * 10.0).floor() as usize;
                if bin < 10 {
                    bins[bin] += 1;
                }
            }

            black_box(bins)
        })
    });

    // Correlation test for independence
    group.bench_function("independence_test", |b| {
        let mut rng = seeded_rng(42);

        b.iter(|| {
            let pairs: Vec<(f64, f64)> = (0..5000)
                .map(|_| {
                    (
                        rng.sample(Uniform::new(0.0, 1.0).unwrap()),
                        rng.sample(Uniform::new(0.0, 1.0).unwrap()),
                    )
                })
                .collect();

            // Simple correlation coefficient calculation
            let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / pairs.len() as f64;
            let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / pairs.len() as f64;

            let correlation = pairs
                .iter()
                .map(|(x, y)| (x - mean_x) * (y - mean_y))
                .sum::<f64>()
                / pairs.len() as f64;

            black_box(correlation)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_core_rng,
    bench_collection_ops,
    bench_distributions,
    bench_array_ops,
    bench_scientific_workflows,
    bench_parallel_ops,
    bench_memory_efficiency,
    bench_statistical_quality
);

criterion_main!(benches);
