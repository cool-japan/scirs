//! Benchmarks for the FFT planning system
//!
//! This module provides benchmarks to compare different planning strategies,
//! measuring their performance across various scenarios.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use scirs2_fft::{
    AdvancedFftPlanner, FftPlanExecutor, PlanBuilder, PlanningConfig, PlanningStrategy,
};
use std::time::Duration;
use tempfile::tempdir;

/// Generate a test signal of the given size
fn generate_test_signal(size: usize) -> Vec<Complex64> {
    let mut signal = Vec::with_capacity(size);
    for i in 0..size {
        let phase = 2.0 * std::f64::consts::PI * (i as f64) / (size as f64);
        signal.push(Complex64::new(phase.cos(), phase.sin()));
    }
    signal
}

/// Benchmark the cost of creating a new plan each time
fn bench_always_new(c: &mut Criterion) {
    let mut group = c.benchmark_group("AlwaysNew");
    group.measurement_time(Duration::from_secs(10));
    let sizes = [1024, 2048, 4096, 8192];

    for size in &sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let signal = generate_test_signal(size);
            let mut output = vec![Complex64::default(); size];

            b.iter(|| {
                // Create a new plan every time
                let plan = PlanBuilder::new()
                    .shape(&[size])
                    .forward(true)
                    .strategy(PlanningStrategy::AlwaysNew)
                    .build()
                    .unwrap();

                let executor = FftPlanExecutor::new(plan);
                executor
                    .execute(black_box(&signal), black_box(&mut output))
                    .unwrap();
            });
        });
    }
    group.finish();
}

/// Benchmark the performance of the cache-first approach
fn bench_cache_first(c: &mut Criterion) {
    let mut group = c.benchmark_group("CacheFirst");
    group.measurement_time(Duration::from_secs(10));
    let sizes = [1024, 2048, 4096, 8192];

    for size in &sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let signal = generate_test_signal(size);
            let mut output = vec![Complex64::default(); size];

            // Setup: create a planner with cache
            let mut planner = AdvancedFftPlanner::with_config(PlanningConfig {
                strategy: PlanningStrategy::CacheFirst,
                ..Default::default()
            });

            b.iter_with_setup(
                || {
                    // No additional setup for each iteration
                },
                |_| {
                    // Get plan from cache if available
                    let plan = planner.plan_fft(&[size], true, Default::default()).unwrap();
                    let executor = FftPlanExecutor::new(plan);
                    executor
                        .execute(black_box(&signal), black_box(&mut output))
                        .unwrap();
                },
            );
        });
    }
    group.finish();
}

/// Benchmark the performance of serialized plans
fn bench_serialized(c: &mut Criterion) {
    let mut group = c.benchmark_group("SerializedFirst");
    group.measurement_time(Duration::from_secs(10));
    let sizes = [1024, 2048, 4096, 8192];

    // Create a temporary directory for serialized plans
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("benchmark_plans.json");

    for size in &sizes {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let signal = generate_test_signal(size);
            let mut output = vec![Complex64::default(); size];

            // Setup: create a planner with serialization
            let mut config = PlanningConfig::default();
            config.strategy = PlanningStrategy::SerializedFirst;
            config.serialized_db_path = Some(db_path.to_str().unwrap().to_string());

            let mut planner = AdvancedFftPlanner::with_config(config);

            // Pre-create plans to simulate previous runs
            let _ = planner
                .plan_fft(&[*size], true, Default::default())
                .unwrap();
            planner.save_plans().unwrap();

            b.iter_with_setup(
                || {
                    // No additional setup for each iteration
                },
                |_| {
                    // Get plan from serialized cache
                    let plan = planner.plan_fft(&[size], true, Default::default()).unwrap();
                    let executor = FftPlanExecutor::new(plan);
                    executor
                        .execute(black_box(&signal), black_box(&mut output))
                        .unwrap();
                },
            );
        });
    }
    group.finish();
}

/// Compare repeated executions with different planning strategies
fn bench_repeated_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("RepeatedExecution");
    group.measurement_time(Duration::from_secs(15));

    // Fixed size for this benchmark
    let size = 2048;
    let signal = generate_test_signal(size);
    let repetitions = 10; // Number of FFTs to perform in each iteration

    // Create a temporary directory for serialized plans
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("benchmark_repeated.json");

    // Benchmark with AlwaysNew
    group.bench_function("AlwaysNew", |b| {
        b.iter(|| {
            let mut output = vec![Complex64::default(); size];
            for _ in 0..repetitions {
                let plan = PlanBuilder::new()
                    .shape(&[size])
                    .forward(true)
                    .strategy(PlanningStrategy::AlwaysNew)
                    .build()
                    .unwrap();

                let executor = FftPlanExecutor::new(plan);
                executor
                    .execute(black_box(&signal), black_box(&mut output))
                    .unwrap();
            }
        });
    });

    // Benchmark with CacheFirst
    group.bench_function("CacheFirst", |b| {
        let mut planner = AdvancedFftPlanner::with_config(PlanningConfig {
            strategy: PlanningStrategy::CacheFirst,
            ..Default::default()
        });

        b.iter(|| {
            let mut output = vec![Complex64::default(); size];
            for _ in 0..repetitions {
                let plan = planner.plan_fft(&[size], true, Default::default()).unwrap();
                let executor = FftPlanExecutor::new(plan);
                executor
                    .execute(black_box(&signal), black_box(&mut output))
                    .unwrap();
            }
        });
    });

    // Benchmark with SerializedFirst
    group.bench_function("SerializedFirst", |b| {
        // Setup: create a planner with serialization
        let mut config = PlanningConfig::default();
        config.strategy = PlanningStrategy::SerializedFirst;
        config.serialized_db_path = Some(db_path.to_str().unwrap().to_string());

        let mut planner = AdvancedFftPlanner::with_config(config);

        // Pre-create plans to simulate previous runs
        let _ = planner.plan_fft(&[size], true, Default::default()).unwrap();
        planner.save_plans().unwrap();

        b.iter(|| {
            let mut output = vec![Complex64::default(); size];
            for _ in 0..repetitions {
                let plan = planner.plan_fft(&[size], true, Default::default()).unwrap();
                let executor = FftPlanExecutor::new(plan);
                executor
                    .execute(black_box(&signal), black_box(&mut output))
                    .unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark different cache sizes
fn bench_cache_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("CacheSizes");
    group.measurement_time(Duration::from_secs(15));

    // We'll use different sized FFTs in a sequence
    let sizes = [1024, 2048, 4096, 8192, 16384];
    let signals: Vec<Vec<Complex64>> = sizes.iter().map(|&s| generate_test_signal(s)).collect();
    let repetitions = 5; // Number of iterations through all sizes

    // Test with different cache sizes
    for cache_size in [1, 3, 5, 10] {
        group.bench_with_input(
            BenchmarkId::from_parameter(cache_size),
            &cache_size,
            |b, &cache_size| {
                let mut planner = AdvancedFftPlanner::with_config(PlanningConfig {
                    strategy: PlanningStrategy::CacheFirst,
                    max_cached_plans: cache_size,
                    ..Default::default()
                });

                b.iter(|| {
                    for _ in 0..repetitions {
                        for (size, signal) in sizes.iter().zip(signals.iter()) {
                            let mut output = vec![Complex64::default(); *size];
                            let plan = planner
                                .plan_fft(&[*size], true, Default::default())
                                .unwrap();
                            let executor = FftPlanExecutor::new(plan);
                            executor
                                .execute(black_box(signal), black_box(&mut output))
                                .unwrap();
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = planning_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_always_new, bench_cache_first, bench_serialized, bench_repeated_execution, bench_cache_sizes
}

criterion_main!(planning_benches);
