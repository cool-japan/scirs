use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, ShapeBuilder};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scirs2_linalg::prelude::*;

fn bench_matmul_optimizations(c: &mut Criterion) {
    let sizes = [64, 128, 256, 512, 1024];
    let mut group = c.benchmark_group("matmul_optimizations");
    group.sample_size(10); // Reduce sample size for large matrices

    let config_blocked = OptConfig::default()
        .with_block_size(64)
        .with_parallel_threshold(256);

    let config_adaptive = OptConfig::default()
        .with_block_size(64)
        .with_parallel_threshold(256)
        .with_algorithm(OptAlgorithm::Adaptive);

    for size in &sizes {
        let a = Array2::<f64>::random((*size, *size).f(), Uniform::new(-1.0, 1.0));
        let b = Array2::<f64>::random((*size, *size).f(), Uniform::new(-1.0, 1.0));

        // Standard ndarray matrix multiplication
        group.bench_with_input(BenchmarkId::new("standard", size), size, |bench, _| {
            bench.iter(|| {
                let _result = black_box(a.dot(&b));
            });
        });

        // Our blocked matrix multiplication
        group.bench_with_input(BenchmarkId::new("blocked", size), size, |bench, _| {
            bench.iter(|| {
                let _result =
                    black_box(blocked_matmul(&a.view(), &b.view(), &config_blocked).unwrap());
            });
        });

        // Adaptive algorithm selection
        group.bench_with_input(BenchmarkId::new("adaptive", size), size, |bench, _| {
            bench.iter(|| {
                let _result =
                    black_box(blocked_matmul(&a.view(), &b.view(), &config_adaptive).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_inplace_operations(c: &mut Criterion) {
    let sizes = [128, 256, 512, 1024, 2048];
    let mut group = c.benchmark_group("inplace_operations");

    for size in &sizes {
        let a = Array2::<f64>::random((*size, *size).f(), Uniform::new(-1.0, 1.0));
        let b = Array2::<f64>::random((*size, *size).f(), Uniform::new(-1.0, 1.0));

        // Standard addition (creates new array)
        group.bench_with_input(BenchmarkId::new("standard_add", size), size, |bench, _| {
            bench.iter(|| {
                let _result = black_box(&a + &b);
            });
        });

        // In-place addition
        group.bench_with_input(BenchmarkId::new("inplace_add", size), size, |bench, _| {
            bench.iter(|| {
                let mut a_copy = a.clone();
                inplace_add(&mut a_copy.view_mut(), &b.view()).unwrap();
                black_box(&a_copy);
            });
        });

        // Standard scaling (creates new array)
        group.bench_with_input(
            BenchmarkId::new("standard_scale", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    let _result = black_box(&a * 2.5);
                });
            },
        );

        // In-place scaling
        group.bench_with_input(BenchmarkId::new("inplace_scale", size), size, |bench, _| {
            bench.iter(|| {
                let mut a_copy = a.clone();
                let _ = black_box(inplace_scale(&mut a_copy.view_mut(), 2.5));
            });
        });
    }

    group.finish();
}

fn bench_transpose_optimizations(c: &mut Criterion) {
    let sizes = [128, 256, 512, 1024, 2048];
    let mut group = c.benchmark_group("transpose_optimizations");

    for size in &sizes {
        let a = Array2::<f64>::random((*size, *size).f(), Uniform::new(-1.0, 1.0));

        // Standard transpose
        group.bench_with_input(BenchmarkId::new("standard", size), size, |bench, _| {
            bench.iter(|| {
                let _result = black_box(a.t().to_owned());
            });
        });

        // Optimized transpose
        group.bench_with_input(BenchmarkId::new("optimized", size), size, |bench, _| {
            bench.iter(|| {
                let _result = black_box(optimized_transpose(&a.view()).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_parallel_vs_serial(c: &mut Criterion) {
    let sizes = [256, 512, 1024, 2048];
    let mut group = c.benchmark_group("parallel_vs_serial");
    group.sample_size(10);

    for size in &sizes {
        let a = Array2::<f64>::random((*size, *size).f(), Uniform::new(-1.0, 1.0));
        let b = Array2::<f64>::random((*size, *size).f(), Uniform::new(-1.0, 1.0));

        // Serial execution
        let config_serial = OptConfig::default()
            .with_block_size(64)
            .with_algorithm(OptAlgorithm::Blocked);

        group.bench_with_input(BenchmarkId::new("serial", size), size, |bench, _| {
            bench.iter(|| {
                let _result =
                    black_box(blocked_matmul(&a.view(), &b.view(), &config_serial).unwrap());
            });
        });

        // Parallel execution
        let config_parallel = OptConfig::default()
            .with_block_size(64)
            .with_parallel_threshold(0) // Always use parallel
            .with_algorithm(OptAlgorithm::Blocked);

        group.bench_with_input(BenchmarkId::new("parallel", size), size, |bench, _| {
            bench.iter(|| {
                let _result =
                    black_box(blocked_matmul(&a.view(), &b.view(), &config_parallel).unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_optimizations,
    bench_inplace_operations,
    bench_transpose_optimizations,
    bench_parallel_vs_serial
);

criterion_main!(benches);
