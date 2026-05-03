//! Criterion benchmarks for large-array allocation patterns in scirs2-numpy.
//!
//! Profiles the cost of allocating `Vec<f32>`, `Vec<f64>`, and
//! `ndarray::Array1` at sizes from 1 K to 1 M elements, which is the typical
//! overhead of array round-trips between Python and Rust.
//!
//! Run with:
//! ```bash
//! cargo bench -p scirs2-numpy --bench allocation_bench
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

/// Benchmark allocation cost for raw `Vec` and `ndarray::Array1`.
fn bench_array_allocation(c: &mut Criterion) {
    let sizes = [1_000_usize, 10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("array_allocation");

    for &n in &sizes {
        // Vec<f32> allocation
        group.bench_with_input(BenchmarkId::new("alloc_f32", n), &n, |b, &n| {
            b.iter(|| {
                let v: Vec<f32> = black_box(vec![0.0_f32; n]);
                black_box(v)
            });
        });

        // Vec<f64> allocation
        group.bench_with_input(BenchmarkId::new("alloc_f64", n), &n, |b, &n| {
            b.iter(|| {
                let v: Vec<f64> = black_box(vec![0.0_f64; n]);
                black_box(v)
            });
        });

        // ndarray::Array1<f32> allocation
        group.bench_with_input(BenchmarkId::new("ndarray_alloc_f32", n), &n, |b, &n| {
            b.iter(|| {
                let arr = ndarray::Array1::<f32>::zeros(black_box(n));
                black_box(arr)
            });
        });

        // ndarray::Array1<f64> allocation
        group.bench_with_input(BenchmarkId::new("ndarray_alloc_f64", n), &n, |b, &n| {
            b.iter(|| {
                let arr = ndarray::Array1::<f64>::zeros(black_box(n));
                black_box(arr)
            });
        });

        // Vec<f32> allocation + clone (models a data copy round-trip)
        group.bench_with_input(BenchmarkId::new("alloc_clone_f32", n), &n, |b, &n| {
            let src: Vec<f32> = vec![1.0_f32; n];
            b.iter(|| {
                let v = black_box(src.clone());
                black_box(v)
            });
        });

        // Vec<f64> allocation + clone
        group.bench_with_input(BenchmarkId::new("alloc_clone_f64", n), &n, |b, &n| {
            let src: Vec<f64> = vec![1.0_f64; n];
            b.iter(|| {
                let v = black_box(src.clone());
                black_box(v)
            });
        });
    }

    group.finish();
}

/// Benchmark dynamic-shape (IxDyn) ndarray allocation, matching the pattern
/// used when consuming DLPack tensors with unknown dimensionality at compile time.
fn bench_dyn_array_allocation(c: &mut Criterion) {
    let sizes = [1_000_usize, 100_000, 1_000_000];
    let mut group = c.benchmark_group("dyn_array_allocation");

    for &n in &sizes {
        // 1-D dynamic shape
        group.bench_with_input(BenchmarkId::new("ixdyn_1d_f32", n), &n, |b, &n| {
            b.iter(|| {
                let arr = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[black_box(n)]));
                black_box(arr)
            });
        });

        // 2-D dynamic shape (sqrt(n) × sqrt(n), rounded)
        let side = (n as f64).sqrt() as usize;
        let n2 = side * side;
        group.bench_with_input(BenchmarkId::new("ixdyn_2d_f32", n2), &n2, |b, &n2| {
            let s = (n2 as f64).sqrt() as usize;
            b.iter(|| {
                let arr =
                    ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[black_box(s), black_box(s)]));
                black_box(arr)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_array_allocation, bench_dyn_array_allocation);
criterion_main!(benches);
