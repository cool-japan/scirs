//! Benchmarks for Bessel functions
//!
//! This module benchmarks performance of the Bessel function implementations.
//! It compares:
//! 1. Different input ranges (small, medium, large)
//! 2. Different function types (first kind, modified, etc.)
//! 3. Integer vs. non-integer orders

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_special::bessel::{i0, j0, j1, jn, jv, spherical_jn};
use std::hint::black_box;

#[allow(dead_code)]
fn bench_j0(c: &mut Criterion) {
    let mut group = c.benchmark_group("j0");

    // Small inputs
    group.bench_function("small", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.01;
                black_box(j0(black_box(x)));
            }
        })
    });

    // Medium inputs
    group.bench_function("medium", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 10.0;
                black_box(j0(black_box(x)));
            }
        })
    });

    // Large inputs
    group.bench_function("large", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 10.0 + 100.0;
                black_box(j0(black_box(x)));
            }
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_j1(c: &mut Criterion) {
    let mut group = c.benchmark_group("j1");

    // Small inputs
    group.bench_function("small", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.01;
                black_box(j1(black_box(x)));
            }
        })
    });

    // Medium inputs
    group.bench_function("medium", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 10.0;
                black_box(j1(black_box(x)));
            }
        })
    });

    // Large inputs
    group.bench_function("large", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 10.0 + 100.0;
                black_box(j1(black_box(x)));
            }
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_jn(c: &mut Criterion) {
    let mut group = c.benchmark_group("jn");

    // Different orders with medium inputs
    for n in [2, 5, 10, 20] {
        group.bench_with_input(BenchmarkId::new("order", n), &n, |b, &n| {
            b.iter(|| {
                for i in 0..50 {
                    let x = i as f64 * 0.2 + 5.0;
                    black_box(jn(black_box(n), black_box(x)));
                }
            })
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_jv(c: &mut Criterion) {
    let mut group = c.benchmark_group("jv");

    // Integer vs. half-integer orders
    let orders = [0.0, 1.0, 2.0, 0.5, 1.5, 2.5];

    for v in orders {
        group.bench_with_input(BenchmarkId::new("order", v), &v, |b, &v| {
            b.iter(|| {
                for i in 0..50 {
                    let x = i as f64 * 0.2 + 5.0;
                    black_box(jv(black_box(v), black_box(x)));
                }
            })
        });
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_i0(c: &mut Criterion) {
    let mut group = c.benchmark_group("i0");

    // Small inputs
    group.bench_function("small", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.01;
                black_box(i0(black_box(x)));
            }
        })
    });

    // Medium inputs
    group.bench_function("medium", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 1.0;
                black_box(i0(black_box(x)));
            }
        })
    });

    group.finish();
}

#[allow(dead_code)]
fn bench_spherical(c: &mut Criterion) {
    let mut group = c.benchmark_group("spherical");

    // Compare spherical_jn vs jn for equivalent computations
    group.bench_function("spherical_j0", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 1.0;
                black_box(spherical_jn(black_box(0), black_box(x)));
            }
        })
    });

    group.bench_function("regular_j0_equiv", |b| {
        b.iter(|| {
            for i in 0..100 {
                let x = i as f64 * 0.1 + 1.0;
                // j_{n+1/2}(x) = sqrt(pi/(2x)) * spherical_jn(n, x)
                black_box(j0(black_box(x)));
            }
        })
    });

    group.finish();
}

criterion_group!(
    bessel_benches,
    bench_j0,
    bench_j1,
    bench_jn,
    bench_jv,
    bench_i0,
    bench_spherical
);
criterion_main!(bessel_benches);
