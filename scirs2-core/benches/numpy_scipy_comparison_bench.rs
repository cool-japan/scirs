//! NumPy/SciPy Performance Comparison Benchmarks
//!
//! This benchmark suite compares scirs2-core performance against equivalent
//! NumPy/SciPy operations to validate Beta 1 performance targets.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::ndarray_ext::{Array1, Array2, Axis, Ix1, Ix2};
use scirs2_core::random::{arrays::random_uniform_array, seeded_rng};
// use statrs::statistics::Statistics; // Not available in dev-dependencies
use std::hint::black_box;
use std::time::Duration;

const SIZES: &[usize] = &[100, 1000, 10000, 100000];
const MATRIX_SIZES: &[usize] = &[10, 50, 100, 500, 1000];

/// Benchmark array creation and initialization
#[allow(dead_code)]
fn bench_array_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation");
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // Zeros initialization
        group.bench_with_input(BenchmarkId::new("zeros", size), &size, |b, &s| {
            b.iter(|| {
                let arr = Array1::<f64>::zeros(s);
                black_box(arr)
            })
        });

        // Ones initialization
        group.bench_with_input(BenchmarkId::new("ones", size), &size, |b, &s| {
            b.iter(|| {
                let arr = Array1::<f64>::ones(s);
                black_box(arr)
            })
        });

        // Random initialization
        group.bench_with_input(BenchmarkId::new("random", size), &size, |b, &s| {
            b.iter(|| {
                let arr = {
                    let mut rng = seeded_rng(42);
                    random_uniform_array(Ix1(s), &mut rng)
                };
                black_box(arr)
            })
        });

        // Linspace equivalent
        group.bench_with_input(BenchmarkId::new("linspace", size), &size, |b, &s| {
            b.iter(|| {
                let arr = Array1::linspace(0.0, 100.0, s);
                black_box(arr)
            })
        });
    }
    group.finish();
}

/// Benchmark element-wise operations (ufuncs)
#[allow(dead_code)]
fn bench_element_wise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_wise_ops");
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES {
        let arr1 = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix1(size), &mut rng)
        };
        let arr2 = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix1(size), &mut rng)
        };

        group.throughput(Throughput::Elements(size as u64));

        // Addition
        group.bench_with_input(
            BenchmarkId::new("add", size),
            &(&arr1, &arr2),
            |b, (a1, a2)| {
                b.iter(|| {
                    let result = &**a1 + &**a2;
                    black_box(result)
                })
            },
        );

        // Multiplication
        group.bench_with_input(
            BenchmarkId::new("multiply", size),
            &(&arr1, &arr2),
            |b, (a1, a2)| {
                b.iter(|| {
                    let result = &**a1 * &**a2;
                    black_box(result)
                })
            },
        );

        // Square root
        group.bench_with_input(BenchmarkId::new("sqrt", size), &arr1, |b, a| {
            b.iter(|| {
                let result = a.mapv(|x| x.sqrt());
                black_box(result)
            })
        });

        // Exponential
        group.bench_with_input(BenchmarkId::new("exp", size), &arr1, |b, a| {
            b.iter(|| {
                let result = a.mapv(|x| x.exp());
                black_box(result)
            })
        });

        // Trigonometric (sin)
        group.bench_with_input(BenchmarkId::new("sin", size), &arr1, |b, a| {
            b.iter(|| {
                let result = a.mapv(|x| x.sin());
                black_box(result)
            })
        });
    }
    group.finish();
}

/// Benchmark reduction operations
#[allow(dead_code)]
fn bench_reduction_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduction_ops");
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES {
        let arr = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix1(size), &mut rng)
        };

        group.throughput(Throughput::Elements(size as u64));

        // Sum reduction
        group.bench_with_input(BenchmarkId::new("sum", size), &arr, |b, a| {
            b.iter(|| {
                let result = a.sum();
                black_box(result)
            })
        });

        // Mean calculation
        group.bench_with_input(BenchmarkId::new("mean", size), &arr, |b, a| {
            b.iter(|| {
                let result = a.mean().unwrap();
                black_box(result)
            })
        });

        // Standard deviation
        group.bench_with_input(BenchmarkId::new("std", size), &arr, |b, a| {
            b.iter(|| {
                let mean = a.mean().unwrap();
                let variance = a.mapv(|x| (x - mean).powi(2)).mean().unwrap();
                let std = variance.sqrt();
                black_box(std)
            })
        });

        // Min/max
        group.bench_with_input(BenchmarkId::new("min_max", size), &arr, |b, a| {
            b.iter(|| {
                let min = a.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                black_box((min, max))
            })
        });
    }
    group.finish();
}

/// Benchmark matrix operations
#[allow(dead_code)]
fn benchmatrix_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_ops");
    group.measurement_time(Duration::from_secs(20));

    for &size in MATRIX_SIZES {
        let mat_a = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix2(size, size), &mut rng)
        };
        let mat_b = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix2(size, size), &mut rng)
        };
        let vec = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix1(size), &mut rng)
        };

        group.throughput(Throughput::Elements((size * size) as u64));

        // Matrix multiplication
        group.bench_with_input(
            BenchmarkId::new("matmul", size),
            &(&mat_a, &mat_b),
            |b, (a, b_mat)| {
                b.iter(|| {
                    let result = a.dot(&**b_mat);
                    black_box(result)
                })
            },
        );

        // Matrix-vector multiplication
        group.bench_with_input(
            BenchmarkId::new("matvec", size),
            &(&mat_a, &vec),
            |b, (a, v)| {
                b.iter(|| {
                    let result = a.dot(&**v);
                    black_box(result)
                })
            },
        );

        // Transpose
        group.bench_with_input(BenchmarkId::new("transpose", size), &mat_a, |b, a| {
            b.iter(|| {
                let result = a.t().to_owned();
                black_box(result)
            })
        });

        // Diagonal extraction
        group.bench_with_input(BenchmarkId::new("diagonal", size), &mat_a, |b, a| {
            b.iter(|| {
                let result = a.diag().to_owned();
                black_box(result)
            })
        });
    }
    group.finish();
}

/// Benchmark array manipulation operations
#[allow(dead_code)]
fn bench_array_manipulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_manipulation");
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES {
        let arr = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix1(size), &mut rng)
        };

        group.throughput(Throughput::Elements(size as u64));

        // Reshape (1D to 2D)
        if size >= 100 {
            let rows = 10;
            let cols = size / rows;
            if rows * cols == size {
                group.bench_with_input(BenchmarkId::new("reshape", size), &arr, |b, a| {
                    b.iter(|| {
                        let result = a.clone().into_shape_with_order((rows, cols)).unwrap();
                        black_box(result)
                    })
                });
            }
        }

        // Concatenation
        let arr2 = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix1(size), &mut rng)
        };
        group.bench_with_input(
            BenchmarkId::new("concatenate", size),
            &(&arr, &arr2),
            |b, (a1, a2)| {
                b.iter(|| {
                    let views = vec![a1.view(), a2.view()];
                    let result = ndarray::concatenate(Axis(0), &views).unwrap();
                    black_box(result)
                })
            },
        );

        // Slicing
        group.bench_with_input(BenchmarkId::new("slice", size), &arr, |b, a| {
            b.iter(|| {
                let mid = size / 2;
                let result = a.slice(scirs2_core::ndarray::s![..mid]).to_owned();
                black_box(result)
            })
        });

        // Sorting (via conversion to vec)
        group.bench_with_input(BenchmarkId::new("sort", size), &arr, |b, a| {
            b.iter(|| {
                let mut vec = a.to_vec();
                vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let result = Array1::from(vec);
                black_box(result)
            })
        });
    }
    group.finish();
}

/// Benchmark statistical operations
#[allow(dead_code)]
fn bench_statistical_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_ops");
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES {
        let arr1 = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix1(size), &mut rng)
        };
        let arr2 = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix1(size), &mut rng)
        };

        group.throughput(Throughput::Elements(size as u64));

        // Variance
        group.bench_with_input(BenchmarkId::new("variance", size), &arr1, |b, a| {
            b.iter(|| {
                let mean = a.mean().unwrap();
                let variance = a.mapv(|x| (x - mean).powi(2)).mean().unwrap();
                black_box(variance)
            })
        });

        // Covariance (simplified)
        group.bench_with_input(
            BenchmarkId::new("covariance", size),
            &(&arr1, &arr2),
            |b, (a1, a2)| {
                b.iter(|| {
                    let mean1 = a1.mean().unwrap();
                    let mean2 = a2.mean().unwrap();
                    let cov = a1
                        .iter()
                        .zip(a2.iter())
                        .map(|(x1, x2)| (x1 - mean1) * (x2 - mean2))
                        .sum::<f64>()
                        / (size as f64);
                    black_box(cov)
                })
            },
        );

        // Percentile (median as 50th percentile)
        group.bench_with_input(BenchmarkId::new("median", size), &arr1, |b, a| {
            b.iter(|| {
                let mut sorted = a.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if size % 2 == 0 {
                    (sorted[size / 2 - 1] + sorted[size / 2]) / 2.0
                } else {
                    sorted[size / 2]
                };
                black_box(median)
            })
        });
    }
    group.finish();
}

/// Benchmark memory-intensive operations
#[allow(dead_code)]
fn bench_memory_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_ops");
    group.measurement_time(Duration::from_secs(15));

    for &size in &[1000, 10000, 50000] {
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f64>()) as u64,
        ));

        // Array copy
        let arr = {
            let mut rng = seeded_rng(42);
            random_uniform_array(Ix1(size), &mut rng)
        };
        group.bench_with_input(BenchmarkId::new("copy", size), &arr, |b, a| {
            b.iter(|| {
                let result = a.to_owned();
                black_box(result)
            })
        });

        // View creation
        group.bench_with_input(BenchmarkId::new("view", size), &arr, |b, a| {
            b.iter(|| {
                let result = a.view();
                black_box(result)
            })
        });

        // Memory allocation pattern
        group.bench_with_input(BenchmarkId::new("alloc_pattern", size), &size, |b, &s| {
            b.iter(|| {
                let mut arrays = Vec::with_capacity(10);
                for _ in 0..10 {
                    arrays.push(Array1::<f64>::zeros(s / 10));
                }
                black_box(arrays)
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_array_creation,
    bench_element_wise_ops,
    bench_reduction_ops,
    benchmatrix_ops,
    bench_array_manipulation,
    bench_statistical_ops,
    bench_memory_ops
);
criterion_main!(benches);
