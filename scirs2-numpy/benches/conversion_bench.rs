//! Criterion benchmarks for array conversion overhead in scirs2-numpy.
//!
//! These benchmarks measure the cost of copying data from strided (non-contiguous)
//! and contiguous sources into a contiguous `Vec`, which is the hot path when
//! Python passes non-contiguous NumPy arrays to Rust functions.
//!
//! Run with:
//! ```bash
//! cargo bench -p scirs2-numpy --bench conversion_bench
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_numpy::simd_copy::{copy_strided_to_contiguous_f32, copy_strided_to_contiguous_f64};

// ── helpers ───────────────────────────────────────────────────────────────────

fn bench_f32_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_copy");
    for size in [1_000_usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(size as u64));

        // Contiguous copy (stride = 1)
        let src_contiguous: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let mut dst = vec![0.0_f32; size];
        group.bench_with_input(
            BenchmarkId::new("stride_1_contiguous", size),
            &size,
            |b, &n| {
                b.iter(|| unsafe {
                    copy_strided_to_contiguous_f32(src_contiguous.as_ptr(), &mut dst, n, 1);
                });
            },
        );

        // Strided copy (stride = 2, gather every other element)
        let src_strided: Vec<f32> = (0..(size * 2)).map(|i| i as f32).collect();
        group.bench_with_input(BenchmarkId::new("stride_2", size), &size, |b, &n| {
            b.iter(|| unsafe {
                copy_strided_to_contiguous_f32(src_strided.as_ptr(), &mut dst, n, 2);
            });
        });

        // Strided copy (stride = 4)
        let src_stride4: Vec<f32> = (0..(size * 4)).map(|i| i as f32).collect();
        group.bench_with_input(BenchmarkId::new("stride_4", size), &size, |b, &n| {
            b.iter(|| unsafe {
                copy_strided_to_contiguous_f32(src_stride4.as_ptr(), &mut dst, n, 4);
            });
        });
    }
    group.finish();
}

fn bench_f64_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_copy");
    for size in [1_000_usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(size as u64));

        // Contiguous copy
        let src_contiguous: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let mut dst = vec![0.0_f64; size];
        group.bench_with_input(
            BenchmarkId::new("stride_1_contiguous", size),
            &size,
            |b, &n| {
                b.iter(|| unsafe {
                    copy_strided_to_contiguous_f64(src_contiguous.as_ptr(), &mut dst, n, 1);
                });
            },
        );

        // Strided copy (stride = 2)
        let src_strided: Vec<f64> = (0..(size * 2)).map(|i| i as f64).collect();
        group.bench_with_input(BenchmarkId::new("stride_2", size), &size, |b, &n| {
            b.iter(|| unsafe {
                copy_strided_to_contiguous_f64(src_strided.as_ptr(), &mut dst, n, 2);
            });
        });

        // Strided copy (stride = 3)
        let src_stride3: Vec<f64> = (0..(size * 3)).map(|i| i as f64).collect();
        group.bench_with_input(BenchmarkId::new("stride_3", size), &size, |b, &n| {
            b.iter(|| unsafe {
                copy_strided_to_contiguous_f64(src_stride3.as_ptr(), &mut dst, n, 3);
            });
        });
    }
    group.finish();
}

/// 1M-element strided f32 copy — documents the overhead of a large gather.
fn bench_1m_strided_f32(c: &mut Criterion) {
    let n = 1_000_000_usize;
    let stride = 3_usize;
    let src: Vec<f32> = (0..(n * stride)).map(|i| i as f32).collect();
    let mut dst = vec![0.0_f32; n];

    let mut group = c.benchmark_group("large_gather");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("f32_1m_stride3", |b| {
        b.iter(|| unsafe {
            copy_strided_to_contiguous_f32(src.as_ptr(), &mut dst, n, stride);
        });
    });

    // 1M-element f64 for comparison
    let src_f64: Vec<f64> = (0..(n * stride)).map(|i| i as f64).collect();
    let mut dst_f64 = vec![0.0_f64; n];
    group.bench_function("f64_1m_stride3", |b| {
        b.iter(|| unsafe {
            copy_strided_to_contiguous_f64(src_f64.as_ptr(), &mut dst_f64, n, stride);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_f32_copy,
    bench_f64_copy,
    bench_1m_strided_f32
);
criterion_main!(benches);
