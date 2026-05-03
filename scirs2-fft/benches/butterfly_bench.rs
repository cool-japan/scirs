//! Criterion benchmarks for scalar vs AVX-512 butterfly kernels.
//!
//! On x86_64 hosts with AVX-512F: measures both scalar and AVX-512 paths.
//! On all other hosts (or x86_64 without AVX-512F): only the scalar path
//! is measured; the AVX-512 bench is skipped at runtime.
//!
//! Run with:
//! ```sh
//! cargo bench -p scirs2-fft --bench butterfly_bench
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;
use std::hint::black_box;

#[cfg(target_arch = "x86_64")]
use scirs2_fft::simd_fft::avx512;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_radix4_input() -> [Complex64; 4] {
    [
        Complex64::new(1.23, -0.45),
        Complex64::new(-0.78, 2.34),
        Complex64::new(0.56, -1.23),
        Complex64::new(-0.12, 0.89),
    ]
}

fn make_radix4_twiddles() -> [Complex64; 3] {
    [
        Complex64::new(0.0, -1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ]
}

fn make_radix8_input() -> [Complex64; 8] {
    std::array::from_fn(|k| {
        let t = k as f64 * 0.5;
        Complex64::new(t.cos() + 0.5, t.sin() - 0.3)
    })
}

fn make_radix8_twiddles() -> [Complex64; 7] {
    std::array::from_fn(|k| {
        let angle = -2.0 * PI * (k + 1) as f64 / 8.0;
        Complex64::new(angle.cos(), angle.sin())
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  Benchmarks
// ─────────────────────────────────────────────────────────────────────────────

fn bench_radix4(c: &mut Criterion) {
    let twiddles = make_radix4_twiddles();
    let mut group = c.benchmark_group("butterfly/radix4");
    group.throughput(Throughput::Elements(4));

    // Scalar reference
    group.bench_with_input(BenchmarkId::new("scalar", "4-point"), &twiddles, |b, tw| {
        b.iter(|| {
            let mut a = black_box(make_radix4_input());
            #[cfg(target_arch = "x86_64")]
            avx512::radix4_butterfly_scalar(black_box(&mut a), black_box(tw));
            #[cfg(not(target_arch = "x86_64"))]
            scirs2_fft::butterfly4(black_box(&mut a), black_box(tw));
            a
        })
    });

    // AVX-512 path (x86_64 only, runtime-guarded)
    #[cfg(target_arch = "x86_64")]
    {
        if avx512::is_avx512_available() {
            group.bench_with_input(BenchmarkId::new("avx512", "4-point"), &twiddles, |b, tw| {
                b.iter(|| {
                    let mut a = black_box(make_radix4_input());
                    // Safety: is_avx512_available() guard above.
                    unsafe {
                        avx512::radix4_butterfly_avx512(
                            black_box(a.as_mut_ptr()),
                            black_box(tw.as_ptr()),
                        );
                    }
                    a
                })
            });

            group.bench_with_input(
                BenchmarkId::new("avx512_2x", "4-point×2"),
                &twiddles,
                |b, tw| {
                    b.iter(|| {
                        let mut a0 = black_box(make_radix4_input());
                        let mut a1 = black_box(make_radix4_input());
                        // Safety: is_avx512_available() guard above.
                        unsafe {
                            avx512::radix4_butterfly_x2_avx512(
                                a0.as_mut_ptr(),
                                tw.as_ptr(),
                                a1.as_mut_ptr(),
                                tw.as_ptr(),
                            );
                        }
                        (a0, a1)
                    })
                },
            );
        }
    }

    group.finish();
}

fn bench_radix8(c: &mut Criterion) {
    let twiddles = make_radix8_twiddles();
    let mut group = c.benchmark_group("butterfly/radix8");
    group.throughput(Throughput::Elements(8));

    // Scalar reference
    group.bench_with_input(BenchmarkId::new("scalar", "8-point"), &twiddles, |b, tw| {
        b.iter(|| {
            let mut a = black_box(make_radix8_input());
            #[cfg(target_arch = "x86_64")]
            avx512::radix8_butterfly_scalar(black_box(&mut a), black_box(tw));
            #[cfg(not(target_arch = "x86_64"))]
            scirs2_fft::butterfly8(black_box(&mut a), black_box(tw));
            a
        })
    });

    // AVX-512 path (x86_64 only, runtime-guarded)
    #[cfg(target_arch = "x86_64")]
    {
        if avx512::is_avx512_available() {
            group.bench_with_input(BenchmarkId::new("avx512", "8-point"), &twiddles, |b, tw| {
                b.iter(|| {
                    let mut a = black_box(make_radix8_input());
                    // Safety: is_avx512_available() guard above.
                    unsafe {
                        avx512::radix8_butterfly_avx512(
                            black_box(a.as_mut_ptr()),
                            black_box(tw.as_ptr()),
                        );
                    }
                    a
                })
            });
        }
    }

    group.finish();
}

/// Benchmark the dispatch wrappers to measure the overhead of the runtime guard.
fn bench_dispatch(c: &mut Criterion) {
    let tw4 = make_radix4_twiddles();
    let tw8 = make_radix8_twiddles();
    let mut group = c.benchmark_group("butterfly/dispatch");

    group.bench_function("radix4_dispatch", |b| {
        b.iter(|| {
            let mut a = black_box(make_radix4_input());
            #[cfg(target_arch = "x86_64")]
            avx512::radix4_butterfly_dispatch(black_box(&mut a), black_box(&tw4));
            #[cfg(not(target_arch = "x86_64"))]
            scirs2_fft::butterfly4(black_box(&mut a), black_box(&tw4));
            a
        })
    });

    group.bench_function("radix8_dispatch", |b| {
        b.iter(|| {
            let mut a = black_box(make_radix8_input());
            #[cfg(target_arch = "x86_64")]
            avx512::radix8_butterfly_dispatch(black_box(&mut a), black_box(&tw8));
            #[cfg(not(target_arch = "x86_64"))]
            scirs2_fft::butterfly8(black_box(&mut a), black_box(&tw8));
            a
        })
    });

    group.finish();
}

criterion_group!(benches, bench_radix4, bench_radix8, bench_dispatch);
criterion_main!(benches);
