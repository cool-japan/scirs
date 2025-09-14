//! Benchmarks for FFT operations
//!
//! This module contains benchmarks comparing various FFT implementations
//! and measuring performance across different input sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use num_complex::Complex64;
use scirs2_fft::{fft, fft2, fftn, frft, ifft, irfft, rfft};
use std::hint::black_box;

/// Benchmark basic 1D FFT operations
#[allow(dead_code)]
fn bench_fft_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT-1D");

    // Use smaller sizes to avoid timeouts during testing
    for size in [64, 128, 256].iter() {
        let size = *size;

        // Generate test signal
        let signal: Vec<f64> = (0..size)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / size as f64).sin())
            .collect();

        // Complex signal
        let complex_signal: Vec<Complex64> =
            signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        // Benchmark regular FFT
        group.bench_with_input(
            BenchmarkId::new("fft", size),
            &complex_signal,
            |b, signal| b.iter(|| fft(black_box(signal), None)),
        );

        // Benchmark real FFT
        group.bench_with_input(BenchmarkId::new("rfft", size), &signal, |b, signal| {
            b.iter(|| rfft(black_box(signal), None))
        });

        // Benchmark inverse FFT
        group.bench_with_input(
            BenchmarkId::new("ifft", size),
            &complex_signal,
            |b, signal| b.iter(|| ifft(black_box(signal), None)),
        );

        // Benchmark inverse real FFT
        let spectrum = rfft(&signal, None).unwrap();
        group.bench_with_input(BenchmarkId::new("irfft", size), &spectrum, |b, spectrum| {
            b.iter(|| irfft(black_box(spectrum), Some(size)))
        });
    }

    group.finish();
}

/// Benchmark 2D FFT operations
#[allow(dead_code)]
fn bench_fft_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT-2D");

    // Use smaller sizes to avoid timeouts
    for size in [16, 32, 64].iter() {
        let size = *size;

        // Generate 2D test data
        let data: Array1<f64> = Array1::from_shape_fn(size * size, |i| {
            let x = (i % size) as f64 / size as f64;
            let y = (i / size) as f64 / size as f64;
            (2.0 * PI * (5.0 * x + 3.0 * y)).sin()
        });

        // Reshape to 2D
        let data_2d = data.intoshape_with_order((size, size)).unwrap();

        // Benchmark 2D FFT
        group.bench_with_input(BenchmarkId::new("fft2", size), &data_2d, |b, data| {
            b.iter(|| fft2(black_box(data), None, None, None))
        });

        // Benchmark N-dimensional FFT
        group.bench_with_input(BenchmarkId::new("fftn", size), &data_2d, |b, data| {
            b.iter(|| {
                fftn(
                    black_box(&data.clone().into_dyn()),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            })
        });
    }

    group.finish();
}

/// Benchmark Fractional Fourier Transform
#[allow(dead_code)]
fn bench_frft(c: &mut Criterion) {
    let mut group = c.benchmark_group("FrFT");

    // Use smaller sizes to avoid timeouts
    for size in [64, 128].iter() {
        let size = *size;

        // Generate test signal
        let signal: Vec<f64> = (0..size)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / size as f64).sin())
            .collect();

        // Different fractional orders
        for &alpha in [0.25, 0.5, 0.75, 1.0].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("alpha_{alpha}"), size),
                &signal,
                |b, signal| b.iter(|| frft(black_box(signal), alpha, None)),
            );
        }
    }

    group.finish();
}

/// Benchmark memory-efficient FFT operations
#[allow(dead_code)]
fn bench_memory_efficient(c: &mut Criterion) {
    // Memory efficient FFT mode is used inline below

    let mut group = c.benchmark_group("Memory-Efficient");

    // 1D in-place FFT
    // Use smaller sizes to avoid timeouts
    for size in [64, 128].iter() {
        let size = *size;

        // Generate complex signal
        let mut signal: Vec<Complex64> = (0..size)
            .map(|i| {
                let x = (2.0 * PI * 10.0 * i as f64 / size as f64).sin();
                Complex64::new(x, 0.0)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("fft_inplace", size),
            &mut signal,
            |b, signal| {
                b.iter(|| {
                    let mut data = signal.clone();
                    use scirs2_fft::memory_efficient::{fft_inplace, FftMode};
                    let mut workspace = vec![Complex64::new(0.0, 0.0); data.len()];
                    fft_inplace(
                        black_box(&mut data),
                        &mut workspace,
                        FftMode::Forward,
                        false,
                    )
                })
            },
        );
    }

    // 2D memory-efficient FFT (using smaller sizes to avoid timeouts)
    for size in [16, 32].iter() {
        let size = *size;

        // Generate 2D test data
        let data: Array1<f64> = Array1::from_shape_fn(size * size, |i| {
            let x = (i % size) as f64 / size as f64;
            let y = (i / size) as f64 / size as f64;
            (2.0 * PI * (5.0 * x + 3.0 * y)).sin()
        });

        let data_2d = data.intoshape_with_order((size, size)).unwrap();

        group.bench_with_input(
            BenchmarkId::new("fft2_efficient", size),
            &data_2d,
            |b, data| {
                b.iter(|| {
                    use scirs2_fft::memory_efficient::{fft2_efficient, FftMode};
                    fft2_efficient(black_box(&data.view()), None, FftMode::Forward, false)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fft_1d,
    bench_fft_2d,
    bench_frft,
    bench_memory_efficient
);
criterion_main!(benches);
