#![feature(test)]
extern crate test;

use num_complex::Complex64;
use scirs2_fft::rfft;
use scirs2_fft::simd_rfft::{irfft_adaptive, irfft_simd, rfft_adaptive, rfft_simd};
use std::f64::consts::PI;
use test::Bencher;

// Regular Real FFT benchmark - standard implementation
#[bench]
fn bench_rfft_standard_1024(b: &mut Bencher) {
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = rfft(&signal, None).unwrap();
    });
}

// SIMD-accelerated Real FFT benchmark
#[bench]
fn bench_rfft_simd_1024(b: &mut Bencher) {
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = rfft_simd(&signal, None, None).unwrap();
    });
}

// Adaptive Real FFT benchmark (should choose SIMD when available)
#[bench]
fn bench_rfft_adaptive_1024(b: &mut Bencher) {
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = rfft_adaptive(&signal, None, None).unwrap();
    });
}

// Larger Real FFT benchmark - standard implementation
#[bench]
fn bench_rfft_standard_8192(b: &mut Bencher) {
    let n = 8192;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 1024.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = rfft(&signal, None).unwrap();
    });
}

// Larger SIMD-accelerated Real FFT benchmark
#[bench]
fn bench_rfft_simd_8192(b: &mut Bencher) {
    let n = 8192;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 1024.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = rfft_simd(&signal, None, None).unwrap();
    });
}

// Full roundtrip benchmark (RFFT + IRFFT) - standard implementation
#[bench]
fn bench_rfft_irfft_roundtrip_standard_1024(b: &mut Bencher) {
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
        .collect();

    b.iter(|| {
        let spectrum = rfft(&signal, None).unwrap();
        let _recovered = scirs2_fft::irfft(&spectrum, Some(n)).unwrap();
    });
}

// Full roundtrip benchmark (RFFT + IRFFT) - SIMD implementation
#[bench]
fn bench_rfft_irfft_roundtrip_simd_1024(b: &mut Bencher) {
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
        .collect();

    b.iter(|| {
        let spectrum = rfft_simd(&signal, None, None).unwrap();
        let _recovered = irfft_simd(&spectrum, Some(n), None).unwrap();
    });
}

// ARM-specific RFFT benchmark
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_rfft_arm_neon_2048(b: &mut Bencher) {
    let n = 2048;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 256.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = rfft_adaptive(&signal, None, None).unwrap();
    });
}

// ARM-specific IRFFT benchmark
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_irfft_arm_neon_2048(b: &mut Bencher) {
    let n = 2048;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 256.0).sin())
        .collect();
    let spectrum = rfft_adaptive(&signal, None, None).unwrap();

    b.iter(|| {
        let _recovered = irfft_adaptive(&spectrum, Some(n), None).unwrap();
    });
}

// Multi-frequency signal benchmark for RFFT
#[bench]
fn bench_rfft_multi_freq_signal_simd(b: &mut Bencher) {
    let n = 4096;
    // Signal with multiple frequency components
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * PI * 10.0 * t).sin()
                + 0.5 * (2.0 * PI * 50.0 * t).sin()
                + 0.25 * (2.0 * PI * 100.0 * t).sin()
        })
        .collect();

    b.iter(|| {
        let _spectrum = rfft_simd(&signal, None, None).unwrap();
    });
}
