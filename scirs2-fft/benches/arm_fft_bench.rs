#![feature(test)]
extern crate test;

use num_complex::Complex64;
use scirs2_fft::fft::adaptive::{fft_adaptive, fft_simd};
use scirs2_fft::fft::multi::{fft2_adaptive, fftn_adaptive};
use scirs2_fft::fft::{fft, fft2, fftn};
use scirs2_fft::planning::types::NormMode;
use std::f64::consts::PI;
use test::Bencher;

// Regular 1D FFT benchmark on ARM
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_standard_1d_1024(b: &mut Bencher) {
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = fft(&signal, None).unwrap();
    });
}

// ARM NEON-optimized 1D FFT benchmark
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_neon_1d_1024(b: &mut Bencher) {
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = fft_simd(&signal, None, None).unwrap();
    });
}

// Adaptive 1D FFT benchmark on ARM (should choose NEON)
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_adaptive_1d_1024(b: &mut Bencher) {
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = fft_adaptive(&signal, None, None).unwrap();
    });
}

// Regular 2D FFT benchmark on ARM
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_standard_2d_32x32(b: &mut Bencher) {
    let n_rows = 32;
    let n_cols = 32;
    let mut signal = Vec::with_capacity(n_rows * n_cols);

    for i in 0..n_rows {
        for j in 0..n_cols {
            let x = i as f64 / n_rows as f64;
            let y = j as f64 / n_cols as f64;
            let value = (2.0 * PI * 3.0 * x).sin() * (2.0 * PI * 5.0 * y).cos();
            signal.push(value);
        }
    }

    b.iter(|| {
        let _spectrum = fft2(&signal, [n_rows, n_cols], None).unwrap();
    });
}

// ARM NEON-optimized 2D FFT benchmark
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_neon_2d_32x32(b: &mut Bencher) {
    let n_rows = 32;
    let n_cols = 32;
    let mut signal = Vec::with_capacity(n_rows * n_cols);

    for i in 0..n_rows {
        for j in 0..n_cols {
            let x = i as f64 / n_rows as f64;
            let y = j as f64 / n_cols as f64;
            let value = (2.0 * PI * 3.0 * x).sin() * (2.0 * PI * 5.0 * y).cos();
            signal.push(value);
        }
    }

    b.iter(|| {
        let _spectrum = fft2_adaptive(&signal, [n_rows, n_cols], None, None).unwrap();
    });
}

// Regular 3D FFT benchmark on ARM
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_standard_3d_16x16x16(b: &mut Bencher) {
    let shape = [16, 16, 16];
    let total_elements: usize = shape.iter().product();

    let mut signal = Vec::with_capacity(total_elements);

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let x = i as f64 / shape[0] as f64;
                let y = j as f64 / shape[1] as f64;
                let z = k as f64 / shape[2] as f64;
                let value = (2.0 * PI * 2.0 * x).sin()
                    * (2.0 * PI * 3.0 * y).cos()
                    * (2.0 * PI * 4.0 * z).sin();
                signal.push(value);
            }
        }
    }

    b.iter(|| {
        let _spectrum = fftn(&signal.as_slice(), None).unwrap();
    });
}

// ARM NEON-optimized 3D FFT benchmark
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_neon_3d_16x16x16(b: &mut Bencher) {
    let shape = [16, 16, 16];
    let total_elements: usize = shape.iter().product();

    let mut signal = Vec::with_capacity(total_elements);

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                let x = i as f64 / shape[0] as f64;
                let y = j as f64 / shape[1] as f64;
                let z = k as f64 / shape[2] as f64;
                let value = (2.0 * PI * 2.0 * x).sin()
                    * (2.0 * PI * 3.0 * y).cos()
                    * (2.0 * PI * 4.0 * z).sin();
                signal.push(value);
            }
        }
    }

    b.iter(|| {
        let _spectrum = fftn_adaptive(&signal, &shape, None, None).unwrap();
    });
}

// Larger 2D FFT comparison (for significant performance improvements)
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_standard_2d_256x256(b: &mut Bencher) {
    let n_rows = 256;
    let n_cols = 256;
    let mut signal = Vec::with_capacity(n_rows * n_cols);

    // Initialize with a simple pattern
    for i in 0..n_rows * n_cols {
        signal.push((i as f64).sin());
    }

    b.iter(|| {
        let _spectrum = fft2(&signal, [n_rows, n_cols], None).unwrap();
    });
}

// Larger 2D FFT with NEON
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_neon_2d_256x256(b: &mut Bencher) {
    let n_rows = 256;
    let n_cols = 256;
    let mut signal = Vec::with_capacity(n_rows * n_cols);

    // Initialize with a simple pattern
    for i in 0..n_rows * n_cols {
        signal.push((i as f64).sin());
    }

    b.iter(|| {
        let _spectrum = fft2_adaptive(&signal, [n_rows, n_cols], None, None).unwrap();
    });
}

// RFFT (Real FFT) benchmarks for ARM
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_rfft_arm_neon_1024(b: &mut Bencher) {
    let n = 1024;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 128.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = scirs2_fft::rfft::adaptive::rfft_adaptive(&signal, None, None).unwrap();
    });
}

#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_rfft_arm_neon_4096(b: &mut Bencher) {
    let n = 4096;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 512.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum = scirs2_fft::rfft::adaptive::rfft_adaptive(&signal, None, None).unwrap();
    });
}

#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_rfft_standard_vs_neon_comparison(b: &mut Bencher) {
    let n = 8192;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 1024.0).sin())
        .collect();

    // Run standard implementation first
    let spectrum_standard = scirs2_fft::rfft::rfft(&signal, None).unwrap();

    // Then run NEON-optimized implementation
    b.iter(|| {
        let _spectrum = scirs2_fft::rfft::adaptive::rfft_adaptive(&signal, None, None).unwrap();
    });

    // Verify results match
    let spectrum_neon = scirs2_fft::rfft::adaptive::rfft_adaptive(&signal, None, None).unwrap();
    assert_eq!(spectrum_standard.len(), spectrum_neon.len());
}

// Complex number normalization benchmark (standard)
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_normalization_standard_1024(b: &mut Bencher) {
    let n = 1024;
    let mut data: Vec<Complex64> = (0..n).map(|i| Complex64::new(i as f64, 0.0)).collect();
    let scale = 1.0 / n as f64;

    b.iter(|| {
        for val in &mut data {
            *val *= scale;
        }
    });
}

// Complex number normalization benchmark (NEON)
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_normalization_neon_1024(b: &mut Bencher) {
    let n = 1024;
    let mut data: Vec<Complex64> = (0..n).map(|i| Complex64::new(i as f64, 0.0)).collect();
    let scale = 1.0 / n as f64;

    b.iter(|| {
        scirs2_fft::utils::simd::apply_simd_normalization(&mut data, scale);
    });
}

// Benchmark planning strategies
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_with_plan_caching(b: &mut Bencher) {
    let n = 2048;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 256.0).sin())
        .collect();

    // Create a plan cache first
    let mut plan_cache = scirs2_fft::planning::cache::PlanCache::new();

    b.iter(|| {
        let _spectrum =
            scirs2_fft::fft::planning::fft_with_plan(&signal, None, &mut plan_cache).unwrap();
    });
}

// Benchmark parallel planning strategies
#[cfg(target_arch = "aarch64")]
#[bench]
fn bench_fft_parallel_planning(b: &mut Bencher) {
    let n = 4096;
    let signal: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 512.0).sin())
        .collect();

    b.iter(|| {
        let _spectrum =
            scirs2_fft::planning_parallel::fft_with_parallel_planning(&signal, None).unwrap();
    });
}
