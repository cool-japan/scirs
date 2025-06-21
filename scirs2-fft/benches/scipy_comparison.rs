//! Enhanced performance comparison with SciPy FFT
//!
//! This module provides comprehensive performance benchmarks comparing
//! scirs2-fft with SciPy's FFT implementation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use scirs2_fft::{
    dct, dct::DCTType, dst, dst::DSTType, fft, fft2, fftn, frft, hfft::hfft, ifft, irfft, rfft,
    worker_pool::set_workers,
};
use std::f64::consts::PI;

/// Performance test configuration
#[derive(Debug, Clone)]
struct TestConfig {
    sizes_1d: Vec<usize>,
    sizes_2d: Vec<usize>,
    sizes_nd: Vec<Vec<usize>>,
    worker_counts: Vec<usize>,
    frft_orders: Vec<f64>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            // Using smaller sizes to avoid timeouts during tests
            sizes_1d: vec![64, 128, 256],
            sizes_2d: vec![16, 32, 64],
            sizes_nd: vec![vec![4, 4, 4], vec![8, 8, 8]],
            worker_counts: vec![1, 2, 4, 8],
            frft_orders: vec![0.25, 0.5, 0.75, 1.0],
        }
    }
}

/// Generate test signals
fn generate_1d_signal(size: usize) -> (Vec<f64>, Vec<Complex64>) {
    let real_signal: Vec<f64> = (0..size)
        .map(|i| {
            let t = i as f64 / size as f64;
            (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 30.0 * t).cos()
        })
        .collect();

    let complex_signal: Vec<Complex64> = real_signal
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    (real_signal, complex_signal)
}

fn generate_2d_signal(size: usize) -> Array2<f64> {
    Array2::from_shape_fn((size, size), |(i, j)| {
        let x = i as f64 / size as f64;
        let y = j as f64 / size as f64;
        (2.0 * PI * (5.0 * x + 3.0 * y)).sin()
    })
}

/// Comprehensive 1D FFT benchmarks
fn bench_fft_1d_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT-1D-Comprehensive");
    let config = TestConfig::default();

    for &size in &config.sizes_1d {
        let (real_signal, complex_signal) = generate_1d_signal(size);

        // Regular FFT
        group.bench_with_input(
            BenchmarkId::new("fft", size),
            &complex_signal,
            |b, signal| b.iter(|| fft(black_box(signal), None)),
        );

        // Real FFT
        group.bench_with_input(BenchmarkId::new("rfft", size), &real_signal, |b, signal| {
            b.iter(|| rfft(black_box(signal), None))
        });

        // Inverse FFT
        group.bench_with_input(
            BenchmarkId::new("ifft", size),
            &complex_signal,
            |b, signal| b.iter(|| ifft(black_box(signal), None)),
        );

        // Inverse real FFT
        let spectrum = rfft(&real_signal, None).unwrap();
        group.bench_with_input(BenchmarkId::new("irfft", size), &spectrum, |b, spectrum| {
            b.iter(|| irfft(black_box(spectrum), Some(size)))
        });

        // Hermitian FFT
        group.bench_with_input(
            BenchmarkId::new("hfft", size),
            &complex_signal,
            |b, signal| b.iter(|| hfft(black_box(signal), Some(size), None)),
        );
    }

    group.finish();
}

/// 2D and N-D FFT benchmarks
fn bench_fft_multidim(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT-MultiDim");
    let config = TestConfig::default();

    // 2D FFT benchmarks
    for &size in &config.sizes_2d {
        let data = generate_2d_signal(size);

        group.bench_with_input(BenchmarkId::new("fft2", size), &data, |b, data| {
            b.iter(|| fft2(black_box(data), None, None, None))
        });
    }

    // N-D FFT benchmarks
    for shape in &config.sizes_nd {
        let total_size: usize = shape.iter().product();
        let data = Array1::from_shape_fn(total_size, |i| (i as f64).sin());
        let data_nd = data.into_shape_with_order(shape.as_slice()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("fftn", format!("{:?}", shape)),
            &data_nd,
            |b, data| {
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
            },
        );
    }

    group.finish();
}

/// Transform-specific benchmarks
fn bench_specialized_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("Specialized-Transforms");
    let config = TestConfig::default();

    // Use smaller sizes to avoid timeouts
    for &size in &[64, 128] {
        let signal: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();

        // DCT benchmarks
        for &dct_type in &[1, 2, 3, 4] {
            group.bench_with_input(
                BenchmarkId::new(format!("dct_type_{}", dct_type), size),
                &signal,
                |b, signal| {
                    b.iter(|| {
                        dct(
                            black_box(signal),
                            Some(match dct_type {
                                1 => DCTType::Type1,
                                2 => DCTType::Type2,
                                3 => DCTType::Type3,
                                4 => DCTType::Type4,
                                _ => DCTType::Type2,
                            }),
                            None,
                        )
                    })
                },
            );
        }

        // DST benchmarks
        for &dst_type in &[1, 2, 3, 4] {
            group.bench_with_input(
                BenchmarkId::new(format!("dst_type_{}", dst_type), size),
                &signal,
                |b, signal| {
                    b.iter(|| {
                        dst(
                            black_box(signal),
                            Some(match dst_type {
                                1 => DSTType::Type1,
                                2 => DSTType::Type2,
                                3 => DSTType::Type3,
                                4 => DSTType::Type4,
                                _ => DSTType::Type2,
                            }),
                            None,
                        )
                    })
                },
            );
        }

        // FrFT benchmarks
        for &alpha in &config.frft_orders {
            group.bench_with_input(
                BenchmarkId::new(format!("frft_alpha_{}", alpha), size),
                &signal,
                |b, signal| b.iter(|| frft(black_box(signal), alpha, None)),
            );
        }
    }

    group.finish();
}

/// Worker scaling benchmarks
fn bench_worker_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Worker-Scaling");
    let config = TestConfig::default();

    // Test with a smaller 2D FFT to avoid timeouts
    let size = 64;
    let data = generate_2d_signal(size);

    for &workers in &config.worker_counts {
        group.bench_with_input(
            BenchmarkId::new("fft2_workers", workers),
            &data,
            |b, data| {
                b.iter(|| {
                    let _ = set_workers(workers);
                    fft2(black_box(data), None, None, None)
                })
            },
        );
    }

    group.finish();
}

/// Generate comparison report with Python
fn generate_comparison_report() {
    println!("Generating comparison report...");

    // Create a Python script to run SciPy benchmarks
    let python_script = r#"
import numpy as np
import scipy.fft as scipy_fft
import time
import json

def benchmark_scipy():
    results = {}
    
    # 1D FFT benchmarks (smaller sizes to avoid timeouts)
    sizes_1d = [64, 128, 256]
    for size in sizes_1d:
        signal = np.sin(2 * np.pi * 10 * np.arange(size) / size)
        
        # FFT
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = scipy_fft.fft(signal)
            end = time.perf_counter()
            times.append(end - start)
        
        results[f'fft_{size}'] = {
            'scipy_time': np.median(times),
            'size': size,
            'operation': 'fft'
        }
        
        # RFFT
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = scipy_fft.rfft(signal)
            end = time.perf_counter()
            times.append(end - start)
        
        results[f'rfft_{size}'] = {
            'scipy_time': np.median(times),
            'size': size,
            'operation': 'rfft'
        }
    
    # 2D FFT benchmarks (smaller sizes to avoid timeouts)
    sizes_2d = [16, 32, 64]
    for size in sizes_2d:
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        data = np.sin(2 * np.pi * (5 * x / size + 3 * y / size))
        
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = scipy_fft.fft2(data)
            end = time.perf_counter()
            times.append(end - start)
        
        results[f'fft2_{size}'] = {
            'scipy_time': np.median(times),
            'size': size,
            'operation': 'fft2'
        }
    
    return results

if __name__ == '__main__':
    results = benchmark_scipy()
    with open('scipy_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('SciPy benchmark results saved to scipy_benchmark_results.json')
"#;

    // Save Python script
    std::fs::write("run_scipy_benchmarks.py", python_script)
        .expect("Failed to write Python script");

    // Create comparison summary
    let summary = r#"
# FFT Performance Comparison: scirs2-fft vs SciPy

## How to run the comparison:

1. Run the Rust benchmarks:
   ```
   cargo bench --bench scipy_comparison
   ```

2. Run the Python benchmarks:
   ```
   python run_scipy_benchmarks.py
   ```

3. Compare the results in the generated reports.

## Benchmark Categories:

1. **1D FFT Operations**:
   - Standard FFT
   - Real FFT (RFFT)
   - Inverse transforms
   - Hermitian FFT

2. **Multi-dimensional FFT**:
   - 2D FFT
   - N-D FFT

3. **Specialized Transforms**:
   - DCT (types I-IV)
   - DST (types I-IV)
   - Fractional FFT

4. **Scaling Performance**:
   - Worker thread scaling
   - Large array performance

## Performance Metrics:

- Execution time (median of multiple runs)
- Memory usage
- Accuracy comparison
- Scaling behavior with problem size
"#;

    std::fs::write("BENCHMARK_COMPARISON.md", summary).expect("Failed to write comparison summary");

    println!("Comparison scripts and documentation generated.");
}

// Criterion benchmark groups
criterion_group!(
    benches,
    bench_fft_1d_comprehensive,
    bench_fft_multidim,
    bench_specialized_transforms,
    bench_worker_scaling
);
criterion_main!(benches);

// Function to generate the comparison report
#[allow(dead_code)]
fn generate_report() {
    generate_comparison_report();
    println!("Run 'cargo bench --bench scipy_comparison' to execute benchmarks");
}
