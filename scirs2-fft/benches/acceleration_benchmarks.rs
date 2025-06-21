//! Comprehensive Acceleration Benchmarks
//!
//! This benchmark suite provides formal performance measurement for all acceleration
//! features including multi-GPU processing and specialized hardware support.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_fft::{
    // GPU backends
    gpu_sparse_fft,
    is_cuda_available,
    is_hip_available,
    is_sycl_available,

    // Multi-GPU processing
    multi_gpu_sparse_fft,

    // CPU reference
    sparse_fft::sparse_fft,
    // Sparse FFT configuration
    sparse_fft::{SparseFFTAlgorithm, SparseFFTConfig, SparsityEstimationMethod},

    // Specialized hardware
    specialized_hardware_sparse_fft,

    GPUBackend,
};
use std::f64::consts::PI;

/// Create a test signal with specified sparse frequency components
fn create_sparse_signal(n: usize, sparsity: usize) -> Vec<f64> {
    let mut signal = vec![0.0; n];

    // Add sparse frequency components
    for k in 0..sparsity {
        let freq = (k + 1) * (n / (sparsity * 4)); // Spread frequencies
        let amplitude = 1.0 / (k as f64 + 1.0); // Decreasing amplitudes

        for i in 0..n {
            let t = 2.0 * PI * (i as f64) / (n as f64);
            signal[i] += amplitude * (freq as f64 * t).sin();
        }
    }

    // Add minimal noise for realism
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(42);

    for sample in &mut signal {
        *sample += 0.01 * (rng.random::<f64>() - 0.5);
    }

    signal
}

/// Benchmark CPU sparse FFT implementation
fn bench_cpu_sparse_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_sparse_fft");

    for signal_size in [512, 1024, 2048, 4096, 8192] {
        let sparsity = signal_size / 64;
        let signal = create_sparse_signal(signal_size, sparsity);

        group.bench_with_input(
            BenchmarkId::new("sublinear", signal_size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    sparse_fft(
                        black_box(signal),
                        black_box(sparsity),
                        black_box(Some(SparseFFTAlgorithm::Sublinear)),
                        black_box(None),
                    )
                    .unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compressed_sensing", signal_size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    sparse_fft(
                        black_box(signal),
                        black_box(sparsity),
                        black_box(Some(SparseFFTAlgorithm::CompressedSensing)),
                        black_box(None),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark GPU sparse FFT implementations
fn bench_gpu_sparse_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_sparse_fft");

    let backends = [
        ("cuda", GPUBackend::CUDA, is_cuda_available()),
        ("hip", GPUBackend::HIP, is_hip_available()),
        ("sycl", GPUBackend::SYCL, is_sycl_available()),
    ];

    for signal_size in [1024, 2048, 4096, 8192] {
        let sparsity = signal_size / 64;
        let signal = create_sparse_signal(signal_size, sparsity);

        for (backend_name, backend, available) in &backends {
            if *available {
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_sublinear", backend_name), signal_size),
                    &signal,
                    |b, signal| {
                        b.iter(|| {
                            gpu_sparse_fft(
                                black_box(signal),
                                black_box(sparsity),
                                black_box(*backend),
                                black_box(Some(SparseFFTAlgorithm::Sublinear)),
                                black_box(None),
                            )
                            .unwrap()
                        })
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new(format!("{}_compressed_sensing", backend_name), signal_size),
                    &signal,
                    |b, signal| {
                        b.iter(|| {
                            gpu_sparse_fft(
                                black_box(signal),
                                black_box(sparsity),
                                black_box(*backend),
                                black_box(Some(SparseFFTAlgorithm::CompressedSensing)),
                                black_box(None),
                            )
                            .unwrap()
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark multi-GPU sparse FFT processing
fn bench_multi_gpu_sparse_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_gpu_sparse_fft");

    // Only benchmark larger signals where multi-GPU makes sense
    for signal_size in [4096, 8192, 16384, 32768] {
        let sparsity = signal_size / 64;
        let signal = create_sparse_signal(signal_size, sparsity);

        group.bench_with_input(
            BenchmarkId::new("adaptive_distribution", signal_size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    multi_gpu_sparse_fft(
                        black_box(signal),
                        black_box(sparsity),
                        black_box(Some(SparseFFTAlgorithm::Sublinear)),
                        black_box(None),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark specialized hardware acceleration
fn bench_specialized_hardware(c: &mut Criterion) {
    let mut group = c.benchmark_group("specialized_hardware");

    for signal_size in [1024, 2048, 4096, 8192, 16384] {
        let sparsity = signal_size / 64;
        let signal = create_sparse_signal(signal_size, sparsity);

        let config = SparseFFTConfig {
            sparsity,
            algorithm: SparseFFTAlgorithm::Sublinear,
            estimation_method: SparsityEstimationMethod::Manual,
            ..SparseFFTConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::new("fpga_asic", signal_size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    specialized_hardware_sparse_fft(black_box(signal), black_box(config.clone()))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark scaling behavior across different sparsity levels
fn bench_sparsity_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity_scaling");

    let signal_size = 4096;

    for sparsity in [4, 8, 16, 32, 64, 128] {
        let signal = create_sparse_signal(signal_size, sparsity);

        // CPU reference
        group.bench_with_input(BenchmarkId::new("cpu", sparsity), &signal, |b, signal| {
            b.iter(|| {
                sparse_fft(
                    black_box(signal),
                    black_box(sparsity),
                    black_box(Some(SparseFFTAlgorithm::Sublinear)),
                    black_box(None),
                )
                .unwrap()
            })
        });

        // Multi-GPU
        group.bench_with_input(
            BenchmarkId::new("multi_gpu", sparsity),
            &signal,
            |b, signal| {
                b.iter(|| {
                    multi_gpu_sparse_fft(
                        black_box(signal),
                        black_box(sparsity),
                        black_box(Some(SparseFFTAlgorithm::Sublinear)),
                        black_box(None),
                    )
                    .unwrap()
                })
            },
        );

        // Specialized hardware
        let config = SparseFFTConfig {
            sparsity,
            algorithm: SparseFFTAlgorithm::Sublinear,
            estimation_method: SparsityEstimationMethod::Manual,
            ..SparseFFTConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::new("specialized_hardware", sparsity),
            &signal,
            |b, signal| {
                b.iter(|| {
                    specialized_hardware_sparse_fft(black_box(signal), black_box(config.clone()))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark algorithm comparison across acceleration methods
fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_comparison");

    let signal_size = 2048;
    let sparsity = 32;
    let signal = create_sparse_signal(signal_size, sparsity);

    let algorithms = [
        SparseFFTAlgorithm::Sublinear,
        SparseFFTAlgorithm::CompressedSensing,
        SparseFFTAlgorithm::Iterative,
        SparseFFTAlgorithm::FrequencyPruning,
    ];

    for algorithm in &algorithms {
        // CPU implementation
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{:?}", algorithm)),
            &signal,
            |b, signal| {
                b.iter(|| {
                    sparse_fft(
                        black_box(signal),
                        black_box(sparsity),
                        black_box(Some(*algorithm)),
                        black_box(None),
                    )
                    .unwrap()
                })
            },
        );

        // Multi-GPU implementation
        group.bench_with_input(
            BenchmarkId::new("multi_gpu", format!("{:?}", algorithm)),
            &signal,
            |b, signal| {
                b.iter(|| {
                    multi_gpu_sparse_fft(
                        black_box(signal),
                        black_box(sparsity),
                        black_box(Some(*algorithm)),
                        black_box(None),
                    )
                    .unwrap()
                })
            },
        );

        // Specialized hardware implementation
        let config = SparseFFTConfig {
            sparsity,
            algorithm: *algorithm,
            estimation_method: SparsityEstimationMethod::Manual,
            ..SparseFFTConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::new("specialized_hardware", format!("{:?}", algorithm)),
            &signal,
            |b, signal| {
                b.iter(|| {
                    specialized_hardware_sparse_fft(black_box(signal), black_box(config.clone()))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency across different acceleration methods
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    // Test with very large signals to stress memory systems
    for signal_size in [8192, 16384, 32768] {
        let sparsity = signal_size / 128; // Lower sparsity for large signals
        let signal = create_sparse_signal(signal_size, sparsity);

        // CPU reference (most memory efficient)
        group.bench_with_input(
            BenchmarkId::new("cpu_memory_efficient", signal_size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    sparse_fft(
                        black_box(signal),
                        black_box(sparsity),
                        black_box(Some(SparseFFTAlgorithm::Sublinear)),
                        black_box(None),
                    )
                    .unwrap()
                })
            },
        );

        // Multi-GPU (distributed memory)
        group.bench_with_input(
            BenchmarkId::new("multi_gpu_distributed", signal_size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    multi_gpu_sparse_fft(
                        black_box(signal),
                        black_box(sparsity),
                        black_box(Some(SparseFFTAlgorithm::Sublinear)),
                        black_box(None),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    acceleration_benches,
    bench_cpu_sparse_fft,
    bench_gpu_sparse_fft,
    bench_multi_gpu_sparse_fft,
    bench_specialized_hardware,
    bench_sparsity_scaling,
    bench_algorithm_comparison,
    bench_memory_efficiency,
);

criterion_main!(acceleration_benches);
