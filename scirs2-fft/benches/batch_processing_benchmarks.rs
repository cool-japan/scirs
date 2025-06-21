use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use scirs2_fft::sparse_fft::{sparse_fft, SparseFFTAlgorithm, WindowFunction};
use scirs2_fft::sparse_fft_batch::{batch_sparse_fft, BatchConfig};
use scirs2_fft::sparse_fft_gpu::{gpu_batch_sparse_fft, gpu_sparse_fft, GPUBackend};
use std::f64::consts::PI;

// Helper function to create a sparse signal with specified frequencies
fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
    let mut signal = vec![0.0; n];
    for i in 0..n {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        for &(freq, amp) in frequencies {
            signal[i] += amp * (freq as f64 * t).sin();
        }
    }
    signal
}

// Helper to add noise to signals
fn add_noise(signal: &[f64], noise_level: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    signal
        .iter()
        .map(|&x| x + rng.random_range(-noise_level..noise_level))
        .collect()
}

// Helper to create a batch of similar signals with different noise
fn create_signal_batch(
    count: usize,
    n: usize,
    frequencies: &[(usize, f64)],
    noise_level: f64,
) -> Vec<Vec<f64>> {
    let base_signal = create_sparse_signal(n, frequencies);
    (0..count)
        .map(|_| add_noise(&base_signal, noise_level))
        .collect()
}

fn sequential_processing_small(c: &mut Criterion) {
    let n = 1024;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signals = create_signal_batch(5, n, &frequencies, 0.1);

    c.bench_function("sequential_processing_small", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for signal in &signals {
                let result = sparse_fft(
                    signal,
                    10, // Look for up to 10 components
                    Some(SparseFFTAlgorithm::Sublinear),
                    None, // seed parameter
                )
                .unwrap();
                results.push(result);
            }
            results
        })
    });
}

fn sequential_processing_medium(c: &mut Criterion) {
    let n = 4096;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signals = create_signal_batch(3, n, &frequencies, 0.1);

    c.bench_function("sequential_processing_medium", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for signal in &signals {
                let result = sparse_fft(
                    signal,
                    10, // Look for up to 10 components
                    Some(SparseFFTAlgorithm::Sublinear),
                    None, // seed parameter
                )
                .unwrap();
                results.push(result);
            }
            results
        })
    });
}

fn sequential_processing_large(c: &mut Criterion) {
    let n = 16384;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signals = create_signal_batch(1, n, &frequencies, 0.1);

    c.bench_function("sequential_processing_large", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for signal in &signals {
                let result = sparse_fft(
                    signal,
                    10, // Look for up to 10 components
                    Some(SparseFFTAlgorithm::Sublinear),
                    None, // seed parameter
                )
                .unwrap();
                results.push(result);
            }
            results
        })
    });
}

fn batch_processing_cpu_small(c: &mut Criterion) {
    let n = 1024;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signals = create_signal_batch(10, n, &frequencies, 0.1);

    let config = BatchConfig {
        max_batch_size: 4,
        use_parallel: true,
        max_memory_per_batch: 0,
        use_mixed_precision: false,
        use_inplace: false,
        preserve_input: true,
    };

    c.bench_function("batch_processing_cpu_small", |b| {
        b.iter(|| {
            batch_sparse_fft(
                &signals,
                10, // Look for up to 10 components each
                Some(SparseFFTAlgorithm::Sublinear),
                Some(WindowFunction::Hann),
                Some(config.clone()),
            )
            .unwrap()
        })
    });
}

fn batch_processing_cpu_medium(c: &mut Criterion) {
    let n = 4096;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signals = create_signal_batch(5, n, &frequencies, 0.1);

    let config = BatchConfig {
        max_batch_size: 4,
        use_parallel: true,
        max_memory_per_batch: 0,
        use_mixed_precision: false,
        use_inplace: false,
        preserve_input: true,
    };

    c.bench_function("batch_processing_cpu_medium", |b| {
        b.iter(|| {
            batch_sparse_fft(
                &signals,
                10, // Look for up to 10 components each
                Some(SparseFFTAlgorithm::Sublinear),
                Some(WindowFunction::Hann),
                Some(config.clone()),
            )
            .unwrap()
        })
    });
}

fn batch_processing_gpu_small(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        // Skip GPU benchmarks if CUDA is not available
        return;
    }

    let n = 1024;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signals = create_signal_batch(10, n, &frequencies, 0.1);

    c.bench_function("batch_processing_gpu_small", |b| {
        b.iter(|| {
            gpu_batch_sparse_fft(
                &signals,
                10, // Look for up to 10 components each
                GPUBackend::CUDA,
                Some(SparseFFTAlgorithm::Sublinear),
                Some(WindowFunction::Hann),
            )
            .unwrap()
        })
    });
}

fn batch_processing_gpu_medium(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        // Skip GPU benchmarks if CUDA is not available
        return;
    }

    let n = 4096;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signals = create_signal_batch(5, n, &frequencies, 0.1);

    c.bench_function("batch_processing_gpu_medium", |b| {
        b.iter(|| {
            gpu_batch_sparse_fft(
                &signals,
                10, // Look for up to 10 components each
                GPUBackend::CUDA,
                Some(SparseFFTAlgorithm::Sublinear),
                Some(WindowFunction::Hann),
            )
            .unwrap()
        })
    });
}

fn gpu_vs_cpu_single_signal(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        // Skip GPU benchmarks if CUDA is not available
        return;
    }

    let n = 16384;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);
    let noisy_signal = add_noise(&signal, 0.1);

    c.bench_function("gpu_sparse_fft", |b| {
        b.iter(|| {
            gpu_sparse_fft(
                &noisy_signal,
                10, // Look for up to 10 components
                GPUBackend::CUDA,
                Some(SparseFFTAlgorithm::Sublinear),
                Some(WindowFunction::Hann),
            )
            .unwrap()
        })
    });

    c.bench_function("cpu_sparse_fft", |b| {
        b.iter(|| {
            sparse_fft(
                &noisy_signal,
                10, // Look for up to 10 components
                Some(SparseFFTAlgorithm::Sublinear),
                None, // seed parameter
            )
            .unwrap()
        })
    });
}

fn batch_size_comparison(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        // Skip GPU benchmarks if CUDA is not available
        return;
    }

    let n = 4096;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let batch_sizes = vec![1, 2, 4, 8, 16];

    for batch_size in batch_sizes {
        let signals = create_signal_batch(batch_size, n, &frequencies, 0.1);

        c.bench_with_input(
            BenchmarkId::new("gpu_batch_sparse_fft", batch_size),
            &signals,
            |b, signals| {
                b.iter(|| {
                    gpu_batch_sparse_fft(
                        signals,
                        10, // Look for up to 10 components each
                        GPUBackend::CUDA,
                        Some(SparseFFTAlgorithm::Sublinear),
                        Some(WindowFunction::Hann),
                    )
                    .unwrap()
                })
            },
        );
    }
}

fn memory_optimization_comparison(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        // Skip GPU benchmarks if CUDA is not available
        return;
    }

    // Create a large signal to stress memory usage
    let n = 65536;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);
    let noisy_signal = add_noise(&signal, 0.1);

    // Create multiple small signals that collectively equal the large signal
    let small_n = 4096;
    let small_signals = create_signal_batch(16, small_n, &frequencies, 0.1);

    c.bench_function("single_large_gpu_sparse_fft", |b| {
        b.iter(|| {
            gpu_sparse_fft(
                &noisy_signal,
                10, // Look for up to 10 components
                GPUBackend::CUDA,
                Some(SparseFFTAlgorithm::Sublinear),
                Some(WindowFunction::Hann),
            )
            .unwrap()
        })
    });

    c.bench_function("multiple_small_gpu_batch_sparse_fft", |b| {
        b.iter(|| {
            gpu_batch_sparse_fft(
                &small_signals,
                10, // Look for up to 10 components each
                GPUBackend::CUDA,
                Some(SparseFFTAlgorithm::Sublinear),
                Some(WindowFunction::Hann),
            )
            .unwrap()
        })
    });
}

criterion_group!(
    basic_benches,
    sequential_processing_small,
    sequential_processing_medium,
    sequential_processing_large,
    batch_processing_cpu_small,
    batch_processing_cpu_medium,
);

criterion_group!(
    gpu_benches,
    batch_processing_gpu_small,
    batch_processing_gpu_medium,
    gpu_vs_cpu_single_signal,
    batch_size_comparison,
    memory_optimization_comparison,
);

criterion_main!(basic_benches, gpu_benches);
