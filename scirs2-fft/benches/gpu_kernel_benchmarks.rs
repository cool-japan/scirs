use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use scirs2_fft::{
    sparse_fft::{sparse_fft, SparseFFTAlgorithm, WindowFunction},
    sparse_fft_gpu::{gpu_sparse_fft, GPUBackend},
    sparse_fft_gpu_cuda::cuda_sparse_fft,
};
use std::f64::consts::PI;

// Helper function to create a sparse signal
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

// Helper to add noise to a signal
fn add_noise(signal: &[f64], noise_level: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    signal
        .iter()
        .map(|&x| x + noise_level * rng.random_range(-1.0..1.0))
        .collect()
}

// Basic kernel benchmarks
fn bench_gpu_kernel_sublinear(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }

    let n = 1024;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);
    let noisy_signal = add_noise(&signal, 0.1);

    c.bench_function("gpu_kernel_sublinear", |b| {
        b.iter(|| {
            cuda_sparse_fft(
                &noisy_signal,
                5, // Look for 5 frequency components
                0, // Use first CUDA device
                Some(SparseFFTAlgorithm::Sublinear),
                Some(WindowFunction::Hann),
            )
            .unwrap()
        })
    });
}

fn bench_gpu_kernel_compressed_sensing(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }

    let n = 1024;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);
    let noisy_signal = add_noise(&signal, 0.1);

    c.bench_function("gpu_kernel_compressed_sensing", |b| {
        b.iter(|| {
            cuda_sparse_fft(
                &noisy_signal,
                5, // Look for 5 frequency components
                0, // Use first CUDA device
                Some(SparseFFTAlgorithm::CompressedSensing),
                Some(WindowFunction::Hann),
            )
            .unwrap()
        })
    });
}

fn bench_gpu_kernel_iterative(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }

    let n = 1024;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);
    let noisy_signal = add_noise(&signal, 0.1);

    c.bench_function("gpu_kernel_iterative", |b| {
        b.iter(|| {
            cuda_sparse_fft(
                &noisy_signal,
                5, // Look for 5 frequency components
                0, // Use first CUDA device
                Some(SparseFFTAlgorithm::Iterative),
                Some(WindowFunction::Hann),
            )
            .unwrap()
        })
    });
}

// Window function comparison
fn bench_window_functions(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }

    let n = 4096;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);
    let noisy_signal = add_noise(&signal, 0.1);

    let windows = [
        WindowFunction::None,
        WindowFunction::Hann,
        WindowFunction::Hamming,
        WindowFunction::Blackman,
        WindowFunction::FlatTop,
        WindowFunction::Kaiser,
    ];

    for window in windows {
        c.bench_with_input(
            BenchmarkId::new("gpu_kernel_window", format!("{:?}", window)),
            &window,
            |b, window| {
                b.iter(|| {
                    cuda_sparse_fft(
                        &noisy_signal,
                        5, // Look for 5 frequency components
                        0, // Use first CUDA device
                        Some(SparseFFTAlgorithm::Sublinear),
                        Some(*window),
                    )
                    .unwrap()
                })
            },
        );
    }
}

// Signal size comparison
fn bench_signal_sizes(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }

    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let sizes = [1024, 4096, 16384, 65536];

    for size in sizes {
        let signal = create_sparse_signal(size, &frequencies);
        let noisy_signal = add_noise(&signal, 0.1);

        c.bench_with_input(BenchmarkId::new("gpu_kernel_size", size), &size, |b, _| {
            b.iter(|| {
                cuda_sparse_fft(
                    &noisy_signal,
                    5, // Look for 5 frequency components
                    0, // Use first CUDA device
                    Some(SparseFFTAlgorithm::Sublinear),
                    Some(WindowFunction::Hann),
                )
                .unwrap()
            })
        });
    }
}

// GPU vs CPU comparison
fn bench_gpu_vs_cpu(c: &mut Criterion) {
    let n = 16384;
    let frequencies = vec![(30, 1.0), (70, 0.5), (150, 0.25)];
    let signal = create_sparse_signal(n, &frequencies);
    let noisy_signal = add_noise(&signal, 0.1);

    // CPU benchmark
    c.bench_function("cpu_sparse_fft_kernel", |b| {
        b.iter(|| {
            sparse_fft(
                &noisy_signal,
                5, // Look for 5 frequency components
                Some(SparseFFTAlgorithm::Sublinear),
                None, // seed parameter
            )
            .unwrap()
        })
    });

    // GPU benchmark (only if CUDA is available)
    if cfg!(feature = "cuda") {
        c.bench_function("gpu_sparse_fft_kernel", |b| {
            b.iter(|| {
                gpu_sparse_fft(
                    &noisy_signal,
                    5, // Look for 5 frequency components
                    GPUBackend::CUDA,
                    Some(SparseFFTAlgorithm::Sublinear),
                    Some(WindowFunction::Hann),
                )
                .unwrap()
            })
        });
    }
}

// Algorithm comparison on GPU
fn bench_gpu_algorithms(c: &mut Criterion) {
    if !cfg!(feature = "cuda") {
        return;
    }

    let n = 8192;
    let frequencies = vec![
        (30, 1.0),   // Very large component
        (70, 0.5),   // Large component
        (150, 0.25), // Medium component
        (350, 0.1),  // Small component
        (700, 0.05), // Very small component
    ];
    let signal = create_sparse_signal(n, &frequencies);
    let noisy_signal = add_noise(&signal, 0.01);

    let algorithms = [
        SparseFFTAlgorithm::Sublinear,
        SparseFFTAlgorithm::CompressedSensing,
        SparseFFTAlgorithm::Iterative,
        SparseFFTAlgorithm::FrequencyPruning,
        SparseFFTAlgorithm::SpectralFlatness,
    ];

    for algorithm in algorithms {
        c.bench_with_input(
            BenchmarkId::new("gpu_algorithm", format!("{:?}", algorithm)),
            &algorithm,
            |b, algorithm| {
                b.iter(|| {
                    cuda_sparse_fft(
                        &noisy_signal,
                        5, // Look for 5 frequency components
                        0, // Use first CUDA device
                        Some(*algorithm),
                        Some(WindowFunction::Hann),
                    )
                    .unwrap()
                })
            },
        );
    }
}

criterion_group!(
    kernel_benches,
    bench_gpu_kernel_sublinear,
    bench_gpu_kernel_compressed_sensing,
    bench_gpu_kernel_iterative,
    bench_window_functions,
    bench_signal_sizes,
    bench_gpu_vs_cpu,
    bench_gpu_algorithms,
);

criterion_main!(kernel_benches);
