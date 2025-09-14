use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use rand::Rng;
use scirs2_fft::{
    sparse_fft,
    sparse_fft::{SparseFFTAlgorithm, WindowFunction},
    sparse_fft_gpu::{gpu_batch_sparse_fft, gpu_sparse_fft, GPUBackend},
};
use std::f64::consts::PI;
use std::hint::black_box;

// Helper function to create a sparse signal
#[allow(dead_code)]
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

// Helper function to create a noisy sparse signal
#[allow(dead_code)]
fn create_noisy_sparse_signal(
    n: usize,
    frequencies: &[(usize, f64)],
    noise_level: f64,
) -> Vec<f64> {
    let mut signal = create_sparse_signal(n, frequencies);

    // Add noise
    let mut rng = rand::rng();

    for i in 0..n {
        signal[i] += noise_level * rng.gen_range(-1.0..1.0);
    }

    signal
}

// Benchmark CPU sparse FFT with different signal sizes
#[allow(dead_code)]
fn bench_cpu_sparse_fft_small(b: &mut Bencher) {
    let n = 1024;
    let frequencies = vec![(10..1.0), (50, 0.5), (100, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = sparse_fft::sparse_fft(
            black_box(&signal),
            sparsity,
            Some(SparseFFTAlgorithm::Sublinear),
            None, // seed parameter
        )
        .unwrap();
    });
}

#[allow(dead_code)]
fn bench_cpu_sparse_fft_medium(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = sparse_fft::sparse_fft(
            black_box(&signal),
            sparsity,
            Some(SparseFFTAlgorithm::Sublinear),
            None, // seed parameter
        )
        .unwrap();
    });
}

#[allow(dead_code)]
fn bench_cpu_sparse_fft_large(b: &mut Bencher) {
    let n = 65536;
    let frequencies = vec![(1000, 1.0), (5000, 0.5), (10000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = sparse_fft::sparse_fft(
            black_box(&signal),
            sparsity,
            Some(SparseFFTAlgorithm::Sublinear),
            None, // seed parameter
        )
        .unwrap();
    });
}

// Benchmark GPU sparse FFT with different algorithms
#[allow(dead_code)]
fn bench_gpu_sparse_fft_sublinear(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = gpu_sparse_fft(
            black_box(&signal),
            sparsity,
            GPUBackend::CUDA,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

#[allow(dead_code)]
fn bench_gpu_sparse_fft_iterative(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = gpu_sparse_fft(
            black_box(&signal),
            sparsity,
            GPUBackend::CUDA,
            Some(SparseFFTAlgorithm::Iterative),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

#[allow(dead_code)]
fn bench_gpu_sparse_fft_compressed_sensing(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = gpu_sparse_fft(
            black_box(&signal),
            sparsity,
            GPUBackend::CUDA,
            Some(SparseFFTAlgorithm::CompressedSensing),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

// Benchmark noisy signals
#[allow(dead_code)]
fn bench_gpu_sparse_fft_noisy(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_noisy_sparse_signal(n, &frequencies, 0.1);
    let sparsity = 10;

    b.iter(|| {
        let _ = gpu_sparse_fft(
            black_box(&signal),
            sparsity,
            GPUBackend::CUDA,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

// Benchmark batch operations
#[allow(dead_code)]
fn bench_gpu_batch_sparse_fft(b: &mut Bencher) {
    let n = 4096;
    let batch_size = 16;
    let frequencies = vec![(50, 1.0), (250, 0.5), (500, 0.3)];

    // Create multiple signals
    let signals: Vec<Vec<f64>> = (0..batch_size)
        .map(|_| create_sparse_signal(n, &frequencies))
        .collect();

    let sparsity = 10;

    b.iter(|| {
        let _ = gpu_batch_sparse_fft(
            black_box(&signals),
            sparsity,
            GPUBackend::CUDA,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

// Benchmark different window functions
#[allow(dead_code)]
fn bench_gpu_sparse_fft_hann_window(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = gpu_sparse_fft(
            black_box(&signal),
            sparsity,
            GPUBackend::CUDA,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

#[allow(dead_code)]
fn bench_gpu_sparse_fft_hamming_window(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = gpu_sparse_fft(
            black_box(&signal),
            sparsity,
            GPUBackend::CUDA,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hamming),
        )
        .unwrap();
    });
}

#[allow(dead_code)]
fn gpu_sparse_fft_benches(c: &mut Criterion) {
    c.bench_function("cpu_sparse_fft_small", bench_cpu_sparse_fft_small);
    c.bench_function("cpu_sparse_fft_medium", bench_cpu_sparse_fft_medium);
    c.bench_function("cpu_sparse_fft_large", bench_cpu_sparse_fft_large);
    c.bench_function("gpu_sparse_fft_sublinear", bench_gpu_sparse_fft_sublinear);
    c.bench_function("gpu_sparse_fft_iterative", bench_gpu_sparse_fft_iterative);
    c.bench_function(
        "gpu_sparse_fft_compressed_sensing",
        bench_gpu_sparse_fft_compressed_sensing,
    );
    c.bench_function("gpu_sparse_fft_noisy", bench_gpu_sparse_fft_noisy);
    c.bench_function("gpu_batch_sparse_fft", bench_gpu_batch_sparse_fft);
    c.bench_function(
        "gpu_sparse_fft_hann_window",
        bench_gpu_sparse_fft_hann_window,
    );
    c.bench_function(
        "gpu_sparse_fft_hamming_window",
        bench_gpu_sparse_fft_hamming_window,
    );
}

criterion_group!(benches, gpu_sparse_fft_benches);
criterion_main!(benches);
