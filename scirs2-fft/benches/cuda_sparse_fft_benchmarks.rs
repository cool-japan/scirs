use criterion::{black_box, criterion_group, criterion_main, Bencher, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use scirs2_fft::{
    sparse_fft::sparse_fft,
    sparse_fft::{SparseFFTAlgorithm, WindowFunction},
    sparse_fft_gpu::GPUBackend,
    sparse_fft_gpu_cuda::cuda_sparse_fft,
    sparse_fft_gpu_memory::{init_global_memory_manager, AllocationStrategy},
};
use std::f64::consts::PI;

// Helper function to create a sparse signal
fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
    let mut signal = vec![0.0; n];

    for (i, sample) in signal.iter_mut().enumerate() {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        for &(freq, amp) in frequencies {
            *sample += amp * (freq as f64 * t).sin();
        }
    }

    signal
}

// Helper function to create a signal with noise
fn create_noisy_sparse_signal(
    n: usize,
    frequencies: &[(usize, f64)],
    noise_level: f64,
) -> Vec<f64> {
    // Use a fixed seed for reproducible benchmarks
    let mut rng = StdRng::seed_from_u64(12345);
    let normal = Normal::new(0.0, noise_level).unwrap();

    let mut signal = create_sparse_signal(n, frequencies);

    // Add Gaussian noise
    for sample in &mut signal {
        *sample += normal.sample(&mut rng);
    }

    signal
}

fn setup_memory_manager() {
    init_global_memory_manager(
        GPUBackend::CPUFallback, // Use CPU fallback for benchmarks if CUDA is not available
        0,                       // Device ID 0
        AllocationStrategy::CacheBySize, // Use cache by size strategy
        1_073_741_824,           // 1GB of memory
    )
    .unwrap();
}

/// CPU Benchmarks
fn bench_cpu_sparse_fft_8192(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = sparse_fft(
            black_box(&signal),
            sparsity,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

fn bench_cpu_sparse_fft_16384(b: &mut Bencher) {
    let n = 16384;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = sparse_fft(
            black_box(&signal),
            sparsity,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

// GPU Benchmarks
fn bench_cuda_sparse_fft_8192_sublinear(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = cuda_sparse_fft(
            black_box(&signal),
            sparsity,
            0, // device_id
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

fn bench_cuda_sparse_fft_16384_sublinear(b: &mut Bencher) {
    let n = 16384;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = cuda_sparse_fft(
            black_box(&signal),
            sparsity,
            0, // device_id
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

fn bench_cuda_sparse_fft_8192_iterative(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = cuda_sparse_fft(
            black_box(&signal),
            sparsity,
            0, // device_id
            Some(SparseFFTAlgorithm::Iterative),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

fn bench_cuda_sparse_fft_16384_iterative(b: &mut Bencher) {
    let n = 16384;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 10;

    b.iter(|| {
        let _ = cuda_sparse_fft(
            black_box(&signal),
            sparsity,
            0, // device_id
            Some(SparseFFTAlgorithm::Iterative),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

fn bench_cuda_sparse_fft_with_noise(b: &mut Bencher) {
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3)];
    let signal = create_noisy_sparse_signal(n, &frequencies, 0.1);
    let sparsity = 10;

    b.iter(|| {
        let _ = cuda_sparse_fft(
            black_box(&signal),
            sparsity,
            0, // device_id
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

fn bench_cuda_sparse_fft_large(b: &mut Bencher) {
    let n = 32768;
    let frequencies = vec![(100, 1.0), (500, 0.5), (1000, 0.3), (1500, 0.2)];
    let signal = create_sparse_signal(n, &frequencies);
    let sparsity = 15;

    b.iter(|| {
        let _ = cuda_sparse_fft(
            black_box(&signal),
            sparsity,
            0, // device_id
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )
        .unwrap();
    });
}

fn cuda_sparse_fft_benches(c: &mut Criterion) {
    // Initialize memory manager once
    setup_memory_manager();

    c.bench_function("cpu_sparse_fft_8192", bench_cpu_sparse_fft_8192);
    c.bench_function("cpu_sparse_fft_16384", bench_cpu_sparse_fft_16384);
    c.bench_function(
        "cuda_sparse_fft_8192_sublinear",
        bench_cuda_sparse_fft_8192_sublinear,
    );
    c.bench_function(
        "cuda_sparse_fft_16384_sublinear",
        bench_cuda_sparse_fft_16384_sublinear,
    );
    c.bench_function(
        "cuda_sparse_fft_8192_iterative",
        bench_cuda_sparse_fft_8192_iterative,
    );
    c.bench_function(
        "cuda_sparse_fft_16384_iterative",
        bench_cuda_sparse_fft_16384_iterative,
    );
    c.bench_function(
        "cuda_sparse_fft_with_noise",
        bench_cuda_sparse_fft_with_noise,
    );
    c.bench_function("cuda_sparse_fft_large", bench_cuda_sparse_fft_large);
}

criterion_group!(benches, cuda_sparse_fft_benches);
criterion_main!(benches);
