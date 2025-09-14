use num_complex::Complex64;
use rand::Rng;
use scirs2_fft::{
    execute_cuda_spectral_flatness_sparse_fft, gpu_sparse_fft, sparse_fft,
    sparse_fft::SparseFFTAlgorithm, spectral_flatness_sparse_fft, GPUBackend, WindowFunction,
};
use std::time::Instant;

#[allow(dead_code)]
fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
    let mut signal = vec![0.0; n];
    for &(freq, amplitude) in frequencies {
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (freq as f64) * (i as f64) / (n as f64);
            signal[i] += amplitude * angle.sin();
        }
    }
    signal
}

#[allow(dead_code)]
fn add_noise(_signal: &[f64], noise_level: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    _signal
        .iter()
        .map(|&x| x + rng.random_range(-noise_level..noise_level))
        .collect()
}

#[allow(dead_code)]
fn print_result(
    name: &str,
    result: &scirs2_fft::sparse_fft::SparseFFTResult,
    expected_freqs: &[usize],
) {
    println!("=== {name} Results ===");
    println!("Algorithm: {:?}", result.algorithm);
    println!(
        "Computation Time: {:.3} ms",
        result.computation_time.as_millis()
    );
    println!("Found {} significant components", result.values.len());

    println!("Top frequencies found: {:?}", result.indices);

    // Check if the expected frequencies were found
    let found_count = expected_freqs
        .iter()
        .filter(|&&f| {
            result.indices.contains(&f) || result.indices.contains(&(result.indices.len() - f))
        })
        .count();

    println!(
        "Found {}/{} expected frequencies",
        found_count,
        expected_freqs.len()
    );
    println!();
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters
    let n = 8192;
    let frequencies = vec![(100, 1.0), (500, 0.7), (1200, 0.4)];
    let expected_freqs = vec![100, 500, 1200];
    let sparsity = 6; // Max number of components to search for
    let noise_level = 0.2;

    println!("Generating signal with {n} samples");
    println!("Frequencies: {frequencies:?}");
    println!("Noise level: {noise_level}");
    println!();

    // Create a sparse signal with noise
    let signal = create_sparse_signal(n, &frequencies);
    let noisy_signal = add_noise(&signal, noise_level);

    // CPU implementation with standard algorithm
    println!("Running CPU sparse FFT (standard algorithm)...");
    let start = Instant::now();
    let result_cpu = sparse_fft(
        &noisy_signal,
        sparsity,
        Some(SparseFFTAlgorithm::Sublinear),
        None,
    )?;
    let duration_cpu = start.elapsed();
    println!("CPU time: {:.3} ms", duration_cpu.as_millis());

    // CPU implementation with spectral flatness algorithm
    println!("Running CPU sparse FFT with spectral flatness algorithm...");
    let start = Instant::now();
    let result_spectral = spectral_flatness_sparse_fft(&noisy_signal, 0.3, 32)?;
    let duration_spectral = start.elapsed();
    println!(
        "CPU time with spectral flatness: {:.3} ms",
        duration_spectral.as_millis()
    );

    // CUDA implementation with spectral flatness algorithm
    println!("Running CUDA sparse FFT with spectral flatness algorithm...");
    let start = Instant::now();
    let result_cuda = gpu_sparse_fft(
        &noisy_signal,
        sparsity,
        GPUBackend::CUDA,
        Some(SparseFFTAlgorithm::SpectralFlatness),
        Some(WindowFunction::Hann),
    )?;
    let duration_cuda = start.elapsed();
    println!(
        "CUDA time with spectral flatness: {:.3} ms",
        duration_cuda.as_millis()
    );

    // Direct CUDA kernel execution
    println!("Running CUDA kernel directly...");
    let signal_complex: Vec<Complex64> = noisy_signal
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    let start = Instant::now();
    let result_direct = execute_cuda_spectral_flatness_sparse_fft(
        &signal_complex,
        sparsity,
        0.3, // Flatness threshold
    )?;
    let duration_direct = start.elapsed();
    println!(
        "Direct CUDA kernel time: {:.3} ms",
        duration_direct.as_millis()
    );

    // Print results
    print_result("CPU Standard", &result_cpu, &expected_freqs);
    print_result("CPU Spectral Flatness", &result_spectral, &expected_freqs);
    print_result("CUDA Spectral Flatness", &result_cuda, &expected_freqs);
    print_result("Direct CUDA Kernel", &result_direct, &expected_freqs);

    // Performance comparison
    println!("=== Performance Comparison ===");
    println!("CPU Standard:         {:.3} ms", duration_cpu.as_millis());
    println!(
        "CPU Spectral Flatness: {:.3} ms",
        duration_spectral.as_millis()
    );
    println!(
        "CUDA Spectral Flatness: {:.3} ms",
        duration_cuda.as_millis()
    );
    println!(
        "Direct CUDA Kernel:    {:.3} ms",
        duration_direct.as_millis()
    );

    // Speedup calculations
    let speedup_cpu = duration_cpu.as_secs_f64() / duration_cuda.as_secs_f64();
    let speedup_spectral = duration_spectral.as_secs_f64() / duration_cuda.as_secs_f64();

    println!("Speedup over CPU standard: {speedup_cpu:.2}x");
    println!("Speedup over CPU spectral flatness: {speedup_spectral:.2}x");

    Ok(())
}
