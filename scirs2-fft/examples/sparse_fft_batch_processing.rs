use rand::Rng;
use scirs2_fft::sparse_fft::SparseFFTAlgorithm;
/// Batch Processing Example for Sparse FFT
///
/// This example demonstrates the batch processing capabilities for sparse FFT,
/// which can significantly improve performance when processing multiple signals.
use scirs2_fft::{
    batch_sparse_fft, gpu_batch_sparse_fft, spectral_flatness_batch_sparse_fft, BatchConfig,
    GPUBackend, WindowFunction,
};
use std::f64::consts::PI;
use std::time::Instant;

// Helper function to create a sparse signal with specified frequencies
#[allow(dead_code)]
fn create_sparse_signal(n: usize, frequencies: &[(usize, f64)]) -> Vec<f64> {
    let mut signal = vec![0.0; n];
    for (i, sample) in signal.iter_mut().enumerate().take(n) {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        for &(freq, amp) in frequencies {
            *sample += amp * (freq as f64 * t).sin();
        }
    }
    signal
}

// Helper to add noise to signals
#[allow(dead_code)]
fn add_noise(signal: &[f64], noise_level: f64) -> Vec<f64> {
    let mut rng = rand::rng();
    signal
        .iter()
        .map(|&x| x + rng.random_range(-noise_level..noise_level))
        .collect()
}

// Helper to create a batch of signals with varying parameters
#[allow(dead_code)]
fn create_test_batch(count: usize) -> Vec<Vec<f64>> {
    let mut signals = Vec::with_capacity(count);
    let _rng = rand::rng();

    for i in 0..count {
        // Vary signal parameters slightly for each signal
        let n = 1024 + (i % 5) * 256; // Different sizes
        let noise_level = 0.05 + (i as f64 * 0.01); // Gradually increasing noise

        // Different frequency components for each signal
        let freq1 = 10 + i % 20;
        let freq2 = 50 + i % 30;
        let freq3 = 100 + i % 50;

        let frequencies = vec![(freq1, 1.0), (freq2, 0.7), (freq3, 0.4)];

        // Create signal and add noise
        let base_signal = create_sparse_signal(n, &frequencies);
        let noisy_signal = add_noise(&base_signal, noise_level);

        signals.push(noisy_signal);
    }

    signals
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Sparse FFT Batch Processing Example");
    println!("===================================\n");

    // Create a batch of test signals
    let batchsize = 20;
    println!("Creating {batchsize} test signals with varying parameters...");
    let signals = create_test_batch(batchsize);

    // 1. Sequential CPU processing (baseline)
    println!("\n1. Sequential CPU Processing:");
    let start = Instant::now();
    let results_sequential = process_signals_sequentially(&signals)?;
    let sequential_time = start.elapsed();
    println!("   Time: {:.3} ms", sequential_time.as_millis());

    // 2. Parallel CPU batch processing
    println!("\n2. Parallel CPU Batch Processing:");
    let start = Instant::now();
    let config = BatchConfig {
        use_parallel: true,
        max_batch_size: batchsize,
        ..BatchConfig::default()
    };
    let results_cpu_batch = batch_sparse_fft(
        &signals,
        10, // Look for up to 10 components
        Some(SparseFFTAlgorithm::Sublinear),
        Some(WindowFunction::Hann),
        Some(config),
    )?;
    let cpu_batch_time = start.elapsed();
    println!("   Time: {:.3} ms", cpu_batch_time.as_millis());
    println!(
        "   Speedup over sequential: {:.2}x",
        sequential_time.as_secs_f64() / cpu_batch_time.as_secs_f64()
    );

    // 3. Spectral flatness CPU batch processing
    println!("\n3. Spectral Flatness CPU Batch Processing:");
    let start = Instant::now();
    let config = BatchConfig {
        use_parallel: true,
        max_batch_size: batchsize,
        ..BatchConfig::default()
    };
    let results_spectral = spectral_flatness_batch_sparse_fft(
        &signals,
        0.3, // Flatness threshold
        32,  // Window size
        Some(WindowFunction::Hann),
        None, // Use CPU
        Some(config),
    )?;
    let spectral_time = start.elapsed();
    println!("   Time: {:.3} ms", spectral_time.as_millis());

    // 4. GPU batch processing (if available)
    if scirs2_fft::sparse_fft_gpu_memory::is_cuda_available() {
        println!("\n4. GPU Batch Processing (CUDA):");
        let start = Instant::now();
        let _config = BatchConfig {
            max_batch_size: 5,                        // Process in smaller batches
            max_memory_per_batch: 1024 * 1024 * 1024, // 1 GB limit
            use_mixed_precision: true,
            ..BatchConfig::default()
        };
        let results_gpu_batch = gpu_batch_sparse_fft(
            &signals,
            10, // Look for up to 10 components
            GPUBackend::CUDA,
            Some(SparseFFTAlgorithm::Sublinear),
            Some(WindowFunction::Hann),
        )?;
        let gpu_batch_time = start.elapsed();
        println!("   Time: {:.3} ms", gpu_batch_time.as_millis());
        println!(
            "   Speedup over sequential: {:.2}x",
            sequential_time.as_secs_f64() / gpu_batch_time.as_secs_f64()
        );
        println!(
            "   Speedup over CPU batch: {:.2}x",
            cpu_batch_time.as_secs_f64() / gpu_batch_time.as_secs_f64()
        );

        // Verify consistency of results
        verify_results(&results_sequential, &results_gpu_batch);
    } else {
        println!("\n4. GPU Processing: CUDA not available");
    }

    // Verify consistency of results
    verify_results(&results_sequential, &results_cpu_batch);

    // Advanced analysis
    println!("\nResult Analysis:");
    println!("----------------");
    analyze_results(&results_sequential, &results_cpu_batch, &results_spectral);

    println!("\nBatch processing experiment complete!");

    Ok(())
}

// Process signals sequentially (baseline)
#[allow(dead_code)]
fn process_signals_sequentially(
    signals: &[Vec<f64>],
) -> Result<Vec<scirs2_fft::sparse_fft::SparseFFTResult>, Box<dyn std::error::Error>> {
    let mut results = Vec::with_capacity(signals.len());

    for (i, signal) in signals.iter().enumerate() {
        if i % 5 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout())?;
        }

        let result = scirs2_fft::sparse_fft(
            signal,
            10, // Look for up to 10 components
            Some(SparseFFTAlgorithm::Sublinear),
            Some(42), // Random seed
        )?;

        results.push(result);
    }
    println!();

    Ok(results)
}

// Verify that different processing methods give similar results
#[allow(dead_code)]
fn verify_results(
    baseline_results: &[scirs2_fft::sparse_fft::SparseFFTResult],
    test_results: &[scirs2_fft::sparse_fft::SparseFFTResult],
) {
    assert_eq!(
        baseline_results.len(),
        test_results.len(),
        "Result count mismatch! Expected {}, got {}",
        baseline_results.len(),
        test_results.len()
    );

    let mut match_count = 0;
    let mut partial_match_count = 0;

    for (base, test) in baseline_results.iter().zip(test_results.iter()) {
        // Two main checks:
        // 1. Are the top 3 frequencies the same?
        let mut found_main_frequencies = 0;
        for idx in &base.indices[0..3.min(base.indices.len())] {
            if test.indices.contains(idx) {
                found_main_frequencies += 1;
            }
        }

        if found_main_frequencies == base.indices[0..3.min(base.indices.len())].len() {
            match_count += 1;
        } else if found_main_frequencies > 0 {
            partial_match_count += 1;
        }
    }

    let total = baseline_results.len();
    println!("Results comparison:");
    println!(
        "  Exact matches (top frequencies): {}/{} ({:.1}%)",
        match_count,
        total,
        100.0 * match_count as f64 / total as f64
    );
    println!(
        "  Partial matches: {}/{} ({:.1}%)",
        partial_match_count,
        total,
        100.0 * partial_match_count as f64 / total as f64
    );
    println!(
        "  Total consistency: {:.1}%",
        100.0 * (match_count + partial_match_count) as f64 / total as f64
    );
}

// Analyze results from different processing methods
#[allow(dead_code)]
fn analyze_results(
    sequential_results: &[scirs2_fft::sparse_fft::SparseFFTResult],
    parallel_results: &[scirs2_fft::sparse_fft::SparseFFTResult],
    spectral_results: &[scirs2_fft::sparse_fft::SparseFFTResult],
) {
    // Calculate average number of components found
    let seq_avg = sequential_results
        .iter()
        .map(|r| r.values.len())
        .sum::<usize>() as f64
        / sequential_results.len() as f64;
    let par_avg = parallel_results
        .iter()
        .map(|r| r.values.len())
        .sum::<usize>() as f64
        / parallel_results.len() as f64;
    let spec_avg = spectral_results
        .iter()
        .map(|r| r.values.len())
        .sum::<usize>() as f64
        / spectral_results.len() as f64;

    println!("Average components found:");
    println!("  Sequential: {seq_avg:.1}");
    println!("  Parallel batch: {par_avg:.1}");
    println!("  Spectral flatness: {spec_avg:.1}");

    // Calculate average computation time
    let seq_time = sequential_results
        .iter()
        .map(|r| r.computation_time.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / sequential_results.len() as f64;
    let par_time = parallel_results
        .iter()
        .map(|r| r.computation_time.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / parallel_results.len() as f64;
    let spec_time = spectral_results
        .iter()
        .map(|r| r.computation_time.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / spectral_results.len() as f64;

    println!("Average per-signal processing time (ms):");
    println!("  Sequential: {seq_time:.3}");
    println!("  Parallel batch: {par_time:.3}");
    println!("  Spectral flatness: {spec_time:.3}");
}
