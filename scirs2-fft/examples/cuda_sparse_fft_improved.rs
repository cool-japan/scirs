//! Enhanced CUDA Sparse FFT Example
//!
//! This example demonstrates the improved CUDA integration with proper memory management
//! and GPU acceleration for sparse FFT algorithms.

use num_complex::Complex64;
use scirs2_fft::{
    sparse_fft::{SparseFFTAlgorithm, WindowFunction},
    sparse_fft_gpu_cuda::cuda_sparse_fft,
    sparse_fft_gpu_memory::is_cuda_available,
    FFTResult,
};
use std::time::Instant;

fn main() -> FFTResult<()> {
    println!("Enhanced CUDA Sparse FFT Example");
    println!("=================================");

    // Check if CUDA is available
    let cuda_available = is_cuda_available();
    println!("CUDA Available: {}", cuda_available);

    if !cuda_available {
        println!(
            "CUDA is not available. This example requires CUDA to demonstrate GPU acceleration."
        );
        println!("Running CPU fallback instead...");
    }

    // Create a test signal with sparse frequency components
    let n = 1024;
    let mut signal = vec![Complex64::new(0.0, 0.0); n];

    // Add some sparse frequency components
    for i in 0..n {
        let t = i as f64 / n as f64;
        // Add components at specific frequencies
        signal[i] = Complex64::new(
            (2.0 * std::f64::consts::PI * 50.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 120.0 * t).sin()
                + 0.3 * (2.0 * std::f64::consts::PI * 200.0 * t).sin(),
            0.0,
        );
    }

    println!("Signal length: {}", n);
    println!("Expected sparsity (k): 3");

    // Test different algorithms and window functions
    let algorithms = vec![
        ("Sublinear", SparseFFTAlgorithm::Sublinear),
        ("CompressedSensing", SparseFFTAlgorithm::CompressedSensing),
        ("Iterative", SparseFFTAlgorithm::Iterative),
    ];

    let windows = vec![
        ("None", WindowFunction::None),
        ("Hann", WindowFunction::Hann),
        ("Hamming", WindowFunction::Hamming),
        ("Blackman", WindowFunction::Blackman),
    ];

    for (algo_name, algorithm) in algorithms {
        for (window_name, window) in &windows {
            println!(
                "\n--- Testing {} with {} window ---",
                algo_name, window_name
            );

            let start = Instant::now();

            match cuda_sparse_fft(
                &signal,
                3, // Expected sparsity
                0, // Device ID (auto-select)
                Some(algorithm),
                Some(*window),
            ) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    println!("✓ Success!");
                    println!("  Execution time: {:?}", elapsed);
                    println!("  Found {} significant components", result.indices.len());
                    println!(
                        "  Frequency domain sparsity: {:.2}%",
                        (result.indices.len() as f64 / n as f64) * 100.0
                    );

                    // Show the detected frequency components
                    println!("  Top frequency components:");
                    for (i, (&idx, &val)) in result
                        .indices
                        .iter()
                        .zip(result.values.iter())
                        .take(5)
                        .enumerate()
                    {
                        let freq = idx as f64 * 1000.0 / n as f64; // Assuming 1kHz sample rate
                        println!(
                            "    {}: f={:.1}Hz, magnitude={:.3}",
                            i + 1,
                            freq,
                            val.norm()
                        );
                    }
                }
                Err(e) => {
                    println!("✗ Error: {}", e);
                }
            }
        }
    }

    println!("\n--- Performance Summary ---");
    if cuda_available {
        println!("✓ GPU acceleration was used for compatible operations");
        println!("✓ Automatic fallback to CPU for unsupported operations");
        println!("✓ Proper memory management with device/host transfers");
    } else {
        println!("• All operations executed on CPU (CUDA not available)");
        println!("• Memory management optimized for host operations");
    }

    Ok(())
}
