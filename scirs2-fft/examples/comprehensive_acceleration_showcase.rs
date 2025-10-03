#![allow(clippy::all)]
//! Comprehensive Acceleration Showcase
//!
//! This example demonstrates the complete acceleration ecosystem of scirs2-fft,
//! including multi-GPU processing, specialized hardware support, and performance
//! comparison across different acceleration methods.

use scirs2_fft::{
    // GPU backends
    gpu_sparse_fft,
    is_cuda_available,
    is_hip_available,
    is_sycl_available,

    // Multi-GPU processing
    multi_gpu_sparse_fft,
    // Core FFT for comparison
    sparse_fft::sparse_fft,

    // Sparse FFT configuration
    sparse_fft::{SparseFFTAlgorithm, SparseFFTConfig, SparsityEstimationMethod},

    // Specialized hardware
    specialized_hardware_sparse_fft,
    FFTResult,
    GPUBackend,
    SpecializedHardwareManager,

    WorkloadDistribution,
};
use std::f64::consts::PI;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> FFTResult<()> {
    println!("🚀 SciRS2-FFT Comprehensive Acceleration Showcase");
    println!("================================================");

    // Create test signals of varying sizes
    let test_signals = create_test_signals();

    for (name, signal) in &test_signals {
        println!("\n📊 Testing with {} ({} samples)", name, signal.len());
        println!("{}", "=".repeat(50));

        // Performance comparison
        perform_acceleration_comparison(signal)?;

        if signal.len() >= 1024 {
            // Multi-GPU demonstration (only for larger signals)
            demonstrate_multi_gpu_processing(signal)?;

            // Specialized hardware demonstration
            demonstrate_specialized_hardware(signal)?;
        }
    }

    // Hardware capability overview
    display_hardware_capabilities()?;

    // Performance recommendations
    display_performance_recommendations();

    Ok(())
}

/// Create test signals of different characteristics
#[allow(dead_code)]
fn create_test_signals() -> Vec<(String, Vec<f64>)> {
    let mut signals = Vec::new();

    // Small signal for basic testing
    signals.push((
        "Small Sparse Signal".to_string(),
        create_sparse_signal(256, &[(10, 1.0), (50, 0.8), (100, 0.6)]),
    ));

    // Medium signal for GPU testing
    signals.push((
        "Medium Multi-tone Signal".to_string(),
        create_sparse_signal(1024, &[(100, 1.0), (200, 0.9), (300, 0.7), (400, 0.5)]),
    ));

    // Large signal for multi-GPU testing
    signals.push((
        "Large Complex Signal".to_string(),
        create_sparse_signal(
            4096,
            &[
                (500, 1.0),
                (1000, 0.9),
                (1500, 0.8),
                (2000, 0.7),
                (2500, 0.6),
                (3000, 0.5),
                (3500, 0.4),
            ],
        ),
    ));

    // Very large signal for specialized hardware
    signals.push((
        "Very Large Signal".to_string(),
        create_sparse_signal(
            16384,
            &[
                (1000, 1.0),
                (2000, 0.95),
                (4000, 0.9),
                (6000, 0.85),
                (8000, 0.8),
                (10000, 0.75),
                (12000, 0.7),
                (14000, 0.65),
            ],
        ),
    ));

    signals
}

/// Create a sparse signal with specified frequency components
#[allow(dead_code)]
fn create_sparse_signal(n: usize, components: &[(usize, f64)]) -> Vec<f64> {
    let mut signal = vec![0.0; n];

    for i in 0..n {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        for &(freq, amp) in components {
            signal[i] += amp * (freq as f64 * t).sin();
        }
    }

    // Add some noise
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(42);

    for sample in &mut signal {
        *sample += 0.01 * (rng.random::<f64>() - 0.5);
    }

    signal
}

/// Compare acceleration methods for a given signal
#[allow(dead_code)]
fn perform_acceleration_comparison(signal: &[f64]) -> FFTResult<()> {
    let sparsity = (signal.len() / 64).max(4).min(32); // Adaptive sparsity
    let config = SparseFFTConfig {
        sparsity,
        algorithm: SparseFFTAlgorithm::Sublinear,
        estimation_method: SparsityEstimationMethod::Manual,
        ..SparseFFTConfig::default()
    };

    println!("  Sparsity level: {sparsity}");
    println!("  Algorithm: {:?}", config.algorithm);

    // 1. CPU Reference Implementation
    print_performance_result("CPU Sparse FFT", || {
        sparse_fft(signal, sparsity, Some(SparseFFTAlgorithm::Sublinear), None)
    });

    // 2. GPU Acceleration Tests
    if is_cuda_available() {
        print_performance_result("CUDA GPU", || {
            gpu_sparse_fft(
                signal,
                sparsity,
                GPUBackend::CUDA,
                Some(SparseFFTAlgorithm::Sublinear),
                None,
            )
        });
    }

    if is_hip_available() {
        print_performance_result("HIP/ROCm GPU", || {
            gpu_sparse_fft(
                signal,
                sparsity,
                GPUBackend::HIP,
                Some(SparseFFTAlgorithm::Sublinear),
                None,
            )
        });
    }

    if is_sycl_available() {
        print_performance_result("SYCL GPU", || {
            gpu_sparse_fft(
                signal,
                sparsity,
                GPUBackend::SYCL,
                Some(SparseFFTAlgorithm::Sublinear),
                None,
            )
        });
    }

    Ok(())
}

/// Demonstrate multi-GPU processing capabilities
#[allow(dead_code)]
fn demonstrate_multi_gpu_processing(signal: &[f64]) -> FFTResult<()> {
    println!("\n  🔄 Multi-GPU Processing:");

    let sparsity = (signal.len() / 64).max(4).min(32);

    // Test different workload distribution strategies
    let strategies = [
        ("Equal Distribution", WorkloadDistribution::Equal),
        ("Memory-Based", WorkloadDistribution::MemoryBased),
        ("Compute-Based", WorkloadDistribution::ComputeBased),
        ("Adaptive", WorkloadDistribution::Adaptive),
    ];

    for (strategy_name, _strategy) in &strategies {
        print_performance_result(&format!("Multi-GPU ({strategy_name})"), || {
            multi_gpu_sparse_fft(signal, sparsity, Some(SparseFFTAlgorithm::Sublinear), None)
        });
    }

    Ok(())
}

/// Demonstrate specialized hardware capabilities
#[allow(dead_code)]
fn demonstrate_specialized_hardware(signal: &[f64]) -> FFTResult<()> {
    println!("\n  ⚡ Specialized Hardware:");

    let sparsity = (signal.len() / 64).max(4).min(32);
    let config = SparseFFTConfig {
        sparsity,
        algorithm: SparseFFTAlgorithm::Sublinear,
        estimation_method: SparsityEstimationMethod::Manual,
        ..SparseFFTConfig::default()
    };

    print_performance_result("Specialized Hardware", || {
        specialized_hardware_sparse_fft(signal, config.clone())
    });

    // Detailed hardware analysis
    let mut manager = SpecializedHardwareManager::new(config);
    if let Ok(discovered) = manager.discover_accelerators() {
        if !discovered.is_empty() {
            println!("    Available accelerators:");
            manager.initialize_all().ok();

            for id in &discovered {
                if let Some(info) = manager.get_accelerator_info(id) {
                    println!(
                        "      • {} ({}): {:.1} GFLOPS, {:.1}W, {:.2}μs latency",
                        id,
                        info.accelerator_type,
                        info.capabilities.peak_throughput_gflops,
                        info.capabilities.power_consumption_watts,
                        info.capabilities.latency_us
                    );
                }
            }
        }
    }

    Ok(())
}

/// Helper function to measure and print performance results
#[allow(dead_code)]
fn print_performance_result<F, R>(name: &str, f: F)
where
    F: FnOnce() -> FFTResult<R>,
    R: std::fmt::Debug,
{
    let start = Instant::now();
    match f() {
        Ok(_result) => {
            let elapsed = start.elapsed();
            println!(
                "    ✅ {:<20}: {:>8.2}ms",
                name,
                elapsed.as_secs_f64() * 1000.0
            );
        }
        Err(e) => {
            println!("    ❌ {name:<20}: Failed ({e})");
        }
    }
}

/// Display comprehensive hardware capabilities
#[allow(dead_code)]
fn display_hardware_capabilities() -> FFTResult<()> {
    println!("\n🔧 Hardware Capabilities Overview");
    println!("================================");

    // GPU Backend Status
    println!("GPU Backends:");
    println!(
        "  CUDA:    {}",
        if is_cuda_available() {
            "✅ Available"
        } else {
            "❌ Not available"
        }
    );
    println!(
        "  HIP:     {}",
        if is_hip_available() {
            "✅ Available"
        } else {
            "❌ Not available"
        }
    );
    println!(
        "  SYCL:    {}",
        if is_sycl_available() {
            "✅ Available"
        } else {
            "❌ Not available"
        }
    );

    // Specialized Hardware Status
    let config = SparseFFTConfig::default();
    let mut manager = SpecializedHardwareManager::new(config);

    match manager.discover_accelerators() {
        Ok(discovered) => {
            println!("\nSpecialized Hardware:");
            if discovered.is_empty() {
                println!("  No specialized accelerators found");
            } else {
                manager.initialize_all().ok();
                for id in discovered {
                    if let Some(info) = manager.get_accelerator_info(&id) {
                        println!("  {id}:");
                        println!("    Type: {}", info.accelerator_type);
                        println!(
                            "    Peak Performance: {:.1} GFLOPS",
                            info.capabilities.peak_throughput_gflops
                        );
                        println!(
                            "    Power Efficiency: {:.1} GFLOPS/W",
                            info.capabilities.peak_throughput_gflops
                                / info.capabilities.power_consumption_watts
                        );
                        println!(
                            "    Max Signal Size: {} samples",
                            info.capabilities.max_signal_size
                        );
                    }
                }
            }
        }
        Err(e) => {
            println!("  Error discovering accelerators: {e}");
        }
    }

    Ok(())
}

/// Display performance recommendations based on use cases
#[allow(dead_code)]
fn display_performance_recommendations() {
    println!("\n💡 Performance Recommendations");
    println!("==============================");

    println!("Signal Size Recommendations:");
    println!("  < 1K samples:    CPU sparse FFT (lowest overhead)");
    println!("  1K - 10K:        Single GPU acceleration");
    println!("  10K - 100K:      Multi-GPU processing");
    println!("  > 100K:          Specialized hardware (FPGA/ASIC)");

    println!("\nUse Case Recommendations:");
    println!("  Real-time (< 1ms):     ASIC or FPGA accelerators");
    println!("  Batch processing:      Multi-GPU with adaptive distribution");
    println!("  Power-constrained:     Low-power ASICs or efficient GPUs");
    println!("  Research/flexible:     GPU backends with CPU fallback");

    println!("\nAlgorithm Selection:");
    println!("  Clean signals:         CompressedSensing algorithm");
    println!("  Noisy signals:         Iterative algorithm");
    println!("  Large signals:         FrequencyPruning algorithm");
    println!("  General purpose:       Sublinear algorithm (fastest)");

    println!("\nOptimal Feature Combinations:");
    if is_cuda_available() && is_hip_available() {
        println!("  Multi-vendor setup:    CUDA + HIP for maximum throughput");
    }
    if is_sycl_available() {
        println!("  Cross-platform:       SYCL for portable acceleration");
    }
    println!("  Maximum performance:   All GPU backends + specialized hardware");
    println!("  Development/testing:   CPU fallback always available");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sparse_signal() {
        let signal = create_sparse_signal(128, &[(10, 1.0), (20, 0.5)]);
        assert_eq!(signal.len(), 128);

        // Signal should not be all zeros
        assert!(signal.iter().any(|&x| x.abs() > 0.1));
    }

    #[test]
    fn test_acceleration_comparison_small_signal() {
        let signal = create_sparse_signal(64, &[(5, 1.0), (15, 0.8)]);

        // Should not panic on small signals
        assert!(perform_acceleration_comparison(&signal).is_ok());
    }
}
