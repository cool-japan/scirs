#![allow(clippy::needless_range_loop)]
//! Specialized Hardware Accelerator Example
//!
//! This example demonstrates the use of specialized hardware accelerators
//! including FPGAs and ASICs for high-performance sparse FFT computation.

use scirs2_fft::{
    sparse_fft::{SparseFFTAlgorithm, SparseFFTConfig, SparsityEstimationMethod},
    sparse_fft_specialized_hardware::{
        specialized_hardware_sparse_fft, SpecializedHardwareManager,
    },
    FFTResult,
};
use std::f64::consts::PI;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> FFTResult<()> {
    println!("Specialized Hardware Accelerator Example");
    println!("========================================");

    // Test hardware discovery
    test_hardware_discovery()?;

    // Test performance comparison
    test_performance_comparison()?;

    // Test different signal sizes
    test_signalsize_scaling()?;

    // Test power efficiency
    test_power_efficiency()?;

    Ok(())
}

/// Test hardware discovery and capabilities
#[allow(dead_code)]
fn test_hardware_discovery() -> FFTResult<()> {
    println!("\n--- Hardware Discovery ---");

    let config = SparseFFTConfig {
        sparsity: 10,
        algorithm: SparseFFTAlgorithm::Sublinear,
        estimation_method: SparsityEstimationMethod::Manual,
        ..SparseFFTConfig::default()
    };

    let mut manager = SpecializedHardwareManager::new(config);
    let discovered = manager.discover_accelerators()?;

    println!("Discovered {} accelerator(s):", discovered.len());
    for id in &discovered {
        println!("  - {id}");
    }

    // Initialize all accelerators
    manager.initialize_all()?;

    // Show detailed information for each accelerator
    for id in &discovered {
        if let Some(info) = manager.get_accelerator_info(id) {
            println!("\nAccelerator: {id}");
            println!("  Type: {}", info.accelerator_type);
            println!("  Name: {}", info.name);
            println!("  Vendor: {}", info.vendor);
            println!("  Driver: {}", info.driver_version);
            println!("  Available: {}", info.is_available);

            println!("  Capabilities:");
            println!(
                "    Max signal size: {} samples",
                info.capabilities.max_signal_size
            );
            println!("    Max sparsity: {}", info.capabilities.max_sparsity);
            println!(
                "    Memory bandwidth: {:.1} GB/s",
                info.capabilities.memory_bandwidth_gb_s
            );
            println!(
                "    Peak throughput: {:.1} GFLOPS",
                info.capabilities.peak_throughput_gflops
            );
            println!(
                "    Power consumption: {:.1} W",
                info.capabilities.power_consumption_watts
            );
            println!("    Latency: {:.2} μs", info.capabilities.latency_us);
            println!(
                "    Parallel support: {}",
                info.capabilities.supports_parallel
            );
            println!(
                "    Pipeline support: {}",
                info.capabilities.supports_pipeline
            );

            if !info.capabilities.custom_features.is_empty() {
                println!("  Custom features:");
                for (feature, value) in &info.capabilities.custom_features {
                    println!("    {feature}: {value}");
                }
            }
        }
    }

    Ok(())
}

/// Test performance comparison between different accelerators
#[allow(dead_code)]
fn test_performance_comparison() -> FFTResult<()> {
    println!("\n--- Performance Comparison ---");

    let signalsizes = vec![256, 512, 1024, 2048];
    let sparsity_levels = vec![4, 8, 16, 32];

    for signalsize in signalsizes {
        for &sparsity in &sparsity_levels {
            println!("\nSignal size: {signalsize}, Sparsity: {sparsity}");

            let signal = create_test_signal(signalsize);
            let config = SparseFFTConfig {
                sparsity,
                algorithm: SparseFFTAlgorithm::Sublinear,
                estimation_method: SparsityEstimationMethod::Manual,
                ..SparseFFTConfig::default()
            };

            // Test with specialized hardware
            let start = Instant::now();
            match specialized_hardware_sparse_fft(&signal, config) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    let throughput = signalsize as f64 / elapsed.as_secs_f64();

                    println!("  Specialized Hardware: {elapsed:?}");
                    println!("    Throughput: {throughput:.0} samples/sec");
                    println!("    Found components: {}", result.values.len());
                    println!("    Execution time: {:?}", result.computation_time);
                }
                Err(e) => {
                    println!("  Specialized Hardware: Failed ({e})");
                }
            }
        }
    }

    Ok(())
}

/// Test scaling behavior with different signal sizes
#[allow(dead_code)]
fn test_signalsize_scaling() -> FFTResult<()> {
    println!("\n--- Signal Size Scaling ---");

    let signalsizes = vec![1024, 4096, 16384, 65536, 262144];
    let sparsity = 16;

    println!("Testing scaling with sparsity = {sparsity}");
    println!("Signal Size\tExecution Time\tThroughput (samples/sec)\tEfficiency");
    println!("===========\t==============\t=======================\t==========");

    for signalsize in signalsizes {
        let signal = create_test_signal(signalsize);
        let config = SparseFFTConfig {
            sparsity,
            algorithm: SparseFFTAlgorithm::Sublinear,
            estimation_method: SparsityEstimationMethod::Manual,
            ..SparseFFTConfig::default()
        };

        let start = Instant::now();
        match specialized_hardware_sparse_fft(&signal, config) {
            Ok(_result) => {
                let elapsed = start.elapsed();
                let throughput = signalsize as f64 / elapsed.as_secs_f64();
                let efficiency = throughput / (signalsize as f64); // Relative efficiency

                println!("{signalsize}\t\t{elapsed:?}\t\t{throughput:.0}\t\t\t{efficiency:.3}");
            }
            Err(e) => {
                println!("{signalsize}\t\tFailed: {e}");
            }
        }
    }

    Ok(())
}

/// Test power efficiency of different accelerators
#[allow(dead_code)]
fn test_power_efficiency() -> FFTResult<()> {
    println!("\n--- Power Efficiency Analysis ---");

    let config = SparseFFTConfig {
        sparsity: 20,
        algorithm: SparseFFTAlgorithm::Sublinear,
        estimation_method: SparsityEstimationMethod::Manual,
        ..SparseFFTConfig::default()
    };

    let mut manager = SpecializedHardwareManager::new(config);
    manager.discover_accelerators()?;
    manager.initialize_all()?;

    let available = manager.get_available_accelerators();

    println!("Accelerator\t\tType\t\tPower (W)\tThroughput (GFLOPS)\tEfficiency (GFLOPS/W)");
    println!("===========\t\t====\t\t=========\t==================\t====================");

    for id in available {
        if let Some(info) = manager.get_accelerator_info(&id) {
            let power = info.capabilities.power_consumption_watts;
            let throughput = info.capabilities.peak_throughput_gflops;
            let efficiency = throughput / power;

            println!(
                "{}\t\t{}\t\t{:.1}\t\t{:.1}\t\t\t{:.2}",
                id, info.accelerator_type, power, throughput, efficiency
            );
        }
    }

    // Analyze power vs performance trade-offs
    println!("\nPower vs Performance Analysis:");

    let signal = create_test_signal(4096);

    for id in manager.get_available_accelerators() {
        // Get info first to avoid borrow conflicts
        let (power, id_clone) = if let Some(info) = manager.get_accelerator_info(&id) {
            (info.capabilities.power_consumption_watts, id.clone())
        } else {
            continue;
        };

        let start = Instant::now();

        // Simulate execution on specific accelerator
        if let Ok(result) = manager.execute_sparse_fft(&signal) {
            let elapsed = start.elapsed();
            let energy_per_operation = power * elapsed.as_secs_f64(); // Joules

            println!(
                "  {}: {:.3} J/operation, {} components found",
                id_clone,
                energy_per_operation,
                result.values.len()
            );
        }
    }

    Ok(())
}

/// Create a test signal with sparse frequency components
#[allow(dead_code)]
fn create_test_signal(n: usize) -> Vec<f64> {
    let mut signal = vec![0.0; n];

    // Add sparse frequency components
    let frequencies = vec![
        (50, 1.0),  // 50 Hz, amplitude 1.0
        (150, 0.8), // 150 Hz, amplitude 0.8
        (300, 0.6), // 300 Hz, amplitude 0.6
        (450, 0.4), // 450 Hz, amplitude 0.4
        (600, 0.3), // 600 Hz, amplitude 0.3
    ];

    for i in 0..n {
        let t = 2.0 * PI * (i as f64) / (n as f64);
        for &(freq, amp) in &frequencies {
            signal[i] += amp * (freq as f64 * t).sin();
        }
    }

    // Add some noise
    use scirs2_core::random::rngs::StdRng;
    use scirs2_core::random::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(12345);

    for sample in &mut signal {
        *sample += 0.05 * (rng.random::<f64>() - 0.5);
    }

    signal
}

/// Display accelerator recommendations
#[allow(dead_code)]
fn display_accelerator_recommendations() {
    println!("\n--- Accelerator Recommendations ---");

    println!("For different use cases:");
    println!();

    println!("Advanced-Low Latency Applications (< 1 μs):");
    println!("  ✓ ASIC accelerators - Purpose-built for sparse FFT");
    println!("  ✓ FPGA with custom bitstreams - Configurable for specific needs");
    println!("  • Consider: Power consumption vs latency trade-offs");
    println!();

    println!("High-Throughput Batch Processing:");
    println!("  ✓ Multi-accelerator setups with pipeline processing");
    println!("  ✓ FPGA with parallel processing units");
    println!("  • Consider: Memory bandwidth limitations");
    println!();

    println!("Power-Constrained Environments:");
    println!("  ✓ Low-power ASICs with optimized algorithms");
    println!("  ✓ DSP processors with sparse FFT optimizations");
    println!("  • Consider: Performance vs power efficiency");
    println!();

    println!("Flexible/Research Applications:");
    println!("  ✓ FPGA with reconfigurable logic");
    println!("  ✓ GPU + FPGA hybrid systems");
    println!("  • Consider: Development time vs performance");
    println!();

    println!("Real-Time Signal Processing:");
    println!("  ✓ ASIC with guaranteed latency bounds");
    println!("  ✓ DSP with real-time OS support");
    println!("  • Consider: Jitter and deterministic behavior");
}
