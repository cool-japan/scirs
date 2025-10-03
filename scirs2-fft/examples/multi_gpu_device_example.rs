#![allow(clippy::needless_range_loop)]
//! Multi-GPU Device Enumeration and Management Example
//!
//! This example demonstrates device enumeration, selection, and workload
//! distribution across multiple GPU devices for high-performance sparse FFT.

use scirs2_fft::{
    sparse_fft_multi_gpu::{MultiGPUConfig, MultiGPUSparseFFT, WorkloadDistribution},
    FFTResult,
};
use std::f64::consts::PI;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> FFTResult<()> {
    println!("Multi-GPU Device Enumeration and Management Example");
    println!("===================================================");

    // Test device enumeration
    test_device_enumeration()?;

    // Test workload distribution strategies
    test_workload_distribution_strategies()?;

    // Test performance scaling
    test_performance_scaling()?;

    // Test adaptive load balancing
    test_adaptive_load_balancing()?;

    Ok(())
}

/// Test device enumeration and selection
#[allow(dead_code)]
fn test_device_enumeration() -> FFTResult<()> {
    println!("\n--- Device Enumeration Test ---");

    let mut processor = MultiGPUSparseFFT::new(MultiGPUConfig::default());
    processor.initialize()?;

    let devices = processor.get_devices();
    println!("Found {} devices:", devices.len());

    for (i, device) in devices.iter().enumerate() {
        println!("  Device {}: {}", i, device.device_name);
        println!("    Backend: {:?}", device.backend);
        println!("    Device ID: {}", device.device_id);
        println!(
            "    Memory: {:.1} GB total, {:.1} GB free",
            device.memory_total as f64 / (1024.0 * 1024.0 * 1024.0),
            device.memory_free as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("    Compute capability: {:.1}", device.compute_capability);
        println!("    Compute units: {}", device.compute_units);
        println!(
            "    Max threads per block: {}",
            device.max_threads_per_block
        );
        println!("    Available: {}", device.is_available);
    }

    let selected_devices = processor.get_selected_devices();
    println!(
        "\nSelected {} devices for processing:",
        selected_devices.len()
    );
    for device in selected_devices {
        println!("  - {} ({})", device.device_name, device.backend as u8);
    }

    Ok(())
}

/// Test different workload distribution strategies
#[allow(dead_code)]
fn test_workload_distribution_strategies() -> FFTResult<()> {
    println!("\n--- Workload Distribution Strategies Test ---");

    // Create a test signal
    let signal = create_test_signal(8192);
    let sparsity = 10;

    let strategies = vec![
        ("Equal Distribution", WorkloadDistribution::Equal),
        (
            "Memory-Based Distribution",
            WorkloadDistribution::MemoryBased,
        ),
        (
            "Compute-Based Distribution",
            WorkloadDistribution::ComputeBased,
        ),
        ("Adaptive Distribution", WorkloadDistribution::Adaptive),
    ];

    for (name, strategy) in strategies {
        println!("\nTesting {}:", name);

        let config = MultiGPUConfig {
            distribution: strategy,
            max_devices: 4,        // Use up to 4 devices
            min_signal_size: 1024, // Lower threshold for testing
            enable_load_balancing: true,
            ..MultiGPUConfig::default()
        };

        let start = Instant::now();

        match test_strategy(config, &signal, sparsity) {
            Ok((result, processor_info)) => {
                let elapsed = start.elapsed();
                println!("  ✓ Success!");
                println!("  Execution time: {:?}", elapsed);
                println!("  Found {} frequency components", result.indices.len());
                println!("  Device configuration: {}", processor_info);

                // Show performance metrics
                if !processor_info.is_empty() {
                    println!("  Performance per device:");
                    for line in processor_info.lines().skip(1) {
                        if !line.trim().is_empty() {
                            println!("    {}", line.trim());
                        }
                    }
                }
            }
            Err(e) => {
                println!("  ✗ Error: {}", e);
            }
        }
    }

    Ok(())
}

/// Test strategy helper function
#[allow(dead_code)]
fn test_strategy(
    config: MultiGPUConfig,
    signal: &[f64],
    _sparsity: usize,
) -> FFTResult<(scirs2_fft::sparse_fft::SparseFFTResult, String)> {
    let mut processor = MultiGPUSparseFFT::new(config);
    processor.initialize()?;

    let result = processor.sparse_fft(signal)?;

    // Generate device info summary
    let devices = processor.get_selected_devices();
    let mut info = format!("Using {} devices", devices.len());
    for device in devices {
        info.push_str(&format!(
            "\n    {} ({})",
            device.device_name, device.backend as u8
        ));
    }

    Ok((result, info))
}

/// Test performance scaling with different numbers of devices
#[allow(dead_code)]
fn test_performance_scaling() -> FFTResult<()> {
    println!("\n--- Performance Scaling Test ---");

    let signalsizes = vec![2048, 4096, 8192, 16384];
    let max_devices_configs = vec![1, 2, 4];

    for signalsize in signalsizes {
        println!("\nSignal size: {} elements", signalsize);
        let signal = create_test_signal(signalsize);

        for max_devices in &max_devices_configs {
            let config = MultiGPUConfig {
                distribution: WorkloadDistribution::ComputeBased,
                max_devices: *max_devices,
                min_signal_size: 1024,
                enable_load_balancing: true,
                ..MultiGPUConfig::default()
            };

            let start = Instant::now();

            match test_strategy(config, &signal, 10) {
                Ok((result_, _)) => {
                    let elapsed = start.elapsed();
                    let throughput = signalsize as f64 / elapsed.as_secs_f64();

                    println!(
                        "  {} device(s): {:?} ({:.0} samples/sec, {} components)",
                        *max_devices,
                        elapsed,
                        throughput,
                        result_.indices.len()
                    );
                }
                Err(e) => {
                    println!("  {} device(s): Failed ({})", *max_devices, e);
                }
            }
        }
    }

    Ok(())
}

/// Test adaptive load balancing
#[allow(dead_code)]
fn test_adaptive_load_balancing() -> FFTResult<()> {
    println!("\n--- Adaptive Load Balancing Test ---");

    let config = MultiGPUConfig {
        distribution: WorkloadDistribution::Adaptive,
        max_devices: 0, // Use all available devices
        min_signal_size: 1024,
        enable_load_balancing: true,
        ..MultiGPUConfig::default()
    };

    let mut processor = MultiGPUSparseFFT::new(config);
    processor.initialize()?;

    // Run multiple iterations to build performance history
    let iterations = 5;
    let signal = create_test_signal(4096);

    println!(
        "Running {} iterations to build performance history...",
        iterations
    );

    for i in 1..=iterations {
        let start = Instant::now();

        match processor.sparse_fft(&signal) {
            Ok(result) => {
                let elapsed = start.elapsed();
                println!(
                    "  Iteration {}: {:?} ({} components)",
                    i,
                    elapsed,
                    result.indices.len()
                );
            }
            Err(e) => {
                println!("  Iteration {} failed: {}", i, e);
            }
        }
    }

    // Show performance statistics
    let stats = processor.get_performance_stats();
    println!("\nPerformance Statistics:");
    for (device_id, times) in stats {
        if !times.is_empty() {
            let avg_time = times.iter().sum::<f64>() / times.len() as f64;
            let min_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_time = times.iter().fold(0.0f64, |a, &b| a.max(b));

            println!(
                "  Device {}: avg={:.3}s, min={:.3}s, max={:.3}s ({} runs)",
                device_id,
                avg_time,
                min_time,
                max_time,
                times.len()
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
        (120, 0.7), // 120 Hz, amplitude 0.7
        (200, 0.5), // 200 Hz, amplitude 0.5
        (350, 0.3), // 350 Hz, amplitude 0.3
        (500, 0.2), // 500 Hz, amplitude 0.2
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
    let mut rng = StdRng::seed_from_u64(42);

    for sample in &mut signal {
        *sample += 0.1 * (rng.random::<f64>() - 0.5);
    }

    signal
}

/// Display system information for multi-GPU setup
#[allow(dead_code)]
fn display_system_info() {
    println!("\n--- Multi-GPU System Information ---");

    let mut processor = MultiGPUSparseFFT::new(MultiGPUConfig::default());
    if let Ok(()) = processor.initialize() {
        let devices = processor.get_devices();

        let gpu_count = devices
            .iter()
            .filter(|d| d.backend != scirs2_fft::sparse_fft_gpu::GPUBackend::CPUFallback)
            .count();
        let total_memory: usize = devices.iter().map(|d| d.memory_total).sum();
        let total_compute_units: usize = devices.iter().map(|d| d.compute_units).sum();

        println!("GPU Devices: {}", gpu_count);
        println!(
            "Total GPU Memory: {:.1} GB",
            total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("Total Compute Units: {}", total_compute_units);

        println!("\nOptimal Configuration Recommendations:");
        if gpu_count >= 4 {
            println!("  ✓ Excellent multi-GPU setup for large-scale processing");
            println!("  ✓ Recommended: Use ComputeBased or Adaptive distribution");
        } else if gpu_count >= 2 {
            println!("  ✓ Good multi-GPU setup for medium-scale processing");
            println!("  ✓ Recommended: Use ComputeBased distribution");
        } else if gpu_count == 1 {
            println!("  • Single GPU detected - multi-GPU benefits limited");
            println!("  • Consider using single-GPU optimizations instead");
        } else {
            println!("  • No GPU devices detected - using CPU fallback");
            println!("  • Multi-GPU features will not provide benefits");
        }
    } else {
        println!("Failed to initialize multi-GPU system");
    }
}
