//! Auto-tuning for hardware configurations example
//!
//! This example demonstrates how to use the auto-tuning system to automatically
//! detect hardware characteristics and optimize algorithm parameters for best
//! performance on the current system.

use ndarray::{Array1, ArrayView1};
use scirs2_integrate::{
    autotuning::{AlgorithmTuner, AutoTuner, HardwareDetector, TuningProfile},
    memory::{CacheAwareAlgorithms, CacheFriendlyMatrix, MatrixLayout},
    monte_carlo::{monte_carlo, MonteCarloOptions},
};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Auto-Tuning for Hardware Configurations ===\n");

    // Example 1: Hardware detection
    hardware_detection_example()?;

    // Example 2: Algorithm-specific auto-tuning
    algorithm_tuning_example()?;

    // Example 3: Benchmark-based tuning
    benchmark_tuning_example()?;

    // Example 4: Memory-aware optimization
    memory_optimization_example()?;

    Ok(())
}

fn hardware_detection_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🖥️  Hardware Detection");
    println!("{}", "=".repeat(50));

    // Detect hardware characteristics
    let hardware = HardwareDetector::detect();

    println!("Hardware Information:");
    println!("CPU Model: {}", hardware.cpu_model);
    println!("Physical Cores: {}", hardware.cpu_cores);
    println!("Logical Threads: {}", hardware.cpu_threads);
    println!("L1 Cache: {:.0} KB", hardware.l1_cache_size as f64 / 1024.0);
    println!("L2 Cache: {:.0} KB", hardware.l2_cache_size as f64 / 1024.0);
    println!(
        "L3 Cache: {:.0} MB",
        hardware.l3_cache_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "System Memory: {:.1} GB",
        hardware.memory_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    if !hardware.simd_features.is_empty() {
        println!("SIMD Features: {:?}", hardware.simd_features);
    } else {
        println!("SIMD Features: None detected");
    }

    if let Some(bandwidth) = hardware.memory_bandwidth {
        println!(
            "Estimated Memory Bandwidth: {:.1} GB/s",
            bandwidth / (1024.0 * 1024.0 * 1024.0)
        );
    }

    if let Some(ref gpu) = hardware.gpu_info {
        println!(
            "GPU: {} {} ({:.1} GB)",
            gpu.vendor,
            gpu.model,
            gpu.memory_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        println!("GPU: Not detected");
    }

    println!();
    Ok(())
}

fn algorithm_tuning_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚙️  Algorithm-Specific Auto-Tuning");
    println!("{}", "=".repeat(50));

    let hardware = HardwareDetector::detect();
    let tuner = AutoTuner::new(hardware.clone());

    // Test different problem sizes
    let problem_sizes = vec![100, 1000, 10000, 100000];

    println!("Problem Size   Threads   Block Size   Chunk Size   SIMD   Memory Pool");
    println!("{}", "─".repeat(70));

    for &size in &problem_sizes {
        let profile = tuner.tune_for_problem_size(size);

        println!(
            "{:10}      {:3}      {:6}      {:6}     {:3}    {:7} MB",
            size,
            profile.num_threads,
            profile.block_size,
            profile.chunk_size,
            if profile.use_simd { "Yes" } else { "No" },
            profile.memory_pool_size / (1024 * 1024)
        );
    }

    println!("\nAlgorithm-Specific Tuning:");

    // Matrix operations tuning
    let matrix_profile = AlgorithmTuner::tune_matrix_operations(&hardware, 1000);
    println!(
        "Matrix (1000×1000): {} threads, {} block size, SIMD: {}",
        matrix_profile.num_threads, matrix_profile.block_size, matrix_profile.use_simd
    );

    // ODE solver tuning
    let ode_profile = AlgorithmTuner::tune_ode_solver(&hardware, 100, 10000);
    println!(
        "ODE (100 vars, 10k steps): {} threads, tolerance: {:.0e}, max iter: {}",
        ode_profile.num_threads, ode_profile.default_tolerance, ode_profile.max_iterations
    );

    // Monte Carlo tuning
    let mc_profile = AlgorithmTuner::tune_monte_carlo(&hardware, 5, 1000000);
    println!(
        "Monte Carlo (5D, 1M samples): {} threads, {} chunk size, GPU: {}",
        mc_profile.num_threads, mc_profile.chunk_size, mc_profile.use_gpu
    );

    println!();
    Ok(())
}

fn benchmark_tuning_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Benchmark-Based Tuning");
    println!("{}", "=".repeat(50));

    let hardware = HardwareDetector::detect();
    let mut tuner = AutoTuner::new(hardware);

    // Define a simple benchmark function for matrix-vector multiplication
    let benchmark_fn = |_profile: &TuningProfile| -> Duration {
        let size = 1000;
        let matrix = CacheFriendlyMatrix::<f64>::new(size, size, MatrixLayout::RowMajor);
        let vector = Array1::ones(size);

        let start = Instant::now();

        // Simulate some work with the tuning parameters
        for _ in 0..10 {
            let _result = matrix.matvec(vector.view());
        }

        start.elapsed()
    };

    println!("Tuning matrix-vector multiplication for 1000×1000 matrices...");

    let base_profile = tuner.tune_for_problem_size(1000 * 1000);
    let base_time = benchmark_fn(&base_profile);

    println!("Base configuration: {:.2?}", base_time);

    let optimized_profile = tuner.benchmark_tune::<f64>("matvec", benchmark_fn, 1000 * 1000);
    let optimized_time = benchmark_fn(&optimized_profile);

    println!("Optimized configuration: {:.2?}", optimized_time);

    let speedup = base_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
    println!("Speedup: {:.2}x", speedup);

    println!("Optimized parameters:");
    println!("  Threads: {}", optimized_profile.num_threads);
    println!("  Block size: {}", optimized_profile.block_size);
    println!("  Chunk size: {}", optimized_profile.chunk_size);
    println!("  SIMD enabled: {}", optimized_profile.use_simd);

    println!();
    Ok(())
}

fn memory_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Memory-Aware Optimization");
    println!("{}", "=".repeat(50));

    let hardware = HardwareDetector::detect();

    // Demonstrate memory pool optimization
    println!("Memory Pool Optimization:");

    let small_profile = AutoTuner::new(hardware.clone()).tune_for_problem_size(1000);
    let large_profile = AutoTuner::new(hardware.clone()).tune_for_problem_size(1000000);

    println!(
        "Small problem (1K): Pool size = {:.1} MB",
        small_profile.memory_pool_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "Large problem (1M): Pool size = {:.1} MB",
        large_profile.memory_pool_size as f64 / (1024.0 * 1024.0)
    );

    // Demonstrate cache-aware algorithms
    println!("\nCache-Aware Algorithm Optimization:");

    let data_size = 10000;
    let data = Array1::from_iter((0..data_size).map(|i| i as f64));

    // Test different block sizes for reduction
    let cache_sizes = vec![
        ("L1-sized", hardware.l1_cache_size / 8),
        ("L2-sized", hardware.l2_cache_size / 8),
        ("L3-sized", hardware.l3_cache_size / 8),
    ];

    println!("Block Size        Time       Efficiency");
    println!("{}", "─".repeat(40));

    let mut baseline_time = None;

    for (name, block_size) in cache_sizes {
        let start = Instant::now();

        // Perform blocked reduction multiple times
        for _ in 0..100 {
            let _result = CacheAwareAlgorithms::reduction_blocked(data.view(), block_size);
        }

        let elapsed = start.elapsed();

        if baseline_time.is_none() {
            baseline_time = Some(elapsed);
        }

        let efficiency = baseline_time.unwrap().as_nanos() as f64 / elapsed.as_nanos() as f64;

        println!("{:10}   {:8.2?}   {:6.2}x", name, elapsed, efficiency);
    }

    // Demonstrate Monte Carlo integration tuning
    println!("\nMonte Carlo Integration Tuning:");

    let mc_profile = AlgorithmTuner::tune_monte_carlo(&hardware, 3, 100000);

    // Simple 3D integration test function: ∫∫∫ x²+y²+z² dxdydz over [0,1]³
    let integrand = |coords: ArrayView1<f64>| -> f64 {
        coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2]
    };

    let ranges = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];

    let options = MonteCarloOptions {
        n_samples: mc_profile.chunk_size,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let result = monte_carlo(integrand, &ranges, Some(options))?;
    let duration = start.elapsed();

    println!(
        "MC Integration ({}D, {} samples): {:.2?}",
        ranges.len(),
        mc_profile.chunk_size,
        duration
    );
    println!("Result: {:.6} (exact: 1.0)", result.value);
    println!("Error estimate: {:.2e}", result.std_error);

    println!();
    Ok(())
}
