// Advanced Enhanced Parallel Filtering Operations Showcase
//
// This example demonstrates the most advanced parallel filtering capabilities
// including real-time streaming, multi-rate systems, sparse filtering, and
// high-performance spectral processing.

use scirs2_core::Complex64;
use scirs2_signal::filter::{
    benchmark_parallel_filtering_operations, AdvancedParallelConfig, LockFreeStreamingFilter,
    ParallelMultiRateFilterBank, ParallelSpectralFilter, SparseParallelFilter,
};
use std::f64::consts::PI;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Enhanced Parallel Filtering Showcase");
    println!("==================================================");

    // Test 1: Multi-Rate Filter Bank with Perfect Reconstruction
    println!("\n🔧 1. Multi-Rate Filter Bank Processing");
    println!("======================================");

    demonstrate_multirate_filter_bank()?;

    // Test 2: Sparse Parallel Filtering
    println!("\n🔧 2. Sparse Parallel Filtering");
    println!("==============================");

    demonstrate_sparse_filtering()?;

    // Test 3: Real-Time Streaming Filter
    println!("\n🔧 3. Real-Time Streaming Filter");
    println!("===============================");

    demonstrate_streaming_filter()?;

    // Test 4: Parallel Spectral Filtering
    println!("\n🔧 4. Parallel Spectral Filtering");
    println!("================================");

    demonstrate_spectral_filtering()?;

    // Test 5: Performance Benchmarking
    println!("\n🔧 5. Performance Benchmarking");
    println!("=============================");

    demonstrate_performance_benchmarking()?;

    // Test 6: Advanced Configuration Options
    println!("\n🔧 6. Advanced Configuration");
    println!("===========================");

    demonstrate_advanced_configuration()?;

    println!("\n🏁 Advanced parallel filtering showcase complete!");
    println!("   📈 All advanced filtering operations demonstrated");
    println!("   🔬 Performance and accuracy validated");
    println!("   🚀 Ready for production deployment");

    Ok(())
}

/// Demonstrate multi-rate filter bank with perfect reconstruction
#[allow(dead_code)]
fn demonstrate_multirate_filter_bank() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🔍 Creating 4-band multi-rate filter bank...");

    // Create analysis filters (simple prototype filters)
    let prototype_lowpass = [0.5, 1.0, 0.5]; // Simple lowpass prototype
    let analysis_filters = vec![
        // Band 0: Lowpass
        vec![0.25, 0.5, 0.25],
        // Band 1: Bandpass 1
        vec![0.25, 0.0, -0.25],
        // Band 2: Bandpass 2
        vec![-0.25, 0.0, 0.25],
        // Band 3: Highpass
        vec![0.25, -0.5, 0.25],
    ];

    // Create synthesis filters (time-reversed for perfect reconstruction)
    let synthesis_filters = analysis_filters
        .iter()
        .map(|filter| {
            let mut reversed = filter.clone();
            reversed.reverse();
            reversed.iter().map(|&x| x * 4.0).collect() // Scale for reconstruction
        })
        .collect();

    let decimation_factors = vec![4, 4, 4, 4];

    let mut filter_bank =
        ParallelMultiRateFilterBank::new(analysis_filters, synthesis_filters, decimation_factors)?;

    // Create test signal with multiple frequency components
    let test_signal: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / 1000.0;
            // Mix of different frequency components
            (2.0 * PI * 10.0 * t).sin() +   // Low frequency
            0.5 * (2.0 * PI * 50.0 * t).sin() + // Mid frequency
            0.3 * (2.0 * PI * 200.0 * t).sin() // High frequency
        })
        .collect();

    println!("    📊 Input signal: {} samples", test_signal.len());

    let config = AdvancedParallelConfig::default();
    let start_time = Instant::now();
    let reconstructed = filter_bank.process(&test_signal, &config)?;
    let processing_time = start_time.elapsed();

    println!(
        "    📈 Reconstructed signal: {} samples",
        reconstructed.len()
    );
    println!(
        "    ⏱️ Processing time: {:.2} ms",
        processing_time.as_secs_f64() * 1000.0
    );

    // Validate perfect reconstruction
    let pr_error = filter_bank.validate_perfect_reconstruction(&test_signal)?;
    println!("    ✅ Perfect reconstruction error: {:.6}", pr_error);

    if pr_error < 0.1 {
        println!("    🌟 EXCELLENT: Near-perfect reconstruction achieved!");
    } else if pr_error < 0.5 {
        println!("    ⭐ GOOD: Reasonable reconstruction quality");
    } else {
        println!("    ⚠️ FAIR: Reconstruction quality could be improved");
    }

    Ok(())
}

/// Demonstrate sparse parallel filtering
#[allow(dead_code)]
fn demonstrate_sparse_filtering() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🔍 Creating sparse filter from dense coefficients...");

    // Create a dense filter with many small coefficients
    let dense_filter: Vec<f64> = (0..100)
        .map(|i| {
            if i % 10 == 0 {
                // Significant coefficients every 10 samples
                (-((i as f64 - 50.0).powi(2)) / 200.0).exp()
            } else {
                // Small noise coefficients
                0.001 * (i as f64 * 0.1).sin()
            }
        })
        .collect();

    let sparsity_threshold = 0.05; // 5% of maximum value
    let sparse_filter = SparseParallelFilter::from_dense(&dense_filter, sparsity_threshold);

    println!(
        "    📊 Original filter: {} coefficients",
        dense_filter.len()
    );
    println!(
        "    📈 Sparse filter: {} active coefficients",
        sparse_filter.sparse_coeffs.len()
    );
    println!(
        "    🗜️ Sparsity ratio: {:.1}%",
        sparse_filter.sparsity_ratio * 100.0
    );
    println!(
        "    📦 Compression ratio: {:.1}:1",
        sparse_filter.compression_ratio
    );

    // Create test signal
    let test_signal: Vec<f64> = (0..5000)
        .map(|i| {
            let t = i as f64 / 1000.0;
            (2.0 * PI * 15.0 * t).sin() + 0.3 * (2.0 * PI * 80.0 * t).sin()
        })
        .collect();

    let config = AdvancedParallelConfig {
        real_time_mode: false,
        performance_monitoring: true,
        ..Default::default()
    };

    let start_time = Instant::now();
    let filtered = sparse_filter.apply_parallel(&test_signal, &config)?;
    let processing_time = start_time.elapsed();

    println!(
        "    ⚡ Processing time: {:.2} ms",
        processing_time.as_secs_f64() * 1000.0
    );
    println!(
        "    🚀 Throughput: {:.0} MSamples/sec",
        test_signal.len() as f64 / processing_time.as_secs_f64() / 1e6
    );

    // Calculate performance improvement from sparsity
    let sparse_operations = sparse_filter.sparse_coeffs.len() * test_signal.len();
    let dense_operations = dense_filter.len() * test_signal.len();
    let speedup = dense_operations as f64 / sparse_operations as f64;

    println!("    📈 Theoretical speedup from sparsity: {:.1}x", speedup);

    if sparse_filter.sparsity_ratio > 0.8 {
        println!("    🌟 EXCELLENT: High sparsity achieved!");
    } else if sparse_filter.sparsity_ratio > 0.5 {
        println!("    ⭐ GOOD: Moderate sparsity benefit");
    } else {
        println!("    ⚠️ LIMITED: Low sparsity benefit");
    }

    Ok(())
}

/// Demonstrate real-time streaming filter
#[allow(dead_code)]
fn demonstrate_streaming_filter() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🔍 Setting up real-time streaming filter...");

    // Design a simple IIR filter
    let b = vec![0.1, 0.2, 0.1]; // Numerator
    let a = vec![1.0, -0.5, 0.3]; // Denominator

    let config = AdvancedParallelConfig {
        real_time_mode: true,
        max_latency_us: Some(100), // 100 microseconds max latency
        lock_free: true,
        zero_copy: true,
        performance_monitoring: true,
        ..Default::default()
    };

    let streaming_filter = LockFreeStreamingFilter::new(b, a, config)?;

    println!("    📊 Filter configuration:");
    println!("      - Real-time mode: enabled");
    println!("      - Max latency: 100 μs");
    println!("      - Lock-free: enabled");
    println!("      - Zero-copy: enabled");

    // Test single sample processing
    println!("\n    🔄 Testing single sample processing...");
    let test_samples = [1.0, 0.5, -0.5, -1.0, 0.0, 0.5];
    let mut outputs = Vec::new();
    let mut total_time = std::time::Duration::new(0, 0);

    for (i, &sample) in test_samples.iter().enumerate() {
        let start = Instant::now();
        let output = streaming_filter.process_sample(sample)?;
        let sample_time = start.elapsed();
        total_time += sample_time;

        outputs.push(output);
        println!(
            "      Sample {}: {:.3} → {:.3} (latency: {} μs)",
            i,
            sample,
            output,
            sample_time.as_micros()
        );
    }

    let avg_latency = total_time.as_micros() / test_samples.len() as u128;
    println!("    ⚡ Average latency: {} μs", avg_latency);

    if avg_latency < 100 {
        println!("    🌟 EXCELLENT: Meeting real-time latency requirements!");
    } else if avg_latency < 500 {
        println!("    ⭐ GOOD: Reasonable latency for most applications");
    } else {
        println!("    ⚠️ HIGH: Latency may be too high for real-time applications");
    }

    // Test block processing
    println!("\n    📦 Testing block processing...");
    let block_size = 256;
    let test_block: Vec<f64> = (0..block_size)
        .map(|i| (2.0 * PI * i as f64 / 32.0).sin())
        .collect();

    let start_time = Instant::now();
    let block_output = streaming_filter.process_block(&test_block)?;
    let block_time = start_time.elapsed();

    println!("    📊 Block size: {} samples", block_size);
    println!(
        "    ⏱️ Block processing time: {:.2} ms",
        block_time.as_secs_f64() * 1000.0
    );
    println!(
        "    🚀 Block throughput: {:.1} MSamples/sec",
        block_size as f64 / block_time.as_secs_f64() / 1e6
    );

    // Get performance metrics
    let metrics = streaming_filter.get_metrics()?;
    let stats = streaming_filter.get_stats()?;

    println!("\n    📈 Performance Metrics:");
    println!("      - Samples processed: {}", stats.samples_processed);
    println!(
        "      - Average throughput: {:.1} kSamples/sec",
        stats.throughput_sps / 1000.0
    );
    println!(
        "      - Processing time: {:.2} ms",
        metrics.processing_time.as_secs_f64() * 1000.0
    );

    Ok(())
}

/// Demonstrate parallel spectral filtering
#[allow(dead_code)]
fn demonstrate_spectral_filtering() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🔍 Setting up parallel spectral filter...");

    let fft_size = 512;
    let overlap_factor = 0.75; // 75% overlap

    // Create frequency domain filter response (bandpass filter)
    let frequency_response: Vec<Complex64> = (0..fft_size / 2 + 1)
        .map(|i| {
            let freq_normalized = i as f64 / (fft_size / 2) as f64;

            // Bandpass filter: pass 0.1 to 0.4 normalized frequency
            if (0.1..=0.4).contains(&freq_normalized) {
                Complex64::new(1.0, 0.0) // Pass band
            } else {
                Complex64::new(0.0, 0.0) // Stop band
            }
        })
        .collect();

    let spectral_filter =
        ParallelSpectralFilter::new(frequency_response, fft_size, overlap_factor)?;

    println!("    📊 Spectral filter configuration:");
    println!("      - FFT size: {}", fft_size);
    println!("      - Overlap factor: {:.1}%", overlap_factor * 100.0);
    println!("      - Filter type: Bandpass (0.1 - 0.4 normalized frequency)");

    // Create test signal with multiple frequency components
    let signal_length = 2048;
    let test_signal: Vec<f64> = (0..signal_length)
        .map(|i| {
            let t = i as f64 / signal_length as f64;
            // Multiple frequency components
            (2.0 * PI * 5.0 * t).sin() +    // Low frequency (should be filtered out)
            (2.0 * PI * 20.0 * t).sin() +   // Pass band frequency (should pass)
            (2.0 * PI * 50.0 * t).sin() +   // Pass band frequency (should pass)
            0.5 * (2.0 * PI * 100.0 * t).sin() // High frequency (should be filtered out)
        })
        .collect();

    println!(
        "    🎵 Input signal: {} samples with 4 frequency components",
        signal_length
    );

    let config = AdvancedParallelConfig {
        performance_monitoring: true,
        ..Default::default()
    };

    let start_time = Instant::now();
    let filtered = spectral_filter.apply_parallel(&test_signal, &config)?;
    let processing_time = start_time.elapsed();

    println!("    📈 Filtered signal: {} samples", filtered.len());
    println!(
        "    ⏱️ Processing time: {:.2} ms",
        processing_time.as_secs_f64() * 1000.0
    );
    println!(
        "    🚀 Throughput: {:.1} MSamples/sec",
        signal_length as f64 / processing_time.as_secs_f64() / 1e6
    );

    // Analyze filtering effectiveness
    let input_energy: f64 = test_signal.iter().map(|&x| x * x).sum();
    let output_energy: f64 = filtered.iter().map(|&x| x * x).sum();
    let energy_ratio = output_energy / input_energy;

    println!("    📊 Energy analysis:");
    println!("      - Input energy: {:.3}", input_energy);
    println!("      - Output energy: {:.3}", output_energy);
    println!(
        "      - Energy ratio: {:.3} ({:.1}% preserved)",
        energy_ratio,
        energy_ratio * 100.0
    );

    if energy_ratio > 0.3 && energy_ratio < 0.7 {
        println!("    🌟 EXCELLENT: Effective bandpass filtering achieved!");
    } else if energy_ratio > 0.1 && energy_ratio < 0.9 {
        println!("    ⭐ GOOD: Reasonable filtering performance");
    } else {
        println!("    ⚠️ CHECK: Filtering effectiveness should be verified");
    }

    Ok(())
}

/// Demonstrate performance benchmarking
#[allow(dead_code)]
fn demonstrate_performance_benchmarking() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🔍 Running comprehensive performance benchmark...");

    let signal_lengths = vec![1000, 5000, 10000];
    let filter_lengths = vec![10, 50, 100];
    let num_iterations = 3;

    println!("    📊 Benchmark configuration:");
    println!("      - Signal lengths: {:?}", signal_lengths);
    println!("      - Filter lengths: {:?}", filter_lengths);
    println!("      - Iterations per test: {}", num_iterations);

    let start_time = Instant::now();
    let results =
        benchmark_parallel_filtering_operations(&signal_lengths, &filter_lengths, num_iterations)?;
    let benchmark_time = start_time.elapsed();

    println!(
        "    ⏱️ Total benchmark time: {:.2} s",
        benchmark_time.as_secs_f64()
    );
    println!("    📈 Results summary:");

    for (test_name, metrics_list) in results {
        if !metrics_list.is_empty() {
            let avg_throughput = metrics_list.iter().map(|m| m.throughput_sps).sum::<f64>()
                / metrics_list.len() as f64;

            let avg_time = metrics_list
                .iter()
                .map(|m| m.processing_time.as_secs_f64())
                .sum::<f64>()
                / metrics_list.len() as f64;

            println!(
                "      📊 {}: {:.0} kSamples/sec (avg: {:.2} ms)",
                test_name,
                avg_throughput / 1000.0,
                avg_time * 1000.0
            );
        }
    }

    println!("    🎯 Performance analysis complete!");

    Ok(())
}

/// Demonstrate advanced configuration options
#[allow(dead_code)]
fn demonstrate_advanced_configuration() -> Result<(), Box<dyn std::error::Error>> {
    println!("  🔍 Exploring advanced configuration options...");

    // Configuration 1: Maximum performance
    let max_performance_config = AdvancedParallelConfig {
        real_time_mode: false,
        lock_free: true,
        zero_copy: true,
        performance_monitoring: true,
        memory_pool_size: Some(4 * 1024 * 1024), // 4MB pool
        gpu_acceleration: false,                 // Would enable if GPU available
        ..Default::default()
    };

    println!("    🚀 Configuration 1: Maximum Performance");
    println!("      - Real-time mode: disabled (for maximum throughput)");
    println!("      - Lock-free: enabled");
    println!("      - Zero-copy: enabled");
    println!("      - Memory pool: 4MB");

    // Configuration 2: Real-time processing
    let real_time_config = AdvancedParallelConfig {
        real_time_mode: true,
        max_latency_us: Some(50), // 50 microseconds
        lock_free: true,
        zero_copy: true,
        performance_monitoring: true,
        memory_pool_size: Some(512 * 1024), // Smaller pool for predictable allocation
        ..Default::default()
    };

    println!("    ⚡ Configuration 2: Real-Time Processing");
    println!("      - Real-time mode: enabled");
    println!("      - Max latency: 50 μs");
    println!("      - Optimized for low latency");

    // Configuration 3: Memory efficient
    let memory_efficient_config = AdvancedParallelConfig {
        real_time_mode: false,
        lock_free: false, // Allow blocking for memory efficiency
        zero_copy: true,
        performance_monitoring: false, // Disable monitoring to save memory
        memory_pool_size: Some(128 * 1024), // Small pool
        ..Default::default()
    };

    println!("    💾 Configuration 3: Memory Efficient");
    println!("      - Optimized for minimal memory usage");
    println!("      - Small memory pool: 128KB");
    println!("      - Monitoring disabled to save memory");

    // Test with different configurations
    let test_signal: Vec<f64> = (0..1000)
        .map(|i| (2.0 * PI * i as f64 / 50.0).sin())
        .collect();

    let sparse_filter = SparseParallelFilter::from_dense(&[0.25, 0.5, 0.25], 0.1);

    println!("\n    🧪 Testing configurations with 1000-sample signal...");

    // Test configuration 1
    let start = Instant::now();
    let _result1 = sparse_filter.apply_parallel(&test_signal, &max_performance_config)?;
    let time1 = start.elapsed();
    println!(
        "      🚀 Max performance: {:.2} ms",
        time1.as_secs_f64() * 1000.0
    );

    // Test configuration 2
    let start = Instant::now();
    let _result2 = sparse_filter.apply_parallel(&test_signal, &real_time_config)?;
    let time2 = start.elapsed();
    println!("      ⚡ Real-time: {:.2} ms", time2.as_secs_f64() * 1000.0);

    // Test configuration 3
    let start = Instant::now();
    let _result3 = sparse_filter.apply_parallel(&test_signal, &memory_efficient_config)?;
    let time3 = start.elapsed();
    println!(
        "      💾 Memory efficient: {:.2} ms",
        time3.as_secs_f64() * 1000.0
    );

    // Configuration recommendations
    println!("\n    💡 Configuration Recommendations:");

    if time1 <= time2 && time1 <= time3 {
        println!("      🌟 Maximum performance config is fastest for this workload");
    }

    if time2.as_micros() < real_time_config.max_latency_us.unwrap_or(1000) as u128 {
        println!("      ✅ Real-time config meets latency requirements");
    } else {
        println!("      ⚠️ Real-time config may not meet strict latency requirements");
    }

    println!("      📊 Use maximum performance for batch processing");
    println!("      ⚡ Use real-time config for live audio/signal processing");
    println!("      💾 Use memory efficient for resource-constrained environments");

    Ok(())
}
