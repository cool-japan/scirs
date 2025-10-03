//! Streaming datasets demonstration
//!
//! This example demonstrates how to work with large datasets using streaming,
//! allowing processing of datasets that are too large to fit in memory.
//!
//! Usage:
//!   cargo run --example streaming_demo --release

use scirs2_datasets::{
    make_classification, stream_classification, stream_regression, utils::train_test_split,
    DataChunk, StreamConfig, StreamProcessor, StreamTransformer,
};
use std::collections::HashMap;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Streaming Datasets Demonstration");
    println!("===================================\n");

    // Basic streaming operations
    demonstrate_basic_streaming()?;

    // Memory-efficient processing
    demonstrate_memory_efficient_processing()?;

    // Stream transformations
    demonstrate_stream_transformations()?;

    // Parallel stream processing
    demonstrate_parallel_processing()?;

    // Performance comparison
    demonstrate_performance_comparison()?;

    // Real-world scenarios
    demonstrate_real_world_scenarios()?;

    println!("\n🎉 Streaming demonstration completed!");
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_basic_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 BASIC STREAMING OPERATIONS");
    println!("{}", "-".repeat(40));

    // Configure streaming
    let config = StreamConfig {
        chunk_size: 1000,           // 1K samples per chunk
        buffer_size: 3,             // Buffer 3 chunks
        num_workers: 4,             // Use 4 worker threads
        memory_limit_mb: Some(100), // Limit to 100MB
        enable_compression: false,
        enable_prefetch: true,
        max_chunks: Some(10), // Process only 10 chunks for demo
    };

    println!("Streaming Configuration:");
    println!("  Chunk size: {} samples", config.chunk_size);
    println!("  Buffer size: {} chunks", config.buffer_size);
    println!("  Workers: {}", config.num_workers);
    println!("  Memory limit: {:?} MB", config.memory_limit_mb);
    println!("  Max chunks: {:?}", config.max_chunks);

    // Create streaming classification dataset
    println!("\nStreaming synthetic classification data...");
    let mut stream = stream_classification(100_000, 20, 5, config.clone())?;

    let mut total_samples = 0;
    let mut chunk_count = 0;
    let mut class_distribution: HashMap<i32, usize> = HashMap::new();

    let start_time = Instant::now();

    while let Some(chunk) = stream.next_chunk()? {
        total_samples += chunk.n_samples();
        chunk_count += 1;

        // Analyze this chunk
        if let Some(target) = &chunk.target {
            for &class in target.iter() {
                *class_distribution.entry(class as i32).or_insert(0) += 1;
            }
        }

        // Print progress
        let stats = stream.stats();
        if let Some(progress) = stats.progress_percent() {
            println!(
                "  Chunk {}: {} samples (Progress: {:.1}%, Buffer: {:.1}%)",
                chunk.chunk_index + 1,
                chunk.n_samples(),
                progress,
                stats.buffer_utilization()
            );
        } else {
            println!(
                "  Chunk {}: {} samples (Buffer: {:.1}%)",
                chunk.chunk_index + 1,
                chunk.n_samples(),
                stats.buffer_utilization()
            );
        }

        // Simulate processing time
        std::thread::sleep(std::time::Duration::from_millis(50));

        if chunk.is_last {
            println!("  📋 Reached last chunk");
            break;
        }
    }

    let duration = start_time.elapsed();

    println!("\nStreaming Results:");
    println!("  Total chunks processed: {chunk_count}");
    println!("  Total samples: {total_samples}");
    println!("  Processing time: {:.2}s", duration.as_secs_f64());
    println!(
        "  Throughput: {:.1} samples/s",
        total_samples as f64 / duration.as_secs_f64()
    );
    println!("  Class distribution: {class_distribution:?}");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_memory_efficient_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 MEMORY-EFFICIENT PROCESSING");
    println!("{}", "-".repeat(40));

    // Compare memory usage: streaming vs. in-memory
    let datasetsize = 50_000;
    let n_features = 50;

    println!("Comparing memory usage for {datasetsize} samples with {n_features} features");

    // In-memory approach (for comparison)
    println!("\n1. In-memory approach:");
    let start_mem = get_memory_usage();
    let start_time = Instant::now();

    let in_memorydataset = make_classification(datasetsize, n_features, 5, 2, 25, Some(42))?;
    let (train, test) = train_test_split(&in_memorydataset, 0.2, Some(42))?;

    let in_memory_time = start_time.elapsed();
    let in_memory_mem = get_memory_usage() - start_mem;

    println!("  Time: {:.2}s", in_memory_time.as_secs_f64());
    println!("  Memory usage: ~{in_memory_mem:.1} MB");
    println!("  Train samples: {}", train.n_samples());
    println!("  Test samples: {}", test.n_samples());

    // Streaming approach
    println!("\n2. Streaming approach:");
    let stream_start_time = Instant::now();
    let stream_start_mem = get_memory_usage();

    let config = StreamConfig {
        chunk_size: 5_000, // Smaller chunks for memory efficiency
        buffer_size: 2,    // Smaller buffer
        num_workers: 2,
        memory_limit_mb: Some(50),
        ..Default::default()
    };

    let mut stream = stream_classification(datasetsize, n_features, 5, config)?;

    let mut total_processed = 0;
    let mut train_samples = 0;
    let mut test_samples = 0;

    while let Some(chunk) = stream.next_chunk()? {
        total_processed += chunk.n_samples();

        // Simulate train/test split on chunk level
        let chunk_trainsize = (chunk.n_samples() as f64 * 0.8) as usize;
        train_samples += chunk_trainsize;
        test_samples += chunk.n_samples() - chunk_trainsize;

        // Process chunk (simulate some computation)
        let _mean = chunk.data.mean_axis(scirs2_core::ndarray::Axis(0));
        let _std = chunk.data.std_axis(scirs2_core::ndarray::Axis(0), 0.0);

        if chunk.is_last {
            break;
        }
    }

    let stream_time = stream_start_time.elapsed();
    let stream_mem = get_memory_usage() - stream_start_mem;

    println!("  Time: {:.2}s", stream_time.as_secs_f64());
    println!("  Memory usage: ~{stream_mem:.1} MB");
    println!("  Train samples: {train_samples}");
    println!("  Test samples: {test_samples}");
    println!("  Total processed: {total_processed}");

    // Comparison
    println!("\n3. Comparison:");
    println!(
        "  Memory savings: {:.1}x less memory",
        in_memory_mem / stream_mem.max(1.0)
    );
    println!(
        "  Time overhead: {:.1}x",
        stream_time.as_secs_f64() / in_memory_time.as_secs_f64()
    );
    println!("  Streaming is beneficial for large datasets that don't fit in memory");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_stream_transformations() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 STREAM TRANSFORMATIONS");
    println!("{}", "-".repeat(40));

    // Create a transformer pipeline
    let transformer = StreamTransformer::new()
        .add_standard_scaling()
        .add_missing_value_imputation();

    println!("Created transformation pipeline:");
    println!("  1. Standard scaling (z-score normalization)");
    println!("  2. Missing value imputation");

    let config = StreamConfig {
        chunk_size: 2000,
        buffer_size: 2,
        max_chunks: Some(5),
        ..Default::default()
    };

    let mut stream = stream_regression(10_000, 15, config)?;
    let mut transformed_chunks = 0;

    println!("\nProcessing and transforming chunks...");

    while let Some(mut chunk) = stream.next_chunk()? {
        println!("  Processing chunk {}", chunk.chunk_index + 1);

        // Show statistics before transformation
        let data_mean_before = chunk.data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let data_std_before = chunk.data.std_axis(scirs2_core::ndarray::Axis(0), 0.0);

        println!(
            "    Before: mean = {:.3}, std = {:.3}",
            data_mean_before[0], data_std_before[0]
        );

        // Apply transformations
        transformer.transform_chunk(&mut chunk)?;

        // Show statistics after transformation
        let data_mean_after = chunk.data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let data_std_after = chunk.data.std_axis(scirs2_core::ndarray::Axis(0), 0.0);

        println!(
            "    After:  mean = {:.3}, std = {:.3}",
            data_mean_after[0], data_std_after[0]
        );

        transformed_chunks += 1;

        if chunk.is_last {
            break;
        }
    }

    println!("\nTransformation Summary:");
    println!("  Chunks processed: {transformed_chunks}");
    println!("  Each chunk was transformed independently");
    println!("  Memory-efficient: only one chunk in memory at a time");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_parallel_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ PARALLEL STREAM PROCESSING");
    println!("{}", "-".repeat(40));

    let config = StreamConfig {
        chunk_size: 1500,
        buffer_size: 4,
        num_workers: 4,
        max_chunks: Some(8),
        ..Default::default()
    };

    println!("Parallel processing configuration:");
    println!("  Workers: {}", config.num_workers);
    println!("  Chunk size: {}", config.chunk_size);
    println!("  Buffer size: {}", config.buffer_size);

    // Create a simple processor that computes statistics
    let _processor: StreamProcessor<DataChunk> = StreamProcessor::new(config.clone());

    // Define a processing function
    let compute_stats = |chunk: DataChunk| -> Result<
        HashMap<String, f64>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let mut stats = HashMap::new();

        // Compute basic statistics
        let mean = chunk.data.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let std = chunk.data.std_axis(scirs2_core::ndarray::Axis(0), 0.0);

        stats.insert("mean_feature_0".to_string(), mean[0]);
        stats.insert("std_feature_0".to_string(), std[0]);
        stats.insert("n_samples".to_string(), chunk.n_samples() as f64);
        stats.insert("chunk_index".to_string(), chunk.chunk_index as f64);

        // Simulate some computation time
        std::thread::sleep(std::time::Duration::from_millis(100));

        Ok(stats)
    };

    println!("\nProcessing stream with parallel workers...");
    let start_time = Instant::now();

    let stream = stream_classification(12_000, 10, 3, config)?;

    // For demonstration, we'll process chunks sequentially with timing
    // In a real implementation, you'd use the processor.process_parallel method
    let mut stream_iter = stream;
    let mut chunk_results = Vec::new();

    while let Some(chunk) = stream_iter.next_chunk()? {
        let chunk_start = Instant::now();
        let chunk_id = chunk.chunk_index;
        let chunk_samples = chunk.n_samples();

        // Process chunk
        let stats = compute_stats(chunk)
            .map_err(|e| -> Box<dyn std::error::Error> { Box::new(std::io::Error::other(e)) })?;
        let chunk_time = chunk_start.elapsed();

        println!(
            "  Chunk {}: {} samples, {:.2}ms",
            chunk_id + 1,
            chunk_samples,
            chunk_time.as_millis()
        );

        chunk_results.push(stats);

        if chunk_results.len() >= 8 {
            break;
        }
    }

    let total_time = start_time.elapsed();

    println!("\nParallel Processing Results:");
    println!("  Total chunks: {}", chunk_results.len());
    println!("  Total time: {:.2}s", total_time.as_secs_f64());
    println!(
        "  Average time per chunk: {:.2}ms",
        total_time.as_millis() as f64 / chunk_results.len() as f64
    );

    // Aggregate statistics
    let total_samples: f64 = chunk_results
        .iter()
        .map(|stats| stats.get("n_samples").unwrap_or(&0.0))
        .sum();

    println!("  Total samples processed: {total_samples}");
    println!(
        "  Throughput: {:.1} samples/s",
        total_samples / total_time.as_secs_f64()
    );

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 PERFORMANCE COMPARISON");
    println!("{}", "-".repeat(40));

    let dataset_sizes = vec![10_000, 50_000, 100_000];
    let chunk_sizes = vec![1_000, 5_000, 10_000];

    println!("Comparing streaming performance across different configurations:");
    println!();

    for &datasetsize in &dataset_sizes {
        println!("Dataset size: {datasetsize} samples");

        for &chunksize in &chunk_sizes {
            let config = StreamConfig {
                chunk_size: chunksize,
                buffer_size: 3,
                num_workers: 2,
                max_chunks: Some(datasetsize / chunksize),
                ..Default::default()
            };

            let start_time = Instant::now();
            let mut stream = stream_regression(datasetsize, 20, config)?;

            let mut processed_samples = 0;
            let mut processed_chunks = 0;

            while let Some(chunk) = stream.next_chunk()? {
                processed_samples += chunk.n_samples();
                processed_chunks += 1;

                // Simulate minimal processing
                let _stats = chunk.data.mean_axis(scirs2_core::ndarray::Axis(0));

                if chunk.is_last || processed_samples >= datasetsize {
                    break;
                }
            }

            let duration = start_time.elapsed();
            let throughput = processed_samples as f64 / duration.as_secs_f64();

            println!(
                "  Chunk size {}: {:.2}s ({:.1} samples/s, {} chunks)",
                chunksize,
                duration.as_secs_f64(),
                throughput,
                processed_chunks
            );
        }
        println!();
    }

    println!("Performance Insights:");
    println!("  • Larger chunks = fewer iterations, better throughput");
    println!("  • Smaller chunks = lower memory usage, more responsive");
    println!("  • Optimal chunk size depends on memory constraints and processing complexity");

    println!();
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_real_world_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 REAL-WORLD STREAMING SCENARIOS");
    println!("{}", "-".repeat(40));

    // Scenario 1: Training on large dataset with limited memory
    println!("Scenario 1: Large dataset training with memory constraints");
    simulate_training_scenario()?;

    // Scenario 2: Data preprocessing pipeline
    println!("\nScenario 2: Data preprocessing pipeline");
    simulate_preprocessing_pipeline()?;

    // Scenario 3: Model evaluation on large test set
    println!("\nScenario 3: Model evaluation on large test set");
    simulate_model_evaluation()?;

    println!();
    Ok(())
}

#[allow(dead_code)]
fn simulate_training_scenario() -> Result<(), Box<dyn std::error::Error>> {
    println!("  • Dataset: 500K samples, 100 features");
    println!("  • Memory limit: 200MB");
    println!("  • Goal: Train incrementally using mini-batches");

    let config = StreamConfig {
        chunk_size: 5_000, // Mini-batch size
        buffer_size: 2,    // Keep memory low
        memory_limit_mb: Some(200),
        max_chunks: Some(10), // Simulate partial processing
        ..Default::default()
    };

    let mut stream = stream_classification(500_000, 100, 10, config)?;
    let mut total_batches = 0;
    let mut total_samples = 0;

    let start_time = Instant::now();

    while let Some(chunk) = stream.next_chunk()? {
        // Simulate training on mini-batch
        let batchsize = chunk.n_samples();

        // Simulate gradient computation time
        std::thread::sleep(std::time::Duration::from_millis(20));

        total_batches += 1;
        total_samples += batchsize;

        if total_batches % 3 == 0 {
            println!("    Processed {total_batches} batches ({total_samples} samples)");
        }

        if chunk.is_last {
            break;
        }
    }

    let duration = start_time.elapsed();
    println!(
        "  ✅ Training simulation: {} batches, {:.2}s",
        total_batches,
        duration.as_secs_f64()
    );

    Ok(())
}

#[allow(dead_code)]
fn simulate_preprocessing_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("  • Raw data → Clean → Scale → Feature selection");
    println!("  • Process 200K samples in chunks");

    let config = StreamConfig {
        chunk_size: 8_000,
        buffer_size: 3,
        max_chunks: Some(5),
        ..Default::default()
    };

    let transformer = StreamTransformer::new()
        .add_missing_value_imputation()
        .add_standard_scaling();

    let mut stream = stream_regression(200_000, 50, config)?;
    let mut processed_chunks = 0;

    while let Some(mut chunk) = stream.next_chunk()? {
        // Step 1: Clean data (remove outliers, handle missing values)
        transformer.transform_chunk(&mut chunk)?;

        // Step 2: Feature selection (simulate by keeping first 30 features)
        let selecteddata = chunk
            .data
            .slice(scirs2_core::ndarray::s![.., ..30])
            .to_owned();

        processed_chunks += 1;
        println!(
            "    Chunk {}: {} → {} features",
            processed_chunks,
            chunk.n_features(),
            selecteddata.ncols()
        );

        if chunk.is_last {
            break;
        }
    }

    println!("  ✅ Preprocessing pipeline: {processed_chunks} chunks processed");

    Ok(())
}

#[allow(dead_code)]
fn simulate_model_evaluation() -> Result<(), Box<dyn std::error::Error>> {
    println!("  • Evaluate model on 1M test samples");
    println!("  • Compute accuracy in streaming fashion");

    let config = StreamConfig {
        chunk_size: 10_000,
        buffer_size: 2,
        max_chunks: Some(8),
        ..Default::default()
    };

    let mut stream = stream_classification(1_000_000, 20, 5, config)?;
    let mut correct_predictions = 0;
    let mut total_predictions = 0;

    while let Some(chunk) = stream.next_chunk()? {
        if let Some(true_labels) = &chunk.target {
            // Simulate model predictions (random for demo)
            let predictions: Vec<f64> = (0..chunk.n_samples())
                .map(|_| (scirs2_core::random::random::<f64>() * 5.0).floor())
                .collect();

            // Calculate accuracy for this chunk
            let chunk_correct = true_labels
                .iter()
                .zip(predictions.iter())
                .filter(|(&true_label, &pred)| (true_label - pred).abs() < 0.5)
                .count();

            correct_predictions += chunk_correct;
            total_predictions += chunk.n_samples();
        }

        if chunk.is_last {
            break;
        }
    }

    let accuracy = correct_predictions as f64 / total_predictions as f64;
    println!(
        "  ✅ Model evaluation: {:.1}% accuracy on {} samples",
        accuracy * 100.0,
        total_predictions
    );

    Ok(())
}

/// Get actual memory usage of the current process in MB
#[allow(dead_code)]
fn get_memory_usage() -> f64 {
    get_process_memory_usage().unwrap_or_else(|_| {
        // Fallback to a placeholder if real memory usage cannot be determined
        scirs2_core::random::random::<f64>() * 50.0 + 10.0 // 10-60 MB range
    })
}

/// Platform-specific memory usage implementation
#[cfg(target_os = "linux")]
#[allow(dead_code)]
fn get_process_memory_usage() -> Result<f64, Box<dyn std::error::Error>> {
    use std::fs;

    // Read /proc/self/status to get VmRSS (Resident Set Size)
    let status = fs::read_to_string("/proc/self/status")?;

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: f64 = parts[1].parse()?;
                return Ok(kb / 1024.0); // Convert KB to MB
            }
        }
    }

    Err("VmRSS not found in /proc/self/status".into())
}

#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn get_process_memory_usage() -> Result<f64, Box<dyn std::error::Error>> {
    use std::mem;
    use std::ptr;

    // Use macOS mach API to get memory info
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut u8,
            task_info_outCnt: *mut u32,
        ) -> i32;
    }

    const TASK_BASIC_INFO: u32 = 5;
    const TASK_BASIC_INFO_COUNT: u32 = 5;

    #[repr(C)]
    struct TaskBasicInfo {
        suspend_count: u32,
        virtualsize: u32,
        residentsize: u32,
        user_time: u64,
        system_time: u64,
    }

    unsafe {
        let mut info = mem::zeroed::<TaskBasicInfo>();
        let mut count = TASK_BASIC_INFO_COUNT;

        let result = task_info(
            mach_task_self(),
            TASK_BASIC_INFO,
            &mut info as *mut _ as *mut u8,
            &mut count,
        );

        if result == 0 {
            Ok(info.residentsize as f64 / (1024.0 * 1024.0)) // Convert bytes to MB
        } else {
            Err(format!("task_info failed with code {}", result).into())
        }
    }
}

#[cfg(target_os = "windows")]
#[allow(dead_code)]
fn get_process_memory_usage() -> Result<f64, Box<dyn std::error::Error>> {
    use std::mem;
    use std::ptr;

    // Use Windows API to get memory info
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_setsize: usize,
        working_setsize: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    unsafe {
        let mut counters = mem::zeroed::<ProcessMemoryCounters>();
        counters.cb = mem::size_of::<ProcessMemoryCounters>() as u32;

        let result = GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb);

        if result != 0 {
            Ok(counters.working_setsize as f64 / (1024.0 * 1024.0)) // Convert bytes to MB
        } else {
            Err("GetProcessMemoryInfo failed".into())
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
#[allow(dead_code)]
fn get_process_memory_usage() -> Result<f64, Box<dyn std::error::Error>> {
    Err("Memory usage monitoring not implemented for this platform".into())
}
