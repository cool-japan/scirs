//! Advanced Memory-Mapped Array Prefetching Example
//!
//! This example demonstrates the advanced prefetching features:
//! 1. Basic prefetching for sequential and strided patterns
//! 2. Adaptive prefetching with reinforcement learning
//! 3. Complex pattern recognition for matrix traversals
//! 4. Resource-aware prefetching that adapts to system load
//! 5. Cross-file prefetching for correlated datasets
//!
//! Run with:
//! ```bash
//! cargo run --example advanced_prefetching_example
//! ```

use ndarray::{Array2, Array3};
use scirs2_core::memory_efficient::{
    AccessType,
    // Advanced prefetching
    AdaptivePatternTracker,
    AdaptivePrefetchConfig,
    AdaptivePrefetchConfigBuilder,
    ComplexPattern,
    CompressedMemMapBuilder,
    // Basic prefetching
    CompressionAlgorithm,
    Confidence,
    CrossFilePrefetchConfig,
    CrossFilePrefetchConfigBuilder,
    CrossFilePrefetchManager,
    DataAccess,
    DatasetId,
    DatasetPrefetcher,
    PatternRecognitionConfig,
    PatternRecognizer,
    PrefetchConfig,
    PrefetchConfigBuilder,
    PrefetchStrategy,
    Prefetching,

    PrefetchingCompressedArray,
    ResourceAwareConfig,
    ResourceAwareConfigBuilder,
    ResourceAwarePrefetcher,
    ResourceMonitor,
};
use std::fs::File;
use std::io::Write;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Advanced Memory-Mapped Array Prefetching Example");
    println!("================================================\n");

    // Create a temporary directory for our example files
    let dir = tempdir()?;
    let matrix_file_path = dir.path().join("matrix.cmm");
    let tensor_file_path = dir.path().join("tensor.cmm");
    let weights_file_path = dir.path().join("weights.cmm");

    // Create test data
    println!("Creating test datasets...");
    let matrix_size = 1000;
    let matrix = Array2::<f64>::from_shape_fn((matrix_size, matrix_size), |(i, j)| {
        (i * matrix_size + j) as f64
    });

    let tensor_size = 200;
    let tensor =
        Array3::<f64>::from_shape_fn((tensor_size, tensor_size, tensor_size), |(i, j, k)| {
            (i * tensor_size * tensor_size + j * tensor_size + k) as f64
        });

    let weights_rows = 1000;
    let weights_cols = 500;
    let weights = Array2::<f64>::from_shape_fn((weights_rows, weights_cols), |(i, j)| {
        ((i + j) % 100) as f64 / 100.0
    });

    // Create compressed memory-mapped arrays
    println!("Creating compressed memory-mapped arrays...");
    let builder = CompressedMemMapBuilder::new()
        .with_block_size(2000)
        .with_algorithm(CompressionAlgorithm::Lz4)
        .with_level(1)
        .with_cache_size(20)
        .with_description("Matrix dataset");

    let matrix_cmm = builder.create(&matrix, &matrix_file_path)?;

    let builder = CompressedMemMapBuilder::new()
        .with_block_size(2000)
        .with_algorithm(CompressionAlgorithm::Lz4)
        .with_level(1)
        .with_cache_size(20)
        .with_description("Tensor dataset");

    let tensor_cmm = builder.create(&tensor, &tensor_file_path)?;

    let builder = CompressedMemMapBuilder::new()
        .with_block_size(1000)
        .with_algorithm(CompressionAlgorithm::Lz4)
        .with_level(1)
        .with_cache_size(10)
        .with_description("Weights dataset");

    let weights_cmm = builder.create(&weights, &weights_file_path)?;

    println!("All datasets created successfully");

    //==========================================================================
    // Part 1: Basic Prefetching vs No Prefetching
    //==========================================================================
    println!("\nPart 1: Basic Prefetching vs No Prefetching");
    println!("------------------------------------------");

    // Without prefetching
    println!("Sequential row-major access without prefetching...");
    let start = Instant::now();

    let mut sum = 0.0;
    for i in 0..matrix_size {
        for j in 0..matrix_size {
            let val = matrix_cmm.get(&[i, j])?;
            sum += val;
        }
    }

    let no_prefetch_time = start.elapsed();
    println!("Sum: {}", sum);
    println!("Time without prefetching: {:?}", no_prefetch_time);

    // With basic prefetching
    println!("\nSequential row-major access with basic prefetching...");

    let basic_config = PrefetchConfigBuilder::new()
        .enabled(true)
        .prefetch_count(5)
        .min_pattern_length(3)
        .async_prefetch(true)
        .prefetch_timeout(Duration::from_millis(50))
        .build();

    let prefetching_cmm = matrix_cmm.clone().with_prefetching_config(basic_config)?;

    let start = Instant::now();

    let mut sum = 0.0;
    for i in 0..matrix_size {
        for j in 0..matrix_size {
            let val = prefetching_cmm.get(&[i, j])?;
            sum += val;
        }
    }

    let basic_prefetch_time = start.elapsed();
    println!("Sum: {}", sum);
    println!("Time with basic prefetching: {:?}", basic_prefetch_time);
    println!(
        "Improvement: {:.2}x faster",
        no_prefetch_time.as_secs_f64() / basic_prefetch_time.as_secs_f64()
    );

    // Print basic prefetching statistics
    let stats = prefetching_cmm.prefetch_stats()?;
    println!("\nBasic Prefetching Statistics:");
    println!("Total prefetch operations: {}", stats.prefetch_count);
    println!("Prefetch hits: {}", stats.prefetch_hits);
    println!("Prefetch misses: {}", stats.prefetch_misses);
    println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);

    //==========================================================================
    // Part 2: Adaptive Prefetching with Reinforcement Learning
    //==========================================================================
    println!("\nPart 2: Adaptive Prefetching with Reinforcement Learning");
    println!("------------------------------------------------------");

    // Create adaptive prefetching configuration
    let adaptive_config = AdaptivePrefetchingConfig::new()
        .with_learning_rate(0.1)
        .with_exploration_rate(0.2)
        .with_discount_factor(0.9)
        .with_reward_for_hit(1.0)
        .with_penalty_for_miss(-0.2)
        .with_max_prefetch_count(10)
        .with_performance_metrics_window(100);

    // Convert to adaptive prefetching array
    let adaptive_cmm =
        AdaptivePrefetchingCompressedArray::from_array(matrix_cmm.clone(), adaptive_config)?;

    println!("Training adaptive prefetching with sequential pattern...");

    // Run sequential access to train the adaptive prefetcher
    for _ in 0..3 {
        let mut sum = 0.0;
        for i in 0..matrix_size {
            for j in 0..matrix_size {
                let val = adaptive_cmm.get(&[i, j])?;
                sum += val;
            }
        }
    }

    println!("Training complete. Testing performance...");
    let start = Instant::now();

    let mut sum = 0.0;
    for i in 0..matrix_size {
        for j in 0..matrix_size {
            let val = adaptive_cmm.get(&[i, j])?;
            sum += val;
        }
    }

    let adaptive_time = start.elapsed();
    println!("Sum: {}", sum);
    println!("Time with adaptive prefetching: {:?}", adaptive_time);
    println!(
        "Improvement over basic: {:.2}x faster",
        basic_prefetch_time.as_secs_f64() / adaptive_time.as_secs_f64()
    );

    // Print adaptive prefetching statistics
    let stats = adaptive_cmm.prefetch_stats()?;
    println!("\nAdaptive Prefetching Statistics:");
    println!("Total prefetch operations: {}", stats.prefetch_count);
    println!("Prefetch hits: {}", stats.prefetch_hits);
    println!("Prefetch misses: {}", stats.prefetch_misses);
    println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
    println!("Learning iterations: {}", stats.learning_iterations);
    println!("Current prefetch strategy: {}", stats.current_strategy);

    //==========================================================================
    // Part 3: Complex Pattern Recognition
    //==========================================================================
    println!("\nPart 3: Complex Pattern Recognition");
    println!("----------------------------------");

    // Create pattern recognition configuration
    let pattern_config = PatternRecognitionConfig::new()
        .detect_matrix_traversals(true)
        .detect_stencil_operations(true)
        .detect_blockwise_operations(true)
        .pattern_history_size(200)
        .confidence_threshold(0.7);

    // Create detector with matrix pattern recognition
    let mut pattern_detector = PatternDetector::new(pattern_config);

    // Let's do some different access patterns to demonstrate detection
    println!("Testing different matrix access patterns...");

    // 1. Row-major pattern
    println!("\nDetecting row-major pattern...");
    pattern_detector.reset();
    for i in 0..100 {
        for j in 0..100 {
            pattern_detector.record_access(&[i, j]);
        }
    }
    let pattern = pattern_detector.detect_pattern();
    println!("Detected pattern: {:?}", pattern);

    // 2. Column-major pattern
    println!("\nDetecting column-major pattern...");
    pattern_detector.reset();
    for j in 0..100 {
        for i in 0..100 {
            pattern_detector.record_access(&[i, j]);
        }
    }
    let pattern = pattern_detector.detect_pattern();
    println!("Detected pattern: {:?}", pattern);

    // 3. Diagonal pattern
    println!("\nDetecting diagonal pattern...");
    pattern_detector.reset();
    let size = 100;
    for d in 0..size {
        for i in 0..=d {
            let j = d - i;
            if j < size {
                pattern_detector.record_access(&[i, j]);
            }
        }
    }
    let pattern = pattern_detector.detect_pattern();
    println!("Detected pattern: {:?}", pattern);

    // 4. Block pattern
    println!("\nDetecting block pattern...");
    pattern_detector.reset();
    let block_size = 10;
    for block_i in 0..10 {
        for block_j in 0..10 {
            for i in 0..block_size {
                for j in 0..block_size {
                    let row = block_i * block_size + i;
                    let col = block_j * block_size + j;
                    pattern_detector.record_access(&[row, col]);
                }
            }
        }
    }
    let pattern = pattern_detector.detect_pattern();
    println!("Detected pattern: {:?}", pattern);

    //==========================================================================
    // Part 4: Resource-Aware Prefetching
    //==========================================================================
    println!("\nPart 4: Resource-Aware Prefetching");
    println!("--------------------------------");

    // Create resource thresholds
    let thresholds = ResourceThresholds::new()
        .with_max_cpu_usage(0.8)       // 80% CPU usage
        .with_max_memory_usage(0.7)    // 70% memory usage
        .with_max_io_operations(1000); // 1000 IO ops per second

    // Create resource monitor
    let resource_monitor = ResourceMonitor::new(
        Duration::from_millis(500), // Check resources every 500ms
        thresholds,
    );

    // Create resource-aware prefetching array
    let resource_aware_cmm = ResourceAwarePrefetchingArray::new(
        matrix_cmm.clone(),
        resource_monitor,
        PrefetchConfigBuilder::new()
            .enabled(true)
            .prefetch_count(8)
            .min_pattern_length(3)
            .async_prefetch(true)
            .prefetch_timeout(Duration::from_millis(50))
            .build(),
    )?;

    println!("Testing resource-aware prefetching...");

    // Create a background CPU load (to demonstrate resource awareness)
    let cpu_load_thread = thread::spawn(|| {
        println!("Starting background CPU load...");
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(5) {
            // Busy loop to create CPU load
            let mut x = 0.0;
            for i in 0..10_000_000 {
                x += (i as f64).sin();
            }
            // Prevent compiler from optimizing away the loop
            if x < 0.0 {
                println!("This should never print: {}", x);
            }
        }
        println!("Background CPU load finished");
    });

    // Give the CPU load thread time to start
    thread::sleep(Duration::from_millis(100));

    let start = Instant::now();

    let mut sum = 0.0;
    for i in 0..matrix_size {
        for j in 0..matrix_size {
            let val = resource_aware_cmm.get(&[i, j])?;
            sum += val;
        }
    }

    let resource_aware_time = start.elapsed();
    println!("Sum: {}", sum);
    println!(
        "Time with resource-aware prefetching: {:?}",
        resource_aware_time
    );

    // Print resource statistics
    let stats = resource_aware_cmm.prefetch_stats()?;
    println!("\nResource-Aware Prefetching Statistics:");
    println!("Total prefetch operations: {}", stats.prefetch_count);
    println!("Prefetch hits: {}", stats.prefetch_hits);
    println!("Prefetch misses: {}", stats.prefetch_misses);
    println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
    println!("Prefetching throttled: {} times", stats.throttle_count);
    println!("Average CPU usage: {:.2}%", stats.avg_cpu_usage * 100.0);
    println!(
        "Average memory usage: {:.2}%",
        stats.avg_memory_usage * 100.0
    );

    // Wait for the CPU load thread to finish
    if let Err(e) = cpu_load_thread.join() {
        eprintln!("Error joining CPU load thread: {:?}", e);
    }

    //==========================================================================
    // Part 5: Cross-File Prefetching
    //==========================================================================
    println!("\nPart 5: Cross-File Prefetching");
    println!("----------------------------");

    // Create dataset correlation tracker
    let mut dataset_manager = CrossFilePrefetchManager::new();

    // Add our datasets
    dataset_manager.register_dataset("matrix", matrix_cmm.clone())?;
    dataset_manager.register_dataset("tensor", tensor_cmm.clone())?;
    dataset_manager.register_dataset("weights", weights_cmm.clone())?;

    // Create correlations between datasets
    dataset_manager.add_correlation(
        "matrix",
        "weights",
        DatasetCorrelation::with_index_mapping(|matrix_indices| {
            // When accessing matrix[i, j], also prefetch weights[i, j/2]
            if matrix_indices.len() == 2 {
                let i = matrix_indices[0];
                let j = matrix_indices[1];
                if j < weights_cols * 2 {
                    return Some(vec![i, j / 2]);
                }
            }
            None
        }),
    );

    dataset_manager.add_correlation(
        "matrix",
        "tensor",
        DatasetCorrelation::with_index_mapping(|matrix_indices| {
            // When accessing matrix[i, j], also prefetch tensor[i/5, j/5, 0]
            if matrix_indices.len() == 2 {
                let i = matrix_indices[0];
                let j = matrix_indices[1];
                if i < tensor_size * 5 && j < tensor_size * 5 {
                    return Some(vec![i / 5, j / 5, 0]);
                }
            }
            None
        }),
    );

    println!("Testing cross-file prefetching...");
    println!("Accessing matrix dataset with correlated access to weights and tensor...");

    // First, access without cross-file prefetching to establish baseline
    let start = Instant::now();

    let mut matrix_sum = 0.0;
    let mut weights_sum = 0.0;
    let mut tensor_sum = 0.0;

    for i in 0..500 {
        for j in 0..500 {
            // Access matrix
            let matrix_val = matrix_cmm.get(&[i, j])?;
            matrix_sum += matrix_val;

            // Access correlated weights
            if j / 2 < weights_cols {
                let weights_val = weights_cmm.get(&[i, j / 2])?;
                weights_sum += weights_val;
            }

            // Access correlated tensor
            if i / 5 < tensor_size && j / 5 < tensor_size {
                let tensor_val = tensor_cmm.get(&[i / 5, j / 5, 0])?;
                tensor_sum += tensor_val;
            }
        }
    }

    let normal_time = start.elapsed();
    println!("Time without cross-file prefetching: {:?}", normal_time);
    println!(
        "Sums - Matrix: {}, Weights: {}, Tensor: {}",
        matrix_sum, weights_sum, tensor_sum
    );

    // Now with cross-file prefetching
    println!("\nNow with cross-file prefetching...");
    let start = Instant::now();

    let mut matrix_sum = 0.0;
    let mut weights_sum = 0.0;
    let mut tensor_sum = 0.0;

    for i in 0..500 {
        for j in 0..500 {
            // Access matrix - this will trigger prefetching in other datasets
            let dataset_key = "matrix";
            let indices = vec![i, j];
            let matrix_val = dataset_manager.get(dataset_key, &indices)?;
            matrix_sum += matrix_val;

            // Access correlated weights - should have better hit rate
            if j / 2 < weights_cols {
                let weights_val = dataset_manager.get("weights", &[i, j / 2])?;
                weights_sum += weights_val;
            }

            // Access correlated tensor - should have better hit rate
            if i / 5 < tensor_size && j / 5 < tensor_size {
                let tensor_val = dataset_manager.get("tensor", &[i / 5, j / 5, 0])?;
                tensor_sum += tensor_val;
            }
        }
    }

    let cross_file_time = start.elapsed();
    println!("Time with cross-file prefetching: {:?}", cross_file_time);
    println!(
        "Improvement: {:.2}x faster",
        normal_time.as_secs_f64() / cross_file_time.as_secs_f64()
    );
    println!(
        "Sums - Matrix: {}, Weights: {}, Tensor: {}",
        matrix_sum, weights_sum, tensor_sum
    );

    // Print cross-file prefetching statistics
    let stats = dataset_manager.statistics();
    println!("\nCross-File Prefetching Statistics:");
    println!(
        "Total cross-file prefetch operations: {}",
        stats.total_prefetch_operations
    );
    println!("Cross-file prefetch hits: {}", stats.total_hits);
    println!("Cross-file prefetch misses: {}", stats.total_misses);
    println!(
        "Overall hit rate: {:.2}%",
        if stats.total_hits + stats.total_misses > 0 {
            100.0 * stats.total_hits as f64 / (stats.total_hits + stats.total_misses) as f64
        } else {
            0.0
        }
    );

    for (dataset_name, dataset_stats) in &stats.dataset_stats {
        println!("\nDataset '{}' Statistics:", dataset_name);
        println!("  Prefetch operations: {}", dataset_stats.prefetch_count);
        println!("  Hits: {}", dataset_stats.hits);
        println!("  Misses: {}", dataset_stats.misses);
        println!(
            "  Hit rate: {:.2}%",
            if dataset_stats.hits + dataset_stats.misses > 0 {
                100.0 * dataset_stats.hits as f64
                    / (dataset_stats.hits + dataset_stats.misses) as f64
            } else {
                0.0
            }
        );
    }

    println!("\nAdvanced prefetching example completed successfully!");
    Ok(())
}
