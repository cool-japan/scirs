//! Memory-efficient neural network training example
//!
//! This example demonstrates various memory optimization techniques including:
//! - Memory pool for tensor reuse
//! - Gradient checkpointing for reduced memory usage
//! - In-place operations to minimize allocations
//! - Memory-aware batch processing
//! - Memory usage tracking and monitoring

use ndarray::{Array2, Array3};
use scirs2_neural::error::Result;
use scirs2_neural::memory_efficient::{
    BatchProcessorStats, GradientCheckpointing, InPlaceOperations, MemoryAwareBatchProcessor,
    MemoryEfficientLayer, MemoryPool, MemoryUsage, PoolStatistics,
};
use std::time::Instant;

fn main() -> Result<()> {
    println!("Memory-Efficient Neural Network Operations Demo");
    println!("===============================================");

    // Demo 1: Memory Pool Usage
    demo_memory_pool()?;

    // Demo 2: Gradient Checkpointing
    demo_gradient_checkpointing()?;

    // Demo 3: In-place Operations
    demo_in_place_operations()?;

    // Demo 4: Memory-aware Batch Processing
    demo_memory_aware_batch_processing()?;

    // Demo 5: Memory-efficient Layer
    demo_memory_efficient_layer()?;

    // Demo 6: Memory Usage Tracking
    demo_memory_usage_tracking()?;

    Ok(())
}

fn demo_memory_pool() -> Result<()> {
    println!("\n🔄 Memory Pool Demo");
    println!("------------------");

    let mut pool = MemoryPool::<f32>::new(50); // 50MB max pool size

    // Allocate several tensors
    println!("Allocating tensors...");
    let tensor1 = pool.allocate(&[1000, 500]); // ~2MB
    let tensor2 = pool.allocate(&[500, 200]); // ~400KB
    let tensor3 = pool.allocate(&[100, 100]); // ~40KB

    let stats = pool.get_pool_stats();
    println!("Pool stats after allocation:");
    print_pool_stats(&stats);

    // Return tensors to pool
    println!("Returning tensors to pool...");
    pool.deallocate(tensor1);
    pool.deallocate(tensor2);
    pool.deallocate(tensor3);

    let stats = pool.get_pool_stats();
    println!("Pool stats after deallocation:");
    print_pool_stats(&stats);

    // Reuse tensors
    println!("Reusing tensors (should be faster)...");
    let start = Instant::now();
    let _reused1 = pool.allocate(&[1000, 500]);
    let _reused2 = pool.allocate(&[500, 200]);
    let reuse_time = start.elapsed();
    println!("Reuse time: {:?}", reuse_time);

    let stats = pool.get_pool_stats();
    println!("Final pool stats:");
    print_pool_stats(&stats);

    Ok(())
}

fn demo_gradient_checkpointing() -> Result<()> {
    println!("\n📊 Gradient Checkpointing Demo");
    println!("------------------------------");

    let mut checkpointing = GradientCheckpointing::<f64>::new(100.0); // 100MB threshold

    // Set up checkpoint layers
    checkpointing.add_checkpoint_layer("conv1".to_string());
    checkpointing.add_checkpoint_layer("conv3".to_string());
    checkpointing.add_checkpoint_layer("fc1".to_string());

    println!("Storing activations at checkpoints...");

    // Simulate storing activations during forward pass
    let conv1_activation = Array3::from_elem((32, 64, 64), 0.5).into_dyn(); // Batch=32, 64x64 feature maps
    let conv3_activation = Array3::from_elem((32, 128, 32), 0.3).into_dyn(); // Reduced spatial size
    let fc1_activation = Array2::from_elem((32, 512), 0.2).into_dyn(); // Fully connected

    checkpointing.store_checkpoint("conv1", conv1_activation)?;
    checkpointing.store_checkpoint("conv3", conv3_activation)?;
    checkpointing.store_checkpoint("fc1", fc1_activation)?;

    let usage = checkpointing.get_memory_usage();
    println!("Memory usage after checkpointing:");
    print_memory_usage(&usage);

    // Simulate retrieving checkpoints during backward pass
    println!("Retrieving checkpoints for gradient computation...");
    if let Some(checkpoint) = checkpointing.get_checkpoint("conv1") {
        println!("Retrieved conv1 checkpoint: shape {:?}", checkpoint.shape());
    }

    // Clear checkpoints to free memory
    println!("Clearing checkpoints...");
    checkpointing.clear_checkpoints();

    let usage = checkpointing.get_memory_usage();
    println!("Memory usage after clearing:");
    print_memory_usage(&usage);

    Ok(())
}

fn demo_in_place_operations() -> Result<()> {
    println!("\n⚡ In-place Operations Demo");
    println!("--------------------------");

    // Create test arrays
    let mut relu_test = Array2::from_shape_vec(
        (3, 4),
        vec![
            -1.0, 2.0, -3.0, 4.0, 0.5, -0.5, 1.5, -2.5, 3.0, -1.0, 0.0, 2.0,
        ],
    )?
    .into_dyn();

    let mut sigmoid_test =
        Array2::from_shape_vec((2, 3), vec![-2.0, 0.0, 2.0, -1.0, 1.0, 3.0])?.into_dyn();

    let mut add_test = Array2::from_elem((2, 2), 1.0).into_dyn();
    let add_source = Array2::from_elem((2, 2), 0.5).into_dyn();

    let mut norm_test =
        Array2::from_shape_vec((2, 3), vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0])?.into_dyn();

    println!("Before operations:");
    println!("ReLU input (should clip negatives): {:?}", relu_test);
    println!("Sigmoid input: {:?}", sigmoid_test);
    println!("Addition target: {:?}", add_test);
    println!("Normalization input: {:?}", norm_test);

    // Apply in-place operations
    println!("\nApplying in-place operations...");
    InPlaceOperations::relu_inplace(&mut relu_test);
    InPlaceOperations::sigmoid_inplace(&mut sigmoid_test);
    InPlaceOperations::add_inplace(&mut add_test, &add_source)?;
    InPlaceOperations::normalize_inplace(&mut norm_test)?;

    println!("\nAfter operations:");
    println!("ReLU result: {:?}", relu_test);
    println!("Sigmoid result: {:?}", sigmoid_test);
    println!("Addition result: {:?}", add_test);
    println!("Normalized result: {:?}", norm_test);

    // Test scaling
    let mut scale_test = Array2::from_elem((2, 2), 2.0).into_dyn();
    println!("\nScaling test - before: {:?}", scale_test);
    InPlaceOperations::scale_inplace(&mut scale_test, 3.0);
    println!("Scaling test - after: {:?}", scale_test);

    Ok(())
}

fn demo_memory_aware_batch_processing() -> Result<()> {
    println!("\n🔀 Memory-Aware Batch Processing Demo");
    println!("------------------------------------");

    let mut processor = MemoryAwareBatchProcessor::<f32>::new(
        200,   // 200MB max memory
        150.0, // 150MB threshold
        50,    // 50MB pool size
    );

    // Create a large dataset that needs to be processed in chunks
    println!("Creating large dataset (1000 samples x 784 features)...");
    let large_dataset = Array2::from_shape_fn((1000, 784), |(i, j)| {
        (i as f32 * 0.01 + j as f32 * 0.001).sin()
    })
    .into_dyn();

    println!("Dataset shape: {:?}", large_dataset.shape());
    println!(
        "Estimated memory: {:.2} MB",
        (large_dataset.len() * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0)
    );

    // Process in memory-aware batches
    println!("Processing with automatic batch size adjustment...");
    let start = Instant::now();

    let results = processor.process_batches(&large_dataset, |batch| {
        // Simulate some processing (e.g., forward pass through a layer)
        let processed = batch.mapv(|x| x.tanh()); // Apply activation
        Ok(processed.to_owned())
    })?;

    let processing_time = start.elapsed();

    println!("Processing completed in {:?}", processing_time);
    println!("Number of result batches: {}", results.len());

    // Print statistics
    let stats = processor.get_stats();
    println!("Batch processor statistics:");
    print_batch_processor_stats(&stats);

    Ok(())
}

fn demo_memory_efficient_layer() -> Result<()> {
    println!("\n🧠 Memory-Efficient Layer Demo");
    println!("------------------------------");

    // Create a memory-efficient layer
    let layer = MemoryEfficientLayer::new(
        784,      // Input size (e.g., 28x28 MNIST)
        128,      // Output size
        Some(64), // Chunk size
    )?;

    println!("Created memory-efficient layer: 784 -> 128");

    // Create input data
    let input =
        Array2::from_shape_fn((256, 784), |(i, j)| ((i + j) as f32 * 0.01).sin()).into_dyn();

    println!("Input shape: {:?}", input.shape());

    // Forward pass
    println!("Performing forward pass...");
    let start = Instant::now();
    let output = layer.forward(&input)?;
    let forward_time = start.elapsed();

    println!("Forward pass completed in {:?}", forward_time);
    println!("Output shape: {:?}", output.shape());

    // Verify output statistics
    let mean = output.mean().unwrap_or(0.0);
    let std = {
        let variance = output.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        variance.sqrt()
    };

    println!("Output statistics:");
    println!("  Mean: {:.6}", mean);
    println!("  Std: {:.6}", std);
    println!(
        "  Min: {:.6}",
        output.iter().cloned().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Max: {:.6}",
        output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    Ok(())
}

fn demo_memory_usage_tracking() -> Result<()> {
    println!("\n📈 Memory Usage Tracking Demo");
    println!("-----------------------------");

    let mut usage = MemoryUsage::new();

    println!("Initial state:");
    print_memory_usage(&usage);

    // Simulate various allocation patterns
    println!("\nSimulating allocation patterns...");

    // Large allocation
    usage.allocate(50 * 1024 * 1024); // 50MB
    println!("After 50MB allocation:");
    print_memory_usage(&usage);

    // Multiple small allocations
    for i in 1..=10 {
        usage.allocate(1024 * 1024); // 1MB each
        if i % 3 == 0 {
            println!("After {} small allocations:", i);
            print_memory_usage(&usage);
        }
    }

    // Peak usage reached
    println!("Peak memory usage reached:");
    print_memory_usage(&usage);

    // Simulate deallocations
    println!("\nSimulating deallocations...");
    for i in 1..=8 {
        usage.deallocate(5 * 1024 * 1024); // 5MB each
        if i % 2 == 0 {
            println!("After {} deallocations:", i);
            print_memory_usage(&usage);
        }
    }

    println!("\nFinal state (note peak is preserved):");
    print_memory_usage(&usage);

    Ok(())
}

// Helper functions for pretty printing

fn print_pool_stats(stats: &PoolStatistics) {
    println!("  Cached tensors: {}", stats.total_cached_tensors);
    println!("  Unique shapes: {}", stats.unique_shapes);
    println!(
        "  Pool size: {:.2}/{:.2} MB",
        stats.current_pool_size_mb, stats.max_pool_size_mb
    );
}

fn print_memory_usage(usage: &MemoryUsage) {
    println!("  Current: {:.2} MB", usage.current_mb());
    println!("  Peak: {:.2} MB", usage.peak_mb());
    println!("  Active allocations: {}", usage.active_allocations);
    println!("  Total allocations: {}", usage.total_allocations);
}

fn print_batch_processor_stats(stats: &BatchProcessorStats) {
    println!("  Max batch size: {}", stats.max_batch_size);
    println!("  Current memory: {:.2} MB", stats.current_memory_mb);
    println!("  Peak memory: {:.2} MB", stats.peak_memory_mb);
    println!("  Memory threshold: {:.2} MB", stats.memory_threshold_mb);
    println!("  Pool stats:");
    print_pool_stats(&stats.pool_stats);
}
