//! Example demonstrating smart prefetching for memory-mapped arrays.
//!
//! This example compares the performance of accessing elements in a compressed
//! memory-mapped array with and without smart prefetching.
//!
//! Run with:
//! ```bash
//! cargo run --example memory_mapped_prefetching
//! ```

use ndarray::Array2;
use scirs2_core::memory_efficient::{
    CompressedMemMapBuilder, CompressionAlgorithm, PrefetchConfig, PrefetchConfigBuilder,
    Prefetching,
};
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Mapped Array Smart Prefetching Example");
    println!("=============================================\n");

    // Create a temporary directory for our example files
    let dir = tempdir()?;
    let file_path = dir.path().join("test_compressed.cmm");

    // Create test data - 1000x1000 matrix (8MB)
    println!("Creating test data...");
    let rows = 1000;
    let cols = 1000;
    let data = Array2::<f64>::from_shape_fn((rows, cols), |(i, j)| (i * cols + j) as f64);

    // Create compressed memory-mapped array
    println!("Creating compressed memory-mapped array...");
    let builder = CompressedMemMapBuilder::new()
        .with_block_size(1000) // 1000 elements per block
        .with_algorithm(CompressionAlgorithm::Lz4)
        .with_level(1)
        .with_cache_size(10) // 10 blocks maximum in cache
        .with_description("Test matrix for prefetching example");

    let cmm = builder.create(&data, &file_path)?;
    println!("Compressed memory-mapped array created successfully");

    // Part 1: Access without prefetching
    println!("\nPart 1: Sequential access without prefetching");
    let start = Instant::now();

    // Access elements sequentially
    let mut sum = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let val = cmm.get(&[i, j])?;
            sum += val;
        }
    }

    let elapsed = start.elapsed();
    println!("Sum: {}", sum);
    println!("Time without prefetching: {:?}", elapsed);

    // Part 2: Access with prefetching
    println!("\nPart 2: Sequential access with prefetching");

    // Create prefetching configuration
    let config = PrefetchConfigBuilder::new()
        .enabled(true)
        .prefetch_count(5) // Prefetch 5 blocks ahead
        .min_pattern_length(3) // Detect patterns after 3 accesses
        .async_prefetch(true) // Use background thread for prefetching
        .prefetch_timeout(Duration::from_millis(50))
        .build();

    // Convert to prefetching array
    let prefetching_cmm = cmm.clone().with_prefetching_config(config)?;

    // Measure performance with prefetching
    let start = Instant::now();

    // Access elements sequentially (same pattern as before)
    let mut sum = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let val = prefetching_cmm.get(&[i, j])?;
            sum += val;
        }
    }

    let elapsed = start.elapsed();
    println!("Sum: {}", sum);
    println!("Time with prefetching: {:?}", elapsed);

    // Print prefetching statistics
    let stats = prefetching_cmm.prefetch_stats()?;
    println!("\nPrefetching Statistics:");
    println!("Total prefetch operations: {}", stats.prefetch_count);
    println!("Prefetch hits: {}", stats.prefetch_hits);
    println!("Prefetch misses: {}", stats.prefetch_misses);
    println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);

    // Part 3: Strided access pattern
    println!("\nPart 3: Strided access pattern");

    // Create new prefetching configuration with longer history
    let config = PrefetchConfigBuilder::new()
        .enabled(true)
        .prefetch_count(5)
        .history_size(50) // Larger history to better detect strides
        .min_pattern_length(5)
        .async_prefetch(true)
        .prefetch_timeout(Duration::from_millis(50))
        .build();

    // Convert to prefetching array with new config
    let mut prefetching_cmm = cmm.clone().with_prefetching_config(config)?;

    // Clear any previous prefetch state
    prefetching_cmm.clear_prefetch_state()?;

    println!("Accessing with stride 10 pattern...");
    let start = Instant::now();

    // Access elements with stride of 10
    let mut sum = 0.0;
    let stride = 10;

    for i in (0..rows).step_by(stride) {
        for j in (0..cols).step_by(stride) {
            let val = prefetching_cmm.get(&[i, j])?;
            sum += val;
        }
    }

    let elapsed = start.elapsed();
    println!("Sum of strided elements: {}", sum);
    println!("Time for strided access with prefetching: {:?}", elapsed);

    // Print prefetching statistics
    let stats = prefetching_cmm.prefetch_stats()?;
    println!("\nPrefetching Statistics for Strided Access:");
    println!("Total prefetch operations: {}", stats.prefetch_count);
    println!("Prefetch hits: {}", stats.prefetch_hits);
    println!("Prefetch misses: {}", stats.prefetch_misses);
    println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);

    // Part 4: Random access pattern
    println!("\nPart 4: Random access pattern");

    // Create new prefetching configuration
    let config = PrefetchConfigBuilder::new()
        .enabled(true)
        .prefetch_count(10) // Prefetch more blocks for random access
        .history_size(100)  // Larger history to try to detect any patterns
        .min_pattern_length(3)
        .async_prefetch(true)
        .prefetch_timeout(Duration::from_millis(50))
        .build();

    // Convert to prefetching array with new config
    let mut prefetching_cmm = cmm.clone().with_prefetching_config(config)?;

    // Clear any previous prefetch state
    prefetching_cmm.clear_prefetch_state()?;

    println!("Accessing with random pattern...");
    let start = Instant::now();

    // Use a simple pseudo-random pattern
    let mut sum = 0.0;
    let mut x = 123456789;
    let mut y = 362436069;
    let mut z = 521288629;

    for _ in 0..10000 {
        // Simple xorshift random number generation
        let t = x ^ (x << 11);
        x = y;
        y = z;
        z = z ^ (z >> 19) ^ (t ^ (t >> 8));

        // Map to matrix indices
        let i = (z % rows as u32) as usize;
        let j = (x % cols as u32) as usize;

        let val = prefetching_cmm.get(&[i, j])?;
        sum += val;
    }

    let elapsed = start.elapsed();
    println!("Sum of random elements: {}", sum);
    println!("Time for random access with prefetching: {:?}", elapsed);

    // Print prefetching statistics
    let stats = prefetching_cmm.prefetch_stats()?;
    println!("\nPrefetching Statistics for Random Access:");
    println!("Total prefetch operations: {}", stats.prefetch_count);
    println!("Prefetch hits: {}", stats.prefetch_hits);
    println!("Prefetch misses: {}", stats.prefetch_misses);
    println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);

    println!("\nExample completed successfully!");
    Ok(())
}
