//! Example demonstrating chunked processing of memory-mapped arrays.
//!
//! This example shows how to:
//! 1. Create memory-mapped arrays
//! 2. Process them in memory-efficient chunks
//! 3. Use iterators for convenient chunk-wise operations
//!
//! Note: This example requires the `memory_efficient` feature to be enabled.
//! Run with: `cargo run --example memory_mapped_chunks --features memory_efficient`

#[cfg(feature = "memory_efficient")]
use ndarray::Array1;
#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{
    create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunkIter, MemoryMappedChunks,
};
#[cfg(feature = "memory_efficient")]
use std::path::Path;
#[cfg(feature = "memory_efficient")]
use std::time::Instant;
#[cfg(feature = "memory_efficient")]
use tempfile::tempdir;

#[cfg(not(feature = "memory_efficient"))]
fn main() {
    println!("This example requires the memory_efficient feature.");
    println!("Run with: cargo run --example memory_mapped_chunks --features memory_efficient");
}

#[cfg(feature = "memory_efficient")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Mapped Chunked Processing Example");
    println!("========================================\n");

    // Create a temporary directory for our example files
    let temp_dir = tempdir()?;
    println!("Using temporary directory: {:?}", temp_dir.path());

    // Basic chunking example
    basic_chunk_example(temp_dir.path())?;

    // Demonstrate chunk-wise aggregation
    aggregate_example(temp_dir.path())?;

    // Performance comparison with and without chunking
    performance_comparison(temp_dir.path())?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

/// Basic example of chunked processing
#[cfg(feature = "memory_efficient")]
fn basic_chunk_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Basic Chunk Processing Example");
    println!("--------------------------------");

    // Create a large 1D array
    let size = 1_000_000;
    println!("Creating a 1D array with {} elements", size);
    let data = Array1::<f64>::linspace(0., (size - 1) as f64, size);

    // Create a memory-mapped file
    let file_path = temp_dir.join("chunk_example.bin");
    let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0)?;
    println!("Created memory-mapped array at: {:?}", file_path);

    // Process in chunks of 100,000 elements
    let chunk_size = 100_000;
    let strategy = ChunkingStrategy::Fixed(chunk_size);
    let num_chunks = mmap.chunk_count(strategy);

    println!(
        "Processing array in {} chunks of {} elements each",
        num_chunks, chunk_size
    );

    // Calculate sum using process_chunks
    let chunk_sums = mmap.process_chunks(strategy, |chunk_data, idx| {
        let sum: f64 = chunk_data.iter().sum();

        // Print progress for a few chunks
        if idx < 3 || idx >= num_chunks - 3 {
            println!("  Chunk {}: Sum = {:.2}", idx, sum);
        } else if idx == 3 {
            println!("  ... (processing remaining chunks) ...");
        }

        sum
    });

    let total_sum: f64 = chunk_sums.iter().sum();
    println!("Total sum: {}", total_sum);
    println!("Expected sum: {}", (size - 1) as f64 * size as f64 / 2.0); // Sum of arithmetic sequence formula

    Ok(())
}

/// Example showing how to aggregate data from chunks
#[cfg(feature = "memory_efficient")]
fn aggregate_example(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Chunk Aggregation Example");
    println!("---------------------------");

    // Create test data - integers from 0 to 999,999
    let size = 1_000_000;
    println!("Creating a 1D array with {} elements", size);
    let data = Array1::<i32>::from_shape_fn(size, |i| i as i32);

    // Create a memory-mapped array
    let file_path = temp_dir.join("aggregate_example.bin");
    let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0)?;
    println!("Created memory-mapped array at: {:?}", file_path);

    // Use process_chunks to collect statistics from each chunk
    println!("Collecting statistics in {} chunks", 10);
    let chunk_stats = mmap.process_chunks(ChunkingStrategy::NumChunks(10), |chunk_data, idx| {
        let chunk = Array1::<i32>::from_vec(chunk_data.to_vec()); // Convert to Array1 for convenience
        let min = *chunk.iter().min().unwrap_or(&i32::MAX);
        let max = *chunk.iter().max().unwrap_or(&i32::MIN);
        let sum: i64 = chunk.iter().map(|&x| x as i64).sum();
        let count = chunk.len();
        let mean = sum as f64 / count as f64;

        println!(
            "  Chunk {}: Min = {}, Max = {}, Mean = {:.1}",
            idx, min, max, mean
        );

        // Return the stats as a tuple
        (min, max, sum, count, mean)
    });

    // Calculate global statistics from the chunk results
    let global_min = chunk_stats
        .iter()
        .map(|&(min, _, _, _, _)| min)
        .min()
        .unwrap();
    let global_max = chunk_stats
        .iter()
        .map(|&(_, max, _, _, _)| max)
        .max()
        .unwrap();
    let global_sum = chunk_stats
        .iter()
        .map(|&(_, _, sum, _, _)| sum)
        .sum::<i64>();
    let global_count = chunk_stats
        .iter()
        .map(|&(_, _, _, count, _)| count)
        .sum::<usize>();
    let global_mean = global_sum as f64 / global_count as f64;

    println!("\nGlobal statistics:");
    println!("  Min = {}", global_min);
    println!("  Max = {}", global_max);
    println!("  Mean = {:.1}", global_mean);
    println!("  Count = {}", global_count);

    Ok(())
}

/// Performance comparison between chunked and unchunked processing
#[cfg(feature = "memory_efficient")]
fn performance_comparison(temp_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Performance Comparison");
    println!("-----------------------");

    // Create a large array to demonstrate the difference
    let size = 10_000_000;
    println!("Creating a 1D array with {} elements", size);
    let data = Array1::<f32>::linspace(0., (size - 1) as f32, size);

    // Create a memory-mapped file
    let file_path = temp_dir.join("perf_comparison.bin");
    let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0)?;
    println!("Created memory-mapped array at: {:?}", file_path);

    // Vector to store results for verification
    let mut results = Vec::new();

    // 1. Process without chunking (load entire array into memory)
    println!("\nMethod 1: Process without chunking (whole array in memory)");
    let start = Instant::now();

    let array = mmap.as_array::<ndarray::Ix1>()?;
    let sum_no_chunks: f32 = array.sum();
    let min_no_chunks = *array
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_no_chunks = *array
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let time_no_chunks = start.elapsed();
    println!("  Time: {:?}", time_no_chunks);
    println!("  Sum: {}", sum_no_chunks);
    println!("  Min: {}", min_no_chunks);
    println!("  Max: {}", max_no_chunks);

    results.push((sum_no_chunks, min_no_chunks, max_no_chunks));

    // 2. Process with chunk-wise processing
    println!("\nMethod 2: Process with chunked processing");
    let start = Instant::now();

    let chunk_size = 1_000_000;
    let strategy = ChunkingStrategy::Fixed(chunk_size);

    let chunk_results = mmap.process_chunks(strategy, |chunk_data, _| {
        let chunk = Array1::<f32>::from_vec(chunk_data.to_vec()); // Convert to Array1 for convenience
        let sum = chunk.sum();
        let min = *chunk
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max = *chunk
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        (sum, min, max)
    });

    let sum_chunks = chunk_results.iter().map(|&(sum, _, _)| sum).sum::<f32>();
    let min_chunks = chunk_results
        .iter()
        .map(|&(_, min, _)| min)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_chunks = chunk_results
        .iter()
        .map(|&(_, _, max)| max)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let time_chunks = start.elapsed();
    println!("  Time: {:?}", time_chunks);
    println!("  Sum: {}", sum_chunks);
    println!("  Min: {}", min_chunks);
    println!("  Max: {}", max_chunks);

    results.push((sum_chunks, min_chunks, max_chunks));

    // 3. Process with iterators (if available)
    println!("\nMethod 3: Process with chunked iterators");
    let start = Instant::now();

    let mut sum_iter = 0.0;
    let mut min_iter = f32::INFINITY;
    let mut max_iter = f32::NEG_INFINITY;

    for chunk in mmap.chunks(strategy) {
        // Convert to f32 array (we know it's f32 from context)
        let array = Array1::<f32>::from_vec(chunk.to_vec());
        let sum = array.sum();
        let min = *array
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max = *array
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        sum_iter += sum;
        min_iter = min_iter.min(min);
        max_iter = max_iter.max(max);
    }

    let time_iter = start.elapsed();
    println!("  Time: {:?}", time_iter);
    println!("  Sum: {}", sum_iter);
    println!("  Min: {}", min_iter);
    println!("  Max: {}", max_iter);

    results.push((sum_iter, min_iter, max_iter));

    // Verify all methods produced the same results
    println!("\nResults verification:");
    let all_equal = results.windows(2).all(|w| {
        let (sum1, min1, max1) = w[0];
        let (sum2, min2, max2) = w[1];
        (sum1 - sum2).abs() < 1e-3 && (min1 - min2).abs() < 1e-6 && (max1 - max2).abs() < 1e-6
    });

    if all_equal {
        println!("  All methods produced identical results ✓");
    } else {
        println!("  Warning: Methods produced different results!");
    }

    // Performance summary
    println!("\nPerformance Summary:");
    println!("  Method 1 (No chunking): {:?}", time_no_chunks);
    println!(
        "  Method 2 (Chunked processing): {:?} ({:.1}x)",
        time_chunks,
        time_chunks.as_secs_f64() / time_no_chunks.as_secs_f64()
    );
    println!(
        "  Method 3 (Chunk iterator): {:?} ({:.1}x)",
        time_iter,
        time_iter.as_secs_f64() / time_no_chunks.as_secs_f64()
    );

    println!("\nNote: Chunked processing may be slower for small arrays due to overhead,");
    println!("      but enables processing arrays larger than available RAM and can improve");
    println!("      cache efficiency for certain operations.");

    Ok(())
}
