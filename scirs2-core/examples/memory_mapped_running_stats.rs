//! Example demonstrating advanced usage of memory-mapped chunks for running statistics.
//!
//! This example shows how to:
//! 1. Create a large memory-mapped array that may not fit in memory
//! 2. Process it in chunks to calculate running statistics (mean, variance)
//! 3. Update the array with normalized values using the calculated statistics
//!
//! Note: This example requires the `memory_efficient` feature to be enabled.
//! Run with: `cargo run --example memory_mapped_running_stats --features memory_efficient`

#[cfg(feature = "memory_efficient")]
use ndarray::{Array1, ArrayView1};
#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{
    create_mmap, AccessMode, ChunkingStrategy, MemoryMappedArray, MemoryMappedChunks,
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
    println!(
        "Run with: cargo run --example memory_mapped_running_stats --features memory_efficient"
    );
}

// Struct to accumulate statistics in a numerically stable way
#[cfg(feature = "memory_efficient")]
struct OnlineStats {
    count: usize,
    mean: f64,
    m2: f64, // For variance calculation
}

#[cfg(feature = "memory_efficient")]
impl OnlineStats {
    fn new() -> Self {
        OnlineStats {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    // Update stats with a batch of values
    fn update_batch(&mut self, values: ArrayView1<f64>) {
        for &x in values.iter() {
            self.update(x);
        }
    }

    // Update stats with a single value using Welford's algorithm
    fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    // Get the current mean
    fn mean(&self) -> f64 {
        self.mean
    }

    // Get the current variance
    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    // Get the current standard deviation
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    // Merge another OnlineStats into this one
    fn merge(&mut self, other: &OnlineStats) {
        if other.count == 0 {
            return;
        }

        if self.count == 0 {
            self.count = other.count;
            self.mean = other.mean;
            self.m2 = other.m2;
            return;
        }

        let total_count = self.count + other.count;
        let delta = other.mean - self.mean;

        self.mean =
            (self.mean * self.count as f64 + other.mean * other.count as f64) / total_count as f64;

        // Combine M2 values
        self.m2 = self.m2
            + other.m2
            + delta * delta * (self.count * other.count) as f64 / total_count as f64;

        self.count = total_count;
    }
}

#[cfg(feature = "memory_efficient")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Mapped Running Statistics Example");
    println!("========================================\n");

    // Create a temporary directory for our example files
    let temp_dir = tempdir()?;
    println!("Using temporary directory: {:?}", temp_dir.path());

    // Create a large dataset for demonstration
    let mut mmap = create_large_dataset(temp_dir.path())?;

    // Calculate statistics using chunk-wise processing
    let stats = calculate_statistics(&mut mmap)?;

    // Normalize the data using the calculated statistics
    normalize_data(&mut mmap, stats.mean(), stats.std_dev())?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

/// Create a large dataset for demonstration purposes
#[cfg(feature = "memory_efficient")]
fn create_large_dataset(
    temp_dir: &Path,
) -> Result<MemoryMappedArray<f64>, Box<dyn std::error::Error>> {
    println!("\n1. Creating Large Dataset");
    println!("------------------------");

    // Create a large array with random-like but deterministic data
    let size: usize = 10_000_000; // 10 million elements (~80MB for f64)
    println!("Creating a dataset with {} elements", size);

    let start = Instant::now();

    // Generate data algorithmically to avoid allocating the entire array at once
    let chunk_size: usize = 1_000_000;
    let num_chunks = size.div_ceil(chunk_size);

    println!(
        "Generating data in {} chunks of {} elements each",
        num_chunks, chunk_size
    );

    // Create a memory-mapped file large enough for our dataset
    let file_path = temp_dir.join("large_dataset.bin");

    // First create an array with the first chunk to initialize the file
    let initial_data = Array1::<f64>::from_shape_fn(chunk_size, |i| {
        // Generate a deterministic pattern
        let x = i as f64;
        (x / 1000.0).sin() * 10.0 + (x / 5000.0).cos() * 5.0 + (i % 100) as f64 / 10.0
    });

    let mut mmap = create_mmap(&initial_data, &file_path, AccessMode::Write, 0)?;

    // Generate the rest of the data in chunks
    if num_chunks > 1 {
        for chunk_idx in 1..num_chunks {
            println!("  Generating chunk {}/{}", chunk_idx + 1, num_chunks);

            // Calculate chunk boundary
            let start_idx = chunk_idx * chunk_size;
            let end_idx = ((chunk_idx + 1) * chunk_size).min(size);
            let actual_chunk_size = end_idx - start_idx;

            // Generate this chunk's data
            let mut chunk_data = Vec::with_capacity(actual_chunk_size);
            for i in 0..actual_chunk_size {
                let global_idx = start_idx + i;
                let x = global_idx as f64;
                let value = (x / 1000.0).sin() * 10.0
                    + (x / 5000.0).cos() * 5.0
                    + (global_idx % 100) as f64 / 10.0;
                chunk_data.push(value);
            }

            // Process the chunk to append it to the file
            // We need to create a new array with the full size
            let full_array = Array1::<f64>::zeros(size);
            let mut new_mmap = create_mmap(&full_array, &file_path, AccessMode::ReadWrite, 0)?;

            // Write just this chunk
            new_mmap.process_chunks_mut(
                ChunkingStrategy::Fixed(actual_chunk_size),
                move |buffer, idx| {
                    if idx == chunk_idx {
                        // Copy our generated data into the buffer
                        for (i, value) in chunk_data.iter().enumerate() {
                            buffer[i] = *value;
                        }
                    }
                },
            );

            // Replace our mmap with the updated one
            mmap = new_mmap;
        }
    }

    let duration = start.elapsed();
    println!("Dataset created in {:?}", duration);
    println!(
        "Dataset size: {:.2} MB",
        (size * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0)
    );

    Ok(mmap)
}

/// Calculate running statistics on the large dataset
#[cfg(feature = "memory_efficient")]
fn calculate_statistics(
    mmap: &mut MemoryMappedArray<f64>,
) -> Result<OnlineStats, Box<dyn std::error::Error>> {
    println!("\n2. Calculating Statistics");
    println!("------------------------");

    println!("Dataset size: {} elements", mmap.size);

    // Choose an appropriate chunk size
    let chunk_size = 1_000_000; // 1 million elements per chunk
    let strategy = ChunkingStrategy::Fixed(chunk_size);
    let num_chunks = mmap.chunk_count(strategy);

    println!(
        "Processing dataset in {} chunks of {} elements each",
        num_chunks, chunk_size
    );

    let start = Instant::now();

    // Initialize stats accumulator
    let mut global_stats = OnlineStats::new();

    // Process each chunk
    let chunk_stats = mmap.process_chunks(strategy, |chunk_data, chunk_idx| {
        // Create a temporary view for this chunk
        let chunk = ArrayView1::from_shape(chunk_data.len(), chunk_data).unwrap();

        // Create a stats accumulator for this chunk
        let mut chunk_stats = OnlineStats::new();
        chunk_stats.update_batch(chunk);

        println!(
            "  Chunk {}: Mean = {:.4}, StdDev = {:.4}",
            chunk_idx,
            chunk_stats.mean(),
            chunk_stats.std_dev()
        );

        // Return the chunk stats for merging
        chunk_stats
    });

    // Merge all chunk stats
    for stats in chunk_stats {
        global_stats.merge(&stats);
    }

    let duration = start.elapsed();

    println!("\nFinal statistics:");
    println!("  Mean:        {:.6}", global_stats.mean());
    println!("  Variance:    {:.6}", global_stats.variance());
    println!("  Std Dev:     {:.6}", global_stats.std_dev());
    println!("  Sample size: {}", global_stats.count);
    println!("Calculation time: {:?}", duration);

    Ok(global_stats)
}

/// Normalize the data using the calculated statistics
#[cfg(feature = "memory_efficient")]
fn normalize_data(
    mmap: &mut MemoryMappedArray<f64>,
    mean: f64,
    stddev: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. Normalizing Data");
    println!("------------------");

    println!(
        "Using statistics: Mean = {:.6}, StdDev = {:.6}",
        mean, stddev
    );
    println!("Dataset size: {} elements", mmap.size);

    // Choose an appropriate chunk size
    let chunk_size = 1_000_000; // 1 million elements per chunk
    let strategy = ChunkingStrategy::Fixed(chunk_size);
    let num_chunks = mmap.chunk_count(strategy);

    println!(
        "Normalizing dataset in {} chunks of {} elements each",
        num_chunks, chunk_size
    );

    let start = Instant::now();

    // Process each chunk and normalize the data
    mmap.process_chunks_mut(strategy, |chunk_data, chunk_idx| {
        println!("  Normalizing chunk {}/{}", chunk_idx + 1, num_chunks);

        // Skip normalization if stddev is too small (to avoid division by zero)
        if stddev < 1e-10 {
            return;
        }

        // Normalize each value in the chunk
        for value in chunk_data.iter_mut() {
            // z-score normalization: (x - mean) / stddev
            *value = (*value - mean) / stddev;
        }

        // Calculate some metrics on the normalized chunk
        if chunk_idx == 0 {
            // For the first chunk, print some sample values
            let limit = 5.min(chunk_data.len());
            println!(
                "    First {} normalized values: {:?}",
                limit,
                &chunk_data[..limit]
            );

            // Calculate and print statistics of normalized data
            let mut norm_stats = OnlineStats::new();
            for &x in chunk_data.iter() {
                norm_stats.update(x);
            }

            println!(
                "    Normalized sample stats: Mean = {:.6}, StdDev = {:.6}",
                norm_stats.mean(),
                norm_stats.std_dev()
            );
        }
    });

    let duration = start.elapsed();
    println!("Normalization completed in {:?}", duration);

    // Verify normalization by calculating stats on the normalized data
    let norm_stats = calculate_verification_stats(mmap)?;

    println!("\nVerification statistics on normalized data:");
    println!("  Mean:        {:.6}", norm_stats.mean());
    println!("  Std Dev:     {:.6}", norm_stats.std_dev());
    println!("  Sample size: {}", norm_stats.count);

    println!(
        "\nNormalization successful! The data now has approximately zero mean and unit variance."
    );

    Ok(())
}

/// Calculate verification statistics on the normalized data
#[cfg(feature = "memory_efficient")]
fn calculate_verification_stats(
    mmap: &MemoryMappedArray<f64>,
) -> Result<OnlineStats, Box<dyn std::error::Error>> {
    // Sample a subset of the data to verify normalization
    let sample_size = mmap.size.min(1_000_000); // Max 1 million elements
    let stride = (mmap.size / sample_size).max(1);

    let array = mmap.as_array::<ndarray::Ix1>()?;
    let mut stats = OnlineStats::new();

    for i in (0..mmap.size).step_by(stride) {
        stats.update(array[i]);
    }

    Ok(stats)
}
