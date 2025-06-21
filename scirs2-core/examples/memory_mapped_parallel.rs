//! Example demonstrating parallel processing of memory-mapped arrays.
//!
//! This example creates a memory-mapped array and demonstrates how to use
//! the parallel processing capabilities of the `MemoryMappedChunksParallel` trait
//! to process chunks of the array in parallel.
//!
//! Run with:
//! ```bash
//! cargo run --example memory_mapped_parallel --features="memory_efficient parallel"
//! ```

#[cfg(all(feature = "memory_efficient", feature = "parallel"))]
fn main() {
    use ndarray::Array1;
    use scirs2_core::memory_efficient::{
        create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunks, MemoryMappedChunksParallel,
    };
    // Path is unused
    use std::time::Instant;

    println!("Memory-Mapped Array Parallel Processing Example");
    println!("==============================================");

    // Create a large array for our example
    let size = 10_000_000;
    let data = Array1::<f64>::linspace(0.0, (size as f64) - 1.0, size);
    println!("Created array with {} elements", size);

    // Create a temporary file for the memory-mapped array
    let temp_file = tempfile::NamedTempFile::new().unwrap();
    let temp_path = temp_file.path();
    println!("Created temporary file at: {}", temp_path.display());

    // Create a memory-mapped array
    let mut mmap = create_mmap(&data, temp_path, AccessMode::ReadWrite, 0).unwrap();
    println!("Created memory-mapped array");

    // First, let's measure the time it takes to process the chunks sequentially
    println!("\nProcessing chunks sequentially...");
    let start = Instant::now();

    // Process each chunk and calculate its sum
    let chunk_size = 100_000;
    let sequential_sums =
        mmap.process_chunks(ChunkingStrategy::Fixed(chunk_size), |chunk, chunk_idx| {
            // Simulate some computation by squaring each element
            let result: f64 = chunk.iter().map(|&x| x * x).sum();
            if chunk_idx % 20 == 0 {
                println!(
                    "Sequential - Processed chunk {}: sum of squares = {:.2}",
                    chunk_idx, result
                );
            }
            result
        });

    let sequential_time = start.elapsed();
    println!("Sequential processing time: {:?}", sequential_time);
    println!("Number of chunks processed: {}", sequential_sums.len());

    // Now, let's measure the time it takes to process the chunks in parallel
    println!("\nProcessing chunks in parallel...");
    let start = Instant::now();

    // Process each chunk in parallel and calculate its sum
    let parallel_sums =
        mmap.process_chunks_parallel(ChunkingStrategy::Fixed(chunk_size), |chunk, chunk_idx| {
            // Simulate some computation by squaring each element
            let result: f64 = chunk.iter().map(|&x| x * x).sum();
            if chunk_idx % 20 == 0 {
                println!(
                    "Parallel - Processed chunk {}: sum of squares = {:.2}",
                    chunk_idx, result
                );
            }
            result
        });

    let parallel_time = start.elapsed();
    println!("Parallel processing time: {:?}", parallel_time);
    println!("Number of chunks processed: {}", parallel_sums.len());

    // Compare the speedup
    let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("\nSpeedup with parallel processing: {:.2}x", speedup);

    // Verify that the results are the same
    let sequential_total: f64 = sequential_sums.iter().sum();
    let parallel_total: f64 = parallel_sums.iter().sum();
    println!("Sequential total: {:.2}", sequential_total);
    println!("Parallel total: {:.2}", parallel_total);
    assert!(
        (sequential_total - parallel_total).abs() < 1e-10,
        "Sequential and parallel results differ!"
    );

    // Let's also demonstrate mutating the array in parallel
    println!("\nModifying chunks in parallel...");
    let start = Instant::now();

    // Modify each element in parallel to be its square
    mmap.process_chunks_mut_parallel(ChunkingStrategy::Fixed(chunk_size), |chunk, chunk_idx| {
        for elem in chunk.iter_mut() {
            *elem = *elem * *elem;
        }
        if chunk_idx % 20 == 0 {
            println!("Modified chunk {}", chunk_idx);
        }
    });

    let mutation_time = start.elapsed();
    println!("Parallel mutation time: {:?}", mutation_time);

    // Verify that the mutation worked by computing the sum
    println!("\nVerifying the mutation worked...");
    let sum_after_mutation: f64 = mmap
        .process_chunks(ChunkingStrategy::Fixed(chunk_size), |chunk, _| {
            chunk.iter().sum::<f64>()
        })
        .iter()
        .sum();

    println!("Sum after mutation: {:.2}", sum_after_mutation);

    // This should be approximately equal to the sum of squares we calculated earlier
    assert!(
        (sum_after_mutation - sequential_total).abs() < 1e-10,
        "Mutation did not produce expected results!"
    );

    println!("\nSuccessfully completed all operations!");
}

#[cfg(not(all(feature = "memory_efficient", feature = "parallel")))]
fn main() {
    println!("This example requires the 'memory_efficient' and 'parallel' features.");
    println!("Please run with:");
    println!("cargo run --example memory_mapped_parallel --features=\"memory_efficient parallel\"");
}
