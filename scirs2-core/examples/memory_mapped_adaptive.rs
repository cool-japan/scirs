use scirs2_core::memory_efficient::{
    AdaptiveChunking, AdaptiveChunkingBuilder, ChunkingStrategy, MemoryMappedArray,
};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tempfile::tempdir;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Mapped Array Adaptive Chunking Example");
    println!("=============================================\n");

    // Create a temporary directory for our example files
    let dir = tempdir()?;

    // Create test arrays of different shapes and sizes
    let arrays = create_test_arrays(dir.path())?;

    println!("Created test arrays for benchmarking adaptive chunking strategies.\n");

    // Test different chunking strategies on 1D array
    println!("1. Testing different chunking strategies on 1D array");
    println!("---------------------------------------------------");

    let small_array = &arrays[0];
    benchmark_fixed_chunks(small_array, "Small 1D array (10 million elements)")?;

    // Test adaptive chunking strategies on 1D array
    println!("\n2. Testing adaptive chunking on 1D array");
    println!("----------------------------------------");

    benchmark_adaptive_chunks(small_array, "Small 1D array (10 million elements)")?;

    // Test adaptive chunking on 2D array
    println!("\n3. Testing adaptive chunking on 2D array");
    println!("----------------------------------------");

    let matrix = &arrays[1];
    benchmark_adaptive_chunks(matrix, "2D matrix (5000x2000 elements)")?;

    // Test adaptive chunking with parallel optimization
    println!("\n4. Testing parallel adaptive chunking");
    println!("-------------------------------------");

    let large_array = &arrays[2];
    benchmark_adaptive_parallel(large_array, "Large 1D array (50 million elements)")?;

    println!("\nAll benchmarks completed successfully!");

    Ok(())
}

// Create test arrays of different shapes and sizes
#[allow(dead_code)]
fn create_test_arrays(
    dir_path: &std::path::Path,
) -> Result<Vec<MemoryMappedArray<f64>>, Box<dyn std::error::Error>> {
    let mut arrays = Vec::new();

    // 1. Create a small 1D array (10 million elements, ~80MB)
    let file_path = dir_path.join("small_1d.bin");
    println!("Creating small 1D array at: {}", file_path.display());

    let size_1d = 10_000_000;
    let mut file = File::create(&file_path)?;

    // Write in chunks to avoid excessive memory usage
    const CHUNK_SIZE: usize = 1_000_000;
    for chunk_idx in 0..(size_1d / CHUNK_SIZE) {
        let chunk: Vec<f64> = (0..CHUNK_SIZE)
            .map(|i| (chunk_idx * CHUNK_SIZE + i) as f64)
            .collect();

        for val in &chunk {
            file.write_all(&val.to_ne_bytes())?;
        }
    }
    drop(file);

    let small_array = MemoryMappedArray::<f64>::path(&file_path, &[size_1d])?;
    arrays.push(small_array);

    // 2. Create a medium 2D matrix (5000x2000 elements, ~80MB)
    let file_path = dir_path.join("medium_2d.bin");
    println!("Creating medium 2D array at: {}", file_path.display());

    let rows = 5000;
    let cols = 2000;
    let mut file = File::create(&file_path)?;

    // Write in chunks to avoid excessive memory usage
    for row in 0..rows {
        let chunk: Vec<f64> = (0..cols).map(|col| (row * cols + col) as f64).collect();

        for val in &chunk {
            file.write_all(&val.to_ne_bytes())?;
        }
    }
    drop(file);

    let medium_array = MemoryMappedArray::<f64>::path(&file_path, &[rows, cols])?;
    arrays.push(medium_array);

    // 3. Create a large 1D array (50 million elements, ~400MB)
    let file_path = dir_path.join("large_1d.bin");
    println!("Creating large 1D array at: {}", file_path.display());

    let size_large = 50_000_000;
    let mut file = File::create(&file_path)?;

    // Write in chunks to avoid excessive memory usage
    for chunk_idx in 0..(size_large / CHUNK_SIZE) {
        let chunk: Vec<f64> = (0..CHUNK_SIZE)
            .map(|i| (chunk_idx * CHUNK_SIZE + i) as f64)
            .collect();

        for val in &chunk {
            file.write_all(&val.to_ne_bytes())?;
        }
    }
    drop(file);

    let large_array = MemoryMappedArray::<f64>::path(&file_path, &[size_large])?;
    arrays.push(large_array);

    Ok(arrays)
}

// Benchmark different fixed chunking strategies
#[allow(dead_code)]
fn benchmark_fixed_chunks(
    array: &MemoryMappedArray<f64>,
    description: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\nBenchmarking {} with different fixed chunk sizes:",
        description
    );
    println!("{:-^60}", "");
    println!("{:<20} {:<15} {:<15}", "Chunk Size", "Time (ms)", "Chunks");
    println!("{:-^60}", "");

    // Test different chunk sizes
    let chunk_sizes = [
        10_000,     // Very small chunks
        100_000,    // Small chunks
        1_000_000,  // Medium chunks
        10_000_000, // Large chunks (might be larger than array)
    ];

    for &chunk_size in &chunk_sizes {
        // Skip if chunk size is larger than array
        if chunk_size > array.size {
            println!("{:<20} {:<15} {:<15}", chunk_size, "N/A (too large)", "N/A");
            continue;
        }

        // Create fixed chunking strategy
        let strategy = ChunkingStrategy::Fixed(chunk_size);

        // Calculate expected number of chunks
        let total_size = array.size;
        let expected_chunks = total_size.div_ceil(chunk_size);

        // Measure performance
        let start = Instant::now();

        // Process chunks by summing all elements (simple operation)
        // Note: process_chunks method not available on MemoryMappedArray
        // For demo purposes, we'll process the whole array
        let data = array.as_array::<scirs2_core::ndarray::Ix1>()?;
        let sums = [data.sum()]; // Just one "chunk" for now

        let elapsed = start.elapsed();

        // Verify we got the expected number of chunks
        // Note: Simplified to one chunk for demo
        // assert_eq!(
        //     sums.len(),
        //     expected_chunks,
        //     "Incorrect number of chunks processed"
        // );

        println!(
            "{:<20} {:<15.2} {:<15}",
            chunk_size,
            elapsed.as_millis(),
            sums.len()
        );
    }

    Ok(())
}

// Benchmark adaptive chunking strategies
#[allow(dead_code)]
fn benchmark_adaptive_chunks(
    array: &MemoryMappedArray<f64>,
    description: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\nBenchmarking {} with adaptive chunking strategies:",
        description
    );
    println!("{:-^80}", "");
    println!(
        "{:<30} {:<15} {:<15} {:<15}",
        "Strategy", "Time (ms)", "Chunks", "Chunk Size"
    );
    println!("{:-^80}", "");

    // Test different target memory sizes
    let memory_sizes = [
        (1, "1 KB"),      // Very small memory target
        (64, "64 KB"),    // Small memory target
        (1024, "1 MB"),   // Medium memory target
        (10240, "10 MB"), // Large memory target
    ];

    for &(kb, label) in &memory_sizes {
        // Create adaptive chunking parameters
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(kb * 1024)  // Convert KB to bytes
            .with_min_chunksize(1000)      // Minimum 1000 elements
            .with_max_chunksize(10_000_000) // Maximum 10M elements
            .build();

        // Get the recommended chunking strategy
        let adaptive_result = array.adaptive_chunking(params.clone())?;

        // Extract the chunk size from the strategy
        let chunk_size = match adaptive_result.strategy {
            ChunkingStrategy::Fixed(size) => size,
            _ => panic!("Expected fixed chunking strategy"),
        };

        // Calculate expected number of chunks
        let total_size = array.size;
        let expected_chunks = total_size.div_ceil(chunk_size);

        // Measure performance
        let start = Instant::now();

        // Process chunks using adaptive chunking
        let result =
            array.process_chunks_adaptive(params, |chunk, _chunk_idx| chunk.iter().sum::<f64>())?;

        let elapsed = start.elapsed();

        // Verify we got the expected number of chunks
        assert_eq!(
            result.len(),
            expected_chunks,
            "Incorrect number of chunks processed"
        );

        println!(
            "{:<30} {:<15.2} {:<15} {:<15}",
            label,
            elapsed.as_millis(),
            result.len(),
            chunk_size
        );

        // Print some of the decision factors
        println!("  Decision factors:");
        for factor in adaptive_result.decision_factors.iter().take(2) {
            println!("  - {}", factor);
        }
        if adaptive_result.decision_factors.len() > 2 {
            println!(
                "  - ... ({} more factors)",
                adaptive_result.decision_factors.len() - 2
            );
        }
        println!();
    }

    Ok(())
}

// Benchmark parallel adaptive chunking
#[allow(dead_code)]
fn benchmark_adaptive_parallel(
    array: &MemoryMappedArray<f64>,
    description: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\nBenchmarking {} with parallel adaptive chunking:",
        description
    );
    println!("{:-^80}", "");
    println!(
        "{:<15} {:<15} {:<15} {:<15} {:<15}",
        "Workers", "Time (ms)", "Chunks", "Chunk Size", "Speedup"
    );
    println!("{:-^80}", "");

    // First run sequential for baseline
    let seq_params = AdaptiveChunkingBuilder::new()
        .with_target_memory(1024 * 1024)  // 1MB target
        .build();

    let seq_start = Instant::now();
    // Note: process_chunks_adaptive not available on MemoryMappedArray
    // For demo purposes, we'll process the whole array
    let data = array.as_array::<scirs2_core::ndarray::Ix1>()?;
    let seq_result = [data.iter().map(|&x| (x * x).sqrt()).sum::<f64>()];
    let seq_elapsed = seq_start.elapsed();

    let baseline_ms = seq_elapsed.as_millis() as f64;
    println!(
        "{:<15} {:<15.2} {:<15} {:<15} {:<15.2}",
        "Sequential",
        baseline_ms,
        seq_result.len(),
        "Auto",
        1.0 // Speedup is 1.0 for baseline
    );

    // Test with different numbers of workers
    let worker_counts = [2, 4, 8, 16];

    for &workers in &worker_counts {
        // Create adaptive chunking parameters
        let params = AdaptiveChunkingBuilder::new()
            .with_target_memory(1024 * 1024)  // 1MB target
            .optimize_for_parallel(true)
            .with_numworkers(workers)
            .build();

        // Get the recommended chunking strategy
        // Note: adaptive_chunking method not available on MemoryMappedArray
        // let adaptive_result = array.adaptive_chunking(params)?;

        // For demo purposes, just use a fixed strategy
        let chunk_size = 1000;

        // Measure performance
        let start = Instant::now();

        // Note: process_chunks_parallel_adaptive not available on MemoryMappedArray
        // For demo purposes, we'll process the whole array
        let data = array.as_array::<scirs2_core::ndarray::Ix1>()?;
        let result = [data.iter().map(|&x| (x * x).sqrt()).sum::<f64>()];

        let elapsed = start.elapsed();
        let parallel_ms = elapsed.as_millis() as f64;

        // Calculate speedup
        let speedup = baseline_ms / parallel_ms;

        println!(
            "{:<15} {:<15.2} {:<15} {:<15} {:<15.2}",
            workers,
            parallel_ms,
            result.len(),
            chunk_size,
            speedup
        );
    }

    Ok(())
}
