use scirs2_core::memory_efficient::{
    CompressedMemMapBuilder, CompressedMemMappedArray, CompressionAlgorithm, MemoryMappedArray,
};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use tempfile::tempdir;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Compressed Memory-Mapped Array Example");
    println!("======================================\n");

    // Create a temporary directory for our example files
    let dir = tempdir()?;
    let raw_file_path = dir.path().join("large_array.bin");
    let compressed_file_path = dir.path().join("large_array.cmm");

    println!("Creating test data...");

    // Create a large test array (10 million elements)
    let size = 10_000_000;
    println!("Creating a 1D array with {} elements", size);

    // Create and save the array in chunks to avoid excessive memory usage
    let mut file = File::create(&raw_file_path)?;
    let chunk_size = 1_000_000;

    for chunk_idx in 0..(size / chunk_size) {
        let start = chunk_idx * chunk_size;
        let chunk: Vec<f64> = (0..chunk_size).map(|i| (start + i) as f64).collect();

        for val in &chunk {
            file.write_all(&val.to_ne_bytes())?;
        }
    }
    drop(file);

    let raw_file_size = std::fs::metadata(&raw_file_path)?.len();
    println!(
        "Uncompressed array saved to file (size: {} bytes)",
        raw_file_size
    );

    // Open as standard memory-mapped array for comparison
    let mmap = MemoryMappedArray::<f64>::path(&raw_file_path, &[size])?;

    // Create a compressed memory-mapped array
    println!("\nCreating compressed memory-mapped arrays with different algorithms...");

    // Create a table for compression results
    println!("{:-^80}", "");
    println!(
        "{:<15} {:<10} {:<15} {:<15} {:<15} {:<10}",
        "Algorithm", "Level", "File Size (MB)", "Ratio", "Compress Time", "Block Size"
    );
    println!("{:-^80}", "");

    // Compression algorithms to test
    let algorithms = [
        (CompressionAlgorithm::Lz4, "LZ4", 1),
        (CompressionAlgorithm::Lz4, "LZ4", 9),
        (CompressionAlgorithm::Zstd, "Zstd", 1),
        (CompressionAlgorithm::Zstd, "Zstd", 9),
        (CompressionAlgorithm::Snappy, "Snappy", 1),
    ];

    // Block sizes to test
    let block_sizes = [(65536, "64K"), (262144, "256K"), (1048576, "1M")];

    let mut compressed_arrays: Vec<CompressedMemMappedArray<f64>> = Vec::new();

    for &(algorithm, name, level) in &algorithms {
        for &(block_size, block_name) in &block_sizes {
            let output_path = dir
                .path()
                .join(format!("array_{}_{}_l{}.cmm", name, block_name, level));

            // Create a builder
            let builder = CompressedMemMapBuilder::new()
                .with_block_size(block_size)
                .with_algorithm(algorithm)
                .with_level(level)
                .with_description(format!("{} compression level {}", name, level));

            // Measure compression time
            let start = Instant::now();

            // Create compressed array from standard memory-mapped array
            let array = mmap.readonlyarray::<scirs2_core::ndarray::Ix1>()?;
            let cmm = builder.create(&array, &output_path)?;

            let elapsed = start.elapsed();

            // Get compressed file size
            let compressed_size = std::fs::metadata(&output_path)?.len();
            let compression_ratio = raw_file_size as f64 / compressed_size as f64;

            // Print results
            println!(
                "{:<15} {:<10} {:<15.2} {:<15.2}x {:<15.2?} {:<10}",
                name,
                level,
                compressed_size as f64 / (1024.0 * 1024.0),
                compression_ratio,
                elapsed,
                block_name
            );

            // Save for performance tests
            if block_size == 262144 {
                // Only keep the middle block size for performance tests
                compressed_arrays.push(cmm);
            }
        }
    }

    println!("\nPerformance Comparison: Random Access");
    println!("{:-^60}", "");
    println!(
        "{:<20} {:<20} {:<15}",
        "Method", "Time (ms)", "Memory Usage"
    );
    println!("{:-^60}", "");

    // Test random access performance (memory-mapped)
    let start = Instant::now();
    let mut _sum = 0.0;
    for i in 0..1000 {
        let idx = (i * 10000) % size; // Random-ish access
        let val = mmap.readonlyarray::<scirs2_core::ndarray::Ix1>()?[idx];
        _sum += val;
    }
    let elapsed = start.elapsed();
    println!(
        "{:<20} {:<20.2} {:<15}",
        "Memory-mapped",
        elapsed.as_millis() as f64,
        "High"
    );

    // Test random access performance (compressed)
    for (i, cmm) in compressed_arrays.iter().enumerate() {
        let algorithm_name = match i {
            0 => "LZ4 (L1)",
            1 => "LZ4 (L9)",
            2 => "Zstd (L1)",
            3 => "Zstd (L9)",
            4 => "Snappy",
            _ => "Unknown",
        };

        let start = Instant::now();
        let mut _sum = 0.0;
        for i in 0..1000 {
            let idx = (i * 10000) % size; // Random-ish access
            let val = cmm.get(&[idx])?;
            _sum += val;
        }
        let elapsed = start.elapsed();
        println!(
            "{:<20} {:<20.2} {:<15}",
            algorithm_name,
            elapsed.as_millis() as f64,
            "Low"
        );
    }

    println!("\nPerformance Comparison: Sequential Access (full array load)");
    println!("{:-^60}", "");
    println!("{:<20} {:<20} {:<15}", "Method", "Time (ms)", "Sum Result");
    println!("{:-^60}", "");

    // Test sequential access performance (memory-mapped)
    let start = Instant::now();
    let mmap_array = mmap.readonlyarray::<scirs2_core::ndarray::Ix1>()?;
    let mmap_sum: f64 = mmap_array.iter().sum();
    let elapsed = start.elapsed();
    println!(
        "{:<20} {:<20.2} {:<15.0}",
        "Memory-mapped",
        elapsed.as_millis() as f64,
        mmap_sum
    );

    // Test sequential access performance (compressed)
    for (i, cmm) in compressed_arrays.iter().enumerate() {
        let algorithm_name = match i {
            0 => "LZ4 (L1)",
            1 => "LZ4 (L9)",
            2 => "Zstd (L1)",
            3 => "Zstd (L9)",
            4 => "Snappy",
            _ => "Unknown",
        };

        let start = Instant::now();
        let cmm_array = cmm.readonly_array()?;
        let cmm_sum: f64 = cmm_array.iter().sum();
        let elapsed = start.elapsed();
        println!(
            "{:<20} {:<20.2} {:<15.0}",
            algorithm_name,
            elapsed.as_millis() as f64,
            cmm_sum
        );
    }

    println!("\nPerformance Comparison: Block Processing");
    println!("{:-^60}", "");
    println!("{:<20} {:<20} {:<15}", "Method", "Time (ms)", "Sum Result");
    println!("{:-^60}", "");

    // Standard memory-mapped block processing
    let start = Instant::now();
    let block_size = 262144;
    let num_blocks = size.div_ceil(block_size);
    let mut mmap_sum = 0.0;

    for block_idx in 0..num_blocks {
        let start_idx = block_idx * block_size;
        let end_idx = (start_idx + block_size).min(size);

        let block = mmap
            .readonlyarray::<scirs2_core::ndarray::Ix1>()?
            .slice(scirs2_core::ndarray::s![start_idx..end_idx])
            .to_owned();

        mmap_sum += block.iter().sum::<f64>();
    }
    let elapsed = start.elapsed();
    println!(
        "{:<20} {:<20.2} {:<15.0}",
        "Memory-mapped",
        elapsed.as_millis() as f64,
        mmap_sum
    );

    // Test block processing performance (compressed)
    for (i, cmm) in compressed_arrays.iter().enumerate() {
        let algorithm_name = match i {
            0 => "LZ4 (L1)",
            1 => "LZ4 (L9)",
            2 => "Zstd (L1)",
            3 => "Zstd (L9)",
            4 => "Snappy",
            _ => "Unknown",
        };

        let start = Instant::now();
        let results = cmm.process_blocks(|block, _idx| block.iter().sum::<f64>())?;
        let cmm_sum: f64 = results.iter().sum();
        let elapsed = start.elapsed();
        println!(
            "{:<20} {:<20.2} {:<15.0}",
            algorithm_name,
            elapsed.as_millis() as f64,
            cmm_sum
        );
    }

    // Demonstrate preloading blocks for better performance
    println!("\nPerformance Improvement with Block Preloading");
    println!("{:-^70}", "");
    println!(
        "{:<20} {:<15} {:<15} {:<15}",
        "Method", "No Preload (ms)", "With Preload (ms)", "Improvement"
    );
    println!("{:-^70}", "");

    // Choose one algorithm for preload testing
    let cmm = &compressed_arrays[0]; // LZ4 level 1

    // Without preloading
    let start = Instant::now();
    let mut _sum = 0.0;
    for i in 0..100 {
        let slice = cmm.slice(&[(i * 1000, (i + 1) * 1000)])?;
        _sum += slice.iter().sum::<f64>();
    }
    let no_preload_time = start.elapsed();

    // With preloading
    let start = Instant::now();
    let mut _sum = 0.0;
    for i in 0..100 {
        // Preload the block containing this slice
        let block_idx = (i * 1000) / cmm.metadata().block_size;
        cmm.preload_block(block_idx)?;

        // Now access the slice
        let slice = cmm.slice(&[(i * 1000, (i + 1) * 1000)])?;
        _sum += slice.iter().sum::<f64>();
    }
    let with_preload_time = start.elapsed();

    // Calculate improvement
    let improvement = no_preload_time.as_millis() as f64 / with_preload_time.as_millis() as f64;

    println!(
        "{:<20} {:<15.2} {:<15.2} {:<15.2}x",
        "LZ4 (L1)",
        no_preload_time.as_millis() as f64,
        with_preload_time.as_millis() as f64,
        improvement
    );

    println!("\nAll examples completed successfully!");

    Ok(())
}
