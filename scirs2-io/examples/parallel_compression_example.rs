//! Parallel compression and decompression example
//!
//! This example demonstrates the new parallel compression capabilities
//! that can significantly improve performance when working with large datasets.

use scirs2_io::compression::{
    self, benchmark_compression_algorithms, compress_data_parallel, decompress_data_parallel,
    CompressionAlgorithm, ParallelCompressionConfig,
};
use std::time::Instant;
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Parallel Compression and Decompression Example");
    println!("==============================================");

    // Create temporary directory for test files
    let temp_dir = tempdir()?;
    println!("📁 Using temporary directory: {:?}", temp_dir.path());

    // Generate large test data
    println!("\n🏗️  Generating test data...");
    let test_data = generate_test_data(10_000_000); // 10MB of data
    println!("📊 Generated {} bytes of test data", test_data.len());

    // Demonstrate basic parallel compression
    demonstrate_basic_parallel_compression(&test_data)?;

    // Compare sequential vs parallel performance
    demonstrate_performance_comparison(&test_data)?;

    // Demonstrate file operations
    demonstrate_parallel_file_operations(&test_data, &temp_dir)?;

    // Benchmark different algorithms and configurations
    demonstrate_algorithm_benchmarking(&test_data)?;

    println!("\n✅ All parallel compression demonstrations completed successfully!");
    println!("💡 Parallel compression provides significant speedups for large datasets");

    Ok(())
}

fn generate_test_data(size: usize) -> Vec<u8> {
    // Generate semi-random data that compresses well
    let mut data = Vec::with_capacity(size);

    // Create patterns that will compress well but still represent realistic data
    for i in 0..size {
        let pattern = match i % 4 {
            0 => (i / 1000) as u8, // Slowly changing values
            1 => 0x42,             // Repeated bytes
            2 => (i % 256) as u8,  // Cycling pattern
            3 => {
                if i % 10 == 0 {
                    0xFF
                } else {
                    0x00
                }
            } // Sparse pattern
            _ => unreachable!(),
        };
        data.push(pattern);
    }

    data
}

fn demonstrate_basic_parallel_compression(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔧 Demonstrating Basic Parallel Compression...");

    let algorithm = CompressionAlgorithm::Zstd;
    let level = Some(6);
    let config = ParallelCompressionConfig::default();

    println!(
        "  📝 Compressing {} bytes with {:?} (level {})...",
        data.len(),
        algorithm,
        level.unwrap()
    );

    let start_time = Instant::now();
    let (compressed_data, compression_stats) =
        compress_data_parallel(data, algorithm, level, config.clone())?;
    let _compression_time = start_time.elapsed();

    println!("  📊 Compression Results:");
    println!(
        "    Original size: {} bytes",
        compression_stats.bytes_processed
    );
    println!(
        "    Compressed size: {} bytes",
        compression_stats.bytes_output
    );
    println!(
        "    Compression ratio: {:.2}x",
        compression_stats.compression_ratio
    );
    println!(
        "    Chunks processed: {}",
        compression_stats.chunks_processed
    );
    println!("    Threads used: {}", compression_stats.threads_used);
    println!("    Time: {:.2}ms", compression_stats.operation_time_ms);
    println!(
        "    Throughput: {:.2} MB/s",
        compression_stats.throughput_bps / 1_000_000.0
    );

    println!("  📖 Decompressing data...");
    let start_time = Instant::now();
    let (decompressed_data, decompression_stats) =
        decompress_data_parallel(&compressed_data, algorithm, config)?;
    let _decompression_time = start_time.elapsed();

    println!("  📊 Decompression Results:");
    println!(
        "    Compressed size: {} bytes",
        decompression_stats.bytes_processed
    );
    println!(
        "    Decompressed size: {} bytes",
        decompression_stats.bytes_output
    );
    println!(
        "    Chunks processed: {}",
        decompression_stats.chunks_processed
    );
    println!("    Threads used: {}", decompression_stats.threads_used);
    println!("    Time: {:.2}ms", decompression_stats.operation_time_ms);
    println!(
        "    Throughput: {:.2} MB/s",
        decompression_stats.throughput_bps / 1_000_000.0
    );

    // Verify data integrity
    assert_eq!(data, &decompressed_data, "Data integrity check failed!");
    println!("  ✅ Data integrity verified - perfect round-trip!");

    Ok(())
}

fn demonstrate_performance_comparison(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚖️  Comparing Sequential vs Parallel Performance...");

    let algorithm = CompressionAlgorithm::Zstd;
    let level = Some(6);
    let config = ParallelCompressionConfig::default();

    // Sequential compression
    println!("  🐌 Sequential compression...");
    let start_time = Instant::now();
    let compressed_sequential = compression::compress_data(data, algorithm, level)?;
    let sequential_compression_time = start_time.elapsed().as_secs_f64() * 1000.0;

    let start_time = Instant::now();
    let decompressed_sequential = compression::decompress_data(&compressed_sequential, algorithm)?;
    let sequential_decompression_time = start_time.elapsed().as_secs_f64() * 1000.0;

    // Parallel compression
    println!("  ⚡ Parallel compression...");
    let (compressed_parallel, parallel_compression_stats) =
        compress_data_parallel(data, algorithm, level, config.clone())?;
    let (decompressed_parallel, parallel_decompression_stats) =
        decompress_data_parallel(&compressed_parallel, algorithm, config)?;

    // Verify both methods produce correct results
    assert_eq!(
        data, &decompressed_sequential,
        "Sequential round-trip failed!"
    );
    assert_eq!(data, &decompressed_parallel, "Parallel round-trip failed!");

    // Compare results
    println!("  📊 Performance Comparison:");
    println!(
        "    Sequential compression: {:.2}ms",
        sequential_compression_time
    );
    println!(
        "    Parallel compression: {:.2}ms",
        parallel_compression_stats.operation_time_ms
    );
    println!(
        "    Compression speedup: {:.2}x",
        sequential_compression_time / parallel_compression_stats.operation_time_ms
    );

    println!(
        "    Sequential decompression: {:.2}ms",
        sequential_decompression_time
    );
    println!(
        "    Parallel decompression: {:.2}ms",
        parallel_decompression_stats.operation_time_ms
    );
    println!(
        "    Decompression speedup: {:.2}x",
        sequential_decompression_time / parallel_decompression_stats.operation_time_ms
    );

    println!(
        "    Sequential compressed size: {} bytes",
        compressed_sequential.len()
    );
    println!(
        "    Parallel compressed size: {} bytes",
        compressed_parallel.len()
    );
    println!(
        "    Size overhead: {:.2}%",
        ((compressed_parallel.len() as f64 / compressed_sequential.len() as f64) - 1.0) * 100.0
    );

    Ok(())
}

fn demonstrate_parallel_file_operations(
    data: &[u8],
    temp_dir: &tempfile::TempDir,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💾 Demonstrating Parallel File Operations...");

    let input_file = temp_dir.path().join("test_data.bin");
    let compressed_file = temp_dir.path().join("test_data.bin.zst");
    let decompressed_file = temp_dir.path().join("test_data_restored.bin");

    // Write original data to file
    std::fs::write(&input_file, data)?;
    println!("  📝 Wrote {} bytes to input file", data.len());

    // Compress file in parallel
    let algorithm = CompressionAlgorithm::Zstd;
    let level = Some(6);
    let config = ParallelCompressionConfig {
        num_threads: 4,
        chunk_size: 512 * 1024, // 512KB chunks
        buffer_size: 64 * 1024,
        enable_memory_mapping: true,
    };

    println!(
        "  🗜️  Compressing file in parallel ({} threads, {}KB chunks)...",
        config.num_threads,
        config.chunk_size / 1024
    );

    let (compressed_path, compression_stats) = compression::compress_file_parallel(
        &input_file,
        Some(&compressed_file),
        algorithm,
        level,
        config.clone(),
    )?;

    println!("  📊 File Compression Results:");
    println!("    Output file: {}", compressed_path);
    println!(
        "    Compression ratio: {:.2}x",
        compression_stats.compression_ratio
    );
    println!("    Time: {:.2}ms", compression_stats.operation_time_ms);
    println!(
        "    Throughput: {:.2} MB/s",
        compression_stats.throughput_bps / 1_000_000.0
    );

    // Decompress file in parallel
    println!("  📦 Decompressing file in parallel...");
    let (decompressed_path, decompression_stats) = compression::decompress_file_parallel(
        &compressed_file,
        Some(&decompressed_file),
        Some(algorithm),
        config,
    )?;

    println!("  📊 File Decompression Results:");
    println!("    Output file: {}", decompressed_path);
    println!("    Time: {:.2}ms", decompression_stats.operation_time_ms);
    println!(
        "    Throughput: {:.2} MB/s",
        decompression_stats.throughput_bps / 1_000_000.0
    );

    // Verify file integrity
    let restored_data = std::fs::read(&decompressed_file)?;
    assert_eq!(
        data, &restored_data,
        "File round-trip integrity check failed!"
    );
    println!("  ✅ File integrity verified - perfect round-trip!");

    Ok(())
}

fn demonstrate_algorithm_benchmarking(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🏁 Benchmarking Different Algorithms and Configurations...");

    // Use a smaller dataset for benchmarking to keep runtime reasonable
    let benchmark_data = &data[0..(data.len() / 4).min(2_500_000)]; // Max 2.5MB for benchmarking

    let algorithms = vec![
        CompressionAlgorithm::Lz4,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Gzip,
    ];

    let levels = vec![1, 6, 9];

    let configs = vec![
        ParallelCompressionConfig {
            num_threads: 2,
            chunk_size: 256 * 1024,
            buffer_size: 32 * 1024,
            enable_memory_mapping: false,
        },
        ParallelCompressionConfig {
            num_threads: 4,
            chunk_size: 512 * 1024,
            buffer_size: 64 * 1024,
            enable_memory_mapping: true,
        },
    ];

    println!(
        "  🔬 Running benchmark with {} bytes of data...",
        benchmark_data.len()
    );
    println!(
        "  📏 Testing {} algorithms × {} levels × {} configurations = {} combinations",
        algorithms.len(),
        levels.len(),
        configs.len(),
        algorithms.len() * levels.len() * configs.len()
    );

    let results = benchmark_compression_algorithms(benchmark_data, &algorithms, &levels, &configs)?;

    println!("  📊 Benchmark Results:");
    println!(
        "  {:<8} {:<6} {:<8} {:<12} {:<12} {:<12} {:<12}",
        "Algorithm", "Level", "Threads", "Comp_Speed", "Decomp_Speed", "Comp_Ratio", "Overhead"
    );
    println!("  {}", "-".repeat(80));

    for result in &results {
        println!(
            "  {:<8} {:<6} {:<8} {:<12.1} {:<12.1} {:<12.2} {:<12.2}",
            format!("{:?}", result.algorithm),
            result.level,
            result.config.num_threads,
            result.compression_speedup(),
            result.decompression_speedup(),
            result.compression_ratio,
            result.compression_overhead()
        );
    }

    // Find best configurations
    if let Some(best_speed) = results.iter().max_by(|a, b| {
        (a.compression_speedup() + a.decompression_speedup())
            .partial_cmp(&(b.compression_speedup() + b.decompression_speedup()))
            .unwrap()
    }) {
        println!("  🏆 Best overall speed: {:?} level {} with {} threads (speedup: {:.1}x comp, {:.1}x decomp)",
                 best_speed.algorithm, best_speed.level, best_speed.config.num_threads,
                 best_speed.compression_speedup(), best_speed.decompression_speedup());
    }

    if let Some(best_ratio) = results.iter().max_by(|a, b| {
        a.compression_ratio
            .partial_cmp(&b.compression_ratio)
            .unwrap()
    }) {
        println!(
            "  📐 Best compression ratio: {:?} level {} ({:.2}x compression)",
            best_ratio.algorithm, best_ratio.level, best_ratio.compression_ratio
        );
    }

    Ok(())
}
