//! Batch operations demonstration
//!
//! This example demonstrates the batch processing utilities and selective cache management
//! features of the scirs2-datasets caching system.

use scirs2_datasets::{BatchOperations, CacheManager};
use std::time::Duration;

fn main() {
    println!("=== Batch Operations Demonstration ===\n");

    // Create a cache manager for demonstration
    let cache_dir = scirs2_datasets::get_cache_dir().unwrap();
    let cache_manager = CacheManager::new(cache_dir, 100, 3600);

    println!("=== Setting up Batch Operations Manager =====");
    let batch_ops = BatchOperations::new(cache_manager)
        .with_parallel(false) // Use sequential for deterministic demo output
        .with_retry_config(2, Duration::from_millis(500));

    println!("Batch operations manager configured:");
    println!("  - Parallel processing: disabled (for demo)");
    println!("  - Max retries: 2");
    println!("  - Retry delay: 500ms");

    // Demonstrate cache setup with sample data
    println!("\n=== Sample Data Setup ========================");
    setup_sample_cache_data(&batch_ops);

    // Demonstrate batch statistics
    println!("\n=== Cache Statistics ==========================");
    demonstrate_cache_statistics(&batch_ops);

    // Demonstrate batch processing
    println!("\n=== Batch Processing =========================");
    demonstrate_batch_processing(&batch_ops);

    // Demonstrate selective cleanup
    println!("\n=== Selective Cache Cleanup ==================");
    demonstrate_selective_cleanup(&batch_ops);

    // Show final cache state
    println!("\n=== Final Cache State =========================");
    show_final_cache_state(&batch_ops);

    // Performance considerations
    println!("\n=== Performance Considerations ================");
    demonstrate_performance_features();

    println!("\n=== Batch Operations Demo Complete ===========");
}

fn setup_sample_cache_data(batch_ops: &BatchOperations) {
    println!("Creating sample cached datasets...");

    // Create various types of sample data
    let sample_datasets = [
        ("iris_processed.csv", create_csv_data()),
        ("experiment_001.json", create_json_data()),
        ("temp_file_001.tmp", create_binary_data(100)),
        ("temp_file_002.tmp", create_binary_data(200)),
        ("large_dataset.dat", create_binary_data(1024)),
        ("model_weights.bin", create_binary_data(512)),
        ("results_summary.txt", create_text_data()),
    ];

    for (name, data) in sample_datasets {
        if let Err(e) = batch_ops.write_cached(name, &data) {
            println!("  Warning: Failed to cache {}: {}", name, e);
        } else {
            println!("  ✓ Cached {} ({} bytes)", name, data.len());
        }
    }
}

fn demonstrate_cache_statistics(batch_ops: &BatchOperations) {
    match batch_ops.get_cache_statistics() {
        Ok(result) => {
            println!("{}", result.summary());
            println!("Cache analysis:");
            println!("  - Files processed: {}", result.success_count);
            println!("  - Total cache size: {}", format_bytes(result.total_bytes));
            println!(
                "  - Analysis time: {:.2}ms",
                result.elapsed_time.as_millis()
            );

            if result.failure_count > 0 {
                println!("  - Failed files: {}", result.failure_count);
                for (file, error) in &result.failures {
                    println!("    • {}: {}", file, error);
                }
            }
        }
        Err(e) => println!("Failed to get cache statistics: {}", e),
    }
}

fn demonstrate_batch_processing(batch_ops: &BatchOperations) {
    println!("Processing multiple cached files in batch...");

    // Get list of cached files
    let cached_files = batch_ops.list_cached_files().unwrap_or_default();

    if cached_files.is_empty() {
        println!("No cached files found for processing");
        return;
    }

    println!("Found {} files to process", cached_files.len());

    // Example 1: Validate file sizes
    println!("\n1. File Size Validation:");
    let result = batch_ops.batch_process(&cached_files, |name, data| {
        if data.len() < 10 {
            Err(format!("File {} too small ({} bytes)", name, data.len()))
        } else {
            Ok(data.len())
        }
    });

    println!("   {}", result.summary());
    if result.failure_count > 0 {
        for (file, error) in &result.failures {
            println!("   ⚠ {}: {}", file, error);
        }
    }

    // Example 2: Content type detection
    println!("\n2. Content Type Detection:");
    let result = batch_ops.batch_process(&cached_files, |name, data| {
        let content_type = detect_content_type(name, data);
        println!("   {} -> {}", name, content_type);
        Ok::<String, String>(content_type)
    });

    println!("   {}", result.summary());

    // Example 3: Data integrity check
    println!("\n3. Data Integrity Check:");
    let result = batch_ops.batch_process(&cached_files, |name, data| {
        // Simple check: ensure data is not all zeros
        let all_zeros = data.iter().all(|&b| b == 0);
        if all_zeros && data.len() > 100 {
            Err("Suspicious: large file with all zeros".to_string())
        } else {
            let checksum = data.iter().map(|&b| b as u32).sum::<u32>();
            println!("   {} checksum: {}", name, checksum);
            Ok(checksum)
        }
    });

    println!("   {}", result.summary());
}

fn demonstrate_selective_cleanup(batch_ops: &BatchOperations) {
    println!("Demonstrating selective cache cleanup...");

    // Show current cache state
    let initial_stats = batch_ops.get_cache_statistics().unwrap();
    println!(
        "Before cleanup: {} files, {}",
        initial_stats.success_count,
        format_bytes(initial_stats.total_bytes)
    );

    // Example 1: Clean up temporary files
    println!("\n1. Cleaning up temporary files (*.tmp):");
    match batch_ops.selective_cleanup(&["*.tmp"], None) {
        Ok(result) => {
            println!("   {}", result.summary());
            if result.success_count > 0 {
                println!("   Removed {} temporary files", result.success_count);
            }
        }
        Err(e) => println!("   Failed: {}", e),
    }

    // Example 2: Clean up old files (demo with 0 days to show functionality)
    println!("\n2. Age-based cleanup (files older than 0 days - for demo):");
    match batch_ops.selective_cleanup(&["*"], Some(0)) {
        Ok(result) => {
            println!("   {}", result.summary());
            println!("   (Note: Using 0 days for demonstration - all files are 'old')");
        }
        Err(e) => println!("   Failed: {}", e),
    }

    // Show final cache state
    let final_stats = batch_ops.get_cache_statistics().unwrap_or_default();
    println!(
        "\nAfter cleanup: {} files, {}",
        final_stats.success_count,
        format_bytes(final_stats.total_bytes)
    );

    let freed_space = initial_stats
        .total_bytes
        .saturating_sub(final_stats.total_bytes);
    if freed_space > 0 {
        println!("Space freed: {}", format_bytes(freed_space));
    }
}

fn show_final_cache_state(batch_ops: &BatchOperations) {
    println!("Final cache contents:");

    match batch_ops.list_cached_files() {
        Ok(files) => {
            if files.is_empty() {
                println!("  Cache is empty");
            } else {
                for file in files {
                    if let Ok(data) = batch_ops.read_cached(&file) {
                        println!("  {} ({} bytes)", file, data.len());
                    }
                }
            }
        }
        Err(e) => println!("  Failed to list files: {}", e),
    }

    // Print detailed cache report
    if let Err(e) = batch_ops.print_cache_report() {
        println!("Failed to generate cache report: {}", e);
    }
}

fn demonstrate_performance_features() {
    println!("Performance and configuration options:");

    println!("\n**Parallel vs Sequential Processing:**");
    println!("- Parallel: Faster for I/O-bound operations, multiple files");
    println!("- Sequential: Better for CPU-bound operations, deterministic order");
    println!("- Configure with: batch_ops.with_parallel(true/false)");

    println!("\n**Retry Configuration:**");
    println!("- Configurable retry count and delay for robust operations");
    println!("- Useful for network downloads or temporary file locks");
    println!("- Configure with: batch_ops.with_retry_config(max_retries, delay)");

    println!("\n**Selective Cleanup Patterns:**");
    println!("- Glob patterns: *.tmp, *.cache, dataset_*");
    println!("- Age-based cleanup: Remove files older than N days");
    println!("- Pattern examples:");
    println!("  • '*.tmp' - all temporary files");
    println!("  • 'old_*' - files starting with 'old_'");
    println!("  • '*_backup' - files ending with '_backup'");

    println!("\n**Use Cases:**");
    println!("- **Batch Downloads**: Download multiple datasets efficiently");
    println!("- **Data Validation**: Verify integrity of multiple cached files");
    println!("- **Cleanup Operations**: Remove outdated or temporary files");
    println!("- **Data Processing**: Apply transformations to multiple datasets");
    println!("- **Cache Maintenance**: Monitor and manage cache size and content");
}

// Helper functions for creating sample data

fn create_csv_data() -> Vec<u8> {
    "sepal_length,sepal_width,petal_length,petal_width,species\n\
     5.1,3.5,1.4,0.2,setosa\n\
     4.9,3.0,1.4,0.2,setosa\n\
     4.7,3.2,1.3,0.2,setosa\n"
        .as_bytes()
        .to_vec()
}

fn create_json_data() -> Vec<u8> {
    r#"{"experiment_id": "001", "results": {"accuracy": 0.95, "precision": 0.92}, "timestamp": "2024-01-01T12:00:00Z"}"#
        .as_bytes().to_vec()
}

fn create_binary_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

fn create_text_data() -> Vec<u8> {
    "Experimental Results Summary\n\
     ============================\n\
     Total samples: 1000\n\
     Success rate: 95.2%\n\
     Processing time: 12.3s\n"
        .as_bytes()
        .to_vec()
}

fn detect_content_type(name: &str, data: &[u8]) -> String {
    if name.ends_with(".csv") {
        "text/csv".to_string()
    } else if name.ends_with(".json") {
        "application/json".to_string()
    } else if name.ends_with(".txt") {
        "text/plain".to_string()
    } else if data.iter().all(|&b| b.is_ascii()) {
        "text/plain (detected)".to_string()
    } else {
        "application/octet-stream".to_string()
    }
}

fn format_bytes(bytes: u64) -> String {
    let size = bytes as f64;
    if size < 1024.0 {
        format!("{} B", size)
    } else if size < 1024.0 * 1024.0 {
        format!("{:.1} KB", size / 1024.0)
    } else if size < 1024.0 * 1024.0 * 1024.0 {
        format!("{:.1} MB", size / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", size / (1024.0 * 1024.0 * 1024.0))
    }
}
