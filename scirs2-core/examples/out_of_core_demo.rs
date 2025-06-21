//! Example demonstrating out-of-core memory management functionality
//!
//! This example shows how to work with datasets larger than available memory
//! using the out-of-core array system with automatic dirty chunk tracking
//! and persistence.

#[cfg(not(feature = "memory_management"))]
fn main() {
    println!("This example requires the 'memory_management' feature to be enabled.");
    println!("Run with: cargo run --example out_of_core_demo --features memory_management");
}

#[cfg(feature = "memory_management")]
use ndarray::{Array, IxDyn};
#[cfg(feature = "memory_management")]
use scirs2_core::memory::out_of_core::{
    CachePolicy, FileStorageBackend, OutOfCoreArray, OutOfCoreConfig, OutOfCoreManager,
};
#[cfg(feature = "memory_management")]
use std::sync::Arc;
#[cfg(feature = "memory_management")]
use tempfile::TempDir;

#[cfg(feature = "memory_management")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Out-of-Core Memory Management Demo ===\n");

    // Create a temporary directory for storage
    let temp_dir = TempDir::new()?;
    println!("1. Created temporary storage at: {:?}", temp_dir.path());

    // Create storage backend
    let storage = Arc::new(FileStorageBackend::new(temp_dir.path())?);

    // Configure out-of-core processing
    let config = OutOfCoreConfig {
        chunk_shape: vec![100, 100],    // 100x100 chunks
        max_cached_chunks: 4,           // Keep up to 4 chunks in memory
        max_cache_memory: 10_000_000,   // 10MB cache limit
        cache_policy: CachePolicy::Lru, // Least Recently Used eviction
        enable_prefetching: true,
        prefetch_count: 2,
        ..Default::default()
    };

    println!("\n2. Configuration:");
    println!("   - Chunk shape: {:?}", config.chunk_shape);
    println!("   - Max cached chunks: {}", config.max_cached_chunks);
    println!("   - Cache policy: {:?}", config.cache_policy);

    // Create an out-of-core array
    let array_shape = vec![1000, 1000]; // 1M elements
    let array = OutOfCoreArray::<f64>::new(
        "large_dataset".to_string(),
        array_shape.clone(),
        storage,
        config.clone(),
    );

    println!("\n3. Created out-of-core array:");
    println!("   - Shape: {:?}", array.shape());
    println!("   - Total elements: {}", array.len());

    // Get array statistics before operations
    println!("\n4. Array statistics:");
    let stats = array.get_statistics();
    println!("   - Total chunks: {}", stats.total_chunks);
    println!("   - Cached chunks: {}", stats.cache_stats.cached_chunks);

    // Note: Chunks are created on-demand, not upfront
    println!("\n5. Chunk processing would create chunks on-demand");
    println!("   - Total array shape: {:?}", array.shape());
    println!("   - Chunk shape: {:?}", config.chunk_shape);
    println!("   - Would create 10x10 = 100 chunks total");

    // Demonstrate the manager API
    println!("\n6. Using OutOfCoreManager:");
    let manager = OutOfCoreManager::default();

    // Create an array through the manager
    let managed_array: Arc<OutOfCoreArray<f64>> =
        manager.create_array("managed_dataset".to_string(), vec![500, 500], None, None)?;

    println!("   - Created managed array: {:?}", managed_array.shape());

    // List all arrays
    let arrays = manager.list_arrays();
    println!("   - Arrays managed: {:?}", arrays);

    // Convert in-memory array to out-of-core
    println!("\n7. Converting in-memory array to out-of-core:");
    let in_memory = Array::<f64, IxDyn>::from_elem(IxDyn(&[200, 200]), std::f64::consts::PI);

    let out_of_core = scirs2_core::memory::out_of_core::utils::convert_to_out_of_core(
        &in_memory,
        "converted_array".to_string(),
        vec![50, 50], // Smaller chunks
    )?;

    println!("   - Converted array shape: {:?}", out_of_core.shape());
    println!("   - Array successfully persisted to disk");

    // Demonstrate memory efficiency
    println!("\n8. Memory efficiency demonstration:");
    let stats = out_of_core.get_statistics();
    println!("   - Array ID: {}", stats.array_id);
    println!("   - Total elements: {}", stats.total_elements);
    println!("   - Total chunks: {}", stats.total_chunks);
    println!("   - Cached chunks: {}", stats.cache_stats.cached_chunks);
    println!(
        "   - Memory usage: {} bytes",
        stats.cache_stats.memory_usage
    );

    println!("\n✅ Out-of-core memory management demonstration complete!");
    println!("\nKey benefits:");
    println!("- Process datasets larger than available RAM");
    println!("- Automatic dirty chunk tracking and persistence");
    println!("- Configurable caching with multiple eviction policies");
    println!("- Seamless integration with ndarray");

    Ok(())
}
