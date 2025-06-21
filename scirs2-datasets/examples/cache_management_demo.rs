//! Enhanced cache management utilities demonstration
//!
//! This example demonstrates the advanced cache management features including
//! platform-specific cache directories, cache size limits, offline mode, and
//! detailed cache statistics.

use scirs2_datasets::{get_cache_dir, CacheManager, DatasetCache};

fn main() {
    println!("=== Enhanced Cache Management Demonstration ===\n");

    // Demonstrate platform-specific cache directory detection
    println!("=== Platform-Specific Cache Directory =========");
    match get_cache_dir() {
        Ok(cache_dir) => {
            println!("Default cache directory: {}", cache_dir.display());
            println!("Platform: {}", std::env::consts::OS);
        }
        Err(e) => {
            println!("Error getting cache directory: {}", e);
        }
    }
    println!();

    // Demonstrate environment variable override
    println!("=== Environment Variable Configuration =========");
    println!("Set SCIRS2_CACHE_DIR to override default cache location");
    println!("Set SCIRS2_OFFLINE=true to enable offline mode");
    if let Ok(cache_env) = std::env::var("SCIRS2_CACHE_DIR") {
        println!("Custom cache directory: {}", cache_env);
    } else {
        println!("Using default cache directory");
    }

    if let Ok(offline_env) = std::env::var("SCIRS2_OFFLINE") {
        println!("Offline mode: {}", offline_env);
    } else {
        println!("Offline mode: Not set (defaults to false)");
    }
    println!();

    // Create a temporary cache for demonstration
    let temp_dir = tempfile::tempdir().unwrap();
    let demo_cache_dir = temp_dir.path().join("demo_cache");

    // Demonstrate cache with size limits
    println!("=== Cache with Size Limits =====================");
    let mut cache_manager = CacheManager::with_full_config(
        demo_cache_dir.clone(),
        50,          // 50 items in memory cache
        3600,        // 1 hour TTL
        1024 * 1024, // 1MB disk cache limit
        false,       // Not in offline mode
    );

    println!("Created cache with 1MB size limit");
    println!("Cache directory: {}", demo_cache_dir.display());

    // Add some test data to the cache
    let cache =
        DatasetCache::with_full_config(demo_cache_dir.clone(), 50, 3600, 1024 * 1024, false);

    // Write several files of different sizes
    let small_data = vec![0u8; 1024]; // 1KB
    let medium_data = vec![1u8; 10240]; // 10KB
    let large_data = vec![2u8; 102400]; // 100KB

    cache.write_cached("small_file.dat", &small_data).unwrap();
    cache.write_cached("medium_file.dat", &medium_data).unwrap();
    cache.write_cached("large_file.dat", &large_data).unwrap();

    println!("Added test files to cache");
    println!();

    // Demonstrate basic cache statistics
    println!("=== Basic Cache Statistics ====================");
    let basic_stats = cache_manager.get_stats();
    println!("Files: {}", basic_stats.file_count);
    println!("Total size: {}", basic_stats.formatted_size());
    println!();

    // Demonstrate detailed cache statistics
    println!("=== Detailed Cache Statistics ==================");
    match cache_manager.get_detailed_stats() {
        Ok(detailed_stats) => {
            println!("Cache Directory: {}", detailed_stats.cache_dir.display());
            println!(
                "Total Size: {} ({} files)",
                detailed_stats.formatted_size(),
                detailed_stats.file_count
            );
            println!("Max Size: {}", detailed_stats.formatted_max_size());
            println!("Usage: {:.1}%", detailed_stats.usage_percentage() * 100.0);
            println!(
                "Offline Mode: {}",
                if detailed_stats.offline_mode {
                    "Enabled"
                } else {
                    "Disabled"
                }
            );

            if !detailed_stats.files.is_empty() {
                println!("\nCached Files (sorted by size):");
                for file in &detailed_stats.files {
                    println!(
                        "  {} - {} (modified {})",
                        file.name,
                        file.formatted_size(),
                        file.formatted_modified()
                    );
                }
            }
        }
        Err(e) => {
            println!("Error getting detailed stats: {}", e);
        }
    }
    println!();

    // Demonstrate cache management operations
    println!("=== Cache Management Operations ===============");
    println!("Available operations:");
    println!("1. List cached files");
    let cached_files = cache_manager.list_cached_files().unwrap();
    for file in &cached_files {
        println!("   - {}", file);
    }

    println!("2. Check if specific files are cached");
    println!(
        "   small_file.dat: {}",
        cache_manager.is_cached("small_file.dat")
    );
    println!(
        "   nonexistent.dat: {}",
        cache_manager.is_cached("nonexistent.dat")
    );

    println!("3. Remove specific file");
    cache_manager.remove("medium_file.dat").unwrap();
    println!("   Removed medium_file.dat");
    println!(
        "   Files remaining: {}",
        cache_manager.list_cached_files().unwrap().len()
    );
    println!();

    // Demonstrate offline mode
    println!("=== Offline Mode Configuration ================");
    println!("Current offline mode: {}", cache_manager.is_offline());
    cache_manager.set_offline_mode(true);
    println!("Enabled offline mode: {}", cache_manager.is_offline());
    cache_manager.set_offline_mode(false);
    println!("Disabled offline mode: {}", cache_manager.is_offline());
    println!();

    // Demonstrate cache size management
    println!("=== Cache Size Management =====================");
    println!(
        "Current max cache size: {} bytes",
        cache_manager.max_cache_size()
    );
    cache_manager.set_max_cache_size(512 * 1024); // 512KB
    println!(
        "Set max cache size to: {} bytes",
        cache_manager.max_cache_size()
    );

    // Add a large file that would exceed the new limit
    let very_large_data = vec![3u8; 400 * 1024]; // 400KB
    cache
        .write_cached("very_large_file.dat", &very_large_data)
        .unwrap();

    let final_stats = cache_manager.get_detailed_stats().unwrap();
    println!(
        "Final cache size: {} (should be within limit)",
        final_stats.formatted_size()
    );
    println!(
        "Final usage: {:.1}%",
        final_stats.usage_percentage() * 100.0
    );
    println!();

    // Demonstrate cache cleanup
    println!("=== Cache Cleanup ==============================");
    println!(
        "Files before cleanup: {}",
        cache_manager.list_cached_files().unwrap().len()
    );
    cache_manager.cleanup_old_files(100 * 1024).unwrap(); // Clean up to fit 100KB
    let cleanup_stats = cache_manager.get_detailed_stats().unwrap();
    println!("Files after cleanup: {}", cleanup_stats.file_count);
    println!("Size after cleanup: {}", cleanup_stats.formatted_size());
    println!();

    // Demonstrate cache report
    println!("=== Complete Cache Report ======================");
    cache_manager.print_cache_report().unwrap();
    println!();

    // Clear all cache data
    println!("=== Cache Clearing =============================");
    cache_manager.clear_all().unwrap();
    let empty_stats = cache_manager.get_stats();
    println!("Files after clearing: {}", empty_stats.file_count);
    println!("Size after clearing: {}", empty_stats.formatted_size());
    println!();

    // Demonstrate configuration examples
    println!("=== Configuration Examples =====================");
    println!("Example configurations for different use cases:");
    println!();

    println!("1. Development (small cache, frequent cleanup):");
    println!("   CacheManager::with_full_config(cache_dir, 20, 1800, 50*1024*1024, false)");
    println!("   - 20 items in memory, 30 min TTL, 50MB disk limit");
    println!();

    println!("2. Production (large cache, longer retention):");
    println!("   CacheManager::with_full_config(cache_dir, 500, 86400, 1024*1024*1024, false)");
    println!("   - 500 items in memory, 24 hour TTL, 1GB disk limit");
    println!();

    println!("3. Offline environment:");
    println!("   CacheManager::with_full_config(cache_dir, 100, 3600, 0, true)");
    println!("   - Offline mode enabled, unlimited disk cache");
    println!();

    println!("4. Memory-constrained (minimal cache):");
    println!("   CacheManager::with_full_config(cache_dir, 10, 900, 10*1024*1024, false)");
    println!("   - 10 items in memory, 15 min TTL, 10MB disk limit");
    println!();

    println!("=== Cache Management Demo Complete =============");
}
