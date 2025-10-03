//! Example demonstrating mutation of memory-mapped arrays in chunks.
//!
//! This example shows how to:
//! 1. Create memory-mapped arrays
//! 2. Apply mutations using the chunked processing approach
//! 3. Verify that changes are properly saved to disk
//!
//! Note: This example requires the `memory_efficient` feature to be enabled.
//! Run with: `cargo run --example memory_mapped_mutation --features memory_efficient`

#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{
    create_mmap, AccessMode, ChunkingStrategy, MemoryMappedChunks,
};
#[cfg(feature = "memory_efficient")]
use scirs2_core::ndarray_ext::Array1;
#[cfg(feature = "memory_efficient")]
use std::path::Path;
#[cfg(feature = "memory_efficient")]
use tempfile::tempdir;

#[cfg(not(feature = "memory_efficient"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the memory_efficient feature.");
    println!("Run with: cargo run --example memory_mapped_mutation --features memory_efficient");
}

#[cfg(feature = "memory_efficient")]
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Mapped Array Mutation Example");
    println!("====================================\n");

    // Create a temporary directory for our example files
    let temp_dir = tempdir()?;
    println!("Using temporary directory: {:?}", temp_dir.path());

    // Basic mutation example
    basic_mutation_example(temp_dir.path())?;

    // Chunked mutation example
    chunked_mutation_example(temp_dir.path())?;

    println!("\nAll examples completed successfully!");
    Ok(())
}

/// Basic example of mutating a memory-mapped array
#[cfg(feature = "memory_efficient")]
#[allow(dead_code)]
fn basic_mutation_example(tempdir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Basic Mutation Example");
    println!("-------------------------");

    // Create a 1D array of zeros
    let size = 100;
    println!("Creating a 1D array with {} zeros", size);
    let data = Array1::<i32>::zeros(size);

    // Create a memory-mapped file
    let file_path = tempdir.join("basic_mutation.bin");
    let mut mmap = create_mmap(&data, &file_path, AccessMode::Write, 0)?;
    println!("Created memory-mapped array at: {:?}", file_path);

    // Get a view of the initial data
    {
        let array_view = mmap.as_array::<scirs2_core::ndarray::Ix1>()?;
        println!(
            "Before mutation (first 10 elements): {:?}",
            array_view.slice(scirs2_core::ndarray::s![0..10])
        );
    }

    // Modify the array using process_chunks_mut (more reliable)
    mmap.process_chunks_mut(
        ChunkingStrategy::Fixed(100), // Process the entire array in one chunk since it's small
        |chunk_data, _chunk_idx| {
            // Set every 10th element to its index * 100
            for i in 0..10 {
                if i * 10 < chunk_data.len() {
                    chunk_data[i * 10] = i as i32 * 100;
                }
            }
        },
    );

    // View the array after mutation
    {
        let array_view = mmap.as_array::<scirs2_core::ndarray::Ix1>()?;
        println!(
            "After mutation (first 10 elements): {:?}",
            array_view.slice(scirs2_core::ndarray::s![0..10])
        );
        println!("Modified positions: 0, 10, 20, ..., 90");
    }
    println!("Changes flushed to disk");

    // Reopen the file to verify changes were saved
    let mmap_reopened = create_mmap(&data, &file_path, AccessMode::ReadOnly, 0)?;
    let array_reopened = mmap_reopened.as_array::<scirs2_core::ndarray::Ix1>()?;

    println!(
        "Reopened array (first 10 elements): {:?}",
        array_reopened.slice(scirs2_core::ndarray::s![0..10])
    );
    println!("Verification of selected elements:");
    for i in 0..10 {
        let pos = i * 10;
        let val = array_reopened[pos];
        println!("  Element at position {}: {}", pos, val);
        assert_eq!(
            val,
            i as i32 * 100,
            "Element at position {} should be {}",
            pos,
            i * 100
        );
    }

    Ok(())
}

/// Example of mutating a memory-mapped array using chunk-wise processing
#[cfg(feature = "memory_efficient")]
#[allow(dead_code)]
fn chunked_mutation_example(tempdir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. Chunked Mutation Example");
    println!("---------------------------");

    // Create a 1D array of zeros
    let size = 1_000_000;
    println!("Creating a 1D array with {} zeros", size);
    let data = Array1::<i32>::zeros(size);

    // Create a memory-mapped file
    let file_path = tempdir.join("chunked_mutation.bin");
    let mut mmap = create_mmap(&data, &file_path, AccessMode::Write, 0)?;
    println!("Created memory-mapped array at: {:?}", file_path);

    // Process in chunks of 100,000 elements
    let chunk_size = 100_000;
    let strategy = ChunkingStrategy::Fixed(chunk_size);
    let num_chunks = mmap.chunk_count(strategy);

    println!(
        "Modifying array in {} chunks of {} elements each",
        num_chunks, chunk_size
    );

    // Modify each chunk with a different pattern
    mmap.process_chunks_mut(strategy, |chunk_data, chunk_idx| {
        println!(
            "  Processing chunk {} (elements {}-{})",
            chunk_idx,
            chunk_idx * chunk_size,
            chunk_idx * chunk_size + chunk_data.len() - 1
        );

        // Set each element to a function of its index and chunk number
        for (i, item) in chunk_data.iter_mut().enumerate() {
            let global_idx = chunk_idx * chunk_size + i;

            if i % 10000 == 0 {
                // Only modify every 10000th element to save processing time
                *item = (global_idx / 1000) as i32;
            }
        }
    });

    println!("All chunks processed");

    // Reopen the file to verify changes were saved
    let mmap_reopened = create_mmap(&data, &file_path, AccessMode::ReadOnly, 0)?;

    // Verify some key points in the array
    println!("Verifying modifications at key points:");

    let array_reopened = mmap_reopened.as_array::<scirs2_core::ndarray::Ix1>()?;

    for check_idx in 0..num_chunks {
        let pos = check_idx * chunk_size;
        let expected = (pos / 1000) as i32;
        let actual = array_reopened[pos];

        println!(
            "  Element at position {}: {} (expected {})",
            pos, actual, expected
        );

        assert_eq!(
            actual, expected,
            "Element at position {} should be {}",
            pos, expected
        );
    }

    println!("All modifications verified successfully!");

    Ok(())
}
