//! Simple example for memory-mapped arrays
//!
//! Note: This example requires the `memory_efficient` feature to be enabled.
//! Run with: `cargo run --example memory_mapped_simple --features memory_efficient`

#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{create_mmap, AccessMode, MemoryMappedChunks};
#[cfg(feature = "memory_efficient")]
use scirs2_core::ndarray_ext::Array1;
#[cfg(feature = "memory_efficient")]
use tempfile::tempdir;

#[cfg(not(feature = "memory_efficient"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the memory_efficient feature.");
    println!("Run with: cargo run --example memory_mapped_simple --features memory_efficient");
}

#[cfg(feature = "memory_efficient")]
#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory-Mapped Array Simple Example");
    println!("=================================");

    // Create a temporary directory for our example
    let temp_dir = tempdir()?;
    let file_path = temp_dir.path().join("simple_example.bin");
    println!("Using file: {:?}", file_path);

    // Create some simple data
    let data = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("Original data: {:?}", data);

    // Create a memory-mapped array
    let mmap = create_mmap(&data, &file_path, AccessMode::Write, 0)?;
    println!("Created memory-mapped array with {} elements", mmap.size);

    // Read the array back
    let array = mmap.as_array::<scirs2_core::ndarray::Ix1>()?;
    println!("Read data: {:?}", array);

    // Process the data in chunks
    let chunk_sums = mmap.process_chunks(
        scirs2_core::memory_efficient::ChunkingStrategy::Fixed(2),
        |chunk_data, idx| {
            let sum: f64 = chunk_data.iter().sum();
            println!("Chunk {}: data = {:?}, sum = {}", idx, chunk_data, sum);
            sum
        },
    );

    println!("Chunk sums: {:?}", chunk_sums);
    println!("Total sum: {}", chunk_sums.iter().sum::<f64>());

    Ok(())
}
