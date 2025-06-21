//! # Memory Metrics with TrackedBufferPool
//!
//! This example demonstrates how to use the TrackedBufferPool to automatically
//! track memory allocations and deallocations in buffer pool operations.

#[cfg(not(feature = "memory_management"))]
fn main() {
    println!("This example requires the 'memory_management' feature to be enabled.");
    println!(
        "Run with: cargo run --example memory_metrics_bufferpool --features memory_management"
    );
}

#[cfg(feature = "memory_management")]
use scirs2_core::memory::metrics::{
    format_bytes, format_memory_report, reset_memory_metrics, TrackedBufferPool,
};
#[cfg(feature = "memory_management")]
use std::thread;
#[cfg(feature = "memory_management")]
use std::time::Duration;

#[cfg(feature = "memory_management")]
fn main() {
    println!("Memory Metrics with TrackedBufferPool Example");
    println!("=============================================\n");

    // Reset metrics to start fresh
    reset_memory_metrics();

    // Create a tracked buffer pool
    let mut pool = TrackedBufferPool::<f64>::new("NumericalComputation");
    println!("Created TrackedBufferPool for numerical computations");

    // Acquire and release vectors of different sizes
    println!("\nAcquiring and releasing buffers:");
    for i in 1..=5 {
        let size = i * 1000;

        // Acquire a vector
        let vec = pool.acquire_vec(size);
        println!(
            "  Acquired vector of size {} ({})",
            size,
            format_bytes(size * 8)
        );

        // Simulate some computation
        thread::sleep(Duration::from_millis(100));

        // Release the vector back to the pool
        pool.release_vec(vec);
        println!("  Released vector of size {}", size);

        // Print the current memory report
        if i == 1 || i == 5 {
            println!("\nMemory report after iteration {}:", i);
            println!("{}", format_memory_report());
        }
    }

    // Working with ndarray objects
    println!("\nWorking with ndarray objects:");

    for i in 1..=3 {
        let size = i * 5000;

        // Acquire an array
        let array = pool.acquire_array(size);
        println!(
            "  Acquired array of size {} ({})",
            size,
            format_bytes(size * 8)
        );

        // Simulate computation
        thread::sleep(Duration::from_millis(200));

        // Release the array back to the pool
        pool.release_array(array);
        println!("  Released array of size {}", size);
    }

    // Final memory report
    println!("\nFinal memory report:");
    println!("{}", format_memory_report());

    // The memory pool operations should show zero net memory usage
    // since all buffers were returned to the pool
}
