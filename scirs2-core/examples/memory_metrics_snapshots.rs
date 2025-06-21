//! # Memory Snapshots and Leak Detection
//!
//! This example demonstrates how to use memory snapshots to monitor memory usage
//! over time and detect potential memory leaks.

#[cfg(not(feature = "memory_management"))]
fn main() {
    println!("This example requires the 'memory_management' feature to be enabled.");
    println!("Run with: cargo run --example memory_metrics_snapshots --features memory_management");
}

#[cfg(feature = "memory_management")]
use std::path::Path;
#[cfg(feature = "memory_management")]
use std::thread;
#[cfg(feature = "memory_management")]
use std::time::Duration;

#[cfg(feature = "memory_management")]
use scirs2_core::memory::metrics::{
    clear_snapshots, compare_snapshots, format_bytes, format_memory_report, load_snapshots,
    reset_memory_metrics, save_snapshots, take_snapshot, track_allocation, track_deallocation,
};

#[cfg(feature = "memory_management")]
fn main() {
    println!("Memory Snapshots and Leak Detection Example");
    println!("===========================================\n");

    // Reset metrics and clear any existing snapshots
    reset_memory_metrics();
    clear_snapshots();

    // Example 1: Basic Snapshot Usage
    println!("Example 1: Basic Snapshot Usage");
    println!("-------------------------------");

    // Take an initial snapshot
    println!("Taking initial snapshot...");
    let _snapshot1 = take_snapshot("baseline", "Initial memory state");

    // Allocate some memory
    println!("\nAllocating memory...");
    let allocations = vec![
        ("ComponentA", 1024, 0x1000),
        ("ComponentA", 2048, 0x2000),
        ("ComponentB", 4096, 0x3000),
    ];

    for &(component, size, address) in &allocations {
        track_allocation(component, size, address);
        println!("  Allocated {} for {}", format_bytes(size), component);
    }

    // Take a second snapshot
    println!("\nTaking snapshot after allocations...");
    let _snapshot2 = take_snapshot("allocated", "After memory allocations");

    // Compare snapshots
    println!("\nComparing snapshots (baseline -> allocated):");
    let diff = compare_snapshots("baseline", "allocated").unwrap();
    println!("{}", diff.format());

    // Example 2: Memory Leak Detection
    println!("\n\nExample 2: Memory Leak Detection");
    println!("--------------------------------");

    // Take a snapshot before potential leak
    println!("Taking snapshot before potential leak...");
    let _snapshot_before_leak = take_snapshot("before_leak", "Before potential memory leak");

    // Simulate a memory leak (allocation without matching deallocation)
    println!("\nSimulating a memory leak...");
    println!("  Allocating memory that won't be properly deallocated");
    track_allocation("LeakyComponent", 8192, 0x4000);

    // Simulate some work
    simulate_work();

    // Take a snapshot after operations that should have cleaned up
    println!("\nTaking snapshot after operations...");
    let _snapshot_after_leak = take_snapshot(
        "after_leak",
        "After operations that should have freed memory",
    );

    // Compare snapshots to detect leaks
    println!("\nAnalyzing for memory leaks:");
    let leak_diff = compare_snapshots("before_leak", "after_leak").unwrap();

    println!("{}", leak_diff.format());

    // Check if there are potential leaks
    if leak_diff.has_potential_leaks() {
        println!("\nPOTENTIAL MEMORY LEAKS DETECTED!");
        println!("Leaking components:");
        for component in leak_diff.get_potential_leak_components() {
            println!("  - {}", component);
        }
    }

    // Display visualization if enabled
    #[cfg(feature = "memory_visualization")]
    {
        println!("\nMemory Usage Visualization:");
        println!("{}", leak_diff.visualize());
    }

    // Example 3: Saving and Loading Snapshots
    println!("\n\nExample 3: Saving and Loading Snapshots");
    println!("---------------------------------------");

    // Create a directory for snapshots if it doesn't exist
    let snapshot_dir = Path::new("/tmp/memory_snapshots");

    // Save all snapshots
    println!("Saving snapshots to {:?}...", snapshot_dir);
    match save_snapshots(snapshot_dir) {
        Ok(_) => println!("  Snapshots saved successfully"),
        Err(e) => println!("  Error saving snapshots: {}", e),
    }

    // Clear existing snapshots
    clear_snapshots();
    println!("\nCleared in-memory snapshots");

    // Load snapshots
    println!("\nLoading snapshots from {:?}...", snapshot_dir);
    match load_snapshots(snapshot_dir) {
        Ok(_) => println!("  Snapshots loaded successfully"),
        Err(e) => println!("  Error loading snapshots: {}", e),
    }

    // Check if we can still access the snapshots
    let diff = compare_snapshots("baseline", "allocated");
    if let Some(diff) = diff {
        println!("\nSuccessfully loaded and compared snapshots:");
        println!(
            "  Total current usage delta: {}",
            format_bytes(diff.current_usage_delta.unsigned_abs())
        );
    } else {
        println!("\nFailed to load snapshots correctly");
    }

    // Cleanup
    println!("\nCleaning up memory...");
    for &(component, size, address) in &allocations {
        track_deallocation(component, size, address);
    }

    println!("\nFinal memory state:");
    println!("{}", format_memory_report());
}

// Simulate doing some work
#[cfg(feature = "memory_management")]
fn simulate_work() {
    println!("  Performing operations...");

    // Allocate and deallocate to simulate normal memory usage
    for i in 0..5 {
        track_allocation("WorkComponent", 512 * (i + 1), 0x5000 + i);
        thread::sleep(Duration::from_millis(50));
        track_deallocation("WorkComponent", 512 * (i + 1), 0x5000 + i);
    }
}
