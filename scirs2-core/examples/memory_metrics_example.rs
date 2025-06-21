//! # Memory Metrics Example
//!
//! This example demonstrates the use of the memory metrics system to track
//! and analyze memory usage in a scientific computing application.

#[cfg(not(feature = "memory_management"))]
fn main() {
    println!("This example requires the 'memory_management' feature to be enabled.");
    println!("Run with: cargo run --example memory_metrics_example --features memory_management");
}

#[cfg(feature = "memory_management")]
use ndarray::{Array2, Array3};
#[cfg(feature = "memory_management")]
use scirs2_core::memory::metrics::{
    format_bytes, format_memory_report, generate_memory_report, track_allocation,
    track_deallocation, track_resize, MemoryEvent, MemoryEventType, MemoryMetricsCollector,
    MemoryMetricsConfig,
};
#[cfg(feature = "memory_management")]
use std::thread;
#[cfg(feature = "memory_management")]
use std::time::Duration;

#[cfg(feature = "memory_management")]
fn main() {
    println!("Memory Metrics Example");
    println!("======================\n");

    // Example 1: Using the global collector for simple tracking
    println!("Example 1: Basic Memory Tracking");
    println!("--------------------------------");

    // Track some memory operations
    for i in 0..5 {
        let size = 1024 * (i + 1);
        track_allocation("Computation", size, 0x1000 + i);
        println!("Allocated {} for Computation", format_bytes(size));

        // Simulate some work
        thread::sleep(Duration::from_millis(100));
    }

    // Deallocate some memory
    track_deallocation("Computation", 1024, 0x1000);
    println!("Deallocated 1024 bytes from Computation");

    // Resize some memory
    track_resize("Computation", 4096, 2048, 0x1002);
    println!("Resized memory from 2048 to 4096 bytes");

    // Print a report
    println!("\nMemory Report:");
    println!("{}", format_memory_report());

    // Example 2: Creating a custom collector for a specific component
    println!("\n\nExample 2: Component-Specific Tracking");
    println!("--------------------------------------");

    // Create a custom configuration with sampling
    let config = MemoryMetricsConfig {
        enabled: true,
        capture_call_stacks: false,
        max_events: 100,
        real_time_aggregation: true,
        sampling_rate: 0.75, // Only track 75% of events
    };

    // Create a collector
    let collector = MemoryMetricsCollector::new(config);

    // Track array allocations for matrix operations
    println!("Creating and tracking matrices...");
    simulate_matrix_operations(&collector);

    // Get and print the report
    let report = collector.generate_report();
    println!("\nMatrix Operations Memory Report:");
    println!("{}", report.format());

    // Example 3: Tracking real ndarray allocations
    println!("\n\nExample 3: Tracking ndarray Allocations");
    println!("-----------------------------------------");

    // Reset the global collector
    scirs2_core::memory::metrics::reset_memory_metrics();

    // Create and track arrays
    println!("Creating arrays and tracking memory usage...");
    create_arrays();

    // Print the final report
    println!("\nFinal Memory Report:");
    println!("{}", format_memory_report());

    // Get the report as JSON for machine processing
    let report = generate_memory_report();

    #[cfg(feature = "memory_metrics")]
    {
        // If memory_metrics feature is enabled, we get a serde_json::Value
        let json = report.to_json();
        println!("\nJSON Report Format (excerpt):");
        println!("{{");
        println!(
            "  \"total_current_usage\": {},",
            json["total_current_usage"]
        );
        println!("  \"total_peak_usage\": {},", json["total_peak_usage"]);
        println!(
            "  \"total_allocation_count\": {}",
            json["total_allocation_count"]
        );
        println!("  ...");
        println!("}}");
    }

    #[cfg(not(feature = "memory_metrics"))]
    {
        // If memory_metrics feature is disabled, we get a String
        let json = report.to_json();
        println!("\nJSON Report Format (feature disabled):");
        println!("{}", json);
    }
}

// Simulate matrix operations with memory tracking
#[cfg(feature = "memory_management")]
fn simulate_matrix_operations(collector: &MemoryMetricsCollector) {
    // Track memory for matrix A
    let matrix_a_size = 8 * 1024 * 1024; // 8MB
    collector.record_event(
        MemoryEvent::new(
            MemoryEventType::Allocation,
            "MatrixOperations",
            matrix_a_size,
            0xa000,
        )
        .with_context("matrix_multiply")
        .with_metadata("matrix", "A"),
    );

    println!("Allocated Matrix A: {}", format_bytes(matrix_a_size));

    // Track memory for matrix B
    let matrix_b_size = 4 * 1024 * 1024; // 4MB
    collector.record_event(
        MemoryEvent::new(
            MemoryEventType::Allocation,
            "MatrixOperations",
            matrix_b_size,
            0xb000,
        )
        .with_context("matrix_multiply")
        .with_metadata("matrix", "B"),
    );

    println!("Allocated Matrix B: {}", format_bytes(matrix_b_size));

    // Simulate computation
    thread::sleep(Duration::from_millis(200));

    // Track memory for result matrix C
    let matrix_c_size = 12 * 1024 * 1024; // 12MB
    collector.record_event(
        MemoryEvent::new(
            MemoryEventType::Allocation,
            "MatrixOperations",
            matrix_c_size,
            0xc000,
        )
        .with_context("matrix_multiply")
        .with_metadata("matrix", "C"),
    );

    println!(
        "Allocated Matrix C (result): {}",
        format_bytes(matrix_c_size)
    );

    // Simulate more computation
    thread::sleep(Duration::from_millis(200));

    // Deallocate temporary matrices
    collector.record_event(
        MemoryEvent::new(
            MemoryEventType::Deallocation,
            "MatrixOperations",
            matrix_a_size,
            0xa000,
        )
        .with_context("matrix_multiply")
        .with_metadata("matrix", "A"),
    );

    collector.record_event(
        MemoryEvent::new(
            MemoryEventType::Deallocation,
            "MatrixOperations",
            matrix_b_size,
            0xb000,
        )
        .with_context("matrix_multiply")
        .with_metadata("matrix", "B"),
    );

    println!("Deallocated Matrix A and B after computation");
}

// Create and track actual ndarray arrays
#[cfg(feature = "memory_management")]
fn create_arrays() {
    // Create a 2D array (1000 x 1000 f64 values = ~8MB)
    let dims = (1000, 1000);
    let size_2d = 8 * dims.0 * dims.1;
    track_allocation("ndarray", size_2d, 0xd000);
    let _array_2d = Array2::<f64>::zeros(dims);
    println!("Created 2D array: {}", format_bytes(size_2d));

    // Create a 3D array (100 x 100 x 100 f64 values = ~8MB)
    let dims_3d = (100, 100, 100);
    let size_3d = 8 * dims_3d.0 * dims_3d.1 * dims_3d.2;
    track_allocation("ndarray", size_3d, 0xe000);
    let _array_3d = Array3::<f64>::zeros(dims_3d);
    println!("Created 3D array: {}", format_bytes(size_3d));

    // Simulate some computation
    thread::sleep(Duration::from_millis(200));

    // Deallocate the 2D array
    track_deallocation("ndarray", size_2d, 0xd000);
    println!("Deallocated 2D array");

    // Later the 3D array will be deallocated automatically,
    // but we won't track it explicitly in this example
}
