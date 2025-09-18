//! # Memory Metrics with GPU Operations
//!
//! This example demonstrates memory tracking for GPU operations using the
//! memory metrics system.

#[cfg(not(all(feature = "memory_management", feature = "gpu")))]
#[allow(dead_code)]
fn main() {
    println!("This example requires both 'memory_management' and 'gpu' features to be enabled.");
    println!("Run with: cargo run --example memory_metrics_gpu --features memory_management,gpu");
}

#[cfg(all(feature = "memory_management", feature = "gpu"))]
use scirs2_core::gpu::GpuBackend;
#[cfg(all(feature = "memory_management", feature = "gpu"))]
use scirs2_core::memory::metrics::{
    format_bytes, format_memory_report, generate_memory_report, reset_memory_metrics,
    setup_gpu_memory_tracking, TrackedGpuContext,
};
#[cfg(all(feature = "memory_management", feature = "gpu"))]
use std::time::Instant;

#[cfg(all(feature = "memory_management", feature = "gpu"))]
#[allow(dead_code)]
fn main() {
    println!("Memory Metrics with GPU Operations Example");
    println!("==========================================\n");

    // Reset metrics to start fresh
    reset_memory_metrics();

    // Set up GPU memory tracking
    setup_gpu_memory_tracking();
    println!("GPU memory tracking set up");

    // Create a tracked GPU context (use CPU backend for compatibility)
    let context = match TrackedGpuContext::with_backend(GpuBackend::Cpu, "GPUDemo") {
        Ok(ctx) => ctx,
        Err(err) => {
            println!("Failed to create GPU context: {}", err);
            return;
        }
    };

    println!(
        "Created GPU context with {} backend",
        context.backend_name()
    );

    // Example 1: Basic Buffer Operations
    println!("\nExample 1: Basic Buffer Operations");
    println!("----------------------------------");

    // Create some buffers
    let buffersizes = [1000, 5000, 10000, 50000];
    let mut buffers = Vec::new();

    for &size in &buffersizes {
        let bytes = size * std::mem::size_of::<f32>();
        println!(
            "Creating buffer with {} elements ({})",
            size,
            format_bytes(bytes)
        );

        let buffer = context.create_buffer::<f32>(size);
        buffers.push(buffer);

        // Print memory usage after each allocation
        let report = generate_memory_report();
        println!(
            "  Current GPU memory: {}",
            format_bytes(report.total_current_usage)
        );
    }

    // Print memory report
    println!("\nMemory Report after buffer allocations:");
    println!("{}", format_memory_report());

    // Release some buffers
    println!("\nReleasing first two buffers");
    buffers.drain(0..2);

    // Print memory report after releasing buffers
    println!("\nMemory Report after releasing buffers:");
    println!("{}", format_memory_report());

    // Example 2: Buffer Data Transfer
    println!("\nExample 2: Buffer Data Transfer");
    println!("-------------------------------");

    // Create a buffer with initial data
    let host_data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
    let data_size = host_data.len() * std::mem::size_of::<f32>();

    println!(
        "Creating buffer from {} elements ({})",
        host_data.len(),
        format_bytes(data_size)
    );
    let buffer = context.create_buffer_from_slice(&host_data);

    // Read back data and verify
    let device_data = buffer.to_vec();
    let matching = host_data
        .iter()
        .zip(device_data.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);

    println!(
        "Data transfer validation: {}",
        if matching { "PASSED" } else { "FAILED" }
    );

    // Example 3: Simulating a GPU Computation
    println!("\nExample 3: Simulating a GPU Computation");
    println!("--------------------------------------");

    // NOTE: simulatematrix_multiplication function not yet implemented
    // simulatematrix_multiplication(&context);
    println!("Matrix multiplication simulation not yet implemented");

    // Release all remaining buffers
    println!("\nReleasing all remaining buffers");
    drop(buffers);
    drop(buffer);

    // Final memory report
    println!("\nFinal Memory Report:");
    println!("{}", format_memory_report());
}

// Simulates a GPU matrix multiplication operation
#[cfg(all(feature = "memory_management", feature = "gpu"))]
#[allow(dead_code)]
fn gpu_operation_example(context: &TrackedGpuContext) {
    let start = Instant::now();

    // Matrix dimensions
    let m = 1000;
    let n = 1000;
    let k = 1000;

    // Create matrices
    println!("Creating matrices:");
    println!("  Matrix A: {}x{}", m, k);
    println!("  Matrix B: {}x{}", k, n);
    println!("  Matrix C: {}x{}", m, n);

    // Allocate GPU buffers for matrices
    let a_size = m * k;
    let b_size = k * n;
    let c_size = m * n;

    let buffer_a = context.create_buffer::<f32>(a_size);
    println!(
        "  Allocated buffer for matrix A: {}",
        format_bytes(a_size * std::mem::size_of::<f32>())
    );

    let buffer_b = context.create_buffer::<f32>(b_size);
    println!(
        "  Allocated buffer for matrix B: {}",
        format_bytes(b_size * std::mem::size_of::<f32>())
    );

    let buffer_c = context.create_buffer::<f32>(c_size);
    println!(
        "  Allocated buffer for matrix C: {}",
        format_bytes(c_size * std::mem::size_of::<f32>())
    );

    // Print memory usage after allocations
    let report = generate_memory_report();
    println!(
        "\nGPU memory usage for matrices: {}",
        format_bytes(report.total_current_usage)
    );

    // Initialize matrices with some data (in real code, we'd upload actual data)
    let a_data = vec![1.0f32; a_size];
    let b_data = vec![2.0f32; b_size];

    // Upload matrices to GPU
    println!("\nUploading matrices to GPU");
    buffer_a.copy_from_host(&a_data);
    buffer_b.copy_from_host(&b_data);

    // Simulate computation (would be a kernel execution in real code)
    println!("Executing matrix multiplication...");
    std::thread::sleep(std::time::Duration::from_millis(100)); // Simulate computation time

    // Download result (in real code, we'd use the actual result)
    println!("Downloading result matrix");
    let mut c_data = vec![0.0f32; c_size];
    buffer_c.copy_to_host(&mut c_data);

    // Report execution time
    let elapsed = start.elapsed();
    println!("Matrix multiplication completed in {:?}", elapsed);

    // Clean up buffers
    println!("Cleaning up GPU buffers");
    drop(buffer_a);
    drop(buffer_b);
    drop(buffer_c);

    // Check memory after cleanup
    let report = generate_memory_report();
    println!(
        "GPU memory after cleanup: {}",
        format_bytes(report.total_current_usage)
    );
}
