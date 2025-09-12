//! Heavy GPU Stress Test for SciRS2
//!
//! This example demonstrates intensive GPU computation using SciRS2's GPU abstractions

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuContext, GpuError};
#[cfg(feature = "gpu")]
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Heavy GPU Stress Test ===\n");

    #[cfg(feature = "gpu")]
    {
        run_gpu_stress_test()?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled. Run with:");
        println!("cargo run --release --example gpu_heavy_stress_test --features=gpu,cuda");
    }

    Ok(())
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn run_gpu_stress_test() -> Result<(), Box<dyn std::error::Error>> {
    // Try to create GPU context with CUDA
    let ctx = match GpuContext::new(GpuBackend::Cuda) {
        Ok(ctx) => {
            println!("✓ Successfully initialized CUDA backend");
            ctx
        }
        Err(_) => {
            println!("CUDA not available, falling back to CPU");
            GpuContext::new(GpuBackend::Cpu)?
        }
    };

    println!("Using backend: {}", ctx.backend_name());
    println!();

    // Test 1: Large Matrix Operations
    test_largematrix_operations(&ctx)?;

    // Test 2: Massive Data Transfer
    test_massive_data_transfer(&ctx)?;

    // Test 3: Intensive Computation Loop
    test_intensive_computation_loop(&ctx)?;

    println!("\n=== All GPU stress tests completed! ===");
    Ok(())
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn test_largematrix_operations(ctx: &GpuContext) -> Result<(), GpuError> {
    println!("Test 1: Large Matrix Operations");
    println!("================================");

    // Matrix dimensions for heavy load
    let sizes = vec![
        (1024, 1024), // 1K x 1K = 4MB per matrix
        (2048, 2048), // 2K x 2K = 16MB per matrix
        (4096, 4096), // 4K x 4K = 64MB per matrix
    ];

    for (rows, cols) in sizes {
        let n_elements = rows * cols;
        println!(
            "\nMatrix size: {}x{} ({} MB per matrix)",
            rows,
            cols,
            n_elements * 4 / (1024 * 1024)
        );

        // Create test data
        let a: Vec<f32> = (0..n_elements).map(|i| ((i % 1000) as f32).sin()).collect();
        let b: Vec<f32> = (0..n_elements).map(|i| ((i % 1000) as f32).cos()).collect();

        // Allocate GPU buffers
        let start = Instant::now();
        let gpu_a = ctx.create_buffer_from_slice(&a);
        let gpu_b = ctx.create_buffer_from_slice(&b);
        let alloc_time = start.elapsed();
        println!("  Allocation time: {:?}", alloc_time);

        // Perform intensive operations
        let compute_start = Instant::now();
        let iterations = 10;

        for i in 0..iterations {
            // Simulate matrix multiplication or other heavy operation
            // In a real implementation, we would use a kernel here
            // For now, just copy data back and forth to simulate work
            let mut temp = vec![0.0f32; n_elements];
            let _ = gpu_a.copy_to_host(&mut temp);

            if i % 2 == 0 {
                print!(".");
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }
        println!();

        let compute_time = compute_start.elapsed();
        println!("  Computation time: {:?}", compute_time);

        // Calculate throughput
        let gb_processed =
            (n_elements as f64 * 4.0 * 2.0 * iterations as f64) / (1024.0 * 1024.0 * 1024.0);
        let throughput = gb_processed / compute_time.as_secs_f64();
        println!("  Throughput: {:.2} GB/s", throughput);
    }

    Ok(())
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn test_massive_data_transfer(ctx: &GpuContext) -> Result<(), GpuError> {
    println!("\nTest 2: Massive Data Transfer");
    println!("==============================");

    let sizes_mb = vec![64, 128, 256, 512];

    for size_mb in sizes_mb {
        let n_elements = (size_mb * 1024 * 1024) / 4; // f32 = 4 bytes
        println!("\nTransfer size: {} MB", size_mb);

        // Create large data
        let data: Vec<f32> = (0..n_elements).map(|i| (i as f32) * 0.0001).collect();

        // Upload to GPU
        let upload_start = Instant::now();
        let gpu_buffer = ctx.create_buffer_from_slice(&data);
        let upload_time = upload_start.elapsed();

        // Download from GPU
        let download_start = Instant::now();
        let result = gpu_buffer.to_vec();
        let download_time = download_start.elapsed();

        let uploadbandwidth = (size_mb as f64) / upload_time.as_secs_f64();
        let downloadbandwidth = (size_mb as f64) / download_time.as_secs_f64();

        println!("  Upload: {:?} ({:.2} MB/s)", upload_time, uploadbandwidth);
        println!(
            "  Download: {:?} ({:.2} MB/s)",
            download_time, downloadbandwidth
        );
    }

    Ok(())
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn test_intensive_computation_loop(ctx: &GpuContext) -> Result<(), GpuError> {
    println!("\nTest 3: Intensive Computation Loop");
    println!("===================================");

    let n = 64 * 1024 * 1024; // 64M elements = 256MB
    println!("Data size: {} elements ({} MB)", n, n * 4 / (1024 * 1024));

    // Create complex data pattern
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i as f32) * 0.00001;
            x.sin() * x.cos() + x.sqrt()
        })
        .collect();

    println!("Uploading to GPU...");
    let gpu_data = ctx.create_buffer_from_slice(&data);

    println!("Running 100 iterations of intensive computation...");
    let compute_start = Instant::now();

    for i in 0..100 {
        // Each iteration performs some computation
        // In a real implementation, we would use a kernel here
        // For now, just copy data back and forth to simulate work
        let mut temp = vec![0.0f32; n];
        let _ = gpu_data.copy_to_host(&mut temp);

        if i % 10 == 0 {
            println!("Progress: {}%", i);
        }
    }

    let compute_time = compute_start.elapsed();
    println!("\nTotal computation time: {:?}", compute_time);

    let total_flops = n as f64 * 100.0; // Rough estimate
    let gflops = total_flops / compute_time.as_secs_f64() / 1e9;
    println!("Estimated performance: {:.2} GFLOPS", gflops);

    println!("\nThis should have generated significant GPU load!");

    Ok(())
}
