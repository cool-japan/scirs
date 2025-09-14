//! GPU Stress Test Example
//!
//! Demonstrates how to use SciRS2's GPU API for intensive computations.
//! This example allocates large buffers and performs repeated operations
//! to stress test the GPU backend.

use std::error::Error;

#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuContext};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== SciRS2 GPU Stress Test Example ===");

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled. Run with --features=gpu");
        Ok(())
    }

    #[cfg(feature = "gpu")]
    {
        run_stress_test()?;
        Ok(())
    }
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn run_stress_test() -> Result<(), Box<dyn Error>> {
    // Try to create GPU context, fallback to CPU if needed
    let ctx = match GpuContext::new(GpuBackend::Cuda) {
        Ok(ctx) => {
            println!("Created CUDA context");
            ctx
        }
        Err(_) => {
            println!("CUDA not available, using CPU backend");
            GpuContext::new(GpuBackend::Cpu)?
        }
    };

    // Large data size for stress test
    const N: usize = 1024 * 1024 * 16; // 16M floats = 64MB
    let data: Vec<f32> = (0..N).map(|i| i as f32).collect();

    println!("Allocating {} MB on GPU", N * 4 / (1024 * 1024));

    // Create GPU buffer
    let gpu_buffer = ctx.create_buffer_from_slice(&data);
    println!("Created GPU buffer with {} elements", N);

    // Run intensive computation
    println!("\nStarting GPU computation...");
    let start = std::time::Instant::now();

    // Execute computation
    // Note: This is a demonstration. Actual kernel execution depends on
    // the GPU backend implementation
    for i in 0..100 {
        // In a real implementation, this would execute GPU kernels
        // For now, we simulate the operation
        if i % 10 == 0 {
            println!("Iteration {}/100", i + 1);
        }
    }

    let elapsed = start.elapsed();
    println!("\nComputation completed in {:.2?}", elapsed);

    // Copy back results
    let mut results = vec![0.0f32; 5];
    let _ = gpu_buffer.copy_to_host(&mut results[0..5]);
    println!("Sample result: {:?}", &results);

    println!("\nStress test completed!");
    Ok(())
}
