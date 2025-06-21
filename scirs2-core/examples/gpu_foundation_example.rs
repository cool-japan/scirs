//! GPU Foundation Example
//!
//! Demonstrates the basic GPU acceleration foundation including:
//! - Context creation with backend detection
//! - Kernel compilation and execution
//! - Buffer management and data transfer
//! - Error handling and fallback to CPU

use ndarray::Array2;
use scirs2_core::gpu::kernels::{DataType, KernelParams};
use scirs2_core::gpu::{GpuBackend, GpuContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 GPU Foundation Example ===\n");

    #[cfg(feature = "gpu")]
    run_gpu_foundation_demo()?;

    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with --features=\"gpu\" to see the GPU examples.");

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_gpu_foundation_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. GPU Backend Detection");
    println!("------------------------");

    // Test different backends
    let backends = [
        GpuBackend::Cuda,
        GpuBackend::Wgpu,
        GpuBackend::Metal,
        GpuBackend::OpenCL,
        GpuBackend::Cpu,
    ];

    for backend in backends.iter() {
        println!(
            "Backend {}: Available = {}",
            backend,
            backend.is_available()
        );
    }

    // Use the preferred backend
    let preferred = GpuBackend::preferred();
    println!("Preferred backend: {}", preferred);

    println!("\n2. GPU Context Creation");
    println!("-----------------------");

    let ctx = match GpuContext::new(preferred) {
        Ok(ctx) => {
            println!(
                "✓ Successfully created GPU context with {} backend",
                ctx.backend_name()
            );
            ctx
        }
        Err(e) => {
            println!("✗ Failed to create GPU context: {}", e);
            println!("Falling back to CPU backend...");
            GpuContext::new(GpuBackend::Cpu)?
        }
    };

    println!("\n3. Memory Management");
    println!("--------------------");

    // Create test data
    let size = 1024;
    let host_data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();

    // Allocate GPU buffer
    let buffer = ctx.create_buffer_from_slice(&host_data);
    println!("✓ Created GPU buffer with {} elements", buffer.len());

    // Test buffer operations
    let retrieved_data = buffer.to_vec();
    let data_matches = host_data
        .iter()
        .zip(retrieved_data.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    if data_matches {
        println!("✓ Buffer data transfer successful");
    } else {
        println!("✗ Buffer data transfer failed");
    }

    println!("\n4. Kernel Registry");
    println!("------------------");

    // Test kernel retrieval
    let kernel_names = ["gemm_standard", "sum_kernel", "relu_kernel"];

    for name in kernel_names.iter() {
        match ctx.get_kernel(name) {
            Ok(kernel) => {
                println!("✓ Found kernel: {}", name);

                // Test kernel compilation (this will use CPU fallback)
                // In a real GPU implementation, this would compile actual GPU code
                kernel.set_f32("test_param", 1.0);
                kernel.dispatch([1, 1, 1]);
                println!("  - Kernel executed successfully");
            }
            Err(e) => {
                println!("✗ Kernel '{}' not found: {}", name, e);
            }
        }
    }

    println!("\n5. Kernel Specialization");
    println!("-------------------------");

    // Create parameters for GEMM specialization
    let params = KernelParams::new(DataType::Float32)
        .with_input_dims(vec![128, 256])
        .with_output_dims(vec![128, 512])
        .with_numeric_param("alpha", 1.0)
        .with_numeric_param("beta", 0.0);

    match ctx.get_specialized_kernel("gemm_standard", &params) {
        Ok(_kernel) => {
            println!("✓ Created specialized GEMM kernel for 128x256 * 256x512 matrices");
            println!("  - Kernel compiled successfully");
            println!("  - Optimized for the specified matrix dimensions");
            println!("  - Ready for execution");
        }
        Err(e) => {
            println!("✗ Failed to create specialized kernel: {}", e);
        }
    }

    println!("\n6. Matrix Operations Example");
    println!("-----------------------------");

    // Create small matrices for testing
    let a = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f32);
    let b = Array2::from_shape_fn((4, 4), |(i, j)| ((i + j) as f32).cos());

    println!("Matrix A (4x4):");
    for row in a.outer_iter() {
        println!("  {:?}", row.to_vec());
    }

    println!("Matrix B (4x4):");
    for row in b.outer_iter() {
        println!("  {:?}", row.to_vec());
    }

    // In a real implementation, we would:
    // 1. Copy matrices to GPU
    // 2. Execute GEMM kernel
    // 3. Copy result back to CPU
    // For now, we'll just demonstrate the CPU fallback
    let c = a.dot(&b);

    println!("Result C = A * B (4x4) [CPU computation]:");
    for row in c.outer_iter() {
        println!(
            "  {:?}",
            row.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>()
        );
    }

    println!("\n7. Error Handling");
    println!("-----------------");

    // Test error conditions
    match ctx.get_kernel("nonexistent_kernel") {
        Ok(_) => println!("✗ Unexpected success for nonexistent kernel"),
        Err(e) => println!("✓ Proper error handling: {}", e),
    }

    // Test backend not available
    match GpuContext::new(GpuBackend::Metal) {
        Ok(_) => println!("✓ Metal backend available"),
        Err(e) => println!("✓ Proper error for unavailable backend: {}", e),
    }

    println!("\n=== GPU Foundation Demo Complete ===");
    println!("The GPU foundation provides:");
    println!("• Multi-backend support (CUDA, WebGPU, Metal, OpenCL, CPU fallback)");
    println!("• Kernel library with optimized implementations");
    println!("• Automatic kernel specialization");
    println!("• Memory management abstraction");
    println!("• Comprehensive error handling");
    println!("\nNext steps for production:");
    println!("• Implement actual CUDA/WebGPU/Metal backends");
    println!("• Add JIT compilation for dynamic kernels");
    println!("• Integrate with linear algebra operations");
    println!("• Add performance monitoring and auto-tuning");

    Ok(())
}
