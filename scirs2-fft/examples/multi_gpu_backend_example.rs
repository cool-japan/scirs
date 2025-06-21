//! Multi-GPU Backend Example
//!
//! This example demonstrates the unified GPU backend system supporting both
//! NVIDIA CUDA and AMD ROCm/HIP backends for sparse FFT acceleration.

use num_complex::Complex64;
use scirs2_fft::{
    sparse_fft_gpu::GPUBackend,
    sparse_fft_gpu_memory::{
        init_cuda_device, init_gpu_backend, init_hip_device, init_sycl_device, is_cuda_available,
        is_gpu_available, is_hip_available, is_sycl_available, BufferDescriptor, BufferLocation,
        BufferType,
    },
    FFTResult,
};
use std::time::Instant;

fn main() -> FFTResult<()> {
    println!("Multi-GPU Backend Example");
    println!("=========================");

    // Test GPU backend detection
    test_gpu_backend_detection()?;

    // Test memory management across backends
    test_memory_management()?;

    // Performance comparison
    test_performance_comparison()?;

    Ok(())
}

/// Test GPU backend detection and initialization
fn test_gpu_backend_detection() -> FFTResult<()> {
    println!("\n--- GPU Backend Detection ---");

    // Test individual backend availability
    println!("Testing CUDA availability...");
    let cuda_available = init_cuda_device()?;
    println!("CUDA Available: {}", cuda_available);

    println!("Testing HIP availability...");
    let hip_available = init_hip_device()?;
    println!("HIP Available: {}", hip_available);

    println!("Testing SYCL availability...");
    let sycl_available = init_sycl_device()?;
    println!("SYCL Available: {}", sycl_available);

    // Test unified GPU detection
    println!("Testing unified GPU detection...");
    let any_gpu_available = is_gpu_available();
    println!("Any GPU Available: {}", any_gpu_available);

    // Auto-detect best backend
    println!("Auto-detecting best backend...");
    let best_backend = init_gpu_backend()?;
    println!("Best Backend: {:?}", best_backend);

    match best_backend {
        GPUBackend::CUDA => println!("✓ Using NVIDIA CUDA for GPU acceleration"),
        GPUBackend::HIP => println!("✓ Using AMD ROCm/HIP for GPU acceleration"),
        GPUBackend::SYCL => println!("✓ Using SYCL for cross-platform GPU acceleration"),
        GPUBackend::CPUFallback => println!("• Using CPU fallback (no GPU detected)"),
    }

    Ok(())
}

/// Test memory management across different backends
fn test_memory_management() -> FFTResult<()> {
    println!("\n--- Memory Management Test ---");

    let backends = vec![
        ("CUDA", GPUBackend::CUDA),
        ("HIP", GPUBackend::HIP),
        ("SYCL", GPUBackend::SYCL),
        ("CPU Fallback", GPUBackend::CPUFallback),
    ];

    for (name, backend) in backends {
        println!("\nTesting {} backend:", name);

        let buffer_size = 1024;
        let element_size = std::mem::size_of::<Complex64>();

        // Test buffer creation
        match BufferDescriptor::new(
            buffer_size,
            element_size,
            BufferLocation::Device,
            BufferType::Input,
            0,
            backend,
        ) {
            Ok(buffer) => {
                println!("✓ Buffer allocation successful");
                println!("  Backend: {:?}", buffer.backend);
                println!("  Location: {:?}", buffer.location);
                println!(
                    "  Size: {} elements ({} bytes)",
                    buffer.size,
                    buffer.size * buffer.element_size
                );
                println!("  Has device memory: {}", buffer.has_device_memory());

                // Test memory transfers
                let test_data = vec![0u8; buffer_size * element_size];
                match buffer.copy_host_to_device(&test_data) {
                    Ok(_) => println!("✓ Host-to-device transfer successful"),
                    Err(e) => println!("⚠ Host-to-device transfer failed: {}", e),
                }

                let mut result_data = vec![0u8; buffer_size * element_size];
                match buffer.copy_device_to_host(&mut result_data) {
                    Ok(_) => println!("✓ Device-to-host transfer successful"),
                    Err(e) => println!("⚠ Device-to-host transfer failed: {}", e),
                }
            }
            Err(e) => {
                println!("⚠ Buffer allocation failed: {}", e);
            }
        }
    }

    Ok(())
}

/// Test performance comparison between backends
fn test_performance_comparison() -> FFTResult<()> {
    println!("\n--- Performance Comparison ---");

    let signal_sizes = vec![256, 512, 1024, 2048];
    let backends = vec![
        GPUBackend::CUDA,
        GPUBackend::HIP,
        GPUBackend::SYCL,
        GPUBackend::CPUFallback,
    ];

    for &size in &signal_sizes {
        println!("\nSignal size: {} elements", size);

        for backend in &backends {
            let backend_name = match backend {
                GPUBackend::CUDA => "CUDA",
                GPUBackend::HIP => "HIP",
                GPUBackend::SYCL => "SYCL",
                GPUBackend::CPUFallback => "CPU",
            };

            let start = Instant::now();

            match create_and_test_buffer(size, *backend) {
                Ok(stats) => {
                    let total_time = start.elapsed();
                    println!(
                        "  {}: {:?} (allocation: {:?})",
                        backend_name, total_time, stats
                    );
                }
                Err(e) => {
                    println!("  {}: Failed ({})", backend_name, e);
                }
            }
        }
    }

    Ok(())
}

/// Helper function to create and test a buffer
fn create_and_test_buffer(size: usize, backend: GPUBackend) -> FFTResult<std::time::Duration> {
    let start = Instant::now();

    let buffer = BufferDescriptor::new(
        size,
        std::mem::size_of::<Complex64>(),
        BufferLocation::Device,
        BufferType::Input,
        0,
        backend,
    )?;

    let allocation_time = start.elapsed();

    // Simulate some work
    let test_data = vec![0u8; size * std::mem::size_of::<Complex64>()];
    buffer.copy_host_to_device(&test_data)?;

    let mut result_data = vec![0u8; size * std::mem::size_of::<Complex64>()];
    buffer.copy_device_to_host(&mut result_data)?;

    Ok(allocation_time)
}

/// Display system GPU information
#[allow(dead_code)]
fn display_gpu_info() {
    println!("\n--- GPU System Information ---");

    println!("CUDA Support:");
    if is_cuda_available() {
        println!("  ✓ CUDA runtime detected");
        println!("  ✓ NVIDIA GPU acceleration available");
    } else {
        println!("  ✗ CUDA not available");
        println!("    Possible reasons:");
        println!("    - CUDA feature not enabled (--features cuda)");
        println!("    - CUDA toolkit not installed");
        println!("    - No NVIDIA GPU detected");
    }

    println!("\nHIP Support:");
    if is_hip_available() {
        println!("  ✓ HIP runtime detected");
        println!("  ✓ AMD GPU acceleration available");
    } else {
        println!("  ✗ HIP not available");
        println!("    Possible reasons:");
        println!("    - HIP feature not enabled (--features hip)");
        println!("    - ROCm toolkit not installed");
        println!("    - No AMD GPU detected");
    }

    println!("\nSYCL Support:");
    if is_sycl_available() {
        println!("  ✓ SYCL runtime detected");
        println!("  ✓ Cross-platform GPU acceleration available");
    } else {
        println!("  ✗ SYCL not available");
        println!("    Possible reasons:");
        println!("    - SYCL feature not enabled (--features sycl)");
        println!("    - SYCL toolkit not installed");
        println!("    - No compatible SYCL device detected");
    }

    println!("\nOverall GPU Status:");
    if is_gpu_available() {
        println!("  ✓ GPU acceleration available");
    } else {
        println!("  ✗ No GPU acceleration available - using CPU fallback");
    }
}
