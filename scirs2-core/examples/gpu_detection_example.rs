//! GPU Detection Example
//!
//! Demonstrates GPU backend detection and device information gathering

use scirs2_core::gpu::backends::{
    check_backend_installation, detect_gpu_backends, get_device_info,
};
use scirs2_core::gpu::GpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Detection Example ===\n");

    #[cfg(feature = "gpu")]
    run_detection_demo()?;

    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with --features=\"gpu\" to see the detection example.");

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_detection_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Backend Installation Check");
    println!("-----------------------------");

    let backends = [
        GpuBackend::Cuda,
        GpuBackend::Metal,
        GpuBackend::OpenCL,
        GpuBackend::Wgpu,
        GpuBackend::Cpu,
    ];

    for backend in backends.iter() {
        match check_backend_installation(*backend) {
            Ok(installed) => {
                let status = if installed {
                    "✓ Installed"
                } else {
                    "✗ Not installed"
                };
                println!("{}: {}", backend, status);
            }
            Err(e) => {
                println!("{}: ✗ Error checking - {}", backend, e);
            }
        }
    }

    println!("\n2. GPU Device Detection");
    println!("-----------------------");

    let detection_result = detect_gpu_backends();

    println!(
        "Recommended backend: {}",
        detection_result.recommended_backend
    );
    println!("Detected {} GPU device(s):", detection_result.devices.len());

    for (i, device) in detection_result.devices.iter().enumerate() {
        println!("\nDevice {}: {}", i, device.device_name);
        println!("  Backend: {}", device.backend);

        if let Some(memory) = device.memory_bytes {
            let memory_gb = memory as f64 / (1024.0 * 1024.0 * 1024.0);
            println!("  Memory: {:.1} GB", memory_gb);
        } else {
            println!("  Memory: Unknown");
        }

        if let Some(ref capability) = device.compute_capability {
            println!("  Compute capability: {}", capability);
        }

        println!(
            "  Tensor support: {}",
            if device.supports_tensors { "Yes" } else { "No" }
        );
    }

    println!("\n3. Specific Device Information");
    println!("------------------------------");

    // Try to get detailed info for the first CUDA device (if available)
    match get_device_info(GpuBackend::Cuda, 0) {
        Ok(device) => {
            println!("CUDA Device 0 details:");
            println!("  Name: {}", device.device_name);
            if let Some(memory) = device.memory_bytes {
                println!("  Memory: {} bytes", memory);
            }
            if let Some(ref capability) = device.compute_capability {
                println!("  Compute capability: {}", capability);
            }
        }
        Err(e) => {
            println!("No CUDA device found: {}", e);
        }
    }

    println!("\n4. Backend Recommendations");
    println!("---------------------------");

    let cuda_devices = detection_result
        .devices
        .iter()
        .filter(|d| d.backend == GpuBackend::Cuda)
        .count();
    let metal_devices = detection_result
        .devices
        .iter()
        .filter(|d| d.backend == GpuBackend::Metal)
        .count();
    let opencl_devices = detection_result
        .devices
        .iter()
        .filter(|d| d.backend == GpuBackend::OpenCL)
        .count();

    println!("Backend suitability for scientific computing:");

    if cuda_devices > 0 {
        println!("✓ CUDA: Excellent (found {} device(s))", cuda_devices);
        println!("  - Best performance for deep learning and HPC");
        println!("  - Extensive library ecosystem (cuBLAS, cuFFT, etc.)");
        println!("  - Mature compute platform");
    } else {
        println!("✗ CUDA: Not available");
    }

    if metal_devices > 0 {
        println!("✓ Metal: Good (found {} device(s))", metal_devices);
        println!("  - Excellent integration with Apple hardware");
        println!("  - Unified memory architecture benefits");
        println!("  - Good for machine learning on Apple Silicon");
    } else {
        println!("✗ Metal: Not available (requires macOS)");
    }

    if opencl_devices > 0 {
        println!("○ OpenCL: Fair (found {} device(s))", opencl_devices);
        println!("  - Cross-platform compatibility");
        println!("  - Good for basic parallel computing");
        println!("  - Limited ecosystem compared to CUDA");
    } else {
        println!("✗ OpenCL: Not available");
    }

    println!("○ WebGPU: Fair (browser-compatible)");
    println!("  - Excellent for web deployment");
    println!("  - Growing ecosystem");
    println!("  - Good for lightweight computations");

    println!("✓ CPU: Always available (fallback)");
    println!("  - Guaranteed compatibility");
    println!("  - Good for development and testing");
    println!("  - SIMD acceleration available");

    println!("\n5. Performance Characteristics");
    println!("------------------------------");

    println!("Expected relative performance for matrix operations:");

    if cuda_devices > 0 {
        let has_tensor_cores = detection_result
            .devices
            .iter()
            .any(|d| d.backend == GpuBackend::Cuda && d.supports_tensors);

        if has_tensor_cores {
            println!("  CUDA with Tensor Cores: 100x (excellent for FP16/BF16)");
        } else {
            println!("  CUDA without Tensor Cores: 50x");
        }
    }

    if metal_devices > 0 {
        println!("  Metal: 30-60x (varies by Apple Silicon generation)");
    }

    if opencl_devices > 0 {
        println!("  OpenCL: 10-40x (varies greatly by hardware)");
    }

    println!("  WebGPU: 5-20x (good for web applications)");
    println!("  CPU (SIMD): 1x (baseline)");

    println!("\nNote: Performance ratios are approximate and vary by:");
    println!("- Problem size and complexity");
    println!("- Memory access patterns");
    println!("- Specific hardware configuration");
    println!("- Driver and software optimization");

    Ok(())
}
