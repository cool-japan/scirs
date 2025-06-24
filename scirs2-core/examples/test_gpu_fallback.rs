#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuContext};

fn main() {
    println!("Testing GPU backend detection and fallback...\n");

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled. Run with --features=gpu");
        return;
    }

    #[cfg(feature = "gpu")]
    {
        // Test 1: Check preferred backend
        let preferred = GpuBackend::preferred();
        println!("Preferred backend: {:?}", preferred);

        // Test 2: Check if CUDA is available (compile-time)
        println!("\nCUDA feature enabled: {}", cfg!(feature = "cuda"));
        println!("CUDA is_available(): {}", GpuBackend::Cuda.is_available());

        // Test 3: Try to create context with default backend
        println!("\nTrying to create GPU context with default backend...");
        match GpuContext::new(GpuBackend::default()) {
            Ok(ctx) => {
                println!(
                    "✓ Successfully created context with backend: {}",
                    ctx.backend()
                );
            }
            Err(e) => {
                println!("✗ Failed to create context: {}", e);
            }
        }

        // Test 4: Try to create context with CUDA explicitly
        println!("\nTrying to create GPU context with CUDA backend...");
        match GpuContext::new(GpuBackend::Cuda) {
            Ok(ctx) => {
                println!(
                    "✓ Successfully created context with backend: {}",
                    ctx.backend()
                );
            }
            Err(e) => {
                println!("✗ Failed to create context: {}", e);
            }
        }

        // Test 5: Try to create context with CPU backend
        println!("\nTrying to create GPU context with CPU backend...");
        match GpuContext::new(GpuBackend::Cpu) {
            Ok(ctx) => {
                println!(
                    "✓ Successfully created context with backend: {}",
                    ctx.backend()
                );
            }
            Err(e) => {
                println!("✗ Failed to create context: {}", e);
            }
        }

        // Test 6: Run the actual backend detection
        println!("\nRunning backend detection...");
        use scirs2_core::gpu::backends;
        let detection_result = backends::detect_gpu_backends();
        println!("Detected {} devices:", detection_result.devices.len());
        for device in &detection_result.devices {
            println!("  - {} ({})", device.device_name, device.backend);
        }
        println!(
            "Recommended backend: {:?}",
            detection_result.recommended_backend
        );
    }
}
