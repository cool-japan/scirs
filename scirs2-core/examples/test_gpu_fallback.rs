#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuContext};

#[allow(dead_code)]
fn main() {
    println!("Testing GPU backend detection and fallback...\n");

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled. Run with --features=gpu");
    }

    #[cfg(feature = "gpu")]
    {
        // Test 1: Check preferred backend
        let preferred = GpuBackend::preferred();
        println!("Preferred backend: {:?}", preferred);

        // Test 2: Check compile-time wgpu support and CUDA runtime availability.
        // NOTE: scirs2-core dropped its bundled CUDA backend in 0.6.x -- CUDA
        // acceleration now lives in the separate oxicuda-* crates. The `Cuda`
        // variant stays in `GpuBackend` for API compatibility, but
        // `is_available()` always reports `false` here; scirs2-core's own GPU
        // story is the portable `wgpu` backend plus a CPU fallback.
        println!("\nwgpu feature enabled: {}", cfg!(feature = "wgpu"));
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

        // Test 4: Try to create a context with CUDA explicitly. CUDA support was
        // retired from scirs2-core in 0.6.x, so this is expected to fail with a
        // migration hint pointing at the oxicuda-* crates (see Test 2 above).
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
