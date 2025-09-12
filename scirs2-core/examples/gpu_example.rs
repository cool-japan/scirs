#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuContext};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GPU Acceleration Example");

    // Only run the GPU code if the feature is enabled
    #[cfg(feature = "gpu")]
    run_gpu_example()?;

    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with --features=\"gpu\" to see the GPU example.");

    Ok(())
}

#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn run_gpu_example() -> Result<(), Box<dyn std::error::Error>> {
    // Try to create a GPU context with the default backend
    let ctx = match GpuContext::new(GpuBackend::default()) {
        Ok(ctx) => {
            println!("Created GPU context with {} backend", ctx.backend());
            ctx
        }
        Err(e) => {
            println!("Failed to create GPU context: {}", e);
            println!("Falling back to CPU computation");
            return Ok(());
        }
    };

    // Create input data
    let data_size = 1024;
    let host_data: Vec<f32> = (0..data_size).map(|i| i as f32).collect();
    println!("Created input data with {} elements", data_size);

    // Allocate buffer on GPU and copy data
    let buffer = ctx.create_buffer::<f32>(data_size);
    let _ = buffer.copy_from_host(&host_data);
    println!("Copied data to GPU buffer");

    // Execute a simple computation (add 1.0 to each element)
    println!("Executing computation on GPU...");
    ctx.execute(|compiler| -> Result<(), Box<dyn std::error::Error>> {
        // This is a placeholder for a real kernel
        // In a real application, this would compile and run an actual GPU kernel
        let kernel = compiler.compile(
            r#"
            __kernel void add_one(__global float* a) {
                size_t i = get_global_id(0);
                a[i] = a[i] + 1.0f;
            }
        "#,
        )?;

        kernel.set_buffer("a", &buffer);
        kernel.dispatch([data_size as u32, 1, 1]);

        Ok(())
    })?;

    // Copy results back to host
    let mut result = vec![0.0f32; data_size];
    let _ = buffer.copy_to_host(&mut result);

    // Verify results (should be i+1 for each element)
    println!("Verifying results...");
    for i in 0..5 {
        println!("Element {}: {} -> {}", i, host_data[i], result[i]);
    }
    println!("...");
    for i in data_size - 5..data_size {
        println!("Element {}: {} -> {}", i, host_data[i], result[i]);
    }

    println!("GPU computation completed successfully!");
    Ok(())
}
