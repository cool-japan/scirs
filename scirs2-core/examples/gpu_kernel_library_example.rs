//! GPU Kernel Library Example
//!
//! Demonstrates the usage of the comprehensive GPU kernel library for
//! common scientific computing operations across different GPU backends.

#[cfg(feature = "gpu")]
use scirs2_core::gpu::kernels::{DataType, KernelParams};
#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuContext};

#[cfg(feature = "gpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Kernel Library Demonstration ===");

    // Initialize GPU context - automatically selects the best available backend
    let context = match GpuContext::new(GpuBackend::preferred()) {
        Ok(ctx) => {
            println!(
                "Successfully initialized GPU context with backend: {}",
                ctx.backend_name()
            );
            ctx
        }
        Err(e) => {
            println!("Failed to initialize preferred GPU backend: {}", e);
            println!("Falling back to CPU backend...");
            GpuContext::new(GpuBackend::Cpu)?
        }
    };

    println!();

    // Demo 1: BLAS Operations
    demo_blas_operations(&context)?;

    // Demo 2: Reduction Operations
    demo_reduction_operations(&context)?;

    // Demo 3: Machine Learning Operations
    demo_ml_operations(&context)?;

    // Demo 4: Transform Operations
    demo_transform_operations(&context)?;

    // Demo 5: Kernel Specialization
    demo_kernel_specialization(&context)?;

    println!("\n=== GPU Kernel Library Demo Complete ===");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU Kernel Library Example ===");
    println!("This example requires the 'gpu' feature to be enabled.");
    println!("Please run with: cargo run --example gpu_kernel_library_example --features gpu");
    Ok(())
}

/// Demonstrate BLAS operations (GEMM, AXPY)
#[cfg(feature = "gpu")]
fn demo_blas_operations(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 1: BLAS Operations");
    println!("-----------------------");

    // Matrix multiplication (GEMM): C = A * B
    println!("1.1 General Matrix Multiplication (GEMM)");

    // Create test matrices
    let m = 64;
    let k = 32;
    let n = 48;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();

    // Create GPU buffers
    let a_buffer = context.create_buffer_from_slice(&a_data);
    let b_buffer = context.create_buffer_from_slice(&b_data);
    let c_buffer = context.create_buffer::<f32>(m * n);

    // Get GEMM kernel
    let gemm_kernel = context.get_kernel("gemm")?;

    // Set kernel parameters
    gemm_kernel.set_buffer("a", &a_buffer);
    gemm_kernel.set_buffer("b", &b_buffer);
    gemm_kernel.set_buffer("c", &c_buffer);
    gemm_kernel.set_u32("m", m as u32);
    gemm_kernel.set_u32("n", n as u32);
    gemm_kernel.set_u32("k", k as u32);
    gemm_kernel.set_f32("alpha", 1.0);
    gemm_kernel.set_f32("beta", 0.0);

    // Execute kernel
    let work_groups_x = (n as u32).div_ceil(16);
    let work_groups_y = (m as u32).div_ceil(16);
    gemm_kernel.dispatch([work_groups_x, work_groups_y, 1]);

    // Read results
    let result = c_buffer.to_vec();
    println!("   GEMM completed: {}x{} * {}x{} = {}x{}", m, k, k, n, m, n);
    println!("   First few results: {:?}", &result[0..5]);

    // AXPY operation: Y = alpha * X + Y
    println!("\n1.2 AXPY Operation (Y = alpha * X + Y)");

    let size = 1024;
    let alpha = 2.5f32;

    let x_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let y_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();

    let x_buffer = context.create_buffer_from_slice(&x_data);
    let y_buffer = context.create_buffer_from_slice(&y_data);

    // Get AXPY kernel
    let axpy_kernel = context.get_kernel("axpy")?;

    // Set kernel parameters
    axpy_kernel.set_buffer("x", &x_buffer);
    axpy_kernel.set_buffer("y", &y_buffer);
    axpy_kernel.set_f32("alpha", alpha);
    axpy_kernel.set_u32("n", size as u32);

    // Execute kernel
    let work_groups = (size as u32).div_ceil(256);
    axpy_kernel.dispatch([work_groups, 1, 1]);

    // Read results
    let result = y_buffer.to_vec();
    println!("   AXPY completed: Y = {} * X + Y", alpha);
    println!("   First few results: {:?}", &result[0..5]);

    println!();
    Ok(())
}

/// Demonstrate reduction operations (sum, min, max, mean, std)
#[cfg(feature = "gpu")]
fn demo_reduction_operations(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 2: Reduction Operations");
    println!("----------------------------");

    let size = 1024;
    let data: Vec<f32> = (0..size).map(|i| ((i as f32) * 0.1).sin()).collect();
    let input_buffer = context.create_buffer_from_slice(&data);

    // Sum reduction
    println!("2.1 Sum Reduction");
    let sum_kernel = context.get_kernel("sum_reduce")?;
    let sum_buffer = context.create_buffer::<f32>(1);

    sum_kernel.set_buffer("input", &input_buffer);
    sum_kernel.set_buffer("output", &sum_buffer);
    sum_kernel.set_u32("n", size as u32);

    let work_groups = (size as u32).div_ceil(512); // 2 elements per thread
    sum_kernel.dispatch([work_groups, 1, 1]);

    let sum_result = sum_buffer.to_vec();
    println!("   Sum: {:.4}", sum_result[0]);

    // Min reduction
    println!("\n2.2 Min Reduction");
    let min_kernel = context.get_kernel("min_reduce")?;
    let min_buffer = context.create_buffer::<f32>(1);

    min_kernel.set_buffer("input", &input_buffer);
    min_kernel.set_buffer("output", &min_buffer);
    min_kernel.set_u32("n", size as u32);

    min_kernel.dispatch([work_groups, 1, 1]);

    let min_result = min_buffer.to_vec();
    println!("   Min: {:.4}", min_result[0]);

    // Max reduction
    println!("\n2.3 Max Reduction");
    let max_kernel = context.get_kernel("max_reduce")?;
    let max_buffer = context.create_buffer::<f32>(1);

    max_kernel.set_buffer("input", &input_buffer);
    max_kernel.set_buffer("output", &max_buffer);
    max_kernel.set_u32("n", size as u32);

    max_kernel.dispatch([work_groups, 1, 1]);

    let max_result = max_buffer.to_vec();
    println!("   Max: {:.4}", max_result[0]);

    // Mean reduction
    println!("\n2.4 Mean Reduction");
    let mean_kernel = context.get_kernel("mean_reduce")?;
    let mean_buffer = context.create_buffer::<f32>(1);

    mean_kernel.set_buffer("input", &input_buffer);
    mean_kernel.set_buffer("output", &mean_buffer);
    mean_kernel.set_u32("n", size as u32);
    mean_kernel.set_u32("total_elements", size as u32);

    mean_kernel.dispatch([work_groups, 1, 1]);

    let mean_result = mean_buffer.to_vec();
    println!("   Mean: {:.4}", mean_result[0]);

    println!();
    Ok(())
}

/// Demonstrate machine learning operations (activations, pooling, softmax)
#[cfg(feature = "gpu")]
fn demo_ml_operations(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 3: Machine Learning Operations");
    println!("----------------------------------");

    let size = 256;
    let data: Vec<f32> = (0..size).map(|i| ((i as f32) - 128.0) * 0.1).collect();

    // ReLU activation
    println!("3.1 ReLU Activation");
    let input_buffer = context.create_buffer_from_slice(&data);
    let output_buffer = context.create_buffer::<f32>(size);

    let relu_kernel = context.get_kernel("relu")?;
    relu_kernel.set_buffer("input", &input_buffer);
    relu_kernel.set_buffer("output", &output_buffer);
    relu_kernel.set_u32("n", size as u32);

    let work_groups = (size as u32).div_ceil(256);
    relu_kernel.dispatch([work_groups, 1, 1]);

    let relu_result = output_buffer.to_vec();
    println!(
        "   Input range: [{:.2}, {:.2}]",
        data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "   ReLU output range: [{:.2}, {:.2}]",
        relu_result.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        relu_result.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Sigmoid activation
    println!("\n3.2 Sigmoid Activation");
    let sigmoid_kernel = context.get_kernel("sigmoid")?;
    sigmoid_kernel.set_buffer("input", &input_buffer);
    sigmoid_kernel.set_buffer("output", &output_buffer);
    sigmoid_kernel.set_u32("n", size as u32);

    sigmoid_kernel.dispatch([work_groups, 1, 1]);

    let sigmoid_result = output_buffer.to_vec();
    println!(
        "   Sigmoid output range: [{:.4}, {:.4}]",
        sigmoid_result.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        sigmoid_result
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Tanh activation
    println!("\n3.3 Tanh Activation");
    let tanh_kernel = context.get_kernel("tanh")?;
    tanh_kernel.set_buffer("input", &input_buffer);
    tanh_kernel.set_buffer("output", &output_buffer);
    tanh_kernel.set_u32("n", size as u32);

    tanh_kernel.dispatch([work_groups, 1, 1]);

    let tanh_result = output_buffer.to_vec();
    println!(
        "   Tanh output range: [{:.4}, {:.4}]",
        tanh_result.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        tanh_result.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    println!();
    Ok(())
}

/// Demonstrate transform operations (FFT)
#[cfg(feature = "gpu")]
fn demo_transform_operations(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 4: Transform Operations");
    println!("---------------------------");

    println!("4.1 Fast Fourier Transform (1D)");

    // Create a simple test signal
    let size = 64;
    let mut real_data: Vec<f32> = Vec::new();
    let mut imag_data: Vec<f32> = Vec::new();

    for i in 0..size {
        let t = (i as f32) / (size as f32);
        // Simple sinusoid + noise
        real_data.push(
            (2.0 * std::f32::consts::PI * 5.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 20.0 * t).sin(),
        );
        imag_data.push(0.0); // Start with real signal
    }

    // Interleave real and imaginary parts for complex input
    let mut complex_data: Vec<f32> = Vec::new();
    for i in 0..size {
        complex_data.push(real_data[i]);
        complex_data.push(imag_data[i]);
    }

    let input_buffer = context.create_buffer_from_slice(&complex_data);
    let output_buffer = context.create_buffer::<f32>(complex_data.len());

    // Get FFT kernel (note: this is currently a placeholder implementation)
    let fft_kernel = context.get_kernel("fft_1d_forward")?;
    fft_kernel.set_buffer("input", &input_buffer);
    fft_kernel.set_buffer("output", &output_buffer);
    fft_kernel.set_u32("n", size as u32);

    let work_groups = (size as u32).div_ceil(256);
    fft_kernel.dispatch([work_groups, 1, 1]);

    let fft_result = output_buffer.to_vec();
    println!("   FFT completed for {} complex samples", size);
    println!("   First few output values: {:?}", &fft_result[0..8]);
    println!(
        "   Note: This is a placeholder implementation - real FFT would show frequency domain"
    );

    println!();
    Ok(())
}

/// Demonstrate kernel specialization capabilities
#[cfg(feature = "gpu")]
fn demo_kernel_specialization(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo 5: Kernel Specialization");
    println!("-----------------------------");

    println!("5.1 GEMM Kernel Specialization");

    // Create parameters for a large matrix multiplication
    let params = KernelParams::new(DataType::Float32)
        .with_input_dims(vec![1024, 512])  // Matrix A: 1024x512
        .with_output_dims(vec![1024, 768]) // Matrix C: 1024x768 (implies B is 512x768)
        .with_numeric_param("alpha", 1.0)
        .with_numeric_param("beta", 0.0);

    // Get specialized kernel for large matrices
    match context.get_specialized_kernel("gemm", &params) {
        Ok(_specialized_kernel) => {
            println!("   Successfully created specialized GEMM kernel for large matrices");
            println!("   Dimensions: 1024x512 * 512x768 = 1024x768");

            // The specialized kernel would use optimized algorithms for large matrices
            // such as larger tile sizes, different memory access patterns, etc.
        }
        Err(e) => {
            println!("   Specialization not available: {}", e);
            println!("   Falling back to general purpose GEMM kernel");
        }
    }

    println!("\n5.2 FFT Kernel Specialization");

    let fft_params = KernelParams::new(DataType::Float32)
        .with_input_dims(vec![256])  // 256-point FFT
        .with_string_param("direction", "forward")
        .with_string_param("dimension", "1d");

    match context.get_specialized_kernel("fft_1d_forward", &fft_params) {
        Ok(_specialized_fft) => {
            println!("   Successfully created specialized FFT kernel for 256-point transform");
            println!("   This could use radix-4 or radix-8 algorithms optimized for this size");
        }
        Err(e) => {
            println!("   FFT specialization not available: {}", e);
        }
    }

    println!("\n5.3 AXPY Kernel with Hardcoded Alpha");

    let axpy_params =
        KernelParams::new(DataType::Float32).with_numeric_param("alpha", std::f64::consts::PI);

    match context.get_specialized_kernel("axpy", &axpy_params) {
        Ok(_specialized_axpy) => {
            println!("   Successfully created specialized AXPY kernel with alpha = PI");
            println!("   This kernel has the alpha value hardcoded for better performance");
        }
        Err(e) => {
            println!("   AXPY specialization not available: {}", e);
        }
    }

    println!();
    Ok(())
}

/// Helper function to compare CPU and GPU results
#[allow(dead_code)]
fn compare_results(cpu_result: &[f32], gpu_result: &[f32], tolerance: f32) -> bool {
    if cpu_result.len() != gpu_result.len() {
        return false;
    }

    for (cpu_val, gpu_val) in cpu_result.iter().zip(gpu_result.iter()) {
        if (cpu_val - gpu_val).abs() > tolerance {
            return false;
        }
    }

    true
}

/// Helper function to print performance comparison
#[allow(dead_code)]
fn print_performance_comparison(operation: &str, cpu_time: f64, gpu_time: f64) {
    let speedup = cpu_time / gpu_time;
    println!("   {} Performance:", operation);
    println!("     CPU: {:.2} ms", cpu_time * 1000.0);
    println!("     GPU: {:.2} ms", gpu_time * 1000.0);
    println!("     Speedup: {:.2}x", speedup);
}
