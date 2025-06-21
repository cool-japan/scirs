//! Comprehensive GPU Kernel Library Example
//!
//! Demonstrates advanced usage of the GPU kernel library including:
//! - All available kernels (BLAS, ML, Reduction, Transform)
//! - Kernel specialization and optimization
//! - Performance comparison between CPU and GPU
//! - Multi-kernel workflows
//! - Error handling and fallback strategies

use scirs2_core::gpu::kernels::{DataType, KernelParams};
use scirs2_core::gpu::{GpuBackend, GpuContext};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comprehensive GPU Kernel Library Example ===");

    // Initialize GPU context with fallback strategy
    let context = initialize_gpu_context()?;
    println!(
        "Successfully initialized GPU context with backend: {}\n",
        context.backend_name()
    );

    // Run comprehensive kernel demos
    demo_blas_advanced(&context)?;
    demo_reduction_comprehensive(&context)?;
    demo_ml_kernels_complete(&context)?;
    demo_transform_operations(&context)?;
    demo_kernel_chaining(&context)?;
    demo_performance_analysis(&context)?;

    println!("\n=== Comprehensive GPU Kernel Demo Complete ===");
    Ok(())
}

/// Initialize GPU context with intelligent fallback
fn initialize_gpu_context() -> Result<GpuContext, Box<dyn std::error::Error>> {
    // Try different backends in order of preference
    let backends = [
        GpuBackend::Cuda,
        GpuBackend::Metal,
        GpuBackend::Wgpu,
        GpuBackend::OpenCL,
        GpuBackend::Cpu,
    ];

    for backend in &backends {
        if backend.is_available() {
            match GpuContext::new(*backend) {
                Ok(context) => {
                    println!("Successfully initialized {} backend", backend);
                    return Ok(context);
                }
                Err(e) => {
                    println!("Failed to initialize {} backend: {}", backend, e);
                    continue;
                }
            }
        } else {
            println!("{} backend is not available on this system", backend);
        }
    }

    Err("No GPU backends available".into())
}

/// Advanced BLAS operations with different matrix sizes
fn demo_blas_advanced(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demo: Advanced BLAS Operations");
    println!("===============================");

    // Test different matrix sizes to demonstrate performance characteristics
    let test_sizes = [
        (64, 32, 48),
        (128, 128, 128),
        (256, 256, 256),
        (512, 256, 384),
    ];

    for (m, k, n) in test_sizes {
        println!(
            "\n1. GEMM Performance Test: {}x{} * {}x{} = {}x{}",
            m, k, k, n, m, n
        );

        let start = Instant::now();

        // Create test data
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();

        // Create GPU buffers
        let a_buffer = context.create_buffer_from_slice(&a_data);
        let b_buffer = context.create_buffer_from_slice(&b_data);
        let c_buffer = context.create_buffer::<f32>(m * n);

        // Get GEMM kernel
        let gemm_kernel = context.get_kernel("gemm")?;

        // Set parameters
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

        let duration = start.elapsed();
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / duration.as_secs_f64() / 1e9;

        println!(
            "   Time: {:.2} ms, Performance: {:.2} GFLOPS",
            duration.as_secs_f64() * 1000.0,
            gflops
        );
    }

    println!("\n2. Specialized AXPY Operations");

    // Test AXPY with different alpha values and specialization
    let alphas = [1.0, 2.0, std::f32::consts::PI, -1.0];
    let size = 1024;

    for alpha in alphas {
        let x_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();

        let x_buffer = context.create_buffer_from_slice(&x_data);
        let y_buffer = context.create_buffer_from_slice(&y_data);

        // Try to get a specialized kernel for this alpha value
        let params = KernelParams::new(DataType::Float32).with_numeric_param("alpha", alpha as f64);

        let kernel = match context.get_specialized_kernel("axpy", &params) {
            Ok(specialized) => {
                println!("   Using specialized AXPY kernel for alpha = {}", alpha);
                specialized
            }
            Err(_) => {
                println!("   Using general AXPY kernel for alpha = {}", alpha);
                context.get_kernel("axpy")?
            }
        };

        kernel.set_buffer("x", &x_buffer);
        kernel.set_buffer("y", &y_buffer);
        kernel.set_f32("alpha", alpha);
        kernel.set_u32("n", size as u32);

        let work_groups = (size as u32).div_ceil(256);
        kernel.dispatch([work_groups, 1, 1]);

        let result = y_buffer.to_vec();
        println!(
            "   AXPY with alpha = {}: result[0:5] = {:?}",
            alpha,
            &result[0..5]
        );
    }

    Ok(())
}

/// Comprehensive reduction operations testing
fn demo_reduction_comprehensive(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\nDemo: Comprehensive Reduction Operations");
    println!("========================================");

    type TestCase = (&'static str, usize, fn(usize) -> f32);

    // Test with different data patterns and sizes
    let test_cases: &[TestCase] = &[
        ("Random sine wave", 1024, |i: usize| {
            ((i as f32) * 0.1).sin()
        }),
        ("Linear ramp", 2048, |i: usize| i as f32),
        ("Exponential decay", 512, |i: usize| {
            (-0.01 * i as f32).exp()
        }),
        ("Large dataset", 8192, |i: usize| ((i as f32) * 0.001).cos()),
    ];

    for (case_idx, (name, size, data_gen)) in test_cases.iter().enumerate() {
        println!("\n{}. Testing: {} (size: {})", case_idx + 1, name, size);

        let data: Vec<f32> = (0..*size).map(data_gen).collect();
        let input_buffer = context.create_buffer_from_slice(&data);

        // CPU reference calculations
        let cpu_sum: f32 = data.iter().sum();
        let cpu_min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let cpu_max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let cpu_mean = cpu_sum / data.len() as f32;

        println!(
            "   CPU Reference - Sum: {:.4}, Min: {:.4}, Max: {:.4}, Mean: {:.4}",
            cpu_sum, cpu_min, cpu_max, cpu_mean
        );

        // Test all reduction operations
        let reductions = [
            ("sum_reduce", "Sum"),
            ("min_reduce", "Min"),
            ("max_reduce", "Max"),
            ("mean_reduce", "Mean"),
        ];

        for (kernel_name, op_name) in reductions {
            if let Ok(kernel) = context.get_kernel(kernel_name) {
                let output_buffer = context.create_buffer::<f32>(1);

                kernel.set_buffer("input", &input_buffer);
                kernel.set_buffer("output", &output_buffer);
                kernel.set_u32("n", *size as u32);

                if kernel_name == "mean_reduce" {
                    kernel.set_u32("total_elements", *size as u32);
                }

                let work_groups = (*size as u32).div_ceil(512);
                kernel.dispatch([work_groups, 1, 1]);

                let result = output_buffer.to_vec();
                println!("   GPU {}: {:.4}", op_name, result[0]);
            }
        }
    }

    Ok(())
}

/// Complete machine learning kernels demonstration
fn demo_ml_kernels_complete(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\nDemo: Complete Machine Learning Kernels");
    println!("=======================================");

    // Test activation functions
    println!("\n1. Activation Functions");
    let test_inputs = [
        ("Small values", vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
        ("Large values", vec![-10.0, -5.0, 0.0, 5.0, 10.0]),
        (
            "Edge cases",
            vec![f32::NEG_INFINITY, -100.0, 100.0, f32::INFINITY],
        ),
    ];

    for (test_name, inputs) in test_inputs {
        println!("\n   Testing: {}", test_name);
        let input_buffer = context.create_buffer_from_slice(&inputs);
        let output_buffer = context.create_buffer::<f32>(inputs.len());

        let activations = [("relu", "ReLU"), ("sigmoid", "Sigmoid"), ("tanh", "Tanh")];

        for (kernel_name, activation_name) in activations {
            if let Ok(kernel) = context.get_kernel(kernel_name) {
                kernel.set_buffer("input", &input_buffer);
                kernel.set_buffer("output", &output_buffer);
                kernel.set_u32("n", inputs.len() as u32);

                let work_groups = (inputs.len() as u32).div_ceil(256);
                kernel.dispatch([work_groups, 1, 1]);

                let result = output_buffer.to_vec();
                println!("     {}: {:?}", activation_name, result);
            }
        }
    }

    // Test pooling operations
    println!("\n2. Pooling Operations");

    // Create a 2D feature map for pooling
    let batch_size = 1;
    let channels = 2;
    let height = 4;
    let width = 4;
    let total_size = batch_size * channels * height * width;

    let feature_map: Vec<f32> = (0..total_size).map(|i| i as f32).collect();
    let input_buffer = context.create_buffer_from_slice(&feature_map);

    println!(
        "   Input feature map ({}x{}x{}x{}):",
        batch_size, channels, height, width
    );
    for c in 0..channels {
        println!("     Channel {}:", c);
        for h in 0..height {
            for w in 0..width {
                let idx = c * height * width + h * width + w;
                print!("{:6.1} ", feature_map[idx]);
            }
            println!();
        }
    }

    let pooling_ops = [
        ("max_pool2d", "Max Pooling"),
        ("avg_pool2d", "Average Pooling"),
    ];

    for (kernel_name, pool_name) in pooling_ops {
        if let Ok(kernel) = context.get_kernel(kernel_name) {
            // 2x2 pooling with stride 2
            let pool_size = 2;
            let stride = 2;
            let output_height = height.div_ceil(stride);
            let output_width = width.div_ceil(stride);
            let output_size = batch_size * channels * output_height * output_width;

            let output_buffer = context.create_buffer::<f32>(output_size);

            kernel.set_buffer("input", &input_buffer);
            kernel.set_buffer("output", &output_buffer);
            kernel.set_u32("batch_size", batch_size as u32);
            kernel.set_u32("channels", channels as u32);
            kernel.set_u32("input_height", height as u32);
            kernel.set_u32("input_width", width as u32);
            kernel.set_u32("output_height", output_height as u32);
            kernel.set_u32("output_width", output_width as u32);
            kernel.set_u32("pool_height", pool_size as u32);
            kernel.set_u32("pool_width", pool_size as u32);
            kernel.set_u32("stride_y", stride as u32);
            kernel.set_u32("stride_x", stride as u32);

            let work_groups_x = (output_width as u32).div_ceil(16);
            let work_groups_y = (output_height as u32).div_ceil(16);
            let work_groups_z = channels as u32;
            kernel.dispatch([work_groups_x, work_groups_y, work_groups_z]);

            let result = output_buffer.to_vec();
            println!(
                "\n   {} result ({}x{}x{}x{}):",
                pool_name, batch_size, channels, output_height, output_width
            );
            for c in 0..channels {
                println!("     Channel {}:", c);
                for h in 0..output_height {
                    for w in 0..output_width {
                        let idx = c * output_height * output_width + h * output_width + w;
                        print!("{:6.1} ", result[idx]);
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}

/// Transform operations including FFT and convolution
fn demo_transform_operations(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\nDemo: Transform Operations");
    println!("==========================");

    // 1D Convolution demonstration
    println!("\n1. 1D Convolution");

    let signal_length = 16;
    let kernel_length = 5;
    let stride = 1;
    let padding = 2;
    let output_length = (signal_length + 2 * padding - kernel_length) / stride + 1;

    let signal: Vec<f32> = (0..signal_length).map(|i| (i as f32 * 0.5).sin()).collect();
    let conv_kernel: Vec<f32> = vec![0.2, 0.2, 0.2, 0.2, 0.2]; // Simple averaging kernel

    println!("   Input signal: {:?}", signal);
    println!("   Convolution kernel: {:?}", conv_kernel);

    if let Ok(kernel) = context.get_kernel("conv1d") {
        let signal_buffer = context.create_buffer_from_slice(&signal);
        let kernel_buffer = context.create_buffer_from_slice(&conv_kernel);
        let output_buffer = context.create_buffer::<f32>(output_length);

        kernel.set_buffer("input", &signal_buffer);
        kernel.set_buffer("kernel", &kernel_buffer);
        kernel.set_buffer("output", &output_buffer);
        kernel.set_u32("input_length", signal_length as u32);
        kernel.set_u32("kernel_length", kernel_length as u32);
        kernel.set_u32("output_length", output_length as u32);
        kernel.set_u32("stride", stride as u32);
        kernel.set_u32("padding", padding as u32);

        let work_groups = (output_length as u32).div_ceil(256);
        kernel.dispatch([work_groups, 1, 1]);

        let result = output_buffer.to_vec();
        println!("   Convolution result: {:?}", result);
    }

    // 2D Convolution demonstration
    println!("\n2. 2D Convolution (simplified CNN layer)");

    if let Ok(kernel) = context.get_kernel("conv2d") {
        let batch_size = 1;
        let in_channels = 1;
        let out_channels = 1;
        let input_height = 4;
        let input_width = 4;
        let kernel_height = 3;
        let kernel_width = 3;
        let stride_y = 1;
        let stride_x = 1;
        let padding_y = 1;
        let padding_x = 1;

        let output_height = (input_height + 2 * padding_y - kernel_height) / stride_y + 1;
        let output_width = (input_width + 2 * padding_x - kernel_width) / stride_x + 1;

        // Create input image
        let input_image: Vec<f32> = (0..input_height * input_width)
            .map(|i| (i % 3) as f32)
            .collect();

        // Edge detection kernel
        let conv_kernel: Vec<f32> = vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

        println!("   Input image ({}x{}):", input_height, input_width);
        for h in 0..input_height {
            for w in 0..input_width {
                print!("{:3.0} ", input_image[h * input_width + w]);
            }
            println!();
        }

        let input_buffer = context.create_buffer_from_slice(&input_image);
        let kernel_buffer = context.create_buffer_from_slice(&conv_kernel);
        let output_buffer = context.create_buffer::<f32>(output_height * output_width);

        kernel.set_buffer("input", &input_buffer);
        kernel.set_buffer("kernel", &kernel_buffer);
        kernel.set_buffer("output", &output_buffer);
        kernel.set_u32("batch_size", batch_size as u32);
        kernel.set_u32("in_channels", in_channels as u32);
        kernel.set_u32("out_channels", out_channels as u32);
        kernel.set_u32("input_height", input_height as u32);
        kernel.set_u32("input_width", input_width as u32);
        kernel.set_u32("output_height", output_height as u32);
        kernel.set_u32("output_width", output_width as u32);
        kernel.set_u32("kernel_height", kernel_height as u32);
        kernel.set_u32("kernel_width", kernel_width as u32);
        kernel.set_u32("stride_y", stride_y as u32);
        kernel.set_u32("stride_x", stride_x as u32);
        kernel.set_u32("padding_y", padding_y as u32);
        kernel.set_u32("padding_x", padding_x as u32);

        let work_groups_x = (output_width as u32).div_ceil(16);
        let work_groups_y = (output_height as u32).div_ceil(16);
        kernel.dispatch([work_groups_x, work_groups_y, 1]);

        let result = output_buffer.to_vec();
        println!(
            "   Edge detection result ({}x{}):",
            output_height, output_width
        );
        for h in 0..output_height {
            for w in 0..output_width {
                print!("{:6.1} ", result[h * output_width + w]);
            }
            println!();
        }
    }

    Ok(())
}

/// Demonstrate kernel chaining and complex workflows
fn demo_kernel_chaining(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\nDemo: Kernel Chaining and Complex Workflows");
    println!("============================================");

    // Simulate a simple neural network forward pass
    println!("\n1. Simulated Neural Network Forward Pass");

    let input_size = 128;
    let hidden_size = 64;
    let output_size = 32;

    // Generate input data
    let input_data: Vec<f32> = (0..input_size).map(|i| (i as f32 * 0.1).sin()).collect();
    println!(
        "   Input data range: [{:.3}, {:.3}]",
        input_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Create weight matrices (normally these would be learned)
    let w1: Vec<f32> = (0..input_size * hidden_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let w2: Vec<f32> = (0..hidden_size * output_size)
        .map(|i| (i as f32 * 0.01).cos())
        .collect();

    // Step 1: First linear layer (GEMM)
    let input_buffer = context.create_buffer_from_slice(&input_data);
    let w1_buffer = context.create_buffer_from_slice(&w1);
    let hidden_buffer = context.create_buffer::<f32>(hidden_size);

    if let Ok(gemm_kernel) = context.get_kernel("gemm") {
        gemm_kernel.set_buffer("a", &input_buffer);
        gemm_kernel.set_buffer("b", &w1_buffer);
        gemm_kernel.set_buffer("c", &hidden_buffer);
        gemm_kernel.set_u32("m", 1); // batch size 1
        gemm_kernel.set_u32("n", hidden_size as u32);
        gemm_kernel.set_u32("k", input_size as u32);
        gemm_kernel.set_f32("alpha", 1.0);
        gemm_kernel.set_f32("beta", 0.0);

        let work_groups_x = (hidden_size as u32).div_ceil(16);
        gemm_kernel.dispatch([work_groups_x, 1, 1]);

        println!("   ✓ First linear layer computed");
    }

    // Step 2: Apply ReLU activation
    let hidden_activated_buffer = context.create_buffer::<f32>(hidden_size);

    if let Ok(relu_kernel) = context.get_kernel("relu") {
        relu_kernel.set_buffer("input", &hidden_buffer);
        relu_kernel.set_buffer("output", &hidden_activated_buffer);
        relu_kernel.set_u32("n", hidden_size as u32);

        let work_groups = (hidden_size as u32).div_ceil(256);
        relu_kernel.dispatch([work_groups, 1, 1]);

        println!("   ✓ ReLU activation applied");
    }

    // Step 3: Second linear layer
    let w2_buffer = context.create_buffer_from_slice(&w2);
    let output_buffer = context.create_buffer::<f32>(output_size);

    if let Ok(gemm_kernel) = context.get_kernel("gemm") {
        gemm_kernel.set_buffer("a", &hidden_activated_buffer);
        gemm_kernel.set_buffer("b", &w2_buffer);
        gemm_kernel.set_buffer("c", &output_buffer);
        gemm_kernel.set_u32("m", 1);
        gemm_kernel.set_u32("n", output_size as u32);
        gemm_kernel.set_u32("k", hidden_size as u32);
        gemm_kernel.set_f32("alpha", 1.0);
        gemm_kernel.set_f32("beta", 0.0);

        let work_groups_x = (output_size as u32).div_ceil(16);
        gemm_kernel.dispatch([work_groups_x, 1, 1]);

        println!("   ✓ Second linear layer computed");
    }

    // Step 4: Apply Softmax
    let softmax_buffer = context.create_buffer::<f32>(output_size);

    if let Ok(softmax_kernel) = context.get_kernel("softmax") {
        softmax_kernel.set_buffer("input", &output_buffer);
        softmax_kernel.set_buffer("output", &softmax_buffer);
        softmax_kernel.set_u32("n", output_size as u32);
        softmax_kernel.set_u32("batch_size", 1);

        let work_groups = (output_size as u32).div_ceil(256);
        softmax_kernel.dispatch([work_groups, 1, 1]);

        let final_output = softmax_buffer.to_vec();
        let sum: f32 = final_output.iter().sum();
        let max_prob = final_output
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        println!("   ✓ Softmax applied");
        println!("   Final output sum: {:.6} (should be ~1.0)", sum);
        println!("   Max probability: {:.6}", max_prob);
        println!("   First 10 probabilities: {:?}", &final_output[0..10]);
    }

    Ok(())
}

/// Performance analysis and comparison
fn demo_performance_analysis(context: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n\nDemo: Performance Analysis");
    println!("==========================");

    // Matrix multiplication performance scaling
    println!("\n1. GEMM Performance Scaling");
    let matrix_sizes = [64, 128, 256, 512];

    for size in matrix_sizes {
        let m = size;
        let k = size;
        let n = size;

        // CPU implementation (simple)
        let start = Instant::now();
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let mut c_cpu = vec![0.0f32; m * n];

        // Simple CPU matrix multiplication
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c_cpu[i * n + j] += a_data[i * k + l] * b_data[l * n + j];
                }
            }
        }
        let cpu_time = start.elapsed();

        // GPU implementation
        let start = Instant::now();
        let a_buffer = context.create_buffer_from_slice(&a_data);
        let b_buffer = context.create_buffer_from_slice(&b_data);
        let c_buffer = context.create_buffer::<f32>(m * n);

        if let Ok(gemm_kernel) = context.get_kernel("gemm") {
            gemm_kernel.set_buffer("a", &a_buffer);
            gemm_kernel.set_buffer("b", &b_buffer);
            gemm_kernel.set_buffer("c", &c_buffer);
            gemm_kernel.set_u32("m", m as u32);
            gemm_kernel.set_u32("n", n as u32);
            gemm_kernel.set_u32("k", k as u32);
            gemm_kernel.set_f32("alpha", 1.0);
            gemm_kernel.set_f32("beta", 0.0);

            let work_groups_x = (n as u32).div_ceil(16);
            let work_groups_y = (m as u32).div_ceil(16);
            gemm_kernel.dispatch([work_groups_x, work_groups_y, 1]);

            let _c_gpu = c_buffer.to_vec();
        }
        let gpu_time = start.elapsed();

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let cpu_gflops = flops / cpu_time.as_secs_f64() / 1e9;
        let gpu_gflops = flops / gpu_time.as_secs_f64() / 1e9;
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "   Size {}x{}: CPU {:.2} GFLOPS, GPU {:.2} GFLOPS, Speedup: {:.2}x",
            size, size, cpu_gflops, gpu_gflops, speedup
        );
    }

    // Memory bandwidth test
    println!("\n2. Memory Bandwidth Test (AXPY)");
    let sizes = [1024, 4096, 16384, 65536];

    for size in sizes {
        let x_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();

        let start = Instant::now();
        let x_buffer = context.create_buffer_from_slice(&x_data);
        let y_buffer = context.create_buffer_from_slice(&y_data);

        if let Ok(axpy_kernel) = context.get_kernel("axpy") {
            axpy_kernel.set_buffer("x", &x_buffer);
            axpy_kernel.set_buffer("y", &y_buffer);
            axpy_kernel.set_f32("alpha", 2.0);
            axpy_kernel.set_u32("n", size as u32);

            let work_groups = (size as u32).div_ceil(256);
            axpy_kernel.dispatch([work_groups, 1, 1]);

            let _result = y_buffer.to_vec();
        }
        let gpu_time = start.elapsed();

        let bytes_read = size * std::mem::size_of::<f32>() * 2; // x and y
        let bytes_written = size * std::mem::size_of::<f32>(); // y
        let total_bytes = bytes_read + bytes_written;
        let bandwidth_gb_s = total_bytes as f64 / gpu_time.as_secs_f64() / 1e9;

        println!(
            "   Size {}: {:.2} GB/s effective bandwidth",
            size, bandwidth_gb_s
        );
    }

    Ok(())
}
