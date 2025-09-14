//! Example demonstrating the GPU kernel library
//!
//! This example shows how to use the kernel library for common operations.

use ndarray::{Array1, Array2};
#[cfg(feature = "gpu")]
use scirs2_core::gpu::kernels::{DataType, KernelParams};
#[cfg(feature = "gpu")]
use scirs2_core::gpu::{GpuBackend, GpuContext, GpuError};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "gpu")]
    {
        // Try to create a GPU context with the preferred backend
        let ctx = match GpuContext::new(GpuBackend::preferred()) {
            Ok(ctx) => {
                println!("Using GPU acceleration with {} backend", ctx.backend_name());
                ctx
            }
            Err(e) => {
                println!("GPU acceleration not available: {}. Using CPU fallback.", e);
                GpuContext::new(GpuBackend::Cpu)?
            }
        };

        // Example 1: Matrix multiplication with GEMM kernel
        println!("\nExample 1: Matrix multiplication (GEMM)");
        demo_gemm_kernel(&ctx)?;

        // Example 2: Vector addition with AXPY kernel
        println!("\nExample 2: Vector addition (AXPY)");
        demo_axpy_kernel(&ctx)?;

        // Example 3: Vector sum reduction
        println!("\nExample 3: Vector sum reduction");
        demo_sum_reduction(&ctx)?;

        // Example 4: Using a specialized kernel for L2 norm
        println!("\nExample 4: Vector L2 norm");
        demo_l2_norm(&ctx)?;

        // Example 5: Neural network activation functions
        println!("\nExample 5: Neural network activation functions");
        demo_activation_functions(&ctx)?;

        println!("\nAll examples completed successfully!");
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled. Run with --features=\"gpu\" to see the GPU examples.");
    }

    Ok(())
}

/// Example demonstrating matrix multiplication with the GEMM kernel
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn demo_gemm_kernel(ctx: &GpuContext) -> Result<(), GpuError> {
    // Create two matrices
    let a = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    let b =
        Array2::from_shape_vec((4, 2), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

    println!("Matrix A ({}x{}):", a.shape()[0], a.shape()[1]);
    for i in 0..a.shape()[0] {
        let mut row = String::new();
        for j in 0..a.shape()[1] {
            row.push_str(&format!("{:.1} ", a[[i, j]]));
        }
        println!("  {}", row);
    }

    println!("Matrix B ({}x{}):", b.shape()[0], b.shape()[1]);
    for i in 0..b.shape()[0] {
        let mut row = String::new();
        for j in 0..b.shape()[1] {
            row.push_str(&format!("{:.1} ", b[[i, j]]));
        }
        println!("  {}", row);
    }

    // Get a specialized GEMM kernel for these matrix dimensions
    let params = KernelParams::new(DataType::Float32)
        .with_input_dims(vec![a.shape()[0], a.shape()[1]])
        .with_output_dims(vec![a.shape()[0], b.shape()[1]])
        .with_numeric_param("alpha", 1.0)
        .with_numeric_param("beta", 0.0);

    let kernel = ctx.get_specialized_kernel("gemm", &params)?;

    // Create GPU buffers
    let a_buffer = ctx.create_buffer_from_slice(a.as_slice().unwrap());
    let b_buffer = ctx.create_buffer_from_slice(b.as_slice().unwrap());
    let c_buffer = ctx.create_buffer::<f32>(a.shape()[0] * b.shape()[1]);

    // Set kernel parameters
    kernel.set_buffer("a", &a_buffer);
    kernel.set_buffer("b", &b_buffer);
    kernel.set_buffer("c", &c_buffer);
    kernel.set_u32("m", a.shape()[0] as u32);
    kernel.set_u32("n", b.shape()[1] as u32);
    kernel.set_u32("k", a.shape()[1] as u32);
    kernel.set_f32("alpha", 1.0);
    kernel.set_f32("beta", 0.0);

    // Execute kernel
    kernel.dispatch([
        (b.shape()[1] as u32).div_ceil(16),
        (a.shape()[0] as u32).div_ceil(16),
        1,
    ]);

    // Get result
    let result_vec = c_buffer.to_vec();
    let result = Array2::from_shape_vec((a.shape()[0], b.shape()[1]), result_vec).unwrap();

    println!(
        "Result C = A * B ({}x{}):",
        result.shape()[0],
        result.shape()[1]
    );
    for i in 0..result.shape()[0] {
        let mut row = String::new();
        for j in 0..result.shape()[1] {
            row.push_str(&format!("{:.1} ", result[[i, j]]));
        }
        println!("  {}", row);
    }

    // Verify with CPU implementation
    let expected = a.dot(&b);
    println!(
        "Result matches CPU implementation: {}",
        result
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );

    Ok(())
}

/// Example demonstrating vector addition with the AXPY kernel
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn demo_axpy_kernel(ctx: &GpuContext) -> Result<(), GpuError> {
    // Create two vectors
    let x = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let mut y = Array1::from_vec(vec![5.0f32, 4.0, 3.0, 2.0, 1.0]);

    println!("Vector x: {:?}", x);
    println!("Vector y: {:?}", y);

    // Get the AXPY kernel
    let kernel = ctx.get_kernel("axpy")?;

    // Create GPU buffers
    let x_buffer = ctx.create_buffer_from_slice(x.as_slice().unwrap());
    let y_buffer = ctx.create_buffer_from_slice(y.as_slice().unwrap());

    // Alpha value for the operation y = alpha * x + y
    let alpha: f32 = 2.0;
    println!("Computing y = {} * x + y", alpha);

    // Set kernel parameters
    kernel.set_buffer("x", &x_buffer);
    kernel.set_buffer("y", &y_buffer);
    kernel.set_f32("alpha", alpha);
    kernel.set_u32("n", x.len() as u32);

    // Execute kernel
    kernel.dispatch([(x.len() as u32).div_ceil(256), 1, 1]);

    // Get result (overwrite y)
    let _ = y_buffer.copy_to_host(y.as_slice_mut().unwrap());

    println!("Result: {:?}", y);

    // Verify with CPU implementation
    let expected = alpha * &x + &y;
    println!(
        "Result matches CPU implementation: {}",
        y.iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );

    Ok(())
}

/// Example demonstrating vector sum reduction
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn demo_sum_reduction(ctx: &GpuContext) -> Result<(), GpuError> {
    // Create a vector
    let x = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    println!("Vector: {:?}", x);

    // Get the sum reduction kernel
    let kernel = ctx.get_kernel("sum_reduce")?;

    // For simplicity, we'll assume the vector fits in a single workgroup
    // In practice, we'd need to do multiple passes for large vectors

    // Create GPU buffers
    let input_buffer = ctx.create_buffer_from_slice(x.as_slice().unwrap());
    let output_buffer = ctx.create_buffer::<f32>(1);

    // Set kernel parameters
    kernel.set_buffer("input", &input_buffer);
    kernel.set_buffer("output", &output_buffer);
    kernel.set_u32("n", x.len() as u32);

    // Execute kernel
    kernel.dispatch([1, 1, 1]);

    // Get result
    let result = output_buffer.to_vec()[0];

    println!("Sum: {}", result);

    // Verify with CPU implementation
    let expected: f32 = x.sum();
    println!(
        "Matches CPU implementation: {}",
        (result - expected).abs() < 1e-5
    );

    Ok(())
}

/// Example demonstrating vector L2 norm
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn demo_l2_norm(ctx: &GpuContext) -> Result<(), GpuError> {
    // Create a vector
    let x = Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);

    println!("Vector: {:?}", x);

    // Get specialized L2 norm kernel
    let params = KernelParams::new(DataType::Float32).with_string_param("norm_type", "l2");

    let kernel = ctx.get_specialized_kernel("norm_l2", &params)?;

    // Create GPU buffers
    let input_buffer = ctx.create_buffer_from_slice(x.as_slice().unwrap());
    let output_buffer = ctx.create_buffer::<f32>(1);

    // Set kernel parameters
    kernel.set_buffer("input", &input_buffer);
    kernel.set_buffer("output", &output_buffer);
    kernel.set_u32("n", x.len() as u32);

    // Execute kernel
    kernel.dispatch([1, 1, 1]);

    // Get result (this is the sum of squares, we need to take sqrt)
    let sum_squares = output_buffer.to_vec()[0];
    let norm = sum_squares.sqrt();

    println!("L2 norm: {}", norm);

    // Verify with CPU implementation
    let expected: f32 = x.dot(&x).sqrt();
    println!(
        "Matches CPU implementation: {}",
        (norm - expected).abs() < 1e-5
    );

    Ok(())
}

/// Example demonstrating neural network activation functions
#[cfg(feature = "gpu")]
#[allow(dead_code)]
fn demo_activation_functions(ctx: &GpuContext) -> Result<(), GpuError> {
    // Create an input vector
    let x = Array1::from_vec(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0]);

    println!("Input: {:?}", x);

    // Get the ReLU kernel
    let relu_kernel = ctx.get_kernel("relu")?;

    // Create GPU buffers
    let input_buffer = ctx.create_buffer_from_slice(x.as_slice().unwrap());
    let output_buffer = ctx.create_buffer::<f32>(x.len());

    // Set kernel parameters
    relu_kernel.set_buffer("input", &input_buffer);
    relu_kernel.set_buffer("output", &output_buffer);
    relu_kernel.set_u32("n", x.len() as u32);

    // Execute kernel
    relu_kernel.dispatch([(x.len() as u32).div_ceil(256), 1, 1]);

    // Get result
    let relu_result = output_buffer.to_vec();

    println!("ReLU output: {:?}", relu_result);

    // Now try sigmoid
    let sigmoid_kernel = ctx.get_kernel("sigmoid")?;

    // Set kernel parameters
    sigmoid_kernel.set_buffer("input", &input_buffer);
    sigmoid_kernel.set_buffer("output", &output_buffer);
    sigmoid_kernel.set_u32("n", x.len() as u32);

    // Execute kernel
    sigmoid_kernel.dispatch([(x.len() as u32).div_ceil(256), 1, 1]);

    // Get result
    let sigmoid_result = output_buffer.to_vec();

    println!("Sigmoid output: {:?}", sigmoid_result);

    // Verify with CPU implementation
    let expected_relu: Vec<f32> = x.iter().map(|&x| x.max(0.0)).collect();
    println!(
        "ReLU matches CPU implementation: {}",
        relu_result
            .iter()
            .zip(expected_relu.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );

    let expected_sigmoid: Vec<f32> = x.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    println!(
        "Sigmoid matches CPU implementation: {}",
        sigmoid_result
            .iter()
            .zip(expected_sigmoid.iter())
            .all(|(a, b)| (a - b).abs() < 1e-5)
    );

    Ok(())
}
