//! Example demonstrating accelerated neural operations
//!
//! This example shows how to use the enhanced neural operations framework
//! that provides a foundation for GPU acceleration while currently using
//! optimized CPU implementations.

use ndarray::{array, Array2};
use scirs2_neural::error::Result;
use scirs2_neural::gpu::{create_neural_ops, create_neural_ops_with_backend};

fn main() -> Result<()> {
    println!("=== Accelerated Neural Operations Demo ===\n");

    // Create neural operations context
    let ops = create_neural_ops()?;
    println!("{}\n", ops.backend_info());

    // Demonstrate matrix multiplication
    println!("1. Matrix Multiplication:");
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

    println!("Matrix A (2x3):\n{:?}", a);
    println!("Matrix B (3x2):\n{:?}", b);

    let result = ops.matrix_multiply(&a, &b)?;
    println!("Result A * B (2x2):\n{:?}\n", result);

    // Demonstrate batch matrix multiplication
    println!("2. Batch Matrix Multiplication:");
    let batch_a = array![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]].into_dyn();
    let batch_b = array![[[2.0, 0.0], [1.0, 1.0]], [[1.0, 2.0], [0.0, 1.0]]].into_dyn();

    println!("Batch A shape: {:?}", batch_a.shape());
    println!("Batch B shape: {:?}", batch_b.shape());

    let batch_result = ops.batch_matrix_multiply(&batch_a, &batch_b)?;
    println!("Batch result shape: {:?}", batch_result.shape());
    println!("Batch result:\n{:?}\n", batch_result);

    // Demonstrate activation functions
    println!("3. Activation Functions:");
    let input = array![[-2.0, -1.0, 0.0, 1.0, 2.0]].into_dyn();
    println!("Input: {:?}", input);

    let relu_output = ops.relu_forward(&input)?;
    println!("ReLU output: {:?}", relu_output);

    let sigmoid_output = ops.sigmoid_forward(&input)?;
    println!("Sigmoid output: {:?}", sigmoid_output);

    // Demonstrate ReLU backward pass
    let grad_output = array![[1.0, 1.0, 1.0, 1.0, 1.0]].into_dyn();
    let relu_grad = ops.relu_backward(&input, &grad_output)?;
    println!("ReLU gradient: {:?}\n", relu_grad);

    // Demonstrate softmax
    println!("4. Softmax Activation:");
    let logits = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
    println!("Logits: {:?}", logits);

    let softmax_output = ops.softmax_forward(&logits)?;
    println!("Softmax output: {:?}", softmax_output);

    // Verify softmax properties (each row sums to 1)
    for (i, row) in softmax_output.axis_iter(ndarray::Axis(0)).enumerate() {
        let sum: f32 = row.sum();
        println!("Row {} sum: {:.6}", i, sum);
    }
    println!();

    // Demonstrate batch normalization
    println!("5. Batch Normalization:");
    let batch_input = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn();
    let mean = array![3.0, 4.0]; // Per-channel mean
    let variance = array![2.0, 2.0]; // Per-channel variance
    let gamma = array![1.0, 1.0]; // Scale parameter
    let beta = array![0.0, 0.0]; // Shift parameter

    println!("Input: {:?}", batch_input);
    println!("Mean: {:?}", mean);
    println!("Variance: {:?}", variance);

    let normalized = ops.batch_normalize(&batch_input, &mean, &variance, &gamma, &beta, 1e-5)?;
    println!("Normalized output: {:?}\n", normalized);

    // Demonstrate convolution
    println!("6. 2D Convolution:");
    let conv_input = array![[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]].into_dyn(); // Shape: (1, 1, 3, 3) - (batch, channels, height, width)

    let kernel = array![[[[1.0, 0.0], [-1.0, 0.0]]]].into_dyn(); // Shape: (1, 1, 2, 2) - (out_channels, in_channels, kernel_h, kernel_w)

    println!("Input shape: {:?}", conv_input.shape());
    println!("Kernel shape: {:?}", kernel.shape());

    let conv_output = ops.conv2d_forward(&conv_input, &kernel, (1, 1), (0, 0))?;
    println!("Convolution output shape: {:?}", conv_output.shape());
    println!("Convolution output: {:?}\n", conv_output);

    // Demonstrate different backend preferences
    println!("7. Backend Selection:");
    let cpu_ops = create_neural_ops_with_backend("CPU")?;
    println!("{}", cpu_ops.backend_info());

    let gpu_ops = create_neural_ops_with_backend("GPU")?;
    println!("{}", gpu_ops.backend_info());

    let custom_ops = create_neural_ops_with_backend("Custom-Accelerator")?;
    println!("{}\n", custom_ops.backend_info());

    // Performance comparison example
    println!("8. Performance Demonstration:");
    demonstrate_performance_scaling()?;

    println!("=== Demo Complete ===");
    Ok(())
}

fn demonstrate_performance_scaling() -> Result<()> {
    let ops = create_neural_ops()?;

    // Create progressively larger matrices to show scaling
    let sizes = vec![10, 50, 100];

    for size in sizes {
        let a = Array2::ones((size, size));
        let b = Array2::ones((size, size));

        let start = std::time::Instant::now();
        let _result = ops.matrix_multiply(&a, &b)?;
        let duration = start.elapsed();

        println!(
            "Matrix {}x{} multiplication took: {:?}",
            size, size, duration
        );
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_runs() {
        // Test that the example runs without panicking
        assert!(main().is_ok());
    }

    #[test]
    fn test_performance_demo() {
        assert!(demonstrate_performance_scaling().is_ok());
    }
}
