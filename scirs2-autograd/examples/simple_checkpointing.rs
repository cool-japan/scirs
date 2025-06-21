use ag::tensor_ops::{self, checkpoint, matmul, ones, sum_all, tanh, CheckpointProfiler};
use scirs2_autograd as ag;
use std::time::Instant;

// This is a simplified example that demonstrates gradient checkpointing
// It creates a small linear chain of operations and shows how checkpointing
// affects memory usage and computation time

fn main() {
    println!("Simple Gradient Checkpointing Example");
    println!("===================================");
    println!("This example demonstrates the basic functionality of gradient checkpointing");
    println!(
        "by comparing memory usage between regular backward pass and checkpointed backward pass."
    );

    // Create a computation graph with a chain of operations
    ag::run::<f32, _, _>(|ctx| {
        // Start tracking memory with the checkpoint profiler
        CheckpointProfiler::start_tracking();

        // Create input tensor
        let x: ag::Tensor<f32> = ones(&[10, 10], ctx);
        let weight: ag::Tensor<f32> = ones(&[10, 10], ctx);

        // Create a chain of matrix multiplications to simulate a deep network
        // We'll make 10 layers
        let mut layer = x;
        for i in 0..10 {
            println!("Adding layer {}", i);
            layer = matmul(layer, weight);
            // Add non-linear activation
            layer = tanh(layer);
        }

        // Compute loss
        let loss = sum_all(layer);

        // Compute gradients (without checkpointing)
        println!("\nComputing gradients without checkpointing");
        let start_time = Instant::now();
        let grad = tensor_ops::grad(&[loss], &[&x])[0];
        let _ = grad.eval(ctx);
        let normal_time = start_time.elapsed();
        println!("Time without checkpointing: {:?}", normal_time);

        // Stop tracking and get memory usage
        let normal_memory = CheckpointProfiler::memory_saved();
        CheckpointProfiler::stop_tracking();
        CheckpointProfiler::reset_statistics();

        // Now repeat with checkpointing
        CheckpointProfiler::start_tracking();

        // Create input tensor
        let x: ag::Tensor<f32> = ones(&[10, 10], ctx);
        let weight: ag::Tensor<f32> = ones(&[10, 10], ctx);

        // Create a chain of matrix multiplications but with checkpointing
        let mut layer = x;
        for i in 0..10 {
            println!("Adding checkpointed layer {}", i);
            let next_layer = matmul(layer, weight);
            let activated = tanh(next_layer);

            // Apply checkpointing every other layer
            if i % 2 == 1 {
                layer = checkpoint(&activated);
            } else {
                layer = activated;
            }
        }

        // Compute loss
        let loss = sum_all(layer);

        // Compute gradients (with checkpointing)
        println!("\nComputing gradients with checkpointing");
        let start_time = Instant::now();
        let grad = tensor_ops::grad(&[loss], &[&x])[0];
        let _ = grad.eval(ctx);
        let checkpoint_time = start_time.elapsed();
        println!("Time with checkpointing: {:?}", checkpoint_time);

        // Get memory usage
        let checkpoint_memory = CheckpointProfiler::memory_saved();
        let num_checkpoints = CheckpointProfiler::checkpoint_count();
        CheckpointProfiler::stop_tracking();

        // Compare results
        println!("\nResults:");
        println!("- Without checkpointing:");
        println!("  Memory usage: {} bytes", normal_memory);
        println!("  Time: {:?}", normal_time);
        println!("- With checkpointing ({} checkpoints):", num_checkpoints);
        println!("  Memory usage: {} bytes", checkpoint_memory);
        println!("  Time: {:?}", checkpoint_time);
        println!(
            "- Time overhead: {:.1}%",
            100.0 * ((checkpoint_time.as_millis() as f64 / normal_time.as_millis() as f64) - 1.0)
        );

        // Note: The checkpoint_segment API has lifetime issues in this simplified example
        // Refer to examples/gradient_checkpointing.rs for a working example of checkpoint_segment
        println!("\nFor a complete example of checkpoint_segment, see examples/gradient_checkpointing.rs");
    });
}
