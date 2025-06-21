use ag::tensor_ops as T;
use ndarray::Array2;
use scirs2_autograd as ag;
use std::time::Instant;

fn main() {
    println!("Enhanced Gradient Checkpointing Example");
    println!("======================================");
    println!("This example demonstrates memory optimization using gradient checkpointing");
    println!("by comparing regular backprop, basic checkpointing, and enhanced checkpointing techniques.");
    println!();

    // Create a simple deep network with multiple layers
    let depth = 50; // Number of layers to simulate a deep network
    let feature_size = 128; // Size of hidden features

    println!(
        "Creating a deep network with {} layers and feature size {}...",
        depth, feature_size
    );

    // Create weights
    let weights: Vec<Array2<f32>> = (0..depth)
        .map(|_| {
            let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
            rng.standard_normal(&[feature_size, feature_size])
                .mapv(|x| x * 0.01) // Scale down to prevent explosion
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap()
        })
        .collect();

    println!("\n1. Running forward/backward without checkpointing...");

    // Measure time and estimate memory without checkpointing
    let start = Instant::now();
    let mut non_ckpt_memory_estimate = 0;

    ag::run::<f32, _, _>(|ctx| {
        // Start tracking memory usage
        T::CheckpointProfiler::start_tracking();

        // Convert weights to tensors
        let weight_tensors: Vec<_> = weights
            .iter()
            .map(|w| T::convert_to_tensor(w.clone(), ctx))
            .collect();

        // Create input
        let input = T::ones(&[1, feature_size], ctx);

        // Forward pass
        let mut activations = Vec::with_capacity(depth + 1);
        activations.push(input);

        for i in 0..depth {
            let layer_output = T::matmul(activations[i], weight_tensors[i]);
            let activation = T::relu(layer_output);
            activations.push(activation);

            // Estimate memory: Each activation stores a tensor of shape [1, feature_size]
            non_ckpt_memory_estimate += feature_size * std::mem::size_of::<f32>();
        }

        let output = activations.last().unwrap();
        let loss = T::sum_all(output);

        // Backward pass
        let grads = T::grad(&[loss], &weight_tensors.iter().collect::<Vec<_>>());

        // Evaluate gradients
        for grad in grads {
            let _ = grad.eval(ctx);
        }

        T::CheckpointProfiler::stop_tracking();
    });

    let non_ckpt_time = start.elapsed();
    println!("  Time: {:?}", non_ckpt_time);
    println!(
        "  Estimated activation memory: {:?} KB",
        non_ckpt_memory_estimate / 1024
    );

    println!("\n2. Running forward/backward with basic checkpointing...");

    // Measure time and estimate memory with basic checkpointing
    let start = Instant::now();
    let mut basic_ckpt_memory_estimate = 0;

    ag::run::<f32, _, _>(|ctx| {
        // Reset and start tracking memory usage
        T::CheckpointProfiler::reset_statistics();
        T::CheckpointProfiler::start_tracking();

        // Convert weights to tensors
        let weight_tensors: Vec<_> = weights
            .iter()
            .map(|w| T::convert_to_tensor(w.clone(), ctx))
            .collect();

        // Create input
        let input = T::ones(&[1, feature_size], ctx);

        // Forward pass with checkpointing every other layer
        let mut activations = Vec::with_capacity(depth + 1);
        activations.push(input);

        for i in 0..depth {
            let layer_output = T::matmul(activations[i], weight_tensors[i]);

            // Apply checkpointing every other layer
            let activation = if i % 2 == 0 {
                // Normal activation - store in memory
                T::relu(layer_output)
            } else {
                // Checkpointed activation - will be recomputed during backward pass
                T::checkpoint(&T::relu(layer_output))
            };

            activations.push(activation);

            // Estimate memory: Only non-checkpointed activations are stored
            if i % 2 == 0 {
                basic_ckpt_memory_estimate += feature_size * std::mem::size_of::<f32>();
            }
        }

        let output = activations.last().unwrap();
        let loss = T::sum_all(output);

        // Backward pass
        let grads = T::grad(&[loss], &weight_tensors.iter().collect::<Vec<_>>());

        // Evaluate gradients
        for grad in grads {
            let _ = grad.eval(ctx);
        }

        let memory_saved = T::CheckpointProfiler::memory_saved();
        println!(
            "  Memory saved by checkpointing: {:?} KB",
            memory_saved / 1024
        );
        println!(
            "  Number of checkpoint operations: {}",
            T::CheckpointProfiler::checkpoint_count()
        );

        T::CheckpointProfiler::stop_tracking();
    });

    let basic_ckpt_time = start.elapsed();
    println!("  Time: {:?}", basic_ckpt_time);
    println!(
        "  Estimated activation memory: {:?} KB",
        basic_ckpt_memory_estimate / 1024
    );

    println!("\n3. Running forward/backward with adaptive checkpointing...");

    // Measure time and estimate memory with adaptive checkpointing
    let start = Instant::now();
    let mut adaptive_ckpt_memory_estimate = 0;
    let memory_threshold = 2048; // 2KB threshold for adaptive checkpointing

    ag::run::<f32, _, _>(|ctx| {
        // Reset and start tracking memory usage
        T::CheckpointProfiler::reset_statistics();
        T::CheckpointProfiler::start_tracking();

        // Convert weights to tensors
        let weight_tensors: Vec<_> = weights
            .iter()
            .map(|w| T::convert_to_tensor(w.clone(), ctx))
            .collect();

        // Create input
        let input = T::ones(&[1, feature_size], ctx);

        // Forward pass with adaptive checkpointing
        let mut activations = Vec::with_capacity(depth + 1);
        activations.push(input);

        for i in 0..depth {
            let layer_output = T::matmul(activations[i], weight_tensors[i]);
            let relu_output = T::relu(layer_output);

            // Use adaptive checkpointing based on tensor size
            let activation = T::adaptive_checkpoint(&relu_output, memory_threshold);

            activations.push(activation);

            // For memory estimation - we'll compute this after the run
            // based on the CheckpointProfiler results
        }

        let output = activations.last().unwrap();
        let loss = T::sum_all(output);

        // Backward pass
        let grads = T::grad(&[loss], &weight_tensors.iter().collect::<Vec<_>>());

        // Evaluate gradients
        for grad in grads {
            let _ = grad.eval(ctx);
        }

        let memory_saved = T::CheckpointProfiler::memory_saved();
        println!(
            "  Memory saved by adaptive checkpointing: {:?} KB",
            memory_saved / 1024
        );
        println!(
            "  Number of checkpoint operations: {}",
            T::CheckpointProfiler::checkpoint_count()
        );

        // Calculate adaptive checkpointing memory estimate
        adaptive_ckpt_memory_estimate = non_ckpt_memory_estimate - memory_saved;

        T::CheckpointProfiler::stop_tracking();
    });

    let adaptive_ckpt_time = start.elapsed();
    println!("  Time: {:?}", adaptive_ckpt_time);
    println!(
        "  Estimated activation memory: {:?} KB",
        adaptive_ckpt_memory_estimate / 1024
    );

    println!("\n4. Running with checkpoint group for multi-output operations...");

    // Example using checkpoint groups for functions with multiple outputs
    ag::run::<f32, _, _>(|ctx| {
        // Create inputs for a multi-output operation
        let a = T::convert_to_tensor(Array2::<f32>::eye(feature_size).into_dyn(), ctx);
        let b = T::convert_to_tensor(
            Array2::<f32>::ones((feature_size, feature_size)).into_dyn(),
            ctx,
        );

        println!("  Running multi-output operation without checkpointing...");
        let start = Instant::now();

        // Run without checkpointing
        let c1 = T::matmul(a, b);
        let c2 = T::transpose(c1, &[1, 0]);
        let c3 = T::matmul(c1, c2);

        let loss1 = T::sum_all(c1) + T::sum_all(c2) + T::sum_all(c3);
        let grad1 = T::grad(&[loss1], &[&a])[0];
        let _ = grad1.eval(ctx);

        let normal_time = start.elapsed();
        println!("    Time: {:?}", normal_time);

        println!("  Running with adaptive checkpoints...");
        let start = Instant::now();

        // Set a memory threshold (in bytes) for when to apply checkpointing
        let memory_threshold = 1024; // 1KB threshold

        // Manually create checkpoint operations for each step
        let c1 = T::matmul(a, b);
        let c2 = T::transpose(c1, &[1, 0]);
        let c3 = T::matmul(c1, c2);

        // Apply adaptive checkpoints to intermediate results
        let c1_checkpoint = T::adaptive_checkpoint(&c1, memory_threshold);
        let c2_checkpoint = T::adaptive_checkpoint(&c2, memory_threshold);
        let c3_checkpoint = T::adaptive_checkpoint(&c3, memory_threshold);

        let loss2 =
            T::sum_all(c1_checkpoint) + T::sum_all(c2_checkpoint) + T::sum_all(c3_checkpoint);
        let grad2 = T::grad(&[loss2], &[&a])[0];
        let _ = grad2.eval(ctx);

        let adaptive_time = start.elapsed();
        println!("    Time: {:?}", adaptive_time);
        println!(
            "    Time ratio: {:.2}x",
            adaptive_time.as_millis() as f64 / normal_time.as_millis() as f64
        );
    });

    println!("\nComparison Summary:");
    println!("--------------------");
    println!("1. No checkpointing:");
    println!("   - Memory: {} KB", non_ckpt_memory_estimate / 1024);
    println!("   - Time: {:?}", non_ckpt_time);
    println!();
    println!("2. Basic checkpointing (every other layer):");
    println!(
        "   - Memory: {} KB ({:.1}% of original)",
        basic_ckpt_memory_estimate / 1024,
        100.0 * (basic_ckpt_memory_estimate as f64 / non_ckpt_memory_estimate as f64)
    );
    println!(
        "   - Time: {:?} ({:.1}% increase)",
        basic_ckpt_time,
        100.0 * ((basic_ckpt_time.as_millis() as f64 / non_ckpt_time.as_millis() as f64) - 1.0)
    );
    println!();
    println!("3. Adaptive checkpointing:");
    println!(
        "   - Memory: {} KB ({:.1}% of original)",
        adaptive_ckpt_memory_estimate / 1024,
        100.0 * (adaptive_ckpt_memory_estimate as f64 / non_ckpt_memory_estimate as f64)
    );
    println!(
        "   - Time: {:?} ({:.1}% increase)",
        adaptive_ckpt_time,
        100.0 * ((adaptive_ckpt_time.as_millis() as f64 / non_ckpt_time.as_millis() as f64) - 1.0)
    );
}
