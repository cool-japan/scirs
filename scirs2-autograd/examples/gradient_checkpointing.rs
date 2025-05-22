use ag::tensor_ops as T;
use ndarray::Array2;
use scirs2_autograd as ag;
use std::time::Instant;

fn main() {
    println!("Gradient Checkpointing Example");
    println!("==============================");
    println!("This example demonstrates memory optimization using gradient checkpointing");
    println!(
        "by comparing memory usage between regular backward pass and checkpointed backward pass."
    );
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

    println!("Running forward/backward without checkpointing...");

    // Measure time and estimate memory without checkpointing
    let start = Instant::now();
    let mut non_ckpt_memory_estimate = 0;

    ag::run::<f32, _, _>(|ctx| {
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
            let layer_output = T::matmul(&activations[i], &weight_tensors[i]);
            let activation = T::relu(&layer_output);
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
    });

    let non_ckpt_time = start.elapsed();
    println!("  Time: {:?}", non_ckpt_time);
    println!(
        "  Estimated activation memory: {:?} KB",
        non_ckpt_memory_estimate / 1024
    );

    println!("\nRunning forward/backward with checkpointing...");

    // Measure time and estimate memory with checkpointing
    let start = Instant::now();
    let mut ckpt_memory_estimate = 0;

    ag::run::<f32, _, _>(|ctx| {
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
            let layer_output = T::matmul(&activations[i], &weight_tensors[i]);

            // Apply checkpointing every other layer
            let activation = if i % 2 == 0 {
                // Normal activation - store in memory
                T::relu(&layer_output)
            } else {
                // Checkpointed activation - will be recomputed during backward pass
                T::checkpoint(&T::relu(&layer_output))
            };

            activations.push(activation);

            // Estimate memory: Only non-checkpointed activations are stored
            if i % 2 == 0 {
                ckpt_memory_estimate += feature_size * std::mem::size_of::<f32>();
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
    });

    let ckpt_time = start.elapsed();
    println!("  Time: {:?}", ckpt_time);
    println!(
        "  Estimated activation memory: {:?} KB",
        ckpt_memory_estimate / 1024
    );

    println!("\nComparison:");
    println!(
        "  Memory reduction: {:.1}%",
        100.0 * (1.0 - (ckpt_memory_estimate as f64 / non_ckpt_memory_estimate as f64))
    );
    println!(
        "  Time increase: {:.1}%",
        100.0 * ((ckpt_time.as_millis() as f64 / non_ckpt_time.as_millis() as f64) - 1.0)
    );

    println!("\nCheckpoint Segment Example");
    println!("-------------------------");

    ag::run::<f32, _, _>(|ctx| {
        // Create two matrices for a segment computation
        let a = T::convert_to_tensor(Array2::<f32>::eye(feature_size).into_dyn(), ctx);
        let b = T::convert_to_tensor(
            Array2::<f32>::ones((feature_size, feature_size)).into_dyn(),
            ctx,
        );

        println!("Running computation segment...");

        // Run the segment normally
        let start = Instant::now();

        // This simulates a complex computation
        let c1 = T::matmul(&a, &b);
        let d1 = T::relu(&c1);
        let e1 = T::matmul(&d1, &b);
        let f1 = T::relu(&e1);
        let result1 = T::sum_all(&f1);

        let val1 = result1.eval(ctx).unwrap();
        let normal_time = start.elapsed();

        // Run with checkpoint operations manually
        let start = Instant::now();

        // Use individual checkpoint operations
        let c2 = T::matmul(&a, &b);
        let c2_ckpt = T::checkpoint(&c2);
        let d2 = T::relu(&c2_ckpt);
        let d2_ckpt = T::checkpoint(&d2);
        let e2 = T::matmul(&d2_ckpt, &b);
        let e2_ckpt = T::checkpoint(&e2);
        let f2 = T::relu(&e2_ckpt);
        let result2 = T::sum_all(&f2);

        let val2 = result2.eval(ctx).unwrap();
        let checkpoint_time = start.elapsed();

        println!("  Normal result: {}", val1[[]]);
        println!("  Checkpointed result: {}", val2[[]]);
        println!("  Results match: {}", (val1[[]] - val2[[]]).abs() < 1e-5);
        println!("  Normal execution time: {:?}", normal_time);
        println!("  Checkpointed execution time: {:?}", checkpoint_time);

        // Test gradients
        let start = Instant::now();
        let grad1 = T::grad(&[result1], &[&a])[0];
        let grad_val1 = grad1.eval(ctx).unwrap();
        let grad1_time = start.elapsed();

        let start = Instant::now();
        let grad2 = T::grad(&[result2], &[&a])[0];
        let grad_val2 = grad2.eval(ctx).unwrap();
        let grad2_time = start.elapsed();

        println!("\nGradient computation:");
        println!("  Normal gradient computation time: {:?}", grad1_time);
        println!("  Checkpointed gradient computation time: {:?}", grad2_time);

        if grad1_time.as_millis() > 0 {
            println!(
                "  Gradient computation time ratio: {:.1}x",
                grad2_time.as_millis() as f64 / grad1_time.as_millis() as f64
            );
        }

        // Compare a few elements of the gradients
        let match_count = grad_val1
            .iter()
            .zip(grad_val2.iter())
            .filter(|(a, b)| (*a - *b).abs() < 1e-5)
            .count();

        println!(
            "  Gradient elements that match: {}/{}",
            match_count,
            grad_val1.len()
        );
    });
}
