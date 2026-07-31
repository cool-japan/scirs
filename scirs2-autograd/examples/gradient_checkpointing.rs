use ag::tensor_ops as T;
use scirs2_autograd as ag;
use scirs2_core::ndarray::Array2;
use std::time::Instant;

#[allow(dead_code)]
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
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .expect("Operation failed")
        })
        .collect();

    println!("Running forward/backward without checkpointing...");

    // Measure time and estimate memory without checkpointing
    let start = Instant::now();
    let mut non_ckpt_memory_estimate = 0;

    ag::run(|ctx| {
        // Convert weights to tensors. Differentiated below via T::grad, so
        // they must be `T::variable`: `T::convert_to_tensor` marks each node
        // non-differentiable, so `on_backprop_path` would be false all the
        // way up to the loss and the backward pass would never touch any of
        // the `depth` simulated layers -- defeating the point of this
        // benchmark.
        let weight_tensors: Vec<_> = weights
            .iter()
            .map(|w| T::variable(w.clone(), ctx))
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

        let output = activations.last().expect("Operation failed");
        let loss = T::sum_all(output);

        // Backward pass
        let grads = T::grad(&[loss], &weight_tensors.iter().collect::<Vec<_>>());

        // Evaluate gradients. With `depth` layers of relu(matmul(.., 0.01-scaled
        // weight)), the activations legitimately underflow to exact zero well
        // before layer `depth` (a real vanishing-gradient effect of this toy
        // network's hyperparameters, not a footgun), so only finiteness is
        // asserted here.
        for grad in grads {
            let grad_val = grad.eval(ctx).expect("gradient should be evaluable");
            assert!(
                grad_val.iter().all(|v: &f32| v.is_finite()),
                "gradient must be finite even where it legitimately vanishes"
            );
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

    ag::run(|ctx| {
        // Convert weights to tensors. Differentiated below via T::grad, so
        // `T::variable` is required (see the non-checkpointed run above).
        let weight_tensors: Vec<_> = weights
            .iter()
            .map(|w| T::variable(w.clone(), ctx))
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
                ckpt_memory_estimate += feature_size * std::mem::size_of::<f32>();
            }
        }

        let output = activations.last().expect("Operation failed");
        let loss = T::sum_all(output);

        // Backward pass
        let grads = T::grad(&[loss], &weight_tensors.iter().collect::<Vec<_>>());

        // Evaluate gradients (see the non-checkpointed run above for why only
        // finiteness, not non-zero-ness, is asserted).
        for grad in grads {
            let grad_val = grad.eval(ctx).expect("gradient should be evaluable");
            assert!(
                grad_val.iter().all(|v: &f32| v.is_finite()),
                "gradient must be finite even where it legitimately vanishes"
            );
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

    ag::run(|ctx| {
        // Create two matrices for a segment computation. Both are
        // differentiated below via T::grad, so they must be `T::variable`:
        // `T::convert_to_tensor` would silently zero every gradient in this
        // section regardless of whether checkpointing preserves it correctly.
        let a = T::variable(Array2::<f32>::eye(feature_size).into_dyn(), ctx);
        let b = T::variable(
            Array2::<f32>::ones((feature_size, feature_size)).into_dyn(),
            ctx,
        );

        println!("Running computation segment...");

        // Run the segment normally
        let start = Instant::now();

        // This simulates a complex computation
        let c1 = T::matmul(a, b);
        let d1 = T::relu(c1);
        let e1 = T::matmul(d1, b);
        let f1 = T::relu(e1);
        let result1 = T::sum_all(f1);

        let val1 = result1.eval(ctx).expect("Operation failed");
        let normal_time = start.elapsed();

        // Run with checkpoint operations manually
        let start = Instant::now();

        // Use individual checkpoint operations
        let c2 = T::matmul(a, b);
        let c2_ckpt = T::checkpoint(&c2);
        let d2 = T::relu(c2_ckpt);
        let d2_ckpt = T::checkpoint(&d2);
        let e2 = T::matmul(d2_ckpt, b);
        let e2_ckpt = T::checkpoint(&e2);
        let f2 = T::relu(e2_ckpt);
        let result2 = T::sum_all(f2);

        let val2 = result2.eval(ctx).expect("Operation failed");
        let checkpoint_time = start.elapsed();

        println!("  Normal result: {}", val1[[]]);
        println!("  Checkpointed result: {}", val2[[]]);
        println!("  Results match: {}", (val1[[]] - val2[[]]).abs() < 1e-5);
        println!("  Normal execution time: {:?}", normal_time);
        println!("  Checkpointed execution time: {:?}", checkpoint_time);

        // Hand-derived: with a = I and b = ones(n,n), both relus are no-ops
        // (every intermediate value stays positive), so
        // result1 = sum(a @ (b@b)) = n^3 for n = feature_size.
        let n = feature_size as f32;
        let expected_result = n * n * n;
        assert!(
            (val1[[]] - expected_result).abs() < 1.0,
            "forward result = {}, expected {}",
            val1[[]],
            expected_result
        );
        assert!(
            (val1[[]] - val2[[]]).abs() < 1e-3,
            "checkpointing must preserve the forward result exactly"
        );

        // Test gradients
        let start = Instant::now();
        let grad1 = T::grad(&[result1], &[&a])[0];
        let grad_val1 = grad1.eval(ctx).expect("Operation failed");
        let grad1_time = start.elapsed();

        let start = Instant::now();
        let grad2 = T::grad(&[result2], &[&a])[0];
        let grad_val2 = grad2.eval(ctx).expect("Operation failed");
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

        // This is the real, non-vacuous point of the comparison above: with
        // `a`/`b` now genuinely differentiable, checkpointing must reproduce
        // EVERY element of the gradient exactly, not just "some".
        assert_eq!(
            match_count,
            grad_val1.len(),
            "checkpointing must preserve every gradient element exactly"
        );

        // Hand-derived: d(result1)/da_{ij} = sum_k bb_{jk} where bb = b@b is
        // uniformly n (feature_size), so the gradient is the constant
        // n*n = feature_size^2 everywhere.
        let expected = (feature_size * feature_size) as f32;
        for &g in grad_val1.iter() {
            assert!(
                (g - expected).abs() < 1.0,
                "gradient element = {g}, expected {expected}"
            );
        }
    });
}
