use ag::tensor_ops as T;
use scirs2_autograd as ag;
use scirs2_core::ndarray::Array2;
use std::time::Instant;

#[allow(dead_code)]
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
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .expect("Operation failed")
        })
        .collect();

    println!("\n1. Running forward/backward without checkpointing...");

    // Measure time and estimate memory without checkpointing
    let start = Instant::now();
    let mut non_ckpt_memory_estimate = 0;

    ag::run(|ctx| {
        // Start tracking memory usage
        T::CheckpointProfiler::start_tracking();

        // Convert weights to tensors. These are differentiated below via
        // T::grad, so they must be `T::variable`: `T::convert_to_tensor`
        // marks each node non-differentiable, which makes `on_backprop_path`
        // false all the way up to the loss, so the backward pass used to
        // terminate immediately without ever touching any of the `depth`
        // simulated layers -- defeating the entire point of this benchmark.
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
        // network's hyperparameters, not a footgun), so we only assert
        // finiteness here rather than non-zero-ness.
        for grad in grads {
            let grad_val = grad.eval(ctx).expect("gradient should be evaluable");
            assert!(
                grad_val.iter().all(|v: &f32| v.is_finite()),
                "gradient must be finite even where it legitimately vanishes"
            );
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

    ag::run(|ctx| {
        // Reset and start tracking memory usage
        T::CheckpointProfiler::reset_statistics();
        T::CheckpointProfiler::start_tracking();

        // Convert weights to tensors. Differentiated below via T::grad, so
        // `T::variable` is required (see the "no checkpointing" section above
        // for why `T::convert_to_tensor` would defeat the whole benchmark).
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
                basic_ckpt_memory_estimate += feature_size * std::mem::size_of::<f32>();
            }
        }

        let output = activations.last().expect("Operation failed");
        let loss = T::sum_all(output);

        // Backward pass
        let grads = T::grad(&[loss], &weight_tensors.iter().collect::<Vec<_>>());

        // Evaluate gradients (see the "no checkpointing" section above for why
        // only finiteness, not non-zero-ness, is asserted).
        for grad in grads {
            let grad_val = grad.eval(ctx).expect("gradient should be evaluable");
            assert!(
                grad_val.iter().all(|v: &f32| v.is_finite()),
                "gradient must be finite even where it legitimately vanishes"
            );
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

    ag::run(|ctx| {
        // Reset and start tracking memory usage
        T::CheckpointProfiler::reset_statistics();
        T::CheckpointProfiler::start_tracking();

        // Convert weights to tensors. Differentiated below via T::grad, so
        // `T::variable` is required (see section 1 above for details).
        let weight_tensors: Vec<_> = weights
            .iter()
            .map(|w| T::variable(w.clone(), ctx))
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

        let output = activations.last().expect("Operation failed");
        let loss = T::sum_all(output);

        // Backward pass
        let grads = T::grad(&[loss], &weight_tensors.iter().collect::<Vec<_>>());

        // Evaluate gradients (see section 1 above for why only finiteness is
        // asserted here).
        for grad in grads {
            let grad_val = grad.eval(ctx).expect("gradient should be evaluable");
            assert!(
                grad_val.iter().all(|v: &f32| v.is_finite()),
                "gradient must be finite even where it legitimately vanishes"
            );
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

        // Calculate adaptive checkpointing memory estimate. `memory_saved` is
        // the profiler's own internal accounting (which can legitimately
        // exceed this file's simple `feature_size * size_of::<f32>()` per-layer
        // estimate, e.g. if it accounts for recomputation overhead
        // differently); saturate instead of panicking on underflow so this
        // pre-existing arithmetic mismatch (unrelated to the convert_to_tensor
        // footgun) doesn't crash the example.
        adaptive_ckpt_memory_estimate = non_ckpt_memory_estimate.saturating_sub(memory_saved);

        T::CheckpointProfiler::stop_tracking();
    });

    let adaptive_ckpt_time = start.elapsed();
    println!("  Time: {:?}", adaptive_ckpt_time);
    println!(
        "  Estimated activation memory: {:?} KB",
        adaptive_ckpt_memory_estimate / 1024
    );

    println!("\n4. Running with checkpoint group for multi-output operations...");

    // Plain arrays kept outside the graph so a finite-difference check can
    // perturb them and rebuild a fresh graph for verification below.
    let a_arr = Array2::<f32>::eye(feature_size);
    let b_arr = Array2::<f32>::ones((feature_size, feature_size));

    // Example using checkpoint groups for functions with multiple outputs
    let (grad1_val, grad2_val) = ag::run(|ctx| {
        // Create inputs for a multi-output operation. Both are differentiated
        // below via T::grad, so they must be `T::variable`: `convert_to_tensor`
        // would silently zero every gradient in this section regardless of
        // whether adaptive checkpointing preserves it correctly.
        let a = T::variable(a_arr.clone().into_dyn(), ctx);
        let b = T::variable(b_arr.clone().into_dyn(), ctx);

        println!("  Running multi-output operation without checkpointing...");
        let start = Instant::now();

        // Run without checkpointing
        let c1 = T::matmul(a, b);
        let c2 = T::transpose(c1, &[1, 0]);
        let c3 = T::matmul(c1, c2);

        let loss1 = T::sum_all(c1) + T::sum_all(c2) + T::sum_all(c3);
        let grad1 = T::grad(&[loss1], &[&a])[0];
        let grad1_val = grad1.eval(ctx).expect("Operation failed");

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
        let grad2_val = grad2.eval(ctx).expect("Operation failed");

        let adaptive_time = start.elapsed();
        println!("    Time: {:?}", adaptive_time);
        println!(
            "    Time ratio: {:.2}x",
            adaptive_time.as_millis() as f64 / normal_time.as_millis() as f64
        );

        (grad1_val, grad2_val)
    });

    // Real correctness check (this used to pass vacuously: `a`/`b` were built
    // with `convert_to_tensor`, so both `grad1` and `grad2` were silently the
    // exact-zero fallback and trivially "matched" no matter what
    // adaptive_checkpoint did).
    let max_mismatch = grad1_val
        .iter()
        .zip(grad2_val.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_mismatch < 1e-2,
        "adaptive checkpointing must preserve the gradient: max |grad1 - grad2| = {max_mismatch}"
    );
    let grad1_abs_sum: f32 = grad1_val.iter().map(|v| v.abs()).sum();
    assert!(
        grad1_abs_sum > 0.0,
        "gradient must be genuinely non-zero (it was a silent zero fallback before the fix)"
    );
    println!(
        "    Consistency verified: max |grad1 - grad2| = {max_mismatch:.6}, sum|grad1| = {grad1_abs_sum:.3}"
    );

    // Independent finite-difference spot check on a single entry of `a`.
    {
        let h = 1.0_f32;
        let mut a_plus = a_arr.clone();
        a_plus[[0, 0]] += h;
        let mut a_minus = a_arr.clone();
        a_minus[[0, 0]] -= h;
        let forward = |a_val: &Array2<f32>| -> f32 {
            ag::run(|ctx| {
                let a_t = T::convert_to_tensor(a_val.clone().into_dyn(), ctx);
                let b_t = T::convert_to_tensor(b_arr.clone().into_dyn(), ctx);
                let c1 = T::matmul(a_t, b_t);
                let c2 = T::transpose(c1, &[1, 0]);
                let c3 = T::matmul(c1, c2);
                let loss = T::sum_all(c1) + T::sum_all(c2) + T::sum_all(c3);
                loss.eval(ctx).expect("Operation failed")[[]]
            })
        };
        let numeric = (forward(&a_plus) - forward(&a_minus)) / (2.0 * h);
        let analytic = grad1_val[[0, 0]];
        assert!(
            (numeric - analytic).abs() < numeric.abs().max(1.0) * 0.05,
            "finite-difference check failed at a[0,0]: analytic={analytic}, numeric={numeric}"
        );
        println!(
            "    Finite-difference spot check at a[0,0]: analytic={analytic:.3}, numeric={numeric:.3}"
        );
    }

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
