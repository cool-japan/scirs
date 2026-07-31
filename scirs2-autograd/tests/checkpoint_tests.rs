use ag::tensor_ops as T;
use scirs2_autograd as ag;
use scirs2_core::ndarray::{array, Array2};

#[test]
#[allow(dead_code)]
fn test_checkpoint_basic() {
    ag::run(|ctx| {
        // Both `a` and `b` are differentiated below (via T::grad), so they must
        // be built with `T::variable`. `T::convert_to_tensor` marks the tensor
        // non-differentiable, which makes every gradient below silently
        // collapse to the `x * scalar(0)` zero fallback -- completely
        // independent of whether checkpointing actually works.
        let a = T::variable(array![[1.0, 2.0], [3.0, 4.0]], ctx);
        let b = T::variable(array![[5.0, 6.0], [7.0, 8.0]], ctx);

        // Regular computation without checkpointing
        let c1 = T::matmul(a, b);
        let d1 = T::sum_all(c1);

        // Same computation with checkpointing
        let c2 = T::checkpoint(&T::matmul(a, b));
        let d2 = T::sum_all(c2);

        // Both computations should yield the same result
        let result1 = d1.eval(ctx).expect("Test: operation failed");
        let result2 = d2.eval(ctx).expect("Test: operation failed");

        assert_eq!(result1[[]], result2[[]]);
        assert_eq!(result1[[]], 134.0); // sum([[19,22],[43,50]]) = 134

        // Test gradients: this is the actual point of checkpointing -- it must
        // reproduce exactly the same gradient as the non-checkpointed path.
        let grad1 = T::grad(&[d1], &[&a])[0];
        let grad2 = T::grad(&[d2], &[&a])[0];

        let grad1_2d = grad1
            .eval(ctx)
            .expect("First gradient should be evaluable")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");
        let grad2_2d = grad2
            .eval(ctx)
            .expect("Second gradient should be evaluable")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");

        // Hand-derived: for L = sum(A @ B), dL/dA_{ij} = sum_k B_{jk} (row sums
        // of B, broadcast over rows of A). Row sums of b: [5+6, 7+8] = [11, 15].
        let expected = [[11.0_f64, 15.0], [11.0, 15.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (grad1_2d[[i, j]] - expected[i][j]).abs() < 1e-9_f64,
                    "non-checkpointed gradient[{i}][{j}] = {}, expected {}",
                    grad1_2d[[i, j]],
                    expected[i][j]
                );
                assert!(
                    (grad2_2d[[i, j]] - expected[i][j]).abs() < 1e-9_f64,
                    "checkpointed gradient[{i}][{j}] = {}, expected {}",
                    grad2_2d[[i, j]],
                    expected[i][j]
                );
                assert!(
                    (grad1_2d[[i, j]] - grad2_2d[[i, j]]).abs() < 1e-9_f64,
                    "checkpointing must preserve the gradient exactly: [{i}][{j}] {} vs {}",
                    grad1_2d[[i, j]],
                    grad2_2d[[i, j]]
                );
            }
        }
    });
}

#[test]
#[allow(dead_code)]
fn test_detach() {
    ag::run(|ctx| {
        // `a` must be a real variable: only then does a provably non-zero
        // "undetached" gradient (checked immediately below) demonstrate that
        // `detach` genuinely blocks backprop, rather than the final zero-check
        // being trivially true because `a` was never differentiable at all.
        let a = T::variable(array![[1.0, 2.0], [3.0, 4.0]], ctx);

        // Sanity check: without detach, d(sum(a))/da is exactly 1 everywhere.
        let direct_grad = T::grad(&[T::sum_all(a)], &[&a])[0];
        let direct_2d = direct_grad
            .eval(ctx)
            .expect("Test: operation failed")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    direct_2d[[i, j]],
                    1.0,
                    "sanity check: undetached gradient must be 1 at [{i}][{j}]"
                );
            }
        }

        // Create a computation that uses the detached tensor
        let b = T::detach(&a);
        let c = T::sum_all(b);

        // The forward computation should work as normal
        let result = c.eval(ctx).expect("Test: operation failed");
        assert_eq!(result[[]], 10.0); // 1+2+3+4 = 10

        // But gradients should not propagate through the detached tensor
        let grad = T::grad(&[c], &[&a])[0];
        let grad_result = grad.eval(ctx).expect("Test: operation failed");

        // Gradient should be zeros since we detached
        let grad_2d = grad_result
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(grad_2d[[i, j]], 0.0);
            }
        }
    });
}

#[test]
#[allow(dead_code)]
fn test_checkpoint_segment() {
    ag::run(|ctx| {
        // Differentiated below, so `T::variable` is required (see
        // test_checkpoint_basic for why `convert_to_tensor` would be wrong).
        let a = T::variable(array![[1.0, 2.0], [3.0, 4.0]], ctx);
        let b = T::variable(array![[5.0, 6.0], [7.0, 8.0]], ctx);

        // First approach: run computations directly
        let c1 = T::matmul(a, b);
        let d1 = T::square(c1);
        let result1 = T::sum_all(d1);

        // Second approach: use manual checkpointing
        let c2 = T::checkpoint(&T::matmul(a, b));
        let d2 = T::square(c2);
        let result2 = T::sum_all(d2);

        // Both should produce the same result
        let val1 = result1.eval(ctx).expect("Test: operation failed");
        let val2 = result2.eval(ctx).expect("Test: operation failed");

        assert!((val1[[]] - val2[[]] as f64).abs() < 1e-10_f64);
        assert!((val1[[]] - 5194.0_f64).abs() < 1e-9); // sum(([[19,22],[43,50]])^2)

        // Test gradients: checkpointing must preserve the gradient exactly.
        let grad1 = T::grad(&[result1], &[&a])[0];
        let grad2 = T::grad(&[result2], &[&a])[0];

        let grad1_2d = grad1
            .eval(ctx)
            .expect("First gradient should be evaluable")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");
        let grad2_2d = grad2
            .eval(ctx)
            .expect("Second gradient should be evaluable")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");

        // Hand-derived: for L = sum((A@B)^2), dL/dA_{ij} = sum_k 2*C_{ik}*B_{jk}
        // where C = A@B = [[19,22],[43,50]].
        let expected = [[454.0, 618.0], [1030.0, 1402.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (grad1_2d[[i, j]] - expected[i][j]).abs() < 1e-6,
                    "non-checkpointed gradient[{i}][{j}] = {}, expected {}",
                    grad1_2d[[i, j]],
                    expected[i][j]
                );
                assert!(
                    (grad1_2d[[i, j]] - grad2_2d[[i, j]]).abs() < 1e-6,
                    "checkpointing must preserve the gradient exactly: [{i}][{j}] {} vs {}",
                    grad1_2d[[i, j]],
                    grad2_2d[[i, j]]
                );
            }
        }
    });
}

/// Plain (non-graph) forward pass used by `test_checkpoint_deep_network` to
/// independently verify the analytic gradient via central finite differences.
fn deep_network_forward(
    input: &Array2<f64>,
    w1: &Array2<f64>,
    w2: &Array2<f64>,
    w3: &Array2<f64>,
) -> f64 {
    ag::run(|ctx| {
        let input_t = T::convert_to_tensor(input.clone(), ctx);
        let w1_t = T::convert_to_tensor(w1.clone(), ctx);
        let w2_t = T::convert_to_tensor(w2.clone(), ctx);
        let w3_t = T::convert_to_tensor(w3.clone(), ctx);
        let layer1 = T::matmul(input_t, w1_t);
        let act1 = T::relu(layer1);
        let layer2 = T::matmul(act1, w2_t);
        let act2 = T::relu(layer2);
        let output = T::matmul(act2, w3_t);
        let loss = T::sum_all(output);
        loss.eval(ctx).expect("Test: forward eval failed")[[]]
    })
}

/// Central-difference gradient of `deep_network_forward` w.r.t. one of the
/// three weight matrices (`which` = 0 for w1, 1 for w2, 2 for w3).
fn deep_network_fd_grad(
    input: &Array2<f64>,
    w1: &Array2<f64>,
    w2: &Array2<f64>,
    w3: &Array2<f64>,
    which: usize,
) -> Array2<f64> {
    let h = 1e-6;
    let base = match which {
        0 => w1,
        1 => w2,
        _ => w3,
    };
    let mut fd_grad = Array2::<f64>::zeros(base.raw_dim());
    for ((r, c), _) in base.indexed_iter() {
        let mut plus = (w1.clone(), w2.clone(), w3.clone());
        let mut minus = (w1.clone(), w2.clone(), w3.clone());
        match which {
            0 => {
                plus.0[[r, c]] += h;
                minus.0[[r, c]] -= h;
            }
            1 => {
                plus.1[[r, c]] += h;
                minus.1[[r, c]] -= h;
            }
            _ => {
                plus.2[[r, c]] += h;
                minus.2[[r, c]] -= h;
            }
        }
        let f_plus = deep_network_forward(input, &plus.0, &plus.1, &plus.2);
        let f_minus = deep_network_forward(input, &minus.0, &minus.1, &minus.2);
        fd_grad[[r, c]] = (f_plus - f_minus) / (2.0 * h);
    }
    fd_grad
}

#[test]
#[allow(dead_code)]
fn test_checkpoint_deep_network() {
    // Plain arrays kept outside the graph so a fresh graph can be rebuilt per
    // perturbed forward pass for the finite-difference check below.
    let input_arr = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
    let w1_arr = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
    let w2_arr = array![[0.7, 0.8], [0.9, 1.0]];
    let w3_arr = array![[1.1], [1.2]];

    let (grads_val, grads_ckpt_val): (Vec<Array2<f64>>, Vec<Array2<f64>>) = ag::run(|ctx| {
        // Create a simple "deep network" with multiple layers. All three
        // weights are differentiated below via T::grad, so they must be
        // `T::variable` (see test_checkpoint_basic for why `convert_to_tensor`
        // would silently zero every one of these gradients).
        let input = T::variable(input_arr.clone(), ctx);
        let w1 = T::variable(w1_arr.clone(), ctx);
        let w2 = T::variable(w2_arr.clone(), ctx);
        let w3 = T::variable(w3_arr.clone(), ctx);

        // Regular computation
        let layer1 = T::matmul(input, w1);
        let act1 = T::relu(layer1);
        let layer2 = T::matmul(act1, w2);
        let act2 = T::relu(layer2);
        let output = T::matmul(act2, w3);
        let loss = T::sum_all(output);

        // Same computation with checkpointing
        let layer1_ckpt = T::matmul(input, w1);
        let act1_ckpt = T::checkpoint(&T::relu(layer1_ckpt));
        let layer2_ckpt = T::matmul(act1_ckpt, w2);
        let act2_ckpt = T::checkpoint(&T::relu(layer2_ckpt));
        let output_ckpt = T::matmul(act2_ckpt, w3);
        let loss_ckpt = T::sum_all(output_ckpt);

        // Both computations should produce the same result
        let result = loss.eval(ctx).expect("Test: operation failed");
        let result_ckpt = loss_ckpt.eval(ctx).expect("Test: operation failed");

        assert!((result[[]] - result_ckpt[[]] as f64).abs() < 1e-10_f64);

        // Test gradients for all weights
        let grads = T::grad(&[loss], &[&w1, &w2, &w3]);
        let grads_ckpt = T::grad(&[loss_ckpt], &[&w1, &w2, &w3]);

        let to_2d = |t: &ag::Tensor<f64>| -> Array2<f64> {
            t.eval(ctx)
                .expect("Test: gradient should be evaluable")
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .expect("Test: gradient should be 2D")
        };

        let grads_val: Vec<_> = grads.iter().map(to_2d).collect();
        let grads_ckpt_val: Vec<_> = grads_ckpt.iter().map(to_2d).collect();
        (grads_val, grads_ckpt_val)
    });

    // 1. The actual point of checkpointing: the checkpointed backward path
    // must reproduce the exact same gradient as the non-checkpointed one, for
    // all three weight matrices.
    for (i, (gv, gcv)) in grads_val.iter().zip(grads_ckpt_val.iter()).enumerate() {
        for (g, gc) in gv.iter().zip(gcv.iter()) {
            assert!(
                (g - gc).abs() < 1e-9,
                "checkpointed vs non-checkpointed gradient mismatch for weight {i}: {g} vs {gc}"
            );
        }
    }

    // 2. Independent real-value check: compare the non-checkpointed analytic
    // gradient against a central finite difference of the same forward pass
    // (this is the check that a silent zero-fallback, or any other backward
    // bug, cannot pass: the network has no relu dead-zones here since every
    // input/weight is strictly positive).
    for (i, gv) in grads_val.iter().enumerate() {
        let fd_grad = deep_network_fd_grad(&input_arr, &w1_arr, &w2_arr, &w3_arr, i);
        assert_eq!(gv.shape(), fd_grad.shape());
        for (a, f) in gv.iter().zip(fd_grad.iter()) {
            assert!(
                (a - f).abs() < 1e-4,
                "weight {i}: analytic gradient {a} does not match finite-difference {f}"
            );
        }
    }
}

#[test]
#[allow(dead_code)]
fn test_adaptive_checkpoint() {
    ag::run(|ctx| {
        // Differentiated below, so `T::variable` is required.
        let a = T::variable(array![[1.0, 2.0], [3.0, 4.0]], ctx);
        let b = T::variable(array![[5.0, 6.0], [7.0, 8.0]], ctx);

        // Create a large tensor that should be checkpointed. This one is never
        // differentiated, so a plain constant is fine.
        let large_tensor = T::ones(&[100, 100], ctx);

        // Set threshold between small and large
        let threshold = 1000;

        // Regular computation
        let c1 = T::matmul(a, b);
        let d1 = T::sum_all(c1);

        // Small tensor with adaptive checkpoint (should not checkpoint)
        let c2 = T::adaptive_checkpoint(&T::matmul(a, b), threshold);
        let d2 = T::sum_all(c2);

        // Large tensor with adaptive checkpoint (should checkpoint)
        let _large_result = T::adaptive_checkpoint(&large_tensor, threshold);

        // Both computations should yield the same result for small tensors
        let result1 = d1.eval(ctx).expect("Test: operation failed");
        let result2 = d2.eval(ctx).expect("Test: operation failed");

        assert_eq!(result1[[]], result2[[]]);

        // Test gradients for small tensors: adaptive checkpointing must
        // preserve the gradient exactly, whether or not it decides to
        // actually checkpoint.
        let grad1 = T::grad(&[d1], &[&a])[0];
        let grad2 = T::grad(&[d2], &[&a])[0];

        let grad1_2d = grad1
            .eval(ctx)
            .expect("First gradient should be evaluable")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");
        let grad2_2d = grad2
            .eval(ctx)
            .expect("Second gradient should be evaluable")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");

        // Same hand-derived value as test_checkpoint_basic (identical a, b).
        let expected = [[11.0_f64, 15.0], [11.0, 15.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (grad1_2d[[i, j]] - expected[i][j]).abs() < 1e-9_f64,
                    "non-checkpointed gradient[{i}][{j}] = {}, expected {}",
                    grad1_2d[[i, j]],
                    expected[i][j]
                );
                assert!(
                    (grad1_2d[[i, j]] - grad2_2d[[i, j]]).abs() < 1e-9_f64,
                    "adaptive checkpointing must preserve the gradient exactly: [{i}][{j}] {} vs {}",
                    grad1_2d[[i, j]],
                    grad2_2d[[i, j]]
                );
            }
        }
    });
}

#[test]
#[allow(dead_code)]
fn test_checkpoint_group() {
    ag::run(|ctx| {
        // Differentiated below, so `T::variable` is required.
        let a = T::variable(array![[1.0, 2.0], [3.0, 4.0]], ctx);
        let b = T::variable(array![[5.0, 6.0], [7.0, 8.0]], ctx);

        // Create a checkpoint group
        let ckpt_group = T::CheckpointGroup::new(ctx);

        // Run functions directly without using a separate closure

        // Regular computation
        let c1 = T::matmul(a, b);
        let d1 = T::transpose(c1, &[1, 0]);

        // Checkpoint group computation
        let (c2, d2) = ckpt_group.checkpoint_fn((a, &b), |inputs| {
            let c = T::matmul(inputs.0, inputs.1);
            let d = T::transpose(c, &[1, 0]);
            (c, d)
        });

        // Verify results are the same
        let c1_val = c1.eval(ctx).expect("Test: operation failed");
        let c2_val = c2.eval(ctx).expect("Test: operation failed");
        let d1_val = d1.eval(ctx).expect("Test: operation failed");
        let d2_val = d2.eval(ctx).expect("Test: operation failed");

        // Compare c1 and c2
        let c1_2d = c1_val
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");
        let c2_2d = c2_val
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");

        for i in 0..2 {
            for j in 0..2 {
                let diff: f64 = c1_2d[[i, j]] - c2_2d[[i, j]];
                assert!(diff.abs() < 1e-10_f64);
            }
        }

        // Compare d1 and d2
        let d1_2d = d1_val
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");
        let d2_2d = d2_val
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");

        for i in 0..2 {
            for j in 0..2 {
                let diff: f64 = d1_2d[[i, j]] - d2_2d[[i, j]];
                assert!(diff.abs() < 1e-10_f64);
            }
        }

        // Compute gradients through both outputs
        let loss1 = T::sum_all(c1) + T::sum_all(d1);
        let loss2 = T::sum_all(c2) + T::sum_all(d2);

        let grad1 = T::grad(&[loss1], &[&a])[0];
        let grad2 = T::grad(&[loss2], &[&a])[0];

        let grad1_2d = grad1
            .eval(ctx)
            .expect("First gradient should be evaluable")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");
        let grad2_2d = grad2
            .eval(ctx)
            .expect("Second gradient should be evaluable")
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");

        // Hand-derived: transpose doesn't change the elementwise sum, so
        // loss = sum(c1) + sum(transpose(c1)) = 2 * sum(matmul(a,b)); its
        // gradient is twice test_checkpoint_basic's [[11,15],[11,15]].
        let expected = [[22.0, 30.0], [22.0, 30.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (grad1_2d[[i, j]] - expected[i][j]).abs() < 1e-9,
                    "non-checkpointed gradient[{i}][{j}] = {}, expected {}",
                    grad1_2d[[i, j]],
                    expected[i][j]
                );
                assert!(
                    (grad1_2d[[i, j]] - grad2_2d[[i, j]]).abs() < 1e-9,
                    "checkpoint group must preserve the gradient exactly: [{i}][{j}] {} vs {}",
                    grad1_2d[[i, j]],
                    grad2_2d[[i, j]]
                );
            }
        }
    });
}

#[test]
#[allow(dead_code)]
fn test_stop_gradient() {
    ag::run(|ctx| {
        // `a` must be a real variable: only then is the "gradient blocked"
        // check below meaningful (see test_detach for the same reasoning).
        let a = T::variable(array![[1.0, 2.0], [3.0, 4.0]], ctx);

        // Apply stop_gradient (which is an alias for detach)
        let b = T::stop_gradient(a);
        let c = T::square(b);
        let d = T::sum_all(c);

        // Forward pass should work normally
        let result = d.eval(ctx).expect("Test: operation failed");
        assert_eq!(result[[]], 30.0); // 1²+2²+3²+4² = 1+4+9+16 = 30

        // But gradients should be zero
        let grad = T::grad(&[d], &[&a])[0];
        let grad_val = grad.eval(ctx).expect("Test: operation failed");

        let grad_2d = grad_val
            .view()
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .expect("Test: operation failed");
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(grad_2d[[i, j]], 0.0);
            }
        }
    });
}

#[test]
#[allow(dead_code)]
fn test_checkpoint_profiler() {
    ag::run(|ctx| {
        // Reset statistics
        T::CheckpointProfiler::reset_statistics();

        // Enable tracking
        T::CheckpointProfiler::start_tracking();

        // Create a tensor and checkpoint it. Never differentiated here, so a
        // plain constant is the right (and honest) choice.
        let a = T::convert_to_tensor(array![[1.0, 2.0], [3.0, 4.0]], ctx);
        let b = T::checkpoint(&a);
        let _ = b.eval(ctx).expect("Test: operation failed");

        // Check that one checkpoint was recorded
        assert!(T::CheckpointProfiler::checkpoint_count() > 0);

        // Memory saved should be approximately the size of the tensor
        // 4 elements * 8 bytes (f64) = 32 bytes
        assert!(T::CheckpointProfiler::memory_saved() >= 32);

        // Reset and verify
        T::CheckpointProfiler::reset_statistics();
        assert_eq!(T::CheckpointProfiler::checkpoint_count(), 0);
        assert_eq!(T::CheckpointProfiler::memory_saved(), 0);

        // Stop tracking
        T::CheckpointProfiler::stop_tracking();
    });
}
