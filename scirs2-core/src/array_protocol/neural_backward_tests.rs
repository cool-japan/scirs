// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Tests for `Layer::backward`/`Sequential::backward` — split out to keep
//! neural.rs under 2000 lines (see grad.rs/grad_tests.rs for the same
//! pattern).
//!
//! Before these were implemented, `Sequential::backward` always returned an
//! empty `GradientDict` and every `Layer::backward` simply didn't exist, so
//! none of these tests could have passed. Each one uses non-constant data
//! and (where practical) verifies the analytic gradient against an
//! independent central finite-difference computation, which a fabricated,
//! zero, or all-constant gradient could not survive.

use super::*;
use ::ndarray::{Array2, ArrayD};

/// Reads layer `layer_idx`'s first parameter (assumed 2D) out of `model`.
fn get_weight_ix2(model: &Sequential, layer_idx: usize) -> Array2<f64> {
    model.layers()[layer_idx].parameters()[0]
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        .expect("weight should be a NdarrayWrapper<f64, Ix2>")
        .as_array()
        .clone()
}

/// Overwrites layer `layer_idx`'s first parameter in `model` with `w`.
fn set_weight_ix2(model: &mut Sequential, layer_idx: usize, w: Array2<f64>) {
    let mut params = model.layers_mut()[layer_idx].parameters_mut();
    *params[0] = Box::new(NdarrayWrapper::new(w));
}

/// L = sum(forward(input)^2), assuming a 2D output.
fn model_loss(model: &Sequential, input: &dyn ArrayProtocol) -> f64 {
    let output = model.forward(input).expect("forward should succeed");
    let out_arr = output
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        .expect("output should be a NdarrayWrapper<f64, Ix2>")
        .as_array();
    out_arr.iter().map(|v| v * v).sum()
}

/// The headline regression test: a tiny 2-layer (Linear -> Linear) model,
/// with non-constant weights and input, whose `Sequential::backward` output
/// is checked element-by-element against central finite differences of the
/// same scalar loss. Bias/activation are intentionally left off both layers
/// so this test isolates exactly what `Sequential::backward` is responsible
/// for: recomputing the forward pass and chaining `Layer::backward` calls
/// correctly across layers (their own gradient math is separately covered
/// by the per-layer tests further down).
#[test]
fn test_sequential_backward_matches_finite_differences() {
    crate::array_protocol::init();

    // in=3 -> hidden=2 -> out=1.
    let w0 = Array2::from_shape_fn((2, 3), |(i, j)| ((i * 3 + j) as f64) * 0.3 - 0.7);
    let w1 = Array2::from_shape_fn((1, 2), |(i, j)| ((i * 2 + j) as f64) * 0.5 + 0.2);

    let mut model = Sequential::new(
        "tiny_mlp",
        vec![
            Box::new(Linear::new(
                "fc0",
                Box::new(NdarrayWrapper::new(w0)),
                None,
                None,
            )),
            Box::new(Linear::new(
                "fc1",
                Box::new(NdarrayWrapper::new(w1)),
                None,
                None,
            )),
        ],
    );

    // Non-constant input: 3 features, batch of 2.
    let input = Array2::from_shape_fn((3, 2), |(i, j)| {
        (i as f64 + 1.0) * 0.4 + (j as f64) * 0.9 - 0.3
    });
    let wrapped_input = NdarrayWrapper::new(input);

    let output = model
        .forward(&wrapped_input)
        .expect("forward should succeed");
    let out_arr = output
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
        .expect("output should be Ix2")
        .as_array()
        .clone();

    // L = sum(output^2)  =>  dL/d(output) = 2 * output (non-constant, since
    // `output` is non-constant for this non-constant input/weights).
    let grad_output = out_arr.mapv(|v| 2.0 * v);
    let wrapped_grad_output = NdarrayWrapper::new(grad_output);

    let gradients = model
        .backward(&wrapped_input, &wrapped_grad_output)
        .expect("backward should succeed");
    assert_eq!(
        gradients.len(),
        2,
        "expected exactly one weight gradient per layer"
    );

    let eps = 1e-5;
    let tol = 1e-4;

    for (key, layer_idx) in [("0.weights", 0usize), ("1.weights", 1usize)] {
        let analytic = gradients
            .get(key)
            .unwrap_or_else(|| panic!("missing gradient for '{key}'"))
            .as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix2>>()
            .unwrap_or_else(|| panic!("'{key}' gradient should be a NdarrayWrapper<f64, Ix2>"))
            .as_array()
            .clone();

        let original = get_weight_ix2(&model, layer_idx);
        let (rows, cols) = original.dim();
        let mut any_nonzero = false;
        for i in 0..rows {
            for j in 0..cols {
                let mut plus = original.clone();
                plus[[i, j]] += eps;
                set_weight_ix2(&mut model, layer_idx, plus);
                let loss_plus = model_loss(&model, &wrapped_input);

                let mut minus = original.clone();
                minus[[i, j]] -= eps;
                set_weight_ix2(&mut model, layer_idx, minus);
                let loss_minus = model_loss(&model, &wrapped_input);

                // Restore the original weight before the next iteration.
                set_weight_ix2(&mut model, layer_idx, original.clone());

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let got = analytic[[i, j]];
                if got != 0.0 {
                    any_nonzero = true;
                }
                assert!(
                    (numeric - got).abs() < tol,
                    "{key}[{i},{j}]: analytic={got}, finite-diff={numeric}"
                );
            }
        }
        assert!(
            any_nonzero,
            "'{key}' gradient is all-zero — cannot distinguish from a fabricated stub"
        );
    }
}

/// `MaxPool2D::backward` should route each output cell's gradient to
/// exactly the input cell that was the max in its pooling window (and
/// nowhere else) — verified exactly (not just via finite differences,
/// since max-pooling's gradient is a routing function, not a smooth one).
#[test]
fn test_max_pool2d_backward_routes_to_argmax() {
    let pool = MaxPool2D::new("pool_test", (2, 2), Some((2, 2)), (0, 0));

    // Distinct, non-constant values so each 2x2 block's max is unambiguous:
    // block maxima end up at (1,1)=5, (1,3)=7, (3,1)=13, (3,3)=15.
    let input = Array4::from_shape_fn((1, 4, 4, 1), |(_, h, w, _)| (h * 4 + w) as f64);
    let wrapped_input = NdarrayWrapper::new(input);

    let grad_output =
        Array4::from_shape_fn((1, 2, 2, 1), |(_, oh, ow, _)| (oh * 2 + ow + 1) as f64);
    let wrapped_grad = NdarrayWrapper::new(grad_output.clone());

    let layer_grad = pool
        .backward(&wrapped_input, &wrapped_grad)
        .expect("max_pool2d backward should succeed");
    let grad_input = layer_grad
        .grad_input
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix4>>()
        .expect("grad_input should be Ix4")
        .as_array()
        .clone();

    let mut expected = Array4::<f64>::zeros((1, 4, 4, 1));
    expected[[0, 1, 1, 0]] = grad_output[[0, 0, 0, 0]];
    expected[[0, 1, 3, 0]] = grad_output[[0, 0, 1, 0]];
    expected[[0, 3, 1, 0]] = grad_output[[0, 1, 0, 0]];
    expected[[0, 3, 3, 0]] = grad_output[[0, 1, 1, 0]];

    assert_eq!(grad_input, expected);
    assert!(layer_grad.grad_params.is_empty());
}

/// `Conv2D::backward`'s gradients (w.r.t. both the filters and the input)
/// checked against central finite differences, on a tiny non-constant
/// example.
#[test]
fn test_conv2d_backward_matches_finite_differences() {
    let filters = Array4::from_shape_fn((2, 2, 1, 1), |(i, j, _, _)| {
        ((i * 2 + j) as f64) * 0.4 - 0.3
    });
    let input = Array4::from_shape_fn((1, 3, 3, 1), |(_, h, w, _)| {
        (h as f64) * 0.7 - (w as f64) * 0.5 + 0.2
    });
    let wrapped_input = NdarrayWrapper::new(input.clone());

    let loss_fn = |filt: &Array4<f64>, inp: &Array4<f64>| -> f64 {
        let conv = Conv2D::new(
            "t",
            Box::new(NdarrayWrapper::new(filt.clone())),
            None,
            (1, 1),
            (0, 0),
            None,
        );
        let wi = NdarrayWrapper::new(inp.clone());
        let out = conv.forward(&wi).expect("conv2d forward should succeed");
        out.as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix4>>()
            .expect("output should be Ix4")
            .as_array()
            .iter()
            .map(|v| v * v)
            .sum()
    };

    let conv = Conv2D::new(
        "conv_test",
        Box::new(NdarrayWrapper::new(filters.clone())),
        None,
        (1, 1),
        (0, 0),
        None,
    );
    let output = conv
        .forward(&wrapped_input)
        .expect("conv2d forward should succeed");
    let out_arr = output
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix4>>()
        .expect("output should be Ix4")
        .as_array()
        .clone();
    let grad_output = out_arr.mapv(|v| 2.0 * v);
    let wrapped_grad_output = NdarrayWrapper::new(grad_output);

    let layer_grad = conv
        .backward(&wrapped_input, &wrapped_grad_output)
        .expect("conv2d backward should succeed");
    let grad_filters = layer_grad.grad_params[0]
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix4>>()
        .expect("grad_filters should be Ix4")
        .as_array()
        .clone();
    let grad_input = layer_grad
        .grad_input
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix4>>()
        .expect("grad_input should be Ix4")
        .as_array()
        .clone();

    let eps = 1e-5;
    let tol = 1e-4;

    for i in 0..2 {
        for j in 0..2 {
            let mut plus = filters.clone();
            plus[[i, j, 0, 0]] += eps;
            let mut minus = filters.clone();
            minus[[i, j, 0, 0]] -= eps;
            let numeric = (loss_fn(&plus, &input) - loss_fn(&minus, &input)) / (2.0 * eps);
            let got = grad_filters[[i, j, 0, 0]];
            assert!(
                (numeric - got).abs() < tol,
                "grad_filters[{i},{j},0,0]: analytic={got}, finite-diff={numeric}"
            );
        }
    }

    for h in 0..3 {
        for w in 0..3 {
            let mut plus = input.clone();
            plus[[0, h, w, 0]] += eps;
            let mut minus = input.clone();
            minus[[0, h, w, 0]] -= eps;
            let numeric = (loss_fn(&filters, &plus) - loss_fn(&filters, &minus)) / (2.0 * eps);
            let got = grad_input[[0, h, w, 0]];
            assert!(
                (numeric - got).abs() < tol,
                "grad_input[0,{h},{w},0]: analytic={got}, finite-diff={numeric}"
            );
        }
    }
}

/// `BatchNorm::backward`'s gradients (w.r.t. `scale` and `offset`) checked
/// against central finite differences. `running_mean`/`running_var` are
/// treated as fixed buffers (not differentiated), matching
/// `ml_ops::batch_norm`'s own forward semantics.
#[test]
fn test_batch_norm_backward_matches_finite_differences() {
    let channels = 2;
    // `ml_ops::batch_norm`'s fallback requires `scale`/`offset`/`mean`/
    // `variance` to be `NdarrayWrapper<f64, IxDyn>` specifically (not
    // `Ix1`), unlike `BatchNorm::withshape` — construct them directly here.
    let scale: ArrayD<f64> = Array::from_shape_fn(channels, |c| 1.0 + 0.3 * c as f64).into_dyn();
    let offset: ArrayD<f64> = Array::from_shape_fn(channels, |c| 0.1 - 0.2 * c as f64).into_dyn();
    let running_mean: ArrayD<f64> = Array::from_shape_fn(channels, |c| 0.05 * c as f64).into_dyn();
    let running_var: ArrayD<f64> =
        Array::from_shape_fn(channels, |c| 1.0 + 0.5 * c as f64).into_dyn();
    let epsilon = 1e-4;

    let make_bn = |s: &ArrayD<f64>, o: &ArrayD<f64>| {
        BatchNorm::new(
            "bn_test",
            Box::new(NdarrayWrapper::new(s.clone())),
            Box::new(NdarrayWrapper::new(o.clone())),
            Box::new(NdarrayWrapper::new(running_mean.clone())),
            Box::new(NdarrayWrapper::new(running_var.clone())),
            epsilon,
        )
    };

    let input = Array4::from_shape_fn((2, 2, 2, 2), |(b, h, w, c)| {
        (b as f64) * 0.7 + (h as f64) * 0.3 - (w as f64) * 0.2 + (c as f64) * 1.1 + 0.4
    });
    let wrapped_input = NdarrayWrapper::new(input);

    let loss_fn = |s: &ArrayD<f64>, o: &ArrayD<f64>| -> f64 {
        let bn = make_bn(s, o);
        let out = bn
            .forward(&wrapped_input)
            .expect("batch_norm forward should succeed");
        out.as_any()
            .downcast_ref::<NdarrayWrapper<f64, Ix4>>()
            .expect("output should be Ix4")
            .as_array()
            .iter()
            .map(|v| v * v)
            .sum()
    };

    let bn = make_bn(&scale, &offset);
    let output = bn
        .forward(&wrapped_input)
        .expect("batch_norm forward should succeed");
    let out_arr = output
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, Ix4>>()
        .expect("output should be Ix4")
        .as_array()
        .clone();
    let grad_output = out_arr.mapv(|v| 2.0 * v);
    let wrapped_grad_output = NdarrayWrapper::new(grad_output);

    let layer_grad = bn
        .backward(&wrapped_input, &wrapped_grad_output)
        .expect("batch_norm backward should succeed");
    let grad_scale = layer_grad.grad_params[0]
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("grad_scale should be IxDyn (scale itself was constructed as IxDyn)")
        .as_array()
        .clone();
    let grad_offset = layer_grad.grad_params[1]
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("grad_offset should be IxDyn (offset itself was constructed as IxDyn)")
        .as_array()
        .clone();

    let eps = 1e-5;
    let tol = 1e-4;
    for c in 0..channels {
        let mut splus = scale.clone();
        splus[[c]] += eps;
        let mut sminus = scale.clone();
        sminus[[c]] -= eps;
        let numeric = (loss_fn(&splus, &offset) - loss_fn(&sminus, &offset)) / (2.0 * eps);
        let got = grad_scale[[c]];
        assert!(
            (numeric - got).abs() < tol,
            "grad_scale[{c}]: analytic={got}, finite-diff={numeric}"
        );

        let mut oplus = offset.clone();
        oplus[[c]] += eps;
        let mut ominus = offset.clone();
        ominus[[c]] -= eps;
        let numeric_o = (loss_fn(&scale, &oplus) - loss_fn(&scale, &ominus)) / (2.0 * eps);
        let got_o = grad_offset[[c]];
        assert!(
            (numeric_o - got_o).abs() < tol,
            "grad_offset[{c}]: analytic={got_o}, finite-diff={numeric_o}"
        );
    }
}

/// With a fixed seed, `Dropout::backward` must reproduce exactly the same
/// mask `forward()` drew: wherever forward zeroed an element, backward's
/// gradient there must also be exactly zero; wherever forward kept (and
/// scaled) an element, backward's gradient there must carry the same scale.
#[test]
fn test_dropout_backward_matches_seeded_forward_mask() {
    let seed = 42u64;
    let rate = 0.5;
    let dropout = Dropout::new("drop_test", rate, Some(seed));
    assert!(dropout.is_training());

    let input = Array::from_shape_fn(IxDyn(&[6]), |idx| (idx[0] as f64) * 1.3 + 0.4);
    let wrapped_input = NdarrayWrapper::new(input);
    let output = dropout
        .forward(&wrapped_input)
        .expect("dropout forward should succeed");
    let out_arr = output
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("output should be IxDyn")
        .as_array()
        .clone();

    let grad_output = Array::from_elem(IxDyn(&[6]), 1.0);
    let wrapped_grad = NdarrayWrapper::new(grad_output);
    let layer_grad = dropout
        .backward(&wrapped_input, &wrapped_grad)
        .expect("dropout backward should succeed");
    let grad_input = layer_grad
        .grad_input
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("grad_input should be IxDyn")
        .as_array()
        .clone();

    let scale = 1.0 / (1.0 - rate);
    let mut num_kept = 0;
    for i in 0..6 {
        if out_arr[[i]] == 0.0 {
            assert_eq!(
                grad_input[[i]],
                0.0,
                "index {i} was dropped by forward() but not by backward()"
            );
        } else {
            num_kept += 1;
            assert!(
                (grad_input[[i]] - scale).abs() < 1e-12,
                "index {i}: expected gradient scaled by {scale}, got {}",
                grad_input[[i]]
            );
        }
    }
    // A degenerate all-dropped or all-kept draw wouldn't actually exercise
    // mask reproduction; this seed/rate/length combination is known to
    // produce a genuine mix (and this assertion pins that down).
    assert!(
        num_kept > 0 && num_kept < 6,
        "expected a non-trivial mix of dropped/kept elements for seed {seed}, got {num_kept}/6 kept"
    );
}

/// Training-mode dropout without a fixed seed cannot be backpropagated
/// through faithfully (forward doesn't cache the mask it drew), so
/// `backward` must return an honest error rather than silently fabricating
/// a plausible-looking gradient.
#[test]
fn test_dropout_backward_without_seed_in_training_is_honest_error() {
    let dropout = Dropout::new("drop_test_noseed", 0.5, None);
    let input = Array::from_elem(IxDyn(&[3]), 1.0);
    let wrapped = NdarrayWrapper::new(input);
    let grad_output = Array::from_elem(IxDyn(&[3]), 1.0);
    let wrapped_grad = NdarrayWrapper::new(grad_output);

    let result = dropout.backward(&wrapped, &wrapped_grad);
    assert!(
        result.is_err(),
        "backward() without a fixed seed in training mode must error, not fabricate a gradient"
    );
}

/// Eval-mode dropout is the identity function, so its backward must be too.
#[test]
fn test_dropout_backward_eval_mode_is_identity() {
    let mut dropout = Dropout::new("drop_test_eval", 0.5, None);
    dropout.eval();

    let input = Array::from_shape_fn(IxDyn(&[4]), |idx| (idx[0] as f64) * 2.1);
    let wrapped = NdarrayWrapper::new(input);
    let grad_output = Array::from_shape_fn(IxDyn(&[4]), |idx| (idx[0] as f64) * 0.3 + 1.0);
    let wrapped_grad = NdarrayWrapper::new(grad_output.clone());

    let layer_grad = dropout
        .backward(&wrapped, &wrapped_grad)
        .expect("eval-mode backward should succeed");
    let grad_input = layer_grad
        .grad_input
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
        .expect("grad_input should be IxDyn")
        .as_array()
        .clone();

    assert_eq!(grad_input, grad_output);
    assert!(layer_grad.grad_params.is_empty());
}

/// `MultiHeadAttention::forward` is already documented as a simplified
/// placeholder (no real per-head splitting), so `backward` must say so
/// honestly rather than fabricate a gradient for an operation it doesn't
/// actually perform.
#[test]
fn test_multi_head_attention_backward_is_honest_error() {
    let mha = MultiHeadAttention::with_params("mha_test", 2, 4);
    let input = Array2::<f64>::ones((4, 4));
    let wrapped = NdarrayWrapper::new(input.clone());
    let wrapped_grad = NdarrayWrapper::new(input);

    let result = mha.backward(&wrapped, &wrapped_grad);
    assert!(
        result.is_err(),
        "MultiHeadAttention::backward must not silently fabricate a gradient"
    );
}
