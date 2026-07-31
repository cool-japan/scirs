//! Numerical verification of backpropagation through the normalization
//! layers (`BatchNorm`, `LayerNorm`, and the `norm_variants` family:
//! `RMSNorm`, `GroupNorm`, `InstanceNorm`, `WeightNorm`).
//!
//! Every analytic gradient produced by `Layer::backward` is compared against
//! a central finite-difference estimate of the same derivative, and every
//! parameterized layer is trained for a few SGD steps on a synthetic
//! regression task to prove that `update()` actually consumes the computed
//! gradients: a no-op `update()` or an identity-passthrough `backward()`
//! cannot make the loss sequence strictly decrease. Mirrors the conventions
//! established in `tests/recurrent_backprop.rs`.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::SeedableRng;
use scirs2_neural::layers::{
    BatchNorm, GroupNorm, InstanceNorm, Layer, LayerNorm, RMSNorm, WeightNorm,
};

/// Relative tolerance for the analytic-vs-numeric gradient comparison.
const RTOL: f64 = 1e-4;
/// Step used by the central finite differences.
const EPS: f64 = 1e-5;

/// Deterministic, non-constant test data.
///
/// All-ones/all-zero data cannot distinguish a real gradient from a
/// fabricated one, so every element here is distinct and both signs occur.
fn varied(shape: &[usize], seed: f64) -> Array<f64, IxDyn> {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.7 + seed;
            0.9 * x.sin() + 0.4 * (0.31 * x).cos() - 0.15
        })
        .collect();
    Array::from_shape_vec(IxDyn(shape), values).expect("test data shape must be valid")
}

fn assert_close(analytic: f64, numeric: f64, what: &str) {
    let scale = 1.0 + analytic.abs().max(numeric.abs());
    assert!(
        (analytic - numeric).abs() <= RTOL * scale,
        "{what}: analytic {analytic:.10e} vs numeric {numeric:.10e}"
    );
}

/// Loss `sum(output * weights)`, whose gradient w.r.t. the output is `weights`.
fn weighted_loss(output: &Array<f64, IxDyn>, weights: &Array<f64, IxDyn>) -> f64 {
    output
        .iter()
        .zip(weights.iter())
        .map(|(&o, &w)| o * w)
        .sum()
}

/// Check `backward`'s input gradient against central finite differences.
fn check_input_gradient<L: Layer<f64>>(
    layer: &L,
    input: &Array<f64, IxDyn>,
    grad_weights: &Array<f64, IxDyn>,
    label: &str,
) {
    let mut numeric = Array::<f64, IxDyn>::zeros(input.dim());
    let mut probe = input.clone();
    for idx in 0..input.len() {
        let original = probe.as_slice_mut().expect("contiguous input")[idx];

        probe.as_slice_mut().expect("contiguous input")[idx] = original + EPS;
        let plus = weighted_loss(
            &layer.forward(&probe).expect("forward must succeed"),
            grad_weights,
        );

        probe.as_slice_mut().expect("contiguous input")[idx] = original - EPS;
        let minus = weighted_loss(
            &layer.forward(&probe).expect("forward must succeed"),
            grad_weights,
        );

        probe.as_slice_mut().expect("contiguous input")[idx] = original;
        numeric.as_slice_mut().expect("contiguous grad")[idx] = (plus - minus) / (2.0 * EPS);
    }

    // Restore the forward caches for the unperturbed input before backprop.
    layer.forward(input).expect("forward must succeed");
    let analytic = layer
        .backward(input, grad_weights)
        .expect("backward must succeed");

    assert_eq!(analytic.shape(), input.shape());
    for idx in 0..input.len() {
        assert_close(
            analytic.as_slice().expect("contiguous grad")[idx],
            numeric.as_slice().expect("contiguous grad")[idx],
            &format!("{label} d(loss)/d(input[{idx}])"),
        );
    }
}

/// Train a layer for a few SGD steps on a fixed regression target and return
/// the loss after each step. A zero-gradient backward or a gradient-ignoring
/// update cannot make this sequence strictly decreasing.
fn train_losses<L: Layer<f64>>(
    layer: &mut L,
    input: &Array<f64, IxDyn>,
    target: &Array<f64, IxDyn>,
    learning_rate: f64,
    steps: usize,
) -> Vec<f64> {
    let scale = 2.0 / target.len() as f64;
    let mut losses = Vec::with_capacity(steps);
    for _ in 0..steps {
        let output = layer.forward(input).expect("forward must succeed");
        let diff = &output - target;
        losses.push(diff.iter().map(|d| d * d).sum::<f64>() / target.len() as f64);
        let grad_output = diff.mapv(|d| d * scale);
        layer
            .backward(input, &grad_output)
            .expect("backward must succeed");
        layer.update(learning_rate).expect("update must succeed");
    }
    losses
}

fn assert_strictly_decreasing(losses: &[f64], label: &str) {
    for pair in losses.windows(2) {
        assert!(
            pair[1] < pair[0],
            "{label}: loss must strictly decrease, got {losses:?}"
        );
    }
}

// =======================================================================
// LayerNorm (bonus sanity check; the implementation predates this file)
// =======================================================================

#[test]
fn layernorm_input_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([3; 32]);
    let ln = LayerNorm::<f64>::new(5, 1e-5, &mut rng).expect("LayerNorm construction");
    let input = varied(&[3, 5], 0.55);
    let grad_weights = varied(&[3, 5], 2.05);
    check_input_gradient(&ln, &input, &grad_weights, "LayerNorm");
}

#[test]
fn layernorm_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([103; 32]);
    let mut ln = LayerNorm::<f64>::new(4, 1e-5, &mut rng).expect("LayerNorm construction");
    let input = varied(&[3, 4], 0.9);
    let target = varied(&[3, 4], 6.5).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut ln, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "LayerNorm");
}

// =======================================================================
// BatchNorm
// =======================================================================

#[test]
fn batchnorm_training_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([41; 32]);
    let bn = BatchNorm::<f64>::new(4, 0.1, 1e-5, &mut rng).expect("BatchNorm construction");
    let input = varied(&[5, 4], 0.3);
    let grad_weights = varied(&[5, 4], 2.2);
    check_input_gradient(&bn, &input, &grad_weights, "BatchNorm(train)");
}

#[test]
fn batchnorm_training_gradient_matches_finite_differences_with_spatial_dims() {
    // Exercises the "S > 1" trailing-spatial-dimension path (`[N, C, H, W]`).
    let mut rng = SmallRng::from_seed([43; 32]);
    let bn = BatchNorm::<f64>::new(2, 0.1, 1e-5, &mut rng).expect("BatchNorm construction");
    let input = varied(&[2, 2, 2, 2], 0.5);
    let grad_weights = varied(&[2, 2, 2, 2], 1.5);
    check_input_gradient(&bn, &input, &grad_weights, "BatchNorm(train,4D)");
}

#[test]
fn batchnorm_eval_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([47; 32]);
    let mut bn = BatchNorm::<f64>::new(3, 0.5, 1e-5, &mut rng).expect("BatchNorm construction");
    // Populate running stats away from the (0, 1) defaults before switching.
    let warmup = varied(&[6, 3], 1.1);
    bn.forward(&warmup).expect("forward must succeed");
    bn.set_training(false);

    let input = varied(&[2, 3], 4.4);
    let grad_weights = varied(&[2, 3], 0.9);
    check_input_gradient(&bn, &input, &grad_weights, "BatchNorm(eval)");
}

#[test]
fn batchnorm_normalizes_batch_statistics() {
    // With gamma=1, beta=0 (the initial values), a training-mode forward
    // pass must leave each channel with batch mean ~0 and (biased) variance
    // ~1 over the batch -- the defining property of batch normalization.
    let mut rng = SmallRng::from_seed([53; 32]);
    let bn = BatchNorm::<f64>::new(3, 0.1, 1e-8, &mut rng).expect("BatchNorm construction");
    let input = varied(&[8, 3], 0.2);
    let output = bn.forward(&input).expect("forward must succeed");

    for c in 0..3 {
        let col: Vec<f64> = (0..8).map(|b| output[[b, c]]).collect();
        let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
        let var: f64 = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;
        assert!(mean.abs() < 1e-6, "channel {c}: post-BN mean {mean} not ~0");
        assert!(
            (var - 1.0).abs() < 1e-4,
            "channel {c}: post-BN var {var} not ~1"
        );
    }
}

#[test]
fn batchnorm_running_stats_converge_to_batch_statistics() {
    let mut rng = SmallRng::from_seed([59; 32]);
    let momentum = 0.3;
    let bn = BatchNorm::<f64>::new(2, momentum, 1e-8, &mut rng).expect("BatchNorm construction");
    let input = varied(&[10, 2], 3.0);

    // The batch statistics of this fixed input (ground truth to converge to).
    let mut true_mean = [0.0f64; 2];
    let mut true_var_unbiased = [0.0f64; 2];
    for c in 0..2 {
        let col: Vec<f64> = (0..10).map(|b| input[[b, c]]).collect();
        let mean = col.iter().sum::<f64>() / col.len() as f64;
        let biased_var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;
        true_mean[c] = mean;
        true_var_unbiased[c] = biased_var * 10.0 / 9.0;
    }

    // Feed the *same* batch repeatedly: exponential-moving-average running
    // stats must converge geometrically to that batch's own statistics
    // ((1 - momentum)^iterations shrinks the initial-state error to ~1e-14
    // well before 80 iterations).
    for _ in 0..80 {
        bn.forward(&input).expect("forward must succeed");
    }

    let running_mean = bn.running_mean();
    let running_var = bn.running_var();
    for c in 0..2 {
        assert!(
            (running_mean[[c]] - true_mean[c]).abs() < 1e-6,
            "channel {c}: running mean {} did not converge to {}",
            running_mean[[c]],
            true_mean[c]
        );
        assert!(
            (running_var[[c]] - true_var_unbiased[c]).abs() < 1e-6,
            "channel {c}: running var {} did not converge to {}",
            running_var[[c]],
            true_var_unbiased[c]
        );
    }
}

#[test]
fn batchnorm_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([61; 32]);
    let mut bn = BatchNorm::<f64>::new(3, 0.5, 1e-5, &mut rng).expect("BatchNorm construction");
    let input = varied(&[4, 3], 0.75);
    let target = varied(&[4, 3], 5.25).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut bn, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "BatchNorm");
}

#[test]
fn batchnorm_set_training_dispatches_through_trait_object() {
    // `Layer::set_training`/`is_training` must be overridden by `BatchNorm`,
    // or a `Box<dyn Layer<F>>` (exactly how `Sequential` stores its layers)
    // could never actually switch a `BatchNorm` to evaluation mode, since
    // the default trait methods are no-ops.
    let mut rng = SmallRng::from_seed([67; 32]);
    let bn = BatchNorm::<f64>::new(2, 0.5, 1e-5, &mut rng).expect("BatchNorm construction");
    let mut boxed: Box<dyn Layer<f64>> = Box::new(bn);

    let warmup = varied(&[6, 2], 0.5);
    boxed.forward(&warmup).expect("forward must succeed");

    assert!(boxed.is_training(), "BatchNorm defaults to training mode");
    boxed.set_training(false);
    assert!(
        !boxed.is_training(),
        "set_training(false) through a trait object must actually switch modes"
    );

    let running_mean = boxed
        .as_any()
        .downcast_ref::<BatchNorm<f64>>()
        .expect("downcast back to BatchNorm")
        .running_mean();
    let running_var = boxed
        .as_any()
        .downcast_ref::<BatchNorm<f64>>()
        .expect("downcast back to BatchNorm")
        .running_var();
    let params = boxed.params();
    let gamma = &params[0];
    let beta = &params[1];

    let single = varied(&[1, 2], 9.0);
    let out = boxed
        .forward(&single)
        .expect("forward must succeed in eval mode");

    let eps = 1e-5;
    for c in 0..2 {
        let inv_std = 1.0 / (running_var[[c]] + eps).sqrt();
        let expected = (single[[0, c]] - running_mean[[c]]) * inv_std * gamma[[c]] + beta[[c]];
        assert!(
            (out[[0, c]] - expected).abs() < 1e-9,
            "channel {c}: eval-mode output {} did not match the running-statistics formula {expected}",
            out[[0, c]]
        );
    }
}

#[test]
fn batchnorm_backward_requires_forward_first() {
    let mut rng = SmallRng::from_seed([71; 32]);
    let bn = BatchNorm::<f64>::new(2, 0.1, 1e-5, &mut rng).expect("BatchNorm construction");
    let input = varied(&[2, 2], 0.0);
    let grad = varied(&[2, 2], 1.0);
    assert!(bn.backward(&input, &grad).is_err());
}

#[test]
fn batchnorm_rejects_wrong_channel_count() {
    let mut rng = SmallRng::from_seed([73; 32]);
    let bn = BatchNorm::<f64>::new(3, 0.1, 1e-5, &mut rng).expect("BatchNorm construction");
    let wrong = varied(&[2, 4], 0.0);
    assert!(bn.forward(&wrong).is_err());
}

// =======================================================================
// RMSNorm
// =======================================================================

#[test]
fn rmsnorm_input_gradient_matches_finite_differences() {
    let rms = RMSNorm::<f64>::new(6, 1e-6).expect("RMSNorm construction");
    let input = varied(&[3, 6], 0.4);
    let grad_weights = varied(&[3, 6], 2.7);
    check_input_gradient(&rms, &input, &grad_weights, "RMSNorm");
}

#[test]
fn rmsnorm_training_reduces_loss() {
    let mut rms = RMSNorm::<f64>::new(4, 1e-6).expect("RMSNorm construction");
    let input = varied(&[3, 4], 1.0);
    let target = varied(&[3, 4], 6.0).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut rms, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "RMSNorm");
}

// =======================================================================
// GroupNorm
// =======================================================================

#[test]
fn groupnorm_3d_input_gradient_matches_finite_differences() {
    let gn = GroupNorm::<f64>::new(2, 4, 1e-5, 2).expect("GroupNorm construction");
    let input = varied(&[2, 3, 4], 0.6);
    let grad_weights = varied(&[2, 3, 4], 1.4);
    check_input_gradient(&gn, &input, &grad_weights, "GroupNorm(3D)");
}

#[test]
fn groupnorm_4d_input_gradient_matches_finite_differences() {
    let gn = GroupNorm::<f64>::new(2, 4, 1e-5, 1).expect("GroupNorm construction");
    let input = varied(&[2, 4, 2, 2], 0.25);
    let grad_weights = varied(&[2, 4, 2, 2], 1.85);
    check_input_gradient(&gn, &input, &grad_weights, "GroupNorm(4D)");
}

#[test]
fn groupnorm_training_reduces_loss() {
    let mut gn = GroupNorm::<f64>::new(2, 4, 1e-5, 2).expect("GroupNorm construction");
    let input = varied(&[2, 3, 4], 0.15);
    let target = varied(&[2, 3, 4], 5.5).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut gn, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "GroupNorm");
}

// =======================================================================
// InstanceNorm
// =======================================================================

#[test]
fn instancenorm_input_gradient_matches_finite_differences() {
    let inst = InstanceNorm::<f64>::new(3, 1e-5, true).expect("InstanceNorm construction");
    let input = varied(&[2, 3, 4], 0.35);
    let grad_weights = varied(&[2, 3, 4], 1.95);
    check_input_gradient(&inst, &input, &grad_weights, "InstanceNorm(affine)");
}

#[test]
fn instancenorm_no_affine_input_gradient_matches_finite_differences() {
    let inst = InstanceNorm::<f64>::new(2, 1e-5, false).expect("InstanceNorm construction");
    let input = varied(&[2, 2, 5], 0.75);
    let grad_weights = varied(&[2, 2, 5], 2.35);
    check_input_gradient(&inst, &input, &grad_weights, "InstanceNorm(no affine)");
}

#[test]
fn instancenorm_training_reduces_loss() {
    let mut inst = InstanceNorm::<f64>::new(3, 1e-5, true).expect("InstanceNorm construction");
    let input = varied(&[2, 3, 4], 0.65);
    let target = varied(&[2, 3, 4], 7.5).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut inst, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "InstanceNorm");
}

// =======================================================================
// WeightNorm
// =======================================================================

#[test]
fn weightnorm_input_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([31; 32]);
    let wn = WeightNorm::<f64>::new(5, 3, &mut rng).expect("WeightNorm construction");
    let input = varied(&[4, 5], 0.45);
    let grad_weights = varied(&[4, 3], 1.65);
    check_input_gradient(&wn, &input, &grad_weights, "WeightNorm");
}

#[test]
fn weightnorm_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([37; 32]);
    let mut wn = WeightNorm::<f64>::new(4, 3, &mut rng).expect("WeightNorm construction");
    let input = varied(&[3, 4], 0.85);
    let target = varied(&[3, 3], 4.5).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut wn, &input, &target, 0.1, 6);
    assert_strictly_decreasing(&losses, "WeightNorm");
}
