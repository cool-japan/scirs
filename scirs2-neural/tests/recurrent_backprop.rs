//! Numerical verification of backpropagation through time in the recurrent layers.
//!
//! Every analytic gradient produced by `Layer::backward` is compared against a
//! central finite-difference estimate of the same derivative, and every layer is
//! trained for a few SGD steps on a synthetic task to prove that `update()`
//! actually consumes the computed gradients.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::SeedableRng;
use scirs2_neural::layers::recurrent::rnn::RecurrentActivation;
use scirs2_neural::layers::recurrent::{Bidirectional, GRU, LSTM, RNN};
use scirs2_neural::layers::rnn_thread_safe::{
    RecurrentActivation as ThreadSafeActivation, ThreadSafeBidirectional, ThreadSafeRNN,
};
use scirs2_neural::layers::{Layer, ParamLayer};

/// Relative tolerance for the analytic-vs-numeric gradient comparison.
const RTOL: f64 = 1e-4;
/// Step used by the central finite differences.
const EPS: f64 = 1e-5;

/// Deterministic, non-constant test data.
///
/// All-ones data cannot distinguish a real gradient from a fabricated one, so
/// every element here is distinct and both signs occur.
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

/// Check every parameter gradient against central finite differences.
fn check_parameter_gradients<L: ParamLayer<f64>>(
    layer: &mut L,
    input: &Array<f64, IxDyn>,
    grad_weights: &Array<f64, IxDyn>,
    label: &str,
) {
    layer.forward(input).expect("forward must succeed");
    layer
        .backward(input, grad_weights)
        .expect("backward must succeed");
    let analytic = ParamLayer::get_gradients(layer);
    let base_params = ParamLayer::get_parameters(layer);
    assert_eq!(
        analytic.len(),
        base_params.len(),
        "{label}: one gradient per parameter tensor is required"
    );

    for (p, param) in base_params.iter().enumerate() {
        assert_eq!(
            analytic[p].shape(),
            param.shape(),
            "{label}: gradient {p} shape mismatch"
        );
        for idx in 0..param.len() {
            let mut perturbed = base_params.clone();
            let original = perturbed[p].as_slice().expect("contiguous param")[idx];

            perturbed[p].as_slice_mut().expect("contiguous param")[idx] = original + EPS;
            layer
                .set_parameters(perturbed.clone())
                .expect("set_parameters must succeed");
            let plus = weighted_loss(
                &layer.forward(input).expect("forward must succeed"),
                grad_weights,
            );

            perturbed[p].as_slice_mut().expect("contiguous param")[idx] = original - EPS;
            layer
                .set_parameters(perturbed.clone())
                .expect("set_parameters must succeed");
            let minus = weighted_loss(
                &layer.forward(input).expect("forward must succeed"),
                grad_weights,
            );

            assert_close(
                analytic[p].as_slice().expect("contiguous grad")[idx],
                (plus - minus) / (2.0 * EPS),
                &format!("{label} d(loss)/d(param[{p}][{idx}])"),
            );
        }
    }

    layer
        .set_parameters(base_params)
        .expect("set_parameters must succeed");
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

#[test]
fn lstm_input_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([7; 32]);
    let lstm = LSTM::<f64>::new(3, 4, &mut rng).expect("LSTM construction");
    let input = varied(&[2, 3, 3], 0.4);
    let grad_weights = varied(&[2, 3, 4], 2.1);
    check_input_gradient(&lstm, &input, &grad_weights, "LSTM");
}

#[test]
fn lstm_parameter_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([11; 32]);
    let mut lstm = LSTM::<f64>::new(2, 3, &mut rng).expect("LSTM construction");
    let input = varied(&[2, 3, 2], 1.3);
    let grad_weights = varied(&[2, 3, 3], 0.6);
    check_parameter_gradients(&mut lstm, &input, &grad_weights, "LSTM");
}

#[test]
fn lstm_simd_path_gradient_matches_finite_differences() {
    // input_size + hidden_size >= 32 selects the SIMD forward kernel; the
    // analytic backward must agree with it too.
    let mut rng = SmallRng::from_seed([23; 32]);
    let lstm = LSTM::<f64>::new(16, 16, &mut rng).expect("LSTM construction");
    let input = varied(&[1, 2, 16], 0.9);
    let grad_weights = varied(&[1, 2, 16], 3.3);
    check_input_gradient(&lstm, &input, &grad_weights, "LSTM(simd)");
}

#[test]
fn lstm_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([13; 32]);
    let mut lstm = LSTM::<f64>::new(3, 2, &mut rng).expect("LSTM construction");
    let input = varied(&[2, 4, 3], 0.2);
    let target = varied(&[2, 4, 2], 5.0).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut lstm, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "LSTM");
}

#[test]
fn lstm_backward_requires_forward_first() {
    let mut rng = SmallRng::from_seed([17; 32]);
    let lstm = LSTM::<f64>::new(2, 2, &mut rng).expect("LSTM construction");
    let input = varied(&[1, 2, 2], 0.0);
    let grad = varied(&[1, 2, 2], 1.0);
    assert!(lstm.backward(&input, &grad).is_err());
}

#[test]
fn lstm_backward_rejects_mismatched_gradient_shape() {
    let mut rng = SmallRng::from_seed([19; 32]);
    let lstm = LSTM::<f64>::new(2, 3, &mut rng).expect("LSTM construction");
    let input = varied(&[1, 2, 2], 0.0);
    lstm.forward(&input).expect("forward must succeed");
    let wrong = varied(&[1, 2, 2], 1.0);
    assert!(lstm.backward(&input, &wrong).is_err());
}

#[test]
fn gru_input_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([29; 32]);
    let gru = GRU::<f64>::new(3, 4, &mut rng).expect("GRU construction");
    let input = varied(&[2, 3, 3], 0.8);
    let grad_weights = varied(&[2, 3, 4], 1.7);
    check_input_gradient(&gru, &input, &grad_weights, "GRU");
}

#[test]
fn gru_parameter_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([31; 32]);
    let mut gru = GRU::<f64>::new(2, 3, &mut rng).expect("GRU construction");
    let input = varied(&[2, 3, 2], 2.5);
    let grad_weights = varied(&[2, 3, 3], 0.15);
    check_parameter_gradients(&mut gru, &input, &grad_weights, "GRU");
}

#[test]
fn gru_simd_path_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([37; 32]);
    let gru = GRU::<f64>::new(16, 16, &mut rng).expect("GRU construction");
    let input = varied(&[1, 2, 16], 1.1);
    let grad_weights = varied(&[1, 2, 16], 4.2);
    check_input_gradient(&gru, &input, &grad_weights, "GRU(simd)");
}

#[test]
fn gru_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([41; 32]);
    let mut gru = GRU::<f64>::new(3, 2, &mut rng).expect("GRU construction");
    let input = varied(&[2, 4, 3], 0.6);
    let target = varied(&[2, 4, 2], 3.0).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut gru, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "GRU");
}

#[test]
fn rnn_input_gradient_matches_finite_differences() {
    for activation in [
        RecurrentActivation::Tanh,
        RecurrentActivation::Sigmoid,
        RecurrentActivation::ReLU,
    ] {
        let mut rng = SmallRng::from_seed([43; 32]);
        let rnn = RNN::<f64>::new(3, 4, activation, &mut rng).expect("RNN construction");
        let input = varied(&[2, 3, 3], 0.35);
        let grad_weights = varied(&[2, 3, 4], 1.25);
        check_input_gradient(&rnn, &input, &grad_weights, &format!("RNN({activation:?})"));
    }
}

#[test]
fn rnn_parameter_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([47; 32]);
    let mut rnn =
        RNN::<f64>::new(2, 3, RecurrentActivation::Tanh, &mut rng).expect("RNN construction");
    let input = varied(&[2, 3, 2], 1.9);
    let grad_weights = varied(&[2, 3, 3], 0.45);
    check_parameter_gradients(&mut rnn, &input, &grad_weights, "RNN");
}

#[test]
fn rnn_simd_path_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([53; 32]);
    let rnn =
        RNN::<f64>::new(16, 16, RecurrentActivation::Tanh, &mut rng).expect("RNN construction");
    let input = varied(&[1, 2, 16], 0.25);
    let grad_weights = varied(&[1, 2, 16], 2.75);
    check_input_gradient(&rnn, &input, &grad_weights, "RNN(simd)");
}

#[test]
fn rnn_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([59; 32]);
    let mut rnn =
        RNN::<f64>::new(3, 2, RecurrentActivation::Tanh, &mut rng).expect("RNN construction");
    let input = varied(&[2, 4, 3], 1.4);
    let target = varied(&[2, 4, 2], 6.0).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut rnn, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "RNN");
}

#[test]
fn thread_safe_rnn_input_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([61; 32]);
    let rnn = ThreadSafeRNN::<f64>::new(3, 4, ThreadSafeActivation::Tanh, &mut rng)
        .expect("ThreadSafeRNN construction");
    let input = varied(&[2, 3, 3], 0.55);
    let grad_weights = varied(&[2, 3, 4], 1.05);
    check_input_gradient(&rnn, &input, &grad_weights, "ThreadSafeRNN");
}

#[test]
fn thread_safe_rnn_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([67; 32]);
    let mut rnn = ThreadSafeRNN::<f64>::new(3, 2, ThreadSafeActivation::Tanh, &mut rng)
        .expect("ThreadSafeRNN construction");
    let input = varied(&[2, 4, 3], 0.85);
    let target = varied(&[2, 4, 2], 4.0).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut rnn, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "ThreadSafeRNN");
}

#[test]
fn thread_safe_rnn_clone_is_independent_and_faithful() {
    let mut rng = SmallRng::from_seed([71; 32]);
    let mut rnn = ThreadSafeRNN::<f64>::new(3, 2, ThreadSafeActivation::Tanh, &mut rng)
        .expect("ThreadSafeRNN construction");
    let input = varied(&[1, 3, 3], 0.12);

    let clone = rnn.clone();
    let original_out = rnn.forward(&input).expect("forward must succeed");
    let clone_out = clone.forward(&input).expect("forward must succeed");
    for (a, b) in original_out.iter().zip(clone_out.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "a clone must reproduce the original's weights exactly"
        );
    }

    // Training the original must not disturb the clone.
    let target = varied(&[1, 3, 2], 2.0).mapv(|v| 0.3 * v);
    train_losses(&mut rnn, &input, &target, 0.5, 3);
    let clone_out_after = clone.forward(&input).expect("forward must succeed");
    for (a, b) in clone_out.iter().zip(clone_out_after.iter()) {
        assert!((a - b).abs() < 1e-12, "the clone must stay independent");
    }
    let original_out_after = rnn.forward(&input).expect("forward must succeed");
    let moved: f64 = original_out
        .iter()
        .zip(original_out_after.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(moved > 1e-6, "training must actually change the original");
}

#[test]
fn thread_safe_bidirectional_clone_reproduces_both_directions() {
    let mut rng = SmallRng::from_seed([73; 32]);
    let inner = ThreadSafeRNN::<f64>::new(3, 2, ThreadSafeActivation::Tanh, &mut rng)
        .expect("ThreadSafeRNN construction");
    let bidi = ThreadSafeBidirectional::new_with_rng(Box::new(inner), Some("bidi"), &mut rng)
        .expect("bidirectional construction");
    let clone = bidi.clone();

    let input = varied(&[1, 4, 3], 0.33);
    let original = bidi.forward(&input).expect("forward must succeed");
    let cloned = clone.forward(&input).expect("forward must succeed");

    assert_eq!(original.shape(), &[1, 4, 4]);
    assert_eq!(original.shape(), cloned.shape());
    for (a, b) in original.iter().zip(cloned.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "the bidirectional clone must reproduce both directions exactly"
        );
    }
    // The two directions must have independent parameters, so the stacked
    // halves of the output must differ.
    let forward_half: f64 = (0..4).map(|t| original[[0, t, 0]].abs()).sum();
    let backward_half: f64 = (0..4).map(|t| original[[0, t, 2]].abs()).sum();
    assert!(
        (forward_half - backward_half).abs() > 1e-9,
        "the two directions must not be identical"
    );
}

#[test]
fn thread_safe_bidirectional_rejects_unsupported_inner_layer() {
    let mut rng = SmallRng::from_seed([79; 32]);
    let dense =
        scirs2_neural::layers::Dense::<f64>::new(3, 3, None, &mut rng).expect("dense construction");
    let result = ThreadSafeBidirectional::new(Box::new(dense), None);
    assert!(
        result.is_err(),
        "a non-RNN inner layer must be reported rather than silently made unidirectional"
    );
}

#[test]
fn thread_safe_bidirectional_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([83; 32]);
    let inner = ThreadSafeRNN::<f64>::new(2, 2, ThreadSafeActivation::Tanh, &mut rng)
        .expect("ThreadSafeRNN construction");
    let bidi = ThreadSafeBidirectional::new_with_rng(Box::new(inner), None, &mut rng)
        .expect("bidirectional construction");
    let input = varied(&[1, 3, 2], 0.77);
    let grad_weights = varied(&[1, 3, 4], 1.45);
    check_input_gradient(&bidi, &input, &grad_weights, "ThreadSafeBidirectional");
}

#[test]
fn bidirectional_lstm_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([89; 32]);
    let forward = LSTM::<f64>::new(2, 3, &mut rng).expect("forward LSTM");
    let backward = LSTM::<f64>::new(2, 3, &mut rng).expect("backward LSTM");
    let bidi = Bidirectional::new(Box::new(forward), Some(Box::new(backward)), Some("bilstm"))
        .expect("bidirectional construction");

    let input = varied(&[1, 4, 2], 0.42);
    let grad_weights = varied(&[1, 4, 6], 1.85);
    check_input_gradient(&bidi, &input, &grad_weights, "Bidirectional(LSTM)");
}

#[test]
fn bidirectional_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([97; 32]);
    let forward = RNN::<f64>::new(2, 2, RecurrentActivation::Tanh, &mut rng).expect("forward RNN");
    let backward =
        RNN::<f64>::new(2, 2, RecurrentActivation::Tanh, &mut rng).expect("backward RNN");
    let mut bidi = Bidirectional::new(Box::new(forward), Some(Box::new(backward)), None)
        .expect("bidirectional construction");

    let input = varied(&[2, 3, 2], 0.31);
    let target = varied(&[2, 3, 4], 8.0).mapv(|v| 0.3 * v);
    let losses = train_losses(&mut bidi, &input, &target, 0.5, 6);
    assert_strictly_decreasing(&losses, "Bidirectional(RNN)");
}

#[test]
fn bidirectional_rejects_a_shared_direction_layer() {
    let mut rng = SmallRng::from_seed([101; 32]);
    let forward = RNN::<f64>::new(2, 2, RecurrentActivation::Tanh, &mut rng).expect("forward RNN");
    // Sharing one layer for both directions cannot produce correct gradients,
    // so it must be reported instead of silently training on wrong values.
    let result = Bidirectional::new(Box::new(forward), None, None);
    assert!(result.is_err());
}
