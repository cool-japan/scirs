//! Numerical verification of backpropagation through the transformer stack.
//!
//! Analytic gradients from `Layer::backward` (and the decoder's dedicated
//! `backward_with_encoder`) are compared against central finite differences,
//! and a tiny transformer is trained for a few steps to prove that the
//! parameter gradients actually drive `update()`.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::SeedableRng;
use scirs2_neural::layers::{AttentionConfig, Layer, LayerNorm, MultiHeadAttention};
use scirs2_neural::transformer::{
    FeedForward, Transformer, TransformerConfig, TransformerDecoder, TransformerDecoderLayer,
    TransformerEncoder, TransformerEncoderLayer,
};
use scirs2_neural::utils::PositionalEncodingType;

/// Relative tolerance for the analytic-vs-numeric gradient comparison.
const RTOL: f64 = 1e-4;
/// Step used by the central finite differences.
const EPS: f64 = 1e-5;

/// Deterministic, non-constant test data: every element differs and both signs
/// occur, so a fabricated gradient cannot accidentally match.
fn varied(shape: &[usize], seed: f64) -> Array<f64, IxDyn> {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.63 + seed;
            0.8 * x.sin() + 0.35 * (0.27 * x).cos() - 0.1
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

fn weighted_loss(output: &Array<f64, IxDyn>, weights: &Array<f64, IxDyn>) -> f64 {
    output
        .iter()
        .zip(weights.iter())
        .map(|(&o, &w)| o * w)
        .sum()
}

/// Numeric gradient of `loss(input)` with respect to every input element.
fn numeric_input_gradient<Fwd>(input: &Array<f64, IxDyn>, mut loss: Fwd) -> Array<f64, IxDyn>
where
    Fwd: FnMut(&Array<f64, IxDyn>) -> f64,
{
    let mut numeric = Array::<f64, IxDyn>::zeros(input.dim());
    let mut probe = input.clone();
    for idx in 0..input.len() {
        let original = probe.as_slice_mut().expect("contiguous input")[idx];
        probe.as_slice_mut().expect("contiguous input")[idx] = original + EPS;
        let plus = loss(&probe);
        probe.as_slice_mut().expect("contiguous input")[idx] = original - EPS;
        let minus = loss(&probe);
        probe.as_slice_mut().expect("contiguous input")[idx] = original;
        numeric.as_slice_mut().expect("contiguous grad")[idx] = (plus - minus) / (2.0 * EPS);
    }
    numeric
}

fn assert_gradients_match(analytic: &Array<f64, IxDyn>, numeric: &Array<f64, IxDyn>, label: &str) {
    assert_eq!(analytic.shape(), numeric.shape(), "{label}: shape mismatch");
    for idx in 0..numeric.len() {
        assert_close(
            analytic.as_slice().expect("contiguous grad")[idx],
            numeric.as_slice().expect("contiguous grad")[idx],
            &format!("{label}[{idx}]"),
        );
    }
}

/// Check `Layer::backward`'s input gradient against finite differences.
fn check_input_gradient<L: Layer<f64>>(
    layer: &L,
    input: &Array<f64, IxDyn>,
    grad_weights: &Array<f64, IxDyn>,
    label: &str,
) {
    let numeric = numeric_input_gradient(input, |probe| {
        weighted_loss(
            &layer.forward(probe).expect("forward must succeed"),
            grad_weights,
        )
    });
    layer.forward(input).expect("forward must succeed");
    let analytic = layer
        .backward(input, grad_weights)
        .expect("backward must succeed");
    assert_gradients_match(&analytic, &numeric, label);
}

/// Check every parameter gradient reported through `Layer::gradients`.
fn check_parameter_gradients<L: Layer<f64>>(
    layer: &mut L,
    input: &Array<f64, IxDyn>,
    grad_weights: &Array<f64, IxDyn>,
    label: &str,
) {
    layer.forward(input).expect("forward must succeed");
    layer
        .backward(input, grad_weights)
        .expect("backward must succeed");
    let analytic = layer.gradients();
    let base = layer.params();
    assert!(!base.is_empty(), "{label}: layer must expose parameters");
    assert_eq!(
        analytic.len(),
        base.len(),
        "{label}: one gradient per parameter tensor is required"
    );

    for (p, param) in base.iter().enumerate() {
        assert_eq!(
            analytic[p].shape(),
            param.shape(),
            "{label}: gradient {p} shape mismatch"
        );
        for idx in 0..param.len() {
            let mut perturbed = base.clone();
            let original = perturbed[p].as_slice().expect("contiguous param")[idx];

            perturbed[p].as_slice_mut().expect("contiguous param")[idx] = original + EPS;
            layer
                .set_params(&perturbed)
                .expect("set_params must succeed");
            let plus = weighted_loss(
                &layer.forward(input).expect("forward must succeed"),
                grad_weights,
            );

            perturbed[p].as_slice_mut().expect("contiguous param")[idx] = original - EPS;
            layer
                .set_params(&perturbed)
                .expect("set_params must succeed");
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

    layer.set_params(&base).expect("set_params must succeed");
}

fn attention_config(num_heads: usize, head_dim: usize) -> AttentionConfig {
    AttentionConfig {
        num_heads,
        head_dim,
        dropout_prob: 0.0,
        causal: false,
        scale: None,
    }
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
fn layer_norm_input_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([3; 32]);
    let mut norm = LayerNorm::<f64>::new(4, 1e-5, &mut rng).expect("LayerNorm construction");
    // A non-trivial affine so gamma/beta really participate.
    let gamma = varied(&[4], 1.0).mapv(|v| 1.0 + 0.4 * v);
    let beta = varied(&[4], 2.0).mapv(|v| 0.2 * v);
    norm.set_params(&[gamma, beta]).expect("set_params");

    let input = varied(&[2, 3, 4], 0.5);
    let grad_weights = varied(&[2, 3, 4], 4.5);
    check_input_gradient(&norm, &input, &grad_weights, "LayerNorm input");
    check_parameter_gradients(&mut norm, &input, &grad_weights, "LayerNorm params");
}

#[test]
fn feed_forward_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([5; 32]);
    let mut ff = FeedForward::<f64>::new(4, 6, 0.0, &mut rng).expect("FeedForward construction");
    let input = varied(&[2, 3, 4], 1.25);
    let grad_weights = varied(&[2, 3, 4], 3.1);
    check_input_gradient(&ff, &input, &grad_weights, "FeedForward input");
    check_parameter_gradients(&mut ff, &input, &grad_weights, "FeedForward params");
}

#[test]
fn feed_forward_gradients_account_for_dropout_rescaling() {
    // The forward pass divides the hidden state by keep_prob; the backward pass
    // has to undo the same factor, otherwise the gradient is off by 1/(1-p).
    let mut rng = SmallRng::from_seed([7; 32]);
    let mut ff = FeedForward::<f64>::new(4, 5, 0.25, &mut rng).expect("FeedForward construction");
    let input = varied(&[1, 3, 4], 0.9);
    let grad_weights = varied(&[1, 3, 4], 2.4);
    check_input_gradient(&ff, &input, &grad_weights, "FeedForward(dropout) input");
    check_parameter_gradients(
        &mut ff,
        &input,
        &grad_weights,
        "FeedForward(dropout) params",
    );
}

#[test]
fn multi_head_self_attention_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([11; 32]);
    let mut mha = MultiHeadAttention::<f64>::new(4, attention_config(2, 2), &mut rng)
        .expect("MultiHeadAttention construction");
    let input = varied(&[2, 3, 4], 0.15);
    let grad_weights = varied(&[2, 3, 4], 5.6);
    check_input_gradient(&mha, &input, &grad_weights, "MHA input");
    check_parameter_gradients(&mut mha, &input, &grad_weights, "MHA params");
}

#[test]
fn causal_self_attention_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([13; 32]);
    let config = AttentionConfig {
        causal: true,
        ..attention_config(2, 2)
    };
    let mha = MultiHeadAttention::<f64>::new(4, config, &mut rng)
        .expect("MultiHeadAttention construction");
    let input = varied(&[1, 4, 4], 0.7);
    let grad_weights = varied(&[1, 4, 4], 1.9);
    check_input_gradient(&mha, &input, &grad_weights, "causal MHA input");
}

#[test]
fn cross_attention_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([17; 32]);
    let mha = MultiHeadAttention::<f64>::new(4, attention_config(2, 2), &mut rng)
        .expect("MultiHeadAttention construction");

    let query = varied(&[1, 3, 4], 0.4);
    let keyvalue = varied(&[1, 5, 4], 2.8);
    let grad_weights = varied(&[1, 3, 4], 6.2);

    // The key/value tensor has a different sequence length than the query, so
    // this could not work at all if cross-attention were secretly self-attention.
    let output = mha
        .forward_with_kv(&query, &keyvalue)
        .expect("cross-attention forward");
    assert_eq!(output.shape(), &[1, 3, 4]);

    let numeric_q = numeric_input_gradient(&query, |probe| {
        weighted_loss(
            &mha.forward_with_kv(probe, &keyvalue)
                .expect("forward must succeed"),
            &grad_weights,
        )
    });
    let numeric_kv = numeric_input_gradient(&keyvalue, |probe| {
        weighted_loss(
            &mha.forward_with_kv(&query, probe)
                .expect("forward must succeed"),
            &grad_weights,
        )
    });

    mha.forward_with_kv(&query, &keyvalue)
        .expect("forward must succeed");
    let (analytic_q, analytic_kv) = mha
        .backward_with_kv(&grad_weights)
        .expect("backward must succeed");

    assert_gradients_match(&analytic_q, &numeric_q, "cross-attention query");
    assert_gradients_match(&analytic_kv, &numeric_kv, "cross-attention key/value");
}

#[test]
fn encoder_layer_input_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([19; 32]);
    let layer = TransformerEncoderLayer::<f64>::new(4, 2, 6, 0.0, 1e-5, &mut rng)
        .expect("encoder layer construction");
    let input = varied(&[1, 3, 4], 0.6);
    let grad_weights = varied(&[1, 3, 4], 2.2);
    check_input_gradient(&layer, &input, &grad_weights, "TransformerEncoderLayer");
}

#[test]
fn encoder_stack_input_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([23; 32]);
    let encoder = TransformerEncoder::<f64>::new(4, 2, 2, 6, 0.0, 1e-5, &mut rng)
        .expect("encoder construction");
    let input = varied(&[1, 3, 4], 1.05);
    let grad_weights = varied(&[1, 3, 4], 0.85);
    check_input_gradient(&encoder, &input, &grad_weights, "TransformerEncoder");
}

#[test]
fn decoder_layer_simplified_input_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([29; 32]);
    let layer = TransformerDecoderLayer::<f64>::new(4, 2, 6, 0.0, 1e-5, &mut rng)
        .expect("decoder layer construction");
    let input = varied(&[1, 3, 4], 0.2);
    let grad_weights = varied(&[1, 3, 4], 3.7);
    check_input_gradient(&layer, &input, &grad_weights, "TransformerDecoderLayer");
}

#[test]
fn decoder_layer_cross_attention_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([31; 32]);
    let layer = TransformerDecoderLayer::<f64>::new(4, 2, 6, 0.0, 1e-5, &mut rng)
        .expect("decoder layer construction");

    let input = varied(&[1, 3, 4], 0.45);
    // Different source length: only real cross-attention can consume this.
    let encoder_output = varied(&[1, 5, 4], 1.85);
    let grad_weights = varied(&[1, 3, 4], 2.95);

    let numeric_input = numeric_input_gradient(&input, |probe| {
        weighted_loss(
            &layer
                .forward_with_encoder(probe, &encoder_output)
                .expect("forward must succeed"),
            &grad_weights,
        )
    });
    let numeric_encoder = numeric_input_gradient(&encoder_output, |probe| {
        weighted_loss(
            &layer
                .forward_with_encoder(&input, probe)
                .expect("forward must succeed"),
            &grad_weights,
        )
    });

    layer
        .forward_with_encoder(&input, &encoder_output)
        .expect("forward must succeed");
    let (analytic_input, analytic_encoder) = layer
        .backward_with_encoder(&input, &grad_weights)
        .expect("backward must succeed");

    assert_gradients_match(&analytic_input, &numeric_input, "decoder layer input");
    assert_gradients_match(
        &analytic_encoder,
        &numeric_encoder,
        "decoder layer encoder output",
    );
}

#[test]
fn decoder_stack_cross_attention_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([37; 32]);
    let decoder = TransformerDecoder::<f64>::new(4, 2, 2, 6, 0.0, 1e-5, &mut rng)
        .expect("decoder construction");

    let input = varied(&[1, 3, 4], 0.33);
    let encoder_output = varied(&[1, 4, 4], 2.05);
    let grad_weights = varied(&[1, 3, 4], 1.15);

    let numeric_input = numeric_input_gradient(&input, |probe| {
        weighted_loss(
            &decoder
                .forward_with_encoder(probe, &encoder_output)
                .expect("forward must succeed"),
            &grad_weights,
        )
    });
    let numeric_encoder = numeric_input_gradient(&encoder_output, |probe| {
        weighted_loss(
            &decoder
                .forward_with_encoder(&input, probe)
                .expect("forward must succeed"),
            &grad_weights,
        )
    });

    decoder
        .forward_with_encoder(&input, &encoder_output)
        .expect("forward must succeed");
    let (analytic_input, analytic_encoder) = decoder
        .backward_with_encoder(&input, &grad_weights)
        .expect("backward must succeed");

    assert_gradients_match(&analytic_input, &numeric_input, "decoder stack input");
    assert_gradients_match(
        &analytic_encoder,
        &numeric_encoder,
        "decoder stack encoder output",
    );
}

fn tiny_transformer_config() -> TransformerConfig {
    TransformerConfig {
        d_model: 4,
        n_encoder_layers: 1,
        n_decoder_layers: 1,
        n_heads: 2,
        d_ff: 6,
        max_seq_len: 16,
        dropout: 0.0,
        pos_encoding_type: PositionalEncodingType::Sinusoidal,
        epsilon: 1e-5,
    }
}

#[test]
fn transformer_encoder_path_gradient_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([41; 32]);
    let transformer =
        Transformer::<f64>::new(tiny_transformer_config(), &mut rng).expect("transformer");
    let input = varied(&[1, 3, 4], 0.55);
    let grad_weights = varied(&[1, 3, 4], 2.35);
    check_input_gradient(&transformer, &input, &grad_weights, "Transformer");
}

#[test]
fn transformer_train_path_gradients_match_finite_differences() {
    let mut rng = SmallRng::from_seed([43; 32]);
    let transformer =
        Transformer::<f64>::new(tiny_transformer_config(), &mut rng).expect("transformer");

    let src = varied(&[1, 4, 4], 0.18);
    let tgt = varied(&[1, 3, 4], 1.62);
    let grad_weights = varied(&[1, 3, 4], 3.44);

    let numeric_src = numeric_input_gradient(&src, |probe| {
        weighted_loss(
            &transformer
                .forward_train(probe, &tgt)
                .expect("forward must succeed"),
            &grad_weights,
        )
    });
    let numeric_tgt = numeric_input_gradient(&tgt, |probe| {
        weighted_loss(
            &transformer
                .forward_train(&src, probe)
                .expect("forward must succeed"),
            &grad_weights,
        )
    });

    transformer
        .forward_train(&src, &tgt)
        .expect("forward must succeed");
    let (analytic_src, analytic_tgt) = transformer
        .backward_train(&src, &tgt, &grad_weights)
        .expect("backward must succeed");

    assert_gradients_match(&analytic_src, &numeric_src, "transformer src");
    assert_gradients_match(&analytic_tgt, &numeric_tgt, "transformer tgt");
}

#[test]
fn transformer_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([47; 32]);
    let mut transformer =
        Transformer::<f64>::new(tiny_transformer_config(), &mut rng).expect("transformer");

    let input = varied(&[2, 3, 4], 0.24);
    // Layer normalization makes each output position zero-mean, so the target
    // is built the same way; the model still has to learn the pattern.
    let mut target = varied(&[2, 3, 4], 7.1).mapv(|v| 0.5 * v);
    for b in 0..2 {
        for t in 0..3 {
            let mean: f64 = (0..4).map(|k| target[[b, t, k]]).sum::<f64>() / 4.0;
            for k in 0..4 {
                target[[b, t, k]] -= mean;
            }
        }
    }

    let n = target.len() as f64;
    let mut losses = Vec::new();
    for _ in 0..5 {
        let output = transformer.forward(&input).expect("forward must succeed");
        let diff = &output - &target;
        losses.push(diff.iter().map(|d| d * d).sum::<f64>() / n);
        let grad_output = diff.mapv(|d| 2.0 * d / n);
        transformer
            .backward(&input, &grad_output)
            .expect("backward must succeed");
        transformer.update(0.05).expect("update must succeed");
    }
    assert_strictly_decreasing(&losses, "Transformer");
}

#[test]
fn transformer_encoder_layer_training_reduces_loss() {
    let mut rng = SmallRng::from_seed([53; 32]);
    let mut layer = TransformerEncoderLayer::<f64>::new(4, 2, 6, 0.0, 1e-5, &mut rng)
        .expect("encoder layer construction");

    let input = varied(&[2, 3, 4], 0.72);
    let mut target = varied(&[2, 3, 4], 9.3).mapv(|v| 0.5 * v);
    for b in 0..2 {
        for t in 0..3 {
            let mean: f64 = (0..4).map(|k| target[[b, t, k]]).sum::<f64>() / 4.0;
            for k in 0..4 {
                target[[b, t, k]] -= mean;
            }
        }
    }

    let n = target.len() as f64;
    let mut losses = Vec::new();
    for _ in 0..6 {
        let output = layer.forward(&input).expect("forward must succeed");
        let diff = &output - &target;
        losses.push(diff.iter().map(|d| d * d).sum::<f64>() / n);
        let grad_output = diff.mapv(|d| 2.0 * d / n);
        layer
            .backward(&input, &grad_output)
            .expect("backward must succeed");
        layer.update(0.1).expect("update must succeed");
    }
    assert_strictly_decreasing(&losses, "TransformerEncoderLayer");
}

#[test]
fn transformer_backward_requires_forward_first() {
    let mut rng = SmallRng::from_seed([59; 32]);
    let transformer =
        Transformer::<f64>::new(tiny_transformer_config(), &mut rng).expect("transformer");
    let src = varied(&[1, 3, 4], 0.0);
    let tgt = varied(&[1, 2, 4], 1.0);
    let grad = varied(&[1, 2, 4], 2.0);
    assert!(transformer.backward_train(&src, &tgt, &grad).is_err());
}
