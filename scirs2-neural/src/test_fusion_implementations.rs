//! Tests for the multi-modal fusion architectures in
//! [`crate::models::architectures::fusion`].
//!
//! [`FeatureAlignment`]'s `Layer::backward`, and [`CrossModalAttention`],
//! [`FiLMModule`], and [`BilinearFusion`]'s dedicated `backward_with_*`
//! methods, already have numerical gradient-check coverage in
//! `tests/fusion_backprop.rs`. This module instead exercises
//! [`FeatureAlignment`] directly and, more importantly,
//! [`FeatureFusion::backward_multi`] — the top-level multi-modality
//! orchestrator whose `Layer::backward` used to silently fake an identity
//! pass-through (`Ok(grad_output.clone())`) instead of actually
//! backpropagating through the fusion pipeline, and whose `Layer::update`
//! therefore never actually trained any of its sub-modules.
//!
//! Declared as a `#[cfg(test)]`-only module from `lib.rs`.

use crate::layers::Layer;
use crate::models::architectures::fusion::*;
use scirs2_core::ndarray::{Array, IxDyn};

/// Relative tolerance for the analytic-vs-numeric gradient comparison.
const RTOL: f64 = 1e-4;
/// Step used by the central finite differences.
const EPS: f64 = 1e-5;

/// Deterministic, non-constant test data (see `tests/fusion_backprop.rs`):
/// all-ones/all-zero data cannot distinguish a real gradient from a
/// fabricated one.
fn varied(shape: &[usize], seed: f64) -> Array<f64, IxDyn> {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.61 + seed;
            0.8 * x.sin() + 0.35 * (0.37 * x).cos() - 0.1
        })
        .collect();
    Array::from_shape_vec(IxDyn(shape), values).expect("test data shape must be valid")
}

fn weighted_loss(output: &Array<f64, IxDyn>, weights: &Array<f64, IxDyn>) -> f64 {
    output
        .iter()
        .zip(weights.iter())
        .map(|(&o, &w)| o * w)
        .sum()
}

fn numeric_gradient<Fwd>(input: &Array<f64, IxDyn>, mut loss: Fwd) -> Array<f64, IxDyn>
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
        let a = analytic.as_slice().expect("contiguous grad")[idx];
        let n = numeric.as_slice().expect("contiguous grad")[idx];
        let scale = 1.0 + a.abs().max(n.abs());
        assert!(
            (a - n).abs() <= RTOL * scale,
            "{label}[{idx}]: analytic {a:.10e} vs numeric {n:.10e}"
        );
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

// ---------------------------------------------------------------------
// FeatureAlignment
// ---------------------------------------------------------------------

#[test]
fn feature_alignment_forward_backward_update() -> crate::error::Result<()> {
    let mut alignment: FeatureAlignment<f32> = FeatureAlignment::new(10, 8, Some("test"))?;

    let input = Array::ones((2, 10)).into_dyn();
    let output = alignment.forward(&input)?;
    assert_eq!(output.shape(), &[2, 8]);

    let grad_output = Array::ones((2, 8)).into_dyn();
    let grad_input = alignment.backward(&input, &grad_output)?;
    assert_eq!(grad_input.shape(), input.shape());

    let before = alignment.forward(&input)?;
    alignment.update(0.1)?;
    let after = alignment.forward(&input)?;
    let moved: f32 = before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        moved > 1e-6,
        "update() must actually change the layer's parameters, moved {moved}"
    );
    Ok(())
}

#[test]
fn feature_alignment_gradient_matches_finite_differences() {
    let alignment: FeatureAlignment<f64> = FeatureAlignment::new(5, 4, None).expect("construction");
    let input = varied(&[3, 5], 0.4);
    let grad_weights = varied(&[3, 4], 1.9);

    let numeric = numeric_gradient(&input, |probe| {
        weighted_loss(&alignment.forward(probe).expect("forward"), &grad_weights)
    });

    alignment.forward(&input).expect("forward");
    let analytic = alignment.backward(&input, &grad_weights).expect("backward");
    assert_gradients_match(&analytic, &numeric, "FeatureAlignment");
}

#[test]
fn feature_alignment_training_reduces_loss() {
    let mut alignment: FeatureAlignment<f64> =
        FeatureAlignment::new(4, 3, None).expect("construction");
    let input = varied(&[3, 4], 0.2);
    let target = varied(&[3, 3], 2.0).mapv(|v| 0.3 * v);

    let mut losses = Vec::new();
    for _ in 0..6 {
        let output = alignment.forward(&input).expect("forward");
        let diff = &output - &target;
        losses.push(diff.iter().map(|d| d * d).sum::<f64>() / target.len() as f64);
        let grad = diff.mapv(|d| d * 2.0 / target.len() as f64);
        alignment.backward(&input, &grad).expect("backward");
        alignment.update(0.5).expect("update");
    }
    assert_strictly_decreasing(&losses, "FeatureAlignment");
}

// ---------------------------------------------------------------------
// FeatureFusion::backward_multi
//
// This was the actual defect behind this file's original malformed tests:
// `Layer::backward` faked `Ok(grad_output.clone())` and there was no
// multi-modality backward at all, so `FeatureFusion::update` could never
// train the aligners, the fusion module, or the post-fusion network.
// ---------------------------------------------------------------------

fn concat_config() -> FeatureFusionConfig {
    FeatureFusionConfig {
        input_dims: vec![5, 4],
        hidden_dim: 6,
        fusion_method: FusionMethod::Concatenation,
        dropout_rate: 0.0,
        num_classes: 3,
        include_head: true,
    }
}

/// Checks both modalities' `backward_multi` gradients against finite
/// differences for a given fusion config, batch, and output-gradient shape.
fn check_fusion_gradients(
    fusion: &FeatureFusion<f64>,
    input_a: &Array<f64, IxDyn>,
    input_b: &Array<f64, IxDyn>,
    grad_weights: &Array<f64, IxDyn>,
    label: &str,
) {
    let numeric_a = numeric_gradient(input_a, |probe| {
        weighted_loss(
            &fusion
                .forward_multi(&[probe.clone(), input_b.clone()])
                .expect("forward_multi"),
            grad_weights,
        )
    });
    let numeric_b = numeric_gradient(input_b, |probe| {
        weighted_loss(
            &fusion
                .forward_multi(&[input_a.clone(), probe.clone()])
                .expect("forward_multi"),
            grad_weights,
        )
    });

    fusion
        .forward_multi(&[input_a.clone(), input_b.clone()])
        .expect("forward_multi");
    let grads = fusion.backward_multi(grad_weights).expect("backward_multi");
    assert_eq!(grads.len(), 2);
    assert_gradients_match(&grads[0], &numeric_a, &format!("{label} modality A"));
    assert_gradients_match(&grads[1], &numeric_b, &format!("{label} modality B"));
}

#[test]
fn feature_fusion_concatenation_gradient_matches_finite_differences() {
    let fusion: FeatureFusion<f64> = FeatureFusion::new(concat_config()).expect("construction");
    let input_a = varied(&[2, 5], 0.3);
    let input_b = varied(&[2, 4], 1.7);
    let grad_weights = varied(&[2, 3], 0.9);
    check_fusion_gradients(
        &fusion,
        &input_a,
        &input_b,
        &grad_weights,
        "FeatureFusion(concat)",
    );
}

#[test]
fn feature_fusion_sum_and_product_gradients_match_finite_differences() {
    for method in [FusionMethod::Sum, FusionMethod::Product] {
        let config = FeatureFusionConfig {
            input_dims: vec![4, 4],
            hidden_dim: 3,
            fusion_method: method,
            dropout_rate: 0.0,
            num_classes: 2,
            include_head: true,
        };
        let fusion: FeatureFusion<f64> = FeatureFusion::new(config).expect("construction");
        let input_a = varied(&[2, 4], 0.55);
        let input_b = varied(&[2, 4], 2.05);
        let grad_weights = varied(&[2, 2], 1.15);
        check_fusion_gradients(
            &fusion,
            &input_a,
            &input_b,
            &grad_weights,
            &format!("FeatureFusion({method:?})"),
        );
    }
}

#[test]
fn feature_fusion_attention_gradient_matches_finite_differences() {
    // `CrossModalAttention` (unlike the other fusion modules) is inherently
    // sequence-shaped, so its two modalities are 3D `[batch, seq, features]`
    // here rather than the 2D `[batch, features]` used by the other fusion
    // methods below.
    let config = FeatureFusionConfig {
        input_dims: vec![5, 4],
        hidden_dim: 6,
        fusion_method: FusionMethod::Attention,
        dropout_rate: 0.0,
        num_classes: 2,
        include_head: true,
    };
    let fusion: FeatureFusion<f64> = FeatureFusion::new(config).expect("construction");
    let input_a = varied(&[2, 3, 5], 0.15);
    let input_b = varied(&[2, 4, 4], 1.35);
    let grad_weights = varied(&[2, 2], 0.75);
    check_fusion_gradients(
        &fusion,
        &input_a,
        &input_b,
        &grad_weights,
        "FeatureFusion(attention)",
    );
}

#[test]
fn feature_fusion_film_and_bilinear_gradients_match_finite_differences() {
    for method in [FusionMethod::FiLM, FusionMethod::Bilinear] {
        let config = FeatureFusionConfig {
            input_dims: vec![5, 4],
            hidden_dim: 6,
            fusion_method: method,
            dropout_rate: 0.0,
            num_classes: 2,
            include_head: true,
        };
        let fusion: FeatureFusion<f64> = FeatureFusion::new(config).expect("construction");
        let input_a = varied(&[2, 5], 0.85);
        let input_b = varied(&[2, 4], 2.35);
        let grad_weights = varied(&[2, 2], 1.55);
        check_fusion_gradients(
            &fusion,
            &input_a,
            &input_b,
            &grad_weights,
            &format!("FeatureFusion({method:?})"),
        );
    }
}

#[test]
fn feature_fusion_training_reduces_loss() {
    let mut fusion: FeatureFusion<f64> = FeatureFusion::new(concat_config()).expect("construction");
    let input_a = varied(&[3, 5], 0.45);
    let input_b = varied(&[3, 4], 1.25);
    let target = varied(&[3, 3], 4.0).mapv(|v| 0.3 * v);

    let mut losses = Vec::new();
    for _ in 0..6 {
        let output = fusion
            .forward_multi(&[input_a.clone(), input_b.clone()])
            .expect("forward_multi");
        let diff = &output - &target;
        losses.push(diff.iter().map(|d| d * d).sum::<f64>() / target.len() as f64);
        let grad = diff.mapv(|d| d * 2.0 / target.len() as f64);
        fusion.backward_multi(&grad).expect("backward_multi");
        fusion.update(0.3).expect("update");
    }
    assert_strictly_decreasing(&losses, "FeatureFusion(concat, end-to-end)");
}

#[test]
fn feature_fusion_backward_multi_requires_forward_multi_first() {
    let fusion: FeatureFusion<f64> = FeatureFusion::new(concat_config()).expect("construction");
    let grad = varied(&[2, 3], 0.0);
    assert!(fusion.backward_multi(&grad).is_err());
}

#[test]
fn feature_fusion_layer_trait_backward_delegates_to_first_modality() {
    // The single-tensor `Layer::backward` cannot express every modality's
    // gradient, so it must delegate to `backward_multi` and return the first
    // modality's real gradient -- not silently fake an identity pass-through
    // of `grad_output` (which has a *different* shape from either input
    // modality whenever `hidden_dim != input_dims[i]`, as is the case here).
    let fusion: FeatureFusion<f64> = FeatureFusion::new(concat_config()).expect("construction");
    let input_a = varied(&[2, 5], 0.65);
    let input_b = varied(&[2, 4], 1.05);
    fusion
        .forward_multi(&[input_a.clone(), input_b.clone()])
        .expect("forward_multi");

    let grad_weights = varied(&[2, 3], 0.35);
    let via_multi = fusion
        .backward_multi(&grad_weights)
        .expect("backward_multi");

    let dummy_input = Array::<f64, IxDyn>::zeros(input_a.dim());
    let via_trait = fusion
        .backward(&dummy_input, &grad_weights)
        .expect("backward");

    assert_eq!(via_trait.shape(), input_a.shape());
    assert_ne!(
        via_trait.shape(),
        grad_weights.shape(),
        "backward() must not fake an identity pass-through of grad_output"
    );
    for (a, b) in via_trait.iter().zip(via_multi[0].iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}
