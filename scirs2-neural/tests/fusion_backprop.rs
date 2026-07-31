//! Numerical verification of the multi-modal fusion modules' gradients.
//!
//! Each dedicated `backward_*` method is compared against central finite
//! differences of the matching dedicated `forward` method, using non-constant
//! data on both modalities.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_neural::models::architectures::fusion::{
    BilinearFusion, CrossModalAttention, FiLMModule,
};

/// Relative tolerance for the analytic-vs-numeric gradient comparison.
const RTOL: f64 = 1e-4;
/// Step used by the central finite differences.
const EPS: f64 = 1e-5;

fn varied(shape: &[usize], seed: f64) -> Array<f64, IxDyn> {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.53 + seed;
            0.7 * x.sin() + 0.3 * (0.41 * x).cos() - 0.05
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

#[test]
fn film_module_gradients_match_finite_differences() {
    let film = FiLMModule::<f64>::new(4, 3).expect("FiLM construction");
    let features = varied(&[2, 4], 0.3);
    let conditioning = varied(&[2, 3], 1.7);
    let grad_weights = varied(&[2, 4], 3.9);

    let numeric_features = numeric_gradient(&features, |probe| {
        weighted_loss(
            &film.forward(probe, &conditioning).expect("forward"),
            &grad_weights,
        )
    });
    let numeric_cond = numeric_gradient(&conditioning, |probe| {
        weighted_loss(
            &film.forward(&features, probe).expect("forward"),
            &grad_weights,
        )
    });

    film.forward(&features, &conditioning).expect("forward");
    let (analytic_features, analytic_cond) = film
        .backward_with_conditioning(&grad_weights)
        .expect("backward");

    assert_gradients_match(&analytic_features, &numeric_features, "FiLM features");
    assert_gradients_match(&analytic_cond, &numeric_cond, "FiLM conditioning");
}

#[test]
fn bilinear_fusion_gradients_match_finite_differences() {
    let fusion = BilinearFusion::<f64>::new(4, 3, 5, 3).expect("BilinearFusion construction");
    let features_a = varied(&[2, 4], 0.9);
    let features_b = varied(&[2, 3], 2.4);
    let grad_weights = varied(&[2, 5], 1.1);

    let numeric_a = numeric_gradient(&features_a, |probe| {
        weighted_loss(
            &fusion.forward(probe, &features_b).expect("forward"),
            &grad_weights,
        )
    });
    let numeric_b = numeric_gradient(&features_b, |probe| {
        weighted_loss(
            &fusion.forward(&features_a, probe).expect("forward"),
            &grad_weights,
        )
    });

    fusion.forward(&features_a, &features_b).expect("forward");
    let (analytic_a, analytic_b) = fusion
        .backward_with_features(&grad_weights)
        .expect("backward");

    assert_gradients_match(&analytic_a, &numeric_a, "BilinearFusion features_a");
    assert_gradients_match(&analytic_b, &numeric_b, "BilinearFusion features_b");
}

#[test]
fn cross_modal_attention_gradients_match_finite_differences() {
    let attention = CrossModalAttention::<f64>::new(4, 3, 4).expect("attention construction");
    // Two batch elements with different context lengths per modality exercise
    // the per-sample attention: a query must not see another sample's context.
    let query = varied(&[2, 3, 4], 0.25);
    let context = varied(&[2, 5, 3], 1.35);
    let grad_weights = varied(&[2, 3, 4], 2.65);

    let output = attention.forward(&query, &context).expect("forward");
    assert_eq!(output.shape(), &[2, 3, 4]);

    let numeric_query = numeric_gradient(&query, |probe| {
        weighted_loss(
            &attention.forward(probe, &context).expect("forward"),
            &grad_weights,
        )
    });
    let numeric_context = numeric_gradient(&context, |probe| {
        weighted_loss(
            &attention.forward(&query, probe).expect("forward"),
            &grad_weights,
        )
    });

    attention.forward(&query, &context).expect("forward");
    let (analytic_query, analytic_context) = attention
        .backward_with_context(&grad_weights)
        .expect("backward");

    assert_gradients_match(&analytic_query, &numeric_query, "cross-modal query");
    assert_gradients_match(&analytic_context, &numeric_context, "cross-modal context");
}

#[test]
fn cross_modal_attention_does_not_leak_across_batch_elements() {
    let attention = CrossModalAttention::<f64>::new(3, 3, 3).expect("attention construction");
    let query = varied(&[2, 2, 3], 0.65);
    let mut context = varied(&[2, 4, 3], 1.95);

    let baseline = attention.forward(&query, &context).expect("forward");

    // Perturbing the second sample's context must not change the first
    // sample's output.
    for j in 0..4 {
        for k in 0..3 {
            context[[1, j, k]] += 0.5;
        }
    }
    let perturbed = attention.forward(&query, &context).expect("forward");

    let mut first_sample_change = 0.0f64;
    let mut second_sample_change = 0.0f64;
    for i in 0..2 {
        for k in 0..3 {
            first_sample_change += (baseline[[0, i, k]] - perturbed[[0, i, k]]).abs();
            second_sample_change += (baseline[[1, i, k]] - perturbed[[1, i, k]]).abs();
        }
    }
    assert!(
        first_sample_change < 1e-12,
        "sample 0 must be unaffected by sample 1's context, changed by {first_sample_change}"
    );
    assert!(
        second_sample_change > 1e-6,
        "sample 1 must react to its own context, changed by {second_sample_change}"
    );
}

#[test]
fn fusion_backward_requires_forward_first() {
    let film = FiLMModule::<f64>::new(3, 2).expect("FiLM construction");
    assert!(film
        .backward_with_conditioning(&varied(&[2, 3], 0.0))
        .is_err());

    let fusion = BilinearFusion::<f64>::new(3, 2, 4, 2).expect("BilinearFusion construction");
    assert!(fusion
        .backward_with_features(&varied(&[2, 4], 0.0))
        .is_err());

    let attention = CrossModalAttention::<f64>::new(3, 3, 3).expect("attention construction");
    assert!(attention
        .backward_with_context(&varied(&[1, 2, 3], 0.0))
        .is_err());
}
