//! Verifies that the `layers::Sequential` container really propagates the
//! gradient through its layers in reverse instead of passing it straight
//! through.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::SeedableRng;
use scirs2_neural::layers::{Dense, Layer, Sequential};

/// Relative tolerance for the analytic-vs-numeric gradient comparison.
const RTOL: f64 = 1e-4;
/// Step used by the central finite differences.
const EPS: f64 = 1e-5;

fn varied(shape: &[usize], seed: f64) -> Array<f64, IxDyn> {
    let n: usize = shape.iter().product();
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * 0.47 + seed;
            0.75 * x.sin() + 0.3 * (0.19 * x).cos() - 0.08
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

#[test]
fn sequential_backward_matches_finite_differences() {
    let mut rng = SmallRng::from_seed([5; 32]);
    let mut model = Sequential::<f64>::new();
    model.add(Dense::<f64>::new(3, 4, Some("tanh"), &mut rng).expect("first dense layer"));
    model.add(Dense::<f64>::new(4, 2, None, &mut rng).expect("second dense layer"));

    let input = varied(&[5, 3], 0.9);
    let grad_weights = varied(&[5, 2], 2.6);

    // Numeric reference first: it perturbs the forward caches.
    let mut numeric = Array::<f64, IxDyn>::zeros(input.dim());
    let mut probe = input.clone();
    for idx in 0..input.len() {
        let original = probe.as_slice_mut().expect("contiguous input")[idx];
        probe.as_slice_mut().expect("contiguous input")[idx] = original + EPS;
        let plus = weighted_loss(&model.forward(&probe).expect("forward"), &grad_weights);
        probe.as_slice_mut().expect("contiguous input")[idx] = original - EPS;
        let minus = weighted_loss(&model.forward(&probe).expect("forward"), &grad_weights);
        probe.as_slice_mut().expect("contiguous input")[idx] = original;
        numeric.as_slice_mut().expect("contiguous grad")[idx] = (plus - minus) / (2.0 * EPS);
    }

    model.forward(&input).expect("forward");
    let analytic = model.backward(&input, &grad_weights).expect("backward");

    assert_eq!(
        analytic.shape(),
        input.shape(),
        "the gradient must have the input's shape, not the output's"
    );
    for idx in 0..input.len() {
        let a = analytic.as_slice().expect("contiguous grad")[idx];
        let n = numeric.as_slice().expect("contiguous grad")[idx];
        let scale = 1.0 + a.abs().max(n.abs());
        assert!(
            (a - n).abs() <= RTOL * scale,
            "Sequential grad[{idx}]: analytic {a:.10e} vs numeric {n:.10e}"
        );
    }
}

#[test]
fn empty_sequential_is_the_identity() {
    let model = Sequential::<f64>::new();
    let input = varied(&[2, 3], 0.1);
    let grad = varied(&[2, 3], 1.4);
    let out = model.backward(&input, &grad).expect("backward");
    for (a, b) in out.iter().zip(grad.iter()) {
        assert!((a - b).abs() < 1e-15);
    }
}

#[test]
fn sequential_backward_requires_forward_first() {
    let mut rng = SmallRng::from_seed([9; 32]);
    let mut model = Sequential::<f64>::new();
    model.add(Dense::<f64>::new(3, 2, None, &mut rng).expect("dense layer"));
    let input = varied(&[2, 3], 0.0);
    let grad = varied(&[2, 2], 1.0);
    assert!(model.backward(&input, &grad).is_err());
}
