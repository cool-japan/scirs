//! Integration tests for neural-enhanced interpolation (`ResidualMlpRbf`).

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_interpolate::neural_enhanced::{
    residual_mlp_rbf::{ResidualMlpRbf, ResidualMlpRbfConfig},
    tiny_mlp::{Activation, TinyMlp},
};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build 1-D noisy sin data: y_i = sin(x_i) + noise.
fn noisy_sin_data(n: usize, noise_amp: f64, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut state = if seed == 0 { 1u64 } else { seed };
    let mut pts = Array2::<f64>::zeros((n, 1));
    let mut vals = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x = i as f64 / (n - 1) as f64 * std::f64::consts::PI * 2.0;
        // Simple LCG noise
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = (state >> 33) as f64 / (u32::MAX as f64) * 2.0 * noise_amp - noise_amp;
        pts[[i, 0]] = x;
        vals[i] = x.sin() + noise;
    }
    (pts, vals)
}

/// RMSE over a set of predictions vs. targets.
fn rmse(pred: &[f64], target: &[f64]) -> f64 {
    let mse = pred
        .iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / pred.len() as f64;
    mse.sqrt()
}

// ---------------------------------------------------------------------------
// TinyMlp unit tests
// ---------------------------------------------------------------------------

#[test]
fn tiny_mlp_glorot_init_variance_within_expected_range() {
    let mlp = TinyMlp::new(&[4, 32, 16, 1], Activation::Tanh, 99).expect("construction");

    // First hidden layer: fan_in=4, fan_out=32, Glorot std ≈ sqrt(2/36) ≈ 0.236
    let w0 = &mlp.weights[0];
    let var: f32 = w0.iter().map(|&v| v * v).sum::<f32>() / w0.len() as f32;
    let expected_var = 2.0f32 / (4.0 + 32.0);
    let ratio = var / expected_var;
    assert!(
        ratio > 0.05 && ratio < 20.0,
        "Weight variance ratio out of range: {ratio:.3} (expected ~1.0)"
    );
}

#[test]
fn tiny_mlp_forward_backward_gradient_check() {
    // Verify backprop gradient against finite-difference for one weight entry.
    let mut mlp = TinyMlp::new(&[2, 4, 1], Activation::Tanh, 7).expect("construction");

    // Set non-trivial output layer weights.
    for j in 0..4 {
        mlp.weights[1][[0, j]] = (j as f32 + 1.0) * 0.1;
    }
    // Set non-trivial hidden layer weights.
    mlp.weights[0][[0, 0]] = 0.5;
    mlp.weights[0][[0, 1]] = -0.3;

    let x = Array1::from(vec![0.6f32, -0.4]);
    let target = 0.3f32;

    // Analytic gradient via public wrapper.
    let (_, pres) = mlp.forward_with_cache(&x).expect("fwd");
    let (gw, _) = mlp.backward_pub(&x, target, &pres).expect("bwd");
    let analytic = gw[0][[0, 0]];

    // Numerical gradient.
    let h = 1e-4f32;
    mlp.weights[0][[0, 0]] += h;
    let out_plus = mlp.forward(&x).expect("fwd+")[0];
    mlp.weights[0][[0, 0]] -= 2.0 * h;
    let out_minus = mlp.forward(&x).expect("fwd-")[0];
    mlp.weights[0][[0, 0]] += h;

    let loss_fn = |o: f32| 0.5 * (o - target).powi(2);
    let numerical = (loss_fn(out_plus) - loss_fn(out_minus)) / (2.0 * h);

    assert!(
        (analytic - numerical).abs() < 5e-3,
        "Gradient check failed: analytic={analytic:.6}, numerical={numerical:.6}"
    );
}

// ---------------------------------------------------------------------------
// ResidualMlpRbf tests
// ---------------------------------------------------------------------------

#[test]
fn residual_rbf_matches_pure_rbf_when_epochs_zero() {
    // With 0 training epochs the MLP output layer is zero-initialised → adds nothing.
    // We also set rbf_nugget = 0.0 so the base RBF is an exact interpolant;
    // this lets us compare directly against ScatteredRbf::new (no nugget).
    let (pts, vals) = noisy_sin_data(12, 0.0, 42);

    let config_rbf_only = ResidualMlpRbfConfig {
        epochs: 0,
        hidden_sizes: vec![8],
        seed: 13,
        rbf_nugget: 0.0, // exact interpolation so residuals are 0 and MLP adds nothing
        ..Default::default()
    };
    let mut model_zero = ResidualMlpRbf::new(config_rbf_only);
    model_zero.fit(&pts, &vals).expect("fit zero");

    // Pure-RBF model for comparison (also exact, no nugget).
    use scirs2_interpolate::rbf_interpolation::{RbfKernel, ScatteredRbf};
    let rbf = ScatteredRbf::<f64>::new(&pts, &vals, RbfKernel::Gaussian, None).expect("rbf");

    // Check test points.
    let test_xs = [0.5f64, 1.2, 2.8, 4.0, 5.5];
    for &xi in &test_xs {
        let q = Array1::from(vec![xi]);
        let pred = model_zero.predict(&q).expect("predict");
        let rbf_pred = rbf.evaluate(&[xi]).expect("rbf predict");
        assert!(
            (pred - rbf_pred).abs() < 1e-8,
            "xi={xi}: epochs=0 model differs from pure RBF: {pred:.8} vs {rbf_pred:.8}"
        );
    }
}

#[test]
fn residual_rbf_reduces_error_on_noisy_sin() {
    // Training data with noise.
    let (train_pts, train_vals) = noisy_sin_data(20, 0.05, 11);

    // Clean test data (ground truth).
    let test_n = 15;
    let mut test_pts = Array2::<f64>::zeros((test_n, 1));
    let mut true_vals = Vec::with_capacity(test_n);
    for i in 0..test_n {
        let x = (i as f64 + 0.5) / test_n as f64 * std::f64::consts::PI * 2.0;
        test_pts[[i, 0]] = x;
        true_vals.push(x.sin());
    }

    // Pure RBF baseline (0 epochs).
    let cfg_baseline = ResidualMlpRbfConfig {
        epochs: 0,
        seed: 77,
        hidden_sizes: vec![32, 16],
        ..Default::default()
    };
    let mut baseline = ResidualMlpRbf::new(cfg_baseline);
    baseline.fit(&train_pts, &train_vals).expect("fit baseline");

    let baseline_preds: Vec<f64> = (0..test_n)
        .map(|i| {
            let q = Array1::from(vec![test_pts[[i, 0]]]);
            baseline.predict(&q).expect("baseline predict")
        })
        .collect();

    // Residual model (trained).
    let cfg_trained = ResidualMlpRbfConfig {
        epochs: 300,
        seed: 77,
        lr: 3e-3,
        hidden_sizes: vec![32, 16],
        ..Default::default()
    };
    let mut trained = ResidualMlpRbf::new(cfg_trained);
    trained.fit(&train_pts, &train_vals).expect("fit trained");

    let trained_preds: Vec<f64> = (0..test_n)
        .map(|i| {
            let q = Array1::from(vec![test_pts[[i, 0]]]);
            trained.predict(&q).expect("trained predict")
        })
        .collect();

    let err_baseline = rmse(&baseline_preds, &true_vals);
    let err_trained = rmse(&trained_preds, &true_vals);

    // Both should produce finite predictions.
    for &v in &trained_preds {
        assert!(v.is_finite(), "Trained prediction is not finite");
    }
    // The trained model should have reasonably small error.
    assert!(
        err_trained < 0.5,
        "ResidualMlpRbf RMSE too large: {err_trained:.4} (baseline: {err_baseline:.4})"
    );
}

#[test]
fn residual_rbf_deterministic_with_seed() {
    let (pts, vals) = noisy_sin_data(10, 0.02, 55);

    let cfg = ResidualMlpRbfConfig {
        epochs: 50,
        seed: 123,
        ..Default::default()
    };

    let mut model_a = ResidualMlpRbf::new(cfg.clone());
    model_a.fit(&pts, &vals).expect("fit a");

    let mut model_b = ResidualMlpRbf::new(cfg);
    model_b.fit(&pts, &vals).expect("fit b");

    let q = Array1::from(vec![2.0f64]);
    let pred_a = model_a.predict(&q).expect("pred a");
    let pred_b = model_b.predict(&q).expect("pred b");

    assert!(
        (pred_a - pred_b).abs() < 1e-10,
        "Same seed should give same result: {pred_a:.8} vs {pred_b:.8}"
    );
}

#[test]
fn residual_rbf_predict_without_fit_returns_error() {
    let model = ResidualMlpRbf::new(ResidualMlpRbfConfig::default());
    let q = Array1::from(vec![1.0f64]);
    assert!(model.predict(&q).is_err(), "Should error if not fitted");
}
