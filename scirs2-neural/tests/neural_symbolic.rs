#![cfg(feature = "symbolic")]

//! Integration tests for SymbolicActivation and SymbolicLoss.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_neural::activations::symbolic::SymbolicActivation;
use scirs2_neural::activations::Activation;
use scirs2_neural::losses::symbolic::SymbolicLoss;
use scirs2_neural::losses::Loss;
use scirs2_symbolic::eml::op::LoweredOp;
use std::sync::Arc;

const TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Helpers to build common LoweredOp expressions
// ---------------------------------------------------------------------------

fn var(i: usize) -> LoweredOp {
    LoweredOp::Var(i)
}

fn c(v: f64) -> LoweredOp {
    LoweredOp::Const(v)
}

/// Swish(x) = x / (1 + exp(-x))  expressed as Mul(Var(0), Div(1, 1+exp(-Var(0))))
fn swish_op() -> LoweredOp {
    // sigmoid(x) = 1 / (1 + exp(-x))
    let neg_x = LoweredOp::Neg(Box::new(var(0)));
    let exp_neg_x = LoweredOp::Exp(Box::new(neg_x));
    let one_plus = LoweredOp::Add(Box::new(c(1.0)), Box::new(exp_neg_x));
    let sigmoid = LoweredOp::Div(Box::new(c(1.0)), Box::new(one_plus));
    LoweredOp::Mul(Box::new(var(0)), Box::new(sigmoid))
}

/// Softplus(x) = ln(1 + exp(x))
fn softplus_op() -> LoweredOp {
    let exp_x = LoweredOp::Exp(Box::new(var(0)));
    let one_plus = LoweredOp::Add(Box::new(c(1.0)), Box::new(exp_x));
    LoweredOp::Ln(Box::new(one_plus))
}

/// x^2 = Mul(Var(0), Var(0))
fn x_squared_op() -> LoweredOp {
    LoweredOp::Mul(Box::new(var(0)), Box::new(var(0)))
}

/// 2*x = Mul(Const(2), Var(0))
fn two_x_op() -> LoweredOp {
    LoweredOp::Mul(Box::new(c(2.0)), Box::new(var(0)))
}

/// (p - t)^2
fn mse_op() -> LoweredOp {
    let diff = LoweredOp::Sub(Box::new(var(0)), Box::new(var(1)));
    LoweredOp::Mul(Box::new(diff.clone()), Box::new(diff))
}

fn from_elem(v: f64) -> Array<f64, IxDyn> {
    Array::from_vec(vec![v]).into_dyn()
}

fn from_slice(vs: &[f64]) -> Array<f64, IxDyn> {
    Array::from_vec(vs.to_vec()).into_dyn()
}

// ---------------------------------------------------------------------------
// Activation tests
// ---------------------------------------------------------------------------

/// Test 1: Swish forward matches expected values.
#[test]
fn symbolic_swish_forward_matches_expected() {
    let act = SymbolicActivation::new(Arc::new(swish_op())).expect("build activation");

    // Swish(0.0) = 0.0 * sigmoid(0.0) = 0.0
    let out0 = act.forward(&from_elem(0.0)).expect("forward x=0");
    let idx = scirs2_core::ndarray::IxDyn(&[0]);
    assert!(
        (out0[idx.clone()] - 0.0).abs() < TOL,
        "Swish(0) expected 0.0, got {}",
        out0[idx.clone()]
    );

    // Swish(1.0) = 1.0 * sigmoid(1.0) ≈ 0.7310585786
    let out1 = act.forward(&from_elem(1.0)).expect("forward x=1");
    let expected = 1.0_f64 / (1.0 + (-1.0_f64).exp()); // sigmoid(1)
    assert!(
        (out1[idx] - expected).abs() < 1e-9,
        "Swish(1) expected {expected}, got {}",
        out1[scirs2_core::ndarray::IxDyn(&[0])]
    );
}

/// Test 2: Swish backward chain rule at x = 1.0 with grad_output = 1.0.
///
/// d/dx[x*σ(x)] = σ(x) + x*σ(x)*(1 - σ(x))  ≈ 0.9276 at x = 1.
#[test]
fn symbolic_swish_backward_chain_rule() {
    let act = SymbolicActivation::new(Arc::new(swish_op())).expect("build activation");

    let sigma = 1.0_f64 / (1.0 + (-1.0_f64).exp()); // ≈ 0.7311
    let expected_grad = sigma + 1.0 * sigma * (1.0 - sigma); // ≈ 0.9276

    let input = from_elem(1.0);
    let grad_output = from_elem(1.0);
    let grad_in = act.backward(&grad_output, &input).expect("backward x=1");
    let idx = scirs2_core::ndarray::IxDyn(&[0]);
    assert!(
        (grad_in[idx] - expected_grad).abs() < 1e-6,
        "Swish backward: expected {expected_grad}, got {}",
        grad_in[scirs2_core::ndarray::IxDyn(&[0])]
    );
}

/// Test 3: Softplus(0) = ln(2) ≈ 0.6931.
#[test]
fn symbolic_relu_approx_forward() {
    let act = SymbolicActivation::new(Arc::new(softplus_op())).expect("build activation");
    let out = act.forward(&from_elem(0.0)).expect("forward x=0");
    let expected = (2.0_f64).ln(); // ln(1 + exp(0)) = ln(2)
    let idx = scirs2_core::ndarray::IxDyn(&[0]);
    assert!(
        (out[idx] - expected).abs() < TOL,
        "Softplus(0) expected {expected}, got {}",
        out[scirs2_core::ndarray::IxDyn(&[0])]
    );
}

/// Test 4: Shape is preserved through forward and backward on a (4, 8) array.
#[test]
fn symbolic_activation_shape_preserved() {
    let op = Arc::new(x_squared_op());
    let act = SymbolicActivation::new(op).expect("build activation");

    let input = Array::<f64, _>::zeros((4, 8)).into_dyn();
    let output = act.forward(&input).expect("forward");
    assert_eq!(output.shape(), input.shape(), "forward shape");

    let grad_output = Array::<f64, _>::zeros((4, 8)).into_dyn();
    let grad_in = act.backward(&grad_output, &input).expect("backward");
    assert_eq!(grad_in.shape(), input.shape(), "backward shape");
}

/// Test 5: f(x) = 2*x maps [1, 2, 3] to [2, 4, 6].
#[test]
fn symbolic_activation_forward_matches_batch() {
    let act = SymbolicActivation::new(Arc::new(two_x_op())).expect("build activation");
    let input = from_slice(&[1.0, 2.0, 3.0]);
    let output = act.forward(&input).expect("forward");
    let expected = [2.0, 4.0, 6.0];
    for (i, &exp) in expected.iter().enumerate() {
        let idx = scirs2_core::ndarray::IxDyn(&[i]);
        assert!(
            (output[idx] - exp).abs() < TOL,
            "2*x at i={i}: expected {exp}, got {}",
            output[scirs2_core::ndarray::IxDyn(&[i])]
        );
    }
}

/// Test 6: f(x) = x^2, df/dx = 2x; backward at [1, 2, 3] with grad=1 returns [2, 4, 6].
#[test]
fn symbolic_activation_backward_matches_batch() {
    let act = SymbolicActivation::new(Arc::new(x_squared_op())).expect("build activation");
    let input = from_slice(&[1.0, 2.0, 3.0]);
    let grad_output = from_slice(&[1.0, 1.0, 1.0]);
    let grad_in = act.backward(&grad_output, &input).expect("backward");
    let expected = [2.0, 4.0, 6.0];
    for (i, &exp) in expected.iter().enumerate() {
        let idx = scirs2_core::ndarray::IxDyn(&[i]);
        assert!(
            (grad_in[idx] - exp).abs() < 1e-9,
            "d(x^2)/dx at i={i}: expected {exp}, got {}",
            grad_in[scirs2_core::ndarray::IxDyn(&[i])]
        );
    }
}

// ---------------------------------------------------------------------------
// Loss tests
// ---------------------------------------------------------------------------

/// Test 7: MSE forward: mean((2-1)^2, (3-1)^2) = mean(1, 4) = 2.5.
#[test]
fn symbolic_loss_mse_forward() {
    let loss_fn = SymbolicLoss::new(Arc::new(mse_op())).expect("build loss");
    let preds = from_slice(&[2.0, 3.0]);
    let targets = from_slice(&[1.0, 1.0]);
    let loss = loss_fn.forward(&preds, &targets).expect("forward");
    assert!(
        (loss - 2.5).abs() < TOL,
        "MSE forward: expected 2.5, got {loss}"
    );
}

/// Test 8: MSE backward: dL/dp = 2*(p-t); scaled by 1/N = 0.5.
/// pairs: (2,1) → 2*1*0.5 = 1.0; (3,1) → 2*2*0.5 = 2.0
#[test]
fn symbolic_loss_mse_backward() {
    let loss_fn = SymbolicLoss::new(Arc::new(mse_op())).expect("build loss");
    let preds = from_slice(&[2.0, 3.0]);
    let targets = from_slice(&[1.0, 1.0]);
    let grad = loss_fn.backward(&preds, &targets).expect("backward");
    let expected = [1.0_f64, 2.0];
    for (i, &exp) in expected.iter().enumerate() {
        let idx = scirs2_core::ndarray::IxDyn(&[i]);
        assert!(
            (grad[idx] - exp).abs() < TOL,
            "MSE backward at i={i}: expected {exp}, got {}",
            grad[scirs2_core::ndarray::IxDyn(&[i])]
        );
    }
}

/// Test 9: L(p, t) = (p-t)^2 at p=t=0 → forward=0, backward=[0].
#[test]
fn symbolic_loss_zero_pred_target() {
    let loss_fn = SymbolicLoss::new(Arc::new(mse_op())).expect("build loss");
    let preds = from_elem(0.0);
    let targets = from_elem(0.0);
    let loss = loss_fn.forward(&preds, &targets).expect("forward");
    assert!(loss.abs() < TOL, "zero loss: expected 0.0, got {loss}");
    let grad = loss_fn.backward(&preds, &targets).expect("backward");
    let idx = scirs2_core::ndarray::IxDyn(&[0]);
    assert!(
        grad[idx].abs() < TOL,
        "zero grad: expected 0.0, got {}",
        grad[scirs2_core::ndarray::IxDyn(&[0])]
    );
}

/// Test 10: ln(x) at x = -1.0 must return Err, not Ok(NaN).
#[test]
fn symbolic_activation_eval_err_propagates() {
    let ln_op = Arc::new(LoweredOp::Ln(Box::new(var(0))));
    let act = SymbolicActivation::new(ln_op).expect("build activation");
    let input = from_elem(-1.0);
    let result = act.forward(&input);
    assert!(
        result.is_err(),
        "ln(-1.0) should return Err; got Ok({:?})",
        result
    );
}

// ---------------------------------------------------------------------------
// B.2 — init_weights_from_formula tests
// ---------------------------------------------------------------------------

mod weight_init_tests {
    use scirs2_core::ndarray::Array2;
    use scirs2_neural::symbolic::weight_init::{init_weights_from_formula, InitFromFormulaError};
    use scirs2_symbolic::eml::LoweredOp;
    use std::sync::Arc;

    fn const_op(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    fn var_op(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    /// Build formula: 2*Var(0) + 3*Var(1).
    fn formula_2x_plus_3y() -> Arc<LoweredOp> {
        let two_x = LoweredOp::Mul(Box::new(const_op(2.0)), Box::new(var_op(0)));
        let three_y = LoweredOp::Mul(Box::new(const_op(3.0)), Box::new(var_op(1)));
        Arc::new(LoweredOp::Add(Box::new(two_x), Box::new(three_y)))
    }

    /// Build formula: Var(0) + 1.0
    fn formula_x_plus_one() -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Add(Box::new(var_op(0)), Box::new(const_op(1.0))))
    }

    /// Build formula: Var(0)^2 = Mul(Var(0), Var(0))
    fn formula_x_squared() -> Arc<LoweredOp> {
        Arc::new(LoweredOp::Mul(Box::new(var_op(0)), Box::new(var_op(0))))
    }

    /// Test B.2-1: f = 2*Var(0) + 3*Var(1) → W ≈ [[2, 3]], b ≈ 0.
    #[test]
    fn test_linear_formula_recovered() {
        let formula = formula_2x_plus_3y();
        // Grid: 4 points in 2D
        let grid_data = vec![1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0];
        let grid = Array2::from_shape_vec((4, 2), grid_data).expect("shape");
        let (w, b) = init_weights_from_formula(&formula, grid.view()).expect("init weights");

        assert_eq!(w.shape(), &[1, 2], "W shape should be [1, 2]");
        assert_eq!(b.len(), 1, "b should have length 1");

        assert!(
            (w[(0, 0)] - 2.0).abs() < 1e-6,
            "W[0,0] should be ~2.0, got {}",
            w[(0, 0)]
        );
        assert!(
            (w[(0, 1)] - 3.0).abs() < 1e-6,
            "W[0,1] should be ~3.0, got {}",
            w[(0, 1)]
        );
        assert!(
            b[0].abs() < 1e-6,
            "b should be ~0.0 for a linear formula, got {}",
            b[0]
        );
    }

    /// Test B.2-2: f = Var(0) + 1.0 → W ≈ [[1]], b ≈ 1.0.
    #[test]
    fn test_affine_formula_recovered() {
        let formula = formula_x_plus_one();
        let grid_data = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let grid = Array2::from_shape_vec((5, 1), grid_data).expect("shape");
        let (w, b) = init_weights_from_formula(&formula, grid.view()).expect("init weights");

        assert_eq!(w.shape(), &[1, 1], "W shape should be [1, 1]");
        assert!(
            (w[(0, 0)] - 1.0).abs() < 1e-6,
            "W[0,0] should be ~1.0, got {}",
            w[(0, 0)]
        );
        assert!((b[0] - 1.0).abs() < 1e-6, "b should be ~1.0, got {}", b[0]);
    }

    /// Test B.2-3: f = Var(0)^2 is non-linear; linear fit residual should be
    /// finite (we don't assert accuracy, just that the call succeeds and
    /// residuals are finite — the linear model will approximate the mean).
    #[test]
    fn test_nonlinear_minimizes_residual() {
        let formula = formula_x_squared();
        // Grid: 6 points at {-1, 0, 1, 2, 3, 4}
        let grid_data = vec![-1.0_f64, 0.0, 1.0, 2.0, 3.0, 4.0];
        let grid = Array2::from_shape_vec((6, 1), grid_data).expect("shape");
        let result = init_weights_from_formula(&formula, grid.view());
        // Just verify it returns Ok with finite values
        assert!(
            result.is_ok(),
            "nonlinear formula should not crash, got {:?}",
            result.err().map(|e| e.to_string())
        );
        let (w, b) = result.expect("ok");
        assert!(w[(0, 0)].is_finite(), "W should be finite");
        assert!(b[0].is_finite(), "b should be finite");
    }

    /// Test B.2-4: empty grid returns Err(EmptyGrid).
    #[test]
    fn test_empty_grid_returns_err() {
        let formula = formula_x_plus_one();
        let grid = Array2::<f64>::zeros((0, 1));
        let result = init_weights_from_formula(&formula, grid.view());
        assert!(
            matches!(result, Err(InitFromFormulaError::EmptyGrid)),
            "expected EmptyGrid, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }
}

// ---------------------------------------------------------------------------
// B.3 — extract_formula_from_callable tests
// ---------------------------------------------------------------------------

mod formula_extract_tests {
    use scirs2_core::ndarray::{Array2, ArrayView2};
    use scirs2_neural::symbolic::formula_extract::{
        extract_formula_from_callable, FormulaExtractionConfig, FormulaExtractionError,
    };

    /// Test B.3-1: callable returning 2*col(0); small grid, just verify Ok + non-empty.
    #[test]
    fn test_extract_linear() {
        let grid_data: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let grid = Array2::from_shape_vec((10, 1), grid_data).expect("shape");
        let cfg = FormulaExtractionConfig {
            n_generations: 5,
            population_size: 20,
            n_results: 2,
            ..Default::default()
        };
        let result = extract_formula_from_callable(
            |x: ArrayView2<'_, f64>| {
                let n = x.shape()[0];
                Array2::from_shape_fn((n, 1), |(i, _)| x[(i, 0)] * 2.0)
            },
            grid.view(),
            &cfg,
        );
        assert!(
            result.is_ok(),
            "expected Ok, got {:?}",
            result.err().map(|e| e.to_string())
        );
        let formulas = result.expect("ok");
        // SR may find something; just check not panicked and result is a Vec
        assert!(
            formulas.len() <= cfg.n_results,
            "should return at most n_results"
        );
    }

    /// Test B.3-2: empty grid returns Err(EmptyGrid).
    #[test]
    fn test_extract_empty_grid_err() {
        let grid = Array2::<f64>::zeros((0, 1));
        let cfg = FormulaExtractionConfig::default();
        let result = extract_formula_from_callable(
            |x: ArrayView2<'_, f64>| Array2::zeros((x.shape()[0], 1)),
            grid.view(),
            &cfg,
        );
        assert!(
            matches!(result, Err(FormulaExtractionError::EmptyGrid)),
            "expected EmptyGrid, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    /// Test B.3-3: callable returning wrong shape → DimensionMismatch.
    #[test]
    fn test_extract_dim_mismatch() {
        let grid = Array2::<f64>::zeros((5, 1));
        let cfg = FormulaExtractionConfig::default();
        // Callable returns (3, 1) instead of (5, 1)
        let result = extract_formula_from_callable(
            |_x: ArrayView2<'_, f64>| Array2::zeros((3, 1)),
            grid.view(),
            &cfg,
        );
        assert!(
            matches!(result, Err(FormulaExtractionError::DimensionMismatch)),
            "expected DimensionMismatch, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    /// Test B.3-4: sin callable; just assert returns Ok (SR non-deterministic).
    #[test]
    fn test_extract_sin() {
        let grid_data: Vec<f64> = (0..10).map(|i| i as f64 * 0.3).collect();
        let grid = Array2::from_shape_vec((10, 1), grid_data).expect("shape");
        let cfg = FormulaExtractionConfig {
            n_generations: 5,
            population_size: 20,
            ..Default::default()
        };
        let result = extract_formula_from_callable(
            |x: ArrayView2<'_, f64>| {
                let n = x.shape()[0];
                Array2::from_shape_fn((n, 1), |(i, _)| x[(i, 0)].sin())
            },
            grid.view(),
            &cfg,
        );
        assert!(
            result.is_ok(),
            "expected Ok, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }
}
