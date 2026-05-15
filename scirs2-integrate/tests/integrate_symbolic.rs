//! Integration tests for scirs2-integrate EML symbolic ODE + quadrature.
//!
//! All tests require the `symbolic` feature.

#![cfg(feature = "symbolic")]

use scirs2_core::ndarray::arr1;
use scirs2_integrate::{
    quad_gauss_legendre_symbolic, solve_ivp_symbolic, SymbolicOdeError, SymbolicQuadError,
};
use scirs2_symbolic::eml::LoweredOp;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn const_op(v: f64) -> Arc<LoweredOp> {
    Arc::new(LoweredOp::Const(v))
}

fn var(i: usize) -> Arc<LoweredOp> {
    Arc::new(LoweredOp::Var(i))
}

fn mul(a: Arc<LoweredOp>, b: Arc<LoweredOp>) -> Arc<LoweredOp> {
    Arc::new(LoweredOp::Mul(
        Box::new((*a).clone()),
        Box::new((*b).clone()),
    ))
}

fn sub(a: Arc<LoweredOp>, b: Arc<LoweredOp>) -> Arc<LoweredOp> {
    Arc::new(LoweredOp::Sub(
        Box::new((*a).clone()),
        Box::new((*b).clone()),
    ))
}

fn add(a: Arc<LoweredOp>, b: Arc<LoweredOp>) -> Arc<LoweredOp> {
    Arc::new(LoweredOp::Add(
        Box::new((*a).clone()),
        Box::new((*b).clone()),
    ))
}

fn neg(a: Arc<LoweredOp>) -> Arc<LoweredOp> {
    Arc::new(LoweredOp::Neg(Box::new((*a).clone())))
}

// ---------------------------------------------------------------------------
// ODE tests — BDF1
// ---------------------------------------------------------------------------

/// Test 1: stiff exponential decay x' = -1000·x
/// Exact solution: x(t) = exp(-1000·t)
/// At t = 0.001: x = exp(-1.0) ≈ 0.3679
///
/// BDF1 is first-order — we use a fine step (h=1e-5, 100 steps) to hit 5%
/// relative accuracy on this stiff problem.
#[test]
fn bdf1_stiff_decay_exact() {
    // Var(0) = t, Var(1) = x; RHS = -1000 * x = Const(-1000) * Var(1)
    let rhs = vec![mul(const_op(-1000.0), var(1))];
    let result = solve_ivp_symbolic(
        &rhs,
        [0.0, 0.001],
        arr1(&[1.0_f64]).view(),
        1e-5, // finer step for better BDF1 accuracy
        1e-7,
        1e-10,
        50_000,
    )
    .expect("BDF1 stiff decay should succeed");

    let y_final = result.y.last().expect("should have at least one output");
    let exact = f64::exp(-1.0);
    // BDF1 first-order: with h=1e-5 over t=[0, 0.001] (100 steps),
    // expected error is |1/(1+1000*1e-5)^100 - exp(-1)| ≈ 0.002 — check < 0.01
    assert!(
        (y_final[0] - exact).abs() < 0.01,
        "y_final = {}, exact = {}, error = {}",
        y_final[0],
        exact,
        (y_final[0] - exact).abs()
    );
}

/// Test 2: stiff decay — exact symbolic Jacobian means Newton converges quickly
/// Expect average Newton iters per step < 4 (for linear ODE, 1 Newton iter is exact).
#[test]
fn bdf1_stiff_fewer_newton_iters() {
    let rhs = vec![mul(const_op(-1000.0), var(1))];
    let result = solve_ivp_symbolic(
        &rhs,
        [0.0, 0.001],
        arr1(&[1.0_f64]).view(),
        1e-4,
        1e-4,
        1e-8,
        5000,
    )
    .expect("should converge");

    assert!(result.n_steps > 0, "should have at least one step");
    // With exact symbolic Jacobian, linear ODEs converge in 1 Newton iter;
    // the bound n_newton < n_steps * 4 is generous.
    assert!(
        result.n_newton < result.n_steps * 4,
        "n_newton = {}, n_steps = {}; expected < {} with exact Jacobian",
        result.n_newton,
        result.n_steps,
        result.n_steps * 4
    );
}

/// Test 3: logistic growth x' = x·(1−x)
/// Exact solution: x(t) = 1 / (1 + (1/x0 − 1)·exp(−t))
/// With x0 = 0.1: x(5) = 1 / (1 + 9·exp(−5)) ≈ 0.9933
#[test]
fn bdf1_logistic_growth() {
    // RHS = Var(1) * (1 - Var(1)) = Var(1) * (Const(1) - Var(1))
    let one_minus_x = sub(const_op(1.0), var(1));
    let rhs = vec![mul(var(1), one_minus_x)];

    let result = solve_ivp_symbolic(
        &rhs,
        [0.0, 5.0],
        arr1(&[0.1_f64]).view(),
        0.1,
        1e-5,
        1e-8,
        10_000,
    )
    .expect("logistic growth should converge");

    let y_final = result.y.last().expect("should have output");
    let exact = 1.0 / (1.0 + 9.0 * f64::exp(-5.0));
    assert!(
        (y_final[0] - exact).abs() < 0.01,
        "y_final = {}, exact = {}, error = {}",
        y_final[0],
        exact,
        (y_final[0] - exact).abs()
    );
}

/// Test 4: 2D linear stable system — x' = -x, y' = -2y
/// Exact: x(1) = exp(-1) ≈ 0.3679, y(1) = exp(-2) ≈ 0.1353
///
/// BDF1 is first-order; we use a small step (h=0.01, 100 steps) to achieve
/// ~1% relative accuracy over t=[0,1].
#[test]
fn bdf1_2d_linear_stable() {
    // Var(0) = t, Var(1) = x, Var(2) = y
    // rhs[0] = -Var(1), rhs[1] = -2*Var(2)
    let rhs = vec![neg(var(1)), mul(const_op(-2.0), var(2))];

    let result = solve_ivp_symbolic(
        &rhs,
        [0.0, 1.0],
        arr1(&[1.0_f64, 1.0]).view(),
        0.01, // finer step for better BDF1 accuracy
        1e-6,
        1e-8,
        10_000,
    )
    .expect("2D linear stable system should succeed");

    let y_final = result.y.last().expect("should have output");
    let exact_x = f64::exp(-1.0);
    let exact_y = f64::exp(-2.0);
    // BDF1 first-order: with h=0.01 over t=[0,1] (100 steps),
    // expected errors are ~0.002 for x and ~0.003 for y — check < 0.01
    assert!(
        (y_final[0] - exact_x).abs() < 0.01,
        "x_final = {}, exact = {}, error = {}",
        y_final[0],
        exact_x,
        (y_final[0] - exact_x).abs()
    );
    assert!(
        (y_final[1] - exact_y).abs() < 0.01,
        "y_final = {}, exact = {}, error = {}",
        y_final[1],
        exact_y,
        (y_final[1] - exact_y).abs()
    );
}

/// Test 5: dimension mismatch — rhs.len() != y0.len()
#[test]
fn bdf1_dim_mismatch_returns_err() {
    let rhs = vec![neg(var(1))]; // 1 component
    let err = solve_ivp_symbolic(
        &rhs,
        [0.0, 1.0],
        arr1(&[1.0_f64, 0.0]).view(), // 2 components
        0.1,
        1e-6,
        1e-8,
        100,
    );
    assert!(
        matches!(err, Err(SymbolicOdeError::DimMismatch { .. })),
        "expected DimMismatch, got {:?}",
        err.err().map(|e| e.to_string())
    );
}

/// Test 6: invalid h0 (negative step size)
#[test]
fn bdf1_invalid_h0_returns_err() {
    let rhs = vec![neg(var(1))];
    let err = solve_ivp_symbolic(
        &rhs,
        [0.0, 1.0],
        arr1(&[1.0_f64]).view(),
        -0.1, // invalid
        1e-6,
        1e-8,
        100,
    );
    assert!(
        matches!(err, Err(SymbolicOdeError::InvalidInput(_))),
        "expected InvalidInput, got {:?}",
        err.err().map(|e| e.to_string())
    );
}

// ---------------------------------------------------------------------------
// Quadrature tests — Gauss-Legendre
// ---------------------------------------------------------------------------

/// Test 7: ∫_{-1}^1 x^4 dx with n=3 (degree 5 rule → exact for degree ≤ 5)
/// Exact: 2/5 = 0.4
#[test]
fn quad_legendre_x4_exact() {
    // x^4 = Var(0)^4 = Mul(Mul(Var(0), Var(0)), Mul(Var(0), Var(0)))
    let x = var(0);
    let x2 = mul(x.clone(), x.clone());
    let x4 = mul(x2.clone(), x2.clone());

    let result = quad_gauss_legendre_symbolic(&x4, -1.0, 1.0, 3).expect("quad x^4 should succeed");

    assert!(
        (result - 0.4).abs() < 1e-12,
        "result = {result}, expected 0.4, error = {}",
        (result - 0.4).abs()
    );
}

/// Test 8: ∫_0^π sin(x) dx = 2  (using native Sin node)
/// With n=20 nodes, should be accurate to 1e-8
#[test]
fn quad_legendre_sin_integral() {
    // sin(x) as native LoweredOp::Sin
    let x = var(0);
    let sin_x = Arc::new(LoweredOp::Sin(Box::new((*x).clone())));

    let result = quad_gauss_legendre_symbolic(&sin_x, 0.0, std::f64::consts::PI, 20)
        .expect("quad sin should succeed");

    assert!(
        (result - 2.0).abs() < 1e-8,
        "result = {result}, expected 2.0, error = {}",
        (result - 2.0).abs()
    );
}

/// Test 9: ∫_0^1 3 dx = 3 (constant integrand, exact for any n ≥ 1)
#[test]
fn quad_legendre_constant_fn() {
    let integrand = const_op(3.0);
    let result = quad_gauss_legendre_symbolic(&integrand, 0.0, 1.0, 1)
        .expect("quad constant should succeed");
    assert!(
        (result - 3.0).abs() < 1e-12,
        "result = {result}, expected 3.0"
    );
}

/// Test 10: invalid interval a > b returns Err(InvalidInterval)
#[test]
fn quad_invalid_interval_returns_err() {
    let integrand = var(0);
    let err = quad_gauss_legendre_symbolic(&integrand, 1.0, 0.0, 5);
    assert!(
        matches!(err, Err(SymbolicQuadError::InvalidInterval(_, _))),
        "expected InvalidInterval, got {:?}",
        err.err().map(|e| e.to_string())
    );
}

// ---------------------------------------------------------------------------
// Extra stability / edge-case tests
// ---------------------------------------------------------------------------

/// Verify the ODE solver stores the initial condition as the first output
/// point.
#[test]
fn bdf1_initial_point_stored() {
    let rhs = vec![neg(var(1))]; // x' = -x
    let result = solve_ivp_symbolic(
        &rhs,
        [0.0, 1.0],
        arr1(&[2.0_f64]).view(),
        0.1,
        1e-6,
        1e-8,
        1000,
    )
    .expect("should succeed");
    assert_eq!(result.t[0], 0.0);
    assert!(
        (result.y[0][0] - 2.0).abs() < 1e-12,
        "initial y[0] should be 2.0"
    );
}

/// Quadrature of x² from 0 to 1 with n=2 (exact for degree ≤ 3 polynomial,
/// n=2 gives degree 3 rule, so this is exact).
#[test]
fn quad_legendre_x_squared_exact() {
    let x = var(0);
    let x2 = mul(x.clone(), x.clone());
    let result = quad_gauss_legendre_symbolic(&x2, 0.0, 1.0, 2).expect("quad x² should succeed");
    // ∫_0^1 x² dx = 1/3
    assert!(
        (result - 1.0 / 3.0).abs() < 1e-12,
        "result = {result}, expected 1/3 = {}",
        1.0 / 3.0
    );
}

/// ODE solver with a zero-derivative RHS: x' = 0 → x should stay constant.
#[test]
fn bdf1_zero_rhs_stays_constant() {
    let rhs = vec![const_op(0.0)]; // x' = 0
    let result = solve_ivp_symbolic(
        &rhs,
        [0.0, 1.0],
        arr1(&[5.0_f64]).view(),
        0.1,
        1e-6,
        1e-8,
        1000,
    )
    .expect("zero RHS should succeed");
    let y_final = result.y.last().expect("should have output");
    assert!(
        (y_final[0] - 5.0).abs() < 1e-10,
        "y should stay at 5.0, got {}",
        y_final[0]
    );
}

/// Sum of BDF1 and quadrature: verify they compose — integrate the closed-form
/// solution of x' = -x from 0 to 1 (which is x(t) = exp(-t)), and compare to
/// the exact ∫_0^1 exp(-t) dt = 1 - exp(-1) ≈ 0.6321.
#[test]
fn quad_of_ode_solution_cross_check() {
    // We compute both values independently and compare to the exact result.
    // This test validates that both subsystems give consistent answers to
    // a shared ground truth.

    // Ground truth
    let exact_integral = 1.0 - f64::exp(-1.0); // ≈ 0.6321

    // Quadrature of exp(-x) directly: Var(0) = x, integrand = Exp(-Var(0))
    let x = var(0);
    let neg_x = neg(x.clone());
    let exp_neg_x = Arc::new(LoweredOp::Exp(Box::new((*neg_x).clone())));
    let quad_result = quad_gauss_legendre_symbolic(&exp_neg_x, 0.0, 1.0, 20)
        .expect("quad exp(-x) should succeed");

    assert!(
        (quad_result - exact_integral).abs() < 1e-10,
        "quad result = {quad_result}, exact = {exact_integral}"
    );

    // ODE for x' = -x: final value at t=1 should be exp(-1)
    let rhs = vec![neg(var(1))];
    let ode_result = solve_ivp_symbolic(
        &rhs,
        [0.0, 1.0],
        arr1(&[1.0_f64]).view(),
        0.05,
        1e-7,
        1e-9,
        5000,
    )
    .expect("ODE should succeed");
    let y_final = ode_result.y.last().expect("should have output");
    assert!(
        (y_final[0] - f64::exp(-1.0)).abs() < 0.01,
        "ODE y_final = {}, exact = {}",
        y_final[0],
        f64::exp(-1.0)
    );
}

/// Additive integrand: ∫_0^2 (x + 1) dx = [x²/2 + x]_0^2 = 2 + 2 = 4
#[test]
fn quad_legendre_additive_integrand() {
    let x = var(0);
    let x_plus_1 = add(x.clone(), const_op(1.0));
    let result =
        quad_gauss_legendre_symbolic(&x_plus_1, 0.0, 2.0, 3).expect("quad (x+1) should succeed");
    assert!(
        (result - 4.0).abs() < 1e-12,
        "result = {result}, expected 4.0"
    );
}

// ---------------------------------------------------------------------------
// ODE Discovery facade tests (B.1)
// ---------------------------------------------------------------------------

mod ode_discovery_tests {
    use scirs2_core::ndarray::{Array1, Array2};
    use scirs2_integrate::eml::discover::{
        discover_ode_from_trajectory, OdeDiscoveryConfig, OdeDiscoveryError,
    };

    /// Helper: build a uniform time vector from 0 to (n-1)*dt.
    fn uniform_t(n: usize, dt: f64) -> Array1<f64> {
        Array1::from_vec((0..n).map(|i| i as f64 * dt).collect())
    }

    /// Test: empty t returns OdeDiscoveryError::EmptyTrajectory.
    #[test]
    fn test_empty_t_returns_err() {
        let t = Array1::<f64>::zeros(0);
        let y = Array2::<f64>::zeros((0, 1));
        let cfg = OdeDiscoveryConfig::default();
        let result = discover_ode_from_trajectory(t.view(), y.view(), &cfg);
        assert!(
            matches!(result, Err(OdeDiscoveryError::EmptyTrajectory)),
            "expected EmptyTrajectory, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    /// Test: t has 3 points but y has 4 rows → DimensionMismatch.
    #[test]
    fn test_dim_mismatch_returns_err() {
        let t = uniform_t(3, 0.1);
        let y = Array2::<f64>::zeros((4, 1)); // 4 rows, t has 3
        let cfg = OdeDiscoveryConfig::default();
        let result = discover_ode_from_trajectory(t.view(), y.view(), &cfg);
        assert!(
            matches!(
                result,
                Err(OdeDiscoveryError::DimensionMismatch {
                    t_len: 3,
                    y_rows: 4
                })
            ),
            "expected DimensionMismatch{{t_len:3,y_rows:4}}, got {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    /// Test: small 5-point trajectory doesn't panic.
    /// We only assert the result is Ok or Err(SymbolicError) — not the formula.
    #[test]
    fn test_discover_runs_without_panic() {
        let dt = 0.1;
        let t = uniform_t(5, dt);
        let traj_data: Vec<f64> = (0..5).map(|i| i as f64 * dt).collect();
        let y = Array2::from_shape_vec((5, 1), traj_data).expect("shape");
        let cfg = OdeDiscoveryConfig {
            n_generations: 3,
            population_size: 10,
            ..Default::default()
        };
        let result = discover_ode_from_trajectory(t.view(), y.view(), &cfg);
        assert!(
            result.is_ok() || matches!(result, Err(OdeDiscoveryError::SymbolicError(_))),
            "unexpected error variant: {:?}",
            result.err().map(|e| e.to_string())
        );
    }

    /// Test: 10-point single-state trajectory returns Ok (call doesn't panic).
    #[test]
    fn test_single_state_trajectory() {
        let dt = 0.1;
        let n = 10;
        let t = uniform_t(n, dt);
        let traj_data: Vec<f64> = (0..n).map(|i| (-(i as f64) * dt).exp()).collect();
        let y = Array2::from_shape_vec((n, 1), traj_data).expect("shape");
        let cfg = OdeDiscoveryConfig {
            n_generations: 5,
            population_size: 20,
            top_n: 2,
            ..Default::default()
        };
        let result = discover_ode_from_trajectory(t.view(), y.view(), &cfg);
        assert!(
            result.is_ok(),
            "expected Ok, got {:?}",
            result.err().map(|e| e.to_string())
        );
        if let Ok(formulas) = result {
            // single-state trajectory → at most 1 outer dimension entry
            assert!(
                formulas.len() <= 1,
                "expected at most 1 state dimension, got {}",
                formulas.len()
            );
        }
    }
}
