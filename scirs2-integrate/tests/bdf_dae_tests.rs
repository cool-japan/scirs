//! Tests for BDF-based DAE solvers
//!
//! This module contains tests for the specialized BDF methods for DAE systems.

use approx::assert_relative_eq;
use ndarray::{array, ArrayView1};
use scirs2_integrate::{bdf_implicit_dae, bdf_semi_explicit_dae, DAEIndex, DAEOptions, DAEType};

/// Test the BDF solver for a simple linear semi-explicit DAE
///
/// System:
/// x' = -0.5 * x
/// 0 = x - 2 * y
///
/// Analytical solution: x(t) = e^(-0.5t), y(t) = 0.5 * e^(-0.5t)
#[test]
fn test_bdf_semi_explicit_linear() {
    // System definition
    let f = |_t: f64, x: ArrayView1<f64>, _y: ArrayView1<f64>| {
        let x_val = x[0];
        array![-0.5 * x_val]
    };

    let g = |_t: f64, x: ArrayView1<f64>, y: ArrayView1<f64>| {
        let x_val = x[0];
        let y_val = y[0];
        array![x_val - 2.0 * y_val]
    };

    // Initial conditions
    let x0 = array![1.0];
    let y0 = array![0.5]; // Satisfies the constraint x - 2y = 0

    // Time span
    let t_span = [0.0, 1.0]; // Reduce the integration interval to make it easier

    // Solver options
    let options = DAEOptions {
        dae_type: DAEType::SemiExplicit,
        index: DAEIndex::Index1,
        rtol: 1e-3,
        atol: 1e-4,
        max_steps: 10000,
        max_newton_iterations: 20,
        newton_tol: 1e-4,
        h0: Some(0.01),
        min_step: Some(1e-8),
        max_step: Some(0.1),
        max_order: Some(5),
        ..Default::default()
    };

    // Solve the system
    let result = bdf_semi_explicit_dae(f, g, t_span, x0, y0, options).unwrap();

    // Check the results
    assert!(result.success);
    assert!(result.n_steps > 0);

    // Check some solution points against the analytical solution
    let n_points = result.t.len();
    let final_t = result.t[n_points - 1];
    let final_x = result.x[n_points - 1][0];
    let final_y = result.y[n_points - 1][0];

    let expected_x = (-0.5 * final_t).exp();
    let expected_y = 0.5 * expected_x;

    assert_relative_eq!(final_x, expected_x, epsilon = 1e-3);
    assert_relative_eq!(final_y, expected_y, epsilon = 1e-3);

    // Check constraint satisfaction
    let constraint_error = final_x - 2.0 * final_y;
    assert_relative_eq!(constraint_error, 0.0, epsilon = 1e-6);
}

/// Test the BDF solver for a pendulum semi-explicit DAE
///
/// Simplified pendulum system:
/// x' = vx
/// y' = vy
/// vx' = -λx
/// vy' = -g - λy
/// 0 = x² + y² - L²
#[test]
fn test_bdf_semi_explicit_pendulum() {
    // Constants
    let g = 9.81;
    let length = 1.0;

    // System definition
    let f = |_t: f64, x: ArrayView1<f64>, lambda: ArrayView1<f64>| {
        let pos_x = x[0];
        let pos_y = x[1];
        let vel_x = x[2];
        let vel_y = x[3];
        let lambda_val = lambda[0];

        array![vel_x, vel_y, -lambda_val * pos_x, -g - lambda_val * pos_y]
    };

    let g_constraint = |_t: f64, x: ArrayView1<f64>, _lambda: ArrayView1<f64>| {
        let pos_x = x[0];
        let pos_y = x[1];

        array![pos_x * pos_x + pos_y * pos_y - length * length]
    };

    // Initial conditions (pendulum at rest, hanging down)
    let x0 = array![0.0, -length, 0.0, 0.0];
    let lambda0 = array![g / length]; // From the equations of motion at equilibrium

    // Time span
    let t_span = [0.0, 0.1]; // Short time to keep the test quick

    // Solver options
    let options = DAEOptions {
        dae_type: DAEType::SemiExplicit,
        index: DAEIndex::Index1,
        rtol: 1e-4,
        atol: 1e-6,
        max_steps: 10000,
        max_newton_iterations: 20,
        newton_tol: 1e-6,
        h0: Some(0.01),
        min_step: Some(1e-10),
        max_step: Some(0.05),
        max_order: Some(5),
        ..Default::default()
    };

    // Solve the system
    let result = bdf_semi_explicit_dae(f, g_constraint, t_span, x0, lambda0, options).unwrap();

    // Check the results
    assert!(result.success);
    assert!(result.n_steps > 0);

    // Check constraint satisfaction at all time points
    for i in 0..result.t.len() {
        let pos_x = result.x[i][0];
        let pos_y = result.x[i][1];
        let constraint_error = pos_x * pos_x + pos_y * pos_y - length * length;
        assert_relative_eq!(constraint_error, 0.0, epsilon = 1e-5);
    }

    // Check energy conservation (approximately, since we're solving with finite precision)
    let compute_energy = |i: usize| -> f64 {
        let pos_y = result.x[i][1];
        let vel_x = result.x[i][2];
        let vel_y = result.x[i][3];

        // Kinetic energy: 0.5 * (vx² + vy²)
        let kinetic = 0.5 * (vel_x * vel_x + vel_y * vel_y);

        // Potential energy: gy (assuming y=0 at the origin)
        let potential = g * (pos_y + length);

        kinetic + potential
    };

    let initial_energy = compute_energy(0);
    let final_energy = compute_energy(result.t.len() - 1);

    // Energy should be approximately conserved over short time intervals
    assert_relative_eq!(final_energy, initial_energy, epsilon = 1e-2);
}

/// Test the BDF solver for a simple implicit DAE
///
/// System:
/// y' + y + z = 0
/// y - z = t
///
/// Analytical solution: y(t) = c*e^(-t) + t, z(t) = c*e^(-t)
/// With initial condition y(0) = 1, we get c = 1
#[test]
fn test_bdf_implicit_dae_simple() {
    // System in implicit form: F(t, y, y') = 0
    let f = |t: f64, y: ArrayView1<f64>, y_prime: ArrayView1<f64>| {
        let y_val = y[0];
        let z_val = y[1];
        let y_prime_val = y_prime[0];

        array![
            y_prime_val + y_val + z_val, // y' + y + z = 0
            y_val - z_val - t            // y - z = t
        ]
    };

    // Initial conditions
    let y0 = array![1.0, 0.0]; // y(0) = 1, z(0) = 0 (which satisfies y - z = t at t=0)

    // Initial derivatives (obtained from the system equations)
    // At t=0: y(0) = 1, z(0) = 0
    // From y' + y + z = 0: y'(0) = -y(0) - z(0) = -1
    // z' can be obtained by differentiating y - z = t: y' - z' = 1, so z'(0) = y'(0) - 1 = -2
    let y_prime0 = array![-1.0, -2.0];

    // Time span
    let t_span = [0.0, 1.0];

    // Solver options
    let options = DAEOptions {
        dae_type: DAEType::FullyImplicit,
        index: DAEIndex::Index1,
        rtol: 1e-4,
        atol: 1e-6,
        max_steps: 10000,
        max_newton_iterations: 20,
        newton_tol: 1e-6,
        h0: Some(0.1),
        min_step: Some(1e-10),
        max_step: Some(0.2),
        max_order: Some(5),
        ..Default::default()
    };

    // Solve the system
    let result = bdf_implicit_dae(f, t_span, y0, y_prime0, options).unwrap();

    // Check the results
    assert!(result.success);
    assert!(result.n_steps > 0);

    // Check some solution points against the analytical solution
    let n_points = result.t.len();
    let final_t = result.t[n_points - 1];
    let final_y = result.x[n_points - 1][0];
    let final_z = result.x[n_points - 1][1];

    // Actual analytical solution for the system:
    // y' + y + z = 0 and y - z = t
    // From the second equation: z = y - t
    // Substituting into first: y' + y + (y - t) = 0 => y' + 2y = t
    // The homogeneous solution is y_h = C * e^(-2t)
    // The particular solution is y_p = t/2 - 1/4
    // The general solution is y = C * e^(-2t) + t/2 - 1/4
    // With initial condition y(0) = 1, we get: C = 1 + 1/4 = 5/4
    // So y(t) = (5/4) * e^(-2t) + t/2 - 1/4
    let expected_y = 1.25 * (-2.0 * final_t).exp() + 0.5 * final_t - 0.25;
    let expected_z = expected_y - final_t;

    assert_relative_eq!(final_y, expected_y, epsilon = 1e-2);
    assert_relative_eq!(final_z, expected_z, epsilon = 1e-2);

    // Check algebraic constraint satisfaction: y - z = t
    let constraint_error = final_y - final_z - final_t;
    assert_relative_eq!(constraint_error, 0.0, epsilon = 1e-6);
}

/// Test the BDF solver for an RLC circuit DAE
#[test]
fn test_bdf_implicit_dae_rlc_circuit() {
    // Circuit parameters
    let r = 1.0; // Resistance (Ohms)
    let l = 1.0; // Inductance (Henries)
    let c = 0.1; // Capacitance (Farads)

    // System in implicit form: F(t, y, y') = 0
    let f = |t: f64, y: ArrayView1<f64>, y_prime: ArrayView1<f64>| {
        let v = y[0];
        let i = y[1];
        let v_prime = y_prime[0];
        let i_prime = y_prime[1];

        // External voltage source: sine wave
        let v_source = t.sin();

        array![
            c * v_prime + i + v / r,    // C*v' + i + v/R = 0
            l * i_prime - v + v_source  // L*i' - v + v_source = 0
        ]
    };

    // Initial conditions (at rest, no initial voltage or current)
    let y0 = array![0.0, 0.0];

    // Initial derivatives (computed from the circuit equations at t=0)
    // At t=0: v(0) = 0, i(0) = 0, v_source(0) = sin(0) = 0
    // From the equations:
    // C*v' + i + v/R = 0 => v'(0) = -(i(0) + v(0)/R)/C = 0
    // L*i' - v + v_source = 0 => i'(0) = (v(0) - v_source(0))/L = 0
    let y_prime0 = array![0.0, 0.0];

    // Time span
    let t_span = [0.0, 0.5]; // Short time to keep the test quick

    // Solver options
    let options = DAEOptions {
        dae_type: DAEType::FullyImplicit,
        index: DAEIndex::Index1,
        rtol: 1e-4,
        atol: 1e-6,
        max_steps: 10000,
        max_newton_iterations: 20,
        newton_tol: 1e-6,
        h0: Some(0.01),
        min_step: Some(1e-10),
        max_step: Some(0.05),
        max_order: Some(5),
        ..Default::default()
    };

    // Solve the system
    let result = bdf_implicit_dae(f, t_span, y0, y_prime0, options).unwrap();

    // Check the results
    assert!(result.success);
    assert!(result.n_steps > 0);

    // For RLC circuits with sine input, we know some qualitative behaviors:
    // 1. For t close to 0, the voltage should follow the input roughly
    // 2. The current and voltage should satisfy the circuit equations

    // Check if the voltage follows the input source for small t
    // For the first few time steps, voltage should be close to 0 since sin(t) is small
    let early_index = (result.t.len() / 10).max(1); // About 10% into the simulation
    let early_t = result.t[early_index];
    let early_v = result.x[early_index][0];

    // For small t, voltage should start to build up but not exceed input
    assert!(early_v.abs() < early_t.sin().abs() + 0.1);

    // Verify circuit equations at a few points
    for i in (0..result.t.len()).step_by(result.t.len() / 10 + 1) {
        let t = result.t[i];
        let v = result.x[i][0];
        let i_current = result.x[i][1];

        // Get approximate derivatives
        let v_prime = if i < result.t.len() - 1 {
            (result.x[i + 1][0] - v) / (result.t[i + 1] - t)
        } else {
            (v - result.x[i - 1][0]) / (t - result.t[i - 1])
        };

        let i_prime = if i < result.t.len() - 1 {
            (result.x[i + 1][1] - i_current) / (result.t[i + 1] - t)
        } else {
            (i_current - result.x[i - 1][1]) / (t - result.t[i - 1])
        };

        // External voltage source: sine wave
        let v_source = t.sin();

        // Circuit equations residuals
        let res1 = c * v_prime + i_current + v / r;
        let res2 = l * i_prime - v + v_source;

        // Rough check on residuals (loose tolerance due to numerical differentiation)
        assert_relative_eq!(res1, 0.0, epsilon = 0.1, max_relative = 0.2);
        assert_relative_eq!(res2, 0.0, epsilon = 0.1, max_relative = 0.2);
    }
}
