//! Tests for mass matrix functionality in ODE solvers
//!
//! This module tests the mass matrix support in the ODE solvers.

use approx::assert_relative_eq;
use ndarray::{array, Array2, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{solve_ivp, MassMatrix, MassMatrixType, ODEMethod, ODEOptions};
use std::f64::consts::PI;

/// Test solving an ODE with a constant mass matrix
#[test]
fn test_constant_mass_matrix() -> IntegrateResult<()> {
    // Simple 2D oscillator with a mass matrix
    // M·[x', v']^T = [v, -x]^T
    // where M = [2 0; 0 1]
    //
    // This translates to:
    // 2·x' = v     -> x' = v/2
    // v' = -x
    //
    // Analytical solution: x(t) = cos(t/√2), v(t) = -√2·sin(t/√2)

    // Create mass matrix
    let mut mass_matrix = Array2::<f64>::eye(2);
    mass_matrix[[0, 0]] = 2.0;

    // Create MassMatrix specification
    let mass = MassMatrix::constant(mass_matrix);

    // ODE function
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    // Initial conditions: x(0) = 1, v(0) = 0
    let y0 = array![1.0, 0.0];

    // Integration parameters
    let t_span = [0.0, 1.0];

    // Solve with RK45
    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(mass),
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0, Some(options))?;

    // Verify solution against analytical solution
    let omega = 1.0 / f64::sqrt(2.0); // Natural frequency
    let t_final = result.t.last().unwrap();

    let x_numerical = result.y.last().unwrap()[0];
    let v_numerical = result.y.last().unwrap()[1];

    let x_analytical = (omega * t_final).cos();
    let v_analytical = -f64::sqrt(2.0) * (omega * t_final).sin();

    // Check that numerical solution matches analytical solution
    assert_relative_eq!(x_numerical, x_analytical, epsilon = 1e-5);
    assert_relative_eq!(v_numerical, v_analytical, epsilon = 1e-5);

    Ok(())
}

/// Test solving an ODE with a time-dependent mass matrix
#[test]
fn test_time_dependent_mass_matrix() -> IntegrateResult<()> {
    // Simple time-dependent system
    // M(t)·x'' + x = 0
    // where M(t) = 1 + 0.1·sin(t)
    //
    // As a first-order system:
    // [M(t) 0] [x'] = [ v ]
    // [  0  1] [v']   [-x]

    // Time-dependent mass matrix function
    let time_dependent_mass = |t: f64| {
        let mut m = Array2::<f64>::eye(2);
        m[[0, 0]] = 1.0 + 0.1 * t.sin();
        m
    };

    // Create mass matrix specification
    let mass = MassMatrix::time_dependent(time_dependent_mass);

    // ODE function
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    // Initial conditions: x(0) = 1, v(0) = 0
    let y0 = array![1.0, 0.0];

    // Integration parameters
    let t_span = [0.0, 1.0];

    // Solve with RK45
    let options = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(mass),
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0, Some(options))?;

    // For a time-dependent system, we don't have a simple analytical solution
    // But we can check energy conservation as an approximation

    // Calculate energy at each time point: E = (1/2)·M·v² + (1/2)·x²
    let mut energies = Vec::new();

    for i in 0..result.t.len() {
        let t = result.t[i];
        let x = result.y[i][0];
        let v = result.y[i][1];

        let m = 1.0 + 0.1 * t.sin();
        let energy = 0.5 * m * v * v + 0.5 * x * x;
        energies.push(energy);
    }

    // Energy shouldn't change too much for small time-dependence
    let initial_energy = energies[0];
    let final_energy = energies.last().unwrap();

    // Allow for some energy change due to time-dependence
    // and numerical errors. Since mass varies by 10%, energy can change significantly.
    let rel_energy_change = (final_energy - initial_energy).abs() / initial_energy;
    assert!(
        rel_energy_change < 0.15, // Allow up to 15% energy change for a 10% mass variation
        "Energy changed too much: {}",
        rel_energy_change
    );

    Ok(())
}

/// Test solving an ODE with an identity mass matrix
/// (should be equivalent to standard ODE)
#[test]
fn test_identity_mass_matrix() -> IntegrateResult<()> {
    // Simple harmonic oscillator
    // x'' + x = 0
    //
    // As a first-order system:
    // [1 0] [x'] = [ v ]
    // [0 1] [v']   [-x]

    // Identity mass matrix
    let mass = MassMatrix::identity();

    // ODE function
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    // Initial conditions: x(0) = 1, v(0) = 0
    let y0 = array![1.0, 0.0];

    // Integration parameters
    let t_span = [0.0, 2.0 * PI]; // One full period

    // Solve with mass matrix
    let options_with_mass = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(mass),
        ..Default::default()
    };

    let result_with_mass = solve_ivp(f, t_span, y0.clone(), Some(options_with_mass))?;

    // Solve without mass matrix (standard ODE)
    let options_standard = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };

    let result_standard = solve_ivp(f, t_span, y0, Some(options_standard))?;

    // Verify that solutions are the same at the final time point
    let x_with_mass = result_with_mass.y.last().unwrap()[0];
    let x_standard = result_standard.y.last().unwrap()[0];

    assert_relative_eq!(x_with_mass, x_standard, epsilon = 1e-8);

    // Solutions should also match the analytical solution: x(2π) = 1
    assert_relative_eq!(x_with_mass, 1.0, epsilon = 1e-5);

    Ok(())
}

/// Test creating different types of mass matrices
#[test]
fn test_mass_matrix_creation() {
    // Identity matrix
    let identity = MassMatrix::<f64>::identity();
    assert_eq!(identity.matrix_type, MassMatrixType::Identity);
    assert!(identity.constant_matrix.is_none());

    // Constant matrix
    let m = Array2::<f64>::eye(3);
    let constant = MassMatrix::constant(m.clone());
    assert_eq!(constant.matrix_type, MassMatrixType::Constant);
    assert!(constant.constant_matrix.is_some());
    assert_eq!(constant.constant_matrix.unwrap().dim(), (3, 3));

    // Time-dependent matrix
    let time_func = |_t: f64| Array2::<f64>::eye(2);
    let time_dependent = MassMatrix::time_dependent(time_func);
    assert_eq!(time_dependent.matrix_type, MassMatrixType::TimeDependent);
    assert!(time_dependent.time_function.is_some());
    assert!(time_dependent.constant_matrix.is_none());

    // State-dependent matrix
    let state_func = |_t: f64, _y: ArrayView1<f64>| Array2::<f64>::eye(2);
    let state_dependent = MassMatrix::state_dependent(state_func);
    assert_eq!(state_dependent.matrix_type, MassMatrixType::StateDependent);
    assert!(state_dependent.state_function.is_some());
    assert!(state_dependent.constant_matrix.is_none());
    assert!(state_dependent.time_function.is_none());

    // Bandwidth specification
    let banded = MassMatrix::constant(m).with_bandwidth(1, 2);
    assert!(banded.is_banded);
    assert_eq!(banded.lower_bandwidth, Some(1));
    assert_eq!(banded.upper_bandwidth, Some(2));
}

/// Test mass matrix evaluation
#[test]
fn test_mass_matrix_evaluation() {
    // Constant matrix
    let mut m = Array2::<f64>::eye(2);
    m[[0, 1]] = 1.0;

    let constant = MassMatrix::constant(m);
    let evaluated = constant.evaluate(0.0, array![0.0, 0.0].view()).unwrap();
    assert_eq!(evaluated[[0, 0]], 1.0);
    assert_eq!(evaluated[[0, 1]], 1.0);
    assert_eq!(evaluated[[1, 0]], 0.0);
    assert_eq!(evaluated[[1, 1]], 1.0);

    // Time-dependent matrix
    let time_func = |t: f64| {
        let mut m = Array2::<f64>::eye(2);
        m[[0, 0]] = t + 1.0;
        m
    };

    let time_dependent = MassMatrix::time_dependent(time_func);
    let evaluated = time_dependent
        .evaluate(2.0, array![0.0, 0.0].view())
        .unwrap();
    assert_eq!(evaluated[[0, 0]], 3.0); // t + 1 = 2 + 1 = 3
    assert_eq!(evaluated[[1, 1]], 1.0);

    // State-dependent matrix
    let state_func = |_t: f64, y: ArrayView1<f64>| {
        let mut m = Array2::<f64>::eye(2);
        m[[0, 0]] = y[0] + 1.0;
        m
    };

    let state_dependent = MassMatrix::state_dependent(state_func);
    let evaluated = state_dependent
        .evaluate(0.0, array![2.0, 0.0].view())
        .unwrap();
    assert_eq!(evaluated[[0, 0]], 3.0); // y[0] + 1 = 2 + 1 = 3
    assert_eq!(evaluated[[1, 1]], 1.0);
}
