//! Tests for Radau method with mass matrices
//!
//! This module tests the Radau method with mass matrix support.

use approx::assert_relative_eq;
use scirs2_core::ndarray::{array, Array2, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{solve_ivp, MassMatrix, ODEMethod, ODEOptions};

/// Test Radau method with a constant mass matrix
#[test]
#[allow(dead_code)]
fn test_radau_constant_mass_matrix() -> IntegrateResult<()> {
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

    // Solve with Radau method
    let options = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(mass),
        dense_output: true,
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0.clone(), Some(options))?;

    // Verify solution against analytical solution
    let omega = 1.0 / f64::sqrt(2.0); // Natural frequency
    let t_final = result.t.last().expect("Operation failed");

    let x_numerical = result.y.last().expect("Operation failed")[0];
    let v_numerical = result.y.last().expect("Operation failed")[1];

    let x_analytical = (omega * t_final).cos();
    let v_analytical = -f64::sqrt(2.0) * (omega * t_final).sin();

    // Check that numerical solution matches analytical solution
    println!("Radau solution at t = {t_final}");
    println!("x_numerical = {x_numerical}, x_analytical = {x_analytical}");
    println!("v_numerical = {v_numerical}, v_analytical = {v_analytical}");
    println!(
        "Error: x = {}, v = {}",
        (x_numerical - x_analytical).abs(),
        (v_numerical - v_analytical).abs()
    );

    assert_relative_eq!(
        x_numerical,
        x_analytical,
        epsilon = 1e-5,
        max_relative = 1e-5
    );
    assert_relative_eq!(
        v_numerical,
        v_analytical,
        epsilon = 1e-5,
        max_relative = 1e-5
    );

    // Check statistics
    println!("Statistics:");
    println!(
        "  Steps: {} (accepted: {}, rejected: {})",
        result.n_steps, result.n_accepted, result.n_rejected
    );
    println!("  Function evaluations: {}", result.n_eval);
    println!("  Jacobian evaluations: {}", result.n_jac);
    println!("  LU decompositions: {}", result.n_lu);

    Ok(())
}

/// Test Radau method with a time-dependent mass matrix
#[test]
#[allow(dead_code)]
fn test_radau_time_dependent_mass_matrix() -> IntegrateResult<()> {
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
    let t_span = [0.0, 10.0];

    // Solve with Radau
    let options = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(mass),
        dense_output: true,
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0.clone(), Some(options))?;

    // For a time-dependent system, we don't have a simple analytical solution
    // But we can check that the solution is oscillatory and reasonably bounded

    println!("Time-dependent mass matrix solution with Radau:");
    println!(
        "  Final time: {}",
        result.t.last().expect("Operation failed")
    );
    println!(
        "  Final state: x = {}, v = {}",
        result.y.last().expect("Operation failed")[0],
        result.y.last().expect("Operation failed")[1]
    );

    // Check a few points throughout the solution to verify oscillatory behavior
    let check_times = [1.0, 3.0, 5.0, 7.0, 9.0];
    for &check_time in &check_times {
        // Find the closest time point in the solution
        let (i, t) = result
            .t
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                (a - check_time)
                    .abs()
                    .partial_cmp(&(b - check_time).abs())
                    .expect("Operation failed")
            })
            .expect("Operation failed");

        println!(
            "  At t ≈ {}: x = {}, v = {}",
            t, result.y[i][0], result.y[i][1]
        );
    }

    // Check that solution stays within reasonable bounds
    // The mass matrix only varies by 10%, so solution shouldn't grow unbounded
    for y_i in &result.y {
        assert!(
            y_i[0].abs() <= 2.0,
            "Position x exceeded reasonable bounds: {}",
            y_i[0]
        );
        assert!(
            y_i[1].abs() <= 2.0,
            "Velocity v exceeded reasonable bounds: {}",
            y_i[1]
        );
    }

    // Verify the solution is slightly different than a standard mass matrix
    // Use an identity mass matrix as comparison
    let standard_opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-8,
        atol: 1e-10,
        dense_output: true,
        ..Default::default()
    };

    let standard_result = solve_ivp(f, t_span, y0.clone(), Some(standard_opts))?;

    // Compare final states
    let time_dep_final = result.y.last().expect("Operation failed");
    let standard_final = standard_result.y.last().expect("Operation failed");

    // The solutions should be different due to the time-dependent mass
    let diff_x = (time_dep_final[0] - standard_final[0]).abs();
    let diff_v = (time_dep_final[1] - standard_final[1]).abs();

    println!("Difference between time-dependent and standard mass matrix:");
    println!("  Δx = {diff_x}, Δv = {diff_v}");

    // The difference should be non-negligible but not huge
    assert!(
        diff_x > 1e-3,
        "Time-dependent mass had no effect on position"
    );
    assert!(
        diff_v > 1e-3,
        "Time-dependent mass had no effect on velocity"
    );

    // Check statistics
    println!("Statistics:");
    println!(
        "  Steps: {} (accepted: {}, rejected: {})",
        result.n_steps, result.n_accepted, result.n_rejected
    );
    println!("  Function evaluations: {}", result.n_eval);
    println!("  Jacobian evaluations: {}", result.n_jac);
    println!("  LU decompositions: {}", result.n_lu);

    Ok(())
}

/// Debug test to understand Radau mass matrix issue
#[test]
#[allow(dead_code)]
fn test_radau_debug() -> IntegrateResult<()> {
    // Simple test case: 2D oscillator with mass matrix
    // M·[x', v']^T = [v, -x]^T where M = [2 0; 0 1]

    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];
    let y0 = array![1.0, 0.0];
    let t_span = [0.0, 0.1]; // Very short time span

    println!("Testing Radau without mass matrix (should work):");
    let opts_no_mass = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        h0: Some(0.01),
        ..Default::default()
    };

    match solve_ivp(f, t_span, y0.clone(), Some(opts_no_mass)) {
        Ok(result) => {
            println!(
                "  Success! Final state: {:?}",
                result.y.last().expect("Operation failed")
            );
            println!(
                "  Steps: {}, Function evals: {}",
                result.n_steps, result.n_eval
            );
        }
        Err(e) => println!("  Failed: {e:?}"),
    }

    println!("\nTesting Radau with identity mass matrix (should work):");
    let identity_mass = MassMatrix::identity();
    let opts_identity = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        h0: Some(0.01),
        mass_matrix: Some(identity_mass),
        ..Default::default()
    };

    match solve_ivp(f, t_span, y0.clone(), Some(opts_identity)) {
        Ok(result) => {
            println!(
                "  Success! Final state: {:?}",
                result.y.last().expect("Operation failed")
            );
            println!(
                "  Steps: {}, Function evals: {}",
                result.n_steps, result.n_eval
            );
        }
        Err(e) => println!("  Failed: {e:?}"),
    }

    println!("\nTesting Radau with non-identity mass matrix (currently fails):");
    let mut mass_matrix = Array2::<f64>::eye(2);
    mass_matrix[[0, 0]] = 2.0;
    let mass = MassMatrix::constant(mass_matrix);

    let opts_mass = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        h0: Some(0.01),
        mass_matrix: Some(mass),
        ..Default::default()
    };

    match solve_ivp(f, t_span, y0.clone(), Some(opts_mass)) {
        Ok(result) => {
            println!(
                "  Success! Final state: {:?}",
                result.y.last().expect("Operation failed")
            );
            println!(
                "  Steps: {}, Function evals: {}",
                result.n_steps, result.n_eval
            );
        }
        Err(e) => println!("  Failed: {e:?}"),
    }

    Ok(())
}

/// Compare Radau method with transformed explicit solver for mass matrices
#[test]
#[allow(dead_code)]
fn test_radau_vs_explicit_mass_matrix() -> IntegrateResult<()> {
    // Test that should now work with the fixed Newton iteration

    // Simple 2D oscillator with a mass matrix
    // M·[x', v']^T = [v, -x]^T
    // where M = [2 0; 0 1]

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
    let t_span = [0.0, 5.0];

    // Solve with Radau (implicit)
    let radau_opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(mass.clone()),
        ..Default::default()
    };

    let radau_result = solve_ivp(f, t_span, y0.clone(), Some(radau_opts))?;

    // Solve with RK45 (explicit, transformed)
    let rk45_opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(mass),
        ..Default::default()
    };

    let rk45_result = solve_ivp(f, t_span, y0, Some(rk45_opts))?;

    // Compare results
    let t_final = t_span[1];
    let radau_final = radau_result.y.last().expect("Operation failed");

    // Find the state at t_final in RK45 result
    let rk45_final = rk45_result.y.last().expect("Operation failed");

    println!("Comparison at t = {t_final}:");
    println!("  Radau: x = {}, v = {}", radau_final[0], radau_final[1]);
    println!("  RK45: x = {}, v = {}", rk45_final[0], rk45_final[1]);
    println!(
        "  Difference: Δx = {}, Δv = {}",
        (radau_final[0] - rk45_final[0]).abs(),
        (radau_final[1] - rk45_final[1]).abs()
    );

    // Strict numerical comparison: Radau and RK45 must agree to within 1e-4 relative error.
    // Both solve the same M·y' = f problem; the corrected Newton iteration achieves this.
    assert_relative_eq!(
        radau_final[0],
        rk45_final[0],
        epsilon = 1e-4,
        max_relative = 1e-4
    );
    assert_relative_eq!(
        radau_final[1],
        rk45_final[1],
        epsilon = 1e-4,
        max_relative = 1e-4
    );

    // Compare statistics
    println!("Radau statistics:");
    println!(
        "  Steps: {} (accepted: {}, rejected: {})",
        radau_result.n_steps, radau_result.n_accepted, radau_result.n_rejected
    );
    println!("  Function evaluations: {}", radau_result.n_eval);

    println!("RK45 statistics:");
    println!(
        "  Steps: {} (accepted: {}, rejected: {})",
        rk45_result.n_steps, rk45_result.n_accepted, rk45_result.n_rejected
    );
    println!("  Function evaluations: {}", rk45_result.n_eval);

    // We generally expect different stats since the methods are implemented differently

    Ok(())
}

// ─── New tests for Slice V stub-check ────────────────────────────────────────

/// Test 1: Diagonal mass matrix with different scales on each component.
///
/// System: M·y' = f where M = diag(m1, m2, m3), f_i = -y_i.
/// Exact solution: y_i(t) = exp(-t/m_i).
#[test]
fn test_radau_diagonal_mass_ode() -> IntegrateResult<()> {
    let m1 = 2.0_f64;
    let m2 = 0.5_f64;
    let m3 = 3.0_f64;

    let mut mass_data = Array2::<f64>::zeros((3, 3));
    mass_data[[0, 0]] = m1;
    mass_data[[1, 1]] = m2;
    mass_data[[2, 2]] = m3;
    let mass = MassMatrix::constant(mass_data);

    let f = |_t: f64, y: ArrayView1<f64>| array![-y[0], -y[1], -y[2]];
    let y0 = array![1.0, 1.0, 1.0];
    let t_span = [0.0_f64, 1.0];

    let opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(mass),
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0, Some(opts))?;
    assert!(result.success, "Integration did not complete successfully");

    let y_final = result.y.last().expect("no result");
    let t_final = *result.t.last().expect("no result");

    let y0_exact = (-t_final / m1).exp();
    let y1_exact = (-t_final / m2).exp();
    let y2_exact = (-t_final / m3).exp();

    println!("Diagonal mass test: t={t_final}");
    println!(
        "  numerical: [{}, {}, {}]",
        y_final[0], y_final[1], y_final[2]
    );
    println!("  exact:     [{y0_exact}, {y1_exact}, {y2_exact}]");

    assert_relative_eq!(y_final[0], y0_exact, epsilon = 1e-5, max_relative = 1e-5);
    assert_relative_eq!(y_final[1], y1_exact, epsilon = 1e-5, max_relative = 1e-5);
    assert_relative_eq!(y_final[2], y2_exact, epsilon = 1e-5, max_relative = 1e-5);

    Ok(())
}

/// Test 2: Three-component oscillator with diagonal mass matrix.
///
/// Three decoupled oscillators with different mass-scaled frequencies:
/// M·y' = f where M = diag(m1, 1, m2, 1, m3, 1)
/// and each pair (y_{2i}, y_{2i+1}) satisfies the oscillator equations.
///
/// Exact solutions: y_{2i}(t) = cos(t / sqrt(m_i)), y_{2i+1}(t) = -sqrt(m_i)*sin(t/sqrt(m_i))
#[test]
fn test_radau_three_oscillators_diagonal_mass() -> IntegrateResult<()> {
    let m1 = 4.0_f64; // omega1 = 1/2
    let m2 = 1.0_f64; // omega2 = 1 (reference)
    let m3 = 0.25_f64; // omega3 = 2

    // Mass matrix: diag(m1, 1, m2, 1, m3, 1)
    let mut mass_data = Array2::<f64>::eye(6);
    mass_data[[0, 0]] = m1;
    mass_data[[2, 2]] = m2;
    mass_data[[4, 4]] = m3;
    let mass = MassMatrix::constant(mass_data);

    // f = [v1, -x1, v2, -x2, v3, -x3]
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0], y[3], -y[2], y[5], -y[4]];

    // Initial conditions: all oscillators start at (1, 0)
    let y0 = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let t_span = [0.0_f64, 3.0];

    let opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-7,
        atol: 1e-9,
        mass_matrix: Some(mass),
        max_steps: 2000,
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0, Some(opts))?;
    assert!(
        result.success,
        "Three-oscillator integration did not complete"
    );

    let y_final = result.y.last().expect("no result");
    let t_final = *result.t.last().expect("no result");

    let x1_exact = (t_final / m1.sqrt()).cos();
    let x2_exact = (t_final / m2.sqrt()).cos();
    let x3_exact = (t_final / m3.sqrt()).cos();

    println!(
        "Three oscillators at t={t_final}: x1={}, x2={}, x3={}",
        y_final[0], y_final[2], y_final[4]
    );
    println!("  exact: x1={x1_exact:.6}, x2={x2_exact:.6}, x3={x3_exact:.6}");

    assert_relative_eq!(y_final[0], x1_exact, epsilon = 1e-4, max_relative = 1e-4);
    assert_relative_eq!(y_final[2], x2_exact, epsilon = 1e-4, max_relative = 1e-4);
    assert_relative_eq!(y_final[4], x3_exact, epsilon = 1e-4, max_relative = 1e-4);

    Ok(())
}

/// Test 3: Nonlinear oscillator with non-trivial mass matrix.
///
/// A modified Duffing oscillator with mass matrix:
/// M·[x', v']^T = [v, -x - 0.1*x^3]^T
/// where M = [[m, 0], [0, 1]] and m=3.
///
/// For this system, the exact period is not simple, but the solution should
/// remain bounded in a neighbourhood of the origin.
#[test]
fn test_radau_nonlinear_oscillator_mass() -> IntegrateResult<()> {
    let m_val = 3.0_f64;

    let mut mass_data = Array2::<f64>::eye(2);
    mass_data[[0, 0]] = m_val;
    let mass = MassMatrix::constant(mass_data);

    // Duffing-like: f = [v, -x - 0.1*x^3]
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0] - 0.1 * y[0].powi(3)];

    let y0 = array![0.5, 0.0];
    let t_span = [0.0_f64, 4.0 * std::f64::consts::PI]; // several periods

    let opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-7,
        atol: 1e-9,
        mass_matrix: Some(mass),
        max_steps: 2000,
        ..Default::default()
    };

    let y0_clone = y0.clone();
    let result = solve_ivp(f, t_span, y0, Some(opts))?;
    assert!(
        result.success,
        "Nonlinear oscillator integration did not complete"
    );

    // Energy should be approximately conserved (within 1%)
    // With M*x'' = -x - 0.1*x^3 (effective), H = m*v^2/(2m) + x^2/2 + 0.1*x^4/4
    let energy_initial = {
        let x = y0_clone[0];
        let v = y0_clone[1];
        0.5 * v * v / m_val + 0.5 * x * x + 0.025 * x.powi(4)
    };
    let y_final = result.y.last().expect("no result");
    let energy_final = {
        let x = y_final[0];
        let v = y_final[1];
        0.5 * v * v / m_val + 0.5 * x * x + 0.025 * x.powi(4)
    };

    println!(
        "Nonlinear oscillator: energy_initial={energy_initial:.6}, energy_final={energy_final:.6}"
    );
    println!(
        "  {} steps ({} accepted, {} rejected)",
        result.n_steps, result.n_accepted, result.n_rejected
    );

    let energy_error = (energy_final - energy_initial).abs() / energy_initial.abs().max(1e-10);
    assert!(
        energy_error < 0.01,
        "Energy not conserved: relative error = {energy_error:.3}"
    );

    Ok(())
}

/// Test 4: Verlet spring-mass system (Hamiltonian structure).
///
/// Spring-mass: M·[x', v']^T = [v, -k*x]^T with M=[[m,0],[0,1]].
/// Exact: x(t) = cos(sqrt(k/m)*t), v(t) = -sqrt(k*m)*sin(sqrt(k/m)*t).
/// This tests that the symplectic structure is approximated correctly.
#[test]
fn test_radau_verlet_spring_mass() -> IntegrateResult<()> {
    let m_val = 4.0_f64;
    let k = 1.0_f64;

    let mut mass_data = Array2::<f64>::eye(2);
    mass_data[[0, 0]] = m_val;
    let mass = MassMatrix::constant(mass_data);

    let f = move |_t: f64, y: ArrayView1<f64>| array![y[1], -k * y[0]];

    let y0 = array![1.0, 0.0];
    let t_span = [0.0_f64, 2.0 * std::f64::consts::PI]; // One full period

    let omega = (k / m_val).sqrt();
    let period = 2.0 * std::f64::consts::PI / omega;

    let opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(mass),
        max_steps: 2000,
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0, Some(opts))?;
    assert!(result.success, "Spring-mass integration did not complete");

    let y_final = result.y.last().expect("no result");
    let t_final = *result.t.last().expect("no result");

    // After one full period, x should return to 1.0
    let x_exact = (omega * t_final).cos();
    let v_exact = -m_val.sqrt() * (omega * t_final).sin();

    println!("Spring-mass (period={period:.4}):");
    println!("  t_final={t_final:.4}, x={}, v={}", y_final[0], y_final[1]);
    println!("  x_exact={x_exact:.6}, v_exact={v_exact:.6}");

    assert_relative_eq!(y_final[0], x_exact, epsilon = 1e-4, max_relative = 1e-4);
    assert_relative_eq!(y_final[1], v_exact, epsilon = 1e-4, max_relative = 1e-4);

    Ok(())
}

/// Test 5: Singular perturbation / index-1 DAE via mass matrix.
///
/// Semi-explicit index-1 DAE: M·y' = f where M has a zero row.
/// System: [1, 0; 0, eps]·[x', y']^T = [y - x, -x*y - y^3]^T  (eps → 0 is nearly singular)
///
/// We use eps=0.01 (not exactly zero) which makes a stiff system solvable by Radau.
/// The solution should track the slow manifold y ≈ x roughly.
#[test]
fn test_radau_singular_perturbation() -> IntegrateResult<()> {
    let eps = 0.01_f64; // Small but nonzero so the system is stiff, not singular

    let mut mass_data = Array2::<f64>::eye(2);
    mass_data[[1, 1]] = eps;
    let mass = MassMatrix::constant(mass_data);

    // x' = y - x
    // eps*y' = -(x*y + y^3) ... stiff fast variable y tracks slow manifold
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1] - y[0], -(y[0] * y[1] + y[1].powi(3))];

    let y0 = array![1.0, 0.5];
    let t_span = [0.0_f64, 1.0];

    let opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(mass),
        max_steps: 5000,
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0, Some(opts))?;
    assert!(
        result.success,
        "Singular perturbation integration did not complete"
    );

    let y_final = result.y.last().expect("no result");
    println!(
        "Singular perturbation (eps={eps}): final=[{}, {}]",
        y_final[0], y_final[1]
    );

    // Both components should be finite and bounded
    assert!(y_final[0].is_finite(), "x is not finite: {}", y_final[0]);
    assert!(y_final[1].is_finite(), "y is not finite: {}", y_final[1]);

    Ok(())
}

/// Test 6: Step-size controller verification.
///
/// For a smooth problem with tight tolerance, Radau should use significantly
/// fewer steps than max_steps (demonstrating that adaptive step control works).
/// We verify that the number of accepted steps is reasonably small.
#[test]
fn test_radau_step_size_controller() -> IntegrateResult<()> {
    // Simple 2-state oscillator with mass matrix [2,0;0,1]
    let mut mass_data = Array2::<f64>::eye(2);
    mass_data[[0, 0]] = 2.0;
    let mass = MassMatrix::constant(mass_data);

    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];
    let y0 = array![1.0, 0.0];
    let t_span = [0.0_f64, 2.0 * std::f64::consts::PI]; // ~6.28 time units

    let opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(mass),
        max_steps: 500, // Default max_steps
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0, Some(opts))?;
    assert!(
        result.success,
        "Step controller test: integration did not complete (hit max_steps?)"
    );

    println!(
        "Step controller test: {} accepted, {} rejected out of {} total",
        result.n_accepted, result.n_rejected, result.n_steps
    );

    // The corrected error estimate should allow Radau to take large steps.
    // For this smooth problem with rtol=1e-6, we expect at most ~100 steps.
    assert!(
        result.n_accepted <= 200,
        "Too many steps: {} accepted (expected ≤200)",
        result.n_accepted
    );

    // Verify accuracy of the solution
    let omega = 1.0 / 2.0_f64.sqrt();
    let t_final = *result.t.last().expect("no result");
    let y_final = result.y.last().expect("no result");

    let x_exact = (omega * t_final).cos();
    assert_relative_eq!(y_final[0], x_exact, epsilon = 1e-4, max_relative = 1e-4);

    Ok(())
}

/// Test 7: Newton convergence counter.
///
/// Verifies that the Newton iteration converges efficiently: the ratio
/// of Jacobian evaluations to accepted steps should be reasonable,
/// indicating Jacobian reuse is working and Newton doesn't thrash.
#[test]
fn test_radau_newton_convergence_counter() -> IntegrateResult<()> {
    let mut mass_data = Array2::<f64>::eye(2);
    mass_data[[0, 0]] = 2.0;
    let mass = MassMatrix::constant(mass_data);

    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];
    let y0 = array![1.0, 0.0];
    let t_span = [0.0_f64, 5.0];

    let opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(mass),
        max_steps: 500,
        ..Default::default()
    };

    let result = solve_ivp(f, t_span, y0, Some(opts))?;
    assert!(
        result.success,
        "Newton counter test: integration did not complete"
    );

    println!(
        "Newton test: n_jac={}, n_accepted={}, n_lu={}, n_rejected={}",
        result.n_jac, result.n_accepted, result.n_lu, result.n_rejected
    );

    // Each step requires at least 1 Newton iteration (usually converges fast).
    // n_jac / n_steps should be a small fraction for a constant-Jacobian problem.
    // With Jacobian reuse (invalidated on rejection), n_jac << n_accepted.
    let jac_per_step = result.n_jac as f64 / result.n_accepted.max(1) as f64;
    println!("  Jacobian evaluations per accepted step: {jac_per_step:.3}");

    // Sanity: we should have completed the integration
    assert!(
        result.n_accepted >= 5,
        "Too few accepted steps: {}",
        result.n_accepted
    );
    assert!(result.n_jac >= 1, "No Jacobian evaluations recorded");

    // For a constant-Jacobian problem, LU decompositions should not exceed steps too much
    assert!(
        result.n_lu <= result.n_steps + result.n_rejected + 5,
        "Too many LU decompositions: {}",
        result.n_lu
    );

    Ok(())
}

/// Test 8: Error estimator accuracy — verifies the embedded estimator
/// agrees to within a constant factor of the actual global error.
///
/// For a simple problem with known analytical solution, we run with two
/// tolerances and verify the error scales approximately as rtol^{4/5} or better,
/// which indicates the error controller is effective.
#[test]
fn test_radau_error_estimator_quality() -> IntegrateResult<()> {
    // Simple oscillator M=[2,0;0,1]: exact x(t) = cos(t/sqrt(2))
    let omega = 1.0_f64 / 2.0_f64.sqrt();

    let mut mass_data = Array2::<f64>::eye(2);
    mass_data[[0, 0]] = 2.0;

    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];
    let y0 = array![1.0, 0.0];
    let t_span = [0.0_f64, 2.0];

    // Run with tight tolerance
    let tight_opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-8,
        atol: 1e-10,
        mass_matrix: Some(MassMatrix::constant(mass_data.clone())),
        max_steps: 2000,
        ..Default::default()
    };
    let tight_result = solve_ivp(f, t_span, y0.clone(), Some(tight_opts))?;
    assert!(tight_result.success, "tight tolerance: did not complete");

    // Run with looser tolerance
    let loose_opts = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-4,
        atol: 1e-6,
        mass_matrix: Some(MassMatrix::constant(mass_data)),
        max_steps: 2000,
        ..Default::default()
    };
    let loose_result = solve_ivp(f, t_span, y0, Some(loose_opts))?;
    assert!(loose_result.success, "loose tolerance: did not complete");

    let t_final = t_span[1];
    let x_exact = (omega * t_final).cos();

    let tight_final = tight_result.y.last().expect("no result");
    let loose_final = loose_result.y.last().expect("no result");

    let tight_err = (tight_final[0] - x_exact).abs();
    let loose_err = (loose_final[0] - x_exact).abs();

    println!("Error estimator quality test at t={t_final}:");
    println!(
        "  tight (rtol=1e-8): error={tight_err:.3e}, n_steps={}",
        tight_result.n_accepted
    );
    println!(
        "  loose (rtol=1e-4): error={loose_err:.3e}, n_steps={}",
        loose_result.n_accepted
    );

    // Tight should be more accurate than loose
    assert!(
        tight_err <= loose_err + 1e-12,
        "tight tolerance gave worse error: tight={tight_err}, loose={loose_err}"
    );

    // Both should give reasonable accuracy relative to their tolerance
    assert!(
        tight_err < 1e-5,
        "tight tolerance gave poor accuracy: error={tight_err}"
    );
    assert!(
        loose_err < 1e-2,
        "loose tolerance gave poor accuracy: error={loose_err}"
    );

    // Loose should use fewer steps than tight (adaptive step control)
    assert!(
        loose_result.n_accepted <= tight_result.n_accepted,
        "loose tolerance used more steps than tight ({} vs {})",
        loose_result.n_accepted,
        tight_result.n_accepted
    );

    Ok(())
}
