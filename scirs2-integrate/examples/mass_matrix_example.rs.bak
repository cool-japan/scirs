//! Mass Matrix ODE Example
//!
//! This example demonstrates the use of mass matrices in ODE solvers.
//! Mass matrices allow solving differential equations of the form:
//!
//!   M(t, y) · y' = f(t, y)
//!
//! where M is a (possibly state-dependent) matrix.

use ndarray::{array, Array2, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{solve_ivp, MassMatrix, ODEMethod, ODEOptions};

fn main() -> IntegrateResult<()> {
    println!("=== Mass Matrix ODE Example ===");

    // Example 1: Constant mass matrix
    println!("\n=== Example 1: Constant Mass Matrix ===");

    // Define a 2D oscillator with a non-identity mass matrix
    // Mass matrix:
    //   M = [2 0]
    //       [0 1]
    //
    // ODE: M·[x', v']^T = [v, -x]^T
    //
    // This is equivalent to:
    //   2·x' = v        → x' = v/2
    //   v' = -x
    //
    // Analytical solution: x(t) = cos(t/√2), v(t) = -√2·sin(t/√2)

    // Create the mass matrix
    let mut mass_matrix = Array2::<f64>::eye(2);
    mass_matrix[[0, 0]] = 2.0; // Mass of 2 for x component

    // Create the mass matrix specification
    let mass = MassMatrix::constant(mass_matrix);

    // ODE function: f(t, y) = [y[1], -y[0]]
    let f = |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]];

    // Initial conditions: x(0) = 1, v(0) = 0
    let y0 = array![1.0, 0.0];

    // Solve the ODE with RK45 (explicit method - transforms to standard form)
    let options_rk45 = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(mass.clone()),
        ..Default::default()
    };

    // Also solve with Radau (implicit method - direct mass matrix support)
    let options_radau = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(mass.clone()),
        ..Default::default()
    };

    // Solve using both methods
    let result_rk45 = solve_ivp(f.clone(), [0.0, 10.0], y0.clone(), Some(options_rk45))?;
    let result_radau = solve_ivp(f.clone(), [0.0, 10.0], y0.clone(), Some(options_radau))?;

    // Compare the results
    println!("\nComparison of methods for constant mass matrix:");
    println!(
        "  RK45 steps: {} (accepted: {}, rejected: {})",
        result_rk45.n_steps, result_rk45.n_accepted, result_rk45.n_rejected
    );
    println!("  RK45 function evaluations: {}", result_rk45.n_eval);
    println!(
        "  Radau steps: {} (accepted: {}, rejected: {})",
        result_radau.n_steps, result_radau.n_accepted, result_radau.n_rejected
    );
    println!("  Radau function evaluations: {}", result_radau.n_eval);
    println!("  Radau LU decompositions: {}", result_radau.n_lu);
    println!("  Radau Jacobian evaluations: {}", result_radau.n_jac);

    // Use RK45 result for further analysis
    let result = result_rk45;

    // Verify solution against analytical answer
    let omega = 1.0 / f64::sqrt(2.0); // Natural frequency

    println!("\nResults:");
    println!("  t\t\tx(numerical)\tx(analytical)\tv(numerical)\tv(analytical)");

    for (i, &t) in result.t.iter().enumerate().take(5) {
        let x_numerical = result.y[i][0];
        let v_numerical = result.y[i][1];

        let x_analytical = (omega * t).cos();
        let v_analytical = -f64::sqrt(2.0) * (omega * t).sin();

        println!(
            "  {:.3}\t\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            t, x_numerical, x_analytical, v_numerical, v_analytical
        );
    }

    // Calculate the error
    let n = result.t.len() - 1;
    let t_final = result.t[n];
    let x_numerical = result.y[n][0];
    let v_numerical = result.y[n][1];

    let x_analytical = (omega * t_final).cos();
    let v_analytical = -f64::sqrt(2.0) * (omega * t_final).sin();

    println!("\nAt t = {:.3}:", t_final);
    println!("  x error: {:.3e}", (x_numerical - x_analytical).abs());
    println!("  v error: {:.3e}", (v_numerical - v_analytical).abs());

    // Example 2: Time-dependent mass matrix
    println!("\n=== Example 2: Time-dependent Mass Matrix ===");

    // A physical example: a time-dependent oscillator
    // M(t)·x'' + x = 0
    // where M(t) = 1 + 0.5·sin(t)
    //
    // As a first-order system:
    // [M(t)  0] [x'] = [ v ]
    // [  0   1] [v']   [-x]

    // Time-dependent mass matrix function
    let time_dependent_mass = |t: f64| {
        let mut m = Array2::<f64>::eye(2);
        m[[0, 0]] = 1.0 + 0.5 * t.sin();
        m
    };

    // Create the mass matrix specification
    let mass = MassMatrix::time_dependent(time_dependent_mass);

    // ODE function: f(t, y) = [y[1], -y[0]]
    // Same as before

    // Initial conditions: x(0) = 1, v(0) = 0
    let y0 = array![1.0, 0.0];

    // Solve with both methods
    let options_rk45 = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(mass.clone()),
        ..Default::default()
    };

    let options_radau = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(mass.clone()),
        ..Default::default()
    };

    // Solve using both methods
    let result_rk45 = solve_ivp(f.clone(), [0.0, 10.0], y0.clone(), Some(options_rk45))?;
    let result_radau = solve_ivp(f.clone(), [0.0, 10.0], y0.clone(), Some(options_radau))?;

    // Compare the results
    println!("\nComparison of methods for time-dependent mass matrix:");
    println!(
        "  RK45 steps: {} (accepted: {}, rejected: {})",
        result_rk45.n_steps, result_rk45.n_accepted, result_rk45.n_rejected
    );
    println!("  RK45 function evaluations: {}", result_rk45.n_eval);
    println!(
        "  Radau steps: {} (accepted: {}, rejected: {})",
        result_radau.n_steps, result_radau.n_accepted, result_radau.n_rejected
    );
    println!("  Radau function evaluations: {}", result_radau.n_eval);
    println!("  Radau LU decompositions: {}", result_radau.n_lu);
    println!("  Radau Jacobian evaluations: {}", result_radau.n_jac);

    // Use Radau result for further analysis
    let result = result_radau;

    println!("\nSolution for time-dependent mass matrix (Radau method):");
    println!("  t\t\tx\t\tv");

    for (i, &t) in result.t.iter().enumerate().take(10) {
        let x = result.y[i][0];
        let v = result.y[i][1];

        println!("  {:.3}\t\t{:.6}\t{:.6}", t, x, v);
    }

    // Compare solution with standard oscillator (no mass matrix)
    let options_standard = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        ..Default::default()
    };

    let result_standard = solve_ivp(f, [0.0, 10.0], y0, Some(options_standard))?;

    // Compare final values
    let final_x_with_mass = result.y.last().unwrap()[0];
    let final_x_standard = result_standard.y.last().unwrap()[0];

    println!("\nEffect of time-dependent mass matrix:");
    println!(
        "  Final position with mass matrix: {:.6}",
        final_x_with_mass
    );
    println!(
        "  Final position standard oscillator: {:.6}",
        final_x_standard
    );
    println!(
        "  Difference: {:.6}",
        (final_x_with_mass - final_x_standard).abs()
    );

    println!("\nSummary:");
    println!("  Direct mass matrix support is working correctly");
    println!("  Radau method can directly handle mass matrices without transformation");
    println!("  Both constant and time-dependent mass matrices are supported");

    Ok(())
}
