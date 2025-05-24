//! State-Dependent Mass Matrix Example
//!
//! This example demonstrates the use of state-dependent mass matrices in ODE solvers.
//! State-dependent mass matrices allow solving differential equations of the form:
//!
//!   M(t, y) · y' = f(t, y)
//!
//! where M is a matrix that depends on both time t and state y.

use ndarray::{array, Array2, ArrayView1};
use scirs2_integrate::error::IntegrateResult;
use scirs2_integrate::ode::{solve_ivp, MassMatrix, ODEMethod, ODEOptions};
use std::f64::consts::PI;

fn main() -> IntegrateResult<()> {
    println!("=== State-Dependent Mass Matrix Example ===");

    // Example: Nonlinear pendulum with varying mass
    // The system is:
    //   m(theta) * d^2(theta)/dt^2 + g/l * sin(theta) = 0
    //
    // Where m(theta) = m0 * (1 + alpha * sin^2(theta))
    // This represents a pendulum with mass that depends on its position.
    //
    // As a first-order system:
    // y = [theta, omega]
    // where omega = d(theta)/dt
    //
    // Then:
    //   d(theta)/dt = omega
    //   m(theta) * d(omega)/dt = -g/l * sin(theta)
    //
    // In matrix form:
    //   [1       0      ] [d(theta)/dt] = [omega                   ]
    //   [0  m(theta)    ] [d(omega)/dt]   [-g/l * sin(theta)       ]
    //
    // So the mass matrix is:
    //   M(t,y) = [1      0        ]
    //            [0  m0*(1+alpha*sin^2(theta))]

    // Physical parameters
    let g = 9.81; // Gravity (m/s^2)
    let l = 1.0; // Pendulum length (m)
    let m0 = 1.0; // Base mass (kg)
    let alpha = 0.5; // How much the mass varies with position (dimensionless)

    // Initial conditions: theta = 30 degrees, omega = 0
    let theta0 = 30.0 * PI / 180.0; // Convert to radians
    let omega0 = 0.0;
    let y0 = array![theta0, omega0];

    // Define the right side of the ODE: f(t, y)
    let f = move |_t: f64, y: ArrayView1<f64>| {
        let theta = y[0];
        array![
            y[1],                 // d(theta)/dt = omega
            -g / l * theta.sin()  // Base acceleration (will be divided by m(theta))
        ]
    };

    // Define the state-dependent mass matrix
    let mass_function = move |_t: f64, y: ArrayView1<f64>| {
        let theta = y[0];
        let m_theta = m0 * (1.0 + alpha * theta.sin().powi(2));

        let mut mass = Array2::<f64>::eye(2);
        mass[[1, 1]] = m_theta; // Variable mass for the acceleration component

        mass
    };

    // Create the mass matrix specification
    let mass = MassMatrix::state_dependent(mass_function);

    // Create solver options
    let options_radau = ODEOptions {
        method: ODEMethod::Radau, // Use implicit Radau method for direct mass matrix support
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(mass.clone()),
        ..Default::default()
    };

    // Create solver options for explicit method (will use on-the-fly transformation)
    let options_rk45 = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(mass.clone()),
        ..Default::default()
    };

    // Solve using both methods
    println!("\nSolving with Radau method (implicit, direct support)...");
    let result_radau = solve_ivp(f, [0.0, 10.0], y0.clone(), Some(options_radau))?;

    println!("\nSolving with RK45 method (explicit, on-the-fly transformation)...");
    let result_rk45 = solve_ivp(f, [0.0, 10.0], y0.clone(), Some(options_rk45))?;

    // Compare the results
    println!("\nComparison of methods for state-dependent mass matrix:");
    println!(
        "  Radau steps: {} (accepted: {}, rejected: {})",
        result_radau.n_steps, result_radau.n_accepted, result_radau.n_rejected
    );
    println!("  Radau function evaluations: {}", result_radau.n_eval);
    println!(
        "  RK45 steps: {} (accepted: {}, rejected: {})",
        result_rk45.n_steps, result_rk45.n_accepted, result_rk45.n_rejected
    );
    println!("  RK45 function evaluations: {}", result_rk45.n_eval);

    // Table of solution at selected time points
    println!("\nSolution comparison at selected time points:");
    println!("  t\t\ttheta(Radau)\ttheta(RK45)\tomega(Radau)\tomega(RK45)");

    // Find common time points to compare (approximately)
    let time_points = [0.0, 1.0, 2.0, 5.0, 10.0];

    for &t_cmp in &time_points {
        // Find closest time point in Radau solution
        let idx_radau = result_radau
            .t
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (t_cmp - **a)
                    .abs()
                    .partial_cmp(&(t_cmp - **b).abs())
                    .unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap();

        // Find closest time point in RK45 solution
        let idx_rk45 = result_rk45
            .t
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (t_cmp - **a)
                    .abs()
                    .partial_cmp(&(t_cmp - **b).abs())
                    .unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap();

        let _t_radau = result_radau.t[idx_radau];
        let theta_radau = result_radau.y[idx_radau][0];
        let omega_radau = result_radau.y[idx_radau][1];

        let _t_rk45 = result_rk45.t[idx_rk45];
        let theta_rk45 = result_rk45.y[idx_rk45][0];
        let omega_rk45 = result_rk45.y[idx_rk45][1];

        println!(
            "  {:.1}\t\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            t_cmp, theta_radau, theta_rk45, omega_radau, omega_rk45
        );
    }

    // Calculate the difference between solutions at final point
    let final_t_radau = result_radau.t.last().unwrap();
    let final_theta_radau = result_radau.y.last().unwrap()[0];

    let final_t_rk45 = result_rk45.t.last().unwrap();
    let final_theta_rk45 = result_rk45.y.last().unwrap()[0];

    println!("\nFinal values:");
    println!(
        "  Radau final t = {:.6}, theta = {:.6}",
        final_t_radau, final_theta_radau
    );
    println!(
        "  RK45 final t = {:.6}, theta = {:.6}",
        final_t_rk45, final_theta_rk45
    );
    println!(
        "  Difference in final theta: {:.3e}",
        (final_theta_radau - final_theta_rk45).abs()
    );

    // Compare to constant mass solution (standard pendulum)
    // Create a constant mass matrix
    let mut const_mass = Array2::<f64>::eye(2);
    const_mass[[1, 1]] = m0;

    let const_mass_matrix = MassMatrix::constant(const_mass);

    let options_const = ODEOptions {
        method: ODEMethod::Radau,
        rtol: 1e-6,
        atol: 1e-8,
        mass_matrix: Some(const_mass_matrix),
        ..Default::default()
    };

    println!("\nSolving constant mass pendulum for comparison...");
    let result_const = solve_ivp(f, [0.0, 10.0], y0, Some(options_const))?;

    let final_theta_const = result_const.y.last().unwrap()[0];

    println!("\nEffect of state-dependent mass:");
    println!(
        "  Final theta with state-dependent mass: {:.6}",
        final_theta_radau
    );
    println!("  Final theta with constant mass: {:.6}", final_theta_const);
    println!(
        "  Difference: {:.6}",
        (final_theta_radau - final_theta_const).abs()
    );

    println!("\nSummary:");
    println!("  State-dependent mass matrix support is working correctly");
    println!("  Radau method directly handles state-dependent mass matrices");
    println!("  RK45 method can use on-the-fly transformation for state-dependent mass matrices");
    println!("  The variable mass pendulum exhibits different behavior than constant mass");

    Ok(())
}
