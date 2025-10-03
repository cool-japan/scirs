//! BDF-specific DAE solver for a pendulum system
//!
//! This example demonstrates the usage of specialized BDF methods for
//! solving a pendulum system represented as a semi-explicit DAE.
//!
//! The pendulum system is described by:
//! x' = vx
//! y' = vy
//! vx' = -2*x*λ
//! vy' = -g - 2*y*λ
//! 0 = x² + y² - L²
//!
//! where λ is the Lagrange multiplier that enforces the constraint.

use plotters::prelude::*;
use scirs2_core::ndarray::{array, ArrayView1};
use scirs2_core::numeric::Float;
use scirs2_integrate::{bdf_semi_explicit_dae, DAEIndex, DAEOptions, DAEType};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Constants
    let g = 9.81f64; // Gravitational acceleration (m/s²)
    let length = 1.0; // Pendulum length (m)
    let period = 2.0 * std::f64::consts::PI * (length / g).sqrt();

    println!("Pendulum period (theoretical): {period:.6} seconds");

    // Initial conditions (pendulum released from 45 degrees)
    let theta0 = std::f64::consts::PI / 4.0;
    let x0 = length * theta0.sin();
    let y0 = -length * theta0.cos();
    let vx0 = 0.0;
    let vy0 = 0.0;

    // Combine into a single differential variables array
    let diff_vars_0 = array![x0, y0, vx0, vy0];

    // Initial Lagrange multiplier (algebraic variable)
    // Typically derived from the constraint equation
    let lambda0 = 0.0;
    let alg_vars_0 = array![lambda0];

    // Define the time span for the integration (0 to 10 seconds)
    let t_span = [0.0, 10.0];

    // Define the system of differential equations
    let f = |_t: f64, diff_vars: ArrayView1<f64>, alg_vars: ArrayView1<f64>| {
        let x = diff_vars[0];
        let y = diff_vars[1];
        let vx = diff_vars[2];
        let vy = diff_vars[3];
        let lambda = alg_vars[0];

        // x' = vx
        // y' = vy
        // vx' = -2*x*λ
        // vy' = -g - 2*y*λ
        array![vx, vy, -2.0 * x * lambda, -g - 2.0 * y * lambda]
    };

    // Define the constraint equation
    let g_constraint = |_t: f64, diff_vars: ArrayView1<f64>, _alg_vars: ArrayView1<f64>| {
        let x = diff_vars[0];
        let y = diff_vars[1];

        // Constraint: x² + y² - L² = 0
        array![x * x + y * y - length * length]
    };

    // Set solver options
    let options = DAEOptions {
        dae_type: DAEType::SemiExplicit,
        index: DAEIndex::Index1,
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 10_000,
        max_newton_iterations: 10,
        newton_tol: 1e-8,
        h0: Some(0.01),
        min_step: Some(1e-10),
        max_step: Some(0.1),
        max_order: Some(5),
        ..Default::default()
    };

    // Solve the DAE system using specialized BDF method
    let result = bdf_semi_explicit_dae(f, g_constraint, t_span, diff_vars_0, alg_vars_0, options)?;

    // Print some performance statistics
    println!("BDF DAE solver performance metrics:");
    println!("  Steps taken: {}", result.n_steps);
    println!("  Steps accepted: {}", result.n_accepted);
    println!("  Steps rejected: {}", result.n_rejected);
    println!("  Function evaluations: {}", result.n_eval);
    println!("  Constraint evaluations: {}", result.n_constraint_eval);
    println!("  Jacobian evaluations: {}", result.n_jac);
    println!("  LU decompositions: {}", result.n_lu);
    println!("  Success: {}", result.success);

    if let Some(msg) = &result.message {
        println!("  Message: {msg}");
    }

    // Calculate the energy at each time point
    let mut energy = Vec::new();
    for i in 0..result.t.len() {
        let _x = result.x[i][0];
        let y = result.x[i][1];
        let vx = result.x[i][2];
        let vy = result.x[i][3];

        // Kinetic energy: 0.5 * (vx² + vy²)
        let kinetic = 0.5 * (vx * vx + vy * vy);

        // Potential energy: gy (assuming y=0 at bottom)
        let potential = g * (y + length);

        // Total energy
        energy.push(kinetic + potential);
    }

    // Compute the maximum energy deviation
    let initial_energy = energy[0];
    let max_deviation = energy
        .iter()
        .map(|&e| (e - initial_energy).abs() / initial_energy)
        .fold(0.0, |max, dev| max.max(dev));

    println!("Maximum energy deviation: {max_deviation:.6e}");

    // Create a plot of the pendulum trajectory
    let root = BitMapBackend::new("pendulum_trajectory_bdf.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_x = result
        .x
        .iter()
        .map(|x| x[0])
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_x = result
        .x
        .iter()
        .map(|x| x[0])
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let min_y = result
        .x
        .iter()
        .map(|x| x[1])
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_y = result
        .x
        .iter()
        .map(|x| x[1])
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let x_margin = (max_x - min_x) * 0.1;
    let y_margin = (max_y - min_y) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Pendulum Trajectory", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (min_x - x_margin)..(max_x + x_margin),
            (min_y - y_margin)..(max_y + y_margin),
        )?;

    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    // Draw the trajectory
    chart.draw_series(LineSeries::new(result.x.iter().map(|p| (p[0], p[1])), &RED))?;

    // Plot the energy over time
    let root = BitMapBackend::new("pendulum_energy_bdf.png", (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_energy = energy.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));
    let max_energy = energy
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let energy_range = max_energy - min_energy;
    let energy_margin = energy_range * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Pendulum Energy", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0.0..t_span[1],
            (min_energy - energy_margin)..(max_energy + energy_margin),
        )?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_desc("Time (s)")
        .y_desc("Energy (J)")
        .draw()?;

    // Draw the energy plot
    chart.draw_series(LineSeries::new(
        result.t.iter().zip(energy.iter()).map(|(&t, &e)| (t, e)),
        &BLUE,
    ))?;

    println!("Plots generated: pendulum_trajectory_bdf.png and pendulum_energy_bdf.png");

    Ok(())
}
