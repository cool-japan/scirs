//! Comparison of DAE solver methods
//!
//! This example compares the performance and accuracy of different DAE solver methods:
//! 1. Basic approach: Converting DAE to ODE by solving constraints
//! 2. Standard BDF: Using general BDF methods from ODE solvers
//! 3. Specialized BDF: Using DAE-specific BDF methods
//!
//! The test problem is a simple pendulum DAE system, which we can solve analytically
//! for comparison purposes.

#![allow(dead_code)]

use scirs2_core::ndarray::{array, Array1, ArrayView1};
use scirs2_core::numeric::Float;
// use plotters::prelude::*;
use scirs2_integrate::{
    bdf_semi_explicit_dae, solve_ivp, solve_semi_explicit_dae, DAEIndex, DAEOptions, DAEType,
    ODEMethod, ODEOptions,
};
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing DAE solver methods for a pendulum system");

    // Constants
    let g = 9.81f64; // Gravitational acceleration (m/s²)
    let length = 1.0; // Pendulum length (m)

    // Initial conditions (pendulum released from 45 degrees)
    let theta0 = std::f64::consts::PI / 4.0;
    let x0 = length * theta0.sin();
    let y0 = -length * theta0.cos();
    let vx0 = 0.0;
    let vy0 = 0.0;

    // Combine into a single differential variables array
    let diff_vars_0 = array![x0, y0, vx0, vy0];

    // Initial Lagrange multiplier (algebraic variable)
    let lambda0 = g / (2.0 * length); // Value at equilibrium
    let alg_vars_0 = array![lambda0];

    // Define the time span for the integration (0 to 5 seconds)
    let t_span = [0.0, 5.0];

    // Define the system of differential equations for the pendulum
    let f = |_t: f64, diff_vars: ArrayView1<f64>, alg_vars: ArrayView1<f64>| {
        let x = diff_vars[0];
        let y = diff_vars[1];
        let vx = diff_vars[2];
        let vy = diff_vars[3];
        let lambda = alg_vars[0];

        // x' = vx
        // y' = vy
        // vx' = -lambda*x
        // vy' = -g - lambda*y
        array![vx, vy, -lambda * x, -g - lambda * y]
    };

    // Define the constraint equation (the pendulum has fixed length)
    let g_constraint = |_t: f64, diff_vars: ArrayView1<f64>, _alg_vars: ArrayView1<f64>| {
        let x = diff_vars[0];
        let y = diff_vars[1];

        // Constraint: x² + y² - L² = 0
        array![x * x + y * y - length * length]
    };

    // Sample points for final comparison
    let n_sample = 100;
    let dt = (t_span[1] - t_span[0]) / (n_sample as f64);
    let sample_times: Vec<f64> = (0..=n_sample)
        .map(|i| t_span[0] + dt * (i as f64))
        .collect();

    // Run analytical solution for comparison
    let _analytical = compute_analytical_solution(g, length, theta0, &sample_times);

    // Store results for comparison
    let mut solver_methods = vec![];
    let mut solution_times = vec![];
    let mut solution_errors = vec![];
    let mut constraint_violations = vec![];

    // 1. Run solver using the general DAE approach
    println!("\nRunning general DAE solver...");
    let start_time = Instant::now();

    let options = DAEOptions {
        dae_type: DAEType::SemiExplicit,
        index: DAEIndex::Index1,
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 10000,
        max_newton_iterations: 10,
        newton_tol: 1e-8,
        h0: Some(0.01),
        min_step: Some(1e-10),
        max_step: Some(0.1),
        method: ODEMethod::Bdf,
        ..Default::default()
    };

    let result_general = solve_semi_explicit_dae(
        f,
        g_constraint,
        t_span,
        diff_vars_0.clone(),
        alg_vars_0.clone(),
        Some(options.clone()),
    )?;

    let elapsed_general = start_time.elapsed();

    // Compute error metrics
    let (error_general, violation_general) =
        compute_error_metrics(&result_general, g, length, theta0, g_constraint);

    solver_methods.push("General DAE".to_string());
    solution_times.push(elapsed_general.as_secs_f64());
    solution_errors.push(error_general);
    constraint_violations.push(violation_general);

    println!("  Time: {:.6} seconds", elapsed_general.as_secs_f64());
    println!("  Error: {error_general:.6e}");
    println!("  Constraint violation: {violation_general:.6e}");
    println!("  Steps: {}", result_general.n_steps);

    // 2. Run solver using standard ODE solver with constraints solved at each step
    // This is a simplified approach where we convert the DAE to an ODE by solving
    // the constraint at each step using Newton's method
    println!("\nRunning ODE solver with constraints solved at each step...");
    let start_time = Instant::now();

    // Define the ODE system (without lambdas)
    let f_ode = |_t: f64, y: ArrayView1<f64>| {
        let x = y[0];
        let y_pos = y[1];
        let vx = y[2];
        let vy = y[3];

        // Calculate lambda from the acceleration constraint
        // x*x'' + y*y'' + vx² + vy² = 0
        // x*(-lambda*x) + y*(-g - lambda*y) + vx² + vy² = 0
        // -lambda*(x² + y²) - g*y + vx² + vy² = 0
        // lambda = (vx² + vy² - g*y) / (x² + y²)

        let v_squared = vx * vx + vy * vy;
        let lambda = (v_squared - g * y_pos) / (x * x + y_pos * y_pos);

        array![vx, vy, -lambda * x, -g - lambda * y_pos]
    };

    let ode_options = ODEOptions {
        method: ODEMethod::Bdf,
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 10000,
        h0: Some(0.01),
        min_step: Some(1e-10),
        max_step: Some(0.1),
        ..Default::default()
    };

    let result_ode = solve_ivp(f_ode, t_span, diff_vars_0.clone(), Some(ode_options))?;

    let elapsed_ode = start_time.elapsed();

    // Compute error metrics for ODE solution
    // We'll need to adapt the error calculation for the ODE solution format
    let mut error_ode = 0.0;
    let mut violation_ode = 0.0;

    for i in 0..result_ode.t.len() {
        let t = result_ode.t[i];
        let x = result_ode.y[i][0];
        let y = result_ode.y[i][1];

        // Calculate analytical solution at this time
        let theta_analytical = theta0 * (g / length).sqrt() * t;
        let x_analytical = length * theta_analytical.sin();
        let y_analytical = -length * theta_analytical.cos();

        // Error is the Euclidean distance between numerical and analytical solution
        let error_at_t = ((x - x_analytical).powi(2) + (y - y_analytical).powi(2)).sqrt();
        error_ode = error_ode.max(error_at_t);

        // Constraint violation is how far off we are from the fixed length
        let constraint_at_t = (x * x + y * y - length * length).abs();
        violation_ode = violation_ode.max(constraint_at_t);
    }

    solver_methods.push("ODE with Constraints".to_string());
    solution_times.push(elapsed_ode.as_secs_f64());
    solution_errors.push(error_ode);
    constraint_violations.push(violation_ode);

    println!("  Time: {:.6} seconds", elapsed_ode.as_secs_f64());
    println!("  Error: {error_ode:.6e}");
    println!("  Constraint violation: {violation_ode:.6e}");
    println!("  Steps: {}", result_ode.n_steps);

    // 3. Run solver using specialized BDF method for DAEs
    println!("\nRunning specialized BDF DAE solver...");
    let start_time = Instant::now();

    let result_bdf = bdf_semi_explicit_dae(
        f,
        g_constraint,
        t_span,
        diff_vars_0.clone(),
        alg_vars_0.clone(),
        options,
    )?;

    let elapsed_bdf = start_time.elapsed();

    // Compute error metrics
    let (error_bdf, violation_bdf) =
        compute_error_metrics(&result_bdf, g, length, theta0, g_constraint);

    solver_methods.push("Specialized BDF".to_string());
    solution_times.push(elapsed_bdf.as_secs_f64());
    solution_errors.push(error_bdf);
    constraint_violations.push(violation_bdf);

    println!("  Time: {:.6} seconds", elapsed_bdf.as_secs_f64());
    println!("  Error: {error_bdf:.6e}");
    println!("  Constraint violation: {violation_bdf:.6e}");
    println!("  Steps: {}", result_bdf.n_steps);

    // Print performance comparison
    println!("\nPerformance Comparison:");
    println!(
        "{:<20} {:<15} {:<15} {:<15} {:<10}",
        "Method", "Time (s)", "Max Error", "Constraint Viol.", "Steps"
    );
    println!("{:-<65}", "");

    for i in 0..solver_methods.len() {
        let method = &solver_methods[i];
        let time = solution_times[i];
        let error = solution_errors[i];
        let violation = constraint_violations[i];

        println!(
            "{:<20} {:<15.6} {:<15.6e} {:<15.6e} {:<10}",
            method,
            time,
            error,
            violation,
            match i {
                0 => result_general.n_steps,
                1 => result_ode.n_steps,
                2 => result_bdf.n_steps,
                _ => 0,
            }
        );
    }

    // Save trajectories for visualization (disabled due to missing dependencies)
    /*
    let root = BitMapBackend::new("dae_comparison_trajectory.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_x = -length * 1.1;
    let max_x = length * 1.1;
    let min_y = -length * 1.1;
    let max_y = length * 1.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Pendulum Trajectory Comparison", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    // Analytical trajectory (circle)
    let n_points = 100;
    let analytical_circle: Vec<(f64, f64)> = (0..=n_points)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_points as f64);
            (length * angle.sin(), -length * angle.cos())
        })
        .collect();

    chart
        .draw_series(LineSeries::new(analytical_circle, &BLACK.mix(0.5)))?
        .label("Analytical (Circle)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK.mix(0.5)));

    // General DAE solution
    let general_trajectory: Vec<(f64, f64)> =
        result_general.x.iter().map(|q| (q[0], q[1])).collect();

    chart
        .draw_series(LineSeries::new(general_trajectory, &RED))?
        .label("General DAE")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // ODE solution
    let ode_trajectory: Vec<(f64, f64)> = result_ode.y.iter().map(|q| (q[0], q[1])).collect();

    chart
        .draw_series(LineSeries::new(ode_trajectory, &GREEN))?
        .label("ODE with Constraints")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    // Specialized BDF solution
    let bdf_trajectory: Vec<(f64, f64)> = result_bdf.x.iter().map(|q| (q[0], q[1])).collect();

    chart
        .draw_series(LineSeries::new(bdf_trajectory, &BLUE))?
        .label("Specialized BDF")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Error comparison plot
    let root = BitMapBackend::new("dae_comparison_errors.png", (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_error = solution_errors.iter().fold(f64::INFINITY, |a, &b| a.min(b)) / 10.0;
    let max_error = solution_errors.iter().fold(0.0, |a, &b| a.max(b)) * 10.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("Error Comparison", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..solver_methods.len(), min_error..max_error)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .y_desc("Max Error (log scale)")
        .y_labels(10)
        .label_style(("sans-serif", 15))
        .y_scale(ScaleTransform::Logarithmic(10.0))
        .draw()?;

    chart.draw_series(ColumnSeries::new(
        (0..solver_methods.len()).map(|i| (i, solution_errors[i])),
        20,
        &BLUE.mix(0.6),
    ))?;

    // Add method names to x-axis
    for (i, method) in solver_methods.iter().enumerate() {
        chart.draw_series(vec![Text::new(
            method.clone(),
            (i, min_error),
            ("sans-serif", 15).into_font().color(&BLACK),
        )])?;
    }
    */

    // println!("\nComparison plots saved as:");
    // println!("  dae_comparison_trajectory.png");
    // println!("  dae_comparison_errors.png");

    Ok(())
}

/// Calculate the analytical solution for the pendulum
#[allow(dead_code)]
fn compute_analytical_solution(g: f64, length: f64, theta0: f64, times: &[f64]) -> Vec<(f64, f64)> {
    // For small oscillations, the solution is approximately:
    // theta(t) = theta0 * cos(omega * t)
    // where omega = sqrt(g/L)

    let omega = (g / length).sqrt();

    times
        .iter()
        .map(|&t| {
            let theta = theta0 * (omega * t).cos();
            let x = length * theta.sin();
            let y = -length * theta.cos();
            (x, y)
        })
        .collect()
}

/// Compute error metrics for a DAE solution
#[allow(dead_code)]
fn compute_error_metrics<F>(
    result: &scirs2_integrate::DAEResult<f64>,
    g: f64,
    length: f64,
    theta0: f64,
    g_constraint: F,
) -> (f64, f64)
where
    F: Fn(f64, ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64>,
{
    let omega = (g / length).sqrt();
    let mut max_error = 0.0;
    let mut max_violation = 0.0;

    for i in 0..result.t.len() {
        let t = result.t[i];
        let x = result.x[i][0];
        let y = result.x[i][1];

        // Calculate analytical solution at this time
        let theta_analytical = theta0 * (omega * t).cos();
        let x_analytical = length * theta_analytical.sin();
        let y_analytical = -length * theta_analytical.cos();

        // Error is the Euclidean distance between numerical and analytical solution
        let error_at_t = ((x - x_analytical).powi(2) + (y - y_analytical).powi(2)).sqrt();
        max_error = max_error.max(error_at_t);

        // Constraint violation from the solver's perspective
        let violation = g_constraint(t, result.x[i].view(), result.y[i].view())[0].abs();
        max_violation = max_violation.max(violation);
    }

    (max_error, max_violation)
}
