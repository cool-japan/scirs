//! Large DAE System with Krylov-Enhanced BDF Solver
//!
//! This example demonstrates the use of Krylov-enhanced BDF methods for
//! solving a large sparse differential algebraic equation (DAE) system.
//! The example models a 2D heat equation on a grid with constraints.

use ndarray::{Array1, ArrayView1};
// use plotters::prelude::*;
use scirs2_integrate::{
    bdf_semi_explicit_dae, krylov_bdf_semi_explicit_dae, DAEIndex, DAEOptions, DAEType, ODEMethod,
};
use std::f64::consts::PI;
use std::time::Instant;

const N: usize = 30; // Grid size (NxN)
const TOTAL_VARS: usize = N * N; // Total number of variables

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Solving a large sparse DAE system ({} variables) using Krylov-enhanced BDF methods",
        TOTAL_VARS
    );

    // Grid spacing
    let _h = 1.0 / (N as f64 - 1.0);

    // Initial conditions (temperature distribution)
    let x0 = setup_initial_conditions();

    // Zero initial algebraic variables (Lagrange multipliers for constraints)
    let n_constraints = N / 5; // Every 5th point is constrained
    let y0 = Array1::zeros(n_constraints);

    println!(
        "System size: {} differential variables, {} algebraic constraints",
        TOTAL_VARS, n_constraints
    );

    // Time span
    let t_span = [0.0, 0.1]; // Shorter time for this example

    // Common solver options
    let options = DAEOptions {
        dae_type: DAEType::SemiExplicit,
        index: DAEIndex::Index1,
        rtol: 1e-4,
        atol: 1e-6,
        max_steps: 1000,
        max_newton_iterations: 10,
        newton_tol: 1e-5,
        h0: Some(0.001),
        method: ODEMethod::Bdf,
        min_step: Some(1e-6),
        max_step: Some(0.01),
        max_order: Some(2), // Limit order for this example
        ..Default::default()
    };

    // Solve using standard BDF method
    println!("Solving with standard BDF method...");
    let start_time = Instant::now();
    let result_bdf = bdf_semi_explicit_dae(
        heat_equation_rhs,
        constraint_equation,
        t_span,
        x0.clone(),
        y0.clone(),
        options.clone(),
    )?;
    let elapsed_bdf = start_time.elapsed();

    // Solve using Krylov-enhanced BDF method
    println!("Solving with Krylov-enhanced BDF method...");
    let start_time = Instant::now();
    let result_krylov = krylov_bdf_semi_explicit_dae(
        heat_equation_rhs,
        constraint_equation,
        t_span,
        x0.clone(),
        y0.clone(),
        options,
    )?;
    let elapsed_krylov = start_time.elapsed();

    // Print performance statistics
    println!("\nPerformance Comparison:");
    println!(
        "{:<20} {:<15} {:<10} {:<15} {:<15}",
        "Method", "Time (s)", "Steps", "Function Evals", "LU/GMRES Iters"
    );
    println!("{:-<75}", "");

    println!(
        "{:<20} {:<15.6} {:<10} {:<15} {:<15}",
        "Standard BDF",
        elapsed_bdf.as_secs_f64(),
        result_bdf.n_steps,
        result_bdf.n_eval,
        result_bdf.n_lu
    );

    println!(
        "{:<20} {:<15.6} {:<10} {:<15} {:<15}",
        "Krylov BDF",
        elapsed_krylov.as_secs_f64(),
        result_krylov.n_steps,
        result_krylov.n_eval,
        result_krylov
            .message
            .as_ref()
            .unwrap()
            .split(',')
            .nth(1)
            .unwrap_or("")
            .trim()
            .split(' ')
            .next()
            .unwrap_or("0")
    );

    // Calculate the speedup
    let speedup = elapsed_bdf.as_secs_f64() / elapsed_krylov.as_secs_f64();
    println!("\nSpeedup: {:.2}x", speedup);

    // Visualize results at the final time point
    // create_solution_plot(&result_bdf, "heat_equation_bdf.png", "Standard BDF")?;
    // create_solution_plot(&result_krylov, "heat_equation_krylov.png", "Krylov BDF")?;

    // println!("\nPlots saved as heat_equation_bdf.png and heat_equation_krylov.png");

    Ok(())
}

/// Initial condition setup
fn setup_initial_conditions() -> Array1<f64> {
    let mut x0 = Array1::zeros(TOTAL_VARS);

    // Gaussian heat source in the center
    let center_x = (N as f64) / 2.0;
    let center_y = (N as f64) / 2.0;
    let sigma = (N as f64) / 10.0;

    for i in 0..N {
        for j in 0..N {
            let idx = i * N + j;
            let dist_squared = ((i as f64) - center_x).powi(2) + ((j as f64) - center_y).powi(2);
            x0[idx] = (-dist_squared / (2.0 * sigma * sigma)).exp();
        }
    }

    x0
}

/// Right-hand side of the heat equation (semi-discretized using finite differences)
/// u_t = k * (u_xx + u_yy) + f(x,y,t)
fn heat_equation_rhs(t: f64, u: ArrayView1<f64>, lambda: ArrayView1<f64>) -> Array1<f64> {
    let mut du_dt = Array1::zeros(TOTAL_VARS);
    let h = 1.0 / (N as f64 - 1.0);
    let k = 1.0; // Thermal diffusivity

    // Heat equation discretization using finite differences
    for i in 0..N {
        for j in 0..N {
            let idx = i * N + j;

            if i == 0 || i == N - 1 || j == 0 || j == N - 1 {
                // Boundary condition: u = 0
                du_dt[idx] = 0.0;
            } else {
                // Interior points
                let u_center = u[idx];
                let u_left = u[i * N + (j - 1)];
                let u_right = u[i * N + (j + 1)];
                let u_up = u[(i - 1) * N + j];
                let u_down = u[(i + 1) * N + j];

                // Laplacian: (u_left + u_right + u_up + u_down - 4 * u_center) / h²
                let laplacian = (u_left + u_right + u_up + u_down - 4.0 * u_center) / (h * h);

                // Source term: sin(2πx) * sin(2πy) * sin(t)
                let x = j as f64 * h;
                let y = i as f64 * h;
                let source = (2.0 * PI * x).sin() * (2.0 * PI * y).sin() * t.sin();

                // Heat equation: u_t = k * ∇²u + f
                du_dt[idx] = k * laplacian + source;
            }
        }
    }

    // Apply constraints - every 5th point is constrained to remain at its initial value
    // This is done by adding constraint forces via Lagrange multipliers
    let n_constraints = lambda.len();
    let mut constraint_idx = 0;

    for i in 0..N {
        for j in 0..N {
            if (i * j) % 5 == 0 && constraint_idx < n_constraints {
                let idx = i * N + j;
                du_dt[idx] += lambda[constraint_idx];
                constraint_idx += 1;
            }
        }
    }

    du_dt
}

/// Constraint equation: constrain every 5th point to remain at its initial value
fn constraint_equation(_t: f64, u: ArrayView1<f64>, _lambda: ArrayView1<f64>) -> Array1<f64> {
    let initial_conditions = setup_initial_conditions();
    let n_constraints = N / 5; // Every 5th point is constrained
    let mut constraints = Array1::zeros(n_constraints);

    let mut constraint_idx = 0;
    for i in 0..N {
        for j in 0..N {
            if (i * j) % 5 == 0 && constraint_idx < n_constraints {
                let idx = i * N + j;
                constraints[constraint_idx] = u[idx] - initial_conditions[idx];
                constraint_idx += 1;
            }
        }
    }

    constraints
}

// Create a heatmap visualization of the solution (disabled due to missing dependencies)
/*
fn create_solution_plot(
    result: &DAEResult<f64>,
    filename: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get the final solution (convert to 2D array)
    let final_solution = result.x.last().unwrap();
    let mut solution_grid = Array2::zeros((N, N));

    for i in 0..N {
        for j in 0..N {
            let idx = i * N + j;
            solution_grid[[i, j]] = final_solution[idx];
        }
    }

    // Find min and max values for color scale
    let min_val = solution_grid.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = solution_grid
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Create plot
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create heatmap
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("{} Solution at t={:.3}", title, result.t.last().unwrap()),
            ("sans-serif", 30),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .disable_mesh()
        .x_desc("x")
        .y_desc("y")
        .draw()?;

    // Create color gradient
    let color_gradient = colorous::VIRIDIS;

    // Draw the heatmap
    let cell_size = 1.0 / (N as f64);

    for i in 0..N {
        for j in 0..N {
            let value = solution_grid[[i, j]];
            let normalized_value = (value - min_val) / (max_val - min_val);
            let color = color_gradient.eval_continuous(normalized_value.max(0.0).min(1.0));

            let x0 = j as f64 * cell_size;
            let y0 = i as f64 * cell_size;
            let x1 = x0 + cell_size;
            let y1 = y0 + cell_size;

            chart.draw_series(vec![Rectangle::new(
                [(x0, y0), (x1, y1)],
                RGBColor(color.r, color.g, color.b).filled(),
            )])?;
        }
    }

    // Add color bar
    let color_bar_width = 20;
    let color_bar_height = 400;
    let color_bar_x = 700;
    let color_bar_y = 100;

    for i in 0..color_bar_height {
        let normalized_value = 1.0 - (i as f64) / (color_bar_height as f64);
        let color = color_gradient.eval_continuous(normalized_value);

        root.draw(&Rectangle::new(
            [
                (color_bar_x, color_bar_y + i),
                (color_bar_x + color_bar_width, color_bar_y + i + 1),
            ],
            RGBColor(color.r, color.g, color.b).filled(),
        ))?;
    }

    // Add color bar labels
    root.draw_text(
        &format!("{:.3}", max_val),
        &("sans-serif", 15),
        (color_bar_x + 30, color_bar_y),
    )?;

    root.draw_text(
        &format!("{:.3}", min_val),
        &("sans-serif", 15),
        (color_bar_x + 30, color_bar_y + color_bar_height),
    )?;

    Ok(())
}
*/
