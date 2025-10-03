//! BDF solver for a higher-index DAE system
//!
//! This example demonstrates solving a higher-index DAE system (index-2)
//! using specialized BDF methods with index reduction techniques.
//!
//! The system is a constrained mechanical system (slider-crank mechanism)
//! which naturally forms an index-2 DAE.

use plotters::prelude::*;
use scirs2_core::ndarray::{array, ArrayView1};
use scirs2_core::numeric::Float;
use scirs2_integrate::{
    bdf_semi_explicit_dae, DAEIndex, DAEOptions, DAEStructure, DAEType, ProjectionMethod,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Constants for the slider-crank mechanism
    let m1 = 1.0; // Mass of crank (kg)
    let m2 = 2.0; // Mass of connecting rod (kg)
    let m3 = 1.5; // Mass of slider (kg)
    let l1 = 0.5; // Length of crank (m)
    let l2 = 1.0; // Length of connecting rod (m)
    let g = 9.81; // Gravity (m/s²)

    println!("Slider-crank mechanism parameters:");
    println!("  Crank mass: {m1} kg, length: {l1} m");
    println!("  Connecting rod mass: {m2} kg, length: {l2} m");
    println!("  Slider mass: {m3} kg");

    // Initial configuration (crank at 45°, with angular velocity)
    let theta1_0 = std::f64::consts::PI / 4.0; // Crank angle (rad)
    let omega1_0 = 1.0; // Crank angular velocity (rad/s)

    // Initial positions
    let x1_0 = l1 * theta1_0.cos(); // Crank pivot position (x)
    let y1_0 = l1 * theta1_0.sin(); // Crank pivot position (y)

    // Compute connecting rod position based on constraints
    let x2_0 = x1_0 + l2 * (x1_0 / (x1_0.powi(2) + y1_0.powi(2)).sqrt());
    let y2_0 = y1_0 + l2 * (y1_0 / (x1_0.powi(2) + y1_0.powi(2)).sqrt());

    // Initial velocities
    let vx1_0 = -l1 * theta1_0.sin() * omega1_0;
    let vy1_0 = l1 * theta1_0.cos() * omega1_0;

    // Compute connecting rod velocity based on constraints
    let vx2_0 = vx1_0; // Simplified for this example
    let vy2_0 = 0.0; // Slider moves horizontally only

    // Differential variables: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    let diff_vars_0 = array![x1_0, y1_0, x2_0, y2_0, vx1_0, vy1_0, vx2_0, vy2_0];

    // Initial Lagrange multipliers (algebraic variables) - we'll start with zeros
    // and let the solver find the correct values
    let lambda1_0 = 0.0; // For fixed length constraint of crank
    let lambda2_0 = 0.0; // For fixed length constraint of connecting rod
    let lambda3_0 = 0.0; // For horizontal constraint of slider
    let alg_vars_0 = array![lambda1_0, lambda2_0, lambda3_0];

    // Time span for the simulation
    let t_span = [0.0, 5.0];

    // Define the system of differential equations
    let f = |_t: f64, q: ArrayView1<f64>, lambda: ArrayView1<f64>| {
        let x1 = q[0];
        let y1 = q[1];
        let x2 = q[2];
        let y2 = q[3];
        let vx1 = q[4];
        let vy1 = q[5];
        let vx2 = q[6];
        let vy2 = q[7];

        let lambda1 = lambda[0];
        let lambda2 = lambda[1];
        let lambda3 = lambda[2];

        // The mass matrix (diagonal)
        let m_inv1 = 1.0 / m1;
        let m_inv2 = 1.0 / m2;
        let _m_inv3 = 1.0 / m3;

        // Constraint gradients
        // Constraint 1: (x1)² + (y1)² - l1² = 0 (crank length)
        let grad_c1_x1 = 2.0 * x1;
        let grad_c1_y1 = 2.0 * y1;

        // Constraint 2: (x2 - x1)² + (y2 - y1)² - l2² = 0 (rod length)
        let grad_c2_x1 = -2.0 * (x2 - x1);
        let grad_c2_y1 = -2.0 * (y2 - y1);
        let grad_c2_x2 = 2.0 * (x2 - x1);
        let grad_c2_y2 = 2.0 * (y2 - y1);

        // Constraint 3: y2 = 0 (horizontal slider constraint)
        let grad_c3_y2 = 1.0;

        // Accelerations (with gravity and constraint forces)
        let ax1 = m_inv1 * (-lambda1 * grad_c1_x1 - lambda2 * grad_c2_x1);
        let ay1 = m_inv1 * (-lambda1 * grad_c1_y1 - lambda2 * grad_c2_y1 - m1 * g);
        let ax2 = m_inv2 * (-lambda2 * grad_c2_x2);
        let ay2 = m_inv2 * (-lambda2 * grad_c2_y2 - lambda3 * grad_c3_y2 - m2 * g);

        // Return the derivatives
        // x' = v, v' = a
        array![
            vx1, vy1, vx2, vy2, // Positions' derivatives = velocities
            ax1, ay1, ax2, ay2 // Velocities' derivatives = accelerations
        ]
    };

    // Define the constraint equations
    let g_constraint = |_t: f64, q: ArrayView1<f64>, _lambda: ArrayView1<f64>| {
        let x1 = q[0];
        let y1 = q[1];
        let x2 = q[2];
        let y2 = q[3];

        // Constraint 1: Fixed length of crank
        let c1 = x1 * x1 + y1 * y1 - l1 * l1;

        // Constraint 2: Fixed length of connecting rod
        let c2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) - l2 * l2;

        // Constraint 3: Slider moves horizontally
        let c3 = y2;

        array![c1, c2, c3]
    };

    // Set solver options
    let options = DAEOptions {
        dae_type: DAEType::SemiExplicit,
        index: DAEIndex::Index2, // This is a higher-index DAE
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 10_000,
        max_newton_iterations: 20,
        newton_tol: 1e-8,
        h0: Some(0.01),
        min_step: Some(1e-10),
        max_step: Some(0.1),
        max_order: Some(5),
        ..Default::default()
    };

    // Create a DAE structure for analyzing the system
    let dae_structure: DAEStructure<f64> =
        DAEStructure::new_semi_explicit(diff_vars_0.len(), alg_vars_0.len());

    // Create a projection method for constraint stabilization
    let _projection = ProjectionMethod::new(dae_structure);

    println!("Solving the slider-crank mechanism DAE with BDF method...");

    // Solve the DAE system using specialized BDF method
    let result = bdf_semi_explicit_dae(f, g_constraint, t_span, diff_vars_0, alg_vars_0, options)?;

    // Print performance statistics
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

    // Check constraint satisfaction
    let mut max_constraint_violation = 0.0;
    for i in 0..result.t.len() {
        let constraints = g_constraint(result.t[i], result.x[i].view(), result.y[i].view());
        let violation = constraints
            .iter()
            .map(|&c| c.abs())
            .fold(0.0, |a, b| a.max(b));
        max_constraint_violation = max_constraint_violation.max(violation);
    }

    println!("Maximum constraint violation: {max_constraint_violation:.6e}");

    // Calculate the energy at each time point
    let mut energy = Vec::new();
    for i in 0..result.t.len() {
        let _x1 = result.x[i][0];
        let y1 = result.x[i][1];
        let _x2 = result.x[i][2];
        let y2 = result.x[i][3];
        let vx1 = result.x[i][4];
        let vy1 = result.x[i][5];
        let vx2 = result.x[i][6];
        let vy2 = result.x[i][7];

        // Kinetic energy
        let ke1 = 0.5 * m1 * (vx1 * vx1 + vy1 * vy1);
        let ke2 = 0.5 * m2 * (vx2 * vx2 + vy2 * vy2);

        // Potential energy
        let pe1 = m1 * g * y1;
        let pe2 = m2 * g * y2;

        // Total energy
        let total = ke1 + ke2 + pe1 + pe2;
        energy.push(total);
    }

    // Create a plot of the crank and connecting rod motion
    let root =
        BitMapBackend::new("slider_crank_trajectory_bdf.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_x = result
        .x
        .iter()
        .map(|q| q[0].min(q[2]))
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_x = result
        .x
        .iter()
        .map(|q| q[0].max(q[2]))
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let min_y = result
        .x
        .iter()
        .map(|q| q[1].min(q[3]))
        .fold(f64::INFINITY, |a, b| a.min(b));
    let max_y = result
        .x
        .iter()
        .map(|q| q[1].max(q[3]))
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let x_margin = (max_x - min_x) * 0.1;
    let y_margin = (max_y - min_y) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Slider-Crank Mechanism Motion", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (min_x - x_margin)..(max_x + x_margin),
            (min_y - y_margin)..(max_y + y_margin),
        )?;

    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    // Sample some frames for visualization (e.g., 20 frames)
    let n_frames = 20;
    let frame_indices: Vec<usize> = (0..n_frames)
        .map(|i| i * result.t.len() / n_frames)
        .collect();

    // Draw the mechanism at different time points
    for &idx in &frame_indices {
        let x1 = result.x[idx][0];
        let y1 = result.x[idx][1];
        let x2 = result.x[idx][2];
        let y2 = result.x[idx][3];

        // Origin to crank pivot
        chart.draw_series(LineSeries::new(vec![(0.0, 0.0), (x1, y1)], &RED.mix(0.5)))?;

        // Crank pivot to connecting rod end
        chart.draw_series(LineSeries::new(vec![(x1, y1), (x2, y2)], &BLUE.mix(0.5)))?;

        // Draw points at joints
        chart.draw_series(PointSeries::of_element(
            vec![(0.0, 0.0), (x1, y1), (x2, y2)],
            5,
            &BLACK,
            &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
        ))?;
    }

    // Plot the crank angle over time
    let root = BitMapBackend::new("slider_crank_angle_bdf.png", (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Crank Angle vs Time", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..t_span[1], -std::f64::consts::PI..std::f64::consts::PI)?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .x_desc("Time (s)")
        .y_desc("Crank Angle (rad)")
        .draw()?;

    // Calculate crank angle from positions
    let crank_angles: Vec<(f64, f64)> = result
        .t
        .iter()
        .zip(result.x.iter())
        .map(|(&t, q)| {
            let x1 = q[0];
            let y1 = q[1];
            let angle = y1.atan2(x1);
            (t, angle)
        })
        .collect();

    // Draw the crank angle plot
    chart.draw_series(LineSeries::new(crank_angles, &GREEN))?;

    // Plot the energy over time
    let root = BitMapBackend::new("slider_crank_energy_bdf.png", (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_energy = energy.iter().cloned().fold(0.0, |a, b| a.max(b)) * 1.1;
    let min_energy = energy.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b)) * 0.9;

    let mut chart = ChartBuilder::on(&root)
        .caption("System Energy", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..t_span[1], min_energy..max_energy)?;

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

    println!("Plots generated: slider_crank_trajectory_bdf.png, slider_crank_angle_bdf.png, and slider_crank_energy_bdf.png");

    Ok(())
}
