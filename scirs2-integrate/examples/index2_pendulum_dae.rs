#![allow(dead_code)]

use scirs2_core::ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::dae::{
    solve_higher_index_dae, DAEIndex, DAEOptions, DAEStructure, ProjectionMethod,
};
use scirs2_integrate::ode::ODEMethod;
use std::f64::consts::PI;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Index-2 Pendulum DAE Example");
    println!("============================\n");

    // Physical parameters
    let g = 9.81; // Gravity constant (m/s²)
    let l = 1.0; // Pendulum length (m)

    // Time span: 0 to 10 seconds
    let t_span = [0.0, 10.0];

    // Initial conditions - the pendulum starts at a 45-degree angle
    let theta0 = PI / 4.0;

    // Position variables
    let x0 = l * theta0.sin(); // x coordinate
    let y0 = -l * theta0.cos(); // y coordinate (negative because y-axis points down)

    // Velocity variables - starting from rest
    let vx0 = 0.0;
    let vy0 = 0.0;

    // Unlike the index-1 case, we don't need to compute initial Lagrange multipliers
    // The constraint will be enforced by the index reduction process

    // Differential variables: [x, y, vx, vy]
    let x_vars = array![x0, y0, vx0, vy0];

    // Algebraic variables (none for this formulation)
    let y_vars = array![];

    // Differential equations for the pendulum system
    // x' = vx
    // y' = vy
    // vx' = -λx (λ is implicitly determined)
    // vy' = -λy - g
    let f = |_t: f64, x: ArrayView1<f64>, _y: ArrayView1<f64>| -> Array1<f64> {
        let (px, py, vx, vy) = (x[0], x[1], x[2], x[3]);

        // For the acceleration, we need the Lagrange multiplier λ
        // Instead of explicitly computing it, we'll let the constraint solver handle it
        // This makes it an index-2 problem

        array![
            vx,  // x' = vx
            vy,  // y' = vy
            0.0, // vx' = -λx (implicitly determined)
            -g   // vy' = -λy - g (implicitly determined)
        ]
    };

    // Constraint function: x² + y² = l²
    // This is a holonomic constraint representing the pendulum rod length
    let g_constraint = |_t: f64, x: ArrayView1<f64>, _y: ArrayView1<f64>| -> Array1<f64> {
        array![x[0] * x[0] + x[1] * x[1] - l * l]
    };

    // We also need to enforce the hidden constraint (the time derivative of the constraint)
    // This is what makes it an index-2 problem:
    // d/dt(x² + y² = l²), gives: 2*x*vx + 2*y*vy = 0

    // We could explicitly include this constraint, but instead we'll let the
    // index reduction technique handle it automatically.

    // Setup DAE options
    let options = DAEOptions {
        method: ODEMethod::Radau, // Implicit method handles the constraint better
        rtol: 1e-6,
        atol: 1e-8,
        max_steps: 1000,
        index: DAEIndex::Index2, // Explicitly specify this is an index-2 problem
        ..Default::default()
    };

    // Create a DAE structure to analyze the system
    let mut structure = DAEStructure::new_semi_explicit(4, 0);

    // Print the detected index (should be 2)
    let detected_index = structure
        .compute_index(t_span[0], x_vars.view(), y_vars.view(), &f, &g_constraint)
        .unwrap_or(DAEIndex::Index1);

    println!("Detected DAE index: {detected_index:?}");

    // Create a projection method for the constraint
    let projection = ProjectionMethod::new(structure);

    // Make the initial conditions consistent with the constraint
    let mut x0_copy = x_vars.clone();
    let mut y0_copy = y_vars.clone();

    println!("\nInitial conditions:");
    println!("Position: ({:.6}, {:.6})", x0_copy[0], x0_copy[1]);
    println!("Velocity: ({:.6}, {:.6})", x0_copy[2], x0_copy[3]);

    // Constraint check
    let constraint_value = g_constraint(t_span[0], x0_copy.view(), y0_copy.view())[0];
    println!("Initial constraint error: {:.3e}", constraint_value.abs());

    // Try to make consistent
    let consistency_result =
        projection.make_consistent(t_span[0], &mut x0_copy, &mut y0_copy, &g_constraint);

    if let Err(e) = consistency_result {
        println!("Warning: Failed to make initial conditions fully consistent: {e}");
    } else {
        println!("Initial conditions made consistent.");
    }

    // Solve the DAE system
    println!("\nSolving index-2 DAE system...");
    let result = solve_higher_index_dae(f, g_constraint, t_span, x0_copy, y0_copy, Some(options))?;

    println!(
        "Solution completed with {} steps ({} accepted, {} rejected).\n",
        result.n_steps, result.n_accepted, result.n_rejected
    );

    // Print headers
    println!(
        "{:<10} {:<12} {:<12} {:<12} {:<12} {:<15} {:<12}",
        "Time", "X", "Y", "VX", "VY", "Constraint", "Energy"
    );
    println!("{:-<85}", "");

    // Calculate total energy at each step and check constraint satisfaction
    let num_print = 10.min(result.t.len());

    // Function to calculate energy
    let energy = |_x: f64, y: f64, vx: f64, vy: f64| -> f64 {
        // Kinetic energy: 0.5 * m * (vx² + vy²)
        let kinetic = 0.5 * (vx * vx + vy * vy);

        // Potential energy: m * g * h, h is height (y + l)
        // Since our y is negative, we use -y for height
        let potential = g * (-y);

        kinetic + potential
    };

    // Initial energy for comparison
    let initial_energy = energy(x0, y0, vx0, vy0);

    // Print first few points
    for i in 0..num_print {
        let t = result.t[i];
        let x = result.x[i][0];
        let y = result.x[i][1];
        let vx = result.x[i][2];
        let vy = result.x[i][3];

        // Check constraint satisfaction
        let constraint_value = x * x + y * y - l * l;

        // Calculate total energy and energy error
        let e = energy(x, y, vx, vy);
        let e_rel_error = (e - initial_energy).abs() / initial_energy.abs();

        println!(
            "{t:<10.3} {x:<12.6} {y:<12.6} {vx:<12.6} {vy:<12.6} {constraint_value:<15.3e} {e_rel_error:<12.3e}"
        );
    }

    if result.t.len() > 2 * num_print {
        println!("{:^85}", "...");
    }

    // Print last few points
    if result.t.len() > num_print {
        for i in (result.t.len() - num_print)..result.t.len() {
            let t = result.t[i];
            let x = result.x[i][0];
            let y = result.x[i][1];
            let vx = result.x[i][2];
            let vy = result.x[i][3];

            // Check constraint satisfaction
            let constraint_value = x * x + y * y - l * l;

            // Calculate total energy and energy error
            let e = energy(x, y, vx, vy);
            let e_rel_error = (e - initial_energy).abs() / initial_energy.abs();

            println!(
                "{t:<10.3} {x:<12.6} {y:<12.6} {vx:<12.6} {vy:<12.6} {constraint_value:<15.3e} {e_rel_error:<12.3e}"
            );
        }
    }

    // Analyze the solution
    println!("\nSolution Analysis:");

    // Calculate the period from the solution
    let mut crossings = Vec::new();

    // Detect x-axis crossings (when y = -l)
    for i in 1..result.t.len() {
        if (result.x[i - 1][1] + l) * (result.x[i][1] + l) <= 0.0 && result.x[i][0] > 0.0 {
            // Linear interpolation to find the crossing time
            let t1 = result.t[i - 1];
            let t2 = result.t[i];
            let y1 = result.x[i - 1][1] + l;
            let y2 = result.x[i][1] + l;

            let t_cross = t1 + (t2 - t1) * y1.abs() / (y1.abs() + y2.abs());
            crossings.push(t_cross);
        }
    }

    if crossings.len() >= 2 {
        // Calculate periods
        let mut periods = Vec::new();
        for i in 1..crossings.len() {
            periods.push(crossings[i] - crossings[i - 1]);
        }

        // Calculate average period
        let avg_period = periods.iter().sum::<f64>() / periods.len() as f64;

        // Calculate theoretical period
        let theoretical_period = 2.0 * PI * (l / g).sqrt();

        println!("Detected {} complete oscillations", periods.len());
        println!("Average period: {avg_period:.6} seconds");
        println!("Theoretical period for small oscillations: {theoretical_period:.6} seconds");
        println!(
            "Difference: {:.3}%",
            100.0 * (avg_period - theoretical_period).abs() / theoretical_period
        );

        if periods.len() >= 2 {
            println!("First period: {:.6} seconds", periods[0]);
            println!("Last period: {:.6} seconds", periods[periods.len() - 1]);
        }
    } else {
        println!("Not enough oscillations detected to calculate period.");
    }

    // Analyze constraint violation
    let constraint_errors = result
        .x
        .iter()
        .map(|x| (x[0] * x[0] + x[1] * x[1] - l * l).abs())
        .collect::<Vec<_>>();
    let max_constraint_error = constraint_errors
        .iter()
        .fold(0.0_f64, |max, &err| max.max(err));
    let avg_constraint_error =
        constraint_errors.iter().sum::<f64>() / constraint_errors.len() as f64;

    println!("\nConstraint Analysis:");
    println!("Maximum constraint violation: {max_constraint_error:.3e}");
    println!("Average constraint violation: {avg_constraint_error:.3e}");

    // Analyze energy conservation
    let energies = result
        .x
        .iter()
        .map(|x| energy(x[0], x[1], x[2], x[3]))
        .collect::<Vec<_>>();
    let max_energy_error = energies.iter().fold(0.0_f64, |max, &e| {
        max.max((e - initial_energy).abs() / initial_energy.abs())
    });
    let avg_energy_error = energies
        .iter()
        .map(|e| (e - initial_energy).abs() / initial_energy.abs())
        .sum::<f64>()
        / energies.len() as f64;

    println!("\nEnergy Conservation Analysis:");
    println!("Maximum relative energy error: {max_energy_error:.3e}");
    println!("Average relative energy error: {avg_energy_error:.3e}");

    Ok(())
}
