use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, MOLOptions,
    MOLWaveEquation1D, ODEMethod,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Method of Lines example: 1D Wave equation solver");
    println!("Solving: ∂²u/∂t² = c² ∂²u/∂x²");
    println!("Domain: x ∈ [0, 1], t ∈ [0, 2.0]");
    println!("Boundary conditions: u(0, t) = u(1, t) = 0");
    println!("Initial condition: u(x, 0) = sin(πx)");
    println!("Initial velocity: ∂u/∂t(x, 0) = 0");
    println!("Exact solution: u(x, t) = sin(πx) * cos(πct)");

    // Physical parameters
    let wave_speed = 1.0; // Wave speed c

    // Set up spatial domain
    let nx = 101; // Number of spatial grid points
    let domain = Domain::new(
        vec![0.0..1.0], // Spatial range [0, 1]
        vec![nx],       // Number of grid points
    )?;

    // Time domain
    let t_start = 0.0;
    let t_end = 2.0;
    let time_range = [t_start, t_end];

    // Wave speed squared function (constant)
    let wave_speed_squared = move |_x: f64, _t: f64, _u: f64| -> f64 { wave_speed * wave_speed };

    // Initial condition: sin(πx)
    let initial_condition = |x: f64| -> f64 { (PI * x).sin() };

    // Initial velocity: 0 (starting from rest)
    let initial_velocity = |_x: f64| -> f64 { 0.0 };

    // Boundary conditions: u(0, t) = u(1, t) = 0 (Dirichlet)
    let bcs = vec![
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
    ];

    // Solver options
    let options = MOLOptions {
        ode_method: ODEMethod::RK45,
        atol: 1e-6,
        rtol: 1e-3,
        max_steps: Some(10000),
        verbose: true,
    };

    // Create the MOL solver
    let mol_solver = MOLWaveEquation1D::new(
        domain,
        time_range,
        wave_speed_squared,
        initial_condition,
        initial_velocity,
        bcs,
        Some(options),
    )?;

    // Solve the PDE
    println!("Solving the wave equation...");
    let start_time = Instant::now();
    let result = mol_solver.solve()?;
    let solve_time = start_time.elapsed().as_secs_f64();

    // Extract results
    let t = &result.t;
    let u = &result.u; // Solution values [time, space]
    let u_t = &result.u_t; // Velocity values [time, space]

    println!("Solution computed in {:.4} seconds", solve_time);
    println!("ODE solver info: {:?}", result.ode_info);

    // Print solution at selected time points
    let nt = t.len();
    let time_indices = [
        0,
        nt / 8,
        nt / 4,
        3 * nt / 8,
        nt / 2,
        5 * nt / 8,
        3 * nt / 4,
        7 * nt / 8,
        nt - 1,
    ];
    let dx = 1.0 / (nx as f64 - 1.0);

    println!("\nSolution at selected time points:");
    println!(
        "{:<10} {:<10} {:<15} {:<15} {:<10}",
        "Time", "x", "Numerical", "Exact", "Error"
    );

    for &ti in &time_indices {
        let time = t[ti];

        // Print solution at x = 0.25, 0.5, 0.75
        for xi in [nx / 4, nx / 2, 3 * nx / 4] {
            let x = xi as f64 * dx;
            let numerical = u[[ti, xi]];

            // Exact solution: sin(πx) * cos(πct)
            let exact = (PI * x).sin() * (PI * wave_speed * time).cos();
            let error = (numerical - exact).abs();

            println!(
                "{:<10.4} {:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
                time, x, numerical, exact, error
            );
        }
        println!(); // Empty line between time points
    }

    // Calculate maximum error
    let mut max_error = 0.0;
    let mut max_error_time = 0.0;
    let mut max_error_pos = 0.0;

    for ti in 0..nt {
        let time = t[ti];
        for xi in 0..nx {
            let x = xi as f64 * dx;
            let numerical = u[[ti, xi]];
            let exact = (PI * x).sin() * (PI * wave_speed * time).cos();
            let error = (numerical - exact).abs();

            if error > max_error {
                max_error = error;
                max_error_time = time;
                max_error_pos = x;
            }
        }
    }

    println!(
        "Maximum error: {:.2e} at t = {:.4}, x = {:.4}",
        max_error, max_error_time, max_error_pos
    );

    // Analyze energy conservation
    println!("\nEnergy conservation analysis:");

    for &ti in &time_indices {
        let time = t[ti];

        // Calculate total energy = 1/2 ∫(u_t² + c²u_x²)dx
        // Using trapezoidal rule for the integral
        let mut kinetic_energy = 0.0; // 1/2 ∫u_t²dx
        let mut potential_energy = 0.0; // 1/2 ∫c²u_x²dx

        for i in 0..nx - 1 {
            // Kinetic energy term
            let ut_i = u_t[[ti, i]];
            let ut_i1 = u_t[[ti, i + 1]];
            let ke_term = 0.5 * dx * (ut_i * ut_i + ut_i1 * ut_i1);
            kinetic_energy += ke_term;

            // Potential energy term (using finite difference for u_x)
            let x_i = i as f64 * dx;
            let x_i1 = (i + 1) as f64 * dx;
            let ux_i = (u[[ti, i + 1]] - u[[ti, i]]) / dx;
            let ux_i1 = if i < nx - 2 {
                (u[[ti, i + 2]] - u[[ti, i + 1]]) / dx
            } else {
                ux_i // Use same value at boundary
            };

            let c2_i = (wave_speed_squared)(x_i, time, u[[ti, i]]);
            let c2_i1 = (wave_speed_squared)(x_i1, time, u[[ti, i + 1]]);

            let pe_term = 0.5 * dx * (c2_i * ux_i * ux_i + c2_i1 * ux_i1 * ux_i1);
            potential_energy += pe_term;
        }

        let total_energy = kinetic_energy + potential_energy;

        // For t=0, store the initial energy
        if ti == 0 {
            println!("Initial energy: {:.8}", total_energy);
            println!("  - Kinetic energy: {:.8}", kinetic_energy);
            println!("  - Potential energy: {:.8}", potential_energy);
            println!("\nTime      Total Energy  Relative Error");
        }

        // Calculate relative error in energy conservation
        let initial_energy = (PI * PI) / 4.0; // Analytical value for sin(πx)
        let energy_error = (total_energy - initial_energy).abs() / initial_energy;

        println!(
            "{:<10.4} {:<15.8e} {:<15.2e}",
            time, total_energy, energy_error
        );
    }

    // Analyze wave propagation
    println!("\nWave propagation analysis:");
    println!("Time      Amplitude at x=0.5");

    let midpoint_index = nx / 2;
    for &ti in &time_indices {
        let time = t[ti];
        let amplitude = u[[ti, midpoint_index]];
        let exact_amplitude = (PI * 0.5).sin() * (PI * wave_speed * time).cos();

        println!(
            "{:<10.4} {:<15.8e} (exact: {:<15.8e})",
            time, amplitude, exact_amplitude
        );
    }

    Ok(())
}
