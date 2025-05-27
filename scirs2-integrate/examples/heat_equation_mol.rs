use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, MOLOptions,
    MOLParabolicSolver1D, ODEMethod,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Method of Lines example: Heat equation solver");
    println!("Solving: ∂u/∂t = α ∂²u/∂x²");
    println!("Domain: x ∈ [0, 1], t ∈ [0, 0.5]");
    println!("Boundary conditions: u(0, t) = 0, u(1, t) = 0");
    println!("Initial condition: u(x, 0) = sin(πx)");
    println!("Exact solution: u(x, t) = sin(πx) * exp(-π² * α * t)");

    // Physical parameters
    let alpha = 0.01; // Thermal diffusivity

    // Set up spatial domain
    let nx = 101; // Number of spatial grid points
    let domain = Domain::new(
        vec![0.0..1.0], // Spatial range [0, 1]
        vec![nx],       // Number of grid points
    )?;

    // Time domain
    let t_start = 0.0;
    let t_end = 0.5;
    let time_range = [t_start, t_end];

    // Diffusion coefficient (constant)
    let diffusion_coeff = move |_x: f64, _t: f64, _u: f64| -> f64 { alpha };

    // Initial condition: sin(πx)
    let initial_condition = |x: f64| -> f64 { (PI * x).sin() };

    // Boundary conditions: u(0, t) = 0, u(1, t) = 0 (Dirichlet)
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
    let mol_solver = MOLParabolicSolver1D::new(
        domain,
        time_range,
        diffusion_coeff,
        initial_condition,
        bcs,
        Some(options),
    )?;

    // Solve the PDE
    println!("Solving the heat equation...");
    let result = mol_solver.solve()?;

    // Extract results
    let t = &result.t;
    let u = &result.u[0]; // Solution values

    println!(
        "Solution computed in {:.4} seconds",
        result.computation_time
    );
    println!("ODE solver info: {:?}", result.ode_info);

    // Print solution at selected time points
    let nt = t.len();
    let time_indices = [0, nt / 4, nt / 2, 3 * nt / 4, nt - 1];
    let dx = 1.0 / (nx as f64 - 1.0);

    println!("\nSolution at selected time points:");
    println!(
        "{:<10} {:<10} {:<15} {:<15} {:<10}",
        "Time", "x", "Numerical", "Exact", "Error"
    );

    for &ti in &time_indices {
        let time = t[ti];

        // Print solution at x = 0.0, 0.25, 0.5, 0.75, 1.0
        for xi in [0, nx / 4, nx / 2, 3 * nx / 4, nx - 1] {
            let x = xi as f64 * dx;
            let numerical = u[[ti, xi]];

            // Exact solution: sin(πx) * exp(-π² * α * t)
            let exact = (PI * x).sin() * (-PI * PI * alpha * time).exp();
            let error = (numerical - exact).abs();

            println!(
                "{:<10.4} {:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
                time, x, numerical, exact, error
            );
        }
        println!(); // Empty line between time points
    }

    // Plot spatial profile at final time (if plotting library available)
    println!("\nFinal solution profile (t = {:.4}):", t[nt - 1]);
    println!(
        "{:<10} {:<15} {:<15} {:<10}",
        "x", "Numerical", "Exact", "Error"
    );

    let final_time = t[nt - 1];
    for i in 0..nx {
        if i % 10 == 0 || i == nx - 1 {
            // Print every 10th point
            let x = i as f64 * dx;
            let numerical = u[[nt - 1, i]];

            // Exact solution
            let exact = (PI * x).sin() * (-PI * PI * alpha * final_time).exp();
            let error = (numerical - exact).abs();

            println!(
                "{:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
                x, numerical, exact, error
            );
        }
    }

    // Calculate and print maximum error
    let mut max_error: f64 = 0.0;
    for i in 0..nx {
        let x = i as f64 * dx;
        let numerical = u[[nt - 1, i]];
        let exact = (PI * x).sin() * (-PI * PI * alpha * final_time).exp();
        let error = (numerical - exact).abs();
        max_error = max_error.max(error);
    }

    println!("\nMaximum error at final time: {:.2e}", max_error);

    Ok(())
}
