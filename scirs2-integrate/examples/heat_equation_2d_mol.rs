use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, MOLOptions,
    MOLParabolicSolver2D, ODEMethod,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Method of Lines example: 2D Heat equation solver");
    println!("Solving: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)");
    println!("Domain: (x,y) ∈ [0, 1] × [0, 1], t ∈ [0, 0.1]");
    println!("Boundary conditions: u = 0 on all boundaries");
    println!("Initial condition: u(x, y, 0) = sin(πx) * sin(πy)");
    println!("Exact solution: u(x, y, t) = sin(πx) * sin(πy) * exp(-2π²αt)");

    // Physical parameters
    let alpha = 0.01; // Thermal diffusivity

    // Set up spatial domain
    let nx = 51; // Number of grid points in x direction
    let ny = 51; // Number of grid points in y direction
    let domain = Domain::new(
        vec![0.0..1.0, 0.0..1.0], // Spatial ranges [0, 1] × [0, 1]
        vec![nx, ny],             // Number of grid points
    )?;

    // Time domain
    let t_start = 0.0;
    let t_end = 0.1;
    let time_range = [t_start, t_end];

    // Diffusion coefficients (constant in both directions)
    let diffusion_x = move |_x: f64, _y: f64, _t: f64, _u: f64| -> f64 { alpha };

    let diffusion_y = move |_x: f64, _y: f64, _t: f64, _u: f64| -> f64 { alpha };

    // Initial condition: sin(πx) * sin(πy)
    let initial_condition = |x: f64, y: f64| -> f64 { (PI * x).sin() * (PI * y).sin() };

    // Boundary conditions: u = 0 on all boundaries (Dirichlet)
    let bcs = vec![
        // X-direction boundaries
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0, // x dimension
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 0, // x dimension
            value: 0.0,
            coefficients: None,
        },
        // Y-direction boundaries
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1, // y dimension
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 1, // y dimension
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
    let mol_solver = MOLParabolicSolver2D::new(
        domain,
        time_range,
        diffusion_x,
        diffusion_y,
        initial_condition,
        bcs,
        Some(options),
    )?;

    // Solve the PDE
    println!("Solving the 2D heat equation...");
    let start_time = Instant::now();
    let result = mol_solver.solve()?;
    let solve_time = start_time.elapsed().as_secs_f64();

    // Extract results
    let t = &result.t;
    let u = &result.u; // Solution values [time, y, x]

    println!("Solution computed in {:.4} seconds", solve_time);
    println!("ODE solver info: {:?}", result.ode_info);

    // Calculate maximum error at final time
    let nt = t.len();
    let final_time = t[nt - 1];

    // Create grid points for error calculation
    let dx = 1.0 / (nx as f64 - 1.0);
    let dy = 1.0 / (ny as f64 - 1.0);

    println!("\nAnalyzing solution accuracy...");
    let mut max_error: f64 = 0.0;
    let mut l2_error = 0.0;

    for j in 0..ny {
        for i in 0..nx {
            let x = i as f64 * dx;
            let y = j as f64 * dy;

            // Numerical solution at final time
            let numerical = u[[nt - 1, j, i]];

            // Exact solution: sin(πx) * sin(πy) * exp(-2π² * α * t)
            let exact =
                (PI * x).sin() * (PI * y).sin() * (-2.0 * PI * PI * alpha * final_time).exp();

            // Error
            let error = (numerical - exact).abs();
            max_error = max_error.max(error);
            l2_error += error * error;
        }
    }

    // Calculate L2 error norm
    l2_error = (l2_error * dx * dy).sqrt();

    println!(
        "Maximum error at final time (t = {:.4}): {:.2e}",
        final_time, max_error
    );
    println!("L2 error norm at final time: {:.2e}", l2_error);

    // Print solution at selected points and times
    let time_indices = [0, nt / 4, nt / 2, 3 * nt / 4, nt - 1];

    println!("\nSolution at selected points:");
    println!(
        "{:<10} {:<10} {:<10} {:<15} {:<15} {:<10}",
        "Time", "x", "y", "Numerical", "Exact", "Error"
    );

    for &ti in &time_indices {
        let time = t[ti];

        // Print solution at center of domain (x=0.5, y=0.5)
        let i = nx / 2;
        let j = ny / 2;
        let x = i as f64 * dx;
        let y = j as f64 * dy;

        let numerical = u[[ti, j, i]];
        let exact = (PI * x).sin() * (PI * y).sin() * (-2.0 * PI * PI * alpha * time).exp();
        let error = (numerical - exact).abs();

        println!(
            "{:<10.4} {:<10.4} {:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
            time, x, y, numerical, exact, error
        );
    }

    // Print solution profile along x-axis at final time (y=0.5)
    println!(
        "\nSolution profile along x-axis (y=0.5) at final time (t = {:.4}):",
        final_time
    );
    println!(
        "{:<10} {:<15} {:<15} {:<10}",
        "x", "Numerical", "Exact", "Error"
    );

    let j = ny / 2; // y=0.5
    for i in (0..nx).step_by(nx / 10) {
        let x = i as f64 * dx;
        let y = j as f64 * dy;

        let numerical = u[[nt - 1, j, i]];
        let exact = (PI * x).sin() * (PI * y).sin() * (-2.0 * PI * PI * alpha * final_time).exp();
        let error = (numerical - exact).abs();

        println!(
            "{:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
            x, numerical, exact, error
        );
    }

    // Print solution profile along y-axis at final time (x=0.5)
    println!(
        "\nSolution profile along y-axis (x=0.5) at final time (t = {:.4}):",
        final_time
    );
    println!(
        "{:<10} {:<15} {:<15} {:<10}",
        "y", "Numerical", "Exact", "Error"
    );

    let i = nx / 2; // x=0.5
    for j in (0..ny).step_by(ny / 10) {
        let x = i as f64 * dx;
        let y = j as f64 * dy;

        let numerical = u[[nt - 1, j, i]];
        let exact = (PI * x).sin() * (PI * y).sin() * (-2.0 * PI * PI * alpha * final_time).exp();
        let error = (numerical - exact).abs();

        println!(
            "{:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
            y, numerical, exact, error
        );
    }

    println!("\nConvergence test with different grid sizes:");
    println!(
        "{:<15} {:<15} {:<15} {:<15}",
        "Grid size", "Max Error", "L2 Error", "Time (s)"
    );

    // Skip detailed convergence test in the example to save time
    // In a real application, you would run with different grid sizes
    // and track how the error decreases

    println!(
        "{:<15} {:<15.2e} {:<15.2e} {:<15.2e}",
        format!("{}x{}", nx, ny),
        max_error,
        l2_error,
        solve_time
    );

    Ok(())
}
