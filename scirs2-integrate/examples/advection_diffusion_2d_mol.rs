use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, MOLOptions,
    MOLParabolicSolver2D, ODEMethod,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Method of Lines example: 2D Advection-Diffusion equation solver");
    println!("Solving: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) - vx∂u/∂x - vy∂u/∂y");
    println!("Domain: (x,y) ∈ [0, 1] × [0, 1], t ∈ [0, 0.5]");
    println!("Boundary conditions:");
    println!("  Left (x=0): u = 1 (Dirichlet)");
    println!("  Right (x=1): ∂u/∂x = 0 (Neumann)");
    println!("  Bottom (y=0): u = 0 (Dirichlet)");
    println!("  Top (y=1): ∂u/∂y = 0 (Neumann)");
    println!("Initial condition: u(x, y, 0) = e^(-50*((x-0.2)² + (y-0.2)²))");

    // Physical parameters
    let alpha = 0.01; // Diffusion coefficient
    let vx = 0.2; // Advection velocity in x direction
    let vy = 0.1; // Advection velocity in y direction

    // Set up spatial domain
    let nx = 61; // Number of grid points in x direction
    let ny = 61; // Number of grid points in y direction
    let domain = Domain::new(
        vec![0.0..1.0, 0.0..1.0], // Spatial ranges [0, 1] × [0, 1]
        vec![nx, ny],             // Number of grid points
    )?;

    // Time domain
    let t_start = 0.0;
    let t_end = 0.5;
    let time_range = [t_start, t_end];

    // Diffusion coefficients (constant in both directions)
    let diffusion_x = move |_x: f64, _y: f64, _t: f64, _u: f64| -> f64 { alpha };

    let diffusion_y = move |_x: f64, _y: f64, _t: f64, _u: f64| -> f64 { alpha };

    // Initial condition: Gaussian pulse centered at (0.2, 0.2)
    let initial_condition =
        |x: f64, y: f64| -> f64 { (-50.0 * ((x - 0.2).powi(2) + (y - 0.2).powi(2))).exp() };

    // Boundary conditions: Mixed
    let bcs = vec![
        // Left boundary (x=0): Dirichlet, u = 1
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0, // x dimension
            value: 1.0,
            coefficients: None,
        },
        // Right boundary (x=1): Neumann, ∂u/∂x = 0
        BoundaryCondition {
            bc_type: BoundaryConditionType::Neumann,
            location: BoundaryLocation::Upper,
            dimension: 0, // x dimension
            value: 0.0,
            coefficients: None,
        },
        // Bottom boundary (y=0): Dirichlet, u = 0
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1, // y dimension
            value: 0.0,
            coefficients: None,
        },
        // Top boundary (y=1): Neumann, ∂u/∂y = 0
        BoundaryCondition {
            bc_type: BoundaryConditionType::Neumann,
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

    // Create the MOL solver with advection terms
    let mol_solver = MOLParabolicSolver2D::new(
        domain,
        time_range,
        diffusion_x,
        diffusion_y,
        initial_condition,
        bcs,
        Some(options),
    )?
    .with_advection(
        // Advection in x-direction
        move |_x: f64, _y: f64, _t: f64, _u: f64| -> f64 { vx },
        // Advection in y-direction
        move |_x: f64, _y: f64, _t: f64, _u: f64| -> f64 { vy },
    );

    // Solve the PDE
    println!("Solving the 2D advection-diffusion equation...");
    let start_time = Instant::now();
    let result = mol_solver.solve()?;
    let solve_time = start_time.elapsed().as_secs_f64();

    // Extract results
    let t = &result.t;
    let u = &result.u; // Solution values [time, y, x]

    println!("Solution computed in {:.4} seconds", solve_time);
    println!("ODE solver info: {:?}", result.ode_info);

    // Print solution at selected times
    let nt = t.len();
    let time_indices = [0, nt / 5, 2 * nt / 5, 3 * nt / 5, 4 * nt / 5, nt - 1];

    // Create grid points
    let dx = 1.0 / (nx as f64 - 1.0);
    let dy = 1.0 / (ny as f64 - 1.0);

    println!("\nMaximum solution values at selected times:");
    println!(
        "{:<10} {:<15} {:<15} {:<15}",
        "Time", "Max Value", "At (x,y)", "Initial Max"
    );

    let initial_max = u[[0, 0, 0]];
    let mut max_i = 0;
    let mut max_j = 0;

    for &ti in &time_indices {
        let time = t[ti];

        // Find maximum value in the solution at this time
        let mut max_val = 0.0;
        let mut max_val_i = 0;
        let mut max_val_j = 0;

        for j in 0..ny {
            for i in 0..nx {
                let val = u[[ti, j, i]];
                if val > max_val {
                    max_val = val;
                    max_val_i = i;
                    max_val_j = j;
                }
            }
        }

        let max_x = max_val_i as f64 * dx;
        let max_y = max_val_j as f64 * dy;

        println!(
            "{:<10.4} {:<15.8e} {:<15} {:<15.8e}",
            time,
            max_val,
            format!("({:.2}, {:.2})", max_x, max_y),
            initial_max
        );

        if ti == nt - 1 {
            max_i = max_val_i;
            max_j = max_val_j;
        }
    }

    // Analyze pulse advection
    println!("\nPulse advection analysis:");
    println!("Initial pulse center: (0.20, 0.20)");

    // In the absence of diffusion, the pulse center would move with the advection velocity
    let final_time = t[nt - 1];
    let expected_x = 0.2 + vx * final_time;
    let expected_y = 0.2 + vy * final_time;
    let actual_x = max_i as f64 * dx;
    let actual_y = max_j as f64 * dy;

    println!(
        "Expected final center with pure advection: ({:.2}, {:.2})",
        expected_x, expected_y
    );
    println!(
        "Actual final maximum location: ({:.2}, {:.2})",
        actual_x, actual_y
    );
    println!(
        "Difference due to diffusion: ({:.2}, {:.2})",
        actual_x - expected_x,
        actual_y - expected_y
    );

    // Print solution profile along x-axis at final time (y=0.5)
    println!(
        "\nSolution profile along x-axis (y=0.5) at final time (t = {:.4}):",
        final_time
    );
    println!("{:<10} {:<15}", "x", "u(x,0.5,t)");

    let j = ny / 2; // y=0.5
    for i in (0..nx).step_by(5) {
        let x = i as f64 * dx;
        let numerical = u[[nt - 1, j, i]];

        println!("{:<10.4} {:<15.8e}", x, numerical);
    }

    // Print solution profile along y-axis at final time (x=0.5)
    println!(
        "\nSolution profile along y-axis (x=0.5) at final time (t = {:.4}):",
        final_time
    );
    println!("{:<10} {:<15}", "y", "u(0.5,y,t)");

    let i = nx / 2; // x=0.5
    for j in (0..ny).step_by(5) {
        let y = j as f64 * dy;
        let numerical = u[[nt - 1, j, i]];

        println!("{:<10.4} {:<15.8e}", y, numerical);
    }

    // Calculate mass conservation
    println!("\nMass conservation check:");

    let mut initial_mass = 0.0;
    let mut final_mass = 0.0;

    // Approximate integral using the trapezoid rule
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            // Area of the grid cell
            let area = dx * dy;

            // Initial mass (average of 4 corners of the cell)
            let u0_avg =
                (u[[0, j, i]] + u[[0, j + 1, i]] + u[[0, j, i + 1]] + u[[0, j + 1, i + 1]]) / 4.0;
            initial_mass += u0_avg * area;

            // Final mass
            let uf_avg = (u[[nt - 1, j, i]]
                + u[[nt - 1, j + 1, i]]
                + u[[nt - 1, j, i + 1]]
                + u[[nt - 1, j + 1, i + 1]])
                / 4.0;
            final_mass += uf_avg * area;
        }
    }

    println!("Initial total mass: {:.8}", initial_mass);
    println!("Final total mass: {:.8}", final_mass);
    println!(
        "Relative change: {:.2e}%",
        100.0 * (final_mass - initial_mass) / initial_mass
    );

    Ok(())
}
