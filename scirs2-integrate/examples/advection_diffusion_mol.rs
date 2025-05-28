use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, MOLOptions,
    MOLParabolicSolver1D, ODEMethod,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Method of Lines example: Advection-Diffusion equation solver");
    println!("Solving: ∂u/∂t = α ∂²u/∂x² - v ∂u/∂x");
    println!("Domain: x ∈ [0, 1], t ∈ [0, 0.5]");
    println!("Boundary conditions: u(0, t) = 1, ∂u/∂x|_{{x=1}} = 0");
    println!("Initial condition: u(x, 0) = 1 for x ≤ 0.1, 0 otherwise");

    // Physical parameters
    let alpha = 0.01; // Diffusion coefficient
    let velocity = 0.5; // Advection velocity

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

    // Initial condition: step function
    let initial_condition = |x: f64| -> f64 {
        if x <= 0.1 {
            1.0
        } else {
            0.0
        }
    };

    // Boundary conditions:
    // - Left: Dirichlet u(0, t) = 1
    // - Right: Neumann ∂u/∂x|_{x=1} = 0
    let bcs = vec![
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0,
            value: 1.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Neumann,
            location: BoundaryLocation::Upper,
            dimension: 0,
            value: 0.0, // Zero gradient at right boundary
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

    // Create the MOL solver with advection term
    let mol_solver = MOLParabolicSolver1D::new(
        domain,
        time_range,
        diffusion_coeff,
        initial_condition,
        bcs,
        Some(options),
    )?
    .with_advection(move |_x: f64, _t: f64, _u: f64| -> f64 {
        velocity // Advection velocity (positive = right to left)
    });

    // Solve the PDE
    println!("Solving the advection-diffusion equation...");
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
    let time_indices = [0, nt / 5, 2 * nt / 5, 3 * nt / 5, 4 * nt / 5, nt - 1];
    let dx = 1.0 / (nx as f64 - 1.0);

    println!("\nSolution at selected time points:");

    for &ti in &time_indices {
        let time = t[ti];
        println!("Time t = {:.4}", time);
        println!("{:<10} {:<15}", "x", "u(x,t)");

        // Print solution at regular intervals
        for xi in (0..nx).step_by(5) {
            let x = xi as f64 * dx;
            let value = u[[ti, xi]];

            println!("{:<10.4} {:<15.8e}", x, value);
        }
        println!(); // Empty line between time points
    }

    // Analyze solution: mass conservation check
    println!("\nMass conservation check:");

    let mut initial_mass = 0.0;
    let mut final_mass = 0.0;

    // Approximate the integral of u(x,t) over the domain using trapezoidal rule
    for i in 0..nx - 1 {
        let _x = i as f64 * dx;
        let h = dx;

        // Initial mass
        let u0_i = u[[0, i]];
        let u0_i1 = u[[0, i + 1]];
        initial_mass += 0.5 * h * (u0_i + u0_i1);

        // Final mass
        let uf_i = u[[nt - 1, i]];
        let uf_i1 = u[[nt - 1, i + 1]];
        final_mass += 0.5 * h * (uf_i + uf_i1);
    }

    println!("Initial total mass: {:.8}", initial_mass);
    println!("Final total mass: {:.8}", final_mass);
    println!(
        "Relative change: {:.2e}%",
        100.0 * (final_mass - initial_mass) / initial_mass
    );

    // Analyze the front propagation
    println!("\nFront propagation analysis:");

    for &ti in &time_indices {
        let time = t[ti];

        // Find the position where u = 0.5 (midpoint of the front)
        let mut front_position = 0.0;
        let mut found = false;

        for i in 0..nx - 1 {
            let x_i = i as f64 * dx;
            let x_i1 = (i + 1) as f64 * dx;
            let u_i = u[[ti, i]];
            let u_i1 = u[[ti, i + 1]];

            if (u_i > 0.5 && u_i1 < 0.5) || (u_i < 0.5 && u_i1 > 0.5) {
                // Linear interpolation to find the point where u = 0.5
                front_position = x_i + (0.5 - u_i) * (x_i1 - x_i) / (u_i1 - u_i);
                found = true;
                break;
            }
        }

        if found {
            // Theoretical front position for pure advection would be x0 + v*t
            let theoretical_advection = 0.1 + velocity * time;
            println!(
                "Time t = {:.4}: Front at x = {:.4} (pure advection would be at x = {:.4})",
                time, front_position, theoretical_advection
            );
        } else {
            println!("Time t = {:.4}: Front not detected", time);
        }
    }

    // Calculate maximum concentration gradient
    println!("\nMaximum concentration gradient analysis:");

    for &ti in &time_indices {
        let time = t[ti];
        let mut max_gradient = 0.0;
        let mut max_gradient_position = 0.0;

        for i in 0..nx - 1 {
            let x_i = i as f64 * dx;
            let gradient = (u[[ti, i + 1]] - u[[ti, i]]) / dx;

            if gradient.abs() > max_gradient {
                max_gradient = gradient.abs();
                max_gradient_position = x_i + 0.5 * dx;
            }
        }

        println!(
            "Time t = {:.4}: Max gradient {:.4e} at x = {:.4}",
            time, max_gradient, max_gradient_position
        );
    }

    Ok(())
}
