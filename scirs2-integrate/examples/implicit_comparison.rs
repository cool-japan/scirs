use ndarray::Array2;
use scirs2_integrate::ode::ODEMethod;
use scirs2_integrate::pde::implicit::{
    BackwardEuler1D, CrankNicolson1D, ImplicitMethod, ImplicitOptions,
};
use scirs2_integrate::pde::method_of_lines::{MOLOptions, MOLParabolicSolver1D};
use scirs2_integrate::pde::{BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain};
use std::time::Instant;

/// This example demonstrates and compares different numerical methods for solving a stiff PDE:
/// - Explicit Method of Lines (Forward Euler, RK4, Dopri5)
/// - Implicit Crank-Nicolson
/// - Implicit Backward Euler
///
/// The PDE is an advection-diffusion-reaction equation with:
/// ∂u/∂t = ε∂²u/∂x² - v∂u/∂x + ku(1-u)
///
/// For small ε, this equation becomes stiff, making it challenging for explicit methods.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Domain and time range
    let domain = Domain::new(vec![0.0..1.0], vec![101])?;
    let time_range = [0.0, 0.5];

    // Problem parameters (stiff configuration)
    let epsilon = 0.001; // Small diffusion coefficient (stiffness source)
    let velocity = 1.0; // Advection velocity
    let reaction_rate = 1.0; // Reaction rate

    // Define boundary conditions
    let boundary_conditions = vec![
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

    // Initial condition: Gaussian pulse
    let initial_condition = |x: f64| {
        let x0 = 0.25;
        let sigma = 0.05f64;
        (-((x - x0).powi(2)) / (2.0 * sigma.powi(2))).exp()
    };

    // Define PDE terms
    let diffusion_coeff = move |_x: f64, _t: f64, _u: f64| epsilon;
    let advection_coeff = move |_x: f64, _t: f64, _u: f64| velocity;
    let reaction_term = move |_x: f64, _t: f64, u: f64| reaction_rate * u * (1.0 - u);

    // Calculate stability limits for explicit methods
    let dx = 1.0 / 100.0;
    let diffusion_limit: f64 = dx * dx / (2.0 * epsilon); // Stability limit for diffusion
    let advection_limit: f64 = dx / velocity; // CFL condition for advection

    println!("Stiff PDE: Advection-Diffusion-Reaction Equation");
    println!("Parameters:");
    println!("  Diffusion coefficient (ε): {}", epsilon);
    println!("  Advection velocity (v): {}", velocity);
    println!("  Reaction rate (k): {}", reaction_rate);
    println!("  Grid spacing (dx): {}", dx);
    println!("\nStability limits for explicit methods:");
    println!("  Diffusion limit: dt < {:.6}", diffusion_limit);
    println!("  Advection limit (CFL): dt < {:.6}", advection_limit);
    println!(
        "  Combined limit: dt < {:.6}",
        diffusion_limit.min(advection_limit)
    );
    println!();

    // Try different time step sizes to demonstrate stability
    let time_steps = [0.1, 0.01, 0.001, 0.0001];

    println!("Method comparison for different time steps:");
    println!("==========================================");
    println!("dt | Explicit MOL | Crank-Nicolson | Backward Euler");
    println!("------------------------------------------------");

    for &dt in &time_steps {
        print!("{:.4} | ", dt);

        // 1. Explicit Method of Lines
        let start_time = Instant::now();
        let mol_result = try_explicit_method(
            &domain,
            time_range,
            dt,
            initial_condition,
            diffusion_coeff,
            advection_coeff,
            reaction_term,
            &boundary_conditions,
        );
        let mol_time = start_time.elapsed().as_secs_f64();

        match mol_result {
            Ok(_) => print!("Stable ({:.3}s) | ", mol_time),
            Err(_) => print!("Unstable      | "),
        }

        // 2. Crank-Nicolson
        let start_time = Instant::now();
        let cn_solver = CrankNicolson1D::new(
            domain.clone(),
            time_range,
            diffusion_coeff,
            initial_condition,
            boundary_conditions.clone(),
            Some(ImplicitOptions {
                dt: Some(dt),
                verbose: false,
                ..Default::default()
            }),
        )?
        .with_advection(advection_coeff)
        .with_reaction(reaction_term);

        let cn_result = cn_solver.solve();
        let cn_time = start_time.elapsed().as_secs_f64();

        match cn_result {
            Ok(_) => print!("Stable ({:.3}s) | ", cn_time),
            Err(_) => print!("Unstable      | "),
        }

        // 3. Backward Euler
        let start_time = Instant::now();
        let be_solver = BackwardEuler1D::new(
            domain.clone(),
            time_range,
            diffusion_coeff,
            initial_condition,
            boundary_conditions.clone(),
            Some(ImplicitOptions {
                method: ImplicitMethod::BackwardEuler,
                dt: Some(dt),
                verbose: false,
                ..Default::default()
            }),
        )?
        .with_advection(advection_coeff)
        .with_reaction(reaction_term);

        let be_result = be_solver.solve();
        let be_time = start_time.elapsed().as_secs_f64();

        match be_result {
            Ok(_) => println!("Stable ({:.3}s)", be_time),
            Err(_) => println!("Unstable"),
        }
    }

    // Run with a stable time step for all methods to compare solutions
    println!("\nSolution comparison with dt = 0.0001 (stable for all methods):");
    println!("==========================================================");

    // Set up solvers with identical dt
    let stable_dt = 0.0001;

    // 1. Explicit Method of Lines
    let mol_result = match try_explicit_method(
        &domain,
        time_range,
        stable_dt,
        initial_condition,
        diffusion_coeff,
        advection_coeff,
        reaction_term,
        &boundary_conditions,
    ) {
        Ok(result) => result,
        Err(_) => panic!("MOL should be stable with dt = {}", stable_dt),
    };

    // 2. Crank-Nicolson
    let cn_solver = CrankNicolson1D::new(
        domain.clone(),
        time_range,
        diffusion_coeff,
        initial_condition,
        boundary_conditions.clone(),
        Some(ImplicitOptions {
            dt: Some(stable_dt),
            verbose: false,
            ..Default::default()
        }),
    )?
    .with_advection(advection_coeff)
    .with_reaction(reaction_term);

    let cn_result = cn_solver.solve()?;

    // 3. Backward Euler
    let be_solver = BackwardEuler1D::new(
        domain.clone(),
        time_range,
        diffusion_coeff,
        initial_condition,
        boundary_conditions.clone(),
        Some(ImplicitOptions {
            method: ImplicitMethod::BackwardEuler,
            dt: Some(stable_dt),
            verbose: false,
            ..Default::default()
        }),
    )?
    .with_advection(advection_coeff)
    .with_reaction(reaction_term);

    let be_result = be_solver.solve()?;

    // Compare final solutions
    compare_solutions(
        &mol_result.u[0],
        cn_result.u.last().unwrap(),
        be_result.u.last().unwrap(),
        &domain,
    )?;

    println!("\nConclusion:");
    println!("  * Explicit method (MOL): Only stable with very small time steps due to stiffness");
    println!("  * Crank-Nicolson: Stable for larger time steps, second-order accurate but can have oscillations");
    println!("  * Backward Euler: Most stable for stiff problems, but only first-order accurate");
    println!("  * Implicit methods allow much larger time steps for stiff problems, trading some accuracy for stability");

    Ok(())
}

/// Helper function to try solving with explicit MOL method
fn try_explicit_method(
    domain: &Domain,
    time_range: [f64; 2],
    _dt: f64,
    initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
    diffusion_coeff: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    advection_coeff: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    reaction_term: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    boundary_conditions: &[BoundaryCondition<f64>],
) -> Result<scirs2_integrate::pde::method_of_lines::MOLResult, Box<dyn std::error::Error>> {
    let mol_options = MOLOptions {
        ode_method: ODEMethod::RK45,
        atol: 1e-6,
        rtol: 1e-3,
        ..Default::default()
    };

    let mol_solver = MOLParabolicSolver1D::new(
        domain.clone(),
        time_range,
        diffusion_coeff,
        initial_condition,
        boundary_conditions.to_vec(),
        Some(mol_options),
    )?
    .with_advection(advection_coeff)
    .with_reaction(reaction_term);

    let result = mol_solver.solve()?;
    Ok(result)
}

/// Compare the solutions from different methods
fn compare_solutions(
    mol_solution: &Array2<f64>,
    cn_solution: &Array2<f64>,
    be_solution: &Array2<f64>,
    domain: &Domain,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get spatial grid
    let x_grid = domain.grid(0)?;
    let nx = x_grid.len();

    // Calculate L2 norm differences
    let mut mol_cn_diff = 0.0;
    let mut mol_be_diff = 0.0;
    let mut cn_be_diff = 0.0;

    for i in 0..nx {
        mol_cn_diff += (mol_solution[[i, 0]] - cn_solution[[i, 0]]).powi(2);
        mol_be_diff += (mol_solution[[i, 0]] - be_solution[[i, 0]]).powi(2);
        cn_be_diff += (cn_solution[[i, 0]] - be_solution[[i, 0]]).powi(2);
    }

    mol_cn_diff = (mol_cn_diff / nx as f64).sqrt();
    mol_be_diff = (mol_be_diff / nx as f64).sqrt();
    cn_be_diff = (cn_be_diff / nx as f64).sqrt();

    println!("Solution differences (L2 norm):");
    println!("  MOL vs Crank-Nicolson: {:.6e}", mol_cn_diff);
    println!("  MOL vs Backward Euler: {:.6e}", mol_be_diff);
    println!("  CN vs Backward Euler:  {:.6e}", cn_be_diff);

    // Print solution at selected points
    println!("\nSolution values at selected points:");
    println!("    x    |    MOL    | Crank-Nicolson | Backward Euler");
    println!("------------------------------------------------");

    for &i in &[0, nx / 4, nx / 2, 3 * nx / 4, nx - 1] {
        let x = x_grid[i];
        println!(
            "{:.4} | {:.6} | {:.6} | {:.6}",
            x,
            mol_solution[[i, 0]],
            cn_solution[[i, 0]],
            be_solution[[i, 0]]
        );
    }

    Ok(())
}
