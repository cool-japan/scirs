// Removed unused import Array2
use scirs2_integrate::pde::implicit::{ImplicitOptions, ADI2D};
use scirs2_integrate::pde::{BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain};

/// Example demonstrating the ADI method for a 2D advection-diffusion problem
///
/// This example solves the advection-diffusion equation:
/// ∂u/∂t + v_x ∂u/∂x + v_y ∂u/∂y = D_x ∂²u/∂x² + D_y ∂²u/∂y²
///
/// with a Gaussian initial condition.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define 2D domain: (x,y) ∈ [0,2]×[0,2]
    let domain = Domain::new(vec![0.0..2.0, 0.0..2.0], vec![101, 101])?;

    // Define time range: t ∈ [0, 1.0]
    let time_range = [0.0, 1.0];

    // Problem parameters
    let diffusion_coeff_x = 0.01; // Small diffusion for visible advection effects
    let diffusion_coeff_y = 0.01;
    let velocity_x: f64 = 0.5; // Advection velocities
    let velocity_y: f64 = 0.5;

    // Create boundary conditions (Dirichlet zeros on all boundaries)
    let boundary_conditions = vec![
        // x-direction boundaries
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
        // y-direction boundaries
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1,
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 1,
            value: 0.0,
            coefficients: None,
        },
    ];

    // Define diffusion coefficients
    let diffusion_x = move |_x: f64, _y: f64, _t: f64, _u: f64| diffusion_coeff_x;
    let diffusion_y = move |_x: f64, _y: f64, _t: f64, _u: f64| diffusion_coeff_y;

    // Define advection velocities
    let advection_x = move |_x: f64, _y: f64, _t: f64, _u: f64| velocity_x;
    let advection_y = move |_x: f64, _y: f64, _t: f64, _u: f64| velocity_y;

    // Define initial condition: Gaussian pulse at (0.5, 0.5)
    let initial_condition = |x: f64, y: f64| {
        let x0 = 0.5;
        let y0 = 0.5;
        let sigma: f64 = 0.1;
        (-((x - x0).powi(2) + (y - y0).powi(2)) / (2.0 * sigma.powi(2))).exp()
    };

    // Define solver options
    // Choose a time step that satisfies CFL condition
    let dx = 2.0 / 100.0;
    let cfl_dt = 0.8 * dx / (velocity_x.max(velocity_y));

    let options = ImplicitOptions {
        dt: Some(cfl_dt),
        save_every: Some(10),
        verbose: true,
        ..Default::default()
    };

    // Create ADI solver with advection terms
    println!("Solving 2D advection-diffusion equation using ADI method...");
    println!("Parameters:");
    println!("  Domain: [0,2]×[0,2], t ∈ [0,1.0]");
    println!("  Grid: 101×101 points");
    println!(
        "  Diffusion: D_x = {}, D_y = {}",
        diffusion_coeff_x, diffusion_coeff_y
    );
    println!(
        "  Advection velocities: v_x = {}, v_y = {}",
        velocity_x, velocity_y
    );
    println!("  Time step: dt = {:.5} (CFL condition)", cfl_dt);

    let adi_solver = ADI2D::new(
        domain.clone(),
        time_range,
        diffusion_x,
        diffusion_y,
        initial_condition,
        boundary_conditions.clone(),
        Some(options),
    )?
    .with_advection(advection_x, advection_y);

    let result = adi_solver.solve()?;

    // Analyze solution and visualize movement of the Gaussian pulse
    analyze_solution(&result, &domain, velocity_x, velocity_y)?;

    println!("\nADI method solution completed successfully.");
    println!("Computation time: {:.4} seconds", result.computation_time);
    println!("Number of time steps: {}", result.num_steps);
    println!("Number of linear solves: {}", result.num_linear_solves);

    if let Some(info) = result.info {
        println!("{}", info);
    }

    Ok(())
}

/// Analyze the numerical solution and track the movement of the Gaussian pulse
fn analyze_solution(
    result: &scirs2_integrate::pde::implicit::ADIResult,
    domain: &Domain,
    velocity_x: f64,
    velocity_y: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get spatial grids
    let x_grid = domain.grid(0)?;
    let y_grid = domain.grid(1)?;
    let nx = x_grid.len();
    let ny = y_grid.len();

    // Track the pulse peak location over time
    println!("\nTracking pulse peak location over time:");
    println!("  Time   |   (x,y)   | Expected (x,y) |  Difference ");
    println!("---------|-----------|----------------|-------------");

    // Initial peak location
    let x0 = 0.5;
    let y0 = 0.5;

    // Select a few time points to check
    let num_times = result.t.len();
    let time_indices = [
        0,
        num_times / 5,
        2 * num_times / 5,
        3 * num_times / 5,
        4 * num_times / 5,
        num_times - 1,
    ];

    for &idx in &time_indices {
        let t = result.t[idx];
        let solution = &result.u[idx];

        // Find maximum value in the solution (peak of Gaussian)
        let mut max_val = 0.0;
        let mut max_i = 0;
        let mut max_j = 0;

        for i in 0..nx {
            for j in 0..ny {
                if solution[[i, j, 0]] > max_val {
                    max_val = solution[[i, j, 0]];
                    max_i = i;
                    max_j = j;
                }
            }
        }

        // Convert indices to coordinates
        let x_peak = x_grid[max_i];
        let y_peak = y_grid[max_j];

        // Expected peak location based on advection velocity
        let x_expected = x0 + velocity_x * t;
        let y_expected = y0 + velocity_y * t;

        // Calculate difference
        let diff_x = (x_peak - x_expected).abs();
        let diff_y = (y_peak - y_expected).abs();
        let diff_total = (diff_x.powi(2) + diff_y.powi(2)).sqrt();

        println!(
            " {:.4} | ({:.3},{:.3}) | ({:.3},{:.3}) | {:.4e}",
            t, x_peak, y_peak, x_expected, y_expected, diff_total
        );
    }

    // Calculate maximum solution value over time (showing diffusion effects)
    println!("\nMaximum solution value over time (showing diffusion effects):");
    println!("  Time   | Max Value | Decay Factor ");
    println!("---------|-----------|-------------");

    let initial_max = find_max_value(&result.u[0]);

    for &idx in &time_indices {
        let t = result.t[idx];
        let max_val = find_max_value(&result.u[idx]);
        let decay_factor = max_val / initial_max;

        println!(" {:.4} | {:.6} | {:.6}", t, max_val, decay_factor);
    }

    // Compute the total mass (integral of solution) over time
    println!("\nTotal mass conservation check:");
    println!("  Time   | Total Mass | Relative Change");
    println!("---------|------------|----------------");

    let dx = domain.grid_spacing(0)?;
    let dy = domain.grid_spacing(1)?;
    let cell_area = dx * dy;

    let initial_mass = compute_total_mass(&result.u[0], cell_area);

    for &idx in &time_indices {
        let t = result.t[idx];
        let mass = compute_total_mass(&result.u[idx], cell_area);
        let rel_change = (mass - initial_mass) / initial_mass;

        println!(" {:.4} | {:.6} | {:.3e}", t, mass, rel_change);
    }

    // Check CFL condition
    let dx = domain.grid_spacing(0)?;
    let dt = result.t[1] - result.t[0];

    // Use the same values as defined at the beginning of main
    let velocity_x: f64 = 0.5;
    let velocity_y: f64 = 0.5;
    let diffusion_coeff_x = 0.01;
    let diffusion_coeff_y = 0.01;

    let advective_cfl = dt * f64::max(velocity_x, velocity_y) / dx;
    let diffusive_cfl = dt * f64::max(diffusion_coeff_x, diffusion_coeff_y) / (dx * dx);

    println!("\nStability analysis:");
    println!(
        "  Advective CFL number: {:.4} (should be < 1)",
        advective_cfl
    );
    println!(
        "  Diffusive CFL number: {:.4} (should be < 0.5 for explicit schemes)",
        diffusive_cfl
    );
    println!("  Grid spacing (dx): {:.4}", dx);
    println!("  Time step (dt): {:.4}", dt);

    Ok(())
}

/// Find the maximum value in a 3D array
fn find_max_value(array: &ndarray::Array3<f64>) -> f64 {
    array.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
}

/// Compute the total mass (integral of solution)
fn compute_total_mass(array: &ndarray::Array3<f64>, cell_area: f64) -> f64 {
    let sum = array.iter().sum::<f64>();
    sum * cell_area
}
