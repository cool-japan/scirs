use scirs2_integrate::pde::implicit::{
    BackwardEuler1D, CrankNicolson1D, ImplicitMethod, ImplicitOptions, ImplicitResult,
};
use scirs2_integrate::pde::{BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain};

/// Example of using implicit methods to solve a nonlinear reaction-diffusion equation
///
/// This example demonstrates solving a Fisher-KPP equation:
/// ∂u/∂t = D * ∂²u/∂x² + r * u * (1 - u)
///
/// The Fisher-KPP equation is a nonlinear PDE that models the spread of a gene
/// in a population. It combines diffusion with logistic growth.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define domain: x ∈ [0, 10]
    let domain = Domain::new(vec![0.0..10.0], vec![201])?;

    // Define time range: t ∈ [0, 5.0]
    let time_range = [0.0, 5.0];

    // Create boundary conditions (zero-flux Neumann conditions)
    let boundary_conditions = vec![
        BoundaryCondition {
            bc_type: BoundaryConditionType::Neumann,
            location: BoundaryLocation::Lower,
            dimension: 0,
            value: 0.0, // du/dx = 0 at x = 0
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Neumann,
            location: BoundaryLocation::Upper,
            dimension: 0,
            value: 0.0, // du/dx = 0 at x = 10
            coefficients: None,
        },
    ];

    // Parameters for Fisher-KPP equation
    let diffusion = 0.5; // Diffusion coefficient D
    let growth_rate = 1.0; // Growth rate r

    // Define coefficients for the Fisher-KPP equation
    // ∂u/∂t = D * ∂²u/∂x² + r * u * (1 - u)
    let diffusion_coeff = move |_x: f64, _t: f64, _u: f64| diffusion;

    // For nonlinear reaction terms, we need to be careful with the implementation
    // Here, we linearize the nonlinear term r*u*(1-u) using the previous time step value
    // For full nonlinear treatment, Newton iterations would be needed
    let reaction_term = move |_x: f64, _t: f64, u: f64| {
        // Return the effective reaction rate k for linearization around u
        // From r*u*(1-u) = ku + c, where k = r*(1-2u) and c = r*u²
        growth_rate * (1.0 - 2.0 * u)
    };

    // Initial condition: localized population at center
    // u(x, 0) = exp(-(x-5)²)
    let initial_condition = |x: f64| (-((x - 5.0) * (x - 5.0))).exp();

    // Define solver options
    let options_cn = ImplicitOptions {
        method: ImplicitMethod::CrankNicolson,
        dt: Some(0.05),
        save_every: Some(20),
        verbose: true,
        ..Default::default()
    };

    let options_be = ImplicitOptions {
        method: ImplicitMethod::BackwardEuler,
        dt: Some(0.05),
        save_every: Some(20),
        verbose: true,
        ..Default::default()
    };

    // Solve using Crank-Nicolson method
    println!("Solving Fisher-KPP equation with Crank-Nicolson method...");
    let cn_solver = CrankNicolson1D::new(
        domain.clone(),
        time_range,
        diffusion_coeff,
        initial_condition,
        boundary_conditions.clone(),
        Some(options_cn),
    )?
    .with_reaction(reaction_term);

    let cn_result = cn_solver.solve()?;

    // Solve using Backward Euler method
    println!("\nSolving Fisher-KPP equation with Backward Euler method...");
    let be_solver = BackwardEuler1D::new(
        domain.clone(),
        time_range,
        diffusion_coeff,
        initial_condition,
        boundary_conditions.clone(),
        Some(options_be),
    )?
    .with_reaction(reaction_term);

    let be_result = be_solver.solve()?;

    // Output solution summary
    println!("\nFisher-KPP Simulation Results:");
    println!("==============================");
    println!("Domain: x ∈ [0, 10], t ∈ [0, 5.0]");
    println!(
        "Parameters: Diffusion D = {}, Growth rate r = {}",
        diffusion, growth_rate
    );

    // Analyze wave propagation (Fisher-KPP generates traveling waves)
    analyze_wave_propagation(&cn_result, &domain)?;

    // Compare the solutions
    compare_solutions(&cn_result, &be_result, &domain)?;

    println!("\nThis example demonstrates how implicit methods can effectively");
    println!("simulate nonlinear reaction-diffusion systems like Fisher-KPP equation,");
    println!("which develop traveling wave solutions.");

    Ok(())
}

/// Analyze the propagation of the traveling wave solution
fn analyze_wave_propagation(
    result: &ImplicitResult,
    domain: &Domain,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get spatial grid
    let x_grid = domain.grid(0)?;
    let nx = x_grid.len();

    // Get times to analyze
    let times = &result.t;
    let num_times = result.u.len();

    // Selected time indices for analysis
    let time_indices = [
        0,
        num_times / 4,
        num_times / 2,
        3 * num_times / 4,
        num_times - 1,
    ];

    println!("\nWave Propagation Analysis:");
    println!("=========================");

    // Track wave front position (where u = 0.5) at different times
    println!("Position of wave front (u = 0.5) at selected times:");
    println!("Time | Position | Theoretical Speed");
    println!("-----|----------|------------------");

    let mut prev_pos = 5.0; // Center of domain
    let mut prev_time = 0.0;

    for &idx in &time_indices {
        let time = times[idx];
        let sol = &result.u[idx];

        // Find position where u = 0.5
        let mut front_pos = 5.0; // Default to center

        for i in 0..nx - 1 {
            if sol[[i, 0]] > 0.5 && sol[[i + 1, 0]] <= 0.5 {
                // Linear interpolation for better precision
                let x1 = x_grid[i];
                let x2 = x_grid[i + 1];
                let y1 = sol[[i, 0]];
                let y2 = sol[[i + 1, 0]];

                // Find x where y = 0.5 using linear interpolation
                front_pos = x1 + (x2 - x1) * (0.5 - y1) / (y2 - y1);
                break;
            }

            if sol[[i, 0]] <= 0.5 && sol[[i + 1, 0]] > 0.5 {
                // Linear interpolation for better precision
                let x1 = x_grid[i];
                let x2 = x_grid[i + 1];
                let y1 = sol[[i, 0]];
                let y2 = sol[[i + 1, 0]];

                // Find x where y = 0.5 using linear interpolation
                front_pos = x1 + (x2 - x1) * (0.5 - y1) / (y2 - y1);
                break;
            }
        }

        // Calculate observed speed
        let speed = if time > prev_time {
            (front_pos - prev_pos) / (time - prev_time)
        } else {
            0.0
        };

        // Theoretical Fisher-KPP wave speed = 2√(D*r)
        let theoretical_speed = 2.0 * (0.5_f64 * 1.0).sqrt();

        println!(
            "{:.1} | {:.4} | {:.4} (expected: {:.4})",
            time, front_pos, speed, theoretical_speed
        );

        prev_pos = front_pos;
        prev_time = time;
    }

    // Print values at specific locations for the final time
    let final_sol = result.u.last().unwrap();
    println!("\nFinal solution values at specific locations:");
    println!("Position | Value");
    println!("---------|------");

    for &x_pos in &[0.0, 2.5, 5.0, 7.5, 10.0] {
        // Find closest grid point
        let i = ((x_pos - domain.ranges[0].start) / (domain.ranges[0].end - domain.ranges[0].start)
            * (nx as f64 - 1.0))
            .round() as usize;

        println!("{:.1} | {:.4}", x_grid[i], final_sol[[i, 0]]);
    }

    Ok(())
}

/// Compare the two solutions
fn compare_solutions(
    cn_result: &ImplicitResult,
    be_result: &ImplicitResult,
    _domain: &Domain,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get final solutions
    let cn_final = cn_result.u.last().unwrap();
    let be_final = be_result.u.last().unwrap();

    // Calculate difference
    let mut max_diff: f64 = 0.0;
    let mut avg_diff = 0.0;

    let nx = cn_final.shape()[0];
    for i in 0..nx {
        let diff = (cn_final[[i, 0]] - be_final[[i, 0]]).abs();
        max_diff = max_diff.max(diff);
        avg_diff += diff;
    }
    avg_diff /= nx as f64;

    println!("\nComparison between Crank-Nicolson and Backward Euler:");
    println!("===============================================");
    println!("Maximum absolute difference: {:.6e}", max_diff);
    println!("Average absolute difference: {:.6e}", avg_diff);
    println!(
        "Relative difference: {:.2}%",
        100.0 * avg_diff / cn_final.iter().sum::<f64>() * nx as f64
    );

    // Performance comparison
    println!("\nPerformance comparison:");
    println!(
        "Crank-Nicolson: {:.4} seconds, {} linear solves",
        cn_result.computation_time, cn_result.num_linear_solves
    );
    println!(
        "Backward Euler: {:.4} seconds, {} linear solves",
        be_result.computation_time, be_result.num_linear_solves
    );
    println!(
        "Speed ratio (BE/CN): {:.2}",
        cn_result.computation_time / be_result.computation_time
    );

    Ok(())
}
