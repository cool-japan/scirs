use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, SpectralElementOptions,
    SpectralElementPoisson2D,
};
use std::f64::consts::PI;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Spectral Element Method example for the 2D Poisson equation");
    println!("Solving: ∇²u = f(x,y) with Dirichlet boundary conditions");
    println!("Domain: [0, 1] × [0, 1]");
    println!("Source term: f(x,y) = -2π² sin(πx) sin(πy)");
    println!("Boundary conditions: u = 0 on all boundaries");
    println!("Exact solution: u(x,y) = sin(πx) sin(πy)");

    // Set up domain
    let domain = Domain::new(
        vec![0.0..1.0, 0.0..1.0], // 2D domain [0,1] × [0,1]
        vec![41, 41],             // Grid resolution (not directly used by SEM)
    )?;

    // Define source term - a function that makes sin(πx)sin(πy) the exact solution
    let source_term = |x: f64, y: f64| -> f64 { -2.0 * PI * PI * (PI * x).sin() * (PI * y).sin() };

    // Define boundary conditions
    let bcs = vec![
        // Bottom boundary (y = 0)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1,
            value: 0.0,
            coefficients: None,
        },
        // Top boundary (y = 1)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 1,
            value: 0.0,
            coefficients: None,
        },
        // Left boundary (x = 0)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
        // Right boundary (x = 1)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
    ];

    // Solver options
    let options = SpectralElementOptions {
        order: 4,          // Polynomial order (4 = 5 points per element in each direction)
        nx: 4,             // Number of elements in x direction
        ny: 4,             // Number of elements in y direction
        max_iterations: 1, // Direct solver
        tolerance: 1e-10,
        save_convergence_history: false,
        verbose: true,
    };

    // Create the spectral element solver
    let solver = SpectralElementPoisson2D::new(
        domain.clone(),
        source_term,
        bcs.clone(),
        Some(options.clone()),
    )?;

    // Solve the PDE
    println!("\nSolving with Spectral Element Method...");
    println!(
        "Using {} x {} elements with polynomial order {}",
        options.nx, options.ny, options.order
    );
    let start_time = Instant::now();
    let result = solver.solve()?;
    let solve_time = start_time.elapsed().as_secs_f64();

    println!("Solution computed in {solve_time:.4} seconds");
    println!("Residual norm: {:.6e}", result.residual_norm);
    println!("Number of elements: {}", result.elements.len());
    println!("Total number of nodes: {}", result.nodes.len());

    // Calculate errors at nodes
    let mut max_error: f64 = 0.0;
    let mut l2_error = 0.0;

    for (i, (x, y)) in result.nodes.iter().enumerate() {
        // Exact solution: sin(πx)sin(πy)
        let exact = (PI * x).sin() * (PI * y).sin();

        // Error
        let error = (result.u[i] - exact).abs();
        max_error = max_error.max(error);
        l2_error += error * error;
    }

    // Compute L2 error norm
    l2_error = (l2_error / result.nodes.len() as f64).sqrt();

    println!("\nError analysis:");
    println!("  - Maximum error: {max_error:.6e}");
    println!("  - L2 error norm: {l2_error:.6e}");

    // Print solution at selected points
    println!("\nSolution at selected points:");
    println!(
        "{:<10} {:<10} {:<15} {:<15} {:<10}",
        "x", "y", "Numerical", "Exact", "Error"
    );

    let points = [
        (0.25, 0.25),
        (0.50, 0.50),
        (0.75, 0.75),
        (0.25, 0.75),
        (0.75, 0.25),
    ];

    for &(point_x, point_y) in &points {
        // Find the closest node
        let mut closest_idx = 0;
        let mut min_dist = f64::MAX;

        for (i, (x, y)) in result.nodes.iter().enumerate() {
            let dist = ((x - point_x).powi(2) + (y - point_y).powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                closest_idx = i;
            }
        }

        let (x, y) = result.nodes[closest_idx];
        let numerical = result.u[closest_idx];
        let exact = (PI * x).sin() * (PI * y).sin();
        let error = (numerical - exact).abs();

        println!("{x:<10.4} {y:<10.4} {numerical:<15.8e} {exact:<15.8e} {error:<10.2e}");
    }

    // Test convergence with p-refinement (increasing polynomial order)
    println!("\np-Convergence Test (increasing polynomial order):");
    println!("Fixing 2x2 elements and increasing polynomial order");

    for order in [2, 4, 6, 8] {
        let options = SpectralElementOptions {
            order,
            nx: 2,
            ny: 2,
            max_iterations: 1,
            tolerance: 1e-10,
            save_convergence_history: false,
            verbose: false,
        };

        let solver =
            SpectralElementPoisson2D::new(domain.clone(), source_term, bcs.clone(), Some(options))?;

        let result = solver.solve()?;

        // Calculate L2 error
        let mut l2_error = 0.0;

        for (i, (x, y)) in result.nodes.iter().enumerate() {
            let exact = (PI * x).sin() * (PI * y).sin();
            let error = (result.u[i] - exact).abs();
            l2_error += error * error;
        }

        l2_error = (l2_error / result.nodes.len() as f64).sqrt();

        println!(
            "  - Order p = {}: L2 error = {:.6e}, DOFs = {}",
            order,
            l2_error,
            result.nodes.len()
        );
    }

    // Test convergence with h-refinement (increasing number of elements)
    println!("\nh-Convergence Test (increasing number of elements):");
    println!("Fixing polynomial order 4 and increasing element count");

    for n in [2, 4, 6, 8] {
        let options = SpectralElementOptions {
            order: 4,
            nx: n,
            ny: n,
            max_iterations: 1,
            tolerance: 1e-10,
            save_convergence_history: false,
            verbose: false,
        };

        let solver =
            SpectralElementPoisson2D::new(domain.clone(), source_term, bcs.clone(), Some(options))?;

        let result = solver.solve()?;

        // Calculate L2 error
        let mut l2_error = 0.0;

        for (i, (x, y)) in result.nodes.iter().enumerate() {
            let exact = (PI * x).sin() * (PI * y).sin();
            let error = (result.u[i] - exact).abs();
            l2_error += error * error;
        }

        l2_error = (l2_error / result.nodes.len() as f64).sqrt();

        println!(
            "  - Grid {}x{}: L2 error = {:.6e}, DOFs = {}",
            n,
            n,
            l2_error,
            result.nodes.len()
        );
    }

    // Example with mixed boundary conditions
    println!("\nBonus example: Solving with mixed boundary conditions");
    println!("Bottom (y=0): Dirichlet u=0");
    println!("Top (y=1): Neumann du/dy=0");
    println!("Left (x=0): Dirichlet u=0");
    println!("Right (x=1): Neumann du/dx=0");

    // Define mixed boundary conditions
    let mixed_bcs = vec![
        // Bottom boundary (y = 0): Dirichlet
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1,
            value: 0.0,
            coefficients: None,
        },
        // Top boundary (y = 1): Neumann
        BoundaryCondition {
            bc_type: BoundaryConditionType::Neumann,
            location: BoundaryLocation::Upper,
            dimension: 1,
            value: 0.0,
            coefficients: None,
        },
        // Left boundary (x = 0): Dirichlet
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
        // Right boundary (x = 1): Neumann
        BoundaryCondition {
            bc_type: BoundaryConditionType::Neumann,
            location: BoundaryLocation::Upper,
            dimension: 0,
            value: 0.0,
            coefficients: None,
        },
    ];

    let mixed_options = SpectralElementOptions {
        order: 4,
        nx: 4,
        ny: 4,
        max_iterations: 1,
        tolerance: 1e-10,
        save_convergence_history: false,
        verbose: false,
    };

    // Create solver with mixed boundary conditions
    let mixed_solver =
        SpectralElementPoisson2D::new(domain.clone(), source_term, mixed_bcs, Some(mixed_options))?;

    // Solve with mixed boundary conditions
    let mixed_result = mixed_solver.solve()?;

    // Print results for mixed boundary conditions
    println!("\nSolution with mixed boundary conditions at selected points:");
    println!("{:<10} {:<10} {:<15}", "x", "y", "u(x,y)");

    for &(point_x, point_y) in &points {
        // Find the closest node
        let mut closest_idx = 0;
        let mut min_dist = f64::MAX;

        for (i, (x, y)) in mixed_result.nodes.iter().enumerate() {
            let dist = ((x - point_x).powi(2) + (y - point_y).powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                closest_idx = i;
            }
        }

        let (x, y) = mixed_result.nodes[closest_idx];
        let value = mixed_result.u[closest_idx];

        println!("{x:<10.4} {y:<10.4} {value:<15.8e}");
    }

    // Explain key advantages of spectral element methods
    println!("\nSpectral Element Method Advantages:");
    println!("1. Combines high-order accuracy of spectral methods with");
    println!("   geometric flexibility of finite element methods");
    println!("2. Exponential convergence for smooth solutions (p-refinement)");
    println!("3. Good parallel efficiency due to element-wise operations");
    println!("4. Handles complex geometries better than global spectral methods");
    println!("5. Can be extended to unstructured meshes and 3D problems");

    Ok(())
}
