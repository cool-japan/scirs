use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, ElementType, FEMOptions,
    FEMPoissonSolver, TriangularMesh,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Poisson equation solver using Finite Element Method");
    println!("Solving: ∇²u = -2π² sin(πx) sin(πy)");
    println!("Domain: (x,y) ∈ [0, 1] × [0, 1]");
    println!("Boundary conditions: u = 0 on all boundaries");
    println!("Exact solution: u(x, y) = sin(πx) sin(πy)");

    // Create a triangular mesh on a rectangular domain
    let nx = 20; // Number of divisions in x-direction
    let ny = 20; // Number of divisions in y-direction
    let mesh = TriangularMesh::generate_rectangular(
        (0.0, 1.0), // x range
        (0.0, 1.0), // y range
        nx,
        ny,
    );

    println!("Mesh generated with:");
    println!("  - {} nodes", mesh.points.len());
    println!("  - {} triangular elements", mesh.elements.len());
    println!("  - {} boundary edges", mesh.boundary_edges.len());

    // Source term: -2π² sin(πx) sin(πy)
    let source_term = |x: f64, y: f64| -> f64 { -2.0 * PI * PI * (PI * x).sin() * (PI * y).sin() };

    // Boundary conditions: u = 0 on all boundaries (Dirichlet)
    let bcs = vec![
        // Bottom boundary (y=0)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1, // y dimension
            value: 0.0,
            coefficients: None,
        },
        // Right boundary (x=1)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 0, // x dimension
            value: 0.0,
            coefficients: None,
        },
        // Top boundary (y=1)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 1, // y dimension
            value: 0.0,
            coefficients: None,
        },
        // Left boundary (x=0)
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0, // x dimension
            value: 0.0,
            coefficients: None,
        },
    ];

    // Solver options
    let options = FEMOptions {
        element_type: ElementType::Linear,
        max_iterations: 1000,
        tolerance: 1e-6,
        save_convergence_history: false,
        verbose: true,
    };

    // Create the FEM solver
    let mut fem_solver = FEMPoissonSolver::new(mesh, source_term, bcs, Some(options))?;

    // Solve the problem
    println!("\nSolving with Finite Element Method...");
    let start_time = Instant::now();
    let result = fem_solver.solve()?;
    let solve_time = start_time.elapsed().as_secs_f64();

    // Extract solution
    let u = &result.u;
    let mesh = &result.mesh;

    println!("\nResults:");
    println!("  - Computation time: {:.4} seconds", solve_time);
    println!("  - Residual norm: {:.6e}", result.residual_norm);

    // Calculate errors
    let mut max_error: f64 = 0.0;
    let mut l2_error = 0.0;
    let mut total_area = 0.0;

    for (i, point) in mesh.points.iter().enumerate() {
        // Exact solution: sin(πx) sin(πy)
        let exact = (PI * point.x).sin() * (PI * point.y).sin();

        // Error at this node
        let error = (u[i] - exact).abs();
        max_error = max_error.max(error);

        // We'll compute an approximate L2 error
        // For more accuracy, we should integrate over each element
        l2_error += error * error;
    }

    // Compute area and average error
    for element in &mesh.elements {
        total_area += mesh.triangle_area(element);
    }

    // Approximate L2 error
    l2_error = (l2_error * total_area / mesh.points.len() as f64).sqrt();

    println!("\nError analysis:");
    println!("  - Maximum error: {:.6e}", max_error);
    println!("  - Approximate L2 error: {:.6e}", l2_error);

    // Print solution at selected points
    println!("\nSolution values at selected points:");
    println!(
        "{:<10} {:<10} {:<15} {:<15} {:<10}",
        "x", "y", "Numerical", "Exact", "Error"
    );

    // Find nodes near specific points of interest
    let points_of_interest = [
        (0.25, 0.25),
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
        (0.75, 0.75),
    ];

    for &(target_x, target_y) in &points_of_interest {
        // Find the closest node
        let mut closest_idx = 0;
        let mut min_dist = f64::MAX;

        for (i, point) in mesh.points.iter().enumerate() {
            let dist = ((point.x - target_x).powi(2) + (point.y - target_y).powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                closest_idx = i;
            }
        }

        let point = &mesh.points[closest_idx];
        let x = point.x;
        let y = point.y;

        let numerical = u[closest_idx];
        let exact = (PI * x).sin() * (PI * y).sin();
        let error = (numerical - exact).abs();

        println!(
            "{:<10.4} {:<10.4} {:<15.8e} {:<15.8e} {:<10.2e}",
            x, y, numerical, exact, error
        );
    }

    // Print convergence information
    println!("\nConvergence analysis:");
    println!("  - Number of nodes: {}", mesh.points.len());
    println!("  - Number of elements: {}", mesh.elements.len());
    println!("  - Element type: {:?}", ElementType::Linear);
    println!(
        "  - Maximum error for mesh size h ≈ {:.4}: {:.6e}",
        1.0 / nx as f64,
        max_error
    );

    Ok(())
}
