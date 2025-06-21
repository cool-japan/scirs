//! Example: Solving Poisson equation on irregular domains with ghost points
//!
//! This example demonstrates how to solve the Poisson equation:
//!   ∇²u = f(x,y)
//! on a circular domain with Dirichlet boundary conditions using
//! the irregular domain finite difference method with ghost points.

use ndarray::{Array1, Array2};
use scirs2_integrate::pde::finite_difference::{BoundaryCondition, IrregularGrid, PointType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Poisson Equation on Circular Domain ===");

    // Define the circular domain: x² + y² ≤ R²
    let radius = 1.0;
    let domain_function =
        Box::new(move |x: f64, y: f64| -> bool { x * x + y * y <= radius * radius });

    // Create irregular grid
    let mut grid = IrregularGrid::new(
        (-1.2, 1.2), // x range (slightly larger than domain)
        (-1.2, 1.2), // y range
        25,          // nx points
        25,          // ny points
        domain_function,
    )?;

    println!(
        "Grid created with {} interior/boundary points",
        grid.count_interior_points()
    );

    // Set up boundary conditions on the circular boundary
    // u = sin(2π(x² + y²)) on ∂Ω
    let mut boundary_count = 0;
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let point = &grid.points[[j, i]];
            if point.point_type == PointType::Boundary {
                let (x, y) = point.coords;
                let boundary_value = (2.0 * std::f64::consts::PI * (x * x + y * y)).sin();
                let bc = BoundaryCondition::Dirichlet(boundary_value);
                grid.set_boundary_condition(i, j, bc)?;
                boundary_count += 1;
            }
        }
    }

    println!(
        "Set boundary conditions on {} boundary points",
        boundary_count
    );

    // Create the Laplacian matrix for the irregular domain
    let laplacian_matrix = grid.create_laplacian_matrix()?;
    println!(
        "Laplacian matrix size: {}x{}",
        laplacian_matrix.shape()[0],
        laplacian_matrix.shape()[1]
    );

    // Define the right-hand side: f(x,y) = -8π²sin(2π(x² + y²)) + 4π cos(2π(x² + y²))
    let n_points = grid.count_interior_points();
    let mut rhs = Array1::zeros(n_points);

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let point = &grid.points[[j, i]];
            if point.solution_index >= 0 {
                let (x, y) = point.coords;
                let r_squared = x * x + y * y;
                let idx = point.solution_index as usize;

                // Analytical RHS for exact solution u = sin(2π(x² + y²))
                rhs[idx] = -8.0
                    * std::f64::consts::PI.powi(2)
                    * (2.0 * std::f64::consts::PI * r_squared).sin()
                    + 4.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * r_squared).cos();
            }
        }
    }

    // For this example, we'll use a simple iterative solver (Jacobi method)
    // In practice, you would use a more sophisticated solver like CG or multigrid
    let solution = solve_jacobi(&laplacian_matrix, &rhs, 1000, 1e-6)?;

    println!("Solution computed with {} iterations", 1000);

    // Extract solution back to 2D grid
    let solution_2d = grid.extract_domain_solution(&solution);

    // Compute error against analytical solution
    let mut max_error = 0.0;
    let mut error_count = 0;

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let point = &grid.points[[j, i]];
            if point.solution_index >= 0 {
                let (x, y) = point.coords;
                let numerical = solution_2d[[j, i]];
                let analytical = (2.0 * std::f64::consts::PI * (x * x + y * y)).sin();
                let error = (numerical - analytical).abs();

                if error > max_error {
                    max_error = error;
                }
                error_count += 1;
            }
        }
    }

    println!(
        "Maximum error: {:.2e} (over {} points)",
        max_error, error_count
    );

    // Print a cross-section of the solution
    println!("\nSolution cross-section along y=0:");
    println!("x\t\tNumerical\tAnalytical\tError");

    let center_j = grid.ny / 2;
    for i in 0..grid.nx {
        let point = &grid.points[[center_j, i]];
        if point.solution_index >= 0 {
            let (x, _y) = point.coords;
            if x.abs() < 0.9 {
                // Only print points well inside the domain
                let numerical = solution_2d[[center_j, i]];
                let analytical = (2.0 * std::f64::consts::PI * (x * x)).sin();
                let error = (numerical - analytical).abs();

                println!(
                    "{:.3}\t\t{:.6}\t{:.6}\t{:.2e}",
                    x, numerical, analytical, error
                );
            }
        }
    }

    println!("\n=== Irregular Domain Analysis ===");

    // Analyze domain characteristics
    let mut interior_count = 0;
    let mut boundary_count = 0;
    let mut ghost_count = 0;

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            match grid.points[[j, i]].point_type {
                PointType::Interior => interior_count += 1,
                PointType::Boundary => boundary_count += 1,
                PointType::Ghost => ghost_count += 1,
                _ => {}
            }
        }
    }

    println!("Domain statistics:");
    println!("  Interior points: {}", interior_count);
    println!("  Boundary points: {}", boundary_count);
    println!("  Ghost points: {}", ghost_count);
    println!("  Total grid points: {}", grid.nx * grid.ny);
    println!(
        "  Domain coverage: {:.1}%",
        100.0 * (interior_count + boundary_count) as f64 / (grid.nx * grid.ny) as f64
    );

    Ok(())
}

/// Simple Jacobi iterative solver for demonstration
/// In practice, use more sophisticated methods like CG, GMRES, or multigrid
fn solve_jacobi(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let n = matrix.shape()[0];
    let mut x = Array1::zeros(n);
    let mut x_new = Array1::zeros(n);

    for _iter in 0..max_iterations {
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += matrix[[i, j]] * x[j];
                }
            }
            x_new[i] = (rhs[i] - sum) / matrix[[i, i]];
        }

        // Check convergence
        let mut residual_norm = 0.0f64;
        for i in 0..n {
            let diff = x_new[i] - x[i];
            residual_norm += diff * diff;
        }
        residual_norm = residual_norm.sqrt();

        if residual_norm < tolerance {
            break;
        }

        x.assign(&x_new);
    }

    Ok(x_new)
}
