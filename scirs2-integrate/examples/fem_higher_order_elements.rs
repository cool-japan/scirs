//! Example: Higher-order finite elements for Poisson's equation
//!
//! This example demonstrates solving the Poisson equation:
//!   ∇²u = f(x,y)
//! on a unit square with different element types (linear, quadratic, cubic)
//! to show convergence improvement with higher-order elements.

use scirs2_integrate::pde::{
    finite_element::{ElementType, FEMOptions, FEMPoissonSolver, TriangularMesh},
    BoundaryCondition as GenericBoundaryCondition, BoundaryConditionType, BoundaryLocation,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Higher-Order Finite Elements for Poisson Equation ===");

    // Define the exact solution: u(x,y) = sin(π*x) * sin(π*y)
    // Then ∇²u = -2π²*sin(π*x)*sin(π*y) = -2π²*u
    // So f(x,y) = 2π²*sin(π*x)*sin(π*y)

    let pi = std::f64::consts::PI;
    let exact_solution = move |x: f64, y: f64| -> f64 { (pi * x).sin() * (pi * y).sin() };

    let source_term =
        move |x: f64, y: f64| -> f64 { 2.0 * pi * pi * (pi * x).sin() * (pi * y).sin() };

    // Test different mesh resolutions and element types
    let mesh_sizes = vec![4, 8, 16];
    let element_types = vec![
        (ElementType::Linear, "Linear"),
        (ElementType::Quadratic, "Quadratic"),
        (ElementType::Cubic, "Cubic"),
    ];

    println!("Comparing element types on Poisson equation with exact solution u = sin(πx)sin(πy)");
    println!("Domain: [0,1] × [0,1] with homogeneous Dirichlet boundary conditions");
    println!();

    for &(element_type, type_name) in &element_types {
        println!("=== {} Elements ===", type_name);
        println!("Mesh Size\\tNodes\\t\\tL2 Error\\t\\tMax Error\\t\\tRate");

        let mut prev_error: Option<f64> = None;

        for &nx in &mesh_sizes {
            // Create triangular mesh on unit square
            let mesh = TriangularMesh::generate_rectangular(
                (0.0, 1.0), // x range
                (0.0, 1.0), // y range
                nx,         // nx divisions
                nx,         // ny divisions
            );

            // Set up boundary conditions (homogeneous Dirichlet: u = 0 on all boundaries)
            let boundary_conditions = vec![
                GenericBoundaryCondition {
                    dimension: 0,                      // x-direction
                    location: BoundaryLocation::Lower, // left boundary
                    bc_type: BoundaryConditionType::Dirichlet,
                    value: 0.0,
                    coefficients: None,
                },
                GenericBoundaryCondition {
                    dimension: 0,                      // x-direction
                    location: BoundaryLocation::Upper, // right boundary
                    bc_type: BoundaryConditionType::Dirichlet,
                    value: 0.0,
                    coefficients: None,
                },
                GenericBoundaryCondition {
                    dimension: 1,                      // y-direction
                    location: BoundaryLocation::Lower, // bottom boundary
                    bc_type: BoundaryConditionType::Dirichlet,
                    value: 0.0,
                    coefficients: None,
                },
                GenericBoundaryCondition {
                    dimension: 1,                      // y-direction
                    location: BoundaryLocation::Upper, // top boundary
                    bc_type: BoundaryConditionType::Dirichlet,
                    value: 0.0,
                    coefficients: None,
                },
            ];

            // Configure solver options
            let quadrature_order = match element_type {
                ElementType::Linear => 3,    // 3 points sufficient for linear
                ElementType::Quadratic => 6, // 6 points for quadratic
                ElementType::Cubic => 12,    // 12 points for cubic
            };

            let options = FEMOptions {
                element_type,
                quadrature_order,
                max_iterations: 1000,
                tolerance: 1e-8,
                save_convergence_history: false,
                verbose: false,
            };

            // Create solver
            let mut solver =
                FEMPoissonSolver::new(mesh, source_term, boundary_conditions, Some(options))?;

            // Solve the system
            let result = solver.solve()?;

            // Compute error against exact solution
            let (l2_error, max_error) = compute_errors(&result, &exact_solution)?;

            // Compute convergence rate
            let rate: f64 = if let Some(prev_err) = prev_error {
                (prev_err / l2_error).log2()
            } else {
                0.0
            };

            // Count total number of nodes
            let n_nodes = result.u.len();

            println!(
                "{}x{}\\t\\t{}\\t\\t{:.2e}\\t\\t{:.2e}\\t\\t{:.2}",
                nx, nx, n_nodes, l2_error, max_error, rate
            );

            prev_error = Some(l2_error);
        }

        println!();
    }

    // Demonstrate convergence rates
    println!("=== Theoretical Convergence Rates ===");
    println!("Linear elements:    O(h²) in L² norm, O(h) in maximum norm");
    println!("Quadratic elements: O(h³) in L² norm, O(h²) in maximum norm");
    println!("Cubic elements:     O(h⁴) in L² norm, O(h³) in maximum norm");
    println!();
    println!("Where h is the mesh spacing. Higher-order elements should show");
    println!("faster convergence rates, especially on smooth solutions.");

    Ok(())
}

/// Compute L2 and maximum errors against exact solution
fn compute_errors(
    result: &scirs2_integrate::pde::finite_element::FEMResult,
    exact_solution: &dyn Fn(f64, f64) -> f64,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let mut l2_error_squared = 0.0;
    let mut max_error = 0.0;
    let mut _total_area = 0.0;

    // For each triangle in the mesh, compute the error
    for element in &result.mesh.elements {
        let [i, j, k] = element.nodes;
        let pi = &result.mesh.points[i];
        let pj = &result.mesh.points[j];
        let pk = &result.mesh.points[k];

        // Triangle area
        let area = result.mesh.triangle_area(element);
        _total_area += area;

        // Simple 3-point quadrature for error computation
        let quad_points = vec![
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), // centroid
            (0.5, 0.5, 0.0),                   // edge midpoints
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
        ];

        for &(a, b, c) in &quad_points {
            // Physical coordinates
            let x = a * pi.x + b * pj.x + c * pk.x;
            let y = a * pi.y + b * pj.y + c * pk.y;

            // Interpolated numerical solution (linear interpolation)
            let u_numerical = a * result.u[i] + b * result.u[j] + c * result.u[k];

            // Exact solution
            let u_exact = exact_solution(x, y);

            // Error
            let error = u_numerical - u_exact;
            let abs_error = error.abs();

            // Update maximum error
            if abs_error > max_error {
                max_error = abs_error;
            }

            // Update L2 error (approximate integration)
            l2_error_squared += error * error * area / 4.0; // /4 because 4 quadrature points
        }
    }

    let l2_error = l2_error_squared.sqrt();

    Ok((l2_error, max_error))
}
