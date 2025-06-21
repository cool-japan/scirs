use scirs2_integrate::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, ElementType, FEMOptions,
    FEMPoissonSolver, Point, Triangle, TriangularMesh,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("FEM solver example for irregular domain (L-shaped domain)");
    println!("Solving: ∇²u = f");
    println!("Domain: L-shaped domain (0,1)×(0,1) minus (0.5,1)×(0.5,1)");
    println!("Boundary conditions: u = 0 on outer boundaries");

    // Create an L-shaped domain
    let mesh = create_l_shaped_mesh(20);

    println!("Mesh generated with:");
    println!("  - {} nodes", mesh.points.len());
    println!("  - {} triangular elements", mesh.elements.len());
    println!("  - {} boundary edges", mesh.boundary_edges.len());

    // Source term: 1.0 (simplified Poisson problem)
    let source_term = |_x: f64, _y: f64| -> f64 { 1.0 };

    // Boundary conditions: u = 0 on all boundaries (Dirichlet)
    let bcs = vec![
        // All boundaries have Dirichlet condition u = 0
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 1, // y dimension (bottom)
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 0, // x dimension (right)
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Upper,
            dimension: 1, // y dimension (top)
            value: 0.0,
            coefficients: None,
        },
        BoundaryCondition {
            bc_type: BoundaryConditionType::Dirichlet,
            location: BoundaryLocation::Lower,
            dimension: 0, // x dimension (left)
            value: 0.0,
            coefficients: None,
        },
    ];

    // Solver options
    let options = FEMOptions {
        element_type: ElementType::Linear,
        quadrature_order: 3,
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

    // Find maximum solution value and its location
    let mut max_u = 0.0;
    let mut max_u_x = 0.0;
    let mut max_u_y = 0.0;

    for (i, point) in mesh.points.iter().enumerate() {
        if u[i] > max_u {
            max_u = u[i];
            max_u_x = point.x;
            max_u_y = point.y;
        }
    }

    println!("\nSolution analysis:");
    println!("  - Maximum solution value: {:.6e}", max_u);
    println!("  - Location of maximum: ({:.4}, {:.4})", max_u_x, max_u_y);

    // Print solution at selected points (e.g., along a diagonal)
    println!("\nSolution along y = x diagonal:");
    println!("{:<10} {:<10} {:<15}", "x", "y", "u(x,y)");

    // Points along the diagonal y = x, for x in [0, 0.5]
    for x in (0..=10).map(|i| i as f64 * 0.05) {
        // Find the closest node to (x, x)
        let mut closest_idx = 0;
        let mut min_dist = f64::MAX;

        for (i, point) in mesh.points.iter().enumerate() {
            let dist = ((point.x - x).powi(2) + (point.y - x).powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                closest_idx = i;
            }
        }

        let point = &mesh.points[closest_idx];
        let value = u[closest_idx];

        println!("{:<10.4} {:<10.4} {:<15.8e}", point.x, point.y, value);
    }

    // Print solution along the inner corner (re-entrant corner)
    println!("\nSolution near the re-entrant corner (0.5, 0.5):");
    println!(
        "{:<10} {:<10} {:<15} {:<15}",
        "Distance", "Angle", "Point (x,y)", "u(x,y)"
    );

    // Calculate solution values around the re-entrant corner
    let corner_x = 0.5;
    let corner_y = 0.5;
    let radii = [0.05, 0.1, 0.15];
    let angles = [0.0, PI / 8.0, PI / 4.0, 3.0 * PI / 8.0, PI / 2.0];

    for &r in &radii {
        for &theta in &angles {
            let x = corner_x + r * theta.cos();
            let y = corner_y + r * theta.sin();

            // Only consider points inside the L-shaped domain
            if x <= 0.5 || y <= 0.5 {
                // Find the closest node
                let mut closest_idx = 0;
                let mut min_dist = f64::MAX;

                for (i, point) in mesh.points.iter().enumerate() {
                    let dist = ((point.x - x).powi(2) + (point.y - y).powi(2)).sqrt();
                    if dist < min_dist {
                        min_dist = dist;
                        closest_idx = i;
                    }
                }

                let point = &mesh.points[closest_idx];
                let value = u[closest_idx];

                println!(
                    "{:<10.4} {:<10.4} ({:.4}, {:.4}) {:<15.8e}",
                    r, theta, point.x, point.y, value
                );
            }
        }
        println!(); // Empty line between radii
    }

    Ok(())
}

/// Create a mesh for an L-shaped domain
/// The domain is [0,1]×[0,1] minus [0.5,1]×[0.5,1]
fn create_l_shaped_mesh(divisions: usize) -> TriangularMesh {
    let mut mesh = TriangularMesh::new();

    // Calculate step size
    let h = 1.0 / divisions as f64;

    // Create nodes for the L-shaped domain
    // We first create a uniform grid, then remove nodes in the upper-right corner

    // Node indices (i,j) -> flattened index
    // For bookkeeping during mesh creation
    let mut node_map = vec![vec![None; divisions + 1]; divisions + 1];
    let mut node_count = 0;

    // Create nodes
    #[allow(clippy::needless_range_loop)]
    for j in 0..=divisions {
        for i in 0..=divisions {
            let x = i as f64 * h;
            let y = j as f64 * h;

            // Skip points in the upper-right corner (where x > 0.5 and y > 0.5)
            if x > 0.5 + 1e-10 && y > 0.5 + 1e-10 {
                continue;
            }

            // Add node
            mesh.points.push(Point::new(x, y));
            node_map[j][i] = Some(node_count);
            node_count += 1;
        }
    }

    // Create triangular elements
    for j in 0..divisions {
        for i in 0..divisions {
            // Skip cells in the upper-right corner
            if i >= divisions / 2 && j >= divisions / 2 {
                continue;
            }

            // Node indices of the cell
            let n00 = node_map[j][i];
            let n10 = node_map[j][i + 1];
            let n01 = node_map[j + 1][i];
            let n11 = node_map[j + 1][i + 1];

            // Create two triangles per grid cell
            if let (Some(n00), Some(n10), Some(n01), Some(n11)) = (n00, n10, n01, n11) {
                // Triangle 1: Bottom-left, Bottom-right, Top-left
                mesh.elements.push(Triangle::new([n00, n10, n01], None));

                // Triangle 2: Top-right, Top-left, Bottom-right
                mesh.elements.push(Triangle::new([n11, n01, n10], None));
            }
        }
    }

    // Create boundary edges

    // Bottom edge (y = 0)
    for i in 0..divisions {
        let n1 = node_map[0][i].unwrap();
        let n2 = node_map[0][i + 1].unwrap();
        mesh.boundary_edges.push((n1, n2, Some(1))); // Marker 1 for bottom
    }

    // Right edges (there are two separated by the corner)
    // Lower portion (x = 1, y <= 0.5)
    for j in 0..divisions / 2 {
        let n1 = node_map[j][divisions].unwrap();
        let n2 = node_map[j + 1][divisions].unwrap();
        mesh.boundary_edges.push((n1, n2, Some(2))); // Marker 2 for right
    }
    // Upper portion (x = 0.5, y > 0.5)
    for j in divisions / 2..divisions {
        let n1 = node_map[j][divisions / 2].unwrap();
        let n2 = node_map[j + 1][divisions / 2].unwrap();
        mesh.boundary_edges.push((n1, n2, Some(5))); // Marker 5 for inner vertical
    }

    // Top edges (there are two separated by the corner)
    // Left portion (y = 1, x <= 0.5)
    for i in 0..divisions / 2 {
        let n1 = node_map[divisions][i + 1].unwrap();
        let n2 = node_map[divisions][i].unwrap();
        mesh.boundary_edges.push((n1, n2, Some(3))); // Marker 3 for top
    }
    // Right portion (y = 0.5, x > 0.5)
    for i in divisions / 2..divisions {
        let n1 = node_map[divisions / 2][i + 1].unwrap();
        let n2 = node_map[divisions / 2][i].unwrap();
        mesh.boundary_edges.push((n1, n2, Some(6))); // Marker 6 for inner horizontal
    }

    // Left edge (x = 0)
    for j in 0..divisions {
        let n1 = node_map[j + 1][0].unwrap();
        let n2 = node_map[j][0].unwrap();
        mesh.boundary_edges.push((n1, n2, Some(4))); // Marker 4 for left
    }

    mesh
}
