//! Finite Element Method (FEM) for solving PDEs
//!
//! This module provides implementations of the Finite Element Method for
//! solving partial differential equations on structured and unstructured meshes.
//!
//! Key features:
//! - Linear, quadratic, and cubic element types
//! - Triangular elements for 2D problems
//! - Mesh generation and manipulation
//! - Support for irregular domains
//! - Various boundary condition types

pub mod higher_order;

#[cfg(test)]
mod higher_order_tests;

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::Instant;

use crate::pde::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, PDEError, PDEResult, PDESolution,
    PDESolverInfo,
};

// Re-export higher-order functionality
pub use higher_order::{
    HigherOrderMeshGenerator, HigherOrderTriangle, ShapeFunctions, TriangularQuadrature,
};

/// A point in 2D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// x-coordinate
    pub x: f64,

    /// y-coordinate
    pub y: f64,
}

impl Point {
    /// Create a new point
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    /// Calculate the distance to another point
    pub fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// A triangle element defined by three nodes
#[derive(Debug, Clone)]
pub struct Triangle {
    /// Node indices (vertices of the triangle)
    pub nodes: [usize; 3],

    /// Marker for domain/boundary identification
    pub marker: Option<i32>,
}

impl Triangle {
    /// Create a new triangle
    pub fn new(nodes: [usize; 3], marker: Option<i32>) -> Self {
        Triangle { nodes, marker }
    }
}

/// A mesh of triangular elements
#[derive(Debug, Clone)]
pub struct TriangularMesh {
    /// Points/nodes in the mesh
    pub points: Vec<Point>,

    /// Triangular elements
    pub elements: Vec<Triangle>,

    /// Boundary edges (node indices for each edge)
    pub boundary_edges: Vec<(usize, usize, Option<i32>)>,

    /// Map from node index to its boundary condition type (if on boundary)
    pub boundary_nodes: HashMap<usize, BoundaryNodeInfo>,
}

/// Information about a boundary node
#[derive(Debug, Clone)]
pub struct BoundaryNodeInfo {
    /// Boundary type
    pub bc_type: BoundaryConditionType,

    /// Value for Dirichlet or flux for Neumann boundaries
    pub value: f64,

    /// Additional coefficients for Robin boundaries
    pub coefficients: Option<[f64; 3]>,

    /// Marker for boundary identification
    pub marker: Option<i32>,
}

impl Default for TriangularMesh {
    fn default() -> Self {
        Self::new()
    }
}

impl TriangularMesh {
    /// Create a new empty triangular mesh
    pub fn new() -> Self {
        TriangularMesh {
            points: Vec::new(),
            elements: Vec::new(),
            boundary_edges: Vec::new(),
            boundary_nodes: HashMap::new(),
        }
    }

    /// Generate a simple triangular mesh on a rectangular domain
    pub fn generate_rectangular(
        x_range: (f64, f64),
        y_range: (f64, f64),
        nx: usize,
        ny: usize,
    ) -> Self {
        let mut mesh = TriangularMesh::new();

        // Generate grid points
        let dx = (x_range.1 - x_range.0) / (nx as f64);
        let dy = (y_range.1 - y_range.0) / (ny as f64);

        // Create points
        for j in 0..=ny {
            for i in 0..=nx {
                let x = x_range.0 + i as f64 * dx;
                let y = y_range.0 + j as f64 * dy;
                mesh.points.push(Point::new(x, y));
            }
        }

        // Create triangular elements
        for j in 0..ny {
            for i in 0..nx {
                // Node indices at the corners of the grid cell
                let n00 = j * (nx + 1) + i; // Bottom-left
                let n10 = j * (nx + 1) + (i + 1); // Bottom-right
                let n01 = (j + 1) * (nx + 1) + i; // Top-left
                let n11 = (j + 1) * (nx + 1) + (i + 1); // Top-right

                // Create two triangles per grid cell
                // Triangle 1: Bottom-left, Bottom-right, Top-left
                mesh.elements.push(Triangle::new([n00, n10, n01], None));

                // Triangle 2: Top-right, Top-left, Bottom-right
                mesh.elements.push(Triangle::new([n11, n01, n10], None));
            }
        }

        // Identify boundary edges

        // Bottom edge (y = y_range.0)
        for i in 0..nx {
            let n1 = i;
            let n2 = i + 1;
            mesh.boundary_edges.push((n1, n2, Some(1))); // Marker 1 for bottom
        }

        // Right edge (x = x_range.1)
        for j in 0..ny {
            let n1 = (j + 1) * (nx + 1) - 1;
            let n2 = (j + 2) * (nx + 1) - 1;
            mesh.boundary_edges.push((n1, n2, Some(2))); // Marker 2 for right
        }

        // Top edge (y = y_range.1)
        for i in 0..nx {
            let n1 = (ny + 1) * (nx + 1) - i - 1;
            let n2 = (ny + 1) * (nx + 1) - i - 2;
            mesh.boundary_edges.push((n1, n2, Some(3))); // Marker 3 for top
        }

        // Left edge (x = x_range.0)
        for j in 0..ny {
            let n1 = (ny - j) * (nx + 1);
            let n2 = (ny - j - 1) * (nx + 1);
            mesh.boundary_edges.push((n1, n2, Some(4))); // Marker 4 for left
        }

        mesh
    }

    /// Set boundary conditions based on boundary markers
    pub fn set_boundary_conditions(
        &mut self,
        boundary_conditions: &[BoundaryCondition<f64>],
    ) -> PDEResult<()> {
        // Clear existing boundary nodes
        self.boundary_nodes.clear();

        // Process all boundary edges
        for &(n1, n2, marker) in &self.boundary_edges {
            // Find the matching boundary condition by marker
            for bc in boundary_conditions {
                // Map dimension and location to marker (simplified approach for example)
                let bc_marker = match (bc.dimension, bc.location) {
                    (1, BoundaryLocation::Lower) => Some(1), // Bottom
                    (0, BoundaryLocation::Upper) => Some(2), // Right
                    (1, BoundaryLocation::Upper) => Some(3), // Top
                    (0, BoundaryLocation::Lower) => Some(4), // Left
                    _ => None,
                };

                // If this boundary condition matches the edge marker
                if bc_marker == marker {
                    // Add both nodes of the edge to boundary_nodes
                    let bc_info = BoundaryNodeInfo {
                        bc_type: bc.bc_type,
                        value: bc.value,
                        coefficients: bc.coefficients,
                        marker,
                    };

                    self.boundary_nodes.insert(n1, bc_info.clone());
                    self.boundary_nodes.insert(n2, bc_info);
                }
            }
        }

        Ok(())
    }

    /// Compute area of a triangle
    pub fn triangle_area(&self, element: &Triangle) -> f64 {
        let [i, j, k] = element.nodes;
        let pi = &self.points[i];
        let pj = &self.points[j];
        let pk = &self.points[k];

        // Area using cross product
        0.5 * ((pj.x - pi.x) * (pk.y - pi.y) - (pk.x - pi.x) * (pj.y - pi.y)).abs()
    }

    /// Compute shape function gradients for a linear triangular element
    pub fn shape_function_gradients(&self, element: &Triangle) -> PDEResult<[Point; 3]> {
        let [i, j, k] = element.nodes;
        let pi = &self.points[i];
        let pj = &self.points[j];
        let pk = &self.points[k];

        let area = self.triangle_area(element);
        if area < 1e-10 {
            return Err(PDEError::FiniteElementError(format!(
                "Element has nearly zero area: {}",
                area
            )));
        }

        // Linear shape function gradients
        let gradients = [
            Point::new((pj.y - pk.y) / (2.0 * area), (pk.x - pj.x) / (2.0 * area)),
            Point::new((pk.y - pi.y) / (2.0 * area), (pi.x - pk.x) / (2.0 * area)),
            Point::new((pi.y - pj.y) / (2.0 * area), (pj.x - pi.x) / (2.0 * area)),
        ];

        Ok(gradients)
    }
}

/// Element type for finite element method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementType {
    /// Linear elements (1st order, 3 nodes for triangles)
    Linear,

    /// Quadratic elements (2nd order, 6 nodes for triangles)
    Quadratic,

    /// Cubic elements (3rd order, 10 nodes for triangles)
    Cubic,
}

/// Options for finite element solvers
#[derive(Debug, Clone)]
pub struct FEMOptions {
    /// Element type to use
    pub element_type: ElementType,

    /// Quadrature rule order (number of integration points)
    pub quadrature_order: usize,

    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,

    /// Tolerance for convergence
    pub tolerance: f64,

    /// Whether to save convergence history
    pub save_convergence_history: bool,

    /// Print detailed progress information
    pub verbose: bool,
}

impl Default for FEMOptions {
    fn default() -> Self {
        FEMOptions {
            element_type: ElementType::Linear,
            quadrature_order: 3, // 3-point rule suitable for quadratic functions
            max_iterations: 1000,
            tolerance: 1e-6,
            save_convergence_history: false,
            verbose: false,
        }
    }
}

/// Result of FEM solution
#[derive(Debug, Clone)]
pub struct FEMResult {
    /// Solution values at nodes
    pub u: Array1<f64>,

    /// Mesh used for the solution
    pub mesh: TriangularMesh,

    /// Residual norm
    pub residual_norm: f64,

    /// Number of iterations performed
    pub num_iterations: usize,

    /// Computation time
    pub computation_time: f64,

    /// Convergence history
    pub convergence_history: Option<Vec<f64>>,
}

/// Finite Element solver for Poisson's equation
pub struct FEMPoissonSolver {
    /// Mesh for finite element discretization
    mesh: TriangularMesh,

    /// Higher-order elements (if using non-linear elements)
    higher_order_elements: Option<Vec<HigherOrderTriangle>>,

    /// Additional points for higher-order elements
    higher_order_points: Option<Vec<Point>>,

    /// Source term function f(x, y)
    source_term: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,

    /// Boundary conditions
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Solver options
    options: FEMOptions,
}

impl FEMPoissonSolver {
    /// Create a new Finite Element solver for Poisson's equation
    pub fn new(
        mesh: TriangularMesh,
        source_term: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<FEMOptions>,
    ) -> PDEResult<Self> {
        // Validate boundary conditions
        if boundary_conditions.is_empty() {
            return Err(PDEError::BoundaryConditions(
                "At least one boundary condition is required".to_string(),
            ));
        }

        let opts = options.unwrap_or_default();

        // Create higher-order elements if needed
        let (higher_order_elements, higher_order_points) = match opts.element_type {
            ElementType::Linear => (None, None),
            ElementType::Quadratic => {
                let (points, elements) = HigherOrderMeshGenerator::linear_to_quadratic(&mesh)?;
                (Some(elements), Some(points))
            }
            ElementType::Cubic => {
                let (points, elements) = HigherOrderMeshGenerator::linear_to_cubic(&mesh)?;
                (Some(elements), Some(points))
            }
        };

        Ok(FEMPoissonSolver {
            mesh,
            higher_order_elements,
            higher_order_points,
            source_term: Box::new(source_term),
            boundary_conditions,
            options: opts,
        })
    }

    /// Solve Poisson's equation using the Finite Element Method
    pub fn solve(&mut self) -> PDEResult<FEMResult> {
        let start_time = Instant::now();

        // Apply boundary conditions to the mesh
        self.mesh
            .set_boundary_conditions(&self.boundary_conditions)?;

        // Number of nodes (degrees of freedom)
        let _n = if let Some(ref higher_order_points) = self.higher_order_points {
            higher_order_points.len()
        } else {
            self.mesh.points.len()
        };

        // Assemble stiffness matrix and load vector
        let (mut a, mut b) = self.assemble_system()?;

        // Apply Dirichlet boundary conditions
        self.apply_dirichlet_boundary_conditions(&mut a, &mut b)?;

        // Solve the linear system
        let u = self.solve_linear_system(&a, &b)?;

        // Compute residual norm
        let residual_norm = self.compute_residual(&a, &b, &u);

        let computation_time = start_time.elapsed().as_secs_f64();

        Ok(FEMResult {
            u,
            mesh: self.mesh.clone(),
            residual_norm,
            num_iterations: 1, // Direct solver counts as one iteration
            computation_time,
            convergence_history: None,
        })
    }

    /// Assemble the stiffness matrix and load vector for the FEM system
    fn assemble_system(&self) -> PDEResult<(Array2<f64>, Array1<f64>)> {
        let n = if let Some(ref higher_order_points) = self.higher_order_points {
            higher_order_points.len()
        } else {
            self.mesh.points.len()
        };

        // Initialize stiffness matrix and load vector
        let mut a = Array2::zeros((n, n));
        let mut b = Array1::zeros(n);

        match self.options.element_type {
            ElementType::Linear => {
                // Use existing linear element assembly
                for element in &self.mesh.elements {
                    let (a_e, b_e) = self.element_matrices_linear(element)?;

                    // Assemble into global matrices
                    let [i, j, k] = element.nodes;

                    // Diagonal terms
                    a[[i, i]] += a_e[0][0];
                    a[[j, j]] += a_e[1][1];
                    a[[k, k]] += a_e[2][2];

                    // Off-diagonal terms
                    a[[i, j]] += a_e[0][1];
                    a[[i, k]] += a_e[0][2];
                    a[[j, i]] += a_e[1][0];
                    a[[j, k]] += a_e[1][2];
                    a[[k, i]] += a_e[2][0];
                    a[[k, j]] += a_e[2][1];

                    // Load vector
                    b[i] += b_e[0];
                    b[j] += b_e[1];
                    b[k] += b_e[2];
                }
            }
            _ => {
                // Use higher-order element assembly
                if let Some(ref higher_order_elements) = self.higher_order_elements {
                    for element in higher_order_elements {
                        let (a_e, b_e) = self.element_matrices_higher_order(element)?;

                        // Assemble into global matrices
                        for (i, node_i) in element.nodes.iter().enumerate() {
                            b[*node_i] += b_e[i];
                            for (j, node_j) in element.nodes.iter().enumerate() {
                                a[[*node_i, *node_j]] += a_e[[i, j]];
                            }
                        }
                    }
                }
            }
        }

        Ok((a, b))
    }

    /// Compute element stiffness matrix and load vector for linear elements
    fn element_matrices_linear(&self, element: &Triangle) -> PDEResult<([[f64; 3]; 3], [f64; 3])> {
        // Get nodes
        let [i, j, k] = element.nodes;
        let pi = &self.mesh.points[i];
        let pj = &self.mesh.points[j];
        let pk = &self.mesh.points[k];

        // Element area
        let area = self.mesh.triangle_area(element);

        // Shape function gradients
        let gradients = self.mesh.shape_function_gradients(element)?;

        // Stiffness matrix - For Poisson's equation: Integral of (∇φᵢ · ∇φⱼ) over element
        let mut a_e = [[0.0; 3]; 3];

        for m in 0..3 {
            for n in 0..3 {
                // Dot product of shape function gradients
                a_e[m][n] =
                    (gradients[m].x * gradients[n].x + gradients[m].y * gradients[n].y) * area;
            }
        }

        // Load vector - For Poisson's equation: Integral of (f · φᵢ) over element
        let mut b_e = [0.0; 3];

        // Approximate the source term at the centroid of the triangle
        let centroid_x = (pi.x + pj.x + pk.x) / 3.0;
        let centroid_y = (pi.y + pj.y + pk.y) / 3.0;
        let f_centroid = (self.source_term)(centroid_x, centroid_y);

        // For linear elements, the integral of each shape function over the element is area/3
        b_e.iter_mut().for_each(|value| {
            *value = f_centroid * (area / 3.0);
        });

        Ok((a_e, b_e))
    }

    /// Compute element stiffness matrix and load vector for higher-order elements
    fn element_matrices_higher_order(
        &self,
        element: &HigherOrderTriangle,
    ) -> PDEResult<(Array2<f64>, Array1<f64>)> {
        let num_nodes = element.nodes.len();
        let mut a_e = Array2::zeros((num_nodes, num_nodes));
        let mut b_e = Array1::zeros(num_nodes);

        // Get the points for this element type
        let points = if let Some(ref ho_points) = self.higher_order_points {
            ho_points
        } else {
            return Err(PDEError::FiniteElementError(
                "Higher-order points not available".to_string(),
            ));
        };

        // Get corner nodes to compute element area and coordinate transformation
        let corner_nodes = element.corner_nodes();
        let p1 = &points[corner_nodes[0]];
        let p2 = &points[corner_nodes[1]];
        let p3 = &points[corner_nodes[2]];

        // Compute Jacobian for coordinate transformation from reference to physical element
        let jacobian = Array2::from_shape_vec(
            (2, 2),
            vec![p2.x - p1.x, p3.x - p1.x, p2.y - p1.y, p3.y - p1.y],
        )
        .unwrap();

        let det_j = jacobian[[0, 0]] * jacobian[[1, 1]] - jacobian[[0, 1]] * jacobian[[1, 0]];
        if det_j.abs() < 1e-12 {
            return Err(PDEError::FiniteElementError(
                "Degenerate element with zero Jacobian determinant".to_string(),
            ));
        }

        // Inverse of Jacobian
        let inv_j = Array2::from_shape_vec(
            (2, 2),
            vec![
                jacobian[[1, 1]] / det_j,
                -jacobian[[0, 1]] / det_j,
                -jacobian[[1, 0]] / det_j,
                jacobian[[0, 0]] / det_j,
            ],
        )
        .unwrap();

        // Get quadrature rule
        let (xi_coords, eta_coords, weights) =
            TriangularQuadrature::get_rule(self.options.quadrature_order)?;

        // Integrate over the element using quadrature
        for q in 0..xi_coords.len() {
            let xi = xi_coords[q];
            let eta = eta_coords[q];
            let weight = weights[q];

            // Evaluate shape functions and their derivatives at quadrature point
            let shape_funcs = ShapeFunctions::evaluate(element.element_type, xi, eta)?;
            let (d_n_dxi, d_n_deta) =
                ShapeFunctions::evaluate_derivatives(element.element_type, xi, eta)?;

            // Transform derivatives from reference to physical coordinates
            let mut d_n_dx = Array1::zeros(num_nodes);
            let mut d_n_dy = Array1::zeros(num_nodes);

            for i in 0..num_nodes {
                d_n_dx[i] = inv_j[[0, 0]] * d_n_dxi[i] + inv_j[[0, 1]] * d_n_deta[i];
                d_n_dy[i] = inv_j[[1, 0]] * d_n_dxi[i] + inv_j[[1, 1]] * d_n_deta[i];
            }

            // Compute physical coordinates of quadrature point for source term evaluation
            let mut x_phys = 0.0;
            let mut y_phys = 0.0;
            for i in 0..num_nodes {
                x_phys += shape_funcs[i] * points[element.nodes[i]].x;
                y_phys += shape_funcs[i] * points[element.nodes[i]].y;
            }

            // Evaluate source term at quadrature point
            let f_val = (self.source_term)(x_phys, y_phys);

            // Add contributions to element matrices
            for i in 0..num_nodes {
                // Load vector: ∫ f * N_i * dV
                b_e[i] += f_val * shape_funcs[i] * weight * det_j.abs();

                for j in 0..num_nodes {
                    // Stiffness matrix: ∫ (∇N_i · ∇N_j) * dV
                    a_e[[i, j]] +=
                        (d_n_dx[i] * d_n_dx[j] + d_n_dy[i] * d_n_dy[j]) * weight * det_j.abs();
                }
            }
        }

        Ok((a_e, b_e))
    }

    /// Apply Dirichlet boundary conditions to the system
    fn apply_dirichlet_boundary_conditions(
        &self,
        a: &mut Array2<f64>,
        b: &mut Array1<f64>,
    ) -> PDEResult<()> {
        let n = self.mesh.points.len();

        // Loop over boundary nodes
        for (&node_idx, bc_info) in &self.mesh.boundary_nodes {
            if bc_info.bc_type == BoundaryConditionType::Dirichlet {
                // Set row to identity
                for j in 0..n {
                    a[[node_idx, j]] = 0.0;
                }
                a[[node_idx, node_idx]] = 1.0;

                // Set right-hand side to boundary value
                b[node_idx] = bc_info.value;
            } else if bc_info.bc_type == BoundaryConditionType::Neumann {
                // Neumann boundary conditions are handled in the assembly process
                // For linear elements on a flat boundary, this is equivalent to
                // modifying the right-hand side vector

                // Get all boundary edges containing this node
                let boundary_edges: Vec<_> = self
                    .mesh
                    .boundary_edges
                    .iter()
                    .filter(|&&(n1, n2, _)| n1 == node_idx || n2 == node_idx)
                    .collect();

                // For each boundary edge, apply the Neumann condition
                for &(n1, n2, _) in &boundary_edges {
                    let other_node = if *n1 == node_idx { *n2 } else { *n1 };

                    // Get the coordinates of the nodes
                    let p1 = &self.mesh.points[node_idx];
                    let p2 = &self.mesh.points[other_node];

                    // Length of the edge
                    let edge_length = p1.distance(p2);

                    // Contribution to the load vector: g * (edge_length / 2)
                    // where g is the Neumann boundary value
                    b[node_idx] += bc_info.value * (edge_length / 2.0);
                }
            } else if bc_info.bc_type == BoundaryConditionType::Robin {
                // Robin boundary conditions (a*u + b*∂u/∂n = c)
                if let Some([a_coef, _b_coef, c_coef]) = bc_info.coefficients {
                    // Similar to Neumann, we need to find boundary edges
                    let boundary_edges: Vec<_> = self
                        .mesh
                        .boundary_edges
                        .iter()
                        .filter(|&&(n1, n2, _)| n1 == node_idx || n2 == node_idx)
                        .collect();

                    for &(n1, n2, _) in &boundary_edges {
                        let other_node = if *n1 == node_idx { *n2 } else { *n1 };

                        // Get the coordinates of the nodes
                        let p1 = &self.mesh.points[node_idx];
                        let p2 = &self.mesh.points[other_node];

                        // Length of the edge
                        let edge_length = p1.distance(p2);

                        // Contribution to the stiffness matrix and load vector
                        // This is simplified - a more accurate implementation would
                        // involve integrating along the boundary edge
                        a[[node_idx, node_idx]] += a_coef * edge_length / 2.0;

                        // Right-hand side contribution
                        b[node_idx] += c_coef * edge_length / 2.0;
                    }
                }
            }
        }

        Ok(())
    }

    /// Solve the linear system Ax = b
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> PDEResult<Array1<f64>> {
        let n = b.len();

        // Simple Gaussian elimination for demonstration purposes
        // For a real implementation, use a sparse matrix solver library

        // Create copies of A and b
        let mut a_copy = a.clone();
        let mut b_copy = b.clone();

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_val = a_copy[[i, i]].abs();
            let mut max_row = i;

            for k in i + 1..n {
                if a_copy[[k, i]].abs() > max_val {
                    max_val = a_copy[[k, i]].abs();
                    max_row = k;
                }
            }

            // Check if matrix is singular
            if max_val < 1e-10 {
                return Err(PDEError::Other(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Swap rows if necessary
            if max_row != i {
                for j in i..n {
                    let temp = a_copy[[i, j]];
                    a_copy[[i, j]] = a_copy[[max_row, j]];
                    a_copy[[max_row, j]] = temp;
                }

                let temp = b_copy[i];
                b_copy[i] = b_copy[max_row];
                b_copy[max_row] = temp;
            }

            // Eliminate below
            for k in i + 1..n {
                let factor = a_copy[[k, i]] / a_copy[[i, i]];

                for j in i..n {
                    a_copy[[k, j]] -= factor * a_copy[[i, j]];
                }

                b_copy[k] -= factor * b_copy[i];
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += a_copy[[i, j]] * x[j];
            }

            x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
        }

        Ok(x)
    }

    /// Compute residual norm ||Ax - b||₂
    fn compute_residual(&self, a: &Array2<f64>, b: &Array1<f64>, x: &Array1<f64>) -> f64 {
        let n = b.len();
        let mut residual = 0.0;

        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += a[[i, j]] * x[j];
            }

            let diff = row_sum - b[i];
            residual += diff * diff;
        }

        residual.sqrt()
    }
}

/// Convert FEMResult to PDESolution
impl From<FEMResult> for PDESolution<f64> {
    fn from(result: FEMResult) -> Self {
        let mut grids = Vec::new();
        let n = result.mesh.points.len();

        // Extract x and y coordinates as separate grids
        let mut x_coords = Array1::zeros(n);
        let mut y_coords = Array1::zeros(n);

        for (i, point) in result.mesh.points.iter().enumerate() {
            x_coords[i] = point.x;
            y_coords[i] = point.y;
        }

        grids.push(x_coords);
        grids.push(y_coords);

        // Create solution values as a 2D array with one column
        let mut values = Vec::new();
        let u_reshaped = result.u.into_shape_with_order((n, 1)).unwrap();
        values.push(u_reshaped);

        // Create solver info
        let info = PDESolverInfo {
            num_iterations: result.num_iterations,
            computation_time: result.computation_time,
            residual_norm: Some(result.residual_norm),
            convergence_history: result.convergence_history,
            method: "Finite Element Method".to_string(),
        };

        PDESolution {
            grids,
            values,
            error_estimate: None,
            info,
        }
    }
}

// Add PDE error types
impl PDEError {
    /// Create a finite element error
    pub fn finite_element_error(msg: String) -> Self {
        PDEError::Other(format!("Finite element error: {}", msg))
    }
}
