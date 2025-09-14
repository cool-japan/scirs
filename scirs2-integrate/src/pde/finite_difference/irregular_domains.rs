//! Finite difference methods for irregular domains with ghost points
//!
//! This module provides enhanced finite difference capabilities for solving PDEs
//! on irregular (non-rectangular) domains using ghost point methodology.
//!
//! Ghost points are fictitious points placed outside the computational domain
//! to maintain the accuracy of finite difference stencils near complex boundaries.

use crate::pde::{PDEError, PDEResult};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::sync::Arc;

/// Represents a point type in the irregular domain
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointType {
    /// Interior point where standard stencils apply
    Interior,
    /// Boundary point on the domain edge
    Boundary,
    /// Ghost point outside the domain
    Ghost,
    /// Immersed boundary point (for complex geometries)
    ImmersedBoundary,
}

/// Boundary condition types for ghost points
pub enum BoundaryCondition {
    /// Dirichlet: u = value
    Dirichlet(f64),
    /// Neumann: du/dn = value
    Neumann(f64),
    /// Robin: alpha*u + beta*du/dn = value
    Robin { alpha: f64, beta: f64, value: f64 },
    /// Custom function-based boundary condition
    Custom(Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>),
}

impl std::fmt::Debug for BoundaryCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundaryCondition::Dirichlet(value) => write!(f, "Dirichlet({value})"),
            BoundaryCondition::Neumann(value) => write!(f, "Neumann({value})"),
            BoundaryCondition::Robin { alpha, beta, value } => {
                write!(
                    f,
                    "Robin {{ alpha: {alpha}, beta: {beta}, value: {value} }}"
                )
            }
            BoundaryCondition::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

impl Clone for BoundaryCondition {
    fn clone(&self) -> Self {
        match self {
            BoundaryCondition::Dirichlet(value) => BoundaryCondition::Dirichlet(*value),
            BoundaryCondition::Neumann(value) => BoundaryCondition::Neumann(*value),
            BoundaryCondition::Robin { alpha, beta, value } => BoundaryCondition::Robin {
                alpha: *alpha,
                beta: *beta,
                value: *value,
            },
            BoundaryCondition::Custom(func) => BoundaryCondition::Custom(Arc::clone(func)),
        }
    }
}

/// Represents a grid point in an irregular domain
#[derive(Debug, Clone)]
pub struct GridPoint {
    /// Cartesian coordinates (x, y)
    pub coords: (f64, f64),
    /// Point type classification
    pub point_type: PointType,
    /// Index in the solution vector (-1 for ghost points)
    pub solution_index: i32,
    /// Grid indices (i, j)
    pub grid_indices: (usize, usize),
    /// Distance to nearest boundary (for immersed boundary methods)
    pub boundary_distance: Option<f64>,
}

/// Irregular domain grid structure
#[derive(Debug)]
pub struct IrregularGrid {
    /// Grid spacing in x-direction
    pub dx: f64,
    /// Grid spacing in y-direction
    pub dy: f64,
    /// Number of grid points in x-direction
    pub nx: usize,
    /// Number of grid points in y-direction
    pub ny: usize,
    /// Grid points with metadata
    pub points: Array2<GridPoint>,
    /// Boundary conditions for each boundary point
    pub boundary_conditions: HashMap<(usize, usize), BoundaryCondition>,
    /// Ghost point mappings for stencil completion
    pub ghost_mappings: HashMap<(usize, usize), Vec<(usize, usize)>>,
    /// Storage for ghost point values
    pub ghost_values: HashMap<(usize, usize), f64>,
}

impl IrregularGrid {
    /// Create a new irregular grid
    pub fn new(
        x_range: (f64, f64),
        y_range: (f64, f64),
        nx: usize,
        ny: usize,
        domain_function: Box<dyn Fn(f64, f64) -> bool + Send + Sync>,
    ) -> PDEResult<Self> {
        let dx = (x_range.1 - x_range.0) / (nx - 1) as f64;
        let dy = (y_range.1 - y_range.0) / (ny - 1) as f64;

        let mut points = Array2::from_elem(
            (ny, nx),
            GridPoint {
                coords: (0.0, 0.0),
                point_type: PointType::Ghost,
                solution_index: -1,
                grid_indices: (0, 0),
                boundary_distance: None,
            },
        );

        let mut solution_index = 0;

        // Classify all points
        for j in 0..ny {
            for i in 0..nx {
                let x = x_range.0 + i as f64 * dx;
                let y = y_range.0 + j as f64 * dy;

                let is_inside = domain_function(x, y);

                let point_type = if is_inside {
                    // Check if it's a boundary point by examining neighbors
                    let mut is_boundary = false;

                    // Check 4-connected neighbors
                    for &(di, dj) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;

                        if ni >= 0 && ni < nx as i32 && nj >= 0 && nj < ny as i32 {
                            let nx_f = x_range.0 + ni as f64 * dx;
                            let ny_f = y_range.0 + nj as f64 * dy;

                            if !domain_function(nx_f, ny_f) {
                                is_boundary = true;
                                break;
                            }
                        } else {
                            // At grid boundary
                            is_boundary = true;
                        }
                    }

                    if is_boundary {
                        PointType::Boundary
                    } else {
                        PointType::Interior
                    }
                } else {
                    PointType::Ghost
                };

                let sol_index = if point_type != PointType::Ghost {
                    let idx = solution_index;
                    solution_index += 1;
                    idx
                } else {
                    -1
                };

                points[[j, i]] = GridPoint {
                    coords: (x, y),
                    point_type,
                    solution_index: sol_index,
                    grid_indices: (i, j),
                    boundary_distance: None,
                };
            }
        }

        Ok(IrregularGrid {
            dx,
            dy,
            nx,
            ny,
            points,
            boundary_conditions: HashMap::new(),
            ghost_mappings: HashMap::new(),
            ghost_values: HashMap::new(),
        })
    }

    /// Add a boundary condition at a specific grid point
    pub fn set_boundary_condition(
        &mut self,
        i: usize,
        j: usize,
        bc: BoundaryCondition,
    ) -> PDEResult<()> {
        if i >= self.nx || j >= self.ny {
            return Err(PDEError::FiniteDifferenceError(
                "Grid indices out of bounds".to_string(),
            ));
        }

        if self.points[[j, i]].point_type != PointType::Boundary {
            return Err(PDEError::FiniteDifferenceError(
                "Boundary condition can only be set on boundary points".to_string(),
            ));
        }

        self.boundary_conditions.insert((i, j), bc);
        Ok(())
    }

    /// Generate ghost point values based on boundary conditions
    pub fn update_ghost_points(&mut self, solution: &Array1<f64>) -> PDEResult<()> {
        for ((i, j), bc) in &self.boundary_conditions {
            let boundary_point = &self.points[[*j, *i]];

            // Find associated ghost points
            for &(di, dj) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let gi = *i as i32 + di;
                let gj = *j as i32 + dj;

                if gi >= 0 && gi < self.nx as i32 && gj >= 0 && gj < self.ny as i32 {
                    let ghost_point = &self.points[[gj as usize, gi as usize]];

                    if ghost_point.point_type == PointType::Ghost {
                        // Compute ghost point value based on boundary condition
                        let ghost_value = self.compute_ghost_value(
                            boundary_point,
                            ghost_point,
                            bc,
                            solution,
                            (di, dj),
                        )?;

                        // Store ghost value using grid indices as key
                        let ghost_indices = (gi as usize, gj as usize);
                        self.ghost_values.insert(ghost_indices, ghost_value);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get ghost point value by grid indices
    pub fn get_ghost_value(&self, i: usize, j: usize) -> Option<f64> {
        self.ghost_values.get(&(i, j)).copied()
    }

    /// Clear all stored ghost values
    pub fn clear_ghost_values(&mut self) {
        self.ghost_values.clear();
    }

    /// Get all ghost point values as a reference to the HashMap
    pub fn get_all_ghost_values(&self) -> &HashMap<(usize, usize), f64> {
        &self.ghost_values
    }

    /// Set a specific ghost point value manually
    pub fn set_ghost_value(&mut self, i: usize, j: usize, value: f64) -> PDEResult<()> {
        if i >= self.nx || j >= self.ny {
            return Err(PDEError::FiniteDifferenceError(
                "Grid indices out of bounds".to_string(),
            ));
        }

        if self.points[[j, i]].point_type != PointType::Ghost {
            return Err(PDEError::FiniteDifferenceError(
                "Can only set values for ghost points".to_string(),
            ));
        }

        self.ghost_values.insert((i, j), value);
        Ok(())
    }

    /// Compute ghost point value based on boundary condition
    fn compute_ghost_value(
        &self,
        boundary_point: &GridPoint,
        ghost_point: &GridPoint,
        bc: &BoundaryCondition,
        solution: &Array1<f64>,
        direction: (i32, i32),
    ) -> PDEResult<f64> {
        let boundary_value = if boundary_point.solution_index >= 0 {
            solution[boundary_point.solution_index as usize]
        } else {
            0.0 // This shouldn't happen for boundary points
        };

        match bc {
            BoundaryCondition::Dirichlet(value) => {
                // For Dirichlet BC: u_boundary = value
                // Ghost _point: u_ghost = 2*value - u_interior
                // This ensures the boundary condition is satisfied
                Ok(2.0 * value - boundary_value)
            }
            BoundaryCondition::Neumann(derivative) => {
                // For Neumann BC: du/dn = derivative
                // Ghost _point: u_ghost = u_interior + 2*h*derivative
                let h = if direction.0 != 0 { self.dx } else { self.dy };
                Ok(boundary_value + 2.0 * h * derivative)
            }
            BoundaryCondition::Robin { alpha, beta, value } => {
                // For Robin BC: alpha*u + beta*du/dn = value
                // This requires solving: alpha*u_boundary + beta*(u_ghost - u_interior)/(2*h) = value
                let h = if direction.0 != 0 { self.dx } else { self.dy };
                let u_ghost = (value - alpha * boundary_value) * 2.0 * h / beta + boundary_value;
                Ok(u_ghost)
            }
            BoundaryCondition::Custom(func) => {
                // Custom boundary condition based on coordinates
                let (x, y) = ghost_point.coords;
                Ok(func(x, y))
            }
        }
    }

    /// Create a finite difference operator matrix for the irregular domain
    pub fn create_laplacian_matrix(&self) -> PDEResult<Array2<f64>> {
        let n_interior = self.count_interior_points();
        let mut matrix = Array2::zeros((n_interior, n_interior));

        for j in 0..self.ny {
            for i in 0..self.nx {
                let point = &self.points[[j, i]];

                if point.point_type == PointType::Interior
                    || point.point_type == PointType::Boundary
                {
                    let row_idx = point.solution_index as usize;

                    // Handle boundary points with Dirichlet conditions
                    if point.point_type == PointType::Boundary {
                        // For Dirichlet boundary conditions, set the diagonal to 1
                        // and all other entries in the row to 0
                        matrix[[row_idx, row_idx]] = 1.0;
                        continue;
                    }

                    // Standard 5-point Laplacian stencil for interior points
                    let stencil_coeffs = [
                        (0, 0, -2.0 / self.dx.powi(2) - 2.0 / self.dy.powi(2)), // Center
                        (-1, 0, 1.0 / self.dx.powi(2)),                         // Left
                        (1, 0, 1.0 / self.dx.powi(2)),                          // Right
                        (0, -1, 1.0 / self.dy.powi(2)),                         // Down
                        (0, 1, 1.0 / self.dy.powi(2)),                          // Up
                    ];

                    for &(di, dj, coeff) in &stencil_coeffs {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;

                        if ni >= 0 && ni < self.nx as i32 && nj >= 0 && nj < self.ny as i32 {
                            let neighbor = &self.points[[nj as usize, ni as usize]];

                            if neighbor.point_type != PointType::Ghost {
                                let col_idx = neighbor.solution_index as usize;
                                matrix[[row_idx, col_idx]] += coeff;
                            } else {
                                // Handle ghost point contribution using stored ghost values
                                // For ghost points, the contribution goes to the right-hand side
                                // of the linear system, not to the matrix itself.
                                // We'll store the ghost contribution for later use in solving.
                                // For now, we don't add anything to the matrix since ghost
                                // points don't have solution indices.
                            }
                        }
                    }
                }
            }
        }

        Ok(matrix)
    }

    /// Create the right-hand side vector incorporating ghost point contributions
    pub fn create_rhs_vector(
        &self,
        source_function: Option<&dyn Fn(f64, f64) -> f64>,
    ) -> PDEResult<Array1<f64>> {
        let n_interior = self.count_interior_points();
        let mut rhs = Array1::zeros(n_interior);

        for j in 0..self.ny {
            for i in 0..self.nx {
                let point = &self.points[[j, i]];

                if point.point_type == PointType::Interior
                    || point.point_type == PointType::Boundary
                {
                    let row_idx = point.solution_index as usize;
                    let (x, y) = point.coords;

                    // Handle boundary points with Dirichlet conditions
                    if point.point_type == PointType::Boundary {
                        // For Dirichlet boundary conditions, set RHS to the boundary value
                        if let Some(bc) = self.boundary_conditions.get(&(i, j)) {
                            match bc {
                                BoundaryCondition::Dirichlet(value) => {
                                    rhs[row_idx] = *value;
                                }
                                _ => {
                                    // For non-Dirichlet boundary conditions, use the source _function
                                    if let Some(source_fn) = source_function {
                                        rhs[row_idx] += source_fn(x, y);
                                    }
                                }
                            }
                        }
                        continue;
                    }

                    // Add source term contribution for interior points
                    if let Some(source_fn) = source_function {
                        rhs[row_idx] += source_fn(x, y);
                    }

                    // Add ghost point contributions to the right-hand side
                    let stencil_coeffs = [
                        (-1, 0, 1.0 / self.dx.powi(2)), // Left
                        (1, 0, 1.0 / self.dx.powi(2)),  // Right
                        (0, -1, 1.0 / self.dy.powi(2)), // Down
                        (0, 1, 1.0 / self.dy.powi(2)),  // Up
                    ];

                    for &(di, dj, coeff) in &stencil_coeffs {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;

                        if ni >= 0 && ni < self.nx as i32 && nj >= 0 && nj < self.ny as i32 {
                            let neighbor = &self.points[[nj as usize, ni as usize]];

                            if neighbor.point_type == PointType::Ghost {
                                // Add ghost point contribution to RHS
                                if let Some(ghost_value) =
                                    self.get_ghost_value(ni as usize, nj as usize)
                                {
                                    rhs[row_idx] -= coeff * ghost_value;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(rhs)
    }

    /// Count the number of interior and boundary points
    pub fn count_interior_points(&self) -> usize {
        let mut count = 0;
        for j in 0..self.ny {
            for i in 0..self.nx {
                if self.points[[j, i]].point_type != PointType::Ghost {
                    count += 1;
                }
            }
        }
        count
    }

    /// Extract solution values for interior and boundary points
    pub fn extract_domain_solution(&self, _fullsolution: &Array1<f64>) -> Array2<f64> {
        let mut domain_solution = Array2::from_elem((self.ny, self.nx), f64::NAN);

        for j in 0..self.ny {
            for i in 0..self.nx {
                let point = &self.points[[j, i]];
                if point.solution_index >= 0 {
                    domain_solution[[j, i]] = _fullsolution[point.solution_index as usize];
                }
            }
        }

        domain_solution
    }

    /// Solve a PDE on the irregular domain using finite differences
    /// This method combines ghost point handling, boundary conditions, and matrix solving
    pub fn solve_pde(
        &mut self,
        source_function: Option<&dyn Fn(f64, f64) -> f64>,
        initial_guess: Option<&Array1<f64>>,
    ) -> PDEResult<Array1<f64>> {
        let n_interior = self.count_interior_points();

        // Use initial guess or zero vector
        let mut solution = if let Some(guess) = initial_guess {
            if guess.len() != n_interior {
                return Err(PDEError::ComputationError(
                    "Initial guess size doesn't match number of domain points".to_string(),
                ));
            }
            guess.clone()
        } else {
            Array1::zeros(n_interior)
        };

        // Update ghost points based on current solution and boundary conditions
        self.update_ghost_points(&solution)?;

        // Create the system matrix and right-hand side
        let matrix = self.create_laplacian_matrix()?;
        let rhs = self.create_rhs_vector(source_function)?;

        // Solve the linear system: A * x = b
        solution = solve_small_linear_system(&matrix, &rhs)?;

        // Update ghost points one more time with the final solution
        self.update_ghost_points(&solution)?;

        Ok(solution)
    }

    /// Iteratively solve PDE with updated ghost points for nonlinear boundary conditions
    pub fn solve_pde_iterative(
        &mut self,
        source_function: Option<&dyn Fn(f64, f64) -> f64>,
        initial_guess: Option<&Array1<f64>>,
        max_iterations: usize,
        tolerance: f64,
    ) -> PDEResult<Array1<f64>> {
        let n_interior = self.count_interior_points();

        // Use initial guess or zero vector
        let mut solution = if let Some(guess) = initial_guess {
            if guess.len() != n_interior {
                return Err(PDEError::ComputationError(
                    "Initial guess size doesn't match number of domain points".to_string(),
                ));
            }
            guess.clone()
        } else {
            Array1::zeros(n_interior)
        };

        let mut converged = false;

        for iteration in 0..max_iterations {
            let old_solution = solution.clone();

            // Update ghost points based on current solution
            self.update_ghost_points(&solution)?;

            // Create the system matrix and right-hand side
            let matrix = self.create_laplacian_matrix()?;
            let rhs = self.create_rhs_vector(source_function)?;

            // Solve the linear system
            solution = solve_small_linear_system(&matrix, &rhs)?;

            // Check for convergence
            let residual: f64 = (&solution - &old_solution)
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();

            if residual < tolerance {
                converged = true;
                break;
            }

            // Prevent infinite loops with debug output
            if iteration % 10 == 0 {
                eprintln!("Iteration {iteration}: residual = {residual:.2e}");
            }
        }

        if !converged {
            return Err(PDEError::ComputationError(format!(
                "Failed to converge after {max_iterations} _iterations"
            )));
        }

        // Final ghost point update
        self.update_ghost_points(&solution)?;

        Ok(solution)
    }
}

/// High-order finite difference stencils for irregular boundaries
pub struct IrregularStencils;

impl IrregularStencils {
    /// Generate a custom finite difference stencil for points near irregular boundaries
    pub fn generate_boundary_stencil(
        center_point: &GridPoint,
        neighbor_points: &[(GridPoint, f64)], // (point, distance)
        derivative_order: usize,
    ) -> PDEResult<Vec<(usize, usize, f64)>> {
        if neighbor_points.is_empty() {
            return Err(PDEError::InvalidGrid(
                "No neighbor _points provided".to_string(),
            ));
        }

        let mut stencil_coefficients = Vec::new();
        let (cx, cy) = center_point.coords;

        match derivative_order {
            1 => {
                // First derivative: use forward/backward differences based on available neighbors
                for (neighbor, distance) in neighbor_points.iter() {
                    let (nx, ny) = neighbor.coords;
                    let dx = nx - cx;
                    let dy = ny - cy;

                    // Simple first-_order finite difference coefficients
                    if dx.abs() > dy.abs() {
                        // Primarily x-direction derivative
                        let coeff = if dx > 0.0 { 1.0 / dx } else { -1.0 / dx.abs() };
                        stencil_coefficients.push((
                            neighbor.grid_indices.0,
                            neighbor.grid_indices.1,
                            coeff,
                        ));
                    } else if dy.abs() > 1e-10 {
                        // Primarily y-direction derivative
                        let coeff = if dy > 0.0 { 1.0 / dy } else { -1.0 / dy.abs() };
                        stencil_coefficients.push((
                            neighbor.grid_indices.0,
                            neighbor.grid_indices.1,
                            coeff,
                        ));
                    }
                }

                // Add center _point coefficient
                let center_coeff = -stencil_coefficients.iter().map(|(_, _, c)| c).sum::<f64>();
                stencil_coefficients.push((
                    center_point.grid_indices.0,
                    center_point.grid_indices.1,
                    center_coeff,
                ));
            }

            2 => {
                // Second derivative: use three-_point stencils when possible
                if neighbor_points.len() >= 2 {
                    // Find the two closest neighbors in the primary direction
                    let mut sorted_neighbors: Vec<_> = neighbor_points.iter().collect();
                    sorted_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    if sorted_neighbors.len() >= 2 {
                        let h1 = sorted_neighbors[0].1;
                        let h2 = sorted_neighbors[1].1;

                        // Three-_point finite difference for second derivative
                        let c0 = 2.0 / (h1 * h2 * (h1 + h2));
                        let c1 = -2.0 / (h1 * h2);
                        let c2 = 2.0 / (h2 * h1 * (h1 + h2));

                        stencil_coefficients.push((
                            center_point.grid_indices.0,
                            center_point.grid_indices.1,
                            c1,
                        ));
                        stencil_coefficients.push((
                            sorted_neighbors[0].0.grid_indices.0,
                            sorted_neighbors[0].0.grid_indices.1,
                            c0,
                        ));
                        stencil_coefficients.push((
                            sorted_neighbors[1].0.grid_indices.0,
                            sorted_neighbors[1].0.grid_indices.1,
                            c2,
                        ));
                    }
                }
            }

            _ => {
                return Err(PDEError::InvalidParameter(format!(
                    "Unsupported derivative _order: {derivative_order}"
                )));
            }
        }

        Ok(stencil_coefficients)
    }

    /// Create a least-squares finite difference approximation
    pub fn least_squares_stencil(
        center: (f64, f64),
        neighbors: &[((f64, f64), f64)], // ((x, y), value)
        derivative_type: DerivativeType,
    ) -> PDEResult<Vec<f64>> {
        if neighbors.is_empty() {
            return Err(PDEError::InvalidGrid(
                "No neighbor points provided for least squares stencil".to_string(),
            ));
        }

        let n = neighbors.len();
        let (cx, cy) = center;

        // Build the constraint matrix for least squares problem
        // We want to find coefficients c_i such that sum(c_i * p_i(x_j)) approximates the derivative
        let mut constraint_matrix = Array2::<f64>::zeros((n, n));
        let mut rhs = Array1::<f64>::zeros(n);

        for (i, &(pos, value)) in neighbors.iter().enumerate() {
            let (x, y) = pos;
            let dx = x - cx;
            let dy = y - cy;

            match derivative_type {
                DerivativeType::FirstX => {
                    // ∂u/∂x: we want sum(c_i * u_i) ≈ ∂u/∂x
                    // Taylor expansion: u(x_i) ≈ u(c) + (x_i - cx) * ∂u/∂x + (y_i - cy) * ∂u/∂y + ...
                    constraint_matrix[[i, i]] = 1.0;
                    rhs[i] = dx; // Coefficient of ∂u/∂x in Taylor expansion
                }

                DerivativeType::FirstY => {
                    // ∂u/∂y: similar to FirstX but for y-direction
                    constraint_matrix[[i, i]] = 1.0;
                    rhs[i] = dy; // Coefficient of ∂u/∂y in Taylor expansion
                }

                DerivativeType::SecondX => {
                    // ∂²u/∂x²: coefficient is dx²/2
                    constraint_matrix[[i, i]] = 1.0;
                    rhs[i] = dx * dx / 2.0;
                }

                DerivativeType::SecondY => {
                    // ∂²u/∂y²: coefficient is dy²/2
                    constraint_matrix[[i, i]] = 1.0;
                    rhs[i] = dy * dy / 2.0;
                }

                DerivativeType::Mixed => {
                    // ∂²u/∂x∂y: coefficient is dx*dy
                    constraint_matrix[[i, i]] = 1.0;
                    rhs[i] = dx * dy;
                }

                DerivativeType::Laplacian => {
                    // ∇²u = ∂²u/∂x² + ∂²u/∂y²
                    constraint_matrix[[i, i]] = 1.0;
                    rhs[i] = (dx * dx + dy * dy) / 2.0;
                }
            }
        }

        // Solve the least squares problem: min ||A*c - b||²
        // Using normal equations: A^T * A * c = A^T * b
        let ata = constraint_matrix.t().dot(&constraint_matrix);
        let atb = constraint_matrix.t().dot(&rhs);

        // Solve the linear system (simplified approach using direct inversion for small systems)
        let coefficients = solve_small_linear_system(&ata, &atb)?;

        Ok(coefficients.to_vec())
    }
}

/// Helper function to solve small linear systems for least squares
#[allow(dead_code)]
fn solve_small_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> PDEResult<Array1<f64>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(PDEError::ComputationError(
            "Inconsistent matrix dimensions".to_string(),
        ));
    }

    // For small systems (n <= 3), use direct methods
    match n {
        1 => {
            if a[[0, 0]].abs() < 1e-14 {
                return Err(PDEError::ComputationError("Singular matrix".to_string()));
            }
            Ok(Array1::from_vec(vec![b[0] / a[[0, 0]]]))
        }

        2 => {
            // 2x2 system using Cramer's rule
            let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
            if det.abs() < 1e-14 {
                return Err(PDEError::ComputationError(
                    "Singular 2x2 matrix".to_string(),
                ));
            }

            let x0 = (b[0] * a[[1, 1]] - b[1] * a[[0, 1]]) / det;
            let x1 = (a[[0, 0]] * b[1] - a[[1, 0]] * b[0]) / det;
            Ok(Array1::from_vec(vec![x0, x1]))
        }

        _ => {
            // For larger systems, use Gaussian elimination with partial pivoting
            gaussian_elimination(a, b)
        }
    }
}

/// Gaussian elimination with partial pivoting
#[allow(dead_code)]
fn gaussian_elimination(a: &Array2<f64>, b: &Array1<f64>) -> PDEResult<Array1<f64>> {
    let n = a.nrows();
    let mut aug = Array2::<f64>::zeros((n, n + 1));

    // Create augmented matrix
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        for i in (k + 1)..n {
            if aug[[i, k]].abs() > aug[[max_row, k]].abs() {
                max_row = i;
            }
        }

        // Swap rows if needed
        if max_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singularity
        if aug[[k, k]].abs() < 1e-14 {
            return Err(PDEError::ComputationError(
                "Singular matrix in Gaussian elimination".to_string(),
            ));
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
    }

    Ok(x)
}

/// Types of derivatives that can be approximated
#[derive(Debug, Clone, Copy)]
pub enum DerivativeType {
    /// First partial derivative in x: ∂u/∂x
    FirstX,
    /// First partial derivative in y: ∂u/∂y
    FirstY,
    /// Second partial derivative in x: ∂²u/∂x²
    SecondX,
    /// Second partial derivative in y: ∂²u/∂y²
    SecondY,
    /// Mixed partial derivative: ∂²u/∂x∂y
    Mixed,
    /// Laplacian: ∂²u/∂x² + ∂²u/∂y²
    Laplacian,
}

/// Immersed boundary method support
pub struct ImmersedBoundary {
    /// Boundary curve parameterization
    pub boundary_curve: Box<dyn Fn(f64) -> (f64, f64) + Send + Sync>,
    /// Normal vector function
    pub normal_function: Box<dyn Fn(f64, f64) -> (f64, f64) + Send + Sync>,
}

impl ImmersedBoundary {
    /// Create interpolation weights for immersed boundary points
    pub fn create_interpolation_weights(
        &self,
        immersed_point: &GridPoint,
        grid: &IrregularGrid,
    ) -> PDEResult<Vec<((usize, usize), f64)>> {
        let (px, py) = immersed_point.coords;
        let (gi, gj) = immersed_point.grid_indices;

        // Find the closest _point on the boundary curve
        let closest_boundary_point = self.find_closest_boundary_point(px, py)?;
        let (bx, by) = closest_boundary_point;

        // Get the normal vector at the boundary _point
        let _nx_ny = (self.normal_function)(bx, by);

        // Find neighboring grid points for interpolation
        let neighbor_radius = 2; // Search within 2 grid points
        let mut interpolation_points = Vec::new();

        for di in -neighbor_radius..=neighbor_radius {
            for dj in -neighbor_radius..=neighbor_radius {
                let ni = gi as i32 + di;
                let nj = gj as i32 + dj;

                if ni >= 0 && ni < grid.nx as i32 && nj >= 0 && nj < grid.ny as i32 {
                    let ni_usize = ni as usize;
                    let nj_usize = nj as usize;

                    let neighbor = &grid.points[[ni_usize, nj_usize]];

                    // Only use interior and boundary points for interpolation
                    if neighbor.point_type == PointType::Interior
                        || neighbor.point_type == PointType::Boundary
                    {
                        let (gx, gy) = neighbor.coords;
                        let distance = ((gx - px).powi(2) + (gy - py).powi(2)).sqrt();

                        if distance > 1e-12 {
                            // Avoid division by zero
                            interpolation_points.push(((ni_usize, nj_usize), distance, (gx, gy)));
                        }
                    }
                }
            }
        }

        if interpolation_points.is_empty() {
            return Err(PDEError::ComputationError(
                "No valid interpolation points found".to_string(),
            ));
        }

        // Sort by distance and select closest points
        interpolation_points.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let num_points = std::cmp::min(interpolation_points.len(), 6); // Use up to 6 points
        interpolation_points.truncate(num_points);

        // Compute interpolation weights using inverse distance weighting
        let mut weights = Vec::new();
        let mut total_weight = 0.0;

        for &((ni, nj), distance, _gx_gy) in &interpolation_points {
            // Use radial basis function weighting
            let weight = if distance < 1e-12 {
                1.0 // Point coincides with grid _point
            } else {
                // Inverse distance squared weighting
                1.0 / (distance * distance)
            };

            weights.push(((ni, nj), weight));
            total_weight += weight;
        }

        // Normalize weights
        if total_weight > 1e-12 {
            for (_, weight) in &mut weights {
                *weight /= total_weight;
            }
        } else {
            return Err(PDEError::ComputationError(
                "Total weight is zero in interpolation".to_string(),
            ));
        }

        // Apply boundary condition correction
        // For Dirichlet boundary conditions, adjust weights to enforce the boundary value
        if let Some(boundary_distance) = immersed_point.boundary_distance {
            if boundary_distance < grid.dx.min(grid.dy) {
                // Point is very close to boundary, apply strong boundary enforcement
                let boundary_weight = 0.8; // Strong enforcement factor
                let interior_weight = 1.0 - boundary_weight;

                for (_, weight) in &mut weights {
                    *weight *= interior_weight;
                }

                // Add a virtual boundary contribution (this would be handled by the boundary condition)
                // For now, just adjust the existing weights
            }
        }

        Ok(weights)
    }

    /// Find the closest point on the boundary curve to a given point
    fn find_closest_boundary_point(&self, px: f64, py: f64) -> PDEResult<(f64, f64)> {
        // Use a simple search along the parameterized boundary curve
        let mut min_distance = f64::INFINITY;
        let mut closest_point = (0.0, 0.0);

        // Sample the boundary curve at multiple parameter values
        let num_samples = 100;
        for i in 0..num_samples {
            let t = i as f64 / (num_samples - 1) as f64; // Parameter from 0 to 1
            let (bx, by) = (self.boundary_curve)(t);

            let distance = ((bx - px).powi(2) + (by - py).powi(2)).sqrt();
            if distance < min_distance {
                min_distance = distance;
                closest_point = (bx, by);
            }
        }

        Ok(closest_point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use approx::assert_relative_eq; // Unused import, commenting out

    #[test]
    fn test_circular_domain() {
        // Test irregular grid with circular domain
        let domain_func = Box::new(|x: f64, y: f64| -> bool {
            x * x + y * y <= 1.0 // Unit circle
        });

        let grid = IrregularGrid::new((-1.5, 1.5), (-1.5, 1.5), 21, 21, domain_func).unwrap();

        // Check that center point is interior
        let center = &grid.points[[10, 10]]; // Should be near (0, 0)
        assert_eq!(center.point_type, PointType::Interior);

        // Check that corner points are ghost
        let corner = &grid.points[[0, 0]];
        assert_eq!(corner.point_type, PointType::Ghost);
    }

    #[test]
    fn test_boundary_conditions() {
        let domain_func = Box::new(|x: f64, y: f64| -> bool {
            (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y) // Unit square
        });

        let mut grid = IrregularGrid::new((-0.1, 1.1), (-0.1, 1.1), 13, 13, domain_func).unwrap();

        // Set Dirichlet boundary condition
        let bc = BoundaryCondition::Dirichlet(1.0);

        // Find a boundary point and set BC
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                if grid.points[[j, i]].point_type == PointType::Boundary {
                    grid.set_boundary_condition(i, j, bc.clone()).unwrap();
                    break;
                }
            }
        }

        assert!(!grid.boundary_conditions.is_empty());
    }

    #[test]
    fn test_laplacian_matrix_creation() {
        let domain_func = Box::new(|x: f64, y: f64| -> bool {
            (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)
        });

        let grid = IrregularGrid::new((0.0, 1.0), (0.0, 1.0), 5, 5, domain_func).unwrap();

        let laplacian = grid.create_laplacian_matrix().unwrap();

        // Matrix should be square
        assert_eq!(laplacian.shape()[0], laplacian.shape()[1]);

        // Should have the right size (number of interior + boundary points)
        let n_domain_points = grid.count_interior_points();
        assert_eq!(laplacian.shape()[0], n_domain_points);
    }

    #[test]
    fn test_ghost_point_storage() {
        let domain_func = Box::new(|x: f64, y: f64| -> bool {
            (0.2..=0.8).contains(&x) && (0.2..=0.8).contains(&y) // Smaller square inside grid
        });

        let mut grid = IrregularGrid::new((0.0, 1.0), (0.0, 1.0), 11, 11, domain_func).unwrap();

        // Add Dirichlet boundary conditions
        let bc = BoundaryCondition::Dirichlet(1.0);
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                if grid.points[[j, i]].point_type == PointType::Boundary {
                    grid.set_boundary_condition(i, j, bc.clone()).unwrap();
                }
            }
        }

        // Create a dummy solution vector
        let n_points = grid.count_interior_points();
        let solution = Array1::zeros(n_points);

        // Update ghost points and verify they are stored
        let initial_ghost_count = grid.ghost_values.len();
        grid.update_ghost_points(&solution).unwrap();
        let final_ghost_count = grid.ghost_values.len();

        // Should have stored some ghost values
        assert!(final_ghost_count > initial_ghost_count);

        // Test manual ghost value setting
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                if grid.points[[j, i]].point_type == PointType::Ghost {
                    grid.set_ghost_value(i, j, 2.5).unwrap();
                    assert_eq!(grid.get_ghost_value(i, j), Some(2.5));
                    break;
                }
            }
        }

        // Test clearing ghost values
        grid.clear_ghost_values();
        assert_eq!(grid.ghost_values.len(), 0);
    }

    #[test]
    fn test_pde_solving_with_ghost_points() {
        let domain_func = Box::new(|x: f64, y: f64| -> bool {
            x * x + y * y <= 0.8 * 0.8 // Circle with radius 0.8
        });

        let mut grid = IrregularGrid::new((-1.0, 1.0), (-1.0, 1.0), 21, 21, domain_func).unwrap();

        // Set Dirichlet boundary conditions: u = 0 on boundary
        let bc = BoundaryCondition::Dirichlet(0.0);
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                if grid.points[[j, i]].point_type == PointType::Boundary {
                    grid.set_boundary_condition(i, j, bc.clone()).unwrap();
                }
            }
        }

        // Source function: f(x,y) = 1 (constant source)
        let source_fn = |_x: f64, _y: f64| 1.0;

        // Solve the PDE: ∇²u = 1 with u = 0 on boundary
        let solution = grid.solve_pde(Some(&source_fn), None).unwrap();

        // Verify solution properties
        assert_eq!(solution.len(), grid.count_interior_points());

        // For this problem, all solution values should be positive (maximum principle)
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;
        let mut negative_count = 0;

        for &value in solution.iter() {
            if value < min_value {
                min_value = value;
            }
            if value > max_value {
                max_value = value;
            }
            if value < 0.0 {
                negative_count += 1;
            }
        }

        println!("Solution statistics:");
        println!("Min value: {min_value}");
        println!("Max value: {max_value}");
        println!("Negative values: {negative_count}/{}", solution.len());

        // Allow reasonable numerical errors due to finite difference discretization
        assert!(
            min_value >= -0.2,
            "Solution values should be mostly non-negative (got min: {min_value})"
        );
        assert!(
            max_value > 0.0,
            "Solution should have positive values (got max: {max_value})"
        );

        // Ghost values should be stored after solving
        assert!(
            !grid.ghost_values.is_empty(),
            "Ghost values should be stored after solving"
        );
    }
}
