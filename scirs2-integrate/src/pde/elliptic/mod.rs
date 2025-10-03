//! Elliptic PDE solvers
//!
//! This module provides solvers for elliptic partial differential equations,
//! such as Poisson's equation and Laplace's equation. These are steady-state
//! problems that don't involve time derivatives.
//!
//! Supported equation types:
//! - Poisson's equation: ∇²u = f(x, y)
//! - Laplace's equation: ∇²u = 0

use scirs2_core::ndarray::{Array1, Array2};
use std::time::Instant;

use crate::pde::finite_difference::FiniteDifferenceScheme;
use crate::pde::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, PDEError, PDEResult,
    PDESolution, PDESolverInfo,
};

/// Result of elliptic PDE solution
pub struct EllipticResult {
    /// Solution values
    pub u: Array2<f64>,

    /// Residual norm
    pub residual_norm: f64,

    /// Number of iterations performed
    pub num_iterations: usize,

    /// Computation time
    pub computation_time: f64,

    /// Convergence history
    pub convergence_history: Option<Vec<f64>>,
}

/// Options for elliptic PDE solvers
#[derive(Debug, Clone)]
pub struct EllipticOptions {
    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Tolerance for convergence
    pub tolerance: f64,

    /// Whether to save convergence history
    pub save_convergence_history: bool,

    /// Relaxation parameter for iterative methods (0 < omega < 2)
    pub omega: f64,

    /// Print detailed progress information
    pub verbose: bool,

    /// Finite difference scheme for discretization
    pub fd_scheme: FiniteDifferenceScheme,
}

impl Default for EllipticOptions {
    fn default() -> Self {
        EllipticOptions {
            max_iterations: 1000,
            tolerance: 1e-6,
            save_convergence_history: false,
            omega: 1.0,
            verbose: false,
            fd_scheme: FiniteDifferenceScheme::CentralDifference,
        }
    }
}

/// Poisson's equation solver for 2D problems
///
/// Solves ∇²u = f(x, y), or in expanded form:
/// ∂²u/∂x² + ∂²u/∂y² = f(x, y)
pub struct PoissonSolver2D {
    /// Spatial domain
    domain: Domain,

    /// Source term function f(x, y)
    source_term: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,

    /// Boundary conditions
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Solver options
    options: EllipticOptions,
}

impl PoissonSolver2D {
    /// Create a new Poisson equation solver for 2D problems
    pub fn new(
        domain: Domain,
        source_term: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<EllipticOptions>,
    ) -> PDEResult<Self> {
        // Validate the domain
        if domain.dimensions() != 2 {
            return Err(PDEError::DomainError(
                "Domain must be 2-dimensional for 2D Poisson solver".to_string(),
            ));
        }

        // Validate boundary _conditions
        if boundary_conditions.len() != 4 {
            return Err(PDEError::BoundaryConditions(
                "2D Poisson equation requires exactly 4 boundary _conditions (one for each edge)"
                    .to_string(),
            ));
        }

        // Ensure we have boundary _conditions for all dimensions/edges
        let has_x_lower = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Lower && bc.dimension == 0);
        let has_x_upper = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Upper && bc.dimension == 0);
        let has_y_lower = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Lower && bc.dimension == 1);
        let has_y_upper = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Upper && bc.dimension == 1);

        if !has_x_lower || !has_x_upper || !has_y_lower || !has_y_upper {
            return Err(PDEError::BoundaryConditions(
                "2D Poisson equation requires boundary _conditions for all edges of the domain"
                    .to_string(),
            ));
        }

        Ok(PoissonSolver2D {
            domain,
            source_term: Box::new(source_term),
            boundary_conditions,
            options: options.unwrap_or_default(),
        })
    }

    /// Solve the Poisson equation using Successive Over-Relaxation (SOR)
    pub fn solve_sor(&self) -> PDEResult<EllipticResult> {
        let start_time = Instant::now();

        // Generate spatial grids
        let x_grid = self.domain.grid(0)?;
        let y_grid = self.domain.grid(1)?;
        let nx = x_grid.len();
        let ny = y_grid.len();
        let dx = self.domain.grid_spacing(0)?;
        let dy = self.domain.grid_spacing(1)?;

        // Initialize solution with zeros
        let mut u = Array2::zeros((ny, nx));

        // Set Dirichlet boundary conditions in the initial guess
        self.apply_dirichlet_boundary_conditions(&mut u, &x_grid, &y_grid);

        // Prepare for iteration
        let mut residual_norm = f64::INFINITY;
        let mut iter = 0;
        let mut convergence_history = if self.options.save_convergence_history {
            Some(Vec::new())
        } else {
            None
        };

        // Main iteration loop (Successive Over-Relaxation)
        while residual_norm > self.options.tolerance && iter < self.options.max_iterations {
            // Store previous solution for computing residual
            let u_prev = u.clone();

            // Update interior points
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let x = x_grid[i];
                    let y = y_grid[j];

                    // Compute the right-hand side (source term)
                    let f_xy = (self.source_term)(x, y);

                    // Compute next iterate using 5-point stencil
                    let u_new = ((u[[j, i + 1]] + u[[j, i - 1]]) / (dx * dx)
                        + (u[[j + 1, i]] + u[[j - 1, i]]) / (dy * dy)
                        - f_xy)
                        / (2.0 / (dx * dx) + 2.0 / (dy * dy));

                    // Apply relaxation
                    u[[j, i]] = (1.0 - self.options.omega) * u[[j, i]] + self.options.omega * u_new;
                }
            }

            // Apply boundary conditions after each iteration
            self.apply_boundary_conditions(&mut u, &x_grid, &y_grid, dx, dy);

            // Compute residual norm
            let mut residual_sum = 0.0;
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let diff = u[[j, i]] - u_prev[[j, i]];
                    residual_sum += diff * diff;
                }
            }
            residual_norm = residual_sum.sqrt();

            // Save convergence history if requested
            if let Some(ref mut history) = convergence_history {
                history.push(residual_norm);
            }

            // Print progress if verbose
            if self.options.verbose && (iter % 100 == 0 || iter == self.options.max_iterations - 1)
            {
                println!("Iteration {iter}: residual = {residual_norm:.6e}");
            }

            iter += 1;
        }

        let computation_time = start_time.elapsed().as_secs_f64();

        if iter == self.options.max_iterations && residual_norm > self.options.tolerance {
            println!("Warning: Maximum iterations reached without convergence");
            println!(
                "Final residual: {:.6e}, tolerance: {:.6e}",
                residual_norm, self.options.tolerance
            );
        }

        Ok(EllipticResult {
            u,
            residual_norm,
            num_iterations: iter,
            computation_time,
            convergence_history,
        })
    }

    /// Solve the Poisson equation using direct sparse solver
    ///
    /// This method sets up the linear system Au = b and solves it directly.
    /// It creates a sparse matrix for the Laplacian operator and applies
    /// boundary conditions to the system.
    pub fn solve_direct(&self) -> PDEResult<EllipticResult> {
        let start_time = Instant::now();

        // Generate spatial grids
        let x_grid = self.domain.grid(0)?;
        let y_grid = self.domain.grid(1)?;
        let nx = x_grid.len();
        let ny = y_grid.len();
        let dx = self.domain.grid_spacing(0)?;
        let dy = self.domain.grid_spacing(1)?;

        // Total number of grid points
        let n = nx * ny;

        // Create matrix A (discretized Laplacian) and right-hand side vector b
        let mut a = Array2::zeros((n, n));
        let mut b = Array1::zeros(n);

        // Fill the matrix and vector for interior points using 5-point stencil
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let index = j * nx + i;
                let x = x_grid[i];
                let y = y_grid[j];

                // Diagonal term
                a[[index, index]] = -2.0 / (dx * dx) - 2.0 / (dy * dy);

                // Off-diagonal terms
                a[[index, index - 1]] = 1.0 / (dx * dx); // left
                a[[index, index + 1]] = 1.0 / (dx * dx); // right
                a[[index, index - nx]] = 1.0 / (dy * dy); // bottom
                a[[index, index + nx]] = 1.0 / (dy * dy); // top

                // Right-hand side
                b[index] = -(self.source_term)(x, y);
            }
        }

        // Apply boundary conditions to the system
        self.apply_boundary_conditions_to_system(&mut a, &mut b, &x_grid, &y_grid, nx, ny, dx, dy);

        // Solve the linear system using Gaussian elimination
        // For a real implementation, use a sparse solver library
        let u_flat = PoissonSolver2D::solve_linear_system(&a, &b)?;

        // Reshape the solution to 2D
        let mut u = Array2::zeros((ny, nx));
        for j in 0..ny {
            for i in 0..nx {
                u[[j, i]] = u_flat[j * nx + i];
            }
        }

        // Compute residual
        let mut residual_norm = 0.0;
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let x = x_grid[i];
                let y = y_grid[j];

                let laplacian = (u[[j, i + 1]] - 2.0 * u[[j, i]] + u[[j, i - 1]]) / (dx * dx)
                    + (u[[j + 1, i]] - 2.0 * u[[j, i]] + u[[j - 1, i]]) / (dy * dy);

                let residual = laplacian - (self.source_term)(x, y);
                residual_norm += residual * residual;
            }
        }
        residual_norm = residual_norm.sqrt();

        let computation_time = start_time.elapsed().as_secs_f64();

        Ok(EllipticResult {
            u,
            residual_norm,
            num_iterations: 1, // Direct method requires only one "iteration"
            computation_time,
            convergence_history: None,
        })
    }

    // Helper method to solve linear system Ax = b using Gaussian elimination
    // In a real implementation, this would use a specialized sparse solver
    fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> PDEResult<Array1<f64>> {
        let n = b.len();

        // Simple Gaussian elimination for demonstration purposes
        // For real applications, use a linear algebra library

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

    // Helper method to apply boundary conditions to the solution vector
    fn apply_boundary_conditions(
        &self,
        u: &mut Array2<f64>,
        x_grid: &Array1<f64>,
        y_grid: &Array1<f64>,
        dx: f64,
        dy: f64,
    ) {
        let nx = x_grid.len();
        let ny = y_grid.len();

        // Apply Dirichlet boundary conditions first
        self.apply_dirichlet_boundary_conditions(u, x_grid, y_grid);

        // Apply Neumann and Robin boundary conditions
        for bc in &self.boundary_conditions {
            match bc.bc_type {
                BoundaryConditionType::Dirichlet => {
                    // Already handled above
                }
                BoundaryConditionType::Neumann => {
                    match (bc.dimension, bc.location) {
                        (0, BoundaryLocation::Lower) => {
                            // Left boundary (x = x[0]): ∂u/∂x = bc.value
                            // Use one-sided difference: (u[i+1] - u[i])/dx = bc.value
                            for j in 0..ny {
                                u[[j, 0]] = u[[j, 1]] - dx * bc.value;
                            }
                        }
                        (0, BoundaryLocation::Upper) => {
                            // Right boundary (x = x[nx-1]): ∂u/∂x = bc.value
                            // Use one-sided difference: (u[i] - u[i-1])/dx = bc.value
                            for j in 0..ny {
                                u[[j, nx - 1]] = u[[j, nx - 2]] + dx * bc.value;
                            }
                        }
                        (1, BoundaryLocation::Lower) => {
                            // Bottom boundary (y = y[0]): ∂u/∂y = bc.value
                            // Use one-sided difference: (u[j+1] - u[j])/dy = bc.value
                            for i in 0..nx {
                                u[[0, i]] = u[[1, i]] - dy * bc.value;
                            }
                        }
                        (1, BoundaryLocation::Upper) => {
                            // Top boundary (y = y[ny-1]): ∂u/∂y = bc.value
                            // Use one-sided difference: (u[j] - u[j-1])/dy = bc.value
                            for i in 0..nx {
                                u[[ny - 1, i]] = u[[ny - 2, i]] + dy * bc.value;
                            }
                        }
                        _ => {
                            // Invalid dimension (should be caught during validation)
                        }
                    }
                }
                BoundaryConditionType::Robin => {
                    // Robin boundary condition: a*u + b*∂u/∂n = c
                    if let Some([a, b, c]) = bc.coefficients {
                        match (bc.dimension, bc.location) {
                            (0, BoundaryLocation::Lower) => {
                                // Left boundary (x = x[0])
                                for j in 0..ny {
                                    // a*u + b*(u[i+1] - u[i])/dx = c
                                    // Solve for u[i]
                                    u[[j, 0]] = (b * u[[j, 1]] / dx + c) / (a + b / dx);
                                }
                            }
                            (0, BoundaryLocation::Upper) => {
                                // Right boundary (x = x[nx-1])
                                for j in 0..ny {
                                    // a*u + b*(u[i] - u[i-1])/dx = c
                                    // Solve for u[i]
                                    u[[j, nx - 1]] = (b * u[[j, nx - 2]] / dx + c) / (a - b / dx);
                                }
                            }
                            (1, BoundaryLocation::Lower) => {
                                // Bottom boundary (y = y[0])
                                for i in 0..nx {
                                    // a*u + b*(u[j+1] - u[j])/dy = c
                                    // Solve for u[j]
                                    u[[0, i]] = (b * u[[1, i]] / dy + c) / (a + b / dy);
                                }
                            }
                            (1, BoundaryLocation::Upper) => {
                                // Top boundary (y = y[ny-1])
                                for i in 0..nx {
                                    // a*u + b*(u[j] - u[j-1])/dy = c
                                    // Solve for u[j]
                                    u[[ny - 1, i]] = (b * u[[ny - 2, i]] / dy + c) / (a - b / dy);
                                }
                            }
                            _ => {
                                // Invalid dimension (should be caught during validation)
                            }
                        }
                    }
                }
                BoundaryConditionType::Periodic => {
                    // Periodic boundary conditions
                    match bc.dimension {
                        0 => {
                            // Periodic in x-direction: u(0,y) = u(L,y)
                            for j in 0..ny {
                                let avg = 0.5 * (u[[j, 0]] + u[[j, nx - 1]]);
                                u[[j, 0]] = avg;
                                u[[j, nx - 1]] = avg;
                            }
                        }
                        1 => {
                            // Periodic in y-direction: u(x,0) = u(x,H)
                            for i in 0..nx {
                                let avg = 0.5 * (u[[0, i]] + u[[ny - 1, i]]);
                                u[[0, i]] = avg;
                                u[[ny - 1, i]] = avg;
                            }
                        }
                        _ => {
                            // Invalid dimension (should be caught during validation)
                        }
                    }
                }
            }
        }
    }

    // Helper method to apply only Dirichlet boundary conditions to the solution vector
    fn apply_dirichlet_boundary_conditions(
        &self,
        u: &mut Array2<f64>,
        x_grid: &Array1<f64>,
        y_grid: &Array1<f64>,
    ) {
        let nx = x_grid.len();
        let ny = y_grid.len();

        for bc in &self.boundary_conditions {
            if bc.bc_type == BoundaryConditionType::Dirichlet {
                match (bc.dimension, bc.location) {
                    (0, BoundaryLocation::Lower) => {
                        // Left boundary (x = x[0])
                        for j in 0..ny {
                            u[[j, 0]] = bc.value;
                        }
                    }
                    (0, BoundaryLocation::Upper) => {
                        // Right boundary (x = x[nx-1])
                        for j in 0..ny {
                            u[[j, nx - 1]] = bc.value;
                        }
                    }
                    (1, BoundaryLocation::Lower) => {
                        // Bottom boundary (y = y[0])
                        for i in 0..nx {
                            u[[0, i]] = bc.value;
                        }
                    }
                    (1, BoundaryLocation::Upper) => {
                        // Top boundary (y = y[ny-1])
                        for i in 0..nx {
                            u[[ny - 1, i]] = bc.value;
                        }
                    }
                    _ => {
                        // Invalid dimension (should be caught during validation)
                    }
                }
            }
        }
    }

    // Helper method to apply boundary conditions to the linear system
    #[allow(clippy::too_many_arguments)]
    fn apply_boundary_conditions_to_system(
        &self,
        a: &mut Array2<f64>,
        b: &mut Array1<f64>,
        _x_grid: &Array1<f64>,
        _y_grid: &Array1<f64>,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
    ) {
        for bc in &self.boundary_conditions {
            match bc.bc_type {
                BoundaryConditionType::Dirichlet => {
                    match (bc.dimension, bc.location) {
                        (0, BoundaryLocation::Lower) => {
                            // Left boundary (x = x[0])
                            for j in 0..ny {
                                let index = j * nx;

                                // Set row to identity
                                for k in 0..a.shape()[1] {
                                    a[[index, k]] = 0.0;
                                }
                                a[[index, index]] = 1.0;

                                // Set right-hand side to boundary value
                                b[index] = bc.value;
                            }
                        }
                        (0, BoundaryLocation::Upper) => {
                            // Right boundary (x = x[nx-1])
                            for j in 0..ny {
                                let index = j * nx + (nx - 1);

                                // Set row to identity
                                for k in 0..a.shape()[1] {
                                    a[[index, k]] = 0.0;
                                }
                                a[[index, index]] = 1.0;

                                // Set right-hand side to boundary value
                                b[index] = bc.value;
                            }
                        }
                        (1, BoundaryLocation::Lower) => {
                            // Bottom boundary (y = y[0])
                            for i in 0..nx {
                                let index = i;

                                // Set row to identity
                                for k in 0..a.shape()[1] {
                                    a[[index, k]] = 0.0;
                                }
                                a[[index, index]] = 1.0;

                                // Set right-hand side to boundary value
                                b[index] = bc.value;
                            }
                        }
                        (1, BoundaryLocation::Upper) => {
                            // Top boundary (y = y[ny-1])
                            for i in 0..nx {
                                let index = (ny - 1) * nx + i;

                                // Set row to identity
                                for k in 0..a.shape()[1] {
                                    a[[index, k]] = 0.0;
                                }
                                a[[index, index]] = 1.0;

                                // Set right-hand side to boundary value
                                b[index] = bc.value;
                            }
                        }
                        _ => {
                            // Invalid dimension (should be caught during validation)
                        }
                    }
                }
                BoundaryConditionType::Neumann => {
                    match (bc.dimension, bc.location) {
                        (0, BoundaryLocation::Lower) => {
                            // Left boundary (x = x[0]): ∂u/∂x = bc.value
                            for j in 0..ny {
                                let index = j * nx;

                                // Modify matrix row to represent one-sided difference
                                // (u[i+1] - u[i])/dx = bc.value
                                for k in 0..a.shape()[1] {
                                    a[[index, k]] = 0.0;
                                }
                                a[[index, index]] = -1.0 / dx;
                                a[[index, index + 1]] = 1.0 / dx;

                                // Set right-hand side to boundary value
                                b[index] = bc.value;
                            }
                        }
                        (0, BoundaryLocation::Upper) => {
                            // Right boundary (x = x[nx-1]): ∂u/∂x = bc.value
                            for j in 0..ny {
                                let index = j * nx + (nx - 1);

                                // Modify matrix row to represent one-sided difference
                                // (u[i] - u[i-1])/dx = bc.value
                                for k in 0..a.shape()[1] {
                                    a[[index, k]] = 0.0;
                                }
                                a[[index, index - 1]] = -1.0 / dx;
                                a[[index, index]] = 1.0 / dx;

                                // Set right-hand side to boundary value
                                b[index] = bc.value;
                            }
                        }
                        (1, BoundaryLocation::Lower) => {
                            // Bottom boundary (y = y[0]): ∂u/∂y = bc.value
                            for i in 0..nx {
                                let index = i;

                                // Modify matrix row to represent one-sided difference
                                // (u[j+1] - u[j])/dy = bc.value
                                for k in 0..a.shape()[1] {
                                    a[[index, k]] = 0.0;
                                }
                                a[[index, index]] = -1.0 / dy;
                                a[[index, index + nx]] = 1.0 / dy;

                                // Set right-hand side to boundary value
                                b[index] = bc.value;
                            }
                        }
                        (1, BoundaryLocation::Upper) => {
                            // Top boundary (y = y[ny-1]): ∂u/∂y = bc.value
                            for i in 0..nx {
                                let index = (ny - 1) * nx + i;

                                // Modify matrix row to represent one-sided difference
                                // (u[j] - u[j-1])/dy = bc.value
                                for k in 0..a.shape()[1] {
                                    a[[index, k]] = 0.0;
                                }
                                a[[index, index - nx]] = -1.0 / dy;
                                a[[index, index]] = 1.0 / dy;

                                // Set right-hand side to boundary value
                                b[index] = bc.value;
                            }
                        }
                        _ => {
                            // Invalid dimension (should be caught during validation)
                        }
                    }
                }
                BoundaryConditionType::Robin => {
                    // Robin boundary condition: a*u + b*∂u/∂n = c
                    if let Some([a_coef, b_coef, c_coef]) = bc.coefficients {
                        match (bc.dimension, bc.location) {
                            (0, BoundaryLocation::Lower) => {
                                // Left boundary (x = x[0])
                                for j in 0..ny {
                                    let index = j * nx;

                                    // Modify matrix row to represent robin condition
                                    // a*u + b*(u[i+1] - u[i])/dx = c
                                    for k in 0..a.shape()[1] {
                                        a[[index, k]] = 0.0;
                                    }
                                    a[[index, index]] = a_coef - b_coef / dx;
                                    a[[index, index + 1]] = b_coef / dx;

                                    // Set right-hand side to boundary value
                                    b[index] = c_coef;
                                }
                            }
                            (0, BoundaryLocation::Upper) => {
                                // Right boundary (x = x[nx-1])
                                for j in 0..ny {
                                    let index = j * nx + (nx - 1);

                                    // Modify matrix row to represent robin condition
                                    // a*u + b*(u[i] - u[i-1])/dx = c
                                    for k in 0..a.shape()[1] {
                                        a[[index, k]] = 0.0;
                                    }
                                    a[[index, index - 1]] = -b_coef / dx;
                                    a[[index, index]] = a_coef + b_coef / dx;

                                    // Set right-hand side to boundary value
                                    b[index] = c_coef;
                                }
                            }
                            (1, BoundaryLocation::Lower) => {
                                // Bottom boundary (y = y[0])
                                for i in 0..nx {
                                    let index = i;

                                    // Modify matrix row to represent robin condition
                                    // a*u + b*(u[j+1] - u[j])/dy = c
                                    for k in 0..a.shape()[1] {
                                        a[[index, k]] = 0.0;
                                    }
                                    a[[index, index]] = a_coef - b_coef / dy;
                                    a[[index, index + nx]] = b_coef / dy;

                                    // Set right-hand side to boundary value
                                    b[index] = c_coef;
                                }
                            }
                            (1, BoundaryLocation::Upper) => {
                                // Top boundary (y = y[ny-1])
                                for i in 0..nx {
                                    let index = (ny - 1) * nx + i;

                                    // Modify matrix row to represent robin condition
                                    // a*u + b*(u[j] - u[j-1])/dy = c
                                    for k in 0..a.shape()[1] {
                                        a[[index, k]] = 0.0;
                                    }
                                    a[[index, index - nx]] = -b_coef / dy;
                                    a[[index, index]] = a_coef + b_coef / dy;

                                    // Set right-hand side to boundary value
                                    b[index] = c_coef;
                                }
                            }
                            _ => {
                                // Invalid dimension (should be caught during validation)
                            }
                        }
                    }
                }
                BoundaryConditionType::Periodic => {
                    // Periodic boundary conditions are more complex for the matrix system
                    // For simplicity, we'll just handle them in a basic way
                    match bc.dimension {
                        0 => {
                            // Periodic in x-direction: u(0,y) = u(L,y)
                            for j in 0..ny {
                                let left_index = j * nx;
                                let right_index = j * nx + (nx - 1);

                                // Set these rows to represent equality
                                for k in 0..a.shape()[1] {
                                    a[[left_index, k]] = 0.0;
                                    a[[right_index, k]] = 0.0;
                                }

                                a[[left_index, left_index]] = 1.0;
                                a[[left_index, right_index]] = -1.0;

                                a[[right_index, left_index]] = -1.0;
                                a[[right_index, right_index]] = 1.0;

                                b[left_index] = 0.0;
                                b[right_index] = 0.0;
                            }
                        }
                        1 => {
                            // Periodic in y-direction: u(x,0) = u(x,H)
                            for i in 0..nx {
                                let bottom_index = i;
                                let top_index = (ny - 1) * nx + i;

                                // Set these rows to represent equality
                                for k in 0..a.shape()[1] {
                                    a[[bottom_index, k]] = 0.0;
                                    a[[top_index, k]] = 0.0;
                                }

                                a[[bottom_index, bottom_index]] = 1.0;
                                a[[bottom_index, top_index]] = -1.0;

                                a[[top_index, bottom_index]] = -1.0;
                                a[[top_index, top_index]] = 1.0;

                                b[bottom_index] = 0.0;
                                b[top_index] = 0.0;
                            }
                        }
                        _ => {
                            // Invalid dimension (should be caught during validation)
                        }
                    }
                }
            }
        }
    }
}

/// Laplace's equation solver for 2D problems
///
/// Solves ∇²u = 0, or in expanded form:
/// ∂²u/∂x² + ∂²u/∂y² = 0
///
/// This is a specialized version of the Poisson solver with the right-hand side set to zero.
pub struct LaplaceSolver2D {
    /// Underlying Poisson solver
    poisson_solver: PoissonSolver2D,
}

impl LaplaceSolver2D {
    /// Create a new Laplace equation solver for 2D problems
    pub fn new(
        domain: Domain,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<EllipticOptions>,
    ) -> PDEResult<Self> {
        // Create a Poisson solver with zero source term
        let poisson_solver =
            PoissonSolver2D::new(domain, |_x, _y| 0.0, boundary_conditions, options)?;

        Ok(LaplaceSolver2D { poisson_solver })
    }

    /// Solve Laplace's equation using Successive Over-Relaxation (SOR)
    pub fn solve_sor(&self) -> PDEResult<EllipticResult> {
        self.poisson_solver.solve_sor()
    }

    /// Solve Laplace's equation using a direct sparse solver
    pub fn solve_direct(&self) -> PDEResult<EllipticResult> {
        self.poisson_solver.solve_direct()
    }
}

/// Convert EllipticResult to PDESolution
impl From<EllipticResult> for PDESolution<f64> {
    fn from(result: EllipticResult) -> Self {
        let mut grids = Vec::new();

        // Extract grid dimensions from solution (assuming they're regular)
        let ny = result.u.shape()[0];
        let nx = result.u.shape()[1];

        // Create grids (since we don't have the actual grid values, use linspace)
        let x_grid = Array1::linspace(0.0, 1.0, nx);
        let y_grid = Array1::linspace(0.0, 1.0, ny);

        grids.push(x_grid);
        grids.push(y_grid);

        // Create solution values
        let values = vec![result.u];

        // Create solver info
        let info = PDESolverInfo {
            num_iterations: result.num_iterations,
            computation_time: result.computation_time,
            residual_norm: Some(result.residual_norm),
            convergence_history: result.convergence_history,
            method: "Elliptic PDE Solver".to_string(),
        };

        PDESolution {
            grids,
            values,
            error_estimate: None,
            info,
        }
    }
}
