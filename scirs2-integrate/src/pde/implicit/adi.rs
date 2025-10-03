//! Alternating Direction Implicit (ADI) methods for 2D and 3D PDEs
//!
//! This module provides implementations of ADI methods for solving
//! two-dimensional and three-dimensional partial differential equations.
//! ADI methods split multi-dimensional problems into sequences of one-dimensional
//! problems, making them computationally efficient.

use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView1};
use std::time::Instant;

use super::ImplicitOptions;
use crate::pde::finite_difference::FiniteDifferenceScheme;
use crate::pde::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, PDEError, PDEResult,
    PDESolution, PDESolverInfo,
};

/// Type alias for 2D coefficient function taking (x, y, t, u) and returning a value
type CoeffFn2D = Box<dyn Fn(f64, f64, f64, f64) -> f64 + Send + Sync>;

/// Result of ADI method solution
pub struct ADIResult {
    /// Time points
    pub t: Array1<f64>,

    /// Solution values, indexed as [time, x, y]
    pub u: Vec<Array3<f64>>,

    /// Solver information
    pub info: Option<String>,

    /// Computation time
    pub computation_time: f64,

    /// Number of time steps
    pub num_steps: usize,

    /// Number of linear system solves
    pub num_linear_solves: usize,
}

/// ADI solver for 2D parabolic PDEs
///
/// This solver implements the Peaceman-Rachford ADI method for solving
/// two-dimensional parabolic PDEs of the form:
/// ∂u/∂t = Dx*∂²u/∂x² + Dy*∂²u/∂y² + f(x,y,t,u)
pub struct ADI2D {
    /// Spatial domain
    domain: Domain,

    /// Time range [t_start, t_end]
    time_range: [f64; 2],

    /// Diffusion coefficient function in x-direction: Dx(x, y, t, u)
    diffusion_x: CoeffFn2D,

    /// Diffusion coefficient function in y-direction: Dy(x, y, t, u)
    diffusion_y: CoeffFn2D,

    /// Advection coefficient function in x-direction: vx(x, y, t, u)
    advection_x: Option<CoeffFn2D>,

    /// Advection coefficient function in y-direction: vy(x, y, t, u)
    advection_y: Option<CoeffFn2D>,

    /// Reaction term function: f(x, y, t, u)
    reaction_term: Option<CoeffFn2D>,

    /// Initial condition function: u(x, y, 0)
    initial_condition: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,

    /// Boundary conditions
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Finite difference scheme for spatial discretization
    fd_scheme: FiniteDifferenceScheme,

    /// Solver options
    options: ImplicitOptions,
}

impl ADI2D {
    /// Create a new ADI solver for 2D parabolic PDEs
    pub fn new(
        domain: Domain,
        time_range: [f64; 2],
        diffusion_x: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        diffusion_y: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        initial_condition: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<ImplicitOptions>,
    ) -> PDEResult<Self> {
        // Validate the domain
        if domain.dimensions() != 2 {
            return Err(PDEError::DomainError(
                "Domain must be 2-dimensional for 2D ADI solver".to_string(),
            ));
        }

        // Validate time _range
        if time_range[0] >= time_range[1] {
            return Err(PDEError::DomainError(
                "Invalid time _range: start must be less than end".to_string(),
            ));
        }

        // Validate boundary _conditions
        if boundary_conditions.len() != 4 {
            return Err(PDEError::BoundaryConditions(
                "2D parabolic PDE requires exactly 4 boundary _conditions".to_string(),
            ));
        }

        // Ensure we have boundary _conditions for all four boundaries
        let has_lower_x = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Lower && bc.dimension == 0);
        let has_upper_x = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Upper && bc.dimension == 0);
        let has_lower_y = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Lower && bc.dimension == 1);
        let has_upper_y = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Upper && bc.dimension == 1);

        if !has_lower_x || !has_upper_x || !has_lower_y || !has_upper_y {
            return Err(PDEError::BoundaryConditions(
                "2D parabolic PDE requires boundary _conditions for all four sides".to_string(),
            ));
        }

        // Use default options if none provided
        let options = options.unwrap_or_default();

        Ok(ADI2D {
            domain,
            time_range,
            diffusion_x: Box::new(diffusion_x),
            diffusion_y: Box::new(diffusion_y),
            advection_x: None,
            advection_y: None,
            reaction_term: None,
            initial_condition: Box::new(initial_condition),
            boundary_conditions,
            fd_scheme: FiniteDifferenceScheme::CentralDifference,
            options,
        })
    }

    /// Add advection terms to the PDE
    pub fn with_advection(
        mut self,
        advection_x: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        advection_y: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.advection_x = Some(Box::new(advection_x));
        self.advection_y = Some(Box::new(advection_y));
        self
    }

    /// Add a reaction term to the PDE
    pub fn with_reaction(
        mut self,
        reaction_term: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.reaction_term = Some(Box::new(reaction_term));
        self
    }

    /// Set the finite difference scheme for spatial discretization
    pub fn with_fd_scheme(mut self, scheme: FiniteDifferenceScheme) -> Self {
        self.fd_scheme = scheme;
        self
    }

    /// Solve the PDE using the Peaceman-Rachford ADI method
    pub fn solve(&self) -> PDEResult<ADIResult> {
        let start_time = Instant::now();

        // Generate spatial grids
        let x_grid = self.domain.grid(0)?;
        let y_grid = self.domain.grid(1)?;
        let nx = x_grid.len();
        let ny = y_grid.len();

        // Grid spacing
        let dx = self.domain.grid_spacing(0)?;
        let dy = self.domain.grid_spacing(1)?;

        // Time step
        let dt = self.options.dt.unwrap_or(0.01);

        // Calculate number of time steps
        let t_start = self.time_range[0];
        let t_end = self.time_range[1];
        let num_steps = ((t_end - t_start) / dt).ceil() as usize;

        // Initialize time array
        let mut t_values = Vec::with_capacity(num_steps + 1);
        t_values.push(t_start);

        // Initialize solution array with initial condition
        let mut u_current = Array2::zeros((nx, ny));

        // Apply initial condition
        for (i, &x) in x_grid.iter().enumerate() {
            for (j, &y) in y_grid.iter().enumerate() {
                u_current[[i, j]] = (self.initial_condition)(x, y);
            }
        }

        // Apply boundary conditions to initial state
        self.apply_boundary_conditions(&mut u_current, &x_grid, &y_grid, t_start);

        // Store solutions
        let save_every = self.options.save_every.unwrap_or(1);
        let mut solutions = Vec::with_capacity((num_steps + 1) / save_every + 1);

        // Add initial condition to solutions
        let mut u3d = Array3::zeros((nx, ny, 1));
        for i in 0..nx {
            for j in 0..ny {
                u3d[[i, j, 0]] = u_current[[i, j]];
            }
        }
        solutions.push(u3d);

        // Initialize coefficient matrices for both directions
        let mut a_x = Array2::zeros((nx, nx)); // For x-direction sweep
        let mut b_x = Array2::zeros((nx, nx));
        let mut a_y = Array2::zeros((ny, ny)); // For y-direction sweep
        let mut b_y = Array2::zeros((ny, ny));

        // Track solver statistics
        let mut num_linear_solves = 0;

        // Time-stepping loop
        for step in 0..num_steps {
            let t_current = t_start + step as f64 * dt;
            let t_mid = t_current + 0.5 * dt;
            let t_next = t_current + dt;

            // Intermediate solution after x-sweep
            let mut u_intermediate = Array2::zeros((nx, ny));

            // 1. First half-step: Implicit in x-direction, explicit in y-direction
            for j in 0..ny {
                // 1.1 Set up coefficient matrices for x-direction
                self.setup_coefficient_matrices_x(
                    &mut a_x,
                    &mut b_x,
                    &x_grid,
                    y_grid[j],
                    dx,
                    0.5 * dt,
                    t_current,
                    &u_current.slice(s![.., j]),
                );

                // 1.2 Extract the current solution row
                let mut u_row = Array1::zeros(nx);
                for i in 0..nx {
                    u_row[i] = u_current[[i, j]];
                }

                // 1.3 Right-hand side vector for x-direction
                let rhs_x = b_x.dot(&u_row);

                // 1.4 Solve the linear system for x-direction
                let u_x_next = ADI2D::solve_tridiagonal(&a_x, &rhs_x)?;
                num_linear_solves += 1;

                // 1.5 Update intermediate solution row
                for i in 0..nx {
                    u_intermediate[[i, j]] = u_x_next[i];
                }
            }

            // Apply boundary conditions to intermediate solution
            self.apply_boundary_conditions(&mut u_intermediate, &x_grid, &y_grid, t_mid);

            // 2. Second half-step: Implicit in y-direction, explicit in x-direction
            for i in 0..nx {
                // 2.1 Set up coefficient matrices for y-direction
                self.setup_coefficient_matrices_y(
                    &mut a_y,
                    &mut b_y,
                    x_grid[i],
                    &y_grid,
                    dy,
                    0.5 * dt,
                    t_mid,
                    &u_intermediate.slice(s![i, ..]),
                );

                // 2.2 Extract the intermediate solution column
                let mut u_col = Array1::zeros(ny);
                for j in 0..ny {
                    u_col[j] = u_intermediate[[i, j]];
                }

                // 2.3 Right-hand side vector for y-direction
                let rhs_y = b_y.dot(&u_col);

                // 2.4 Solve the linear system for y-direction
                let u_y_next = ADI2D::solve_tridiagonal(&a_y, &rhs_y)?;
                num_linear_solves += 1;

                // 2.5 Update solution column
                for j in 0..ny {
                    u_current[[i, j]] = u_y_next[j];
                }
            }

            // Apply boundary conditions to final solution for this time step
            self.apply_boundary_conditions(&mut u_current, &x_grid, &y_grid, t_next);

            // Update time
            t_values.push(t_next);

            // Save solution if needed
            if (step + 1) % save_every == 0 || step == num_steps - 1 {
                let mut u3d = Array3::zeros((nx, ny, 1));
                for i in 0..nx {
                    for j in 0..ny {
                        u3d[[i, j, 0]] = u_current[[i, j]];
                    }
                }
                solutions.push(u3d);
            }

            // Print progress if verbose
            if self.options.verbose && (step + 1) % 10 == 0 {
                println!(
                    "Step {}/{} completed, t = {:.4}",
                    step + 1,
                    num_steps,
                    t_next
                );
            }
        }

        // Convert time values to Array1
        let t_array = Array1::from_vec(t_values);

        // Compute solution time
        let computation_time = start_time.elapsed().as_secs_f64();

        // Create result
        let info = Some(format!(
            "Time steps: {num_steps}, Linear system solves: {num_linear_solves}"
        ));

        Ok(ADIResult {
            t: t_array,
            u: solutions,
            info,
            computation_time,
            num_steps,
            num_linear_solves,
        })
    }

    /// Set up coefficient matrices for the x-direction sweep
    #[allow(clippy::too_many_arguments)]
    fn setup_coefficient_matrices_x(
        &self,
        a_matrix: &mut Array2<f64>,
        b_matrix: &mut Array2<f64>,
        x_grid: &Array1<f64>,
        y: f64,
        dx: f64,
        half_dt: f64,
        t: f64,
        u_row: &ArrayView1<f64>,
    ) {
        let nx = x_grid.len();

        // Clear matrices
        a_matrix.fill(0.0);
        b_matrix.fill(0.0);

        // Set up matrices for interior points
        for i in 1..nx - 1 {
            let x = x_grid[i];
            let u_val = u_row[i];

            // Diffusion coefficient at the current point
            let d = (self.diffusion_x)(x, y, t, u_val);

            // Coefficient for diffusion term
            let r = 0.5 * d * half_dt / (dx * dx);

            // Implicit part (left-hand side)
            a_matrix[[i, i - 1]] = -r; // Coefficient for u_{i-1,j}^{n+1/2}
            a_matrix[[i, i]] = 1.0 + 2.0 * r; // Coefficient for u_{i,j}^{n+1/2}
            a_matrix[[i, i + 1]] = -r; // Coefficient for u_{i+1,j}^{n+1/2}

            // Explicit part (right-hand side)
            b_matrix[[i, i - 1]] = r; // Coefficient for u_{i-1,j}^{n}
            b_matrix[[i, i]] = 1.0 - 2.0 * r; // Coefficient for u_{i,j}^{n}
            b_matrix[[i, i + 1]] = r; // Coefficient for u_{i+1,j}^{n}

            // Add advection term in x-direction if present
            if let Some(advection_x) = &self.advection_x {
                let vx = advection_x(x, y, t, u_val);

                // Coefficient for advection term
                let c = 0.25 * vx * half_dt / dx;

                // Implicit part
                a_matrix[[i, i - 1]] -= c; // Additional term for u_{i-1,j}^{n+1/2}
                a_matrix[[i, i + 1]] += c; // Additional term for u_{i+1,j}^{n+1/2}

                // Explicit part
                b_matrix[[i, i - 1]] -= c; // Additional term for u_{i-1,j}^{n}
                b_matrix[[i, i + 1]] += c; // Additional term for u_{i+1,j}^{n}
            }

            // Add half of the reaction term if present
            if let Some(reaction) = &self.reaction_term {
                let k = reaction(x, y, t, u_val);

                // Coefficient for reaction term (half applied to each direction)
                let s = 0.25 * k * half_dt;

                // Implicit part
                a_matrix[[i, i]] += s; // Additional term for u_{i,j}^{n+1/2}

                // Explicit part
                b_matrix[[i, i]] += s; // Additional term for u_{i,j}^{n}
            }
        }

        // Apply boundary conditions in x-direction
        for bc in &self.boundary_conditions {
            if bc.dimension == 0 {
                // x-direction
                match bc.location {
                    BoundaryLocation::Lower => {
                        // Apply boundary condition at x[0]
                        match bc.bc_type {
                            BoundaryConditionType::Dirichlet => {
                                // u(a, y, t) = bc.value
                                for j in 0..nx {
                                    a_matrix[[0, j]] = 0.0;
                                    b_matrix[[0, j]] = 0.0;
                                }
                                a_matrix[[0, 0]] = 1.0;
                                b_matrix[[0, 0]] = bc.value;
                            }
                            BoundaryConditionType::Neumann => {
                                // du/dx(a, y, t) = bc.value
                                // Use second-order one-sided difference
                                for j in 0..nx {
                                    a_matrix[[0, j]] = 0.0;
                                    b_matrix[[0, j]] = 0.0;
                                }
                                a_matrix[[0, 0]] = -3.0;
                                a_matrix[[0, 1]] = 4.0;
                                a_matrix[[0, 2]] = -1.0;
                                b_matrix[[0, 0]] = 2.0 * dx * bc.value;
                            }
                            BoundaryConditionType::Robin => {
                                // a*u + b*du/dx = c
                                if let Some([a_val, b_val, c_val]) = bc.coefficients {
                                    for j in 0..nx {
                                        a_matrix[[0, j]] = 0.0;
                                        b_matrix[[0, j]] = 0.0;
                                    }
                                    a_matrix[[0, 0]] = a_val - 3.0 * b_val / (2.0 * dx);
                                    a_matrix[[0, 1]] = 4.0 * b_val / (2.0 * dx);
                                    a_matrix[[0, 2]] = -b_val / (2.0 * dx);
                                    b_matrix[[0, 0]] = c_val;
                                }
                            }
                            BoundaryConditionType::Periodic => {
                                // For periodic BCs, handled together with upper boundary
                            }
                        }
                    }
                    BoundaryLocation::Upper => {
                        // Apply boundary condition at x[nx-1]
                        let i = nx - 1;

                        match bc.bc_type {
                            BoundaryConditionType::Dirichlet => {
                                // u(b, y, t) = bc.value
                                for j in 0..nx {
                                    a_matrix[[i, j]] = 0.0;
                                    b_matrix[[i, j]] = 0.0;
                                }
                                a_matrix[[i, i]] = 1.0;
                                b_matrix[[i, i]] = bc.value;
                            }
                            BoundaryConditionType::Neumann => {
                                // du/dx(b, y, t) = bc.value
                                // Use second-order one-sided difference
                                for j in 0..nx {
                                    a_matrix[[i, j]] = 0.0;
                                    b_matrix[[i, j]] = 0.0;
                                }
                                a_matrix[[i, i]] = 3.0;
                                a_matrix[[i, i - 1]] = -4.0;
                                a_matrix[[i, i - 2]] = 1.0;
                                b_matrix[[i, i]] = 2.0 * dx * bc.value;
                            }
                            BoundaryConditionType::Robin => {
                                // a*u + b*du/dx = c
                                if let Some([a_val, b_val, c_val]) = bc.coefficients {
                                    for j in 0..nx {
                                        a_matrix[[i, j]] = 0.0;
                                        b_matrix[[i, j]] = 0.0;
                                    }
                                    a_matrix[[i, i]] = a_val + 3.0 * b_val / (2.0 * dx);
                                    a_matrix[[i, i - 1]] = -4.0 * b_val / (2.0 * dx);
                                    a_matrix[[i, i - 2]] = b_val / (2.0 * dx);
                                    b_matrix[[i, i]] = c_val;
                                }
                            }
                            BoundaryConditionType::Periodic => {
                                // Handle periodic boundary conditions in x-direction

                                // First, clear boundary rows
                                for j in 0..nx {
                                    a_matrix[[0, j]] = 0.0;
                                    a_matrix[[i, j]] = 0.0;
                                    b_matrix[[0, j]] = 0.0;
                                    b_matrix[[i, j]] = 0.0;
                                }

                                // Extract diffusion coefficient
                                let x_lower = x_grid[0];
                                let x_upper = x_grid[i];
                                let u_lower = u_row[0];
                                let u_upper = u_row[i];

                                let d_lower = (self.diffusion_x)(x_lower, y, t, u_lower);
                                let d_upper = (self.diffusion_x)(x_upper, y, t, u_upper);

                                let r_lower = 0.5 * d_lower * half_dt / (dx * dx);
                                let r_upper = 0.5 * d_upper * half_dt / (dx * dx);

                                // Lower boundary (connects to upper)
                                a_matrix[[0, i]] = -r_lower;
                                a_matrix[[0, 0]] = 1.0 + 2.0 * r_lower;
                                a_matrix[[0, 1]] = -r_lower;

                                b_matrix[[0, i]] = r_lower;
                                b_matrix[[0, 0]] = 1.0 - 2.0 * r_lower;
                                b_matrix[[0, 1]] = r_lower;

                                // Upper boundary (connects to lower)
                                a_matrix[[i, i - 1]] = -r_upper;
                                a_matrix[[i, i]] = 1.0 + 2.0 * r_upper;
                                a_matrix[[i, 0]] = -r_upper;

                                b_matrix[[i, i - 1]] = r_upper;
                                b_matrix[[i, i]] = 1.0 - 2.0 * r_upper;
                                b_matrix[[i, 0]] = r_upper;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Set up coefficient matrices for the y-direction sweep
    #[allow(clippy::too_many_arguments)]
    fn setup_coefficient_matrices_y(
        &self,
        a_matrix: &mut Array2<f64>,
        b_matrix: &mut Array2<f64>,
        x: f64,
        y_grid: &Array1<f64>,
        dy: f64,
        half_dt: f64,
        t: f64,
        u_col: &ArrayView1<f64>,
    ) {
        let ny = y_grid.len();

        // Clear matrices
        a_matrix.fill(0.0);
        b_matrix.fill(0.0);

        // Set up matrices for interior points
        for j in 1..ny - 1 {
            let y = y_grid[j];
            let u_val = u_col[j];

            // Diffusion coefficient at the current point
            let d = (self.diffusion_y)(x, y, t, u_val);

            // Coefficient for diffusion term
            let r = 0.5 * d * half_dt / (dy * dy);

            // Implicit part (left-hand side)
            a_matrix[[j, j - 1]] = -r; // Coefficient for u_{i,j-1}^{n+1}
            a_matrix[[j, j]] = 1.0 + 2.0 * r; // Coefficient for u_{i,j}^{n+1}
            a_matrix[[j, j + 1]] = -r; // Coefficient for u_{i,j+1}^{n+1}

            // Explicit part (right-hand side)
            b_matrix[[j, j - 1]] = r; // Coefficient for u_{i,j-1}^{n+1/2}
            b_matrix[[j, j]] = 1.0 - 2.0 * r; // Coefficient for u_{i,j}^{n+1/2}
            b_matrix[[j, j + 1]] = r; // Coefficient for u_{i,j+1}^{n+1/2}

            // Add advection term in y-direction if present
            if let Some(advection_y) = &self.advection_y {
                let vy = advection_y(x, y, t, u_val);

                // Coefficient for advection term
                let c = 0.25 * vy * half_dt / dy;

                // Implicit part
                a_matrix[[j, j - 1]] -= c; // Additional term for u_{i,j-1}^{n+1}
                a_matrix[[j, j + 1]] += c; // Additional term for u_{i,j+1}^{n+1}

                // Explicit part
                b_matrix[[j, j - 1]] -= c; // Additional term for u_{i,j-1}^{n+1/2}
                b_matrix[[j, j + 1]] += c; // Additional term for u_{i,j+1}^{n+1/2}
            }

            // Add half of the reaction term if present
            if let Some(reaction) = &self.reaction_term {
                let k = reaction(x, y, t, u_val);

                // Coefficient for reaction term (half applied to each direction)
                let s = 0.25 * k * half_dt;

                // Implicit part
                a_matrix[[j, j]] += s; // Additional term for u_{i,j}^{n+1}

                // Explicit part
                b_matrix[[j, j]] += s; // Additional term for u_{i,j}^{n+1/2}
            }
        }

        // Apply boundary conditions in y-direction
        for bc in &self.boundary_conditions {
            if bc.dimension == 1 {
                // y-direction
                match bc.location {
                    BoundaryLocation::Lower => {
                        // Apply boundary condition at y[0]
                        match bc.bc_type {
                            BoundaryConditionType::Dirichlet => {
                                // u(x, c, t) = bc.value
                                for j in 0..ny {
                                    a_matrix[[0, j]] = 0.0;
                                    b_matrix[[0, j]] = 0.0;
                                }
                                a_matrix[[0, 0]] = 1.0;
                                b_matrix[[0, 0]] = bc.value;
                            }
                            BoundaryConditionType::Neumann => {
                                // du/dy(x, c, t) = bc.value
                                // Use second-order one-sided difference
                                for j in 0..ny {
                                    a_matrix[[0, j]] = 0.0;
                                    b_matrix[[0, j]] = 0.0;
                                }
                                a_matrix[[0, 0]] = -3.0;
                                a_matrix[[0, 1]] = 4.0;
                                a_matrix[[0, 2]] = -1.0;
                                b_matrix[[0, 0]] = 2.0 * dy * bc.value;
                            }
                            BoundaryConditionType::Robin => {
                                // a*u + b*du/dy = c
                                if let Some([a_val, b_val, c_val]) = bc.coefficients {
                                    for j in 0..ny {
                                        a_matrix[[0, j]] = 0.0;
                                        b_matrix[[0, j]] = 0.0;
                                    }
                                    a_matrix[[0, 0]] = a_val - 3.0 * b_val / (2.0 * dy);
                                    a_matrix[[0, 1]] = 4.0 * b_val / (2.0 * dy);
                                    a_matrix[[0, 2]] = -b_val / (2.0 * dy);
                                    b_matrix[[0, 0]] = c_val;
                                }
                            }
                            BoundaryConditionType::Periodic => {
                                // For periodic BCs, handled together with upper boundary
                            }
                        }
                    }
                    BoundaryLocation::Upper => {
                        // Apply boundary condition at y[ny-1]
                        let j = ny - 1;

                        match bc.bc_type {
                            BoundaryConditionType::Dirichlet => {
                                // u(x, d, t) = bc.value
                                for i in 0..ny {
                                    a_matrix[[j, i]] = 0.0;
                                    b_matrix[[j, i]] = 0.0;
                                }
                                a_matrix[[j, j]] = 1.0;
                                b_matrix[[j, j]] = bc.value;
                            }
                            BoundaryConditionType::Neumann => {
                                // du/dy(x, d, t) = bc.value
                                // Use second-order one-sided difference
                                for i in 0..ny {
                                    a_matrix[[j, i]] = 0.0;
                                    b_matrix[[j, i]] = 0.0;
                                }
                                a_matrix[[j, j]] = 3.0;
                                a_matrix[[j, j - 1]] = -4.0;
                                a_matrix[[j, j - 2]] = 1.0;
                                b_matrix[[j, j]] = 2.0 * dy * bc.value;
                            }
                            BoundaryConditionType::Robin => {
                                // a*u + b*du/dy = c
                                if let Some([a_val, b_val, c_val]) = bc.coefficients {
                                    for i in 0..ny {
                                        a_matrix[[j, i]] = 0.0;
                                        b_matrix[[j, i]] = 0.0;
                                    }
                                    a_matrix[[j, j]] = a_val + 3.0 * b_val / (2.0 * dy);
                                    a_matrix[[j, j - 1]] = -4.0 * b_val / (2.0 * dy);
                                    a_matrix[[j, j - 2]] = b_val / (2.0 * dy);
                                    b_matrix[[j, j]] = c_val;
                                }
                            }
                            BoundaryConditionType::Periodic => {
                                // Handle periodic boundary conditions in y-direction

                                // First, clear boundary rows
                                for i in 0..ny {
                                    a_matrix[[0, i]] = 0.0;
                                    a_matrix[[j, i]] = 0.0;
                                    b_matrix[[0, i]] = 0.0;
                                    b_matrix[[j, i]] = 0.0;
                                }

                                // Extract diffusion coefficient
                                let y_lower = y_grid[0];
                                let y_upper = y_grid[j];
                                let u_lower = u_col[0];
                                let u_upper = u_col[j];

                                let d_lower = (self.diffusion_y)(x, y_lower, t, u_lower);
                                let d_upper = (self.diffusion_y)(x, y_upper, t, u_upper);

                                let r_lower = 0.5 * d_lower * half_dt / (dy * dy);
                                let r_upper = 0.5 * d_upper * half_dt / (dy * dy);

                                // Lower boundary (connects to upper)
                                a_matrix[[0, j]] = -r_lower;
                                a_matrix[[0, 0]] = 1.0 + 2.0 * r_lower;
                                a_matrix[[0, 1]] = -r_lower;

                                b_matrix[[0, j]] = r_lower;
                                b_matrix[[0, 0]] = 1.0 - 2.0 * r_lower;
                                b_matrix[[0, 1]] = r_lower;

                                // Upper boundary (connects to lower)
                                a_matrix[[j, j - 1]] = -r_upper;
                                a_matrix[[j, j]] = 1.0 + 2.0 * r_upper;
                                a_matrix[[j, 0]] = -r_upper;

                                b_matrix[[j, j - 1]] = r_upper;
                                b_matrix[[j, j]] = 1.0 - 2.0 * r_upper;
                                b_matrix[[j, 0]] = r_upper;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Apply boundary conditions to the solution
    fn apply_boundary_conditions(
        &self,
        u: &mut Array2<f64>,
        x_grid: &Array1<f64>,
        y_grid: &Array1<f64>,
        _t: f64,
    ) {
        let nx = x_grid.len();
        let ny = y_grid.len();

        for bc in &self.boundary_conditions {
            match (bc.dimension, bc.location, bc.bc_type) {
                // x-direction Dirichlet boundary conditions
                (0, BoundaryLocation::Lower, BoundaryConditionType::Dirichlet) => {
                    for j in 0..ny {
                        u[[0, j]] = bc.value;
                    }
                }
                (0, BoundaryLocation::Upper, BoundaryConditionType::Dirichlet) => {
                    for j in 0..ny {
                        u[[nx - 1, j]] = bc.value;
                    }
                }

                // y-direction Dirichlet boundary conditions
                (1, BoundaryLocation::Lower, BoundaryConditionType::Dirichlet) => {
                    for i in 0..nx {
                        u[[i, 0]] = bc.value;
                    }
                }
                (1, BoundaryLocation::Upper, BoundaryConditionType::Dirichlet) => {
                    for i in 0..nx {
                        u[[i, ny - 1]] = bc.value;
                    }
                }

                // Other boundary conditions are handled in the coefficient matrices
                _ => {}
            }
        }
    }

    /// Solve a tridiagonal linear system using the Thomas algorithm
    fn solve_tridiagonal(a: &Array2<f64>, b: &Array1<f64>) -> PDEResult<Array1<f64>> {
        let n = b.len();

        // Extract the tridiagonal elements
        let mut lower = Array1::zeros(n - 1);
        let mut diagonal = Array1::zeros(n);
        let mut upper = Array1::zeros(n - 1);

        for i in 0..n {
            diagonal[i] = a[[i, i]];
            if i < n - 1 {
                upper[i] = a[[i, i + 1]];
            }
            if i > 0 {
                lower[i - 1] = a[[i, i - 1]];
            }
        }

        // Solve tridiagonal system using Thomas algorithm
        let mut solution = Array1::zeros(n);
        let mut temp_diag = diagonal.clone();
        let mut temp_rhs = b.to_owned();

        // Forward sweep
        for i in 1..n {
            let w = lower[i - 1] / temp_diag[i - 1];
            temp_diag[i] -= w * upper[i - 1];
            temp_rhs[i] -= w * temp_rhs[i - 1];
        }

        // Backward sweep
        solution[n - 1] = temp_rhs[n - 1] / temp_diag[n - 1];
        for i in (0..n - 1).rev() {
            solution[i] = (temp_rhs[i] - upper[i] * solution[i + 1]) / temp_diag[i];
        }

        Ok(solution)
    }
}

/// Convert an ADIResult to a PDESolution
impl From<ADIResult> for PDESolution<f64> {
    fn from(result: ADIResult) -> Self {
        let mut grids = Vec::new();

        // Add time grid
        grids.push(result.t.clone());

        // Extract spatial grids from solution shape
        let nx = result.u[0].shape()[0];
        let ny = result.u[0].shape()[1];

        // Create spatial grids (we don't have the actual grid values, so use linspace)
        let x_grid = Array1::linspace(0.0, 1.0, nx);
        let y_grid = Array1::linspace(0.0, 1.0, ny);

        grids.push(x_grid);
        grids.push(y_grid);

        // Convert 3D arrays to 2D arrays for PDESolution format
        let mut values = Vec::new();
        for u3d in result.u {
            let mut u2d = Array2::zeros((nx, ny));
            for i in 0..nx {
                for j in 0..ny {
                    u2d[[i, j]] = u3d[[i, j, 0]];
                }
            }
            values.push(u2d);
        }

        // Create solver info
        let info = PDESolverInfo {
            num_iterations: result.num_linear_solves,
            computation_time: result.computation_time,
            residual_norm: None,
            convergence_history: None,
            method: "ADI Method".to_string(),
        };

        PDESolution {
            grids,
            values,
            error_estimate: None,
            info,
        }
    }
}
