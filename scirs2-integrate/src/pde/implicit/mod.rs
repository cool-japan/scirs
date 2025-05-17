//! Implicit methods for solving PDEs
//!
//! This module provides implementations of implicit time-stepping schemes for
//! solving partial differential equations (PDEs). Implicit methods are generally 
//! more stable than explicit methods and can handle stiff problems more efficiently.
//!
//! ## Supported Methods
//!
//! * Crank-Nicolson method: A second-order implicit method that is A-stable
//! * Backward Euler method: A first-order fully implicit method that is L-stable
//! * Trapezoidal rule: A second-order implicit method
//! * Alternating Direction Implicit (ADI) method: An efficient operator splitting method for
//!   multi-dimensional problems

use ndarray::{Array1, Array2, ArrayView1};
use std::time::Instant;

use crate::pde::{
    Domain, BoundaryCondition, PDEError, PDEResult, PDESolution, PDESolverInfo,
    BoundaryConditionType, BoundaryLocation
};
use crate::pde::finite_difference::FiniteDifferenceScheme;

// Re-export ADI implementation
mod adi;
pub use adi::{ADI2D, ADIResult};

/// Available implicit time-stepping schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImplicitMethod {
    /// Backward Euler method (first-order, L-stable)
    BackwardEuler,
    
    /// Crank-Nicolson method (second-order, A-stable)
    CrankNicolson,
    
    /// Trapezoidal rule (second-order, A-stable)
    TrapezoidalRule,
    
    /// Alternating Direction Implicit method (efficient for multi-dimensional problems)
    ADI,
}

/// Options for implicit PDE solvers
#[derive(Debug, Clone)]
pub struct ImplicitOptions {
    /// Implicit time-stepping method to use
    pub method: ImplicitMethod,
    
    /// Tolerance for iterative solvers
    pub tolerance: f64,
    
    /// Maximum number of iterations for linear system solver
    pub max_iterations: usize,
    
    /// Fixed time step (if None, adaptive time-stepping will be used)
    pub dt: Option<f64>,
    
    /// Minimum time step for adaptive time-stepping
    pub min_dt: Option<f64>,
    
    /// Maximum time step for adaptive time-stepping
    pub max_dt: Option<f64>,
    
    /// Time steps to save (if None, all steps will be saved)
    pub save_every: Option<usize>,
    
    /// Print detailed progress information
    pub verbose: bool,
}

impl Default for ImplicitOptions {
    fn default() -> Self {
        ImplicitOptions {
            method: ImplicitMethod::CrankNicolson,
            tolerance: 1e-6,
            max_iterations: 100,
            dt: Some(0.01),
            min_dt: Some(1e-6),
            max_dt: Some(0.1),
            save_every: None,
            verbose: false,
        }
    }
}

/// Result of implicit method solution
pub struct ImplicitResult {
    /// Time points
    pub t: Array1<f64>,
    
    /// Solution values, indexed as [time, space...]
    pub u: Vec<Array2<f64>>,
    
    /// Solver information
    pub info: Option<String>,
    
    /// Computation time
    pub computation_time: f64,
    
    /// Number of time steps
    pub num_steps: usize,
    
    /// Number of linear system solves
    pub num_linear_solves: usize,
}

/// Crank-Nicolson solver for 1D parabolic PDEs
///
/// This solver uses the Crank-Nicolson method, which is a second-order
/// implicit method that is unconditionally stable for the heat equation.
pub struct CrankNicolson1D {
    /// Spatial domain
    domain: Domain,
    
    /// Time range [t_start, t_end]
    time_range: [f64; 2],
    
    /// Diffusion coefficient function D(x, t, u) for ∂²u/∂x²
    diffusion_coeff: Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>,
    
    /// Advection coefficient function v(x, t, u) for ∂u/∂x
    advection_coeff: Option<Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>>,
    
    /// Reaction term function f(x, t, u)
    reaction_term: Option<Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>>,
    
    /// Initial condition function u(x, 0)
    initial_condition: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    
    /// Boundary conditions
    boundary_conditions: Vec<BoundaryCondition<f64>>,
    
    /// Finite difference scheme for spatial discretization
    fd_scheme: FiniteDifferenceScheme,
    
    /// Solver options
    options: ImplicitOptions,
}

impl CrankNicolson1D {
    /// Create a new Crank-Nicolson solver for 1D parabolic PDEs
    pub fn new(
        domain: Domain,
        time_range: [f64; 2],
        diffusion_coeff: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
        initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<ImplicitOptions>,
    ) -> PDEResult<Self> {
        // Validate the domain
        if domain.dimensions() != 1 {
            return Err(PDEError::DomainError(
                "Domain must be 1-dimensional for 1D Crank-Nicolson solver".to_string()
            ));
        }
        
        // Validate time range
        if time_range[0] >= time_range[1] {
            return Err(PDEError::DomainError(
                "Invalid time range: start must be less than end".to_string()
            ));
        }
        
        // Validate boundary conditions
        if boundary_conditions.len() != 2 {
            return Err(PDEError::BoundaryConditions(
                "1D parabolic PDE requires exactly 2 boundary conditions".to_string()
            ));
        }
        
        // Ensure we have both lower and upper boundary conditions
        let has_lower = boundary_conditions.iter()
            .any(|bc| bc.location == BoundaryLocation::Lower);
        let has_upper = boundary_conditions.iter()
            .any(|bc| bc.location == BoundaryLocation::Upper);
            
        if !has_lower || !has_upper {
            return Err(PDEError::BoundaryConditions(
                "1D parabolic PDE requires both lower and upper boundary conditions".to_string()
            ));
        }
        
        let mut options = options.unwrap_or_default();
        options.method = ImplicitMethod::CrankNicolson;
        
        Ok(CrankNicolson1D {
            domain,
            time_range,
            diffusion_coeff: Box::new(diffusion_coeff),
            advection_coeff: None,
            reaction_term: None,
            initial_condition: Box::new(initial_condition),
            boundary_conditions,
            fd_scheme: FiniteDifferenceScheme::CentralDifference,
            options,
        })
    }
    
    /// Add an advection term to the PDE
    pub fn with_advection(
        mut self,
        advection_coeff: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.advection_coeff = Some(Box::new(advection_coeff));
        self
    }
    
    /// Add a reaction term to the PDE
    pub fn with_reaction(
        mut self,
        reaction_term: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.reaction_term = Some(Box::new(reaction_term));
        self
    }
    
    /// Set the finite difference scheme for spatial discretization
    pub fn with_fd_scheme(mut self, scheme: FiniteDifferenceScheme) -> Self {
        self.fd_scheme = scheme;
        self
    }
    
    /// Solve the PDE using the Crank-Nicolson method
    pub fn solve(&self) -> PDEResult<ImplicitResult> {
        let start_time = Instant::now();
        
        // Generate spatial grid
        let x_grid = self.domain.grid(0)?;
        let nx = x_grid.len();
        let dx = self.domain.grid_spacing(0)?;
        
        // Time step
        let dt = self.options.dt.unwrap_or(0.01);
        
        // Calculate number of time steps
        let t_start = self.time_range[0];
        let t_end = self.time_range[1];
        let num_steps = ((t_end - t_start) / dt).ceil() as usize;
        
        // Initialize time array
        let mut t_values = Vec::with_capacity(num_steps + 1);
        t_values.push(t_start);
        
        // Initialize solution arrays
        let mut u_current = Array1::zeros(nx);
        
        // Apply initial condition
        for (i, &x) in x_grid.iter().enumerate() {
            u_current[i] = (self.initial_condition)(x);
        }
        
        // Apply Dirichlet boundary conditions to initial condition
        apply_dirichlet_conditions_to_initial_1d(&mut u_current, &self.boundary_conditions);
        
        // Store solutions
        let save_every = self.options.save_every.unwrap_or(1);
        let mut solutions = Vec::with_capacity((num_steps + 1) / save_every + 1);
        solutions.push(u_current.clone().into_shape_with_order((nx, 1)).unwrap());
        
        // Initialize matrices for Crank-Nicolson method
        let mut a_matrix = Array2::zeros((nx, nx));
        let mut b_matrix = Array2::zeros((nx, nx));
        
        // Track solver statistics
        let mut num_linear_solves = 0;
        
        // Time-stepping loop
        for step in 0..num_steps {
            let t_current = t_start + step as f64 * dt;
            let t_next = t_current + dt;
            
            // Set up coefficient matrices based on PDE terms and boundary conditions
            self.setup_coefficient_matrices(
                &mut a_matrix, 
                &mut b_matrix, 
                &x_grid, 
                dx, 
                dt, 
                t_current,
            );
            
            // Right-hand side vector
            let rhs = b_matrix.dot(&u_current);
            
            // Solve the linear system: A * u_{n+1} = B * u_n
            let u_next = self.solve_linear_system(&a_matrix, &rhs.view())?;
            num_linear_solves += 1;
            
            // Update current solution and time
            u_current = u_next;
            t_values.push(t_next);
            
            // Save solution if needed
            if (step + 1) % save_every == 0 || step == num_steps - 1 {
                solutions.push(u_current.clone().into_shape_with_order((nx, 1)).unwrap());
            }
            
            // Print progress if verbose
            if self.options.verbose && (step + 1) % 10 == 0 {
                println!("Step {}/{} completed, t = {:.4}", step + 1, num_steps, t_next);
            }
        }
        
        // Convert time values to Array1
        let t_array = Array1::from_vec(t_values);
        
        // Compute solution time
        let computation_time = start_time.elapsed().as_secs_f64();
        
        // Create result
        let info = Some(format!(
            "Time steps: {}, Linear system solves: {}",
            num_steps, num_linear_solves
        ));
        
        Ok(ImplicitResult {
            t: t_array,
            u: solutions,
            info,
            computation_time,
            num_steps,
            num_linear_solves,
        })
    }
    
    /// Set up coefficient matrices for the Crank-Nicolson method
    fn setup_coefficient_matrices(
        &self,
        a_matrix: &mut Array2<f64>,
        b_matrix: &mut Array2<f64>,
        x_grid: &Array1<f64>,
        dx: f64,
        dt: f64,
        t: f64,
    ) {
        let nx = x_grid.len();
        
        // Clear matrices
        a_matrix.fill(0.0);
        b_matrix.fill(0.0);
        
        // Set up implicit (A) and explicit (B) matrices for interior points
        for i in 1..nx-1 {
            let x = x_grid[i];
            let u_val = 0.0; // Used for linearization around previous state if needed
            
            // Diffusion coefficient at the current point
            let d = (self.diffusion_coeff)(x, t, u_val);
            
            // Crank-Nicolson coefficients for diffusion term
            let r = 0.5 * d * dt / (dx * dx);
            
            // Implicit part (left-hand side)
            a_matrix[[i, i-1]] = -r;                // Coefficient for u_{i-1}^{n+1}
            a_matrix[[i, i]] = 1.0 + 2.0 * r;       // Coefficient for u_{i}^{n+1}
            a_matrix[[i, i+1]] = -r;                // Coefficient for u_{i+1}^{n+1}
            
            // Explicit part (right-hand side)
            b_matrix[[i, i-1]] = r;                 // Coefficient for u_{i-1}^{n}
            b_matrix[[i, i]] = 1.0 - 2.0 * r;       // Coefficient for u_{i}^{n}
            b_matrix[[i, i+1]] = r;                 // Coefficient for u_{i+1}^{n}
            
            // Add advection term if present
            if let Some(advection) = &self.advection_coeff {
                let v = advection(x, t, u_val);
                
                // Coefficient for advection term (central difference)
                let c = 0.25 * v * dt / dx;
                
                // Implicit part
                a_matrix[[i, i-1]] -= c;            // Additional term for u_{i-1}^{n+1}
                a_matrix[[i, i+1]] += c;            // Additional term for u_{i+1}^{n+1}
                
                // Explicit part
                b_matrix[[i, i-1]] -= c;            // Additional term for u_{i-1}^{n}
                b_matrix[[i, i+1]] += c;            // Additional term for u_{i+1}^{n}
            }
            
            // Add reaction term if present
            if let Some(reaction) = &self.reaction_term {
                // For linear reaction terms of the form R(u) = ku
                // For nonlinear terms, would need to linearize around previous state
                let k = reaction(x, t, u_val);
                
                // Coefficient for reaction term
                let s = 0.5 * k * dt;
                
                // Implicit part
                a_matrix[[i, i]] += s;              // Additional term for u_{i}^{n+1}
                
                // Explicit part
                b_matrix[[i, i]] += s;              // Additional term for u_{i}^{n}
            }
        }
        
        // Apply boundary conditions
        for bc in &self.boundary_conditions {
            match bc.location {
                BoundaryLocation::Lower => {
                    // Apply boundary condition at x[0]
                    match bc.bc_type {
                        BoundaryConditionType::Dirichlet => {
                            // u(a, t) = bc.value
                            // Set the row to enforce u_0^{n+1} = bc.value
                            for j in 0..nx {
                                a_matrix[[0, j]] = 0.0;
                                b_matrix[[0, j]] = 0.0;
                            }
                            a_matrix[[0, 0]] = 1.0;
                            b_matrix[[0, 0]] = 0.0;
                            
                            // Add boundary value to RHS
                            b_matrix[[0, 0]] = bc.value;
                        },
                        BoundaryConditionType::Neumann => {
                            // du/dx(a, t) = bc.value
                            // Use second-order one-sided difference: 
                            // (-3u_0 + 4u_1 - u_2)/(2dx) = bc.value
                            
                            for j in 0..nx {
                                a_matrix[[0, j]] = 0.0;
                                b_matrix[[0, j]] = 0.0;
                            }
                            
                            // Implicit part
                            a_matrix[[0, 0]] = -3.0;
                            a_matrix[[0, 1]] = 4.0;
                            a_matrix[[0, 2]] = -1.0;
                            
                            // RHS is the boundary value scaled by 2*dx
                            b_matrix[[0, 0]] = 2.0 * dx * bc.value;
                        },
                        BoundaryConditionType::Robin => {
                            // a*u + b*du/dx = c
                            if let Some([a_val, b_val, c_val]) = bc.coefficients {
                                // Use second-order one-sided difference for the derivative:
                                // (-3u_0 + 4u_1 - u_2)/(2dx)
                                
                                for j in 0..nx {
                                    a_matrix[[0, j]] = 0.0;
                                    b_matrix[[0, j]] = 0.0;
                                }
                                
                                // Implicit part: a*u_0 + b*(-3u_0 + 4u_1 - u_2)/(2dx) = c
                                a_matrix[[0, 0]] = a_val - 3.0 * b_val / (2.0 * dx);
                                a_matrix[[0, 1]] = 4.0 * b_val / (2.0 * dx);
                                a_matrix[[0, 2]] = -b_val / (2.0 * dx);
                                
                                // RHS is the boundary value c
                                b_matrix[[0, 0]] = c_val;
                            }
                        },
                        BoundaryConditionType::Periodic => {
                            // For periodic BCs, special handling at both boundaries together is needed
                            // (see below)
                        },
                    }
                },
                BoundaryLocation::Upper => {
                    // Apply boundary condition at x[nx-1]
                    let i = nx - 1;
                    
                    match bc.bc_type {
                        BoundaryConditionType::Dirichlet => {
                            // u(b, t) = bc.value
                            // Set the row to enforce u_{nx-1}^{n+1} = bc.value
                            for j in 0..nx {
                                a_matrix[[i, j]] = 0.0;
                                b_matrix[[i, j]] = 0.0;
                            }
                            a_matrix[[i, i]] = 1.0;
                            b_matrix[[i, i]] = 0.0;
                            
                            // Add boundary value to RHS
                            b_matrix[[i, i]] = bc.value;
                        },
                        BoundaryConditionType::Neumann => {
                            // du/dx(b, t) = bc.value
                            // Use second-order one-sided difference:
                            // (3u_{nx-1} - 4u_{nx-2} + u_{nx-3})/(2dx) = bc.value
                            
                            for j in 0..nx {
                                a_matrix[[i, j]] = 0.0;
                                b_matrix[[i, j]] = 0.0;
                            }
                            
                            // Implicit part
                            a_matrix[[i, i]] = 3.0;
                            a_matrix[[i, i-1]] = -4.0;
                            a_matrix[[i, i-2]] = 1.0;
                            
                            // RHS is the boundary value scaled by 2*dx
                            b_matrix[[i, i]] = 2.0 * dx * bc.value;
                        },
                        BoundaryConditionType::Robin => {
                            // a*u + b*du/dx = c
                            if let Some([a_val, b_val, c_val]) = bc.coefficients {
                                // Use second-order one-sided difference for the derivative:
                                // (3u_{nx-1} - 4u_{nx-2} + u_{nx-3})/(2dx)
                                
                                for j in 0..nx {
                                    a_matrix[[i, j]] = 0.0;
                                    b_matrix[[i, j]] = 0.0;
                                }
                                
                                // Implicit part: a*u_{nx-1} + b*(3u_{nx-1} - 4u_{nx-2} + u_{nx-3})/(2dx) = c
                                a_matrix[[i, i]] = a_val + 3.0 * b_val / (2.0 * dx);
                                a_matrix[[i, i-1]] = -4.0 * b_val / (2.0 * dx);
                                a_matrix[[i, i-2]] = b_val / (2.0 * dx);
                                
                                // RHS is the boundary value c
                                b_matrix[[i, i]] = c_val;
                            }
                        },
                        BoundaryConditionType::Periodic => {
                            // Handle periodic boundary conditions
                            // We need to make u_0 = u_{nx-1} and ensure the fluxes match at the boundaries
                            
                            // First, clear the boundary rows
                            for j in 0..nx {
                                a_matrix[[0, j]] = 0.0;
                                a_matrix[[i, j]] = 0.0;
                                b_matrix[[0, j]] = 0.0;
                                b_matrix[[i, j]] = 0.0;
                            }
                            
                            // For the lower boundary (i=0), set up equation connecting to upper boundary
                            // Use periodic stencil for diffusion
                            let x = x_grid[0];
                            let u_val = 0.0; // Used for linearization if needed
                            let d = (self.diffusion_coeff)(x, t, u_val);
                            let r = 0.5 * d * dt / (dx * dx);
                            
                            a_matrix[[0, i]] = -r;                // Coefficient for u_{nx-1}^{n+1}
                            a_matrix[[0, 0]] = 1.0 + 2.0 * r;     // Coefficient for u_{0}^{n+1}
                            a_matrix[[0, 1]] = -r;                // Coefficient for u_{1}^{n+1}
                            
                            b_matrix[[0, i]] = r;                 // Coefficient for u_{nx-1}^{n}
                            b_matrix[[0, 0]] = 1.0 - 2.0 * r;     // Coefficient for u_{0}^{n}
                            b_matrix[[0, 1]] = r;                 // Coefficient for u_{1}^{n}
                            
                            // For the upper boundary (i=nx-1), set up equation connecting to lower boundary
                            let x = x_grid[i];
                            let d = (self.diffusion_coeff)(x, t, u_val);
                            let r = 0.5 * d * dt / (dx * dx);
                            
                            a_matrix[[i, i-1]] = -r;              // Coefficient for u_{nx-2}^{n+1}
                            a_matrix[[i, i]] = 1.0 + 2.0 * r;     // Coefficient for u_{nx-1}^{n+1}
                            a_matrix[[i, 0]] = -r;                // Coefficient for u_{0}^{n+1}
                            
                            b_matrix[[i, i-1]] = r;               // Coefficient for u_{nx-2}^{n}
                            b_matrix[[i, i]] = 1.0 - 2.0 * r;     // Coefficient for u_{nx-1}^{n}
                            b_matrix[[i, 0]] = r;                 // Coefficient for u_{0}^{n}
                            
                            // Add advection term if present
                            if let Some(advection) = &self.advection_coeff {
                                // Lower boundary
                                let v = advection(x_grid[0], t, u_val);
                                let c = 0.25 * v * dt / dx;
                                
                                a_matrix[[0, i]] -= c;            // Additional term for u_{nx-1}^{n+1}
                                a_matrix[[0, 1]] += c;            // Additional term for u_{1}^{n+1}
                                
                                b_matrix[[0, i]] -= c;            // Additional term for u_{nx-1}^{n}
                                b_matrix[[0, 1]] += c;            // Additional term for u_{1}^{n}
                                
                                // Upper boundary
                                let v = advection(x_grid[i], t, u_val);
                                let c = 0.25 * v * dt / dx;
                                
                                a_matrix[[i, i-1]] -= c;          // Additional term for u_{nx-2}^{n+1}
                                a_matrix[[i, 0]] += c;            // Additional term for u_{0}^{n+1}
                                
                                b_matrix[[i, i-1]] -= c;          // Additional term for u_{nx-2}^{n}
                                b_matrix[[i, 0]] += c;            // Additional term for u_{0}^{n}
                            }
                        },
                    }
                },
            }
        }
        
        // Special case for periodic boundary conditions
        // If we have both lower and upper periodic boundary conditions,
        // we need to make sure they're consistent
        let has_periodic_lower = self.boundary_conditions.iter()
            .any(|bc| bc.location == BoundaryLocation::Lower && 
                 bc.bc_type == BoundaryConditionType::Periodic);
        
        let has_periodic_upper = self.boundary_conditions.iter()
            .any(|bc| bc.location == BoundaryLocation::Upper && 
                 bc.bc_type == BoundaryConditionType::Periodic);
        
        if has_periodic_lower && has_periodic_upper {
            // Already handled in the boundary condition loop
        }
    }
    
    /// Solve the linear system Ax = b
    fn solve_linear_system(&self, a: &Array2<f64>, b: &ArrayView1<f64>) -> PDEResult<Array1<f64>> {
        let n = b.len();
        
        // Simple tridiagonal solver for Crank-Nicolson matrices
        // For a general solver, use a specialized linear algebra library
        
        // Check if the matrix is tridiagonal
        let is_tridiagonal = a.indexed_iter()
            .filter(|((i, j), &val)| val != 0.0 && (*i as isize - *j as isize).abs() > 1)
            .count() == 0;
        
        if is_tridiagonal {
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
            for i in (0..n-1).rev() {
                solution[i] = (temp_rhs[i] - upper[i] * solution[i + 1]) / temp_diag[i];
            }
            
            Ok(solution)
        } else {
            // General case: Gaussian elimination with partial pivoting
            // For a real implementation, use a specialized linear algebra library
            
            // Create copies of A and b
            let mut a_copy = a.clone();
            let mut b_copy = b.to_owned();
            
            // Forward elimination
            for i in 0..n {
                // Find pivot
                let mut max_val = a_copy[[i, i]].abs();
                let mut max_row = i;
                
                for k in i+1..n {
                    if a_copy[[k, i]].abs() > max_val {
                        max_val = a_copy[[k, i]].abs();
                        max_row = k;
                    }
                }
                
                // Check if matrix is singular
                if max_val < 1e-10 {
                    return Err(PDEError::Other(
                        "Matrix is singular or nearly singular".to_string()
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
                for k in i+1..n {
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
                for j in i+1..n {
                    sum += a_copy[[i, j]] * x[j];
                }
                
                x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
            }
            
            Ok(x)
        }
    }
}

/// Backward Euler solver for 1D parabolic PDEs
///
/// This solver uses the Backward Euler method, which is a first-order
/// fully implicit method that is L-stable and well-suited for stiff problems.
pub struct BackwardEuler1D {
    /// Spatial domain
    domain: Domain,
    
    /// Time range [t_start, t_end]
    time_range: [f64; 2],
    
    /// Diffusion coefficient function D(x, t, u) for ∂²u/∂x²
    diffusion_coeff: Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>,
    
    /// Advection coefficient function v(x, t, u) for ∂u/∂x
    advection_coeff: Option<Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>>,
    
    /// Reaction term function f(x, t, u)
    reaction_term: Option<Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>>,
    
    /// Initial condition function u(x, 0)
    initial_condition: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    
    /// Boundary conditions
    boundary_conditions: Vec<BoundaryCondition<f64>>,
    
    /// Finite difference scheme for spatial discretization
    fd_scheme: FiniteDifferenceScheme,
    
    /// Solver options
    options: ImplicitOptions,
}

impl BackwardEuler1D {
    /// Create a new Backward Euler solver for 1D parabolic PDEs
    pub fn new(
        domain: Domain,
        time_range: [f64; 2],
        diffusion_coeff: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
        initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<ImplicitOptions>,
    ) -> PDEResult<Self> {
        // Validate the domain
        if domain.dimensions() != 1 {
            return Err(PDEError::DomainError(
                "Domain must be 1-dimensional for 1D Backward Euler solver".to_string()
            ));
        }
        
        // Validate time range
        if time_range[0] >= time_range[1] {
            return Err(PDEError::DomainError(
                "Invalid time range: start must be less than end".to_string()
            ));
        }
        
        // Validate boundary conditions
        if boundary_conditions.len() != 2 {
            return Err(PDEError::BoundaryConditions(
                "1D parabolic PDE requires exactly 2 boundary conditions".to_string()
            ));
        }
        
        // Ensure we have both lower and upper boundary conditions
        let has_lower = boundary_conditions.iter()
            .any(|bc| bc.location == BoundaryLocation::Lower);
        let has_upper = boundary_conditions.iter()
            .any(|bc| bc.location == BoundaryLocation::Upper);
            
        if !has_lower || !has_upper {
            return Err(PDEError::BoundaryConditions(
                "1D parabolic PDE requires both lower and upper boundary conditions".to_string()
            ));
        }
        
        let mut options = options.unwrap_or_default();
        options.method = ImplicitMethod::BackwardEuler;
        
        Ok(BackwardEuler1D {
            domain,
            time_range,
            diffusion_coeff: Box::new(diffusion_coeff),
            advection_coeff: None,
            reaction_term: None,
            initial_condition: Box::new(initial_condition),
            boundary_conditions,
            fd_scheme: FiniteDifferenceScheme::CentralDifference,
            options,
        })
    }
    
    /// Add an advection term to the PDE
    pub fn with_advection(
        mut self,
        advection_coeff: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.advection_coeff = Some(Box::new(advection_coeff));
        self
    }
    
    /// Add a reaction term to the PDE
    pub fn with_reaction(
        mut self,
        reaction_term: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.reaction_term = Some(Box::new(reaction_term));
        self
    }
    
    /// Set the finite difference scheme for spatial discretization
    pub fn with_fd_scheme(mut self, scheme: FiniteDifferenceScheme) -> Self {
        self.fd_scheme = scheme;
        self
    }
    
    /// Solve the PDE using the Backward Euler method
    pub fn solve(&self) -> PDEResult<ImplicitResult> {
        let start_time = Instant::now();
        
        // Generate spatial grid
        let x_grid = self.domain.grid(0)?;
        let nx = x_grid.len();
        let dx = self.domain.grid_spacing(0)?;
        
        // Time step
        let dt = self.options.dt.unwrap_or(0.01);
        
        // Calculate number of time steps
        let t_start = self.time_range[0];
        let t_end = self.time_range[1];
        let num_steps = ((t_end - t_start) / dt).ceil() as usize;
        
        // Initialize time array
        let mut t_values = Vec::with_capacity(num_steps + 1);
        t_values.push(t_start);
        
        // Initialize solution arrays
        let mut u_current = Array1::zeros(nx);
        
        // Apply initial condition
        for (i, &x) in x_grid.iter().enumerate() {
            u_current[i] = (self.initial_condition)(x);
        }
        
        // Apply Dirichlet boundary conditions to initial condition
        apply_dirichlet_conditions_to_initial_1d(&mut u_current, &self.boundary_conditions);
        
        // Store solutions
        let save_every = self.options.save_every.unwrap_or(1);
        let mut solutions = Vec::with_capacity((num_steps + 1) / save_every + 1);
        solutions.push(u_current.clone().into_shape_with_order((nx, 1)).unwrap());
        
        // Initialize coefficient matrix for Backward Euler method
        let mut a_matrix = Array2::zeros((nx, nx));
        
        // Track solver statistics
        let mut num_linear_solves = 0;
        
        // Time-stepping loop
        for step in 0..num_steps {
            let t_current = t_start + step as f64 * dt;
            let t_next = t_current + dt;
            
            // Set up coefficient matrix based on PDE terms and boundary conditions
            self.setup_coefficient_matrix(
                &mut a_matrix,
                &x_grid,
                dx,
                dt,
                t_next, // Use t_{n+1} for fully implicit scheme
            );
            
            // Right-hand side vector is just the current solution
            let rhs = u_current.clone();
            
            // Solve the linear system: A * u_{n+1} = u_n
            let u_next = self.solve_linear_system(&a_matrix, &rhs.view())?;
            num_linear_solves += 1;
            
            // Update current solution and time
            u_current = u_next;
            t_values.push(t_next);
            
            // Save solution if needed
            if (step + 1) % save_every == 0 || step == num_steps - 1 {
                solutions.push(u_current.clone().into_shape_with_order((nx, 1)).unwrap());
            }
            
            // Print progress if verbose
            if self.options.verbose && (step + 1) % 10 == 0 {
                println!("Step {}/{} completed, t = {:.4}", step + 1, num_steps, t_next);
            }
        }
        
        // Convert time values to Array1
        let t_array = Array1::from_vec(t_values);
        
        // Compute solution time
        let computation_time = start_time.elapsed().as_secs_f64();
        
        // Create result
        let info = Some(format!(
            "Time steps: {}, Linear system solves: {}",
            num_steps, num_linear_solves
        ));
        
        Ok(ImplicitResult {
            t: t_array,
            u: solutions,
            info,
            computation_time,
            num_steps,
            num_linear_solves,
        })
    }
    
    /// Set up coefficient matrix for the Backward Euler method
    fn setup_coefficient_matrix(
        &self,
        a_matrix: &mut Array2<f64>,
        x_grid: &Array1<f64>,
        dx: f64,
        dt: f64,
        t: f64,
    ) {
        let nx = x_grid.len();
        
        // Clear matrix
        a_matrix.fill(0.0);
        
        // Set up implicit matrix for interior points
        for i in 1..nx-1 {
            let x = x_grid[i];
            let u_val = 0.0; // Used for linearization around previous state if needed
            
            // Diffusion coefficient at the current point
            let d = (self.diffusion_coeff)(x, t, u_val);
            
            // Backward Euler coefficients for diffusion term
            let r = d * dt / (dx * dx);
            
            // Coefficient matrix for implicit scheme
            a_matrix[[i, i-1]] = -r;                // Coefficient for u_{i-1}^{n+1}
            a_matrix[[i, i]] = 1.0 + 2.0 * r;       // Coefficient for u_{i}^{n+1}
            a_matrix[[i, i+1]] = -r;                // Coefficient for u_{i+1}^{n+1}
            
            // Add advection term if present
            if let Some(advection) = &self.advection_coeff {
                let v = advection(x, t, u_val);
                
                // Coefficient for advection term (upwind for stability)
                if v > 0.0 {
                    // Use backward difference for positive velocity
                    let c = v * dt / dx;
                    a_matrix[[i, i]] += c;
                    a_matrix[[i, i-1]] -= c;
                } else {
                    // Use forward difference for negative velocity
                    let c = -v * dt / dx;
                    a_matrix[[i, i]] += c;
                    a_matrix[[i, i+1]] -= c;
                }
            }
            
            // Add reaction term if present
            if let Some(reaction) = &self.reaction_term {
                // For linear reaction terms of the form R(u) = ku
                // For nonlinear terms, would need to linearize around previous state
                let k = reaction(x, t, u_val);
                
                // Coefficient for reaction term
                let s = k * dt;
                a_matrix[[i, i]] += s;              // Additional term for u_{i}^{n+1}
            }
        }
        
        // Apply boundary conditions
        for bc in &self.boundary_conditions {
            match bc.location {
                BoundaryLocation::Lower => {
                    // Apply boundary condition at x[0]
                    match bc.bc_type {
                        BoundaryConditionType::Dirichlet => {
                            // u(a, t) = bc.value
                            // Replace the equation with u_0^{n+1} = bc.value
                            for j in 0..nx {
                                a_matrix[[0, j]] = 0.0;
                            }
                            a_matrix[[0, 0]] = 1.0;
                            
                            // Right-hand side will have the boundary value
                            // For Backward Euler, we directly modify the solution vector
                            // to enforce the boundary condition
                        },
                        BoundaryConditionType::Neumann => {
                            // du/dx(a, t) = bc.value
                            // Use second-order one-sided difference:
                            // (-3u_0 + 4u_1 - u_2)/(2dx) = bc.value
                            
                            for j in 0..nx {
                                a_matrix[[0, j]] = 0.0;
                            }
                            
                            a_matrix[[0, 0]] = -3.0;
                            a_matrix[[0, 1]] = 4.0;
                            a_matrix[[0, 2]] = -1.0;
                            
                            // Right-hand side will have the boundary value scaled by 2*dx
                            // For Backward Euler, we directly modify the solution vector
                        },
                        BoundaryConditionType::Robin => {
                            // a*u + b*du/dx = c
                            if let Some([a_val, b_val, _c_val]) = bc.coefficients {
                                // Use second-order one-sided difference for the derivative:
                                // (-3u_0 + 4u_1 - u_2)/(2dx)
                                
                                for j in 0..nx {
                                    a_matrix[[0, j]] = 0.0;
                                }
                                
                                a_matrix[[0, 0]] = a_val - 3.0 * b_val / (2.0 * dx);
                                a_matrix[[0, 1]] = 4.0 * b_val / (2.0 * dx);
                                a_matrix[[0, 2]] = -b_val / (2.0 * dx);
                                
                                // Right-hand side will have the boundary value c
                                // For Backward Euler, we directly modify the solution vector
                            }
                        },
                        BoundaryConditionType::Periodic => {
                            // For periodic BCs, special handling at both boundaries together is needed
                            // (see below)
                        },
                    }
                },
                BoundaryLocation::Upper => {
                    // Apply boundary condition at x[nx-1]
                    let i = nx - 1;
                    
                    match bc.bc_type {
                        BoundaryConditionType::Dirichlet => {
                            // u(b, t) = bc.value
                            // Replace the equation with u_{nx-1}^{n+1} = bc.value
                            for j in 0..nx {
                                a_matrix[[i, j]] = 0.0;
                            }
                            a_matrix[[i, i]] = 1.0;
                            
                            // Right-hand side will have the boundary value
                            // For Backward Euler, we directly modify the solution vector
                        },
                        BoundaryConditionType::Neumann => {
                            // du/dx(b, t) = bc.value
                            // Use second-order one-sided difference:
                            // (3u_{nx-1} - 4u_{nx-2} + u_{nx-3})/(2dx) = bc.value
                            
                            for j in 0..nx {
                                a_matrix[[i, j]] = 0.0;
                            }
                            
                            a_matrix[[i, i]] = 3.0;
                            a_matrix[[i, i-1]] = -4.0;
                            a_matrix[[i, i-2]] = 1.0;
                            
                            // Right-hand side will have the boundary value scaled by 2*dx
                            // For Backward Euler, we directly modify the solution vector
                        },
                        BoundaryConditionType::Robin => {
                            // a*u + b*du/dx = c
                            if let Some([a_val, b_val, _c_val]) = bc.coefficients {
                                // Use second-order one-sided difference for the derivative:
                                // (3u_{nx-1} - 4u_{nx-2} + u_{nx-3})/(2dx)
                                
                                for j in 0..nx {
                                    a_matrix[[i, j]] = 0.0;
                                }
                                
                                a_matrix[[i, i]] = a_val + 3.0 * b_val / (2.0 * dx);
                                a_matrix[[i, i-1]] = -4.0 * b_val / (2.0 * dx);
                                a_matrix[[i, i-2]] = b_val / (2.0 * dx);
                                
                                // Right-hand side will have the boundary value c
                                // For Backward Euler, we directly modify the solution vector
                            }
                        },
                        BoundaryConditionType::Periodic => {
                            // Handle periodic boundary conditions
                            // We need to make u_0 = u_{nx-1} and ensure the fluxes match at the boundaries
                            
                            // First, clear the boundary rows
                            for j in 0..nx {
                                a_matrix[[0, j]] = 0.0;
                                a_matrix[[i, j]] = 0.0;
                            }
                            
                            // For the lower boundary (i=0), set up equation connecting to upper boundary
                            // Use periodic stencil for diffusion
                            let x = x_grid[0];
                            let u_val = 0.0; // Used for linearization if needed
                            let d = (self.diffusion_coeff)(x, t, u_val);
                            let r = d * dt / (dx * dx);
                            
                            a_matrix[[0, i]] = -r;              // Coefficient for u_{nx-1}^{n+1}
                            a_matrix[[0, 0]] = 1.0 + 2.0 * r;   // Coefficient for u_{0}^{n+1}
                            a_matrix[[0, 1]] = -r;              // Coefficient for u_{1}^{n+1}
                            
                            // For the upper boundary (i=nx-1), set up equation connecting to lower boundary
                            let x = x_grid[i];
                            let d = (self.diffusion_coeff)(x, t, u_val);
                            let r = d * dt / (dx * dx);
                            
                            a_matrix[[i, i-1]] = -r;            // Coefficient for u_{nx-2}^{n+1}
                            a_matrix[[i, i]] = 1.0 + 2.0 * r;   // Coefficient for u_{nx-1}^{n+1}
                            a_matrix[[i, 0]] = -r;              // Coefficient for u_{0}^{n+1}
                            
                            // Add advection term if present
                            if let Some(advection) = &self.advection_coeff {
                                // Lower boundary
                                let v = advection(x_grid[0], t, u_val);
                                
                                if v > 0.0 {
                                    // Use backward difference for positive velocity
                                    let c = v * dt / dx;
                                    a_matrix[[0, 0]] += c;
                                    a_matrix[[0, i]] -= c;
                                } else {
                                    // Use forward difference for negative velocity
                                    let c = -v * dt / dx;
                                    a_matrix[[0, 0]] += c;
                                    a_matrix[[0, 1]] -= c;
                                }
                                
                                // Upper boundary
                                let v = advection(x_grid[i], t, u_val);
                                
                                if v > 0.0 {
                                    // Use backward difference for positive velocity
                                    let c = v * dt / dx;
                                    a_matrix[[i, i]] += c;
                                    a_matrix[[i, i-1]] -= c;
                                } else {
                                    // Use forward difference for negative velocity
                                    let c = -v * dt / dx;
                                    a_matrix[[i, i]] += c;
                                    a_matrix[[i, 0]] -= c;
                                }
                            }
                        },
                    }
                },
            }
        }
        
        // Special handling for Dirichlet boundary conditions
        // For Backward Euler, we need to modify the RHS vector with the boundary values
        // This is done in the solve method when applying the boundary conditions to the solution
    }
    
    /// Solve the linear system Ax = b
    fn solve_linear_system(&self, a: &Array2<f64>, b: &ArrayView1<f64>) -> PDEResult<Array1<f64>> {
        let n = b.len();
        
        // Simple tridiagonal solver for Backward Euler matrices
        // For a general solver, use a specialized linear algebra library
        
        // Check if the matrix is tridiagonal
        let is_tridiagonal = a.indexed_iter()
            .filter(|((i, j), &val)| val != 0.0 && (*i as isize - *j as isize).abs() > 1)
            .count() == 0;
        
        if is_tridiagonal {
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
            for i in (0..n-1).rev() {
                solution[i] = (temp_rhs[i] - upper[i] * solution[i + 1]) / temp_diag[i];
            }
            
            // Apply Dirichlet boundary conditions directly
            for bc in &self.boundary_conditions {
                if bc.bc_type == BoundaryConditionType::Dirichlet {
                    match bc.location {
                        BoundaryLocation::Lower => solution[0] = bc.value,
                        BoundaryLocation::Upper => solution[n - 1] = bc.value,
                    }
                }
            }
            
            Ok(solution)
        } else {
            // General case: Gaussian elimination with partial pivoting
            // For a real implementation, use a specialized linear algebra library
            
            // Create copies of A and b
            let mut a_copy = a.clone();
            let mut b_copy = b.to_owned();
            
            // Forward elimination
            for i in 0..n {
                // Find pivot
                let mut max_val = a_copy[[i, i]].abs();
                let mut max_row = i;
                
                for k in i+1..n {
                    if a_copy[[k, i]].abs() > max_val {
                        max_val = a_copy[[k, i]].abs();
                        max_row = k;
                    }
                }
                
                // Check if matrix is singular
                if max_val < 1e-10 {
                    return Err(PDEError::Other(
                        "Matrix is singular or nearly singular".to_string()
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
                for k in i+1..n {
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
                for j in i+1..n {
                    sum += a_copy[[i, j]] * x[j];
                }
                
                x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
            }
            
            // Apply Dirichlet boundary conditions directly
            for bc in &self.boundary_conditions {
                if bc.bc_type == BoundaryConditionType::Dirichlet {
                    match bc.location {
                        BoundaryLocation::Lower => x[0] = bc.value,
                        BoundaryLocation::Upper => x[n - 1] = bc.value,
                    }
                }
            }
            
            Ok(x)
        }
    }
}

/// Helper function to apply Dirichlet boundary conditions to initial condition
fn apply_dirichlet_conditions_to_initial_1d(
    u0: &mut Array1<f64>,
    boundary_conditions: &[BoundaryCondition<f64>],
) {
    let nx = u0.len();
    
    for bc in boundary_conditions {
        if bc.bc_type == BoundaryConditionType::Dirichlet {
            match bc.location {
                BoundaryLocation::Lower => {
                    // Lower boundary (x = x[0])
                    u0[0] = bc.value;
                },
                BoundaryLocation::Upper => {
                    // Upper boundary (x = x[nx-1])
                    u0[nx - 1] = bc.value;
                }
            }
        }
    }
}

/// Convert an ImplicitResult to a PDESolution
impl From<ImplicitResult> for PDESolution<f64> {
    fn from(result: ImplicitResult) -> Self {
        let mut grids = Vec::new();
        
        // Add time grid
        grids.push(result.t.clone());
        
        // Extract spatial grid from solution shape
        let nx = result.u[0].shape()[0];
        
        // Create spatial grid (we don't have the actual grid values, so use linspace)
        let x_grid = Array1::linspace(0.0, 1.0, nx);
        grids.push(x_grid);
        
        // Create solver info
        let info = PDESolverInfo {
            num_iterations: result.num_linear_solves,
            computation_time: result.computation_time,
            residual_norm: None,
            convergence_history: None,
            method: "Implicit Method".to_string(),
        };
        
        PDESolution {
            grids,
            values: result.u,
            error_estimate: None,
            info,
        }
    }
}