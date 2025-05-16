//! Method of Lines (MOL) approach for solving PDEs
//!
//! The Method of Lines transforms PDEs into systems of ODEs by discretizing
//! all but one dimension (typically space), leaving a system of ODEs in the
//! remaining dimension (typically time).
//!
//! This approach allows using well-established ODE solvers for time integration
//! after the spatial discretization is performed.
//!
//! ## Supported PDE Types and Dimensions
//!
//! - 1D Parabolic PDEs (heat equation, advection-diffusion)
//! - 2D Parabolic PDEs (2D heat equation, 2D advection-diffusion)
//! - 3D Parabolic PDEs (3D heat equation, 3D advection-diffusion)
//! - 1D Hyperbolic PDEs (wave equation)

pub mod mod2d;
pub use mod2d::{MOL2DResult, MOLParabolicSolver2D};

pub mod mod3d;
pub use mod3d::{MOL3DResult, MOLParabolicSolver3D};

pub mod hyperbolic;
pub use hyperbolic::{MOLHyperbolicResult, MOLWaveEquation1D};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::time::Instant;

use crate::ode::{solve_ivp, ODEMethod, ODEOptions, ODEResult};
use crate::pde::finite_difference::FiniteDifferenceScheme;
use crate::pde::{
    BoundaryCondition, Domain, PDEError, PDEResult, PDESolution, PDESolverInfo, PDEType,
};

/// Options for the Method of Lines PDE solver
#[derive(Debug, Clone)]
pub struct MOLOptions {
    /// ODE solver method to use for time integration
    pub ode_method: ODEMethod,

    /// Absolute tolerance for the ODE solver
    pub atol: f64,

    /// Relative tolerance for the ODE solver
    pub rtol: f64,

    /// Maximum number of ODE steps
    pub max_steps: Option<usize>,

    /// Print detailed progress information
    pub verbose: bool,
}

impl Default for MOLOptions {
    fn default() -> Self {
        MOLOptions {
            ode_method: ODEMethod::RK45,
            atol: 1e-6,
            rtol: 1e-3,
            max_steps: None,
            verbose: false,
        }
    }
}

/// Result of method of lines solution
pub struct MOLResult {
    /// Time points
    pub t: Array1<f64>,

    /// Solution values, indexed as [time, space, variable]
    pub u: Vec<Array2<f64>>,

    /// ODE solver information
    pub ode_info: Option<String>,

    /// Computation time
    pub computation_time: f64,
}

/// Method of Lines solver for 1D parabolic PDEs (e.g., heat equation)
///
/// Solves equations of the form: u/t = �(ab(x, t, u) u/x) + f(x, t, u)
pub struct MOLParabolicSolver1D {
    /// Spatial domain
    domain: Domain,

    /// Time range [t_start, t_end]
    time_range: [f64; 2],

    /// Diffusion coefficient function a(x, t, u) for �u/x�
    diffusion_coeff: Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>,

    /// Advection coefficient function b(x, t, u) for u/x
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
    options: MOLOptions,
}

impl MOLParabolicSolver1D {
    /// Create a new Method of Lines solver for 1D parabolic PDEs
    pub fn new(
        domain: Domain,
        time_range: [f64; 2],
        diffusion_coeff: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
        initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<MOLOptions>,
    ) -> PDEResult<Self> {
        // Validate the domain
        if domain.dimensions() != 1 {
            return Err(PDEError::DomainError(
                "Domain must be 1-dimensional for 1D parabolic solver".to_string(),
            ));
        }

        // Validate time range
        if time_range[0] >= time_range[1] {
            return Err(PDEError::DomainError(
                "Invalid time range: start must be less than end".to_string(),
            ));
        }

        // Validate boundary conditions
        if boundary_conditions.len() != 2 {
            return Err(PDEError::BoundaryConditions(
                "1D parabolic PDE requires exactly 2 boundary conditions".to_string(),
            ));
        }

        // Ensure we have both lower and upper boundary conditions
        let has_lower = boundary_conditions
            .iter()
            .any(|bc| bc.location == crate::pde::BoundaryLocation::Lower);
        let has_upper = boundary_conditions
            .iter()
            .any(|bc| bc.location == crate::pde::BoundaryLocation::Upper);

        if !has_lower || !has_upper {
            return Err(PDEError::BoundaryConditions(
                "1D parabolic PDE requires both lower and upper boundary conditions".to_string(),
            ));
        }

        Ok(MOLParabolicSolver1D {
            domain,
            time_range,
            diffusion_coeff: Box::new(diffusion_coeff),
            advection_coeff: None,
            reaction_term: None,
            initial_condition: Box::new(initial_condition),
            boundary_conditions,
            fd_scheme: FiniteDifferenceScheme::CentralDifference,
            options: options.unwrap_or_default(),
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

    /// Solve the PDE
    pub fn solve(&self) -> PDEResult<MOLResult> {
        let start_time = Instant::now();

        // Generate spatial grid
        let x_grid = self.domain.grid(0)?;
        let nx = x_grid.len();
        let dx = self.domain.grid_spacing(0)?;

        // Create initial condition vector
        let mut u0 = Array1::zeros(nx);
        for (i, &x) in x_grid.iter().enumerate() {
            u0[i] = (self.initial_condition)(x);
        }

        // Construct the ODE function that represents the PDE after spatial discretization
        let ode_func = move |t: f64, u: ArrayView1<f64>| -> Array1<f64> {
            let mut dudt = Array1::zeros(nx);

            // Apply finite difference approximations for interior points
            for i in 1..nx - 1 {
                let x = x_grid[i];
                let u_i = u[i];

                // Second derivative term (diffusion)
                let d2u_dx2 = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx);
                let diffusion_term = (self.diffusion_coeff)(x, t, u_i) * d2u_dx2;

                // First derivative term (advection)
                let advection_term = if let Some(advection) = &self.advection_coeff {
                    let du_dx = (u[i + 1] - u[i - 1]) / (2.0 * dx);
                    advection(x, t, u_i) * du_dx
                } else {
                    0.0
                };

                // Reaction term
                let reaction_term = if let Some(reaction) = &self.reaction_term {
                    reaction(x, t, u_i)
                } else {
                    0.0
                };

                dudt[i] = diffusion_term + advection_term + reaction_term;
            }

            // Apply boundary conditions
            for bc in &self.boundary_conditions {
                match bc.location {
                    crate::pde::BoundaryLocation::Lower => {
                        // Apply boundary condition at x[0]
                        match bc.bc_type {
                            crate::pde::BoundaryConditionType::Dirichlet => {
                                // Fixed value: u(x_0, t) = bc.value
                                // For Dirichlet, we set dudt[0] = 0 to maintain the fixed value
                                dudt[0] = 0.0;
                            }
                            crate::pde::BoundaryConditionType::Neumann => {
                                // Fixed gradient: u/x|_{x_0} = bc.value
                                // Use one-sided difference to approximate second derivative
                                let du_dx = bc.value; // Given Neumann value
                                let u_ghost = u[0] - dx * du_dx; // Ghost point value

                                // Now use central difference for diffusion term
                                let d2u_dx2 = (u[1] - 2.0 * u[0] + u_ghost) / (dx * dx);
                                let diffusion_term =
                                    (self.diffusion_coeff)(x_grid[0], t, u[0]) * d2u_dx2;

                                // Other terms
                                let advection_term = if let Some(advection) = &self.advection_coeff
                                {
                                    let du_dx_forward = (u[1] - u[0]) / dx;
                                    advection(x_grid[0], t, u[0]) * du_dx_forward
                                } else {
                                    0.0
                                };

                                let reaction_term = if let Some(reaction) = &self.reaction_term {
                                    reaction(x_grid[0], t, u[0])
                                } else {
                                    0.0
                                };

                                dudt[0] = diffusion_term + advection_term + reaction_term;
                            }
                            crate::pde::BoundaryConditionType::Robin => {
                                // Robin boundary condition: a*u + b*du/dx = c
                                if let Some([a, b, c]) = bc.coefficients {
                                    // Solve for ghost point value using Robin condition
                                    let du_dx = (c - a * u[0]) / b;
                                    let u_ghost = u[0] - dx * du_dx;

                                    // Use central difference with ghost point
                                    let d2u_dx2 = (u[1] - 2.0 * u[0] + u_ghost) / (dx * dx);
                                    let diffusion_term =
                                        (self.diffusion_coeff)(x_grid[0], t, u[0]) * d2u_dx2;

                                    // Other terms
                                    let advection_term =
                                        if let Some(advection) = &self.advection_coeff {
                                            let du_dx_forward = (u[1] - u[0]) / dx;
                                            advection(x_grid[0], t, u[0]) * du_dx_forward
                                        } else {
                                            0.0
                                        };

                                    let reaction_term = if let Some(reaction) = &self.reaction_term
                                    {
                                        reaction(x_grid[0], t, u[0])
                                    } else {
                                        0.0
                                    };

                                    dudt[0] = diffusion_term + advection_term + reaction_term;
                                }
                            }
                            crate::pde::BoundaryConditionType::Periodic => {
                                // Periodic boundary: u(x_0, t) = u(x_n, t)
                                // Use values from the other end of the domain
                                let d2u_dx2 = (u[1] - 2.0 * u[0] + u[nx - 1]) / (dx * dx);
                                let diffusion_term =
                                    (self.diffusion_coeff)(x_grid[0], t, u[0]) * d2u_dx2;

                                // Other terms
                                let advection_term = if let Some(advection) = &self.advection_coeff
                                {
                                    let du_dx = (u[1] - u[nx - 1]) / (2.0 * dx);
                                    advection(x_grid[0], t, u[0]) * du_dx
                                } else {
                                    0.0
                                };

                                let reaction_term = if let Some(reaction) = &self.reaction_term {
                                    reaction(x_grid[0], t, u[0])
                                } else {
                                    0.0
                                };

                                dudt[0] = diffusion_term + advection_term + reaction_term;
                            }
                        }
                    }
                    crate::pde::BoundaryLocation::Upper => {
                        // Apply boundary condition at x[nx-1]
                        match bc.bc_type {
                            crate::pde::BoundaryConditionType::Dirichlet => {
                                // Fixed value: u(x_n, t) = bc.value
                                // For Dirichlet, we set dudt[nx-1] = 0 to maintain the fixed value
                                dudt[nx - 1] = 0.0;
                            }
                            crate::pde::BoundaryConditionType::Neumann => {
                                // Fixed gradient: u/x|_{x_n} = bc.value
                                // Use one-sided difference to approximate second derivative
                                let du_dx = bc.value; // Given Neumann value
                                let u_ghost = u[nx - 1] + dx * du_dx; // Ghost point value

                                // Now use central difference for diffusion term
                                let d2u_dx2 = (u_ghost - 2.0 * u[nx - 1] + u[nx - 2]) / (dx * dx);
                                let diffusion_term =
                                    (self.diffusion_coeff)(x_grid[nx - 1], t, u[nx - 1]) * d2u_dx2;

                                // Other terms
                                let advection_term = if let Some(advection) = &self.advection_coeff
                                {
                                    let du_dx_backward = (u[nx - 1] - u[nx - 2]) / dx;
                                    advection(x_grid[nx - 1], t, u[nx - 1]) * du_dx_backward
                                } else {
                                    0.0
                                };

                                let reaction_term = if let Some(reaction) = &self.reaction_term {
                                    reaction(x_grid[nx - 1], t, u[nx - 1])
                                } else {
                                    0.0
                                };

                                dudt[nx - 1] = diffusion_term + advection_term + reaction_term;
                            }
                            crate::pde::BoundaryConditionType::Robin => {
                                // Robin boundary condition: a*u + b*du/dx = c
                                if let Some([a, b, c]) = bc.coefficients {
                                    // Solve for ghost point value using Robin condition
                                    let du_dx = (c - a * u[nx - 1]) / b;
                                    let u_ghost = u[nx - 1] + dx * du_dx;

                                    // Use central difference with ghost point
                                    let d2u_dx2 =
                                        (u_ghost - 2.0 * u[nx - 1] + u[nx - 2]) / (dx * dx);
                                    let diffusion_term =
                                        (self.diffusion_coeff)(x_grid[nx - 1], t, u[nx - 1])
                                            * d2u_dx2;

                                    // Other terms
                                    let advection_term =
                                        if let Some(advection) = &self.advection_coeff {
                                            let du_dx_backward = (u[nx - 1] - u[nx - 2]) / dx;
                                            advection(x_grid[nx - 1], t, u[nx - 1]) * du_dx_backward
                                        } else {
                                            0.0
                                        };

                                    let reaction_term = if let Some(reaction) = &self.reaction_term
                                    {
                                        reaction(x_grid[nx - 1], t, u[nx - 1])
                                    } else {
                                        0.0
                                    };

                                    dudt[nx - 1] = diffusion_term + advection_term + reaction_term;
                                }
                            }
                            crate::pde::BoundaryConditionType::Periodic => {
                                // Periodic boundary: u(x_n, t) = u(x_0, t)
                                // Use values from the other end of the domain
                                let d2u_dx2 = (u[0] - 2.0 * u[nx - 1] + u[nx - 2]) / (dx * dx);
                                let diffusion_term =
                                    (self.diffusion_coeff)(x_grid[nx - 1], t, u[nx - 1]) * d2u_dx2;

                                // Other terms
                                let advection_term = if let Some(advection) = &self.advection_coeff
                                {
                                    let du_dx = (u[0] - u[nx - 2]) / (2.0 * dx);
                                    advection(x_grid[nx - 1], t, u[nx - 1]) * du_dx
                                } else {
                                    0.0
                                };

                                let reaction_term = if let Some(reaction) = &self.reaction_term {
                                    reaction(x_grid[nx - 1], t, u[nx - 1])
                                } else {
                                    0.0
                                };

                                dudt[nx - 1] = diffusion_term + advection_term + reaction_term;
                            }
                        }
                    }
                }
            }

            dudt
        };

        // Set up ODE solver options
        let ode_options = ODEOptions {
            method: self.options.ode_method,
            rtol: self.options.rtol,
            atol: Some(self.options.atol),
            max_num_steps: self.options.max_steps,
            first_step: None,
            max_step: None,
            min_step: None,
            dense_output: true,
        };

        // Apply Dirichlet boundary conditions to initial condition
        for bc in &self.boundary_conditions {
            if bc.bc_type == crate::pde::BoundaryConditionType::Dirichlet {
                match bc.location {
                    crate::pde::BoundaryLocation::Lower => u0[0] = bc.value,
                    crate::pde::BoundaryLocation::Upper => u0[nx - 1] = bc.value,
                }
            }
        }

        // Solve the ODE system
        let ode_result = solve_ivp(ode_func, self.time_range, u0, Some(ode_options))?;

        // Extract results
        let computation_time = start_time.elapsed().as_secs_f64();

        // Reshape the ODE result to match the spatial grid
        let t = ode_result.t;
        let mut u = Vec::new();

        // Create a single 2D array with dimensions [time, space]
        let u_2d =
            Array2::from_shape_vec((t.len(), nx), ode_result.y.into_iter().flatten().collect())
                .map_err(|e| PDEError::Other(format!("Error reshaping result: {}", e)))?;

        u.push(u_2d);

        let ode_info = Some(format!(
            "ODE steps: {}, function evaluations: {}, successful steps: {}",
            ode_result.stats.num_steps,
            ode_result.stats.num_function_evaluations,
            ode_result.stats.num_accepted_steps,
        ));

        Ok(MOLResult {
            t,
            u,
            ode_info,
            computation_time,
        })
    }
}

/// Convert a MOLResult to a PDESolution
impl From<MOLResult> for PDESolution<f64> {
    fn from(result: MOLResult) -> Self {
        let mut grids = Vec::new();

        // Add time grid
        grids.push(result.t.clone());

        // Extract spatial grid from solution
        let nx = result.u[0].shape()[1];
        // Note: For a proper implementation, the spatial grid should be provided
        let spatial_grid = Array1::linspace(0.0, 1.0, nx);
        grids.push(spatial_grid);

        // Create solver info
        let info = PDESolverInfo {
            num_iterations: 0, // This information is not available directly
            computation_time: result.computation_time,
            residual_norm: None,
            convergence_history: None,
            method: "Method of Lines".to_string(),
        };

        PDESolution {
            grids,
            values: result.u,
            error_estimate: None,
            info,
        }
    }
}
