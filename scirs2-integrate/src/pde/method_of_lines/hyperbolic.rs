//! Method of Lines for hyperbolic PDEs
//!
//! This module implements the Method of Lines (MOL) approach for solving
//! hyperbolic PDEs, such as the wave equation.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use std::sync::Arc;
use std::time::Instant;

use crate::ode::{solve_ivp, ODEOptions};
use crate::pde::finite_difference::FiniteDifferenceScheme;
use crate::pde::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, PDEError, PDEResult,
    PDESolution, PDESolverInfo,
};

/// Type alias for 1D coefficient function taking (x, t, u) and returning a value
type CoeffFn1D = Arc<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>;

/// Result of hyperbolic PDE solution
pub struct MOLHyperbolicResult {
    /// Time points
    pub t: Array1<f64>,

    /// Solution values, indexed as [time, space]
    pub u: Array2<f64>,

    /// First-order time derivative values (∂u/∂t)
    pub u_t: Array2<f64>,

    /// ODE solver information
    pub ode_info: Option<String>,

    /// Computation time
    pub computation_time: f64,
}

/// Method of Lines solver for 1D Wave Equation
///
/// Solves the equation: ∂²u/∂t² = c² ∂²u/∂x² + f(x,t,u)
#[derive(Clone)]
pub struct MOLWaveEquation1D {
    /// Spatial domain
    domain: Domain,

    /// Time range [t_start, t_end]
    time_range: [f64; 2],

    /// Wave speed (squared) coefficient c²(x, t, u)
    wave_speed_squared: CoeffFn1D,

    /// Source term function f(x, t, u)
    source_term: Option<CoeffFn1D>,

    /// Initial condition function u(x, 0)
    initial_condition: Arc<dyn Fn(f64) -> f64 + Send + Sync>,

    /// Initial velocity function ∂u/∂t(x, 0)
    initial_velocity: Arc<dyn Fn(f64) -> f64 + Send + Sync>,

    /// Boundary conditions
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Finite difference scheme for spatial discretization
    fd_scheme: FiniteDifferenceScheme,

    /// Solver options
    options: super::MOLOptions,
}

impl MOLWaveEquation1D {
    /// Create a new Method of Lines solver for the 1D wave equation
    pub fn new(
        domain: Domain,
        time_range: [f64; 2],
        wave_speed_squared: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
        initial_condition: impl Fn(f64) -> f64 + Send + Sync + 'static,
        initial_velocity: impl Fn(f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<super::MOLOptions>,
    ) -> PDEResult<Self> {
        // Validate the domain
        if domain.dimensions() != 1 {
            return Err(PDEError::DomainError(
                "Domain must be 1-dimensional for 1D wave equation solver".to_string(),
            ));
        }

        // Validate time _range
        if time_range[0] >= time_range[1] {
            return Err(PDEError::DomainError(
                "Invalid time _range: start must be less than end".to_string(),
            ));
        }

        // Validate boundary _conditions
        if boundary_conditions.len() != 2 {
            return Err(PDEError::BoundaryConditions(
                "1D wave equation requires exactly 2 boundary _conditions".to_string(),
            ));
        }

        // Ensure we have both lower and upper boundary _conditions
        let has_lower = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Lower);
        let has_upper = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Upper);

        if !has_lower || !has_upper {
            return Err(PDEError::BoundaryConditions(
                "1D wave equation requires both lower and upper boundary _conditions".to_string(),
            ));
        }

        Ok(MOLWaveEquation1D {
            domain,
            time_range,
            wave_speed_squared: Arc::new(wave_speed_squared),
            source_term: None,
            initial_condition: Arc::new(initial_condition),
            initial_velocity: Arc::new(initial_velocity),
            boundary_conditions,
            fd_scheme: FiniteDifferenceScheme::CentralDifference,
            options: options.unwrap_or_default(),
        })
    }

    /// Add a source term to the wave equation
    pub fn with_source(
        mut self,
        source_term: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.source_term = Some(Arc::new(source_term));
        self
    }

    /// Set the finite difference scheme for spatial discretization
    pub fn with_fd_scheme(mut self, scheme: FiniteDifferenceScheme) -> Self {
        self.fd_scheme = scheme;
        self
    }

    /// Solve the wave equation
    pub fn solve(&self) -> PDEResult<MOLHyperbolicResult> {
        let start_time = Instant::now();

        // Generate spatial grid
        let x_grid = self.domain.grid(0)?;
        let nx = x_grid.len();
        let dx = self.domain.grid_spacing(0)?;

        // Create initial condition and velocity vectors
        let mut u0 = Array1::zeros(nx);
        let mut v0 = Array1::zeros(nx);

        for (i, &x) in x_grid.iter().enumerate() {
            u0[i] = (self.initial_condition)(x);
            v0[i] = (self.initial_velocity)(x);
        }

        // The wave equation is a second-order in time PDE, so we convert it
        // to a first-order system by introducing v = ∂u/∂t
        // This gives us:
        // ∂u/∂t = v
        // ∂v/∂t = c² ∂²u/∂x² + f

        // Combine u and v into a single state vector for the ODE solver
        let mut y0 = Array1::zeros(2 * nx);
        for i in 0..nx {
            y0[i] = u0[i]; // First nx elements are u
            y0[i + nx] = v0[i]; // Next nx elements are v = ∂u/∂t
        }

        // Extract data before moving self
        let x_grid = x_grid.clone();
        let time_range = self.time_range;
        let boundary_conditions = self.boundary_conditions.clone();
        let boundary_conditions_copy = boundary_conditions.clone();
        let options = self.options.clone();

        // Move self into closure
        let solver = self;

        // Construct the ODE function for the first-order system
        let ode_func = move |t: f64, y: ArrayView1<f64>| -> Array1<f64> {
            // Extract u and v from the combined state vector
            let u = y.slice(s![0..nx]);
            let v = y.slice(s![nx..2 * nx]);

            let mut dydt = Array1::zeros(2 * nx);

            // First part: ∂u/∂t = v
            for i in 0..nx {
                dydt[i] = v[i];
            }

            // Second part: ∂v/∂t = c² ∂²u/∂x² + f

            // Apply finite difference approximations for interior points
            for i in 1..nx - 1 {
                let x = x_grid[i];
                let u_i = u[i];

                // Second derivative term
                let d2u_dx2 = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx);
                let c_squared = (solver.wave_speed_squared)(x, t, u_i);
                let wave_term = c_squared * d2u_dx2;

                // Source term
                let source_term = if let Some(source) = &solver.source_term {
                    source(x, t, u_i)
                } else {
                    0.0
                };

                dydt[i + nx] = wave_term + source_term;
            }

            // Apply boundary conditions
            for bc in &boundary_conditions_copy {
                match bc.location {
                    BoundaryLocation::Lower => {
                        // Apply boundary condition at x[0]
                        match bc.bc_type {
                            BoundaryConditionType::Dirichlet => {
                                // Fixed value: u(x_0, t) = bc.value
                                // For Dirichlet, we set v[0] = 0 to maintain the fixed value
                                // and to ensure u[0] doesn't change
                                dydt[0] = 0.0; // ∂u/∂t = 0
                                dydt[nx] = 0.0; // ∂v/∂t = 0
                            }
                            BoundaryConditionType::Neumann => {
                                // Fixed gradient: ∂u/∂x|_{x_0} = bc.value

                                // Calculate the ghost point value based on the Neumann condition
                                let du_dx = bc.value;
                                let u_ghost = u[0] - dx * du_dx; // Ghost point value

                                // Use central difference for the second derivative
                                let d2u_dx2 = (u[1] - 2.0 * u[0] + u_ghost) / (dx * dx);
                                let c_squared = (solver.wave_speed_squared)(x_grid[0], t, u[0]);
                                let wave_term = c_squared * d2u_dx2;

                                // Source term
                                let source_term = if let Some(source) = &solver.source_term {
                                    source(x_grid[0], t, u[0])
                                } else {
                                    0.0
                                };

                                dydt[0] = v[0]; // ∂u/∂t = v
                                dydt[nx] = wave_term + source_term; // ∂v/∂t
                            }
                            BoundaryConditionType::Robin => {
                                // Robin boundary condition: a*u + b*du/dx = c
                                if let Some([a, b, c]) = bc.coefficients {
                                    // Solve for ghost point value using Robin condition
                                    let du_dx = (c - a * u[0]) / b;
                                    let u_ghost = u[0] - dx * du_dx;

                                    // Use central difference for the second derivative
                                    let d2u_dx2 = (u[1] - 2.0 * u[0] + u_ghost) / (dx * dx);
                                    let c_squared = (solver.wave_speed_squared)(x_grid[0], t, u[0]);
                                    let wave_term = c_squared * d2u_dx2;

                                    // Source term
                                    let source_term = if let Some(source) = &solver.source_term {
                                        source(x_grid[0], t, u[0])
                                    } else {
                                        0.0
                                    };

                                    dydt[0] = v[0]; // ∂u/∂t = v
                                    dydt[nx] = wave_term + source_term; // ∂v/∂t
                                }
                            }
                            BoundaryConditionType::Periodic => {
                                // Periodic boundary: u(x_0, t) = u(x_n, t)

                                // Use values from the other end of the domain
                                let d2u_dx2 = (u[1] - 2.0 * u[0] + u[nx - 1]) / (dx * dx);
                                let c_squared = (solver.wave_speed_squared)(x_grid[0], t, u[0]);
                                let wave_term = c_squared * d2u_dx2;

                                // Source term
                                let source_term = if let Some(source) = &solver.source_term {
                                    source(x_grid[0], t, u[0])
                                } else {
                                    0.0
                                };

                                dydt[0] = v[0]; // ∂u/∂t = v
                                dydt[nx] = wave_term + source_term; // ∂v/∂t
                            }
                        }
                    }
                    BoundaryLocation::Upper => {
                        // Apply boundary condition at x[nx-1]
                        match bc.bc_type {
                            BoundaryConditionType::Dirichlet => {
                                // Fixed value: u(x_n, t) = bc.value
                                dydt[nx - 1] = 0.0; // ∂u/∂t = 0
                                dydt[nx - 1 + nx] = 0.0; // ∂v/∂t = 0
                            }
                            BoundaryConditionType::Neumann => {
                                // Fixed gradient: ∂u/∂x|_{x_n} = bc.value

                                // Calculate the ghost point value based on the Neumann condition
                                let du_dx = bc.value;
                                let u_ghost = u[nx - 1] + dx * du_dx; // Ghost point value

                                // Use central difference for the second derivative
                                let d2u_dx2 = (u_ghost - 2.0 * u[nx - 1] + u[nx - 2]) / (dx * dx);
                                let c_squared =
                                    (solver.wave_speed_squared)(x_grid[nx - 1], t, u[nx - 1]);
                                let wave_term = c_squared * d2u_dx2;

                                // Source term
                                let source_term = if let Some(source) = &solver.source_term {
                                    source(x_grid[nx - 1], t, u[nx - 1])
                                } else {
                                    0.0
                                };

                                dydt[nx - 1] = v[nx - 1]; // ∂u/∂t = v
                                dydt[nx - 1 + nx] = wave_term + source_term; // ∂v/∂t
                            }
                            BoundaryConditionType::Robin => {
                                // Robin boundary condition: a*u + b*du/dx = c
                                if let Some([a, b, c]) = bc.coefficients {
                                    // Solve for ghost point value using Robin condition
                                    let du_dx = (c - a * u[nx - 1]) / b;
                                    let u_ghost = u[nx - 1] + dx * du_dx;

                                    // Use central difference for the second derivative
                                    let d2u_dx2 =
                                        (u_ghost - 2.0 * u[nx - 1] + u[nx - 2]) / (dx * dx);
                                    let c_squared =
                                        (solver.wave_speed_squared)(x_grid[nx - 1], t, u[nx - 1]);
                                    let wave_term = c_squared * d2u_dx2;

                                    // Source term
                                    let source_term = if let Some(source) = &solver.source_term {
                                        source(x_grid[nx - 1], t, u[nx - 1])
                                    } else {
                                        0.0
                                    };

                                    dydt[nx - 1] = v[nx - 1]; // ∂u/∂t = v
                                    dydt[nx - 1 + nx] = wave_term + source_term;
                                    // ∂v/∂t
                                }
                            }
                            BoundaryConditionType::Periodic => {
                                // Periodic boundary: u(x_n, t) = u(x_0, t)

                                // Use values from the other end of the domain
                                let d2u_dx2 = (u[0] - 2.0 * u[nx - 1] + u[nx - 2]) / (dx * dx);
                                let c_squared =
                                    (solver.wave_speed_squared)(x_grid[nx - 1], t, u[nx - 1]);
                                let wave_term = c_squared * d2u_dx2;

                                // Source term
                                let source_term = if let Some(source) = &solver.source_term {
                                    source(x_grid[nx - 1], t, u[nx - 1])
                                } else {
                                    0.0
                                };

                                dydt[nx - 1] = v[nx - 1]; // ∂u/∂t = v
                                dydt[nx - 1 + nx] = wave_term + source_term; // ∂v/∂t
                            }
                        }
                    }
                }
            }

            dydt
        };

        // Set up ODE solver options
        let ode_options = ODEOptions {
            method: options.ode_method,
            rtol: options.rtol,
            atol: options.atol,
            h0: None,
            max_steps: options.max_steps.unwrap_or(500),
            max_step: None,
            min_step: None,
            dense_output: true,
            max_order: None,
            jac: None,
            use_banded_jacobian: false,
            ml: None,
            mu: None,
            mass_matrix: None,
            jacobian_strategy: None,
        };

        // Apply Dirichlet boundary conditions to initial condition
        for bc in &boundary_conditions {
            if bc.bc_type == BoundaryConditionType::Dirichlet {
                match bc.location {
                    BoundaryLocation::Lower => {
                        y0[0] = bc.value; // u(x_0, 0) = bc.value
                        y0[nx] = 0.0; // v(x_0, 0) = 0
                    }
                    BoundaryLocation::Upper => {
                        y0[nx - 1] = bc.value; // u(x_n, 0) = bc.value
                        y0[nx - 1 + nx] = 0.0; // v(x_n, 0) = 0
                    }
                }
            }
        }

        // Solve the ODE system
        let ode_result = solve_ivp(ode_func, time_range, y0, Some(ode_options))?;

        // Extract results
        let computation_time = start_time.elapsed().as_secs_f64();

        // Reshape the ODE result to separate u and v
        let t = ode_result.t;
        let nt = t.len();

        let mut u = Array2::zeros((nt, nx));
        let mut u_t = Array2::zeros((nt, nx));

        for (i, y) in ode_result.y.iter().enumerate() {
            // Split the state vector into u and v
            for j in 0..nx {
                u[[i, j]] = y[j]; // u values
                u_t[[i, j]] = y[j + nx]; // v = ∂u/∂t values
            }
        }

        let ode_info = Some(format!(
            "ODE steps: {}, function evaluations: {}, successful steps: {}",
            ode_result.n_steps, ode_result.n_eval, ode_result.n_accepted,
        ));

        Ok(MOLHyperbolicResult {
            t: t.into(),
            u,
            u_t,
            ode_info,
            computation_time,
        })
    }
}

/// Convert a MOLHyperbolicResult to a PDESolution
impl From<MOLHyperbolicResult> for PDESolution<f64> {
    fn from(result: MOLHyperbolicResult) -> Self {
        let mut grids = Vec::new();

        // Add time grid
        grids.push(result.t.clone());

        // Extract spatial grid from solution
        let nx = result.u.shape()[1];

        // Note: For a proper implementation, the spatial grid should be provided
        let spatial_grid = Array1::linspace(0.0, 1.0, nx);
        grids.push(spatial_grid);

        // Create solver info
        let info = PDESolverInfo {
            num_iterations: 0, // This information is not available directly
            computation_time: result.computation_time,
            residual_norm: None,
            convergence_history: None,
            method: "Method of Lines (Hyperbolic)".to_string(),
        };

        // For hyperbolic PDEs, we return both u and u_t as values
        let values = vec![result.u, result.u_t];

        PDESolution {
            grids,
            values,
            error_estimate: None,
            info,
        }
    }
}
