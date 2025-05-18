//! Method of Lines for 2D parabolic PDEs
//!
//! This module implements the Method of Lines (MOL) approach for solving
//! 2D parabolic PDEs, such as the 2D heat equation and 2D advection-diffusion.

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2};
use std::sync::Arc;
use std::time::Instant;

use crate::ode::{solve_ivp, ODEOptions};
use crate::pde::finite_difference::FiniteDifferenceScheme;
use crate::pde::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, PDEError, PDEResult,
    PDESolution, PDESolverInfo,
};

/// Result of 2D method of lines solution
pub struct MOL2DResult {
    /// Time points
    pub t: Array1<f64>,

    /// Solution values, indexed as [time, y, x]
    pub u: Array3<f64>,

    /// ODE solver information
    pub ode_info: Option<String>,

    /// Computation time
    pub computation_time: f64,
}

/// Method of Lines solver for 2D parabolic PDEs
///
/// Solves equations of the form:
/// ∂u/∂t = ∂/∂x(D_x(x,y,t,u) ∂u/∂x) + ∂/∂y(D_y(x,y,t,u) ∂u/∂y) + v_x(x,y,t,u) ∂u/∂x + v_y(x,y,t,u) ∂u/∂y + f(x,y,t,u)
#[derive(Clone)]
pub struct MOLParabolicSolver2D {
    /// Spatial domain
    domain: Domain,

    /// Time range [t_start, t_end]
    time_range: [f64; 2],

    /// Diffusion coefficient function D_x(x, y, t, u) for ∂²u/∂x²
    diffusion_x: Arc<dyn Fn(f64, f64, f64, f64) -> f64 + Send + Sync>,

    /// Diffusion coefficient function D_y(x, y, t, u) for ∂²u/∂y²
    diffusion_y: Arc<dyn Fn(f64, f64, f64, f64) -> f64 + Send + Sync>,

    /// Advection coefficient function v_x(x, y, t, u) for ∂u/∂x
    advection_x: Option<Arc<dyn Fn(f64, f64, f64, f64) -> f64 + Send + Sync>>,

    /// Advection coefficient function v_y(x, y, t, u) for ∂u/∂y
    advection_y: Option<Arc<dyn Fn(f64, f64, f64, f64) -> f64 + Send + Sync>>,

    /// Reaction term function f(x, y, t, u)
    reaction_term: Option<Arc<dyn Fn(f64, f64, f64, f64) -> f64 + Send + Sync>>,

    /// Initial condition function u(x, y, 0)
    initial_condition: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>,

    /// Boundary conditions
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Finite difference scheme for spatial discretization
    fd_scheme: FiniteDifferenceScheme,

    /// Solver options
    options: super::MOLOptions,
}

impl MOLParabolicSolver2D {
    /// Create a new Method of Lines solver for 2D parabolic PDEs
    pub fn new(
        domain: Domain,
        time_range: [f64; 2],
        diffusion_x: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        diffusion_y: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        initial_condition: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<super::MOLOptions>,
    ) -> PDEResult<Self> {
        // Validate the domain
        if domain.dimensions() != 2 {
            return Err(PDEError::DomainError(
                "Domain must be 2-dimensional for 2D parabolic solver".to_string(),
            ));
        }

        // Validate time range
        if time_range[0] >= time_range[1] {
            return Err(PDEError::DomainError(
                "Invalid time range: start must be less than end".to_string(),
            ));
        }

        // Validate boundary conditions
        if boundary_conditions.len() != 4 {
            return Err(PDEError::BoundaryConditions(
                "2D parabolic PDE requires exactly 4 boundary conditions (one for each edge)"
                    .to_string(),
            ));
        }

        // Ensure we have boundary conditions for all dimensions/edges
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
                "2D parabolic PDE requires boundary conditions for all edges of the domain"
                    .to_string(),
            ));
        }

        Ok(MOLParabolicSolver2D {
            domain,
            time_range,
            diffusion_x: Arc::new(diffusion_x),
            diffusion_y: Arc::new(diffusion_y),
            advection_x: None,
            advection_y: None,
            reaction_term: None,
            initial_condition: Arc::new(initial_condition),
            boundary_conditions,
            fd_scheme: FiniteDifferenceScheme::CentralDifference,
            options: options.unwrap_or_default(),
        })
    }

    /// Add advection terms to the PDE
    pub fn with_advection(
        mut self,
        advection_x: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        advection_y: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.advection_x = Some(Arc::new(advection_x));
        self.advection_y = Some(Arc::new(advection_y));
        self
    }

    /// Add a reaction term to the PDE
    pub fn with_reaction(
        mut self,
        reaction_term: impl Fn(f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.reaction_term = Some(Arc::new(reaction_term));
        self
    }

    /// Set the finite difference scheme for spatial discretization
    pub fn with_fd_scheme(mut self, scheme: FiniteDifferenceScheme) -> Self {
        self.fd_scheme = scheme;
        self
    }

    /// Solve the 2D parabolic PDE
    pub fn solve(self) -> PDEResult<MOL2DResult> {
        let start_time = Instant::now();

        // Generate spatial grids
        let x_grid = self.domain.grid(0)?;
        let y_grid = self.domain.grid(1)?;
        let nx = x_grid.len();
        let ny = y_grid.len();
        let dx = self.domain.grid_spacing(0)?;
        let dy = self.domain.grid_spacing(1)?;

        // Create initial condition 2D grid and flatten it for the ODE solver
        let mut u0 = Array2::zeros((ny, nx));
        for j in 0..ny {
            for i in 0..nx {
                u0[[j, i]] = (self.initial_condition)(x_grid[i], y_grid[j]);
            }
        }

        // Flatten the 2D grid into a 1D array for the ODE solver
        let _u0_flat = u0.clone().into_shape_with_order(nx * ny).unwrap();

        // Clone grids for the closure
        let x_grid_closure = x_grid.clone();
        let y_grid_closure = y_grid.clone();

        // Clone grids for later use
        let x_grid_apply = x_grid.clone();
        let y_grid_apply = y_grid.clone();

        // Extract options before moving self
        let ode_options = ODEOptions {
            method: self.options.ode_method,
            rtol: self.options.rtol,
            atol: self.options.atol,
            max_steps: self.options.max_steps.unwrap_or(500),
            h0: None,
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

        let time_range = self.time_range;
        let boundary_conditions = self.boundary_conditions.clone();

        // Move self into closure
        let solver = self;

        // Construct the ODE function that represents the PDE after spatial discretization
        let ode_func = move |t: f64, u_flat: ArrayView1<f64>| -> Array1<f64> {
            // Reshape the flattened array back to 2D for easier indexing
            let u = u_flat.into_shape_with_order((ny, nx)).unwrap();
            let mut dudt = Array2::zeros((ny, nx));

            // Apply finite difference approximations for interior points
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let y = y_grid[j];
                    let x = x_grid[i];
                    let u_val = u[[j, i]];

                    // Diffusion terms
                    let d2u_dx2 = (u[[j, i + 1]] - 2.0 * u[[j, i]] + u[[j, i - 1]]) / (dx * dx);
                    let d2u_dy2 = (u[[j + 1, i]] - 2.0 * u[[j, i]] + u[[j - 1, i]]) / (dy * dy);

                    let diffusion_term_x = (solver.diffusion_x)(x, y, t, u_val) * d2u_dx2;
                    let diffusion_term_y = (solver.diffusion_y)(x, y, t, u_val) * d2u_dy2;

                    // Advection terms
                    let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                        let du_dx = (u[[j, i + 1]] - u[[j, i - 1]]) / (2.0 * dx);
                        advection_x(x, y, t, u_val) * du_dx
                    } else {
                        0.0
                    };

                    let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                        let du_dy = (u[[j + 1, i]] - u[[j - 1, i]]) / (2.0 * dy);
                        advection_y(x, y, t, u_val) * du_dy
                    } else {
                        0.0
                    };

                    // Reaction term
                    let reaction_val = if let Some(reaction) = &solver.reaction_term {
                        reaction(x, y, t, u_val)
                    } else {
                        0.0
                    };

                    dudt[[j, i]] = diffusion_term_x
                        + diffusion_term_y
                        + advection_term_x
                        + advection_term_y
                        + reaction_val;
                }
            }

            // Apply boundary conditions
            for bc in &solver.boundary_conditions {
                match (bc.dimension, bc.location) {
                    // X-direction boundaries
                    (0, BoundaryLocation::Lower) => {
                        // Apply boundary condition at x[0] (left edge)
                        apply_boundary_condition_2d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            bc,
                            dx,
                            dy,
                            Some(0),
                            None,
                            None,
                            Some(ny),
                            &solver,
                        );
                    }
                    (0, BoundaryLocation::Upper) => {
                        // Apply boundary condition at x[nx-1] (right edge)
                        apply_boundary_condition_2d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            bc,
                            dx,
                            dy,
                            Some(nx - 1),
                            None,
                            None,
                            Some(ny),
                            &solver,
                        );
                    }
                    // Y-direction boundaries
                    (1, BoundaryLocation::Lower) => {
                        // Apply boundary condition at y[0] (bottom edge)
                        apply_boundary_condition_2d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            bc,
                            dx,
                            dy,
                            None,
                            Some(0),
                            Some(nx),
                            None,
                            &solver,
                        );
                    }
                    (1, BoundaryLocation::Upper) => {
                        // Apply boundary condition at y[ny-1] (top edge)
                        apply_boundary_condition_2d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            bc,
                            dx,
                            dy,
                            None,
                            Some(ny - 1),
                            Some(nx),
                            None,
                            &solver,
                        );
                    }
                    _ => {
                        // Invalid dimension
                        // We'll just ignore this case for now - validation occurs during initialization
                    }
                }
            }

            // Flatten the 2D dudt back to 1D for the ODE solver
            dudt.into_shape_with_order(nx * ny).unwrap()
        };

        // Apply Dirichlet boundary conditions to initial condition
        apply_dirichlet_conditions_to_initial(
            &mut u0,
            &boundary_conditions,
            &x_grid_apply,
            &y_grid_apply,
        );

        let u0_flat = u0.clone().into_shape_with_order(nx * ny).unwrap();

        // Solve the ODE system
        let ode_result = solve_ivp(ode_func, time_range, u0_flat, Some(ode_options))?;

        // Extract results
        let computation_time = start_time.elapsed().as_secs_f64();

        // Reshape the ODE result to match the spatial grid
        let t = ode_result.t.clone();
        let nt = t.len();

        // Create a 3D array with dimensions [time, y, x]
        let mut u_3d = Array3::zeros((nt, ny, nx));

        for (time_idx, y_flat) in ode_result.y.iter().enumerate() {
            let u_2d = y_flat.clone().into_shape_with_order((ny, nx)).unwrap();
            for j in 0..ny {
                for i in 0..nx {
                    u_3d[[time_idx, j, i]] = u_2d[[j, i]];
                }
            }
        }

        let ode_info = Some(format!(
            "ODE steps: {}, function evaluations: {}, successful steps: {}",
            ode_result.n_steps, ode_result.n_eval, ode_result.n_accepted,
        ));

        Ok(MOL2DResult {
            t: ode_result.t.into(),
            u: u_3d,
            ode_info,
            computation_time,
        })
    }
}

// Helper function to apply boundary conditions in 2D
#[allow(clippy::too_many_arguments)]
fn apply_boundary_condition_2d(
    dudt: &mut Array2<f64>,
    u: &ArrayView2<f64>,
    x_grid: &Array1<f64>,
    y_grid: &Array1<f64>,
    bc: &BoundaryCondition<f64>,
    dx: f64,
    dy: f64,
    i_fixed: Option<usize>,
    j_fixed: Option<usize>,
    i_range: Option<usize>,
    j_range: Option<usize>,
    solver: &MOLParabolicSolver2D,
) {
    let nx = x_grid.len();
    let ny = y_grid.len();

    let i_range = i_range.unwrap_or(1);
    let j_range = j_range.unwrap_or(1);

    match bc.bc_type {
        BoundaryConditionType::Dirichlet => {
            // Fixed value: u = bc.value
            // For Dirichlet, we set dudt = 0 to maintain the fixed value
            if let Some(i) = i_fixed {
                for j in 0..j_range {
                    dudt[[j, i]] = 0.0;
                }
            } else if let Some(j) = j_fixed {
                for i in 0..i_range {
                    dudt[[j, i]] = 0.0;
                }
            }
        }
        BoundaryConditionType::Neumann => {
            // Gradient in the direction normal to the boundary
            if let Some(i) = i_fixed {
                // x-direction boundary (left or right)
                for j in 1..ny - 1 {
                    let y = y_grid[j];
                    let x = x_grid[i];
                    let u_val = u[[j, i]];
                    let t = 0.0; // Time is not used here

                    // Use one-sided difference for the normal direction
                    let du_dn = bc.value; // Given Neumann value

                    // For left boundary (i=0), ghost point is at i-1
                    // For right boundary (i=nx-1), ghost point is at i+1
                    let u_ghost = if i == 0 {
                        u[[j, i]] - dx * du_dn
                    } else {
                        u[[j, i]] + dx * du_dn
                    };

                    // Diffusion term in x direction
                    let d2u_dx2 = if i == 0 {
                        (u[[j, i + 1]] - 2.0 * u[[j, i]] + u_ghost) / (dx * dx)
                    } else {
                        (u_ghost - 2.0 * u[[j, i]] + u[[j, i - 1]]) / (dx * dx)
                    };

                    // Diffusion term in y direction (use central difference)
                    let d2u_dy2 = (u[[j + 1, i]] - 2.0 * u[[j, i]] + u[[j - 1, i]]) / (dy * dy);

                    let diffusion_term_x = (solver.diffusion_x)(x, y, t, u_val) * d2u_dx2;
                    let diffusion_term_y = (solver.diffusion_y)(x, y, t, u_val) * d2u_dy2;

                    // Advection terms
                    let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                        // Use one-sided difference for advection in x
                        let du_dx = if i == 0 {
                            (u[[j, i + 1]] - u[[j, i]]) / dx
                        } else {
                            (u[[j, i]] - u[[j, i - 1]]) / dx
                        };
                        advection_x(x, y, t, u_val) * du_dx
                    } else {
                        0.0
                    };

                    let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                        // Use central difference for advection in y
                        let du_dy = (u[[j + 1, i]] - u[[j - 1, i]]) / (2.0 * dy);
                        advection_y(x, y, t, u_val) * du_dy
                    } else {
                        0.0
                    };

                    // Reaction term
                    let reaction_term = if let Some(reaction) = &solver.reaction_term {
                        reaction(x, y, t, u_val)
                    } else {
                        0.0
                    };

                    dudt[[j, i]] = diffusion_term_x
                        + diffusion_term_y
                        + advection_term_x
                        + advection_term_y
                        + reaction_term;
                }
            } else if let Some(j) = j_fixed {
                // y-direction boundary (bottom or top)
                for i in 1..nx - 1 {
                    let y = y_grid[j];
                    let x = x_grid[i];
                    let u_val = u[[j, i]];
                    let t = 0.0; // Time is not used here

                    // Use one-sided difference for the normal direction
                    let du_dn = bc.value; // Given Neumann value

                    // For bottom boundary (j=0), ghost point is at j-1
                    // For top boundary (j=ny-1), ghost point is at j+1
                    let u_ghost = if j == 0 {
                        u[[j, i]] - dy * du_dn
                    } else {
                        u[[j, i]] + dy * du_dn
                    };

                    // Diffusion term in y direction
                    let d2u_dy2 = if j == 0 {
                        (u[[j + 1, i]] - 2.0 * u[[j, i]] + u_ghost) / (dy * dy)
                    } else {
                        (u_ghost - 2.0 * u[[j, i]] + u[[j - 1, i]]) / (dy * dy)
                    };

                    // Diffusion term in x direction (use central difference)
                    let d2u_dx2 = (u[[j, i + 1]] - 2.0 * u[[j, i]] + u[[j, i - 1]]) / (dx * dx);

                    let diffusion_term_x = (solver.diffusion_x)(x, y, t, u_val) * d2u_dx2;
                    let diffusion_term_y = (solver.diffusion_y)(x, y, t, u_val) * d2u_dy2;

                    // Advection terms
                    let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                        // Use central difference for advection in x
                        let du_dx = (u[[j, i + 1]] - u[[j, i - 1]]) / (2.0 * dx);
                        advection_x(x, y, t, u_val) * du_dx
                    } else {
                        0.0
                    };

                    let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                        // Use one-sided difference for advection in y
                        let du_dy = if j == 0 {
                            (u[[j + 1, i]] - u[[j, i]]) / dy
                        } else {
                            (u[[j, i]] - u[[j - 1, i]]) / dy
                        };
                        advection_y(x, y, t, u_val) * du_dy
                    } else {
                        0.0
                    };

                    // Reaction term
                    let reaction_term = if let Some(reaction) = &solver.reaction_term {
                        reaction(x, y, t, u_val)
                    } else {
                        0.0
                    };

                    dudt[[j, i]] = diffusion_term_x
                        + diffusion_term_y
                        + advection_term_x
                        + advection_term_y
                        + reaction_term;
                }
            }
        }
        BoundaryConditionType::Robin => {
            // Robin boundary condition: a*u + b*du/dn = c
            if let Some([a, b, c]) = bc.coefficients {
                if let Some(i) = i_fixed {
                    // x-direction boundary (left or right)
                    for j in 1..ny - 1 {
                        let y = y_grid[j];
                        let x = x_grid[i];
                        let u_val = u[[j, i]];
                        let t = 0.0; // Time is not used here

                        // Solve for ghost point value using Robin condition
                        let du_dn = (c - a * u_val) / b;

                        // For left boundary (i=0), ghost point is at i-1
                        // For right boundary (i=nx-1), ghost point is at i+1
                        let u_ghost = if i == 0 {
                            u[[j, i]] - dx * du_dn
                        } else {
                            u[[j, i]] + dx * du_dn
                        };

                        // Diffusion term in x direction
                        let d2u_dx2 = if i == 0 {
                            (u[[j, i + 1]] - 2.0 * u[[j, i]] + u_ghost) / (dx * dx)
                        } else {
                            (u_ghost - 2.0 * u[[j, i]] + u[[j, i - 1]]) / (dx * dx)
                        };

                        // Diffusion term in y direction (use central difference)
                        let d2u_dy2 = (u[[j + 1, i]] - 2.0 * u[[j, i]] + u[[j - 1, i]]) / (dy * dy);

                        let diffusion_term_x = (solver.diffusion_x)(x, y, t, u_val) * d2u_dx2;
                        let diffusion_term_y = (solver.diffusion_y)(x, y, t, u_val) * d2u_dy2;

                        // Advection terms
                        let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                            // Use one-sided difference for advection in x
                            let du_dx = if i == 0 {
                                (u[[j, i + 1]] - u[[j, i]]) / dx
                            } else {
                                (u[[j, i]] - u[[j, i - 1]]) / dx
                            };
                            advection_x(x, y, t, u_val) * du_dx
                        } else {
                            0.0
                        };

                        let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                            // Use central difference for advection in y
                            let du_dy = (u[[j + 1, i]] - u[[j - 1, i]]) / (2.0 * dy);
                            advection_y(x, y, t, u_val) * du_dy
                        } else {
                            0.0
                        };

                        // Reaction term
                        let reaction_term = if let Some(reaction) = &solver.reaction_term {
                            reaction(x, y, t, u_val)
                        } else {
                            0.0
                        };

                        dudt[[j, i]] = diffusion_term_x
                            + diffusion_term_y
                            + advection_term_x
                            + advection_term_y
                            + reaction_term;
                    }
                } else if let Some(j) = j_fixed {
                    // y-direction boundary (bottom or top)
                    for i in 1..nx - 1 {
                        let y = y_grid[j];
                        let x = x_grid[i];
                        let u_val = u[[j, i]];
                        let t = 0.0; // Time is not used here

                        // Solve for ghost point value using Robin condition
                        let du_dn = (c - a * u_val) / b;

                        // For bottom boundary (j=0), ghost point is at j-1
                        // For top boundary (j=ny-1), ghost point is at j+1
                        let u_ghost = if j == 0 {
                            u[[j, i]] - dy * du_dn
                        } else {
                            u[[j, i]] + dy * du_dn
                        };

                        // Diffusion term in y direction
                        let d2u_dy2 = if j == 0 {
                            (u[[j + 1, i]] - 2.0 * u[[j, i]] + u_ghost) / (dy * dy)
                        } else {
                            (u_ghost - 2.0 * u[[j, i]] + u[[j - 1, i]]) / (dy * dy)
                        };

                        // Diffusion term in x direction (use central difference)
                        let d2u_dx2 = (u[[j, i + 1]] - 2.0 * u[[j, i]] + u[[j, i - 1]]) / (dx * dx);

                        let diffusion_term_x = (solver.diffusion_x)(x, y, t, u_val) * d2u_dx2;
                        let diffusion_term_y = (solver.diffusion_y)(x, y, t, u_val) * d2u_dy2;

                        // Advection terms
                        let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                            // Use central difference for advection in x
                            let du_dx = (u[[j, i + 1]] - u[[j, i - 1]]) / (2.0 * dx);
                            advection_x(x, y, t, u_val) * du_dx
                        } else {
                            0.0
                        };

                        let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                            // Use one-sided difference for advection in y
                            let du_dy = if j == 0 {
                                (u[[j + 1, i]] - u[[j, i]]) / dy
                            } else {
                                (u[[j, i]] - u[[j - 1, i]]) / dy
                            };
                            advection_y(x, y, t, u_val) * du_dy
                        } else {
                            0.0
                        };

                        // Reaction term
                        let reaction_term = if let Some(reaction) = &solver.reaction_term {
                            reaction(x, y, t, u_val)
                        } else {
                            0.0
                        };

                        dudt[[j, i]] = diffusion_term_x
                            + diffusion_term_y
                            + advection_term_x
                            + advection_term_y
                            + reaction_term;
                    }
                }
            }
        }
        BoundaryConditionType::Periodic => {
            // Periodic boundary - handle in the ODE function itself
            // This is more complex for 2D problems and requires special handling of corners
            // For now, we'll leave this unimplemented
        }
    }
}

// Helper function to apply Dirichlet boundary conditions to initial condition
fn apply_dirichlet_conditions_to_initial(
    u0: &mut Array2<f64>,
    boundary_conditions: &[BoundaryCondition<f64>],
    x_grid: &Array1<f64>,
    y_grid: &Array1<f64>,
) {
    let nx = x_grid.len();
    let ny = y_grid.len();

    for bc in boundary_conditions {
        if bc.bc_type == BoundaryConditionType::Dirichlet {
            match (bc.dimension, bc.location) {
                (0, BoundaryLocation::Lower) => {
                    // Left boundary (x = x[0])
                    for j in 0..ny {
                        u0[[j, 0]] = bc.value;
                    }
                }
                (0, BoundaryLocation::Upper) => {
                    // Right boundary (x = x[nx-1])
                    for j in 0..ny {
                        u0[[j, nx - 1]] = bc.value;
                    }
                }
                (1, BoundaryLocation::Lower) => {
                    // Bottom boundary (y = y[0])
                    for i in 0..nx {
                        u0[[0, i]] = bc.value;
                    }
                }
                (1, BoundaryLocation::Upper) => {
                    // Top boundary (y = y[ny-1])
                    for i in 0..nx {
                        u0[[ny - 1, i]] = bc.value;
                    }
                }
                _ => {
                    // Invalid dimension
                    // We'll just ignore this case - validation occurs during initialization
                }
            }
        }
    }
}

/// Convert a MOL2DResult to a PDESolution
impl From<MOL2DResult> for PDESolution<f64> {
    fn from(result: MOL2DResult) -> Self {
        let mut grids = Vec::new();

        // Add time grid
        grids.push(result.t.clone());

        // Extract spatial grids from solution shape
        let nt = result.t.len();
        let ny = result.u.shape()[1];
        let nx = result.u.shape()[2];

        // Create spatial grids (we don't have the actual grid values, so use linspace)
        let y_grid = Array1::linspace(0.0, 1.0, ny);
        let x_grid = Array1::linspace(0.0, 1.0, nx);
        grids.push(y_grid);
        grids.push(x_grid);

        // Convert the 3D array to 2D arrays, one per time step
        let mut values = Vec::new();
        for t_idx in 0..nt {
            let time_slice = result.u.slice(s![t_idx, .., ..]).to_owned();
            values.push(time_slice);
        }

        // Create solver info
        let info = PDESolverInfo {
            num_iterations: 0, // This information is not available directly
            computation_time: result.computation_time,
            residual_norm: None,
            convergence_history: None,
            method: "Method of Lines (2D)".to_string(),
        };

        PDESolution {
            grids,
            values,
            error_estimate: None,
            info,
        }
    }
}
