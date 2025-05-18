//! Method of Lines for 3D parabolic PDEs
//!
//! This module implements the Method of Lines (MOL) approach for solving
//! 3D parabolic PDEs, such as the 3D heat equation and 3D advection-diffusion.

use ndarray::{Array1, Array3, Array4, ArrayView1, ArrayView3};
use std::sync::Arc;
use std::time::Instant;

use crate::ode::{solve_ivp, ODEOptions};
use crate::pde::finite_difference::FiniteDifferenceScheme;
use crate::pde::{
    BoundaryCondition, BoundaryConditionType, BoundaryLocation, Domain, PDEError, PDEResult,
    PDESolution, PDESolverInfo,
};

/// Result of 3D method of lines solution
pub struct MOL3DResult {
    /// Time points
    pub t: Array1<f64>,

    /// Solution values, indexed as [time, z, y, x]
    pub u: Array4<f64>,

    /// ODE solver information
    pub ode_info: Option<String>,

    /// Computation time
    pub computation_time: f64,
}

/// Method of Lines solver for 3D parabolic PDEs
///
/// Solves equations of the form:
/// ∂u/∂t = ∂/∂x(D_x ∂u/∂x) + ∂/∂y(D_y ∂u/∂y) + ∂/∂z(D_z ∂u/∂z) +
///         v_x ∂u/∂x + v_y ∂u/∂y + v_z ∂u/∂z + f(x,y,z,t,u)
#[derive(Clone)]
pub struct MOLParabolicSolver3D {
    /// Spatial domain
    domain: Domain,

    /// Time range [t_start, t_end]
    time_range: [f64; 2],

    /// Diffusion coefficient function D_x(x, y, z, t, u) for ∂²u/∂x²
    diffusion_x: Arc<dyn Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync>,

    /// Diffusion coefficient function D_y(x, y, z, t, u) for ∂²u/∂y²
    diffusion_y: Arc<dyn Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync>,

    /// Diffusion coefficient function D_z(x, y, z, t, u) for ∂²u/∂z²
    diffusion_z: Arc<dyn Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync>,

    /// Advection coefficient function v_x(x, y, z, t, u) for ∂u/∂x
    advection_x: Option<Arc<dyn Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync>>,

    /// Advection coefficient function v_y(x, y, z, t, u) for ∂u/∂y
    advection_y: Option<Arc<dyn Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync>>,

    /// Advection coefficient function v_z(x, y, z, t, u) for ∂u/∂z
    advection_z: Option<Arc<dyn Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync>>,

    /// Reaction term function f(x, y, z, t, u)
    reaction_term: Option<Arc<dyn Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync>>,

    /// Initial condition function u(x, y, z, 0)
    initial_condition: Arc<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>,

    /// Boundary conditions
    boundary_conditions: Vec<BoundaryCondition<f64>>,

    /// Finite difference scheme for spatial discretization
    fd_scheme: FiniteDifferenceScheme,

    /// Solver options
    options: super::MOLOptions,
}

impl MOLParabolicSolver3D {
    /// Create a new Method of Lines solver for 3D parabolic PDEs
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        domain: Domain,
        time_range: [f64; 2],
        diffusion_x: impl Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        diffusion_y: impl Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        diffusion_z: impl Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        initial_condition: impl Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
        boundary_conditions: Vec<BoundaryCondition<f64>>,
        options: Option<super::MOLOptions>,
    ) -> PDEResult<Self> {
        // Validate the domain
        if domain.dimensions() != 3 {
            return Err(PDEError::DomainError(
                "Domain must be 3-dimensional for 3D parabolic solver".to_string(),
            ));
        }

        // Validate time range
        if time_range[0] >= time_range[1] {
            return Err(PDEError::DomainError(
                "Invalid time range: start must be less than end".to_string(),
            ));
        }

        // Validate boundary conditions
        if boundary_conditions.len() != 6 {
            return Err(PDEError::BoundaryConditions(
                "3D parabolic PDE requires exactly 6 boundary conditions (one for each face)"
                    .to_string(),
            ));
        }

        // Ensure we have boundary conditions for all dimensions/faces
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
        let has_z_lower = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Lower && bc.dimension == 2);
        let has_z_upper = boundary_conditions
            .iter()
            .any(|bc| bc.location == BoundaryLocation::Upper && bc.dimension == 2);

        if !has_x_lower
            || !has_x_upper
            || !has_y_lower
            || !has_y_upper
            || !has_z_lower
            || !has_z_upper
        {
            return Err(PDEError::BoundaryConditions(
                "3D parabolic PDE requires boundary conditions for all faces of the domain"
                    .to_string(),
            ));
        }

        Ok(MOLParabolicSolver3D {
            domain,
            time_range,
            diffusion_x: Arc::new(diffusion_x),
            diffusion_y: Arc::new(diffusion_y),
            diffusion_z: Arc::new(diffusion_z),
            advection_x: None,
            advection_y: None,
            advection_z: None,
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
        advection_x: impl Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        advection_y: impl Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
        advection_z: impl Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.advection_x = Some(Arc::new(advection_x));
        self.advection_y = Some(Arc::new(advection_y));
        self.advection_z = Some(Arc::new(advection_z));
        self
    }

    /// Add a reaction term to the PDE
    pub fn with_reaction(
        mut self,
        reaction_term: impl Fn(f64, f64, f64, f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.reaction_term = Some(Arc::new(reaction_term));
        self
    }

    /// Set the finite difference scheme for spatial discretization
    pub fn with_fd_scheme(mut self, scheme: FiniteDifferenceScheme) -> Self {
        self.fd_scheme = scheme;
        self
    }

    /// Solve the 3D parabolic PDE
    pub fn solve(self) -> PDEResult<MOL3DResult> {
        let start_time = Instant::now();

        // Generate spatial grids
        let x_grid = self.domain.grid(0)?;
        let y_grid = self.domain.grid(1)?;
        let z_grid = self.domain.grid(2)?;
        let nx = x_grid.len();
        let ny = y_grid.len();
        let nz = z_grid.len();
        let dx = self.domain.grid_spacing(0)?;
        let dy = self.domain.grid_spacing(1)?;
        let dz = self.domain.grid_spacing(2)?;

        // Create initial condition 3D grid and flatten it for the ODE solver
        let mut u0 = Array3::zeros((nz, ny, nx));
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    u0[[k, j, i]] = (self.initial_condition)(x_grid[i], y_grid[j], z_grid[k]);
                }
            }
        }

        // Flatten the 3D grid into a 1D array for the ODE solver
        // Note: This is computed but not used, likely for future use
        let _u0_flat = u0.clone().into_shape_with_order(nx * ny * nz).unwrap();

        // Clone grids for the closure
        let x_grid_closure = x_grid.clone();
        let y_grid_closure = y_grid.clone();
        let z_grid_closure = z_grid.clone();

        // Clone grids for later use outside closure
        let x_grid_apply = x_grid.clone();
        let y_grid_apply = y_grid.clone();
        let z_grid_apply = z_grid.clone();

        // Extract options and other needed values before moving self
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
            // Reshape the flattened array back to 3D for easier indexing
            let u = u_flat.into_shape_with_order((nz, ny, nx)).unwrap();
            let mut dudt = Array3::zeros((nz, ny, nx));

            // Apply finite difference approximations for interior points
            for k in 1..nz - 1 {
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        let z = z_grid[k];
                        let y = y_grid[j];
                        let x = x_grid[i];
                        let u_val = u[[k, j, i]];

                        // Diffusion terms
                        let d2u_dx2 =
                            (u[[k, j, i + 1]] - 2.0 * u[[k, j, i]] + u[[k, j, i - 1]]) / (dx * dx);
                        let d2u_dy2 =
                            (u[[k, j + 1, i]] - 2.0 * u[[k, j, i]] + u[[k, j - 1, i]]) / (dy * dy);
                        let d2u_dz2 =
                            (u[[k + 1, j, i]] - 2.0 * u[[k, j, i]] + u[[k - 1, j, i]]) / (dz * dz);

                        let diffusion_term_x = (solver.diffusion_x)(x, y, z, t, u_val) * d2u_dx2;
                        let diffusion_term_y = (solver.diffusion_y)(x, y, z, t, u_val) * d2u_dy2;
                        let diffusion_term_z = (solver.diffusion_z)(x, y, z, t, u_val) * d2u_dz2;

                        // Advection terms
                        let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                            let du_dx = (u[[k, j, i + 1]] - u[[k, j, i - 1]]) / (2.0 * dx);
                            advection_x(x, y, z, t, u_val) * du_dx
                        } else {
                            0.0
                        };

                        let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                            let du_dy = (u[[k, j + 1, i]] - u[[k, j - 1, i]]) / (2.0 * dy);
                            advection_y(x, y, z, t, u_val) * du_dy
                        } else {
                            0.0
                        };

                        let advection_term_z = if let Some(advection_z) = &solver.advection_z {
                            let du_dz = (u[[k + 1, j, i]] - u[[k - 1, j, i]]) / (2.0 * dz);
                            advection_z(x, y, z, t, u_val) * du_dz
                        } else {
                            0.0
                        };

                        // Reaction term
                        let reaction_term = if let Some(reaction) = &solver.reaction_term {
                            reaction(x, y, z, t, u_val)
                        } else {
                            0.0
                        };

                        dudt[[k, j, i]] = diffusion_term_x
                            + diffusion_term_y
                            + diffusion_term_z
                            + advection_term_x
                            + advection_term_y
                            + advection_term_z
                            + reaction_term;
                    }
                }
            }

            // Apply boundary conditions
            for bc in &solver.boundary_conditions {
                match (bc.dimension, bc.location) {
                    // X-direction boundaries
                    (0, BoundaryLocation::Lower) => {
                        // Apply boundary condition at x[0] (left face)
                        apply_boundary_condition_3d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            &z_grid_closure,
                            bc,
                            dx,
                            dy,
                            dz,
                            Some(0),
                            None,
                            None,
                            Some(ny),
                            Some(nz),
                            &solver,
                        );
                    }
                    (0, BoundaryLocation::Upper) => {
                        // Apply boundary condition at x[nx-1] (right face)
                        apply_boundary_condition_3d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            &z_grid_closure,
                            bc,
                            dx,
                            dy,
                            dz,
                            Some(nx - 1),
                            None,
                            None,
                            Some(ny),
                            Some(nz),
                            &solver,
                        );
                    }
                    // Y-direction boundaries
                    (1, BoundaryLocation::Lower) => {
                        // Apply boundary condition at y[0] (front face)
                        apply_boundary_condition_3d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            &z_grid_closure,
                            bc,
                            dx,
                            dy,
                            dz,
                            None,
                            Some(0),
                            Some(nx),
                            None,
                            Some(nz),
                            &solver,
                        );
                    }
                    (1, BoundaryLocation::Upper) => {
                        // Apply boundary condition at y[ny-1] (back face)
                        apply_boundary_condition_3d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            &z_grid_closure,
                            bc,
                            dx,
                            dy,
                            dz,
                            None,
                            Some(ny - 1),
                            Some(nx),
                            None,
                            Some(nz),
                            &solver,
                        );
                    }
                    // Z-direction boundaries
                    (2, BoundaryLocation::Lower) => {
                        // Apply boundary condition at z[0] (bottom face)
                        apply_boundary_condition_3d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            &z_grid_closure,
                            bc,
                            dx,
                            dy,
                            dz,
                            None,
                            None,
                            Some(nx),
                            Some(ny),
                            Some(0),
                            &solver,
                        );
                    }
                    (2, BoundaryLocation::Upper) => {
                        // Apply boundary condition at z[nz-1] (top face)
                        apply_boundary_condition_3d(
                            &mut dudt,
                            &u,
                            &x_grid_closure,
                            &y_grid_closure,
                            &z_grid_closure,
                            bc,
                            dx,
                            dy,
                            dz,
                            None,
                            None,
                            Some(nx),
                            Some(ny),
                            Some(nz - 1),
                            &solver,
                        );
                    }
                    _ => {
                        // Invalid dimension
                        // We'll just ignore this case for now - validation occurs during initialization
                    }
                }
            }

            // Flatten the 3D dudt back to 1D for the ODE solver
            dudt.into_shape_with_order(nx * ny * nz).unwrap()
        };

        // Use the ode_options from earlier

        // Apply Dirichlet boundary conditions to initial condition
        apply_dirichlet_conditions_to_initial_3d(
            &mut u0,
            &boundary_conditions,
            &x_grid_apply,
            &y_grid_apply,
            &z_grid_apply,
        );

        let u0_flat = u0.into_shape_with_order(nx * ny * nz).unwrap();

        // Solve the ODE system
        let ode_result = solve_ivp(ode_func, time_range, u0_flat, Some(ode_options))?;

        // Extract results
        let computation_time = start_time.elapsed().as_secs_f64();

        // Reshape the ODE result to match the spatial grid
        let t = ode_result.t.clone();
        let nt = t.len();

        // Create a 4D array with dimensions [time, z, y, x]
        let mut u_4d = Array4::zeros((nt, nz, ny, nx));

        for (time_idx, y_flat) in ode_result.y.iter().enumerate() {
            let u_3d = y_flat.clone().into_shape_with_order((nz, ny, nx)).unwrap();
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        u_4d[[time_idx, k, j, i]] = u_3d[[k, j, i]];
                    }
                }
            }
        }

        let ode_info = Some(format!(
            "ODE steps: {}, function evaluations: {}, successful steps: {}",
            ode_result.n_steps, ode_result.n_eval, ode_result.n_accepted,
        ));

        Ok(MOL3DResult {
            t: ode_result.t.into(),
            u: u_4d,
            ode_info,
            computation_time,
        })
    }
}

// Helper function to apply boundary conditions in 3D
#[allow(clippy::too_many_arguments)]
fn apply_boundary_condition_3d(
    dudt: &mut Array3<f64>,
    u: &ArrayView3<f64>,
    x_grid: &Array1<f64>,
    y_grid: &Array1<f64>,
    z_grid: &Array1<f64>,
    bc: &BoundaryCondition<f64>,
    dx: f64,
    dy: f64,
    dz: f64,
    i_fixed: Option<usize>,
    j_fixed: Option<usize>,
    i_range: Option<usize>,
    j_range: Option<usize>,
    k_fixed: Option<usize>,
    solver: &MOLParabolicSolver3D,
) {
    let nx = x_grid.len();
    let ny = y_grid.len();
    let nz = z_grid.len();

    let i_range = i_range.unwrap_or(1);
    let j_range = j_range.unwrap_or(1);
    let t = 0.0; // Time is not used here

    match bc.bc_type {
        BoundaryConditionType::Dirichlet => {
            // Fixed value: u = bc.value
            // For Dirichlet, we set dudt = 0 to maintain the fixed value

            if let Some(i) = i_fixed {
                // X-direction boundary (left or right face)
                for k in 0..nz {
                    for j in 0..ny {
                        dudt[[k, j, i]] = 0.0;
                    }
                }
            } else if let Some(j) = j_fixed {
                // Y-direction boundary (front or back face)
                for k in 0..nz {
                    for i in 0..i_range {
                        dudt[[k, j, i]] = 0.0;
                    }
                }
            } else if let Some(k) = k_fixed {
                // Z-direction boundary (bottom or top face)
                for j in 0..j_range {
                    for i in 0..i_range {
                        dudt[[k, j, i]] = 0.0;
                    }
                }
            }
        }
        BoundaryConditionType::Neumann => {
            // Gradient in the direction normal to the boundary

            if let Some(i) = i_fixed {
                // X-direction boundary (left or right face)
                for k in 1..nz - 1 {
                    for j in 1..ny - 1 {
                        let z = z_grid[k];
                        let y = y_grid[j];
                        let x = x_grid[i];
                        let u_val = u[[k, j, i]];

                        // Use one-sided difference for the normal direction
                        let du_dn = bc.value; // Given Neumann value

                        // For left boundary (i=0), ghost point is at i-1
                        // For right boundary (i=nx-1), ghost point is at i+1
                        let u_ghost = if i == 0 {
                            u[[k, j, i]] - dx * du_dn
                        } else {
                            u[[k, j, i]] + dx * du_dn
                        };

                        // Diffusion term in x direction
                        let d2u_dx2 = if i == 0 {
                            (u[[k, j, i + 1]] - 2.0 * u[[k, j, i]] + u_ghost) / (dx * dx)
                        } else {
                            (u_ghost - 2.0 * u[[k, j, i]] + u[[k, j, i - 1]]) / (dx * dx)
                        };

                        // Diffusion terms in y and z direction (use central difference)
                        let d2u_dy2 =
                            (u[[k, j + 1, i]] - 2.0 * u[[k, j, i]] + u[[k, j - 1, i]]) / (dy * dy);
                        let d2u_dz2 =
                            (u[[k + 1, j, i]] - 2.0 * u[[k, j, i]] + u[[k - 1, j, i]]) / (dz * dz);

                        let diffusion_term_x = (solver.diffusion_x)(x, y, z, t, u_val) * d2u_dx2;
                        let diffusion_term_y = (solver.diffusion_y)(x, y, z, t, u_val) * d2u_dy2;
                        let diffusion_term_z = (solver.diffusion_z)(x, y, z, t, u_val) * d2u_dz2;

                        // Advection terms
                        let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                            // Use one-sided difference for advection in x
                            let du_dx = if i == 0 {
                                (u[[k, j, i + 1]] - u[[k, j, i]]) / dx
                            } else {
                                (u[[k, j, i]] - u[[k, j, i - 1]]) / dx
                            };
                            advection_x(x, y, z, t, u_val) * du_dx
                        } else {
                            0.0
                        };

                        let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                            // Use central difference for advection in y
                            let du_dy = (u[[k, j + 1, i]] - u[[k, j - 1, i]]) / (2.0 * dy);
                            advection_y(x, y, z, t, u_val) * du_dy
                        } else {
                            0.0
                        };

                        let advection_term_z = if let Some(advection_z) = &solver.advection_z {
                            // Use central difference for advection in z
                            let du_dz = (u[[k + 1, j, i]] - u[[k - 1, j, i]]) / (2.0 * dz);
                            advection_z(x, y, z, t, u_val) * du_dz
                        } else {
                            0.0
                        };

                        // Reaction term
                        let reaction_term = if let Some(reaction) = &solver.reaction_term {
                            reaction(x, y, z, t, u_val)
                        } else {
                            0.0
                        };

                        dudt[[k, j, i]] = diffusion_term_x
                            + diffusion_term_y
                            + diffusion_term_z
                            + advection_term_x
                            + advection_term_y
                            + advection_term_z
                            + reaction_term;
                    }
                }
            } else if let Some(j) = j_fixed {
                // Y-direction boundary (front or back face)
                for k in 1..nz - 1 {
                    for i in 1..nx - 1 {
                        let z = z_grid[k];
                        let y = y_grid[j];
                        let x = x_grid[i];
                        let u_val = u[[k, j, i]];

                        // Use one-sided difference for the normal direction
                        let du_dn = bc.value; // Given Neumann value

                        // For front boundary (j=0), ghost point is at j-1
                        // For back boundary (j=ny-1), ghost point is at j+1
                        let u_ghost = if j == 0 {
                            u[[k, j, i]] - dy * du_dn
                        } else {
                            u[[k, j, i]] + dy * du_dn
                        };

                        // Diffusion term in y direction
                        let d2u_dy2 = if j == 0 {
                            (u[[k, j + 1, i]] - 2.0 * u[[k, j, i]] + u_ghost) / (dy * dy)
                        } else {
                            (u_ghost - 2.0 * u[[k, j, i]] + u[[k, j - 1, i]]) / (dy * dy)
                        };

                        // Diffusion terms in x and z direction (use central difference)
                        let d2u_dx2 =
                            (u[[k, j, i + 1]] - 2.0 * u[[k, j, i]] + u[[k, j, i - 1]]) / (dx * dx);
                        let d2u_dz2 =
                            (u[[k + 1, j, i]] - 2.0 * u[[k, j, i]] + u[[k - 1, j, i]]) / (dz * dz);

                        let diffusion_term_x = (solver.diffusion_x)(x, y, z, t, u_val) * d2u_dx2;
                        let diffusion_term_y = (solver.diffusion_y)(x, y, z, t, u_val) * d2u_dy2;
                        let diffusion_term_z = (solver.diffusion_z)(x, y, z, t, u_val) * d2u_dz2;

                        // Advection terms
                        let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                            // Use central difference for advection in x
                            let du_dx = (u[[k, j, i + 1]] - u[[k, j, i - 1]]) / (2.0 * dx);
                            advection_x(x, y, z, t, u_val) * du_dx
                        } else {
                            0.0
                        };

                        let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                            // Use one-sided difference for advection in y
                            let du_dy = if j == 0 {
                                (u[[k, j + 1, i]] - u[[k, j, i]]) / dy
                            } else {
                                (u[[k, j, i]] - u[[k, j - 1, i]]) / dy
                            };
                            advection_y(x, y, z, t, u_val) * du_dy
                        } else {
                            0.0
                        };

                        let advection_term_z = if let Some(advection_z) = &solver.advection_z {
                            // Use central difference for advection in z
                            let du_dz = (u[[k + 1, j, i]] - u[[k - 1, j, i]]) / (2.0 * dz);
                            advection_z(x, y, z, t, u_val) * du_dz
                        } else {
                            0.0
                        };

                        // Reaction term
                        let reaction_term = if let Some(reaction) = &solver.reaction_term {
                            reaction(x, y, z, t, u_val)
                        } else {
                            0.0
                        };

                        dudt[[k, j, i]] = diffusion_term_x
                            + diffusion_term_y
                            + diffusion_term_z
                            + advection_term_x
                            + advection_term_y
                            + advection_term_z
                            + reaction_term;
                    }
                }
            } else if let Some(k) = k_fixed {
                // Z-direction boundary (bottom or top face)
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        let z = z_grid[k];
                        let y = y_grid[j];
                        let x = x_grid[i];
                        let u_val = u[[k, j, i]];

                        // Use one-sided difference for the normal direction
                        let du_dn = bc.value; // Given Neumann value

                        // For bottom boundary (k=0), ghost point is at k-1
                        // For top boundary (k=nz-1), ghost point is at k+1
                        let u_ghost = if k == 0 {
                            u[[k, j, i]] - dz * du_dn
                        } else {
                            u[[k, j, i]] + dz * du_dn
                        };

                        // Diffusion term in z direction
                        let d2u_dz2 = if k == 0 {
                            (u[[k + 1, j, i]] - 2.0 * u[[k, j, i]] + u_ghost) / (dz * dz)
                        } else {
                            (u_ghost - 2.0 * u[[k, j, i]] + u[[k - 1, j, i]]) / (dz * dz)
                        };

                        // Diffusion terms in x and y direction (use central difference)
                        let d2u_dx2 =
                            (u[[k, j, i + 1]] - 2.0 * u[[k, j, i]] + u[[k, j, i - 1]]) / (dx * dx);
                        let d2u_dy2 =
                            (u[[k, j + 1, i]] - 2.0 * u[[k, j, i]] + u[[k, j - 1, i]]) / (dy * dy);

                        let diffusion_term_x = (solver.diffusion_x)(x, y, z, t, u_val) * d2u_dx2;
                        let diffusion_term_y = (solver.diffusion_y)(x, y, z, t, u_val) * d2u_dy2;
                        let diffusion_term_z = (solver.diffusion_z)(x, y, z, t, u_val) * d2u_dz2;

                        // Advection terms
                        let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                            // Use central difference for advection in x
                            let du_dx = (u[[k, j, i + 1]] - u[[k, j, i - 1]]) / (2.0 * dx);
                            advection_x(x, y, z, t, u_val) * du_dx
                        } else {
                            0.0
                        };

                        let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                            // Use central difference for advection in y
                            let du_dy = (u[[k, j + 1, i]] - u[[k, j - 1, i]]) / (2.0 * dy);
                            advection_y(x, y, z, t, u_val) * du_dy
                        } else {
                            0.0
                        };

                        let advection_term_z = if let Some(advection_z) = &solver.advection_z {
                            // Use one-sided difference for advection in z
                            let du_dz = if k == 0 {
                                (u[[k + 1, j, i]] - u[[k, j, i]]) / dz
                            } else {
                                (u[[k, j, i]] - u[[k - 1, j, i]]) / dz
                            };
                            advection_z(x, y, z, t, u_val) * du_dz
                        } else {
                            0.0
                        };

                        // Reaction term
                        let reaction_term = if let Some(reaction) = &solver.reaction_term {
                            reaction(x, y, z, t, u_val)
                        } else {
                            0.0
                        };

                        dudt[[k, j, i]] = diffusion_term_x
                            + diffusion_term_y
                            + diffusion_term_z
                            + advection_term_x
                            + advection_term_y
                            + advection_term_z
                            + reaction_term;
                    }
                }
            }
        }
        BoundaryConditionType::Robin => {
            // Robin boundary condition: a*u + b*du/dn = c
            if let Some([a, b, c]) = bc.coefficients {
                if let Some(i) = i_fixed {
                    // X-direction boundary (left or right face)
                    for k in 1..nz - 1 {
                        for j in 1..ny - 1 {
                            let z = z_grid[k];
                            let y = y_grid[j];
                            let x = x_grid[i];
                            let u_val = u[[k, j, i]];

                            // Solve for ghost point value using Robin condition
                            let du_dn = (c - a * u_val) / b;

                            // For left boundary (i=0), ghost point is at i-1
                            // For right boundary (i=nx-1), ghost point is at i+1
                            let u_ghost = if i == 0 {
                                u[[k, j, i]] - dx * du_dn
                            } else {
                                u[[k, j, i]] + dx * du_dn
                            };

                            // Diffusion term in x direction
                            let d2u_dx2 = if i == 0 {
                                (u[[k, j, i + 1]] - 2.0 * u[[k, j, i]] + u_ghost) / (dx * dx)
                            } else {
                                (u_ghost - 2.0 * u[[k, j, i]] + u[[k, j, i - 1]]) / (dx * dx)
                            };

                            // Diffusion terms in y and z direction (use central difference)
                            let d2u_dy2 = (u[[k, j + 1, i]] - 2.0 * u[[k, j, i]]
                                + u[[k, j - 1, i]])
                                / (dy * dy);
                            let d2u_dz2 = (u[[k + 1, j, i]] - 2.0 * u[[k, j, i]]
                                + u[[k - 1, j, i]])
                                / (dz * dz);

                            let diffusion_term_x =
                                (solver.diffusion_x)(x, y, z, t, u_val) * d2u_dx2;
                            let diffusion_term_y =
                                (solver.diffusion_y)(x, y, z, t, u_val) * d2u_dy2;
                            let diffusion_term_z =
                                (solver.diffusion_z)(x, y, z, t, u_val) * d2u_dz2;

                            // Advection terms
                            let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                                // Use one-sided difference for advection in x
                                let du_dx = if i == 0 {
                                    (u[[k, j, i + 1]] - u[[k, j, i]]) / dx
                                } else {
                                    (u[[k, j, i]] - u[[k, j, i - 1]]) / dx
                                };
                                advection_x(x, y, z, t, u_val) * du_dx
                            } else {
                                0.0
                            };

                            let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                                // Use central difference for advection in y
                                let du_dy = (u[[k, j + 1, i]] - u[[k, j - 1, i]]) / (2.0 * dy);
                                advection_y(x, y, z, t, u_val) * du_dy
                            } else {
                                0.0
                            };

                            let advection_term_z = if let Some(advection_z) = &solver.advection_z {
                                // Use central difference for advection in z
                                let du_dz = (u[[k + 1, j, i]] - u[[k - 1, j, i]]) / (2.0 * dz);
                                advection_z(x, y, z, t, u_val) * du_dz
                            } else {
                                0.0
                            };

                            // Reaction term
                            let reaction_term = if let Some(reaction) = &solver.reaction_term {
                                reaction(x, y, z, t, u_val)
                            } else {
                                0.0
                            };

                            dudt[[k, j, i]] = diffusion_term_x
                                + diffusion_term_y
                                + diffusion_term_z
                                + advection_term_x
                                + advection_term_y
                                + advection_term_z
                                + reaction_term;
                        }
                    }
                } else if let Some(j) = j_fixed {
                    // Y-direction boundary (front or back face)
                    for k in 1..nz - 1 {
                        for i in 1..nx - 1 {
                            let z = z_grid[k];
                            let y = y_grid[j];
                            let x = x_grid[i];
                            let u_val = u[[k, j, i]];

                            // Solve for ghost point value using Robin condition
                            let du_dn = (c - a * u_val) / b;

                            // For front boundary (j=0), ghost point is at j-1
                            // For back boundary (j=ny-1), ghost point is at j+1
                            let u_ghost = if j == 0 {
                                u[[k, j, i]] - dy * du_dn
                            } else {
                                u[[k, j, i]] + dy * du_dn
                            };

                            // Diffusion term in y direction
                            let d2u_dy2 = if j == 0 {
                                (u[[k, j + 1, i]] - 2.0 * u[[k, j, i]] + u_ghost) / (dy * dy)
                            } else {
                                (u_ghost - 2.0 * u[[k, j, i]] + u[[k, j - 1, i]]) / (dy * dy)
                            };

                            // Diffusion terms in x and z direction (use central difference)
                            let d2u_dx2 = (u[[k, j, i + 1]] - 2.0 * u[[k, j, i]]
                                + u[[k, j, i - 1]])
                                / (dx * dx);
                            let d2u_dz2 = (u[[k + 1, j, i]] - 2.0 * u[[k, j, i]]
                                + u[[k - 1, j, i]])
                                / (dz * dz);

                            let diffusion_term_x =
                                (solver.diffusion_x)(x, y, z, t, u_val) * d2u_dx2;
                            let diffusion_term_y =
                                (solver.diffusion_y)(x, y, z, t, u_val) * d2u_dy2;
                            let diffusion_term_z =
                                (solver.diffusion_z)(x, y, z, t, u_val) * d2u_dz2;

                            // Advection terms
                            let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                                // Use central difference for advection in x
                                let du_dx = (u[[k, j, i + 1]] - u[[k, j, i - 1]]) / (2.0 * dx);
                                advection_x(x, y, z, t, u_val) * du_dx
                            } else {
                                0.0
                            };

                            let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                                // Use one-sided difference for advection in y
                                let du_dy = if j == 0 {
                                    (u[[k, j + 1, i]] - u[[k, j, i]]) / dy
                                } else {
                                    (u[[k, j, i]] - u[[k, j - 1, i]]) / dy
                                };
                                advection_y(x, y, z, t, u_val) * du_dy
                            } else {
                                0.0
                            };

                            let advection_term_z = if let Some(advection_z) = &solver.advection_z {
                                // Use central difference for advection in z
                                let du_dz = (u[[k + 1, j, i]] - u[[k - 1, j, i]]) / (2.0 * dz);
                                advection_z(x, y, z, t, u_val) * du_dz
                            } else {
                                0.0
                            };

                            // Reaction term
                            let reaction_term = if let Some(reaction) = &solver.reaction_term {
                                reaction(x, y, z, t, u_val)
                            } else {
                                0.0
                            };

                            dudt[[k, j, i]] = diffusion_term_x
                                + diffusion_term_y
                                + diffusion_term_z
                                + advection_term_x
                                + advection_term_y
                                + advection_term_z
                                + reaction_term;
                        }
                    }
                } else if let Some(k) = k_fixed {
                    // Z-direction boundary (bottom or top face)
                    for j in 1..ny - 1 {
                        for i in 1..nx - 1 {
                            let z = z_grid[k];
                            let y = y_grid[j];
                            let x = x_grid[i];
                            let u_val = u[[k, j, i]];

                            // Solve for ghost point value using Robin condition
                            let du_dn = (c - a * u_val) / b;

                            // For bottom boundary (k=0), ghost point is at k-1
                            // For top boundary (k=nz-1), ghost point is at k+1
                            let u_ghost = if k == 0 {
                                u[[k, j, i]] - dz * du_dn
                            } else {
                                u[[k, j, i]] + dz * du_dn
                            };

                            // Diffusion term in z direction
                            let d2u_dz2 = if k == 0 {
                                (u[[k + 1, j, i]] - 2.0 * u[[k, j, i]] + u_ghost) / (dz * dz)
                            } else {
                                (u_ghost - 2.0 * u[[k, j, i]] + u[[k - 1, j, i]]) / (dz * dz)
                            };

                            // Diffusion terms in x and y direction (use central difference)
                            let d2u_dx2 = (u[[k, j, i + 1]] - 2.0 * u[[k, j, i]]
                                + u[[k, j, i - 1]])
                                / (dx * dx);
                            let d2u_dy2 = (u[[k, j + 1, i]] - 2.0 * u[[k, j, i]]
                                + u[[k, j - 1, i]])
                                / (dy * dy);

                            let diffusion_term_x =
                                (solver.diffusion_x)(x, y, z, t, u_val) * d2u_dx2;
                            let diffusion_term_y =
                                (solver.diffusion_y)(x, y, z, t, u_val) * d2u_dy2;
                            let diffusion_term_z =
                                (solver.diffusion_z)(x, y, z, t, u_val) * d2u_dz2;

                            // Advection terms
                            let advection_term_x = if let Some(advection_x) = &solver.advection_x {
                                // Use central difference for advection in x
                                let du_dx = (u[[k, j, i + 1]] - u[[k, j, i - 1]]) / (2.0 * dx);
                                advection_x(x, y, z, t, u_val) * du_dx
                            } else {
                                0.0
                            };

                            let advection_term_y = if let Some(advection_y) = &solver.advection_y {
                                // Use central difference for advection in y
                                let du_dy = (u[[k, j + 1, i]] - u[[k, j - 1, i]]) / (2.0 * dy);
                                advection_y(x, y, z, t, u_val) * du_dy
                            } else {
                                0.0
                            };

                            let advection_term_z = if let Some(advection_z) = &solver.advection_z {
                                // Use one-sided difference for advection in z
                                let du_dz = if k == 0 {
                                    (u[[k + 1, j, i]] - u[[k, j, i]]) / dz
                                } else {
                                    (u[[k, j, i]] - u[[k - 1, j, i]]) / dz
                                };
                                advection_z(x, y, z, t, u_val) * du_dz
                            } else {
                                0.0
                            };

                            // Reaction term
                            let reaction_term = if let Some(reaction) = &solver.reaction_term {
                                reaction(x, y, z, t, u_val)
                            } else {
                                0.0
                            };

                            dudt[[k, j, i]] = diffusion_term_x
                                + diffusion_term_y
                                + diffusion_term_z
                                + advection_term_x
                                + advection_term_y
                                + advection_term_z
                                + reaction_term;
                        }
                    }
                }
            }
        }
        BoundaryConditionType::Periodic => {
            // Periodic boundary - handle in the ODE function itself
            // This is more complex for 3D problems and requires special handling of edges and corners
            // For now, we'll leave this unimplemented
        }
    }
}

// Helper function to apply Dirichlet boundary conditions to initial condition
fn apply_dirichlet_conditions_to_initial_3d(
    u0: &mut Array3<f64>,
    boundary_conditions: &[BoundaryCondition<f64>],
    x_grid: &Array1<f64>,
    y_grid: &Array1<f64>,
    z_grid: &Array1<f64>,
) {
    let nx = x_grid.len();
    let ny = y_grid.len();
    let nz = z_grid.len();

    for bc in boundary_conditions {
        if bc.bc_type == BoundaryConditionType::Dirichlet {
            match (bc.dimension, bc.location) {
                (0, BoundaryLocation::Lower) => {
                    // Left boundary (x = x[0])
                    for k in 0..nz {
                        for j in 0..ny {
                            u0[[k, j, 0]] = bc.value;
                        }
                    }
                }
                (0, BoundaryLocation::Upper) => {
                    // Right boundary (x = x[nx-1])
                    for k in 0..nz {
                        for j in 0..ny {
                            u0[[k, j, nx - 1]] = bc.value;
                        }
                    }
                }
                (1, BoundaryLocation::Lower) => {
                    // Front boundary (y = y[0])
                    for k in 0..nz {
                        for i in 0..nx {
                            u0[[k, 0, i]] = bc.value;
                        }
                    }
                }
                (1, BoundaryLocation::Upper) => {
                    // Back boundary (y = y[ny-1])
                    for k in 0..nz {
                        for i in 0..nx {
                            u0[[k, ny - 1, i]] = bc.value;
                        }
                    }
                }
                (2, BoundaryLocation::Lower) => {
                    // Bottom boundary (z = z[0])
                    for j in 0..ny {
                        for i in 0..nx {
                            u0[[0, j, i]] = bc.value;
                        }
                    }
                }
                (2, BoundaryLocation::Upper) => {
                    // Top boundary (z = z[nz-1])
                    for j in 0..ny {
                        for i in 0..nx {
                            u0[[nz - 1, j, i]] = bc.value;
                        }
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

/// Convert a MOL3DResult to a PDESolution
impl From<MOL3DResult> for PDESolution<f64> {
    fn from(result: MOL3DResult) -> Self {
        let mut grids = Vec::new();

        // Add time grid
        grids.push(result.t.clone());

        // Extract spatial grids from solution shape
        let nt = result.t.len();
        let nz = result.u.shape()[1];
        let ny = result.u.shape()[2];
        let nx = result.u.shape()[3];

        // Create spatial grids (we don't have the actual grid values, so use linspace)
        let z_grid = Array1::linspace(0.0, 1.0, nz);
        let y_grid = Array1::linspace(0.0, 1.0, ny);
        let x_grid = Array1::linspace(0.0, 1.0, nx);
        grids.push(z_grid);
        grids.push(y_grid);
        grids.push(x_grid);

        // Convert the 4D array to a list of 2D arrays
        // For PDESolution format, we need to flatten the spatial dimensions
        let mut values = Vec::new();
        let total_spatial_points = nx * ny * nz;

        // Reshape the 4D array (time, z, y, x) to 2D (time, spatial_points)
        let u_reshaped = result
            .u
            .into_shape_with_order((nt, total_spatial_points))
            .unwrap();

        // Create a single 2D array with time on one dimension and flattened spatial points on the other
        values.push(u_reshaped.t().to_owned());

        // Create solver info
        let info = PDESolverInfo {
            num_iterations: 0, // This information is not available directly
            computation_time: result.computation_time,
            residual_norm: None,
            convergence_history: None,
            method: "Method of Lines (3D)".to_string(),
        };

        PDESolution {
            grids,
            values,
            error_estimate: None,
            info,
        }
    }
}
