//! Incompressible Navier-Stokes solver implementation
//!
//! This module provides the main solver for incompressible fluid flows using
//! the projection method to enforce the divergence-free constraint.

use crate::error::{IntegrateError, IntegrateResult as Result};
use scirs2_core::ndarray::Array2;

use super::super::core::{FluidBoundaryCondition, FluidState, NavierStokesParams};
use super::boundary_handling::{apply_boundary_conditions_2d, apply_pressure_boundary_conditions};

/// Navier-Stokes solver for incompressible flow
#[derive(Debug, Clone)]
pub struct NavierStokesSolver {
    /// Solver parameters
    pub params: NavierStokesParams,
    /// Boundary conditions for each edge (left, right, top, bottom)
    pub bc_x: (FluidBoundaryCondition, FluidBoundaryCondition),
    pub bc_y: (FluidBoundaryCondition, FluidBoundaryCondition),
}

impl NavierStokesSolver {
    /// Create a new Navier-Stokes solver
    pub fn new(
        params: NavierStokesParams,
        bc_x: (FluidBoundaryCondition, FluidBoundaryCondition),
        bc_y: (FluidBoundaryCondition, FluidBoundaryCondition),
    ) -> Self {
        Self { params, bc_x, bc_y }
    }

    /// Solve 2D incompressible Navier-Stokes using projection method
    pub fn solve_2d(
        &self,
        initial_state: FluidState,
        t_final: f64,
        save_interval: usize,
    ) -> Result<Vec<FluidState>> {
        let mut states = vec![initial_state.clone()];
        let mut current = initial_state;
        let n_steps = (t_final / self.params.dt).ceil() as usize;

        for step in 0..n_steps {
            // Step 1: Compute intermediate velocity (without pressure gradient)
            let (u_star, v_star) = self.compute_intermediate_velocity_2d(&current)?;

            // Step 2: Solve pressure Poisson equation
            let pressure = self.solve_pressure_poisson_2d(&u_star, &v_star, &current)?;

            // Step 3: Correct velocity with pressure gradient
            let (u_new, v_new) =
                self.correct_velocity_2d(&u_star, &v_star, &pressure, current.dx, current.dy)?;

            // Update state
            current.velocity = vec![u_new, v_new];
            current.pressure = pressure;
            current.time += self.params.dt;

            // Save state at intervals
            if (step + 1) % save_interval == 0 {
                states.push(current.clone());
            }
        }

        Ok(states)
    }

    /// Perform a single solving step (useful for interactive simulations)
    pub fn step(&self, state: &mut FluidState) -> Result<()> {
        // Step 1: Compute intermediate velocity (without pressure gradient)
        let (u_star, v_star) = self.compute_intermediate_velocity_2d(state)?;

        // Step 2: Solve pressure Poisson equation
        let pressure = self.solve_pressure_poisson_2d(&u_star, &v_star, state)?;

        // Step 3: Correct velocity with pressure gradient
        let (u_new, v_new) =
            self.correct_velocity_2d(&u_star, &v_star, &pressure, state.dx, state.dy)?;

        // Update state
        state.velocity = vec![u_new, v_new];
        state.pressure = pressure;
        state.time += self.params.dt;

        Ok(())
    }

    /// Compute intermediate velocity (advection + diffusion)
    fn compute_intermediate_velocity_2d(
        &self,
        state: &FluidState,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let u = &state.velocity[0];
        let v = &state.velocity[1];

        let mut u_star = u.clone();
        let mut v_star = v.clone();

        if self.params.semi_lagrangian {
            // Semi-Lagrangian advection
            self.semi_lagrangian_advection_2d(u, v, &mut u_star, &mut v_star, state.dx, state.dy)?;
        } else {
            // Standard advection using upwind or central differences
            self.standard_advection_2d(u, v, &mut u_star, &mut v_star, state.dx, state.dy)?;
        }

        // Add diffusion term
        self.add_diffusion_2d(&mut u_star, &mut v_star, u, v, state.dx, state.dy)?;

        // Apply boundary conditions
        apply_boundary_conditions_2d(&mut u_star, &mut v_star, self.bc_x, self.bc_y)?;

        Ok((u_star, v_star))
    }

    /// Semi-Lagrangian advection
    fn semi_lagrangian_advection_2d(
        &self,
        u: &Array2<f64>,
        v: &Array2<f64>,
        u_new: &mut Array2<f64>,
        v_new: &mut Array2<f64>,
        dx: f64,
        dy: f64,
    ) -> Result<()> {
        let (ny, nx) = u.dim();

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                // Backtrack particle position
                let x_back = i as f64 - u[[j, i]] * self.params.dt / dx;
                let y_back = j as f64 - v[[j, i]] * self.params.dt / dy;

                // Interpolate velocities at backtracked position
                u_new[[j, i]] = Self::bilinear_interpolate(u, x_back, y_back)?;
                v_new[[j, i]] = Self::bilinear_interpolate(v, x_back, y_back)?;
            }
        }

        Ok(())
    }

    /// Standard advection using upwind scheme
    fn standard_advection_2d(
        &self,
        u: &Array2<f64>,
        v: &Array2<f64>,
        u_new: &mut Array2<f64>,
        v_new: &mut Array2<f64>,
        dx: f64,
        dy: f64,
    ) -> Result<()> {
        let (ny, nx) = u.dim();
        let dt = self.params.dt;

        // Parallel computation for better performance
        let indices: Vec<(usize, usize)> = (1..ny - 1)
            .flat_map(|j| (1..nx - 1).map(move |i| (j, i)))
            .collect();

        let u_updates: Vec<((usize, usize), f64)> = indices
            .iter()
            .map(|&(j, i)| {
                // u momentum equation: ∂u/∂t + u∂u/∂x + v∂u/∂y = 0
                let u_val = u[[j, i]];
                let v_val = v[[j, i]];

                // Upwind scheme for u∂u/∂x
                let du_dx = if u_val > 0.0 {
                    (u[[j, i]] - u[[j, i - 1]]) / dx
                } else {
                    (u[[j, i + 1]] - u[[j, i]]) / dx
                };

                // Upwind scheme for v∂u/∂y
                let du_dy = if v_val > 0.0 {
                    (u[[j, i]] - u[[j - 1, i]]) / dy
                } else {
                    (u[[j + 1, i]] - u[[j, i]]) / dy
                };

                let u_update = u[[j, i]] - dt * (u_val * du_dx + v_val * du_dy);
                ((j, i), u_update)
            })
            .collect();

        let v_updates: Vec<((usize, usize), f64)> = indices
            .iter()
            .map(|&(j, i)| {
                // v momentum equation: ∂v/∂t + u∂v/∂x + v∂v/∂y = 0
                let u_val = u[[j, i]];
                let v_val = v[[j, i]];

                // Upwind scheme for u∂v/∂x
                let dv_dx = if u_val > 0.0 {
                    (v[[j, i]] - v[[j, i - 1]]) / dx
                } else {
                    (v[[j, i + 1]] - v[[j, i]]) / dx
                };

                // Upwind scheme for v∂v/∂y
                let dv_dy = if v_val > 0.0 {
                    (v[[j, i]] - v[[j - 1, i]]) / dy
                } else {
                    (v[[j + 1, i]] - v[[j, i]]) / dy
                };

                let v_update = v[[j, i]] - dt * (u_val * dv_dx + v_val * dv_dy);
                ((j, i), v_update)
            })
            .collect();

        // Apply updates
        for ((j, i), val) in u_updates {
            u_new[[j, i]] = val;
        }
        for ((j, i), val) in v_updates {
            v_new[[j, i]] = val;
        }

        Ok(())
    }

    /// Add diffusion term
    fn add_diffusion_2d(
        &self,
        u_new: &mut Array2<f64>,
        v_new: &mut Array2<f64>,
        u: &Array2<f64>,
        v: &Array2<f64>,
        dx: f64,
        dy: f64,
    ) -> Result<()> {
        let (ny, nx) = u.dim();
        let nu_dt = self.params.nu * self.params.dt;

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                // Laplacian of u
                let d2u_dx2 = (u[[j, i + 1]] - 2.0 * u[[j, i]] + u[[j, i - 1]]) / (dx * dx);
                let d2u_dy2 = (u[[j + 1, i]] - 2.0 * u[[j, i]] + u[[j - 1, i]]) / (dy * dy);
                u_new[[j, i]] += nu_dt * (d2u_dx2 + d2u_dy2);

                // Laplacian of v
                let d2v_dx2 = (v[[j, i + 1]] - 2.0 * v[[j, i]] + v[[j, i - 1]]) / (dx * dx);
                let d2v_dy2 = (v[[j + 1, i]] - 2.0 * v[[j, i]] + v[[j - 1, i]]) / (dy * dy);
                v_new[[j, i]] += nu_dt * (d2v_dx2 + d2v_dy2);
            }
        }

        Ok(())
    }

    /// Solve pressure Poisson equation: ∇²p = -ρ/Δt ∇·u*
    fn solve_pressure_poisson_2d(
        &self,
        u_star: &Array2<f64>,
        v_star: &Array2<f64>,
        state: &FluidState,
    ) -> Result<Array2<f64>> {
        let (ny, nx) = u_star.dim();
        let mut pressure = Array2::zeros((ny, nx));
        let mut rhs = Array2::zeros((ny, nx));

        // Compute divergence of intermediate velocity
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let div_u = (u_star[[j, i + 1]] - u_star[[j, i - 1]]) / (2.0 * state.dx)
                    + (v_star[[j + 1, i]] - v_star[[j - 1, i]]) / (2.0 * state.dy);
                rhs[[j, i]] = -self.params.rho * div_u / self.params.dt;
            }
        }

        // Solve using Jacobi iteration
        let mut pressure_new = pressure.clone();
        for _ in 0..self.params.max_pressure_iter {
            let mut max_diff: f64 = 0.0;

            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let p_new: f64 = ((pressure[[j, i + 1]] + pressure[[j, i - 1]])
                        / (state.dx * state.dx)
                        + (pressure[[j + 1, i]] + pressure[[j - 1, i]]) / (state.dy * state.dy)
                        - rhs[[j, i]])
                        / (2.0 / (state.dx * state.dx) + 2.0 / (state.dy * state.dy));

                    let diff = f64::abs(p_new - pressure[[j, i]]);
                    max_diff = max_diff.max(diff);
                    pressure_new[[j, i]] = p_new;
                }
            }

            pressure.assign(&pressure_new);

            // Apply pressure boundary conditions
            apply_pressure_boundary_conditions(&mut pressure)?;

            if max_diff < self.params.pressure_tol {
                break;
            }
        }

        Ok(pressure)
    }

    /// Correct velocity with pressure gradient
    fn correct_velocity_2d(
        &self,
        u_star: &Array2<f64>,
        v_star: &Array2<f64>,
        pressure: &Array2<f64>,
        dx: f64,
        dy: f64,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (ny, nx) = u_star.dim();
        let mut u_new = u_star.clone();
        let mut v_new = v_star.clone();

        let dt_over_rho = self.params.dt / self.params.rho;

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                // u^{n+1} = u* - Δt/ρ ∂p/∂x
                let dp_dx = (pressure[[j, i + 1]] - pressure[[j, i - 1]]) / (2.0 * dx);
                u_new[[j, i]] = u_star[[j, i]] - dt_over_rho * dp_dx;

                // v^{n+1} = v* - Δt/ρ ∂p/∂y
                let dp_dy = (pressure[[j + 1, i]] - pressure[[j - 1, i]]) / (2.0 * dy);
                v_new[[j, i]] = v_star[[j, i]] - dt_over_rho * dp_dy;
            }
        }

        Ok((u_new, v_new))
    }

    /// Bilinear interpolation for semi-Lagrangian advection
    fn bilinear_interpolate(field: &Array2<f64>, x: f64, y: f64) -> Result<f64> {
        let (ny, nx) = field.dim();

        // Clamp to domain
        let x = x.max(0.0).min((nx - 1) as f64);
        let y = y.max(0.0).min((ny - 1) as f64);

        let i = x.floor() as usize;
        let j = y.floor() as usize;
        let fx = x - i as f64;
        let fy = y - j as f64;

        // Ensure we don't go out of bounds
        let i = i.min(nx - 2);
        let j = j.min(ny - 2);

        Ok(field[[j, i]] * (1.0 - fx) * (1.0 - fy)
            + field[[j, i + 1]] * fx * (1.0 - fy)
            + field[[j + 1, i]] * (1.0 - fx) * fy
            + field[[j + 1, i + 1]] * fx * fy)
    }
}
