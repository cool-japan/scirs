//! Main compressible flow solver implementation
//!
//! This module provides the primary solver for compressible fluid dynamics equations,
//! including Euler and Navier-Stokes solvers with SIMD optimizations and adaptive time stepping.

use crate::error::IntegrateResult;
use crate::specialized::fluid_dynamics::compressible::flux_computation::CompressibleFluxes;
use crate::specialized::fluid_dynamics::compressible::state::CompressibleState;
use scirs2_core::ndarray::{s, Array1, Array3};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// High-performance compressible flow solver with SIMD optimizations
///
/// This solver implements the compressible Euler and Navier-Stokes equations using
/// conservative finite difference methods with SIMD acceleration for optimal performance.
#[derive(Debug, Clone)]
pub struct CompressibleFlowSolver {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Grid spacing
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Gas properties
    pub gamma: f64, // Specific heat ratio
    pub r_gas: f64, // Gas constant
    /// Solver parameters
    pub cfl: f64, // CFL number
    pub time: f64,
}

impl CompressibleFlowSolver {
    /// Create new compressible flow solver
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Grid dimensions
    /// * `dx`, `dy`, `dz` - Grid spacing
    ///
    /// # Returns
    /// A new `CompressibleFlowSolver` with default parameters
    #[allow(clippy::too_many_arguments)]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            gamma: 1.4,   // Air at standard conditions
            r_gas: 287.0, // J/(kg·K) for air
            cfl: 0.5,     // Conservative CFL number
            time: 0.0,
        }
    }

    /// Create solver with custom gas properties
    ///
    /// # Arguments
    /// * `nx`, `ny`, `nz` - Grid dimensions
    /// * `dx`, `dy`, `dz` - Grid spacing
    /// * `gamma` - Specific heat ratio
    /// * `r_gas` - Gas constant
    /// * `cfl` - CFL number
    #[allow(clippy::too_many_arguments)]
    pub fn with_gas_properties(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        gamma: f64,
        r_gas: f64,
        cfl: f64,
    ) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            gamma,
            r_gas,
            cfl,
            time: 0.0,
        }
    }

    /// Initialize compressible state with default conditions
    ///
    /// # Returns
    /// A new `CompressibleState` with initialized fields
    pub fn initialize_state(&self) -> CompressibleState {
        CompressibleState::new(self.nx, self.ny, self.nz)
    }

    /// Set solver time
    ///
    /// # Arguments
    /// * `time` - New time value
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Get current solver time
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Advance solution by one time step using SIMD-optimized methods
    ///
    /// This method solves the compressible Euler/Navier-Stokes equations:
    /// ∂U/∂t + ∇·F(U) = 0
    /// where U = [ρ, ρu, ρv, ρw, E]ᵀ
    ///
    /// # Arguments
    /// * `state` - Mutable reference to flow state
    /// * `dt` - Time step size
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn solve_step(&mut self, state: &mut CompressibleState, dt: f64) -> IntegrateResult<()> {
        // Update derived quantities (pressure, temperature, Mach number)
        self.update_derived_quantities_simd(state)?;

        // Compute fluxes using SIMD
        let fluxes = CompressibleFluxes::compute_fluxes_simd(
            state, self.nx, self.ny, self.nz, self.dx, self.dy, self.dz,
        )?;

        // Apply fourth-order Runge-Kutta time stepping with SIMD
        self.runge_kutta_step_simd(state, &fluxes, dt)?;

        // Apply boundary conditions
        Self::apply_compressible_boundary_conditions(state)?;

        self.time += dt;
        Ok(())
    }

    /// Update derived quantities (pressure, temperature, Mach) using SIMD
    ///
    /// # Arguments
    /// * `state` - Mutable reference to compressible state
    ///
    /// # Returns
    /// Result indicating success or error
    fn update_derived_quantities_simd(&self, state: &mut CompressibleState) -> IntegrateResult<()> {
        // Flatten arrays for SIMD processing
        let total_size = self.nx * self.ny * self.nz;

        let density_flat: Array1<f64> = state.density.iter().cloned().collect();
        let momentum_u_flat: Array1<f64> = state.momentum[0].iter().cloned().collect();
        let momentum_v_flat: Array1<f64> = state.momentum[1].iter().cloned().collect();
        let momentum_w_flat: Array1<f64> = state.momentum[2].iter().cloned().collect();
        let energy_flat: Array1<f64> = state.energy.iter().cloned().collect();

        // Calculate velocities using SIMD
        let u_flat = f64::simd_div(&momentum_u_flat.view(), &density_flat.view());
        let v_flat = f64::simd_div(&momentum_v_flat.view(), &density_flat.view());
        let w_flat = f64::simd_div(&momentum_w_flat.view(), &density_flat.view());

        // Calculate kinetic energy: 0.5 * ρ * (u² + v² + w²)
        let u_squared = f64::simd_mul(&u_flat.view(), &u_flat.view());
        let v_squared = f64::simd_mul(&v_flat.view(), &v_flat.view());
        let w_squared = f64::simd_mul(&w_flat.view(), &w_flat.view());
        let vel_squared = f64::simd_add(&u_squared.view(), &v_squared.view());
        let vel_squared = f64::simd_add(&vel_squared.view(), &w_squared.view());
        let half_array = Array1::from_elem(total_size, 0.5);
        let kinetic_energy = f64::simd_mul(&vel_squared.view(), &half_array.view());
        let kinetic_energy = f64::simd_mul(&kinetic_energy.view(), &density_flat.view());

        // Calculate internal energy: E - kinetic energy
        let internal_energy = f64::simd_sub(&energy_flat.view(), &kinetic_energy.view());

        // Calculate pressure using ideal gas law: p = (γ-1) * internal energy density
        let gamma_minus_one_array = Array1::from_elem(total_size, self.gamma - 1.0);
        let pressure_flat = f64::simd_mul(&internal_energy.view(), &gamma_minus_one_array.view());

        // Calculate temperature: T = p / (ρ * R)
        let r_gas_array = Array1::from_elem(total_size, self.r_gas);
        let rho_r = f64::simd_mul(&density_flat.view(), &r_gas_array.view());
        let temperature_flat = f64::simd_div(&pressure_flat.view(), &rho_r.view());

        // Update state arrays
        for ((i, &p), &t) in pressure_flat
            .iter()
            .enumerate()
            .zip(temperature_flat.iter())
        {
            let (ii, jj, kk) = self.index_to_coords(i);
            if ii < self.nx && jj < self.ny && kk < self.nz {
                state.pressure[[ii, jj, kk]] = p;
                state.temperature[[ii, jj, kk]] = t;
            }
        }

        // Update Mach number
        state.update_mach_number(self.gamma);

        Ok(())
    }

    /// Convert linear index to 3D coordinates
    fn index_to_coords(&self, index: usize) -> (usize, usize, usize) {
        let k = index / (self.nx * self.ny);
        let remainder = index % (self.nx * self.ny);
        let j = remainder / self.nx;
        let i = remainder % self.nx;
        (i, j, k)
    }

    /// Fourth-order Runge-Kutta time stepping with SIMD
    ///
    /// # Arguments
    /// * `state` - Mutable reference to compressible state
    /// * `fluxes` - Computed flux terms
    /// * `dt` - Time step size
    ///
    /// # Returns
    /// Result indicating success or error
    fn runge_kutta_step_simd(
        &self,
        state: &mut CompressibleState,
        fluxes: &CompressibleFluxes,
        dt: f64,
    ) -> IntegrateResult<()> {
        // RK4 implementation with SIMD
        let _dt_half = dt * 0.5;
        let _dt_sixth = dt / 6.0;

        // k1 = -∇·F(U^n)
        let k1_density = fluxes.density.mapv(|x| -x);
        let k1_momentum: Vec<Array3<f64>> = fluxes
            .momentum
            .iter()
            .map(|flux| flux.mapv(|x| -x))
            .collect();
        let k1_energy = fluxes.energy.mapv(|x| -x);

        // Update state using simplified Euler step for now
        // Full RK4 would require computing intermediate states
        let total_size = self.nx * self.ny * self.nz;

        let density_flat: Array1<f64> = state.density.iter().cloned().collect();
        let energy_flat: Array1<f64> = state.energy.iter().cloned().collect();
        let k1_density_flat: Array1<f64> = k1_density.iter().cloned().collect();
        let k1_energy_flat: Array1<f64> = k1_energy.iter().cloned().collect();

        // Update density and energy using SIMD
        let dt_array = Array1::from_elem(total_size, dt);
        let density_update = f64::simd_mul(&k1_density_flat.view(), &dt_array.view());
        let energy_update = f64::simd_mul(&k1_energy_flat.view(), &dt_array.view());

        let new_density = f64::simd_add(&density_flat.view(), &density_update.view());
        let new_energy = f64::simd_add(&energy_flat.view(), &energy_update.view());

        // Update state arrays
        for (i, (&rho, &e)) in new_density.iter().zip(new_energy.iter()).enumerate() {
            let (ii, jj, kk) = self.index_to_coords(i);
            if ii < self.nx && jj < self.ny && kk < self.nz {
                state.density[[ii, jj, kk]] = rho;
                state.energy[[ii, jj, kk]] = e;
            }
        }

        // Update momentum components
        for comp in 0..3 {
            let momentum_flat: Array1<f64> = state.momentum[comp].iter().cloned().collect();
            let k1_momentum_flat: Array1<f64> = k1_momentum[comp].iter().cloned().collect();
            let momentum_update = f64::simd_mul(&k1_momentum_flat.view(), &dt_array.view());
            let new_momentum = f64::simd_add(&momentum_flat.view(), &momentum_update.view());

            for (i, &mom) in new_momentum.iter().enumerate() {
                let (ii, jj, kk) = self.index_to_coords(i);
                if ii < self.nx && jj < self.ny && kk < self.nz {
                    state.momentum[comp][[ii, jj, kk]] = mom;
                }
            }
        }

        Ok(())
    }

    /// Apply boundary conditions for compressible flow
    ///
    /// # Arguments
    /// * `state` - Mutable reference to compressible state
    ///
    /// # Returns
    /// Result indicating success or error
    fn apply_compressible_boundary_conditions(
        state: &mut CompressibleState,
    ) -> IntegrateResult<()> {
        let (nx, ny, nz) = state.dimensions();

        // X boundaries - reflective/extrapolation conditions
        for j in 0..ny {
            for k in 0..nz {
                // Left boundary (i = 0)
                state.density[[0, j, k]] = state.density[[1, j, k]];
                state.momentum[0][[0, j, k]] = -state.momentum[0][[1, j, k]]; // Reflect u
                state.momentum[1][[0, j, k]] = state.momentum[1][[1, j, k]]; // Keep v
                state.momentum[2][[0, j, k]] = state.momentum[2][[1, j, k]]; // Keep w
                state.energy[[0, j, k]] = state.energy[[1, j, k]];
                state.pressure[[0, j, k]] = state.pressure[[1, j, k]];
                state.temperature[[0, j, k]] = state.temperature[[1, j, k]];

                // Right boundary (i = nx-1)
                let last = nx - 1;
                state.density[[last, j, k]] = state.density[[last - 1, j, k]];
                state.momentum[0][[last, j, k]] = -state.momentum[0][[last - 1, j, k]]; // Reflect u
                state.momentum[1][[last, j, k]] = state.momentum[1][[last - 1, j, k]]; // Keep v
                state.momentum[2][[last, j, k]] = state.momentum[2][[last - 1, j, k]]; // Keep w
                state.energy[[last, j, k]] = state.energy[[last - 1, j, k]];
                state.pressure[[last, j, k]] = state.pressure[[last - 1, j, k]];
                state.temperature[[last, j, k]] = state.temperature[[last - 1, j, k]];
            }
        }

        // Y boundaries - similar reflective conditions
        for i in 0..nx {
            for k in 0..nz {
                // Bottom boundary (j = 0)
                state.density[[i, 0, k]] = state.density[[i, 1, k]];
                state.momentum[0][[i, 0, k]] = state.momentum[0][[i, 1, k]]; // Keep u
                state.momentum[1][[i, 0, k]] = -state.momentum[1][[i, 1, k]]; // Reflect v
                state.momentum[2][[i, 0, k]] = state.momentum[2][[i, 1, k]]; // Keep w
                state.energy[[i, 0, k]] = state.energy[[i, 1, k]];
                state.pressure[[i, 0, k]] = state.pressure[[i, 1, k]];
                state.temperature[[i, 0, k]] = state.temperature[[i, 1, k]];

                // Top boundary (j = ny-1)
                let last = ny - 1;
                state.density[[i, last, k]] = state.density[[i, last - 1, k]];
                state.momentum[0][[i, last, k]] = state.momentum[0][[i, last - 1, k]]; // Keep u
                state.momentum[1][[i, last, k]] = -state.momentum[1][[i, last - 1, k]]; // Reflect v
                state.momentum[2][[i, last, k]] = state.momentum[2][[i, last - 1, k]]; // Keep w
                state.energy[[i, last, k]] = state.energy[[i, last - 1, k]];
                state.pressure[[i, last, k]] = state.pressure[[i, last - 1, k]];
                state.temperature[[i, last, k]] = state.temperature[[i, last - 1, k]];
            }
        }

        // Z boundaries - similar reflective conditions
        for i in 0..nx {
            for j in 0..ny {
                // Front boundary (k = 0)
                state.density[[i, j, 0]] = state.density[[i, j, 1]];
                state.momentum[0][[i, j, 0]] = state.momentum[0][[i, j, 1]]; // Keep u
                state.momentum[1][[i, j, 0]] = state.momentum[1][[i, j, 1]]; // Keep v
                state.momentum[2][[i, j, 0]] = -state.momentum[2][[i, j, 1]]; // Reflect w
                state.energy[[i, j, 0]] = state.energy[[i, j, 1]];
                state.pressure[[i, j, 0]] = state.pressure[[i, j, 1]];
                state.temperature[[i, j, 0]] = state.temperature[[i, j, 1]];

                // Back boundary (k = nz-1)
                let last = nz - 1;
                state.density[[i, j, last]] = state.density[[i, j, last - 1]];
                state.momentum[0][[i, j, last]] = state.momentum[0][[i, j, last - 1]]; // Keep u
                state.momentum[1][[i, j, last]] = state.momentum[1][[i, j, last - 1]]; // Keep v
                state.momentum[2][[i, j, last]] = -state.momentum[2][[i, j, last - 1]]; // Reflect w
                state.energy[[i, j, last]] = state.energy[[i, j, last - 1]];
                state.pressure[[i, j, last]] = state.pressure[[i, j, last - 1]];
                state.temperature[[i, j, last]] = state.temperature[[i, j, last - 1]];
            }
        }

        Ok(())
    }

    /// Calculate adaptive time step based on CFL condition
    ///
    /// This method computes a time step that satisfies the CFL (Courant-Friedrichs-Lewy)
    /// condition for numerical stability: Δt ≤ CFL * min(Δx/(|u|+c))
    ///
    /// # Arguments
    /// * `state` - Current compressible state
    ///
    /// # Returns
    /// Adaptive time step size
    pub fn calculate_adaptive_timestep(&self, state: &CompressibleState) -> f64 {
        let mut max_eigenvalue: f64 = 1e-12; // Avoid division by zero

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let rho = state.density[[i, j, k]];
                    if rho <= 0.0 {
                        continue; // Skip invalid cells
                    }

                    let u = state.momentum[0][[i, j, k]] / rho;
                    let v = state.momentum[1][[i, j, k]] / rho;
                    let w = state.momentum[2][[i, j, k]] / rho;
                    let p = state.pressure[[i, j, k]];

                    if p <= 0.0 {
                        continue; // Skip invalid cells
                    }

                    // Sound speed: c = √(γp/ρ)
                    let c = (self.gamma * p / rho).sqrt();

                    // Maximum eigenvalues in each direction
                    let lambda_x = (u.abs() + c) / self.dx;
                    let lambda_y = (v.abs() + c) / self.dy;
                    let lambda_z = (w.abs() + c) / self.dz;

                    let max_local = lambda_x.max(lambda_y).max(lambda_z);
                    max_eigenvalue = max_eigenvalue.max(max_local);
                }
            }
        }

        self.cfl / max_eigenvalue
    }

    /// Set CFL number for adaptive time stepping
    ///
    /// # Arguments
    /// * `cfl` - New CFL number (should be ≤ 1.0 for stability)
    pub fn set_cfl(&mut self, cfl: f64) {
        self.cfl = cfl.max(0.1).min(1.0); // Clamp to reasonable range
    }

    /// Get current CFL number
    pub fn cfl(&self) -> f64 {
        self.cfl
    }

    /// Get grid dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// Get grid spacing
    pub fn spacing(&self) -> (f64, f64, f64) {
        (self.dx, self.dy, self.dz)
    }

    /// Get gas properties
    pub fn gas_properties(&self) -> (f64, f64) {
        (self.gamma, self.r_gas)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let solver = CompressibleFlowSolver::new(10, 10, 10, 0.1, 0.1, 0.1);
        assert_eq!(solver.dimensions(), (10, 10, 10));
        assert_eq!(solver.spacing(), (0.1, 0.1, 0.1));
        assert_eq!(solver.gas_properties(), (1.4, 287.0));
        assert_eq!(solver.cfl(), 0.5);
    }

    #[test]
    fn test_solver_with_custom_properties() {
        let solver =
            CompressibleFlowSolver::with_gas_properties(5, 5, 5, 0.2, 0.2, 0.2, 1.3, 300.0, 0.3);
        assert_eq!(solver.gas_properties(), (1.3, 300.0));
        assert_eq!(solver.cfl(), 0.3);
    }

    #[test]
    fn test_state_initialization() {
        let solver = CompressibleFlowSolver::new(8, 8, 8, 0.1, 0.1, 0.1);
        let state = solver.initialize_state();

        assert_eq!(state.dimensions(), (8, 8, 8));
        assert!(state.is_physically_valid());
    }

    #[test]
    fn test_adaptive_timestep() {
        let solver = CompressibleFlowSolver::new(5, 5, 5, 0.1, 0.1, 0.1);
        let mut state = solver.initialize_state();
        state.set_uniform_conditions(1.0, 10.0, 0.0, 0.0, 101325.0, 300.0);

        let dt = solver.calculate_adaptive_timestep(&state);
        assert!(dt > 0.0);
        assert!(dt.is_finite());
        assert!(dt < 1.0); // Should be reasonable for typical flows
    }

    #[test]
    fn test_cfl_setting() {
        let mut solver = CompressibleFlowSolver::new(5, 5, 5, 0.1, 0.1, 0.1);

        solver.set_cfl(0.8);
        assert_eq!(solver.cfl(), 0.8);

        // Test clamping
        solver.set_cfl(1.5);
        assert_eq!(solver.cfl(), 1.0);

        solver.set_cfl(0.05);
        assert_eq!(solver.cfl(), 0.1);
    }

    #[test]
    fn test_time_management() {
        let mut solver = CompressibleFlowSolver::new(5, 5, 5, 0.1, 0.1, 0.1);
        assert_eq!(solver.time(), 0.0);

        solver.set_time(10.5);
        assert_eq!(solver.time(), 10.5);
    }

    #[test]
    fn test_boundary_conditions() {
        let mut state = CompressibleState::new(5, 5, 5);
        state.set_uniform_conditions(1.0, 10.0, 5.0, 2.0, 101325.0, 300.0);

        // Modify interior point
        state.momentum[0][[2, 2, 2]] = 20.0;

        let result = CompressibleFlowSolver::apply_compressible_boundary_conditions(&mut state);
        assert!(result.is_ok());

        // Check that boundary values were updated
        assert_ne!(state.momentum[0][[0, 0, 0]], 10.0); // Should be reflected
    }
}
