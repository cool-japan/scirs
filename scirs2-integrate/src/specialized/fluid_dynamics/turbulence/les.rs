//! Large Eddy Simulation (LES) implementation
//!
//! This module provides LES solvers and subgrid-scale (SGS) models for
//! turbulence simulation using the filtered Navier-Stokes equations.

use super::models::SGSModel as SGSModelTrait;
use crate::error::IntegrateResult;
use scirs2_core::ndarray::{Array2, Array3, Array4};

/// Subgrid-scale models for LES
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SGSModel {
    /// Smagorinsky model
    Smagorinsky,
    /// Dynamic Smagorinsky model
    DynamicSmagorinsky,
    /// Wall-Adapting Local Eddy-viscosity (WALE) model
    WALE,
    /// Vreman model
    Vreman,
}

/// Large Eddy Simulation solver
#[derive(Debug, Clone)]
pub struct LESolver {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Grid spacing
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    /// Subgrid-scale model
    pub sgs_model: SGSModel,
    /// Filter width ratio
    pub filter_ratio: f64,
    /// Smagorinsky constant
    pub cs: f64,
}

/// 3D fluid state for LES
#[derive(Debug, Clone)]
pub struct FluidState3D {
    /// Velocity components [u, v, w]
    pub velocity: Vec<Array3<f64>>,
    /// Pressure field
    pub pressure: Array3<f64>,
    /// Grid spacing
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl LESolver {
    /// Create a new LES solver
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        sgs_model: SGSModel,
    ) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            sgs_model,
            filter_ratio: 2.0,
            cs: 0.1, // Typical Smagorinsky constant
        }
    }

    /// Solve 3D LES equations
    pub fn solve_3d(
        &self,
        initial_state: FluidState3D,
        final_time: f64,
        n_steps: usize,
    ) -> IntegrateResult<Vec<FluidState3D>> {
        let dt = final_time / n_steps as f64;
        let mut state = initial_state;
        let mut results = Vec::with_capacity(n_steps + 1);
        results.push(state.clone());

        for _step in 0..n_steps {
            // Compute SGS stress tensor
            let sgs_stress = self.compute_sgs_stress(&state)?;

            // Update velocity using filtered Navier-Stokes equations
            state = self.update_velocity_3d(&state, &sgs_stress, dt)?;

            // Apply boundary conditions
            state = Self::apply_boundary_conditions_3d(state)?;

            // Store result
            results.push(state.clone());
        }

        Ok(results)
    }

    /// Compute subgrid-scale stress tensor
    pub fn compute_sgs_stress(&self, state: &FluidState3D) -> IntegrateResult<Array4<f64>> {
        let mut sgs_stress = Array4::zeros((3, 3, self.nx, self.ny));

        match self.sgs_model {
            SGSModel::Smagorinsky => {
                self.compute_smagorinsky_stress(&mut sgs_stress, state)?;
            }
            SGSModel::DynamicSmagorinsky => {
                self.compute_dynamic_smagorinsky_stress(&mut sgs_stress, state)?;
            }
            SGSModel::WALE => {
                self.compute_wale_stress(&mut sgs_stress, state)?;
            }
            SGSModel::Vreman => {
                self.compute_vreman_stress(&mut sgs_stress, state)?;
            }
        }

        Ok(sgs_stress)
    }

    /// Compute Smagorinsky SGS stress
    fn compute_smagorinsky_stress(
        &self,
        sgs_stress: &mut Array4<f64>,
        state: &FluidState3D,
    ) -> IntegrateResult<()> {
        let delta = (self.dx * self.dy * self.dz).powf(1.0 / 3.0); // Filter width

        // Compute strain rate tensor
        let strain_rate = self.compute_strain_rate_tensor_3d(state)?;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    // Compute magnitude of strain rate
                    let mut s_mag: f64 = 0.0;
                    for ii in 0..3 {
                        for jj in 0..3 {
                            s_mag +=
                                strain_rate[[ii, jj, i, j, k]] * strain_rate[[ii, jj, i, j, k]];
                        }
                    }
                    s_mag = (2.0 * s_mag).sqrt();

                    // Compute eddy viscosity
                    let nu_sgs = (self.cs * delta).powi(2) * s_mag;

                    // Compute SGS stress tensor
                    for ii in 0..3 {
                        for jj in 0..3 {
                            sgs_stress[[ii, jj, i, j]] =
                                -2.0 * nu_sgs * strain_rate[[ii, jj, i, j, k]];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute dynamic Smagorinsky SGS stress
    fn compute_dynamic_smagorinsky_stress(
        &self,
        sgs_stress: &mut Array4<f64>,
        state: &FluidState3D,
    ) -> IntegrateResult<()> {
        // Dynamic procedure to compute Smagorinsky coefficient
        let test_filter_ratio = 2.0;
        let delta = (self.dx * self.dy * self.dz).powf(1.0 / 3.0);
        let delta_test = test_filter_ratio * delta;

        // Apply test filter to velocity field
        let filtered_velocity = self.apply_test_filter_3d(&state.velocity)?;

        // Compute Leonard stress (resolved stress)
        let leonard_stress = self.compute_leonard_stress_3d(state, &filtered_velocity)?;

        // Compute strain rate for filtered field
        let filtered_state = FluidState3D {
            velocity: filtered_velocity,
            pressure: state.pressure.clone(),
            dx: state.dx,
            dy: state.dy,
            dz: state.dz,
        };
        let filtered_strain = self.compute_strain_rate_tensor_3d(&filtered_state)?;

        // Original strain rate
        let strain_rate = self.compute_strain_rate_tensor_3d(state)?;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    // Compute dynamic coefficient using least-squares
                    let cs_dynamic = self.compute_dynamic_coefficient(
                        &leonard_stress,
                        &strain_rate,
                        &filtered_strain,
                        i,
                        j,
                        k,
                        delta,
                        delta_test,
                    )?;

                    // Compute magnitude of strain rate
                    let mut s_mag: f64 = 0.0;
                    for ii in 0..3 {
                        for jj in 0..3 {
                            s_mag +=
                                strain_rate[[ii, jj, i, j, k]] * strain_rate[[ii, jj, i, j, k]];
                        }
                    }
                    s_mag = (2.0 * s_mag).sqrt();

                    // Compute eddy viscosity with dynamic coefficient
                    let nu_sgs = (cs_dynamic * delta).powi(2) * s_mag;

                    // Compute SGS stress tensor
                    for ii in 0..3 {
                        for jj in 0..3 {
                            sgs_stress[[ii, jj, i, j]] =
                                -2.0 * nu_sgs * strain_rate[[ii, jj, i, j, k]];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute WALE SGS stress
    fn compute_wale_stress(
        &self,
        sgs_stress: &mut Array4<f64>,
        state: &FluidState3D,
    ) -> IntegrateResult<()> {
        let delta = (self.dx * self.dy * self.dz).powf(1.0 / 3.0);
        let cw = 0.5; // WALE constant

        // Compute velocity gradient tensor
        let grad_u = self.compute_velocity_gradient_3d(state)?;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    // Compute symmetric and antisymmetric parts
                    let mut s_d = Array2::zeros((3, 3)); // Traceless symmetric part
                    let mut omega = Array2::zeros((3, 3)); // Antisymmetric part

                    for ii in 0..3 {
                        for jj in 0..3 {
                            let grad_ij = grad_u[[ii, jj, i, j, k]];
                            let grad_ji = grad_u[[jj, ii, i, j, k]];

                            omega[[ii, jj]] = 0.5 * (grad_ij - grad_ji);
                            s_d[[ii, jj]] = 0.5 * (grad_ij + grad_ji);
                        }
                    }

                    // Remove trace from s_d
                    let trace = (s_d[[0, 0]] + s_d[[1, 1]] + s_d[[2, 2]]) / 3.0;
                    for ii in 0..3 {
                        s_d[[ii, ii]] -= trace;
                    }

                    // Compute invariants
                    let mut s_d_mag_sq: f64 = 0.0;
                    let mut omega_mag_sq: f64 = 0.0;

                    for ii in 0..3 {
                        for jj in 0..3 {
                            s_d_mag_sq += s_d[[ii, jj]] * s_d[[ii, jj]];
                            omega_mag_sq += omega[[ii, jj]] * omega[[ii, jj]];
                        }
                    }

                    // WALE eddy viscosity
                    let numerator = s_d_mag_sq.powf(1.5);
                    let denominator = (s_d_mag_sq.powf(2.5) + omega_mag_sq.powf(1.25)).max(1e-12);
                    let nu_sgs = (cw * delta).powi(2) * numerator / denominator;

                    // Compute SGS stress tensor
                    for ii in 0..3 {
                        for jj in 0..3 {
                            sgs_stress[[ii, jj, i, j]] = -2.0 * nu_sgs * s_d[[ii, jj]];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute Vreman SGS stress
    fn compute_vreman_stress(
        &self,
        sgs_stress: &mut Array4<f64>,
        state: &FluidState3D,
    ) -> IntegrateResult<()> {
        let delta = (self.dx * self.dy * self.dz).powf(1.0 / 3.0);
        let cv: f64 = 0.07; // Vreman constant

        // Compute velocity gradient tensor
        let grad_u = self.compute_velocity_gradient_3d(state)?;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    // Compute α and β tensors
                    let mut alpha = Array2::zeros((3, 3));
                    let mut beta: Array2<f64> = Array2::zeros((3, 3));

                    for ii in 0..3 {
                        for jj in 0..3 {
                            alpha[[ii, jj]] = grad_u[[ii, jj, i, j, k]];

                            for kk in 0..3 {
                                beta[[ii, jj]] +=
                                    grad_u[[ii, kk, i, j, k]] * grad_u[[jj, kk, i, j, k]];
                            }
                        }
                    }

                    // Compute Vreman invariants
                    let alpha_norm_sq = alpha.iter().map(|&x| x * x).sum::<f64>();

                    let b_beta = beta[[0, 0]] * beta[[1, 1]]
                        + beta[[1, 1]] * beta[[2, 2]]
                        + beta[[0, 0]] * beta[[2, 2]]
                        - beta[[0, 1]].powi(2)
                        - beta[[1, 2]].powi(2)
                        - beta[[0, 2]].powi(2);

                    // Vreman eddy viscosity
                    let nu_sgs = if alpha_norm_sq > 1e-12 {
                        cv.powi(2) * delta.powi(2) * (b_beta / alpha_norm_sq).sqrt()
                    } else {
                        0.0
                    };

                    // Compute strain rate tensor
                    for ii in 0..3 {
                        for jj in 0..3 {
                            let strain =
                                0.5 * (grad_u[[ii, jj, i, j, k]] + grad_u[[jj, ii, i, j, k]]);
                            sgs_stress[[ii, jj, i, j]] = -2.0 * nu_sgs * strain;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute strain rate tensor for 3D case
    fn compute_strain_rate_tensor_3d(
        &self,
        state: &FluidState3D,
    ) -> IntegrateResult<scirs2_core::ndarray::Array5<f64>> {
        let mut strain_rate =
            scirs2_core::ndarray::Array5::zeros((3, 3, self.nx, self.ny, self.nz));

        // Compute derivatives using central differences
        for i in 1..(self.nx - 1) {
            for j in 1..(self.ny - 1) {
                for k in 1..(self.nz - 1) {
                    // Velocity gradients
                    let dudx = (state.velocity[0][[i + 1, j, k]]
                        - state.velocity[0][[i - 1, j, k]])
                        / (2.0 * self.dx);
                    let dudy = (state.velocity[0][[i, j + 1, k]]
                        - state.velocity[0][[i, j - 1, k]])
                        / (2.0 * self.dy);
                    let dudz = (state.velocity[0][[i, j, k + 1]]
                        - state.velocity[0][[i, j, k - 1]])
                        / (2.0 * self.dz);

                    let dvdx = (state.velocity[1][[i + 1, j, k]]
                        - state.velocity[1][[i - 1, j, k]])
                        / (2.0 * self.dx);
                    let dvdy = (state.velocity[1][[i, j + 1, k]]
                        - state.velocity[1][[i, j - 1, k]])
                        / (2.0 * self.dy);
                    let dvdz = (state.velocity[1][[i, j, k + 1]]
                        - state.velocity[1][[i, j, k - 1]])
                        / (2.0 * self.dz);

                    let dwdx = (state.velocity[2][[i + 1, j, k]]
                        - state.velocity[2][[i - 1, j, k]])
                        / (2.0 * self.dx);
                    let dwdy = (state.velocity[2][[i, j + 1, k]]
                        - state.velocity[2][[i, j - 1, k]])
                        / (2.0 * self.dy);
                    let dwdz = (state.velocity[2][[i, j, k + 1]]
                        - state.velocity[2][[i, j, k - 1]])
                        / (2.0 * self.dz);

                    // Strain rate tensor components: S_ij = 0.5(∂u_i/∂x_j + ∂u_j/∂x_i)
                    strain_rate[[0, 0, i, j, k]] = dudx;
                    strain_rate[[1, 1, i, j, k]] = dvdy;
                    strain_rate[[2, 2, i, j, k]] = dwdz;

                    strain_rate[[0, 1, i, j, k]] = 0.5 * (dudy + dvdx);
                    strain_rate[[1, 0, i, j, k]] = strain_rate[[0, 1, i, j, k]];

                    strain_rate[[0, 2, i, j, k]] = 0.5 * (dudz + dwdx);
                    strain_rate[[2, 0, i, j, k]] = strain_rate[[0, 2, i, j, k]];

                    strain_rate[[1, 2, i, j, k]] = 0.5 * (dvdz + dwdy);
                    strain_rate[[2, 1, i, j, k]] = strain_rate[[1, 2, i, j, k]];
                }
            }
        }

        Ok(strain_rate)
    }

    /// Compute velocity gradient tensor
    fn compute_velocity_gradient_3d(
        &self,
        state: &FluidState3D,
    ) -> IntegrateResult<scirs2_core::ndarray::Array5<f64>> {
        let mut grad_u = scirs2_core::ndarray::Array5::zeros((3, 3, self.nx, self.ny, self.nz));

        // Compute derivatives using central differences
        for i in 1..(self.nx - 1) {
            for j in 1..(self.ny - 1) {
                for k in 1..(self.nz - 1) {
                    // Velocity gradients
                    grad_u[[0, 0, i, j, k]] = (state.velocity[0][[i + 1, j, k]]
                        - state.velocity[0][[i - 1, j, k]])
                        / (2.0 * self.dx);
                    grad_u[[0, 1, i, j, k]] = (state.velocity[0][[i, j + 1, k]]
                        - state.velocity[0][[i, j - 1, k]])
                        / (2.0 * self.dy);
                    grad_u[[0, 2, i, j, k]] = (state.velocity[0][[i, j, k + 1]]
                        - state.velocity[0][[i, j, k - 1]])
                        / (2.0 * self.dz);

                    grad_u[[1, 0, i, j, k]] = (state.velocity[1][[i + 1, j, k]]
                        - state.velocity[1][[i - 1, j, k]])
                        / (2.0 * self.dx);
                    grad_u[[1, 1, i, j, k]] = (state.velocity[1][[i, j + 1, k]]
                        - state.velocity[1][[i, j - 1, k]])
                        / (2.0 * self.dy);
                    grad_u[[1, 2, i, j, k]] = (state.velocity[1][[i, j, k + 1]]
                        - state.velocity[1][[i, j, k - 1]])
                        / (2.0 * self.dz);

                    grad_u[[2, 0, i, j, k]] = (state.velocity[2][[i + 1, j, k]]
                        - state.velocity[2][[i - 1, j, k]])
                        / (2.0 * self.dx);
                    grad_u[[2, 1, i, j, k]] = (state.velocity[2][[i, j + 1, k]]
                        - state.velocity[2][[i, j - 1, k]])
                        / (2.0 * self.dy);
                    grad_u[[2, 2, i, j, k]] = (state.velocity[2][[i, j, k + 1]]
                        - state.velocity[2][[i, j, k - 1]])
                        / (2.0 * self.dz);
                }
            }
        }

        Ok(grad_u)
    }

    /// Apply test filter to velocity field
    fn apply_test_filter_3d(&self, velocity: &[Array3<f64>]) -> IntegrateResult<Vec<Array3<f64>>> {
        let mut filtered = vec![Array3::zeros((self.nx, self.ny, self.nz)); 3];

        // Simple box filter
        let filter_width = 2;
        let filter_weight = 1.0 / (filter_width * filter_width * filter_width) as f64;

        for comp in 0..3 {
            for i in filter_width..(self.nx - filter_width) {
                for j in filter_width..(self.ny - filter_width) {
                    for k in filter_width..(self.nz - filter_width) {
                        let mut sum: f64 = 0.0;
                        for di in 0..filter_width {
                            for dj in 0..filter_width {
                                for dk in 0..filter_width {
                                    sum += velocity[comp][[
                                        i - filter_width / 2 + di,
                                        j - filter_width / 2 + dj,
                                        k - filter_width / 2 + dk,
                                    ]];
                                }
                            }
                        }
                        filtered[comp][[i, j, k]] = sum * filter_weight;
                    }
                }
            }
        }

        Ok(filtered)
    }

    /// Compute Leonard stress tensor
    fn compute_leonard_stress_3d(
        &self,
        state: &FluidState3D,
        filtered_velocity: &[Array3<f64>],
    ) -> IntegrateResult<Array4<f64>> {
        let mut leonard = Array4::zeros((3, 3, self.nx, self.ny));

        // Compute Leonard stress: L_ij = u_i * u_j - filtered(u_i) * filtered(u_j)
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    for ii in 0..3 {
                        for jj in 0..3 {
                            let unfiltered_product =
                                state.velocity[ii][[i, j, k]] * state.velocity[jj][[i, j, k]];
                            let filtered_product =
                                filtered_velocity[ii][[i, j, k]] * filtered_velocity[jj][[i, j, k]];
                            leonard[[ii, jj, i, j]] = unfiltered_product - filtered_product;
                        }
                    }
                }
            }
        }

        Ok(leonard)
    }

    /// Compute dynamic coefficient
    fn compute_dynamic_coefficient(
        &self,
        leonard: &Array4<f64>,
        strain_rate: &scirs2_core::ndarray::Array5<f64>,
        filtered_strain: &scirs2_core::ndarray::Array5<f64>,
        i: usize,
        j: usize,
        k: usize,
        delta: f64,
        delta_test: f64,
    ) -> IntegrateResult<f64> {
        // Simplified dynamic coefficient computation
        // In practice, this would involve spatial averaging and least-squares
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for ii in 0..3 {
            for jj in 0..3 {
                let l_ij = leonard[[ii, jj, i, j]];
                let s_ij = strain_rate[[ii, jj, i, j, k]];
                let s_test_ij = filtered_strain[[ii, jj, i, j, k]];

                let m_ij = delta.powi(2) * s_ij.abs() * s_ij
                    - delta_test.powi(2) * s_test_ij.abs() * s_test_ij;

                numerator += l_ij * m_ij;
                denominator += m_ij * m_ij;
            }
        }

        if denominator > 1e-12 {
            Ok((numerator / denominator).max(0.0).min(0.5)) // Clamp coefficient
        } else {
            Ok(self.cs) // Fall back to constant Smagorinsky
        }
    }

    /// Update velocity using LES equations
    fn update_velocity_3d(
        &self,
        state: &FluidState3D,
        sgs_stress: &Array4<f64>,
        dt: f64,
    ) -> IntegrateResult<FluidState3D> {
        let mut new_velocity = state.velocity.clone();

        // Simplified velocity update (in practice, would involve full Navier-Stokes discretization)
        for comp in 0..3 {
            for i in 1..self.nx - 1 {
                for j in 1..self.ny - 1 {
                    for k in 1..self.nz - 1 {
                        // Add SGS stress divergence term (simplified)
                        let mut sgs_divergence = 0.0;
                        for jj in 0..3 {
                            if jj == 0 && i > 0 && i < self.nx - 1 {
                                sgs_divergence += (sgs_stress[[comp, jj, i + 1, j]]
                                    - sgs_stress[[comp, jj, i - 1, j]])
                                    / (2.0 * self.dx);
                            }
                            if jj == 1 && j > 0 && j < self.ny - 1 {
                                sgs_divergence += (sgs_stress[[comp, jj, i, j + 1]]
                                    - sgs_stress[[comp, jj, i, j - 1]])
                                    / (2.0 * self.dy);
                            }
                        }

                        new_velocity[comp][[i, j, k]] += dt * sgs_divergence;
                    }
                }
            }
        }

        Ok(FluidState3D {
            velocity: new_velocity,
            pressure: state.pressure.clone(),
            dx: state.dx,
            dy: state.dy,
            dz: state.dz,
        })
    }

    /// Apply boundary conditions for 3D LES
    fn apply_boundary_conditions_3d(mut state: FluidState3D) -> IntegrateResult<FluidState3D> {
        let (nx, ny, nz) = state.velocity[0].dim();

        // No-slip boundary conditions on walls
        for comp in 0..3 {
            // x-boundaries
            for j in 0..ny {
                for k in 0..nz {
                    state.velocity[comp][[0, j, k]] = 0.0;
                    state.velocity[comp][[nx - 1, j, k]] = 0.0;
                }
            }

            // y-boundaries
            for i in 0..nx {
                for k in 0..nz {
                    state.velocity[comp][[i, 0, k]] = 0.0;
                    state.velocity[comp][[i, ny - 1, k]] = 0.0;
                }
            }

            // z-boundaries
            for i in 0..nx {
                for j in 0..ny {
                    state.velocity[comp][[i, j, 0]] = 0.0;
                    state.velocity[comp][[i, j, nz - 1]] = 0.0;
                }
            }
        }

        Ok(state)
    }
}

impl SGSModelTrait for LESolver {
    fn compute_sgs_viscosity(
        &self,
        velocity: &[Array3<f64>],
        filter_width: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> IntegrateResult<Array3<f64>> {
        let (nx, ny, nz) = velocity[0].dim();
        let mut nu_sgs = Array3::zeros((nx, ny, nz));

        // Create temporary state for strain rate computation
        let state = FluidState3D {
            velocity: velocity.to_vec(),
            pressure: Array3::zeros((nx, ny, nz)),
            dx,
            dy,
            dz,
        };

        match self.sgs_model {
            SGSModel::Smagorinsky => {
                let strain_rate = self.compute_strain_rate_tensor_3d(&state)?;

                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            let mut s_mag: f64 = 0.0;
                            for ii in 0..3 {
                                for jj in 0..3 {
                                    s_mag += strain_rate[[ii, jj, i, j, k]]
                                        * strain_rate[[ii, jj, i, j, k]];
                                }
                            }
                            s_mag = (2.0 * s_mag).sqrt();
                            nu_sgs[[i, j, k]] = (self.cs * filter_width).powi(2) * s_mag;
                        }
                    }
                }
            }
            _ => {
                // For other models, use a simplified implementation
                nu_sgs.fill(0.1 * filter_width.powi(2));
            }
        }

        Ok(nu_sgs)
    }

    fn compute_sgs_stress(
        &self,
        velocity: &[Array3<f64>],
        filter_width: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> IntegrateResult<Array3<[[f64; 3]; 3]>> {
        let (nx, ny, nz) = velocity[0].dim();
        let mut sgs_stress = Array3::from_elem((nx, ny, nz), [[0.0; 3]; 3]);

        let state = FluidState3D {
            velocity: velocity.to_vec(),
            pressure: Array3::zeros((nx, ny, nz)),
            dx,
            dy,
            dz,
        };

        let stress_tensor = self.compute_sgs_stress(&state)?;

        for i in 0..nx {
            for j in 0..ny {
                for ii in 0..3 {
                    for jj in 0..3 {
                        sgs_stress[[i, j, 0]][ii][jj] = stress_tensor[[ii, jj, i, j]];
                    }
                }
            }
        }

        Ok(sgs_stress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_les_solver_creation() {
        let solver = LESolver::new(8, 8, 8, 0.1, 0.1, 0.1, SGSModel::Smagorinsky);
        assert_eq!(solver.nx, 8);
        assert_eq!(solver.sgs_model, SGSModel::Smagorinsky);
        assert_eq!(solver.cs, 0.1);
    }

    #[test]
    fn test_fluid_state_3d_creation() {
        let velocity = vec![
            Array3::zeros((4, 4, 4)),
            Array3::zeros((4, 4, 4)),
            Array3::zeros((4, 4, 4)),
        ];
        let state = FluidState3D {
            velocity,
            pressure: Array3::zeros((4, 4, 4)),
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
        };

        assert_eq!(state.velocity.len(), 3);
        assert_eq!(state.velocity[0].dim(), (4, 4, 4));
    }

    #[test]
    fn test_sgs_model_enum() {
        assert_eq!(SGSModel::Smagorinsky, SGSModel::Smagorinsky);
        assert_ne!(SGSModel::Smagorinsky, SGSModel::WALE);
    }

    #[test]
    fn test_sgs_viscosity_computation() {
        let solver = LESolver::new(4, 4, 4, 0.1, 0.1, 0.1, SGSModel::Smagorinsky);
        let velocity = vec![
            Array3::ones((4, 4, 4)),
            Array3::zeros((4, 4, 4)),
            Array3::zeros((4, 4, 4)),
        ];

        let nu_sgs = solver.compute_sgs_viscosity(&velocity, 0.2, 0.1, 0.1, 0.1);
        assert!(nu_sgs.is_ok());
        let viscosity = nu_sgs.unwrap();
        assert_eq!(viscosity.dim(), (4, 4, 4));
    }
}
