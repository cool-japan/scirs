//! Spectral Navier-Stokes solver for periodic domains
//!
//! This module provides spectral methods for solving the Navier-Stokes equations
//! in periodic domains using FFT-based approaches.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::numeric::Complex;

use super::dealiasing::DealiasingStrategy;
use super::fft_operations::{FFTOperations, FFTResult};

/// Spectral Navier-Stokes solver for periodic domains
pub struct SpectralNavierStokesSolver {
    /// Number of grid points in x-direction
    pub nx: usize,
    /// Number of grid points in y-direction  
    pub ny: usize,
    /// Number of grid points in z-direction (for 3D)
    pub nz: Option<usize>,
    /// Domain size in x-direction
    pub lx: f64,
    /// Domain size in y-direction
    pub ly: f64,
    /// Domain size in z-direction (for 3D)
    pub lz: Option<f64>,
    /// Kinematic viscosity
    pub nu: f64,
    /// Time step
    pub dt: f64,
    /// Dealiasing strategy
    pub dealiasing: DealiasingStrategy,
}

impl SpectralNavierStokesSolver {
    /// Create new spectral Navier-Stokes solver for periodic domain
    pub fn new(
        nx: usize,
        ny: usize,
        nz: Option<usize>,
        lx: f64,
        ly: f64,
        lz: Option<f64>,
        nu: f64,
        dt: f64,
        dealiasing: DealiasingStrategy,
    ) -> Self {
        Self {
            nx,
            ny,
            nz,
            lx,
            ly,
            lz,
            nu,
            dt,
            dealiasing,
        }
    }

    /// Solve 2D Navier-Stokes with spectral methods
    pub fn solve_2d_spectral(
        &self,
        initial_vorticity: &Array2<f64>,
        t_final: f64,
    ) -> IntegrateResult<Vec<Array2<f64>>> {
        let n_steps = (t_final / self.dt) as usize;
        let mut vorticity_history = Vec::with_capacity(n_steps + 1);
        vorticity_history.push(initial_vorticity.clone());

        let mut omega = initial_vorticity.clone();

        // Wavenumber grids
        let kx = SpectralNavierStokesSolver::wavenumber_grid_1d(self.nx, self.lx);
        let ky = SpectralNavierStokesSolver::wavenumber_grid_1d(self.ny, self.ly);

        for _step in 0..n_steps {
            omega = self.rk4_step_2d(&omega, &kx, &ky)?;
            vorticity_history.push(omega.clone());
        }

        Ok(vorticity_history)
    }

    /// Solve 3D Navier-Stokes with spectral methods
    pub fn solve_3d_spectral(
        &self,
        initial_velocity: &[Array3<f64>; 3],
        t_final: f64,
    ) -> IntegrateResult<Vec<[Array3<f64>; 3]>> {
        let nz = self.nz.ok_or_else(|| {
            IntegrateError::InvalidInput("3D solver requires nz to be specified".to_string())
        })?;
        let lz = self.lz.ok_or_else(|| {
            IntegrateError::InvalidInput("3D solver requires lz to be specified".to_string())
        })?;

        let n_steps = (t_final / self.dt) as usize;
        let mut velocity_history = Vec::with_capacity(n_steps + 1);
        velocity_history.push(initial_velocity.clone());

        let mut u = initial_velocity.clone();

        // Wavenumber grids
        let kx = SpectralNavierStokesSolver::wavenumber_grid_1d(self.nx, self.lx);
        let ky = SpectralNavierStokesSolver::wavenumber_grid_1d(self.ny, self.ly);
        let kz = SpectralNavierStokesSolver::wavenumber_grid_1d(nz, lz);

        for _step in 0..n_steps {
            u = self.rk4_step_3d(&u, &kx, &ky, &kz)?;
            velocity_history.push(u.clone());
        }

        Ok(velocity_history)
    }

    /// RK4 time integration step for 2D vorticity
    fn rk4_step_2d(
        &self,
        omega: &Array2<f64>,
        kx: &Array1<f64>,
        ky: &Array1<f64>,
    ) -> IntegrateResult<Array2<f64>> {
        let k1 = self.vorticity_rhs_2d(omega, kx, ky)?;
        let omega1 = omega + &k1 * (self.dt / 2.0);

        let k2 = self.vorticity_rhs_2d(&omega1, kx, ky)?;
        let omega2 = omega + &k2 * (self.dt / 2.0);

        let k3 = self.vorticity_rhs_2d(&omega2, kx, ky)?;
        let omega3 = omega + &k3 * self.dt;

        let k4 = self.vorticity_rhs_2d(&omega3, kx, ky)?;

        Ok(omega + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (self.dt / 6.0))
    }

    /// RK4 time integration step for 3D velocity
    fn rk4_step_3d(
        &self,
        u: &[Array3<f64>; 3],
        kx: &Array1<f64>,
        ky: &Array1<f64>,
        kz: &Array1<f64>,
    ) -> IntegrateResult<[Array3<f64>; 3]> {
        let k1 = self.velocity_rhs_3d(u, kx, ky, kz)?;
        let u1 = [
            &u[0] + &k1[0] * (self.dt / 2.0),
            &u[1] + &k1[1] * (self.dt / 2.0),
            &u[2] + &k1[2] * (self.dt / 2.0),
        ];

        let k2 = self.velocity_rhs_3d(&u1, kx, ky, kz)?;
        let u2 = [
            &u[0] + &k2[0] * (self.dt / 2.0),
            &u[1] + &k2[1] * (self.dt / 2.0),
            &u[2] + &k2[2] * (self.dt / 2.0),
        ];

        let k3 = self.velocity_rhs_3d(&u2, kx, ky, kz)?;
        let u3 = [
            &u[0] + &k3[0] * self.dt,
            &u[1] + &k3[1] * self.dt,
            &u[2] + &k3[2] * self.dt,
        ];

        let k4 = self.velocity_rhs_3d(&u3, kx, ky, kz)?;

        Ok([
            &u[0] + (&k1[0] + &k2[0] * 2.0 + &k3[0] * 2.0 + &k4[0]) * (self.dt / 6.0),
            &u[1] + (&k1[1] + &k2[1] * 2.0 + &k3[1] * 2.0 + &k4[1]) * (self.dt / 6.0),
            &u[2] + (&k1[2] + &k2[2] * 2.0 + &k3[2] * 2.0 + &k4[2]) * (self.dt / 6.0),
        ])
    }

    /// Right-hand side of 2D vorticity equation in spectral space
    fn vorticity_rhs_2d(
        &self,
        omega: &Array2<f64>,
        kx: &Array1<f64>,
        ky: &Array1<f64>,
    ) -> IntegrateResult<Array2<f64>> {
        // Transform vorticity to spectral space
        let omega_hat = FFTOperations::fft_2d_forward(omega)?;

        // Solve for stream function: ∇²ψ = -ω
        let psi_hat = self.solve_poisson_2d(&omega_hat, kx, ky)?;

        // Compute velocity from stream function: u = ∂ψ/∂y, v = -∂ψ/∂x
        let u_hat = self.derivative_spectral_2d(&psi_hat, kx, ky, 0, 1)?;
        let v_hat = self.derivative_spectral_2d(&psi_hat, kx, ky, 1, 0)?;

        // Transform to physical space
        let u = FFTOperations::fft_2d_backward(&u_hat)?;
        let v = FFTOperations::fft_2d_backward(&v_hat)?;

        // Compute advection terms: ∂ω/∂x and ∂ω/∂y
        let dwdx_hat = self.derivative_spectral_2d(&omega_hat, kx, ky, 1, 0)?;
        let dwdy_hat = self.derivative_spectral_2d(&omega_hat, kx, ky, 0, 1)?;

        let dwdx = FFTOperations::fft_2d_backward(&dwdx_hat)?;
        let dwdy = FFTOperations::fft_2d_backward(&dwdy_hat)?;

        // Compute nonlinear advection: u·∇ω
        let advection = &u * &dwdx + &v * &dwdy;

        // Apply dealiasing and convert to spectral space
        let advection_dealiased = self.apply_dealiasing_2d(&advection)?;
        let advection_hat = FFTOperations::fft_2d_forward(&advection_dealiased)?;

        // Compute diffusion term: ν∇²ω
        let diffusion_hat = self.laplacian_spectral_2d(&omega_hat, kx, ky)?;

        // Right-hand side: -u·∇ω + ν∇²ω
        let rhs_hat = &diffusion_hat * self.nu - &advection_hat;

        FFTOperations::fft_2d_backward(&rhs_hat)
    }

    /// Right-hand side of 3D velocity equations in spectral space
    fn velocity_rhs_3d(
        &self,
        u: &[Array3<f64>; 3],
        kx: &Array1<f64>,
        ky: &Array1<f64>,
        kz: &Array1<f64>,
    ) -> IntegrateResult<[Array3<f64>; 3]> {
        // Transform velocity to spectral space
        let u_hat = [
            FFTOperations::fft_3d_forward(&u[0])?,
            FFTOperations::fft_3d_forward(&u[1])?,
            FFTOperations::fft_3d_forward(&u[2])?,
        ];

        // Solve pressure Poisson equation
        let pressure_hat = self.solve_pressure_poisson_3d(&u_hat, kx, ky, kz)?;

        // Compute pressure gradient
        let dpdx_hat = self.derivative_spectral_3d(&pressure_hat, kx, ky, kz, 1, 0, 0)?;
        let dpdy_hat = self.derivative_spectral_3d(&pressure_hat, kx, ky, kz, 0, 1, 0)?;
        let dpdz_hat = self.derivative_spectral_3d(&pressure_hat, kx, ky, kz, 0, 0, 1)?;

        // Compute nonlinear terms (convection): (u·∇)u
        let mut nonlinear = [
            Array3::zeros(u[0].dim()),
            Array3::zeros(u[1].dim()),
            Array3::zeros(u[2].dim()),
        ];

        for i in 0..3 {
            let dudx_hat = self.derivative_spectral_3d(&u_hat[i], kx, ky, kz, 1, 0, 0)?;
            let dudy_hat = self.derivative_spectral_3d(&u_hat[i], kx, ky, kz, 0, 1, 0)?;
            let dudz_hat = self.derivative_spectral_3d(&u_hat[i], kx, ky, kz, 0, 0, 1)?;

            let dudx = FFTOperations::fft_3d_backward(&dudx_hat)?;
            let dudy = FFTOperations::fft_3d_backward(&dudy_hat)?;
            let dudz = FFTOperations::fft_3d_backward(&dudz_hat)?;

            nonlinear[i] = &u[0] * &dudx + &u[1] * &dudy + &u[2] * &dudz;
        }

        // Apply dealiasing and convert to spectral space
        let nonlinear_hat = [
            FFTOperations::fft_3d_forward(&self.apply_dealiasing_3d(&nonlinear[0])?)?,
            FFTOperations::fft_3d_forward(&self.apply_dealiasing_3d(&nonlinear[1])?)?,
            FFTOperations::fft_3d_forward(&self.apply_dealiasing_3d(&nonlinear[2])?)?,
        ];

        // Compute viscous terms: ν∇²u
        let viscous_hat = [
            &self.laplacian_spectral_3d(&u_hat[0], kx, ky, kz)? * self.nu,
            &self.laplacian_spectral_3d(&u_hat[1], kx, ky, kz)? * self.nu,
            &self.laplacian_spectral_3d(&u_hat[2], kx, ky, kz)? * self.nu,
        ];

        // Right-hand side: -(u·∇)u - ∇p + ν∇²u
        let rhs_hat = [
            &viscous_hat[0] - &nonlinear_hat[0] - &dpdx_hat,
            &viscous_hat[1] - &nonlinear_hat[1] - &dpdy_hat,
            &viscous_hat[2] - &nonlinear_hat[2] - &dpdz_hat,
        ];

        Ok([
            FFTOperations::fft_3d_backward(&rhs_hat[0])?,
            FFTOperations::fft_3d_backward(&rhs_hat[1])?,
            FFTOperations::fft_3d_backward(&rhs_hat[2])?,
        ])
    }

    /// Generate 1D wavenumber grid
    fn wavenumber_grid_1d(n: usize, l: f64) -> Array1<f64> {
        let mut k = Array1::zeros(n);
        let dk = 2.0 * std::f64::consts::PI / l;

        for i in 0..n {
            if i <= n / 2 {
                k[i] = (i as f64) * dk;
            } else {
                k[i] = ((i as i32) - (n as i32)) as f64 * dk;
            }
        }

        k
    }

    /// Solve Poisson equation in 2D spectral space: ∇²φ = f
    fn solve_poisson_2d(
        &self,
        f_hat: &Array2<Complex<f64>>,
        kx: &Array1<f64>,
        ky: &Array1<f64>,
    ) -> IntegrateResult<Array2<Complex<f64>>> {
        let mut phi_hat = Array2::zeros(f_hat.dim());

        for i in 0..self.nx {
            for j in 0..self.ny {
                let k2 = kx[i] * kx[i] + ky[j] * ky[j];
                if k2 > 1e-12 {
                    phi_hat[[i, j]] = -f_hat[[i, j]] / k2;
                }
            }
        }

        Ok(phi_hat)
    }

    /// Solve pressure Poisson equation in 3D spectral space
    fn solve_pressure_poisson_3d(
        &self,
        u_hat: &[Array3<Complex<f64>>; 3],
        kx: &Array1<f64>,
        ky: &Array1<f64>,
        kz: &Array1<f64>,
    ) -> IntegrateResult<Array3<Complex<f64>>> {
        let nz = self.nz.unwrap();
        let mut pressure_hat = Array3::zeros((self.nx, self.ny, nz));

        // Compute divergence in spectral space: ∇·u
        let mut div_hat = Array3::zeros((self.nx, self.ny, nz));

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..nz {
                    div_hat[[i, j, k]] = Complex::new(0.0, 1.0)
                        * (kx[i] * u_hat[0][[i, j, k]]
                            + ky[j] * u_hat[1][[i, j, k]]
                            + kz[k] * u_hat[2][[i, j, k]]);
                }
            }
        }

        // Solve Poisson equation: ∇²p = -∇·u
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..nz {
                    let k2 = kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k];
                    if k2 > 1e-12 {
                        pressure_hat[[i, j, k]] = div_hat[[i, j, k]] / k2;
                    }
                }
            }
        }

        Ok(pressure_hat)
    }

    /// Compute spectral derivative in 2D
    fn derivative_spectral_2d(
        &self,
        f_hat: &Array2<Complex<f64>>,
        kx: &Array1<f64>,
        ky: &Array1<f64>,
        dx: usize,
        dy: usize,
    ) -> IntegrateResult<Array2<Complex<f64>>> {
        let mut df_hat = Array2::zeros(f_hat.dim());

        for i in 0..self.nx {
            for j in 0..self.ny {
                let mut factor = Complex::new(1.0, 0.0);

                if dx > 0 {
                    factor *= Complex::new(0.0, kx[i]).powi(dx as i32);
                }
                if dy > 0 {
                    factor *= Complex::new(0.0, ky[j]).powi(dy as i32);
                }

                df_hat[[i, j]] = f_hat[[i, j]] * factor;
            }
        }

        Ok(df_hat)
    }

    /// Compute spectral derivative in 3D
    fn derivative_spectral_3d(
        &self,
        f_hat: &Array3<Complex<f64>>,
        kx: &Array1<f64>,
        ky: &Array1<f64>,
        kz: &Array1<f64>,
        dx: usize,
        dy: usize,
        dz: usize,
    ) -> IntegrateResult<Array3<Complex<f64>>> {
        let nz = self.nz.unwrap();
        let mut df_hat = Array3::zeros(f_hat.dim());

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..nz {
                    let mut factor = Complex::new(1.0, 0.0);

                    if dx > 0 {
                        factor *= Complex::new(0.0, kx[i]).powi(dx as i32);
                    }
                    if dy > 0 {
                        factor *= Complex::new(0.0, ky[j]).powi(dy as i32);
                    }
                    if dz > 0 {
                        factor *= Complex::new(0.0, kz[k]).powi(dz as i32);
                    }

                    df_hat[[i, j, k]] = f_hat[[i, j, k]] * factor;
                }
            }
        }

        Ok(df_hat)
    }

    /// Compute Laplacian in 2D spectral space
    fn laplacian_spectral_2d(
        &self,
        f_hat: &Array2<Complex<f64>>,
        kx: &Array1<f64>,
        ky: &Array1<f64>,
    ) -> IntegrateResult<Array2<Complex<f64>>> {
        let mut laplacian_hat = Array2::zeros(f_hat.dim());

        for i in 0..self.nx {
            for j in 0..self.ny {
                let k2 = kx[i] * kx[i] + ky[j] * ky[j];
                laplacian_hat[[i, j]] = f_hat[[i, j]] * (-k2);
            }
        }

        Ok(laplacian_hat)
    }

    /// Compute Laplacian in 3D spectral space
    fn laplacian_spectral_3d(
        &self,
        f_hat: &Array3<Complex<f64>>,
        kx: &Array1<f64>,
        ky: &Array1<f64>,
        kz: &Array1<f64>,
    ) -> IntegrateResult<Array3<Complex<f64>>> {
        let nz = self.nz.unwrap();
        let mut laplacian_hat = Array3::zeros(f_hat.dim());

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..nz {
                    let k2 = kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k];
                    laplacian_hat[[i, j, k]] = f_hat[[i, j, k]] * (-k2);
                }
            }
        }

        Ok(laplacian_hat)
    }

    /// Apply dealiasing in 2D
    fn apply_dealiasing_2d(&self, field: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        match self.dealiasing {
            DealiasingStrategy::None => Ok(field.clone()),
            DealiasingStrategy::TwoThirds => {
                // Apply 2/3 rule dealiasing
                let field_hat = FFTOperations::fft_2d_forward(field)?;
                let mut dealiased_hat = field_hat.clone();

                let cutoff_x = (2 * self.nx) / 3;
                let cutoff_y = (2 * self.ny) / 3;

                for i in cutoff_x..self.nx {
                    for j in 0..self.ny {
                        dealiased_hat[[i, j]] = Complex::new(0.0, 0.0);
                    }
                }
                for i in 0..self.nx {
                    for j in cutoff_y..self.ny {
                        dealiased_hat[[i, j]] = Complex::new(0.0, 0.0);
                    }
                }

                FFTOperations::fft_2d_backward(&dealiased_hat)
            }
            _ => {
                // Other dealiasing strategies would be implemented here
                Ok(field.clone())
            }
        }
    }

    /// Apply dealiasing in 3D
    fn apply_dealiasing_3d(&self, field: &Array3<f64>) -> IntegrateResult<Array3<f64>> {
        match self.dealiasing {
            DealiasingStrategy::None => Ok(field.clone()),
            DealiasingStrategy::TwoThirds => {
                let field_hat = FFTOperations::fft_3d_forward(field)?;
                let mut dealiased_hat = field_hat.clone();
                let nz = self.nz.unwrap();

                let cutoff_x = (2 * self.nx) / 3;
                let cutoff_y = (2 * self.ny) / 3;
                let cutoff_z = (2 * nz) / 3;

                for i in cutoff_x..self.nx {
                    for j in 0..self.ny {
                        for k in 0..nz {
                            dealiased_hat[[i, j, k]] = Complex::new(0.0, 0.0);
                        }
                    }
                }
                for i in 0..self.nx {
                    for j in cutoff_y..self.ny {
                        for k in 0..nz {
                            dealiased_hat[[i, j, k]] = Complex::new(0.0, 0.0);
                        }
                    }
                }
                for i in 0..self.nx {
                    for j in 0..self.ny {
                        for k in cutoff_z..nz {
                            dealiased_hat[[i, j, k]] = Complex::new(0.0, 0.0);
                        }
                    }
                }

                FFTOperations::fft_3d_backward(&dealiased_hat)
            }
            _ => Ok(field.clone()),
        }
    }

    /// Initialize Taylor-Green vortex in 2D
    pub fn initialize_taylor_green_vortex_2d(&self) -> Array2<f64> {
        let mut omega = Array2::zeros((self.nx, self.ny));
        let dx = self.lx / (self.nx as f64);
        let dy = self.ly / (self.ny as f64);

        for i in 0..self.nx {
            for j in 0..self.ny {
                let x = (i as f64) * dx;
                let y = (j as f64) * dy;

                // Taylor-Green vortex initialization
                omega[[i, j]] = 2.0
                    * (2.0 * std::f64::consts::PI * x / self.lx).cos()
                    * (2.0 * std::f64::consts::PI * y / self.ly).cos();
            }
        }

        omega
    }

    /// Initialize Taylor-Green vortex in 3D
    pub fn initialize_taylor_green_vortex_3d(&self) -> [Array3<f64>; 3] {
        let nz = self.nz.unwrap();
        let lz = self.lz.unwrap();

        let mut u = Array3::zeros((self.nx, self.ny, nz));
        let mut v = Array3::zeros((self.nx, self.ny, nz));
        let mut w = Array3::zeros((self.nx, self.ny, nz));

        let dx = self.lx / (self.nx as f64);
        let dy = self.ly / (self.ny as f64);
        let dz = lz / (nz as f64);

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..nz {
                    let x = (i as f64) * dx;
                    let y = (j as f64) * dy;
                    let z = (k as f64) * dz;

                    let kx = 2.0 * std::f64::consts::PI / self.lx;
                    let ky = 2.0 * std::f64::consts::PI / self.ly;
                    let kz = 2.0 * std::f64::consts::PI / lz;

                    // Taylor-Green vortex 3D initialization
                    u[[i, j, k]] = (kx * x).sin() * (ky * y).cos() * (kz * z).cos();
                    v[[i, j, k]] = -(kx * x).cos() * (ky * y).sin() * (kz * z).cos();
                    w[[i, j, k]] = 0.0;
                }
            }
        }

        [u, v, w]
    }
}
