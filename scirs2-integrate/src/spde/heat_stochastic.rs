//! Stochastic Heat Equation solver
//!
//! Solves:
//!
//! **Additive noise** (space–time white noise):
//! ```text
//! du = D Δu dt + σ dW(x,t)
//! ```
//!
//! **Multiplicative noise** (Stratonovich):
//! ```text
//! du = D Δu dt + σ u dW(x,t)
//! ```
//!
//! using Euler-Maruyama in time and second-order finite differences in space.
//!
//! ## 1-D scheme (uniform grid, Dirichlet BCs)
//!
//! Let `N` interior nodes, spacing `dx = L/(N+1)`.  With `D` the diffusion
//! coefficient and `dt` the time step the scheme reads:
//!
//! ```text
//! u_i^{n+1} = u_i^n + D * dt / dx^2 * (u_{i-1}^n - 2 u_i^n + u_{i+1}^n)
//!           + sqrt(dt/dx) * σ * ξ_i^n          (additive)
//!           + sqrt(dt/dx) * σ * u_i^n * ξ_i^n  (multiplicative)
//! ```
//!
//! The `sqrt(1/dx)` factor accounts for the spatial correlation of
//! space–time white noise.
//!
//! ## Stability
//!
//! Requires `D * dt / dx^2 ≤ 0.5` for explicit stability.  The solver
//! enforces this at construction and returns an error if violated.
//!
//! ## Karhunen-Loève noise
//!
//! For smoother noise the 1-D solver optionally uses a truncated
//! Karhunen-Loève expansion of the covariance operator to generate
//! spatially correlated noise increments.

use crate::error::{IntegrateError, IntegrateResult};
use crate::spde::random_fields::{CorrelationFunction, RandomField};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::prelude::{Normal, Rng, StdRng};
use scirs2_core::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Choice of noise type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseType {
    /// Additive white noise: `du = D Δu dt + σ dW`
    Additive,
    /// Multiplicative noise: `du = D Δu dt + σ u dW`
    Multiplicative,
}

/// Configuration for the stochastic heat equation solver.
#[derive(Debug, Clone)]
pub struct StochasticHeatConfig {
    /// Thermal diffusivity / diffusion coefficient D > 0.
    pub diffusion: f64,
    /// Noise amplitude σ.
    pub sigma: f64,
    /// Noise type (additive vs. multiplicative).
    pub noise_type: NoiseType,
    /// Optional spatial covariance function for coloured noise.
    /// When `None` the solver uses space-time white noise.
    pub spatial_covariance: Option<CorrelationFunction>,
    /// Number of KL modes used when `spatial_covariance` is `Some`.
    pub kl_terms: usize,
    /// Time step dt.
    pub dt: f64,
}

impl Default for StochasticHeatConfig {
    fn default() -> Self {
        Self {
            diffusion: 1.0,
            sigma: 0.1,
            noise_type: NoiseType::Additive,
            spatial_covariance: None,
            kl_terms: 20,
            dt: 1e-4,
        }
    }
}

/// Solution snapshot: field values at a single time point.
#[derive(Debug, Clone)]
pub struct HeatSnapshot {
    /// Simulation time.
    pub t: f64,
    /// Field values `u(x)` (1-D) or `u(x,y)` (2-D, stored in row-major order).
    pub u: Array1<f64>,
}

/// Full time-series output.
#[derive(Debug, Clone)]
pub struct StochasticHeatSolution {
    /// Time points at which snapshots were saved.
    pub times: Vec<f64>,
    /// Snapshots at each saved time.
    pub snapshots: Vec<Array1<f64>>,
    /// Spatial grid (1-D: x coordinates; 2-D: flattened `[nx, ny]` coordinates).
    pub grid: Array1<f64>,
    /// Problem dimension (1 or 2).
    pub dim: usize,
    /// Grid shape `[nx]` (1-D) or `[nx, ny]` (2-D).
    pub shape: Vec<usize>,
}

impl StochasticHeatSolution {
    /// Number of saved time points.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns true if no snapshots have been saved.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Compute the mean field over all saved snapshots.
    pub fn mean_field(&self) -> Option<Array1<f64>> {
        if self.snapshots.is_empty() {
            return None;
        }
        let n = self.snapshots[0].len();
        let mut mean = Array1::zeros(n);
        for snap in &self.snapshots {
            mean += snap;
        }
        let count = self.snapshots.len() as f64;
        Some(mean / count)
    }

    /// Compute the pointwise variance over all saved snapshots.
    pub fn variance_field(&self) -> Option<Array1<f64>> {
        let mean = self.mean_field()?;
        let n = mean.len();
        let mut var = Array1::zeros(n);
        for snap in &self.snapshots {
            let diff = snap - &mean;
            var = var + diff.mapv(|v| v * v);
        }
        let count = (self.snapshots.len() as f64 - 1.0).max(1.0);
        Some(var / count)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-D solver
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic heat equation solver in 1-D on `[0, L]` with Dirichlet BCs.
///
/// # Method
///
/// Euler-Maruyama in time + second-order central FD in space.
///
/// ```text
/// u_i^{n+1} = u_i^n
///           + D * dt/dx^2 * (u_{i-1}^n - 2 u_i^n + u_{i+1}^n)
///           + noise_amplitude_i * xi_i^n
/// ```
///
/// where `xi_i^n ~ N(0,1)` i.i.d. and `noise_amplitude_i` is set by
/// the noise type.
pub struct StochasticHeatSolver1D {
    cfg: StochasticHeatConfig,
    /// Interior node count N.
    n_nodes: usize,
    /// Spatial grid spacing dx = L / (N+1).
    dx: f64,
    /// Domain length L.
    domain_length: f64,
    /// Number of time steps between snapshots (≥ 1).
    save_every: usize,
    /// x-coordinates of interior nodes.
    x_coords: Array1<f64>,
}

impl StochasticHeatSolver1D {
    /// Construct a new 1-D solver.
    ///
    /// # Arguments
    /// * `config`       – Solver configuration.
    /// * `domain_length`– L: right endpoint of `[0, L]`.
    /// * `n_nodes`      – Number of **interior** nodes (excludes boundaries).
    /// * `save_every`   – Save a snapshot every this many time steps.
    ///
    /// # Errors
    /// Returns an error if the CFL condition `D dt / dx^2 ≤ 0.5` is violated,
    /// or if any parameter is non-positive.
    pub fn new(
        config: StochasticHeatConfig,
        domain_length: f64,
        n_nodes: usize,
        save_every: usize,
    ) -> IntegrateResult<Self> {
        if n_nodes == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_nodes must be at least 1".to_string(),
            ));
        }
        if domain_length <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "domain_length must be positive".to_string(),
            ));
        }
        if config.dt <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "dt must be positive".to_string(),
            ));
        }
        if config.diffusion <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "diffusion coefficient must be positive".to_string(),
            ));
        }
        let dx = domain_length / (n_nodes + 1) as f64;
        let cfl = config.diffusion * config.dt / (dx * dx);
        if cfl > 0.5 {
            return Err(IntegrateError::InvalidInput(format!(
                "CFL condition violated: D*dt/dx^2 = {cfl:.4} > 0.5.  \
                 Reduce dt or increase n_nodes.",
            )));
        }
        let x_coords = Array1::linspace(dx, domain_length - dx, n_nodes);
        Ok(Self {
            cfg: config,
            n_nodes,
            dx,
            domain_length,
            save_every: save_every.max(1),
            x_coords,
        })
    }

    /// Integrate from `t0` to `t_end` with initial condition `u0`.
    ///
    /// `u0` must have length `n_nodes` (interior nodes only; boundary values
    /// are assumed zero by Dirichlet BCs).
    ///
    /// # Errors
    /// Returns an error if `u0.len() != n_nodes` or `t_end ≤ t0`.
    pub fn solve(
        &self,
        u0: ArrayView1<f64>,
        t0: f64,
        t_end: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<StochasticHeatSolution> {
        if u0.len() != self.n_nodes {
            return Err(IntegrateError::DimensionMismatch(format!(
                "u0 length {} ≠ n_nodes {}",
                u0.len(),
                self.n_nodes
            )));
        }
        if t_end <= t0 {
            return Err(IntegrateError::InvalidInput(
                "t_end must be greater than t0".to_string(),
            ));
        }

        let normal = Normal::new(0.0_f64, 1.0)
            .map_err(|e| IntegrateError::ComputationError(format!("Normal distribution: {e}")))?;

        let dt = self.cfg.dt;
        let d = self.cfg.diffusion;
        let sigma = self.cfg.sigma;
        let r = d * dt / (self.dx * self.dx); // diffusion number
                                              // noise scale: sqrt(dt/dx) for space-time white noise
        let noise_scale = (dt / self.dx).sqrt();

        let n_steps = ((t_end - t0) / dt).ceil() as usize;
        let capacity = (n_steps / self.save_every + 2).max(2);

        let mut times = Vec::with_capacity(capacity);
        let mut snapshots = Vec::with_capacity(capacity);

        let mut u = u0.to_owned();
        let mut u_new = Array1::<f64>::zeros(self.n_nodes);

        // Save initial condition
        times.push(t0);
        snapshots.push(u.clone());

        let mut t = t0;
        for step in 0..n_steps {
            // Actual dt for final (possibly smaller) step
            let actual_dt = dt.min(t_end - t);
            if actual_dt <= 0.0 {
                break;
            }
            let r_act = d * actual_dt / (self.dx * self.dx);
            let ns_act = (actual_dt / self.dx).sqrt();

            // Generate spatially correlated or white noise
            let xi: Array1<f64> = if self.cfg.spatial_covariance.is_some() {
                self.sample_kl_noise(rng, actual_dt)?
            } else {
                Array1::from_vec((0..self.n_nodes).map(|_| rng.sample(normal)).collect())
            };

            // Euler-Maruyama step
            for i in 0..self.n_nodes {
                let u_left = if i == 0 { 0.0 } else { u[i - 1] }; // Dirichlet BC = 0
                let u_right = if i == self.n_nodes - 1 { 0.0 } else { u[i + 1] };
                let laplacian = u_left - 2.0 * u[i] + u_right;
                let noise_amp = match self.cfg.noise_type {
                    NoiseType::Additive => sigma * ns_act,
                    NoiseType::Multiplicative => sigma * u[i] * ns_act,
                };
                u_new[i] = u[i] + r_act * laplacian + noise_amp * xi[i];
            }

            u.assign(&u_new);
            t += actual_dt;

            if (step + 1) % self.save_every == 0 || t >= t_end - 1e-14 {
                times.push(t);
                snapshots.push(u.clone());
            }
        }

        Ok(StochasticHeatSolution {
            times,
            snapshots,
            grid: self.x_coords.clone(),
            dim: 1,
            shape: vec![self.n_nodes],
        })
    }

    /// Sample spatially correlated noise via KL expansion, scaled by `sqrt(dt)`.
    fn sample_kl_noise(&self, rng: &mut StdRng, dt: f64) -> IntegrateResult<Array1<f64>> {
        let cov = self
            .cfg
            .spatial_covariance
            .as_ref()
            .ok_or_else(|| IntegrateError::ComputationError("No covariance set".to_string()))?
            .clone();
        let gy = Array1::from_vec(vec![0.0_f64]); // dummy second dimension
        let field = RandomField::sample_kl_expansion(
            self.x_coords.view(),
            gy.view(),
            cov,
            self.cfg.kl_terms,
            rng,
        )?;
        // field has shape [n_nodes, 1]; extract column 0
        let noise = field.column(0).to_owned() * dt.sqrt();
        Ok(noise)
    }

    /// Return the x-coordinates of the interior grid nodes.
    pub fn grid(&self) -> &Array1<f64> {
        &self.x_coords
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-D solver
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic heat equation solver in 2-D on `[0, Lx] × [0, Ly]`.
///
/// Uses a 5-point Laplacian stencil with Dirichlet BCs (zero on boundary).
pub struct StochasticHeatSolver2D {
    cfg: StochasticHeatConfig,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    save_every: usize,
    /// Flat (row-major) x-coordinates of interior nodes.
    x_coords: Vec<f64>,
    /// Flat (row-major) y-coordinates of interior nodes.
    y_coords: Vec<f64>,
}

impl StochasticHeatSolver2D {
    /// Construct a 2-D stochastic heat solver.
    ///
    /// # Arguments
    /// * `config`  – Solver configuration.
    /// * `lx`, `ly`  – Domain dimensions.
    /// * `nx`, `ny`  – Number of **interior** nodes in x and y.
    /// * `save_every` – Snapshot interval.
    ///
    /// # Errors
    /// Returns an error if the 2-D CFL condition is violated:
    /// `D * dt * (1/dx^2 + 1/dy^2) ≤ 0.5`.
    pub fn new(
        config: StochasticHeatConfig,
        lx: f64,
        ly: f64,
        nx: usize,
        ny: usize,
        save_every: usize,
    ) -> IntegrateResult<Self> {
        if nx == 0 || ny == 0 {
            return Err(IntegrateError::InvalidInput(
                "nx and ny must be at least 1".to_string(),
            ));
        }
        if lx <= 0.0 || ly <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "Domain dimensions must be positive".to_string(),
            ));
        }
        if config.dt <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "dt must be positive".to_string(),
            ));
        }
        let dx = lx / (nx + 1) as f64;
        let dy = ly / (ny + 1) as f64;
        let cfl = config.diffusion * config.dt * (1.0 / (dx * dx) + 1.0 / (dy * dy));
        if cfl > 0.5 {
            return Err(IntegrateError::InvalidInput(format!(
                "2-D CFL condition violated: {cfl:.4} > 0.5.  \
                 Reduce dt or increase nx/ny.",
            )));
        }
        let n = nx * ny;
        let mut x_coords = vec![0.0_f64; n];
        let mut y_coords = vec![0.0_f64; n];
        for i in 0..nx {
            for j in 0..ny {
                let xi = (i + 1) as f64 * dx;
                let yj = (j + 1) as f64 * dy;
                x_coords[i * ny + j] = xi;
                y_coords[i * ny + j] = yj;
            }
        }
        Ok(Self {
            cfg: config,
            nx,
            ny,
            dx,
            dy,
            save_every: save_every.max(1),
            x_coords,
            y_coords,
        })
    }

    /// Total number of interior nodes.
    pub fn n_nodes(&self) -> usize {
        self.nx * self.ny
    }

    /// Solve from `t0` to `t_end`.
    ///
    /// `u0` must be a flat `[nx * ny]` array in row-major (i * ny + j) order.
    pub fn solve(
        &self,
        u0: &[f64],
        t0: f64,
        t_end: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<StochasticHeatSolution> {
        let n = self.n_nodes();
        if u0.len() != n {
            return Err(IntegrateError::DimensionMismatch(format!(
                "u0 length {} ≠ n_nodes {}",
                u0.len(),
                n
            )));
        }
        if t_end <= t0 {
            return Err(IntegrateError::InvalidInput(
                "t_end must be greater than t0".to_string(),
            ));
        }

        let normal = Normal::new(0.0_f64, 1.0)
            .map_err(|e| IntegrateError::ComputationError(format!("Normal distribution: {e}")))?;

        let dt = self.cfg.dt;
        let d = self.cfg.diffusion;
        let sigma = self.cfg.sigma;
        let rx = d * dt / (self.dx * self.dx);
        let ry = d * dt / (self.dy * self.dy);
        // noise scale for 2-D space-time white noise: sqrt(dt / (dx * dy))
        let noise_scale = (dt / (self.dx * self.dy)).sqrt();

        let n_steps = ((t_end - t0) / dt).ceil() as usize;
        let capacity = (n_steps / self.save_every + 2).max(2);

        let mut times = Vec::with_capacity(capacity);
        let mut snapshots = Vec::with_capacity(capacity);

        let mut u = u0.to_vec();
        let mut u_new = vec![0.0_f64; n];

        times.push(t0);
        snapshots.push(Array1::from_vec(u.clone()));

        let mut t = t0;
        for step in 0..n_steps {
            let actual_dt = dt.min(t_end - t);
            if actual_dt <= 0.0 {
                break;
            }
            let rx_act = d * actual_dt / (self.dx * self.dx);
            let ry_act = d * actual_dt / (self.dy * self.dy);
            let ns_act = (actual_dt / (self.dx * self.dy)).sqrt();

            for idx in 0..n {
                let i = idx / self.ny;
                let j = idx % self.ny;

                let u_left = if i > 0 { u[(i - 1) * self.ny + j] } else { 0.0 };
                let u_right = if i < self.nx - 1 {
                    u[(i + 1) * self.ny + j]
                } else {
                    0.0
                };
                let u_down = if j > 0 { u[i * self.ny + j - 1] } else { 0.0 };
                let u_up = if j < self.ny - 1 {
                    u[i * self.ny + j + 1]
                } else {
                    0.0
                };

                let lap = (u_left - 2.0 * u[idx] + u_right) / (self.dx * self.dx)
                    + (u_down - 2.0 * u[idx] + u_up) / (self.dy * self.dy);

                let xi = rng.sample(normal);
                let noise_amp = match self.cfg.noise_type {
                    NoiseType::Additive => sigma * ns_act,
                    NoiseType::Multiplicative => sigma * u[idx] * ns_act,
                };
                u_new[idx] = u[idx] + d * actual_dt * lap + noise_amp * xi;
            }

            u.copy_from_slice(&u_new);
            t += actual_dt;

            if (step + 1) % self.save_every == 0 || t >= t_end - 1e-14 {
                times.push(t);
                snapshots.push(Array1::from_vec(u.clone()));
            }
        }

        let grid = Array1::from_vec(self.x_coords.clone()); // x-coords as representative grid
        Ok(StochasticHeatSolution {
            times,
            snapshots,
            grid,
            dim: 2,
            shape: vec![self.nx, self.ny],
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use scirs2_core::random::prelude::*;

    fn make_rng() -> StdRng {
        seeded_rng(12345)
    }

    #[test]
    fn test_1d_additive_runs() {
        let config = StochasticHeatConfig {
            diffusion: 0.1,
            sigma: 0.05,
            noise_type: NoiseType::Additive,
            dt: 1e-4,
            ..Default::default()
        };
        let solver = StochasticHeatSolver1D::new(config, 1.0, 20, 10)
            .expect("StochasticHeatSolver1D::new should succeed");
        let u0 = Array1::from_vec(
            (0..20)
                .map(|i| ((i as f64 + 1.0) * std::f64::consts::PI / 21.0).sin())
                .collect(),
        );
        let mut rng = make_rng();
        let sol = solver
            .solve(u0.view(), 0.0, 0.01, &mut rng)
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
        // Mass should remain finite
        for snap in &sol.snapshots {
            assert!(
                snap.iter().all(|v| v.is_finite()),
                "Non-finite value in snapshot"
            );
        }
    }

    #[test]
    fn test_1d_multiplicative_runs() {
        let config = StochasticHeatConfig {
            diffusion: 0.1,
            sigma: 0.1,
            noise_type: NoiseType::Multiplicative,
            dt: 1e-4,
            ..Default::default()
        };
        let solver = StochasticHeatSolver1D::new(config, 1.0, 10, 5)
            .expect("StochasticHeatSolver1D::new should succeed");
        let u0 = Array1::from_vec(vec![0.5_f64; 10]);
        let mut rng = make_rng();
        let sol = solver
            .solve(u0.view(), 0.0, 0.005, &mut rng)
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
    }

    #[test]
    fn test_cfl_violation_returns_error() {
        // dx = 1/21 ≈ 0.0476, dt = 0.1, D = 1.0 → D*dt/dx^2 ≈ 44 >> 0.5
        let config = StochasticHeatConfig {
            diffusion: 1.0,
            sigma: 0.1,
            dt: 0.1,
            ..Default::default()
        };
        let result = StochasticHeatSolver1D::new(config, 1.0, 20, 1);
        assert!(result.is_err(), "Should fail CFL check");
    }

    #[test]
    fn test_2d_additive_runs() {
        let config = StochasticHeatConfig {
            diffusion: 0.05,
            sigma: 0.02,
            noise_type: NoiseType::Additive,
            dt: 5e-5,
            ..Default::default()
        };
        let solver = StochasticHeatSolver2D::new(config, 1.0, 1.0, 8, 8, 5)
            .expect("StochasticHeatSolver2D::new should succeed");
        let u0 = vec![0.1_f64; 64];
        let mut rng = make_rng();
        let sol = solver
            .solve(&u0, 0.0, 0.001, &mut rng)
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
        for snap in &sol.snapshots {
            assert!(snap.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_mean_variance_field() {
        let config = StochasticHeatConfig {
            diffusion: 0.1,
            sigma: 0.05,
            dt: 1e-4,
            ..Default::default()
        };
        let solver = StochasticHeatSolver1D::new(config, 1.0, 10, 5)
            .expect("StochasticHeatSolver1D::new should succeed");
        let u0 = Array1::from_vec(vec![0.0_f64; 10]);
        let mut rng = make_rng();
        let sol = solver
            .solve(u0.view(), 0.0, 0.01, &mut rng)
            .expect("solver.solve should succeed");
        let mean = sol.mean_field().expect("mean_field should succeed");
        let var = sol.variance_field().expect("variance_field should succeed");
        assert_eq!(mean.len(), 10);
        assert_eq!(var.len(), 10);
        assert!(var.iter().all(|v| *v >= 0.0));
    }
}
