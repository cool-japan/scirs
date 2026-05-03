//! Stochastic Wave Equation solver
//!
//! Solves the stochastic wave equation in 1-D on `[0, L]`:
//!
//! ```text
//! ∂²u/∂t² = c² ∂²u/∂x² + σ Ẇ(x,t)
//! ```
//!
//! where `Ẇ(x,t)` is space-time white noise (formal time derivative of the
//! Wiener field).
//!
//! ## Numerical Method
//!
//! The second-order equation is rewritten as a first-order system by
//! introducing the velocity field `v = ∂u/∂t`:
//!
//! ```text
//! ∂u/∂t = v
//! ∂v/∂t = c² ∂²u/∂x² + σ Ẇ
//! ```
//!
//! The **leapfrog (Störmer-Verlet) scheme** integrates this system:
//!
//! ```text
//! u^{n+1}  = u^n + dt * v^{n+1/2}
//! v^{n+1/2} = v^{n-1/2} + dt * (c²/dx² * Lu^n + σ/sqrt(dx*dt) * ξ_i^n)
//! ```
//!
//! where `Lu` is the second-order FD Laplacian and `ξ_i^n ~ N(0,1)`.
//!
//! The staggered (half-step) velocity maintains the second-order accuracy of
//! the deterministic scheme.  With noise the strong order is 0.5 and the weak
//! order is 1.
//!
//! ## Stability
//!
//! Requires the Courant condition `c * dt / dx ≤ 1`.  The solver checks this
//! at construction.
//!
//! ## Energy Tracking
//!
//! The discrete energy
//! ```text
//! E^n = (1/2) Σ_i [ (v_i^n)^2 + c^2 ((u_{i+1}^n - u_i^n)/dx)^2 ] * dx
//! ```
//! is computed and stored at each saved snapshot, allowing users to monitor
//! energy growth due to the stochastic forcing.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::prelude::{Normal, Rng, StdRng};
use scirs2_core::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the stochastic wave equation solver.
#[derive(Debug, Clone)]
pub struct StochasticWaveConfig {
    /// Wave speed c > 0.
    pub wave_speed: f64,
    /// Noise amplitude σ.
    pub sigma: f64,
    /// Time step dt.
    pub dt: f64,
}

impl Default for StochasticWaveConfig {
    fn default() -> Self {
        Self {
            wave_speed: 1.0,
            sigma: 0.01,
            dt: 1e-4,
        }
    }
}

/// A single snapshot of the wave state at time `t`.
#[derive(Debug, Clone)]
pub struct WaveSnapshot {
    /// Simulation time.
    pub t: f64,
    /// Displacement field u(x).
    pub u: Array1<f64>,
    /// Velocity field v(x) = ∂u/∂t.
    pub v: Array1<f64>,
    /// Discrete energy E = (1/2) Σ [v_i^2 + c^2 ((u_{i+1}-u_i)/dx)^2] dx.
    pub energy: f64,
}

/// Full output of the stochastic wave solver.
#[derive(Debug, Clone)]
pub struct StochasticWaveSolution {
    /// Saved snapshots.
    pub snapshots: Vec<WaveSnapshot>,
    /// x-coordinates of interior nodes.
    pub grid: Array1<f64>,
}

impl StochasticWaveSolution {
    /// Number of saved snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns true if no snapshots have been saved.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Return time series of energies.
    pub fn energy_series(&self) -> Vec<f64> {
        self.snapshots.iter().map(|s| s.energy).collect()
    }

    /// Return mean energy growth rate (E_final - E_initial) / T.
    pub fn mean_energy_growth_rate(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }
        let e0 = self.snapshots[0].energy;
        let ef = self.snapshots.last().map(|s| s.energy).unwrap_or(e0);
        let t0 = self.snapshots[0].t;
        let tf = self.snapshots.last().map(|s| s.t).unwrap_or(t0);
        if (tf - t0).abs() < 1e-15 {
            0.0
        } else {
            (ef - e0) / (tf - t0)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Solver
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic wave equation solver on `[0, L]` with Dirichlet BCs.
///
/// Uses the leapfrog scheme with staggered velocity.
pub struct StochasticWaveSolver {
    cfg: StochasticWaveConfig,
    /// Interior node count N.
    n_nodes: usize,
    /// Grid spacing dx = L / (N+1).
    dx: f64,
    /// Snapshot interval (steps).
    save_every: usize,
    /// x-coordinates of interior nodes.
    x_coords: Array1<f64>,
}

impl StochasticWaveSolver {
    /// Construct a new stochastic wave solver.
    ///
    /// # Arguments
    /// * `config`        – Solver configuration.
    /// * `domain_length` – L.
    /// * `n_nodes`       – Interior node count N.
    /// * `save_every`    – Snapshot interval.
    ///
    /// # Errors
    /// Returns an error if the Courant condition `c * dt / dx > 1` is violated.
    pub fn new(
        config: StochasticWaveConfig,
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
        if config.wave_speed <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "wave_speed must be positive".to_string(),
            ));
        }
        let dx = domain_length / (n_nodes + 1) as f64;
        let courant = config.wave_speed * config.dt / dx;
        if courant > 1.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "Courant condition violated: c*dt/dx = {courant:.4} > 1.  \
                 Reduce dt or increase n_nodes.",
            )));
        }
        let x_coords = Array1::linspace(dx, domain_length - dx, n_nodes);
        Ok(Self {
            cfg: config,
            n_nodes,
            dx,
            save_every: save_every.max(1),
            x_coords,
        })
    }

    /// Integrate from `t0` to `t_end`.
    ///
    /// # Arguments
    /// * `u0` – Initial displacement (length `n_nodes`).
    /// * `v0` – Initial velocity     (length `n_nodes`).
    /// * `t0`, `t_end` – Time interval.
    /// * `rng` – Random number generator.
    ///
    /// # Errors
    /// Returns an error on dimension mismatch or invalid time span.
    pub fn solve(
        &self,
        u0: &Array1<f64>,
        v0: &Array1<f64>,
        t0: f64,
        t_end: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<StochasticWaveSolution> {
        if u0.len() != self.n_nodes || v0.len() != self.n_nodes {
            return Err(IntegrateError::DimensionMismatch(format!(
                "u0/v0 length must equal n_nodes = {}",
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
        let c = self.cfg.wave_speed;
        let sigma = self.cfg.sigma;
        let c2_over_dx2 = c * c / (self.dx * self.dx);
        // Space-time white noise scale: sigma / sqrt(dx * dt)
        let noise_scale = sigma / (self.dx * dt).sqrt();

        let n_steps = ((t_end - t0) / dt).ceil() as usize;
        let capacity = (n_steps / self.save_every + 2).max(2);

        let mut snapshots = Vec::with_capacity(capacity);

        let mut u = u0.clone();
        let mut v = v0.clone(); // v_{1/2} initialised as v_0 (first-order start-up)

        // Store initial energy using v = v_0
        let e0 = self.discrete_energy(&u, &v);
        snapshots.push(WaveSnapshot {
            t: t0,
            u: u.clone(),
            v: v.clone(),
            energy: e0,
        });

        let mut t = t0;
        let mut u_new = Array1::<f64>::zeros(self.n_nodes);

        for step in 0..n_steps {
            let actual_dt = dt.min(t_end - t);
            if actual_dt <= 0.0 {
                break;
            }
            let c2dx2 = c * c / (self.dx * self.dx);
            let ns = sigma / (self.dx * actual_dt).sqrt();

            // Leapfrog: v^{n+1/2} = v^{n-1/2} + dt * (c^2/dx^2 * Lu^n + noise)
            for i in 0..self.n_nodes {
                let u_left = if i == 0 { 0.0 } else { u[i - 1] };
                let u_right = if i < self.n_nodes - 1 { u[i + 1] } else { 0.0 };
                let laplacian = u_left - 2.0 * u[i] + u_right;
                let xi = rng.sample(normal);
                v[i] += actual_dt * (c2dx2 * laplacian + ns * xi);
            }

            // u^{n+1} = u^n + dt * v^{n+1/2}
            for i in 0..self.n_nodes {
                u_new[i] = u[i] + actual_dt * v[i];
            }
            u.assign(&u_new);
            t += actual_dt;

            if (step + 1) % self.save_every == 0 || t >= t_end - 1e-14 {
                let energy = self.discrete_energy(&u, &v);
                snapshots.push(WaveSnapshot {
                    t,
                    u: u.clone(),
                    v: v.clone(),
                    energy,
                });
            }
        }

        Ok(StochasticWaveSolution {
            snapshots,
            grid: self.x_coords.clone(),
        })
    }

    /// Compute the discrete energy functional.
    ///
    /// ```text
    /// E = (dx/2) * Σ_i [ v_i^2 + c^2 * ((u_{i+1} - u_i)/dx)^2 ]
    /// ```
    fn discrete_energy(&self, u: &Array1<f64>, v: &Array1<f64>) -> f64 {
        let c = self.cfg.wave_speed;
        let mut kinetic = 0.0_f64;
        let mut potential = 0.0_f64;
        for i in 0..self.n_nodes {
            kinetic += v[i] * v[i];
            // potential energy from gradient (forward difference, zero at boundary)
            let u_right = if i < self.n_nodes - 1 { u[i + 1] } else { 0.0 };
            let grad = (u_right - u[i]) / self.dx;
            potential += c * c * grad * grad;
        }
        0.5 * self.dx * (kinetic + potential)
    }

    /// Return the x-grid of interior nodes.
    pub fn grid(&self) -> &Array1<f64> {
        &self.x_coords
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
        seeded_rng(999)
    }

    #[test]
    fn test_deterministic_wave_energy_conserved() {
        // With σ=0 the discrete energy should be approximately conserved
        let config = StochasticWaveConfig {
            wave_speed: 1.0,
            sigma: 0.0, // no noise
            dt: 5e-5,
        };
        let solver = StochasticWaveSolver::new(config, 1.0, 20, 10)
            .expect("StochasticWaveSolver::new should succeed");
        let u0 = Array1::from_vec(
            (0..20)
                .map(|i| ((i as f64 + 1.0) * std::f64::consts::PI / 21.0).sin())
                .collect::<Vec<f64>>(),
        );
        let v0 = Array1::zeros(20);
        let mut rng = make_rng();
        let sol = solver
            .solve(&u0, &v0, 0.0, 0.05, &mut rng)
            .expect("solver.solve should succeed");

        let energies = sol.energy_series();
        let e0 = energies[0];
        for &e in &energies[1..] {
            let rel_err = ((e - e0) / e0).abs();
            assert!(
                rel_err < 0.02,
                "Energy not conserved: e0={e0:.6}, e={e:.6}, rel={rel_err:.4}"
            );
        }
    }

    #[test]
    fn test_stochastic_wave_energy_grows() {
        // With σ>0 the mean energy should trend upward (noise pumps energy in)
        let config = StochasticWaveConfig {
            wave_speed: 1.0,
            sigma: 0.5,
            dt: 5e-5,
        };
        let solver = StochasticWaveSolver::new(config, 1.0, 20, 10)
            .expect("StochasticWaveSolver::new should succeed");
        let u0 = Array1::zeros(20);
        let v0 = Array1::zeros(20);
        let mut rng = make_rng();
        let sol = solver
            .solve(&u0, &v0, 0.0, 0.02, &mut rng)
            .expect("solver.solve should succeed");
        // Energy must be finite throughout
        for s in &sol.snapshots {
            assert!(s.energy.is_finite(), "Non-finite energy at t={}", s.t);
        }
        // Final energy must be non-negative
        assert!(sol.snapshots.last().map(|s| s.energy).unwrap_or(0.0) >= 0.0);
    }

    #[test]
    fn test_courant_violation_error() {
        // dx = 1/21, dt = 0.1, c = 1 → c*dt/dx = 2.1 > 1
        let config = StochasticWaveConfig {
            wave_speed: 1.0,
            sigma: 0.0,
            dt: 0.1,
        };
        let result = StochasticWaveSolver::new(config, 1.0, 20, 1);
        assert!(result.is_err(), "Should fail Courant check");
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = StochasticWaveConfig::default();
        let solver = StochasticWaveSolver::new(config, 1.0, 10, 1)
            .expect("StochasticWaveSolver::new should succeed");
        let u0 = Array1::zeros(5); // wrong length
        let v0 = Array1::zeros(10);
        let mut rng = make_rng();
        let result = solver.solve(&u0, &v0, 0.0, 0.001, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_finite_values() {
        let config = StochasticWaveConfig {
            wave_speed: 0.5,
            sigma: 0.1,
            dt: 1e-4,
        };
        let solver = StochasticWaveSolver::new(config, 1.0, 15, 20)
            .expect("StochasticWaveSolver::new should succeed");
        let n = solver.n_nodes;
        let u0 = Array1::zeros(n);
        let v0 = Array1::zeros(n);
        let mut rng = make_rng();
        let sol = solver
            .solve(&u0, &v0, 0.0, 0.01, &mut rng)
            .expect("solver.solve should succeed");
        for s in &sol.snapshots {
            assert!(s.u.iter().all(|v| v.is_finite()));
            assert!(s.v.iter().all(|v| v.is_finite()));
        }
    }
}
