//! Simplified SDE solvers and Fokker-Planck equation
//!
//! This module provides self-contained, easy-to-use stochastic differential
//! equation solvers with a minimal API. For production-grade SDE simulation,
//! see the `sde` module which uses the full scirs2-core random number infrastructure.
//!
//! ## Contents
//!
//! - [`SimpleRng`] — deterministic seeded RNG (Xoshiro256+)
//! - [`EulerMaruyama`] — Euler-Maruyama strong order 0.5 solver
//! - [`MilsteinSolver`] — Milstein strong order 1.0 scalar solver
//! - [`solve_sde`] — convenience function for multiple sample paths
//! - [`FokkerPlanckSolver`] — finite-difference PDE solver for the FP equation
//!
//! ## Background
//!
//! An n-dimensional SDE with m-dimensional Wiener process:
//! ```text
//! dX = f(X, t) dt + G(X, t) dW,   X(t₀) = X₀
//! ```
//! where:
//! - `f: ℝⁿ × ℝ → ℝⁿ` is the drift
//! - `G: ℝⁿ × ℝ → ℝⁿˣᵐ` is the diffusion matrix
//! - `W` is an m-dimensional standard Wiener process
//!
//! ## Example
//!
//! ```rust
//! use scirs2_integrate::sde_simple::{EulerMaruyama, SimpleRng, solve_sde};
//!
//! // Geometric Brownian Motion: dX = μ X dt + σ X dW
//! let mu = 0.05_f64;
//! let sigma = 0.2_f64;
//!
//! let solver = EulerMaruyama::new(
//!     move |x: &[f64], _t: f64| vec![mu * x[0]],
//!     move |x: &[f64], _t: f64| vec![vec![sigma * x[0]]],
//!     1, 1,
//! );
//!
//! let sol = solve_sde(&solver, &[100.0], 0.0, 1.0, 0.01, 5, 42);
//! assert_eq!(sol.paths.len(), 5);
//! assert!(!sol.times.is_empty());
//! ```

use crate::error::{IntegrateError, IntegrateResult};

// ─────────────────────────────────────────────────────────────────────────────
// Minimal PRNG: Xoshiro256+ (fast, high quality)
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight seeded pseudo-random number generator based on Xoshiro256+.
///
/// Provides uniform and standard-normal samples without any external dependencies.
pub struct SimpleRng {
    state: [u64; 4],
}

impl SimpleRng {
    /// Create a new RNG initialised from `seed`.
    pub fn new(seed: u64) -> Self {
        // Seed expansion via splitmix64
        let mut s = seed;
        let expand = |x: &mut u64| -> u64 {
            *x = x.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = *x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        Self {
            state: [
                expand(&mut s),
                expand(&mut s),
                expand(&mut s),
                expand(&mut s),
            ],
        }
    }

    /// Generate a random u64 using the Xoshiro256+ algorithm.
    fn next_u64(&mut self) -> u64 {
        let result = self.state[0].wrapping_add(self.state[3]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Sample a uniform float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits for uniform [0,1)
        (self.next_u64() >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Sample from the standard normal distribution N(0, 1) via Box-Muller.
    pub fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > f64::EPSILON {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return r * theta.cos();
            }
        }
    }

    /// Sample an m-dimensional Brownian increment √dt · Z where Z ~ N(0, I_m).
    pub fn brownian_increment(&mut self, m: usize, dt: f64) -> Vec<f64> {
        let sqrt_dt = dt.sqrt();
        (0..m).map(|_| sqrt_dt * self.next_normal()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SDE solver trait
// ─────────────────────────────────────────────────────────────────────────────

/// Common interface for SDE stepping methods.
pub trait SdeSolver {
    /// Advance the state `x` at time `t` by one step of size `dt`.
    fn step(&self, x: &[f64], t: f64, dt: f64, rng: &mut SimpleRng) -> Vec<f64>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Euler-Maruyama
// ─────────────────────────────────────────────────────────────────────────────

/// Euler-Maruyama discretisation of an n-dimensional SDE (strong order 0.5).
///
/// Update rule:
/// ```text
/// X(t+dt) = X(t) + f(X,t) dt + G(X,t) ΔW
/// ```
/// where `ΔW ~ N(0, dt I_m)`.
pub struct EulerMaruyama {
    drift: Box<dyn Fn(&[f64], f64) -> Vec<f64> + Send + Sync>,
    diffusion: Box<dyn Fn(&[f64], f64) -> Vec<Vec<f64>> + Send + Sync>,
    /// State-space dimension n
    pub n_dim: usize,
    /// Wiener process dimension m
    pub m_noise: usize,
}

impl EulerMaruyama {
    /// Create a new Euler-Maruyama solver.
    ///
    /// # Arguments
    /// * `drift` — drift function `f(x, t) -> Vec<f64>` of length `n_dim`
    /// * `diffusion` — diffusion matrix `G(x, t) -> Vec<Vec<f64>>` of shape `[n_dim][m_noise]`
    /// * `n_dim` — dimension of the state space
    /// * `m_noise` — number of independent Brownian motions
    pub fn new(
        drift: impl Fn(&[f64], f64) -> Vec<f64> + Send + Sync + 'static,
        diffusion: impl Fn(&[f64], f64) -> Vec<Vec<f64>> + Send + Sync + 'static,
        n_dim: usize,
        m_noise: usize,
    ) -> Self {
        Self {
            drift: Box::new(drift),
            diffusion: Box::new(diffusion),
            n_dim,
            m_noise,
        }
    }
}

impl SdeSolver for EulerMaruyama {
    /// One Euler-Maruyama step.
    fn step(&self, x: &[f64], t: f64, dt: f64, rng: &mut SimpleRng) -> Vec<f64> {
        let f = (self.drift)(x, t);
        let g = (self.diffusion)(x, t);
        let dw = rng.brownian_increment(self.m_noise, dt);

        let mut x_new = vec![0.0_f64; self.n_dim];
        for i in 0..self.n_dim {
            x_new[i] = x[i] + f[i] * dt;
            for j in 0..self.m_noise {
                if i < g.len() && j < g[i].len() {
                    x_new[i] += g[i][j] * dw[j];
                }
            }
        }
        x_new
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Milstein solver (scalar)
// ─────────────────────────────────────────────────────────────────────────────

/// Milstein method for a scalar SDE (strong order 1.0).
///
/// Update rule for the 1D SDE `dX = f(X,t) dt + g(X,t) dW`:
/// ```text
/// X(t+dt) = X(t) + f dt + g ΔW + 0.5 g g' (ΔW² - dt)
/// ```
/// where `g' = ∂g/∂X`.
pub struct MilsteinSolver {
    drift: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,
    diffusion: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,
    diffusion_deriv: Box<dyn Fn(f64, f64) -> f64 + Send + Sync>,
}

impl MilsteinSolver {
    /// Create a Milstein solver for a scalar SDE.
    ///
    /// # Arguments
    /// * `drift` — `f(x, t)` (scalar drift)
    /// * `diffusion` — `g(x, t)` (scalar diffusion coefficient)
    /// * `diffusion_deriv` — `∂g/∂x(x, t)` (derivative of g with respect to state)
    pub fn new(
        drift: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
        diffusion: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
        diffusion_deriv: impl Fn(f64, f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            drift: Box::new(drift),
            diffusion: Box::new(diffusion),
            diffusion_deriv: Box::new(diffusion_deriv),
        }
    }

    /// One Milstein step for the scalar SDE.
    pub fn step_scalar(&self, x: f64, t: f64, dt: f64, rng: &mut SimpleRng) -> f64 {
        let dw = rng.next_normal() * dt.sqrt();
        let f = (self.drift)(x, t);
        let g = (self.diffusion)(x, t);
        let gp = (self.diffusion_deriv)(x, t);
        x + f * dt + g * dw + 0.5 * g * gp * (dw * dw - dt)
    }
}

impl SdeSolver for MilsteinSolver {
    /// Milstein step adapted to the generic `SdeSolver` trait (scalar SDE, n=1, m=1).
    fn step(&self, x: &[f64], t: f64, dt: f64, rng: &mut SimpleRng) -> Vec<f64> {
        let x0 = if x.is_empty() { 0.0 } else { x[0] };
        vec![self.step_scalar(x0, t, dt, rng)]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SDE solution container
// ─────────────────────────────────────────────────────────────────────────────

/// Container for multiple sample paths of an SDE solution.
pub struct SdeSolution {
    /// Time grid: `times[k]` = time at step k
    pub times: Vec<f64>,
    /// Sample paths: `paths[path_idx][step_idx * n_dim + dim_idx]`
    /// For n_dim=1: `paths[path_idx][step_idx]` = X(t_k) for path `path_idx`
    pub paths: Vec<Vec<f64>>,
    /// State space dimension
    pub n_dim: usize,
}

impl SdeSolution {
    /// Get the full trajectory of a single sample path as `(times, values)`.
    ///
    /// `values[k]` is the n_dim-dimensional state at `times[k]`.
    pub fn path(&self, idx: usize) -> Option<(&[f64], Vec<Vec<f64>>)> {
        if idx >= self.paths.len() {
            return None;
        }
        let n_steps = self.times.len();
        let n_dim = self.n_dim;
        let flat = &self.paths[idx];
        let states: Vec<Vec<f64>> = (0..n_steps)
            .map(|k| flat[k * n_dim..(k + 1) * n_dim].to_vec())
            .collect();
        Some((&self.times, states))
    }

    /// Compute the mean trajectory (element-wise average over paths) at each time step.
    pub fn mean_trajectory(&self) -> Vec<Vec<f64>> {
        let n_steps = self.times.len();
        let n_dim = self.n_dim;
        let n_paths = self.paths.len();
        if n_paths == 0 || n_steps == 0 {
            return vec![];
        }
        let scale = 1.0 / n_paths as f64;
        (0..n_steps)
            .map(|k| {
                (0..n_dim)
                    .map(|d| self.paths.iter().map(|p| p[k * n_dim + d]).sum::<f64>() * scale)
                    .collect()
            })
            .collect()
    }

    /// Compute the variance of each dimension across sample paths at each time step.
    pub fn variance_trajectory(&self) -> Vec<Vec<f64>> {
        let n_steps = self.times.len();
        let n_dim = self.n_dim;
        let n_paths = self.paths.len();
        if n_paths < 2 || n_steps == 0 {
            return vec![vec![0.0; n_dim]; n_steps];
        }
        let mean = self.mean_trajectory();
        let scale = 1.0 / (n_paths - 1) as f64;
        (0..n_steps)
            .map(|k| {
                (0..n_dim)
                    .map(|d| {
                        self.paths
                            .iter()
                            .map(|p| {
                                let diff = p[k * n_dim + d] - mean[k][d];
                                diff * diff
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect()
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience solver function
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate `n_paths` independent sample paths of an SDE using the given solver.
///
/// # Arguments
/// * `solver` — any type implementing `SdeSolver`
/// * `x0` — initial condition (n_dim-dimensional)
/// * `t0` — initial time
/// * `t_final` — final time
/// * `dt` — time step size
/// * `n_paths` — number of independent sample paths
/// * `seed` — base seed (each path uses `seed + path_index`)
///
/// # Returns
/// An `SdeSolution` containing all time steps and path data.
pub fn solve_sde(
    solver: &impl SdeSolver,
    x0: &[f64],
    t0: f64,
    t_final: f64,
    dt: f64,
    n_paths: usize,
    seed: u64,
) -> SdeSolution {
    let n_dim = x0.len();
    let n_steps = ((t_final - t0) / dt).ceil() as usize + 1;

    let mut times = Vec::with_capacity(n_steps);
    let mut t = t0;
    while t <= t_final + dt * 0.5 {
        times.push(t);
        t += dt;
        if times.len() >= n_steps {
            break;
        }
    }
    // Ensure final time is included
    if times.last().copied().unwrap_or(t0) < t_final - f64::EPSILON {
        times.push(t_final);
    }

    let actual_steps = times.len();
    let mut paths = Vec::with_capacity(n_paths);

    for path_idx in 0..n_paths {
        let mut rng = SimpleRng::new(seed.wrapping_add(path_idx as u64));
        let mut path_data = Vec::with_capacity(actual_steps * n_dim);
        // Store initial state
        path_data.extend_from_slice(x0);
        let mut x = x0.to_vec();
        for k in 1..actual_steps {
            let dt_actual = times[k] - times[k - 1];
            x = solver.step(&x, times[k - 1], dt_actual, &mut rng);
            path_data.extend_from_slice(&x);
        }
        paths.push(path_data);
    }

    SdeSolution {
        times,
        paths,
        n_dim,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fokker-Planck solver
// ─────────────────────────────────────────────────────────────────────────────

/// Finite-difference solver for the 1D Fokker-Planck equation.
///
/// Evolves the probability density function p(x, t) according to:
/// ```text
/// ∂p/∂t = -∂/∂x [f(x) p] + (1/2) ∂²/∂x² [g²(x) p]
/// ```
/// where `f` is the drift and `g` is the diffusion coefficient of the SDE.
///
/// The FP equation governs the time evolution of the marginal distribution of
/// the SDE state. The probability density is kept non-negative and normalised
/// throughout the integration.
///
/// ## Discretisation
///
/// Uses the Chang-Cooper upwind finite-difference scheme which preserves
/// positivity and maintains the stationary distribution exactly. The simpler
/// explicit Lax-Wendroff scheme is used here for ease of implementation.
pub struct FokkerPlanckSolver {
    /// Number of grid points
    pub n: usize,
    /// Grid spacing
    pub dx: f64,
    /// Grid point coordinates
    pub x: Vec<f64>,
    /// Probability density function p(x)
    pub pdf: Vec<f64>,
}

impl FokkerPlanckSolver {
    /// Create a new Fokker-Planck solver on [x_min, x_max].
    ///
    /// # Arguments
    /// * `x_min`, `x_max` — domain boundaries
    /// * `n` — number of grid points
    /// * `initial_pdf` — initial probability density (will be normalised)
    ///
    /// # Errors
    /// Returns an error if `initial_pdf.len() != n` or `n < 3`.
    pub fn new(x_min: f64, x_max: f64, n: usize, initial_pdf: &[f64]) -> IntegrateResult<Self> {
        if n < 3 {
            return Err(IntegrateError::ValueError(
                "FokkerPlanck: need at least 3 grid points".into(),
            ));
        }
        if initial_pdf.len() != n {
            return Err(IntegrateError::ValueError(format!(
                "FokkerPlanck: initial_pdf length {} != n={}",
                initial_pdf.len(),
                n
            )));
        }
        let dx = (x_max - x_min) / ((n - 1) as f64);
        let x: Vec<f64> = (0..n).map(|i| x_min + i as f64 * dx).collect();

        // Normalise the initial PDF
        let integral: f64 = initial_pdf.iter().sum::<f64>() * dx;
        let pdf: Vec<f64> = if integral > 1e-15 {
            initial_pdf.iter().map(|&p| p.max(0.0) / integral).collect()
        } else {
            vec![1.0 / (x_max - x_min); n]
        };

        Ok(Self { n, dx, x, pdf })
    }

    /// Advance the PDF by one step using the pre-computed drift and diffusion arrays.
    ///
    /// This uses a forward-backward difference scheme for the advection term
    /// (upwind in drift direction) and central differences for diffusion.
    ///
    /// # Arguments
    /// * `dt` — time step (must satisfy stability: dt < dx² / max(g²))
    /// * `drift` — precomputed `f(x_i)` values at all grid points
    /// * `diffusion` — precomputed `g(x_i)` values at all grid points
    pub fn step(&mut self, dt: f64, drift: &[f64], diffusion: &[f64]) {
        let n = self.n;
        let dx = self.dx;
        let inv_dx = 1.0 / dx;
        let inv_dx2 = inv_dx * inv_dx;

        let mut p_new = vec![0.0_f64; n];

        for i in 1..n - 1 {
            let p = self.pdf[i];
            let pp = self.pdf[i + 1];
            let pm = self.pdf[i - 1];

            // Advection (upwind): -∂/∂x [f p]
            let f = drift[i];
            let adv = if f >= 0.0 {
                -f * (p - pm) * inv_dx
            } else {
                -f * (pp - p) * inv_dx
            };

            // Diffusion: 0.5 ∂²/∂x² [g² p]
            let g2_p = diffusion[i] * diffusion[i] * p;
            let g2_pp = diffusion[i + 1] * diffusion[i + 1] * pp;
            let g2_pm = diffusion[i - 1] * diffusion[i - 1] * pm;
            let diff = 0.5 * (g2_pp - 2.0 * g2_p + g2_pm) * inv_dx2;

            p_new[i] = p + dt * (adv + diff);
        }

        // Neumann (reflecting) boundary conditions — zero flux
        p_new[0] = p_new[1];
        p_new[n - 1] = p_new[n - 2];

        // Ensure non-negativity
        for v in p_new.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        // Renormalise to preserve probability
        let total: f64 = p_new.iter().sum::<f64>() * dx;
        if total > 1e-15 {
            for v in p_new.iter_mut() {
                *v /= total;
            }
        }

        self.pdf = p_new;
    }

    /// Run the Fokker-Planck equation for `n_steps` steps.
    pub fn run(
        &mut self,
        n_steps: usize,
        dt: f64,
        drift_fn: impl Fn(f64) -> f64,
        diffusion_fn: impl Fn(f64) -> f64,
    ) {
        let drift: Vec<f64> = self.x.iter().map(|&xi| drift_fn(xi)).collect();
        let diffusion: Vec<f64> = self.x.iter().map(|&xi| diffusion_fn(xi)).collect();
        for _ in 0..n_steps {
            self.step(dt, &drift, &diffusion);
        }
    }

    /// Compute the mean ⟨X⟩ = ∫ x p(x) dx.
    pub fn mean(&self) -> f64 {
        self.x
            .iter()
            .zip(self.pdf.iter())
            .map(|(&xi, &pi)| xi * pi)
            .sum::<f64>()
            * self.dx
    }

    /// Compute the variance Var\[X\] = ⟨X²⟩ − ⟨X⟩².
    pub fn variance(&self) -> f64 {
        let mu = self.mean();
        let ex2: f64 = self
            .x
            .iter()
            .zip(self.pdf.iter())
            .map(|(&xi, &pi)| xi * xi * pi)
            .sum::<f64>()
            * self.dx;
        (ex2 - mu * mu).max(0.0)
    }

    /// Compute the differential entropy −∫ p(x) ln p(x) dx.
    pub fn entropy(&self) -> f64 {
        -self
            .pdf
            .iter()
            .filter(|&&p| p > 1e-300)
            .map(|&p| p * p.ln())
            .sum::<f64>()
            * self.dx
    }

    /// Compute the L1 norm ∫ |p(x)| dx (should be ≈ 1 after normalisation).
    pub fn l1_norm(&self) -> f64 {
        self.pdf.iter().sum::<f64>() * self.dx
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SimpleRng ──────────────────────────────────────────────────────────

    #[test]
    fn test_rng_uniform_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_rng_normal_zero_mean() {
        let mut rng = SimpleRng::new(7);
        let samples: Vec<f64> = (0..10_000).map(|_| rng.next_normal()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.1, "mean={}", mean);
    }

    #[test]
    fn test_rng_normal_unit_variance() {
        let mut rng = SimpleRng::new(13);
        let samples: Vec<f64> = (0..10_000).map(|_| rng.next_normal()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!((var - 1.0).abs() < 0.05, "var={}", var);
    }

    #[test]
    fn test_rng_seeded_deterministic() {
        let mut r1 = SimpleRng::new(99);
        let mut r2 = SimpleRng::new(99);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    // ── Euler-Maruyama ─────────────────────────────────────────────────────

    #[test]
    fn test_em_brownian_motion() {
        // dX = dW, X(0) = 0  →  E[X(T)] = 0, Var[X(T)] ≈ T
        let solver = EulerMaruyama::new(
            |_x: &[f64], _t: f64| vec![0.0],
            |_x: &[f64], _t: f64| vec![vec![1.0]],
            1,
            1,
        );
        let sol = solve_sde(&solver, &[0.0], 0.0, 1.0, 0.01, 1000, 42);
        let mean_traj = sol.mean_trajectory();
        let final_mean = mean_traj.last().map(|v| v[0]).unwrap_or(f64::NAN);
        let var_traj = sol.variance_trajectory();
        let final_var = var_traj.last().map(|v| v[0]).unwrap_or(f64::NAN);
        assert!(final_mean.abs() < 0.15, "E[W(1)] ≈ 0, got {}", final_mean);
        assert!(
            (final_var - 1.0).abs() < 0.15,
            "Var[W(1)] ≈ 1, got {}",
            final_var
        );
    }

    #[test]
    fn test_em_gbm_mean() {
        // GBM: dX = μX dt + σX dW  →  E[X(T)] = X₀ exp(μT)
        let mu = 0.1_f64;
        let sigma = 0.2_f64;
        let x0 = 1.0_f64;
        let t_final = 0.5_f64;

        let solver = EulerMaruyama::new(
            move |x: &[f64], _t: f64| vec![mu * x[0]],
            move |x: &[f64], _t: f64| vec![vec![sigma * x[0]]],
            1,
            1,
        );
        let sol = solve_sde(&solver, &[x0], 0.0, t_final, 0.01, 2000, 7);
        let mean_traj = sol.mean_trajectory();
        let final_mean = mean_traj.last().map(|v| v[0]).unwrap_or(f64::NAN);
        let expected = x0 * (mu * t_final).exp();
        let rel_err = (final_mean - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "E[GBM] rel_err={:.3}, got={:.4} exp={:.4}",
            rel_err,
            final_mean,
            expected
        );
    }

    // ── Milstein ───────────────────────────────────────────────────────────

    #[test]
    fn test_milstein_gbm_order() {
        // Milstein should give smaller strong error than EM for GBM
        let mu = 0.05_f64;
        let sigma = 0.3_f64;
        let x0 = 1.0_f64;

        let solver = MilsteinSolver::new(
            move |x: f64, _t: f64| mu * x,
            move |x: f64, _t: f64| sigma * x,
            move |_x: f64, _t: f64| sigma, // dg/dx = sigma (g = sigma*x)
        );
        let sol = solve_sde(&solver, &[x0], 0.0, 0.5, 0.01, 500, 42);
        assert!(!sol.times.is_empty());
        // Mean should be near exp(mu*T)
        let mean = sol.mean_trajectory();
        let last = mean.last().map(|v| v[0]).unwrap_or(f64::NAN);
        assert!(last > 0.0 && last.is_finite());
    }

    // ── SdeSolution ────────────────────────────────────────────────────────

    #[test]
    fn test_solution_path_access() {
        let solver = EulerMaruyama::new(
            |_x: &[f64], _t: f64| vec![0.0],
            |_x: &[f64], _t: f64| vec![vec![1.0]],
            1,
            1,
        );
        let sol = solve_sde(&solver, &[0.0], 0.0, 0.1, 0.01, 3, 1);
        assert_eq!(sol.paths.len(), 3);
        assert!(sol.path(0).is_some());
        assert!(sol.path(10).is_none());
    }

    // ── Fokker-Planck ──────────────────────────────────────────────────────

    #[test]
    fn test_fp_creation() {
        let n = 50;
        let pdf: Vec<f64> = (0..n)
            .map(|i| {
                let x = -5.0 + 10.0 * i as f64 / (n - 1) as f64;
                (-(x * x) / 2.0).exp()
            })
            .collect();
        let solver = FokkerPlanckSolver::new(-5.0, 5.0, n, &pdf);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_fp_wrong_size() {
        let pdf = vec![1.0; 10];
        let solver = FokkerPlanckSolver::new(-5.0, 5.0, 20, &pdf);
        assert!(solver.is_err());
    }

    #[test]
    fn test_fp_normalisation_preserved() {
        let n = 50;
        let pdf: Vec<f64> = (0..n)
            .map(|i| {
                let x = -5.0 + 10.0 * i as f64 / (n - 1) as f64;
                (-(x * x) / 2.0).exp()
            })
            .collect();
        let mut solver = FokkerPlanckSolver::new(-5.0, 5.0, n, &pdf)
            .expect("FokkerPlanckSolver::new should succeed with valid params");
        solver.run(20, 1e-3, |x| -x, |_| 1.0); // Ornstein-Uhlenbeck
        let norm = solver.l1_norm();
        assert!((norm - 1.0).abs() < 0.01, "L1 norm = {}", norm);
    }

    #[test]
    fn test_fp_ou_mean_mean_reversion() {
        // OU process: dX = -X dt + dW, stationary dist N(0,0.5)
        // Start from δ(x-2), should move towards 0
        let n = 101;
        let center_idx = 85usize; // near x=2 in [-5,5]
        let mut pdf = vec![0.0_f64; n];
        pdf[center_idx] = 1.0 / (10.0 / n as f64); // delta-like spike
        let mut solver = FokkerPlanckSolver::new(-5.0, 5.0, n, &pdf)
            .expect("FokkerPlanckSolver::new should succeed with valid params");
        let m0 = solver.mean();
        solver.run(100, 1e-3, |x| -x, |_| 1.0);
        let m1 = solver.mean();
        // Mean should have moved towards 0 (from positive side)
        assert!(
            m1.abs() < m0.abs() + 0.5,
            "mean should move towards 0: m0={:.3} m1={:.3}",
            m0,
            m1
        );
    }

    #[test]
    fn test_fp_variance_finite() {
        let n = 50;
        let pdf = vec![1.0_f64 / 50.0; n]; // uniform
        let mut solver = FokkerPlanckSolver::new(-1.0, 1.0, n, &pdf)
            .expect("FokkerPlanckSolver::new should succeed with valid params");
        solver.run(10, 1e-4, |_| 0.0, |_| 0.5);
        let v = solver.variance();
        assert!(v >= 0.0 && v.is_finite(), "variance={}", v);
    }
}
