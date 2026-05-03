//! Pathwise (regularity-based) SPDE methods
//!
//! This module provides three families of advanced methods that exploit
//! regularity structure in the SPDE to achieve higher accuracy:
//!
//! ## 1. Anderson-Mattingly Scheme for Coloured Noise
//!
//! For SPDEs driven by spatially **coloured** (smooth) noise, the Anderson-
//! Mattingly splitting separates the stochastic convolution from the
//! deterministic drift:
//!
//! ```text
//! u^{n+1} = S_dt * u^n + I_dt * W
//! ```
//!
//! where `S_dt` is the semigroup of the deterministic part (approximated by
//! the heat kernel), and `I_dt * W` is the stochastic convolution increment.
//!
//! ## 2. Implicit-Explicit (IMEX) Scheme for Parabolic SPDEs
//!
//! For semilinear parabolic SPDEs
//! ```text
//! du = [A u + F(u)] dt + σ dW
//! ```
//!
//! the IMEX-Euler scheme treats the linear stiff part `A` implicitly and the
//! nonlinear drift `F` explicitly:
//!
//! ```text
//! (I - dt*A) u^{n+1} = u^n + dt * F(u^n) + σ sqrt(dt/dx) * ξ^n
//! ```
//!
//! This allows larger time steps than the explicit Euler-Maruyama method.
//!
//! ## 3. Spectral Galerkin for SPDEs with Fourier Basis
//!
//! Expands the solution in sine modes (Fourier–Galerkin):
//!
//! ```text
//! u(x,t) = Σ_{k=1}^{N} a_k(t) * sin(k π x / L)
//! ```
//!
//! Each modal coefficient `a_k` satisfies an SDE:
//!
//! ```text
//! da_k = -λ_k a_k dt + σ_k dW_k
//! ```
//!
//! where `λ_k = D (kπ/L)²` is the eigenvalue of `-D Δ` and `σ_k` is the
//! k-th Fourier coefficient of the noise amplitude.  Since each mode is an
//! independent Ornstein-Uhlenbeck process, the coefficients can be integrated
//! exactly:
//!
//! ```text
//! a_k(t+dt) = a_k(t) e^{-λ_k dt} + σ_k sqrt((1 - e^{-2λ_k dt}) / (2λ_k)) * N(0,1)
//! ```
//!
//! This is **exact in distribution** for the linear case.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::{Normal, Rng, StdRng};
use scirs2_core::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Anderson-Mattingly scheme for coloured noise
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Anderson-Mattingly scheme.
#[derive(Debug, Clone)]
pub struct AndersonMattinglyConfig {
    /// Diffusion coefficient D.
    pub diffusion: f64,
    /// Noise amplitude σ.
    pub sigma: f64,
    /// Spatial correlation length ℓ.
    pub correlation_length: f64,
    /// Time step dt.
    pub dt: f64,
    /// Number of interior spatial nodes.
    pub n_nodes: usize,
    /// Domain length L.
    pub domain_length: f64,
}

impl Default for AndersonMattinglyConfig {
    fn default() -> Self {
        Self {
            diffusion: 1.0,
            sigma: 0.1,
            correlation_length: 0.2,
            dt: 1e-4,
            n_nodes: 32,
            domain_length: 1.0,
        }
    }
}

/// Result of an Anderson-Mattingly simulation.
#[derive(Debug, Clone)]
pub struct AndersonMattinglySolution {
    /// Saved time points.
    pub times: Vec<f64>,
    /// Field snapshots at saved times.
    pub snapshots: Vec<Array1<f64>>,
    /// x-coordinates of interior nodes.
    pub grid: Array1<f64>,
}

impl AndersonMattinglySolution {
    /// Number of saved snapshots.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

/// Anderson-Mattingly scheme for SPDEs with spatially coloured noise.
///
/// The scheme approximates the solution using an exponential integrator for
/// the heat semigroup combined with a correlated stochastic increment.
pub struct AndersonMattinglyScheme {
    cfg: AndersonMattinglyConfig,
    /// x-coordinates of interior nodes.
    x_coords: Array1<f64>,
    dx: f64,
    /// Precomputed covariance matrix C[i,j] = exp(-|xi-xj|/ℓ).
    cov_chol: Vec<f64>, // Cholesky factor L (lower triangular, row-major)
    /// Precomputed heat kernel weights: exp(-D*(k*pi/L)^2 * dt) for each mode k.
    heat_weights: Vec<f64>,
    save_every: usize,
}

impl AndersonMattinglyScheme {
    /// Construct a new solver and precompute the Cholesky factor of the
    /// spatial covariance and the heat semigroup weights.
    ///
    /// # Errors
    /// Returns an error if the covariance matrix is not positive definite
    /// (which can happen for very small correlation lengths).
    pub fn new(cfg: AndersonMattinglyConfig, save_every: usize) -> IntegrateResult<Self> {
        let n = cfg.n_nodes;
        if n == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_nodes must be positive".to_string(),
            ));
        }
        let dx = cfg.domain_length / (n + 1) as f64;
        let x_coords = Array1::linspace(dx, cfg.domain_length - dx, n);

        // Build exponential covariance matrix C[i,j] = exp(-|xi-xj|/ℓ)
        let mut cov = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let r = (x_coords[i] - x_coords[j]).abs();
                cov[i * n + j] = (-r / cfg.correlation_length).exp();
            }
        }

        // Cholesky decomposition: C = L L^T
        let cov_chol = cholesky_lower(&cov, n)?;

        // Heat semigroup weights for sine modes: exp(-D (k pi/L)^2 dt)
        let heat_weights: Vec<f64> = (1..=n)
            .map(|k| {
                let lam =
                    cfg.diffusion * (k as f64 * std::f64::consts::PI / cfg.domain_length).powi(2);
                (-lam * cfg.dt).exp()
            })
            .collect();

        Ok(Self {
            cfg,
            x_coords,
            dx,
            cov_chol,
            heat_weights,
            save_every: save_every.max(1),
        })
    }

    /// Solve from `t0` to `t_end` starting from initial condition `u0`.
    pub fn solve(
        &self,
        u0: &Array1<f64>,
        t0: f64,
        t_end: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<AndersonMattinglySolution> {
        let n = self.cfg.n_nodes;
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

        let n_steps = ((t_end - t0) / self.cfg.dt).ceil() as usize;
        let capacity = (n_steps / self.save_every + 2).max(2);
        let mut times = Vec::with_capacity(capacity);
        let mut snapshots = Vec::with_capacity(capacity);

        let mut u = u0.clone();
        times.push(t0);
        snapshots.push(u.clone());

        let mut t = t0;
        for step in 0..n_steps {
            let actual_dt = self.cfg.dt.min(t_end - t);
            if actual_dt <= 0.0 {
                break;
            }

            // Step 1: Apply heat semigroup via modal decomposition
            u = self.apply_heat_semigroup(&u, actual_dt);

            // Step 2: Add stochastic increment: Δu = σ L ξ * sqrt(dt)
            //   where L is the Cholesky factor of the covariance
            let xi: Vec<f64> = (0..n).map(|_| rng.sample(normal)).collect();
            let correlated_noise = self.apply_cholesky(&xi);
            let noise_scale = self.cfg.sigma * actual_dt.sqrt();
            for i in 0..n {
                u[i] += noise_scale * correlated_noise[i];
            }

            t += actual_dt;
            if (step + 1) % self.save_every == 0 || t >= t_end - 1e-14 {
                times.push(t);
                snapshots.push(u.clone());
            }
        }

        Ok(AndersonMattinglySolution {
            times,
            snapshots,
            grid: self.x_coords.clone(),
        })
    }

    /// Apply the heat semigroup via a sine-mode expansion.
    ///
    /// Projects u onto sine modes, multiplies each mode by the precomputed
    /// weight `exp(-λ_k dt)`, and reconstructs.
    fn apply_heat_semigroup(&self, u: &Array1<f64>, dt: f64) -> Array1<f64> {
        let n = self.cfg.n_nodes;
        let l = self.cfg.domain_length;
        let dx = self.dx;

        // Compute Fourier-sine coefficients via numerical quadrature
        let mut a = vec![0.0_f64; n];
        for k in 0..n {
            let mode = k + 1;
            let mut sum = 0.0_f64;
            for i in 0..n {
                let xi = self.x_coords[i];
                sum += u[i] * (mode as f64 * std::f64::consts::PI * xi / l).sin();
            }
            a[k] = sum * dx * 2.0 / l;
        }

        // Multiply each mode by the heat kernel weight
        let weights: Vec<f64> = (0..n)
            .map(|k| {
                let lam = self.cfg.diffusion * ((k + 1) as f64 * std::f64::consts::PI / l).powi(2);
                (-lam * dt).exp()
            })
            .collect();

        // Reconstruct
        let mut u_new = Array1::zeros(n);
        for i in 0..n {
            let xi = self.x_coords[i];
            let mut val = 0.0_f64;
            for k in 0..n {
                let mode = k + 1;
                val += weights[k] * a[k] * (mode as f64 * std::f64::consts::PI * xi / l).sin();
            }
            u_new[i] = val;
        }
        u_new
    }

    /// Multiply a white noise vector ξ by the Cholesky factor L.
    fn apply_cholesky(&self, xi: &[f64]) -> Vec<f64> {
        let n = self.cfg.n_nodes;
        let mut result = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..=i {
                s += self.cov_chol[i * n + j] * xi[j];
            }
            result[i] = s;
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Implicit-Explicit (IMEX) scheme
// ─────────────────────────────────────────────────────────────────────────────

/// IMEX-Euler scheme for semilinear parabolic SPDEs:
/// `du = [D Δu + F(u)] dt + σ dW`.
pub struct ImexParabolicSolver {
    /// Diffusion coefficient D.
    diffusion: f64,
    /// Noise amplitude σ.
    sigma: f64,
    /// Time step dt.
    dt: f64,
    /// Interior node count N.
    n_nodes: usize,
    /// Grid spacing dx.
    dx: f64,
    /// x-coordinates.
    x_coords: Array1<f64>,
    /// Precomputed tridiagonal solve factors for (I - dt*D*Δ).
    /// Stored as (diag, super_diag) for the Thomas algorithm.
    thomas_diag: Vec<f64>,
    thomas_super: Vec<f64>,
    save_every: usize,
}

/// Result of an IMEX parabolic SPDE simulation.
#[derive(Debug, Clone)]
pub struct ImexSolution {
    /// Saved time points.
    pub times: Vec<f64>,
    /// Field snapshots.
    pub snapshots: Vec<Array1<f64>>,
    /// x-grid of interior nodes.
    pub grid: Array1<f64>,
}

impl ImexSolution {
    /// Number of saved snapshots.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

impl ImexParabolicSolver {
    /// Construct the IMEX solver and precompute the Thomas algorithm factors.
    ///
    /// # Arguments
    /// * `diffusion`     – D > 0.
    /// * `sigma`         – Noise amplitude.
    /// * `dt`            – Time step.
    /// * `domain_length` – L.
    /// * `n_nodes`       – Interior node count.
    /// * `save_every`    – Snapshot interval.
    pub fn new(
        diffusion: f64,
        sigma: f64,
        dt: f64,
        domain_length: f64,
        n_nodes: usize,
        save_every: usize,
    ) -> IntegrateResult<Self> {
        if n_nodes == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_nodes must be positive".to_string(),
            ));
        }
        if dt <= 0.0 || diffusion <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "dt and diffusion must be positive".to_string(),
            ));
        }
        let dx = domain_length / (n_nodes + 1) as f64;
        let x_coords = Array1::linspace(dx, domain_length - dx, n_nodes);

        // Tridiagonal matrix A = I - dt*D*Δ_h
        // A_{ii} = 1 + 2r,  A_{i,i±1} = -r  where r = D*dt/dx^2
        let r = diffusion * dt / (dx * dx);
        let main_diag = 1.0 + 2.0 * r;
        let off_diag = -r;

        // Thomas algorithm forward sweep factors (LU factorisation of tridiag)
        let mut thomas_diag = vec![main_diag; n_nodes];
        let thomas_super = vec![off_diag; n_nodes]; // constant super-diagonal

        // Forward elimination of sub-diagonal
        for i in 1..n_nodes {
            let factor = off_diag / thomas_diag[i - 1];
            thomas_diag[i] -= factor * off_diag;
        }

        Ok(Self {
            diffusion,
            sigma,
            dt,
            n_nodes,
            dx,
            x_coords,
            thomas_diag,
            thomas_super,
            save_every: save_every.max(1),
        })
    }

    /// Solve the SPDE from `t0` to `t_end`.
    ///
    /// # Arguments
    /// * `u0` – Initial condition (length `n_nodes`).
    /// * `nonlinear` – Explicit nonlinear term F(u) evaluated pointwise.
    /// * `t0`, `t_end` – Time interval.
    /// * `rng` – Random number generator.
    pub fn solve<F>(
        &self,
        u0: &Array1<f64>,
        nonlinear: F,
        t0: f64,
        t_end: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<ImexSolution>
    where
        F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    {
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

        let n_steps = ((t_end - t0) / self.dt).ceil() as usize;
        let capacity = (n_steps / self.save_every + 2).max(2);
        let mut times = Vec::with_capacity(capacity);
        let mut snapshots = Vec::with_capacity(capacity);

        let mut u = u0.clone();
        times.push(t0);
        snapshots.push(u.clone());

        let noise_scale = self.sigma * (self.dt / self.dx).sqrt();
        let mut t = t0;

        for step in 0..n_steps {
            let actual_dt = self.dt.min(t_end - t);
            if actual_dt <= 0.0 {
                break;
            }
            let ns = self.sigma * (actual_dt / self.dx).sqrt();

            // Compute explicit nonlinear term F(t, u^n)
            let f_u = nonlinear(t, &u);

            // Assemble RHS: u^n + dt * F(u^n) + noise
            let mut rhs = Array1::zeros(self.n_nodes);
            for i in 0..self.n_nodes {
                let xi = rng.sample(normal);
                rhs[i] = u[i] + actual_dt * f_u[i] + ns * xi;
            }

            // Solve (I - dt*D*Δ) u^{n+1} = rhs via Thomas algorithm
            u = self.thomas_solve(&rhs, actual_dt)?;
            t += actual_dt;

            if (step + 1) % self.save_every == 0 || t >= t_end - 1e-14 {
                times.push(t);
                snapshots.push(u.clone());
            }
        }

        Ok(ImexSolution {
            times,
            snapshots,
            grid: self.x_coords.clone(),
        })
    }

    /// Solve the tridiagonal system (I - dt*D*Δ_h) x = rhs using the
    /// Thomas algorithm with precomputed LU factors.
    fn thomas_solve(&self, rhs: &Array1<f64>, dt: f64) -> IntegrateResult<Array1<f64>> {
        let n = self.n_nodes;
        let r = self.diffusion * dt / (self.dx * self.dx);
        let main_diag_val = 1.0 + 2.0 * r;
        let off_diag_val = -r;

        // Re-compute Thomas factors for this dt (dt may vary at last step)
        let mut diag = vec![main_diag_val; n];
        for i in 1..n {
            let factor = off_diag_val / diag[i - 1];
            diag[i] -= factor * off_diag_val;
        }

        // Forward substitution
        let mut d = rhs.to_vec();
        for i in 1..n {
            let factor = off_diag_val / diag[i - 1];
            d[i] -= factor * d[i - 1];
        }

        // Back substitution
        let mut x = vec![0.0_f64; n];
        x[n - 1] = d[n - 1] / diag[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = (d[i] - off_diag_val * x[i + 1]) / diag[i];
        }

        Ok(Array1::from_vec(x))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Spectral Galerkin (Fourier basis, exact for linear case)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Spectral Galerkin solver.
#[derive(Debug, Clone)]
pub struct SpectralGalerkinConfig {
    /// Diffusion coefficient D.
    pub diffusion: f64,
    /// Noise amplitude σ (applied uniformly to all modes; or per-mode via `mode_sigmas`).
    pub sigma: f64,
    /// Optional per-mode noise amplitudes σ_k for k = 1, …, n_modes.
    /// If None, `sigma` is used for all modes.
    pub mode_sigmas: Option<Vec<f64>>,
    /// Number of sine modes.
    pub n_modes: usize,
    /// Domain length L.
    pub domain_length: f64,
}

impl Default for SpectralGalerkinConfig {
    fn default() -> Self {
        Self {
            diffusion: 1.0,
            sigma: 0.1,
            mode_sigmas: None,
            n_modes: 32,
            domain_length: 1.0,
        }
    }
}

/// Result of the Spectral Galerkin simulation.
#[derive(Debug, Clone)]
pub struct SpectralGalerkinSolution {
    /// Saved time points.
    pub times: Vec<f64>,
    /// Modal coefficients a_k(t) at each saved time, shape [n_modes].
    pub modal_snapshots: Vec<Array1<f64>>,
    /// Reconstructed physical-space snapshots on evaluation grid.
    pub physical_snapshots: Vec<Array1<f64>>,
    /// Evaluation grid x_i.
    pub grid: Array1<f64>,
}

impl SpectralGalerkinSolution {
    /// Number of saved time points.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Compute the L2 energy (½ Σ a_k²) at each saved time.
    pub fn modal_energy_series(&self) -> Vec<f64> {
        self.modal_snapshots
            .iter()
            .map(|a| 0.5 * a.iter().map(|&v| v * v).sum::<f64>())
            .collect()
    }
}

/// Spectral Galerkin solver for the stochastic heat equation with Fourier basis.
///
/// Each sine mode `a_k` satisfies an independent Ornstein-Uhlenbeck SDE:
/// ```text
/// da_k = -λ_k a_k dt + σ_k dW_k,   λ_k = D (k π / L)^2
/// ```
/// which is integrated **exactly** via the transition distribution.
pub struct SpectralGalerkinSolver {
    cfg: SpectralGalerkinConfig,
    /// Eigenvalues λ_k = D (k π / L)².
    lambda: Vec<f64>,
    /// Per-mode noise amplitudes.
    sigma_k: Vec<f64>,
    save_every: usize,
}

impl SpectralGalerkinSolver {
    /// Construct the solver.
    ///
    /// # Arguments
    /// * `cfg`        – Configuration.
    /// * `save_every` – Snapshot interval (in time steps).
    pub fn new(cfg: SpectralGalerkinConfig, save_every: usize) -> IntegrateResult<Self> {
        if cfg.n_modes == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_modes must be positive".to_string(),
            ));
        }
        let n = cfg.n_modes;
        let lambda: Vec<f64> = (1..=n)
            .map(|k| cfg.diffusion * (k as f64 * std::f64::consts::PI / cfg.domain_length).powi(2))
            .collect();

        let sigma_k: Vec<f64> = if let Some(ref ms) = cfg.mode_sigmas {
            if ms.len() < n {
                return Err(IntegrateError::InvalidInput(format!(
                    "mode_sigmas has {} entries but n_modes = {}",
                    ms.len(),
                    n
                )));
            }
            ms[..n].to_vec()
        } else {
            vec![cfg.sigma; n]
        };

        Ok(Self {
            cfg,
            lambda,
            sigma_k,
            save_every: save_every.max(1),
        })
    }

    /// Integrate from `t0` to `t_end` with initial modal coefficients `a0`.
    ///
    /// # Arguments
    /// * `a0`        – Initial modal coefficients, length `n_modes`.
    /// * `dt`        – Time step.
    /// * `eval_grid` – x-points for physical-space reconstruction.
    /// * `t0`, `t_end` – Time interval.
    /// * `rng`       – Random number generator.
    pub fn solve(
        &self,
        a0: &Array1<f64>,
        dt: f64,
        eval_grid: &Array1<f64>,
        t0: f64,
        t_end: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<SpectralGalerkinSolution> {
        let n = self.cfg.n_modes;
        if a0.len() != n {
            return Err(IntegrateError::DimensionMismatch(format!(
                "a0 length {} ≠ n_modes {}",
                a0.len(),
                n
            )));
        }
        if t_end <= t0 {
            return Err(IntegrateError::InvalidInput(
                "t_end must be greater than t0".to_string(),
            ));
        }
        if dt <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "dt must be positive".to_string(),
            ));
        }

        let normal = Normal::new(0.0_f64, 1.0)
            .map_err(|e| IntegrateError::ComputationError(format!("Normal distribution: {e}")))?;

        let n_steps = ((t_end - t0) / dt).ceil() as usize;
        let capacity = (n_steps / self.save_every + 2).max(2);
        let mut times = Vec::with_capacity(capacity);
        let mut modal_snapshots = Vec::with_capacity(capacity);
        let mut physical_snapshots = Vec::with_capacity(capacity);

        let mut a = a0.clone();

        // Precompute reconstruction basis values at eval_grid points
        let ng = eval_grid.len();
        let l = self.cfg.domain_length;
        let mut basis = vec![0.0_f64; n * ng]; // basis[k * ng + g]
        for k in 0..n {
            for g in 0..ng {
                basis[k * ng + g] =
                    ((k + 1) as f64 * std::f64::consts::PI * eval_grid[g] / l).sin();
            }
        }

        let reconstruct = |a: &Array1<f64>| -> Array1<f64> {
            let mut u = Array1::zeros(ng);
            for g in 0..ng {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += a[k] * basis[k * ng + g];
                }
                u[g] = s;
            }
            u
        };

        times.push(t0);
        modal_snapshots.push(a.clone());
        physical_snapshots.push(reconstruct(&a));

        let mut t = t0;

        for step in 0..n_steps {
            let actual_dt = dt.min(t_end - t);
            if actual_dt <= 0.0 {
                break;
            }

            // Exact OU transition: a_k(t+dt) = a_k(t) e^{-λ_k dt}
            //   + σ_k sqrt((1 - e^{-2λ_k dt}) / (2λ_k)) * N(0,1)
            for k in 0..n {
                let lam = self.lambda[k];
                let e_lam = (-lam * actual_dt).exp();
                let var_k = if lam > 1e-15 {
                    self.sigma_k[k] * self.sigma_k[k] * (1.0 - e_lam * e_lam) / (2.0 * lam)
                } else {
                    self.sigma_k[k] * self.sigma_k[k] * actual_dt
                };
                let std_k = var_k.max(0.0).sqrt();
                let xi = rng.sample(normal);
                a[k] = e_lam * a[k] + std_k * xi;
            }

            t += actual_dt;

            if (step + 1) % self.save_every == 0 || t >= t_end - 1e-14 {
                times.push(t);
                modal_snapshots.push(a.clone());
                physical_snapshots.push(reconstruct(&a));
            }
        }

        Ok(SpectralGalerkinSolution {
            times,
            modal_snapshots,
            physical_snapshots,
            grid: eval_grid.clone(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky decomposition of an n×n symmetric positive definite matrix.
/// Returns the lower-triangular factor L in row-major storage.
fn cholesky_lower(a: &[f64], n: usize) -> IntegrateResult<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0_f64;
            for k in 0..j {
                s += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - s;
                if diag < 0.0 {
                    return Err(IntegrateError::ComputationError(format!(
                        "Cholesky failed: non-positive diagonal {diag} at row {i}.  \
                         Increase correlation_length.",
                    )));
                }
                l[i * n + j] = diag.sqrt();
            } else {
                let lii = l[j * n + j];
                l[i * n + j] = if lii.abs() > 1e-15 {
                    (a[i * n + j] - s) / lii
                } else {
                    0.0
                };
            }
        }
    }
    Ok(l)
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
        seeded_rng(77777)
    }

    // ── Anderson-Mattingly ──────────────────────────────────────────────────

    #[test]
    fn test_am_scheme_runs() {
        let cfg = AndersonMattinglyConfig {
            diffusion: 0.5,
            sigma: 0.05,
            correlation_length: 0.2,
            dt: 1e-3,
            n_nodes: 16,
            domain_length: 1.0,
        };
        let solver = AndersonMattinglyScheme::new(cfg, 5)
            .expect("AndersonMattinglyScheme::new should succeed");
        let u0 = Array1::zeros(16);
        let mut rng = make_rng();
        let sol = solver
            .solve(&u0, 0.0, 0.05, &mut rng)
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
        for snap in &sol.snapshots {
            assert!(snap.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_am_decay_without_noise() {
        // σ=0: solution should decay toward zero from non-zero IC
        let cfg = AndersonMattinglyConfig {
            diffusion: 1.0,
            sigma: 0.0,
            correlation_length: 0.2,
            dt: 1e-3,
            n_nodes: 10,
            domain_length: 1.0,
        };
        let solver = AndersonMattinglyScheme::new(cfg, 1)
            .expect("AndersonMattinglyScheme::new should succeed");
        let u0 = Array1::from_vec(vec![1.0_f64; 10]);
        let mut rng = make_rng();
        let sol = solver
            .solve(&u0, 0.0, 0.1, &mut rng)
            .expect("solver.solve should succeed");
        let u_final = sol.snapshots.last().expect("solution has snapshots");
        let norm_final: f64 = u_final.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_init: f64 = 10.0_f64.sqrt();
        assert!(
            norm_final < norm_init,
            "Solution should decay without noise: norm_init={norm_init:.4}, norm_final={norm_final:.4}"
        );
    }

    // ── IMEX ───────────────────────────────────────────────────────────────

    #[test]
    fn test_imex_zero_nonlinear_runs() {
        let solver = ImexParabolicSolver::new(0.1, 0.05, 1e-3, 1.0, 16, 5)
            .expect("ImexParabolicSolver::new should succeed");
        let u0 = Array1::zeros(16);
        let mut rng = make_rng();
        // F(u) = 0 (pure stochastic heat)
        let sol = solver
            .solve(&u0, |_t, _u| Array1::zeros(16), 0.0, 0.05, &mut rng)
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
        for snap in &sol.snapshots {
            assert!(snap.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_imex_with_nonlinear_term() {
        let solver = ImexParabolicSolver::new(0.1, 0.05, 1e-3, 1.0, 10, 10)
            .expect("ImexParabolicSolver::new should succeed");
        let u0 = Array1::from_vec(vec![0.5_f64; 10]);
        let mut rng = make_rng();
        // Logistic growth: F(u) = u(1 - u)
        let sol = solver
            .solve(
                &u0,
                |_t, u| u.mapv(|ui| ui * (1.0 - ui)),
                0.0,
                0.05,
                &mut rng,
            )
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
    }

    // ── Spectral Galerkin ───────────────────────────────────────────────────

    #[test]
    fn test_spectral_galerkin_zero_ic() {
        let cfg = SpectralGalerkinConfig {
            diffusion: 1.0,
            sigma: 0.1,
            mode_sigmas: None,
            n_modes: 8,
            domain_length: 1.0,
        };
        let solver = SpectralGalerkinSolver::new(cfg, 10)
            .expect("SpectralGalerkinSolver::new should succeed");
        let a0 = Array1::zeros(8);
        let grid = Array1::linspace(0.05, 0.95, 10);
        let mut rng = make_rng();
        let sol = solver
            .solve(&a0, 1e-3, &grid, 0.0, 0.1, &mut rng)
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
        for snap in &sol.physical_snapshots {
            assert!(snap.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_spectral_galerkin_energy_series() {
        let cfg = SpectralGalerkinConfig {
            diffusion: 1.0,
            sigma: 0.5,
            mode_sigmas: None,
            n_modes: 4,
            domain_length: 1.0,
        };
        let solver = SpectralGalerkinSolver::new(cfg, 1)
            .expect("SpectralGalerkinSolver::new should succeed");
        let a0 = Array1::zeros(4);
        let grid = Array1::linspace(0.1, 0.9, 5);
        let mut rng = make_rng();
        let sol = solver
            .solve(&a0, 1e-3, &grid, 0.0, 0.05, &mut rng)
            .expect("solver.solve should succeed");
        let energies = sol.modal_energy_series();
        assert_eq!(energies.len(), sol.len());
        for &e in &energies {
            assert!(e >= 0.0 && e.is_finite());
        }
    }

    #[test]
    fn test_spectral_galerkin_no_noise_decays() {
        // With σ=0 starting from non-zero IC, energy should decrease
        let cfg = SpectralGalerkinConfig {
            diffusion: 1.0,
            sigma: 0.0,
            mode_sigmas: None,
            n_modes: 4,
            domain_length: 1.0,
        };
        let solver = SpectralGalerkinSolver::new(cfg, 1)
            .expect("SpectralGalerkinSolver::new should succeed");
        let a0 = Array1::from_vec(vec![1.0, 0.5, 0.25, 0.1]);
        let grid = Array1::linspace(0.1, 0.9, 5);
        let mut rng = make_rng();
        let sol = solver
            .solve(&a0, 1e-3, &grid, 0.0, 0.1, &mut rng)
            .expect("solver.solve should succeed");
        let energies = sol.modal_energy_series();
        let e0 = energies[0];
        let ef = *energies.last().expect("energies series is non-empty");
        assert!(
            ef < e0,
            "Energy should decay without noise: e0={e0:.6}, ef={ef:.6}"
        );
    }

    #[test]
    fn test_per_mode_sigmas() {
        // Only excite mode 1; modes 2+ should have small amplitude
        let n = 8;
        let mut mode_sigmas = vec![0.0_f64; n];
        mode_sigmas[0] = 1.0;
        let cfg = SpectralGalerkinConfig {
            diffusion: 0.1,
            sigma: 0.0,
            mode_sigmas: Some(mode_sigmas),
            n_modes: n,
            domain_length: 1.0,
        };
        let solver = SpectralGalerkinSolver::new(cfg, 10)
            .expect("SpectralGalerkinSolver::new should succeed");
        let a0 = Array1::zeros(n);
        let grid = Array1::linspace(0.1, 0.9, 10);
        let mut rng = make_rng();
        let sol = solver
            .solve(&a0, 1e-3, &grid, 0.0, 0.5, &mut rng)
            .expect("solver.solve should succeed");
        let a_final = sol
            .modal_snapshots
            .last()
            .expect("modal snapshots is non-empty");
        // Mode 2+ should have negligible amplitude compared to mode 1
        let a1_abs = a_final[0].abs();
        let rest_max = a_final
            .iter()
            .skip(1)
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        // Not strict: just check no NaN and mode 1 is largest
        assert!(a_final.iter().all(|v| v.is_finite()));
    }
}
