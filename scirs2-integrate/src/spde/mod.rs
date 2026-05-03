//! Stochastic Partial Differential Equation (SPDE) solvers
//!
//! This module provides numerical methods for solving SPDEs — partial
//! differential equations driven by stochastic forcing (Wiener fields or
//! coloured noise).
//!
//! ## Submodules
//!
//! | Submodule | Contents |
//! |-----------|----------|
//! | [`random_fields`] | Gaussian random field generation (circulant embedding, KL expansion, Fourier spectral) |
//! | [`heat_stochastic`] | Stochastic heat equation in 1-D and 2-D |
//! | [`wave_stochastic`] | Stochastic wave equation in 1-D |
//! | [`pathwise`] | Advanced pathwise methods: Anderson-Mattingly, IMEX, Spectral Galerkin |
//!
//! ## Background
//!
//! SPDEs arise in a wide range of scientific domains:
//!
//! - **Fluid dynamics**: stochastic Navier-Stokes, turbulence modelling.
//! - **Finance**: Heath-Jarrow-Morton interest rate models, stochastic volatility surfaces.
//! - **Neuroscience**: stochastic cable equations, neural field models.
//! - **Materials**: stochastic phase-field models (Cahn-Hilliard-Cook equation).
//! - **Climate**: stochastic parameterisations in GCMs.
//!
//! A common prototype is the stochastic heat equation:
//!
//! ```text
//! du = D Δu dt + σ dW(x,t)
//! ```
//!
//! where `W(x,t)` is a cylindrical Wiener process (space-time white noise)
//! or a coloured noise field.
//!
//! ## Numerical Methods Overview
//!
//! ### Euler-Maruyama + Finite Differences
//!
//! The [`heat_stochastic`] module uses explicit Euler-Maruyama in time and
//! second-order central finite differences in space.  The stability constraint
//! (CFL / Von Neumann condition) is checked at construction time.
//!
//! For the **additive noise** case the update is:
//! ```text
//! u_i^{n+1} = u_i^n + D dt/dx^2 (u_{i-1}^n - 2u_i^n + u_{i+1}^n)
//!           + σ sqrt(dt/dx) N_i(0,1)
//! ```
//!
//! For **multiplicative noise**:
//! ```text
//! u_i^{n+1} = u_i^n + D dt/dx^2 (u_{i-1}^n - 2u_i^n + u_{i+1}^n)
//!           + σ u_i^n sqrt(dt/dx) N_i(0,1)
//! ```
//!
//! ### Leapfrog + Stochastic Forcing (Wave Equation)
//!
//! The [`wave_stochastic`] module uses a staggered leapfrog scheme for the
//! first-order system (u, v = ∂u/∂t):
//! ```text
//! v^{n+1/2} = v^{n-1/2} + dt (c^2/dx^2 Lu^n + σ/sqrt(dx dt) ξ^n)
//! u^{n+1}   = u^n + dt v^{n+1/2}
//! ```
//! Energy tracking is built in.
//!
//! ### Pathwise Methods ([`pathwise`])
//!
//! Three approaches for higher accuracy or larger time steps:
//!
//! 1. **Anderson-Mattingly**: Splits into a deterministic heat semigroup step
//!    (applied via Fourier modes) and a spatially correlated stochastic
//!    increment generated from a Cholesky-factored covariance matrix.
//!
//! 2. **IMEX-Euler**: Implicit treatment of the stiff linear part `DΔu` via a
//!    tridiagonal Thomas solve; explicit treatment of the nonlinear drift.
//!    Removes the CFL stability restriction on the time step.
//!
//! 3. **Spectral Galerkin**: Exact (in distribution) integration of the linear
//!    stochastic heat equation via Ornstein-Uhlenbeck transitions for each
//!    Fourier-sine mode.
//!
//! ## Random Field Generation ([`random_fields`])
//!
//! Three methods for sampling Gaussian random fields over 2-D grids:
//!
//! - **Circulant Embedding**: Extends the covariance to a circulant structure
//!   and samples via 2-D FFT.  Exact (in distribution).  O(N log N).
//! - **KL Expansion**: Truncated eigendecomposition of the covariance operator.
//!   Controlled approximation; useful for low-dimensional parameterisations.
//! - **Fourier Spectral**: Samples amplitudes from the spectral density and
//!   assigns random phases.  Fast and simple; assumes periodic BCs.
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_integrate::spde::{
//!     heat_stochastic::{NoiseType, StochasticHeatConfig, StochasticHeatSolver1D},
//! };
//! use scirs2_core::ndarray::Array1;
//! use scirs2_core::random::prelude::*;
//!
//! let config = StochasticHeatConfig {
//!     diffusion: 0.1,
//!     sigma: 0.05,
//!     noise_type: NoiseType::Additive,
//!     dt: 1e-4,
//!     ..Default::default()
//! };
//! let solver = StochasticHeatSolver1D::new(config, 1.0, 20, 10).unwrap();
//! let u0 = Array1::zeros(20);
//! let mut rng = seeded_rng(42);
//! let sol = solver.solve(u0.view(), 0.0, 0.01, &mut rng).unwrap();
//! assert!(!sol.is_empty());
//! ```

pub mod heat_stochastic;
pub mod pathwise;
pub mod random_fields;
pub mod wave_stochastic;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use heat_stochastic::{
    HeatSnapshot, NoiseType, StochasticHeatConfig, StochasticHeatSolution, StochasticHeatSolver1D,
    StochasticHeatSolver2D,
};

pub use wave_stochastic::{
    StochasticWaveConfig, StochasticWaveSolution, StochasticWaveSolver, WaveSnapshot,
};

pub use random_fields::{CorrelationFunction, RandomField};

pub use pathwise::{
    AndersonMattinglyConfig, AndersonMattinglyScheme, AndersonMattinglySolution,
    ImexParabolicSolver, ImexSolution, SpectralGalerkinConfig, SpectralGalerkinSolution,
    SpectralGalerkinSolver,
};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use scirs2_core::random::prelude::*;

    /// Smoke test: solve the stochastic heat equation in 1-D and verify basic
    /// structural properties of the solution.
    #[test]
    fn test_spde_heat_1d_smoke() {
        let config = StochasticHeatConfig {
            diffusion: 0.05,
            sigma: 0.01,
            noise_type: NoiseType::Additive,
            dt: 5e-5,
            ..Default::default()
        };
        let solver = StochasticHeatSolver1D::new(config, 1.0, 16, 5)
            .expect("StochasticHeatSolver1D::new should succeed");
        let u0 = Array1::zeros(16);
        let mut rng = seeded_rng(1234);
        let sol = solver
            .solve(u0.view(), 0.0, 0.01, &mut rng)
            .expect("solver.solve should succeed");

        assert!(
            sol.len() >= 2,
            "Should have at least initial + final snapshot"
        );
        assert_eq!(sol.grid.len(), 16);
        assert_eq!(sol.shape, vec![16]);
        assert_eq!(sol.dim, 1);
        for snap in &sol.snapshots {
            assert_eq!(snap.len(), 16);
            assert!(snap.iter().all(|v| v.is_finite()));
        }
        let mean = sol.mean_field().expect("mean_field should succeed");
        assert_eq!(mean.len(), 16);
    }

    /// Smoke test for the stochastic wave solver.
    #[test]
    fn test_spde_wave_1d_smoke() {
        let config = StochasticWaveConfig {
            wave_speed: 1.0,
            sigma: 0.1,
            dt: 5e-5,
        };
        let solver = StochasticWaveSolver::new(config, 1.0, 20, 10)
            .expect("StochasticWaveSolver::new should succeed");
        let u0 = Array1::from_vec(
            (0..20)
                .map(|i| ((i as f64 + 1.0) * std::f64::consts::PI / 21.0).sin() * 0.1)
                .collect(),
        );
        let v0 = Array1::zeros(20);
        let mut rng = seeded_rng(5678);
        let sol = solver
            .solve(&u0, &v0, 0.0, 0.01, &mut rng)
            .expect("solver.solve should succeed");

        assert!(!sol.is_empty());
        let energies = sol.energy_series();
        assert_eq!(energies.len(), sol.len());
        assert!(energies.iter().all(|&e| e.is_finite() && e >= 0.0));
    }

    /// Cross-module smoke test: random field into heat solver initial condition.
    #[test]
    fn test_random_field_as_initial_condition() {
        let mut rng = seeded_rng(9999);
        let gx = Array1::linspace(1.0 / 17.0, 16.0 / 17.0, 16);
        let gy = Array1::from_vec(vec![0.0]);
        let cov = CorrelationFunction::Gaussian { length_scale: 0.3 };
        let field = RandomField::sample_kl_expansion(gx.view(), gy.view(), cov, 8, &mut rng)
            .expect("sample_kl_expansion should succeed");
        // Extract 1-D slice (column 0)
        let u0_vec: Vec<f64> = (0..16).map(|i| field[[i, 0]] * 0.01).collect();
        let u0 = Array1::from_vec(u0_vec);

        let config = StochasticHeatConfig {
            diffusion: 0.05,
            sigma: 0.005,
            dt: 5e-5,
            ..Default::default()
        };
        let solver = StochasticHeatSolver1D::new(config, 1.0, 16, 10)
            .expect("StochasticHeatSolver1D::new should succeed");
        let sol = solver
            .solve(u0.view(), 0.0, 0.005, &mut rng)
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
    }

    /// Test IMEX solver through the top-level re-export.
    #[test]
    fn test_imex_via_spde_module() {
        let solver = ImexParabolicSolver::new(0.2, 0.05, 2e-3, 1.0, 12, 5)
            .expect("ImexParabolicSolver::new should succeed");
        let u0 = Array1::from_vec(vec![0.1_f64; 12]);
        let mut rng = seeded_rng(42);
        let sol = solver
            .solve(&u0, |_t, _u| Array1::zeros(12), 0.0, 0.05, &mut rng)
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
    }

    /// Test Spectral Galerkin through the top-level re-export.
    #[test]
    fn test_spectral_galerkin_via_spde_module() {
        let cfg = SpectralGalerkinConfig {
            diffusion: 1.0,
            sigma: 0.2,
            n_modes: 8,
            domain_length: 1.0,
            mode_sigmas: None,
        };
        let solver = SpectralGalerkinSolver::new(cfg, 5)
            .expect("SpectralGalerkinSolver::new should succeed");
        let a0 = Array1::zeros(8);
        let grid = Array1::linspace(0.1, 0.9, 10);
        let mut rng = seeded_rng(1111);
        let sol = solver
            .solve(&a0, 1e-3, &grid, 0.0, 0.1, &mut rng)
            .expect("solver.solve should succeed");
        assert!(!sol.is_empty());
        assert_eq!(sol.grid.len(), 10);
    }
}
