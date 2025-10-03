//! Spectral methods for fluid dynamics
//!
//! This module provides spectral methods for solving fluid dynamics problems,
//! particularly the Navier-Stokes equations in periodic domains. Spectral methods
//! offer excellent accuracy for smooth solutions and are particularly well-suited
//! for turbulence simulations and problems with periodic boundary conditions.
//!
//! # Overview
//!
//! Spectral methods represent the solution as a sum of basis functions (typically
//! Fourier modes for periodic domains) and compute derivatives exactly in spectral
//! space using multiplication by wavenumbers. This leads to spectral accuracy
//! (exponential convergence for smooth solutions) but requires periodic boundary
//! conditions.
//!
//! # Key Components
//!
//! - [`SpectralNavierStokesSolver`]: Main solver for Navier-Stokes equations using spectral methods
//! - [`FFTOperations`]: Fast Fourier Transform operations for transforming between physical and spectral space
//! - [`DealiasingStrategy`]: Strategies for preventing aliasing errors in nonlinear terms
//! - [`DealiasingOperations`]: Implementation of various dealiasing techniques
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use scirs2_integrate::specialized::fluid_dynamics::spectral::{
//!     SpectralNavierStokesSolver, DealiasingStrategy
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! // Create a 2D spectral solver for a periodic domain
//! let solver = SpectralNavierStokesSolver::new(
//!     64,    // nx: grid points in x
//!     64,    // ny: grid points in y  
//!     None,  // nz: no z for 2D
//!     2.0 * std::f64::consts::PI, // lx: domain size in x
//!     2.0 * std::f64::consts::PI, // ly: domain size in y
//!     None,  // lz: no z for 2D
//!     0.01,  // nu: kinematic viscosity
//!     0.001, // dt: time step
//!     DealiasingStrategy::TwoThirds, // dealiasing strategy
//! );
//!
//! // Initialize with Taylor-Green vortex
//! let initial_vorticity = solver.initialize_taylor_green_vortex_2d();
//!
//! // Solve for time evolution
//! let solution = solver.solve_2d_spectral(&initial_vorticity, 1.0).unwrap();
//! ```
//!
//! # Advantages of Spectral Methods
//!
//! - **Spectral accuracy**: Exponential convergence for smooth solutions
//! - **Exact derivatives**: No numerical errors in computing derivatives
//! - **Efficient for turbulence**: Excellent resolution of small scales
//! - **Energy conservation**: Natural conservation properties
//!
//! # Limitations
//!
//! - **Periodic boundaries only**: Limited to periodic boundary conditions
//! - **Smooth solutions required**: Poor performance for discontinuous solutions
//! - **Aliasing errors**: Require dealiasing for nonlinear problems
//! - **Memory intensive**: Global operations require significant memory

pub mod dealiasing;
pub mod fft_operations;
pub mod solver;

// Re-export main types and functions
pub use dealiasing::{DealiasingOperations, DealiasingStrategy};
pub use fft_operations::{FFTOperations, FFTResult};
pub use solver::SpectralNavierStokesSolver;

// Re-export common functions for convenience
pub use solver::SpectralNavierStokesSolver as SpectralSolver;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_spectral_module_integration() {
        // Test that all components work together
        let solver = SpectralNavierStokesSolver::new(
            8,
            8,
            None,
            2.0 * std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            None,
            0.01,
            0.001,
            DealiasingStrategy::TwoThirds,
        );

        // Test initialization
        let vorticity = solver.initialize_taylor_green_vortex_2d();
        assert_eq!(vorticity.dim(), (8, 8));

        // Test that solver can be created with different strategies
        let _solver_none = SpectralNavierStokesSolver::new(
            8,
            8,
            None,
            2.0 * std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            None,
            0.01,
            0.001,
            DealiasingStrategy::None,
        );

        let _solver_phase = SpectralNavierStokesSolver::new(
            8,
            8,
            None,
            2.0 * std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            None,
            0.01,
            0.001,
            DealiasingStrategy::PhaseShift,
        );
    }

    #[test]
    fn test_fft_operations_integration() {
        // Test FFT operations work with solver
        let field = Array2::ones((8, 8));

        let field_hat = FFTOperations::fft_2d_forward(&field).unwrap();
        assert_eq!(field_hat.dim(), (8, 8));

        let recovered = FFTOperations::fft_2d_backward(&field_hat).unwrap();
        assert_eq!(recovered.dim(), (8, 8));

        // Test energy spectrum
        let spectrum = FFTOperations::compute_energy_spectrum_2d(&field).unwrap();
        assert!(!spectrum.is_empty());
    }

    #[test]
    fn test_dealiasing_integration() {
        // Test dealiasing operations
        let field = Array2::ones((8, 8));

        let dealiased =
            DealiasingOperations::apply_dealiasing_2d(&field, DealiasingStrategy::TwoThirds)
                .unwrap();
        assert_eq!(dealiased.dim(), field.dim());

        // Test strategy recommendation
        let strategy = DealiasingOperations::recommend_strategy((64, 64), 1000.0, 0.95);
        assert!(matches!(
            strategy,
            DealiasingStrategy::TwoThirds | DealiasingStrategy::ThreeHalves
        ));
    }

    #[test]
    fn test_3d_solver_creation() {
        // Test 3D solver can be created
        let solver_3d = SpectralNavierStokesSolver::new(
            8,
            8,
            Some(8),
            2.0 * std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            Some(2.0 * std::f64::consts::PI),
            0.01,
            0.001,
            DealiasingStrategy::TwoThirds,
        );

        // Test 3D initialization
        let velocity_3d = solver_3d.initialize_taylor_green_vortex_3d();
        assert_eq!(velocity_3d[0].dim(), (8, 8, 8));
        assert_eq!(velocity_3d[1].dim(), (8, 8, 8));
        assert_eq!(velocity_3d[2].dim(), (8, 8, 8));
    }

    #[test]
    fn test_solver_parameters() {
        let solver = SpectralNavierStokesSolver::new(
            16,
            32,
            Some(8),
            4.0,
            6.0,
            Some(2.0),
            0.02,
            0.0005,
            DealiasingStrategy::PhaseShift,
        );

        // Check parameters are stored correctly
        assert_eq!(solver.nx, 16);
        assert_eq!(solver.ny, 32);
        assert_eq!(solver.nz, Some(8));
        assert_eq!(solver.lx, 4.0);
        assert_eq!(solver.ly, 6.0);
        assert_eq!(solver.lz, Some(2.0));
        assert_eq!(solver.nu, 0.02);
        assert_eq!(solver.dt, 0.0005);
        assert_eq!(solver.dealiasing, DealiasingStrategy::PhaseShift);
    }

    #[test]
    fn test_default_dealiasing_strategy() {
        let default_strategy = DealiasingStrategy::default();
        assert_eq!(default_strategy, DealiasingStrategy::TwoThirds);
    }
}
