//! Fluid dynamics module for computational fluid dynamics (CFD)
//!
//! This module provides comprehensive tools for simulating fluid flows,
//! including incompressible and compressible flows, turbulence modeling,
//! and advanced optimization techniques.
//!
//! # Module Structure
//!
//! - [`core`]: Core types and utilities shared across all fluid dynamics modules
//! - [`incompressible`]: Incompressible Navier-Stokes solvers and utilities
//! - [`compressible`]: Compressible flow solvers with SIMD optimizations
//! - [`turbulence`]: Turbulence modeling including LES, RANS, and advanced models
//! - [`multiphase`]: Multi-phase flow simulations (future implementation)
//! - [`spectral`]: Spectral methods for fluid dynamics using FFT-based approaches
//! - [`gpu_acceleration`]: GPU-accelerated fluid dynamics solvers (future implementation)
//! - [`optimization`]: Performance optimization utilities (future implementation)

pub mod compressible;
pub mod core;
pub mod incompressible;
pub mod spectral;
pub mod turbulence;
// pub mod multiphase;
// pub mod gpu_acceleration;
// pub mod optimization;
// pub mod tests;

// Re-export commonly used items
pub use core::{FluidBoundaryCondition, FluidState, FluidState3D, NavierStokesParams};

pub use incompressible::{
    apply_boundary_conditions_2d, apply_periodic_conditions, apply_pressure_boundary_conditions,
    apply_wall_velocity_conditions, channel_flow, couette_flow, lid_driven_cavity, poiseuille_flow,
    stagnation_point_flow, taylor_green_vortex, vortex_pair, NavierStokesSolver,
};

pub use compressible::{CompressibleFlowSolver, CompressibleFluxes, CompressibleState};

pub use turbulence::{
    // Advanced turbulence models
    AdvancedTurbulenceModel,
    // LES components
    LESolver,
    RANSModel,
    // RANS components
    RANSSolver,
    RANSState,
    SGSModel,
    SpalartAllmarasModel,
    TurbulenceConstants,
    TurbulenceModel,
    // Base turbulence types and traits
    TurbulenceModelType,
    TurbulenceUtils,
};

pub use spectral::{
    DealiasingOperations,
    // Dealiasing strategies and operations
    DealiasingStrategy,
    // FFT operations
    FFTOperations,
    FFTResult,
    // Main spectral solver
    SpectralNavierStokesSolver,
    SpectralSolver,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that main types can be imported
        let params = NavierStokesParams::default();
        assert_eq!(params.nu, 0.01);
        assert_eq!(params.rho, 1.0);
    }

    #[test]
    fn test_boundary_condition_enum() {
        let bc = FluidBoundaryCondition::NoSlip;
        matches!(bc, FluidBoundaryCondition::NoSlip);

        let bc2 = FluidBoundaryCondition::Inflow(1.0, 0.5);
        if let FluidBoundaryCondition::Inflow(u, v) = bc2 {
            assert_eq!(u, 1.0);
            assert_eq!(v, 0.5);
        }
    }

    #[test]
    fn test_initialization_functions() {
        // Test lid-driven cavity
        let state = lid_driven_cavity(16, 16, 1.0);
        assert_eq!(state.velocity.len(), 2);
        assert_eq!(state.velocity[0].dim(), (16, 16));

        // Test Taylor-Green vortex
        let state2 = taylor_green_vortex(16, 16, 1.0, 1.0);
        assert_eq!(state2.velocity.len(), 2);
        assert_eq!(state2.velocity[0].dim(), (16, 16));
    }

    #[test]
    fn test_solver_creation() {
        let params = NavierStokesParams::default();
        let bc_x = (
            FluidBoundaryCondition::NoSlip,
            FluidBoundaryCondition::NoSlip,
        );
        let bc_y = (
            FluidBoundaryCondition::NoSlip,
            FluidBoundaryCondition::NoSlip,
        );

        let _solver = NavierStokesSolver::new(params, bc_x, bc_y);
    }

    #[test]
    fn test_compressible_solver_creation() {
        let solver = CompressibleFlowSolver::new(8, 8, 8, 0.1, 0.1, 0.1);
        let state = solver.initialize_state();

        assert_eq!(solver.dimensions(), (8, 8, 8));
        assert_eq!(state.dimensions(), (8, 8, 8));
        assert!(state.is_physically_valid());
    }

    #[test]
    fn test_turbulence_module_integration() {
        // Test RANS solver
        let rans_solver = RANSSolver::new(8, 8, RANSModel::KEpsilon, 1000.0);
        assert_eq!(rans_solver.nx, 8);
        assert_eq!(rans_solver.turbulence_model, RANSModel::KEpsilon);

        // Test LES solver
        let les_solver = LESolver::new(8, 8, 8, 0.1, 0.1, 0.1, SGSModel::Smagorinsky);
        assert_eq!(les_solver.nx, 8);
        assert_eq!(les_solver.sgs_model, SGSModel::Smagorinsky);

        // Test advanced turbulence model
        let advanced_model = AdvancedTurbulenceModel::new(TurbulenceModelType::KEpsilon, 4, 4, 4);
        assert_eq!(advanced_model.model_type, TurbulenceModelType::KEpsilon);
    }

    #[test]
    fn test_turbulence_state_creation() {
        // Test RANS state
        let rans_state = RANSState::new(6, 6, 0.1, 0.1);
        assert_eq!(rans_state.mean_velocity.len(), 2);
        assert_eq!(rans_state.mean_velocity[0].dim(), (6, 6));

        // Test 3D fluid state for LES
        let velocity = vec![
            scirs2_core::ndarray::Array3::zeros((4, 4, 4)),
            scirs2_core::ndarray::Array3::zeros((4, 4, 4)),
            scirs2_core::ndarray::Array3::zeros((4, 4, 4)),
        ];
        let les_state = FluidState3D {
            velocity,
            pressure: scirs2_core::ndarray::Array3::zeros((4, 4, 4)),
            temperature: None,
            time: 0.0,
            dx: 0.1,
            dy: 0.1,
            dz: 0.1,
        };
        assert_eq!(les_state.velocity.len(), 3);
        assert_eq!(les_state.velocity[0].dim(), (4, 4, 4));
    }

    #[test]
    fn test_spectral_solver_creation() {
        // Test 2D spectral solver
        let solver_2d = SpectralNavierStokesSolver::new(
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
        assert_eq!(solver_2d.nx, 8);
        assert_eq!(solver_2d.ny, 8);
        assert_eq!(solver_2d.nz, None);
        assert_eq!(solver_2d.dealiasing, DealiasingStrategy::TwoThirds);

        // Test 3D spectral solver
        let solver_3d = SpectralNavierStokesSolver::new(
            8,
            8,
            Some(8),
            2.0 * std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            Some(2.0 * std::f64::consts::PI),
            0.01,
            0.001,
            DealiasingStrategy::None,
        );
        assert_eq!(solver_3d.nx, 8);
        assert_eq!(solver_3d.ny, 8);
        assert_eq!(solver_3d.nz, Some(8));
        assert_eq!(solver_3d.dealiasing, DealiasingStrategy::None);
    }

    #[test]
    fn test_spectral_initialization() {
        let solver = SpectralNavierStokesSolver::new(
            16,
            16,
            None,
            2.0 * std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            None,
            0.01,
            0.001,
            DealiasingStrategy::TwoThirds,
        );

        // Test 2D Taylor-Green vortex initialization
        let vorticity = solver.initialize_taylor_green_vortex_2d();
        assert_eq!(vorticity.dim(), (16, 16));

        // Check that vorticity is not all zeros
        let max_vorticity = vorticity.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        assert!(max_vorticity > 0.0);
    }

    #[test]
    fn test_spectral_3d_initialization() {
        let solver = SpectralNavierStokesSolver::new(
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

        // Test 3D Taylor-Green vortex initialization
        let velocity = solver.initialize_taylor_green_vortex_3d();
        assert_eq!(velocity[0].dim(), (8, 8, 8));
        assert_eq!(velocity[1].dim(), (8, 8, 8));
        assert_eq!(velocity[2].dim(), (8, 8, 8));

        // Check that velocity fields are not all zeros
        let max_u = velocity[0].iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        let max_v = velocity[1].iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        assert!(max_u > 0.0);
        assert!(max_v > 0.0);
    }

    #[test]
    fn test_fft_operations_integration() {
        // Test that FFT operations can be used directly
        let field = scirs2_core::ndarray::Array2::ones((8, 8));

        let field_hat = FFTOperations::fft_2d_forward(&field).unwrap();
        assert_eq!(field_hat.dim(), (8, 8));

        let recovered = FFTOperations::fft_2d_backward(&field_hat).unwrap();
        assert_eq!(recovered.dim(), (8, 8));

        // Test energy spectrum computation
        let spectrum = FFTOperations::compute_energy_spectrum_2d(&field).unwrap();
        assert!(!spectrum.is_empty());
    }

    #[test]
    fn test_dealiasing_strategy_integration() {
        // Test that dealiasing strategies work
        let field = scirs2_core::ndarray::Array2::ones((8, 8));

        let dealiased =
            DealiasingOperations::apply_dealiasing_2d(&field, DealiasingStrategy::TwoThirds)
                .unwrap();
        assert_eq!(dealiased.dim(), field.dim());

        // Test strategy recommendation
        let strategy = DealiasingOperations::recommend_strategy((64, 64), 1000.0, 0.95);
        assert!(matches!(
            strategy,
            DealiasingStrategy::None
                | DealiasingStrategy::TwoThirds
                | DealiasingStrategy::ThreeHalves
                | DealiasingStrategy::PhaseShift
        ));
    }
}
