//! Turbulence modeling for computational fluid dynamics
//!
//! This module provides comprehensive turbulence modeling capabilities including:
//! - Base turbulence model types and traits
//! - Advanced turbulence models with SIMD optimization
//! - Large Eddy Simulation (LES) with subgrid-scale models
//! - Reynolds-Averaged Navier-Stokes (RANS) methods
//!
//! # Module Structure
//!
//! - [`models`]: Base turbulence model types, traits, and common utilities
//! - [`advanced_models`]: Advanced turbulence models with SIMD acceleration
//! - [`les`]: Large Eddy Simulation solver and subgrid-scale models
//! - [`rans`]: Reynolds-Averaged Navier-Stokes solver and turbulence models
//!
//! # Examples
//!
//! ## Using RANS k-ε model
//!
//! ```rust
//! use scirs2_integrate::specialized::fluid_dynamics::turbulence::{
//!     RANSSolver, RANSModel, RANSState
//! };
//!
//! // Create RANS solver
//! let solver = RANSSolver::new(64, 64, RANSModel::KEpsilon, 1000.0);
//!
//! // Initialize state for lid-driven cavity
//! let initial_state = RANSState::lid_driven_cavity(64, 64, 0.01, 0.01, 1.0);
//!
//! // Solve RANS equations
//! let result = solver.solve_rans(initial_state, 1000, 1e-6);
//! ```
//!
//! ## Using LES with Smagorinsky model
//!
//! ```rust,no_run
//! use scirs2_integrate::specialized::fluid_dynamics::turbulence::{
//!     LESolver, SGSModel, FluidState3D
//! };
//! use scirs2_core::ndarray::Array3;
//!
//! // Create LES solver
//! let solver = LESolver::new(32, 32, 32, 0.1, 0.1, 0.1, SGSModel::Smagorinsky);
//!
//! // Initialize 3D fluid state
//! let velocity = vec![
//!     Array3::zeros((32, 32, 32)),
//!     Array3::zeros((32, 32, 32)),
//!     Array3::zeros((32, 32, 32)),
//! ];
//! let initial_state = FluidState3D {
//!     velocity,
//!     pressure: Array3::zeros((32, 32, 32)),
//!     dx: 0.1,
//!     dy: 0.1,
//!     dz: 0.1,
//! };
//!
//! // Solve LES equations
//! let results = solver.solve_3d(initial_state, 1.0, 100);
//! ```
//!
//! ## Using advanced turbulence models
//!
//! ```rust
//! use scirs2_integrate::specialized::fluid_dynamics::turbulence::{
//!     AdvancedTurbulenceModel, TurbulenceModelType
//! };
//! use scirs2_core::ndarray::Array3;
//!
//! // Create advanced k-ε model
//! let model = AdvancedTurbulenceModel::new(TurbulenceModelType::KEpsilon, 16, 16, 16);
//!
//! // Solve with SIMD optimization
//! let velocity = vec![
//!     Array3::ones((16, 16, 16)),
//!     Array3::zeros((16, 16, 16)),
//!     Array3::zeros((16, 16, 16)),
//! ];
//! let mut k = Array3::from_elem((16, 16, 16), 1e-3);
//! let mut epsilon = Array3::from_elem((16, 16, 16), 1e-4);
//!
//! model.solve_k_epsilon_simd(&velocity, &mut k, &mut epsilon, 0.01, 0.1, 0.1, 0.1);
//! ```

pub mod advanced_models;
pub mod les;
pub mod models;
pub mod rans;

// Re-export commonly used types and traits
pub use models::{
    RANSModel as RANSModelTrait, SGSModel as SGSModelTrait, TurbulenceConstants, TurbulenceModel,
    TurbulenceModelType, TurbulenceUtils,
};

pub use advanced_models::{
    AdvancedTurbulenceModel, SpalartAllmarasConstants, SpalartAllmarasModel,
};

pub use les::{FluidState3D, LESolver, SGSModel};

pub use rans::{RANSModel, RANSSolver, RANSState};

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_turbulence_module_exports() {
        // Test that all main types can be imported and used
        let _model_type = TurbulenceModelType::KEpsilon;
        let _constants = TurbulenceConstants::default();
        let _sgs_model = SGSModel::Smagorinsky;
        let _rans_model = RANSModel::KEpsilon;
    }

    #[test]
    fn test_rans_integration() {
        let solver = RANSSolver::new(8, 8, RANSModel::KEpsilon, 1000.0);
        let state = RANSState::new(8, 8, 0.1, 0.1);

        assert_eq!(solver.nx, 8);
        assert_eq!(state.mean_velocity.len(), 2);
    }

    #[test]
    fn test_les_integration() {
        let solver = LESolver::new(4, 4, 4, 0.1, 0.1, 0.1, SGSModel::Vreman);

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

        assert_eq!(solver.nx, 4);
        assert_eq!(state.velocity.len(), 3);
    }

    #[test]
    fn test_advanced_model_integration() {
        let model = AdvancedTurbulenceModel::new(TurbulenceModelType::KOmega, 4, 4, 4);
        let sa_model = SpalartAllmarasModel::new(4, 4, 4);

        assert_eq!(model.model_type, TurbulenceModelType::KOmega);
        assert_eq!(sa_model.nu_tilde.dim(), (4, 4, 4));
    }

    #[test]
    fn test_turbulence_utils() {
        let velocity = vec![
            Array3::ones((4, 4, 4)),
            Array3::zeros((4, 4, 4)),
            Array3::zeros((4, 4, 4)),
        ];

        let strain_mag = TurbulenceUtils::compute_strain_rate_magnitude(&velocity, 0.1, 0.1, 0.1);
        assert!(strain_mag.is_ok());

        let vorticity_mag = TurbulenceUtils::compute_vorticity_magnitude(&velocity, 0.1, 0.1, 0.1);
        assert!(vorticity_mag.is_ok());

        let wall_distance = TurbulenceUtils::compute_wall_distance(4, 4, 4);
        assert_eq!(wall_distance.dim(), (4, 4, 4));
    }

    #[test]
    fn test_comprehensive_workflow() {
        // Test a complete workflow using multiple components

        // 1. Create RANS solver
        let rans_solver = RANSSolver::new(6, 6, RANSModel::KOmegaSST, 2000.0);
        let rans_state = RANSState::lid_driven_cavity(6, 6, 0.1, 0.1, 1.0);

        // 2. Create LES solver
        let les_solver = LESolver::new(6, 6, 6, 0.1, 0.1, 0.1, SGSModel::DynamicSmagorinsky);

        // 3. Create advanced model
        let advanced_model = AdvancedTurbulenceModel::new(TurbulenceModelType::KEpsilon, 6, 6, 6);

        // Verify all components work together
        assert_eq!(rans_solver.model_type(), TurbulenceModelType::KOmegaSST);
        assert_eq!(les_solver.sgs_model, SGSModel::DynamicSmagorinsky);
        assert_eq!(advanced_model.model_type, TurbulenceModelType::KEpsilon);
        assert!(rans_state.turbulent_kinetic_energy[[3, 3]] > 0.0);
    }

    #[test]
    fn test_boundary_conditions_integration() {
        // Test that boundary condition methods work across modules
        let mut k = Array3::from_elem((4, 4, 4), 1.0);
        let mut epsilon = Array3::from_elem((4, 4, 4), 1.0);

        let advanced_model = AdvancedTurbulenceModel::new(TurbulenceModelType::KEpsilon, 4, 4, 4);
        let result = advanced_model.apply_turbulence_boundary_conditions(&mut k, &mut epsilon);

        assert!(result.is_ok());
        // Check that boundary values were set correctly
        assert_eq!(k[[0, 0, 0]], 0.0);
        assert_eq!(epsilon[[0, 0, 0]], 1e6);
    }
}
