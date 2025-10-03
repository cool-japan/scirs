//! Incompressible fluid dynamics module
//!
//! This module provides comprehensive tools for simulating incompressible fluid flows
//! using the Navier-Stokes equations. It includes solvers, initialization functions,
//! and boundary condition handling for various fluid dynamics scenarios.
//!
//! # Main Components
//!
//! - [`NavierStokesSolver`]: Main solver for incompressible Navier-Stokes equations
//! - [`initialization`]: Functions for setting up initial conditions
//! - [`boundary_handling`]: Boundary condition application methods
//!
//! # Example Usage
//!
//! ```rust
//! use scirs2_integrate::specialized::fluid_dynamics::incompressible::*;
//! use scirs2_integrate::specialized::fluid_dynamics::core::*;
//!
//! // Create solver parameters
//! let params = NavierStokesParams::default();
//! let bc_x = (FluidBoundaryCondition::NoSlip, FluidBoundaryCondition::NoSlip);
//! let bc_y = (FluidBoundaryCondition::NoSlip, FluidBoundaryCondition::NoSlip);
//!
//! // Create solver
//! let solver = NavierStokesSolver::new(params, bc_x, bc_y);
//!
//! // Initialize with lid-driven cavity
//! let initial_state = initialization::lid_driven_cavity(64, 64, 1.0);
//!
//! // Solve
//! let results = solver.solve_2d(initial_state, 1.0, 10).unwrap();
//! ```

pub mod boundary_handling;
pub mod initialization;
pub mod solver;

// Re-export main components
pub use boundary_handling::{
    apply_boundary_conditions_2d, apply_periodic_conditions, apply_pressure_boundary_conditions,
    apply_wall_velocity_conditions,
};
pub use initialization::{
    channel_flow, couette_flow, lid_driven_cavity, poiseuille_flow, stagnation_point_flow,
    taylor_green_vortex, vortex_pair,
};
pub use solver::NavierStokesSolver;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::fluid_dynamics::core::*;

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
    fn test_lid_driven_cavity_initialization() {
        let state = initialization::lid_driven_cavity(32, 32, 1.0);

        assert_eq!(state.velocity.len(), 2);
        assert_eq!(state.velocity[0].dim(), (32, 32));
        assert_eq!(state.velocity[1].dim(), (32, 32));
        assert_eq!(state.pressure.dim(), (32, 32));

        // Check that lid velocity is set correctly
        for i in 0..32 {
            assert_eq!(state.velocity[0][[31, i]], 1.0);
        }
    }

    #[test]
    fn test_taylor_green_vortex_initialization() {
        let state = initialization::taylor_green_vortex(16, 16, 1.0, 1.0);

        assert_eq!(state.velocity.len(), 2);
        assert_eq!(state.velocity[0].dim(), (16, 16));
        assert_eq!(state.velocity[1].dim(), (16, 16));
        assert_eq!(state.pressure.dim(), (16, 16));

        // Basic sanity check - velocity should not be all zeros
        let u_sum: f64 = state.velocity[0].iter().map(|&x| x.abs()).sum();
        let v_sum: f64 = state.velocity[1].iter().map(|&x| x.abs()).sum();

        assert!(u_sum > 0.0);
        assert!(v_sum > 0.0);
    }

    #[test]
    fn test_poiseuille_flow_initialization() {
        let state = initialization::poiseuille_flow(32, 16, 2.0);

        assert_eq!(state.velocity.len(), 2);

        // Check that maximum velocity is approximately correct
        let max_u = state.velocity[0].iter().fold(0.0f64, |a, &b| a.max(b));
        assert!((max_u - 2.0).abs() < 0.1);

        // Check that v-velocity is zero everywhere
        let v_sum: f64 = state.velocity[1].iter().map(|&x| x.abs()).sum();
        assert!(v_sum < 1e-10);
    }

    #[test]
    fn test_boundary_condition_application() {
        let mut u = scirs2_core::ndarray::Array2::ones((5, 5));
        let mut v = scirs2_core::ndarray::Array2::ones((5, 5));

        let bc_x = (
            FluidBoundaryCondition::NoSlip,
            FluidBoundaryCondition::NoSlip,
        );
        let bc_y = (
            FluidBoundaryCondition::NoSlip,
            FluidBoundaryCondition::NoSlip,
        );

        boundary_handling::apply_boundary_conditions_2d(&mut u, &mut v, bc_x, bc_y).unwrap();

        // Check that boundaries are zero for no-slip condition
        for i in 0..5 {
            assert_eq!(u[[0, i]], 0.0); // Bottom
            assert_eq!(u[[4, i]], 0.0); // Top
            assert_eq!(v[[0, i]], 0.0); // Bottom
            assert_eq!(v[[4, i]], 0.0); // Top
        }

        for j in 0..5 {
            assert_eq!(u[[j, 0]], 0.0); // Left
            assert_eq!(u[[j, 4]], 0.0); // Right
            assert_eq!(v[[j, 0]], 0.0); // Left
            assert_eq!(v[[j, 4]], 0.0); // Right
        }
    }

    #[test]
    fn test_solver_step() {
        let params = NavierStokesParams {
            nu: 0.01,
            rho: 1.0,
            dt: 0.001,
            max_pressure_iter: 50,
            pressure_tol: 1e-6,
            semi_lagrangian: false,
        };
        let bc_x = (
            FluidBoundaryCondition::NoSlip,
            FluidBoundaryCondition::NoSlip,
        );
        let bc_y = (
            FluidBoundaryCondition::NoSlip,
            FluidBoundaryCondition::NoSlip,
        );

        let solver = NavierStokesSolver::new(params, bc_x, bc_y);
        let mut state = initialization::lid_driven_cavity(16, 16, 1.0);

        let initial_time = state.time;
        solver.step(&mut state).unwrap();

        // Check that time was advanced
        assert!((state.time - initial_time - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_vortex_pair_initialization() {
        let state = initialization::vortex_pair(32, 32, 1.0, 0.4);

        assert_eq!(state.velocity.len(), 2);

        // Check that velocity field is not zero
        let u_sum: f64 = state.velocity[0].iter().map(|&x| x.abs()).sum();
        let v_sum: f64 = state.velocity[1].iter().map(|&x| x.abs()).sum();

        assert!(u_sum > 0.0);
        assert!(v_sum > 0.0);
    }
}
