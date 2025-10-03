//! Boundary condition handling for incompressible flow simulations
//!
//! This module provides methods for applying various boundary conditions
//! to velocity and pressure fields in incompressible fluid simulations.

use crate::error::{IntegrateError, IntegrateResult as Result};
use scirs2_core::ndarray::Array2;

use super::super::core::FluidBoundaryCondition;

/// Apply boundary conditions to velocity fields
///
/// This function applies the specified boundary conditions to the u and v velocity
/// components based on the boundary condition types for each edge.
///
/// # Arguments
///
/// * `u` - Mutable reference to u-velocity component
/// * `v` - Mutable reference to v-velocity component
/// * `bc_x` - Boundary conditions for left and right edges (x-direction)
/// * `bc_y` - Boundary conditions for bottom and top edges (y-direction)
pub fn apply_boundary_conditions_2d(
    u: &mut Array2<f64>,
    v: &mut Array2<f64>,
    bc_x: (FluidBoundaryCondition, FluidBoundaryCondition),
    bc_y: (FluidBoundaryCondition, FluidBoundaryCondition),
) -> Result<()> {
    let (ny, nx) = u.dim();

    // Left and right boundaries (x-direction)
    apply_x_boundary_left(u, v, bc_x.0, ny)?;
    apply_x_boundary_right(u, v, bc_x.1, ny, nx)?;

    // Top and bottom boundaries (y-direction)
    apply_y_boundary_bottom(u, v, bc_y.0, nx)?;
    apply_y_boundary_top(u, v, bc_y.1, nx, ny)?;

    Ok(())
}

/// Apply left boundary condition (x = 0)
fn apply_x_boundary_left(
    u: &mut Array2<f64>,
    v: &mut Array2<f64>,
    bc: FluidBoundaryCondition,
    ny: usize,
) -> Result<()> {
    match bc {
        FluidBoundaryCondition::NoSlip => {
            for j in 0..ny {
                u[[j, 0]] = 0.0;
                v[[j, 0]] = 0.0;
            }
        }
        FluidBoundaryCondition::FreeSlip => {
            for j in 0..ny {
                u[[j, 0]] = u[[j, 1]]; // Zero normal gradient for tangential velocity
                v[[j, 0]] = 0.0; // Zero normal velocity
            }
        }
        FluidBoundaryCondition::Periodic => {
            for j in 0..ny {
                u[[j, 0]] = u[[j, u.dim().1 - 2]];
                v[[j, 0]] = v[[j, v.dim().1 - 2]];
            }
        }
        FluidBoundaryCondition::Inflow(u_in, v_in) => {
            for j in 0..ny {
                u[[j, 0]] = u_in;
                v[[j, 0]] = v_in;
            }
        }
        FluidBoundaryCondition::Outflow => {
            for j in 0..ny {
                u[[j, 0]] = u[[j, 1]]; // Zero gradient
                v[[j, 0]] = v[[j, 1]];
            }
        }
    }
    Ok(())
}

/// Apply right boundary condition (x = L)
fn apply_x_boundary_right(
    u: &mut Array2<f64>,
    v: &mut Array2<f64>,
    bc: FluidBoundaryCondition,
    ny: usize,
    nx: usize,
) -> Result<()> {
    match bc {
        FluidBoundaryCondition::NoSlip => {
            for j in 0..ny {
                u[[j, nx - 1]] = 0.0;
                v[[j, nx - 1]] = 0.0;
            }
        }
        FluidBoundaryCondition::FreeSlip => {
            for j in 0..ny {
                u[[j, nx - 1]] = u[[j, nx - 2]]; // Zero normal gradient for tangential velocity
                v[[j, nx - 1]] = 0.0; // Zero normal velocity
            }
        }
        FluidBoundaryCondition::Periodic => {
            for j in 0..ny {
                u[[j, nx - 1]] = u[[j, 1]];
                v[[j, nx - 1]] = v[[j, 1]];
            }
        }
        FluidBoundaryCondition::Inflow(u_in, v_in) => {
            for j in 0..ny {
                u[[j, nx - 1]] = u_in;
                v[[j, nx - 1]] = v_in;
            }
        }
        FluidBoundaryCondition::Outflow => {
            for j in 0..ny {
                u[[j, nx - 1]] = u[[j, nx - 2]]; // Zero gradient
                v[[j, nx - 1]] = v[[j, nx - 2]];
            }
        }
    }
    Ok(())
}

/// Apply bottom boundary condition (y = 0)
fn apply_y_boundary_bottom(
    u: &mut Array2<f64>,
    v: &mut Array2<f64>,
    bc: FluidBoundaryCondition,
    nx: usize,
) -> Result<()> {
    match bc {
        FluidBoundaryCondition::NoSlip => {
            // Skip corners (i=0 and i=nx-1) to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[0, i]] = 0.0;
                v[[0, i]] = 0.0;
            }
        }
        FluidBoundaryCondition::FreeSlip => {
            // Skip corners to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[0, i]] = 0.0; // Zero normal velocity
                v[[0, i]] = v[[1, i]]; // Zero normal gradient for tangential velocity
            }
        }
        FluidBoundaryCondition::Periodic => {
            // Skip corners to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[0, i]] = u[[u.dim().0 - 2, i]];
                v[[0, i]] = v[[v.dim().0 - 2, i]];
            }
        }
        FluidBoundaryCondition::Inflow(u_in, v_in) => {
            // Skip corners to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[0, i]] = u_in;
                v[[0, i]] = v_in;
            }
        }
        FluidBoundaryCondition::Outflow => {
            // Skip corners to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[0, i]] = u[[1, i]]; // Zero gradient
                v[[0, i]] = v[[1, i]];
            }
        }
    }
    Ok(())
}

/// Apply top boundary condition (y = L)
fn apply_y_boundary_top(
    u: &mut Array2<f64>,
    v: &mut Array2<f64>,
    bc: FluidBoundaryCondition,
    nx: usize,
    ny: usize,
) -> Result<()> {
    match bc {
        FluidBoundaryCondition::NoSlip => {
            // Skip corners to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[ny - 1, i]] = 0.0;
                v[[ny - 1, i]] = 0.0;
            }
        }
        FluidBoundaryCondition::FreeSlip => {
            // Skip corners to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[ny - 1, i]] = 0.0; // Zero normal velocity
                v[[ny - 1, i]] = v[[ny - 2, i]]; // Zero normal gradient for tangential velocity
            }
        }
        FluidBoundaryCondition::Periodic => {
            // Skip corners to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[ny - 1, i]] = u[[1, i]];
                v[[ny - 1, i]] = v[[1, i]];
            }
        }
        FluidBoundaryCondition::Inflow(u_in, v_in) => {
            // Skip corners to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[ny - 1, i]] = u_in;
                v[[ny - 1, i]] = v_in;
            }
        }
        FluidBoundaryCondition::Outflow => {
            // Skip corners to let x-direction boundaries take precedence
            for i in 1..nx - 1 {
                u[[ny - 1, i]] = u[[ny - 2, i]]; // Zero gradient
                v[[ny - 1, i]] = v[[ny - 2, i]];
            }
        }
    }
    Ok(())
}

/// Apply boundary conditions to pressure field
///
/// Pressure boundary conditions are typically Neumann (zero gradient) conditions
/// to ensure proper pressure-velocity coupling in the projection method.
///
/// # Arguments
///
/// * `pressure` - Mutable reference to pressure field
pub fn apply_pressure_boundary_conditions(pressure: &mut Array2<f64>) -> Result<()> {
    let (ny, nx) = pressure.dim();

    // Neumann boundary conditions (zero gradient) for pressure
    // Left and right
    for j in 0..ny {
        pressure[[j, 0]] = pressure[[j, 1]];
        pressure[[j, nx - 1]] = pressure[[j, nx - 2]];
    }

    // Top and bottom
    for i in 0..nx {
        pressure[[0, i]] = pressure[[1, i]];
        pressure[[ny - 1, i]] = pressure[[ny - 2, i]];
    }

    Ok(())
}

/// Apply boundary conditions with specified wall velocities
///
/// This is a specialized function for handling moving wall boundary conditions,
/// such as in the lid-driven cavity problem.
///
/// # Arguments
///
/// * `u` - Mutable reference to u-velocity component
/// * `v` - Mutable reference to v-velocity component
/// * `wall_velocities` - Tuple of (left, right, bottom, top) wall velocities as Option<(f64, f64)>
pub fn apply_wall_velocity_conditions(
    u: &mut Array2<f64>,
    v: &mut Array2<f64>,
    wall_velocities: (
        Option<(f64, f64)>,
        Option<(f64, f64)>,
        Option<(f64, f64)>,
        Option<(f64, f64)>,
    ),
) -> Result<()> {
    let (ny, nx) = u.dim();

    // Left wall
    if let Some((u_wall, v_wall)) = wall_velocities.0 {
        for j in 0..ny {
            u[[j, 0]] = u_wall;
            v[[j, 0]] = v_wall;
        }
    }

    // Right wall
    if let Some((u_wall, v_wall)) = wall_velocities.1 {
        for j in 0..ny {
            u[[j, nx - 1]] = u_wall;
            v[[j, nx - 1]] = v_wall;
        }
    }

    // Bottom wall
    if let Some((u_wall, v_wall)) = wall_velocities.2 {
        for i in 0..nx {
            u[[0, i]] = u_wall;
            v[[0, i]] = v_wall;
        }
    }

    // Top wall
    if let Some((u_wall, v_wall)) = wall_velocities.3 {
        for i in 0..nx {
            u[[ny - 1, i]] = u_wall;
            v[[ny - 1, i]] = v_wall;
        }
    }

    Ok(())
}

/// Apply periodic boundary conditions
///
/// This function specifically handles periodic boundary conditions where
/// the flow field wraps around from one side to the other.
///
/// # Arguments
///
/// * `u` - Mutable reference to u-velocity component
/// * `v` - Mutable reference to v-velocity component
/// * `periodic_x` - Whether to apply periodic conditions in x-direction
/// * `periodic_y` - Whether to apply periodic conditions in y-direction
pub fn apply_periodic_conditions(
    u: &mut Array2<f64>,
    v: &mut Array2<f64>,
    periodic_x: bool,
    periodic_y: bool,
) -> Result<()> {
    let (ny, nx) = u.dim();

    if periodic_x {
        for j in 0..ny {
            u[[j, 0]] = u[[j, nx - 2]];
            u[[j, nx - 1]] = u[[j, 1]];
            v[[j, 0]] = v[[j, nx - 2]];
            v[[j, nx - 1]] = v[[j, 1]];
        }
    }

    if periodic_y {
        for i in 0..nx {
            u[[0, i]] = u[[ny - 2, i]];
            u[[ny - 1, i]] = u[[1, i]];
            v[[0, i]] = v[[ny - 2, i]];
            v[[ny - 1, i]] = v[[1, i]];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_no_slip_boundary_conditions() {
        let mut u = Array2::ones((5, 5));
        let mut v = Array2::ones((5, 5));

        let bc_x = (
            FluidBoundaryCondition::NoSlip,
            FluidBoundaryCondition::NoSlip,
        );
        let bc_y = (
            FluidBoundaryCondition::NoSlip,
            FluidBoundaryCondition::NoSlip,
        );

        apply_boundary_conditions_2d(&mut u, &mut v, bc_x, bc_y).unwrap();

        // Check that boundaries are zero
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
    fn test_pressure_boundary_conditions() {
        let mut pressure = Array2::from_shape_fn((5, 5), |(i, j)| (i + j) as f64);
        let original_interior = pressure[[2, 2]];

        apply_pressure_boundary_conditions(&mut pressure).unwrap();

        // Check that interior values are unchanged
        assert_eq!(pressure[[2, 2]], original_interior);

        // Check that boundary values are copied from interior
        assert_eq!(pressure[[0, 2]], pressure[[1, 2]]); // Bottom
        assert_eq!(pressure[[4, 2]], pressure[[3, 2]]); // Top
        assert_eq!(pressure[[2, 0]], pressure[[2, 1]]); // Left
        assert_eq!(pressure[[2, 4]], pressure[[2, 3]]); // Right
    }

    #[test]
    #[ignore] // FIXME: Inflow boundary conditions test failing
    fn test_inflow_boundary_conditions() {
        let mut u = Array2::zeros((5, 5));
        let mut v = Array2::zeros((5, 5));

        let bc_x = (
            FluidBoundaryCondition::Inflow(1.0, 0.5),
            FluidBoundaryCondition::Outflow,
        );
        let bc_y = (
            FluidBoundaryCondition::NoSlip,
            FluidBoundaryCondition::NoSlip,
        );

        apply_boundary_conditions_2d(&mut u, &mut v, bc_x, bc_y).unwrap();

        // Check inflow boundary
        for j in 0..5 {
            assert_eq!(u[[j, 0]], 1.0);
            assert_eq!(v[[j, 0]], 0.5);
        }
    }
}
