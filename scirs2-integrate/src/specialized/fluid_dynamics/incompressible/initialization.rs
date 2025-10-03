//! Initialization functions for incompressible flow simulations
//!
//! This module provides various initial condition setups for common fluid dynamics
//! problems, including lid-driven cavity, Taylor-Green vortex, and other benchmark cases.

use scirs2_core::constants::PI;
use scirs2_core::ndarray::Array2;

use super::super::core::FluidState;

/// Create lid-driven cavity initial condition
///
/// This is a classic benchmark problem where the top boundary moves with a constant
/// velocity while all other boundaries are no-slip walls.
///
/// # Arguments
///
/// * `nx` - Number of grid points in x-direction  
/// * `ny` - Number of grid points in y-direction
/// * `lid_velocity` - Velocity of the moving lid at the top boundary
///
/// # Returns
///
/// A `FluidState` with the initial conditions for lid-driven cavity flow
pub fn lid_driven_cavity(nx: usize, ny: usize, lidvelocity: f64) -> FluidState {
    let mut u = Array2::zeros((ny, nx));
    let v = Array2::zeros((ny, nx));

    // Set lid velocity at top boundary
    for i in 0..nx {
        u[[ny - 1, i]] = lidvelocity;
    }

    let pressure = Array2::zeros((ny, nx));
    let dx = 1.0 / (nx - 1) as f64;
    let dy = 1.0 / (ny - 1) as f64;

    FluidState {
        velocity: vec![u, v],
        pressure,
        temperature: None,
        time: 0.0,
        dx,
        dy,
    }
}

/// Create Taylor-Green vortex initial condition
///
/// The Taylor-Green vortex is an exact solution to the 2D incompressible Navier-Stokes
/// equations, commonly used for validation and testing of numerical methods.
///
/// # Arguments
///
/// * `nx` - Number of grid points in x-direction
/// * `ny` - Number of grid points in y-direction  
/// * `a` - Scaling parameter for x-direction
/// * `b` - Scaling parameter for y-direction
///
/// # Returns
///
/// A `FluidState` with the Taylor-Green vortex initial conditions
pub fn taylor_green_vortex(nx: usize, ny: usize, a: f64, b: f64) -> FluidState {
    let dx = 2.0 * PI / (nx - 1) as f64;
    let dy = 2.0 * PI / (ny - 1) as f64;

    let mut u = Array2::zeros((ny, nx));
    let mut v = Array2::zeros((ny, nx));
    let mut pressure = Array2::zeros((ny, nx));

    for j in 0..ny {
        for i in 0..nx {
            let x = i as f64 * dx;
            let y = j as f64 * dy;

            u[[j, i]] = a * (x / a).cos() * (y / b).sin();
            v[[j, i]] = -b * (x / a).sin() * (y / b).cos();
            pressure[[j, i]] = -0.25 * ((2.0 * x / a).cos() + (2.0 * y / b).cos());
        }
    }

    FluidState {
        velocity: vec![u, v],
        pressure,
        temperature: None,
        time: 0.0,
        dx,
        dy,
    }
}

/// Create Poiseuille flow initial condition
///
/// Poiseuille flow is the steady laminar flow between two parallel plates
/// or in a circular pipe. This creates a parabolic velocity profile.
///
/// # Arguments
///
/// * `nx` - Number of grid points in x-direction
/// * `ny` - Number of grid points in y-direction
/// * `max_velocity` - Maximum velocity at the centerline
///
/// # Returns
///
/// A `FluidState` with Poiseuille flow initial conditions
pub fn poiseuille_flow(nx: usize, ny: usize, maxvelocity: f64) -> FluidState {
    let mut u = Array2::zeros((ny, nx));
    let v = Array2::zeros((ny, nx));

    let dx = 1.0 / (nx - 1) as f64;
    let dy = 1.0 / (ny - 1) as f64;

    // Create parabolic velocity profile
    for j in 0..ny {
        for i in 0..nx {
            let y_normalized = j as f64 / (ny - 1) as f64; // Normalized y from 0 to 1
            let y_centered = y_normalized - 0.5; // Center around 0

            // Parabolic profile: u = u_max * (1 - 4y²) for y in [-0.5, 0.5]
            u[[j, i]] = maxvelocity * (0.25 - y_centered * y_centered) * 4.0;
        }
    }

    let pressure = Array2::zeros((ny, nx));

    FluidState {
        velocity: vec![u, v],
        pressure,
        temperature: None,
        time: 0.0,
        dx,
        dy,
    }
}

/// Create channel flow initial condition
///
/// Channel flow is flow between two parallel walls with a specified inlet velocity.
/// This is similar to Poiseuille flow but can be used as an initial condition
/// that develops into the full parabolic profile.
///
/// # Arguments
///
/// * `nx` - Number of grid points in x-direction
/// * `ny` - Number of grid points in y-direction
/// * `inlet_velocity` - Uniform inlet velocity
///
/// # Returns
///
/// A `FluidState` with uniform channel flow initial conditions
pub fn channel_flow(nx: usize, ny: usize, inletvelocity: f64) -> FluidState {
    let mut u = Array2::zeros((ny, nx));
    let v = Array2::zeros((ny, nx));

    let dx = 1.0 / (nx - 1) as f64;
    let dy = 1.0 / (ny - 1) as f64;

    // Set uniform velocity in the interior (excluding boundary layers)
    for j in 1..ny - 1 {
        for i in 0..nx {
            u[[j, i]] = inletvelocity;
        }
    }

    let pressure = Array2::zeros((ny, nx));

    FluidState {
        velocity: vec![u, v],
        pressure,
        temperature: None,
        time: 0.0,
        dx,
        dy,
    }
}

/// Create Couette flow initial condition
///
/// Couette flow is the flow between two parallel plates where one plate moves
/// at a constant velocity relative to the other, creating a linear velocity profile.
///
/// # Arguments
///
/// * `nx` - Number of grid points in x-direction
/// * `ny` - Number of grid points in y-direction
/// * `wall_velocity` - Velocity of the moving wall
///
/// # Returns
///
/// A `FluidState` with Couette flow initial conditions
pub fn couette_flow(nx: usize, ny: usize, wallvelocity: f64) -> FluidState {
    let mut u = Array2::zeros((ny, nx));
    let v = Array2::zeros((ny, nx));

    let dx = 1.0 / (nx - 1) as f64;
    let dy = 1.0 / (ny - 1) as f64;

    // Create linear velocity profile
    for j in 0..ny {
        for i in 0..nx {
            let y_normalized = j as f64 / (ny - 1) as f64; // Normalized y from 0 to 1
            u[[j, i]] = wallvelocity * y_normalized;
        }
    }

    let pressure = Array2::zeros((ny, nx));

    FluidState {
        velocity: vec![u, v],
        pressure,
        temperature: None,
        time: 0.0,
        dx,
        dy,
    }
}

/// Create stagnation point flow initial condition
///
/// Stagnation point flow represents the flow near a stagnation point where
/// fluid approaches and then divides, creating a specific velocity pattern.
///
/// # Arguments
///
/// * `nx` - Number of grid points in x-direction
/// * `ny` - Number of grid points in y-direction
/// * `strength` - Strength of the stagnation point flow
///
/// # Returns
///
/// A `FluidState` with stagnation point flow initial conditions
pub fn stagnation_point_flow(nx: usize, ny: usize, strength: f64) -> FluidState {
    let mut u = Array2::zeros((ny, nx));
    let mut v = Array2::zeros((ny, nx));

    let dx = 1.0 / (nx - 1) as f64;
    let dy = 1.0 / (ny - 1) as f64;

    // Create stagnation point flow pattern
    for j in 0..ny {
        for i in 0..nx {
            let x = (i as f64 / (nx - 1) as f64 - 0.5) * 2.0; // Normalize to [-1, 1]
            let y = (j as f64 / (ny - 1) as f64 - 0.5) * 2.0; // Normalize to [-1, 1]

            u[[j, i]] = strength * x;
            v[[j, i]] = -strength * y;
        }
    }

    let pressure = Array2::zeros((ny, nx));

    FluidState {
        velocity: vec![u, v],
        pressure,
        temperature: None,
        time: 0.0,
        dx,
        dy,
    }
}

/// Create vortex pair initial condition
///
/// Creates a pair of counter-rotating vortices, useful for testing vortex dynamics
/// and interaction phenomena.
///
/// # Arguments
///
/// * `nx` - Number of grid points in x-direction
/// * `ny` - Number of grid points in y-direction
/// * `vortex_strength` - Strength of the vortices
/// * `separation` - Distance between vortex centers (as fraction of domain size)
///
/// # Returns
///
/// A `FluidState` with vortex pair initial conditions
pub fn vortex_pair(nx: usize, ny: usize, vortexstrength: f64, separation: f64) -> FluidState {
    let mut u = Array2::zeros((ny, nx));
    let mut v = Array2::zeros((ny, nx));

    let dx = 1.0 / (nx - 1) as f64;
    let dy = 1.0 / (ny - 1) as f64;

    // Vortex centers
    let center1_x = 0.5 - separation * 0.5;
    let center1_y = 0.5;
    let center2_x = 0.5 + separation * 0.5;
    let center2_y = 0.5;

    for j in 0..ny {
        for i in 0..nx {
            let x = i as f64 / (nx - 1) as f64;
            let y = j as f64 / (ny - 1) as f64;

            // Distance and angle to first vortex
            let dx1 = x - center1_x;
            let dy1 = y - center1_y;
            let r1_sq = dx1 * dx1 + dy1 * dy1;
            let r1 = r1_sq.sqrt();

            // Distance and angle to second vortex
            let dx2 = x - center2_x;
            let dy2 = y - center2_y;
            let r2_sq = dx2 * dx2 + dy2 * dy2;
            let r2 = r2_sq.sqrt();

            // Velocity from first vortex (counter-clockwise)
            if r1 > 1e-10 {
                u[[j, i]] +=
                    vortexstrength * (-dy1) / (2.0 * PI * r1_sq) * (1.0 - (-r1_sq * 10.0).exp());
                v[[j, i]] +=
                    vortexstrength * dx1 / (2.0 * PI * r1_sq) * (1.0 - (-r1_sq * 10.0).exp());
            }

            // Velocity from second vortex (clockwise)
            if r2 > 1e-10 {
                u[[j, i]] +=
                    -vortexstrength * (-dy2) / (2.0 * PI * r2_sq) * (1.0 - (-r2_sq * 10.0).exp());
                v[[j, i]] +=
                    -vortexstrength * dx2 / (2.0 * PI * r2_sq) * (1.0 - (-r2_sq * 10.0).exp());
            }
        }
    }

    let pressure = Array2::zeros((ny, nx));

    FluidState {
        velocity: vec![u, v],
        pressure,
        temperature: None,
        time: 0.0,
        dx,
        dy,
    }
}
