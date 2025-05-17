//! Finite difference methods for spatial discretization of PDEs
//!
//! This module provides various finite difference schemes for approximating
//! spatial derivatives in PDEs. These schemes can be used with the Method of Lines
//! approach to convert PDEs into systems of ODEs.

use crate::pde::{PDEError, PDEResult};
use ndarray::{Array1, Array2, ArrayView1};

/// Enum for different finite difference schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FiniteDifferenceScheme {
    /// Forward difference: (u[i+1] - u[i]) / dx
    ForwardDifference,

    /// Backward difference: (u[i] - u[i-1]) / dx
    BackwardDifference,

    /// Central difference: (u[i+1] - u[i-1]) / (2*dx)
    CentralDifference,

    /// Higher-order central difference (4th order)
    FourthOrderCentral,

    /// Upwind scheme (for advection)
    Upwind,
}

/// Compute first derivative approximation at a point using the specified scheme
pub fn first_derivative(
    u: &ArrayView1<f64>,
    i: usize,
    dx: f64,
    scheme: FiniteDifferenceScheme,
) -> PDEResult<f64> {
    let n = u.len();

    if i >= n {
        return Err(PDEError::FiniteDifferenceError(format!(
            "Index {} out of bounds for array of length {}",
            i, n
        )));
    }

    match scheme {
        FiniteDifferenceScheme::ForwardDifference => {
            if i == n - 1 {
                return Err(PDEError::FiniteDifferenceError(
                    "Cannot use forward difference at the right boundary".to_string(),
                ));
            }
            Ok((u[i + 1] - u[i]) / dx)
        }
        FiniteDifferenceScheme::BackwardDifference => {
            if i == 0 {
                return Err(PDEError::FiniteDifferenceError(
                    "Cannot use backward difference at the left boundary".to_string(),
                ));
            }
            Ok((u[i] - u[i - 1]) / dx)
        }
        FiniteDifferenceScheme::CentralDifference => {
            if i == 0 || i == n - 1 {
                return Err(PDEError::FiniteDifferenceError(
                    "Cannot use central difference at the boundaries".to_string(),
                ));
            }
            Ok((u[i + 1] - u[i - 1]) / (2.0 * dx))
        }
        FiniteDifferenceScheme::FourthOrderCentral => {
            if i < 2 || i > n - 3 {
                return Err(PDEError::FiniteDifferenceError(
                    "Cannot use 4th-order central difference near boundaries".to_string(),
                ));
            }
            // 4th-order central: (-u[i+2] + 8*u[i+1] - 8*u[i-1] + u[i-2]) / (12*dx)
            Ok((-u[i + 2] + 8.0 * u[i + 1] - 8.0 * u[i - 1] + u[i - 2]) / (12.0 * dx))
        }
        FiniteDifferenceScheme::Upwind => {
            // Upwind requires velocity information - default to central for now
            if i == 0 || i == n - 1 {
                return Err(PDEError::FiniteDifferenceError(
                    "Cannot use upwind scheme at the boundaries without ghost points".to_string(),
                ));
            }
            Ok((u[i + 1] - u[i - 1]) / (2.0 * dx))
        }
    }
}

/// Compute upwind first derivative based on sign of velocity
pub fn upwind_first_derivative(
    u: &ArrayView1<f64>,
    i: usize,
    dx: f64,
    velocity: f64,
) -> PDEResult<f64> {
    let n = u.len();

    if i >= n {
        return Err(PDEError::FiniteDifferenceError(format!(
            "Index {} out of bounds for array of length {}",
            i, n
        )));
    }

    if velocity > 0.0 {
        // Use backward difference for positive velocity
        if i == 0 {
            return Err(PDEError::FiniteDifferenceError(
                "Cannot use backward difference at the left boundary".to_string(),
            ));
        }
        Ok((u[i] - u[i - 1]) / dx)
    } else if velocity < 0.0 {
        // Use forward difference for negative velocity
        if i == n - 1 {
            return Err(PDEError::FiniteDifferenceError(
                "Cannot use forward difference at the right boundary".to_string(),
            ));
        }
        Ok((u[i + 1] - u[i]) / dx)
    } else {
        // Velocity is zero, no advection
        Ok(0.0)
    }
}

/// Compute second derivative approximation at a point
pub fn second_derivative(
    u: &ArrayView1<f64>,
    i: usize,
    dx: f64,
    scheme: FiniteDifferenceScheme,
) -> PDEResult<f64> {
    let n = u.len();

    if i >= n {
        return Err(PDEError::FiniteDifferenceError(format!(
            "Index {} out of bounds for array of length {}",
            i, n
        )));
    }

    match scheme {
        FiniteDifferenceScheme::CentralDifference => {
            if i == 0 || i == n - 1 {
                return Err(PDEError::FiniteDifferenceError(
                    "Cannot compute standard second derivative at boundaries".to_string(),
                ));
            }
            // Central second derivative: (u[i+1] - 2*u[i] + u[i-1]) / dx^2
            Ok((u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx))
        }
        FiniteDifferenceScheme::FourthOrderCentral => {
            if i < 2 || i > n - 3 {
                return Err(PDEError::FiniteDifferenceError(
                    "Cannot use 4th-order central difference near boundaries".to_string(),
                ));
            }
            // 4th-order second derivative:
            // (-u[i+2] + 16*u[i+1] - 30*u[i] + 16*u[i-1] - u[i-2]) / (12*dx^2)
            Ok(
                (-u[i + 2] + 16.0 * u[i + 1] - 30.0 * u[i] + 16.0 * u[i - 1] - u[i - 2])
                    / (12.0 * dx * dx),
            )
        }
        _ => {
            // For other schemes, default to central difference
            if i == 0 || i == n - 1 {
                return Err(PDEError::FiniteDifferenceError(
                    "Cannot compute standard second derivative at boundaries".to_string(),
                ));
            }
            Ok((u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx))
        }
    }
}

/// Generate a finite difference differentiation matrix for the first derivative
pub fn first_derivative_matrix(
    n: usize,
    dx: f64,
    scheme: FiniteDifferenceScheme,
) -> PDEResult<Array2<f64>> {
    if n < 3 {
        return Err(PDEError::FiniteDifferenceError(
            "At least 3 grid points are needed for differentiation matrix".to_string(),
        ));
    }

    let mut matrix = Array2::zeros((n, n));

    match scheme {
        FiniteDifferenceScheme::ForwardDifference => {
            for i in 0..n - 1 {
                matrix[[i, i]] = -1.0 / dx;
                matrix[[i, i + 1]] = 1.0 / dx;
            }
            // Special case for the last row (backward difference)
            matrix[[n - 1, n - 2]] = -1.0 / dx;
            matrix[[n - 1, n - 1]] = 1.0 / dx;
        }
        FiniteDifferenceScheme::BackwardDifference => {
            // Special case for the first row (forward difference)
            matrix[[0, 0]] = -1.0 / dx;
            matrix[[0, 1]] = 1.0 / dx;

            for i in 1..n {
                matrix[[i, i - 1]] = -1.0 / dx;
                matrix[[i, i]] = 1.0 / dx;
            }
        }
        FiniteDifferenceScheme::CentralDifference => {
            // Special case for the first and last rows
            matrix[[0, 0]] = -3.0 / (2.0 * dx);
            matrix[[0, 1]] = 4.0 / (2.0 * dx);
            matrix[[0, 2]] = -1.0 / (2.0 * dx);

            for i in 1..n - 1 {
                matrix[[i, i - 1]] = -1.0 / (2.0 * dx);
                matrix[[i, i + 1]] = 1.0 / (2.0 * dx);
            }

            matrix[[n - 1, n - 3]] = 1.0 / (2.0 * dx);
            matrix[[n - 1, n - 2]] = -4.0 / (2.0 * dx);
            matrix[[n - 1, n - 1]] = 3.0 / (2.0 * dx);
        }
        FiniteDifferenceScheme::FourthOrderCentral => {
            // 4th-order central difference
            // Special case for first two and last two rows

            // First row (forward-biased 4th-order)
            matrix[[0, 0]] = -25.0 / (12.0 * dx);
            matrix[[0, 1]] = 48.0 / (12.0 * dx);
            matrix[[0, 2]] = -36.0 / (12.0 * dx);
            matrix[[0, 3]] = 16.0 / (12.0 * dx);
            matrix[[0, 4]] = -3.0 / (12.0 * dx);

            // Second row (forward-biased 4th-order)
            matrix[[1, 0]] = -3.0 / (12.0 * dx);
            matrix[[1, 1]] = -10.0 / (12.0 * dx);
            matrix[[1, 2]] = 18.0 / (12.0 * dx);
            matrix[[1, 3]] = -6.0 / (12.0 * dx);
            matrix[[1, 4]] = 1.0 / (12.0 * dx);

            // Interior points (4th-order central)
            for i in 2..n - 2 {
                matrix[[i, i - 2]] = 1.0 / (12.0 * dx);
                matrix[[i, i - 1]] = -8.0 / (12.0 * dx);
                matrix[[i, i + 1]] = 8.0 / (12.0 * dx);
                matrix[[i, i + 2]] = -1.0 / (12.0 * dx);
            }

            // Second-to-last row (backward-biased 4th-order)
            matrix[[n - 2, n - 5]] = -1.0 / (12.0 * dx);
            matrix[[n - 2, n - 4]] = 6.0 / (12.0 * dx);
            matrix[[n - 2, n - 3]] = -18.0 / (12.0 * dx);
            matrix[[n - 2, n - 2]] = 10.0 / (12.0 * dx);
            matrix[[n - 2, n - 1]] = 3.0 / (12.0 * dx);

            // Last row (backward-biased 4th-order)
            matrix[[n - 1, n - 5]] = 3.0 / (12.0 * dx);
            matrix[[n - 1, n - 4]] = -16.0 / (12.0 * dx);
            matrix[[n - 1, n - 3]] = 36.0 / (12.0 * dx);
            matrix[[n - 1, n - 2]] = -48.0 / (12.0 * dx);
            matrix[[n - 1, n - 1]] = 25.0 / (12.0 * dx);
        }
        FiniteDifferenceScheme::Upwind => {
            // For upwind, we need velocity information, which isn't available
            // Default to central difference
            return first_derivative_matrix(n, dx, FiniteDifferenceScheme::CentralDifference);
        }
    }

    Ok(matrix)
}

/// Generate a finite difference differentiation matrix for the second derivative
pub fn second_derivative_matrix(
    n: usize,
    dx: f64,
    scheme: FiniteDifferenceScheme,
) -> PDEResult<Array2<f64>> {
    if n < 3 {
        return Err(PDEError::FiniteDifferenceError(
            "At least 3 grid points are needed for second derivative matrix".to_string(),
        ));
    }

    let mut matrix = Array2::zeros((n, n));
    let dx2 = dx * dx;

    match scheme {
        FiniteDifferenceScheme::CentralDifference => {
            // Standard second derivative stencil for interior points
            for i in 1..n - 1 {
                matrix[[i, i - 1]] = 1.0 / dx2;
                matrix[[i, i]] = -2.0 / dx2;
                matrix[[i, i + 1]] = 1.0 / dx2;
            }

            // One-sided approximations for boundaries
            // Left boundary (forward stencil)
            matrix[[0, 0]] = 2.0 / dx2;
            matrix[[0, 1]] = -5.0 / dx2;
            matrix[[0, 2]] = 4.0 / dx2;
            matrix[[0, 3]] = -1.0 / dx2;

            // Right boundary (backward stencil)
            matrix[[n - 1, n - 4]] = -1.0 / dx2;
            matrix[[n - 1, n - 3]] = 4.0 / dx2;
            matrix[[n - 1, n - 2]] = -5.0 / dx2;
            matrix[[n - 1, n - 1]] = 2.0 / dx2;
        }
        FiniteDifferenceScheme::FourthOrderCentral => {
            // 4th-order central stencil for interior points
            // (-u[i+2] + 16*u[i+1] - 30*u[i] + 16*u[i-1] - u[i-2]) / (12*dx^2)
            for i in 2..n - 2 {
                matrix[[i, i - 2]] = -1.0 / (12.0 * dx2);
                matrix[[i, i - 1]] = 16.0 / (12.0 * dx2);
                matrix[[i, i]] = -30.0 / (12.0 * dx2);
                matrix[[i, i + 1]] = 16.0 / (12.0 * dx2);
                matrix[[i, i + 2]] = -1.0 / (12.0 * dx2);
            }

            // Special case for first two rows (one-sided approximations)
            matrix[[0, 0]] = 45.0 / (12.0 * dx2);
            matrix[[0, 1]] = -154.0 / (12.0 * dx2);
            matrix[[0, 2]] = 214.0 / (12.0 * dx2);
            matrix[[0, 3]] = -156.0 / (12.0 * dx2);
            matrix[[0, 4]] = 61.0 / (12.0 * dx2);
            matrix[[0, 5]] = -10.0 / (12.0 * dx2);

            matrix[[1, 0]] = 10.0 / (12.0 * dx2);
            matrix[[1, 1]] = -15.0 / (12.0 * dx2);
            matrix[[1, 2]] = -4.0 / (12.0 * dx2);
            matrix[[1, 3]] = 14.0 / (12.0 * dx2);
            matrix[[1, 4]] = -6.0 / (12.0 * dx2);
            matrix[[1, 5]] = 1.0 / (12.0 * dx2);

            // Special case for last two rows (one-sided approximations)
            matrix[[n - 2, n - 6]] = 1.0 / (12.0 * dx2);
            matrix[[n - 2, n - 5]] = -6.0 / (12.0 * dx2);
            matrix[[n - 2, n - 4]] = 14.0 / (12.0 * dx2);
            matrix[[n - 2, n - 3]] = -4.0 / (12.0 * dx2);
            matrix[[n - 2, n - 2]] = -15.0 / (12.0 * dx2);
            matrix[[n - 2, n - 1]] = 10.0 / (12.0 * dx2);

            matrix[[n - 1, n - 6]] = -10.0 / (12.0 * dx2);
            matrix[[n - 1, n - 5]] = 61.0 / (12.0 * dx2);
            matrix[[n - 1, n - 4]] = -156.0 / (12.0 * dx2);
            matrix[[n - 1, n - 3]] = 214.0 / (12.0 * dx2);
            matrix[[n - 1, n - 2]] = -154.0 / (12.0 * dx2);
            matrix[[n - 1, n - 1]] = 45.0 / (12.0 * dx2);
        }
        _ => {
            // For other schemes, default to central difference
            return second_derivative_matrix(n, dx, FiniteDifferenceScheme::CentralDifference);
        }
    }

    Ok(matrix)
}

/// Apply a differentiation matrix to a vector
pub fn apply_diff_matrix(matrix: &Array2<f64>, u: &ArrayView1<f64>) -> PDEResult<Array1<f64>> {
    if matrix.shape()[1] != u.len() {
        return Err(PDEError::FiniteDifferenceError(format!(
            "Matrix columns ({}) must match vector length ({})",
            matrix.shape()[1],
            u.len()
        )));
    }

    Ok(matrix.dot(u))
}
