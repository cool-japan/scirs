//! Diagnostic tools for ODE solvers
//!
//! This module provides diagnostic utilities for ODE solvers.

use crate::IntegrateFloat;
use scirs2_core::ndarray::{Array1, ArrayView1};

/// Calculate stability metrics for a solution
///
/// # Arguments
///
/// * `t` - Time points
/// * `y` - Solution values
///
/// # Returns
///
/// Stability metrics for the solution
#[allow(dead_code)]
pub fn stability_metrics<F: IntegrateFloat>(t: &[F], y: &[Array1<F>]) -> F {
    // Simple implementation for now
    // In the future, this could compute eigenvalues of the Jacobian, etc.

    if t.len() < 2 || y.len() < 2 {
        return F::infinity(); // Not enough data
    }

    // Calculate growth rate
    let mut max_growth = F::zero();

    for i in 1..t.len() {
        let dt = t[i] - t[i - 1];

        if dt <= F::zero() {
            continue;
        }

        for j in 0..y[i].len() {
            if j >= y[i - 1].len() {
                break;
            }

            let dy = (y[i][j] - y[i - 1][j]).abs();
            let growth = dy / dt;

            if growth > max_growth {
                max_growth = growth;
            }
        }
    }

    max_growth
}

/// Detect whether a system is stiff based on solution behavior
///
/// # Arguments
///
/// * `t` - Time points
/// * `y` - Solution values
/// * `f` - ODE function
///
/// # Returns
///
/// A stiffness score (higher values indicate more stiffness)
#[allow(dead_code)]
pub fn stiffness_detector<F, Func>(t: &[F], y: &[Array1<F>], f: &Func) -> F
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Simple implementation - could be enhanced later

    if t.len() < 3 || y.len() < 3 {
        return F::zero(); // Not enough data
    }

    // Compute a basic stiffness measure using step size changes
    let mut stiffness = F::zero();

    for i in 2..t.len() {
        let h1 = t[i - 1] - t[i - 2];
        let h2 = t[i] - t[i - 1];

        if h1 <= F::zero() || h2 <= F::zero() {
            continue;
        }

        // Calculate how much the step size had to be reduced
        if h2 < h1 {
            let ratio = h2 / h1;
            stiffness += F::one() - ratio;
        }
    }

    stiffness
}
