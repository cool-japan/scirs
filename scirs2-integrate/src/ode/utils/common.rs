//! Common utilities for ODE solvers
//!
//! This module provides common utilities used by multiple ODE solvers.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use ndarray::{Array1, Array2, ArrayView1};

/// Result of a single integration step
pub enum StepResult<F: IntegrateFloat> {
    /// Step accepted with the given solution
    Accepted(Array1<F>),
    /// Step rejected
    Rejected,
    /// Should switch methods (for LSODA)
    ShouldSwitch,
}

/// State information for ODE solvers
pub struct ODEState<F: IntegrateFloat> {
    /// Current time
    pub t: F,
    /// Current solution
    pub y: Array1<F>,
    /// Current derivative
    pub dy: Array1<F>,
    /// Current step size
    pub h: F,
    /// Function evaluations
    pub func_evals: usize,
    /// Steps taken
    pub steps: usize,
    /// Accepted steps
    pub accepted_steps: usize,
    /// Rejected steps
    pub rejected_steps: usize,
}

/// Type of ODE problem
pub enum ODEType {
    /// Non-stiff problem
    NonStiff,
    /// Stiff problem
    Stiff,
    /// Problem that changes between stiff and non-stiff
    Mixed,
}

/// Calculate a safe step size based on function derivatives
pub fn estimate_initial_step<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    dy: &Array1<F>,
    tol: F,
    tend: F,
) -> F
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    // Calculate a scaling factor based on the solution magnitude
    let mut d0 = F::zero();
    for i in 0..y.len() {
        let sc = (y[i].abs() + dy[i].abs()).max(tol);
        d0 = d0.max(dy[i].abs() / sc);
    }

    // Set a reasonable default
    if d0 < tol {
        d0 = F::one();
    }

    // Initial step size
    let dt = (F::from_f64(0.01).unwrap() / d0).min(F::from_f64(0.1).unwrap() * (tend - t).abs());

    // Evaluate f at t + small step to estimate second derivative
    let t_new = t + dt * F::from_f64(0.001).unwrap();
    let y_new = y + &(dy * (t_new - t));
    let dy_new = f(t_new, y_new.view());

    // Calculate an estimate of the second derivative
    let mut d1 = F::zero();
    for i in 0..y.len() {
        let sc = (y[i].abs() + dy[i].abs()).max(tol);
        d1 = d1.max((dy_new[i] - dy[i]).abs() / (sc * (t_new - t)));
    }

    // Prevent division by zero
    if d1 < tol {
        d1 = tol;
    }

    // Calculate step size based on error tolerance
    let h1 = (F::from_f64(0.01).unwrap() / d1).sqrt();

    // Choose the smaller of the two estimates
    let mut h = h1.min(dt * F::from_f64(100.0).unwrap());

    // Make sure step size is not too large
    h = h.min((tend - t).abs() * F::from_f64(0.1).unwrap());

    // Ensure the step is in the correct direction
    if tend < t {
        h = -h;
    }

    h
}

/// Calculate finite difference approximation of the jacobian matrix
pub fn finite_difference_jacobian<F, Func>(
    f: &Func,
    t: F,
    y: &Array1<F>,
    f_eval: &Array1<F>,
    _perturbation_scale: F,
) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n_dim = y.len();
    let mut jacobian = Array2::<F>::zeros((n_dim, n_dim));

    // Calculate appropriate perturbation size
    let eps_base = F::from_f64(1e-8).unwrap();

    for i in 0..n_dim {
        // Scale perturbation by variable magnitude
        let eps = eps_base * (F::one() + y[i].abs()).max(F::one());

        // Perturb the i-th component
        let mut y_perturbed = y.clone();
        y_perturbed[i] += eps;

        // Evaluate function at perturbed point
        let f_perturbed = f(t, y_perturbed.view());

        // Calculate the i-th column of the Jacobian using finite differences
        for j in 0..n_dim {
            jacobian[[j, i]] = (f_perturbed[j] - f_eval[j]) / eps;
        }
    }

    jacobian
}

/// Apply a scaled norm to an array
pub fn scaled_norm<F: IntegrateFloat>(v: &Array1<F>, scale: &Array1<F>) -> F {
    let mut max_err = F::zero();
    for i in 0..v.len() {
        let err = v[i].abs() / scale[i];
        max_err = max_err.max(err);
    }
    max_err
}

/// Calculate scaling factors for error control
pub fn calculate_error_weights<F: IntegrateFloat>(y: &Array1<F>, atol: F, rtol: F) -> Array1<F> {
    let mut weights = Array1::<F>::zeros(y.len());
    for i in 0..y.len() {
        weights[i] = atol + rtol * y[i].abs();
    }
    weights
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting
pub fn solve_linear_system<F: IntegrateFloat>(
    a: &Array2<F>,
    b: &Array1<F>,
) -> IntegrateResult<Array1<F>> {
    let n = a.shape()[0];
    if n != a.shape()[1] || n != b.len() {
        return Err(IntegrateError::DimensionMismatch(
            "Matrix dimensions do not match for linear solve".to_string(),
        ));
    }

    // Create augmented matrix [A|b]
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_idx = i;
        let mut max_val = aug[[i, i]].abs();

        for j in i + 1..n {
            if aug[[j, i]].abs() > max_val {
                max_idx = j;
                max_val = aug[[j, i]].abs();
            }
        }

        // Check if matrix is singular
        if max_val < F::from_f64(1e-10).unwrap() {
            return Err(IntegrateError::LinearSolveError(
                "Matrix is singular".to_string(),
            ));
        }

        // Swap rows if necessary
        if max_idx != i {
            for j in 0..n + 1 {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_idx, j]];
                aug[[max_idx, j]] = temp;
            }
        }

        // Eliminate below
        for j in i + 1..n {
            let factor = aug[[j, i]] / aug[[i, i]];
            aug[[j, i]] = F::zero();

            for k in i + 1..n + 1 {
                aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);

    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];

        for j in i + 1..n {
            sum -= aug[[i, j]] * x[j];
        }

        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Extrapolate solution values for use as initial guess
pub fn extrapolate<F: IntegrateFloat>(times: &[F], values: &[Array1<F>], t_target: F) -> Array1<F> {
    let n = values.len();

    if n == 0 {
        panic!("Cannot extrapolate from empty values");
    }

    if n == 1 {
        return values[0].clone();
    }

    // Linear extrapolation if we have 2 points
    if n == 2 {
        let dt = times[1] - times[0];
        if dt.abs() < F::from_f64(1e-10).unwrap() {
            return values[1].clone();
        }

        let t_ratio = (t_target - times[1]) / dt;
        return &values[1] + &((&values[1] - &values[0]) * t_ratio);
    }

    // Quadratic extrapolation if we have 3 or more points
    let t0 = times[n - 3];
    let t1 = times[n - 2];
    let t2 = times[n - 1];

    let y0 = &values[n - 3];
    let y1 = &values[n - 2];
    let y2 = &values[n - 1];

    // Compute quadratic Lagrange extrapolation
    let dt0 = t_target - t0;
    let dt1 = t_target - t1;
    let dt2 = t_target - t2;

    let dt01 = t0 - t1;
    let dt02 = t0 - t2;
    let dt12 = t1 - t2;

    let c0 = dt1 * dt2 / (dt01 * dt02);
    let c1 = dt0 * dt2 / (-dt01 * dt12);
    let c2 = dt0 * dt1 / (dt02 * dt12);

    y0 * c0 + y1 * c1 + y2 * c2
}
