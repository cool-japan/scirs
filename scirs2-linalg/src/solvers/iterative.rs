//! Iterative solvers for large linear systems
//!
//! This module provides implementations of iterative methods for solving linear systems
//! Ax = b, particularly suited for large sparse systems where direct methods are impractical.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_traits::{Float, NumAssign};
use std::fmt::{Debug, Display};

/// Options for iterative solvers
#[derive(Debug, Clone)]
pub struct IterativeSolverOptions<A> {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: A,
    /// Whether to print iteration information
    pub verbose: bool,
    /// Restart parameter for GMRES
    pub restart: Option<usize>,
}

impl<A: Float> Default for IterativeSolverOptions<A> {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: A::from(1e-10).unwrap(),
            verbose: false,
            restart: None,
        }
    }
}

/// Result from iterative solver
#[derive(Debug, Clone)]
pub struct IterativeSolverResult<A> {
    /// Solution vector
    pub solution: Array1<A>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: A,
    /// Whether the solver converged
    pub converged: bool,
}

/// Conjugate Gradient solver for symmetric positive definite systems
///
/// Solves Ax = b where A is symmetric positive definite.
///
/// # Arguments
/// * `a` - Symmetric positive definite matrix
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (optional)
/// * `options` - Solver options
///
/// # Example
/// ```
/// use ndarray::array;
/// use scirs2_linalg::solvers::iterative::{conjugate_gradient, IterativeSolverOptions};
///
/// let a = array![[4.0, 1.0], [1.0, 3.0]];
/// let b = array![1.0, 2.0];
/// let options = IterativeSolverOptions::default();
///
/// let result = conjugate_gradient(&a.view(), &b.view(), None, &options).unwrap();
/// assert!(result.converged);
/// ```
pub fn conjugate_gradient<A>(
    a: &ArrayView2<A>,
    b: &ArrayView1<A>,
    x0: Option<&ArrayView1<A>>,
    options: &IterativeSolverOptions<A>,
) -> LinalgResult<IterativeSolverResult<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + 'static,
{
    let n = a.shape()[0];

    // Validate inputs
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError(
            "Matrix A must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(
            "Vector b must have same length as matrix dimension".to_string(),
        ));
    }

    // Initialize solution
    let mut x = match x0 {
        Some(x0_view) => {
            if x0_view.len() != n {
                return Err(LinalgError::ShapeError(
                    "Initial guess x0 must have same length as b".to_string(),
                ));
            }
            x0_view.to_owned()
        }
        None => Array1::zeros(n),
    };

    // Compute initial residual r = b - Ax
    let mut r = b.to_owned() - a.dot(&x);
    let mut r_norm_sq = r.dot(&r);

    // Check if already converged
    if r_norm_sq.sqrt() < options.tolerance {
        return Ok(IterativeSolverResult {
            solution: x,
            iterations: 0,
            residual_norm: r_norm_sq.sqrt(),
            converged: true,
        });
    }

    // Initialize search direction
    let mut p = r.clone();
    let tolerance_sq = options.tolerance * options.tolerance;

    for iteration in 0..options.max_iterations {
        // Compute Ap
        let ap = a.dot(&p);

        // Compute step size alpha = r^T r / (p^T Ap)
        let p_ap = p.dot(&ap);
        if p_ap.abs() < A::epsilon() {
            return Err(LinalgError::ConvergenceError(
                "Conjugate gradient failed: p^T Ap is nearly zero".to_string(),
            ));
        }
        let alpha = r_norm_sq / p_ap;

        // Update solution: x = x + alpha * p
        x.scaled_add(alpha, &p);

        // Update residual: r = r - alpha * Ap
        r.scaled_add(-alpha, &ap);

        // Compute new residual norm
        let r_norm_sq_new = r.dot(&r);

        if options.verbose && iteration % 10 == 0 {
            println!(
                "CG iteration {}: residual = {}",
                iteration,
                r_norm_sq_new.sqrt()
            );
        }

        // Check convergence
        if r_norm_sq_new < tolerance_sq {
            return Ok(IterativeSolverResult {
                solution: x,
                iterations: iteration + 1,
                residual_norm: r_norm_sq_new.sqrt(),
                converged: true,
            });
        }

        // Compute beta = r_new^T r_new / r^T r
        let beta = r_norm_sq_new / r_norm_sq;

        // Update search direction: p = r + beta * p
        p = &r + &p * beta;

        // Update residual norm
        r_norm_sq = r_norm_sq_new;
    }

    // Maximum iterations reached
    Ok(IterativeSolverResult {
        solution: x,
        iterations: options.max_iterations,
        residual_norm: r_norm_sq.sqrt(),
        converged: false,
    })
}

/// Preconditioned Conjugate Gradient solver
///
/// Solves Ax = b where A is symmetric positive definite, using a preconditioner M.
///
/// # Arguments
/// * `a` - Symmetric positive definite matrix
/// * `b` - Right-hand side vector
/// * `preconditioner` - Function that applies M^{-1} to a vector
/// * `x0` - Initial guess (optional)
/// * `options` - Solver options
pub fn preconditioned_conjugate_gradient<A, F>(
    a: &ArrayView2<A>,
    b: &ArrayView1<A>,
    mut preconditioner: F,
    x0: Option<&ArrayView1<A>>,
    options: &IterativeSolverOptions<A>,
) -> LinalgResult<IterativeSolverResult<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + 'static,
    F: FnMut(&ArrayView1<A>) -> Array1<A>,
{
    let n = a.shape()[0];

    // Validate inputs
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError(
            "Matrix A must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(
            "Vector b must have same length as matrix dimension".to_string(),
        ));
    }

    // Initialize solution
    let mut x = match x0 {
        Some(x0_view) => {
            if x0_view.len() != n {
                return Err(LinalgError::ShapeError(
                    "Initial guess x0 must have same length as b".to_string(),
                ));
            }
            x0_view.to_owned()
        }
        None => Array1::zeros(n),
    };

    // Compute initial residual r = b - Ax
    let mut r = b.to_owned() - a.dot(&x);

    // Apply preconditioner: z = M^{-1} r
    let mut z = preconditioner(&r.view());

    // Initialize search direction
    let mut p = z.clone();
    let mut rz_old = r.dot(&z);

    let tolerance_sq = options.tolerance * options.tolerance;

    for iteration in 0..options.max_iterations {
        // Compute Ap
        let ap = a.dot(&p);

        // Compute step size alpha = r^T z / (p^T Ap)
        let p_ap = p.dot(&ap);
        if p_ap.abs() < A::epsilon() {
            return Err(LinalgError::ConvergenceError(
                "PCG failed: p^T Ap is nearly zero".to_string(),
            ));
        }
        let alpha = rz_old / p_ap;

        // Update solution: x = x + alpha * p
        x.scaled_add(alpha, &p);

        // Update residual: r = r - alpha * Ap
        r.scaled_add(-alpha, &ap);

        // Check convergence
        let r_norm_sq = r.dot(&r);

        if options.verbose && iteration % 10 == 0 {
            println!(
                "PCG iteration {}: residual = {}",
                iteration,
                r_norm_sq.sqrt()
            );
        }

        if r_norm_sq < tolerance_sq {
            return Ok(IterativeSolverResult {
                solution: x,
                iterations: iteration + 1,
                residual_norm: r_norm_sq.sqrt(),
                converged: true,
            });
        }

        // Apply preconditioner: z = M^{-1} r
        z = preconditioner(&r.view());

        // Compute r^T z
        let rz_new = r.dot(&z);

        // Compute beta = r_new^T z_new / r^T z
        let beta = rz_new / rz_old;

        // Update search direction: p = z + beta * p
        p = &z + &p * beta;

        // Update rz_old
        rz_old = rz_new;
    }

    // Maximum iterations reached
    let r_norm = r.dot(&r).sqrt();
    Ok(IterativeSolverResult {
        solution: x,
        iterations: options.max_iterations,
        residual_norm: r_norm,
        converged: false,
    })
}

/// GMRES (Generalized Minimal Residual) solver for general linear systems
///
/// Solves Ax = b for general (possibly non-symmetric) matrices.
///
/// # Arguments
/// * `a` - Matrix (can be non-symmetric)
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (optional)
/// * `options` - Solver options (restart parameter can be specified)
pub fn gmres<A>(
    a: &ArrayView2<A>,
    b: &ArrayView1<A>,
    x0: Option<&ArrayView1<A>>,
    options: &IterativeSolverOptions<A>,
) -> LinalgResult<IterativeSolverResult<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + 'static,
{
    let n = a.shape()[0];

    // Validate inputs
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError(
            "Matrix A must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(
            "Vector b must have same length as matrix dimension".to_string(),
        ));
    }

    // Initialize solution
    let mut x = match x0 {
        Some(x0_view) => {
            if x0_view.len() != n {
                return Err(LinalgError::ShapeError(
                    "Initial guess x0 must have same length as b".to_string(),
                ));
            }
            x0_view.to_owned()
        }
        None => Array1::zeros(n),
    };

    // Set restart parameter (default to min(n, 30) if not specified)
    let restart = options.restart.unwrap_or_else(|| n.min(30));

    let mut total_iterations = 0;

    // Outer restart loop
    for _outer in 0..(options.max_iterations / restart).max(1) {
        // Compute initial residual r = b - Ax
        let r = b.to_owned() - a.dot(&x);
        let beta = r.dot(&r).sqrt();

        // Check if already converged
        if beta < options.tolerance {
            return Ok(IterativeSolverResult {
                solution: x,
                iterations: total_iterations,
                residual_norm: beta,
                converged: true,
            });
        }

        // Initialize Krylov subspace basis
        let mut v = vec![Array1::zeros(n); restart + 1];
        v[0] = &r / beta;

        // Initialize Hessenberg matrix
        let mut h = Array2::<A>::zeros((restart + 1, restart));

        // Arnoldi iteration
        let mut j = 0;
        while j < restart && total_iterations < options.max_iterations {
            // Compute w = A * v_j
            let w = a.dot(&v[j]);

            // Gram-Schmidt orthogonalization
            let mut w_orth = w.clone();
            for i in 0..=j {
                h[[i, j]] = w.dot(&v[i]);
                w_orth.scaled_add(-h[[i, j]], &v[i]);
            }

            // Compute norm
            h[[j + 1, j]] = w_orth.dot(&w_orth).sqrt();

            // Check for breakdown
            if h[[j + 1, j]] < A::epsilon() {
                j += 1;
                break;
            }

            // Normalize
            v[j + 1] = &w_orth / h[[j + 1, j]];

            j += 1;
            total_iterations += 1;
        }

        // Solve least squares problem min ||beta*e1 - H*y||
        // where H is the (j+1) x j Hessenberg matrix
        let y = solve_least_squares_gmres(&h.slice(ndarray::s![..j + 1, ..j]), beta)?;

        // Update solution: x = x0 + V * y
        for (i, yi) in y.iter().enumerate() {
            x.scaled_add(*yi, &v[i]);
        }

        // Compute residual for convergence check
        let r_final = b.to_owned() - a.dot(&x);
        let residual_norm = r_final.dot(&r_final).sqrt();

        if options.verbose {
            println!(
                "GMRES outer iteration {}: residual = {}",
                _outer, residual_norm
            );
        }

        if residual_norm < options.tolerance {
            return Ok(IterativeSolverResult {
                solution: x,
                iterations: total_iterations,
                residual_norm,
                converged: true,
            });
        }
    }

    // Maximum iterations reached
    let r_final = b.to_owned() - a.dot(&x);
    let residual_norm = r_final.dot(&r_final).sqrt();

    Ok(IterativeSolverResult {
        solution: x,
        iterations: total_iterations,
        residual_norm,
        converged: false,
    })
}

/// BiCGSTAB (Biconjugate Gradient Stabilized) solver for non-symmetric systems
///
/// Solves Ax = b for general matrices, often faster than GMRES for certain problems.
///
/// # Arguments
/// * `a` - Matrix (can be non-symmetric)
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (optional)
/// * `options` - Solver options
pub fn bicgstab<A>(
    a: &ArrayView2<A>,
    b: &ArrayView1<A>,
    x0: Option<&ArrayView1<A>>,
    options: &IterativeSolverOptions<A>,
) -> LinalgResult<IterativeSolverResult<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + 'static,
{
    let n = a.shape()[0];

    // Validate inputs
    if a.shape()[1] != n {
        return Err(LinalgError::ShapeError(
            "Matrix A must be square".to_string(),
        ));
    }
    if b.len() != n {
        return Err(LinalgError::ShapeError(
            "Vector b must have same length as matrix dimension".to_string(),
        ));
    }

    // Initialize solution
    let mut x = match x0 {
        Some(x0_view) => {
            if x0_view.len() != n {
                return Err(LinalgError::ShapeError(
                    "Initial guess x0 must have same length as b".to_string(),
                ));
            }
            x0_view.to_owned()
        }
        None => Array1::zeros(n),
    };

    // Compute initial residual r = b - Ax
    let mut r = b.to_owned() - a.dot(&x);
    let r_norm_init = r.dot(&r).sqrt();

    // Check if already converged
    if r_norm_init < options.tolerance {
        return Ok(IterativeSolverResult {
            solution: x,
            iterations: 0,
            residual_norm: r_norm_init,
            converged: true,
        });
    }

    // Choose r_hat (usually r_hat = r)
    let r_hat = r.clone();

    // Initialize vectors
    let mut rho = A::one();
    let mut alpha = A::one();
    let mut omega = A::one();
    let mut v = Array1::zeros(n);
    let mut p = Array1::zeros(n);

    for iteration in 0..options.max_iterations {
        let rho_old = rho;
        rho = r_hat.dot(&r);

        if rho.abs() < A::epsilon() {
            return Err(LinalgError::ConvergenceError(
                "BiCGSTAB failed: rho is nearly zero".to_string(),
            ));
        }

        let beta = (rho / rho_old) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        p = &r + &(&p - &v * omega) * beta;

        // v = A * p
        v = a.dot(&p);

        alpha = rho / r_hat.dot(&v);

        // s = r - alpha * v
        let s = &r - &v * alpha;

        // Check if s is small enough
        let s_norm = s.dot(&s).sqrt();
        if s_norm < options.tolerance {
            x.scaled_add(alpha, &p);
            return Ok(IterativeSolverResult {
                solution: x,
                iterations: iteration + 1,
                residual_norm: s_norm,
                converged: true,
            });
        }

        // t = A * s
        let t = a.dot(&s);

        omega = t.dot(&s) / t.dot(&t);

        // Update solution: x = x + alpha * p + omega * s
        x.scaled_add(alpha, &p);
        x.scaled_add(omega, &s);

        // Update residual: r = s - omega * t
        r = &s - &t * omega;

        // Check convergence
        let r_norm = r.dot(&r).sqrt();

        if options.verbose && iteration % 10 == 0 {
            println!("BiCGSTAB iteration {}: residual = {}", iteration, r_norm);
        }

        if r_norm < options.tolerance {
            return Ok(IterativeSolverResult {
                solution: x,
                iterations: iteration + 1,
                residual_norm: r_norm,
                converged: true,
            });
        }

        if omega.abs() < A::epsilon() {
            return Err(LinalgError::ConvergenceError(
                "BiCGSTAB failed: omega is nearly zero".to_string(),
            ));
        }
    }

    // Maximum iterations reached
    let r_final = b.to_owned() - a.dot(&x);
    let residual_norm = r_final.dot(&r_final).sqrt();

    Ok(IterativeSolverResult {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
    })
}

/// Helper function to solve least squares problem for GMRES
fn solve_least_squares_gmres<A>(h: &ArrayView2<A>, beta: A) -> LinalgResult<Array1<A>>
where
    A: Float + NumAssign + Debug + Display + ndarray::ScalarOperand + 'static,
{
    let (m, n) = h.dim();
    if m <= n {
        return Err(LinalgError::ShapeError(
            "Hessenberg matrix must have more rows than columns".to_string(),
        ));
    }

    // QR factorization using Givens rotations
    let mut h_copy = h.to_owned();
    let mut g = Array1::zeros(m);
    g[0] = beta;

    // Apply Givens rotations
    for j in 0..n {
        for i in (j + 1)..m {
            if h_copy[[i, j]].abs() > A::epsilon() {
                // Compute Givens rotation
                let a = h_copy[[j, j]];
                let b = h_copy[[i, j]];
                let r = (a * a + b * b).sqrt();
                let c = a / r;
                let s = b / r;

                // Apply rotation to H
                for k in j..n {
                    let hjk = h_copy[[j, k]];
                    let hik = h_copy[[i, k]];
                    h_copy[[j, k]] = c * hjk + s * hik;
                    h_copy[[i, k]] = -s * hjk + c * hik;
                }

                // Apply rotation to g
                let gj = g[j];
                let gi = g[i];
                g[j] = c * gj + s * gi;
                g[i] = -s * gj + c * gi;
            }
        }
    }

    // Back substitution
    let mut y = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = g[i];
        for j in (i + 1)..n {
            sum -= h_copy[[i, j]] * y[j];
        }
        if h_copy[[i, i]].abs() < A::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular in GMRES least squares solve".to_string(),
            ));
        }
        y[i] = sum / h_copy[[i, i]];
    }

    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_conjugate_gradient() {
        // Simple 2x2 SPD system
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let options = IterativeSolverOptions::default();

        let result = conjugate_gradient(&a.view(), &b.view(), None, &options).unwrap();

        assert!(result.converged);
        assert!(result.iterations < 10);

        // Verify solution
        let residual = &b - a.dot(&result.solution);
        assert!(residual.dot(&residual).sqrt() < 1e-10);
    }

    #[test]
    fn test_gmres() {
        // Non-symmetric system
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![5.0, 11.0];
        let options = IterativeSolverOptions::default();

        let result = gmres(&a.view(), &b.view(), None, &options).unwrap();

        assert!(result.converged);

        // Verify solution (should be [1, 2])
        assert_abs_diff_eq!(result.solution[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.solution[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bicgstab() {
        // Non-symmetric system
        let a = array![[4.0, 1.0], [2.0, 3.0]];
        let b = array![1.0, 2.0];
        let options = IterativeSolverOptions::default();

        let result = bicgstab(&a.view(), &b.view(), None, &options).unwrap();

        assert!(result.converged);

        // Verify solution
        let residual = &b - a.dot(&result.solution);
        assert!(residual.dot(&residual).sqrt() < 1e-10);
    }

    #[test]
    fn test_preconditioned_cg() {
        // Simple diagonal preconditioner
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];

        // Diagonal preconditioner (Jacobi)
        let preconditioner =
            |r: &ArrayView1<f64>| -> Array1<f64> { array![r[0] / 4.0, r[1] / 3.0] };

        let options = IterativeSolverOptions::default();
        let result =
            preconditioned_conjugate_gradient(&a.view(), &b.view(), preconditioner, None, &options)
                .unwrap();

        assert!(result.converged);
        assert!(result.iterations <= 10); // PCG should converge faster

        // Verify solution
        let residual = &b - a.dot(&result.solution);
        assert!(residual.dot(&residual).sqrt() < 1e-10);
    }
}
