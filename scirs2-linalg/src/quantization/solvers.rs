//! Specialized iterative solvers for quantized matrix representations
//!
//! This module provides optimized iterative solvers designed specifically
//! for quantized matrices. These implementations leverage the properties
//! of quantized data to achieve better performance and memory efficiency.

use crate::error::{LinalgError, LinalgResult};
use crate::matrixfree::MatrixFreeOp;
use crate::norm::vector_norm;
use crate::quantization::quantized_matrixfree::QuantizedMatrixFreeOp;
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::{AsPrimitive, Float, FromPrimitive, NumAssign, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

/// Optimized conjugate gradient solver for quantized matrices
///
/// This implementation is specifically designed for quantized matrices,
/// with optimizations that take advantage of quantization properties.
///
/// # Arguments
///
/// * `a` - Quantized matrix-free operator representing the coefficient matrix
/// * `b` - Right-hand side vector
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
/// * `adaptive_precision` - Whether to adaptively adjust precision during iterations
///
/// # Returns
///
/// * Solution vector x
#[allow(dead_code)]
pub fn quantized_conjugate_gradient<F>(
    a: &QuantizedMatrixFreeOp<F>,
    b: &Array1<F>,
    max_iter: usize,
    tol: F,
    adaptive_precision: bool,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Debug + Send + Sync,
{
    // Check that A is square
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Expected square operator, got shape {}x{}",
            a.nrows(),
            a.ncols()
        )));
    }

    // Check that dimensions are compatible
    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch: operator shape {}x{}, vector shape {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }

    // Verify that A is symmetric positive definite
    if !a.is_symmetric() {
        return Err(LinalgError::ValueError(
            "Quantized conjugate gradient requires a symmetric operator".to_string(),
        ));
    }

    let n = a.nrows();

    // Initialize solution with zeros
    let mut x = Array1::zeros(n);

    // If b is zero, return zero solution
    let b_norm = vector_norm(&b.view(), 2)?;
    if b_norm < F::epsilon() {
        return Ok(x);
    }

    // Initial residual r = b - Ax
    let ax = a.apply(&x.view())?;
    let mut r = b.clone();
    r -= &ax;

    // Initial search direction p = r
    let mut p = r.clone();

    // Initial residual norm squared
    let mut rsold = r.dot(&r);

    // If initial guess is very close to solution
    if rsold.sqrt() < tol * b_norm {
        return Ok(x);
    }

    // Cache for AP product to avoid recomputation in adaptive _precision mode
    let mut ap;

    // Tracking variables for adaptive _precision
    let mut successive_slow_progress = 0;
    let mut previous_residual = rsold;

    for (iteration_, _) in (0..max_iter).enumerate() {
        // Keep track of iteration number
        let _iter = iteration_;

        // Compute A*p
        ap = a.apply(&p.view())?;

        // Compute step size alpha
        let pap = p.dot(&ap);

        // Safety check for numerical stability
        if pap.abs() < F::epsilon() {
            if _iter == 0usize {
                return Err(LinalgError::ComputationError(
                    "Zero curvature detected in first iteration".to_string(),
                ));
            }

            // We've reached a breakdown or convergence
            break;
        }

        let alpha = rsold / pap;

        // Update solution x = x + alpha*p
        x = &x + &(&p * alpha);

        // Update residual r = r - alpha*A*p
        r = &r - &(&ap * alpha);

        // Compute new residual norm squared
        let mut rsnew = r.dot(&r);

        // Check convergence
        if rsnew.sqrt() < tol * b_norm {
            break;
        }

        // Adaptive _precision strategy
        if adaptive_precision {
            // Check if we're making good progress
            let ratio = rsnew / previous_residual;

            // If progress is slow for multiple iterations, we might need to reset
            if ratio > F::from(0.9).unwrap() {
                successive_slow_progress += 1;
            } else {
                successive_slow_progress = 0;
            }

            // If we've had multiple iterations with slow progress, perform a residual refresh
            if successive_slow_progress >= 5 {
                // Re-compute residual directly from r = b - Ax to avoid accumulated error
                let ax = a.apply(&x.view())?;
                r = b.clone();
                r -= &ax;

                // Reset progress counter
                successive_slow_progress = 0;

                // Recompute rsnew
                rsnew = r.dot(&r);

                // Check convergence again after refresh
                if rsnew.sqrt() < tol * b_norm {
                    break;
                }
            }

            previous_residual = rsnew;
        }

        // Compute direction update beta
        let beta = rsnew / rsold;

        // Update search direction p = r + beta*p
        p = &r + &(&p * beta);

        // Update old residual norm
        rsold = rsnew;
    }

    // Return the solution and iteration count
    Ok(x)
}

/// Optimized GMRES method for quantized matrices
///
/// This implementation is specifically designed for quantized matrices,
/// with optimizations that take advantage of quantization properties.
///
/// # Arguments
///
/// * `a` - Quantized matrix-free operator representing the coefficient matrix
/// * `b` - Right-hand side vector
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
/// * `restart` - Number of iterations before restarting (optional)
/// * `adaptive_precision` - Whether to adaptively adjust precision during iterations
///
/// # Returns
///
/// * Solution vector x
#[allow(dead_code)]
pub fn quantized_gmres<F>(
    a: &QuantizedMatrixFreeOp<F>,
    b: &Array1<F>,
    max_iter: usize,
    tol: F,
    restart: Option<usize>,
    adaptive_precision: bool,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Debug + Send + Sync,
{
    // Check that dimensions are compatible
    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch: operator shape {}x{}, vector shape {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }

    let n = a.nrows();
    let restart_iter = restart.unwrap_or(n.min(30)); // Use smaller of n or 30 as default restart

    // Initialize solution with zeros
    let mut x = Array1::zeros(n);

    // If b is zero, return zero solution
    let b_norm = vector_norm(&b.view(), 2)?;
    if b_norm < F::epsilon() {
        return Ok(x);
    }

    // Get quantization parameters for potential _precision adjustments
    // No direct parameter adjustments in this implementation

    // Tracking variable for adaptive _precision
    let mut reorth_step = if adaptive_precision { 1 } else { restart_iter };

    // Outer iteration (restarts)
    for _outer in 0..max_iter {
        // Compute initial residual r = b - Ax
        let ax = a.apply(&x.view())?;
        let mut r = b.clone();
        r -= &ax;

        // Check if we've already converged
        let r_norm = vector_norm(&r.view(), 2)?;
        if r_norm < tol * b_norm {
            return Ok(x);
        }

        // Initialize the Krylov subspace with the normalized residual
        let beta = r_norm;
        let mut v = Array1::zeros(n);
        for i in 0..n {
            v[i] = r[i] / beta;
        }

        // Storage for the Hessenberg matrix
        let mut h = Array2::zeros((restart_iter + 1, restart_iter));

        // Storage for the orthogonal basis of the Krylov subspace
        let mut v_basis = Vec::with_capacity(restart_iter + 1);
        v_basis.push(v);

        // Storage for the rotations in the Hessenberg matrix
        let mut cs: Vec<F> = Vec::with_capacity(restart_iter);
        let mut sn: Vec<F> = Vec::with_capacity(restart_iter);

        // Storage for the right-hand side in the least squares problem
        let mut g = Array1::zeros(restart_iter + 1);
        g[0] = beta;

        // Inner iteration (Arnoldi process and least squares solve)
        let mut i = 0;
        while i < restart_iter {
            // Arnoldi process: Generate a new basis vector and orthogonalize
            let av = a.apply(&v_basis[i].view())?;
            let mut w = av;

            // Modified Gram-Schmidt orthogonalization with potential reorthogonalization
            let reorth_needed = adaptive_precision && (i % reorth_step == 0);

            for j in 0..=i {
                h[[j, i]] = w.dot(&v_basis[j]);
                w = &w - &(&v_basis[j] * h[[j, i]]);

                // Additional reorthogonalization for improved numerical stability
                // Critical for quantized matrices which can have higher numerical errors
                if reorth_needed {
                    let h_correction = w.dot(&v_basis[j]);
                    h[[j, i]] += h_correction;
                    w = &w - &(&v_basis[j] * h_correction);
                }
            }

            // Compute the norm of the new basis vector
            h[[i + 1, i]] = vector_norm(&w.view(), 2)?;

            // If the norm is very small, we've reached a breakdown
            if h[[i + 1, i]] < F::epsilon() {
                // We've converged or encountered a breakdown
                i += 1;
                break;
            }

            // Normalize the new basis vector
            let mut new_v = Array1::zeros(n);
            for j in 0..n {
                new_v[j] = w[j] / h[[i + 1, i]];
            }
            v_basis.push(new_v);

            // Apply previous Givens rotations to the new column of the Hessenberg matrix
            for j in 0..i {
                let temp = h[[j, i]];
                h[[j, i]] = cs[j] * temp + sn[j] * h[[j + 1, i]];
                h[[j + 1, i]] = -sn[j] * temp + cs[j] * h[[j + 1, i]];
            }

            // Compute the new Givens rotation
            let (c, s) = givens_rotation(h[[i, i]], h[[i + 1, i]]);
            cs.push(c);
            sn.push(s);

            // Apply the new Givens rotation to the last element of the new column
            h[[i, i]] = c * h[[i, i]] + s * h[[i + 1, i]];
            h[[i + 1, i]] = F::zero();

            // Apply the new Givens rotation to the right-hand side
            let temp = g[i];
            g[i] = c * temp + s * g[i + 1];
            g[i + 1] = -s * temp + c * g[i + 1];

            // Check convergence
            let residual = g[i + 1].abs();
            if residual < tol * b_norm {
                // We've converged, solve the upper triangular system
                i += 1;
                break;
            }

            // Adaptive strategy: If progress is slowing down, increase reorthogonalization frequency
            if adaptive_precision && i > 2 {
                let progress_ratio = residual / g[i].abs();

                // If progress is slowing, decrease reorthogonalization step
                if progress_ratio > F::from(0.8).unwrap() && reorth_step > 1 {
                    reorth_step = reorth_step.max(1) / 2;
                }
                // If progress is good, can relax reorthogonalization
                else if progress_ratio < F::from(0.5).unwrap() && reorth_step < restart_iter {
                    reorth_step = (reorth_step * 2).min(restart_iter);
                }
            }

            i += 1;
        }

        // Solve the upper triangular system H y = g
        let mut y = Array1::zeros(i);
        for j in (0..i).rev() {
            let mut sum = g[j];
            for k in (j + 1)..i {
                sum -= h[[j, k]] * y[k];
            }
            y[j] = sum / h[[j, j]];
        }

        // Update the solution x = x + V y
        for j in 0..i {
            x = &x + &(&v_basis[j] * y[j]);
        }

        // If we've converged or reached the maximum number of iterations, return
        let ax = a.apply(&x.view())?;
        let mut r = b.clone();
        r -= &ax;
        let r_norm = vector_norm(&r.view(), 2)?;
        if r_norm < tol * b_norm || i < restart_iter {
            return Ok(x);
        }

        // Adaptive _precision strategy - adjust quantization on restart if needed
        if adaptive_precision {
            // Note: In a real implementation, we might adjust the quantization parameters
            // here, but that would require rebuilding the quantized operator, which is beyond
            // the scope of this implementation
        }
    }

    // Return the best solution we have
    Ok(x)
}

/// Helper function for GMRES: compute the Givens rotation matrix parameters
///
/// This function computes the cosine and sine values for a Givens rotation
/// matrix that eliminates the subdiagonal element in the Hessenberg matrix.
#[allow(dead_code)]
fn givens_rotation<F>(a: F, b: F) -> (F, F)
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Send + Sync,
{
    if b == F::zero() {
        (F::one(), F::zero())
    } else if a.abs() < b.abs() {
        let t = a / b;
        let s = F::one() / (F::one() + t * t).sqrt();
        let c = s * t;
        (c, s)
    } else {
        let t = b / a;
        let c = F::one() / (F::one() + t * t).sqrt();
        let s = c * t;
        (c, s)
    }
}

/// Optimized Jacobi preconditioner for quantized matrices
///
/// Creates a diagonal preconditioner that can accelerate convergence
/// of iterative solvers with quantized matrices.
///
/// # Arguments
///
/// * `a` - Quantized matrix-free operator
///
/// # Returns
///
/// A quantized matrix-free operator representing the preconditioner
#[allow(dead_code)]
pub fn quantized_jacobi_preconditioner<F>(
    a: &QuantizedMatrixFreeOp<F>,
) -> LinalgResult<QuantizedMatrixFreeOp<F>>
where
    F: Float
        + NumAssign
        + Zero
        + Sum
        + One
        + ScalarOperand
        + Clone
        + Debug
        + Send
        + Sync
        + FromPrimitive
        + AsPrimitive<f32>
        + 'static,
    f32: AsPrimitive<F>,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(
            "Jacobi preconditioner requires a square operator".to_string(),
        ));
    }

    let n = a.nrows();
    let mut diag = Array1::zeros(n);

    // Estimate the diagonal entries by applying A to unit vectors
    for i in 0..n {
        let mut e_i = Array1::zeros(n);
        e_i[i] = F::one();
        let a_e_i = a.apply(&e_i.view())?;
        diag[i] = a_e_i[i];
    }

    // Check for zeros on the diagonal
    for i in 0..n {
        if diag[i].abs() < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Jacobi preconditioner encountered zero on diagonal".to_string(),
            ));
        }
        // Invert the diagonal for the preconditioner
        diag[i] = F::one() / diag[i];
    }

    // Create a quantized diagonal operator
    // First, we create a function that applies the diagonal operator
    let diag_clone = diag.clone();

    // Create a new quantized matrix-free operator
    QuantizedMatrixFreeOp::new(
        n,
        n,
        a.params().bits,
        a.params().method,
        move |x: &ArrayView1<F>| -> LinalgResult<Array1<F>> {
            if x.len() != n {
                return Err(LinalgError::ShapeError(format!(
                    "Expected vector of length {}, got {}",
                    n,
                    x.len()
                )));
            }

            let mut result = Array1::zeros(n);
            for i in 0..n {
                result[i] = diag_clone[i] * x[i];
            }

            Ok(result)
        },
    )
}

/// Optimized preconditioned conjugate gradient for quantized matrices
///
/// This function implements a preconditioned conjugate gradient solver
/// specifically optimized for quantized matrices.
///
/// # Arguments
///
/// * `a` - Quantized matrix-free operator representing the coefficient matrix
/// * `m` - Quantized matrix-free operator representing the preconditioner
/// * `b` - Right-hand side vector
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
/// * `adaptive_precision` - Whether to adaptively adjust precision during iterations
///
/// # Returns
///
/// * Solution vector x
#[allow(dead_code)]
pub fn quantized_preconditioned_conjugate_gradient<F>(
    a: &QuantizedMatrixFreeOp<F>,
    m: &QuantizedMatrixFreeOp<F>,
    b: &Array1<F>,
    max_iter: usize,
    tol: F,
    adaptive_precision: bool,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Zero + Sum + One + ScalarOperand + Debug + Send + Sync,
{
    // Check that A is square
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Expected square operator, got shape {}x{}",
            a.nrows(),
            a.ncols()
        )));
    }

    // Check that dimensions are compatible
    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Shape mismatch: operator shape {}x{}, vector shape {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }

    // Check that the preconditioner is compatible
    if m.nrows() != a.nrows() || m.ncols() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Preconditioner shape {}x{} doesn't match operator shape {}x{}",
            m.nrows(),
            m.ncols(),
            a.nrows(),
            a.ncols()
        )));
    }

    let n = a.nrows();

    // Initialize solution with zeros
    let mut x = Array1::zeros(n);

    // If b is zero, return zero solution
    let b_norm = vector_norm(&b.view(), 2)?;
    if b_norm < F::epsilon() {
        return Ok(x);
    }

    // Initial residual r = b - Ax
    let ax = a.apply(&x.view())?;
    let mut r = b.clone();
    r -= &ax;

    // Initial preconditioned residual z = M^-1 r
    let mut z = m.apply(&r.view())?;

    // Initial search direction p = z
    let mut p = z.clone();

    // Initial residual inner product
    let mut rz_old = r.dot(&z);

    // If initial guess is very close to solution
    if vector_norm(&r.view(), 2)? < tol * b_norm {
        return Ok(x);
    }

    // Tracking variables for adaptive _precision
    let mut successive_slow_progress = 0;
    let mut previous_residual = r.dot(&r);

    for _iter in 0..max_iter {
        // Compute A*p
        let ap = a.apply(&p.view())?;

        // Compute step size alpha
        let pap = p.dot(&ap);

        // Safety check for numerical stability
        if pap.abs() < F::epsilon() {
            if _iter == 0usize {
                return Err(LinalgError::ComputationError(
                    "Zero curvature detected in first iteration".to_string(),
                ));
            }

            // We've reached a breakdown or convergence
            break;
        }

        let alpha = rz_old / pap;

        // Update solution x = x + alpha*p
        x = &x + &(&p * alpha);

        // Update residual r = r - alpha*A*p
        r = &r - &(&ap * alpha);

        // Check convergence
        let r_norm = vector_norm(&r.view(), 2)?;
        if r_norm < tol * b_norm {
            break;
        }

        // Adaptive _precision strategy
        if adaptive_precision {
            let r_squared = r.dot(&r);

            // Check if we're making good progress
            let ratio = r_squared / previous_residual;

            // If progress is slow for multiple iterations, we might need to reset
            if ratio > F::from(0.9).unwrap() {
                successive_slow_progress += 1;
            } else {
                successive_slow_progress = 0;
            }

            // If we've had multiple iterations with slow progress, perform a residual refresh
            if successive_slow_progress >= 5 {
                // Re-compute residual directly from r = b - Ax to avoid accumulated error
                let ax = a.apply(&x.view())?;
                r = b.clone();
                r -= &ax;

                // Reset progress counter
                successive_slow_progress = 0;

                // Check convergence again after refresh
                if vector_norm(&r.view(), 2)? < tol * b_norm {
                    break;
                }
            }

            previous_residual = r_squared;
        }

        // Update preconditioned residual z = M^-1 r
        z = m.apply(&r.view())?;

        // Compute new residual inner product
        let rz_new = r.dot(&z);

        // Compute direction update beta
        let beta = rz_new / rz_old;

        // Update search direction p = z + beta*p
        p = &z + &(&p * beta);

        // Update old residual inner product
        rz_old = rz_new;
    }

    // Return the solution
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::quantized_matrixfree::QuantizedMatrixFreeOp;
    use crate::quantization::QuantizationMethod;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantized_conjugate_gradient_smallmatrix() {
        // Create a test matrix (symmetric positive definite)
        let matrix = array![[4.0f32, 1.0], [1.0, 3.0]];

        // Create a quantized matrix-free operator
        let op =
            QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
                .unwrap()
                .symmetric()
                .positive_definite();

        // Define the right-hand side
        let b = array![1.0f32, 2.0];

        // Solve using quantized conjugate gradient
        let x = quantized_conjugate_gradient(&op, &b, 10, 1e-6, false).unwrap();

        // Expected solution (computed analytically): [0.181818, 0.636364]
        let expected = array![0.181818f32, 0.636364];

        // Check that the results are close
        assert_eq!(x.len(), expected.len());
        for i in 0..x.len() {
            assert_relative_eq!(x[i], expected[i], epsilon = 0.15);
        }
    }

    #[test]
    fn test_quantized_gmres_smallmatrix() {
        // Create a test matrix (non-symmetric)
        let matrix = array![[3.0f32, 1.0], [1.0, 2.0]];

        // Create a quantized matrix-free operator
        let op =
            QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
                .unwrap();

        // Define the right-hand side
        let b = array![4.0f32, 3.0];

        // Solve using quantized GMRES
        let x = quantized_gmres(&op, &b, 10, 1e-6, None, false).unwrap();

        // Expected solution (computed analytically): [1.0, 1.0]
        let expected = array![1.0f32, 1.0];

        // Check that the results are close
        assert_eq!(x.len(), expected.len());
        for i in 0..x.len() {
            assert_relative_eq!(x[i], expected[i], epsilon = 0.15);
        }
    }

    #[test]
    fn test_quantized_preconditioned_conjugate_gradient() {
        // Create a test matrix (symmetric positive definite)
        let matrix = array![[4.0f32, 1.0], [1.0, 3.0]];

        // Create a quantized matrix-free operator
        let op =
            QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
                .unwrap()
                .symmetric()
                .positive_definite();

        // Create a preconditioner (here, we'll use the Jacobi preconditioner)
        let precond = quantized_jacobi_preconditioner(&op).unwrap();

        // Define the right-hand side
        let b = array![1.0f32, 2.0];

        // Solve using preconditioned conjugate gradient
        let x = quantized_preconditioned_conjugate_gradient(&op, &precond, &b, 10, 1e-6, false)
            .unwrap();

        // Expected solution (computed analytically): [0.181818, 0.636364]
        let expected = array![0.181818f32, 0.636364];

        // Check that the results are close
        assert_eq!(x.len(), expected.len());
        for i in 0..x.len() {
            assert_relative_eq!(x[i], expected[i], epsilon = 0.15);
        }
    }

    #[test]
    fn test_quantized_jacobi_preconditioner() {
        // Create a test matrix with known diagonal
        let matrix = array![[4.0f32, 1.0], [1.0, 3.0]];

        // Create a quantized matrix-free operator
        let op =
            QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
                .unwrap();

        // Create a Jacobi preconditioner
        let precond = quantized_jacobi_preconditioner(&op).unwrap();

        // Apply to a vector
        let x = array![1.0f32, 2.0];
        let y = precond.apply(&x.view()).unwrap();

        // The preconditioner should be the inverse of the diagonal of the matrix
        // Diagonal of the matrix: [4.0, 3.0]
        // Inverse: [1/4.0, 1/3.0]
        // Applied to [1.0, 2.0]: [1.0/4.0, 2.0/3.0] = [0.25, 0.6666...]
        let expected = array![0.25f32, 2.0 / 3.0];

        assert_eq!(y.len(), expected.len());
        for i in 0..y.len() {
            assert_relative_eq!(y[i], expected[i], epsilon = 0.01);
        }
    }

    #[test]
    fn test_adaptive_precision_conjugate_gradient() {
        // Create a test matrix (symmetric positive definite)
        let matrix = array![[4.0f32, 1.0], [1.0, 3.0]];

        // Create a quantized matrix-free operator
        let op =
            QuantizedMatrixFreeOp::frommatrix(&matrix.view(), 8, QuantizationMethod::Symmetric)
                .unwrap()
                .symmetric()
                .positive_definite();

        // Define the right-hand side
        let b = array![1.0f32, 2.0];

        // Solve using adaptive precision conjugate gradient
        let x_adaptive = quantized_conjugate_gradient(&op, &b, 10, 1e-6, true).unwrap();

        // Solve using standard conjugate gradient
        let x_standard = quantized_conjugate_gradient(&op, &b, 10, 1e-6, false).unwrap();

        // Both should give accurate solutions
        let expected = array![0.181818f32, 0.636364];

        // Check adaptive precision result
        assert_eq!(x_adaptive.len(), expected.len());
        for i in 0..x_adaptive.len() {
            assert_relative_eq!(x_adaptive[i], expected[i], epsilon = 0.15);
        }

        // Check standard result
        assert_eq!(x_standard.len(), expected.len());
        for i in 0..x_standard.len() {
            assert_relative_eq!(x_standard[i], expected[i], epsilon = 0.15);
        }
    }
}
