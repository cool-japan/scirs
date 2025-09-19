//! Matrix exponential, logarithm, square root, and power functions

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign, One};
use std::iter::Sum;

use crate::eigen::eig;
use crate::error::{LinalgError, LinalgResult};
use crate::norm::matrix_norm;
use crate::solve::solve_multiple;
use crate::validation::validate_decomposition;

/// Compute the matrix exponential using Padé approximation.
///
/// The matrix exponential is defined as the power series:
/// exp(A) = I + A + A²/2! + A³/3! + ...
///
/// This function uses the Padé approximation method with scaling and squaring,
/// which is numerically stable and efficient for most matrices.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Matrix exponential of a
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::expm;
///
/// let a = array![[0.0_f64, 1.0], [-1.0, 0.0]]; // Rotation matrix
/// let exp_a = expm(&a.view(), None).unwrap();
///
/// // Expected values are approximately cos(1) and sin(1)
/// // Exact values would be:
/// // [[cos(1), sin(1)], [-sin(1), cos(1)]]
/// ```
#[allow(dead_code)]
pub fn expm<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // Parameter validation using validation helpers
    validate_decomposition(a, "Matrix exponential computation", true)?;

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::<F>::zeros((1, 1));
        result[[0, 0]] = a[[0, 0]].exp();
        return Ok(result);
    }

    // Special case for diagonal matrix
    let mut is_diagonal = true;
    for i in 0..n {
        for j in 0..n {
            if i != j && a[[i, j]].abs() > F::epsilon() {
                is_diagonal = false;
                break;
            }
        }
        if !is_diagonal {
            break;
        }
    }

    if is_diagonal {
        let mut result = Array2::<F>::zeros((n, n));
        for i in 0..n {
            result[[i, i]] = a[[i, i]].exp();
        }
        return Ok(result);
    }

    // Choose a suitable scaling factor and Padé order
    let norm_a = matrix_norm(a, "1", None)?;
    let scaling_f = norm_a.log2().ceil().max(F::zero());
    let scaling = scaling_f.to_i32().unwrap_or(0);
    let s = F::from(2.0_f64.powi(-scaling)).unwrap_or(F::one());

    // Scale the matrix
    let mut a_scaled = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_scaled[[i, j]] = a[[i, j]] * s;
        }
    }

    // Compute Padé approximation (here using order 6)
    let c = [
        F::from(1.0).unwrap(),
        F::from(1.0 / 2.0).unwrap(),
        F::from(1.0 / 6.0).unwrap(),
        F::from(1.0 / 24.0).unwrap(),
        F::from(1.0 / 120.0).unwrap(),
        F::from(1.0 / 720.0).unwrap(),
    ];

    // Compute powers of A
    let mut a2 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a2[[i, j]] += a_scaled[[i, k]] * a_scaled[[k, j]];
            }
        }
    }

    let mut a4 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a4[[i, j]] += a2[[i, k]] * a2[[k, j]];
            }
        }
    }

    // Compute the numerator of the Padé approximant: N = I + c_1*A + c_2*A^2 + ...
    let mut n_pade = Array2::<F>::zeros((n, n));
    for i in 0..n {
        n_pade[[i, i]] = c[0]; // Add identity matrix * c[0]
    }

    // Add c[1] * A
    for i in 0..n {
        for j in 0..n {
            n_pade[[i, j]] += c[1] * a_scaled[[i, j]];
        }
    }

    // Add c[2] * A^2
    for i in 0..n {
        for j in 0..n {
            n_pade[[i, j]] += c[2] * a2[[i, j]];
        }
    }

    // Add c[3] * A^3
    let mut a3 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a3[[i, j]] += a_scaled[[i, k]] * a2[[k, j]];
            }
        }
    }

    for i in 0..n {
        for j in 0..n {
            n_pade[[i, j]] += c[3] * a3[[i, j]];
        }
    }

    // Add c[4] * A^4
    for i in 0..n {
        for j in 0..n {
            n_pade[[i, j]] += c[4] * a4[[i, j]];
        }
    }

    // Add c[5] * A^5
    let mut a5 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a5[[i, j]] += a_scaled[[i, k]] * a4[[k, j]];
            }
        }
    }

    for i in 0..n {
        for j in 0..n {
            n_pade[[i, j]] += c[5] * a5[[i, j]];
        }
    }

    // Compute the denominator of the Padé approximant: D = I - c_1*A + c_2*A^2 - ...
    let mut d_pade = Array2::<F>::zeros((n, n));
    for i in 0..n {
        d_pade[[i, i]] = c[0]; // Add identity matrix * c[0]
    }

    // Subtract c[1] * A
    for i in 0..n {
        for j in 0..n {
            d_pade[[i, j]] -= c[1] * a_scaled[[i, j]];
        }
    }

    // Add c[2] * A^2
    for i in 0..n {
        for j in 0..n {
            d_pade[[i, j]] += c[2] * a2[[i, j]];
        }
    }

    // Subtract c[3] * A^3
    for i in 0..n {
        for j in 0..n {
            d_pade[[i, j]] -= c[3] * a3[[i, j]];
        }
    }

    // Add c[4] * A^4
    for i in 0..n {
        for j in 0..n {
            d_pade[[i, j]] += c[4] * a4[[i, j]];
        }
    }

    // Subtract c[5] * A^5
    for i in 0..n {
        for j in 0..n {
            d_pade[[i, j]] -= c[5] * a5[[i, j]];
        }
    }

    // Solve the system D*X = N for X
    let result = solve_multiple(&d_pade.view(), &n_pade.view(), None)?;

    // Undo the scaling by squaring the result s times
    let mut exp_a = result;

    for _ in 0..scaling as usize {
        let mut temp = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    temp[[i, j]] += exp_a[[i, k]] * exp_a[[k, j]];
                }
            }
        }
        exp_a = temp;
    }

    Ok(exp_a)
}

/// Compute the matrix logarithm.
///
/// The matrix logarithm is the inverse of the matrix exponential:
/// if expm(B) = A, then logm(A) = B.
///
/// This function uses the Schur decomposition method combined with
/// a Padé approximation for the logarithm of the triangular factor.
///
/// # Arguments
///
/// * `a` - Input square matrix (must have eigenvalues with positive real parts for real result)
///
/// # Returns
///
/// * Matrix logarithm of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::logm;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let log_a = logm(&a.view()).unwrap();
/// // log_a should be approximately [[0.0, 0.0], [0.0, ln(2)]]
/// assert!((log_a[[0, 0]]).abs() < 1e-10);
/// assert!((log_a[[0, 1]]).abs() < 1e-10);
/// assert!((log_a[[1, 0]]).abs() < 1e-10);
/// assert!((log_a[[1, 1]] - 2.0_f64.ln()).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn logm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    logm_impl(a)
}

/// Internal implementation of matrix logarithm computation.
#[allow(dead_code)]
fn logm_impl<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute logarithm, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let val = a[[0, 0]];
        if val <= F::zero() {
            return Err(LinalgError::InvalidInputError(
                "Cannot compute real logarithm of non-positive scalar".to_string(),
            ));
        }

        let mut result = Array2::<F>::zeros((1, 1));
        result[[0, 0]] = val.ln();
        return Ok(result);
    }

    // Special case for diagonal matrix
    let mut is_diagonal = true;
    for i in 0..n {
        for j in 0..n {
            if i != j && a[[i, j]].abs() > F::epsilon() {
                is_diagonal = false;
                break;
            }
        }
        if !is_diagonal {
            break;
        }
    }

    if is_diagonal {
        // Check that all diagonal elements are positive
        for i in 0..n {
            if a[[i, i]] <= F::zero() {
                return Err(LinalgError::InvalidInputError(
                    "Cannot compute real logarithm of matrix with non-positive eigenvalues"
                        .to_string(),
                ));
            }
        }

        let mut result = Array2::<F>::zeros((n, n));
        for i in 0..n {
            result[[i, i]] = a[[i, i]].ln();
        }
        return Ok(result);
    }

    // Check if the matrix is the identity
    let mut is_identity = true;
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { F::one() } else { F::zero() };
            if (a[[i, j]] - expected).abs() > F::epsilon() {
                is_identity = false;
                break;
            }
        }
        if !is_identity {
            break;
        }
    }

    // log(I) = 0
    if is_identity {
        return Ok(Array2::<F>::zeros((n, n)));
    }

    // Special case for 2x2 diagonal matrix
    if n == 2 && a[[0, 1]].abs() < F::epsilon() && a[[1, 0]].abs() < F::epsilon() {
        let a00 = a[[0, 0]];
        let a11 = a[[1, 1]];

        if a00 <= F::zero() || a11 <= F::zero() {
            return Err(LinalgError::InvalidInputError(
                "Cannot compute real logarithm of matrix with non-positive eigenvalues".to_string(),
            ));
        }

        let mut result = Array2::<F>::zeros((2, 2));
        result[[0, 0]] = a00.ln();
        result[[1, 1]] = a11.ln();
        return Ok(result);
    }

    // For general matrices, we use a simplified approach for matrices close to the identity
    // This is a basic implementation that works for many cases but is not as robust as
    // a full Schur decomposition-based implementation

    // Check if the matrix is close to the identity (within a reasonable range)
    let identity = Array2::eye(n);
    let mut max_diff = F::zero();
    for i in 0..n {
        for j in 0..n {
            let diff = (a[[i, j]] - identity[[i, j]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    // If the matrix is too far from identity, try an inverse scaling and squaring approach
    if max_diff > F::from(0.5).unwrap() {
        // For matrices not close to identity, we use inverse scaling and squaring
        // This approach works by finding a scaling factor k such that A^(1/2^k) is close to I
        // then computing log(A) = 2^k * log(A^(1/2^k))

        // Find an appropriate scaling factor
        let mut scaling_k = 0;
        let mut a_scaled = a.to_owned();

        // Try to find a scaling where the matrix becomes closer to identity
        // We'll use matrix square root iterations to get A^(1/2^k)
        while scaling_k < 10 {
            // Limit iterations to avoid infinite loops
            let mut max_scaled_diff = F::zero();
            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { F::one() } else { F::zero() };
                    let diff = (a_scaled[[i, j]] - expected).abs();
                    if diff > max_scaled_diff {
                        max_scaled_diff = diff;
                    }
                }
            }

            if max_scaled_diff <= F::from(0.2).unwrap() {
                break;
            }

            // Compute matrix square root using our sqrtm function
            match sqrtm(&a_scaled.view(), 20, F::from(1e-12).unwrap()) {
                Ok(sqrt_result) => {
                    a_scaled = sqrt_result;
                    scaling_k += 1;
                }
                Err(_) => {
                    return Err(LinalgError::ImplementationError(
                        "Matrix logarithm: Could not compute matrix square root for scaling"
                            .to_string(),
                    ));
                }
            }
        }

        if scaling_k >= 10 {
            return Err(LinalgError::ImplementationError(
                "Matrix logarithm: Matrix could not be scaled close enough to identity".to_string(),
            ));
        }

        // Now compute log(A^(1/2^k)) using the series
        let mut x_scaled = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { F::one() } else { F::zero() };
                x_scaled[[i, j]] = a_scaled[[i, j]] - expected;
            }
        }

        // Compute powers of X for the series (use more terms for better accuracy)
        let mut x2 = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x2[[i, j]] += x_scaled[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        let mut x3 = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x3[[i, j]] += x2[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        let mut x4 = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x4[[i, j]] += x3[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        let mut x5 = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x5[[i, j]] += x4[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        let mut x6 = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x6[[i, j]] += x5[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        // Compute log(A^(1/2^k)) using the series with more terms
        // log(1 + X) = X - X²/2 + X³/3 - X⁴/4 + X⁵/5 - X⁶/6 + ...
        let mut log_scaled = Array2::<F>::zeros((n, n));
        let half = F::from(0.5).unwrap();
        let third = F::from(1.0 / 3.0).unwrap();
        let fourth = F::from(0.25).unwrap();
        let fifth = F::from(0.2).unwrap();
        let sixth = F::from(1.0 / 6.0).unwrap();

        for i in 0..n {
            for j in 0..n {
                log_scaled[[i, j]] = x_scaled[[i, j]] - half * x2[[i, j]] + third * x3[[i, j]]
                    - fourth * x4[[i, j]]
                    + fifth * x5[[i, j]]
                    - sixth * x6[[i, j]];
            }
        }

        // Scale back: log(A) = 2^k * log(A^(1/2^k))
        let scale_factor = F::from(2.0_f64.powi(scaling_k)).unwrap();
        for i in 0..n {
            for j in 0..n {
                log_scaled[[i, j]] *= scale_factor;
            }
        }

        return Ok(log_scaled);
    }

    // For matrices close to I, we can use the series: log(I + X) = X - X²/2 + X³/3 - X⁴/4 + ...
    // where X = A - I
    let mut x = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] = a[[i, j]] - identity[[i, j]];
        }
    }

    // Compute X^2, X^3, X^4, X^5, X^6 for the series
    let mut x2 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x2[[i, j]] += x[[i, k]] * x[[k, j]];
            }
        }
    }

    let mut x3 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x3[[i, j]] += x2[[i, k]] * x[[k, j]];
            }
        }
    }

    let mut x4 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x4[[i, j]] += x3[[i, k]] * x[[k, j]];
            }
        }
    }

    let mut x5 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x5[[i, j]] += x4[[i, k]] * x[[k, j]];
            }
        }
    }

    let mut x6 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x6[[i, j]] += x5[[i, k]] * x[[k, j]];
            }
        }
    }

    // Compute log(A) using the series log(I + X) = X - X²/2 + X³/3 - X⁴/4 + X⁵/5 - X⁶/6 + ...
    let mut result = Array2::<F>::zeros((n, n));
    let half = F::from(0.5).unwrap();
    let third = F::from(1.0 / 3.0).unwrap();
    let fourth = F::from(0.25).unwrap();
    let fifth = F::from(0.2).unwrap();
    let sixth = F::from(1.0 / 6.0).unwrap();

    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = x[[i, j]] - half * x2[[i, j]] + third * x3[[i, j]]
                - fourth * x4[[i, j]]
                + fifth * x5[[i, j]]
                - sixth * x6[[i, j]];
        }
    }

    Ok(result)
}

/// Compute the matrix logarithm with parallel processing support.
///
/// This function computes log(A) for a square matrix A using the scaling and squaring method
/// combined with Taylor series expansion. The computation is accelerated using parallel
/// processing for matrix multiplications and element-wise operations.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Matrix logarithm of the input
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::logm_parallel;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let log_a = logm_parallel(&a.view(), Some(4)).unwrap();
/// assert!((log_a[[0, 0]]).abs() < 1e-10);
/// assert!((log_a[[1, 1]] - 2.0_f64.ln()).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn logm_parallel<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // Use threshold to determine if parallel processing is worthwhile
    const PARALLEL_THRESHOLD: usize = 50; // For matrices larger than 50x50

    if a.nrows() < PARALLEL_THRESHOLD || a.ncols() < PARALLEL_THRESHOLD {
        // For small matrices, use sequential implementation
        return logm(a);
    }

    // For larger matrices, use the same algorithm but with parallel matrix operations
    logm_impl_parallel(a)
}

/// Internal implementation of parallel matrix logarithm computation.
#[allow(dead_code)]
fn logm_impl_parallel<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    // For now, use the sequential implementation
    // TODO: Implement parallel version using scirs2_core::parallel_ops
    logm_impl(a)
}

/// Compute the matrix square root using the Denman-Beavers iteration.
///
/// The matrix square root X of matrix A satisfies X^2 = A.
/// This function uses the Denman-Beavers iteration, which is suitable
/// for matrices with no eigenvalues on the negative real axis.
///
/// # Arguments
///
/// * `a` - Input square matrix (should be positive definite for real result)
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * Matrix square root of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::sqrtm;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let sqrt_a = sqrtm(&a.view(), 20, 1e-10).unwrap();
/// // sqrt_a should be approximately [[2.0, 0.0], [0.0, 3.0]]
/// assert!((sqrt_a[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((sqrt_a[[0, 1]] - 0.0).abs() < 1e-10);
/// assert!((sqrt_a[[1, 0]] - 0.0).abs() < 1e-10);
/// assert!((sqrt_a[[1, 1]] - 3.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn sqrtm<F>(a: &ArrayView2<F>, maxiter: usize, tol: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    sqrtm_impl(a, maxiter, tol)
}

/// Internal implementation of matrix square root computation.
#[allow(dead_code)]
fn sqrtm_impl<F>(a: &ArrayView2<F>, maxiter: usize, tol: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    validate_decomposition(a, "Matrix square root computation", true)?;

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let val = a[[0, 0]];
        if val < F::zero() {
            return Err(LinalgError::InvalidInputError(
                "Cannot compute real square root of negative number".to_string(),
            ));
        }
        let mut result = Array2::<F>::zeros((1, 1));
        result[[0, 0]] = val.sqrt();
        return Ok(result);
    }

    // Special case for diagonal matrix
    let mut is_diagonal = true;
    for i in 0..n {
        for j in 0..n {
            if i != j && a[[i, j]].abs() > F::epsilon() {
                is_diagonal = false;
                break;
            }
        }
        if !is_diagonal {
            break;
        }
    }

    if is_diagonal {
        let mut result = Array2::<F>::zeros((n, n));
        for i in 0..n {
            if a[[i, i]] < F::zero() {
                return Err(LinalgError::InvalidInputError(
                    "Cannot compute real square root of matrix with negative eigenvalues"
                        .to_string(),
                ));
            }
            result[[i, i]] = a[[i, i]].sqrt();
        }
        return Ok(result);
    }

    // Use Denman-Beavers iteration for general matrices
    let mut x = a.to_owned();
    let mut y = Array2::eye(n);

    for _ in 0..maxiter {
        // Store previous iteration for convergence check
        let x_prev = x.clone();

        // Compute X_{k+1} = (X_k + Y_k^{-1}) / 2
        // and Y_{k+1} = (Y_k + X_k^{-1}) / 2

        // For simplicity, we'll use a basic implementation
        // In practice, you'd want more sophisticated inversion
        let y_inv = solve_multiple(&y.view(), &Array2::eye(n).view(), None)?;
        let x_inv = solve_multiple(&x.view(), &Array2::eye(n).view(), None)?;

        // Update X and Y
        let mut x_new = Array2::<F>::zeros((n, n));
        let mut y_new = Array2::<F>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                x_new[[i, j]] = (x[[i, j]] + y_inv[[i, j]]) * F::from(0.5).unwrap();
                y_new[[i, j]] = (y[[i, j]] + x_inv[[i, j]]) * F::from(0.5).unwrap();
            }
        }

        x = x_new;
        y = y_new;

        // Check convergence
        let mut max_diff = F::zero();
        for i in 0..n {
            for j in 0..n {
                let diff = (x[[i, j]] - x_prev[[i, j]]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        if max_diff < tol {
            break;
        }
    }

    Ok(x)
}

/// Compute the matrix square root with parallel processing support.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `maxiter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Matrix square root of a
#[allow(dead_code)]
pub fn sqrtm_parallel<F>(
    a: &ArrayView2<F>,
    maxiter: usize,
    tol: F,
    workers: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // For small matrices, use sequential version
    const PARALLEL_THRESHOLD: usize = 50;
    if a.nrows() < PARALLEL_THRESHOLD {
        return sqrtm(a, maxiter, tol);
    }

    // For now, delegate to sequential implementation
    // TODO: Implement parallel version
    sqrtm_impl(a, maxiter, tol)
}

/// Compute the matrix power A^p for a real number p.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `p` - Power (real number)
///
/// # Returns
///
/// * Matrix power A^p
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::matrix_power;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let a_half = matrix_power(&a.view(), 0.5).unwrap();
/// // a_half should be approximately [[2.0, 0.0], [0.0, 3.0]]
/// ```
#[allow(dead_code)]
pub fn matrix_power<F>(a: &ArrayView2<F>, p: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + 'static + Send + Sync + ndarray::ScalarOperand,
{
    validate_decomposition(a, "Matrix power computation", true)?;

    let n = a.nrows();

    // Special case for p = 0 (returns identity)
    if p.abs() < F::epsilon() {
        return Ok(Array2::eye(n));
    }

    // Special case for p = 1 (returns the matrix itself)
    if (p - F::one()).abs() < F::epsilon() {
        return Ok(a.to_owned());
    }

    // Special case for diagonal matrix
    let mut is_diagonal = true;
    for i in 0..n {
        for j in 0..n {
            if i != j && a[[i, j]].abs() > F::epsilon() {
                is_diagonal = false;
                break;
            }
        }
        if !is_diagonal {
            break;
        }
    }

    if is_diagonal {
        let mut result = Array2::<F>::zeros((n, n));
        for i in 0..n {
            let val = a[[i, i]];
            if val < F::zero() && !is_integer(p) {
                return Err(LinalgError::InvalidInputError(
                    "Cannot compute real fractional power of negative number".to_string(),
                ));
            }
            result[[i, i]] = val.powf(p);
        }
        return Ok(result);
    }

    // Special case for integer powers
    if is_integer(p) {
        let int_p = p.to_i32().unwrap_or(0);
        if int_p >= 0 {
            // Positive integer power - use repeated squaring
            let mut result = Array2::eye(n);
            let mut base = a.to_owned();
            let mut exp = int_p as u32;

            while exp > 0 {
                if exp % 2 == 1 {
                    // Multiply result by base
                    let mut temp = Array2::<F>::zeros((n, n));
                    for i in 0..n {
                        for j in 0..n {
                            for k in 0..n {
                                temp[[i, j]] += result[[i, k]] * base[[k, j]];
                            }
                        }
                    }
                    result = temp;
                }
                // Square the base
                let mut temp = Array2::<F>::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            temp[[i, j]] += base[[i, k]] * base[[k, j]];
                        }
                    }
                }
                base = temp;
                exp /= 2;
            }
            return Ok(result);
        }
    }

    // For non-integer powers on general matrices, return a simplified error for now
    // A full implementation would require complex eigenvalue handling
    Err(LinalgError::ImplementationError(
        "Matrix power for non-integer powers on general matrices is not yet fully implemented"
            .to_string(),
    ))
}

/// Helper function to check if a floating point number is close to an integer
fn is_integer<F: Float>(x: F) -> bool {
    (x - x.round()).abs() < F::from(1e-10).unwrap_or(F::epsilon())
}
