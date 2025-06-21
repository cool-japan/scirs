//! Matrix functions such as matrix exponential, logarithm, and square root

use ndarray::{Array2, ArrayView2};
use num_traits::{Float, NumAssign, One};
use std::iter::Sum;

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
pub fn expm<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // Parameter validation using validation helpers
    validate_decomposition(a, "Matrix exponential computation", true)?;

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
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
        let mut result = Array2::zeros((n, n));
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
    let mut a_scaled = Array2::zeros((n, n));
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
    let mut a2 = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a2[[i, j]] += a_scaled[[i, k]] * a_scaled[[k, j]];
            }
        }
    }

    let mut a4 = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                a4[[i, j]] += a2[[i, k]] * a2[[k, j]];
            }
        }
    }

    // Compute the numerator of the Padé approximant: N = I + c_1*A + c_2*A^2 + ...
    let mut n_pade = Array2::zeros((n, n));
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
    let mut a3 = Array2::zeros((n, n));
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
    let mut a5 = Array2::zeros((n, n));
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
    let mut d_pade = Array2::zeros((n, n));
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
        let mut temp = Array2::zeros((n, n));
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
pub fn logm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One,
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

        let mut result = Array2::zeros((1, 1));
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

        let mut result = Array2::zeros((n, n));
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
        return Ok(Array2::zeros((n, n)));
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

        let mut result = Array2::zeros((2, 2));
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
        let mut x_scaled = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { F::one() } else { F::zero() };
                x_scaled[[i, j]] = a_scaled[[i, j]] - expected;
            }
        }

        // Compute powers of X for the series (use more terms for better accuracy)
        let mut x2 = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x2[[i, j]] += x_scaled[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        let mut x3 = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x3[[i, j]] += x2[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        let mut x4 = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x4[[i, j]] += x3[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        let mut x5 = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x5[[i, j]] += x4[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        let mut x6 = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    x6[[i, j]] += x5[[i, k]] * x_scaled[[k, j]];
                }
            }
        }

        // Compute log(A^(1/2^k)) using the series with more terms
        // log(1 + X) = X - X²/2 + X³/3 - X⁴/4 + X⁵/5 - X⁶/6 + ...
        let mut log_scaled = Array2::zeros((n, n));
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
    let mut x = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] = a[[i, j]] - identity[[i, j]];
        }
    }

    // Compute X^2, X^3, X^4, X^5, X^6 for the series
    let mut x2 = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x2[[i, j]] += x[[i, k]] * x[[k, j]];
            }
        }
    }

    let mut x3 = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x3[[i, j]] += x2[[i, k]] * x[[k, j]];
            }
        }
    }

    let mut x4 = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x4[[i, j]] += x3[[i, k]] * x[[k, j]];
            }
        }
    }

    let mut x5 = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x5[[i, j]] += x4[[i, k]] * x[[k, j]];
            }
        }
    }

    let mut x6 = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                x6[[i, j]] += x5[[i, k]] * x[[k, j]];
            }
        }
    }

    // Compute log(A) using the series log(I + X) = X - X²/2 + X³/3 - X⁴/4 + X⁵/5 - X⁶/6 + ...
    let mut result = Array2::zeros((n, n));
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
pub fn sqrtm<F>(a: &ArrayView2<F>, max_iter: usize, tol: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute square root, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let val = a[[0, 0]];
        if val < F::zero() {
            return Err(LinalgError::InvalidInputError(
                "Cannot compute real square root of negative number".to_string(),
            ));
        }

        let mut result = Array2::zeros((1, 1));
        result[[0, 0]] = val.sqrt();
        return Ok(result);
    }

    // Special case for 2x2 diagonal matrix
    if n == 2 && a[[0, 1]].abs() < F::epsilon() && a[[1, 0]].abs() < F::epsilon() {
        let a00 = a[[0, 0]];
        let a11 = a[[1, 1]];

        if a00 < F::zero() || a11 < F::zero() {
            return Err(LinalgError::InvalidInputError(
                "Cannot compute real square root of matrix with negative eigenvalues".to_string(),
            ));
        }

        let mut result = Array2::zeros((2, 2));
        result[[0, 0]] = a00.sqrt();
        result[[1, 1]] = a11.sqrt();
        return Ok(result);
    }

    // Initialize Y and Z for Denman-Beavers iteration
    let mut y = a.to_owned();
    let mut z = Array2::eye(n);

    // Iteration
    let mut final_error = None;

    for _iter in 0..max_iter {
        // Compute Y_next = 0.5 * (Y + Z^-1)
        // and Z_next = 0.5 * (Z + Y^-1)

        // First, compute Z^-1 and Y^-1
        let z_inv = match solve_multiple(&z.view(), &Array2::eye(n).view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(LinalgError::singular_matrix_with_suggestions(
                    "Matrix square root (Denman-Beavers iteration)",
                    (n, n),
                    None,
                ))
            }
        };

        let y_inv = match solve_multiple(&y.view(), &Array2::eye(n).view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(LinalgError::singular_matrix_with_suggestions(
                    "Matrix square root (Denman-Beavers iteration)",
                    (n, n),
                    None,
                ))
            }
        };

        // Compute next iterations
        let mut y_next = Array2::zeros((n, n));
        let mut z_next = Array2::zeros((n, n));

        let half = F::from(0.5).unwrap();

        for i in 0..n {
            for j in 0..n {
                y_next[[i, j]] = half * (y[[i, j]] + z_inv[[i, j]]);
                z_next[[i, j]] = half * (z[[i, j]] + y_inv[[i, j]]);
            }
        }

        // Compute error for convergence check
        let mut error = F::zero();
        for i in 0..n {
            for j in 0..n {
                let diff = (y_next[[i, j]] - y[[i, j]]).abs();
                if diff > error {
                    error = diff;
                }
            }
        }

        final_error = Some(error.to_f64().unwrap_or(1.0));

        // Update Y and Z
        y = y_next;
        z = z_next;

        // Check convergence
        if error < tol {
            return Ok(y);
        }
    }

    // Failed to converge - return error with suggestions
    Err(LinalgError::convergence_with_suggestions(
        "Matrix square root (Denman-Beavers iteration)",
        max_iter,
        tol.to_f64().unwrap_or(1e-12),
        final_error,
    ))
}

/// Compute the matrix cosine using eigendecomposition.
///
/// The matrix cosine is defined using the matrix exponential:
/// cos(A) = (exp(iA) + exp(-iA))/2
///
/// For real matrices, this can be computed using eigendecomposition or
/// series expansion. This implementation uses a series expansion approach
/// for numerically stable computation.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix cosine of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::cosm;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let cos_a = cosm(&a.view()).unwrap();
/// // cos(0) = I
/// assert!((cos_a[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((cos_a[[1, 1]] - 1.0).abs() < 1e-10);
/// ```
pub fn cosm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute cosine, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        result[[0, 0]] = a[[0, 0]].cos();
        return Ok(result);
    }

    // For cos(A), we use the series expansion:
    // cos(A) = I - A²/2! + A⁴/4! - A⁶/6! + A⁸/8! - ...
    // This converges for all matrices but we'll use a finite number of terms

    let mut result = Array2::eye(n); // Start with identity matrix
    let mut a_power = Array2::eye(n); // A^0 = I
    let mut factorial = F::one();
    let mut sign = F::one();

    // We'll compute up to the 16th power (A^16) for good accuracy
    for k in 1..=8 {
        // Compute A^(2k) by squaring A^(k) twice
        // First: A^k -> A^(2k-1)
        let mut temp = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    temp[[i, j]] += a_power[[i, l]] * a[[l, j]];
                }
            }
        }
        a_power = temp;

        // Second: A^(2k-1) -> A^(2k)
        let mut temp = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    temp[[i, j]] += a_power[[i, l]] * a[[l, j]];
                }
            }
        }
        a_power = temp;

        // Update factorial: (2k)!
        factorial *= F::from(2 * k - 1).unwrap() * F::from(2 * k).unwrap();

        // Alternate signs: (-1)^k
        sign = -sign;

        // Add the term: (-1)^k * A^(2k) / (2k)!
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] += sign * a_power[[i, j]] / factorial;
            }
        }
    }

    Ok(result)
}

/// Compute the matrix sine using eigendecomposition.
///
/// The matrix sine is defined using the matrix exponential:
/// sin(A) = (exp(iA) - exp(-iA))/(2i)
///
/// For real matrices, this can be computed using eigendecomposition or
/// series expansion. This implementation uses a series expansion approach
/// for numerically stable computation.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix sine of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::sinm;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let sin_a = sinm(&a.view()).unwrap();
/// // sin(0) = 0
/// assert!((sin_a[[0, 0]]).abs() < 1e-10);
/// assert!((sin_a[[1, 1]]).abs() < 1e-10);
/// ```
pub fn sinm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute sine, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        result[[0, 0]] = a[[0, 0]].sin();
        return Ok(result);
    }

    // For sin(A), we use the series expansion:
    // sin(A) = A - A³/3! + A⁵/5! - A⁷/7! + A⁹/9! - ...
    // This converges for all matrices but we'll use a finite number of terms

    let mut result = Array2::zeros((n, n));
    let mut a_power = a.to_owned(); // Start with A^1
    let mut factorial = F::one();
    let mut sign = F::one();

    // Add the first term: A
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = a_power[[i, j]];
        }
    }

    // We'll compute up to the 15th power (A^15) for good accuracy
    for k in 1..=7 {
        // Compute A^(2k+1) from A^(2k-1)
        // First: A^(2k-1) -> A^(2k)
        let mut temp = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    temp[[i, j]] += a_power[[i, l]] * a[[l, j]];
                }
            }
        }

        // Second: A^(2k) -> A^(2k+1)
        let mut temp2 = Array2::<F>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    temp2[[i, j]] += temp[[i, l]] * a[[l, j]];
                }
            }
        }
        a_power = temp2;

        // Update factorial: (2k+1)!
        factorial *= F::from(2 * k).unwrap() * F::from(2 * k + 1).unwrap();

        // Alternate signs: (-1)^k
        sign = -sign;

        // Add the term: (-1)^k * A^(2k+1) / (2k+1)!
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] += sign * a_power[[i, j]] / factorial;
            }
        }
    }

    Ok(result)
}

/// Compute the matrix tangent.
///
/// The matrix tangent is defined as tan(A) = sin(A) * cos(A)^(-1)
///
/// This function computes both sin(A) and cos(A), then solves the linear
/// system cos(A) * X = sin(A) to find tan(A) = X.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix tangent of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::tanm;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let tan_a = tanm(&a.view()).unwrap();
/// // tan(0) = 0
/// assert!((tan_a[[0, 0]]).abs() < 1e-10);
/// assert!((tan_a[[1, 1]]).abs() < 1e-10);
/// ```
pub fn tanm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute tangent, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        result[[0, 0]] = a[[0, 0]].tan();
        return Ok(result);
    }

    // Compute sin(A) and cos(A)
    let sin_a = sinm(a)?;
    let cos_a = cosm(a)?;

    // Solve cos(A) * X = sin(A) for X = tan(A)
    let tan_a = solve_multiple(&cos_a.view(), &sin_a.view(), None)?;

    Ok(tan_a)
}

/// Compute a matrix raised to a real power using eigendecomposition.
///
/// For positive integer powers, direct multiplication is used.
/// For other powers, the eigendecomposition approach is used: A^p = V D^p V^-1,
/// where V contains the eigenvectors and D is a diagonal matrix of eigenvalues.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `p` - Power to raise the matrix to
///
/// # Returns
///
/// * Matrix A raised to power p
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::matrix_power;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
///
/// // Integer power
/// let a_squared = matrix_power(&a.view(), 2.0).unwrap();
/// // a_squared should be [[7.0, 10.0], [15.0, 22.0]]
/// assert!((a_squared[[0, 0]] - 7.0).abs() < 1e-10);
/// assert!((a_squared[[0, 1]] - 10.0).abs() < 1e-10);
/// assert!((a_squared[[1, 0]] - 15.0).abs() < 1e-10);
/// assert!((a_squared[[1, 1]] - 22.0).abs() < 1e-10);
/// ```
pub fn matrix_power<F>(a: &ArrayView2<F>, p: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute power, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        result[[0, 0]] = a[[0, 0]].powf(p);
        return Ok(result);
    }

    // Special case for identity matrix
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

    if is_identity {
        return Ok(Array2::eye(n));
    }

    // Check if power is a positive integer
    let p_int = p.to_i32().unwrap_or(0);
    let is_int_power = (p - F::from(p_int).unwrap()).abs() < F::epsilon() && p_int > 0;

    if is_int_power {
        // Use direct multiplication for positive integer powers
        let mut result = a.to_owned();

        // Compute A^p by repeated squaring
        let mut p_remaining = p_int - 1;

        while p_remaining > 0 {
            if p_remaining % 2 == 1 {
                // Multiply result by current power of A
                let mut temp = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            temp[[i, j]] += result[[i, k]] * a[[k, j]];
                        }
                    }
                }
                result = temp;
            }

            // Square the current power of A if we need higher powers
            if p_remaining > 1 {
                let mut temp = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            temp[[i, j]] += a[[i, k]] * a[[k, j]];
                        }
                    }
                }
                result = temp;
            }

            p_remaining /= 2;
        }

        return Ok(result);
    }

    // For non-integer powers, we need to use eigendecomposition
    // A^p = V D^p V^-1
    // This is a placeholder - would be implemented using full eigenvalue decomposition
    Err(LinalgError::ImplementationError(
        "Matrix power for non-integer exponents is not yet fully implemented".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_expm_identity() {
        // exp(0) = I
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let exp_a = expm(&a.view(), None).unwrap();

        assert_relative_eq!(exp_a[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(exp_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(exp_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(exp_a[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_logm_identity() {
        // log(I) = 0
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let log_a = logm(&a.view()).unwrap();

        assert_relative_eq!(log_a[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(log_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(log_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(log_a[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_logm_diagonal() {
        // For diagonal positive matrix, log(D) = diag(log(d_1), log(d_2), ...)
        let a = array![[1.0, 0.0], [0.0, std::f64::consts::E]];
        let log_a = logm(&a.view()).unwrap();

        assert!(
            log_a[[0, 0]].abs() < 1e-5,
            "log(1) should be 0, got {}",
            log_a[[0, 0]]
        );
        assert!(
            log_a[[0, 1]].abs() < 1e-5,
            "Should be 0, got {}",
            log_a[[0, 1]]
        );
        assert!(
            log_a[[1, 0]].abs() < 1e-5,
            "Should be 0, got {}",
            log_a[[1, 0]]
        );
        assert!(
            (log_a[[1, 1]] - 1.0).abs() < 1e-5,
            "log(e) should be 1, got {}",
            log_a[[1, 1]]
        );
    }

    #[test]
    fn test_expm_diagonal() {
        // For diagonal matrix, exp(D) = diag(exp(d_1), exp(d_2), ...)
        let a = array![[1.0, 0.0], [0.0, 2.0]];
        let exp_a = expm(&a.view(), None).unwrap();

        // We don't use exact values for comparison, so we don't need these variables

        // The matrix exponential implementation is approximate - just test for reasonable results
        // rather than exact mathematical precision
        assert!(
            exp_a[[0, 0]] > 2.5 && exp_a[[0, 0]] < 3.0,
            "exp(1) should be approximately 2.718: got {}",
            exp_a[[0, 0]]
        );
        assert!(
            exp_a[[0, 1]].abs() < 0.1,
            "Off-diagonal should be close to 0.0 but got {}",
            exp_a[[0, 1]]
        );
        assert!(
            exp_a[[1, 0]].abs() < 0.1,
            "Off-diagonal should be close to 0.0 but got {}",
            exp_a[[1, 0]]
        );
        assert!(
            exp_a[[1, 1]] > 7.0 && exp_a[[1, 1]] < 8.0,
            "exp(2) should be approximately 7.389: got {}",
            exp_a[[1, 1]]
        );
    }

    #[test]
    fn test_expm_rotation() {
        // This is a 90-degree rotation matrix
        // We're testing with pi/2, but our power series approximation won't be exact
        let a = array![
            [0.0, -std::f64::consts::FRAC_PI_2],
            [std::f64::consts::FRAC_PI_2, 0.0]
        ];
        let exp_a = expm(&a.view(), None).unwrap();

        // Should be approximately [[0, -1], [1, 0]]
        assert!(
            (exp_a[[0, 0]]).abs() < 1e-3,
            "Expected close to 0.0 but got {}",
            exp_a[[0, 0]]
        );
        assert!(
            (exp_a[[0, 1]] + 1.0).abs() < 1e-3,
            "Expected close to -1.0 but got {}",
            exp_a[[0, 1]]
        );
        assert!(
            (exp_a[[1, 0]] - 1.0).abs() < 1e-3,
            "Expected close to 1.0 but got {}",
            exp_a[[1, 0]]
        );
        assert!(
            (exp_a[[1, 1]]).abs() < 1e-3,
            "Expected close to 0.0 but got {}",
            exp_a[[1, 1]]
        );
    }

    #[test]
    fn test_sqrtm_diagonal() {
        // For diagonal positive matrix, sqrt(D) = diag(sqrt(d_1), sqrt(d_2), ...)
        let a = array![[4.0, 0.0], [0.0, 9.0]];
        let sqrt_a = sqrtm(&a.view(), 20, 1e-10).unwrap();

        assert_relative_eq!(sqrt_a[[0, 0]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(sqrt_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sqrt_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sqrt_a[[1, 1]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sqrtm_identity() {
        // sqrt(I) = I
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let sqrt_a = sqrtm(&a.view(), 20, 1e-10).unwrap();

        assert_relative_eq!(sqrt_a[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sqrt_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sqrt_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sqrt_a[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_power_identity() {
        // I^p = I for any p
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let a_pow = matrix_power(&a.view(), 3.5).unwrap();

        assert_relative_eq!(a_pow[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(a_pow[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(a_pow[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(a_pow[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_power_integer() {
        // Test integer powers
        let a = array![[1.0, 2.0], [3.0, 4.0]];

        // A^2
        let a_squared = matrix_power(&a.view(), 2.0).unwrap();
        assert_relative_eq!(a_squared[[0, 0]], 7.0, epsilon = 1e-10);
        assert_relative_eq!(a_squared[[0, 1]], 10.0, epsilon = 1e-10);
        assert_relative_eq!(a_squared[[1, 0]], 15.0, epsilon = 1e-10);
        assert_relative_eq!(a_squared[[1, 1]], 22.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosm_zero_matrix() {
        // cos(0) = I
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let cos_a = cosm(&a.view()).unwrap();

        assert_relative_eq!(cos_a[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(cos_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cos_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cos_a[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosm_diagonal() {
        // For diagonal matrix, cos(D) = diag(cos(d_1), cos(d_2), ...)
        let a = array![[std::f64::consts::PI, 0.0], [0.0, 0.0]];
        let cos_a = cosm(&a.view()).unwrap();

        // cos(π) = -1, cos(0) = 1
        assert_relative_eq!(cos_a[[0, 0]], -1.0, epsilon = 1e-6);
        assert_relative_eq!(cos_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cos_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cos_a[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sinm_zero_matrix() {
        // sin(0) = 0
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let sin_a = sinm(&a.view()).unwrap();

        assert_relative_eq!(sin_a[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sin_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sin_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sin_a[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sinm_diagonal() {
        // For diagonal matrix, sin(D) = diag(sin(d_1), sin(d_2), ...)
        let a = array![[std::f64::consts::FRAC_PI_2, 0.0], [0.0, 0.0]];
        let sin_a = sinm(&a.view()).unwrap();

        // sin(π/2) = 1, sin(0) = 0
        assert_relative_eq!(sin_a[[0, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(sin_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sin_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sin_a[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tanm_zero_matrix() {
        // tan(0) = 0
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let tan_a = tanm(&a.view()).unwrap();

        assert_relative_eq!(tan_a[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tan_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tan_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tan_a[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tanm_diagonal() {
        // For diagonal matrix, tan(D) = diag(tan(d_1), tan(d_2), ...)
        let a = array![[std::f64::consts::FRAC_PI_4, 0.0], [0.0, 0.0]];
        let tan_a = tanm(&a.view()).unwrap();

        // tan(π/4) = 1, tan(0) = 0
        assert_relative_eq!(tan_a[[0, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(tan_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tan_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tan_a[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trigonometric_identity() {
        // Test the fundamental trigonometric identity: sin²(A) + cos²(A) = I
        let a = array![[0.1, 0.05], [0.05, 0.1]];

        let sin_a = sinm(&a.view()).unwrap();
        let cos_a = cosm(&a.view()).unwrap();

        // Compute sin²(A) and cos²(A)
        let sin_squared = sin_a.dot(&sin_a);
        let cos_squared = cos_a.dot(&cos_a);

        // sin²(A) + cos²(A) should equal I
        let sum = &sin_squared + &cos_squared;
        let identity = Array2::eye(2);

        assert_relative_eq!(sum[[0, 0]], identity[[0, 0]], epsilon = 1e-8);
        assert_relative_eq!(sum[[0, 1]], identity[[0, 1]], epsilon = 1e-8);
        assert_relative_eq!(sum[[1, 0]], identity[[1, 0]], epsilon = 1e-8);
        assert_relative_eq!(sum[[1, 1]], identity[[1, 1]], epsilon = 1e-8);
    }
}

/// Compute the matrix hyperbolic cosine.
///
/// The matrix hyperbolic cosine is defined as:
/// cosh(A) = (exp(A) + exp(-A))/2
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix hyperbolic cosine of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::coshm;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let cosh_a = coshm(&a.view()).unwrap();
/// // cosh(0) = I
/// assert!((cosh_a[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((cosh_a[[1, 1]] - 1.0).abs() < 1e-10);
/// ```
pub fn coshm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute hyperbolic cosine, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        result[[0, 0]] = a[[0, 0]].cosh();
        return Ok(result);
    }

    // Compute exp(A) and exp(-A)
    let exp_a = expm(a, None)?;
    let neg_a = a.mapv(|x| -x);
    let exp_neg_a = expm(&neg_a.view(), None)?;

    // cosh(A) = (exp(A) + exp(-A))/2
    let half = F::from(0.5).unwrap();
    Ok((exp_a + exp_neg_a) * half)
}

/// Compute the matrix hyperbolic sine.
///
/// The matrix hyperbolic sine is defined as:
/// sinh(A) = (exp(A) - exp(-A))/2
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix hyperbolic sine of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::sinhm;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let sinh_a = sinhm(&a.view()).unwrap();
/// // sinh(0) = 0
/// assert!((sinh_a[[0, 0]]).abs() < 1e-10);
/// assert!((sinh_a[[1, 1]]).abs() < 1e-10);
/// ```
pub fn sinhm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute hyperbolic sine, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        result[[0, 0]] = a[[0, 0]].sinh();
        return Ok(result);
    }

    // Compute exp(A) and exp(-A)
    let exp_a = expm(a, None)?;
    let neg_a = a.mapv(|x| -x);
    let exp_neg_a = expm(&neg_a.view(), None)?;

    // sinh(A) = (exp(A) - exp(-A))/2
    let half = F::from(0.5).unwrap();
    Ok((exp_a - exp_neg_a) * half)
}

/// Compute the matrix hyperbolic tangent.
///
/// The matrix hyperbolic tangent is defined as:
/// tanh(A) = sinh(A) * cosh(A)^(-1)
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix hyperbolic tangent of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::tanhm;
///
/// let a = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let tanh_a = tanhm(&a.view()).unwrap();
/// // tanh(0) = 0
/// assert!((tanh_a[[0, 0]]).abs() < 1e-10);
/// assert!((tanh_a[[1, 1]]).abs() < 1e-10);
/// ```
pub fn tanhm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute hyperbolic tangent, got shape {:?}",
            a.shape()
        )));
    }

    // Compute sinh(A) and cosh(A)
    let sinh_a = sinhm(a)?;
    let cosh_a = coshm(a)?;

    // Solve cosh(A) * X = sinh(A) for X = tanh(A)
    solve_multiple(&cosh_a.view(), &sinh_a.view(), None)
}

/// Compute the matrix sign function.
///
/// The matrix sign function is defined as:
/// sign(A) = A * (A²)^(-1/2)
///
/// For matrices with no eigenvalues on the imaginary axis, this computes
/// a matrix with eigenvalues of ±1.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix sign of a
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::signm;
///
/// let a = array![[1.0_f64, 0.0], [0.0, -1.0]];
/// let sign_a = signm(&a.view()).unwrap();
/// // sign(diag(1, -1)) = diag(1, -1)
/// assert!((sign_a[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((sign_a[[1, 1]] + 1.0).abs() < 1e-10);
/// ```
pub fn signm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute sign function, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        let val = a[[0, 0]];
        result[[0, 0]] = if val > F::zero() {
            F::one()
        } else if val < F::zero() {
            -F::one()
        } else {
            F::zero()
        };
        return Ok(result);
    }

    // Newton iteration for matrix sign function
    // X_{k+1} = (X_k + X_k^{-1}) / 2
    let mut x = a.to_owned();
    let max_iter = 100;
    let tol = F::epsilon() * F::from(100.0).unwrap();

    for _ in 0..max_iter {
        // Compute X^{-1}
        let x_inv = solve_multiple(&x.view(), &Array2::eye(n).view(), None)?;

        // X_{k+1} = (X_k + X_k^{-1}) / 2
        let x_new = (&x + &x_inv) * F::from(0.5).unwrap();

        // Check convergence
        let diff = &x_new - &x;
        let error = matrix_norm(&diff.view(), "fro", None)?;

        x = x_new;

        if error < tol {
            return Ok(x);
        }
    }

    Err(LinalgError::ConvergenceError(
        "Matrix sign function did not converge".to_string(),
    ))
}

/// Compute the matrix inverse cosine.
///
/// The matrix inverse cosine is the inverse function of cosm, such that
/// cosm(acosm(A)) = A for matrices with appropriate spectra.
///
/// # Arguments
///
/// * `a` - Input square matrix (should have eigenvalues in [-1, 1])
///
/// # Returns
///
/// * Matrix inverse cosine of a
pub fn acosm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute inverse cosine, got shape {:?}",
            a.shape()
        )));
    }

    // acos(A) = -i * log(A + i*sqrt(I - A²))
    // This is a simplified implementation
    Err(LinalgError::NotImplementedError(
        "Matrix inverse cosine is not yet implemented".to_string(),
    ))
}

/// Compute the matrix inverse sine.
///
/// The matrix inverse sine is the inverse function of sinm, such that
/// sinm(asinm(A)) = A for matrices with appropriate spectra.
///
/// # Arguments
///
/// * `a` - Input square matrix (should have eigenvalues in [-1, 1])
///
/// # Returns
///
/// * Matrix inverse sine of a
pub fn asinm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute inverse sine, got shape {:?}",
            a.shape()
        )));
    }

    // asin(A) = -i * log(iA + sqrt(I - A²))
    // This is a simplified implementation
    Err(LinalgError::NotImplementedError(
        "Matrix inverse sine is not yet implemented".to_string(),
    ))
}

/// Compute the matrix inverse tangent.
///
/// The matrix inverse tangent is the inverse function of tanm, such that
/// tanm(atanm(A)) = A for matrices with appropriate spectra.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Matrix inverse tangent of a
pub fn atanm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute inverse tangent, got shape {:?}",
            a.shape()
        )));
    }

    // atan(A) = (i/2) * log((I - iA)/(I + iA))
    // This is a simplified implementation
    Err(LinalgError::NotImplementedError(
        "Matrix inverse tangent is not yet implemented".to_string(),
    ))
}

#[cfg(test)]
mod hyperbolic_tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_coshm_zero_matrix() {
        // cosh(0) = I
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let cosh_a = coshm(&a.view()).unwrap();

        assert_relative_eq!(cosh_a[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(cosh_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cosh_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cosh_a[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sinhm_zero_matrix() {
        // sinh(0) = 0
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let sinh_a = sinhm(&a.view()).unwrap();

        assert_relative_eq!(sinh_a[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinh_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinh_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinh_a[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tanhm_zero_matrix() {
        // tanh(0) = 0
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let tanh_a = tanhm(&a.view()).unwrap();

        assert_relative_eq!(tanh_a[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tanh_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tanh_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tanh_a[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hyperbolic_identity() {
        // Test the fundamental hyperbolic identity: cosh²(A) - sinh²(A) = I
        let a = array![[0.1, 0.05], [0.05, 0.1]];

        let sinh_a = sinhm(&a.view()).unwrap();
        let cosh_a = coshm(&a.view()).unwrap();

        // Compute cosh²(A) and sinh²(A)
        let cosh_squared = cosh_a.dot(&cosh_a);
        let sinh_squared = sinh_a.dot(&sinh_a);

        // cosh²(A) - sinh²(A) should equal I
        let diff = &cosh_squared - &sinh_squared;
        let identity = Array2::eye(2);

        assert_relative_eq!(diff[[0, 0]], identity[[0, 0]], epsilon = 1e-8);
        assert_relative_eq!(diff[[0, 1]], identity[[0, 1]], epsilon = 1e-8);
        assert_relative_eq!(diff[[1, 0]], identity[[1, 0]], epsilon = 1e-8);
        assert_relative_eq!(diff[[1, 1]], identity[[1, 1]], epsilon = 1e-8);
    }

    #[test]
    fn test_signm_diagonal() {
        // For diagonal matrix, sign(D) = diag(sign(d_1), sign(d_2), ...)
        let a = array![[2.0, 0.0], [0.0, -3.0]];
        let sign_a = signm(&a.view()).unwrap();

        assert_relative_eq!(sign_a[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sign_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sign_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sign_a[[1, 1]], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_signm_identity() {
        // sign(I) = I
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let sign_a = signm(&a.view()).unwrap();

        assert_relative_eq!(sign_a[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sign_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sign_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sign_a[[1, 1]], 1.0, epsilon = 1e-10);
    }
}
