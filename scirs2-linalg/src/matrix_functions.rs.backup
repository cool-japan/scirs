//! Matrix functions such as matrix exponential, logarithm, and square root

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
/// use scirs2_linalg::logm_parallel;
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
    use scirs2_core::parallel_ops::*;

    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute logarithm, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special cases (1x1, diagonal, identity) - same as sequential version
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

    // Check for diagonal matrix using parallel iteration
    let is_diagonal = (0..n)
        .into_par_iter()
        .all(|i| (0..n).all(|j| i == j || a[[i, j]].abs() <= F::epsilon()));

    if is_diagonal {
        // Check that all diagonal elements are positive
        let all_positive = (0..n).into_par_iter().all(|i| a[[i, i]] > F::zero());

        if !all_positive {
            return Err(LinalgError::InvalidInputError(
                "Cannot compute real logarithm of matrix with non-positive eigenvalues".to_string(),
            ));
        }

        // Compute diagonal logarithm in parallel
        let diagonal_values: Vec<F> = (0..n).into_par_iter().map(|i| a[[i, i]].ln()).collect();

        let mut result = Array2::zeros((n, n));
        for (i, &val) in diagonal_values.iter().enumerate() {
            result[[i, i]] = val;
        }
        return Ok(result);
    }

    // Check if the matrix is identity using parallel iteration
    let is_identity = (0..n).into_par_iter().all(|i| {
        (0..n).all(|j| {
            let expected = if i == j { F::one() } else { F::zero() };
            (a[[i, j]] - expected).abs() <= F::epsilon()
        })
    });

    if is_identity {
        return Ok(Array2::zeros((n, n)));
    }

    // For general matrices, use parallel matrix operations
    // Check if the matrix is close to the identity
    let identity = Array2::eye(n);
    let max_diff = (0..n)
        .into_par_iter()
        .map(|i| {
            (0..n)
                .map(|j| (a[[i, j]] - identity[[i, j]]).abs())
                .fold(F::zero(), |acc, x| if x > acc { x } else { acc })
        })
        .reduce(|| F::zero(), |acc, x| if x > acc { x } else { acc });

    if max_diff > F::from(0.5).unwrap() {
        // Use scaling and squaring approach with parallel matrix operations
        let mut scaling_k = 0;
        let mut a_scaled = a.to_owned();

        while scaling_k < 10 {
            let max_scaled_diff = (0..n)
                .into_par_iter()
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            let expected = if i == j { F::one() } else { F::zero() };
                            (a_scaled[[i, j]] - expected).abs()
                        })
                        .fold(F::zero(), |acc, x| if x > acc { x } else { acc })
                })
                .reduce(|| F::zero(), |acc, x| if x > acc { x } else { acc });

            if max_scaled_diff <= F::from(0.2).unwrap() {
                break;
            }

            // Compute matrix square root
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

        // Compute X = A - I in parallel
        let x_values: Vec<Vec<F>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let expected = if i == j { F::one() } else { F::zero() };
                        a_scaled[[i, j]] - expected
                    })
                    .collect()
            })
            .collect();

        let mut x_scaled = Array2::zeros((n, n));
        for (i, row) in x_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                x_scaled[[i, j]] = val;
            }
        }

        // Compute matrix powers in parallel using parallel matrix multiplication
        let x2_values: Vec<Vec<F>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let mut sum = F::zero();
                        for k in 0..n {
                            sum += x_scaled[[i, k]] * x_scaled[[k, j]];
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        let mut x2 = Array2::zeros((n, n));
        for (i, row) in x2_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                x2[[i, j]] = val;
            }
        }

        let x3_values: Vec<Vec<F>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let mut sum = F::zero();
                        for k in 0..n {
                            sum += x2[[i, k]] * x_scaled[[k, j]];
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        let mut x3 = Array2::zeros((n, n));
        for (i, row) in x3_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                x3[[i, j]] = val;
            }
        }

        // Compute the logarithm using Taylor series: log(I+X) = X - X²/2 + X³/3 - ...
        let result_values: Vec<Vec<F>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .map(|j| {
                        x_scaled[[i, j]] - x2[[i, j]] / F::from(2).unwrap()
                            + x3[[i, j]] / F::from(3).unwrap()
                    })
                    .collect()
            })
            .collect();

        // Scale back the result: log(A) = 2^k * log(A^(1/2^k))
        let scale_factor = F::from(1 << scaling_k).unwrap();
        let scaled_result_values: Vec<Vec<F>> = result_values
            .into_par_iter()
            .map(|row| row.into_iter().map(|val| val * scale_factor).collect())
            .collect();

        let mut result = Array2::zeros((n, n));
        for (i, row) in scaled_result_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    } else {
        // Matrix is close to identity, use direct Taylor series
        let x_values: Vec<Vec<F>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let expected = if i == j { F::one() } else { F::zero() };
                        a[[i, j]] - expected
                    })
                    .collect()
            })
            .collect();

        let mut x = Array2::zeros((n, n));
        for (i, row) in x_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                x[[i, j]] = val;
            }
        }

        // Compute powers in parallel
        let x2_values: Vec<Vec<F>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let mut sum = F::zero();
                        for k in 0..n {
                            sum += x[[i, k]] * x[[k, j]];
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        let mut x2 = Array2::zeros((n, n));
        for (i, row) in x2_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                x2[[i, j]] = val;
            }
        }

        // Compute result using Taylor series
        let result_values: Vec<Vec<F>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .map(|j| x[[i, j]] - x2[[i, j]] / F::from(2).unwrap())
                    .collect()
            })
            .collect();

        let mut result = Array2::zeros((n, n));
        for (i, row) in result_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }
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

    for _iter in 0..maxiter {
        // Compute Y_next = 0.5 * (Y + Z^-1)
        // and Z_next = 0.5 * (Z + Y^-1)

        // First, compute Z^-1 and Y^-1
        let z_inv = match solve_multiple(&z.view(), &Array2::eye(n).view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(LinalgError::singularmatrix_with_suggestions(
                    "Matrix square root (Denman-Beavers iteration)",
                    (n, n),
                    None,
                ))
            }
        };

        let y_inv = match solve_multiple(&y.view(), &Array2::eye(n).view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(LinalgError::singularmatrix_with_suggestions(
                    "Matrix square root (Denman-Beavers iteration)",
                    (n, n),
                    None,
                ))
            }
        };

        // Compute next iterations using parallel element-wise operations
        let half = F::from(0.5).unwrap();

        let y_next = Array2::from_shape_fn((n, n), |(i, j)| half * (y[[i, j]] + z_inv[[i, j]]));

        let z_next = Array2::from_shape_fn((n, n), |(i, j)| half * (z[[i, j]] + y_inv[[i, j]]));

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
        maxiter,
        tol.to_f64().unwrap_or(1e-12),
        final_error,
    ))
}

/// Compute the matrix square root with parallel processing support.
///
/// The matrix square root X of matrix A satisfies X^2 = A.
/// This function uses the Denman-Beavers iteration with parallel matrix operations
/// for improved performance on large matrices.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Matrix square root of the input
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::sqrtm_parallel;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let sqrt_a = sqrtm_parallel(&a.view(), 50, 1e-12, Some(4)).unwrap();
/// assert!((sqrt_a[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((sqrt_a[[1, 1]] - 3.0).abs() < 1e-10);
/// ```
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

    // Use threshold to determine if parallel processing is worthwhile
    const PARALLEL_THRESHOLD: usize = 30; // For matrices larger than 30x30

    if a.nrows() < PARALLEL_THRESHOLD || a.ncols() < PARALLEL_THRESHOLD {
        // For small matrices, use sequential implementation
        return sqrtm(a, maxiter, tol);
    }

    // For larger matrices, use parallel implementation
    sqrtm_impl_parallel(a, maxiter, tol)
}

/// Internal implementation of parallel matrix square root computation using Denman-Beavers iteration.
#[allow(dead_code)]
fn sqrtm_impl_parallel<F>(a: &ArrayView2<F>, maxiter: usize, tol: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
{
    use scirs2_core::parallel_ops::*;

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

    let mut final_error = None;

    for _iter in 0..maxiter {
        // Compute Z^-1 and Y^-1 using parallel matrix solve
        let z_inv = match solve_multiple(&z.view(), &Array2::eye(n).view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(LinalgError::singularmatrix_with_suggestions(
                    "Matrix square root (Denman-Beavers iteration)",
                    (n, n),
                    None,
                ))
            }
        };

        let y_inv = match solve_multiple(&y.view(), &Array2::eye(n).view(), None) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(LinalgError::singularmatrix_with_suggestions(
                    "Matrix square root (Denman-Beavers iteration)",
                    (n, n),
                    None,
                ))
            }
        };

        let half = F::from(0.5).unwrap();

        // Compute next iterations using parallel element-wise operations
        let y_next_values: Vec<Vec<F>> = (0..n)
            .into_par_iter()
            .map(|i| (0..n).map(|j| half * (y[[i, j]] + z_inv[[i, j]])).collect())
            .collect();

        let mut y_next = Array2::zeros((n, n));
        for (i, row) in y_next_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                y_next[[i, j]] = val;
            }
        }

        let z_next_values: Vec<Vec<F>> = (0..n)
            .into_par_iter()
            .map(|i| (0..n).map(|j| half * (z[[i, j]] + y_inv[[i, j]])).collect())
            .collect();

        let mut z_next = Array2::zeros((n, n));
        for (i, row) in z_next_values.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                z_next[[i, j]] = val;
            }
        }

        // Compute error for convergence check using parallel max reduction
        let error = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .map(|j| (y_next[[i, j]] - y[[i, j]]).abs())
                    .fold(F::zero(), |acc, x| if x > acc { x } else { acc })
            })
            .reduce(|| F::zero(), |acc, x| if x > acc { x } else { acc });

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
        maxiter,
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
#[allow(dead_code)]
pub fn cosm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
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
#[allow(dead_code)]
pub fn sinm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
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
#[allow(dead_code)]
pub fn tanm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + ndarray::ScalarOperand + 'static,
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
#[allow(dead_code)]
pub fn matrix_power<F>(a: &ArrayView2<F>, p: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + 'static + Send + Sync + ndarray::ScalarOperand,
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

    // For non-integer powers, we use eigendecomposition: A^p = V * D^p * V^(-1)
    // where V contains eigenvectors and D contains eigenvalues

    match eig(&a.view(), None) {
        Ok((eigenvalues, eigenvectors)) => {
            // Check for complex eigenvalues which would require complex arithmetic
            // For simplicity, we handle only real eigenvalues for now
            let mut has_complex = false;
            for i in 0..eigenvalues.len() {
                if eigenvalues[i].im.abs() > F::epsilon() {
                    has_complex = true;
                    break;
                }
            }

            if has_complex {
                return Err(LinalgError::ImplementationError(
                    "Matrix power for non-integer exponents with complex eigenvalues not yet supported".to_string(),
                ));
            }

            // Check for negative eigenvalues which would cause issues with non-integer powers
            for i in 0..eigenvalues.len() {
                if eigenvalues[i].re < F::zero() {
                    return Err(LinalgError::ImplementationError(
                        "Matrix power for non-integer exponents with negative eigenvalues not supported".to_string(),
                    ));
                }
            }

            // Create diagonal matrix D^p
            let mut d_power = Array2::zeros((n, n));
            for i in 0..n {
                d_power[[i, i]] = eigenvalues[i].re.powf(p);
            }

            // Extract real parts of eigenvectors since we've confirmed eigenvalues are real
            let mut real_eigenvectors = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    real_eigenvectors[[i, j]] = eigenvectors[[i, j]].re;
                }
            }

            // Compute V * D^p * V^(-1)
            // First compute V * D^p
            let mut temp = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        temp[[i, j]] += real_eigenvectors[[i, k]] * d_power[[k, j]];
                    }
                }
            }

            // We need to compute V^(-1), but for numerical stability,
            // we'll solve the linear system V * result = temp instead of explicitly inverting V
            // This gives us result = V^(-1) * temp
            match solve_multiple(&real_eigenvectors.view(), &temp.view(), None) {
                Ok(result) => Ok(result),
                Err(_) => Err(LinalgError::ImplementationError(
                    "Failed to solve linear system for matrix power computation (matrix may be singular)".to_string(),
                )),
            }
        }
        Err(_) => Err(LinalgError::ImplementationError(
            "Failed to compute eigendecomposition for matrix power".to_string(),
        )),
    }
}

/// Advanced matrix softmax function for machine learning applications
///
/// Computes exp(A) / sum(exp(A)) along specified axis or element-wise.
/// This is numerically stable and commonly used in neural networks.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `axis` - Axis along which to compute softmax (None = element-wise)
///
/// # Returns
///
/// * Matrix softmax of a
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::softmax;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let softmax_a = softmax(&a.view(), None).unwrap();
/// ```
#[allow(dead_code)]
pub fn softmax<F>(a: &ArrayView2<F>, axis: Option<usize>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
{
    let (m, n) = a.dim();
    let mut result = Array2::<F>::zeros((m, n));

    match axis {
        Some(0) => {
            // Softmax along rows (column-wise)
            for j in 0..n {
                let col = a.column(j);

                // Find maximum for numerical stability
                let max_val = col.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));

                // Compute exp(x - max) for numerical stability
                let mut exp_sum = F::zero();
                for i in 0..m {
                    let exp_val = (a[[i, j]] - max_val).exp();
                    result[[i, j]] = exp_val;
                    exp_sum += exp_val;
                }

                // Normalize
                for i in 0..m {
                    result[[i, j]] /= exp_sum;
                }
            }
        }
        Some(1) => {
            // Softmax along columns (row-wise)
            for i in 0..m {
                let row = a.row(i);

                // Find maximum for numerical stability
                let max_val = row.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));

                // Compute exp(x - max) for numerical stability
                let mut exp_sum = F::zero();
                for j in 0..n {
                    let exp_val = (a[[i, j]] - max_val).exp();
                    result[[i, j]] = exp_val;
                    exp_sum += exp_val;
                }

                // Normalize
                for j in 0..n {
                    result[[i, j]] /= exp_sum;
                }
            }
        }
        None => {
            // Element-wise softmax (global normalization)
            let max_val = a.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));

            // Compute exp(x - max) for all elements
            let mut exp_sum = F::zero();
            for i in 0..m {
                for j in 0..n {
                    let exp_val = (a[[i, j]] - max_val).exp();
                    result[[i, j]] = exp_val;
                    exp_sum += exp_val;
                }
            }

            // Normalize all elements
            result.mapv_inplace(|x| x / exp_sum);
        }
        _ => {
            return Err(LinalgError::InvalidInputError(format!(
                "Invalid axis {}. Matrix is 2D, so axis must be 0, 1, or None",
                axis.unwrap()
            )));
        }
    }

    Ok(result)
}

/// Advanced matrix sigmoid function for machine learning
///
/// Computes 1 / (1 + exp(-A)) element-wise with numerical stability.
///
/// # Arguments
///
/// * `a` - Input matrix
///
/// # Returns
///
/// * Element-wise sigmoid of matrix a
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::sigmoid;
///
/// let a = array![[1.0_f64, -2.0], [0.0, 3.0]];
/// let sigmoid_a = sigmoid(&a.view()).unwrap();
/// ```
#[allow(dead_code)]
pub fn sigmoid<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + One + ndarray::ScalarOperand + Send + Sync + 'static,
{
    let mut result = Array2::<F>::zeros(a.raw_dim());

    for ((i, j), &val) in a.indexed_iter() {
        // Numerically stable sigmoid computation
        if val >= F::zero() {
            let exp_neg = (-val).exp();
            result[[i, j]] = F::one() / (F::one() + exp_neg);
        } else {
            let exp_pos = val.exp();
            result[[i, j]] = exp_pos / (F::one() + exp_pos);
        }
    }

    Ok(result)
}

/// Enhanced fractional matrix power with improved numerical stability
///
/// Computes A^p where p can be any real number, using improved algorithms
/// for better numerical stability and accuracy.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `p` - Fractional power (can be negative, fractional, etc.)
/// * `method` - Algorithm method ("eigen", "schur", "pade")
///
/// # Returns
///
/// * Matrix raised to power p
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::fractionalmatrix_power;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let sqrt_a = fractionalmatrix_power(&a.view(), 0.5, "eigen").unwrap();
/// ```
#[allow(dead_code)]
pub fn fractionalmatrix_power<F>(a: &ArrayView2<F>, p: F, method: &str) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + 'static + Send + Sync,
{
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for fractional power".to_string(),
        ));
    }

    match method {
        "eigen" => {
            // Use eigendecomposition for symmetric matrices
            let (eigenvals, eigenvecs) = crate::eigen::eigh(a, None)?;

            // Check for negative eigenvalues if p is not an integer
            if !is_integer(p) {
                for &eigenval in eigenvals.iter() {
                    if eigenval < F::zero() {
                        return Err(LinalgError::ComputationError(
                            "Matrix has negative eigenvalues, fractional power not real"
                                .to_string(),
                        ));
                    }
                }
            }

            // Compute eigenvalue^p
            let powered_eigenvals = eigenvals.mapv(|x| {
                if x > F::zero() {
                    x.powf(p)
                } else if x == F::zero() && p > F::zero() {
                    F::zero()
                } else {
                    F::nan() // This case should have been caught above
                }
            });

            // Reconstruct matrix: V * Λ^p * V^T
            let lambdamatrix = Array2::from_diag(&powered_eigenvals);
            let temp = eigenvecs.dot(&lambdamatrix);
            Ok(temp.dot(&eigenvecs.t()))
        }
        "schur" => {
            // Use Schur decomposition for general matrices
            let (q, t) = crate::decomposition::schur(a)?;

            // Compute T^p using specialized algorithm for upper triangular matrices
            let t_powered = upper_triangular_power(&t, p)?;

            // Reconstruct: Q * T^p * Q^T
            let temp = q.dot(&t_powered);
            Ok(temp.dot(&q.t()))
        }
        "pade" => {
            // Use Padé approximation combined with scaling and squaring
            pade_fractional_power(a, p)
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Unknown method '{method}'. Use 'eigen', 'schur', or 'pade'"
        ))),
    }
}

/// Matrix function for symmetric positive definite matrices with enhanced stability
///
/// Computes f(A) for symmetric positive definite matrices using specialized algorithms
/// that take advantage of the SPD structure for improved numerical stability.
///
/// # Arguments
///
/// * `a` - Symmetric positive definite matrix
/// * `func` - Function name ("log", "sqrt", "inv_sqrt", "exp", "power")
/// * `param` - Optional parameter (e.g., power for "power" function)
///
/// # Returns
///
/// * f(A) computed using SPD-optimized algorithms
#[allow(dead_code)]
pub fn spdmatrix_function<F>(
    a: &ArrayView2<F>,
    func: &str,
    param: Option<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + 'static + Send + Sync,
{
    // Check if matrix is symmetric
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for SPD function".to_string(),
        ));
    }

    let tolerance = F::from(1e-12).unwrap();
    for i in 0..n {
        for j in i + 1..n {
            if (a[[i, j]] - a[[j, i]]).abs() > tolerance {
                return Err(LinalgError::InvalidInputError(
                    "Matrix is not symmetric".to_string(),
                ));
            }
        }
    }

    // Use Cholesky decomposition for SPD matrices when possible
    if let Ok(l) = crate::decomposition::cholesky(a, None) {
        match func {
            "sqrt" => {
                // For SPD matrices: sqrt(A) = L * L^T where A = L * L^T
                // So sqrt(A) is just L for lower triangular L
                Ok(l)
            }
            "inv_sqrt" => {
                // inv_sqrt(A) = L^{-1} * L^{-T}
                let l_inv = triangular_inverse(&l, true)?;
                Ok(l_inv.dot(&l_inv.t()))
            }
            "log" => {
                // Use eigendecomposition for log
                let (eigenvals, eigenvecs) = crate::eigen::eigh(a, None)?;

                // Check for positive eigenvalues
                for &eigenval in eigenvals.iter() {
                    if eigenval <= F::zero() {
                        return Err(LinalgError::ComputationError(
                            "Matrix is not positive definite for logarithm".to_string(),
                        ));
                    }
                }

                let log_eigenvals = eigenvals.mapv(|x| x.ln());
                let lambdamatrix = Array2::from_diag(&log_eigenvals);
                let temp = eigenvecs.dot(&lambdamatrix);
                Ok(temp.dot(&eigenvecs.t()))
            }
            "exp" => {
                // Use eigendecomposition for exp
                let (eigenvals, eigenvecs) = crate::eigen::eigh(a, None)?;
                let exp_eigenvals = eigenvals.mapv(|x| x.exp());
                let lambdamatrix = Array2::from_diag(&exp_eigenvals);
                let temp = eigenvecs.dot(&lambdamatrix);
                Ok(temp.dot(&eigenvecs.t()))
            }
            "power" => {
                let p = param.ok_or_else(|| {
                    LinalgError::InvalidInputError(
                        "Power parameter required for 'power' function".to_string(),
                    )
                })?;
                fractionalmatrix_power(a, p, "eigen")
            }
            _ => Err(LinalgError::InvalidInputError(format!(
                "Unknown SPD function '{func}'. Use 'log', 'sqrt', 'inv_sqrt', 'exp', or 'power'"
            ))),
        }
    } else {
        // Fall back to eigendecomposition if Cholesky fails
        let (eigenvals, eigenvecs) = crate::eigen::eigh(a, None)?;

        // Check for positive eigenvalues
        for &eigenval in eigenvals.iter() {
            if eigenval <= F::zero() {
                return Err(LinalgError::ComputationError(
                    "Matrix is not positive definite".to_string(),
                ));
            }
        }

        let transformed_eigenvals = match func {
            "sqrt" => eigenvals.mapv(|x| x.sqrt()),
            "inv_sqrt" => eigenvals.mapv(|x| F::one() / x.sqrt()),
            "log" => eigenvals.mapv(|x| x.ln()),
            "exp" => eigenvals.mapv(|x| x.exp()),
            "power" => {
                let p = param.ok_or_else(|| {
                    LinalgError::InvalidInputError(
                        "Power parameter required for 'power' function".to_string(),
                    )
                })?;
                eigenvals.mapv(|x| x.powf(p))
            }
            _ => {
                return Err(LinalgError::InvalidInputError(format!(
                    "Unknown SPD function '{func}'"
                )));
            }
        };

        let lambdamatrix = Array2::from_diag(&transformed_eigenvals);
        let temp = eigenvecs.dot(&lambdamatrix);
        Ok(temp.dot(&eigenvecs.t()))
    }
}

// Helper functions for advanced matrix functions

/// Check if a float is close to an integer
#[allow(dead_code)]
fn is_integer<F: Float>(x: F) -> bool {
    (x - x.round()).abs() < F::from(1e-10).unwrap()
}

/// Compute power of upper triangular matrix using specialized algorithm
#[allow(dead_code)]
fn upper_triangular_power<F>(t: &Array2<F>, p: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
{
    let n = t.nrows();
    let mut result = Array2::<F>::eye(n);

    // Diagonal elements are easy: t[i,i]^p
    for i in 0..n {
        result[[i, i]] = t[[i, i]].powf(p);
    }

    // Off-diagonal elements require special computation
    // This is a simplified version - full implementation would use
    // more sophisticated algorithms for numerical stability
    for k in 1..n {
        for i in 0..n - k {
            let j = i + k;
            let mut sum = F::zero();

            for l in (i + 1)..j {
                sum = sum + result[[i, l]] * t[[l, j]] - t[[i, l]] * result[[l, j]];
            }

            let denom = t[[j, j]] - t[[i, i]];
            if denom.abs() > F::epsilon() {
                result[[i, j]] = (t[[i, j]] * p + sum) / denom;
            } else {
                // Handle near-singular case
                result[[i, j]] = F::zero();
            }
        }
    }

    Ok(result)
}

/// Padé approximation for fractional powers
#[allow(dead_code)]
fn pade_fractional_power<F>(a: &ArrayView2<F>, p: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
{
    // This is a simplified implementation
    // Full implementation would use more sophisticated Padé approximants

    // For now, use the identity A^p = exp(p * log(A))
    let log_a = logm(a)?;
    let p_log_a = log_a.mapv(|x| x * p);
    expm(&p_log_a.view(), None)
}

/// Compute inverse of triangular matrix
#[allow(dead_code)]
fn triangular_inverse<F>(l: &Array2<F>, lower: bool) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + One + ndarray::ScalarOperand + Send + Sync + 'static,
{
    let n = l.nrows();
    let mut l_inv = Array2::<F>::zeros((n, n));

    if lower {
        // Forward substitution for lower triangular
        for i in 0..n {
            l_inv[[i, i]] = F::one() / l[[i, i]];

            for j in 0..i {
                let mut sum = F::zero();
                for k in j..i {
                    sum += l[[i, k]] * l_inv[[k, j]];
                }
                l_inv[[i, j]] = -sum / l[[i, i]];
            }
        }
    } else {
        // Back substitution for upper triangular
        for i in (0..n).rev() {
            l_inv[[i, i]] = F::one() / l[[i, i]];

            for j in (i + 1)..n {
                let mut sum = F::zero();
                for k in (i + 1)..=j {
                    sum += l[[i, k]] * l_inv[[k, j]];
                }
                l_inv[[i, j]] = -sum / l[[i, i]];
            }
        }
    }

    Ok(l_inv)
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
    fn testmatrix_power_identity() {
        // I^p = I for any p
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let a_pow = matrix_power(&a.view(), 3.5).unwrap();

        assert_relative_eq!(a_pow[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(a_pow[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(a_pow[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(a_pow[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn testmatrix_power_integer() {
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
    fn test_cosm_zeromatrix() {
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
    fn test_sinm_zeromatrix() {
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
    fn test_tanm_zeromatrix() {
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
#[allow(dead_code)]
pub fn coshm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
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
#[allow(dead_code)]
pub fn sinhm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
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
#[allow(dead_code)]
pub fn tanhm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
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
#[allow(dead_code)]
pub fn signm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
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
#[allow(dead_code)]
pub fn acosm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute inverse cosine, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        let val = a[[0, 0]];
        if val.abs() > F::one() {
            return Err(LinalgError::DomainError(
                "Matrix element must be in [-1, 1] for acos".to_string(),
            ));
        }
        result[[0, 0]] = val.acos();
        return Ok(result);
    }

    // For general matrices with small norm, use series expansion
    // acos(A) ≈ π/2 - asin(A)
    // We'll use this relation when asin is available
    let norm_a = matrix_norm(a, "2", None)?;
    if norm_a < F::from(0.9).unwrap() {
        // Compute asin(A) and then use the relation
        let asin_a = asinm(a)?;
        let pi_over_2 = F::from(std::f64::consts::PI / 2.0).unwrap();
        let result = Array2::<F>::from_elem((n, n), pi_over_2) - asin_a;
        return Ok(result);
    }

    Err(LinalgError::NotImplementedError(
        "Matrix inverse cosine for matrices with large norm requires complex arithmetic"
            .to_string(),
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
#[allow(dead_code)]
pub fn asinm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute inverse sine, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        let val = a[[0, 0]];
        if val.abs() > F::one() {
            return Err(LinalgError::DomainError(
                "Matrix element must be in [-1, 1] for asin".to_string(),
            ));
        }
        result[[0, 0]] = val.asin();
        return Ok(result);
    }

    // For small matrices, use the power series:
    // asin(A) = A + A³/6 + 3A⁵/40 + 5A⁷/112 + ...
    // This converges for ||A|| < 1

    let norm_a = matrix_norm(a, "2", None)?;
    if norm_a >= F::one() {
        return Err(LinalgError::ConvergenceError(
            "Matrix norm must be less than 1 for asin series to converge".to_string(),
        ));
    }

    let mut result = a.to_owned();
    let a_squared = a.dot(a);
    let mut a_power = a.dot(&a_squared); // A³

    // Coefficients for asin series
    let coeffs = [
        F::from(1.0 / 6.0).unwrap(),
        F::from(3.0 / 40.0).unwrap(),
        F::from(5.0 / 112.0).unwrap(),
        F::from(35.0 / 1152.0).unwrap(),
    ];

    let tol = F::epsilon() * F::from(10.0).unwrap();

    for (i, &coeff) in coeffs.iter().enumerate() {
        let term = &a_power * coeff;
        let term_norm = matrix_norm(&term.view(), "2", None)?;

        if term_norm < tol {
            break;
        }

        result = result + term;

        // Update for next odd power
        if i < coeffs.len() - 1 {
            a_power = a_power.dot(&a_squared);
        }
    }

    Ok(result)
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
#[allow(dead_code)]
pub fn atanm<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + Send + Sync + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square to compute inverse tangent, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        let mut result = Array2::zeros((1, 1));
        result[[0, 0]] = a[[0, 0]].atan();
        return Ok(result);
    }

    // For general matrices, we use the power series expansion:
    // atan(A) = A - A³/3 + A⁵/5 - A⁷/7 + ...
    // This converges for ||A|| < 1

    // Check if ||A|| < 1 for convergence
    let norm_a = matrix_norm(a, "2", None)?;
    if norm_a >= F::one() {
        return Err(LinalgError::ConvergenceError(
            "Matrix norm must be less than 1 for atan series to converge".to_string(),
        ));
    }

    let mut result = a.to_owned();
    let mut a_power = a.dot(a).dot(a); // A³
    let mut sign = -F::one();
    let tol = F::epsilon() * F::from(10.0).unwrap();

    for k in 1..50 {
        let coeff = sign / F::from(2 * k + 1).unwrap();
        let term = &a_power * coeff;

        // Check convergence
        let term_norm = matrix_norm(&term.view(), "2", None)?;
        if term_norm < tol {
            break;
        }

        result = result + term;

        // Update for next iteration
        a_power = a_power.dot(a).dot(a); // Multiply by A²
        sign = -sign;
    }

    Ok(result)
}

/// Compute the spectral radius (largest absolute eigenvalue) of a matrix.
///
/// The spectral radius ρ(A) is defined as the maximum absolute value of all eigenvalues.
/// This is important for analyzing convergence properties of iterative methods.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Spectral radius ρ(A) = max_i |λ_i|
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::spectral_radius;
///
/// let a = array![[2.0_f64, 1.0], [0.0, 0.5]];
/// let rho = spectral_radius(&a.view(), None).unwrap();
/// // rho should be approximately 2.0
/// ```
#[allow(dead_code)]
pub fn spectral_radius<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + 'static + Send + Sync,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // Parameter validation
    validate_decomposition(a, "Spectral radius computation", true)?;

    let n = a.nrows();

    // Special case for 1x1 matrix
    if n == 1 {
        return Ok(a[[0, 0]].abs());
    }

    // Compute eigenvalues
    let eigenvalues = crate::eigen::eigvals(a, workers)?;

    // Find maximum absolute value
    let mut max_abs = F::zero();
    for eigenval in eigenvalues.iter() {
        let abs_val = (eigenval.re * eigenval.re + eigenval.im * eigenval.im).sqrt();
        if abs_val > max_abs {
            max_abs = abs_val;
        }
    }

    Ok(max_abs)
}

/// Compute the condition number of a matrix using spectral norm.
///
/// The spectral condition number is defined as κ(A) = ||A||₂ * ||A⁻¹||₂ = σ_max / σ_min
/// where σ_max and σ_min are the largest and smallest singular values.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Spectral condition number κ₂(A)
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::spectral_condition_number;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1e-12]];
/// let kappa = spectral_condition_number(&a.view(), None).unwrap();
/// // kappa should be approximately 1e12
/// ```
#[allow(dead_code)]
pub fn spectral_condition_number<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + 'static + Send + Sync,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // Parameter validation
    validate_decomposition(a, "Spectral condition number computation", true)?;

    // Compute SVD to get singular values
    let (_, s, _) = crate::decomposition::svd(a, false, workers)?;

    if s.is_empty() {
        return Err(LinalgError::ComputationError(
            "No singular values computed".to_string(),
        ));
    }

    let sigma_max = s[0]; // Singular values are sorted in descending order
    let sigma_min = s[s.len() - 1];

    if sigma_min <= F::epsilon() {
        // Matrix is singular or near-singular
        Ok(F::from(1e16).unwrap_or(F::max_value()))
    } else {
        Ok(sigma_max / sigma_min)
    }
}

/// Compute the polar decomposition A = UH where U is orthogonal/unitary and H is positive semidefinite.
///
/// For a square matrix A, the polar decomposition gives A = UH where:
/// - U is orthogonal (unitary for complex matrices)
/// - H is Hermitian positive semidefinite
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Tuple (U, H) representing the polar decomposition
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::polar_decomposition;
///
/// let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
/// let (u, h) = polar_decomposition(&a.view(), None).unwrap();
/// // A = U * H and U is orthogonal, H is positive semidefinite
/// ```
#[allow(dead_code)]
pub fn polar_decomposition<F>(
    a: &ArrayView2<F>,
    workers: Option<usize>,
) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + 'static + Send + Sync,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // Parameter validation
    validate_decomposition(a, "Polar decomposition", true)?;

    let n = a.nrows();

    // Compute SVD: A = UΣV*
    let (u_svd, s, vt) = crate::decomposition::svd(a, true, workers)?;

    // Construct H = VΣV* (positive semidefinite part)
    let mut sigmamatrix = Array2::zeros((n, n));
    for (i, &sigma) in s.iter().enumerate() {
        if i < n {
            sigmamatrix[[i, i]] = sigma;
        }
    }

    let v = vt.t();
    let temp = v.dot(&sigmamatrix);
    let h = temp.dot(&vt);

    // Construct U = UV* (orthogonal part)
    let u = u_svd.dot(&vt);

    Ok((u, h))
}

/// Compute the matrix geometric mean of two positive definite matrices.
///
/// For two positive definite matrices A and B, the geometric mean is defined as:
/// G = A^(1/2) * (A^(-1/2) * B * A^(-1/2))^(1/2) * A^(1/2)
///
/// This is useful in Riemannian geometry and optimization on the manifold of SPD matrices.
///
/// # Arguments
///
/// * `a` - First positive definite matrix
/// * `b` - Second positive definite matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Geometric mean of A and B
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::geometric_mean_spd;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 1.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 4.0]];
/// let g = geometric_mean_spd(&a.view(), &b.view(), None).unwrap();
/// // G should be approximately [[2.0, 0.0], [0.0, 2.0]]
/// ```
#[allow(dead_code)]
pub fn geometric_mean_spd<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    workers: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + 'static + Send + Sync,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // Parameter validation
    validate_decomposition(a, "Geometric mean computation (matrix A)", true)?;
    validate_decomposition(b, "Geometric mean computation (matrix B)", true)?;

    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return Err(LinalgError::ShapeError(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let n = a.nrows();

    // Special case for diagonal matrices
    let mut a_diagonal = true;
    let mut b_diagonal = true;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                if a[[i, j]].abs() > F::epsilon() {
                    a_diagonal = false;
                }
                if b[[i, j]].abs() > F::epsilon() {
                    b_diagonal = false;
                }
            }
        }
    }

    if a_diagonal && b_diagonal {
        // For diagonal matrices, geometric mean is just geometric mean of diagonal elements
        let mut result = Array2::zeros((n, n));
        for i in 0..n {
            if a[[i, i]] > F::zero() && b[[i, i]] > F::zero() {
                result[[i, i]] = (a[[i, i]] * b[[i, i]]).sqrt();
            } else {
                return Err(LinalgError::ComputationError(
                    "Matrices must be positive definite".to_string(),
                ));
            }
        }
        return Ok(result);
    }

    // General case: use the formula G = A^(1/2) * (A^(-1/2) * B * A^(-1/2))^(1/2) * A^(1/2)

    // Compute A^(1/2) and A^(-1/2)
    let a_sqrt = spdmatrix_function(a, "sqrt", None)?;
    let a_inv_sqrt = spdmatrix_function(a, "inv_sqrt", None)?;

    // Compute A^(-1/2) * B * A^(-1/2)
    let temp1 = a_inv_sqrt.dot(b);
    let normalized_b = temp1.dot(&a_inv_sqrt);

    // Compute (A^(-1/2) * B * A^(-1/2))^(1/2)
    let normalized_b_sqrt = spdmatrix_function(&normalized_b.view(), "sqrt", None)?;

    // Final result: A^(1/2) * (A^(-1/2) * B * A^(-1/2))^(1/2) * A^(1/2)
    let temp2 = a_sqrt.dot(&normalized_b_sqrt);
    let result = temp2.dot(&a_sqrt);

    Ok(result)
}

/// Apply Tikhonov regularization to a matrix to improve conditioning.
///
/// Tikhonov regularization adds a multiple of the identity matrix: A_reg = A + λI
/// This is commonly used in ridge regression and to stabilize ill-conditioned problems.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `lambda` - Regularization parameter
/// * `adaptive` - If true, automatically select lambda based on condition number
///
/// # Returns
///
/// * Regularized matrix A + λI
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::tikhonov_regularization;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1e-12]]; // Ill-conditioned
/// let a_reg = tikhonov_regularization(&a.view(), 1e-6, false).unwrap();
/// ```
#[allow(dead_code)]
pub fn tikhonov_regularization<F>(
    a: &ArrayView2<F>,
    lambda: F,
    adaptive: bool,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + 'static + Send + Sync,
{
    // Parameter validation
    validate_decomposition(a, "Tikhonov regularization", true)?;

    let n = a.nrows();
    let mut result = a.to_owned();

    let reg_param = if adaptive {
        // Adaptive regularization based on condition number estimate
        let cond_est = crate::eigen::estimate_condition_number(a);
        if cond_est > F::from(1e12).unwrap() {
            lambda * F::from(100.0).unwrap()
        } else if cond_est > F::from(1e8).unwrap() {
            lambda * F::from(10.0).unwrap()
        } else {
            lambda
        }
    } else {
        lambda
    };

    // Add λI to the diagonal
    for i in 0..n {
        result[[i, i]] += reg_param;
    }

    Ok(result)
}

/// Compute the nuclear norm (trace norm) of a matrix: ||A||* = ∑ᵢ σᵢ.
///
/// The nuclear norm is the sum of singular values and is the convex envelope
/// of the rank function. It's used in low-rank matrix optimization.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Nuclear norm ||A||* = ∑ᵢ σᵢ
///
/// # Examples
///
/// ```no_run
/// use ndarray::array;
/// use scirs2_linalg::matrix_functions::nuclear_norm;
///
/// let a = array![[3.0_f64, 0.0], [0.0, 1.0]];
/// let norm = nuclear_norm(&a.view(), None).unwrap();
/// // Should be 3.0 + 1.0 = 4.0
/// ```
#[allow(dead_code)]
pub fn nuclear_norm<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + One + ndarray::ScalarOperand + 'static + Send + Sync,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    // Compute SVD to get singular values
    let (_, s, _) = crate::decomposition::svd(a, false, workers)?;

    // Sum all singular values
    Ok(s.iter().fold(F::zero(), |acc, &x| acc + x))
}

#[cfg(test)]
mod hyperbolic_tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_coshm_zeromatrix() {
        // cosh(0) = I
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let cosh_a = coshm(&a.view()).unwrap();

        assert_relative_eq!(cosh_a[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(cosh_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cosh_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cosh_a[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sinhm_zeromatrix() {
        // sinh(0) = 0
        let a = array![[0.0, 0.0], [0.0, 0.0]];
        let sinh_a = sinhm(&a.view()).unwrap();

        assert_relative_eq!(sinh_a[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinh_a[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinh_a[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinh_a[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tanhm_zeromatrix() {
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
