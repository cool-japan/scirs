//! Matrix analysis functions including decompositions and matrix norms

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One};
use std::iter::Sum;

use crate::eigen::eig;
use crate::error::{LinalgError, LinalgResult};
use crate::solve::solve_multiple;
use crate::validation::validate_decomposition;

/// Compute the spectral radius of a matrix.
///
/// The spectral radius is the largest absolute value of the eigenvalues.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Spectral radius (largest eigenvalue magnitude)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::spectral_radius;
///
/// let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
/// let rho = spectral_radius(&a.view(), None).unwrap();
/// // rho should be 3.0
/// ```
#[allow(dead_code)]
pub fn spectral_radius<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    validate_decomposition(a, "Spectral radius computation", true)?;

    // Compute eigenvalues
    let (eigenvals, _) = eig(a, None)?;

    // Find the maximum absolute value
    let mut max_abs = F::zero();
    for &val in eigenvals.iter() {
        let abs_val = (val.re * val.re + val.im * val.im).sqrt();
        if abs_val > max_abs {
            max_abs = abs_val;
        }
    }

    Ok(max_abs)
}

/// Compute the spectral condition number of a matrix.
///
/// The spectral condition number is the ratio of the largest to smallest
/// singular values (or eigenvalues for symmetric matrices).
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Spectral condition number
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::spectral_condition_number;
///
/// let a = array![[2.0_f64, 0.0], [0.0, 1.0]];
/// let cond = spectral_condition_number(&a.view(), None).unwrap();
/// // cond should be 2.0
/// ```
#[allow(dead_code)]
pub fn spectral_condition_number<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    validate_decomposition(a, "Spectral condition number computation", true)?;

    // Compute eigenvalues
    let (eigenvals, _) = eig(a, None)?;

    // Find the maximum and minimum absolute values
    let mut max_abs = F::zero();
    let mut min_abs = F::infinity();

    for &val in eigenvals.iter() {
        let abs_val = (val.re * val.re + val.im * val.im).sqrt();
        if abs_val > max_abs {
            max_abs = abs_val;
        }
        if abs_val < min_abs && abs_val > F::epsilon() {
            min_abs = abs_val;
        }
    }

    // Check for singular matrix
    if min_abs < F::epsilon() {
        return Ok(F::infinity());
    }

    Ok(max_abs / min_abs)
}

/// Compute the polar decomposition of a matrix.
///
/// The polar decomposition factorizes a matrix A as A = UP where U is unitary
/// and P is positive semidefinite.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `side` - Which factor to return ("left" for A = UP, "right" for A = PU)
///
/// # Returns
///
/// * (U, P) - Unitary and positive semidefinite factors
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::polar_decomposition;
///
/// let a = array![[2.0_f64, 1.0], [0.0, 1.0]];
/// let (u, p) = polar_decomposition(&a.view(), "right").unwrap();
/// ```
#[allow(dead_code)]
pub fn polar_decomposition<F>(a: &ArrayView2<F>, side: &str) -> LinalgResult<(Array2<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    use super::exponential::sqrtm;

    validate_decomposition(a, "Polar decomposition", false)?;

    let (m, n) = a.dim();

    match side {
        "right" => {
            // Right polar decomposition: A = UP
            // P = sqrt(A^H * A), U = A * P^{-1}

            // Compute A^H * A
            let mut aha = Array2::<F>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    for k in 0..m {
                        aha[[i, j]] += a[[k, i]] * a[[k, j]]; // A^H * A
                    }
                }
            }

            // Compute P = sqrt(A^H * A)
            let p = sqrtm(&aha.view(), 50, F::from(1e-12).unwrap())?;

            // Compute U = A * P^{-1}
            let p_inv = solve_multiple(&p.view(), &Array2::eye(n).view(), None)?;

            let mut u = Array2::<F>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    for k in 0..n {
                        u[[i, j]] += a[[i, k]] * p_inv[[k, j]];
                    }
                }
            }

            Ok((u, p))
        }
        "left" => {
            // Left polar decomposition: A = PU
            // P = sqrt(A * A^H), U = P^{-1} * A

            // Compute A * A^H
            let mut aah = Array2::<F>::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    for k in 0..n {
                        aah[[i, j]] += a[[i, k]] * a[[j, k]]; // A * A^H
                    }
                }
            }

            // Compute P = sqrt(A * A^H)
            let p = sqrtm(&aah.view(), 50, F::from(1e-12).unwrap())?;

            // Compute U = P^{-1} * A
            let p_inv = solve_multiple(&p.view(), &Array2::eye(m).view(), None)?;

            let mut u = Array2::<F>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    for k in 0..m {
                        u[[i, j]] += p_inv[[i, k]] * a[[k, j]];
                    }
                }
            }

            Ok((p, u))
        }
        _ => Err(LinalgError::InvalidInputError(format!(
            "Invalid side '{}'. Must be 'left' or 'right'",
            side
        ))),
    }
}

/// Compute the geometric mean of symmetric positive definite matrices.
///
/// For two SPD matrices A and B, the geometric mean is:
/// A #_t B = A^{1/2} * (A^{-1/2} * B * A^{-1/2})^t * A^{1/2}
///
/// For t = 1/2, this gives the standard geometric mean.
///
/// # Arguments
///
/// * `a` - First SPD matrix
/// * `b` - Second SPD matrix
/// * `t` - Parameter (0.5 for standard geometric mean)
///
/// # Returns
///
/// * Geometric mean of A and B
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::geometric_mean_spd;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 1.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 4.0]];
/// let mean = geometric_mean_spd(&a.view(), &b.view(), 0.5).unwrap();
/// ```
#[allow(dead_code)]
pub fn geometric_mean_spd<F>(a: &ArrayView2<F>, b: &ArrayView2<F>, t: F) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    use super::exponential::sqrtm;
    use super::fractional::spdmatrix_function;

    validate_decomposition(a, "Geometric mean computation (matrix A)", true)?;
    validate_decomposition(b, "Geometric mean computation (matrix B)", true)?;

    if a.dim() != b.dim() {
        return Err(LinalgError::ShapeError(
            "Matrices must have the same dimensions".to_string(),
        ));
    }

    let n = a.nrows();

    // Check if matrices are symmetric
    for i in 0..n {
        for j in 0..n {
            if (a[[i, j]] - a[[j, i]]).abs() > F::epsilon() {
                return Err(LinalgError::InvalidInputError(
                    "Matrix A must be symmetric".to_string(),
                ));
            }
            if (b[[i, j]] - b[[j, i]]).abs() > F::epsilon() {
                return Err(LinalgError::InvalidInputError(
                    "Matrix B must be symmetric".to_string(),
                ));
            }
        }
    }

    // Compute A^{1/2}
    let a_half = spdmatrix_function(a, |x| x.sqrt(), true)?;

    // Compute A^{-1/2}
    let a_neg_half = spdmatrix_function(a, |x| F::one() / x.sqrt(), true)?;

    // Compute A^{-1/2} * B * A^{-1/2}
    let mut temp1 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                temp1[[i, j]] += a_neg_half[[i, k]] * b[[k, j]];
            }
        }
    }

    let mut similarity = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                similarity[[i, j]] += temp1[[i, k]] * a_neg_half[[k, j]];
            }
        }
    }

    // Compute (A^{-1/2} * B * A^{-1/2})^t
    let powered_similarity = spdmatrix_function(&similarity.view(), |x| x.powf(t), true)?;

    // Compute A^{1/2} * (A^{-1/2} * B * A^{-1/2})^t * A^{1/2}
    let mut temp2 = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                temp2[[i, j]] += a_half[[i, k]] * powered_similarity[[k, j]];
            }
        }
    }

    let mut result = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[[i, j]] += temp2[[i, k]] * a_half[[k, j]];
            }
        }
    }

    Ok(result)
}

/// Compute Tikhonov regularization of a matrix.
///
/// Tikhonov regularization adds a ridge term to improve conditioning:
/// A_reg = A + λI
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `lambda` - Regularization parameter
/// * `identity_like` - Whether to add λI (true) or λ times identity-like term
///
/// # Returns
///
/// * Regularized matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::tikhonov_regularization;
///
/// let a = array![[1.0_f64, 0.5], [0.5, 1.0]];
/// let reg_a = tikhonov_regularization(&a.view(), 0.1, true).unwrap();
/// ```
#[allow(dead_code)]
pub fn tikhonov_regularization<F>(
    a: &ArrayView2<F>,
    lambda: F,
    identity_like: bool,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    let (m, n) = a.dim();

    if identity_like && m != n {
        return Err(LinalgError::ShapeError(
            "Matrix must be square for identity-like regularization".to_string(),
        ));
    }

    let mut result = a.to_owned();

    if identity_like {
        // Add λI to the diagonal
        for i in 0..n {
            result[[i, i]] += lambda;
        }
    } else {
        // Add λ to all elements (different regularization scheme)
        for i in 0..m {
            for j in 0..n {
                result[[i, j]] += lambda;
            }
        }
    }

    Ok(result)
}

/// Compute the nuclear norm (trace norm) of a matrix.
///
/// The nuclear norm is the sum of singular values, which for symmetric matrices
/// is the sum of absolute eigenvalues.
///
/// # Arguments
///
/// * `a` - Input matrix
/// * `workers` - Number of worker threads (None = use default)
///
/// # Returns
///
/// * Nuclear norm of the matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::matrix_functions::nuclear_norm;
///
/// let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
/// let norm = nuclear_norm(&a.view(), None).unwrap();
/// // norm should be 5.0
/// ```
#[allow(dead_code)]
pub fn nuclear_norm<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<F>
where
    F: Float + NumAssign + Sum + One + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    validate_decomposition(a, "Nuclear norm computation", false)?;

    let (m, n) = a.dim();

    if m == n {
        // For square matrices, use eigendecomposition
        // This gives us the singular values for symmetric matrices
        let (eigenvals, _) = eig(a, None)?;

        let mut sum = F::zero();
        for &val in eigenvals.iter() {
            sum += (val.re * val.re + val.im * val.im).sqrt();
        }

        Ok(sum)
    } else {
        // For non-square matrices, we'd need SVD
        // For now, return an error
        Err(LinalgError::ImplementationError(
            "Nuclear norm for non-square matrices requires SVD (not yet implemented)".to_string(),
        ))
    }
}
