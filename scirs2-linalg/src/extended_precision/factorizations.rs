//! Extended precision matrix factorizations
//!
//! This module provides functions for computing matrix factorizations
//! using extended precision arithmetic for improved accuracy.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, One, Zero};

use super::{DemotableTo, PromotableTo};
use crate::error::LinalgResult;

/// LU decomposition with partial pivoting using extended precision
///
/// This function computes the LU decomposition of a matrix using higher precision
/// arithmetic for improved numerical stability.
///
/// # Parameters
///
/// * `a` - Input matrix
///
/// # Returns
///
/// * Tuple containing (P, L, U) where:
///   - P is the permutation matrix
///   - L is the lower triangular matrix with unit diagonal
///   - U is the upper triangular matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ArrayView2};
/// use scirs2_linalg::extended_precision::factorizations::extended_lu;
///
/// let a = array![
///     [2.0_f32, 1.0, 1.0],
///     [4.0, 3.0, 3.0],
///     [8.0, 7.0, 9.0]
/// ];
///
/// // Compute LU decomposition with extended precision
/// let (p, l, u) = extended_lu::<_, f64>(&a.view()).unwrap();
///
/// // Check result: P*A ≈ L*U
/// let pa = p.dot(&a);
/// let lu = l.dot(&u);
/// for i in 0..a.nrows() {
///     for j in 0..a.ncols() {
///         assert!((pa[[i, j]] - lu[[i, j]]).abs() < 1e-5);
///     }
/// }
/// ```
#[allow(dead_code)]
pub fn extended_lu<A, I>(a: &ArrayView2<A>) -> LinalgResult<(Array2<A>, Array2<A>, Array2<A>)>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + PartialOrd
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign,
{
    let m = a.nrows();
    let n = a.ncols();

    // Convert matrix to higher precision
    let mut a_high = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Initialize permutation vector for pivoting
    let mut p_indices = Vec::with_capacity(m);
    for i in 0..m {
        p_indices.push(i);
    }

    // Perform LU decomposition with partial pivoting in high precision
    for k in 0..std::cmp::min(m, n) {
        // Find pivot
        let mut pivot_row = k;
        let mut max_val = a_high[[k, k]].abs();

        for i in k + 1..m {
            let val = a_high[[i, k]].abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }

        // Check for singularity with a more relaxed threshold for ill-conditioned matrices
        // Use a threshold that is much smaller than machine epsilon to handle ill-conditioned cases
        let threshold =
            I::from(1e-30_f64).unwrap_or(I::epsilon() * I::from(1e-10).unwrap_or(I::epsilon()));
        if max_val < threshold {
            return Err(crate::error::LinalgError::SingularMatrixError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows in a_high and p_indices if necessary
        if pivot_row != k {
            p_indices.swap(k, pivot_row);
            for j in 0..n {
                let temp = a_high[[k, j]];
                a_high[[k, j]] = a_high[[pivot_row, j]];
                a_high[[pivot_row, j]] = temp;
            }
        }

        // Compute the Gauss transform
        for i in k + 1..m {
            a_high[[i, k]] = a_high[[i, k]] / a_high[[k, k]];

            for j in k + 1..n {
                a_high[[i, j]] = a_high[[i, j]] - a_high[[i, k]] * a_high[[k, j]];
            }
        }
    }

    // Extract L and U from a_high (still in high precision)
    let mut l_high = Array2::zeros((m, std::cmp::min(m, n)));
    let mut u_high = Array2::zeros((std::cmp::min(m, n), n));

    for i in 0..m {
        for j in 0..std::cmp::min(m, n) {
            match i.cmp(&j) {
                std::cmp::Ordering::Greater => l_high[[i, j]] = a_high[[i, j]],
                std::cmp::Ordering::Equal => l_high[[i, j]] = I::one(),
                std::cmp::Ordering::Less => {} // Do nothing
            }
        }
    }

    for i in 0..std::cmp::min(m, n) {
        for j in 0..n {
            if i <= j {
                u_high[[i, j]] = a_high[[i, j]];
            }
        }
    }

    // Build permutation matrix
    let mut p_high = Array2::zeros((m, m));
    for i in 0..m {
        p_high[[i, p_indices[i]]] = I::one();
    }

    // Convert results back to original precision
    let mut p = Array2::zeros((m, m));
    let mut l = Array2::zeros((m, std::cmp::min(m, n)));
    let mut u = Array2::zeros((std::cmp::min(m, n), n));

    for i in 0..m {
        for j in 0..m {
            p[[i, j]] = p_high[[i, j]].demote();
        }
    }

    for i in 0..m {
        for j in 0..std::cmp::min(m, n) {
            l[[i, j]] = l_high[[i, j]].demote();
        }
    }

    for i in 0..std::cmp::min(m, n) {
        for j in 0..n {
            u[[i, j]] = u_high[[i, j]].demote();
        }
    }

    Ok((p, l, u))
}

/// Internal LU decomposition that returns high precision matrices
/// Used internally for determinant calculations
#[allow(dead_code)]
pub(super) fn extended_lu_internal<A, I>(
    a: &ArrayView2<A>,
) -> LinalgResult<(Array2<I>, Array2<I>, Array2<I>)>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + PartialOrd
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign,
{
    let m = a.nrows();
    let n = a.ncols();

    // Convert matrix to higher precision
    let mut a_high = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Initialize permutation vector for pivoting
    let mut p_indices = Vec::with_capacity(m);
    for i in 0..m {
        p_indices.push(i);
    }

    // Perform LU decomposition with partial pivoting in high precision
    for k in 0..std::cmp::min(m, n) {
        // Find pivot
        let mut pivot_row = k;
        let mut max_val = a_high[[k, k]].abs();

        for i in k + 1..m {
            let val = a_high[[i, k]].abs();
            if val > max_val {
                max_val = val;
                pivot_row = i;
            }
        }

        // Check for singularity with a more relaxed threshold for ill-conditioned matrices
        let threshold =
            I::from(1e-30_f64).unwrap_or(I::epsilon() * I::from(1e-10).unwrap_or(I::epsilon()));
        if max_val < threshold {
            return Err(crate::error::LinalgError::SingularMatrixError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows in a_high and p_indices if necessary
        if pivot_row != k {
            p_indices.swap(k, pivot_row);
            for j in 0..n {
                let temp = a_high[[k, j]];
                a_high[[k, j]] = a_high[[pivot_row, j]];
                a_high[[pivot_row, j]] = temp;
            }
        }

        // Compute the Gauss transform
        for i in k + 1..m {
            a_high[[i, k]] = a_high[[i, k]] / a_high[[k, k]];

            for j in k + 1..n {
                a_high[[i, j]] = a_high[[i, j]] - a_high[[i, k]] * a_high[[k, j]];
            }
        }
    }

    // Extract L and U from a_high (keep in high precision)
    let mut l_high = Array2::zeros((m, std::cmp::min(m, n)));
    let mut u_high = Array2::zeros((std::cmp::min(m, n), n));

    for i in 0..m {
        for j in 0..std::cmp::min(m, n) {
            match i.cmp(&j) {
                std::cmp::Ordering::Greater => l_high[[i, j]] = a_high[[i, j]],
                std::cmp::Ordering::Equal => l_high[[i, j]] = I::one(),
                std::cmp::Ordering::Less => {} // Do nothing
            }
        }
    }

    for i in 0..std::cmp::min(m, n) {
        for j in 0..n {
            if i <= j {
                u_high[[i, j]] = a_high[[i, j]];
            }
        }
    }

    // Build permutation matrix in high precision
    let mut p_high = Array2::zeros((m, m));
    for i in 0..m {
        p_high[[i, p_indices[i]]] = I::one();
    }

    Ok((p_high, l_high, u_high))
}

/// QR decomposition using extended precision
///
/// This function computes the QR decomposition of a matrix using higher precision
/// arithmetic for improved numerical stability.
///
/// # Parameters
///
/// * `a` - Input matrix
///
/// # Returns
///
/// * Tuple containing (Q, R) where:
///   - Q is the orthogonal matrix
///   - R is the upper triangular matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ArrayView2};
/// use scirs2_linalg::extended_precision::factorizations::extended_qr;
///
/// let a = array![
///     [12.0_f32, -51.0, 4.0],
///     [6.0, 167.0, -68.0],
///     [-4.0, 24.0, -41.0]
/// ];
///
/// // Compute QR decomposition with extended precision
/// let (q, r) = extended_qr::<_, f64>(&a.view()).unwrap();
///
/// // Check orthogonality of Q (Q^T * Q ≈ I)
/// let qt = q.t();
/// let qtq = qt.dot(&q);
/// for i in 0..q.nrows() {
///     for j in 0..q.ncols() {
///         if i == j {
///             assert!((qtq[[i, j]] - 1.0).abs() < 1e-5);
///         } else {
///             assert!(qtq[[i, j]].abs() < 1e-5);
///         }
///     }
/// }
///
/// // Check that A = Q*R
/// let qr = q.dot(&r);
/// for i in 0..a.nrows() {
///     for j in 0..a.ncols() {
///         assert!((a[[i, j]] - qr[[i, j]]).abs() < 1e-5);
///     }
/// }
/// ```
#[allow(dead_code)]
pub fn extended_qr<A, I>(a: &ArrayView2<A>) -> LinalgResult<(Array2<A>, Array2<A>)>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + PartialOrd
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign,
{
    let m = a.nrows();
    let n = a.ncols();

    // Convert matrix to higher precision
    let mut a_high = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Initialize Q as identity matrix
    let mut q_high = Array2::zeros((m, m));
    for i in 0..m {
        q_high[[i, i]] = I::one();
    }

    // Apply Householder reflections
    for k in 0..std::cmp::min(m - 1, n) {
        // Extract column vector
        let mut x = Array1::zeros(m - k);
        for i in k..m {
            x[i - k] = a_high[[i, k]];
        }

        // Compute Householder vector
        let norm_x = x.iter().map(|&val| val * val).sum::<I>().sqrt();
        let mut v = x.clone();

        let sign = if v[0] >= I::zero() {
            I::one()
        } else {
            -I::one()
        };
        v[0] += sign * norm_x;

        let norm_v = v.iter().map(|&val| val * val).sum::<I>().sqrt();

        // Normalize v if it's not zero
        if norm_v > I::epsilon() {
            for i in 0..v.len() {
                v[i] = v[i] / norm_v;
            }

            // Apply Householder reflection to A
            for j in k..n {
                let mut dot_product = I::zero();
                for i in 0..v.len() {
                    dot_product += v[i] * a_high[[i + k, j]];
                }

                for i in 0..v.len() {
                    a_high[[i + k, j]] -= I::from(2.0).unwrap() * dot_product * v[i];
                }
            }

            // Apply Householder reflection to Q
            for j in 0..m {
                let mut dot_product = I::zero();
                for i in 0..v.len() {
                    dot_product += v[i] * q_high[[i + k, j]];
                }

                for i in 0..v.len() {
                    q_high[[i + k, j]] -= I::from(2.0).unwrap() * dot_product * v[i];
                }
            }
        }
    }

    // Transpose Q to get the orthogonal matrix (since we've been applying reflections from the right)
    let mut q_high_t = Array2::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            q_high_t[[i, j]] = q_high[[j, i]];
        }
    }

    // Zero out the lower triangular part of A to get R
    let mut r_high = a_high.clone();
    for i in 0..m {
        for j in 0..std::cmp::min(i, n) {
            r_high[[i, j]] = I::zero();
        }
    }

    // Convert results back to original precision
    let mut q = Array2::zeros((m, m));
    let mut r = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..m {
            q[[i, j]] = q_high_t[[i, j]].demote();
        }
    }

    for i in 0..m {
        for j in 0..n {
            r[[i, j]] = r_high[[i, j]].demote();
        }
    }

    Ok((q, r))
}

/// Cholesky decomposition using extended precision
///
/// This function computes the Cholesky decomposition of a symmetric positive definite matrix
/// using higher precision arithmetic for improved numerical stability.
///
/// # Parameters
///
/// * `a` - Input symmetric positive definite matrix
///
/// # Returns
///
/// * Lower triangular matrix L such that A = L * L^T
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ArrayView2};
/// use scirs2_linalg::extended_precision::factorizations::extended_cholesky;
///
/// let a = array![
///     [4.0_f32, 12.0, -16.0],
///     [12.0, 37.0, -43.0],
///     [-16.0, -43.0, 98.0]
/// ];
///
/// // Compute Cholesky decomposition with extended precision
/// let l = extended_cholesky::<_, f64>(&a.view()).unwrap();
///
/// // Check result: A ≈ L*L^T
/// let lt = l.t();
/// let llt = l.dot(&lt);
/// for i in 0..a.nrows() {
///     for j in 0..a.ncols() {
///         assert!((a[[i, j]] - llt[[i, j]]).abs() < 1e-5);
///     }
/// }
/// ```
#[allow(dead_code)]
pub fn extended_cholesky<A, I>(a: &ArrayView2<A>) -> LinalgResult<Array2<A>>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + PartialOrd
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign,
{
    if a.nrows() != a.ncols() {
        return Err(crate::error::LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    // Check symmetry
    let n = a.nrows();
    for i in 0..n {
        for j in i + 1..n {
            if (a[[i, j]] - a[[j, i]]).abs() > A::epsilon() * A::from(10.0).unwrap() {
                return Err(crate::error::LinalgError::InvalidInputError(
                    "Matrix must be symmetric".to_string(),
                ));
            }
        }
    }

    // Convert matrix to higher precision
    let mut a_high = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Initialize the result matrix
    let mut l_high = Array2::zeros((n, n));

    // Compute Cholesky decomposition
    for j in 0..n {
        let mut d = a_high[[j, j]];

        for k in 0..j {
            d -= l_high[[j, k]] * l_high[[j, k]];
        }

        if d <= I::zero() {
            return Err(crate::error::LinalgError::InvalidInputError(
                "Matrix is not positive definite".to_string(),
            ));
        }

        l_high[[j, j]] = d.sqrt();

        for i in j + 1..n {
            let mut s = a_high[[i, j]];

            for k in 0..j {
                s -= l_high[[i, k]] * l_high[[j, k]];
            }

            l_high[[i, j]] = s / l_high[[j, j]];
        }
    }

    // Convert result back to original precision
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            // Only copy the lower triangular part
            l[[i, j]] = l_high[[i, j]].demote();
        }
    }

    Ok(l)
}

/// Singular Value Decomposition (SVD) using extended precision
///
/// This function computes the SVD of a matrix using higher precision
/// arithmetic for improved numerical stability.
///
/// # Parameters
///
/// * `a` - Input matrix
/// * `full_matrices` - If true, returns full U and Vh matrices; if false, returns reduced matrices
/// * `max_iter` - Maximum number of iterations (default: 100)
/// * `tol` - Convergence tolerance (default: 1e-8 for the working precision)
///
/// # Returns
///
/// * Tuple containing (U, S, Vh) where:
///   - U is the left singular vectors
///   - S is the singular values
///   - Vh is the right singular vectors (transposed)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ArrayView2};
/// use scirs2_linalg::extended_precision::factorizations::extended_svd;
///
/// let a = array![
///     [1.0_f32, 0.0, 0.0],
///     [0.0, 0.5, 0.0],
///     [0.0, 0.0, 0.2]
/// ];
///
/// // Compute SVD with extended precision
/// let (u, s, vh) = extended_svd::<_, f64>(&a.view(), true, None, None).unwrap();
///
/// // Check that singular values are correct
/// assert!((s[0] - 1.0).abs() < 1e-5);
/// assert!((s[1] - 0.5).abs() < 1e-5);
/// assert!((s[2] - 0.2).abs() < 1e-5);
/// ```
#[allow(dead_code)]
pub fn extended_svd<A, I>(
    a: &ArrayView2<A>,
    full_matrices: bool,
    max_iter: Option<usize>,
    tol: Option<A>,
) -> LinalgResult<(Array2<A>, Array1<A>, Array2<A>)>
where
    A: Float + Zero + One + PromotableTo<I> + DemotableTo<A> + Copy,
    I: Float
        + Zero
        + One
        + DemotableTo<A>
        + Copy
        + PartialOrd
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign
        + 'static,
{
    // For this example, we'll implement a basic SVD using QR iteration
    // In practice, more advanced algorithms like divide-and-conquer would be used

    let m = a.nrows();
    let n = a.ncols();
    let k = std::cmp::min(m, n);

    let max_iter = max_iter.unwrap_or(100 * k);
    let tol = tol.unwrap_or(A::epsilon().sqrt());

    // Convert matrix to higher precision
    let mut a_high = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a_high[[i, j]] = a[[i, j]].promote();
        }
    }

    // Compute A^T * A (for n ≤ m) or A * A^T (for m < n)
    let mut ata = if n <= m {
        let mut result = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for l in 0..m {
                    result[[i, j]] += a_high[[l, i]] * a_high[[l, j]];
                }
            }
        }
        result
    } else {
        let mut result = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                for l in 0..n {
                    result[[i, j]] += a_high[[i, l]] * a_high[[j, l]];
                }
            }
        }
        result
    };

    // Apply QR algorithm to find eigenvalues and eigenvectors of A^T * A or A * A^T
    // This is a simplified approach; a full implementation would use a more sophisticated algorithm

    // Initialize eigenvectors as identity matrix
    let mut v_high = Array2::zeros((if n <= m { n } else { m }, if n <= m { n } else { m }));
    for i in 0..v_high.nrows() {
        v_high[[i, i]] = I::one();
    }

    for _ in 0..max_iter {
        // QR decomposition of current matrix
        let (q, r) = householder_qr_high_precision(&ata.view());

        // Update matrix: A = R * Q
        ata = r.dot(&q);

        // Update eigenvectors: V = V * Q
        v_high = v_high.dot(&q);

        // Check convergence (off-diagonal elements approaching zero)
        let mut converged = true;
        for i in 0..ata.nrows() {
            for j in 0..ata.ncols() {
                if i != j && ata[[i, j]].abs() > I::from(tol.promote()).unwrap() {
                    converged = false;
                    break;
                }
            }
            if !converged {
                break;
            }
        }

        if converged {
            break;
        }
    }

    // Extract singular values from diagonal of converged matrix
    let mut s_high = Array1::zeros(k);
    for i in 0..k {
        s_high[i] = ata[[i, i]].sqrt();
    }

    // Sort singular values and corresponding vectors
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&i, &j| s_high[j].partial_cmp(&s_high[i]).unwrap());

    let mut sorted_s_high = Array1::zeros(k);
    let mut sorted_v_high = Array2::zeros((if n <= m { n } else { m }, k));

    for (idx, &i) in indices.iter().enumerate() {
        sorted_s_high[idx] = s_high[i];
        for j in 0..v_high.nrows() {
            sorted_v_high[[j, idx]] = v_high[[j, i]];
        }
    }

    // Compute U from V and singular values
    let mut u_high = if n <= m {
        let mut u = Array2::zeros((m, k));
        for j in 0..k {
            if sorted_s_high[j] > I::epsilon() {
                for i in 0..m {
                    for l in 0..n {
                        u[[i, j]] += a_high[[i, l]] * sorted_v_high[[l, j]] / sorted_s_high[j];
                    }
                }
            }
        }
        u
    } else {
        sorted_v_high.clone()
    };

    let mut vh_high = if n <= m {
        sorted_v_high.clone()
    } else {
        let mut vh = Array2::zeros((k, n));
        for j in 0..k {
            if sorted_s_high[j] > I::epsilon() {
                for i in 0..n {
                    for l in 0..m {
                        vh[[j, i]] += a_high[[l, i]] * sorted_v_high[[l, j]] / sorted_s_high[j];
                    }
                }
            }
        }
        vh
    };

    // If full_matrices is true, pad U and Vh appropriately
    if full_matrices {
        let mut u_full = Array2::zeros((m, m));
        let mut vh_full = Array2::zeros((n, n));

        for i in 0..m {
            for j in 0..std::cmp::min(m, k) {
                u_full[[i, j]] = u_high[[i, j]];
            }
        }

        for i in 0..std::cmp::min(n, k) {
            for j in 0..n {
                vh_full[[i, j]] = vh_high[[i, j]];
            }
        }

        // Orthogonalize remaining columns
        if m > k {
            // Orthogonalize remaining columns of U
            for j in k..m {
                let mut v = Array1::zeros(m);
                v[j] = I::one();

                for l in 0..j {
                    let mut dot_prod = I::zero();
                    for i in 0..m {
                        dot_prod += u_full[[i, l]] * v[i];
                    }

                    for i in 0..m {
                        v[i] -= dot_prod * u_full[[i, l]];
                    }
                }

                let norm = v.iter().map(|&x| x * x).sum::<I>().sqrt();
                if norm > I::epsilon() {
                    for i in 0..m {
                        u_full[[i, j]] = v[i] / norm;
                    }
                }
            }
        }

        if n > k {
            // Orthogonalize remaining rows of Vh
            for i in k..n {
                let mut v = Array1::zeros(n);
                v[i] = I::one();

                for l in 0..i {
                    let mut dot_prod = I::zero();
                    for j in 0..n {
                        dot_prod += vh_full[[l, j]] * v[j];
                    }

                    for j in 0..n {
                        v[j] -= dot_prod * vh_full[[l, j]];
                    }
                }

                let norm = v.iter().map(|&x| x * x).sum::<I>().sqrt();
                if norm > I::epsilon() {
                    for j in 0..n {
                        vh_full[[i, j]] = v[j] / norm;
                    }
                }
            }
        }

        u_high = u_full;
        vh_high = vh_full;
    }

    // Convert results back to original precision
    let mut u = Array2::zeros(u_high.dim());
    let mut s = Array1::zeros(k);
    let mut vh = Array2::zeros(vh_high.dim());

    for i in 0..u_high.nrows() {
        for j in 0..u_high.ncols() {
            u[[i, j]] = u_high[[i, j]].demote();
        }
    }

    for i in 0..k {
        s[i] = sorted_s_high[i].demote();
    }

    for i in 0..vh_high.nrows() {
        for j in 0..vh_high.ncols() {
            vh[[i, j]] = vh_high[[i, j]].demote();
        }
    }

    Ok((u, s, vh))
}

// Helper function for QR decomposition using Householder reflections
#[allow(dead_code)]
fn householder_qr_high_precision<I>(a: &ArrayView2<I>) -> (Array2<I>, Array2<I>)
where
    I: Float
        + Zero
        + One
        + Copy
        + PartialOrd
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign,
{
    let m = a.nrows();
    let n = a.ncols();

    let mut q = Array2::eye(m);
    let mut r = a.to_owned();

    for k in 0..std::cmp::min(m - 1, n) {
        // Extract column vector
        let mut x = Array1::zeros(m - k);
        for i in k..m {
            x[i - k] = r[[i, k]];
        }

        // Compute Householder vector
        let norm_x = x.iter().map(|&val| val * val).sum::<I>().sqrt();
        let mut v = x.clone();

        let sign = if v[0] >= I::zero() {
            I::one()
        } else {
            -I::one()
        };
        v[0] += sign * norm_x;

        let norm_v = v.iter().map(|&val| val * val).sum::<I>().sqrt();

        // Normalize v if it's not zero
        if norm_v > I::epsilon() {
            for i in 0..v.len() {
                v[i] = v[i] / norm_v;
            }

            // Apply Householder reflection to R
            for j in k..n {
                let mut dot_product = I::zero();
                for i in 0..v.len() {
                    dot_product += v[i] * r[[i + k, j]];
                }

                for i in 0..v.len() {
                    r[[i + k, j]] -= I::from(2.0).unwrap() * dot_product * v[i];
                }
            }

            // Apply Householder reflection to Q
            for j in 0..m {
                let mut dot_product = I::zero();
                for i in 0..v.len() {
                    dot_product += v[i] * q[[j, i + k]];
                }

                for i in 0..v.len() {
                    q[[j, i + k]] -= I::from(2.0).unwrap() * dot_product * v[i];
                }
            }
        }
    }

    // Zero out the lower triangular part of R
    for i in 1..m {
        for j in 0..std::cmp::min(i, n) {
            r[[i, j]] = I::zero();
        }
    }

    // Transpose Q to get the orthogonal matrix
    let q_t = q.t().to_owned();

    (q_t, r)
}
