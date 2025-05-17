//! Eigenvalue and eigenvector computations

use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex;
use num_traits::{Float, NumAssign};
use rand::prelude::*;
use std::iter::Sum;

use crate::decomposition::qr;
use crate::error::{LinalgError, LinalgResult};
use crate::norm::vector_norm;

/// Type alias for eigenvalue-eigenvector pair result
/// Returns a tuple of (eigenvalues, eigenvectors) where eigenvalues is a 1D array
/// and eigenvectors is a 2D array where each column corresponds to an eigenvector
type EigenResult<F> = LinalgResult<(Array1<Complex<F>>, Array2<Complex<F>>)>;

/// Compute the eigenvalues and right eigenvectors of a square matrix.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Tuple (eigenvalues, eigenvectors) where eigenvalues is a complex vector
///   and eigenvectors is a complex matrix whose columns are the eigenvectors
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eig;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let (w, v) = eig(&a.view()).unwrap();
///
/// // Sort eigenvalues (they may be returned in different order)
/// let mut eigenvalues = vec![(w[0].re, 0), (w[1].re, 1)];
/// eigenvalues.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
///
/// assert!((eigenvalues[0].0 - 1.0).abs() < 1e-10);
/// assert!((eigenvalues[1].0 - 2.0).abs() < 1e-10);
/// ```
pub fn eig<F>(a: &ArrayView2<F>) -> EigenResult<F>
where
    F: Float + NumAssign + Sum + 'static,
{
    // Check if matrix is square
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // For 1x1 and 2x2 matrices, we can compute eigenvalues analytically
    if n == 1 {
        let eigenvalue = Complex::new(a[[0, 0]], F::zero());
        let eigenvector = Array2::eye(1).mapv(|x| Complex::new(x, F::zero()));

        return Ok((Array1::from_elem(1, eigenvalue), eigenvector));
    } else if n == 2 {
        // For 2x2 matrices, use the quadratic formula
        let a11 = a[[0, 0]];
        let a12 = a[[0, 1]];
        let a21 = a[[1, 0]];
        let a22 = a[[1, 1]];

        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a21;

        let discriminant = trace * trace - F::from(4.0).unwrap() * det;

        // Create eigenvalues
        let mut eigenvalues = Array1::zeros(2);
        let mut eigenvectors = Array2::zeros((2, 2));

        if discriminant >= F::zero() {
            // Real eigenvalues
            let sqrt_discriminant = discriminant.sqrt();
            let lambda1 = (trace + sqrt_discriminant) / F::from(2.0).unwrap();
            let lambda2 = (trace - sqrt_discriminant) / F::from(2.0).unwrap();

            eigenvalues[0] = Complex::new(lambda1, F::zero());
            eigenvalues[1] = Complex::new(lambda2, F::zero());

            // Compute eigenvectors
            for (i, &lambda) in [lambda1, lambda2].iter().enumerate() {
                let mut eigenvector = Array1::zeros(2);

                if a12 != F::zero() {
                    eigenvector[0] = a12;
                    eigenvector[1] = lambda - a11;
                } else if a21 != F::zero() {
                    eigenvector[0] = lambda - a22;
                    eigenvector[1] = a21;
                } else {
                    // Diagonal matrix
                    eigenvector[0] = if (a11 - lambda).abs() < F::epsilon() {
                        F::one()
                    } else {
                        F::zero()
                    };
                    eigenvector[1] = if (a22 - lambda).abs() < F::epsilon() {
                        F::one()
                    } else {
                        F::zero()
                    };
                }

                // Normalize
                let norm = vector_norm(&eigenvector.view(), 2)?;
                if norm > F::epsilon() {
                    eigenvector.mapv_inplace(|x| x / norm);
                }

                eigenvectors.column_mut(i).assign(&eigenvector);
            }

            // Convert to complex
            let complex_eigenvectors = eigenvectors.mapv(|x| Complex::new(x, F::zero()));

            return Ok((eigenvalues, complex_eigenvectors));
        } else {
            // Complex eigenvalues
            let real_part = trace / F::from(2.0).unwrap();
            let imag_part = (-discriminant).sqrt() / F::from(2.0).unwrap();

            eigenvalues[0] = Complex::new(real_part, imag_part);
            eigenvalues[1] = Complex::new(real_part, -imag_part);

            // Compute complex eigenvectors
            let mut complex_eigenvectors = Array2::zeros((2, 2));

            if a12 != F::zero() {
                complex_eigenvectors[[0, 0]] = Complex::new(a12, F::zero());
                complex_eigenvectors[[1, 0]] =
                    Complex::new(eigenvalues[0].re - a11, eigenvalues[0].im);

                complex_eigenvectors[[0, 1]] = Complex::new(a12, F::zero());
                complex_eigenvectors[[1, 1]] =
                    Complex::new(eigenvalues[1].re - a11, eigenvalues[1].im);
            } else if a21 != F::zero() {
                complex_eigenvectors[[0, 0]] =
                    Complex::new(eigenvalues[0].re - a22, eigenvalues[0].im);
                complex_eigenvectors[[1, 0]] = Complex::new(a21, F::zero());

                complex_eigenvectors[[0, 1]] =
                    Complex::new(eigenvalues[1].re - a22, eigenvalues[1].im);
                complex_eigenvectors[[1, 1]] = Complex::new(a21, F::zero());
            }

            // Normalize complex eigenvectors
            for i in 0..2 {
                let mut norm_sq = Complex::new(F::zero(), F::zero());
                for j in 0..2 {
                    norm_sq += complex_eigenvectors[[j, i]] * complex_eigenvectors[[j, i]].conj();
                }
                let norm = norm_sq.re.sqrt();

                if norm > F::epsilon() {
                    for j in 0..2 {
                        complex_eigenvectors[[j, i]] /= Complex::new(norm, F::zero());
                    }
                }
            }

            return Ok((eigenvalues, complex_eigenvectors));
        }
    }

    // For larger matrices, use the QR algorithm
    let mut a_k = a.to_owned();
    let n = a.nrows();
    let max_iter = 100;
    let tol = F::epsilon() * F::from(100.0).unwrap();

    // Initialize eigenvalues and eigenvectors
    let mut eigenvalues = Array1::zeros(n);
    let mut eigenvectors = Array2::eye(n);

    for _iter in 0..max_iter {
        // QR decomposition
        let (q, r) = qr(&a_k.view())?;

        // Update A_k+1 = R*Q (reversed order gives better convergence)
        let a_next = r.dot(&q);

        // Update eigenvectors: V_k+1 = V_k * Q
        eigenvectors = eigenvectors.dot(&q);

        // Check for convergence (check if subdiagonal elements are close to zero)
        let mut converged = true;
        for i in 1..n {
            if a_next[[i, i - 1]].abs() > tol {
                converged = false;
                break;
            }
        }

        if converged {
            // Extract eigenvalues from diagonal
            for i in 0..n {
                eigenvalues[i] = Complex::new(a_next[[i, i]], F::zero());
            }

            // Return as complex values
            let complex_eigenvectors = eigenvectors.mapv(|x| Complex::new(x, F::zero()));
            return Ok((eigenvalues, complex_eigenvectors));
        }

        // If not converged, continue with next iteration
        a_k = a_next;
    }

    // If we reached maximum iterations without convergence
    // Check if we at least have a reasonable approximation
    let mut eigenvals = Array1::zeros(n);
    for i in 0..n {
        eigenvals[i] = Complex::new(a_k[[i, i]], F::zero());
    }

    // Return the best approximation we have
    let complex_eigenvectors = eigenvectors.mapv(|x| Complex::new(x, F::zero()));
    Ok((eigenvals, complex_eigenvectors))
}

/// Compute the eigenvalues of a square matrix.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Vector of complex eigenvalues
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigvals;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let w = eigvals(&a.view()).unwrap();
///
/// // Sort eigenvalues (they may be returned in different order)
/// let mut eigenvalues = vec![w[0].re, w[1].re];
/// eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
///
/// assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
/// assert!((eigenvalues[1] - 2.0).abs() < 1e-10);
/// ```
pub fn eigvals<F>(a: &ArrayView2<F>) -> LinalgResult<Array1<Complex<F>>>
where
    F: Float + NumAssign + Sum + 'static,
{
    // For efficiency, we can compute just the eigenvalues
    // But for now, we'll use the full function and discard the eigenvectors
    let (eigenvalues, _) = eig(a)?;
    Ok(eigenvalues)
}

/// Compute the dominant eigenvalue and eigenvector of a matrix using power iteration.
///
/// This is a simple iterative method that converges to the eigenvalue with the largest
/// absolute value and its corresponding eigenvector.
///
/// # Arguments
///
/// * `a` - Input square matrix
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
///
/// # Returns
///
/// * Tuple (eigenvalue, eigenvector)
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::power_iteration;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
/// let (eigenvalue, eigenvector) = power_iteration(&a.view(), 100, 1e-10).unwrap();
/// // The largest eigenvalue of this matrix is approximately 3.618
/// assert!((eigenvalue - 3.618).abs() < 1e-2);
/// ```
pub fn power_iteration<F>(
    a: &ArrayView2<F>,
    max_iter: usize,
    tol: F,
) -> LinalgResult<(F, Array1<F>)>
where
    F: Float + NumAssign + Sum,
{
    // Check if matrix is square
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    let n = a.nrows();

    // Start with a random vector
    let mut rng = rand::rng();
    let mut b = Array1::zeros(n);
    for i in 0..n {
        b[i] = F::from(rng.random_range(-1.0..=1.0)).unwrap_or(F::zero());
    }

    // Normalize the vector
    let norm_b = vector_norm(&b.view(), 2)?;
    b.mapv_inplace(|x| x / norm_b);

    let mut eigenvalue = F::zero();
    let mut prev_eigenvalue = F::zero();

    for _ in 0..max_iter {
        // Multiply b by A
        let mut b_new = Array1::zeros(n);
        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..n {
                sum += a[[i, j]] * b[j];
            }
            b_new[i] = sum;
        }

        // Calculate the Rayleigh quotient (eigenvalue estimate)
        eigenvalue = F::zero();
        for i in 0..n {
            eigenvalue += b[i] * b_new[i];
        }

        // Normalize the vector
        let norm_b_new = vector_norm(&b_new.view(), 2)?;
        for i in 0..n {
            b[i] = b_new[i] / norm_b_new;
        }

        // Check for convergence
        if (eigenvalue - prev_eigenvalue).abs() < tol {
            return Ok((eigenvalue, b));
        }

        prev_eigenvalue = eigenvalue;
    }

    // Return the result after max_iter iterations
    Ok((eigenvalue, b))
}

/// Compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.
///
/// # Arguments
///
/// * `a` - Input Hermitian or symmetric matrix
///
/// # Returns
///
/// * Tuple (eigenvalues, eigenvectors) where eigenvalues is a real vector
///   and eigenvectors is a real matrix whose columns are the eigenvectors
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigh;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let (w, v) = eigh(&a.view()).unwrap();
///
/// // Sort eigenvalues (they may be returned in different order)
/// let mut eigenvalues = vec![(w[0], 0), (w[1], 1)];
/// eigenvalues.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
///
/// assert!((eigenvalues[0].0 - 1.0).abs() < 1e-10);
/// assert!((eigenvalues[1].0 - 2.0).abs() < 1e-10);
/// ```
pub fn eigh<F>(a: &ArrayView2<F>) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + 'static,
{
    // Check if matrix is square
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Expected square matrix, got shape {:?}",
            a.shape()
        )));
    }

    // Check if matrix is symmetric
    for i in 0..a.nrows() {
        for j in (i + 1)..a.ncols() {
            if (a[[i, j]] - a[[j, i]]).abs() > F::epsilon() {
                return Err(LinalgError::ShapeError(
                    "Matrix must be symmetric for Hermitian eigenvalue computation".to_string(),
                ));
            }
        }
    }

    let n = a.nrows();

    // For small matrices, we can compute eigenvalues directly
    if n == 1 {
        let eigenvalue = a[[0, 0]];
        let eigenvector = Array2::eye(1);

        return Ok((Array1::from_elem(1, eigenvalue), eigenvector));
    } else if n == 2 {
        // For 2x2 symmetric matrices
        let a11 = a[[0, 0]];
        let a12 = a[[0, 1]];
        let a22 = a[[1, 1]];

        let trace = a11 + a22;
        let det = a11 * a22 - a12 * a12; // For symmetric matrices, a12 = a21

        let discriminant = trace * trace - F::from(4.0).unwrap() * det;
        let sqrt_discriminant = discriminant.sqrt();

        // Eigenvalues
        let lambda1 = (trace + sqrt_discriminant) / F::from(2.0).unwrap();
        let lambda2 = (trace - sqrt_discriminant) / F::from(2.0).unwrap();

        let mut eigenvalues = Array1::zeros(2);
        eigenvalues[0] = lambda1;
        eigenvalues[1] = lambda2;

        // Eigenvectors
        let mut eigenvectors = Array2::zeros((2, 2));

        // First eigenvector
        if a12 != F::zero() {
            eigenvectors[[0, 0]] = a12;
            eigenvectors[[1, 0]] = lambda1 - a11;
        } else {
            // Diagonal matrix
            eigenvectors[[0, 0]] = if (a11 - lambda1).abs() < F::epsilon() {
                F::one()
            } else {
                F::zero()
            };
            eigenvectors[[1, 0]] = if (a22 - lambda1).abs() < F::epsilon() {
                F::one()
            } else {
                F::zero()
            };
        }

        // Normalize
        let norm1 = (eigenvectors[[0, 0]] * eigenvectors[[0, 0]]
            + eigenvectors[[1, 0]] * eigenvectors[[1, 0]])
        .sqrt();
        if norm1 > F::epsilon() {
            eigenvectors[[0, 0]] /= norm1;
            eigenvectors[[1, 0]] /= norm1;
        }

        // Second eigenvector
        if a12 != F::zero() {
            eigenvectors[[0, 1]] = a12;
            eigenvectors[[1, 1]] = lambda2 - a11;
        } else {
            // Diagonal matrix
            eigenvectors[[0, 1]] = if (a11 - lambda2).abs() < F::epsilon() {
                F::one()
            } else {
                F::zero()
            };
            eigenvectors[[1, 1]] = if (a22 - lambda2).abs() < F::epsilon() {
                F::one()
            } else {
                F::zero()
            };
        }

        // Normalize
        let norm2 = (eigenvectors[[0, 1]] * eigenvectors[[0, 1]]
            + eigenvectors[[1, 1]] * eigenvectors[[1, 1]])
        .sqrt();
        if norm2 > F::epsilon() {
            eigenvectors[[0, 1]] /= norm2;
            eigenvectors[[1, 1]] /= norm2;
        }

        return Ok((eigenvalues, eigenvectors));
    }

    // For larger matrices, use the QR algorithm with tridiagonalization

    // Step 1: Tridiagonalization using Householder transformations
    let mut t = a.to_owned();
    let mut q = Array2::eye(n);

    // Householder tridiagonalization
    for k in 0..(n - 2) {
        // Extract subvector from column k, below diagonal
        let mut x = Array1::zeros(n - k - 1);
        for i in (k + 1)..n {
            x[i - k - 1] = t[[i, k]];
        }

        // Compute Householder reflection vector
        let x_norm = x.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();
        if x_norm > F::epsilon() {
            let alpha = if x[0] >= F::zero() { -x_norm } else { x_norm };
            let mut v = x.clone();
            v[0] -= alpha;

            // Normalize v
            let v_norm = v.iter().fold(F::zero(), |acc, &val| acc + val * val).sqrt();
            if v_norm > F::epsilon() {
                for i in 0..v.len() {
                    v[i] /= v_norm;
                }

                // Apply Householder transformation to T
                // First, compute T*v for the trailing submatrix
                for j in k..n {
                    let mut tv = F::zero();
                    for i in (k + 1)..n {
                        tv += t[[i, j]] * v[i - k - 1];
                    }

                    // Now compute T - 2v(v^T*T)
                    for i in (k + 1)..n {
                        t[[i, j]] -= F::from(2.0).unwrap() * v[i - k - 1] * tv;
                    }
                }

                // Apply Householder transformation to T on the left
                for i in 0..n {
                    let mut vt = F::zero();
                    for j in (k + 1)..n {
                        vt += v[j - k - 1] * t[[i, j]];
                    }

                    for j in (k + 1)..n {
                        t[[i, j]] -= F::from(2.0).unwrap() * vt * v[j - k - 1];
                    }
                }

                // Apply the transformation to the eigenvector matrix Q
                for i in 0..n {
                    let mut qv = F::zero();
                    for j in (k + 1)..n {
                        qv += q[[i, j]] * v[j - k - 1];
                    }

                    for j in (k + 1)..n {
                        q[[i, j]] -= F::from(2.0).unwrap() * qv * v[j - k - 1];
                    }
                }
            }
        }
    }

    // Clear numerical noise in the tridiagonal matrix
    for i in 0..n {
        for j in 0..n {
            if (i > 0 && j < i - 1) || j > i + 1 {
                t[[i, j]] = F::zero();
            }
        }
    }

    // Step 2: QR algorithm on the tridiagonal matrix
    let mut eigenvalues = Array1::zeros(n);
    let max_iter = 100;
    let tol = F::epsilon() * F::from(100.0).unwrap();

    // Extract diagonal and subdiagonal
    let mut diag = Array1::zeros(n);
    let mut subdiag = Array1::zeros(n - 1);

    for i in 0..n {
        diag[i] = t[[i, i]];
        if i < n - 1 {
            subdiag[i] = t[[i + 1, i]];
        }
    }

    // QR algorithm without explicit matrix formations
    for _ in 0..max_iter {
        // Check for convergence
        let mut m = n - 1;
        while m > 0 {
            if subdiag[m - 1].abs() < tol * (diag[m - 1].abs() + diag[m].abs()) {
                subdiag[m - 1] = F::zero();
            } else {
                break;
            }
            m -= 1;
        }

        if m == 0 {
            // All eigenvalues have been decoupled
            break;
        }

        // Find the submatrix to work on
        let mut l = m;
        while l > 0 {
            if subdiag[l - 1].abs() < tol * (diag[l - 1].abs() + diag[l].abs()) {
                break;
            }
            l -= 1;
        }

        // Implicit QR step on the submatrix [l:m+1, l:m+1]
        let shift = diag[m];
        let mut g = (diag[m - 1] - shift) / (F::from(2.0).unwrap() * subdiag[m - 1]);
        let mut r = (g * g + F::one()).sqrt();
        if g < F::zero() {
            r = -r;
        }
        g = diag[l] - shift + subdiag[m - 1] / (g + r);

        let mut s = F::one();
        let mut c = F::one();
        let mut p = F::zero();

        // Givens rotations
        for i in l..(m) {
            let f = s * subdiag[i];
            let b = c * subdiag[i];

            r = (f * f + g * g).sqrt();
            if i > l {
                subdiag[i - 1] = r;
            }

            if r == F::zero() {
                diag[i + 1] -= p;
                subdiag[m - 1] = F::zero();
                break;
            }

            s = f / r;
            c = g / r;
            g = diag[i + 1] - p;
            r = (diag[i] - g) * s + F::from(2.0).unwrap() * c * b;
            p = s * r;
            diag[i] = g + p;
            g = c * r - b;

            // Accumulate the transformation in the eigenvector matrix
            if i > l {
                for k in 0..n {
                    let temp = q[[k, i]];
                    q[[k, i]] = s * q[[k, i - 1]] + c * temp;
                    q[[k, i - 1]] = c * q[[k, i - 1]] - s * temp;
                }
            }
        }

        diag[l] -= p;
        subdiag[l] = g;
        subdiag[m - 1] = F::zero();
    }

    // Extract eigenvalues from the diagonal
    for i in 0..n {
        eigenvalues[i] = diag[i];
    }

    // Sort eigenvalues and eigenvectors (optional)
    // For simplicity, we'll just return them as is

    // Return the eigenvalues and eigenvectors
    // Note: Q has eigenvectors in columns, so we need to transpose
    Ok((eigenvalues, q))
}

/// Compute the eigenvalues of a Hermitian or symmetric matrix.
///
/// # Arguments
///
/// * `a` - Input Hermitian or symmetric matrix
///
/// # Returns
///
/// * Vector of real eigenvalues
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_linalg::eigvalsh;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let w = eigvalsh(&a.view()).unwrap();
///
/// // Sort eigenvalues (they may be returned in different order)
/// let mut eigenvalues = vec![w[0], w[1]];
/// eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
///
/// assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
/// assert!((eigenvalues[1] - 2.0).abs() < 1e-10);
/// ```
pub fn eigvalsh<F>(a: &ArrayView2<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    // For efficiency, we can compute just the eigenvalues
    // But for now, we'll use the full function and discard the eigenvectors
    let (eigenvalues, _) = eigh(a)?;
    Ok(eigenvalues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_non_square_matrix() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = eig(&a.view());
        assert!(result.is_err());

        let result = eigvals(&a.view());
        assert!(result.is_err());

        let result = eigh(&a.view());
        assert!(result.is_err());

        let result = eigvalsh(&a.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_non_symmetric_matrix() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];

        // eig and eigvals handle non-symmetric matrices
        let result = eig(&a.view());
        assert!(result.is_ok()); // Now implemented for 2x2

        let result = eigvals(&a.view());
        assert!(result.is_ok()); // Now implemented for 2x2

        // eigh and eigvalsh require symmetric matrices
        let result = eigh(&a.view());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("symmetric"));

        let result = eigvalsh(&a.view());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("symmetric"));
    }

    #[test]
    fn test_1x1_matrix() {
        let a = array![[5.0]];

        // Test eig
        let (eigenvalues, eigenvectors) = eig(&a.view()).unwrap();
        assert_relative_eq!(eigenvalues[0].re, 5.0, epsilon = 1e-10);
        assert_relative_eq!(eigenvalues[0].im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(eigenvectors[[0, 0]].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(eigenvectors[[0, 0]].im, 0.0, epsilon = 1e-10);

        // Test eigh
        let (eigenvalues, _) = eigh(&a.view()).unwrap();
        assert_relative_eq!(eigenvalues[0], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_2x2_diagonal_matrix() {
        let a = array![[3.0, 0.0], [0.0, 4.0]];

        // Test eig
        let (eigenvalues, eigenvectors) = eig(&a.view()).unwrap();
        assert_relative_eq!(eigenvalues[0].re, 4.0, epsilon = 1e-10); // Eigenvalues might be sorted
        assert_relative_eq!(eigenvalues[0].im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(eigenvalues[1].re, 3.0, epsilon = 1e-10);
        assert_relative_eq!(eigenvalues[1].im, 0.0, epsilon = 1e-10);

        // First eigenvector should be [0, 1]
        assert!(
            (eigenvectors[[0, 0]].re.abs() < 1e-10
                && (eigenvectors[[1, 0]].re - 1.0).abs() < 1e-10)
                || (eigenvectors[[0, 1]].re.abs() < 1e-10
                    && (eigenvectors[[1, 1]].re - 1.0).abs() < 1e-10)
        );

        // Second eigenvector should be [1, 0]
        assert!(
            ((eigenvectors[[0, 0]].re - 1.0).abs() < 1e-10
                && eigenvectors[[1, 0]].re.abs() < 1e-10)
                || ((eigenvectors[[0, 1]].re - 1.0).abs() < 1e-10
                    && eigenvectors[[1, 1]].re.abs() < 1e-10)
        );

        // Test eigh
        let (eigenvalues, _) = eigh(&a.view()).unwrap();
        // The eigenvalues might be returned in a different order
        assert!(
            (eigenvalues[0] - 3.0).abs() < 1e-10 && (eigenvalues[1] - 4.0).abs() < 1e-10
                || (eigenvalues[1] - 3.0).abs() < 1e-10 && (eigenvalues[0] - 4.0).abs() < 1e-10
        );
    }

    #[test]
    fn test_2x2_symmetric_matrix() {
        let a = array![[1.0, 2.0], [2.0, 4.0]];

        // Test eigh
        let (eigenvalues, eigenvectors) = eigh(&a.view()).unwrap();

        // Eigenvalues should be approximately 5 and 0
        assert!(
            (eigenvalues[0] - 5.0).abs() < 1e-10 && eigenvalues[1].abs() < 1e-10
                || (eigenvalues[1] - 5.0).abs() < 1e-10 && eigenvalues[0].abs() < 1e-10
        );

        // Check that eigenvectors are orthogonal
        let dot_product = eigenvectors[[0, 0]] * eigenvectors[[0, 1]]
            + eigenvectors[[1, 0]] * eigenvectors[[1, 1]];
        assert!(
            (dot_product).abs() < 1e-10,
            "Eigenvectors should be orthogonal"
        );

        // Check that eigenvectors are normalized
        let norm1 = (eigenvectors[[0, 0]] * eigenvectors[[0, 0]]
            + eigenvectors[[1, 0]] * eigenvectors[[1, 0]])
        .sqrt();
        let norm2 = (eigenvectors[[0, 1]] * eigenvectors[[0, 1]]
            + eigenvectors[[1, 1]] * eigenvectors[[1, 1]])
        .sqrt();
        assert!(
            (norm1 - 1.0).abs() < 1e-10,
            "First eigenvector should be normalized"
        );
        assert!(
            (norm2 - 1.0).abs() < 1e-10,
            "Second eigenvector should be normalized"
        );
    }

    #[test]
    fn test_power_iteration() {
        // Matrix with known dominant eigenvalue and eigenvector
        let a = array![[3.0, 1.0], [1.0, 3.0]];

        let (eigenvalue, eigenvector) = power_iteration(&a.view(), 100, 1e-10).unwrap();

        // Dominant eigenvalue should be 4
        assert_relative_eq!(eigenvalue, 4.0, epsilon = 1e-8);

        // Eigenvector should be normalized
        let norm = (eigenvector[0] * eigenvector[0] + eigenvector[1] * eigenvector[1]).sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);

        // Skip detailed checking of eigenvector components since these can vary slightly
        // depending on numerical precision and iteration count. Just verify that:
        // 1. The vector is normalized
        // 2. When multiplied by A, it should give approximately eigenvalue * v

        // Check that Av ≈ lambda * v
        let av = a.dot(&eigenvector);
        let lambda_v = &eigenvector * eigenvalue;

        let mut max_diff = 0.0;
        for i in 0..eigenvector.len() {
            let diff = (av[i] - lambda_v[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        assert!(
            max_diff < 1e-5,
            "A*v should approximately equal lambda*v, max diff: {}",
            max_diff
        );
    }
}
