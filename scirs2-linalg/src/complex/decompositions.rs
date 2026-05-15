//! Complex matrix decompositions
//!
//! This module provides implementations of various matrix decompositions
//! for complex-valued matrices.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::Complex;
use scirs2_core::numeric::{Float, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

use crate::complex::hermitian_transpose;
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::validation::check_square;

/// Complex LU decomposition structure
pub struct ComplexLUDecomposition<F: Float> {
    /// LU decomposition result (combined L and U matrices)
    pub lu: Array2<Complex<F>>,
    /// Permutation indices
    pub piv: Vec<usize>,
    /// Permutation sign (+1 or -1)
    pub sign: Complex<F>,
}

/// Complex QR decomposition structure
pub struct ComplexQRDecomposition<F: Float> {
    /// Q matrix (unitary)
    pub q: Array2<Complex<F>>,
    /// R matrix (upper triangular)
    pub r: Array2<Complex<F>>,
}

/// Complex SVD decomposition structure
pub struct ComplexSVDDecomposition<F: Float> {
    /// U matrix (left singular vectors, unitary)
    pub u: Array2<Complex<F>>,
    /// Singular values (real, non-negative)
    pub s: Array1<F>,
    /// V^H matrix (conjugate transpose of right singular vectors)
    pub vh: Array2<Complex<F>>,
}

/// Complex eigenvalue decomposition structure
pub struct ComplexEigDecomposition<F: Float> {
    /// Eigenvalues (complex)
    pub eigenvalues: Array1<Complex<F>>,
    /// Eigenvectors (column-wise, complex)
    pub eigenvectors: Array2<Complex<F>>,
}

/// Performs complex LU decomposition with partial pivoting
///
/// Decomposes A into P * L * U where:
/// - P is a permutation matrix
/// - L is lower triangular with 1s on diagonal
/// - U is upper triangular
#[allow(dead_code)]
pub fn complex_lu<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<ComplexLUDecomposition<F>>
where
    F: Float + Sum + Debug,
{
    let (m, n) = (a.nrows(), a.ncols());
    let mut lu = a.to_owned();
    let mut piv: Vec<usize> = (0..m).collect();
    let mut sign = Complex::one();

    for k in 0..m.min(n) {
        // Find pivot
        let mut max_row = k;
        let mut max_val = lu[[k, k]].norm();

        for i in (k + 1)..m {
            let val = lu[[i, k]].norm();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        if max_val < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != k {
            piv.swap(k, max_row);
            sign = -sign;

            for j in 0..n {
                let temp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = temp;
            }
        }

        // Compute multipliers
        for i in (k + 1)..m {
            lu[[i, k]] = lu[[i, k]] / lu[[k, k]];

            // Update row
            for j in (k + 1)..n {
                lu[[i, j]] = lu[[i, j]] - lu[[i, k]] * lu[[k, j]];
            }
        }
    }

    Ok(ComplexLUDecomposition { lu, piv, sign })
}

/// Performs complex QR decomposition
///
/// Decomposes A into Q * R where:
/// - Q is unitary (Q^H * Q = I)
/// - R is upper triangular
#[allow(dead_code)]
pub fn complex_qr<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<ComplexQRDecomposition<F>>
where
    F: Float + Sum + Debug,
{
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);

    // Start with A
    let mut q = Array2::<Complex<F>>::zeros((m, k));
    let mut r = Array2::<Complex<F>>::zeros((k, n));

    // Copy columns of A
    let mut columns = vec![];
    for j in 0..n {
        let mut col = Array1::<Complex<F>>::zeros(m);
        for i in 0..m {
            col[i] = a[[i, j]];
        }
        columns.push(col);
    }

    // Gram-Schmidt process
    for j in 0..k {
        // Start with the j-th column of A
        let mut q_j = columns[j].clone();

        // Orthogonalize against all previous columns
        for i in 0..j {
            // Calculate projection coefficient: <q_i, a_j>
            let mut projection = Complex::<F>::zero();
            for k in 0..m {
                projection = projection + q[[k, i]].conj() * q_j[k];
            }

            // Store in R
            r[[i, j]] = projection;

            // Subtract projection from q_j
            for k in 0..m {
                q_j[k] = q_j[k] - projection * q[[k, i]];
            }
        }

        // Calculate norm
        let mut norm_sq = F::zero();
        for i in 0..m {
            norm_sq = norm_sq + q_j[i].norm_sqr();
        }

        if norm_sq < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Columns are linearly dependent".to_string(),
            ));
        }

        let norm = norm_sq.sqrt();
        r[[j, j]] = Complex::<F>::new(norm, F::zero());

        // Normalize and store in Q
        for i in 0..m {
            q[[i, j]] = q_j[i] / Complex::<F>::new(norm, F::zero());
        }

        // For columns beyond k, just calculate R entries
        for l in (j + 1)..n {
            let mut projection = Complex::<F>::zero();
            for i in 0..m {
                projection = projection + q[[i, j]].conj() * a[[i, l]];
            }
            r[[j, l]] = projection;
        }
    }

    // Extend Q to full unitary matrix if needed (m > n)
    let final_q = if m > n {
        let mut q_full = Array2::<Complex<F>>::eye(m);
        // Copy the computed columns
        for i in 0..m {
            for j in 0..k {
                q_full[[i, j]] = q[[i, j]];
            }
        }
        q_full
    } else {
        q
    };

    Ok(ComplexQRDecomposition { q: final_q, r })
}

/// Performs complex SVD decomposition
///
/// Decomposes A into U * S * V^H where:
/// - U is unitary (left singular vectors)
/// - S is diagonal with real, non-negative values
/// - V^H is unitary (conjugate transpose of right singular vectors)
#[allow(dead_code)]
pub fn complex_svd<F>(
    a: &ArrayView2<Complex<F>>,
    full_matrices: bool,
) -> LinalgResult<ComplexSVDDecomposition<F>>
where
    F: Float + Sum + Debug,
{
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);

    // Compute A^H * A
    let ah = hermitian_transpose(a);
    let mut aha = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex::<F>::zero();
            for l in 0..m {
                sum = sum + ah[[i, l]] * a[[l, j]];
            }
            aha[[i, j]] = sum;
        }
    }

    // Eigendecomposition of A^H * A
    let eig = complex_eigh(&aha.view())?;
    let eigenvalues = eig.eigenvalues.clone();
    let v = eig.eigenvectors.clone();

    // Sort eigenvalues and eigenvectors in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        eigenvalues[j]
            .re
            .partial_cmp(&eigenvalues[i].re)
            .expect("Operation failed")
    });

    let mut s = Array1::zeros(k);
    let vh_rows = if full_matrices { n } else { k.min(n) };
    let mut vt = Array2::zeros((vh_rows, n));

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        if new_idx < s.len() {
            s[new_idx] = eigenvalues[old_idx].re.sqrt();
        }
        if new_idx < vt.nrows() {
            for j in 0..n {
                vt[[new_idx, j]] = v[[j, old_idx]].conj();
            }
        }
    }

    // Compute U = A * V * S^(-1)
    let u_cols = if full_matrices { m } else { k.min(m) };
    let mut u = Array2::zeros((m, u_cols));

    // Only compute U columns for non-zero singular values
    for j in 0..s.len().min(u_cols) {
        if s[j] > F::epsilon() {
            for i in 0..m {
                let mut sum = Complex::<F>::zero();
                for l in 0..n {
                    sum = sum + a[[i, l]] * v[[l, indices[j]]];
                }
                u[[i, j]] = sum / Complex::<F>::new(s[j], F::zero());
            }
        }
    }

    if full_matrices && k < u_cols {
        // Implement proper orthogonalization for remaining columns using modified Gram-Schmidt

        // Generate random orthogonal columns for the null space
        for j in k..u_cols {
            // Start with a random vector
            for i in 0..m {
                // Use a simple pattern to generate a "random" starting vector
                let real_part = F::from((i + j) as f64 / m as f64).expect("Operation failed");
                let imag_part = F::from((i * j) as f64 / (m * m) as f64).expect("Operation failed");
                u[[i, j]] = Complex::new(real_part, imag_part);
            }

            // Orthogonalize against previous columns using modified Gram-Schmidt
            for prev_j in 0..j {
                // Compute projection coefficient: <u_prev, u_current>
                let mut proj_coeff = Complex::<F>::zero();
                for i in 0..m {
                    proj_coeff = proj_coeff + u[[i, prev_j]].conj() * u[[i, j]];
                }

                // Subtract projection: u_current = u_current - proj_coeff * u_prev
                for i in 0..m {
                    u[[i, j]] = u[[i, j]] - proj_coeff * u[[i, prev_j]];
                }
            }

            // Normalize the column
            let mut norm_sq = F::zero();
            for i in 0..m {
                norm_sq = norm_sq + u[[i, j]].norm_sqr();
            }
            let norm = norm_sq.sqrt();

            if norm > F::epsilon() {
                for i in 0..m {
                    u[[i, j]] = u[[i, j]] / Complex::new(norm, F::zero());
                }
            } else {
                // If the vector becomes zero after orthogonalization, use a different starting vector
                for i in 0..m {
                    let real_part = if i == j % m { F::one() } else { F::zero() };
                    u[[i, j]] = Complex::new(real_part, F::zero());
                }

                // Re-orthogonalize
                for prev_j in 0..j {
                    let mut proj_coeff = Complex::<F>::zero();
                    for i in 0..m {
                        proj_coeff = proj_coeff + u[[i, prev_j]].conj() * u[[i, j]];
                    }

                    for i in 0..m {
                        u[[i, j]] = u[[i, j]] - proj_coeff * u[[i, prev_j]];
                    }
                }

                // Re-normalize
                let mut norm_sq = F::zero();
                for i in 0..m {
                    norm_sq = norm_sq + u[[i, j]].norm_sqr();
                }
                let norm = norm_sq.sqrt();

                if norm > F::epsilon() {
                    for i in 0..m {
                        u[[i, j]] = u[[i, j]] / Complex::new(norm, F::zero());
                    }
                }
            }
        }
    }

    // Return vh with correct dimensions
    Ok(ComplexSVDDecomposition { u, s, vh: vt })
}

/// Performs eigendecomposition of a complex matrix
///
/// Finds eigenvalues and eigenvectors such that A * v = λ * v
#[allow(dead_code)]
pub fn complex_eig<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<ComplexEigDecomposition<F>>
where
    F: Float + Sum + Debug,
{
    check_square(a, "matrix")?;
    let n = a.nrows();

    // Use QR algorithm with shifts for general complex matrices
    let max_iterations = 1000;
    let tolerance = F::from(1e-10).expect("Operation failed");

    let mut h = a.to_owned(); // Working matrix (will become Schur form)
    let mut q_total = Array2::eye(n); // Accumulated transformations

    // First, reduce to Hessenberg form using Householder reflections
    for col in 0..(n - 2) {
        // Find Householder reflector for column
        let mut v = Array1::zeros(n - col - 1);
        let mut norm_sq = F::zero();

        for i in (col + 1)..n {
            v[i - col - 1] = h[[i, col]];
            norm_sq = norm_sq + h[[i, col]].norm_sqr();
        }

        if norm_sq > tolerance {
            let norm = norm_sq.sqrt();
            let first_elem = h[[col + 1, col]];
            let phase = if first_elem.norm() > F::zero() {
                first_elem / Complex::<F>::new(first_elem.norm(), F::zero())
            } else {
                Complex::<F>::one()
            };

            v[0] = v[0] + phase * Complex::<F>::new(norm, F::zero());
            let v_norm_sq = v.iter().map(|x| x.norm_sqr()).sum();

            if v_norm_sq > tolerance {
                // Apply Householder transformation: H = I - 2vv^H/||v||^2
                for j in col..n {
                    let mut sum = Complex::<F>::zero();
                    for i in 0..v.len() {
                        sum = sum + v[i].conj() * h[[i + col + 1, j]];
                    }
                    let factor = Complex::<F>::new(
                        F::from(2.0).expect("Operation failed") / v_norm_sq,
                        F::zero(),
                    ) * sum;

                    for i in 0..v.len() {
                        h[[i + col + 1, j]] = h[[i + col + 1, j]] - factor * v[i];
                    }
                }

                // Apply from right: H' = H'H
                for i in 0..n {
                    let mut sum = Complex::<F>::zero();
                    for j in 0..v.len() {
                        sum = sum + h[[i, j + col + 1]] * v[j];
                    }
                    let factor = Complex::<F>::new(
                        F::from(2.0).expect("Operation failed") / v_norm_sq,
                        F::zero(),
                    ) * sum;

                    for j in 0..v.len() {
                        h[[i, j + col + 1]] = h[[i, j + col + 1]] - factor * v[j].conj();
                    }
                }
            }
        }
    }

    // QR iterations with shifts
    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < max_iterations {
        converged = true;
        iterations += 1;

        // Check for convergence (subdiagonal elements)
        for i in 0..(n - 1) {
            if h[[i + 1, i]].norm() > tolerance {
                converged = false;
                break;
            }
        }

        if !converged {
            // Compute shift (Wilkinson shift for 2x2 lower-right block)
            let mut shift = Complex::<F>::zero();
            if n >= 2 {
                let a11 = h[[n - 2, n - 2]];
                let a12 = h[[n - 2, n - 1]];
                let a21 = h[[n - 1, n - 2]];
                let a22 = h[[n - 1, n - 1]];

                let trace = a11 + a22;
                let det = a11 * a22 - a12 * a21;
                let discriminant = trace * trace
                    - Complex::<F>::new(F::from(4.0).expect("Operation failed"), F::zero()) * det;
                let sqrt_disc = discriminant.sqrt();

                let eig1 = (trace + sqrt_disc)
                    / Complex::<F>::new(F::from(2.0).expect("Operation failed"), F::zero());
                let eig2 = (trace - sqrt_disc)
                    / Complex::<F>::new(F::from(2.0).expect("Operation failed"), F::zero());

                // Choose eigenvalue closer to a22
                shift = if (eig1 - a22).norm() < (eig2 - a22).norm() {
                    eig1
                } else {
                    eig2
                };
            }

            // QR step with shift
            let mut h_shifted = h.clone();
            for i in 0..n {
                h_shifted[[i, i]] = h_shifted[[i, i]] - shift;
            }

            let qr = complex_qr(&h_shifted.view())?;
            h = crate::complex::complex_matmul(&qr.r.view(), &qr.q.view())
                .expect("Operation failed");

            for i in 0..n {
                h[[i, i]] = h[[i, i]] + shift;
            }

            // Update eigenvector matrix
            q_total = crate::complex::complex_matmul(&q_total.view(), &qr.q.view())
                .expect("Operation failed");
        }
    }

    // Extract eigenvalues from diagonal of Schur form
    let mut eigenvalues = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = h[[i, i]];
    }

    Ok(ComplexEigDecomposition {
        eigenvalues,
        eigenvectors: q_total,
    })
}

/// Performs eigendecomposition of a Hermitian complex matrix
///
/// For Hermitian matrices, eigenvalues are real and eigenvectors are orthogonal
#[allow(dead_code)]
pub fn complex_eigh<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<ComplexEigDecomposition<F>>
where
    F: Float + Sum + Debug,
{
    check_square(a, "matrix")?;
    let n = a.nrows();

    // Verify matrix is Hermitian
    let tolerance = F::from(1e-10).expect("Operation failed");

    for i in 0..n {
        for j in i..n {
            let diff = (a[[i, j]] - a[[j, i]].conj()).norm();
            if diff > tolerance {
                return Err(LinalgError::ValueError(
                    "Matrix is not Hermitian".to_string(),
                ));
            }
        }
    }

    // For Hermitian matrices, use specialized tridiagonalization
    // followed by QR algorithm

    // Step 1: Reduce to tridiagonal form using Householder reflections
    let mut t = a.to_owned();
    let mut q_total = Array2::eye(n);

    for k in 0..(n - 2) {
        // Form Householder reflector for column k
        let mut v = Array1::zeros(n - k - 1);
        let mut norm_sq = F::zero();

        for i in (k + 1)..n {
            v[i - k - 1] = t[[i, k]];
            norm_sq = norm_sq + t[[i, k]].norm_sqr();
        }

        if norm_sq > tolerance {
            let norm = norm_sq.sqrt();
            let first_elem = t[[k + 1, k]];
            let phase = if first_elem.norm() > F::zero() {
                first_elem / Complex::<F>::new(first_elem.norm(), F::zero())
            } else {
                Complex::<F>::one()
            };

            v[0] = v[0] + phase * Complex::<F>::new(norm, F::zero());
            let v_norm_sq: F = v.iter().map(|x| x.norm_sqr()).sum();

            if v_norm_sq > tolerance {
                // Apply Householder transformation from left and right
                // T = H * T * H^H where H = I - 2vv^H/||v||^2

                // First: compute T * v
                let mut tv = Array1::zeros(n);
                for i in 0..n {
                    let mut sum = Complex::<F>::zero();
                    for j in 0..v.len() {
                        sum = sum + t[[i, j + k + 1]] * v[j];
                    }
                    tv[i] = sum;
                }

                // Update T = T - (2/||v||^2) * T * v * v^H
                let factor = Complex::<F>::new(
                    F::from(2.0).expect("Operation failed") / v_norm_sq,
                    F::zero(),
                );
                for i in 0..n {
                    for j in 0..v.len() {
                        t[[i, j + k + 1]] = t[[i, j + k + 1]] - factor * tv[i] * v[j].conj();
                    }
                }

                // Apply from left: v^H * T
                let mut vt = Array1::zeros(n);
                for j in 0..n {
                    let mut sum = Complex::<F>::zero();
                    for i in 0..v.len() {
                        sum = sum + v[i].conj() * t[[i + k + 1, j]];
                    }
                    vt[j] = sum;
                }

                // Update T = T - (2/||v||^2) * v * v^H * T
                for i in 0..v.len() {
                    for j in 0..n {
                        t[[i + k + 1, j]] = t[[i + k + 1, j]] - factor * v[i] * vt[j];
                    }
                }

                // Update eigenvector matrix Q
                let mut qv = Array1::zeros(n);
                for i in 0..n {
                    let mut sum = Complex::<F>::zero();
                    for j in 0..v.len() {
                        sum = sum + q_total[[i, j + k + 1]] * v[j];
                    }
                    qv[i] = sum;
                }

                for i in 0..n {
                    for j in 0..v.len() {
                        q_total[[i, j + k + 1]] =
                            q_total[[i, j + k + 1]] - factor * qv[i] * v[j].conj();
                    }
                }
            }
        }
    }

    // Step 2: Apply QR algorithm to tridiagonal matrix
    // Extract diagonal and subdiagonal
    let mut diagonal = Array1::zeros(n);
    let mut subdiagonal = Array1::zeros(n - 1);

    for i in 0..n {
        diagonal[i] = t[[i, i]].re; // Diagonal elements are real for Hermitian matrices
        if i < n - 1 {
            subdiagonal[i] = t[[i + 1, i]].norm(); // Subdiagonal can be made real and positive
        }
    }

    // QR iterations for tridiagonal matrix
    let max_iterations = 1000;
    let mut iterations = 0;

    while iterations < max_iterations {
        iterations += 1;
        let mut converged = true;

        for i in 0..(n - 1) {
            if subdiagonal[i].abs() > tolerance {
                converged = false;
                break;
            }
        }

        if converged {
            break;
        }

        // Wilkinson shift
        let d = (diagonal[n - 2] - diagonal[n - 1]) / F::from(2.0).expect("Operation failed");
        let sign = if d >= F::zero() { F::one() } else { -F::one() };
        let shift = diagonal[n - 1]
            - subdiagonal[n - 2].powi(2)
                / (d + sign * (d.powi(2) + subdiagonal[n - 2].powi(2)).sqrt());

        // Implicit QR step with shift
        let mut g = diagonal[0] - shift;
        let mut s = subdiagonal[0];

        for k in 0..(n - 1) {
            // Givens rotation to zero out s
            let r = (g * g + s * s).sqrt();
            if r > F::zero() {
                let c = g / r;
                let sn = s / r;

                // Apply to subdiagonal
                if k > 0 {
                    subdiagonal[k - 1] = r;
                }

                // Apply to diagonal elements
                let d1 = diagonal[k];
                let d2 = diagonal[k + 1];
                let e = subdiagonal[k];

                diagonal[k] = c * c * d1
                    + F::from(2.0).expect("Operation failed") * c * sn * e
                    + sn * sn * d2;
                diagonal[k + 1] = sn * sn * d1
                    - F::from(2.0).expect("Operation failed") * c * sn * e
                    + c * c * d2;

                if k < n - 2 {
                    subdiagonal[k] = (c * c - sn * sn) * e + c * sn * (d2 - d1);
                    g = subdiagonal[k];
                    s = -sn * subdiagonal[k + 1];
                    subdiagonal[k + 1] = c * subdiagonal[k + 1];
                } else {
                    subdiagonal[k] = (c * c - sn * sn) * e + c * sn * (d2 - d1);
                }

                // Update eigenvector matrix
                for i in 0..n {
                    let q_ik = q_total[[i, k]];
                    let q_ik1 = q_total[[i, k + 1]];
                    q_total[[i, k]] = Complex::<F>::new(c, F::zero()) * q_ik
                        + Complex::<F>::new(sn, F::zero()) * q_ik1;
                    q_total[[i, k + 1]] = Complex::<F>::new(-sn, F::zero()) * q_ik
                        + Complex::<F>::new(c, F::zero()) * q_ik1;
                }
            }
        }
    }

    // Convert real eigenvalues to complex
    let mut eigenvalues = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = Complex::<F>::new(diagonal[i], F::zero());
    }

    Ok(ComplexEigDecomposition {
        eigenvalues,
        eigenvectors: q_total,
    })
}

/// Performs Cholesky decomposition of a positive-definite Hermitian matrix
///
/// Decomposes A into L * L^H where L is lower triangular
#[allow(dead_code)]
pub fn complex_cholesky<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<Array2<Complex<F>>>
where
    F: Float + Sum + Debug,
{
    check_square(a, "matrix")?;
    let n = a.nrows();

    // Verify matrix is Hermitian
    let ah = hermitian_transpose(a);
    let tolerance = F::from(1e-10).expect("Operation failed");

    for i in 0..n {
        for j in 0..n {
            let diff = (a[[i, j]] - ah[[i, j]]).norm();
            if diff > tolerance {
                return Err(LinalgError::ValueError(
                    "Matrix is not Hermitian".to_string(),
                ));
            }
        }
    }

    let mut l = Array2::<Complex<F>>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = Complex::<F>::zero();

            if i == j {
                // Diagonal element
                for k in 0..j {
                    sum = sum + l[[i, k]] * l[[i, k]].conj();
                }

                let diag = (a[[i, i]] - sum).re;
                if diag <= F::zero() {
                    return Err(LinalgError::NonPositiveDefiniteError(
                        "Matrix is not positive definite".to_string(),
                    ));
                }

                l[[i, j]] = Complex::<F>::new(diag.sqrt(), F::zero());
            } else {
                // Off-diagonal element
                for k in 0..j {
                    sum = sum + l[[i, k]] * l[[j, k]].conj();
                }

                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Result type for Schur decomposition with Q and T matrices
pub type SchurResult<F> = LinalgResult<(Array2<Complex<F>>, Array2<Complex<F>>)>;

/// Performs Schur decomposition of a complex matrix.
///
/// Decomposes A = Q T Q^H where:
/// - Q is unitary  (Q^H Q = I)
/// - T is upper triangular (the Schur form)
///
/// The algorithm is:
/// 1. Reduce A to upper-Hessenberg form H = U^H A U via Householder reflectors.
/// 2. Apply Francis single-shift QR iterations on H until all sub-diagonal
///    entries are negligible, giving the upper triangular Schur matrix T.
/// 3. Accumulate all transformations to obtain the unitary factor Q.
///
/// # Errors
/// Returns `LinalgError::NotConvergedError` if the QR iteration exhausts its
/// iteration budget without converging.
#[allow(dead_code)]
pub fn complex_schur<F>(a: &ArrayView2<Complex<F>>) -> SchurResult<F>
where
    F: Float + Sum + Debug,
{
    check_square(a, "matrix")?;
    let n = a.nrows();

    // Trivial cases
    if n == 0 {
        return Ok((Array2::eye(0), Array2::zeros((0, 0))));
    }
    if n == 1 {
        let mut t = Array2::zeros((1, 1));
        t[[0, 0]] = a[[0, 0]];
        return Ok((Array2::eye(1), t));
    }

    let tolerance = F::from(1e-14).expect("tolerance cast failed");
    let two = F::from(2.0).expect("2.0 cast failed");
    let four = F::from(4.0).expect("4.0 cast failed");

    // -----------------------------------------------------------------------
    // Step 1 – Reduce A to upper Hessenberg form using Householder reflectors.
    //
    // Invariant maintained: A = Q_accum * H * Q_accum^H
    // -----------------------------------------------------------------------
    let mut h = a.to_owned();
    let mut q_accum: Array2<Complex<F>> = Array2::eye(n);

    for col in 0..(n.saturating_sub(2)) {
        // Build Householder vector for h[col+1..n, col]
        let sub_len = n - col - 1;
        let mut v = Array1::<Complex<F>>::zeros(sub_len);
        let mut norm_sq = F::zero();

        for i in 0..sub_len {
            v[i] = h[[i + col + 1, col]];
            norm_sq = norm_sq + v[i].norm_sqr();
        }

        if norm_sq <= tolerance {
            continue;
        }

        let norm = norm_sq.sqrt();
        // Phase of first element (choose to introduce cancellation)
        let first = h[[col + 1, col]];
        let phase = if first.norm() > F::zero() {
            first / Complex::<F>::new(first.norm(), F::zero())
        } else {
            Complex::<F>::one()
        };
        // v <- v + phase * ||v|| * e_0
        v[0] = v[0] + phase * Complex::<F>::new(norm, F::zero());

        let v_norm_sq: F = v.iter().map(|x| x.norm_sqr()).sum();
        if v_norm_sq <= tolerance {
            continue;
        }

        let factor_real = two / v_norm_sq;

        // Apply from left: H[col+1..n, :] <- H[col+1..n, :] - factor * v * (v^H H[col+1..n, :])
        for j in 0..n {
            let mut dot = Complex::<F>::zero();
            for i in 0..sub_len {
                dot = dot + v[i].conj() * h[[i + col + 1, j]];
            }
            let f = Complex::<F>::new(factor_real, F::zero()) * dot;
            for i in 0..sub_len {
                h[[i + col + 1, j]] = h[[i + col + 1, j]] - f * v[i];
            }
        }

        // Apply from right: H[:, col+1..n] <- H[:, col+1..n] - (H[:, col+1..n] v) v^H * factor
        for i in 0..n {
            let mut dot = Complex::<F>::zero();
            for j in 0..sub_len {
                dot = dot + h[[i, j + col + 1]] * v[j];
            }
            let f = Complex::<F>::new(factor_real, F::zero()) * dot;
            for j in 0..sub_len {
                h[[i, j + col + 1]] = h[[i, j + col + 1]] - f * v[j].conj();
            }
        }

        // Accumulate Q: q_accum[:, col+1..n] <- q_accum[:, col+1..n] - (q_accum[:, col+1..n] v) v^H * factor
        for i in 0..n {
            let mut dot = Complex::<F>::zero();
            for j in 0..sub_len {
                dot = dot + q_accum[[i, j + col + 1]] * v[j];
            }
            let f = Complex::<F>::new(factor_real, F::zero()) * dot;
            for j in 0..sub_len {
                q_accum[[i, j + col + 1]] = q_accum[[i, j + col + 1]] - f * v[j].conj();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 2 – Shifted QR iterations on the full n×n Hessenberg matrix.
    //
    // Each iteration: apply Wilkinson shift, factor H-σI = QR, update H = RQ+σI,
    // accumulate Q. This mirrors complex_eig's approach and avoids the phase/
    // similarity bugs in a hand-rolled Givens deflation loop.
    // -----------------------------------------------------------------------
    let max_iterations = 1000 * n;
    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < max_iterations {
        iterations += 1;

        // Check convergence: all sub-diagonal entries small
        converged = true;
        for i in 0..(n - 1) {
            let tol_entry = tolerance * (h[[i, i]].norm() + h[[i + 1, i + 1]].norm());
            if h[[i + 1, i]].norm() > tol_entry.max(tolerance) {
                converged = false;
                break;
            }
        }

        if !converged {
            // Wilkinson shift from the full-matrix bottom-right 2×2 block
            let shift = if n >= 2 {
                let a11 = h[[n - 2, n - 2]];
                let a12 = h[[n - 2, n - 1]];
                let a21 = h[[n - 1, n - 2]];
                let a22 = h[[n - 1, n - 1]];

                let trace = a11 + a22;
                let det = a11 * a22 - a12 * a21;
                let disc = trace * trace - Complex::<F>::new(four, F::zero()) * det;
                let sqrt_disc = disc.sqrt();
                let eig1 = (trace + sqrt_disc) / Complex::<F>::new(two, F::zero());
                let eig2 = (trace - sqrt_disc) / Complex::<F>::new(two, F::zero());
                if (eig1 - a22).norm() <= (eig2 - a22).norm() {
                    eig1
                } else {
                    eig2
                }
            } else {
                Complex::<F>::zero()
            };

            // Build shifted matrix, perform QR decomposition, then H = RQ + σI.
            // If the shifted matrix is singular (shift == eigenvalue), fall back to
            // an unshifted step to preserve numerical stability.
            let mut h_shifted = h.clone();
            for i in 0..n {
                h_shifted[[i, i]] = h_shifted[[i, i]] - shift;
            }

            let qr_result = complex_qr(&h_shifted.view());
            let (qr, actual_shift) = match qr_result {
                Ok(qr) => (qr, shift),
                Err(_) => {
                    // Shift caused singularity; try unshifted step
                    let qr_unshifted = complex_qr(&h.view())?;
                    (qr_unshifted, Complex::<F>::zero())
                }
            };

            h = crate::complex::complex_matmul(&qr.r.view(), &qr.q.view())
                .map_err(|e| LinalgError::ComputationError(format!("RQ multiply failed: {e}")))?;

            for i in 0..n {
                h[[i, i]] = h[[i, i]] + actual_shift;
            }

            q_accum =
                crate::complex::complex_matmul(&q_accum.view(), &qr.q.view()).map_err(|e| {
                    LinalgError::ComputationError(format!("Q accumulation failed: {e}"))
                })?;
        }
    }

    if !converged {
        return Err(LinalgError::ConvergenceError(
            "Complex Schur decomposition: QR iteration did not converge".to_string(),
        ));
    }

    // -----------------------------------------------------------------------
    // Step 3 – Clean up: zero all sub-diagonal entries in T explicitly.
    // -----------------------------------------------------------------------
    for i in 1..n {
        for j in 0..i {
            h[[i, j]] = Complex::<F>::zero();
        }
    }

    // q_accum is Q such that T = Q^H A Q  →  A = Q T Q^H
    Ok((q_accum, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_complex_lu() {
        let a = array![
            [Complex::<f64>::new(2.0, 1.0), Complex::<f64>::new(1.0, 0.0)],
            [
                Complex::<f64>::new(1.0, 0.0),
                Complex::<f64>::new(2.0, -1.0)
            ]
        ];

        let lu_result = complex_lu(&a.view()).expect("Operation failed");

        // Verify that P * L * U = A
        let n = a.nrows();
        let mut l = Array2::eye(n);
        let mut u = Array2::zeros((n, n));

        // Extract L and U from combined matrix
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    l[[i, j]] = lu_result.lu[[i, j]];
                } else {
                    u[[i, j]] = lu_result.lu[[i, j]];
                }
            }
        }

        // Apply permutation and verify L * U = P * A
        // First reconstruct the permutation matrix P
        let mut p = Array2::<Complex<f64>>::zeros((n, n));
        for (i, &piv_i) in lu_result.piv.iter().enumerate() {
            p[[i, piv_i]] = Complex::<f64>::one();
        }

        // Compute L * U
        let mut lu_product = Array2::<Complex<f64>>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex::<f64>::zero();
                for k in 0..n {
                    sum += l[[i, k]] * u[[k, j]];
                }
                lu_product[[i, j]] = sum;
            }
        }

        // Compute P * A
        let mut pa = Array2::<Complex<f64>>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex::<f64>::zero();
                for k in 0..n {
                    sum += p[[i, k]] * a[[k, j]];
                }
                pa[[i, j]] = sum;
            }
        }

        // Verify L * U = P * A
        for i in 0..n {
            for j in 0..n {
                let diff = (lu_product[[i, j]] - pa[[i, j]]).norm();
                assert!(
                    diff < 1e-10,
                    "LU decomposition verification failed at ({}, {}): L*U = {}, P*A = {}, diff = {}",
                    i, j, lu_product[[i, j]], pa[[i, j]], diff
                );
            }
        }

        // Verify L is lower triangular with unit diagonal
        for i in 0..n {
            for j in 0..n {
                match i.cmp(&j) {
                    std::cmp::Ordering::Less => {
                        assert!(
                            l[[i, j]].norm() < 1e-10,
                            "L is not lower triangular at ({}, {})",
                            i,
                            j
                        );
                    }
                    std::cmp::Ordering::Equal => {
                        let diff = (l[[i, j]] - Complex::<f64>::one()).norm();
                        assert!(
                            diff < 1e-10,
                            "L does not have unit diagonal at ({}, {}): value = {}",
                            i,
                            j,
                            l[[i, j]]
                        );
                    }
                    std::cmp::Ordering::Greater => {}
                }
            }
        }

        // Verify U is upper triangular
        for i in 0..n {
            for j in 0..i {
                assert!(
                    u[[i, j]].norm() < 1e-10,
                    "U is not upper triangular at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_complex_qr() {
        let a = array![
            [Complex::<f64>::new(1.0, 0.0), Complex::<f64>::new(1.0, 1.0)],
            [Complex::<f64>::new(0.0, 1.0), Complex::<f64>::new(1.0, 0.0)]
        ];

        let qr_result = complex_qr(&a.view()).expect("Operation failed");

        println!(
            "Q shape: {:?}, R shape: {:?}",
            qr_result.q.shape(),
            qr_result.r.shape()
        );
        println!("Q = {:?}", qr_result.q);
        println!("R = {:?}", qr_result.r);

        // Verify Q is unitary: Q^H * Q = I
        let qh = hermitian_transpose(&qr_result.q.view());
        let should_be_i = crate::complex::complex_matmul(&qh.view(), &qr_result.q.view())
            .expect("Operation failed");

        let m = qr_result.q.nrows();
        for i in 0..m {
            for j in 0..m {
                let expected = if i == j {
                    Complex::<f64>::one()
                } else {
                    Complex::<f64>::zero()
                };
                let diff = (should_be_i[[i, j]] - expected).norm();
                assert!(
                    diff < 1e-10,
                    "Q is not unitary at position ({}, {}): diff = {}",
                    i,
                    j,
                    diff
                );
            }
        }

        // Verify Q * R = A
        let qr = crate::complex::complex_matmul(&qr_result.q.view(), &qr_result.r.view())
            .expect("Operation failed");

        println!("Q*R = {:?}", qr);
        println!("A = {:?}", a);

        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_relative_eq!(qr[[i, j]].re, a[[i, j]].re, epsilon = 1e-10);
                assert_relative_eq!(qr[[i, j]].im, a[[i, j]].im, epsilon = 1e-10);
            }
        }

        // Verify R is upper triangular
        for i in 0..qr_result.r.nrows() {
            for j in 0..i.min(qr_result.r.ncols()) {
                assert!(
                    qr_result.r[[i, j]].norm() < 1e-10,
                    "R is not upper triangular at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_complex_cholesky() {
        // Create a positive-definite Hermitian matrix
        let a = array![
            [
                Complex::<f64>::new(3.0, 0.0),
                Complex::<f64>::new(1.0, -1.0)
            ],
            [Complex::<f64>::new(1.0, 1.0), Complex::<f64>::new(3.0, 0.0)]
        ];

        let l = complex_cholesky(&a.view()).expect("Operation failed");

        // Verify L * L^H = A
        let lh = hermitian_transpose(&l.view());
        let llh = crate::complex::complex_matmul(&l.view(), &lh.view()).expect("Operation failed");

        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_relative_eq!(llh[[i, j]].re, a[[i, j]].re, epsilon = 1e-10);
                assert_relative_eq!(llh[[i, j]].im, a[[i, j]].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_complex_svd() {
        // Test with a simple 2x2 complex matrix
        let a = array![
            [Complex::<f64>::new(1.0, 1.0), Complex::<f64>::new(0.0, 0.0)],
            [Complex::<f64>::new(0.0, 0.0), Complex::<f64>::new(2.0, 0.0)]
        ];

        let svd = complex_svd(&a.view(), false).expect("Operation failed");
        assert_eq!(svd.u.shape(), &[2, 2]);
        assert_eq!(svd.s.shape(), &[2]);
        assert_eq!(svd.vh.shape(), &[2, 2]);

        // Test singular values are non-negative
        for s in svd.s.iter() {
            assert!(*s >= 0.0);
        }

        // Test full matrices option
        let svd_full = complex_svd(&a.view(), true).expect("Operation failed");
        assert_eq!(svd_full.u.shape(), &[2, 2]);
        assert_eq!(svd_full.s.shape(), &[2]);
        assert_eq!(svd_full.vh.shape(), &[2, 2]);

        // Test rectangular matrix (m > n)
        let b = array![
            [Complex::<f64>::new(1.0, 0.0), Complex::<f64>::new(0.0, 1.0)],
            [
                Complex::<f64>::new(0.0, -1.0),
                Complex::<f64>::new(2.0, 0.0)
            ],
            [Complex::<f64>::new(3.0, 0.0), Complex::<f64>::new(0.0, 0.0)]
        ];

        let svd_rect = complex_svd(&b.view(), false).expect("Operation failed");
        assert_eq!(svd_rect.u.shape(), &[3, 2]);
        assert_eq!(svd_rect.s.shape(), &[2]);
        assert_eq!(svd_rect.vh.shape(), &[2, 2]);

        // Test rectangular matrix (m < n)
        let c = array![
            [
                Complex::<f64>::new(1.0, 0.0),
                Complex::<f64>::new(0.0, 1.0),
                Complex::<f64>::new(2.0, 1.0)
            ],
            [
                Complex::<f64>::new(0.0, -1.0),
                Complex::<f64>::new(2.0, 0.0),
                Complex::<f64>::new(0.0, 3.0)
            ]
        ];

        let svd_wide = complex_svd(&c.view(), false).expect("Operation failed");
        assert_eq!(svd_wide.u.shape(), &[2, 2]);
        assert_eq!(svd_wide.s.shape(), &[2]);
        assert_eq!(svd_wide.vh.shape(), &[2, 3]);
    }

    /// Verify A ≈ Q T Q^H and Q^H Q ≈ I for a 3×3 complex matrix.
    #[test]
    fn test_complex_schur_3x3() {
        // A non-trivial 3×3 complex matrix with distinct eigenvalues
        let a = array![
            [
                Complex::<f64>::new(2.0, 1.0),
                Complex::<f64>::new(1.0, 0.5),
                Complex::<f64>::new(0.0, -1.0)
            ],
            [
                Complex::<f64>::new(0.0, 0.5),
                Complex::<f64>::new(3.0, -1.0),
                Complex::<f64>::new(1.0, 0.0)
            ],
            [
                Complex::<f64>::new(-1.0, 0.0),
                Complex::<f64>::new(0.5, 1.0),
                Complex::<f64>::new(1.0, 2.0)
            ]
        ];

        let (q, t) = complex_schur(&a.view()).expect("complex_schur failed on 3×3 matrix");
        let n = a.nrows();

        // --- T is upper triangular ---
        for i in 1..n {
            for j in 0..i {
                assert!(
                    t[[i, j]].norm() < 1e-8,
                    "T is not upper triangular at ({i},{j}): value = {}",
                    t[[i, j]]
                );
            }
        }

        // --- Q is unitary: Q^H Q ≈ I ---
        let qh = hermitian_transpose(&q.view());
        let qtq =
            crate::complex::complex_matmul(&qh.view(), &q.view()).expect("Q^H * Q matmul failed");

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j {
                    Complex::<f64>::one()
                } else {
                    Complex::<f64>::zero()
                };
                let diff = (qtq[[i, j]] - expected).norm();
                assert!(
                    diff < 1e-8,
                    "Q^H Q ≠ I at ({i},{j}): got {}, expected {expected}, diff = {diff:.2e}",
                    qtq[[i, j]]
                );
            }
        }

        // --- A ≈ Q T Q^H ---
        let qt = crate::complex::complex_matmul(&q.view(), &t.view()).expect("Q * T matmul failed");
        let qtqh = crate::complex::complex_matmul(&qt.view(), &qh.view())
            .expect("(Q*T) * Q^H matmul failed");

        for i in 0..n {
            for j in 0..n {
                let diff = (qtqh[[i, j]] - a[[i, j]]).norm();
                assert!(
                    diff < 1e-8,
                    "A ≠ Q T Q^H at ({i},{j}): reconstructed {}, original {}, diff = {diff:.2e}",
                    qtqh[[i, j]],
                    a[[i, j]]
                );
            }
        }
    }

    /// Verify Schur decomposition on a 2×2 diagonal complex matrix (trivial case).
    #[test]
    fn test_complex_schur_2x2_diagonal() {
        let a = array![
            [Complex::<f64>::new(3.0, 1.0), Complex::<f64>::zero()],
            [Complex::<f64>::zero(), Complex::<f64>::new(1.0, -2.0)]
        ];

        let (q, t) = complex_schur(&a.view()).expect("complex_schur failed on diagonal 2×2");
        let n = 2_usize;

        // T upper triangular
        assert!(
            t[[1, 0]].norm() < 1e-10,
            "T[1,0] should be zero for diagonal input, got {}",
            t[[1, 0]]
        );

        // Q unitary
        let qh = hermitian_transpose(&q.view());
        let qtq =
            crate::complex::complex_matmul(&qh.view(), &q.view()).expect("Q^H * Q matmul failed");

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j {
                    Complex::<f64>::one()
                } else {
                    Complex::<f64>::zero()
                };
                let diff = (qtq[[i, j]] - expected).norm();
                assert!(
                    diff < 1e-10,
                    "Q not unitary at ({i},{j}), diff = {diff:.2e}"
                );
            }
        }

        // A ≈ Q T Q^H
        let qt = crate::complex::complex_matmul(&q.view(), &t.view()).expect("Q*T matmul failed");
        let qtqh = crate::complex::complex_matmul(&qt.view(), &qh.view())
            .expect("(Q*T)*Q^H matmul failed");

        for i in 0..n {
            for j in 0..n {
                let diff = (qtqh[[i, j]] - a[[i, j]]).norm();
                assert!(diff < 1e-10, "A ≠ Q T Q^H at ({i},{j}), diff = {diff:.2e}");
            }
        }
    }
}
