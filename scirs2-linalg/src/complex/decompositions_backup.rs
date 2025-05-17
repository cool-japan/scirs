//! Complex matrix decompositions
//!
//! This module provides implementations of various matrix decompositions
//! for complex-valued matrices.

use ndarray::{Array1, Array2, ArrayView2};
use num_complex::Complex;
use num_traits::{Float, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};
use crate::complex::hermitian_transpose;
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
pub fn complex_lu<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<ComplexLUDecomposition<F>>
where
    F: Float + Sum + Debug,
{
    let (m, n) = (a.nrows(), a.ncols());
    let mut lu = a.to_owned();
    let mut piv: Vec<usize> = (0..m).collect();
    let mut sign = Complex::<f64>::one();
    
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
pub fn complex_qr<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<ComplexQRDecomposition<F>>
where
    F: Float + Sum + Debug,
{
    let (m, n) = (a.nrows(), a.ncols());
    let mut q = Array2::eye(m);
    let mut r = a.to_owned();
    
    // Gram-Schmidt orthogonalization
    for j in 0..n {
        // Normalize column j of R to get column j of Q
        let mut norm_sq = F::zero();
        for i in 0..m {
            norm_sq = norm_sq + r[[i, j]].norm_sqr();
        }
        
        if norm_sq < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix columns are linearly dependent".to_string(),
            ));
        }
        
        let norm = norm_sq.sqrt();
        
        // Set Q column
        for i in 0..m {
            q[[i, j]] = r[[i, j]] / Complex::<f64>::new(norm, F::zero());
        }
        
        // Set R diagonal element
        r[[j, j]] = Complex::<f64>::new(norm, F::zero());
        
        // Orthogonalize remaining columns
        for k in (j + 1)..n {
            let mut dot = Complex::<f64>::zero();
            for i in 0..m {
                dot = dot + q[[i, j]].conj() * r[[i, k]];
            }
            
            r[[j, k]] = dot;
            
            for i in 0..m {
                r[[i, k]] = r[[i, k]] - dot * q[[i, j]];
            }
        }
        
        // Zero out below diagonal
        for i in (j + 1)..m {
            r[[i, j]] = Complex::<f64>::zero();
        }
    }
    
    Ok(ComplexQRDecomposition { q, r })
}

/// Performs complex SVD decomposition
///
/// Decomposes A into U * S * V^H where:
/// - U is unitary (left singular vectors)
/// - S is diagonal with real, non-negative values
/// - V^H is unitary (conjugate transpose of right singular vectors)
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
            let mut sum = Complex::<f64>::zero();
            for l in 0..m {
                sum = sum + ah[[i, l]] * a[[l, j]];
            }
            aha[[i, j]] = sum;
        }
    }
    
    // Eigendecomposition of A^H * A
    let eig = complex_eigh(&aha.view())?;
    let mut eigenvalues = eig.eigenvalues.clone();
    let mut v = eig.eigenvectors.clone();
    
    // Sort eigenvalues and eigenvectors in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        eigenvalues[j].re.partial_cmp(&eigenvalues[i].re).unwrap()
    });
    
    let mut s = Array1::zeros(k);
    let mut vt = Array2::zeros((if full_matrices { n } else { k }, n));
    
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        if new_idx < k {
            s[new_idx] = eigenvalues[old_idx].re.sqrt();
        }
        if new_idx < vt.nrows() {
            for j in 0..n {
                vt[[new_idx, j]] = v[[j, old_idx]].conj();
            }
        }
    }
    
    // Compute U = A * V * S^(-1)
    let u_cols = if full_matrices { m } else { k };
    let mut u = Array2::zeros((m, u_cols));
    
    for j in 0..u_cols {
        if j < k && s[j] > F::epsilon() {
            for i in 0..m {
                let mut sum = Complex::<f64>::zero();
                for l in 0..n {
                    sum = sum + a[[i, l]] * v[[l, indices[j]]];
                }
                u[[i, j]] = sum / Complex::<f64>::new(s[j], F::zero());
            }
        } else if j >= k && full_matrices {
            // For full matrices, extend with orthogonal vectors
            // This would require implementing Gram-Schmidt or similar
            // For now, we'll leave these as zero
            // TODO: Implement proper orthogonalization
        }
    }
    
    Ok(ComplexSVDDecomposition { u, s, vh: vt })
}

/// Performs eigendecomposition of a complex matrix
///
/// Finds eigenvalues and eigenvectors such that A * v = λ * v
pub fn complex_eig<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<ComplexEigDecomposition<F>>
where
    F: Float + Sum + Debug,
{
    check_square(a, "matrix")?;
    let n = a.nrows();
    
    // Use QR algorithm with shifts for general complex matrices
    let max_iterations = 1000;
    let tolerance = F::from(1e-10).unwrap();
    
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
                first_elem / Complex::<f64>::new(first_elem.norm(), F::zero())
            } else {
                Complex::<f64>::one()
            };
            
            v[0] = v[0] + phase * Complex::<f64>::new(norm, F::zero());
            let v_norm_sq = v.iter().map(|x| x.norm_sqr()).sum();
            
            if v_norm_sq > tolerance {
                // Apply Householder transformation: H = I - 2vv^H/||v||^2
                for j in col..n {
                    let mut sum = Complex::<f64>::zero();
                    for i in 0..v.len() {
                        sum = sum + v[i].conj() * h[[i + col + 1, j]];
                    }
                    let factor = Complex::<f64>::new(F::from(2.0).unwrap() / v_norm_sq, F::zero()) * sum;
                    
                    for i in 0..v.len() {
                        h[[i + col + 1, j]] = h[[i + col + 1, j]] - factor * v[i];
                    }
                }
                
                // Apply from right: H' = H'H
                for i in 0..n {
                    let mut sum = Complex::<f64>::zero();
                    for j in 0..v.len() {
                        sum = sum + h[[i, j + col + 1]] * v[j];
                    }
                    let factor = Complex::<f64>::new(F::from(2.0).unwrap() / v_norm_sq, F::zero()) * sum;
                    
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
            let mut shift = Complex::<f64>::zero();
            if n >= 2 {
                let a11 = h[[n - 2, n - 2]];
                let a12 = h[[n - 2, n - 1]];
                let a21 = h[[n - 1, n - 2]];
                let a22 = h[[n - 1, n - 1]];
                
                let trace = a11 + a22;
                let det = a11 * a22 - a12 * a21;
                let discriminant = trace * trace - Complex::<f64>::new(F::from(4.0).unwrap(), F::zero()) * det;
                let sqrt_disc = discriminant.sqrt();
                
                let eig1 = (trace + sqrt_disc) / Complex::<f64>::new(F::from(2.0).unwrap(), F::zero());
                let eig2 = (trace - sqrt_disc) / Complex::<f64>::new(F::from(2.0).unwrap(), F::zero());
                
                // Choose eigenvalue closer to a22
                shift = if (eig1 - a22).norm() < (eig2 - a22).norm() { eig1 } else { eig2 };
            }
            
            // QR step with shift
            let mut h_shifted = h.clone();
            for i in 0..n {
                h_shifted[[i, i]] = h_shifted[[i, i]] - shift;
            }
            
            let qr = complex_qr(&h_shifted.view())?;
            h = crate::complex::complex_matmul(&qr.r.view(), &qr.q.view()).unwrap();
            
            for i in 0..n {
                h[[i, i]] = h[[i, i]] + shift;
            }
            
            // Update eigenvector matrix
            q_total = crate::complex::complex_matmul(&q_total.view(), &qr.q.view()).unwrap();
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
pub fn complex_eigh<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<ComplexEigDecomposition<F>>
where
    F: Float + Sum + Debug,
{
    check_square(a, "matrix")?;
    let n = a.nrows();
    
    // Verify matrix is Hermitian
    let tolerance = F::from(1e-10).unwrap();
    
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
                first_elem / Complex::<f64>::new(first_elem.norm(), F::zero())
            } else {
                Complex::<f64>::one()
            };
            
            v[0] = v[0] + phase * Complex::<f64>::new(norm, F::zero());
            let v_norm_sq: F = v.iter().map(|x| x.norm_sqr()).sum();
            
            if v_norm_sq > tolerance {
                // Apply Householder transformation from left and right
                // T = H * T * H^H where H = I - 2vv^H/||v||^2
                
                // First: compute T * v
                let mut tv = Array1::zeros(n);
                for i in 0..n {
                    let mut sum = Complex::<f64>::zero();
                    for j in 0..v.len() {
                        sum = sum + t[[i, j + k + 1]] * v[j];
                    }
                    tv[i] = sum;
                }
                
                // Update T = T - (2/||v||^2) * T * v * v^H
                let factor = Complex::<f64>::new(F::from(2.0).unwrap() / v_norm_sq, F::zero());
                for i in 0..n {
                    for j in 0..v.len() {
                        t[[i, j + k + 1]] = t[[i, j + k + 1]] - factor * tv[i] * v[j].conj();
                    }
                }
                
                // Apply from left: v^H * T
                let mut vt = Array1::zeros(n);
                for j in 0..n {
                    let mut sum = Complex::<f64>::zero();
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
                    let mut sum = Complex::<f64>::zero();
                    for j in 0..v.len() {
                        sum = sum + q_total[[i, j + k + 1]] * v[j];
                    }
                    qv[i] = sum;
                }
                
                for i in 0..n {
                    for j in 0..v.len() {
                        q_total[[i, j + k + 1]] = q_total[[i, j + k + 1]] - factor * qv[i] * v[j].conj();
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
        let d = (diagonal[n - 2] - diagonal[n - 1]) / F::from(2.0).unwrap();
        let sign = if d >= F::zero() { F::one() } else { -F::one() };
        let shift = diagonal[n - 1] - subdiagonal[n - 2].powi(2) / (d + sign * (d.powi(2) + subdiagonal[n - 2].powi(2)).sqrt());
        
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
                
                diagonal[k] = c * c * d1 + F::from(2.0).unwrap() * c * sn * e + sn * sn * d2;
                diagonal[k + 1] = sn * sn * d1 - F::from(2.0).unwrap() * c * sn * e + c * c * d2;
                
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
                    q_total[[i, k]] = Complex::<f64>::new(c, F::zero()) * q_ik + Complex::<f64>::new(sn, F::zero()) * q_ik1;
                    q_total[[i, k + 1]] = Complex::<f64>::new(-sn, F::zero()) * q_ik + Complex::<f64>::new(c, F::zero()) * q_ik1;
                }
            }
        }
    }
    
    // Convert real eigenvalues to complex
    let mut eigenvalues = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = Complex::<f64>::new(diagonal[i], F::zero());
    }
    
    Ok(ComplexEigDecomposition {
        eigenvalues,
        eigenvectors: q_total,
    })
}

/// Performs Cholesky decomposition of a positive-definite Hermitian matrix
///
/// Decomposes A into L * L^H where L is lower triangular
pub fn complex_cholesky<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<Array2<Complex<F>>>
where
    F: Float + Sum + Debug,
{
    check_square(a, "matrix")?;
    let n = a.nrows();
    
    // Verify matrix is Hermitian
    let ah = hermitian_transpose(a);
    let tolerance = F::from(1e-10).unwrap();
    
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
            let mut sum = Complex::<f64>::zero();
            
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
                
                l[[i, j]] = Complex::<f64>::new(diag.sqrt(), F::zero());
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

/// Performs Schur decomposition of a complex matrix
///
/// Decomposes A into Q * T * Q^H where T is upper triangular
pub fn complex_schur<F>(a: &ArrayView2<Complex<F>>) -> LinalgResult<(Array2<Complex<F>>, Array2<Complex<F>>)>
where
    F: Float + Sum + Debug,
{
    check_square(a, "matrix")?;
    
    // This is a placeholder for full Schur decomposition
    // A complete implementation would use QR algorithm with shifts
    Err(LinalgError::NotImplementedError(
        "Complex Schur decomposition not yet implemented".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_complex_lu() {
        let a = array![
            [Complex::<f64>::new(2.0, 1.0), Complex::<f64>::new(1.0, 0.0)],
            [Complex::<f64>::new(1.0, 0.0), Complex::<f64>::new(2.0, -1.0)]
        ];
        
        let lu_result = complex_lu(&a.view()).unwrap();
        
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
        
        // Apply permutation and verify
        // TODO: Add proper verification
    }
    
    #[test]
    fn test_complex_qr() {
        let a = array![
            [Complex::<f64>::new(1.0, 0.0), Complex::<f64>::new(1.0, 1.0)],
            [Complex::<f64>::new(0.0, 1.0), Complex::<f64>::new(1.0, 0.0)]
        ];
        
        let qr_result = complex_qr(&a.view()).unwrap();
        
        // Verify Q is unitary: Q^H * Q = I
        let qh = hermitian_transpose(&qr_result.q.view());
        let should_be_i = crate::complex::complex_matmul(&qh.view(), &qr_result.q.view()).unwrap();
        
        let n = a.nrows();
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { Complex::<f64>::one() } else { Complex::<f64>::zero() };
                assert!(
                    (should_be_i[[i, j]] - expected).norm() < 1e-10,
                    "Q is not unitary at position ({}, {})", i, j
                );
            }
        }
        
        // Verify Q * R = A
        let qr = crate::complex::complex_matmul(&qr_result.q.view(), &qr_result.r.view()).unwrap();
        
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_relative_eq!(qr[[i, j]].re, a[[i, j]].re, epsilon = 1e-10);
                assert_relative_eq!(qr[[i, j]].im, a[[i, j]].im, epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_complex_cholesky() {
        // Create a positive-definite Hermitian matrix
        let a = array![
            [Complex::<f64>::new(3.0, 0.0), Complex::<f64>::new(1.0, -1.0)],
            [Complex::<f64>::new(1.0, 1.0), Complex::<f64>::new(3.0, 0.0)]
        ];
        
        let l = complex_cholesky(&a.view()).unwrap();
        
        // Verify L * L^H = A
        let lh = hermitian_transpose(&l.view());
        let llh = crate::complex::complex_matmul(&l.view(), &lh.view()).unwrap();
        
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_relative_eq!(llh[[i, j]].re, a[[i, j]].re, epsilon = 1e-10);
                assert_relative_eq!(llh[[i, j]].im, a[[i, j]].im, epsilon = 1e-10);
            }
        }
    }
}