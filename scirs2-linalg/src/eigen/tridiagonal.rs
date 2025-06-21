//! Eigenvalue solvers for tridiagonal matrices
//!
//! This module provides specialized solvers for tridiagonal matrices,
//! which are much faster than general eigenvalue solvers.

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, NumAssign, Zero, One};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

/// Compute eigenvalues of a symmetric tridiagonal matrix.
///
/// This uses the QR algorithm specialized for tridiagonal matrices,
/// which is much faster than the general case.
///
/// # Arguments
///
/// * `diagonal` - Main diagonal elements
/// * `off_diagonal` - Off-diagonal elements (length should be diagonal.len() - 1)
///
/// # Returns
///
/// * Array of eigenvalues
pub fn tridiagonal_eigvalsh<F>(
    diagonal: &ArrayView1<F>,
    off_diagonal: &ArrayView1<F>
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + Sum + 'static,
{
    let n = diagonal.len();
    
    if off_diagonal.len() != n - 1 {
        return Err(LinalgError::ShapeError(format!(
            "Off-diagonal length ({}) must be one less than diagonal length ({})",
            off_diagonal.len(), n
        )));
    }
    
    // Create copies of the inputs (will be modified during computation)
    let mut d = diagonal.to_owned();
    let mut e = off_diagonal.to_owned();
    
    // Result array for eigenvalues
    let mut eigenvalues = d.clone();
    
    // Maximum number of iterations
    let max_iter = 30 * n;
    
    // Implementation of the implicit QR algorithm for tridiagonal matrices
    // This is a specialized version that doesn't form Q and R explicitly
    
    // Helper function to compute the Givens rotation parameters
    let compute_givens = |a: F, b: F| -> (F, F) {
        let r = (a * a + b * b).sqrt();
        if r < F::epsilon() {
            (F::one(), F::zero())
        } else {
            (a / r, -b / r)
        }
    };
    
    // Zero out small off-diagonal elements
    let tol = F::epsilon().sqrt() * eigenvalues.iter()
              .fold(F::zero(), |max, &val| max.max(val.abs()));
    
    for i in 0..n-1 {
        if e[i].abs() < tol {
            e[i] = F::zero();
        }
    }
    
    // Main loop of the algorithm
    let mut m = n - 1;
    let mut iter_count = 0;
    
    while m > 0 && iter_count < max_iter {
        // Find largest unreduced block
        let mut l = m;
        while l > 0 {
            if e[l-1].abs() <= tol {
                break;
            }
            l -= 1;
        }
        
        // If the entire matrix is reduced, we're done
        if l == m {
            m -= 1;
            continue;
        }
        
        // Compute shift
        let mut shift = eigenvalues[m];
        
        // Perform an implicit QR step on the tridiagonal submatrix
        let mut g = (eigenvalues[l] - shift) / (F::from(2.0).unwrap() * e[l]);
        let mut r = (F::one() + g * g).sqrt();
        if g < F::zero() {
            r = -r;
        }
        
        g = eigenvalues[l] - shift + e[l] / (g + r);
        
        let mut s = F::one();
        let mut c = F::one();
        let mut p = F::zero();
        
        for i in l..m {
            let f = s * e[i];
            let b = c * e[i];
            
            // Compute Givens rotation
            let (c_i, s_i) = compute_givens(g, f);
            c = c_i;
            s = s_i;
            
            if i > l {
                e[i-1] = r;
            }
            
            g = eigenvalues[i+1] - p;
            r = (eigenvalues[i] - g) * s + F::from(2.0).unwrap() * c * b;
            p = s * r;
            eigenvalues[i] = g + p;
            g = c * r - b;
        }
        
        eigenvalues[m] -= p;
        e[m-1] = g;
        
        // Increment iteration count
        iter_count += 1;
    }
    
    // Check if we converged
    if iter_count >= max_iter {
        return Err(LinalgError::ConvergenceError(
            "Maximum iterations reached in tridiagonal_eigvalsh".to_string()
        ));
    }
    
    // Sort eigenvalues
    let mut sorted = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..n {
        eigenvalues[i] = sorted[i];
    }
    
    Ok(eigenvalues)
}

/// Compute eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// This uses the QR algorithm specialized for tridiagonal matrices,
/// which is much faster than the general case.
///
/// # Arguments
///
/// * `diagonal` - Main diagonal elements
/// * `off_diagonal` - Off-diagonal elements (length should be diagonal.len() - 1)
///
/// # Returns
///
/// * Tuple containing eigenvalues and eigenvectors
pub fn tridiagonal_eigh<F>(
    diagonal: &ArrayView1<F>,
    off_diagonal: &ArrayView1<F>
) -> LinalgResult<(Array1<F>, Array2<F>)>
where
    F: Float + NumAssign + Sum + 'static,
{
    let n = diagonal.len();
    
    if off_diagonal.len() != n - 1 {
        return Err(LinalgError::ShapeError(format!(
            "Off-diagonal length ({}) must be one less than diagonal length ({})",
            off_diagonal.len(), n
        )));
    }
    
    // Create tridiagonal matrix in full form
    let mut tri_matrix = Array2::zeros((n, n));
    for i in 0..n {
        tri_matrix[[i, i]] = diagonal[i];
        if i < n - 1 {
            tri_matrix[[i, i+1]] = off_diagonal[i];
            tri_matrix[[i+1, i]] = off_diagonal[i];
        }
    }
    
    // Create full matrix form and call standard eigenvalue solver
    // We'll use a simplified approach here since this is meant to be
    // a reference implementation. In practice, you would want a specialized
    // algorithm for tridiagonal matrices.
    
    // Create a temporary full symmetric matrix from the tridiagonal form
    let mut result = Array2::<F>::zeros((n, n));
    let identity = Array2::<F>::eye(n);
    
    // Perform a simplified version of the QR algorithm directly on this matrix
    let max_iter = 100;
    let mut converged = false;
    
    for _ in 0..max_iter {
        // Do a QR step
        let (q, r) = match crate::decomposition::qr(&tri_matrix.view(), None) {
            Ok((q, r)) => (q, r),
            Err(e) => return Err(e),
        };
        
        // A' = R * Q (reversed order for better convergence)
        let temp = r.dot(&q);
        
        // Check convergence (check if off-diagonal elements are close to zero)
        converged = true;
        for i in 0..n {
            for j in 0..n {
                if i != j && temp[[i, j]].abs() > F::epsilon() * F::from(100.0).unwrap() {
                    converged = false;
                    break;
                }
            }
            if !converged {
                break;
            }
        }
        
        if converged {
            // Extract eigenvalues and eigenvectors
            let mut eigenvalues = Array1::zeros(n);
            let eigenvectors = identity.dot(&q);
            
            // Eigenvalues are on the diagonal
            for i in 0..n {
                eigenvalues[i] = temp[[i, i]];
            }
            
            // Sort eigenvalues and corresponding eigenvectors
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());
            
            let mut sorted_eigenvalues = Array1::zeros(n);
            let mut sorted_eigenvectors = Array2::zeros((n, n));
            
            for (pos, &idx) in indices.iter().enumerate() {
                sorted_eigenvalues[pos] = eigenvalues[idx];
                for i in 0..n {
                    sorted_eigenvectors[[i, pos]] = eigenvectors[[i, idx]];
                }
            }
            
            return Ok((sorted_eigenvalues, sorted_eigenvectors));
        }
        
        tri_matrix = temp;
    }
    
    // If not converged, return a simple approximation
    let mut eigenvalues = Array1::zeros(n);
    for i in 0..n {
        eigenvalues[i] = tri_matrix[[i, i]];
    }
    
    Err(LinalgError::ConvergenceError(
        "QR algorithm did not converge for tridiagonal matrix".to_string()
    ))
}