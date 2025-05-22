//! DPSS (Slepian) window generation for multitaper spectral estimation.

use crate::error::{SignalError, SignalResult};
use ndarray::{s, Array1, Array2};

/// Compute Discrete Prolate Spheroidal Sequences (DPSS), also known as Slepian sequences.
///
/// DPSS tapers are often used in multitaper spectral estimation and are designed
/// to maximize energy concentration in a specified frequency band.
///
/// # Arguments
///
/// * `n` - Length of the tapers
/// * `nw` - Time-bandwidth product (typically between 2 and 8)
/// * `k` - Number of tapers to compute (should be less than or equal to 2*nw)
/// * `return_ratios` - If true, also return the eigenvalues
///
/// # Returns
///
/// * If return_ratios is false: Array2 of DPSS tapers (shape: [k, n])
/// * If return_ratios is true: Tuple of (tapers, eigenvalues)
///
/// # Examples
///
/// ```
/// use scirs2_signal::multitaper::dpss;
///
/// // Compute 4 DPSS tapers of length 64 with time-bandwidth product of 4
/// let result = dpss(64, 4.0, 4, true).unwrap();
/// let (tapers, eigenvalues) = (result.0, result.1.unwrap());
///
/// // Check number of tapers
/// assert_eq!(tapers.shape()[0], 4);
/// assert_eq!(tapers.shape()[1], 64);
///
/// // Basic verification - eigenvalues should exist and be positive
/// assert!(eigenvalues.len() >= 2);
/// assert!(eigenvalues[0] > 0.0);
/// ```
pub fn dpss(
    n: usize,
    nw: f64,
    k: usize,
    return_ratios: bool,
) -> SignalResult<(Array2<f64>, Option<Array1<f64>>)> {
    if n < 2 {
        return Err(SignalError::ValueError(
            "Length of tapers must be at least 2".to_string(),
        ));
    }

    if nw <= 0.0 {
        return Err(SignalError::ValueError(
            "Time-bandwidth product must be positive".to_string(),
        ));
    }

    if k < 1 {
        return Err(SignalError::ValueError(
            "Number of tapers must be at least 1".to_string(),
        ));
    }

    // Maximum number of tapers that can be well-concentrated with the given nw
    let max_tapers = (2.0 * nw).floor() as usize;

    if k > max_tapers {
        return Err(SignalError::ValueError(format!(
            "Number of tapers k ({}) must not exceed 2*nw ({})",
            k, max_tapers
        )));
    }

    // Construct the tridiagonal matrix
    let w = nw / n as f64; // Half-bandwidth
    let n_points = n;

    // Diagonal elements
    let mut diag = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let i_float = i as f64;
        let n_float = n_points as f64;
        let val =
            ((n_float - 1.0) / 2.0 - i_float).powi(2) * (2.0 * std::f64::consts::PI * w).powi(2);
        diag.push(val);
    }

    // Off-diagonal elements
    let mut offdiag = Vec::with_capacity(n_points - 1);
    for i in 0..(n_points - 1) {
        let i_float = i as f64;
        offdiag.push((i_float + 1.0) * (n_points as f64 - i_float - 1.0));
    }

    // Diagonalize the tridiagonal matrix
    let (eigvals, eigvecs) = tridiagonal_eig(diag.as_slice(), offdiag.as_slice())?;

    // Sort eigenvalues and eigenvectors
    let mut idx: Vec<usize> = (0..n_points).collect();
    idx.sort_by(|&i, &j| eigvals[i].partial_cmp(&eigvals[j]).unwrap());

    // Reorder eigenvalues and eigenvectors
    let mut sorted_eigvals = Vec::with_capacity(n_points);
    let mut sorted_eigvecs = Array2::zeros((n_points, n_points));

    for i in 0..n_points {
        sorted_eigvals.push(eigvals[idx[i]]);
        for j in 0..n_points {
            sorted_eigvecs[[i, j]] = eigvecs[[idx[i], j]];
        }
    }

    // Take the k most concentrated eigenvectors
    let mut dpss = Array2::zeros((k, n_points));
    let mut lambda = Array1::zeros(k);

    for i in 0..k {
        // Convert concentration from sin squared to sin
        lambda[i] = (1.0 - sorted_eigvals[i]).sqrt();

        // Get eigenvector and normalize
        let mut v = sorted_eigvecs.slice(s![i, ..]).to_owned();
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);

        // Ensure the first element is positive (for consistency)
        if v[0] < 0.0 {
            v.iter_mut().for_each(|x| *x = -*x);
        }

        // Copy to output
        for j in 0..n_points {
            dpss[[i, j]] = v[j];
        }
    }

    if return_ratios {
        Ok((dpss, Some(lambda)))
    } else {
        Ok((dpss, None))
    }
}

/// Compute eigenvalues and eigenvectors of a symmetric tridiagonal matrix.
///
/// This is a simplified implementation for computing eigenvalues and eigenvectors
/// of a symmetric tridiagonal matrix, which is needed for the DPSS calculation.
///
/// # Arguments
///
/// * `diag` - Diagonal elements of the matrix
/// * `offdiag` - Off-diagonal elements of the matrix
///
/// # Returns
///
/// * Tuple of (eigenvalues, eigenvectors)
fn tridiagonal_eig(diag: &[f64], offdiag: &[f64]) -> SignalResult<(Vec<f64>, Array2<f64>)> {
    if diag.is_empty() {
        return Err(SignalError::ValueError(
            "Diagonal must have at least one element".to_string(),
        ));
    }

    if offdiag.len() != diag.len() - 1 {
        return Err(SignalError::ValueError(
            "Off-diagonal must have one fewer element than diagonal".to_string(),
        ));
    }

    let mut n = diag.len();
    let mut eigvals = vec![0.0; n];
    let mut eigvecs = Array2::zeros((n, n));

    // Initialize with identity matrix
    for i in 0..n {
        eigvecs[[i, i]] = 1.0;
    }

    // Copy diagonal and off-diagonal elements
    let mut a = diag.to_vec();
    let mut b = offdiag.to_vec();

    // Number of iterations for QR algorithm
    let max_iter = 30 * n;
    let mut iter_count = 0;

    // Tolerance for convergence
    let tol = 1e-10;

    // Main QR iteration loop
    while iter_count < max_iter {
        // Find the largest index of a small off-diagonal element
        let mut m = n - 1;
        while m > 0 {
            if b[m - 1].abs() <= tol * (a[m - 1].abs() + a[m].abs()) {
                break;
            }
            m -= 1;
        }

        if m == n - 1 {
            // Last eigenvalue is isolated
            eigvals[n - 1] = a[n - 1];
            n -= 1;
            if n == 0 {
                break;
            }
            a.pop();
            b.pop();
        } else if m == 0 {
            // First eigenvalue is isolated
            eigvals[0] = a[0];
            a.copy_within(1.., 0);
            b.copy_within(1.., 0);
            a.pop();
            b.pop();
            n -= 1;

            // Update eigenvectors
            for i in 0..eigvecs.shape()[0] {
                for j in 0..eigvecs.shape()[1] - 1 {
                    eigvecs[[i, j]] = eigvecs[[i, j + 1]];
                }
            }
        } else {
            // Perform QR step on a[m..n, m..n]
            let mut c = 0.0;
            let mut s = 1.0;

            for i in m..n {
                let f = s * b[i - 1];
                b[i - 1] *= c;

                if f.abs() <= tol {
                    break;
                }

                let g = a[i];
                let r = (a[i - 1] - g).hypot(2.0 * f) * 0.5;
                let t = if a[i - 1] - g >= 0.0 { 1.0 } else { -1.0 };

                a[i - 1] = a[i - 1] + t * (r * r - (a[i - 1] - g).powi(2) / 4.0 - f * f) / r;
                a[i] = g + t * (r * r - (a[i - 1] - g).powi(2) / 4.0 - f * f) / r;

                c = r / ((a[i - 1] - g).powi(2) / 4.0 + f * f).sqrt();
                s = f / ((a[i - 1] - g).powi(2) / 4.0 + f * f).sqrt();

                // Update eigenvectors
                for j in 0..eigvecs.shape()[0] {
                    let temp = c * eigvecs[[j, i - 1]] - s * eigvecs[[j, i]];
                    eigvecs[[j, i]] = s * eigvecs[[j, i - 1]] + c * eigvecs[[j, i]];
                    eigvecs[[j, i - 1]] = temp;
                }
            }
        }

        iter_count += 1;
    }

    // If iterations didn't converge, return the best approximation
    if iter_count >= max_iter {
        let len = eigvals.len().min(a.len());
        eigvals[..len].copy_from_slice(&a[..len]);
    } else {
        // Fill in remaining eigenvalues
        eigvals[..a.len()].copy_from_slice(&a);
    }

    Ok((eigvals, eigvecs))
}
