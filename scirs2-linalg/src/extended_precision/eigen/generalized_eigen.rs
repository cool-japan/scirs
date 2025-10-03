//! Core eigenvalue algorithms for extended precision computations
//!
//! This module contains the fundamental algorithms for eigenvalue computation
//! including Hessenberg reduction, tridiagonalization, and QR algorithm implementations.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, One, Zero};

// Helper function for Hessenberg reduction
#[allow(dead_code)]
pub(super) fn hessenberg_reduction<I>(mut a: Array2<I>) -> Array2<I>
where
    I: Float + Zero + One + Copy + std::ops::AddAssign,
{
    let n = a.nrows();

    for k in 0..n - 2 {
        let mut scale = I::zero();

        // Find scale to avoid underflow/overflow
        for i in k + 1..n {
            scale += a[[i, k]].abs();
        }

        if scale <= I::epsilon() {
            continue; // Skip transformation
        }

        let mut h = I::zero();
        for i in k + 1..n {
            a[[i, k]] = a[[i, k]] / scale;
            h += a[[i, k]] * a[[i, k]];
        }

        let f = a[[k + 1, k]];
        let g = if f >= I::zero() { -h.sqrt() } else { h.sqrt() };

        h = h - f * g;
        a[[k + 1, k]] = f - g;

        for j in k + 1..n {
            let mut f = I::zero();
            for i in k + 1..n {
                f += a[[i, k]] * a[[i, j]];
            }
            f = f / h;

            for i in k + 1..n {
                a[[i, j]] = a[[i, j]] - f * a[[i, k]];
            }
        }

        for i in 0..n {
            let mut f = I::zero();
            for j in k + 1..n {
                f += a[[j, k]] * a[[i, j]];
            }
            f = f / h;

            for j in k + 1..n {
                a[[i, j]] = a[[i, j]] - f * a[[j, k]];
            }
        }

        a[[k + 1, k]] = scale * a[[k + 1, k]];

        for i in k + 2..n {
            a[[i, k]] = I::zero();
        }
    }

    a
}

// Helper function to tridiagonalize a symmetric matrix
#[allow(dead_code)]
pub(super) fn tridiagonalize<I>(mut a: Array2<I>) -> Array2<I>
where
    I: Float + Zero + One + Copy + std::ops::AddAssign + std::ops::SubAssign + std::ops::DivAssign,
{
    let n = a.nrows();

    for k in 0..n - 2 {
        let mut scale = I::zero();

        for i in k + 1..n {
            scale += a[[i, k]].abs();
        }

        if scale <= I::epsilon() {
            continue;
        }

        let mut h = I::zero();
        for i in k + 1..n {
            a[[i, k]] /= scale;
            h += a[[i, k]] * a[[i, k]];
        }

        let f = a[[k + 1, k]];
        let g = if f >= I::zero() { -h.sqrt() } else { h.sqrt() };

        h -= f * g;
        a[[k + 1, k]] = f - g;

        for j in k + 1..n {
            let mut f = I::zero();
            for i in k + 1..n {
                f += a[[i, k]] * a[[i, j]];
            }
            f /= h;

            for i in k + 1..n {
                a[[i, j]] = a[[i, j]] - f * a[[i, k]];
            }
        }

        for i in 0..n {
            let mut f = I::zero();
            for j in k + 1..n {
                f += a[[j, k]] * a[[i, j]];
            }
            f /= h;

            for j in k + 1..n {
                a[[i, j]] = a[[i, j]] - f * a[[j, k]];
            }
        }

        a[[k + 1, k]] = scale * a[[k + 1, k]];

        for i in k + 2..n {
            a[[i, k]] = I::zero();
            a[[k, i]] = I::zero();
        }
    }

    // Make the matrix explicitly tridiagonal
    for i in 0..n {
        for j in 0..n {
            if (i > 0 && j < i - 1) || j > i + 1 {
                a[[i, j]] = I::zero();
            }
        }
    }

    a
}

// Helper function to tridiagonalize a symmetric matrix and return the transformation matrix
#[allow(dead_code)]
pub(super) fn tridiagonalize_with_transform<I>(a: Array2<I>) -> (Array2<I>, Array2<I>)
where
    I: Float + Zero + One + Copy + std::ops::AddAssign + std::ops::SubAssign + std::ops::DivAssign,
{
    let n = a.nrows();
    let mut a_copy = a.clone();
    let mut q = Array2::eye(n);

    for k in 0..n - 2 {
        let mut scale = I::zero();

        for i in k + 1..n {
            scale += a_copy[[i, k]].abs();
        }

        if scale <= I::epsilon() {
            continue;
        }

        // Create Householder vector
        let mut v = Array1::zeros(n - k - 1);
        for i in 0..v.len() {
            v[i] = a_copy[[i + k + 1, k]] / scale;
        }

        let mut h = I::zero();
        for i in 0..v.len() {
            h += v[i] * v[i];
        }

        let f = v[0];
        let g = if f >= I::zero() { -h.sqrt() } else { h.sqrt() };

        h -= f * g;
        v[0] = f - g;

        // Apply Householder reflection to A
        for j in k + 1..n {
            let mut f = I::zero();
            for i in 0..v.len() {
                f += v[i] * a_copy[[i + k + 1, j]];
            }
            f /= h;

            for i in 0..v.len() {
                a_copy[[i + k + 1, j]] -= f * v[i];
            }
        }

        for i in 0..n {
            let mut f = I::zero();
            for j in 0..v.len() {
                f += v[j] * a_copy[[i, j + k + 1]];
            }
            f /= h;

            for j in 0..v.len() {
                a_copy[[i, j + k + 1]] -= f * v[j];
            }
        }

        // Update the transformation matrix
        for i in 0..n {
            let mut f = I::zero();
            for j in 0..v.len() {
                f += v[j] * q[[i, j + k + 1]];
            }
            f /= h;

            for j in 0..v.len() {
                q[[i, j + k + 1]] -= f * v[j];
            }
        }
    }

    // Make a_copy explicitly tridiagonal
    let mut a_tri = Array2::zeros((n, n));
    for i in 0..n {
        a_tri[[i, i]] = a_copy[[i, i]];
        if i > 0 {
            a_tri[[i, i - 1]] = a_copy[[i, i - 1]];
            a_tri[[i - 1, i]] = a_copy[[i - 1, i]];
        }
    }

    (a_tri, q)
}

// QR algorithm for computing eigenvalues of a Hessenberg matrix
#[allow(dead_code)]
pub(super) fn qr_algorithm<I>(
    mut a: Array2<I>,
    maxiter: usize,
    tol: I,
) -> Array1<scirs2_core::numeric::Complex<I>>
where
    I: Float + Zero + One + Copy + std::fmt::Debug + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = a.nrows();
    let mut eigenvalues = Array1::zeros(n);

    // This is a simplified implementation of the QR algorithm
    // A full implementation would use the Francis QR step with implicit shifts

    let mut m = n;
    let mut iter_count = 0;

    while m > 1 && iter_count < maxiter {
        // Check for small off-diagonal element
        let mut l = 0;
        for i in 0..m - 1 {
            if a[[i + 1, i]].abs() < tol * (a[[i, i]].abs() + a[[i + 1, i + 1]].abs()) {
                a[[i + 1, i]] = I::zero();
            }
            if a[[i + 1, i]] == I::zero() {
                l = i + 1;
            }
        }

        if l == m - 1 {
            // 1x1 block - real eigenvalue
            eigenvalues[m - 1] = scirs2_core::numeric::Complex::new(a[[m - 1, m - 1]], I::zero());
            m -= 1;
        } else {
            // Apply QR iteration to the active submatrix
            let mut q = Array2::eye(m);
            let mut r = a.slice(scirs2_core::ndarray::s![0..m, 0..m]).to_owned();

            // QR factorization
            for k in 0..m - 1 {
                if r[[k + 1, k]].abs() > I::epsilon() {
                    let alpha = r[[k, k]];
                    let beta = r[[k + 1, k]];
                    let r_norm = (alpha * alpha + beta * beta).sqrt();

                    let c = alpha / r_norm;
                    let s = -beta / r_norm;

                    // Apply Givens rotation to r
                    for j in k..m {
                        let temp = c * r[[k, j]] - s * r[[k + 1, j]];
                        r[[k + 1, j]] = s * r[[k, j]] + c * r[[k + 1, j]];
                        r[[k, j]] = temp;
                    }

                    // Accumulate the rotations in q
                    for i in 0..m {
                        let temp = c * q[[i, k]] - s * q[[i, k + 1]];
                        q[[i, k + 1]] = s * q[[i, k]] + c * q[[i, k + 1]];
                        q[[i, k]] = temp;
                    }
                }
            }

            // Compute R*Q
            let mut rq = Array2::zeros((m, m));
            for i in 0..m {
                for j in 0..m {
                    for k in 0..m {
                        rq[[i, j]] += r[[i, k]] * q[[k, j]];
                    }
                }
            }

            // Update the active submatrix
            for i in 0..m {
                for j in 0..m {
                    a[[i, j]] = rq[[i, j]];
                }
            }

            iter_count += 1;

            // Check if we've done too many iterations
            if iter_count >= maxiter {
                // In a full implementation, we would use a different strategy here
                // For now, just extract approximate eigenvalues
                for i in 0..m {
                    eigenvalues[i] = scirs2_core::numeric::Complex::new(a[[i, i]], I::zero());
                }
                break;
            }
        }
    }

    // Handle any remaining 1x1 blocks
    for i in 0..m {
        eigenvalues[i] = scirs2_core::numeric::Complex::new(a[[i, i]], I::zero());
    }

    eigenvalues
}

// QR algorithm for symmetric tridiagonal matrices
#[allow(dead_code)]
pub(super) fn qr_algorithm_symmetric<I>(a: Array2<I>, maxiter: usize, tol: I) -> Array1<I>
where
    I: Float + Zero + One + Copy + std::fmt::Debug + std::ops::AddAssign + std::ops::SubAssign,
{
    let n = a.nrows();
    let mut d = Array1::zeros(n); // Diagonal elements
    let mut e = Array1::zeros(n - 1); // Off-diagonal elements

    // Extract diagonal and off-diagonal elements from tridiagonal matrix
    for i in 0..n {
        d[i] = a[[i, i]];
        if i < n - 1 {
            e[i] = a[[i, i + 1]];
        }
    }

    // Apply implicit QL algorithm for symmetric tridiagonal matrices
    // This is more stable than explicit QR for eigenvalues

    for l in 0..n {
        let mut _iter = 0;
        loop {
            // Find small off-diagonal element
            let mut m = n - 1;
            for i in l..n - 1 {
                let dd = d[i].abs() + d[i + 1].abs();
                if e[i].abs() <= tol * dd {
                    m = i;
                    break;
                }
            }

            if m == l {
                break; // Converged for this eigenvalue - e[l] is small
            }

            _iter += 1;
            if _iter > maxiter {
                break; // Max iterations reached
            }

            // Form shift
            let g = (d[l + 1] - d[l]) / (I::from(2.0).unwrap() * e[l]);
            let mut r = (g * g + I::one()).sqrt();
            let mut g = d[m] - d[l] + e[l] / (g + if g >= I::zero() { r } else { -r });

            let mut s = I::one();
            let mut c = I::one();
            let mut p = I::zero();

            // Perform the transformation
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];

                if f.abs() >= g.abs() {
                    c = g / f;
                    r = (c * c + I::one()).sqrt();
                    if i + 1 < n - 1 {
                        e[i + 1] = f * r;
                    }
                    s = I::one() / r;
                    c = c * s;
                } else {
                    s = f / g;
                    r = (s * s + I::one()).sqrt();
                    if i + 1 < n - 1 {
                        e[i + 1] = g * r;
                    }
                    c = I::one() / r;
                    s = s * c;
                }

                g = d[i + 1] - p;
                r = (d[i] - g) * s + I::from(2.0).unwrap() * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }

            d[l] -= p;
            if l < n - 1 {
                e[l] = g;
            }
            if m < n - 1 {
                e[m] = I::zero();
            }
        }
    }

    // Sort eigenvalues in ascending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        if d[i] < d[j] {
            std::cmp::Ordering::Less
        } else if d[i] > d[j] {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });

    let mut sorted_d = Array1::zeros(n);
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        sorted_d[new_idx] = d[old_idx];
    }

    sorted_d
}

// QR algorithm for symmetric tridiagonal matrices with eigenvector computation
#[allow(dead_code)]
pub(super) fn qr_algorithm_symmetric_with_vectors<I>(
    a: Array2<I>,
    q: Array2<I>,
    max_iter: usize,
    tol: I,
) -> (Array1<I>, Array2<I>)
where
    I: Float
        + Zero
        + One
        + Copy
        + std::fmt::Debug
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::DivAssign
        + 'static,
{
    let n = a.nrows();
    let mut d = Array1::zeros(n); // Diagonal elements
    let mut e = Array1::zeros(n); // Off-diagonal elements
    let mut z = q.clone(); // Eigenvector matrix (starts as the tridiagonalization transform)

    // Extract diagonal and off-diagonal elements
    for i in 0..n {
        d[i] = a[[i, i]];
        if i < n - 1 {
            e[i] = a[[i, i + 1]];
        }
    }

    // Apply QR algorithm specialized for symmetric tridiagonal matrices
    for _ in 0..max_iter {
        // Check for convergence
        let mut converged = true;
        for i in 0..n - 1 {
            if e[i].abs() > tol * (d[i].abs() + d[i + 1].abs()) {
                converged = false;
                break;
            }
        }

        if converged {
            break;
        }

        // Find indices for the submatrix to work on
        let mut m = n - 1;
        while m > 0 {
            if e[m - 1].abs() <= tol * (d[m - 1].abs() + d[m].abs()) {
                break;
            }
            m -= 1;
        }

        if m == n - 1 {
            continue; // Already converged for this block
        }

        // Find the extent of the unreduced submatrix
        let mut l = m;
        while l > 0 {
            if e[l - 1].abs() <= tol * (d[l - 1].abs() + d[l].abs()) {
                break;
            }
            l -= 1;
        }

        // Apply implicit QR step to the submatrix
        for i in l..m {
            let h = d[i + 1] - d[i];
            let t = if h.abs() < tol {
                I::one()
            } else {
                I::from(2.0).unwrap() * e[i] / h
            };

            let r = (t * t + I::one()).sqrt();
            let c = I::one() / r;
            let s = t * c;

            // Apply Givens rotation to d and e
            if i > l {
                e[i - 1] = s * e[i - 1] + c * e[i];
            }

            let oldc = c;
            let olds = s;

            // Update diagonal elements
            let c2 = oldc * oldc;
            let s2 = olds * olds;
            let cs = oldc * olds;

            let temp_i = d[i];
            let temp_ip1 = d[i + 1];
            d[i] = c2 * temp_i + s2 * temp_ip1 - I::from(2.0).unwrap() * cs * e[i];
            d[i + 1] = s2 * temp_i + c2 * temp_ip1 + I::from(2.0).unwrap() * cs * e[i];

            // Update off-diagonal elements
            if i < m - 1 {
                let temp = e[i + 1];
                e[i + 1] = oldc * temp;
                e[i] = olds * temp;
            } else {
                e[i] = I::zero();
            }

            // Update eigenvectors
            for k in 0..n {
                let t1 = z[[k, i]];
                let t2 = z[[k, i + 1]];
                z[[k, i]] = oldc * t1 - olds * t2;
                z[[k, i + 1]] = olds * t1 + oldc * t2;
            }
        }
    }

    // Sort eigenvalues and eigenvectors
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));

    let mut sorted_d = Array1::zeros(n);
    let mut sorted_z = Array2::zeros((n, n));

    for (idx, &i) in indices.iter().enumerate() {
        sorted_d[idx] = d[i];
        for j in 0..n {
            sorted_z[[j, idx]] = z[[j, i]];
        }
    }

    // Normalize eigenvectors
    for j in 0..n {
        let mut norm = I::zero();
        for i in 0..n {
            norm += sorted_z[[i, j]] * sorted_z[[i, j]];
        }
        norm = norm.sqrt();

        if norm > I::epsilon() {
            for i in 0..n {
                sorted_z[[i, j]] /= norm;
            }
        }
    }

    (sorted_d, sorted_z)
}
