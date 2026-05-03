//! Internal shared utilities for the random_features module.
//! These are `pub(super)` / `pub(crate)` and not part of the public API.

use crate::error::InterpolateError;

/// Solve `Ax = b` for symmetric positive-definite `A` via Cholesky decomposition.
/// Falls back to conjugate gradient if the matrix is near-singular.
pub(crate) fn cholesky_solve_vec(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, InterpolateError> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s < 0.0 {
                    return conjugate_gradient(a, b);
                }
                l[i][j] = s.sqrt().max(1e-300);
            } else {
                l[i][j] = s / l[j][j].max(1e-300);
            }
        }
    }

    // Forward substitution: Ly = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i].max(1e-300);
    }

    // Backward substitution: Lᵀx = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i].max(1e-300);
    }

    Ok(x)
}

/// Conjugate gradient fallback for near-singular systems.
pub(super) fn conjugate_gradient(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, InterpolateError> {
    let n = a.len();
    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old: f64 = r.iter().map(|v| v * v).sum();

    for _ in 0..(n * 10) {
        let ap: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| a[i][j] * p[j]).sum())
            .collect();
        let pap: f64 = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
        if pap.abs() < 1e-300 {
            break;
        }
        let alpha = rs_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new: f64 = r.iter().map(|v| v * v).sum();
        if rs_new.sqrt() < 1e-10 {
            break;
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Ok(x)
}
