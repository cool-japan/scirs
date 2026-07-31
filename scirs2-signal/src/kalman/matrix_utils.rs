//! Matrix utility functions for Kalman filter implementations.
//!
//! Provides basic matrix operations using `Vec<Vec<f64>>` representation
//! without external dependencies beyond the standard library.

use crate::error::{SignalError, SignalResult};

/// Multiply two matrices A (m×k) and B (k×n) -> C (m×n)
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    let m = a.len();
    if m == 0 {
        return Err(SignalError::ValueError("Empty matrix A".to_string()));
    }
    let k = a[0].len();
    if b.len() != k {
        return Err(SignalError::ValueError(format!(
            "Incompatible dimensions: A is {}×{}, B has {} rows",
            m,
            k,
            b.len()
        )));
    }
    let n = if b.is_empty() { 0 } else { b[0].len() };
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..k {
                s += a[i][l] * b[l][j];
            }
            c[i][j] = s;
        }
    }
    Ok(c)
}

/// Transpose a matrix
pub fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut t = vec![vec![0.0_f64; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Add two matrices element-wise
pub fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    let m = a.len();
    if m != b.len() {
        return Err(SignalError::ValueError(
            "Matrix dimension mismatch in addition".to_string(),
        ));
    }
    let n = if m == 0 { 0 } else { a[0].len() };
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        if a[i].len() != b[i].len() {
            return Err(SignalError::ValueError(
                "Matrix column mismatch in addition".to_string(),
            ));
        }
        for j in 0..n {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    Ok(c)
}

/// Subtract two matrices element-wise: A - B
pub fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    let m = a.len();
    if m != b.len() {
        return Err(SignalError::ValueError(
            "Matrix dimension mismatch in subtraction".to_string(),
        ));
    }
    let n = if m == 0 { 0 } else { a[0].len() };
    let mut c = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        if a[i].len() != b[i].len() {
            return Err(SignalError::ValueError(
                "Matrix column mismatch in subtraction".to_string(),
            ));
        }
        for j in 0..n {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    Ok(c)
}

/// Scale a matrix by a scalar
pub fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|x| x * s).collect())
        .collect()
}

/// Create an identity matrix of size n×n
pub fn mat_eye(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Matrix-vector product: A * v
pub fn mat_vec_mul(a: &[Vec<f64>], v: &[f64]) -> SignalResult<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return Ok(Vec::new());
    }
    let k = a[0].len();
    if v.len() != k {
        return Err(SignalError::ValueError(format!(
            "Incompatible dimensions for mat-vec mul: matrix cols={}, vector len={}",
            k,
            v.len()
        )));
    }
    let mut result = vec![0.0_f64; m];
    for i in 0..m {
        for j in 0..k {
            result[i] += a[i][j] * v[j];
        }
    }
    Ok(result)
}

/// Invert a square matrix using LU decomposition with partial pivoting.
pub fn mat_inv(a: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Err(SignalError::ValueError(
            "Cannot invert empty matrix".to_string(),
        ));
    }
    for row in a.iter() {
        if row.len() != n {
            return Err(SignalError::ValueError(
                "Matrix must be square to invert".to_string(),
            ));
        }
    }

    // Augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.resize(2 * n, 0.0);
            r[n + i] = 1.0;
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "Singular matrix: cannot invert".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    let val = aug[col][j] * factor;
                    aug[row][j] -= val;
                }
            }
        }
    }

    // Extract inverse from right half
    let inv: Vec<Vec<f64>> = aug.iter().map(|row| row[n..].to_vec()).collect();
    Ok(inv)
}

/// Solve the linear system A*x = b using LU decomposition with partial pivoting.
pub fn mat_solve(a: &[Vec<f64>], b: &[f64]) -> SignalResult<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Err(SignalError::ValueError("Empty matrix".to_string()));
    }
    if b.len() != n {
        return Err(SignalError::ValueError(
            "RHS length does not match matrix rows".to_string(),
        ));
    }

    // Build augmented [A|b]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "Singular matrix in linear solve".to_string(),
            ));
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..=n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..=n {
                    let val = aug[col][j] * factor;
                    aug[row][j] -= val;
                }
            }
        }
    }

    Ok(aug.iter().map(|row| row[n]).collect())
}

/// Compute the outer product of two vectors: v1 * v2^T
pub fn outer_product(v1: &[f64], v2: &[f64]) -> Vec<Vec<f64>> {
    v1.iter()
        .map(|&a| v2.iter().map(|&b| a * b).collect())
        .collect()
}

/// Compute the Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns lower triangular L such that A = L * L^T.
pub fn cholesky_decomp(a: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s < 0.0 {
                    return Err(SignalError::ComputationError(
                        "Matrix is not positive definite in Cholesky".to_string(),
                    ));
                }
                l[i][j] = s.sqrt();
            } else {
                if l[j][j].abs() < 1e-14 {
                    return Err(SignalError::ComputationError(
                        "Zero diagonal in Cholesky factorization".to_string(),
                    ));
                }
                l[i][j] = s / l[j][j];
            }
        }
    }
    Ok(l)
}

/// Compute the dot product of two vectors
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Add two vectors element-wise
pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Subtract two vectors element-wise: a - b
pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Scale a vector by a scalar
pub fn vec_scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

/// Compute the Euclidean norm of a vector
pub fn vec_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_mul_identity() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let eye = mat_eye(2);
        let result = mat_mul(&a, &eye).expect("mat_mul should succeed");
        assert!((result[0][0] - 1.0).abs() < 1e-12);
        assert!((result[1][1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_mat_inv_2x2() {
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = mat_inv(&a).expect("mat_inv should succeed");
        let result = mat_mul(&a, &inv).expect("mat_mul should succeed");
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((result[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_cholesky() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky_decomp(&a).expect("Cholesky should succeed");
        let lt = mat_transpose(&l);
        let reconstructed = mat_mul(&l, &lt).expect("mat_mul should succeed");
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[i][j] - a[i][j]).abs() < 1e-10);
            }
        }
    }
}
