//! Strassen's algorithm for matrix multiplication and structured matrix products
//!
//! This module provides the classic Strassen recursive algorithm (O(n^2.807)) as well
//! as specialized routines for banded and triangular matrix-matrix products.
//!
//! ## Background
//!
//! Strassen (1969) showed that two 2×2 matrices can be multiplied using only 7 scalar
//! multiplications instead of the naive 8. Applying this recursively yields complexity
//! O(n^log2(7)) ≈ O(n^2.807), which outperforms the O(n^3) naive algorithm for large n.
//!
//! ## Threshold Selection
//!
//! The recursion bottoms out at `STRASSEN_BASE_THRESHOLD` (64 by default). Below this
//! size the overhead of recursive calls and temporary allocations exceeds the savings
//! from fewer multiplications, so a straightforward three-loop BLAS-style kernel is used.
//!
//! ## References
//!
//! - Strassen, V. (1969). "Gaussian Elimination is not Optimal". Numerische Mathematik. 13: 354–356.
//! - Cohn, H.; Umans, C. (2003). "A Group-theoretic Approach to Matrix Multiplication". FOCS.

use scirs2_core::ndarray::{s, Array2, ArrayView2};

use crate::error::{LinalgError, LinalgResult};

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Default threshold: matrices with max-dimension ≤ this value are multiplied
/// with the plain three-loop algorithm.  Tunable at call-site via the
/// `threshold` argument.
pub const STRASSEN_BASE_THRESHOLD: usize = 64;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the threshold (base-case side length) that `strassen_multiply` uses
/// given the leading dimension `n` of the problem.
///
/// The heuristic is: round down to the largest power-of-2 ≤ max(64, n/16) so
/// that the recursion tree has a bounded depth and each leaf is at least
/// cache-resident.
///
/// # Examples
///
/// ```
/// use scirs2_linalg::strassen::strassen_threshold;
///
/// assert!(strassen_threshold(1024) >= 64);
/// assert!(strassen_threshold(64) == 64);
/// ```
pub fn strassen_threshold(n: usize) -> usize {
    let candidate = n / 16;
    // Find the largest power-of-two >= STRASSEN_BASE_THRESHOLD
    let base = candidate.max(STRASSEN_BASE_THRESHOLD);
    // Round down to a power of two
    if base == 0 {
        return STRASSEN_BASE_THRESHOLD;
    }
    let bits = usize::BITS - base.leading_zeros();
    let pow2 = 1usize << (bits - 1);
    pow2.max(STRASSEN_BASE_THRESHOLD)
}

/// Multiply two matrices using Strassen's algorithm.
///
/// The matrices need not be square or have power-of-2 dimensions.  They are
/// padded internally to the next common power-of-2 size when required and the
/// result is trimmed back to `(m, n)`.
///
/// Falls back to a plain O(n^3) kernel when the dimension drops below
/// `threshold` (default: [`STRASSEN_BASE_THRESHOLD`]).
///
/// # Arguments
///
/// * `a` — `(m, k)` matrix
/// * `b` — `(k, n)` matrix
/// * `threshold` — Optional base-case cutoff; defaults to [`strassen_threshold`]`(max(m,k,n))`
///
/// # Returns
///
/// `(m, n)` result matrix, or a [`LinalgError`] on shape mismatch.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::strassen::strassen_multiply;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// let c = strassen_multiply(&a.view(), &b.view(), None).expect("valid input");
/// assert!((c[[0,0]] - 19.0).abs() < 1e-12);
/// assert!((c[[0,1]] - 22.0).abs() < 1e-12);
/// assert!((c[[1,0]] - 43.0).abs() < 1e-12);
/// assert!((c[[1,1]] - 50.0).abs() < 1e-12);
/// ```
pub fn strassen_multiply(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    threshold: Option<usize>,
) -> LinalgResult<Array2<f64>> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "strassen_multiply: inner dimensions differ: A is ({m}×{k1}), B is ({k2}×{n})"
        )));
    }

    if m == 0 || k1 == 0 || n == 0 {
        return Ok(Array2::zeros((m, n)));
    }

    let thresh = threshold.unwrap_or_else(|| strassen_threshold(m.max(k1).max(n)));

    // Determine padded size — next power of two covering all three dimensions
    let max_dim = m.max(k1).max(n);
    let pad_size = next_power_of_two(max_dim);

    if pad_size <= thresh {
        // Direct base-case without padding overhead
        return Ok(naive_matmul_alloc(a, b));
    }

    // Pad to (pad_size × pad_size) square matrices
    let a_pad = pad_to_square(a, pad_size);
    let b_pad = pad_to_square(b, pad_size);

    // Recursive Strassen on padded matrices
    let c_pad = strassen_recursive_f64(&a_pad.view(), &b_pad.view(), thresh);

    // Extract result sub-matrix
    Ok(c_pad.slice(s![0..m, 0..n]).to_owned())
}

/// Multiply two banded matrices exploiting the band structure.
///
/// A banded matrix with bandwidth `w` has non-zero entries only in columns
/// `[max(0, i-w) .. min(n-1, i+w)]` of row `i`.  This routine skips all
/// zero-band elements, reducing the work from O(n³) to O(n · w²).
///
/// # Arguments
///
/// * `a` — `(m, k)` matrix; assumed to have bandwidth `bandwidth`
/// * `b` — `(k, n)` matrix; assumed to have bandwidth `bandwidth`
/// * `bandwidth` — Half-bandwidth `w` (number of sub/super-diagonals)
///
/// # Returns
///
/// Product matrix of shape `(m, n)`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::strassen::banded_multiply;
///
/// // Tridiagonal (bandwidth = 1)
/// let a = array![
///     [2.0_f64,  1.0,  0.0],
///     [1.0,      2.0,  1.0],
///     [0.0,      1.0,  2.0],
/// ];
/// let b = array![
///     [1.0_f64,  0.0,  0.0],
///     [0.0,      1.0,  0.0],
///     [0.0,      0.0,  1.0],
/// ];
/// let c = banded_multiply(&a.view(), &b.view(), 1).expect("valid input");
/// // A * I == A
/// assert!((c[[0,0]] - 2.0).abs() < 1e-12);
/// assert!((c[[1,1]] - 2.0).abs() < 1e-12);
/// ```
pub fn banded_multiply(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    bandwidth: usize,
) -> LinalgResult<Array2<f64>> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "banded_multiply: inner dimensions differ: A is ({m}×{k1}), B is ({k2}×{n})"
        )));
    }

    let mut result = Array2::<f64>::zeros((m, n));
    let w = bandwidth;

    for i in 0..m {
        // Non-zero columns of row i in A: [max(0, i-w) .. min(k1-1, i+w)]
        let k_lo = i.saturating_sub(w);
        let k_hi = (i + w + 1).min(k1);

        for k in k_lo..k_hi {
            let a_ik = a[[i, k]];
            if a_ik == 0.0 {
                continue;
            }
            // Non-zero columns of row k in B
            let j_lo = k.saturating_sub(w);
            let j_hi = (k + w + 1).min(n);

            for j in j_lo..j_hi {
                result[[i, j]] += a_ik * b[[k, j]];
            }
        }
    }

    Ok(result)
}

/// Multiply a triangular matrix `a` (left operand) by a general matrix `b`.
///
/// Mirrors the BLAS TRMM ("triangular matrix–matrix multiply") operation.
/// Only the relevant triangle of `a` is accessed; the other entries are
/// treated as zero, avoiding unnecessary work.
///
/// # Arguments
///
/// * `a_triangular` — `(n, n)` square triangular matrix
/// * `b` — `(n, p)` general matrix
/// * `upper` — `true` if `a` is upper triangular; `false` for lower triangular
///
/// # Returns
///
/// Product matrix of shape `(n, p)`.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::strassen::triangular_multiply;
///
/// // Upper triangular A
/// let a = array![[1.0_f64, 2.0], [0.0, 3.0]];
/// let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let c = triangular_multiply(&a.view(), &b.view(), true).expect("valid input");
/// assert!((c[[0,0]] - 1.0).abs() < 1e-12);
/// assert!((c[[0,1]] - 2.0).abs() < 1e-12);
/// assert!((c[[1,0]] - 0.0).abs() < 1e-12);
/// assert!((c[[1,1]] - 3.0).abs() < 1e-12);
/// ```
pub fn triangular_multiply(
    a_triangular: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    upper: bool,
) -> LinalgResult<Array2<f64>> {
    let (an, am) = a_triangular.dim();
    let (bm, p) = b.dim();

    if an != am {
        return Err(LinalgError::ShapeError(format!(
            "triangular_multiply: A must be square, got ({an}×{am})"
        )));
    }
    if am != bm {
        return Err(LinalgError::ShapeError(format!(
            "triangular_multiply: inner dimensions differ: A is ({an}×{am}), B is ({bm}×{p})"
        )));
    }

    let n = an;
    let mut result = Array2::<f64>::zeros((n, p));

    if upper {
        // C[i,j] = sum_{k=i}^{n-1} A[i,k] * B[k,j]
        for i in 0..n {
            for j in 0..p {
                let mut sum = 0.0_f64;
                for k in i..n {
                    sum += a_triangular[[i, k]] * b[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }
    } else {
        // Lower triangular: C[i,j] = sum_{k=0}^{i} A[i,k] * B[k,j]
        for i in 0..n {
            for j in 0..p {
                let mut sum = 0.0_f64;
                for k in 0..=i {
                    sum += a_triangular[[i, k]] * b[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Round up to the next power of two (returns 1 for input 0).
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let bits = usize::BITS - (n - 1).leading_zeros();
    1usize << bits
}

/// Zero-pad `a` to a `(size × size)` square matrix.
fn pad_to_square(a: &ArrayView2<f64>, size: usize) -> Array2<f64> {
    let (rows, cols) = a.dim();
    let mut out = Array2::<f64>::zeros((size, size));
    for i in 0..rows {
        for j in 0..cols {
            out[[i, j]] = a[[i, j]];
        }
    }
    out
}

/// Naive O(n³) multiplication — allocates a new result array.
fn naive_matmul_alloc(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let n = b.ncols();
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a[[i, kk]];
            for j in 0..n {
                c[[i, j]] += a_ik * b[[kk, j]];
            }
        }
    }
    c
}

/// Core recursive Strassen implementation for **square** power-of-2 matrices.
///
/// Both `a` and `b` must be `(n × n)` with `n` a power of two.
/// Uses exactly 7 recursive multiplications per level.
fn strassen_recursive_f64(a: &ArrayView2<f64>, b: &ArrayView2<f64>, thresh: usize) -> Array2<f64> {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());
    debug_assert_eq!(n, b.nrows());
    debug_assert_eq!(n, b.ncols());

    if n <= thresh {
        return naive_matmul_alloc(a, b);
    }

    let h = n / 2;

    // Partition A and B into 2×2 block form
    let a11 = a.slice(s![0..h, 0..h]);
    let a12 = a.slice(s![0..h, h..n]);
    let a21 = a.slice(s![h..n, 0..h]);
    let a22 = a.slice(s![h..n, h..n]);

    let b11 = b.slice(s![0..h, 0..h]);
    let b12 = b.slice(s![0..h, h..n]);
    let b21 = b.slice(s![h..n, 0..h]);
    let b22 = b.slice(s![h..n, h..n]);

    // Strassen's 7 products (Winograd variant keeps same count but reduces adds)
    // Using original Strassen formulation:
    // M1 = (A11 + A22)(B11 + B22)
    // M2 = (A21 + A22) B11
    // M3 = A11 (B12 - B22)
    // M4 = A22 (B21 - B11)
    // M5 = (A11 + A12) B22
    // M6 = (A21 - A11)(B11 + B12)
    // M7 = (A12 - A22)(B21 + B22)
    let m1 = strassen_recursive_f64(&(&a11 + &a22).view(), &(&b11 + &b22).view(), thresh);
    let m2 = strassen_recursive_f64(&(&a21 + &a22).view(), &b11.to_owned().view(), thresh);
    let m3 = strassen_recursive_f64(&a11.to_owned().view(), &(&b12 - &b22).view(), thresh);
    let m4 = strassen_recursive_f64(&a22.to_owned().view(), &(&b21 - &b11).view(), thresh);
    let m5 = strassen_recursive_f64(&(&a11 + &a12).view(), &b22.to_owned().view(), thresh);
    let m6 = strassen_recursive_f64(&(&a21 - &a11).view(), &(&b11 + &b12).view(), thresh);
    let m7 = strassen_recursive_f64(&(&a12 - &a22).view(), &(&b21 + &b22).view(), thresh);

    // Reconstruct quadrants
    // C11 = M1 + M4 - M5 + M7
    // C12 = M3 + M5
    // C21 = M2 + M4
    // C22 = M1 - M2 + M3 + M6
    let c11 = &m1 + &m4 - &m5 + &m7;
    let c12 = &m3 + &m5;
    let c21 = &m2 + &m4;
    let c22 = &m1 - &m2 + &m3 + &m6;

    // Assemble result
    let mut c = Array2::<f64>::zeros((n, n));
    for i in 0..h {
        for j in 0..h {
            c[[i, j]] = c11[[i, j]];
            c[[i, j + h]] = c12[[i, j]];
            c[[i + h, j]] = c21[[i, j]];
            c[[i + h, j + h]] = c22[[i, j]];
        }
    }

    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{array, Array2};

    fn reference_matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let (m, k) = a.dim();
        let n = b.ncols();
        let mut c = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for kk in 0..k {
                for j in 0..n {
                    c[[i, j]] += a[[i, kk]] * b[[kk, j]];
                }
            }
        }
        c
    }

    #[test]
    fn test_strassen_2x2() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
        // Force threshold=1 so Strassen is used on 2×2
        let c = strassen_multiply(&a.view(), &b.view(), Some(1)).expect("failed to create c");
        assert_relative_eq!(c[[0, 0]], 19.0, epsilon = 1e-12);
        assert_relative_eq!(c[[0, 1]], 22.0, epsilon = 1e-12);
        assert_relative_eq!(c[[1, 0]], 43.0, epsilon = 1e-12);
        assert_relative_eq!(c[[1, 1]], 50.0, epsilon = 1e-12);
    }

    #[test]
    fn test_strassen_4x4() {
        let n = 4;
        let a = Array2::<f64>::from_shape_fn((n, n), |(i, j)| (i * n + j + 1) as f64);
        let b = Array2::<f64>::from_shape_fn((n, n), |(i, j)| ((n - i) * n + (n - j)) as f64);
        let expected = reference_matmul(&a, &b);
        let got = strassen_multiply(&a.view(), &b.view(), Some(2)).expect("failed to create got");
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(got[[i, j]], expected[[i, j]], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_strassen_non_square_rect() {
        // 3×5 × 5×4
        let m = 3usize;
        let k = 5usize;
        let n = 4usize;
        let a = Array2::<f64>::from_shape_fn((m, k), |(i, j)| (i + j + 1) as f64);
        let b = Array2::<f64>::from_shape_fn((k, n), |(i, j)| (i + j + 2) as f64);
        let expected = reference_matmul(&a, &b);
        let got = strassen_multiply(&a.view(), &b.view(), Some(2)).expect("failed to create got");
        assert_eq!(got.dim(), (m, n));
        for i in 0..m {
            for j in 0..n {
                assert_relative_eq!(got[[i, j]], expected[[i, j]], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_strassen_dimension_mismatch() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[1.0_f64, 2.0, 3.0]];
        assert!(strassen_multiply(&a.view(), &b.view(), None).is_err());
    }

    #[test]
    fn test_strassen_identity() {
        let n = 8;
        let a = Array2::<f64>::from_shape_fn((n, n), |(i, j)| (i * n + j + 1) as f64);
        let eye = Array2::<f64>::eye(n);
        let c = strassen_multiply(&a.view(), &eye.view(), Some(2)).expect("failed to create c");
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(c[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_banded_multiply_identity() {
        let a = array![[2.0_f64, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]];
        let id = Array2::<f64>::eye(3);
        let c = banded_multiply(&a.view(), &id.view(), 1).expect("failed to create c");
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(c[[i, j]], a[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_banded_multiply_tridiagonal_x_tridiagonal() {
        // Both tridiagonal; check against naive
        let n = 5;
        let a = Array2::<f64>::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                2.0
            } else if i.abs_diff(j) == 1 {
                -1.0
            } else {
                0.0
            }
        });
        let b = a.clone();
        let expected = reference_matmul(&a, &b);
        let got = banded_multiply(&a.view(), &b.view(), 1).expect("failed to create got");
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(got[[i, j]], expected[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_banded_multiply_shape_error() {
        let a = array![[1.0_f64, 2.0]];
        let b = array![[1.0_f64], [2.0], [3.0]];
        assert!(banded_multiply(&a.view(), &b.view(), 1).is_err());
    }

    #[test]
    fn test_triangular_multiply_upper() {
        let a = array![[1.0_f64, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]];
        let b = Array2::<f64>::eye(3);
        let c = triangular_multiply(&a.view(), &b.view(), true).expect("failed to create c");
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(c[[i, j]], a[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_triangular_multiply_lower() {
        let a = array![[1.0_f64, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]];
        let b = Array2::<f64>::eye(3);
        let c = triangular_multiply(&a.view(), &b.view(), false).expect("failed to create c");
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(c[[i, j]], a[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_triangular_multiply_non_square_a_error() {
        let a = array![[1.0_f64, 2.0, 3.0]];
        let b = array![[1.0_f64], [2.0], [3.0]];
        assert!(triangular_multiply(&a.view(), &b.view(), true).is_err());
    }

    #[test]
    fn test_triangular_multiply_vs_reference() {
        let a = array![[3.0_f64, 1.0, 0.0], [0.0, 2.0, -1.0], [0.0, 0.0, 4.0]];
        let b = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let expected = reference_matmul(&a, &b);
        let got = triangular_multiply(&a.view(), &b.view(), true).expect("failed to create got");
        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(got[[i, j]], expected[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
    }

    #[test]
    fn test_strassen_threshold_bounds() {
        assert!(strassen_threshold(64) >= 64);
        assert!(strassen_threshold(1024) >= 64);
        assert!(strassen_threshold(4096) >= 64);
    }
}
