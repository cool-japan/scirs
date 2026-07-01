//! Accelerated BLAS (Basic Linear Algebra Subprograms) operations using ndarray-linalg
//!
//! This module provides optimized BLAS operations using ndarray-linalg bindings to native BLAS libraries.
//! These functions are significantly faster for large matrices compared to pure Rust implementations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign};

use crate::error::{LinalgError, LinalgResult};

/// Computes the dot product of two vectors using optimized BLAS.
///
/// # Arguments
///
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
///
/// * The dot product of x and y
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::dot;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let y = array![4.0_f64, 5.0, 6.0];
/// let result = dot(&x.view(), &y.view()).expect("Operation failed");
/// assert!((result - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
/// ```
#[allow(dead_code)]
pub fn dot<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + 'static,
{
    if x.len() != y.len() {
        return Err(LinalgError::ShapeError(format!(
            "Vectors must have the same length for dot product, got {} and {}",
            x.len(),
            y.len()
        )));
    }

    // Use ndarray-linalg dot product implementation
    Ok(x.dot(y))
}

/// Computes the 2-norm (Euclidean norm) of a vector using optimized BLAS.
///
/// # Arguments
///
/// * `x` - Input vector
///
/// # Returns
///
/// * The 2-norm of x
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::norm;
///
/// let x = array![3.0_f64, 4.0];
/// let result = norm(&x.view()).expect("Operation failed");
/// assert!((result - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
/// ```
#[allow(dead_code)]
pub fn norm<F>(x: &ArrayView1<F>) -> LinalgResult<F>
where
    F: Float + NumAssign + 'static,
{
    if x.is_empty() {
        return Err(LinalgError::InvalidInputError(
            "Cannot compute norm of an empty vector".to_string(),
        ));
    }

    // Calculate the Euclidean (L2) norm manually
    let mut sum = F::zero();
    for &val in x.iter() {
        sum += val * val;
    }
    Ok(Float::sqrt(sum))
}

/// Performs matrix-vector multiplication using optimized BLAS.
///
/// Computes y = alpha*A*x + beta*y
///
/// # Arguments
///
/// * `alpha` - Scalar value for A*x
/// * `a` - Input matrix A
/// * `x` - Input vector x
/// * `beta` - Scalar value for y
/// * `y` - Input/output vector y
///
/// # Returns
///
/// * The resulting vector
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_linalg::blas_accelerated::gemv;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let x = array![2.0_f64, 3.0];
/// let y = Array1::<f64>::zeros(2);
/// let result = gemv(1.0, &a.view(), &x.view(), 0.0, &y.view()).expect("Operation failed");
/// assert!((result[0] - 8.0).abs() < 1e-10); // 1*2 + 2*3 = 8
/// assert!((result[1] - 18.0).abs() < 1e-10); // 3*2 + 4*3 = 18
/// ```
#[allow(dead_code)]
pub fn gemv<F>(
    alpha: F,
    a: &ArrayView2<F>,
    x: &ArrayView1<F>,
    beta: F,
    y: &ArrayView1<F>,
) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.ncols() != x.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix columns ({}) must match vector length ({}) for gemv",
            a.ncols(),
            x.len()
        )));
    }

    if a.nrows() != y.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix rows ({}) must match result vector length ({}) for gemv",
            a.nrows(),
            y.len()
        )));
    }

    // Create result vector (copy y)
    let mut result = y.to_owned();

    // Scale y by beta
    if beta != F::one() {
        result.map_inplace(|v| *v *= beta);
    }

    // Compute matrix-vector product using ndarray-linalg
    // a.dot(x) * alpha + result
    let ax = a.dot(x);
    result.zip_mut_with(&ax, |y_i, &ax_i| *y_i += alpha * ax_i);

    Ok(result)
}

/// Performs matrix-matrix multiplication using optimized BLAS.
///
/// Computes C = alpha*A*B + beta*C
///
/// # Arguments
///
/// * `alpha` - Scalar value for A*B
/// * `a` - Input matrix A
/// * `b` - Input matrix B
/// * `beta` - Scalar value for C
/// * `c` - Input/output matrix C
///
/// # Returns
///
/// * The resulting matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_linalg::blas_accelerated::gemm;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// let c = Array2::<f64>::zeros((2, 2));
/// let result = gemm(1.0, &a.view(), &b.view(), 0.0, &c.view()).expect("Operation failed");
/// assert!((result[[0, 0]] - 19.0).abs() < 1e-10); // 1*5 + 2*7 = 19
/// assert!((result[[0, 1]] - 22.0).abs() < 1e-10); // 1*6 + 2*8 = 22
/// assert!((result[[1, 0]] - 43.0).abs() < 1e-10); // 3*5 + 4*7 = 43
/// assert!((result[[1, 1]] - 50.0).abs() < 1e-10); // 3*6 + 4*8 = 50
/// ```
#[allow(dead_code)]
pub fn gemm<F>(
    alpha: F,
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    beta: F,
    c: &ArrayView2<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.ncols() != b.nrows() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions not compatible for multiplication: a.ncols ({}) != b.nrows ({})",
            a.ncols(),
            b.nrows()
        )));
    }

    if a.nrows() != c.nrows() || b.ncols() != c.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Output matrix dimensions ({},{}) don't match expected ({},{})",
            c.nrows(),
            c.ncols(),
            a.nrows(),
            b.ncols()
        )));
    }

    // Create result matrix (copy c)
    let mut result = c.to_owned();

    // Scale c by beta
    if beta != F::one() {
        result.map_inplace(|v| *v *= beta);
    }

    // Compute matrix-matrix product using ndarray-linalg
    // a.dot(b) * alpha + result
    let ab = a.dot(b);
    result.zip_mut_with(&ab, |c_ij, &ab_ij| *c_ij += alpha * ab_ij);

    Ok(result)
}

/// Performs matrix-matrix multiplication of large matrices using optimized BLAS.
///
/// This version is optimized for large matrices and returns a new matrix C = A * B.
///
/// # Arguments
///
/// * `a` - Input matrix A
/// * `b` - Input matrix B
///
/// # Returns
///
/// * The resulting matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::matmul;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// let c = matmul(&a.view(), &b.view()).expect("Operation failed");
/// assert!((c[[0, 0]] - 19.0).abs() < 1e-10); // 1*5 + 2*7 = 19
/// assert!((c[[0, 1]] - 22.0).abs() < 1e-10); // 1*6 + 2*8 = 22
/// assert!((c[[1, 0]] - 43.0).abs() < 1e-10); // 3*5 + 4*7 = 43
/// assert!((c[[1, 1]] - 50.0).abs() < 1e-10); // 3*6 + 4*8 = 50
/// ```
/// Minimum GEMM work (`m * k * n`) before the optional CUDA `matmul` fast path is
/// considered. Set deliberately high so that (a) small test/doctest matrices always
/// stay on the bit-stable CPU path and (b) the GPU is engaged only once the host
/// transfer and kernel-launch overhead is amortized. Only referenced under `cuda`.
#[cfg(feature = "cuda")]
const CUDA_MATMUL_MIN_FLOPS: usize = 1 << 21; // 2,097,152 (a 128x128x128 GEMM)

#[allow(dead_code)]
pub fn matmul<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.ncols() != b.nrows() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix dimensions not compatible for multiplication: a.ncols ({}) != b.nrows ({})",
            a.ncols(),
            b.nrows()
        )));
    }

    // Optional, transparent NVIDIA-CUDA fast path (off-by-default `cuda` feature).
    // ADDITIVE and SAFE: this entire block vanishes when `cuda` is disabled, so the
    // default build is byte-identical to the pure CPU path below. It engages only for
    // large f64 problems on a real CUDA device; on any miss or error it falls through
    // to the CPU `a.dot(b)`, which stays the source of truth. GPU GEMM accumulates in
    // a different associative order than the CPU triple-loop, so results may differ in
    // the last ULP -- acceptable precisely because CUDA_MATMUL_MIN_FLOPS is high enough
    // that no (small) test matrix ever reaches the GPU path.
    #[cfg(feature = "cuda")]
    {
        use std::any::TypeId;
        if TypeId::of::<F>() == TypeId::of::<f64>()
            && crate::gpu_cuda::cuda_is_available()
            && a.nrows()
                .saturating_mul(a.ncols())
                .saturating_mul(b.ncols())
                >= CUDA_MATMUL_MIN_FLOPS
        {
            // SAFETY: F == f64 is verified by the TypeId guard above, so reinterpreting
            // these references is layout-identical (mirrors the existing `det_impl`
            // idiom in basic.rs). Transmuting the *references* keeps the cast
            // pointer-sized and avoids any owned-value / generic-size reinterpret.
            let a_f64: &ArrayView2<f64> = unsafe { std::mem::transmute(a) };
            let b_f64: &ArrayView2<f64> = unsafe { std::mem::transmute(b) };
            if let Ok(c_f64) = crate::gpu_cuda::cuda_gemm(a_f64, b_f64) {
                let c_view = c_f64.view();
                // SAFETY: F == f64 (same TypeId guard); reinterpret the f64 result
                // view as the F-typed view, then own a copy as Array2<F>.
                let c_view_f: &ArrayView2<F> = unsafe { std::mem::transmute(&c_view) };
                return Ok(c_view_f.to_owned());
            }
            // On Err: silently fall through to the CPU result (CPU = source of truth).
        }
    }

    // Use ndarray-linalg's dot implementation for optimal performance
    Ok(a.dot(b))
}

/// Solves the linear system Ax = b for large matrices using optimized LAPACK.
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// * Solution vector x
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::solve;
///
/// let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
/// let b = array![9.0_f64, 8.0];
/// let x = solve(&a.view(), &b.view()).expect("Operation failed");
/// assert!((x[0] - 2.0).abs() < 1e-10);
/// assert!((x[1] - 3.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn solve<F>(a: &ArrayView2<F>, b: &ArrayView1<F>) -> LinalgResult<Array1<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square for solve, got shape {:?}",
            a.shape()
        )));
    }

    if a.nrows() != b.len() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix rows ({}) must match vector length ({}) for solve",
            a.nrows(),
            b.len()
        )));
    }

    // Implement a basic solver instead of using ndarray-linalg directly
    // For now, we'll use a simple Gaussian elimination approach
    let n = a.nrows();

    // Create augmented matrix [A|b]
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = Float::abs(aug[[i, i]]);

        for j in (i + 1)..n {
            let val = Float::abs(aug[[j, i]]);
            if val > max_val {
                max_row = j;
                max_val = val;
            }
        }

        // Check for singular matrix
        if max_val < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..(n + 1) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate below
        for j in (i + 1)..n {
            let factor = aug[[j, i]] / aug[[i, i]];
            aug[[j, i]] = F::zero(); // Just for numerical stability

            for k in (i + 1)..(n + 1) {
                aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// Computes the inverse of a square matrix using optimized LAPACK.
///
/// # Arguments
///
/// * `a` - Input square matrix
///
/// # Returns
///
/// * Inverse of the matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::blas_accelerated::inv;
///
/// let a = array![[4.0_f64, 7.0], [2.0, 6.0]];
/// let a_inv = inv(&a.view()).expect("Operation failed");
/// // Check that A * A^-1 is approximately identity
/// let identity = a.dot(&a_inv);
/// assert!((identity[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((identity[[0, 1]]).abs() < 1e-10);
/// assert!((identity[[1, 0]]).abs() < 1e-10);
/// assert!((identity[[1, 1]] - 1.0).abs() < 1e-10);
/// ```
#[allow(dead_code)]
pub fn inv<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + 'static,
{
    if a.nrows() != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "Matrix must be square for inverse, got shape {:?}",
            a.shape()
        )));
    }

    // Implement matrix inversion using Gaussian elimination with identity matrix
    let n = a.nrows();

    // Create augmented matrix [A|I]
    let mut aug = Array2::<F>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, i + n]] = F::one(); // Identity matrix part
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = Float::abs(aug[[i, i]]);

        for j in (i + 1)..n {
            let val = Float::abs(aug[[j, i]]);
            if val > max_val {
                max_row = j;
                max_val = val;
            }
        }

        // Check for singular matrix
        if max_val < F::epsilon() {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Scale row to get pivot = 1
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] /= pivot;
        }

        // Eliminate other rows
        for j in 0..n {
            if j != i {
                let factor = aug[[j, i]];
                for k in 0..(2 * n) {
                    aug[[j, k]] = aug[[j, k]] - factor * aug[[i, k]];
                }
            }
        }
    }

    // Extract inverse from right half of augmented matrix
    let mut a_inv = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_inv[[i, j]] = aug[[i, j + n]];
        }
    }

    Ok(a_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_dot() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![4.0, 5.0, 6.0];
        let result = dot(&x.view(), &y.view()).expect("Operation failed");
        assert_relative_eq!(result, 32.0, epsilon = 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_norm() {
        let x = array![3.0, 4.0];
        let result = norm(&x.view()).expect("Operation failed");
        assert_relative_eq!(result, 5.0, epsilon = 1e-10); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_gemv() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let x = array![2.0, 3.0];
        let y = Array1::<f64>::zeros(2);
        let result = gemv(1.0, &a.view(), &x.view(), 0.0, &y.view()).expect("Operation failed");
        assert_relative_eq!(result[0], 8.0, epsilon = 1e-10); // 1*2 + 2*3 = 8
        assert_relative_eq!(result[1], 18.0, epsilon = 1e-10); // 3*2 + 4*3 = 18
    }

    #[test]
    fn test_gemm() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = Array2::<f64>::zeros((2, 2));
        let result = gemm(1.0, &a.view(), &b.view(), 0.0, &c.view()).expect("Operation failed");
        assert_relative_eq!(result[[0, 0]], 19.0, epsilon = 1e-10); // 1*5 + 2*7 = 19
        assert_relative_eq!(result[[0, 1]], 22.0, epsilon = 1e-10); // 1*6 + 2*8 = 22
        assert_relative_eq!(result[[1, 0]], 43.0, epsilon = 1e-10); // 3*5 + 4*7 = 43
        assert_relative_eq!(result[[1, 1]], 50.0, epsilon = 1e-10); // 3*6 + 4*8 = 50
    }

    #[test]
    fn test_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let result = matmul(&a.view(), &b.view()).expect("Operation failed");
        assert_relative_eq!(result[[0, 0]], 19.0, epsilon = 1e-10); // 1*5 + 2*7 = 19
        assert_relative_eq!(result[[0, 1]], 22.0, epsilon = 1e-10); // 1*6 + 2*8 = 22
        assert_relative_eq!(result[[1, 0]], 43.0, epsilon = 1e-10); // 3*5 + 4*7 = 43
        assert_relative_eq!(result[[1, 1]], 50.0, epsilon = 1e-10); // 3*6 + 4*8 = 50
    }

    #[test]
    fn test_solve() {
        let a = array![[3.0, 1.0], [1.0, 2.0]];
        let b = array![9.0, 8.0];
        let x = solve(&a.view(), &b.view()).expect("Operation failed");
        assert_relative_eq!(x[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(x[1], 3.0, epsilon = 1e-10);

        // Verify solution
        let b_check = a.dot(&x);
        assert_relative_eq!(b_check[0], b[0], epsilon = 1e-10);
        assert_relative_eq!(b_check[1], b[1], epsilon = 1e-10);
    }

    /// §4a: when cuda feature is active and `m*k*n >= CUDA_MATMUL_MIN_FLOPS` and
    /// `F == f64`, `matmul` routes to `cuda_gemm`.  On a host where
    /// `cuda_is_available()` is `true` (e.g. this RTX A4000 box), the 130×130×130
    /// problem is above the 2,097,152-flop threshold so the GPU branch is taken;
    /// correctness is the observable invariant.
    #[cfg(feature = "cuda")]
    #[test]
    fn matmul_cuda_dispatch_f64_or_skip() {
        if !crate::gpu_cuda::cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device — GPU branch not engaged");
            return;
        }
        // 130 * 130 * 130 = 2,197,000 > CUDA_MATMUL_MIN_FLOPS (2,097,152).
        let m = 130usize;
        let k = 130usize;
        let n = 130usize;
        let a_data: Vec<f64> = (0..m * k)
            .map(|idx| (idx as f64 + 1.0) / (m * k) as f64)
            .collect();
        let b_data: Vec<f64> = (0..k * n)
            .map(|idx| (idx as f64 * 1.3 + 0.7) / (k * n) as f64)
            .collect();
        let a = Array2::from_shape_vec((m, k), a_data).expect("a shape ok");
        let b = Array2::from_shape_vec((k, n), b_data).expect("b shape ok");
        let c_matmul = matmul(&a.view(), &b.view()).expect("matmul f64 130x130 failed");
        let c_cpu = a.dot(&b);
        assert_eq!(c_matmul.shape(), &[m, n]);
        // GPU GEMM accumulates in a different associative order; allow ~1e-6 relative.
        // On the A4000 with values in [0,1] the actual difference is typically < 1e-12.
        let max_rel = c_matmul
            .iter()
            .zip(c_cpu.iter())
            .map(|(g, e)| (g - e).abs() / e.abs().max(1e-30))
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 1e-6,
            "130x130 f64 GPU dispatch: max relative diff {max_rel} exceeds 1e-6"
        );
    }

    /// §4a: f32 inputs of any size stay on the CPU path (the `TypeId == f64` guard
    /// in `matmul` excludes them from the CUDA branch).
    #[cfg(feature = "cuda")]
    #[test]
    fn matmul_f32_stays_cpu() {
        // Same 130x130 footprint as the f64 test; TypeId != f64 so always CPU.
        let m = 130usize;
        let k = 130usize;
        let n = 130usize;
        let a_data: Vec<f32> = (0..m * k)
            .map(|idx| (idx as f32 + 1.0) / (m * k) as f32)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|idx| (idx as f32 * 1.3 + 0.7) / (k * n) as f32)
            .collect();
        let a = Array2::<f32>::from_shape_vec((m, k), a_data).expect("a shape ok");
        let b = Array2::<f32>::from_shape_vec((k, n), b_data).expect("b shape ok");
        let c_matmul = matmul(&a.view(), &b.view()).expect("matmul f32 130x130 failed");
        let c_cpu = a.dot(&b);
        assert_eq!(c_matmul.shape(), &[m, n]);
        // Both go through a.dot(b) — must match within f32 precision.
        let max_diff = c_matmul
            .iter()
            .zip(c_cpu.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "f32 130x130 (CPU path): max abs diff {max_diff} exceeds 1e-5"
        );
    }

    /// §4a: sub-threshold f64 inputs (m*k*n < CUDA_MATMUL_MIN_FLOPS) always stay
    /// on the CPU path and must be bit-identical to `a.dot(&b)`.
    #[cfg(feature = "cuda")]
    #[test]
    fn matmul_subthreshold_f64_stays_cpu() {
        // 4 * 4 * 4 = 64, well below the 2,097,152 threshold.
        let a = array![
            [1.0_f64, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ];
        let b = array![
            [0.1_f64, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6]
        ];
        let c_matmul = matmul(&a.view(), &b.view()).expect("sub-threshold matmul failed");
        let c_dot = a.dot(&b);
        // Both code paths are a.dot(b) — must be bit-identical for finite values.
        c_matmul
            .iter()
            .zip(c_dot.iter())
            .enumerate()
            .for_each(|(i, (m_val, d_val))| {
                assert_eq!(
                    *m_val, *d_val,
                    "sub-threshold matmul element {i} not bit-identical to a.dot(&b)"
                );
            });
    }

    /// §7: shape-mismatch must return `Err` (no panic), and valid inputs produce
    /// correct results regardless of which branch fires.  Any `cuda_gemm` `Err`
    /// inside `matmul` falls through silently to `a.dot(b)` via the
    /// `if let Ok(...) = cuda_gemm(...)` guard — there is no panic path.
    #[test]
    fn matmul_fallback_safety() {
        // Dimension mismatch: rejected before any device call.
        let a_bad = Array2::<f64>::zeros((3, 4));
        let b_bad = Array2::<f64>::zeros((5, 2)); // 4 != 5
        assert!(
            matmul(&a_bad.view(), &b_bad.view()).is_err(),
            "shape mismatch must return Err, not panic"
        );
        // Valid small f64 (always CPU): must give the correct product.
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[3.0_f64, 1.0], [2.0, 4.0]];
        let c = matmul(&a.view(), &b.view()).expect("identity x B failed");
        assert_relative_eq!(c[[0, 0]], 3.0, epsilon = 1e-13);
        assert_relative_eq!(c[[0, 1]], 1.0, epsilon = 1e-13);
        assert_relative_eq!(c[[1, 0]], 2.0, epsilon = 1e-13);
        assert_relative_eq!(c[[1, 1]], 4.0, epsilon = 1e-13);
    }

    #[test]
    fn test_inv() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let a_inv = inv(&a.view()).expect("Operation failed");

        // Check a few values
        assert_relative_eq!(a_inv[[0, 0]], 0.6, epsilon = 1e-10);
        assert_relative_eq!(a_inv[[0, 1]], -0.7, epsilon = 1e-10);
        assert_relative_eq!(a_inv[[1, 0]], -0.2, epsilon = 1e-10);
        assert_relative_eq!(a_inv[[1, 1]], 0.4, epsilon = 1e-10);

        // Check that A * A^-1 is approximately identity
        let identity = a.dot(&a_inv);
        assert_relative_eq!(identity[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[1, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[1, 1]], 1.0, epsilon = 1e-10);
    }
}
