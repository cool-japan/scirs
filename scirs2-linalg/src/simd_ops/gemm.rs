//! SIMD-accelerated General Matrix Multiplication (GEMM) operations
//!
//! This module provides highly optimized SIMD implementations of GEMM operations
//! using cache-friendly blocking strategies, micro-kernels, and vectorized
//! inner loops for maximum performance on modern CPUs.

#[cfg(feature = "simd")]
use crate::error::{LinalgError, LinalgResult};
#[cfg(feature = "simd")]
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
#[cfg(feature = "simd")]
use wide::{f32x8, f64x4};

/// Cache-friendly block sizes for GEMM operations
/// These should be tuned for target CPU cache hierarchy
#[cfg(feature = "simd")]
pub struct GemmBlockSizes {
    /// Block size for M dimension (rows of A, rows of C)
    pub mc: usize,
    /// Block size for K dimension (cols of A, rows of B)  
    pub kc: usize,
    /// Block size for N dimension (cols of B, cols of C)
    pub nc: usize,
    /// Micro-kernel block size for M dimension
    pub mr: usize,
    /// Micro-kernel block size for N dimension  
    pub nr: usize,
}

#[cfg(feature = "simd")]
impl Default for GemmBlockSizes {
    fn default() -> Self {
        Self {
            mc: 64,  // L2 cache friendly
            kc: 256, // L1 cache friendly for B panel
            nc: 512, // L3 cache friendly
            mr: 8,   // SIMD width considerations
            nr: 8,   // SIMD width considerations
        }
    }
}

/// SIMD-accelerated GEMM for f32: C = alpha * A * B + beta * C
///
/// This implementation uses a 3-level blocking strategy:
/// 1. Outer blocks (MC x KC) of A and (KC x NC) of B
/// 2. Panel operations that are cache-friendly
/// 3. Micro-kernels with SIMD vectorization
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Left matrix A (M x K)
/// * `b` - Right matrix B (K x N)
/// * `beta` - Scalar multiplier for C
/// * `c` - Result matrix C (M x N), updated in-place
/// * `block_sizes` - Cache-friendly block size configuration
///
/// # Returns
///
/// * Result indicating success or error
#[cfg(feature = "simd")]
pub fn simd_gemm_f32(
    alpha: f32,
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
    beta: f32,
    c: &mut Array2<f32>,
    block_sizes: Option<GemmBlockSizes>,
) -> LinalgResult<()> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();
    let (cm, cn) = c.dim();

    // Validate matrix dimensions
    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix inner dimensions must match: A({}, {}) * B({}, {})",
            m, k1, k2, n
        )));
    }
    if cm != m || cn != n {
        return Err(LinalgError::ShapeError(format!(
            "Result matrix dimensions must match: C({}, {}) for A({}, {}) * B({}, {})",
            cm, cn, m, k1, k2, n
        )));
    }

    let k = k1;
    let bs = block_sizes.unwrap_or_default();

    // Scale existing C matrix by beta
    if beta != 1.0 {
        if beta == 0.0 {
            c.fill(0.0);
        } else {
            c.mapv_inplace(|x| x * beta);
        }
    }

    // 3-level blocked GEMM algorithm
    for jc in (0..n).step_by(bs.nc) {
        let nc = (bs.nc).min(n - jc);

        for pc in (0..k).step_by(bs.kc) {
            let kc = (bs.kc).min(k - pc);

            // Pack B panel for better cache locality
            let b_panel = b.slice(s![pc..pc + kc, jc..jc + nc]);

            for ic in (0..m).step_by(bs.mc) {
                let mc = (bs.mc).min(m - ic);

                // Pack A panel for better cache locality
                let a_panel = a.slice(s![ic..ic + mc, pc..pc + kc]);

                // Call micro-kernel for this block
                simd_gemm_micro_kernel_f32(
                    alpha,
                    &a_panel,
                    &b_panel,
                    &mut c.slice_mut(s![ic..ic + mc, jc..jc + nc]),
                    &bs,
                )?;
            }
        }
    }

    Ok(())
}

/// SIMD-accelerated GEMM for f64: C = alpha * A * B + beta * C
///
/// This implementation uses a 3-level blocking strategy optimized for f64 precision.
/// Uses 4-wide SIMD vectors for f64 values.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Left matrix A (M x K)
/// * `b` - Right matrix B (K x N)
/// * `beta` - Scalar multiplier for C
/// * `c` - Result matrix C (M x N), updated in-place
/// * `block_sizes` - Cache-friendly block size configuration
///
/// # Returns
///
/// * Result indicating success or error
#[cfg(feature = "simd")]
pub fn simd_gemm_f64(
    alpha: f64,
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    beta: f64,
    c: &mut Array2<f64>,
    block_sizes: Option<GemmBlockSizes>,
) -> LinalgResult<()> {
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();
    let (cm, cn) = c.dim();

    // Validate matrix dimensions
    if k1 != k2 {
        return Err(LinalgError::ShapeError(format!(
            "Matrix inner dimensions must match: A({}, {}) * B({}, {})",
            m, k1, k2, n
        )));
    }
    if cm != m || cn != n {
        return Err(LinalgError::ShapeError(format!(
            "Result matrix dimensions must match: C({}, {}) for A({}, {}) * B({}, {})",
            cm, cn, m, k1, k2, n
        )));
    }

    let k = k1;
    let bs = block_sizes.unwrap_or_default();

    // Scale existing C matrix by beta
    if beta != 1.0 {
        if beta == 0.0 {
            c.fill(0.0);
        } else {
            c.mapv_inplace(|x| x * beta);
        }
    }

    // 3-level blocked GEMM algorithm
    for jc in (0..n).step_by(bs.nc) {
        let nc = (bs.nc).min(n - jc);

        for pc in (0..k).step_by(bs.kc) {
            let kc = (bs.kc).min(k - pc);

            // Pack B panel for better cache locality
            let b_panel = b.slice(s![pc..pc + kc, jc..jc + nc]);

            for ic in (0..m).step_by(bs.mc) {
                let mc = (bs.mc).min(m - ic);

                // Pack A panel for better cache locality
                let a_panel = a.slice(s![ic..ic + mc, pc..pc + kc]);

                // Call micro-kernel for this block
                simd_gemm_micro_kernel_f64(
                    alpha,
                    &a_panel,
                    &b_panel,
                    &mut c.slice_mut(s![ic..ic + mc, jc..jc + nc]),
                    &bs,
                )?;
            }
        }
    }

    Ok(())
}

/// SIMD micro-kernel for f32 GEMM operations
///
/// This performs the innermost computation with SIMD vectorization.
/// Processes blocks of size MR x NR with fully vectorized inner loops.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier
/// * `a` - A panel (MC x KC)
/// * `b` - B panel (KC x NC)
/// * `c` - C block to update (MC x NC)
/// * `block_sizes` - Micro-kernel dimensions
#[cfg(feature = "simd")]
fn simd_gemm_micro_kernel_f32(
    alpha: f32,
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
    c: &mut ndarray::ArrayViewMut2<f32>,
    block_sizes: &GemmBlockSizes,
) -> LinalgResult<()> {
    let (mc, kc) = a.dim();
    let (_, nc) = b.dim();
    let mr = block_sizes.mr;
    let nr = block_sizes.nr;

    // Process in micro-tiles of size MR x NR
    for ir in (0..mc).step_by(mr) {
        let mr_actual = mr.min(mc - ir);

        for jr in (0..nc).step_by(nr) {
            let nr_actual = nr.min(nc - jr);

            // Initialize accumulator registers
            let mut c_regs = vec![f32x8::splat(0.0); (mr_actual * nr_actual).div_ceil(8)];

            // Inner product loop over K dimension
            for p in 0..kc {
                // Load column vector from A
                let mut a_vec = Vec::with_capacity(mr_actual);
                for i in 0..mr_actual {
                    a_vec.push(a[[ir + i, p]]);
                }

                // Load row vector from B and broadcast, then multiply-accumulate
                for j in 0..nr_actual {
                    let b_val = b[[p, jr + j]];
                    let b_broadcast = f32x8::splat(b_val);

                    // Process A elements in chunks of 8
                    let mut a_idx = 0;
                    while a_idx + 8 <= mr_actual {
                        let a_chunk = [
                            a_vec[a_idx],
                            a_vec[a_idx + 1],
                            a_vec[a_idx + 2],
                            a_vec[a_idx + 3],
                            a_vec[a_idx + 4],
                            a_vec[a_idx + 5],
                            a_vec[a_idx + 6],
                            a_vec[a_idx + 7],
                        ];
                        let a_simd = f32x8::new(a_chunk);

                        let reg_idx = (j * mr_actual + a_idx) / 8;
                        c_regs[reg_idx] += a_simd * b_broadcast;

                        a_idx += 8;
                    }

                    // Handle remaining elements
                    #[allow(clippy::needless_range_loop)]
                    for i in a_idx..mr_actual {
                        let reg_idx = (j * mr_actual + i) / 8;
                        let lane = (j * mr_actual + i) % 8;

                        // For simplicity, accumulate remaining elements separately
                        let mut temp_array: [f32; 8] = c_regs[reg_idx].into();
                        temp_array[lane] += a_vec[i] * b_val;
                        c_regs[reg_idx] = f32x8::new(temp_array);
                    }
                }
            }

            // Store results back to C with alpha scaling
            for j in 0..nr_actual {
                for i in 0..mr_actual {
                    let reg_idx = (j * mr_actual + i) / 8;
                    let lane = (j * mr_actual + i) % 8;

                    let temp_array: [f32; 8] = c_regs[reg_idx].into();
                    let result = alpha * temp_array[lane];
                    c[[ir + i, jr + j]] += result;
                }
            }
        }
    }

    Ok(())
}

/// SIMD micro-kernel for f64 GEMM operations
///
/// This performs the innermost computation with SIMD vectorization for f64.
/// Uses 4-wide SIMD vectors for double precision.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier
/// * `a` - A panel (MC x KC)
/// * `b` - B panel (KC x NC)
/// * `c` - C block to update (MC x NC)
/// * `block_sizes` - Micro-kernel dimensions
#[cfg(feature = "simd")]
fn simd_gemm_micro_kernel_f64(
    alpha: f64,
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    c: &mut ndarray::ArrayViewMut2<f64>,
    block_sizes: &GemmBlockSizes,
) -> LinalgResult<()> {
    let (mc, kc) = a.dim();
    let (_, nc) = b.dim();
    let mr = block_sizes.mr;
    let nr = block_sizes.nr;

    // Process in micro-tiles of size MR x NR
    for ir in (0..mc).step_by(mr) {
        let mr_actual = mr.min(mc - ir);

        for jr in (0..nc).step_by(nr) {
            let nr_actual = nr.min(nc - jr);

            // Initialize accumulator registers (4-wide for f64)
            let mut c_regs = vec![f64x4::splat(0.0); (mr_actual * nr_actual).div_ceil(4)];

            // Inner product loop over K dimension
            for p in 0..kc {
                // Load column vector from A
                let mut a_vec = Vec::with_capacity(mr_actual);
                for i in 0..mr_actual {
                    a_vec.push(a[[ir + i, p]]);
                }

                // Load row vector from B and broadcast, then multiply-accumulate
                for j in 0..nr_actual {
                    let b_val = b[[p, jr + j]];
                    let b_broadcast = f64x4::splat(b_val);

                    // Process A elements in chunks of 4
                    let mut a_idx = 0;
                    while a_idx + 4 <= mr_actual {
                        let a_chunk = [
                            a_vec[a_idx],
                            a_vec[a_idx + 1],
                            a_vec[a_idx + 2],
                            a_vec[a_idx + 3],
                        ];
                        let a_simd = f64x4::new(a_chunk);

                        let reg_idx = (j * mr_actual + a_idx) / 4;
                        c_regs[reg_idx] += a_simd * b_broadcast;

                        a_idx += 4;
                    }

                    // Handle remaining elements
                    #[allow(clippy::needless_range_loop)]
                    for i in a_idx..mr_actual {
                        let reg_idx = (j * mr_actual + i) / 4;
                        let lane = (j * mr_actual + i) % 4;

                        // For simplicity, accumulate remaining elements separately
                        let mut temp_array: [f64; 4] = c_regs[reg_idx].into();
                        temp_array[lane] += a_vec[i] * b_val;
                        c_regs[reg_idx] = f64x4::new(temp_array);
                    }
                }
            }

            // Store results back to C with alpha scaling
            for j in 0..nr_actual {
                for i in 0..mr_actual {
                    let reg_idx = (j * mr_actual + i) / 4;
                    let lane = (j * mr_actual + i) % 4;

                    let temp_array: [f64; 4] = c_regs[reg_idx].into();
                    let result = alpha * temp_array[lane];
                    c[[ir + i, jr + j]] += result;
                }
            }
        }
    }

    Ok(())
}

/// Convenience function for SIMD matrix multiplication: C = A * B
///
/// # Arguments
///
/// * `a` - Left matrix (M x K)
/// * `b` - Right matrix (K x N)
///
/// # Returns
///
/// * Result matrix C (M x N)
#[cfg(feature = "simd")]
pub fn simd_matmul_optimized_f32(
    a: &ArrayView2<f32>,
    b: &ArrayView2<f32>,
) -> LinalgResult<Array2<f32>> {
    let (m, _) = a.dim();
    let (_, n) = b.dim();

    let mut c = Array2::zeros((m, n));
    simd_gemm_f32(1.0, a, b, 0.0, &mut c, None)?;
    Ok(c)
}

/// Convenience function for SIMD matrix multiplication: C = A * B (f64)
///
/// # Arguments
///
/// * `a` - Left matrix (M x K)
/// * `b` - Right matrix (K x N)
///
/// # Returns
///
/// * Result matrix C (M x N)
#[cfg(feature = "simd")]
pub fn simd_matmul_optimized_f64(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
) -> LinalgResult<Array2<f64>> {
    let (m, _) = a.dim();
    let (_, n) = b.dim();

    let mut c = Array2::zeros((m, n));
    simd_gemm_f64(1.0, a, b, 0.0, &mut c, None)?;
    Ok(c)
}

/// SIMD-accelerated matrix-vector multiplication: y = alpha * A * x + beta * y
///
/// Optimized version using the GEMM infrastructure for matrix-vector products.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*x
/// * `a` - Matrix A (M x N)
/// * `x` - Vector x (N,)
/// * `beta` - Scalar multiplier for y
/// * `y` - Result vector y (M,), updated in-place
///
/// # Returns
///
/// * Result indicating success or error
#[cfg(feature = "simd")]
pub fn simd_gemv_f32(
    alpha: f32,
    a: &ArrayView2<f32>,
    x: &ArrayView1<f32>,
    beta: f32,
    y: &mut Array1<f32>,
) -> LinalgResult<()> {
    let (m, n) = a.dim();

    if x.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "Vector x length ({}) must match matrix columns ({})",
            x.len(),
            n
        )));
    }
    if y.len() != m {
        return Err(LinalgError::ShapeError(format!(
            "Vector y length ({}) must match matrix rows ({})",
            y.len(),
            m
        )));
    }

    // Reshape x as a column vector for GEMM
    let x_matrix = x.insert_axis(Axis(1));
    let mut y_matrix = y.view().insert_axis(Axis(1)).to_owned();

    simd_gemm_f32(alpha, a, &x_matrix.view(), beta, &mut y_matrix, None)?;

    // Copy results back to y
    for (i, &val) in y_matrix.column(0).iter().enumerate() {
        y[i] = val;
    }

    Ok(())
}

/// SIMD-accelerated matrix-vector multiplication: y = alpha * A * x + beta * y (f64)
///
/// Optimized version using the GEMM infrastructure for matrix-vector products.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*x
/// * `a` - Matrix A (M x N)
/// * `x` - Vector x (N,)
/// * `beta` - Scalar multiplier for y
/// * `y` - Result vector y (M,), updated in-place
///
/// # Returns
///
/// * Result indicating success or error
#[cfg(feature = "simd")]
pub fn simd_gemv_f64(
    alpha: f64,
    a: &ArrayView2<f64>,
    x: &ArrayView1<f64>,
    beta: f64,
    y: &mut Array1<f64>,
) -> LinalgResult<()> {
    let (m, n) = a.dim();

    if x.len() != n {
        return Err(LinalgError::ShapeError(format!(
            "Vector x length ({}) must match matrix columns ({})",
            x.len(),
            n
        )));
    }
    if y.len() != m {
        return Err(LinalgError::ShapeError(format!(
            "Vector y length ({}) must match matrix rows ({})",
            y.len(),
            m
        )));
    }

    // Reshape x as a column vector for GEMM
    let x_matrix = x.insert_axis(Axis(1));
    let mut y_matrix = y.view().insert_axis(Axis(1)).to_owned();

    simd_gemm_f64(alpha, a, &x_matrix.view(), beta, &mut y_matrix, None)?;

    // Copy results back to y
    for (i, &val) in y_matrix.column(0).iter().enumerate() {
        y[i] = val;
    }

    Ok(())
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_gemm_f32_basic() {
        // Test C = A * B
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let mut c = Array2::zeros((2, 2));

        simd_gemm_f32(1.0, &a.view(), &b.view(), 0.0, &mut c, None).unwrap();

        // Expected: [[58, 64], [139, 154]]
        // A*B = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //     = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //     = [[58, 64], [139, 154]]
        assert_relative_eq!(c[[0, 0]], 58.0, epsilon = 1e-6);
        assert_relative_eq!(c[[0, 1]], 64.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 0]], 139.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 1]], 154.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_gemm_f64_basic() {
        // Test C = A * B
        let a = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f64, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let mut c = Array2::zeros((2, 2));

        simd_gemm_f64(1.0, &a.view(), &b.view(), 0.0, &mut c, None).unwrap();

        // Expected: [[58, 64], [139, 154]]
        assert_relative_eq!(c[[0, 0]], 58.0, epsilon = 1e-12);
        assert_relative_eq!(c[[0, 1]], 64.0, epsilon = 1e-12);
        assert_relative_eq!(c[[1, 0]], 139.0, epsilon = 1e-12);
        assert_relative_eq!(c[[1, 1]], 154.0, epsilon = 1e-12);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_gemm_alpha_beta() {
        // Test C = alpha * A * B + beta * C
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0], [7.0, 8.0]];
        let mut c = array![[1.0f32, 2.0], [3.0, 4.0]];

        let alpha = 2.0;
        let beta = 3.0;

        simd_gemm_f32(alpha, &a.view(), &b.view(), beta, &mut c, None).unwrap();

        // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        // Expected: 2.0 * [[19, 22], [43, 50]] + 3.0 * [[1, 2], [3, 4]]
        //         = [[38, 44], [86, 100]] + [[3, 6], [9, 12]]
        //         = [[41, 50], [95, 112]]
        assert_relative_eq!(c[[0, 0]], 41.0, epsilon = 1e-6);
        assert_relative_eq!(c[[0, 1]], 50.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 0]], 95.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 1]], 112.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_matmul_optimized() {
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f32, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let c = simd_matmul_optimized_f32(&a.view(), &b.view()).unwrap();

        assert_relative_eq!(c[[0, 0]], 58.0, epsilon = 1e-6);
        assert_relative_eq!(c[[0, 1]], 64.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 0]], 139.0, epsilon = 1e-6);
        assert_relative_eq!(c[[1, 1]], 154.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_gemv() {
        // Test y = alpha * A * x + beta * y
        let a = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = array![7.0f32, 8.0, 9.0];
        let mut y = array![1.0f32, 2.0];

        let alpha = 2.0;
        let beta = 3.0;

        simd_gemv_f32(alpha, &a.view(), &x.view(), beta, &mut y).unwrap();

        // A*x = [1*7+2*8+3*9, 4*7+5*8+6*9] = [7+16+27, 28+40+54] = [50, 122]
        // Expected: 2.0 * [50, 122] + 3.0 * [1, 2] = [100, 244] + [3, 6] = [103, 250]
        assert_relative_eq!(y[0], 103.0, epsilon = 1e-6);
        assert_relative_eq!(y[1], 250.0, epsilon = 1e-6);
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_gemm_large_matrix() {
        // Test with larger matrices to exercise blocking
        let m = 100;
        let k = 80;
        let n = 60;

        let a = Array2::from_shape_fn((m, k), |(i, j)| (i + j) as f32 * 0.01);
        let b = Array2::from_shape_fn((k, n), |(i, j)| (i * 2 + j) as f32 * 0.01);
        let mut c = Array2::zeros((m, n));

        // Test with custom block sizes
        let block_sizes = GemmBlockSizes {
            mc: 32,
            kc: 64,
            nc: 48,
            mr: 8,
            nr: 8,
        };

        simd_gemm_f32(1.0, &a.view(), &b.view(), 0.0, &mut c, Some(block_sizes)).unwrap();

        // Verify with reference implementation (naive multiplication)
        let c_ref = a.dot(&b);

        for ((i, j), &val) in c.indexed_iter() {
            assert_relative_eq!(val, c_ref[[i, j]], epsilon = 1e-4);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_gemm_error_handling() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]]; // Wrong dimensions
        let mut c = Array2::zeros((2, 3));

        let result = simd_gemm_f32(1.0, &a.view(), &b.view(), 0.0, &mut c, None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LinalgError::ShapeError(_)));
    }
}
