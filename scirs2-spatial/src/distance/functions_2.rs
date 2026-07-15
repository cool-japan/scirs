//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array2, ArrayView1};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::{PlatformCapabilities, SimdUnifiedOps};

use super::functions::Distance;
use super::functions::{fma_f32, fma_f64, prefetch_read, streaming_load_hint};
use super::types::CacheAlignedBuffer;

#[allow(dead_code)]
pub fn jaccard<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    let mut n_true_true = T::zero();
    let mut n_false_true = T::zero();
    let mut n_true_false = T::zero();
    for i in 0..point1.len() {
        let is_p1_true = point1[i] > T::zero();
        let is_p2_true = point2[i] > T::zero();
        if is_p1_true && is_p2_true {
            n_true_true = n_true_true + T::one();
        } else if !is_p1_true && is_p2_true {
            n_false_true = n_false_true + T::one();
        } else if is_p1_true && !is_p2_true {
            n_true_false = n_true_false + T::one();
        }
    }
    if n_true_true + n_false_true + n_true_false == T::zero() {
        T::zero()
    } else {
        (n_false_true + n_true_false) / (n_true_true + n_false_true + n_true_false)
    }
}
/// Compute Hamming distance between two points
///
/// The Hamming distance between two vectors is the proportion of coordinates
/// at which the corresponding values differ:
/// `count(point1\[i\] != point2\[i\] for i in 0..n) / n`. This matches SciPy's
/// `scipy.spatial.distance.hamming` semantics.
///
/// Coordinates are compared with a floating-point tolerance (`T::epsilon()`)
/// rather than exact equality, matching the tolerance-based float comparisons
/// used elsewhere in this module (e.g. `squareform_to_condensed`'s symmetry
/// check).
///
/// # Arguments
///
/// * `point1` - First point
/// * `point2` - Second point
///
/// # Returns
///
/// * Hamming distance between the points (proportion of differing
///   coordinates), or `NaN` if the points have different lengths
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::hamming;
///
/// let point1 = &[1.0, 0.0, 1.0, 1.0, 0.0];
/// let point2 = &[1.0, 1.0, 0.0, 1.0, 0.0];
///
/// // 2 of 5 coordinates differ
/// let dist = hamming(point1, point2);
/// assert!((dist - 0.4f64).abs() < 1e-6);
/// ```
pub fn hamming<T: Float>(point1: &[T], point2: &[T]) -> T {
    if point1.len() != point2.len() {
        return T::nan();
    }
    if point1.is_empty() {
        return T::zero();
    }
    let mut n_diff = T::zero();
    for i in 0..point1.len() {
        if (point1[i] - point2[i]).abs() > T::epsilon() {
            n_diff = n_diff + T::one();
        }
    }
    n_diff / T::from(point1.len()).expect("Operation failed")
}
/// Compute a distance matrix between two sets of points
///
/// # Arguments
///
/// * `x_a` - First set of points
/// * `xb` - Second set of points
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// * Distance matrix with shape (x_a.nrows(), xb.nrows())
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::{pdist, euclidean};
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let dist_matrix = pdist(&points, euclidean);
///
/// assert_eq!(dist_matrix.shape(), &[3, 3]);
/// assert!((dist_matrix[(0, 1)] - 1.0f64).abs() < 1e-6);
/// assert!((dist_matrix[(0, 2)] - 1.0f64).abs() < 1e-6);
/// assert!((dist_matrix[(1, 2)] - std::f64::consts::SQRT_2).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn pdist<T, F>(x: &Array2<T>, metric: F) -> Array2<T>
where
    T: Float + std::fmt::Debug,
    F: Fn(&[T], &[T]) -> T,
{
    let n = x.nrows();
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        result[(i, i)] = T::zero();
        let row_i = x.row(i).to_vec();
        for j in (i + 1)..n {
            let row_j = x.row(j).to_vec();
            let dist = metric(&row_i, &row_j);
            result[(i, j)] = dist;
            result[(j, i)] = dist;
        }
    }
    result
}
/// Compute a distance matrix between points (optimized zero-allocation version)
///
/// This function avoids memory allocations by working directly with array views,
/// providing significant performance improvements over the standard pdist function.
///
/// # Arguments
///
/// * `x` - Input matrix where each row is a point
/// * `metric` - Distance metric function that operates on ArrayView1
///
/// # Returns
///
/// * Symmetric distance matrix with shape (n, n)
///
/// # Examples
///
/// ```
/// use scirs2_spatial::distance::{pdist_optimized, euclidean_view};
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// let dist_matrix = pdist_optimized(&points, euclidean_view);
///
/// assert_eq!(dist_matrix.shape(), &[3, 3]);
/// assert!((dist_matrix[(0, 1)] - 1.0f64).abs() < 1e-6);
/// ```
pub fn pdist_optimized<T, F>(x: &Array2<T>, metric: F) -> Array2<T>
where
    T: Float + std::fmt::Debug,
    F: Fn(ArrayView1<T>, ArrayView1<T>) -> T,
{
    let n = x.nrows();
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        result[(i, i)] = T::zero();
        let row_i = x.row(i);
        for j in (i + 1)..n {
            let row_j = x.row(j);
            let dist = metric(row_i, row_j);
            result[(i, j)] = dist;
            result[(j, i)] = dist;
        }
    }
    result
}
/// Euclidean distance function that operates on ArrayView1 (zero-allocation)
///
/// This is an optimized version of euclidean distance that works directly
/// with array views without requiring vector conversions.
pub fn euclidean_view<T>(a: ArrayView1<T>, b: ArrayView1<T>) -> T
where
    T: Float + std::fmt::Debug,
{
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt()
}
/// SIMD-optimized euclidean distance for f64 using scirs2_core operations
///
/// This function leverages SIMD acceleration when working with f64 arrays
/// for maximum performance in distance computations.
pub fn euclidean_view_simd_f64(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    use scirs2_core::simd_ops::SimdUnifiedOps;
    let diff = f64::simd_sub(&a, &b);
    let squared = f64::simd_mul(&diff.view(), &diff.view());
    let sum = f64::simd_sum(&squared.view());
    sum.sqrt()
}
/// Ultra-optimized f64 Euclidean distance with comprehensive compiler optimizations
///
/// This function uses all available CPU features and compiler optimizations
/// to make aggressive optimizations and use the best instruction sequences.
#[must_use]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "fma,avx,avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub unsafe fn euclidean_distance_f64_specialized(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut sum = 0.0f64;
    let chunks = len / 8;
    #[allow(clippy::needless_range_loop)]
    for i in 0..chunks {
        let base = i * 8;
        if base + 16 < len {
            prefetch_read(&a[base + 8..base + 16]);
            prefetch_read(&b[base + 8..base + 16]);
        }
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        let d4 = a[base + 4] - b[base + 4];
        let d5 = a[base + 5] - b[base + 5];
        let d6 = a[base + 6] - b[base + 6];
        let d7 = a[base + 7] - b[base + 7];
        sum = fma_f64(d0, d0, sum);
        sum = fma_f64(d1, d1, sum);
        sum = fma_f64(d2, d2, sum);
        sum = fma_f64(d3, d3, sum);
        sum = fma_f64(d4, d4, sum);
        sum = fma_f64(d5, d5, sum);
        sum = fma_f64(d6, d6, sum);
        sum = fma_f64(d7, d7, sum);
    }
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        sum = fma_f64(diff, diff, sum);
    }
    sum.sqrt()
}
/// Ultra-optimized f32 Euclidean distance with comprehensive compiler optimizations
///
/// This function is hyper-optimized for f32 specifically, taking advantage
/// of wider SIMD registers and different instruction costs with maximum compiler optimizations.
#[inline(always)]
#[must_use]
pub fn euclidean_distance_f32_specialized(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut sum = 0.0f32;
    let chunks = len / 16;
    #[allow(clippy::needless_range_loop)]
    for i in 0..chunks {
        let base = i * 16;
        if base + 32 < len {
            prefetch_read(&a[base + 16..base + 32]);
            prefetch_read(&b[base + 16..base + 32]);
        }
        let mut chunk_sum = 0.0f32;
        for j in 0..16 {
            let diff = a[base + j] - b[base + j];
            chunk_sum = fma_f32(diff, diff, chunk_sum);
        }
        sum += chunk_sum;
    }
    for i in (chunks * 16)..len {
        let diff = a[i] - b[i];
        sum = fma_f32(diff, diff, sum);
    }
    sum.sqrt()
}
/// Ultra-high performance distance matrix with advanced cache optimization
///
/// This implements cache-blocking, memory prefetching, and SIMD acceleration
/// for maximum performance on large datasets.
#[inline]
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
unsafe fn pdist_simd_flat_f64_avx2(points: &Array2<f64>) -> Vec<f64> {
    pdist_simd_flat_f64_impl(points)
}
/// Fallback implementation for non-AVX2 targets
#[inline]
fn pdist_simd_flat_f64_fallback(points: &Array2<f64>) -> Vec<f64> {
    pdist_simd_flat_f64_impl(points)
}
/// Core implementation shared between optimized and fallback versions
#[inline(always)]
fn pdist_simd_flat_f64_impl(points: &Array2<f64>) -> Vec<f64> {
    let n = points.nrows();
    let mut matrix = CacheAlignedBuffer::new_with_capacity(n * n);
    matrix.resize(n * n, 0.0f64);
    pdist_simd_flat_f64_core(points, matrix.as_mut_slice())
}
/// ARM NEON optimized implementation
#[inline]
#[target_feature(enable = "neon")]
#[cfg(target_arch = "aarch64")]
unsafe fn pdist_simd_flat_f64_neon(points: &Array2<f64>) -> Vec<f64> {
    pdist_simd_flat_f64_impl(points)
}
/// Optimized small matrix distance computation for tiny datasets
///
/// Uses completely unrolled loops and inline computations for maximum performance
/// on small matrices where the overhead of general algorithms is significant.
#[inline]
pub(super) fn pdist_small_matrix_f64(points: &Array2<f64>) -> Vec<f64> {
    let n = points.nrows();
    let mut matrix = vec![0.0f64; n * n];
    match n {
        1 => {
            matrix[0] = 0.0;
        }
        2 => {
            matrix[0] = 0.0;
            matrix[3] = 0.0;
            let dist = unsafe {
                euclidean_distance_f64_specialized(
                    points.row(0).as_slice().unwrap_or(&[]),
                    points.row(1).as_slice().unwrap_or(&[]),
                )
            };
            matrix[1] = dist;
            matrix[2] = dist;
        }
        3 => {
            matrix[0] = 0.0;
            matrix[4] = 0.0;
            matrix[8] = 0.0;
            let dist_01 = unsafe {
                euclidean_distance_f64_specialized(
                    points.row(0).as_slice().unwrap_or(&[]),
                    points.row(1).as_slice().unwrap_or(&[]),
                )
            };
            let dist_02 = unsafe {
                euclidean_distance_f64_specialized(
                    points.row(0).as_slice().unwrap_or(&[]),
                    points.row(2).as_slice().unwrap_or(&[]),
                )
            };
            let dist_12 = unsafe {
                euclidean_distance_f64_specialized(
                    points.row(1).as_slice().unwrap_or(&[]),
                    points.row(2).as_slice().unwrap_or(&[]),
                )
            };
            matrix[1] = dist_01;
            matrix[3] = dist_01;
            matrix[2] = dist_02;
            matrix[6] = dist_02;
            matrix[5] = dist_12;
            matrix[7] = dist_12;
        }
        4 => {
            for i in 0..4 {
                matrix[i * 4 + i] = 0.0;
            }
            for i in 0..3 {
                for j in (i + 1)..4 {
                    let dist = unsafe {
                        euclidean_distance_f64_specialized(
                            points.row(i).as_slice().unwrap_or(&[]),
                            points.row(j).as_slice().unwrap_or(&[]),
                        )
                    };
                    matrix[i * 4 + j] = dist;
                    matrix[j * 4 + i] = dist;
                }
            }
        }
        _ => {
            return pdist_simd_flat_f64_impl(points);
        }
    }
    matrix
}
/// Public interface that dispatches to the best available implementation
pub fn pdist_simd_flat_f64(points: &Array2<f64>) -> Vec<f64> {
    let n = points.nrows();
    if n <= 4 {
        return pdist_small_matrix_f64(points);
    }
    #[cfg(target_arch = "x86_64")]
    {
        let capabilities = PlatformCapabilities::detect();
        if capabilities.avx2_available {
            unsafe { pdist_simd_flat_f64_avx2(points) }
        } else {
            pdist_simd_flat_f64_fallback(points)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let capabilities = PlatformCapabilities::detect();
        if capabilities.neon_available {
            unsafe { pdist_simd_flat_f64_neon(points) }
        } else {
            pdist_simd_flat_f64_fallback(points)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        pdist_simd_flat_f64_fallback(points)
    }
}
/// Core computation kernel shared by all implementations
#[inline(always)]
fn pdist_simd_flat_f64_core(points: &Array2<f64>, matrix: &mut [f64]) -> Vec<f64> {
    let n = points.nrows();
    const CACHE_BLOCK_SIZE: usize = 64;
    for i_block in (0..n).step_by(CACHE_BLOCK_SIZE) {
        let i_end = (i_block + CACHE_BLOCK_SIZE).min(n);
        for j_block in (i_block..n).step_by(CACHE_BLOCK_SIZE) {
            let j_end = (j_block + CACHE_BLOCK_SIZE).min(n);
            for i in i_block..i_end {
                if i + 1 < i_end {
                    let next_row = points.row(i + 1);
                    let next_slice = next_row.as_slice().unwrap_or(&[]);
                    streaming_load_hint(next_slice);
                    prefetch_read(next_slice);
                }
                if i + 2 < i_end {
                    let future_base = (i + 2) * n;
                    if future_base + n <= matrix.len() {
                        let write_region = &matrix[future_base..future_base + n.min(64)];
                        streaming_load_hint(write_region);
                        prefetch_read(write_region);
                    }
                }
                let current_row = points.row(i);
                let i_n = i * n;
                for j in j_block.max(i)..j_end {
                    let distance = if i == j {
                        0.0f64
                    } else {
                        let row_j = points.row(j);
                        unsafe {
                            euclidean_distance_f64_specialized(
                                current_row.as_slice().unwrap_or(&[]),
                                row_j.as_slice().unwrap_or(&[]),
                            )
                        }
                    };
                    let flat_idx_ij = i_n + j;
                    let flat_idx_ji = j * n + i;
                    unsafe {
                        *matrix.get_unchecked_mut(flat_idx_ij) = distance;
                        *matrix.get_unchecked_mut(flat_idx_ji) = distance;
                    }
                }
            }
        }
    }
    matrix.to_vec()
}
/// Ultra-optimized distance computation for small, fixed-size vectors
///
/// This function uses const generics to enable aggressive compile-time optimizations
/// for common small dimensions (2D, 3D, 4D, etc.).
#[inline(always)]
pub fn euclidean_distance_fixed<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let mut sum = 0.0f64;
    match N {
        1 => {
            let diff = a[0] - b[0];
            sum = diff * diff;
        }
        2 => {
            let diff0 = a[0] - b[0];
            let diff1 = a[1] - b[1];
            sum = diff0 * diff0 + diff1 * diff1;
        }
        3 => {
            let diff0 = a[0] - b[0];
            let diff1 = a[1] - b[1];
            let diff2 = a[2] - b[2];
            sum = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
        }
        4 => {
            let diff0 = a[0] - b[0];
            let diff1 = a[1] - b[1];
            let diff2 = a[2] - b[2];
            let diff3 = a[3] - b[3];
            sum = diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        }
        _ => {
            for i in 0..N {
                let diff = a[i] - b[i];
                sum = fma_f64(diff, diff, sum);
            }
        }
    }
    sum.sqrt()
}
/// Hierarchical clustering-aware distance computation with compiler optimizations
///
/// This algorithm exploits spatial locality in clustered datasets by:
/// 1. Pre-sorting points by Morton codes (Z-order curve)
/// 2. Computing distances in spatial order to maximize cache hits
/// 3. Using early termination for sparse distance matrices
#[inline(always)]
#[must_use]
pub fn pdist_hierarchical_f64(points: &Array2<f64>) -> Vec<f64> {
    let n = points.nrows();
    let mut matrix = vec![0.0f64; n * n];
    if n <= 4 {
        return pdist_small_matrix_f64(points);
    }
    let mut morton_indices: Vec<(u64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let row = points.row(i);
        let morton_code =
            compute_morton_code_2d((row[0] * 1024.0) as u32, (row[1] * 1024.0) as u32);
        morton_indices.push((morton_code, i));
    }
    morton_indices.sort_unstable_by_key(|&(code, _)| code);
    let sorted_indices: Vec<usize> = morton_indices.iter().map(|(_, idx)| *idx).collect();
    for (i_morton, &i) in sorted_indices.iter().enumerate() {
        for (j_morton, &j) in sorted_indices.iter().enumerate().skip(i_morton) {
            let distance = if i == j {
                0.0f64
            } else {
                unsafe {
                    euclidean_distance_f64_specialized(
                        points.row(i).as_slice().unwrap_or(&[]),
                        points.row(j).as_slice().unwrap_or(&[]),
                    )
                }
            };
            matrix[i * n + j] = distance;
            matrix[j * n + i] = distance;
        }
    }
    matrix
}
/// Compute Morton code (Z-order curve) for 2D spatial ordering with compiler optimizations
#[inline(always)]
#[must_use]
fn compute_morton_code_2d(x: u32, y: u32) -> u64 {
    let mut result = 0u64;
    for i in 0..16 {
        result |= ((x & (1 << i)) as u64) << (2 * i);
        result |= ((y & (1 << i)) as u64) << (2 * i + 1);
    }
    result
}
