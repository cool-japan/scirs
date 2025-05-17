//! Performance optimizations for large matrices
//!
//! This module provides optimized implementations of linear algebra operations
//! specifically designed for large matrices, including cache-friendly algorithms,
//! parallelization, and memory layout optimizations.

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayView2, Axis};
use num_traits::{Float, NumAssign};
use rayon::prelude::*;
use std::cmp;
use std::iter::Sum;

/// Algorithm selection for optimized operations
#[derive(Debug, Clone, Copy)]
pub enum OptAlgorithm {
    /// Use standard algorithm
    Standard,
    /// Use blocked algorithm
    Blocked,
    /// Use parallel algorithms
    Parallel,
    /// Automatically select based on matrix size
    Adaptive,
}

/// Configuration for performance-optimized operations
#[derive(Debug, Clone)]
pub struct OptConfig {
    /// Block size for cache-friendly algorithms
    pub block_size: usize,
    /// Threshold for using parallel algorithms
    pub parallel_threshold: usize,
    /// Number of threads for parallel operations (None = use default)
    pub num_threads: Option<usize>,
    /// Algorithm selection
    pub algorithm: OptAlgorithm,
}

impl Default for OptConfig {
    fn default() -> Self {
        OptConfig {
            block_size: 64,
            parallel_threshold: 1000,
            num_threads: None,
            algorithm: OptAlgorithm::Blocked,
        }
    }
}

impl OptConfig {
    /// Builder pattern methods
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    pub fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    pub fn with_num_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }

    pub fn with_algorithm(mut self, algorithm: OptAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

/// Cache-friendly blocked matrix multiplication
///
/// This implementation uses loop tiling to improve cache locality
/// for large matrix multiplications.
pub fn blocked_matmul<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    config: &OptConfig,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + 'static,
{
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());

    if k != k2 {
        return Err(LinalgError::DimensionError(format!(
            "Matrix dimensions don't match: ({}, {}) x ({}, {})",
            m, k, k2, n
        )));
    }

    let mut c = Array2::zeros((m, n));
    let block_size = config.block_size;

    match config.algorithm {
        OptAlgorithm::Standard => Ok(a.dot(b)),
        OptAlgorithm::Blocked => {
            serial_blocked_matmul(a, b, &mut c, block_size)?;
            Ok(c)
        }
        OptAlgorithm::Parallel => {
            parallel_blocked_matmul(a, b, &mut c, block_size)?;
            Ok(c)
        }
        OptAlgorithm::Adaptive => {
            // Use parallel processing for large matrices
            if m * n > config.parallel_threshold {
                parallel_blocked_matmul(a, b, &mut c, block_size)?;
            } else if m * n > 10000 {
                serial_blocked_matmul(a, b, &mut c, block_size)?;
            } else {
                return Ok(a.dot(b));
            }
            Ok(c)
        }
    }
}

/// Serial blocked matrix multiplication
fn serial_blocked_matmul<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    c: &mut Array2<F>,
    block_size: usize,
) -> LinalgResult<()>
where
    F: Float + NumAssign + Sum,
{
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();

    // Loop tiling for better cache performance
    for ii in (0..m).step_by(block_size) {
        for jj in (0..n).step_by(block_size) {
            for kk in (0..k).step_by(block_size) {
                // Process block
                let i_end = cmp::min(ii + block_size, m);
                let j_end = cmp::min(jj + block_size, n);
                let k_end = cmp::min(kk + block_size, k);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = c[[i, j]];
                        for ki in kk..k_end {
                            sum = sum + a[[i, ki]] * b[[ki, j]];
                        }
                        c[[i, j]] = sum;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Parallel blocked matrix multiplication using Rayon
fn parallel_blocked_matmul<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    c: &mut Array2<F>,
    block_size: usize,
) -> LinalgResult<()>
where
    F: Float + NumAssign + Sum + Send + Sync,
{
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();

    // Create blocks for parallel processing
    let block_indices: Vec<(usize, usize)> = (0..m)
        .step_by(block_size)
        .flat_map(|i| (0..n).step_by(block_size).map(move |j| (i, j)))
        .collect();

    // Process blocks in parallel and collect results
    let results: Vec<_> = block_indices
        .par_iter()
        .map(|&(ii, jj)| {
            let i_end = cmp::min(ii + block_size, m);
            let j_end = cmp::min(jj + block_size, n);

            // Create local accumulator for this block
            let mut local_c = Array2::zeros((i_end - ii, j_end - jj));

            // Compute block multiplication
            for kk in (0..k).step_by(block_size) {
                let k_end = cmp::min(kk + block_size, k);

                for (i_local, i) in (0..(i_end - ii)).zip(ii..i_end) {
                    for (j_local, j) in (0..(j_end - jj)).zip(jj..j_end) {
                        let mut sum = local_c[[i_local, j_local]];
                        for ki in kk..k_end {
                            sum = sum + a[[i, ki]] * b[[ki, j]];
                        }
                        local_c[[i_local, j_local]] = sum;
                    }
                }
            }

            // Return the block and its position
            ((ii, jj), local_c)
        })
        .collect();

    // Write results back to the main matrix
    for ((ii, jj), local_c) in results {
        let i_end = cmp::min(ii + block_size, m);
        let j_end = cmp::min(jj + block_size, n);

        for (i_local, i) in (0..(i_end - ii)).zip(ii..i_end) {
            for (j_local, j) in (0..(j_end - jj)).zip(jj..j_end) {
                c[[i, j]] = local_c[[i_local, j_local]];
            }
        }
    }

    Ok(())
}

/// Optimized matrix transpose for better memory access patterns
pub fn optimized_transpose<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + Send + Sync,
{
    let (m, n) = (a.nrows(), a.ncols());
    let mut result = Array2::zeros((n, m));

    // Use blocked transpose for better cache performance
    let block_size = 32;

    for i in (0..m).step_by(block_size) {
        for j in (0..n).step_by(block_size) {
            let i_end = cmp::min(i + block_size, m);
            let j_end = cmp::min(j + block_size, n);

            // Transpose block
            for ii in i..i_end {
                for jj in j..j_end {
                    result[[jj, ii]] = a[[ii, jj]];
                }
            }
        }
    }

    Ok(result)
}

/// Parallel matrix-vector multiplication for large matrices
pub fn parallel_matvec<F>(
    a: &ArrayView2<F>,
    x: &ArrayView2<F>,
    config: &OptConfig,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync,
{
    if a.ncols() != x.nrows() {
        return Err(LinalgError::DimensionError(format!(
            "Matrix and vector dimensions don't match: ({}, {}) x ({}, {})",
            a.nrows(),
            a.ncols(),
            x.nrows(),
            x.ncols()
        )));
    }

    let m = a.nrows();
    let n = x.ncols();
    let mut result = Array2::zeros((m, n));

    if m > config.parallel_threshold {
        // Parallel computation for large matrices
        result
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n {
                    let mut sum = F::zero();
                    for k in 0..a.ncols() {
                        sum = sum + a[[i, k]] * x[[k, j]];
                    }
                    row[j] = sum;
                }
            });
    } else {
        // Serial computation for smaller matrices
        for i in 0..m {
            for j in 0..n {
                let mut sum = F::zero();
                for k in 0..a.ncols() {
                    sum = sum + a[[i, k]] * x[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }
    }

    Ok(result)
}

/// Memory-efficient in-place matrix operations
pub mod inplace {
    use super::*;

    /// In-place matrix addition: A += B
    pub fn add_assign<F>(a: &mut Array2<F>, b: &ArrayView2<F>) -> LinalgResult<()>
    where
        F: Float + NumAssign,
    {
        if a.shape() != b.shape() {
            return Err(LinalgError::DimensionError(format!(
                "Matrix dimensions don't match: {:?} != {:?}",
                a.shape(),
                b.shape()
            )));
        }

        for (a_elem, b_elem) in a.iter_mut().zip(b.iter()) {
            *a_elem = *a_elem + *b_elem;
        }

        Ok(())
    }

    /// In-place scalar multiplication: A *= scalar
    pub fn scalar_mul_assign<F>(a: &mut Array2<F>, scalar: F) -> LinalgResult<()>
    where
        F: Float + NumAssign + Send + Sync,
    {
        for elem in a.iter_mut() {
            *elem = *elem * scalar;
        }

        Ok(())
    }

    /// In-place transpose for square matrices
    pub fn transpose_square<F>(a: &mut Array2<F>) -> LinalgResult<()>
    where
        F: Float,
    {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(LinalgError::DimensionError(
                "In-place transpose requires square matrix".to_string(),
            ));
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let temp = a[[i, j]];
                a[[i, j]] = a[[j, i]];
                a[[j, i]] = temp;
            }
        }

        Ok(())
    }
}

/// Adaptive algorithm selection based on matrix properties
pub fn adaptive_matmul<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + 'static,
{
    let size = a.nrows() * a.ncols() + b.nrows() * b.ncols();

    // Choose algorithm based on matrix size
    if size < 10000 {
        // Small matrices: use standard ndarray multiplication
        Ok(a.dot(b))
    } else if size < 1000000 {
        // Medium matrices: use blocked algorithm
        let config = OptConfig::default();
        blocked_matmul(a, b, &config)
    } else {
        // Large matrices: use parallel blocked algorithm
        let mut config = OptConfig::default();
        config.parallel_threshold = 50000;
        blocked_matmul(a, b, &config)
    }
}

/// Convenience function for in-place matrix addition
pub fn inplace_add<F>(a: &mut ndarray::ArrayViewMut2<F>, b: &ArrayView2<F>) -> LinalgResult<()>
where
    F: Float + NumAssign + Send + Sync,
{
    let mut a_owned = a.to_owned();
    inplace::add_assign(&mut a_owned, b)?;
    a.assign(&a_owned);
    Ok(())
}

/// Convenience function for in-place scalar multiplication
pub fn inplace_scale<F>(a: &mut ndarray::ArrayViewMut2<F>, scalar: F) -> LinalgResult<()>
where
    F: Float + NumAssign + Send + Sync,
{
    for elem in a.iter_mut() {
        *elem = *elem * scalar;
    }
    Ok(())
}

/// Simple benchmarking utility for matrix multiplication
pub fn matmul_benchmark<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    config: &OptConfig,
) -> LinalgResult<String>
where
    F: Float + NumAssign + Sum + Send + Sync + 'static,
{
    use std::time::Instant;

    // Benchmark standard multiplication
    let start = Instant::now();
    let _c1 = a.dot(b);
    let time_standard = start.elapsed();

    // Benchmark optimized multiplication
    let start = Instant::now();
    let _c2 = blocked_matmul(a, b, config)?;
    let time_optimized = start.elapsed();

    Ok(format!(
        "Standard: {:?}, Optimized: {:?}, Speedup: {:.2}x",
        time_standard,
        time_optimized,
        time_standard.as_secs_f64() / time_optimized.as_secs_f64()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_blocked_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let config = OptConfig {
            block_size: 1,
            parallel_threshold: 1000,
            num_threads: None,
            algorithm: OptAlgorithm::Blocked,
        };

        let c = blocked_matmul(&a.view(), &b.view(), &config).unwrap();

        assert_relative_eq!(c[[0, 0]], 19.0);
        assert_relative_eq!(c[[0, 1]], 22.0);
        assert_relative_eq!(c[[1, 0]], 43.0);
        assert_relative_eq!(c[[1, 1]], 50.0);
    }

    #[test]
    fn test_optimized_transpose() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let a_t = optimized_transpose(&a.view()).unwrap();

        assert_eq!(a_t.shape(), &[3, 2]);
        assert_relative_eq!(a_t[[0, 0]], 1.0);
        assert_relative_eq!(a_t[[1, 0]], 2.0);
        assert_relative_eq!(a_t[[2, 0]], 3.0);
        assert_relative_eq!(a_t[[0, 1]], 4.0);
        assert_relative_eq!(a_t[[1, 1]], 5.0);
        assert_relative_eq!(a_t[[2, 1]], 6.0);
    }

    #[test]
    fn test_parallel_matvec() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let x = array![[5.0], [6.0]];

        let config = OptConfig::default();
        let y = parallel_matvec(&a.view(), &x.view(), &config).unwrap();

        assert_relative_eq!(y[[0, 0]], 17.0);
        assert_relative_eq!(y[[1, 0]], 39.0);
    }

    #[test]
    fn test_inplace_operations() {
        let mut a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        inplace::add_assign(&mut a, &b.view()).unwrap();

        assert_relative_eq!(a[[0, 0]], 6.0);
        assert_relative_eq!(a[[0, 1]], 8.0);
        assert_relative_eq!(a[[1, 0]], 10.0);
        assert_relative_eq!(a[[1, 1]], 12.0);

        inplace::scalar_mul_assign(&mut a, 2.0).unwrap();

        assert_relative_eq!(a[[0, 0]], 12.0);
        assert_relative_eq!(a[[0, 1]], 16.0);
        assert_relative_eq!(a[[1, 0]], 20.0);
        assert_relative_eq!(a[[1, 1]], 24.0);
    }

    #[test]
    fn test_inplace_transpose() {
        let mut a = array![[1.0, 2.0], [3.0, 4.0]];

        inplace::transpose_square(&mut a).unwrap();

        assert_relative_eq!(a[[0, 0]], 1.0);
        assert_relative_eq!(a[[0, 1]], 3.0);
        assert_relative_eq!(a[[1, 0]], 2.0);
        assert_relative_eq!(a[[1, 1]], 4.0);
    }

    #[test]
    fn test_adaptive_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let c = adaptive_matmul(&a.view(), &b.view()).unwrap();

        assert_relative_eq!(c[[0, 0]], 19.0);
        assert_relative_eq!(c[[0, 1]], 22.0);
        assert_relative_eq!(c[[1, 0]], 43.0);
        assert_relative_eq!(c[[1, 1]], 50.0);
    }

    #[test]
    fn test_large_matrix_blocked() {
        // Test with larger matrix to verify blocking works correctly
        let n = 100;
        let a = Array2::from_shape_fn((n, n), |(i, j)| (i + j) as f64);
        let b = Array2::eye(n);

        let config = OptConfig {
            block_size: 16,
            parallel_threshold: 10000,
            num_threads: None,
            algorithm: OptAlgorithm::Blocked,
        };

        let c = blocked_matmul(&a.view(), &b.view(), &config).unwrap();

        // Multiplying by identity should give original matrix
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(c[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }
}
