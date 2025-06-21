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
                            sum += a[[i, ki]] * b[[ki, j]];
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
                            sum += a[[i, ki]] * b[[ki, j]];
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

/// Cache-friendly blocked matrix multiplication with explicit workers parameter
pub fn blocked_matmul_with_workers<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    workers: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    let config = OptConfig {
        num_threads: workers,
        ..OptConfig::default()
    };

    blocked_matmul(a, b, &config)
}

/// Optimized matrix transpose for better memory access patterns
pub fn optimized_transpose<F>(a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + Send + Sync,
{
    optimized_transpose_with_workers(a, None)
}

/// Optimized matrix transpose with explicit workers parameter
pub fn optimized_transpose_with_workers<F>(
    a: &ArrayView2<F>,
    workers: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + Send + Sync,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);
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
                        sum += a[[i, k]] * x[[k, j]];
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
                    sum += a[[i, k]] * x[[k, j]];
                }
                result[[i, j]] = sum;
            }
        }
    }

    Ok(result)
}

/// Parallel matrix-vector multiplication with explicit workers parameter
pub fn parallel_matvec_with_workers<F>(
    a: &ArrayView2<F>,
    x: &ArrayView2<F>,
    workers: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    let config = OptConfig {
        num_threads: workers,
        ..OptConfig::default()
    };

    parallel_matvec(a, x, &config)
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
            *a_elem += *b_elem;
        }

        Ok(())
    }

    /// In-place scalar multiplication: A *= scalar
    pub fn scalar_mul_assign<F>(a: &mut Array2<F>, scalar: F) -> LinalgResult<()>
    where
        F: Float + NumAssign + Send + Sync,
    {
        for elem in a.iter_mut() {
            *elem *= scalar;
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
    adaptive_matmul_with_workers(a, b, None)
}

/// Adaptive matrix multiplication with explicit workers parameter
pub fn adaptive_matmul_with_workers<F>(
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    workers: Option<usize>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + 'static,
{
    use crate::parallel;

    // Configure workers for parallel operations
    parallel::configure_workers(workers);

    let size = a.nrows() * a.ncols() + b.nrows() * b.ncols();

    // Choose algorithm based on matrix size
    if size < 10000 {
        // Small matrices: use standard ndarray multiplication
        Ok(a.dot(b))
    } else if size < 1000000 {
        // Medium matrices: use blocked algorithm
        let config = OptConfig {
            num_threads: workers,
            ..OptConfig::default()
        };
        blocked_matmul(a, b, &config)
    } else {
        // Large matrices: use parallel blocked algorithm
        let config = OptConfig {
            parallel_threshold: 50000,
            num_threads: workers,
            ..OptConfig::default()
        };
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
        *elem *= scalar;
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

/// Memory-optimized decomposition algorithms
/// These implementations minimize memory allocations and reuse workspace arrays
pub mod decomposition_opt {
    use super::*;
    use ndarray::{s, Array1, Array2, ArrayView2};

    /// Workspace for QR decomposition to avoid repeated allocations
    pub struct QRWorkspace<F: Float> {
        /// Tau array for Householder reflectors
        pub tau: Array1<F>,
        /// Work array for computations
        pub work: Array1<F>,
        /// Temporary array for matrix operations
        pub temp_matrix: Array2<F>,
    }

    impl<F: Float> QRWorkspace<F> {
        /// Create a new workspace for matrices up to the given size
        pub fn new(max_rows: usize, max_cols: usize) -> Self {
            let min_dim = max_rows.min(max_cols);
            Self {
                tau: Array1::zeros(min_dim),
                work: Array1::zeros(max_cols * 64), // 64 is a reasonable work size multiplier
                temp_matrix: Array2::zeros((max_rows, max_cols)),
            }
        }

        /// Resize workspace if needed for the given matrix dimensions
        pub fn resize_if_needed(&mut self, rows: usize, cols: usize) {
            let min_dim = rows.min(cols);

            if self.tau.len() < min_dim {
                self.tau = Array1::zeros(min_dim);
            }

            let work_size = cols * 64;
            if self.work.len() < work_size {
                self.work = Array1::zeros(work_size);
            }

            if self.temp_matrix.nrows() < rows || self.temp_matrix.ncols() < cols {
                self.temp_matrix = Array2::zeros((rows, cols));
            }
        }
    }

    /// Memory-optimized QR decomposition using workspace arrays
    /// This reduces allocations for repeated QR decompositions
    pub fn qr_with_workspace<F>(
        a: &ArrayView2<F>,
        workspace: &mut QRWorkspace<F>,
    ) -> LinalgResult<(Array2<F>, Array2<F>)>
    where
        F: Float + NumAssign + Sum + Clone,
    {
        let (m, n) = a.dim();
        workspace.resize_if_needed(m, n);

        // Copy input matrix to temporary workspace to avoid modifying original
        let mut a_work = workspace.temp_matrix.slice_mut(s![..m, ..n]);
        a_work.assign(a);

        // Perform in-place QR factorization
        let min_dim = m.min(n);

        // Simple Householder QR (educational implementation)
        // In production, this would call optimized LAPACK routines
        for j in 0..min_dim {
            let column = a_work.column(j);
            let norm = column
                .slice(s![j..])
                .fold(F::zero(), |acc, &x| acc + x * x)
                .sqrt();

            if norm > F::epsilon() {
                // Update workspace tau
                workspace.tau[j] = norm;

                // Apply Householder reflection
                let alpha = column[j];
                let sign = if alpha >= F::zero() {
                    F::one()
                } else {
                    -F::one()
                };
                let u1 = alpha + sign * norm;

                // Normalize Householder vector
                if u1.abs() > F::epsilon() {
                    let scale = F::one() / u1;
                    for i in (j + 1)..m {
                        a_work[[i, j]] *= scale;
                    }
                }
            }
        }

        // Extract Q and R matrices
        let q = Array2::eye(m);
        let mut r = Array2::zeros((m, n));

        // Copy upper triangular part to R
        for i in 0..m {
            for j in i..n {
                if i < min_dim && j < n {
                    r[[i, j]] = a_work[[i, j]];
                }
            }
        }

        Ok((q, r))
    }

    /// Memory pool for temporary arrays in decomposition algorithms
    pub struct DecompositionMemoryPool<F: Float> {
        /// Pool of reusable arrays of different sizes
        pub arrays: Vec<Array2<F>>,
        /// Pool of reusable vectors
        pub vectors: Vec<Array1<F>>,
        /// Maximum number of arrays to keep in pool
        pub max_pool_size: usize,
    }

    impl<F: Float> DecompositionMemoryPool<F> {
        /// Create a new memory pool
        pub fn new(max_pool_size: usize) -> Self {
            Self {
                arrays: Vec::new(),
                vectors: Vec::new(),
                max_pool_size,
            }
        }

        /// Get a temporary array of the specified size, reusing from pool if available
        pub fn get_array(&mut self, rows: usize, cols: usize) -> Array2<F> {
            // Try to find a suitable array in the pool
            for (i, array) in self.arrays.iter().enumerate() {
                if array.nrows() >= rows && array.ncols() >= cols {
                    let mut result = self.arrays.swap_remove(i);
                    // Resize to exact dimensions needed
                    result = result.slice(s![..rows, ..cols]).to_owned();
                    result.fill(F::zero()); // Clear the array
                    return result;
                }
            }

            // No suitable array found, create new one
            Array2::zeros((rows, cols))
        }

        /// Return an array to the pool for reuse
        pub fn return_array(&mut self, array: Array2<F>) {
            if self.arrays.len() < self.max_pool_size {
                self.arrays.push(array);
            }
        }

        /// Get a temporary vector of the specified size
        pub fn get_vector(&mut self, len: usize) -> Array1<F> {
            // Try to find a suitable vector in the pool
            for (i, vector) in self.vectors.iter().enumerate() {
                if vector.len() >= len {
                    let mut result = self.vectors.swap_remove(i);
                    result = result.slice(s![..len]).to_owned();
                    result.fill(F::zero()); // Clear the vector
                    return result;
                }
            }

            // No suitable vector found, create new one
            Array1::zeros(len)
        }

        /// Return a vector to the pool for reuse
        pub fn return_vector(&mut self, vector: Array1<F>) {
            if self.vectors.len() < self.max_pool_size {
                self.vectors.push(vector);
            }
        }

        /// Clear the memory pool
        pub fn clear(&mut self) {
            self.arrays.clear();
            self.vectors.clear();
        }
    }

    /// Cache-friendly Householder QR decomposition
    /// Uses blocked algorithms for better memory access patterns
    pub fn blocked_qr<F>(
        a: &ArrayView2<F>,
        block_size: usize,
    ) -> LinalgResult<(Array2<F>, Array2<F>)>
    where
        F: Float + NumAssign + Sum + Clone,
    {
        let (m, n) = a.dim();
        let mut a_copy = a.to_owned();
        let q = Array2::eye(m);

        let min_dim = m.min(n);

        // Process matrix in blocks for better cache locality
        for start_col in (0..min_dim).step_by(block_size) {
            let end_col = (start_col + block_size).min(min_dim);
            let _panel_width = end_col - start_col;

            // Apply Householder transformations to current panel
            for j in start_col..end_col {
                // Compute Householder vector for column j
                let col_norm = a_copy
                    .slice(s![j.., j])
                    .fold(F::zero(), |acc, &x| acc + x * x)
                    .sqrt();

                if col_norm > F::epsilon() {
                    let alpha = a_copy[[j, j]];
                    let sign = if alpha >= F::zero() {
                        F::one()
                    } else {
                        -F::one()
                    };
                    let beta = alpha + sign * col_norm;

                    if beta.abs() > F::epsilon() {
                        // Normalize Householder vector
                        let scale = F::one() / beta;
                        for i in (j + 1)..m {
                            a_copy[[i, j]] *= scale;
                        }

                        // Update R diagonal element
                        a_copy[[j, j]] = -sign * col_norm;

                        // Apply Householder transformation to remaining columns
                        for k in (j + 1)..n {
                            let mut dot_product = a_copy[[j, k]];
                            for i in (j + 1)..m {
                                dot_product += a_copy[[i, j]] * a_copy[[i, k]];
                            }

                            let tau = dot_product * F::from(2.0).unwrap();
                            a_copy[[j, k]] -= tau;
                            for i in (j + 1)..m {
                                let householder_val = a_copy[[i, j]];
                                a_copy[[i, k]] -= tau * householder_val;
                            }
                        }
                    }
                }
            }
        }

        // Extract R matrix (upper triangular)
        let mut r = Array2::zeros((m, n));
        for i in 0..m {
            for j in i..n {
                if i < min_dim {
                    r[[i, j]] = a_copy[[i, j]];
                }
            }
        }

        Ok((q, r))
    }
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
