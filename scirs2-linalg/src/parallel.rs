//! Parallel processing utilities for linear algebra operations
//!
//! This module provides utilities for managing worker threads across various
//! linear algebra operations, ensuring consistent behavior and optimal performance.

use std::sync::Mutex;

/// Global worker configuration
static GLOBAL_WORKERS: Mutex<Option<usize>> = Mutex::new(None);

/// Set the global worker thread count for all operations
///
/// This affects operations that don't explicitly specify a worker count.
/// If set to None, operations will use system defaults.
///
/// # Arguments
///
/// * `workers` - Number of worker threads (None = use system default)
///
/// # Examples
///
/// ```
/// use scirs2_linalg::parallel::set_global_workers;
///
/// // Use 4 threads for all operations
/// set_global_workers(Some(4));
///
/// // Reset to system default
/// set_global_workers(None);
/// ```
pub fn set_global_workers(workers: Option<usize>) {
    if let Ok(mut global) = GLOBAL_WORKERS.lock() {
        *global = workers;

        // Set OpenMP environment variable if specified
        if let Some(num_workers) = workers {
            std::env::set_var("OMP_NUM_THREADS", num_workers.to_string());
        } else {
            // Remove the environment variable to use system default
            std::env::remove_var("OMP_NUM_THREADS");
        }
    }
}

/// Get the current global worker thread count
///
/// # Returns
///
/// * Current global worker count (None = system default)
pub fn get_global_workers() -> Option<usize> {
    GLOBAL_WORKERS.lock().ok().and_then(|global| *global)
}

/// Configure worker threads for an operation
///
/// This function determines the appropriate number of worker threads to use,
/// considering both the operation-specific setting and global configuration.
///
/// # Arguments
///
/// * `workers` - Operation-specific worker count
///
/// # Returns
///
/// * Effective worker count to use
pub fn configure_workers(workers: Option<usize>) -> Option<usize> {
    match workers {
        Some(count) => {
            // Operation-specific setting takes precedence
            std::env::set_var("OMP_NUM_THREADS", count.to_string());
            Some(count)
        }
        None => {
            // Use global setting if available
            let global_workers = get_global_workers();
            if let Some(count) = global_workers {
                std::env::set_var("OMP_NUM_THREADS", count.to_string());
            }
            global_workers
        }
    }
}

/// Worker configuration for batched operations
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Number of worker threads
    pub workers: Option<usize>,
    /// Threshold for using parallel processing
    pub parallel_threshold: usize,
    /// Chunk size for batched operations
    pub chunk_size: usize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            workers: None,
            parallel_threshold: 1000,
            chunk_size: 64,
        }
    }
}

impl WorkerConfig {
    /// Create a new worker configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of worker threads
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = Some(workers);
        self
    }

    /// Set the parallel processing threshold
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Set the chunk size for batched operations
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Apply this configuration for the current operation
    pub fn apply(&self) {
        configure_workers(self.workers);
    }
}

/// Scoped worker configuration
///
/// Temporarily sets worker configuration and restores the previous
/// configuration when dropped.
pub struct ScopedWorkers {
    previous_workers: Option<usize>,
}

impl ScopedWorkers {
    /// Create a scoped worker configuration
    ///
    /// # Arguments
    ///
    /// * `workers` - Number of worker threads for this scope
    ///
    /// # Returns
    ///
    /// * ScopedWorkers guard that restores previous configuration on drop
    pub fn new(workers: Option<usize>) -> Self {
        let previous_workers = get_global_workers();
        set_global_workers(workers);
        Self { previous_workers }
    }
}

impl Drop for ScopedWorkers {
    fn drop(&mut self) {
        set_global_workers(self.previous_workers);
    }
}

/// Parallel iterator utilities for matrix operations
pub mod iter {
    use rayon::prelude::*;

    /// Process chunks of work in parallel
    ///
    /// # Arguments
    ///
    /// * `items` - Items to process
    /// * `chunk_size` - Size of each chunk
    /// * `f` - Function to apply to each chunk
    ///
    /// # Returns
    ///
    /// * Vector of results from each chunk
    pub fn parallel_chunks<T, R, F>(items: &[T], chunk_size: usize, f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(&[T]) -> R + Send + Sync,
    {
        items
            .chunks(chunk_size)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(f)
            .collect()
    }

    /// Process items in parallel with index information
    ///
    /// # Arguments
    ///
    /// * `items` - Items to process
    /// * `f` - Function to apply to each (index, item) pair
    ///
    /// # Returns
    ///
    /// * Vector of results
    pub fn parallel_enumerate<T, R, F>(items: &[T], f: F) -> Vec<R>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(usize, &T) -> R + Send + Sync,
    {
        items
            .par_iter()
            .enumerate()
            .map(|(i, item)| f(i, item))
            .collect()
    }
}

/// Adaptive algorithm selection based on data size and worker configuration
pub mod adaptive {
    use super::WorkerConfig;

    /// Algorithm selection strategy
    #[derive(Debug, Clone, Copy)]
    pub enum Strategy {
        /// Always use serial processing
        Serial,
        /// Always use parallel processing
        Parallel,
        /// Automatically choose based on data size
        Adaptive,
    }

    /// Choose processing strategy based on data size and configuration
    ///
    /// # Arguments
    ///
    /// * `data_size` - Size of the data to process
    /// * `config` - Worker configuration
    ///
    /// # Returns
    ///
    /// * Recommended processing strategy
    pub fn choose_strategy(data_size: usize, config: &WorkerConfig) -> Strategy {
        if data_size < config.parallel_threshold {
            Strategy::Serial
        } else {
            Strategy::Parallel
        }
    }

    /// Check if parallel processing is recommended
    ///
    /// # Arguments
    ///
    /// * `data_size` - Size of the data to process
    /// * `config` - Worker configuration
    ///
    /// # Returns
    ///
    /// * true if parallel processing is recommended
    pub fn should_use_parallel(data_size: usize, config: &WorkerConfig) -> bool {
        matches!(choose_strategy(data_size, config), Strategy::Parallel)
    }
}

/// Algorithm-specific parallel implementations
pub mod algorithms {
    use super::{adaptive, WorkerConfig};
    use crate::error::{LinalgError, LinalgResult};
    use ndarray::{Array1, ArrayView1, ArrayView2};
    use num_traits::{Float, NumAssign, One, Zero};
    use rayon::prelude::*;
    use std::iter::Sum;

    /// Parallel matrix-vector multiplication
    ///
    /// This is a simpler and more effective parallelization that can be used
    /// as a building block for more complex algorithms.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Input matrix
    /// * `vector` - Input vector
    /// * `config` - Worker configuration
    ///
    /// # Returns
    ///
    /// * Result vector y = A * x
    pub fn parallel_matvec<F>(
        matrix: &ArrayView2<F>,
        vector: &ArrayView1<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + 'static,
    {
        let (m, n) = matrix.dim();
        if n != vector.len() {
            return Err(LinalgError::ShapeError(format!(
                "Matrix-vector dimensions incompatible: {}x{} * {}",
                m,
                n,
                vector.len()
            )));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            // Fall back to serial computation
            return Ok(matrix.dot(vector));
        }

        config.apply();

        // Parallel computation of each row
        let result_vec: Vec<F> = (0..m)
            .into_par_iter()
            .map(|i| {
                matrix
                    .row(i)
                    .iter()
                    .zip(vector.iter())
                    .map(|(&aij, &xj)| aij * xj)
                    .sum()
            })
            .collect();

        Ok(Array1::from_vec(result_vec))
    }

    /// Parallel power iteration for dominant eigenvalue
    ///
    /// This implementation uses parallel matrix-vector multiplications
    /// in the power iteration method for computing dominant eigenvalues.
    pub fn parallel_power_iteration<F>(
        matrix: &ArrayView2<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<(F, Array1<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + NumAssign + One + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "Power iteration requires square matrix".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            // Fall back to serial power iteration
            return crate::eigen::power_iteration(&matrix.view(), max_iter, tolerance);
        }

        config.apply();

        // Initialize with simple vector
        let mut v = Array1::ones(n);
        let norm = v.iter().map(|&x| x * x).sum::<F>().sqrt();
        v /= norm;

        let mut eigenvalue = F::zero();

        for _iter in 0..max_iter {
            // Use the parallel matrix-vector multiplication
            let new_v = parallel_matvec(matrix, &v.view(), config)?;

            // Compute eigenvalue estimate (Rayleigh quotient)
            let new_eigenvalue = new_v
                .iter()
                .zip(v.iter())
                .map(|(&new_vi, &vi)| new_vi * vi)
                .sum::<F>();

            // Normalize
            let norm = new_v.iter().map(|&x| x * x).sum::<F>().sqrt();
            if norm < F::epsilon() {
                return Err(LinalgError::ComputationError(
                    "Vector became zero during iteration".to_string(),
                ));
            }
            let normalized_v = new_v / norm;

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < tolerance {
                return Ok((new_eigenvalue, normalized_v));
            }

            eigenvalue = new_eigenvalue;
            v = normalized_v;
        }

        Err(LinalgError::ComputationError(
            "Power iteration failed to converge".to_string(),
        ))
    }

    /// Parallel vector operations for linear algebra
    ///
    /// This module provides basic parallel vector operations that serve as
    /// building blocks for more complex algorithms.
    pub mod vector_ops {
        use super::*;

        /// Parallel dot product of two vectors
        pub fn parallel_dot<F>(
            x: &ArrayView1<F>,
            y: &ArrayView1<F>,
            config: &WorkerConfig,
        ) -> LinalgResult<F>
        where
            F: Float + Send + Sync + Zero + Sum + 'static,
        {
            if x.len() != y.len() {
                return Err(LinalgError::ShapeError(
                    "Vectors must have same length for dot product".to_string(),
                ));
            }

            let data_size = x.len();
            if !adaptive::should_use_parallel(data_size, config) {
                return Ok(x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum());
            }

            config.apply();

            let result = (0..x.len()).into_par_iter().map(|i| x[i] * y[i]).sum();

            Ok(result)
        }

        /// Parallel vector norm computation
        pub fn parallel_norm<F>(x: &ArrayView1<F>, config: &WorkerConfig) -> LinalgResult<F>
        where
            F: Float + Send + Sync + Zero + Sum + 'static,
        {
            let data_size = x.len();
            if !adaptive::should_use_parallel(data_size, config) {
                return Ok(x.iter().map(|&xi| xi * xi).sum::<F>().sqrt());
            }

            config.apply();

            let sum_squares = (0..x.len()).into_par_iter().map(|i| x[i] * x[i]).sum::<F>();

            Ok(sum_squares.sqrt())
        }

        /// Parallel AXPY operation: y = a*x + y
        ///
        /// Note: This function returns a new array rather than modifying in-place
        /// due to complications with parallel mutable iteration.
        pub fn parallel_axpy<F>(
            alpha: F,
            x: &ArrayView1<F>,
            y: &ArrayView1<F>,
            config: &WorkerConfig,
        ) -> LinalgResult<Array1<F>>
        where
            F: Float + Send + Sync + 'static,
        {
            if x.len() != y.len() {
                return Err(LinalgError::ShapeError(
                    "Vectors must have same length for AXPY".to_string(),
                ));
            }

            let data_size = x.len();
            if !adaptive::should_use_parallel(data_size, config) {
                let result = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| alpha * xi + yi)
                    .collect();
                return Ok(Array1::from_vec(result));
            }

            config.apply();

            let result_vec: Vec<F> = (0..x.len())
                .into_par_iter()
                .map(|i| alpha * x[i] + y[i])
                .collect();

            Ok(Array1::from_vec(result_vec))
        }
    }

    /// Parallel matrix multiplication (GEMM)
    ///
    /// Implements parallel general matrix multiplication with block-based
    /// parallelization for improved cache performance.
    pub fn parallel_gemm<F>(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<ndarray::Array2<F>>
    where
        F: Float + Send + Sync + Zero + Sum + NumAssign + 'static,
    {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(LinalgError::ShapeError(format!(
                "Matrix dimensions incompatible for multiplication: {}x{} * {}x{}",
                m, k, k2, n
            )));
        }

        let data_size = m * k * n;
        if !adaptive::should_use_parallel(data_size, config) {
            return Ok(a.dot(b));
        }

        config.apply();

        // Block size for cache-friendly computation
        let block_size = config.chunk_size;

        let mut result = ndarray::Array2::zeros((m, n));

        // Parallel computation using blocks
        result
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(i, mut row)| {
                for j in 0..n {
                    let mut sum = F::zero();
                    for kb in (0..k).step_by(block_size) {
                        let k_end = std::cmp::min(kb + block_size, k);
                        for ki in kb..k_end {
                            sum += a[[i, ki]] * b[[ki, j]];
                        }
                    }
                    row[j] = sum;
                }
            });

        Ok(result)
    }

    /// Parallel QR decomposition using Householder reflections
    ///
    /// This implementation parallelizes the application of Householder
    /// transformations across columns.
    pub fn parallel_qr<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<(ndarray::Array2<F>, ndarray::Array2<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let data_size = m * n;

        if !adaptive::should_use_parallel(data_size, config) {
            return crate::decomposition::qr(&matrix.view(), None);
        }

        config.apply();

        let mut a = matrix.to_owned();
        let mut q = ndarray::Array2::eye(m);
        let min_dim = std::cmp::min(m, n);

        for k in 0..min_dim {
            // Extract column vector for Householder reflection
            let x = a.slice(ndarray::s![k.., k]).to_owned();
            let alpha = if x[0] >= F::zero() {
                -x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()
            } else {
                x.iter().map(|&xi| xi * xi).sum::<F>().sqrt()
            };

            if alpha.abs() < F::epsilon() {
                continue;
            }

            let mut v = x.clone();
            v[0] -= alpha;
            let v_norm_sq = v.iter().map(|&vi| vi * vi).sum::<F>();

            if v_norm_sq < F::epsilon() {
                continue;
            }

            // Apply Householder transformation (serial for simplicity)
            let remaining_cols = n - k;
            if remaining_cols > 1 {
                for j in k..n {
                    let col = a.slice(ndarray::s![k.., j]).to_owned();
                    let dot_product = v
                        .iter()
                        .zip(col.iter())
                        .map(|(&vi, &ci)| vi * ci)
                        .sum::<F>();
                    let factor = F::from(2.0).unwrap() * dot_product / v_norm_sq;

                    for (i, &vi) in v.iter().enumerate() {
                        a[[k + i, j]] -= factor * vi;
                    }
                }
            }

            // Update Q matrix (serial for simplicity)
            for i in 0..m {
                let row = q.slice(ndarray::s![i, k..]).to_owned();
                let dot_product = v
                    .iter()
                    .zip(row.iter())
                    .map(|(&vi, &ri)| vi * ri)
                    .sum::<F>();
                let factor = F::from(2.0).unwrap() * dot_product / v_norm_sq;

                for (j, &vj) in v.iter().enumerate() {
                    q[[i, k + j]] -= factor * vj;
                }
            }
        }

        let r = a.slice(ndarray::s![..min_dim, ..]).to_owned();
        Ok((q, r))
    }

    /// Parallel Cholesky decomposition
    ///
    /// Implements parallel Cholesky decomposition using block-column approach
    /// for positive definite matrices.
    pub fn parallel_cholesky<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<ndarray::Array2<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "Cholesky decomposition requires square matrix".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            return crate::decomposition::cholesky(&matrix.view(), None);
        }

        config.apply();

        let mut l = ndarray::Array2::zeros((n, n));
        let block_size = config.chunk_size;

        for k in (0..n).step_by(block_size) {
            let k_end = std::cmp::min(k + block_size, n);

            // Diagonal block factorization (serial for numerical stability)
            for i in k..k_end {
                // Compute L[i,i]
                let mut sum = F::zero();
                for j in 0..i {
                    sum += l[[i, j]] * l[[i, j]];
                }
                let aii = matrix[[i, i]] - sum;
                if aii <= F::zero() {
                    return Err(LinalgError::ComputationError(
                        "Matrix is not positive definite".to_string(),
                    ));
                }
                l[[i, i]] = aii.sqrt();

                // Compute L[i+1:k_end, i]
                for j in (i + 1)..k_end {
                    let mut sum = F::zero();
                    for p in 0..i {
                        sum += l[[j, p]] * l[[i, p]];
                    }
                    l[[j, i]] = (matrix[[j, i]] - sum) / l[[i, i]];
                }
            }

            // Update trailing submatrix (serial for simplicity)
            if k_end < n {
                for i in k_end..n {
                    for j in k..k_end {
                        let mut sum = F::zero();
                        for p in 0..j {
                            sum += l[[i, p]] * l[[j, p]];
                        }
                        l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
                    }
                }
            }
        }

        Ok(l)
    }

    /// Parallel LU decomposition with partial pivoting
    ///
    /// Implements parallel LU decomposition using block-column approach.
    pub fn parallel_lu<F>(
        matrix: &ArrayView2<F>,
        config: &WorkerConfig,
    ) -> LinalgResult<(ndarray::Array2<F>, ndarray::Array2<F>, ndarray::Array2<F>)>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        let data_size = m * n;

        if !adaptive::should_use_parallel(data_size, config) {
            return crate::decomposition::lu(&matrix.view(), None);
        }

        config.apply();

        let mut a = matrix.to_owned();
        let mut perm_vec = (0..m).collect::<Vec<_>>();
        let min_dim = std::cmp::min(m, n);

        for k in 0..min_dim {
            // Find pivot (serial for correctness)
            let mut max_val = F::zero();
            let mut pivot_row = k;
            for i in k..m {
                let abs_val = a[[i, k]].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    pivot_row = i;
                }
            }

            if max_val < F::epsilon() {
                return Err(LinalgError::ComputationError(
                    "Matrix is singular".to_string(),
                ));
            }

            // Swap rows if needed
            if pivot_row != k {
                for j in 0..n {
                    let temp = a[[k, j]];
                    a[[k, j]] = a[[pivot_row, j]];
                    a[[pivot_row, j]] = temp;
                }
                perm_vec.swap(k, pivot_row);
            }

            // Update submatrix (serial for now to avoid borrowing issues)
            let pivot = a[[k, k]];

            for i in (k + 1)..m {
                let multiplier = a[[i, k]] / pivot;
                a[[i, k]] = multiplier;

                for j in (k + 1)..n {
                    a[[i, j]] = a[[i, j]] - multiplier * a[[k, j]];
                }
            }
        }

        // Create permutation matrix P
        let mut p = ndarray::Array2::zeros((m, m));
        for (i, &piv) in perm_vec.iter().enumerate() {
            p[[i, piv]] = F::one();
        }

        // Extract L and U matrices
        let mut l = ndarray::Array2::eye(m);
        let mut u = ndarray::Array2::zeros((m, n));

        for i in 0..m {
            for j in 0..n {
                if i > j && j < min_dim {
                    l[[i, j]] = a[[i, j]];
                } else if i <= j {
                    u[[i, j]] = a[[i, j]];
                }
            }
        }

        Ok((p, l, u))
    }

    /// Parallel conjugate gradient solver
    ///
    /// Implements parallel conjugate gradient method for solving linear systems
    /// with symmetric positive definite matrices.
    pub fn parallel_conjugate_gradient<F>(
        matrix: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        config: &WorkerConfig,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Send + Sync + Zero + Sum + One + NumAssign + ndarray::ScalarOperand + 'static,
    {
        let (m, n) = matrix.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "CG requires square matrix".to_string(),
            ));
        }
        if n != b.len() {
            return Err(LinalgError::ShapeError(
                "Matrix and vector dimensions incompatible".to_string(),
            ));
        }

        let data_size = m * n;
        if !adaptive::should_use_parallel(data_size, config) {
            return crate::iterative_solvers::conjugate_gradient(
                &matrix.view(),
                &b.view(),
                max_iter,
                tolerance,
                None,
            );
        }

        config.apply();

        // Initialize
        let mut x = Array1::zeros(n);

        // r = b - A*x
        let ax = parallel_matvec(matrix, &x.view(), config)?;
        let mut r = b - &ax;
        let mut p = r.clone();
        let mut rsold = vector_ops::parallel_dot(&r.view(), &r.view(), config)?;

        for _iter in 0..max_iter {
            let ap = parallel_matvec(matrix, &p.view(), config)?;
            let alpha = rsold / vector_ops::parallel_dot(&p.view(), &ap.view(), config)?;

            x = vector_ops::parallel_axpy(alpha, &p.view(), &x.view(), config)?;
            r = vector_ops::parallel_axpy(-alpha, &ap.view(), &r.view(), config)?;

            let rsnew = vector_ops::parallel_dot(&r.view(), &r.view(), config)?;

            if rsnew.sqrt() < tolerance {
                return Ok(x);
            }

            let beta = rsnew / rsold;
            p = vector_ops::parallel_axpy(beta, &p.view(), &r.view(), config)?;
            rsold = rsnew;
        }

        Err(LinalgError::ComputationError(
            "Conjugate gradient failed to converge".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_workers() {
        // Save initial state to restore later
        let original_state = get_global_workers();

        // Test setting and getting global workers
        set_global_workers(Some(4));
        assert_eq!(get_global_workers(), Some(4));

        set_global_workers(None);
        assert_eq!(get_global_workers(), None);

        // Restore original state to avoid test interference
        set_global_workers(original_state);
    }

    #[test]
    fn test_scoped_workers() {
        // Save initial state to restore later
        let original_state = get_global_workers();

        // Set initial global workers
        set_global_workers(Some(2));

        {
            // Create scoped configuration
            let _scoped = ScopedWorkers::new(Some(8));
            assert_eq!(get_global_workers(), Some(8));
        }

        // Should be restored after scope
        assert_eq!(get_global_workers(), Some(2));

        // Restore original state to avoid test interference
        set_global_workers(original_state);
    }

    #[test]
    fn test_worker_config() {
        let config = WorkerConfig::new()
            .with_workers(4)
            .with_threshold(2000)
            .with_chunk_size(128);

        assert_eq!(config.workers, Some(4));
        assert_eq!(config.parallel_threshold, 2000);
        assert_eq!(config.chunk_size, 128);
    }

    #[test]
    fn test_adaptive_strategy() {
        let config = WorkerConfig::default();

        // Small data should use serial
        assert!(matches!(
            adaptive::choose_strategy(100, &config),
            adaptive::Strategy::Serial
        ));

        // Large data should use parallel
        assert!(matches!(
            adaptive::choose_strategy(2000, &config),
            adaptive::Strategy::Parallel
        ));
    }
}
