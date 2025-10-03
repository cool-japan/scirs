//! Parallel algorithm dispatch for linear algebra operations
//!
//! This module provides utilities for automatically selecting and dispatching
//! to parallel implementations when appropriate, based on matrix size and
//! worker configuration.

use crate::error::LinalgResult;
use crate::parallel::{algorithms, WorkerConfig};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::iter::Sum;

/// Parallel-aware matrix decomposition dispatcher
pub struct ParallelDecomposition;

impl ParallelDecomposition {
    /// Choose and execute the appropriate Cholesky decomposition implementation
    pub fn cholesky<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Sum + Send + Sync + scirs2_core::ndarray::ScalarOperand + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_cholesky(a, &config);
            }
        }

        // Fall back to standard implementation
        crate::decomposition::cholesky(a, workers)
    }

    /// Choose and execute the appropriate LU decomposition implementation
    pub fn lu<F>(
        a: &ArrayView2<F>,
        workers: Option<usize>,
    ) -> LinalgResult<(Array2<F>, Array2<F>, Array2<F>)>
    where
        F: Float
            + NumAssign
            + One
            + Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_lu(a, &config);
            }
        }

        // Fall back to standard implementation
        crate::decomposition::lu(a, workers)
    }

    /// Choose and execute the appropriate QR decomposition implementation
    pub fn qr<F>(a: &ArrayView2<F>, workers: Option<usize>) -> LinalgResult<(Array2<F>, Array2<F>)>
    where
        F: Float
            + NumAssign
            + One
            + Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_qr(a, &config);
            }
        }

        // Fall back to standard implementation
        crate::decomposition::qr(a, workers)
    }

    /// Choose and execute the appropriate SVD implementation
    pub fn svd<F>(
        a: &ArrayView2<F>,
        full_matrices: bool,
        workers: Option<usize>,
    ) -> LinalgResult<(Array2<F>, Array1<F>, Array2<F>)>
    where
        F: Float
            + NumAssign
            + One
            + Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large _matrices (only supports reduced form)
            if m * n > config.parallel_threshold && !full_matrices {
                return algorithms::parallel_svd(a, &config);
            }
        }

        // Fall back to standard implementation
        crate::decomposition::svd(a, full_matrices, workers)
    }
}

/// Parallel-aware solver dispatcher
pub struct ParallelSolver;

impl ParallelSolver {
    /// Choose and execute the appropriate conjugate gradient implementation
    pub fn conjugate_gradient<F>(
        a: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        workers: Option<usize>,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float
            + NumAssign
            + One
            + Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_conjugate_gradient(a, b, max_iter, tolerance, &config);
            }
        }

        // Fall back to standard implementation
        crate::iterative_solvers::conjugate_gradient(a, b, max_iter, tolerance, None)
    }

    /// Choose and execute the appropriate GMRES implementation
    pub fn gmres<F>(
        a: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        restart: usize,
        workers: Option<usize>,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float
            + NumAssign
            + One
            + Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + std::fmt::Debug
            + std::fmt::Display
            + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_gmres(a, b, max_iter, tolerance, restart, &config);
            }
        }

        // Fall back to standard implementation
        let options = crate::solvers::iterative::IterativeSolverOptions {
            max_iterations: max_iter,
            tolerance,
            verbose: false,
            restart: Some(restart),
        };
        crate::solvers::iterative::gmres(a, b, None, &options).map(|result| result.solution)
    }

    /// Choose and execute the appropriate BiCGSTAB implementation
    pub fn bicgstab<F>(
        a: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        workers: Option<usize>,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float
            + NumAssign
            + One
            + Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_bicgstab(a, b, max_iter, tolerance, &config);
            }
        }

        // Fall back to standard implementation
        crate::iterative_solvers::bicgstab(a, b, max_iter, tolerance, None)
    }

    /// Choose and execute the appropriate Jacobi method implementation
    pub fn jacobi<F>(
        a: &ArrayView2<F>,
        b: &ArrayView1<F>,
        max_iter: usize,
        tolerance: F,
        workers: Option<usize>,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float
            + NumAssign
            + One
            + Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_jacobi(a, b, max_iter, tolerance, &config);
            }
        }

        // Fall back to standard implementation
        crate::iterative_solvers::jacobi_method(a, b, max_iter, tolerance, None)
    }

    /// Choose and execute the appropriate SOR method implementation
    pub fn sor<F>(
        a: &ArrayView2<F>,
        b: &ArrayView1<F>,
        omega: F,
        max_iter: usize,
        tolerance: F,
        workers: Option<usize>,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float
            + NumAssign
            + One
            + Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_sor(a, b, omega, max_iter, tolerance, &config);
            }
        }

        // Fall back to standard implementation
        crate::iterative_solvers::successive_over_relaxation(a, b, omega, max_iter, tolerance, None)
    }
}

/// Parallel-aware matrix operations dispatcher
pub struct ParallelOperations;

impl ParallelOperations {
    /// Choose and execute the appropriate matrix multiplication implementation
    pub fn matmul<F>(
        a: &ArrayView2<F>,
        b: &ArrayView2<F>,
        workers: Option<usize>,
    ) -> LinalgResult<Array2<F>>
    where
        F: Float + NumAssign + Zero + Sum + Send + Sync + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, k) = a.dim();
            let (_, n) = b.dim();

            // Use parallel implementation for large matrices
            if m * k * n > config.parallel_threshold {
                return algorithms::parallel_gemm(a, b, &config);
            }
        }

        // Fall back to standard implementation
        Ok(a.dot(b))
    }

    /// Choose and execute the appropriate matrix-vector multiplication implementation
    pub fn matvec<F>(
        a: &ArrayView2<F>,
        x: &ArrayView1<F>,
        workers: Option<usize>,
    ) -> LinalgResult<Array1<F>>
    where
        F: Float + Zero + Sum + Send + Sync + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_matvec(a, x, &config);
            }
        }

        // Fall back to standard implementation
        Ok(a.dot(x))
    }

    /// Choose and execute the appropriate power iteration implementation
    pub fn power_iteration<F>(
        a: &ArrayView2<F>,
        max_iter: usize,
        tolerance: F,
        workers: Option<usize>,
    ) -> LinalgResult<(F, Array1<F>)>
    where
        F: Float
            + NumAssign
            + One
            + Zero
            + Sum
            + Send
            + Sync
            + scirs2_core::ndarray::ScalarOperand
            + 'static,
    {
        if let Some(num_workers) = workers {
            let config = WorkerConfig::new().with_workers(num_workers);
            let (m, n) = a.dim();

            // Use parallel implementation for large matrices
            if m * n > config.parallel_threshold {
                return algorithms::parallel_power_iteration(a, max_iter, tolerance, &config);
            }
        }

        // Fall back to standard implementation
        crate::eigen::power_iteration(a, max_iter, tolerance)
    }
}

/// Configuration builder for parallel dispatch
pub struct ParallelConfig {
    workers: Option<usize>,
    threshold_multiplier: f64,
}

impl ParallelConfig {
    /// Create a new parallel configuration
    pub fn new() -> Self {
        Self {
            workers: None,
            threshold_multiplier: 1.0,
        }
    }

    /// Set the number of worker threads
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = Some(workers);
        self
    }

    /// Set the threshold multiplier for parallel execution
    ///
    /// A value of 2.0 means matrices need to be 2x larger than default
    /// threshold to use parallel implementation
    pub fn with_threshold_multiplier(mut self, multiplier: f64) -> Self {
        self.threshold_multiplier = multiplier;
        self
    }

    /// Build a WorkerConfig from this configuration
    pub fn build(&self) -> WorkerConfig {
        let mut config = WorkerConfig::new();

        if let Some(workers) = self.workers {
            config = config.with_workers(workers);
        }

        let base_threshold = config.parallel_threshold;
        config =
            config.with_threshold((base_threshold as f64 * self.threshold_multiplier) as usize);

        config
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_parallel_dispatch_smallmatrix() {
        // Small matrix should use serial implementation
        let a = array![[1.0, 2.0], [2.0, 5.0]];
        let result = ParallelDecomposition::cholesky(&a.view(), Some(4));
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_config_builder() {
        let config = ParallelConfig::new()
            .with_workers(8)
            .with_threshold_multiplier(2.0)
            .build();

        assert_eq!(config.workers, Some(8));
        assert_eq!(config.parallel_threshold, 2000); // Default 1000 * 2.0
    }
}
