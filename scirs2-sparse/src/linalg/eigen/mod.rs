//! Eigenvalue algorithms for sparse matrices
//!
//! This module provides a comprehensive set of eigenvalue solvers for sparse matrices,
//! organized by algorithm type and matrix properties.

pub mod general;
pub mod generalized;
pub mod lanczos;
pub mod power_iteration;
pub mod symmetric;

// Re-export main types and functions for convenience
pub use general::eigs;
pub use generalized::{eigsh_generalized, eigsh_generalized_enhanced};
pub use lanczos::{lanczos, EigenResult, LanczosOptions};
pub use power_iteration::{power_iteration, PowerIterationOptions};
pub use symmetric::{eigsh, eigsh_shift_invert, eigsh_shift_invert_enhanced};

// Common eigenvalue result type

// Compatibility types for missing imports
#[derive(Debug, Clone)]
pub struct ArpackOptions {
    pub max_iter: usize,
    pub tol: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum EigenvalueMethod {
    Lanczos,
    PowerIteration,
    Arpack,
}

#[derive(Debug, Clone, Copy)]
pub enum EigenvalueMode {
    Normal,
    ShiftInvert,
    Buckling,
}

/// Eigenvalue solver configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct EigenSolverConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of eigenvalues to compute
    pub num_eigenvalues: usize,
    /// Whether to compute eigenvectors
    pub compute_eigenvectors: bool,
    /// Which eigenvalues to compute ("LA", "SA", "LM", "SM")
    pub which: String,
}

impl Default for EigenSolverConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tolerance: 1e-8,
            num_eigenvalues: 6,
            compute_eigenvectors: true,
            which: "LA".to_string(),
        }
    }
}

impl EigenSolverConfig {
    /// Create a new eigenvalue solver configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set number of eigenvalues to compute
    pub fn with_num_eigenvalues(mut self, num_eigenvalues: usize) -> Self {
        self.num_eigenvalues = num_eigenvalues;
        self
    }

    /// Set whether to compute eigenvectors
    pub fn with_compute_eigenvectors(mut self, compute_eigenvectors: bool) -> Self {
        self.compute_eigenvectors = compute_eigenvectors;
        self
    }

    /// Set which eigenvalues to compute
    pub fn with_which(mut self, which: &str) -> Self {
        self.which = which.to_string();
        self
    }

    /// Convert to LanczosOptions
    pub fn to_lanczos_options(&self) -> LanczosOptions {
        LanczosOptions {
            max_iter: self.max_iter,
            max_subspace_size: (self.num_eigenvalues * 2 + 10).max(20),
            tol: self.tolerance,
            numeigenvalues: self.num_eigenvalues,
            compute_eigenvectors: self.compute_eigenvectors,
        }
    }

    /// Convert to PowerIterationOptions
    pub fn to_power_iteration_options(&self) -> PowerIterationOptions {
        PowerIterationOptions {
            max_iter: self.max_iter,
            tol: self.tolerance,
            normalize: true,
        }
    }
}

/// Eigenvalue solver selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenSolverStrategy {
    /// Power iteration for single largest eigenvalue
    PowerIteration,
    /// Lanczos algorithm for multiple eigenvalues
    Lanczos,
    /// Symmetric eigenvalue solver (eigsh)
    Symmetric,
    /// Shift-invert mode for eigenvalues near a target
    ShiftInvert,
    /// Automatic selection based on problem characteristics
    Auto,
}

impl Default for EigenSolverStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

/// Unified eigenvalue solver interface
///
/// This function provides a unified interface to various eigenvalue solvers,
/// automatically selecting the best algorithm based on the problem characteristics.
pub fn solve_eigenvalues<T>(
    matrix: &crate::sym_csr::SymCsrMatrix<T>,
    config: &EigenSolverConfig,
    strategy: EigenSolverStrategy,
) -> crate::error::SparseResult<EigenResult<T>>
where
    T: num_traits::Float
        + std::fmt::Debug
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + std::iter::Sum
        + scirs2_core::simd_ops::SimdUnifiedOps
        + Send
        + Sync
        + 'static,
{
    let (n, _) = matrix.shape();

    let selected_strategy = match strategy {
        EigenSolverStrategy::Auto => {
            // Automatic selection based on problem characteristics
            if config.num_eigenvalues == 1 && config.which == "LA" {
                EigenSolverStrategy::PowerIteration
            } else if n < 1000 {
                EigenSolverStrategy::Lanczos
            } else {
                EigenSolverStrategy::Symmetric
            }
        }
        other => other,
    };

    match selected_strategy {
        EigenSolverStrategy::PowerIteration => {
            let opts = config.to_power_iteration_options();
            let power_result = power_iteration::power_iteration(matrix, &opts, None)?;
            // Convert power iteration result to common EigenResult format
            Ok(EigenResult {
                eigenvalues: power_result.eigenvalues,
                eigenvectors: power_result.eigenvectors,
                iterations: power_result.iterations,
                residuals: power_result.residuals,
                converged: power_result.converged,
            })
        }
        EigenSolverStrategy::Lanczos => {
            let opts = config.to_lanczos_options();
            let result = lanczos::lanczos(matrix, &opts, None)?;
            Ok(result)
        }
        EigenSolverStrategy::Symmetric => {
            let opts = config.to_lanczos_options();
            let result = symmetric::eigsh(
                matrix,
                Some(config.num_eigenvalues),
                Some(&config.which),
                Some(opts),
            )?;
            Ok(result)
        }
        EigenSolverStrategy::ShiftInvert => {
            // For shift-invert, we need a target value - use 0.0 as default
            let opts = config.to_lanczos_options();
            let result = symmetric::eigsh_shift_invert(
                matrix,
                T::zero(),
                Some(config.num_eigenvalues),
                Some(&config.which),
                Some(opts),
            )?;
            Ok(result)
        }
        EigenSolverStrategy::Auto => {
            // This case is handled above, but we include it for completeness
            unreachable!("Auto strategy should have been resolved")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sym_csr::SymCsrMatrix;

    #[test]
    fn test_eigen_solver_config() {
        let config = EigenSolverConfig::new()
            .with_max_iter(500)
            .with_tolerance(1e-10)
            .with_num_eigenvalues(3)
            .with_which("SA");

        assert_eq!(config.max_iter, 500);
        assert_eq!(config.tolerance, 1e-10);
        assert_eq!(config.num_eigenvalues, 3);
        assert_eq!(config.which, "SA");
    }

    #[test]
    fn test_unified_solver_auto() {
        // Create a simple 2x2 symmetric matrix
        // Matrix: [[2, 1], [1, 2]] stored as lower: [[2], [1, 2]]
        let data = vec![2.0, 1.0, 2.0];
        let indptr = vec![0, 1, 3];
        let indices = vec![0, 0, 1];
        let matrix = SymCsrMatrix::new(data, indptr, indices, (2, 2)).unwrap();

        let config = EigenSolverConfig::new().with_num_eigenvalues(1);
        let result = solve_eigenvalues(&matrix, &config, EigenSolverStrategy::Auto).unwrap();

        assert!(result.converged);
        assert_eq!(result.eigenvalues.len(), 1);
    }

    #[test]
    fn test_unified_solver_power_iteration() {
        // Matrix: [[3, 1], [1, 2]] stored as lower: [[3], [1, 2]]
        let data = vec![3.0, 1.0, 2.0];
        let indptr = vec![0, 1, 3];
        let indices = vec![0, 0, 1];
        let matrix = SymCsrMatrix::new(data, indptr, indices, (2, 2)).unwrap();

        let config = EigenSolverConfig::new().with_num_eigenvalues(1);
        let result =
            solve_eigenvalues(&matrix, &config, EigenSolverStrategy::PowerIteration).unwrap();

        assert!(result.converged);
        assert_eq!(result.eigenvalues.len(), 1);
    }

    #[test]
    fn test_unified_solver_lanczos() {
        // Matrix: [[4, 2], [2, 3]] stored as lower: [[4], [2, 3]]
        let data = vec![4.0, 2.0, 3.0];
        let indptr = vec![0, 1, 3];
        let indices = vec![0, 0, 1];
        let matrix = SymCsrMatrix::new(data, indptr, indices, (2, 2)).unwrap();

        let config = EigenSolverConfig::new().with_num_eigenvalues(2);
        let result = solve_eigenvalues(&matrix, &config, EigenSolverStrategy::Lanczos).unwrap();

        assert!(result.converged);
        assert!(result.eigenvalues.len() >= 1);
    }
}
