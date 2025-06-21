//! SciRS2 Benchmarking Suite
//!
//! This crate provides comprehensive performance benchmarking for SciRS2,
//! including comparisons with SciPy, memory efficiency analysis, and
//! numerical stability testing.
//!
//! ## Usage
//!
//! Run all benchmarks:
//! ```bash
//! ./run_benchmarks.sh
//! ```
//!
//! Run specific benchmark categories:
//! ```bash
//! cargo bench --bench linalg_benchmarks
//! cargo bench --bench memory_efficiency
//! cargo bench --bench numerical_stability
//! cargo bench --bench scipy_comparison
//! ```
//!
//! ## Benchmark Categories
//!
//! - **Linear Algebra Performance**: Core mathematical operations
//! - **SciPy Comparison**: Direct performance comparison with SciPy
//! - **Memory Efficiency**: Memory usage and optimization analysis  
//! - **Numerical Stability**: Accuracy and robustness testing
//!
//! See the README.md for detailed documentation.

/// Common utilities for benchmarking
pub mod common {
    use ndarray::{Array1, Array2};
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Standard seed for reproducible benchmarks
    pub const BENCHMARK_SEED: u64 = 42;

    /// Generate a random matrix with controlled properties
    pub fn generate_random_matrix(n: usize, seed: u64) -> Array2<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array2::random_using((n, n), Uniform::new(-1.0, 1.0), &mut rng)
    }

    /// Generate a random vector with controlled properties
    pub fn generate_random_vector(n: usize, seed: u64) -> Array1<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array1::random_using(n, Uniform::new(-1.0, 1.0), &mut rng)
    }

    /// Generate a symmetric positive definite matrix
    pub fn generate_spd_matrix(n: usize, seed: u64) -> Array2<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let a = Array2::random_using((n, n), Uniform::new(-1.0, 1.0), &mut rng);

        // A^T * A is always positive definite
        let at = a.t();
        at.dot(&a) + Array2::<f64>::eye(n) * 0.1 // Add small diagonal term for numerical stability
    }

    /// Calculate relative error between two values
    pub fn relative_error(computed: f64, exact: f64) -> f64 {
        if exact.abs() < f64::EPSILON {
            computed.abs()
        } else {
            (computed - exact).abs() / exact.abs()
        }
    }

    /// Check if a computation result is numerically acceptable
    pub fn is_numerically_acceptable(relative_error: f64, tolerance: f64) -> bool {
        relative_error < tolerance && relative_error.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::common::*;

    #[test]
    fn test_random_matrix_generation() {
        let matrix = generate_random_matrix(10, BENCHMARK_SEED);
        assert_eq!(matrix.shape(), &[10, 10]);

        // Test reproducibility
        let matrix2 = generate_random_matrix(10, BENCHMARK_SEED);
        assert_eq!(matrix, matrix2);
    }

    #[test]
    fn test_spd_matrix_generation() {
        let matrix = generate_spd_matrix(5, BENCHMARK_SEED);
        assert_eq!(matrix.shape(), &[5, 5]);

        // Check symmetry
        let diff = &matrix - &matrix.t();
        let max_asymmetry = diff.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(max_asymmetry < 1e-14, "Matrix should be symmetric");
    }

    #[test]
    fn test_relative_error_calculation() {
        assert_eq!(relative_error(1.0, 1.0), 0.0);

        // Use approximate equality for floating point comparisons
        let error1 = relative_error(1.1, 1.0);
        assert!(
            (error1 - 0.1).abs() < 1e-14,
            "Expected ~0.1, got {}",
            error1
        );

        let error2 = relative_error(0.9, 1.0);
        assert!(
            (error2 - 0.1).abs() < 1e-14,
            "Expected ~0.1, got {}",
            error2
        );

        // Test with very small exact values
        let error = relative_error(1e-16, 0.0);
        assert!(error.is_finite());
    }

    #[test]
    fn test_numerical_acceptability() {
        assert!(is_numerically_acceptable(1e-12, 1e-10));
        assert!(!is_numerically_acceptable(1e-8, 1e-10));
        assert!(!is_numerically_acceptable(f64::INFINITY, 1e-10));
        assert!(!is_numerically_acceptable(f64::NAN, 1e-10));
    }
}
