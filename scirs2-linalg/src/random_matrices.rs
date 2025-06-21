//! Random matrix generation utilities
//!
//! This module provides functions for generating various types of random matrices
//! commonly used in scientific computing, testing, and simulations.

use ndarray::{Array1, Array2};
use num_complex::Complex;
use num_traits::{Float, NumAssign};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::fmt::Debug;
use std::iter::Sum;

use crate::decomposition::qr;
use crate::error::{LinalgError, LinalgResult};

/// Standard distributions for matrix elements
#[derive(Debug, Clone, Copy)]
pub enum Distribution1D {
    /// Uniform distribution on [a, b]
    Uniform { a: f64, b: f64 },
    /// Normal distribution with given mean and standard deviation
    Normal { mean: f64, std_dev: f64 },
    /// Standard normal distribution (mean=0, std=1)
    StandardNormal,
}

/// Types of random matrices
#[derive(Debug, Clone, Copy)]
pub enum MatrixType {
    /// General random matrix with given distribution
    General(Distribution1D),
    /// Symmetric matrix
    Symmetric(Distribution1D),
    /// Positive definite matrix
    PositiveDefinite {
        eigenvalue_min: f64,
        eigenvalue_max: f64,
    },
    /// Orthogonal matrix (O^T * O = I)
    Orthogonal,
    /// Correlation matrix (symmetric positive semi-definite with 1s on diagonal)
    Correlation,
    /// Sparse matrix with given density
    Sparse {
        density: f64,
        distribution: Distribution1D,
    },
    /// Diagonal matrix
    Diagonal(Distribution1D),
    /// Triangular matrix (upper or lower)
    Triangular {
        upper: bool,
        distribution: Distribution1D,
    },
}

/// Generate a random matrix of the specified type
pub fn random_matrix<F, R>(
    rows: usize,
    cols: usize,
    matrix_type: MatrixType,
    rng: &mut R,
) -> LinalgResult<Array2<F>>
where
    F: Float + Debug + NumAssign + Sum + 'static,
    R: Rng + ?Sized,
{
    match matrix_type {
        MatrixType::General(dist) => random_general(rows, cols, dist, rng),
        MatrixType::Symmetric(dist) => random_symmetric(rows, dist, rng),
        MatrixType::PositiveDefinite {
            eigenvalue_min,
            eigenvalue_max,
        } => random_positive_definite(rows, eigenvalue_min, eigenvalue_max, rng),
        MatrixType::Orthogonal => random_orthogonal(rows, rng),
        MatrixType::Correlation => random_correlation(rows, rng),
        MatrixType::Sparse {
            density,
            distribution,
        } => random_sparse(rows, cols, density, distribution, rng),
        MatrixType::Diagonal(dist) => random_diagonal(rows, dist, rng),
        MatrixType::Triangular {
            upper,
            distribution,
        } => random_triangular(rows, cols, upper, distribution, rng),
    }
}

/// Generate a general random matrix
fn random_general<F, R>(
    rows: usize,
    cols: usize,
    distribution: Distribution1D,
    rng: &mut R,
) -> LinalgResult<Array2<F>>
where
    F: Float,
    R: Rng + ?Sized,
{
    let mut matrix = Array2::zeros((rows, cols));

    match distribution {
        Distribution1D::Uniform { a, b } => {
            let uniform = Uniform::new(a, b).map_err(|_| {
                LinalgError::ValueError("Invalid uniform distribution range".to_string())
            })?;
            for elem in matrix.iter_mut() {
                *elem = F::from(uniform.sample(rng)).unwrap();
            }
        }
        Distribution1D::Normal { mean, std_dev } => {
            let normal = Normal::new(mean, std_dev).map_err(|_| {
                LinalgError::ValueError("Invalid normal distribution parameters".to_string())
            })?;
            for elem in matrix.iter_mut() {
                *elem = F::from(normal.sample(rng)).unwrap();
            }
        }
        Distribution1D::StandardNormal => {
            let normal = Normal::new(0.0, 1.0).unwrap();
            for elem in matrix.iter_mut() {
                *elem = F::from(normal.sample(rng)).unwrap();
            }
        }
    }

    Ok(matrix)
}

/// Generate a symmetric random matrix
fn random_symmetric<F, R>(
    size: usize,
    distribution: Distribution1D,
    rng: &mut R,
) -> LinalgResult<Array2<F>>
where
    F: Float,
    R: Rng + ?Sized,
{
    let mut matrix = random_general(size, size, distribution, rng)?;

    // Make it symmetric: A = (A + A^T) / 2
    for i in 0..size {
        for j in i + 1..size {
            let avg = (matrix[[i, j]] + matrix[[j, i]]) / F::from(2.0).unwrap();
            matrix[[i, j]] = avg;
            matrix[[j, i]] = avg;
        }
    }

    Ok(matrix)
}

/// Generate a positive definite matrix
fn random_positive_definite<F, R>(
    size: usize,
    eigenvalue_min: f64,
    eigenvalue_max: f64,
    rng: &mut R,
) -> LinalgResult<Array2<F>>
where
    F: Float + Debug + NumAssign + Sum + 'static,
    R: Rng + ?Sized,
{
    if eigenvalue_min <= 0.0 {
        return Err(LinalgError::ValueError(
            "Minimum eigenvalue must be positive".to_string(),
        ));
    }

    // Generate random orthogonal matrix using QR decomposition
    let a = random_general(size, size, Distribution1D::StandardNormal, rng)?;
    let qr_result = qr(&a.view(), None)?;
    let (q, _) = qr_result;

    // Generate random eigenvalues in the specified range
    let uniform = Uniform::new(eigenvalue_min, eigenvalue_max)
        .map_err(|_| LinalgError::ValueError("Invalid eigenvalue range".to_string()))?;
    let mut eigenvalues = Array1::zeros(size);
    for i in 0..size {
        eigenvalues[i] = F::from(uniform.sample(rng)).unwrap();
    }

    // Create diagonal matrix with eigenvalues
    let mut d = Array2::zeros((size, size));
    for i in 0..size {
        d[[i, i]] = eigenvalues[i];
    }

    // A = Q * D * Q^T
    let qd = q.dot(&d);
    let qt = q.t();
    let result = qd.dot(&qt);

    Ok(result)
}

/// Generate a random orthogonal matrix using QR decomposition
fn random_orthogonal<F, R>(size: usize, rng: &mut R) -> LinalgResult<Array2<F>>
where
    F: Float + Debug + NumAssign + Sum + 'static,
    R: Rng + ?Sized,
{
    // Generate random matrix with normal distribution
    let a = random_general(size, size, Distribution1D::StandardNormal, rng)?;

    // Perform QR decomposition
    let qr_result = qr(&a.view(), None)?;

    // Q is orthogonal
    let (q, _) = qr_result;
    Ok(q)
}

/// Generate a random correlation matrix
fn random_correlation<F, R>(size: usize, rng: &mut R) -> LinalgResult<Array2<F>>
where
    F: Float + Debug + NumAssign + Sum + 'static,
    R: Rng + ?Sized,
{
    // Start with a positive definite matrix
    let mut matrix = random_positive_definite(size, 0.1, 10.0, rng)?;

    // Normalize to get correlations
    let diag = matrix.diag().to_owned();
    for i in 0..size {
        for j in 0..size {
            let prod: F = diag[i] * diag[j];
            let denom = prod.sqrt();
            matrix[[i, j]] /= denom;
        }
    }

    // Ensure diagonal is exactly 1
    for i in 0..size {
        matrix[[i, i]] = F::one();
    }

    Ok(matrix)
}

/// Generate a sparse random matrix
fn random_sparse<F, R>(
    rows: usize,
    cols: usize,
    density: f64,
    distribution: Distribution1D,
    rng: &mut R,
) -> LinalgResult<Array2<F>>
where
    F: Float,
    R: Rng + ?Sized,
{
    if !(0.0..=1.0).contains(&density) {
        return Err(LinalgError::ValueError(
            "Density must be between 0 and 1".to_string(),
        ));
    }

    let mut matrix = Array2::zeros((rows, cols));
    let uniform = Uniform::new(0.0, 1.0)
        .map_err(|_| LinalgError::ValueError("Invalid uniform distribution range".to_string()))?;

    for i in 0..rows {
        for j in 0..cols {
            if F::from(uniform.sample(rng)).unwrap() < F::from(density).unwrap() {
                matrix[[i, j]] = match distribution {
                    Distribution1D::Uniform { a, b } => {
                        F::from(Uniform::new(a, b).unwrap().sample(rng)).unwrap()
                    }
                    Distribution1D::Normal { mean, std_dev } => {
                        F::from(Normal::new(mean, std_dev).unwrap().sample(rng)).unwrap()
                    }
                    Distribution1D::StandardNormal => {
                        F::from(Normal::new(0.0, 1.0).unwrap().sample(rng)).unwrap()
                    }
                };
            }
        }
    }

    Ok(matrix)
}

/// Generate a random diagonal matrix
fn random_diagonal<F, R>(
    size: usize,
    distribution: Distribution1D,
    rng: &mut R,
) -> LinalgResult<Array2<F>>
where
    F: Float,
    R: Rng + ?Sized,
{
    let mut matrix = Array2::zeros((size, size));

    match distribution {
        Distribution1D::Uniform { a, b } => {
            let uniform = Uniform::new(a, b).map_err(|_| {
                LinalgError::ValueError("Invalid uniform distribution range".to_string())
            })?;
            for i in 0..size {
                matrix[[i, i]] = F::from(uniform.sample(rng)).unwrap();
            }
        }
        Distribution1D::Normal { mean, std_dev } => {
            let normal = Normal::new(mean, std_dev).map_err(|_| {
                LinalgError::ValueError("Invalid normal distribution parameters".to_string())
            })?;
            for i in 0..size {
                matrix[[i, i]] = F::from(normal.sample(rng)).unwrap();
            }
        }
        Distribution1D::StandardNormal => {
            let normal = Normal::new(0.0, 1.0).unwrap();
            for i in 0..size {
                matrix[[i, i]] = F::from(normal.sample(rng)).unwrap();
            }
        }
    }

    Ok(matrix)
}

/// Generate a random triangular matrix
fn random_triangular<F, R>(
    rows: usize,
    cols: usize,
    upper: bool,
    distribution: Distribution1D,
    rng: &mut R,
) -> LinalgResult<Array2<F>>
where
    F: Float,
    R: Rng + ?Sized,
{
    let mut matrix = Array2::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let should_fill = if upper { j >= i } else { j <= i };
            if should_fill {
                matrix[[i, j]] = match distribution {
                    Distribution1D::Uniform { a, b } => {
                        F::from(Uniform::new(a, b).unwrap().sample(rng)).unwrap()
                    }
                    Distribution1D::Normal { mean, std_dev } => {
                        F::from(Normal::new(mean, std_dev).unwrap().sample(rng)).unwrap()
                    }
                    Distribution1D::StandardNormal => {
                        F::from(Normal::new(0.0, 1.0).unwrap().sample(rng)).unwrap()
                    }
                };
            }
        }
    }

    Ok(matrix)
}

/// Generate a random complex matrix
pub fn random_complex_matrix<F, R>(
    rows: usize,
    cols: usize,
    real_dist: Distribution1D,
    imag_dist: Distribution1D,
    rng: &mut R,
) -> LinalgResult<Array2<Complex<F>>>
where
    F: Float,
    R: Rng + ?Sized,
{
    let real_part = random_general(rows, cols, real_dist, rng)?;
    let imag_part = random_general(rows, cols, imag_dist, rng)?;

    let mut matrix = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            matrix[[i, j]] = Complex::new(real_part[[i, j]], imag_part[[i, j]]);
        }
    }

    Ok(matrix)
}

/// Generate a random Hermitian matrix
pub fn random_hermitian<F, R>(
    size: usize,
    real_dist: Distribution1D,
    imag_dist: Distribution1D,
    rng: &mut R,
) -> LinalgResult<Array2<Complex<F>>>
where
    F: Float,
    R: Rng + ?Sized,
{
    let mut matrix = random_complex_matrix(size, size, real_dist, imag_dist, rng)?;

    // Make it Hermitian: A = (A + A^H) / 2
    for i in 0..size {
        for j in i + 1..size {
            let avg =
                (matrix[[i, j]] + matrix[[j, i]].conj()) / Complex::from(F::from(2.0).unwrap());
            matrix[[i, j]] = avg;
            matrix[[j, i]] = avg.conj();
        }
        // Diagonal elements must be real
        matrix[[i, i]] = Complex::new(matrix[[i, i]].re, F::zero());
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_symmetric_matrix() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let matrix = random_matrix::<f64, _>(
            5,
            5,
            MatrixType::Symmetric(Distribution1D::StandardNormal),
            &mut rng,
        )
        .unwrap();

        // Check symmetry
        for i in 0..5 {
            for j in 0..5 {
                assert_relative_eq!(matrix[[i, j]], matrix[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_orthogonal_matrix() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let q = random_matrix::<f64, _>(4, 4, MatrixType::Orthogonal, &mut rng).unwrap();

        // Check Q^T * Q = I
        let qt = q.t();
        let qtq = qt.dot(&q);

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(qtq[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_positive_definite() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let matrix = random_matrix::<f64, _>(
            3,
            3,
            MatrixType::PositiveDefinite {
                eigenvalue_min: 0.1,
                eigenvalue_max: 5.0,
            },
            &mut rng,
        )
        .unwrap();

        // A positive definite matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(matrix[[i, j]], matrix[[j, i]], epsilon = 1e-10);
            }
        }

        // Should have positive eigenvalues (we won't compute them here, just test Cholesky)
        use crate::cholesky;
        let result = cholesky(&matrix.view(), None);
        assert!(result.is_ok(), "Matrix should be positive definite");
    }

    #[test]
    fn test_correlation_matrix() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let matrix = random_matrix::<f64, _>(4, 4, MatrixType::Correlation, &mut rng).unwrap();

        // Check diagonal elements are 1
        for i in 0..4 {
            assert_relative_eq!(matrix[[i, i]], 1.0, epsilon = 1e-10);
        }

        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(matrix[[i, j]], matrix[[j, i]], epsilon = 1e-10);
            }
        }

        // Check elements are in [-1, 1]
        for elem in matrix.iter() {
            assert!(*elem >= -1.0 && *elem <= 1.0);
        }
    }
}
