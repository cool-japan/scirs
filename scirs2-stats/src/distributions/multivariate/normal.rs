//! Multivariate Normal distribution functions
//!
//! This module provides functionality for the Multivariate Normal (Gaussian) distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use crate::traits::{Distribution as DistributionTrait, MultivariateDistribution};
use scirs2_core::ndarray::{
    s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1, Ix2,
};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution as RandDistribution, Normal as RandNormal};
use statrs::statistics::Statistics;
use std::fmt::Debug;

/// Multivariate Normal distribution structure
#[derive(Debug, Clone)]
pub struct MultivariateNormal {
    /// Mean vector
    pub mean: Array1<f64>,
    /// Covariance matrix
    pub cov: Array2<f64>,
    /// Dimensionality of the distribution
    pub dim: usize,
    /// Cholesky decomposition of the covariance matrix (lower triangular)
    cholesky_l: Array2<f64>,
    /// Determinant of the covariance matrix
    cov_det: f64,
    /// Inverse of the covariance matrix
    cov_inv: Array2<f64>,
}

impl MultivariateNormal {
    /// Create a new multivariate normal distribution with given mean vector and covariance matrix
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean vector (k-dimensional)
    /// * `cov` - Covariance matrix (k x k, symmetric positive-definite)
    ///
    /// # Returns
    ///
    /// * A new MultivariateNormal distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::{array, Array1, Array2};
    /// use scirs2_stats::distributions::multivariate::normal::MultivariateNormal;
    ///
    /// // Create a 2D multivariate normal distribution
    /// let mean = array![0.0, 0.0];
    /// let cov = array![[1.0, 0.5], [0.5, 2.0]];
    /// let mvn = MultivariateNormal::new(mean, cov).unwrap();
    /// ```
    pub fn new<D1, D2>(mean: ArrayBase<D1, Ix1>, cov: ArrayBase<D2, Ix2>) -> StatsResult<Self>
    where
        D1: Data<Elem = f64>,
        D2: Data<Elem = f64>,
    {
        // Validate dimensions
        let dim = mean.len();
        if cov.shape()[0] != dim || cov.shape()[1] != dim {
            return Err(StatsError::DimensionMismatch(format!(
                "Covariance matrix shape ({:?}) must match _mean vector length ({})",
                cov.shape(),
                dim
            )));
        }

        // Create owned copies of inputs
        let _mean = mean.to_owned();
        let cov = cov.to_owned();

        // Compute Cholesky decomposition (lower triangular L where Σ = L·L^T)
        let cholesky_l = compute_cholesky(&cov).map_err(|_| {
            StatsError::DomainError("Covariance matrix must be positive definite".to_string())
        })?;

        // For positive definite matrix, det(Σ) = det(L)^2 = prod(diag(L))^2
        let cov_det = {
            let mut det = 1.0;
            for i in 0..dim {
                det *= cholesky_l[[i, i]];
            }
            det * det // Square it since det(Σ) = det(L)^2
        };

        // Compute inverse using Cholesky decomposition
        let cov_inv = compute_inverse_from_cholesky(&cholesky_l).map_err(|_| {
            StatsError::ComputationError("Failed to compute matrix inverse".to_string())
        })?;

        Ok(MultivariateNormal {
            mean: _mean,
            cov,
            dim,
            cholesky_l,
            cov_det,
            cov_inv,
        })
    }

    /// Calculate the probability density function (PDF) at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the PDF
    ///
    /// # Returns
    ///
    /// * The value of the PDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::normal::MultivariateNormal;
    ///
    /// let mean = array![0.0, 0.0];
    /// let cov = array![[1.0, 0.0], [0.0, 1.0]];
    /// let mvn = MultivariateNormal::new(mean, cov).unwrap();
    ///
    /// // PDF at origin for a standard 2D normal should be 1/(2π)
    /// let pdf_at_origin = mvn.pdf(&array![0.0, 0.0]);
    /// assert!((pdf_at_origin - 0.15915494).abs() < 1e-7); // 1/(2π) ≈ 0.15915494
    /// ```
    pub fn pdf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        if x.len() != self.dim {
            return 0.0; // Return zero for invalid dimensions
        }

        let pi = std::f64::consts::PI;
        let two = 2.0;
        let constant_factor = 1.0 / ((two * pi).powf(self.dim as f64 / two) * self.cov_det.sqrt());

        // Calculate Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
        let diff = x - &self.mean;
        let mahalanobis_squared = self.mahalanobis_distance_squared(&diff.view());

        // PDF = (2π)^(-k/2) |Σ|^(-1/2) exp(-1/2 * mahalanobis^2)
        constant_factor * (-mahalanobis_squared / two).exp()
    }

    /// Calculate the Mahalanobis distance squared: (x - μ)^T Σ^-1 (x - μ)
    fn mahalanobis_distance_squared(&self, diff: &ArrayView1<f64>) -> f64 {
        // Compute (x - μ)^T Σ^-1 (x - μ)
        diff.dot(&self.cov_inv.dot(diff))
    }

    /// Generate random samples from the distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// * Matrix where each row is a random sample
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::normal::MultivariateNormal;
    ///
    /// let mean = array![0.0, 0.0];
    /// let cov = array![[1.0, 0.5], [0.5, 2.0]];
    /// let mvn = MultivariateNormal::new(mean, cov).unwrap();
    ///
    /// let samples = mvn.rvs(100).unwrap();
    /// assert_eq!(samples.shape(), &[100, 2]);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Array2<f64>> {
        let mut rng = thread_rng();
        let normal = RandNormal::new(0.0, 1.0).unwrap();

        // Create a matrix of standard normal samples
        let mut std_normal_samples = Array2::<f64>::zeros((size, self.dim));
        for i in 0..size {
            for j in 0..self.dim {
                let sample = normal.sample(&mut rng);
                std_normal_samples[[i, j]] = sample;
            }
        }

        // Transform to the desired multivariate normal distribution
        // X = μ + L*Z where L is the Cholesky factor of Σ and Z is standard normal
        let mut samples = Array2::<f64>::zeros((size, self.dim));
        for i in 0..size {
            // Get the i-th row of std_normal_samples
            let z = std_normal_samples.slice(s![i, ..]);

            // Compute L*z (matrix-vector product)
            let mut transformed = Array1::<f64>::zeros(self.dim);
            for j in 0..self.dim {
                for k in 0..=j {
                    transformed[j] += self.cholesky_l[[j, k]] * z[k];
                }
            }

            // Add the mean and store in samples
            for j in 0..self.dim {
                samples[[i, j]] = self.mean[j] + transformed[j];
            }
        }

        Ok(samples)
    }

    /// Generate a single random sample from the distribution
    ///
    /// # Returns
    ///
    /// * Vector representing a single sample
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::normal::MultivariateNormal;
    ///
    /// let mean = array![0.0, 0.0];
    /// let cov = array![[1.0, 0.5], [0.5, 2.0]];
    /// let mvn = MultivariateNormal::new(mean, cov).unwrap();
    ///
    /// let sample = mvn.rvs_single().unwrap();
    /// assert_eq!(sample.len(), 2);
    /// ```
    pub fn rvs_single(&self) -> StatsResult<Array1<f64>> {
        let samples = self.rvs(1)?;
        Ok(samples.index_axis(Axis(0), 0).to_owned())
    }

    /// Calculate the log probability density function (log PDF) at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the log PDF
    ///
    /// # Returns
    ///
    /// * The value of the log PDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::normal::MultivariateNormal;
    ///
    /// let mean = array![0.0, 0.0];
    /// let cov = array![[1.0, 0.0], [0.0, 1.0]];
    /// let mvn = MultivariateNormal::new(mean, cov).unwrap();
    ///
    /// let log_pdf = mvn.logpdf(&array![0.0, 0.0]);
    /// assert!((log_pdf - (-1.8378770664093453)).abs() < 1e-7); // ln(1/(2π)) ≈ -1.837877
    /// ```
    pub fn logpdf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        if x.len() != self.dim {
            return f64::NEG_INFINITY; // Return -∞ for invalid dimensions
        }

        let pi = std::f64::consts::PI;
        let two = 2.0;

        // log of normalization constant: -k/2 * ln(2π) - 1/2 * ln(|Σ|)
        let log_const = -(self.dim as f64) / two * (two * pi).ln() - self.cov_det.ln() / two;

        // Calculate Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
        let diff = x - &self.mean;
        let mahalanobis_squared = self.mahalanobis_distance_squared(&diff.view());

        // logpdf = -k/2 * ln(2π) - 1/2 * ln(|Σ|) - 1/2 * mahalanobis^2
        log_const - mahalanobis_squared / two
    }

    /// Get the dimension of the distribution
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the covariance matrix of the distribution
    pub fn cov(&self) -> ArrayView2<f64> {
        self.cov.view()
    }

    /// Get the mean vector of the distribution
    pub fn mean(&self) -> ArrayView1<f64> {
        self.mean.view()
    }
}

/// Compute the Cholesky decomposition L of a positive definite matrix A such that A = L·L^T
#[allow(dead_code)]
pub fn compute_cholesky(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.shape()[0];
    let mut l = Array2::<f64>::zeros((n, n));

    // Cholesky-Banachiewicz algorithm
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;

            if j == i {
                // Diagonal element
                for k in 0..j {
                    sum += l[[j, k]] * l[[j, k]];
                }
                let diag_value = a[[j, j]] - sum;
                if diag_value <= 0.0 {
                    return Err("Matrix is not positive definite".to_string());
                }
                l[[j, j]] = diag_value.sqrt();
            } else {
                // Off-diagonal element
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Compute the inverse of a symmetric positive definite matrix A from its Cholesky decomposition L
#[allow(dead_code)]
pub fn compute_inverse_from_cholesky(l: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = l.shape()[0];
    let mut inv = Array2::<f64>::zeros((n, n));

    // First, compute the inverse of L (lower triangular)
    let mut l_inv = Array2::<f64>::zeros((n, n));

    // Diagonal elements of L⁻¹
    for i in 0..n {
        l_inv[[i, i]] = 1.0 / l[[i, i]];
    }

    // Off-diagonal elements of L⁻¹
    for i in 1..n {
        for j in 0..i {
            let mut sum = 0.0;
            for k in j..i {
                sum += l[[i, k]] * l_inv[[k, j]];
            }
            l_inv[[i, j]] = -sum / l[[i, i]];
        }
    }

    // Compute A⁻¹ = (L·L^T)⁻¹ = (L^T)⁻¹ · L⁻¹
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            // We only use elements k where k ≥ max(i,j) because L⁻¹ is lower triangular
            let max_idx = i.max(j);
            for k in max_idx..n {
                sum += l_inv[[k, i]] * l_inv[[k, j]];
            }
            inv[[i, j]] = sum;
        }
    }

    Ok(inv)
}

/// Create a multivariate normal distribution with the given parameters.
///
/// This is a convenience function to create a multivariate normal distribution with
/// the given mean vector and covariance matrix.
///
/// # Arguments
///
/// * `mean` - Mean vector (k-dimensional)
/// * `cov` - Covariance matrix (k x k, symmetric positive-definite)
///
/// # Returns
///
/// * A multivariate normal distribution object
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distributions::multivariate;
///
/// let mean = array![0.0, 0.0];
/// let cov = array![[1.0, 0.5], [0.5, 2.0]];
/// let mvn = multivariate::multivariate_normal(mean, cov).unwrap();
/// let pdf_at_origin = mvn.pdf(&array![0.0, 0.0]);
/// ```
#[allow(dead_code)]
pub fn multivariate_normal<D1, D2>(
    mean: ArrayBase<D1, Ix1>,
    cov: ArrayBase<D2, Ix2>,
) -> StatsResult<MultivariateNormal>
where
    D1: Data<Elem = f64>,
    D2: Data<Elem = f64>,
{
    MultivariateNormal::new(mean, cov)
}

// Implement the Distribution trait for MultivariateNormal
impl DistributionTrait<f64> for MultivariateNormal {
    fn mean(&self) -> f64 {
        // Return mean of first dimension as representative value
        if self.dim > 0 {
            self.mean[0]
        } else {
            0.0
        }
    }

    fn var(&self) -> f64 {
        // Return variance of first dimension as representative value
        if self.dim > 0 {
            self.cov[[0, 0]]
        } else {
            0.0
        }
    }

    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    fn rvs(&self, size: usize) -> StatsResult<Array1<f64>> {
        // For the base Distribution trait, return the first dimension of samples
        let samples_matrix = self.rvs(size)?;
        Ok(samples_matrix.column(0).to_owned())
    }

    fn entropy(&self) -> f64 {
        // Entropy of multivariate normal: k/2 + k/2*ln(2π) + 1/2*ln(|Σ|)
        let k = self.dim as f64;
        let pi = std::f64::consts::PI;

        k / 2.0 + k / 2.0 * (2.0 * pi).ln() + 0.5 * self.cov_det.ln()
    }
}

// Implement the MultivariateDistribution trait for MultivariateNormal
impl MultivariateDistribution<f64> for MultivariateNormal {
    fn pdf(&self, x: &Array1<f64>) -> f64 {
        self.pdf(x)
    }

    fn rvs(&self, size: usize) -> StatsResult<scirs2_core::ndarray::Array2<f64>> {
        self.rvs(size)
    }

    fn mean(&self) -> Array1<f64> {
        self.mean.clone()
    }

    fn cov(&self) -> scirs2_core::ndarray::Array2<f64> {
        self.cov.clone()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn logpdf(&self, x: &Array1<f64>) -> f64 {
        self.logpdf(x)
    }

    fn rvs_single(&self) -> StatsResult<Vec<f64>> {
        let sample = self.rvs(1)?;
        Ok(sample.row(0).to_vec())
    }
}

/// Implementation of SampleableDistribution for MultivariateNormal
impl SampleableDistribution<Array1<f64>> for MultivariateNormal {
    fn rvs(&self, size: usize) -> StatsResult<Vec<Array1<f64>>> {
        let samples_matrix = self.rvs(size)?;
        let mut result = Vec::with_capacity(size);

        for i in 0..size {
            let row = samples_matrix.slice(s![i, ..]).to_owned();
            result.push(row);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{array, Axis};

    #[test]
    fn test_mvn_creation() {
        // 2D standard multivariate normal
        let mean = array![0.0, 0.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        let mvn = MultivariateNormal::new(mean.clone(), cov.clone()).unwrap();

        assert_eq!(mvn.dim, 2);
        assert_eq!(mvn.mean, mean);
        assert_eq!(mvn.cov, cov);

        // Custom 3D multivariate normal
        let mean3 = array![1.0, 2.0, 3.0];
        let cov3 = array![[1.0, 0.5, 0.3], [0.5, 2.0, 0.2], [0.3, 0.2, 1.5]];
        let mvn3 = MultivariateNormal::new(mean3.clone(), cov3.clone()).unwrap();

        assert_eq!(mvn3.dim, 3);
        assert_eq!(mvn3.mean, mean3);
        assert_eq!(mvn3.cov, cov3);
    }

    #[test]
    fn test_mvn_creation_errors() {
        // Dimension mismatch
        let mean = array![0.0, 0.0, 0.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        assert!(MultivariateNormal::new(mean, cov).is_err());

        // Non-positive definite covariance
        let mean = array![0.0, 0.0];
        let cov = array![[1.0, 2.0], [2.0, 1.0]]; // Not positive definite
        assert!(MultivariateNormal::new(mean, cov).is_err());
    }

    #[test]
    fn test_mvn_pdf() {
        // 2D standard multivariate normal
        let mean = array![0.0, 0.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        let mvn = MultivariateNormal::new(mean, cov).unwrap();

        // PDF at origin should be 1/(2π) ≈ 0.15915494
        let pdf_at_origin = mvn.pdf(&array![0.0, 0.0]);
        assert_relative_eq!(pdf_at_origin, 0.15915494, epsilon = 1e-7);

        // PDF at [1, 1] should be 1/(2π*e) ≈ 0.05854983
        let pdf_at_one = mvn.pdf(&array![1.0, 1.0]);
        assert_relative_eq!(pdf_at_one, 0.05854983, epsilon = 1e-7);
    }

    #[test]
    fn test_mvn_logpdf() {
        // 2D standard multivariate normal
        let mean = array![0.0, 0.0];
        let cov = array![[1.0, 0.0], [0.0, 1.0]];
        let mvn = MultivariateNormal::new(mean, cov).unwrap();

        // logPDF at origin should be ln(1/(2π)) ≈ -1.837877
        let logpdf_at_origin = mvn.logpdf(&array![0.0, 0.0]);
        assert_relative_eq!(logpdf_at_origin, -1.837877, epsilon = 1e-6);

        // Check that exp(logPDF) = PDF
        let x = array![1.0, 1.0];
        let pdf = mvn.pdf(&x);
        let logpdf = mvn.logpdf(&x);
        assert_relative_eq!(logpdf.exp(), pdf, epsilon = 1e-7);
    }

    #[test]
    fn test_mvn_rvs() {
        // 2D multivariate normal with correlation
        let mean = array![1.0, 2.0];
        let cov = array![[1.0, 0.5], [0.5, 2.0]];
        let mvn = MultivariateNormal::new(mean, cov).unwrap();

        // Generate samples and check dimensions
        let n_samples_ = 500;
        let samples = mvn.rvs(n_samples_).unwrap();
        assert_eq!(samples.shape(), &[n_samples_, 2]);

        // Check statistics (rough check as it's random)
        let sample_mean = samples.mean_axis(Axis(0)).unwrap();
        assert_relative_eq!(sample_mean[0], 1.0, epsilon = 0.3);
        assert_relative_eq!(sample_mean[1], 2.0, epsilon = 0.3);

        // Calculate sample covariance
        let centered = samples.mapv(|x| x) - &sample_mean;
        let sample_cov = centered.t().dot(&centered) / (n_samples_ as f64 - 1.0);
        assert_relative_eq!(sample_cov[[0, 0]], 1.0, epsilon = 0.5);
        assert_relative_eq!(sample_cov[[1, 1]], 2.0, epsilon = 0.5);
        assert_relative_eq!(sample_cov[[0, 1]].abs(), 0.5, epsilon = 0.3);
    }

    #[test]
    fn test_mvn_rvs_single() {
        let mean = array![1.0, 2.0];
        let cov = array![[1.0, 0.5], [0.5, 2.0]];
        let mvn = MultivariateNormal::new(mean, cov).unwrap();

        let sample = mvn.rvs_single().unwrap();
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_cholesky() {
        // Test the Cholesky decomposition
        let a = array![[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]];

        let l = compute_cholesky(&a).unwrap();

        // Reconstruct A = L·L^T
        let mut a_reconstructed = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..=j.min(i) {
                    a_reconstructed[[i, j]] += l[[i, k]] * l[[j, k]];
                }
            }
        }

        // Check that the reconstruction is close to the original matrix
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(a[[i, j]], a_reconstructed[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_inverse() {
        // Test the matrix inverse calculation
        let a = array![[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]];

        // Compute Cholesky decomposition
        let l = compute_cholesky(&a).unwrap();

        // Compute inverse from Cholesky
        let a_inv = compute_inverse_from_cholesky(&l).unwrap();

        // Check that A·A⁻¹ ≈ I
        let identity = a.dot(&a_inv);

        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(identity[[i, j]], 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(identity[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }
}
