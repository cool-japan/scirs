//! Multivariate Lognormal distribution functions
//!
//! This module provides functionality for the Multivariate Lognormal distribution.

use crate::distributions::multivariate::normal::MultivariateNormal;
use crate::error::StatsResult;
use crate::sampling::SampleableDistribution;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use std::fmt::Debug;

/// Multivariate Lognormal distribution structure
///
/// The multivariate lognormal distribution arises when taking the exponential
/// of a multivariate normal random variable.
#[derive(Debug, Clone)]
pub struct MultivariateLognormal {
    /// Mean vector of the underlying multivariate normal distribution
    pub mu: Array1<f64>,
    /// Covariance matrix of the underlying multivariate normal distribution
    pub sigma: Array2<f64>,
    /// Dimensionality of the distribution
    pub dim: usize,
    /// Associated multivariate normal distribution
    mvn: MultivariateNormal,
}

impl MultivariateLognormal {
    /// Create a new multivariate lognormal distribution with given parameters.
    ///
    /// The parameters represent the mean vector and covariance matrix of the
    /// underlying multivariate normal distribution, not of the lognormal distribution itself.
    ///
    /// # Arguments
    ///
    /// * `mu` - Mean vector of the underlying multivariate normal (k-dimensional)
    /// * `sigma` - Covariance matrix of the underlying multivariate normal (k x k, symmetric positive-definite)
    ///
    /// # Returns
    ///
    /// * A new MultivariateLognormal distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::{array, Array1, Array2};
    /// use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
    ///
    /// // Create a 2D multivariate lognormal distribution
    /// let mu = array![0.0, 0.0];
    /// let sigma = array![[0.5, 0.2], [0.2, 0.5]];
    /// let mvln = MultivariateLognormal::new(mu, sigma).unwrap();
    /// ```
    pub fn new<D1, D2>(mu: ArrayBase<D1, Ix1>, sigma: ArrayBase<D2, Ix2>) -> StatsResult<Self>
    where
        D1: Data<Elem = f64>,
        D2: Data<Elem = f64>,
    {
        // Create the underlying MultivariateNormal distribution
        let mvn = MultivariateNormal::new(mu.to_owned(), sigma.to_owned())?;

        // Get the dimension of the distribution
        let dim = mvn.dim();

        Ok(MultivariateLognormal {
            mu: mu.to_owned(),
            sigma: sigma.to_owned(),
            dim,
            mvn,
        })
    }

    /// Calculate the probability density function (PDF) at a given point.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the PDF (must have all positive components)
    ///
    /// # Returns
    ///
    /// * The value of the PDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
    ///
    /// let mu = array![0.0, 0.0];
    /// let sigma = array![[0.5, 0.0], [0.0, 0.5]];
    /// let mvln = MultivariateLognormal::new(mu, sigma).unwrap();
    ///
    /// let x = array![1.0, 1.0];
    /// let pdf_value = mvln.pdf(&x);
    /// ```
    pub fn pdf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        if x.len() != self.dim {
            return 0.0; // Invalid dimensions
        }

        // Check if all components are positive
        for &xi in x.iter() {
            if xi <= 0.0 {
                return 0.0; // Log-normal is only defined for positive values
            }
        }

        // Convert x to log space (element-wise log)
        let log_x = x.mapv(|xi| xi.ln());

        // Calculate normal PDF in log space
        let normal_pdf = self.mvn.pdf(&log_x);

        // Apply Jacobian correction: multiply by 1/(x1*x2*...*xn)
        let jacobian_factor = x.iter().fold(1.0, |acc, &xi| acc * xi);

        normal_pdf / jacobian_factor
    }

    /// Calculate the log probability density function (log PDF) at a given point.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the log PDF (must have all positive components)
    ///
    /// # Returns
    ///
    /// * The value of the log PDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
    ///
    /// let mu = array![0.0, 0.0];
    /// let sigma = array![[0.5, 0.0], [0.0, 0.5]];
    /// let mvln = MultivariateLognormal::new(mu, sigma).unwrap();
    ///
    /// let x = array![1.0, 1.0];
    /// let logpdf_value = mvln.logpdf(&x);
    /// ```
    pub fn logpdf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        if x.len() != self.dim {
            return f64::NEG_INFINITY; // Invalid dimensions
        }

        // Check if all components are positive
        for &xi in x.iter() {
            if xi <= 0.0 {
                return f64::NEG_INFINITY; // Log-normal is only defined for positive values
            }
        }

        // Convert x to log space (element-wise log)
        let log_x = x.mapv(|xi| xi.ln());

        // Calculate normal log PDF in log space
        let normal_logpdf = self.mvn.logpdf(&log_x);

        // Apply Jacobian correction: subtract sum of log(xi)
        let sum_log_x = x.iter().fold(0.0, |acc, &xi| acc + xi.ln());

        normal_logpdf - sum_log_x
    }

    /// Generate random samples from the distribution.
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
    /// use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
    ///
    /// let mu = array![0.0, 0.0];
    /// let sigma = array![[0.5, 0.2], [0.2, 0.5]];
    /// let mvln = MultivariateLognormal::new(mu, sigma).unwrap();
    ///
    /// let samples = mvln.rvs(100).unwrap();
    /// assert_eq!(samples.shape(), &[100, 2]);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Array2<f64>> {
        // Generate samples from the underlying normal distribution
        let normal_samples = self.mvn.rvs(size)?;

        // Transform the samples by taking element-wise exponential
        let lognormal_samples = normal_samples.mapv(|x| x.exp());

        Ok(lognormal_samples)
    }

    /// Generate a single random sample from the distribution.
    ///
    /// # Returns
    ///
    /// * Vector representing a single sample
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
    ///
    /// let mu = array![0.0, 0.0];
    /// let sigma = array![[0.5, 0.2], [0.2, 0.5]];
    /// let mvln = MultivariateLognormal::new(mu, sigma).unwrap();
    ///
    /// let sample = mvln.rvs_single().unwrap();
    /// assert_eq!(sample.len(), 2);
    /// ```
    pub fn rvs_single(&self) -> StatsResult<Array1<f64>> {
        // Generate a sample from the underlying normal distribution
        let normal_sample = self.mvn.rvs_single()?;

        // Transform the sample by taking element-wise exponential
        let lognormal_sample = normal_sample.mapv(|x| x.exp());

        Ok(lognormal_sample)
    }

    /// Calculate the mean of the distribution.
    ///
    /// For a multivariate lognormal distribution with parameters μ and Σ,
    /// the mean of component i is exp(μ_i + Σ_ii/2).
    ///
    /// # Returns
    ///
    /// * Mean vector
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
    ///
    /// let mu = array![0.0, 0.0];
    /// let sigma = array![[0.5, 0.0], [0.0, 0.5]];
    /// let mvln = MultivariateLognormal::new(mu, sigma).unwrap();
    ///
    /// let mean = mvln.mean();
    /// ```
    pub fn mean(&self) -> Array1<f64> {
        let mut mean = Array1::zeros(self.dim);

        for i in 0..self.dim {
            mean[i] = (self.mu[i] + self.sigma[[i, i]] / 2.0).exp();
        }

        mean
    }

    /// Calculate the median of the distribution.
    ///
    /// For a multivariate lognormal distribution with parameters μ and Σ,
    /// the median of component i is exp(μ_i).
    ///
    /// # Returns
    ///
    /// * Median vector
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
    ///
    /// let mu = array![0.0, 0.0];
    /// let sigma = array![[0.5, 0.0], [0.0, 0.5]];
    /// let mvln = MultivariateLognormal::new(mu, sigma).unwrap();
    ///
    /// let median = mvln.median();
    /// ```
    pub fn median(&self) -> Array1<f64> {
        // Median of lognormal is exp(μ)
        self.mu.mapv(|mu_i| mu_i.exp())
    }

    /// Calculate the mode of the distribution.
    ///
    /// For a multivariate lognormal distribution with parameters μ and Σ,
    /// the mode of component i is exp(μ_i - Σ_ii).
    ///
    /// # Returns
    ///
    /// * Mode vector
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
    ///
    /// let mu = array![0.0, 0.0];
    /// let sigma = array![[0.5, 0.0], [0.0, 0.5]];
    /// let mvln = MultivariateLognormal::new(mu, sigma).unwrap();
    ///
    /// let mode = mvln.mode();
    /// ```
    pub fn mode(&self) -> Array1<f64> {
        let mut mode = Array1::zeros(self.dim);

        for i in 0..self.dim {
            mode[i] = (self.mu[i] - self.sigma[[i, i]]).exp();
        }

        mode
    }

    /// Calculate the covariance matrix of the distribution.
    ///
    /// For a multivariate lognormal distribution with parameters μ and Σ,
    /// the covariance between components i and j is:
    /// Cov(X_i, X_j) = exp(μ_i + μ_j + (Σ_ii + Σ_jj)/2) * (exp(Σ_ij) - 1)
    ///
    /// # Returns
    ///
    /// * Covariance matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::multivariate_lognormal::MultivariateLognormal;
    ///
    /// let mu = array![0.0, 0.0];
    /// let sigma = array![[0.5, 0.2], [0.2, 0.5]];
    /// let mvln = MultivariateLognormal::new(mu, sigma).unwrap();
    ///
    /// let cov = mvln.cov();
    /// ```
    pub fn cov(&self) -> Array2<f64> {
        let mut cov = Array2::zeros((self.dim, self.dim));

        for i in 0..self.dim {
            for j in 0..self.dim {
                // For a lognormal, Var(X_i) = exp(2*μ_i + Σ_ii) * (exp(Σ_ii) - 1)
                // Cov(X_i, X_j) = exp(μ_i + μ_j + (Σ_ii + Σ_jj)/2) * (exp(Σ_ij) - 1)

                // Mean of X_i = exp(μ_i + Σ_ii/2)
                let mean_i = (self.mu[i] + self.sigma[[i, i]] / 2.0).exp();
                let mean_j = (self.mu[j] + self.sigma[[j, j]] / 2.0).exp();

                // Formula from Aitchison & Brown (1957)
                let term = (self.sigma[[i, j]]).exp() - 1.0;
                cov[[i, j]] = mean_i * mean_j * term;
            }
        }

        cov
    }

    /// Get the dimension of the distribution.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the mean vector of the underlying normal distribution.
    pub fn mu(&self) -> ArrayView1<f64> {
        self.mu.view()
    }

    /// Get the covariance matrix of the underlying normal distribution.
    pub fn sigma(&self) -> ArrayView2<f64> {
        self.sigma.view()
    }
}

/// Create a multivariate lognormal distribution with the given parameters.
///
/// This is a convenience function to create a multivariate lognormal distribution with
/// the given mean vector and covariance matrix of the underlying multivariate normal distribution.
///
/// # Arguments
///
/// * `mu` - Mean vector of the underlying multivariate normal
/// * `sigma` - Covariance matrix of the underlying multivariate normal
///
/// # Returns
///
/// * A multivariate lognormal distribution object
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distributions::multivariate;
///
/// let mu = array![0.0, 0.0];
/// let sigma = array![[0.5, 0.2], [0.2, 0.5]];
/// let mvln = multivariate::multivariate_lognormal(mu, sigma).unwrap();
/// ```
#[allow(dead_code)]
pub fn multivariate_lognormal<D1, D2>(
    mu: ArrayBase<D1, Ix1>,
    sigma: ArrayBase<D2, Ix2>,
) -> StatsResult<MultivariateLognormal>
where
    D1: Data<Elem = f64>,
    D2: Data<Elem = f64>,
{
    MultivariateLognormal::new(mu, sigma)
}

/// Implementation of SampleableDistribution for MultivariateLognormal
impl SampleableDistribution<Array1<f64>> for MultivariateLognormal {
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
    fn test_mvln_creation() {
        // 2D multivariate lognormal
        let mu = array![0.0, 0.0];
        let sigma = array![[0.5, 0.0], [0.0, 0.5]];
        let mvln = MultivariateLognormal::new(mu.clone(), sigma.clone()).unwrap();

        assert_eq!(mvln.dim, 2);
        assert_eq!(mvln.mu, mu);
        assert_eq!(mvln.sigma, sigma);

        // 3D multivariate lognormal
        let mu3 = array![1.0, 2.0, 3.0];
        let sigma3 = array![[0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5]];
        let mvln3 = MultivariateLognormal::new(mu3.clone(), sigma3.clone()).unwrap();

        assert_eq!(mvln3.dim, 3);
        assert_eq!(mvln3.mu, mu3);
        assert_eq!(mvln3.sigma, sigma3);
    }

    #[test]
    fn test_mvln_creation_errors() {
        // Dimension mismatch
        let mu = array![0.0, 0.0, 0.0];
        let sigma = array![[0.5, 0.0], [0.0, 0.5]];
        assert!(MultivariateLognormal::new(mu, sigma).is_err());

        // Non-positive definite covariance
        let mu = array![0.0, 0.0];
        let sigma = array![[1.0, 2.0], [2.0, 1.0]]; // Not positive definite
        assert!(MultivariateLognormal::new(mu, sigma).is_err());
    }

    #[test]
    fn test_mvln_pdf() {
        // 2D independent lognormal
        let mu = array![0.0, 0.0];
        let sigma = array![[0.5, 0.0], [0.0, 0.5]];
        let mvln = MultivariateLognormal::new(mu, sigma).unwrap();

        // PDF at various points
        let x1 = array![1.0, 1.0];
        let pdf1 = mvln.pdf(&x1);
        assert!(pdf1 > 0.0);

        // PDF at a negative point should be 0
        let x2 = array![-1.0, 1.0];
        let pdf2 = mvln.pdf(&x2);
        assert_eq!(pdf2, 0.0);

        // PDF at a zero point should be 0
        let x3 = array![0.0, 1.0];
        let pdf3 = mvln.pdf(&x3);
        assert_eq!(pdf3, 0.0);

        // Wrong dimension
        let x4 = array![1.0, 1.0, 1.0];
        let pdf4 = mvln.pdf(&x4);
        assert_eq!(pdf4, 0.0);
    }

    #[test]
    fn test_mvln_logpdf() {
        // 2D independent lognormal
        let mu = array![0.0, 0.0];
        let sigma = array![[0.5, 0.0], [0.0, 0.5]];
        let mvln = MultivariateLognormal::new(mu, sigma).unwrap();

        // LogPDF at various points
        let x1 = array![1.0, 1.0];
        let pdf1 = mvln.pdf(&x1);
        let logpdf1 = mvln.logpdf(&x1);
        assert_relative_eq!(logpdf1.exp(), pdf1, epsilon = 1e-10);

        // LogPDF at a negative point should be -inf
        let x2 = array![-1.0, 1.0];
        let logpdf2 = mvln.logpdf(&x2);
        assert_eq!(logpdf2, f64::NEG_INFINITY);
    }

    #[test]
    fn test_mvln_statistics() {
        // Create a 2D lognormal with known parameters
        let mu = array![0.0, 0.0];
        let sigma = array![[0.5, 0.0], [0.0, 0.5]];
        let mvln = MultivariateLognormal::new(mu, sigma).unwrap();

        // Mean
        let mean = mvln.mean();
        let expected_mean = array![(0.5_f64 / 2.0).exp(), (0.5_f64 / 2.0).exp()];
        assert_relative_eq!(mean[0], expected_mean[0], epsilon = 1e-10);
        assert_relative_eq!(mean[1], expected_mean[1], epsilon = 1e-10);

        // Median
        let median = mvln.median();
        let expected_median = array![1.0, 1.0]; // exp(0.0) = 1.0
        assert_relative_eq!(median[0], expected_median[0], epsilon = 1e-10);
        assert_relative_eq!(median[1], expected_median[1], epsilon = 1e-10);

        // Mode
        let mode = mvln.mode();
        let expected_mode = array![(-0.5_f64).exp(), (-0.5_f64).exp()]; // exp(μ - σ^2)
        assert_relative_eq!(mode[0], expected_mode[0], epsilon = 1e-10);
        assert_relative_eq!(mode[1], expected_mode[1], epsilon = 1e-10);

        // Test covariance for a lognormal distribution with independent variables
        let cov = mvln.cov();

        // For a lognormal with μ = 0, Var(X_i) = mean_i^2 * (exp(Σ_ii) - 1)
        // where mean_i = exp(Σ_ii/2)
        let mean_i = (0.5_f64 / 2.0).exp();
        let var0 = mean_i * mean_i * ((0.5_f64).exp() - 1.0);

        assert_relative_eq!(cov[[0, 0]], var0, epsilon = 1e-10);
        assert_relative_eq!(cov[[1, 1]], var0, epsilon = 1e-10);

        // Since our Σ has 0 off-diagonal elements (independent case), the covariance is 0
        assert_relative_eq!(cov[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cov[[1, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mvln_rvs() {
        // Create a 2D lognormal
        let mu = array![0.0, 0.0];
        let sigma = array![[0.5, 0.2], [0.2, 0.5]];
        let mvln = MultivariateLognormal::new(mu, sigma).unwrap();

        // Generate samples
        let n_samples_ = 1000;
        let samples = mvln.rvs(n_samples_).unwrap();
        assert_eq!(samples.shape(), &[n_samples_, 2]);

        // Check all samples are positive
        for i in 0..n_samples_ {
            for j in 0..2 {
                assert!(samples[[i, j]] > 0.0);
            }
        }

        // Verify sample statistics (rough check due to randomness)
        // Calculate sample means
        let sample_mean = samples.mean_axis(Axis(0)).unwrap();
        let expected_mean = mvln.mean();
        assert_relative_eq!(sample_mean[0], expected_mean[0], epsilon = 0.2);
        assert_relative_eq!(sample_mean[1], expected_mean[1], epsilon = 0.2);

        // Single sample
        let single_sample = mvln.rvs_single().unwrap();
        assert_eq!(single_sample.len(), 2);
        assert!(single_sample[0] > 0.0);
        assert!(single_sample[1] > 0.0);
    }
}
