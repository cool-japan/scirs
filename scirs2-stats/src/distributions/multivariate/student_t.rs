//! Multivariate Student's t-distribution functions
//!
//! This module provides functionality for the Multivariate Student's t-distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::ndarray::{
    s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1, Ix2,
};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{ChiSquared, Distribution, Normal as RandNormal};
use std::fmt::Debug;

// Import the helper functions used by MultivariateNormal
use super::normal::{compute_cholesky, compute_inverse_from_cholesky};

// Implementation of the natural logarithm of the gamma function
// This is a workaround for the unstable gamma function in Rust
#[allow(dead_code)]
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        panic!("lgamma requires positive input");
    }

    // For integers, we can use a simpler calculation
    if x.fract() == 0.0 && x <= 20.0 {
        let n = x as usize;
        if n == 1 || n == 2 {
            return 0.0; // ln(1) = 0
        }

        let mut result = 0.0;
        for i in 2..n {
            result += (i as f64).ln();
        }
        return result;
    }

    // For x = 0.5, we have Γ(0.5) = sqrt(π)
    if (x - 0.5).abs() < 1e-10 {
        return (std::f64::consts::PI.sqrt()).ln();
    }

    // For x > 1, use the recurrence relation: Γ(x+1) = x * Γ(x)
    if x > 1.0 {
        return (x - 1.0).ln() + lgamma(x - 1.0);
    }

    // For 0 < x < 1, use the reflection formula: Γ(x) * Γ(1-x) = π/sin(πx)
    if x < 1.0 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x);
    }

    // Lanczos approximation for other values around 1
    let p = [
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    let x_adj = x - 1.0;
    let t = x_adj + 7.5;

    let mut sum = 0.0;
    for (i, &coef) in p.iter().enumerate() {
        sum += coef / (x_adj + (i + 1) as f64);
    }

    let pi = std::f64::consts::PI;
    let sqrt_2pi = (2.0 * pi).sqrt();

    sqrt_2pi.ln() + sum.ln() + (x_adj + 0.5) * t.ln() - t
}

/// Multivariate Student's t-distribution structure
#[derive(Debug, Clone)]
pub struct MultivariateT {
    /// Mean vector
    pub mean: Array1<f64>,
    /// Scale matrix (like covariance but scaled by df/(df-2) for df > 2)
    pub scale: Array2<f64>,
    /// Dimensionality of the distribution
    pub dim: usize,
    /// Degrees of freedom
    pub df: f64,
    /// Cholesky decomposition of the scale matrix (lower triangular)
    cholesky_l: Array2<f64>,
    /// Determinant of the scale matrix
    scale_det: f64,
    /// Inverse of the scale matrix
    scale_inv: Array2<f64>,
}

impl MultivariateT {
    /// Create a new multivariate Student's t-distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean vector (k-dimensional)
    /// * `scale` - Scale matrix (k x k, symmetric positive-definite)
    /// * `df` - Degrees of freedom (> 0)
    ///
    /// # Returns
    ///
    /// * A new MultivariateT distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::student_t::MultivariateT;
    ///
    /// // Create a 2D multivariate Student's t-distribution with 5 degrees of freedom
    /// let mean = array![0.0, 0.0];
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let mvt = MultivariateT::new(mean, scale, 5.0).unwrap();
    /// ```
    pub fn new<D1, D2>(
        mean: ArrayBase<D1, Ix1>,
        scale: ArrayBase<D2, Ix2>,
        df: f64,
    ) -> StatsResult<Self>
    where
        D1: Data<Elem = f64>,
        D2: Data<Elem = f64>,
    {
        // Validate dimensions
        let dim = mean.len();
        if scale.shape()[0] != dim || scale.shape()[1] != dim {
            return Err(StatsError::DimensionMismatch(format!(
                "Scale matrix shape ({:?}) must match mean vector length ({})",
                scale.shape(),
                dim
            )));
        }

        // Validate degrees of freedom
        if df <= 0.0 {
            return Err(StatsError::DomainError(
                "Degrees of freedom must be positive".to_string(),
            ));
        }

        // Create owned copies of inputs
        let mean = mean.to_owned();
        let scale = scale.to_owned();

        // Compute Cholesky decomposition (lower triangular L where Σ = L·L^T)
        let cholesky_l = compute_cholesky(&scale).map_err(|_| {
            StatsError::DomainError("Scale matrix must be positive definite".to_string())
        })?;

        // For positive definite matrix, det(Σ) = det(L)^2 = prod(diag(L))^2
        let scale_det = {
            let mut det = 1.0;
            for i in 0..dim {
                det *= cholesky_l[[i, i]];
            }
            det * det // Square it since det(Σ) = det(L)^2
        };

        // Compute inverse using Cholesky decomposition
        let scale_inv = compute_inverse_from_cholesky(&cholesky_l).map_err(|_| {
            StatsError::ComputationError("Failed to compute matrix inverse".to_string())
        })?;

        Ok(MultivariateT {
            mean,
            scale,
            dim,
            df,
            cholesky_l,
            scale_det,
            scale_inv,
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
    /// use scirs2_stats::distributions::multivariate::student_t::MultivariateT;
    ///
    /// let mean = array![0.0, 0.0];
    /// let scale = array![[1.0, 0.0], [0.0, 1.0]];
    /// let mvt = MultivariateT::new(mean, scale, 5.0).unwrap();
    ///
    /// // PDF at origin
    /// let pdf_at_origin = mvt.pdf(&array![0.0, 0.0]);
    /// ```
    pub fn pdf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        if x.len() != self.dim {
            return 0.0; // Return zero for invalid dimensions
        }

        let pi = std::f64::consts::PI;

        // Calculate the constant part of the PDF
        let gamma_term_num = lgamma((self.df + self.dim as f64) / 2.0).exp();
        let gamma_term_denom = lgamma(self.df / 2.0).exp()
            * lgamma(self.dim as f64 / 2.0).exp()
            * self.df.powf(self.dim as f64 / 2.0);
        let constant_factor = gamma_term_num
            / (gamma_term_denom * pi.powf(self.dim as f64 / 2.0) * self.scale_det.sqrt());

        // Calculate Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
        let diff = x - &self.mean;
        let mahalanobis_squared = self.mahalanobis_distance_squared(&diff.view());

        // PDF = C * [1 + (1/v) * dist]^(-(v+p)/2)
        // where C is a normalization constant, v is df, p is dimension, and dist is Mahalanobis distance squared
        constant_factor
            * (1.0 + mahalanobis_squared / self.df).powf(-(self.df + self.dim as f64) / 2.0)
    }

    /// Calculate the Mahalanobis distance squared: (x - μ)^T Σ^-1 (x - μ)
    fn mahalanobis_distance_squared(&self, diff: &ArrayView1<f64>) -> f64 {
        // Compute (x - μ)^T Σ^-1 (x - μ)
        diff.dot(&self.scale_inv.dot(diff))
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
    /// use scirs2_stats::distributions::multivariate::student_t::MultivariateT;
    ///
    /// let mean = array![0.0, 0.0];
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let mvt = MultivariateT::new(mean, scale, 5.0).unwrap();
    ///
    /// let samples = mvt.rvs(100).unwrap();
    /// assert_eq!(samples.shape(), &[100, 2]);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Array2<f64>> {
        let mut rng = thread_rng();
        let normal_dist = RandNormal::new(0.0, 1.0).unwrap();
        let chi2_dist = ChiSquared::new(self.df).unwrap();

        // Create a matrix for the samples
        let mut samples = Array2::<f64>::zeros((size, self.dim));

        // For each sample
        for i in 0..size {
            // Generate standard normal samples for each dimension
            let mut z = Array1::<f64>::zeros(self.dim);
            for j in 0..self.dim {
                z[j] = normal_dist.sample(&mut rng);
            }

            // Generate chi-square sample with df degrees of freedom
            let w = chi2_dist.sample(&mut rng);

            // Transform Z using the Cholesky decomposition
            let mut transformed = Array1::<f64>::zeros(self.dim);
            for j in 0..self.dim {
                for k in 0..=j {
                    transformed[j] += self.cholesky_l[[j, k]] * z[k];
                }
            }

            // Apply the t-distribution scaling
            let scaling_factor = (self.df / w).sqrt();
            for j in 0..self.dim {
                samples[[i, j]] = self.mean[j] + transformed[j] * scaling_factor;
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
    /// use scirs2_stats::distributions::multivariate::student_t::MultivariateT;
    ///
    /// let mean = array![0.0, 0.0];
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let mvt = MultivariateT::new(mean, scale, 5.0).unwrap();
    ///
    /// let sample = mvt.rvs_single().unwrap();
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
    /// use scirs2_stats::distributions::multivariate::student_t::MultivariateT;
    ///
    /// let mean = array![0.0, 0.0];
    /// let scale = array![[1.0, 0.0], [0.0, 1.0]];
    /// let mvt = MultivariateT::new(mean, scale, 5.0).unwrap();
    ///
    /// let log_pdf = mvt.logpdf(&array![0.0, 0.0]);
    /// ```
    pub fn logpdf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        if x.len() != self.dim {
            return f64::NEG_INFINITY; // Return -∞ for invalid dimensions
        }

        let pi = std::f64::consts::PI;

        // Calculate the constant part of the log PDF
        let gamma_term_num = lgamma((self.df + self.dim as f64) / 2.0);
        let gamma_term_denom = lgamma(self.df / 2.0)
            + lgamma(self.dim as f64 / 2.0)
            + (self.dim as f64 / 2.0) * self.df.ln();
        let log_const = gamma_term_num
            - gamma_term_denom
            - (self.dim as f64 / 2.0) * pi.ln()
            - 0.5 * self.scale_det.ln();

        // Calculate Mahalanobis distance: (x - μ)^T Σ^-1 (x - μ)
        let diff = x - &self.mean;
        let mahalanobis_squared = self.mahalanobis_distance_squared(&diff.view());

        // log(PDF) = log(C) - ((v+p)/2) * log(1 + (1/v) * dist)
        log_const - ((self.df + self.dim as f64) / 2.0) * (1.0 + mahalanobis_squared / self.df).ln()
    }

    /// Get the dimension of the distribution
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the scale matrix of the distribution
    pub fn scale(&self) -> ArrayView2<f64> {
        self.scale.view()
    }

    /// Get the mean vector of the distribution
    pub fn mean(&self) -> ArrayView1<f64> {
        self.mean.view()
    }

    /// Get the degrees of freedom of the distribution
    pub fn df(&self) -> f64 {
        self.df
    }
}

/// Create a multivariate Student's t-distribution with the given parameters.
///
/// This is a convenience function to create a multivariate Student's t-distribution with
/// the given mean vector, scale matrix, and degrees of freedom.
///
/// # Arguments
///
/// * `mean` - Mean vector (k-dimensional)
/// * `scale` - Scale matrix (k x k, symmetric positive-definite)
/// * `df` - Degrees of freedom (> 0)
///
/// # Returns
///
/// * A multivariate Student's t-distribution object
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distributions::multivariate;
///
/// let mean = array![0.0, 0.0];
/// let scale = array![[1.0, 0.5], [0.5, 2.0]];
/// let mvt = multivariate::multivariate_t(mean, scale, 5.0).unwrap();
/// let pdf_at_origin = mvt.pdf(&array![0.0, 0.0]);
/// ```
#[allow(dead_code)]
pub fn multivariate_t<D1, D2>(
    mean: ArrayBase<D1, Ix1>,
    scale: ArrayBase<D2, Ix2>,
    df: f64,
) -> StatsResult<MultivariateT>
where
    D1: Data<Elem = f64>,
    D2: Data<Elem = f64>,
{
    MultivariateT::new(mean, scale, df)
}

/// Implementation of SampleableDistribution for MultivariateT
impl SampleableDistribution<Array1<f64>> for MultivariateT {
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
    fn test_mvt_creation() {
        // 2D standard multivariate t
        let mean = array![0.0, 0.0];
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        let mvt = MultivariateT::new(mean.clone(), scale.clone(), 5.0).unwrap();

        assert_eq!(mvt.dim, 2);
        assert_eq!(mvt.mean, mean);
        assert_eq!(mvt.scale, scale);
        assert_eq!(mvt.df, 5.0);

        // Custom 3D multivariate t
        let mean3 = array![1.0, 2.0, 3.0];
        let scale3 = array![[1.0, 0.5, 0.3], [0.5, 2.0, 0.2], [0.3, 0.2, 1.5]];
        let mvt3 = MultivariateT::new(mean3.clone(), scale3.clone(), 10.0).unwrap();

        assert_eq!(mvt3.dim, 3);
        assert_eq!(mvt3.mean, mean3);
        assert_eq!(mvt3.scale, scale3);
        assert_eq!(mvt3.df, 10.0);
    }

    #[test]
    fn test_mvt_creation_errors() {
        // Dimension mismatch
        let mean = array![0.0, 0.0, 0.0];
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        assert!(MultivariateT::new(mean, scale, 5.0).is_err());

        // Non-positive definite scale matrix
        let mean = array![0.0, 0.0];
        let scale = array![[1.0, 2.0], [2.0, 1.0]]; // Not positive definite
        assert!(MultivariateT::new(mean, scale, 5.0).is_err());

        // Invalid degrees of freedom
        let mean = array![0.0, 0.0];
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        assert!(MultivariateT::new(mean.clone(), scale.clone(), 0.0).is_err());
        assert!(MultivariateT::new(mean, scale, -1.0).is_err());
    }

    #[test]
    fn test_mvt_pdf() {
        // 2D standard multivariate t with 5 degrees of freedom
        let mean = array![0.0, 0.0];
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        let mvt = MultivariateT::new(mean, scale, 5.0).unwrap();

        // PDF at origin should be calculable
        let pdf_at_origin = mvt.pdf(&array![0.0, 0.0]);
        assert!(pdf_at_origin > 0.0);

        // PDF at origin should be greater than at [1, 1]
        let pdf_at_one = mvt.pdf(&array![1.0, 1.0]);
        assert!(pdf_at_origin > pdf_at_one);

        // PDF should be symmetric
        let pdf_at_pos = mvt.pdf(&array![2.0, 1.0]);
        let pdf_at_neg = mvt.pdf(&array![-2.0, -1.0]);
        assert_relative_eq!(pdf_at_pos, pdf_at_neg, epsilon = 1e-10);
    }

    #[test]
    fn test_mvt_logpdf() {
        // 2D standard multivariate t with 5 degrees of freedom
        let mean = array![0.0, 0.0];
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        let mvt = MultivariateT::new(mean, scale, 5.0).unwrap();

        // Check that exp(logPDF) = PDF
        let x = array![1.0, 1.0];
        let pdf = mvt.pdf(&x);
        let logpdf = mvt.logpdf(&x);
        assert_relative_eq!(logpdf.exp(), pdf, epsilon = 1e-7);
    }

    #[test]
    fn test_mvt_rvs() {
        // 2D multivariate t with correlation and 10 degrees of freedom
        let mean = array![1.0, 2.0];
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let mvt = MultivariateT::new(mean, scale, 10.0).unwrap();

        // Generate samples and check dimensions
        let n_samples_ = 1000;
        let samples = mvt.rvs(n_samples_).unwrap();
        assert_eq!(samples.shape(), &[n_samples_, 2]);

        // Check statistics (rough check as it's random and t-distribution has heavier tails)
        let sample_mean = samples.mean_axis(Axis(0)).unwrap();
        assert_relative_eq!(sample_mean[0], 1.0, epsilon = 0.3);
        assert_relative_eq!(sample_mean[1], 2.0, epsilon = 0.3);
    }

    #[test]
    fn test_mvt_rvs_single() {
        let mean = array![1.0, 2.0];
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let mvt = MultivariateT::new(mean, scale, 5.0).unwrap();

        let sample = mvt.rvs_single().unwrap();
        assert_eq!(sample.len(), 2);
    }
}
