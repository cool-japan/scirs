//! Dirichlet distribution functions
//!
//! This module provides functionality for the Dirichlet distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::ndarray::{Array1, ArrayBase, Data, Ix1};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Gamma as RandGamma};
use std::fmt::Debug;

/// Implementation of the natural logarithm of the gamma function
///
/// This is a workaround for the unstable gamma function in Rust
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

/// Dirichlet distribution structure
#[derive(Debug, Clone)]
pub struct Dirichlet {
    /// Concentration parameters (alpha values)
    pub alpha: Array1<f64>,
    /// Dimension of the distribution (number of categories)
    pub dim: usize,
    /// Natural log of the normalization constant (cached for efficiency)
    log_norm_const: f64,
}

impl Dirichlet {
    /// Create a new Dirichlet distribution with given concentration parameters
    ///
    /// # Arguments
    ///
    /// * `alpha` - Concentration parameters (all values must be positive)
    ///
    /// # Returns
    ///
    /// * A new Dirichlet distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::dirichlet::Dirichlet;
    ///
    /// // Create a 3D Dirichlet distribution with symmetric parameters (equivalent to a uniform distribution over the simplex)
    /// let alpha = array![1.0, 1.0, 1.0];
    /// let dirichlet = Dirichlet::new(alpha).unwrap();
    /// ```
    pub fn new<D>(alpha: ArrayBase<D, Ix1>) -> StatsResult<Self>
    where
        D: Data<Elem = f64>,
    {
        let alpha_owned = alpha.to_owned();
        let dim = alpha_owned.len();

        // Check that all _alpha values are positive
        for &a in alpha_owned.iter() {
            if a <= 0.0 {
                return Err(StatsError::DomainError(
                    "All concentration parameters must be positive".to_string(),
                ));
            }
        }

        let alpha_sum = alpha_owned.sum();

        // Compute the log normalization constant:
        // ln[B(α)] = sum(ln[Γ(αᵢ)]) - ln[Γ(sum(αᵢ))]
        let mut log_norm_const = 0.0;

        // Sum of log(Gamma(alpha_i))
        for &a in alpha_owned.iter() {
            log_norm_const += lgamma(a);
        }

        // Subtract log(Gamma(sum(alpha_i)))
        log_norm_const -= lgamma(alpha_sum);

        Ok(Dirichlet {
            alpha: alpha_owned,
            dim,
            log_norm_const,
        })
    }

    /// Calculate the probability density function (PDF) at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the PDF (must sum to 1)
    ///
    /// # Returns
    ///
    /// * The value of the PDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::dirichlet::Dirichlet;
    ///
    /// let alpha = array![1.0, 1.0, 1.0];
    /// let dirichlet = Dirichlet::new(alpha).unwrap();
    ///
    /// // PDF for a uniform Dirichlet at any point on the simplex is 2 (in 3D)
    /// let point = array![0.3, 0.3, 0.4];
    /// let pdf_value = dirichlet.pdf(&point);
    /// assert!((pdf_value - 2.0).abs() < 1e-10);
    /// ```
    pub fn pdf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        if x.len() != self.dim {
            return 0.0; // Return zero for invalid dimensions
        }

        // Check if x is on the simplex (all values > 0 and sum to 1)
        let sum: f64 = x.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return 0.0; // Point not on the simplex
        }

        for &val in x.iter() {
            if val <= 0.0 || val >= 1.0 {
                return 0.0; // Values must be in (0, 1)
            }
        }

        // Calculate the PDF using the formula:
        // p(x|α) = [∏ xᵢ^(αᵢ-1)] / B(α)
        // where B(α) is the multivariate beta function

        // We'll work in log space for numerical stability
        let log_pdf = self.logpdf(x);
        log_pdf.exp()
    }

    /// Calculate the log probability density function (log PDF) at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the log PDF (must sum to 1)
    ///
    /// # Returns
    ///
    /// * The value of the log PDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::dirichlet::Dirichlet;
    ///
    /// let alpha = array![1.0, 1.0, 1.0];
    /// let dirichlet = Dirichlet::new(alpha).unwrap();
    ///
    /// let point = array![0.3, 0.3, 0.4];
    /// let logpdf_value = dirichlet.logpdf(&point);
    /// assert!((logpdf_value - 0.693).abs() < 1e-3);  // ln(2) ≈ 0.693
    /// ```
    pub fn logpdf<D>(&self, x: &ArrayBase<D, Ix1>) -> f64
    where
        D: Data<Elem = f64>,
    {
        if x.len() != self.dim {
            return f64::NEG_INFINITY; // Return -∞ for invalid dimensions
        }

        // Check if x is on the simplex (all values > 0 and sum to 1)
        let sum: f64 = x.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return f64::NEG_INFINITY; // Point not on the simplex
        }

        for &val in x.iter() {
            if val <= 0.0 || val >= 1.0 {
                return f64::NEG_INFINITY; // Values must be in (0, 1)
            }
        }

        // Calculate the log PDF using the formula:
        // log p(x|α) = sum[(αᵢ-1)log(xᵢ)] - log B(α)
        let mut log_pdf = -self.log_norm_const;

        for i in 0..self.dim {
            log_pdf += (self.alpha[i] - 1.0) * x[i].ln();
        }

        log_pdf
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
    /// use scirs2_stats::distributions::multivariate::dirichlet::Dirichlet;
    ///
    /// let alpha = array![1.0, 2.0, 3.0];
    /// let dirichlet = Dirichlet::new(alpha).unwrap();
    ///
    /// let samples = dirichlet.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// assert_eq!(samples[0].len(), 3);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<Array1<f64>>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(size);

        // Generate samples using the gamma method:
        // 1. Generate independent gamma samples with shape αᵢ and scale=1
        // 2. Normalize by their sum

        for _ in 0..size {
            let mut sample = Array1::<f64>::zeros(self.dim);
            let mut sum = 0.0;

            // Generate gamma samples
            for i in 0..self.dim {
                let gamma_dist = RandGamma::new(self.alpha[i], 1.0).map_err(|_| {
                    StatsError::ComputationError("Failed to create gamma distribution".to_string())
                })?;

                let gamma_sample = gamma_dist.sample(&mut rng);
                sample[i] = gamma_sample;
                sum += gamma_sample;
            }

            // Normalize to get a point on the simplex
            sample.mapv_inplace(|x| x / sum);
            samples.push(sample);
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
    /// use scirs2_stats::distributions::multivariate::dirichlet::Dirichlet;
    ///
    /// let alpha = array![1.0, 2.0, 3.0];
    /// let dirichlet = Dirichlet::new(alpha).unwrap();
    ///
    /// let sample = dirichlet.rvs_single().unwrap();
    /// assert_eq!(sample.len(), 3);
    /// ```
    pub fn rvs_single(&self) -> StatsResult<Array1<f64>> {
        let samples = self.rvs(1)?;
        Ok(samples[0].clone())
    }
}

/// Create a Dirichlet distribution with the given parameters.
///
/// This is a convenience function to create a Dirichlet distribution with
/// the given concentration parameters.
///
/// # Arguments
///
/// * `alpha` - Concentration parameters (all values must be positive)
///
/// # Returns
///
/// * A Dirichlet distribution object
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distributions::multivariate;
///
/// let alpha = array![1.0, 1.0, 1.0];
/// let dirichlet = multivariate::dirichlet(&alpha).unwrap();
/// let point = array![0.3, 0.3, 0.4];
/// let pdf_at_point = dirichlet.pdf(&point);
/// ```
#[allow(dead_code)]
pub fn dirichlet<D>(alpha: &ArrayBase<D, Ix1>) -> StatsResult<Dirichlet>
where
    D: Data<Elem = f64>,
{
    Dirichlet::new(alpha.to_owned())
}

/// Implementation of SampleableDistribution for Dirichlet
impl SampleableDistribution<Array1<f64>> for Dirichlet {
    fn rvs(&self, size: usize) -> StatsResult<Vec<Array1<f64>>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_dirichlet_creation() {
        // Uniform Dirichlet
        let alpha = array![1.0, 1.0, 1.0];
        let dirichlet = Dirichlet::new(alpha.clone()).unwrap();

        assert_eq!(dirichlet.dim, 3);
        assert_eq!(dirichlet.alpha, alpha);

        // Non-uniform Dirichlet
        let alpha2 = array![2.0, 3.0, 4.0];
        let dirichlet2 = Dirichlet::new(alpha2.clone()).unwrap();

        assert_eq!(dirichlet2.dim, 3);
        assert_eq!(dirichlet2.alpha, alpha2);
    }

    #[test]
    fn test_dirichlet_creation_errors() {
        // Zero alpha value
        let alpha = array![1.0, 0.0, 1.0];
        assert!(Dirichlet::new(alpha).is_err());

        // Negative alpha value
        let alpha = array![1.0, -1.0, 1.0];
        assert!(Dirichlet::new(alpha).is_err());
    }

    #[test]
    fn test_dirichlet_pdf() {
        // Uniform Dirichlet (alpha = [1,1,1])
        // PDF value should be constant on the simplex: 2 for 3D
        let alpha = array![1.0, 1.0, 1.0];
        let dirichlet = Dirichlet::new(alpha).unwrap();

        let point1 = array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let point2 = array![0.2, 0.3, 0.5];

        assert_relative_eq!(dirichlet.pdf(&point1), 2.0, epsilon = 1e-10);
        assert_relative_eq!(dirichlet.pdf(&point2), 2.0, epsilon = 1e-10);

        // Concentrated Dirichlet
        let alpha = array![5.0, 5.0, 5.0];
        let concentrated = Dirichlet::new(alpha).unwrap();

        // PDF should be higher at the center than at the edges
        let center = array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let edge = array![0.01, 0.01, 0.98];

        assert!(concentrated.pdf(&center) > concentrated.pdf(&edge));
    }

    #[test]
    fn test_dirichlet_pdf_edge_cases() {
        let alpha = array![1.0, 1.0, 1.0];
        let dirichlet = Dirichlet::new(alpha).unwrap();

        // Points not on the simplex
        let invalid1 = array![0.3, 0.3, 0.3]; // Sum != 1
        let invalid2 = array![0.5, 0.6, 0.2]; // Sum > 1
        let invalid3 = array![0.0, 0.5, 0.5]; // Contains 0
        let invalid4 = array![1.0, 0.0, 0.0]; // Contains 0

        assert_eq!(dirichlet.pdf(&invalid1), 0.0);
        assert_eq!(dirichlet.pdf(&invalid2), 0.0);
        assert_eq!(dirichlet.pdf(&invalid3), 0.0);
        assert_eq!(dirichlet.pdf(&invalid4), 0.0);
    }

    #[test]
    fn test_dirichlet_logpdf() {
        let alpha = array![1.0, 1.0, 1.0];
        let dirichlet = Dirichlet::new(alpha).unwrap();

        let point = array![0.3, 0.3, 0.4];

        // Log of uniform Dirichlet with alpha=[1,1,1] is ln(2) ≈ 0.693
        assert_relative_eq!(dirichlet.logpdf(&point), 0.693, epsilon = 1e-3);

        // Check that exp(logPDF) = PDF
        assert_relative_eq!(
            dirichlet.logpdf(&point).exp(),
            dirichlet.pdf(&point),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_dirichlet_rvs() {
        let alpha = array![1.0, 2.0, 3.0];
        let dirichlet = Dirichlet::new(alpha.clone()).unwrap();

        // Generate samples
        let n_samples_ = 1000;
        let samples = dirichlet.rvs(n_samples_).unwrap();

        // Check number of samples
        assert_eq!(samples.len(), n_samples_);

        // Check that all samples sum to 1 (within floating point error)
        for sample in &samples {
            let sum: f64 = sample.iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

            // Check all values are in [0,1]
            for &val in sample.iter() {
                assert!(val >= 0.0 && val <= 1.0);
            }
        }

        // Check sample mean is close to expected mean: E[X_i] = alpha_i / sum(alpha)
        let mut sample_mean = [0.0; 3];
        for sample in &samples {
            for i in 0..3 {
                sample_mean[i] += sample[i];
            }
        }

        let alpha_sum = alpha.sum();
        for i in 0..3 {
            sample_mean[i] /= n_samples_ as f64;
            let expected_mean = alpha[i] / alpha_sum;
            assert_relative_eq!(sample_mean[i], expected_mean, epsilon = 0.05);
        }
    }

    #[test]
    fn test_dirichlet_rvs_single() {
        let alpha = array![1.0, 2.0, 3.0];
        let dirichlet = Dirichlet::new(alpha.clone()).unwrap();

        let sample = dirichlet.rvs_single().unwrap();

        // Check sample dimension
        assert_eq!(sample.len(), 3);

        // Check sample sums to 1
        let sum: f64 = sample.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

        // Check all values in [0,1]
        for &val in sample.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }
}
