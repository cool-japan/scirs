//! Inverse Wishart distribution functions
//!
//! This module provides functionality for the Inverse Wishart distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::ndarray::{Array2, ArrayBase, Data, Ix2};
use std::fmt::Debug;

// Import helper functions from the multivariate module
use super::normal::{compute_cholesky, compute_inverse_from_cholesky};
use super::wishart::Wishart;

/// Inverse Wishart distribution structure
#[derive(Debug, Clone)]
pub struct InverseWishart {
    /// Scale matrix
    pub scale: Array2<f64>,
    /// Degrees of freedom
    pub df: f64,
    /// Dimension of the distribution (p x p)
    pub dim: usize,
    /// Cholesky decomposition of the scale matrix
    #[allow(dead_code)]
    scale_chol: Array2<f64>,
    /// Determinant of the scale matrix
    scale_det: f64,
}

impl InverseWishart {
    /// Create a new Inverse Wishart distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale matrix (p x p, symmetric positive-definite)
    /// * `df` - Degrees of freedom (must be greater than p + 1)
    ///
    /// # Returns
    ///
    /// * A new Inverse Wishart distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::inverse_wishart::InverseWishart;
    ///
    /// // Create a 2D Inverse Wishart distribution with 5 degrees of freedom
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;
    /// let inv_wishart = InverseWishart::new(scale, df).unwrap();
    /// ```
    pub fn new<D>(scale: ArrayBase<D, Ix2>, df: f64) -> StatsResult<Self>
    where
        D: Data<Elem = f64>,
    {
        let scale_owned = scale.to_owned();
        let dim = scale_owned.shape()[0];

        // Check if the matrix is square
        if scale_owned.shape()[1] != dim {
            return Err(StatsError::DimensionMismatch(
                "Scale matrix must be square".to_string(),
            ));
        }

        // Check the degrees of freedom constraint: df > p + 1
        if df <= dim as f64 + 1.0 {
            return Err(StatsError::DomainError(format!(
                "Degrees of freedom ({}) must be greater than dimension + 1 ({})",
                df,
                dim + 1
            )));
        }

        // Compute Cholesky decomposition to check if the matrix is positive definite
        let scale_chol = compute_cholesky(&scale_owned).map_err(|_| {
            StatsError::DomainError("Scale matrix must be positive definite".to_string())
        })?;

        // Compute determinant of the _scale matrix
        let scale_det = {
            let mut det = 1.0;
            for i in 0..dim {
                det *= scale_chol[[i, i]];
            }
            det * det // Square it since det(Σ) = det(L)^2
        };

        Ok(InverseWishart {
            scale: scale_owned,
            df,
            dim,
            scale_chol,
            scale_det,
        })
    }

    /// Calculate the probability density function (PDF) at a given matrix point
    ///
    /// # Arguments
    ///
    /// * `x` - The matrix at which to evaluate the PDF (p x p)
    ///
    /// # Returns
    ///
    /// * The value of the PDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::inverse_wishart::InverseWishart;
    ///
    /// let scale = array![[1.0, 0.0], [0.0, 1.0]];
    /// let df = 5.0;
    /// let inv_wishart = InverseWishart::new(scale, df).unwrap();
    ///
    /// let x = array![[0.2, 0.05], [0.05, 0.3]];
    /// let pdf_value = inv_wishart.pdf(&x);
    /// ```
    pub fn pdf<D>(&self, x: &ArrayBase<D, Ix2>) -> f64
    where
        D: Data<Elem = f64>,
    {
        // Check if x has the right dimensions
        if x.shape()[0] != self.dim || x.shape()[1] != self.dim {
            return 0.0;
        }

        // Check if x is symmetric and positive definite
        let x_owned = x.to_owned();
        let x_chol = match compute_cholesky(&x_owned) {
            Ok(chol) => chol,
            Err(_) => return 0.0, // Not positive definite, PDF is 0
        };

        // Compute the log PDF and exponentiate
        self.logpdf_with_cholesky(x, &x_chol).exp()
    }

    /// Calculate the log PDF with precomputed Cholesky decomposition of x
    fn logpdf_with_cholesky<D>(&self, x: &ArrayBase<D, Ix2>, xchol: &Array2<f64>) -> f64
    where
        D: Data<Elem = f64>,
    {
        // Calculate determinant of _x
        let mut x_det = 1.0;
        for i in 0..self.dim {
            x_det *= xchol[[i, i]];
        }
        x_det = x_det * x_det; // Square it since det(X) = det(L)^2

        // Compute x_inv using its Cholesky decomposition
        let x_inv = compute_inverse_from_cholesky(xchol).expect("Failed to compute matrix inverse");

        // Calculate trace(Ψ·X^-1)
        let mut trace = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                trace += self.scale[[i, j]] * x_inv[[j, i]];
            }
        }

        // Calculate log PDF
        // ln(PDF) = -0.5 * [tr(Ψ·X^-1) + ln|Ψ| + p ln 2 + 2 ln Γₚ((n+p+1)/2)]
        //           - 0.5 * (n + p + 1) ln|X|
        // Note: For Inverse Wishart, we use n+p+1 as the effective degree of freedom
        let p = self.dim as f64;
        let n = self.df;

        let term1 = -0.5 * trace;
        let term2 = -0.5 * self.scale_det.ln();
        let term3 = -0.5 * p * (2.0f64).ln();
        let term4 = -super::wishart::lmultigamma(n, self.dim);
        let term5 = -0.5 * (n + p + 1.0) * x_det.ln();

        term1 + term2 + term3 + term4 + term5
    }

    /// Calculate the log probability density function (log PDF) at a given matrix point
    ///
    /// # Arguments
    ///
    /// * `x` - The matrix at which to evaluate the log PDF (p x p)
    ///
    /// # Returns
    ///
    /// * The value of the log PDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::inverse_wishart::InverseWishart;
    ///
    /// let scale = array![[1.0, 0.0], [0.0, 1.0]];
    /// let df = 5.0;
    /// let inv_wishart = InverseWishart::new(scale, df).unwrap();
    ///
    /// let x = array![[0.2, 0.05], [0.05, 0.3]];
    /// let logpdf_value = inv_wishart.logpdf(&x);
    /// ```
    pub fn logpdf<D>(&self, x: &ArrayBase<D, Ix2>) -> f64
    where
        D: Data<Elem = f64>,
    {
        // Check if x has the right dimensions
        if x.shape()[0] != self.dim || x.shape()[1] != self.dim {
            return f64::NEG_INFINITY;
        }

        // Check if x is symmetric and positive definite
        let x_owned = x.to_owned();
        let x_chol = match compute_cholesky(&x_owned) {
            Ok(chol) => chol,
            Err(_) => return f64::NEG_INFINITY, // Not positive definite, log PDF is -∞
        };

        // Compute the log PDF
        self.logpdf_with_cholesky(x, &x_chol)
    }

    /// Generate random samples from the distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// * Vector of random matrix samples
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::inverse_wishart::InverseWishart;
    ///
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;
    /// let inv_wishart = InverseWishart::new(scale, df).unwrap();
    ///
    /// let samples = inv_wishart.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// assert_eq!(samples[0].shape(), &[2, 2]);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<Array2<f64>>> {
        // Create a corresponding Wishart distribution
        let wishart = Wishart::new(self.scale.clone(), self.df)?;

        // Generate samples from the Wishart distribution and invert them
        let wishart_samples = wishart.rvs(size)?;
        let mut inv_wishart_samples = Vec::with_capacity(size);

        for sample in wishart_samples {
            // Compute Cholesky decomposition of the Wishart sample
            let sample_chol = compute_cholesky(&sample).map_err(|_| {
                StatsError::ComputationError("Failed to compute Cholesky decomposition".to_string())
            })?;

            // Compute the inverse
            let inv_sample = compute_inverse_from_cholesky(&sample_chol).map_err(|_| {
                StatsError::ComputationError("Failed to compute matrix inverse".to_string())
            })?;

            inv_wishart_samples.push(inv_sample);
        }

        Ok(inv_wishart_samples)
    }

    /// Generate a single random sample from the distribution
    ///
    /// # Returns
    ///
    /// * A random matrix sample
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::inverse_wishart::InverseWishart;
    ///
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;
    /// let inv_wishart = InverseWishart::new(scale, df).unwrap();
    ///
    /// let sample = inv_wishart.rvs_single().unwrap();
    /// assert_eq!(sample.shape(), &[2, 2]);
    /// ```
    pub fn rvs_single(&self) -> StatsResult<Array2<f64>> {
        let samples = self.rvs(1)?;
        Ok(samples[0].clone())
    }

    /// Mean of the Inverse Wishart distribution
    ///
    /// # Returns
    ///
    /// * Mean matrix (Ψ / (ν - p - 1))
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::inverse_wishart::InverseWishart;
    ///
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;
    /// let inv_wishart = InverseWishart::new(scale, df).unwrap();
    ///
    /// if let Ok(mean) = inv_wishart.mean() {
    ///     println!("Mean: {:?}", mean);
    /// }
    /// ```
    pub fn mean(&self) -> StatsResult<Array2<f64>> {
        let p = self.dim as f64;
        let nu = self.df;

        // Mean exists only when ν > p + 1
        if nu <= p + 1.0 {
            return Err(StatsError::DomainError(
                "Mean is undefined for degrees of freedom <= dimension + 1".to_string(),
            ));
        }

        // E[X] = Ψ / (ν - p - 1)
        let mut mean = self.scale.clone();
        mean /= nu - p - 1.0;
        Ok(mean)
    }

    /// Mode of the Inverse Wishart distribution
    ///
    /// # Returns
    ///
    /// * Mode matrix (Ψ / (ν + p + 1))
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::inverse_wishart::InverseWishart;
    ///
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;
    /// let inv_wishart = InverseWishart::new(scale, df).unwrap();
    ///
    /// let mode = inv_wishart.mode();
    /// ```
    pub fn mode(&self) -> Array2<f64> {
        let p = self.dim as f64;
        let nu = self.df;

        // Mode = Ψ / (ν + p + 1)
        let mut mode = self.scale.clone();
        mode /= nu + p + 1.0;
        mode
    }
}

/// Create an Inverse Wishart distribution with the given parameters.
///
/// This is a convenience function to create an Inverse Wishart distribution with
/// the given scale matrix and degrees of freedom.
///
/// # Arguments
///
/// * `scale` - Scale matrix (p x p, symmetric positive-definite)
/// * `df` - Degrees of freedom (must be greater than dimension + 1)
///
/// # Returns
///
/// * An Inverse Wishart distribution object
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distributions::multivariate;
///
/// let scale = array![[1.0, 0.5], [0.5, 2.0]];
/// let df = 5.0;
/// let inv_wishart = multivariate::inverse_wishart(scale, df).unwrap();
/// ```
#[allow(dead_code)]
pub fn inverse_wishart<D>(scale: ArrayBase<D, Ix2>, df: f64) -> StatsResult<InverseWishart>
where
    D: Data<Elem = f64>,
{
    InverseWishart::new(scale, df)
}

/// Implementation of SampleableDistribution for InverseWishart
impl SampleableDistribution<Array2<f64>> for InverseWishart {
    fn rvs(&self, size: usize) -> StatsResult<Vec<Array2<f64>>> {
        self.rvs(size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_inverse_wishart_creation() {
        // 2x2 Inverse Wishart with identity scale
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        let df = 5.0;
        let inv_wishart = InverseWishart::new(scale.clone(), df).unwrap();

        assert_eq!(inv_wishart.dim, 2);
        assert_eq!(inv_wishart.df, df);
        assert_eq!(inv_wishart.scale, scale);

        // 3x3 Inverse Wishart with custom scale
        let scale3 = array![[2.0, 0.5, 0.3], [0.5, 1.0, 0.1], [0.3, 0.1, 1.5]];
        let df3 = 10.0;
        let inv_wishart3 = InverseWishart::new(scale3.clone(), df3).unwrap();

        assert_eq!(inv_wishart3.dim, 3);
        assert_eq!(inv_wishart3.df, df3);
        assert_eq!(inv_wishart3.scale, scale3);
    }

    #[test]
    fn test_inverse_wishart_creation_errors() {
        // Non-square scale matrix
        let non_square_scale = array![[1.0, 0.5, 0.3], [0.5, 1.0, 0.1]];
        assert!(InverseWishart::new(non_square_scale, 5.0).is_err());

        // Degrees of freedom too small
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        assert!(InverseWishart::new(scale.clone(), 3.0).is_err()); // df <= dim + 1

        // Non-positive definite scale matrix
        let non_pd_scale = array![[1.0, 2.0], [2.0, 1.0]]; // Not positive definite
        assert!(InverseWishart::new(non_pd_scale, 5.0).is_err());
    }

    #[test]
    fn test_inverse_wishart_mean() {
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let df = 5.0;
        let inv_wishart = InverseWishart::new(scale.clone(), df).unwrap();

        let mean = inv_wishart.mean().unwrap();
        let expected_mean = scale.clone() / (df - 3.0); // (df - p - 1) where p = 2

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(mean[[i, j]], expected_mean[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_inverse_wishart_mode() {
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let df = 5.0;
        let inv_wishart = InverseWishart::new(scale.clone(), df).unwrap();

        let mode = inv_wishart.mode();
        let expected_mode = scale.clone() / (df + 3.0); // (df + p + 1) where p = 2

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(mode[[i, j]], expected_mode[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_inverse_wishart_pdf() {
        // Simple identity scale case
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        let df = 5.0;
        let inv_wishart = InverseWishart::new(scale, df).unwrap();

        // PDF at identity matrix (should be positive)
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let pdf_at_id = inv_wishart.pdf(&x);
        assert!(pdf_at_id > 0.0);

        // PDF at another matrix
        let x2 = array![[0.5, 0.1], [0.1, 0.8]];
        let pdf_at_x2 = inv_wishart.pdf(&x2);
        assert!(pdf_at_x2 > 0.0);

        // Check pdf of non-positive definite matrix is zero
        let non_pd = array![[1.0, 2.0], [2.0, 1.0]];
        assert_eq!(inv_wishart.pdf(&non_pd), 0.0);
    }

    #[test]
    fn test_inverse_wishart_logpdf() {
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        let df = 5.0;
        let inv_wishart = InverseWishart::new(scale, df).unwrap();

        // Check that exp(logPDF) = PDF
        let x = array![[0.5, 0.1], [0.1, 0.8]];
        let pdf = inv_wishart.pdf(&x);
        let logpdf = inv_wishart.logpdf(&x);
        assert_relative_eq!(logpdf.exp(), pdf, epsilon = 1e-10);

        // Check logpdf of non-positive definite matrix is -∞
        let non_pd = array![[1.0, 2.0], [2.0, 1.0]];
        assert_eq!(inv_wishart.logpdf(&non_pd), f64::NEG_INFINITY);
    }

    #[test]
    fn test_inverse_wishart_rvs() {
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let df = 5.0;
        let inv_wishart = InverseWishart::new(scale.clone(), df).unwrap();

        // Generate samples
        let n_samples_ = 100;
        let samples = inv_wishart.rvs(n_samples_).unwrap();

        // Check number of samples
        assert_eq!(samples.len(), n_samples_);

        // Check dimensions of each sample
        for sample in &samples {
            assert_eq!(sample.shape(), &[2, 2]);
        }

        // Compute sample mean
        let mut sample_mean = Array2::<f64>::zeros((2, 2));
        for sample in &samples {
            sample_mean += sample;
        }
        sample_mean /= n_samples_ as f64;

        // Expected mean is Ψ/(ν-p-1) where p=2
        let expected_mean = scale.clone() / (df - 3.0);

        // Check that sample mean is close to expected mean (allowing for sampling variation)
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    sample_mean[[i, j]],
                    expected_mean[[i, j]],
                    epsilon = 1.0, // Larger epsilon for random sampling with higher variance
                    max_relative = 0.8  // Higher tolerance for relative error
                );
            }
        }
    }

    #[test]
    fn test_inverse_wishart_rvs_single() {
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let df = 5.0;
        let inv_wishart = InverseWishart::new(scale, df).unwrap();

        let sample = inv_wishart.rvs_single().unwrap();

        // Check dimensions
        assert_eq!(sample.shape(), &[2, 2]);

        // Check that sample is symmetric
        assert_relative_eq!(sample[[0, 1]], sample[[1, 0]], epsilon = 1e-10);

        // Check that sample is positive definite (can compute Cholesky)
        assert!(compute_cholesky(&sample).is_ok());
    }
}
