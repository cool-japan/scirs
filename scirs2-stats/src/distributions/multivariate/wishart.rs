//! Wishart distribution functions
//!
//! This module provides functionality for the Wishart distribution.

use crate::error::{StatsError, StatsResult};
use crate::sampling::SampleableDistribution;
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{ChiSquared, Distribution, Normal as RandNormal};
use std::fmt::Debug;

// Import helper functions from the multivariate module
use super::normal::{compute_cholesky, compute_inverse_from_cholesky};

/// Implementation of the natural logarithm of the gamma function
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

/// Calculates the multivariate gamma function (natural log)
///
/// This is defined as:
/// ln Γₚ(n/2) = ln (π^(p(p-1)/4) ∏ᵢ₌₁ᵖ Γ((n+1-i)/2))
///
/// where p is the dimension and n is the degrees of freedom
#[allow(dead_code)]
pub fn lmultigamma(n: f64, p: usize) -> f64 {
    // Calculate ln(π^(p(p-1)/4))
    let pi = std::f64::consts::PI;
    let term1 = (p * (p - 1)) as f64 / 4.0 * pi.ln();

    // Calculate ln(∏ᵢ₌₁ᵖ Γ((n+1-i)/2))
    let mut term2 = 0.0;
    for i in 1..=p {
        let arg = (n + 1.0 - i as f64) / 2.0;
        term2 += lgamma(arg);
    }

    term1 + term2
}

/// Wishart distribution structure
#[derive(Debug, Clone)]
pub struct Wishart {
    /// Scale matrix
    pub scale: Array2<f64>,
    /// Degrees of freedom
    pub df: f64,
    /// Dimension of the distribution (p x p)
    pub dim: usize,
    /// Cholesky decomposition of the scale matrix
    scale_chol: Array2<f64>,
    /// Determinant of the scale matrix
    scale_det: f64,
}

impl Wishart {
    /// Create a new Wishart distribution with given parameters
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale matrix (p x p, symmetric positive-definite)
    /// * `df` - Degrees of freedom (must be greater than or equal to the dimension)
    ///
    /// # Returns
    ///
    /// * A new Wishart distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::wishart::Wishart;
    ///
    /// // Create a 2D Wishart distribution with 5 degrees of freedom
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;
    /// let wishart = Wishart::new(scale, df).unwrap();
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

        // Check the degrees of freedom constraint: df >= dim
        if df < dim as f64 {
            return Err(StatsError::DomainError(format!(
                "Degrees of freedom ({}) must be greater than or equal to dimension ({})",
                df, dim
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

        Ok(Wishart {
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
    /// use scirs2_stats::distributions::multivariate::wishart::Wishart;
    ///
    /// let scale = array![[1.0, 0.0], [0.0, 1.0]];
    /// let df = 5.0;
    /// let wishart = Wishart::new(scale, df).unwrap();
    ///
    /// let x = array![[5.0, 1.0], [1.0, 5.0]];
    /// let pdf_value = wishart.pdf(&x);
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
        // Calculate determinant of x
        let mut x_det = 1.0;
        for i in 0..self.dim {
            x_det *= xchol[[i, i]];
        }
        x_det = x_det * x_det; // Square it since det(X) = det(L)^2

        // Calculate trace(Σ^-1 X)
        let scale_inv = compute_inverse_from_cholesky(&self.scale_chol)
            .expect("Failed to compute matrix inverse");

        let mut trace = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                trace += scale_inv[[i, j]] * x[[j, i]];
            }
        }

        // Calculate log PDF
        // ln(PDF) = -0.5 * [tr(Σ^-1 X) + n ln|Σ| + p ln 2 + 2 ln Γₚ(n/2)] + 0.5 * (n - p - 1) ln|X|
        let p = self.dim as f64;
        let n = self.df;

        let term1 = -0.5 * trace;
        let term2 = -0.5 * n * self.scale_det.ln();
        let term3 = -0.5 * p * (2.0f64).ln();
        let term4 = -lmultigamma(n, self.dim);
        let term5 = 0.5 * (n - p - 1.0) * x_det.ln();

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
    /// use scirs2_stats::distributions::multivariate::wishart::Wishart;
    ///
    /// let scale = array![[1.0, 0.0], [0.0, 1.0]];
    /// let df = 5.0;
    /// let wishart = Wishart::new(scale, df).unwrap();
    ///
    /// let x = array![[5.0, 1.0], [1.0, 5.0]];
    /// let logpdf_value = wishart.logpdf(&x);
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
    /// use scirs2_stats::distributions::multivariate::wishart::Wishart;
    ///
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;
    /// let wishart = Wishart::new(scale, df).unwrap();
    ///
    /// let samples = wishart.rvs(10).unwrap();
    /// assert_eq!(samples.len(), 10);
    /// assert_eq!(samples[0].shape(), &[2, 2]);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<Array2<f64>>> {
        let mut rng = thread_rng();
        let normal_dist = RandNormal::new(0.0, 1.0).unwrap();
        let mut samples = Vec::with_capacity(size);

        for _ in 0..size {
            // For integer degrees of freedom, we use the sum of outer products method
            if self.df.fract() == 0.0 {
                let n = self.df as usize;
                let mut x = Array2::<f64>::zeros((self.dim, self.dim));

                // Generate n independent vectors
                for _ in 0..n {
                    // Generate standard normal vector
                    let mut z = Array1::<f64>::zeros(self.dim);
                    for j in 0..self.dim {
                        z[j] = normal_dist.sample(&mut rng);
                    }

                    // Transform using Cholesky decomposition
                    let az = self.scale_chol.dot(&z);

                    // Add outer product to X
                    for i in 0..self.dim {
                        for j in 0..self.dim {
                            x[[i, j]] += az[i] * az[j];
                        }
                    }
                }

                samples.push(x);
            } else {
                // For non-integer df, we use Bartlett's decomposition
                // Generate lower triangular matrix A
                let mut a = Array2::<f64>::zeros((self.dim, self.dim));

                // Diagonal elements from chi-square distributions
                for i in 0..self.dim {
                    let df_i = self.df - (i as f64);
                    let chi2_dist = ChiSquared::new(df_i).map_err(|_| {
                        StatsError::ComputationError(
                            "Failed to create chi-square distribution".to_string(),
                        )
                    })?;

                    // sqrt because we need to sample from sqrt(χ²) here
                    a[[i, i]] = chi2_dist.sample(&mut rng).sqrt();
                }

                // Off-diagonal elements from standard normal distribution
                for i in 0..self.dim {
                    for j in 0..i {
                        a[[i, j]] = normal_dist.sample(&mut rng);
                    }
                }

                // Compute B = L·A where L is the Cholesky decomposition of scale matrix
                let b = self.scale_chol.dot(&a);

                // Compute X = B·B'
                let mut x = Array2::<f64>::zeros((self.dim, self.dim));
                for i in 0..self.dim {
                    for j in 0..=i {
                        // Only compute lower triangular part
                        let mut sum = 0.0;
                        for k in 0..self.dim {
                            sum += b[[i, k]] * b[[j, k]];
                        }
                        x[[i, j]] = sum;
                        if i != j {
                            x[[j, i]] = sum; // Symmetric
                        }
                    }
                }

                samples.push(x);
            }
        }

        Ok(samples)
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
    /// use scirs2_stats::distributions::multivariate::wishart::Wishart;
    ///
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;
    /// let wishart = Wishart::new(scale, df).unwrap();
    ///
    /// let sample = wishart.rvs_single().unwrap();
    /// assert_eq!(sample.shape(), &[2, 2]);
    /// ```
    pub fn rvs_single(&self) -> StatsResult<Array2<f64>> {
        let samples = self.rvs(1)?;
        Ok(samples[0].clone())
    }

    /// Mean of the Wishart distribution
    ///
    /// # Returns
    ///
    /// * Mean matrix (ν × Σ)
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::wishart::Wishart;
    ///
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;
    /// let wishart = Wishart::new(scale, df).unwrap();
    ///
    /// let mean = wishart.mean();
    /// ```
    pub fn mean(&self) -> Array2<f64> {
        // E[X] = ν × Σ
        let mut mean = self.scale.clone();
        mean *= self.df;
        mean
    }

    /// Mode of the Wishart distribution (only defined for ν ≥ p + 1)
    ///
    /// # Returns
    ///
    /// * Mode matrix ((ν - p - 1) × Σ) if ν ≥ p + 1, otherwise None
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_stats::distributions::multivariate::wishart::Wishart;
    ///
    /// let scale = array![[1.0, 0.5], [0.5, 2.0]];
    /// let df = 5.0;  // For 2x2 matrix, mode exists when df ≥ 3
    /// let wishart = Wishart::new(scale, df).unwrap();
    ///
    /// if let Some(mode) = wishart.mode() {
    ///     println!("Mode exists: {:?}", mode);
    /// }
    /// ```
    pub fn mode(&self) -> Option<Array2<f64>> {
        let p = self.dim as f64;
        if self.df < p + 1.0 {
            None // Mode doesn't exist
        } else {
            let mut mode = self.scale.clone();
            mode *= self.df - p - 1.0;
            Some(mode)
        }
    }
}

/// Create a Wishart distribution with the given parameters.
///
/// This is a convenience function to create a Wishart distribution with
/// the given scale matrix and degrees of freedom.
///
/// # Arguments
///
/// * `scale` - Scale matrix (p x p, symmetric positive-definite)
/// * `df` - Degrees of freedom (must be greater than or equal to the dimension)
///
/// # Returns
///
/// * A Wishart distribution object
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::distributions::multivariate;
///
/// let scale = array![[1.0, 0.5], [0.5, 2.0]];
/// let df = 5.0;
/// let wishart = multivariate::wishart(scale, df).unwrap();
/// ```
#[allow(dead_code)]
pub fn wishart<D>(scale: ArrayBase<D, Ix2>, df: f64) -> StatsResult<Wishart>
where
    D: Data<Elem = f64>,
{
    Wishart::new(scale, df)
}

/// Implementation of SampleableDistribution for Wishart
impl SampleableDistribution<Array2<f64>> for Wishart {
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
    fn test_wishart_creation() {
        // 2x2 Wishart with identity scale
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        let df = 5.0;
        let wishart = Wishart::new(scale.clone(), df).unwrap();

        assert_eq!(wishart.dim, 2);
        assert_eq!(wishart.df, df);
        assert_eq!(wishart.scale, scale);

        // 3x3 Wishart with custom scale
        let scale3 = array![[2.0, 0.5, 0.3], [0.5, 1.0, 0.1], [0.3, 0.1, 1.5]];
        let df3 = 10.0;
        let wishart3 = Wishart::new(scale3.clone(), df3).unwrap();

        assert_eq!(wishart3.dim, 3);
        assert_eq!(wishart3.df, df3);
        assert_eq!(wishart3.scale, scale3);
    }

    #[test]
    fn test_wishart_creation_errors() {
        // Non-square scale matrix
        let non_square_scale = array![[1.0, 0.5, 0.3], [0.5, 1.0, 0.1]];
        assert!(Wishart::new(non_square_scale, 5.0).is_err());

        // Degrees of freedom too small
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        assert!(Wishart::new(scale.clone(), 1.0).is_err()); // df < dim

        // Non-positive definite scale matrix
        let non_pd_scale = array![[1.0, 2.0], [2.0, 1.0]]; // Not positive definite
        assert!(Wishart::new(non_pd_scale, 5.0).is_err());
    }

    #[test]
    fn test_wishart_mean() {
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let df = 5.0;
        let wishart = Wishart::new(scale.clone(), df).unwrap();

        let mean = wishart.mean();
        let expected_mean = scale * df;

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(mean[[i, j]], expected_mean[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_wishart_mode() {
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let df = 5.0; // df = 5 > p + 1 = 3, so mode exists
        let wishart = Wishart::new(scale.clone(), df).unwrap();

        let mode = wishart.mode().unwrap(); // Mode should exist
        let expected_mode = scale.clone() * (df - 3.0); // (ν - p - 1) × Σ where p = 2

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(mode[[i, j]], expected_mode[[i, j]], epsilon = 1e-10);
            }
        }

        // Test case when mode doesn't exist
        let wishart2 = Wishart::new(scale, 2.5).unwrap(); // df = 2.5 < p + 1 = 3
        assert!(wishart2.mode().is_none()); // Mode should not exist
    }

    #[test]
    fn test_wishart_pdf() {
        // Simple identity scale case
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        let df = 5.0;
        let wishart = Wishart::new(scale, df).unwrap();

        // PDF at identity matrix
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let pdf_at_id = wishart.pdf(&x);
        assert!(pdf_at_id > 0.0);

        // PDF at another matrix
        let x2 = array![[2.0, 0.5], [0.5, 3.0]];
        let pdf_at_x2 = wishart.pdf(&x2);
        assert!(pdf_at_x2 > 0.0);

        // Check pdf of non-positive definite matrix is zero
        let non_pd = array![[1.0, 2.0], [2.0, 1.0]];
        assert_eq!(wishart.pdf(&non_pd), 0.0);
    }

    #[test]
    fn test_wishart_logpdf() {
        let scale = array![[1.0, 0.0], [0.0, 1.0]];
        let df = 5.0;
        let wishart = Wishart::new(scale, df).unwrap();

        // Check that exp(logPDF) = PDF
        let x = array![[2.0, 0.5], [0.5, 3.0]];
        let pdf = wishart.pdf(&x);
        let logpdf = wishart.logpdf(&x);
        assert_relative_eq!(logpdf.exp(), pdf, epsilon = 1e-10);

        // Check logpdf of non-positive definite matrix is -∞
        let non_pd = array![[1.0, 2.0], [2.0, 1.0]];
        assert_eq!(wishart.logpdf(&non_pd), f64::NEG_INFINITY);
    }

    #[test]
    fn test_wishart_rvs() {
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let df = 5.0;
        let wishart = Wishart::new(scale.clone(), df).unwrap();

        // Generate samples
        let n_samples_ = 100;
        let samples = wishart.rvs(n_samples_).unwrap();

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

        // Expected mean is ν × Σ
        let expected_mean = scale * df;

        // Check that sample mean is close to expected mean (allowing for sampling variation)
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    sample_mean[[i, j]],
                    expected_mean[[i, j]],
                    epsilon = 0.8, // Larger epsilon for random sampling
                    max_relative = 0.3
                );
            }
        }
    }

    #[test]
    fn test_wishart_rvs_single() {
        let scale = array![[1.0, 0.5], [0.5, 2.0]];
        let df = 5.0;
        let wishart = Wishart::new(scale, df).unwrap();

        let sample = wishart.rvs_single().unwrap();

        // Check dimensions
        assert_eq!(sample.shape(), &[2, 2]);

        // Check that sample is symmetric
        assert_relative_eq!(sample[[0, 1]], sample[[1, 0]], epsilon = 1e-10);

        // Check that sample is positive definite (can compute Cholesky)
        assert!(compute_cholesky(&sample).is_ok());
    }
}
