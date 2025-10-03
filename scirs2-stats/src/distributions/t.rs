//! Student's t distribution functions
//!
//! This module provides functionality for the Student's t distribution.

use crate::error::{StatsError, StatsResult};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::{Distribution, StudentT as RandStudentT};
use std::fmt::Debug;

/// Student's t distribution structure
pub struct StudentT<F: Float> {
    /// Degrees of freedom
    pub df: F,
    /// Location parameter
    pub loc: F,
    /// Scale parameter
    pub scale: F,
    /// Random number generator for this distribution
    rand_distr: RandStudentT<f64>,
}

impl<F: Float + NumCast + Debug + std::fmt::Display> StudentT<F> {
    /// Create a new Student's t distribution with given degrees of freedom, location, and scale
    ///
    /// # Arguments
    ///
    /// * `df` - Degrees of freedom
    /// * `loc` - Location parameter
    /// * `scale` - Scale parameter
    ///
    /// # Returns
    ///
    /// * A new Student's t distribution instance
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::t::StudentT;
    ///
    /// let t_dist = StudentT::new(10.0f64, 0.0, 1.0).unwrap();
    /// ```
    pub fn new(df: F, loc: F, scale: F) -> StatsResult<Self> {
        if df <= F::zero() {
            return Err(StatsError::DomainError(
                "Degrees of freedom must be positive".to_string(),
            ));
        }
        if scale <= F::zero() {
            return Err(StatsError::DomainError(
                "Scale parameter must be positive".to_string(),
            ));
        }

        // Convert to f64 for rand_distr
        let df_f64 = <f64 as NumCast>::from(df).unwrap();
        
        match RandStudentT::new(df_f64) {
            Ok(rand_distr) => Ok(StudentT {
                df,
                loc,
                scale,
                rand_distr,
            }),
            Err(_) => Err(StatsError::ComputationError(
                "Failed to create Student's t distribution".to_string(),
            )),
        }
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
    /// use scirs2_stats::distributions::t::StudentT;
    ///
    /// let t_dist = StudentT::new(10.0f64, 0.0, 1.0).unwrap();
    /// let pdf_at_zero = t_dist.pdf(0.0);
    /// assert!((pdf_at_zero - 0.3940886).abs() < 1e-6);
    /// ```
    pub fn pdf(&self, x: F) -> F {
        // Standardize the value
        let z = (x - self.loc) / self.scale;
        
        // Calculate PDF for standard t-distribution
        let pi = F::from(std::f64::consts::PI).unwrap();
        let one = F::one();
        let two = F::from(2.0).unwrap();
        
        // Gamma function approximation
        let df_half = self.df / two;
        let df_plus_half = (self.df + one) / two;
        
        // Use the PDF formula for Student's t distribution
        // f(t) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + t²/ν)^(-(ν+1)/2)
        
        // For numerical stability
        let gamma_ratio = gamma_function(df_plus_half) / gamma_function(df_half);
        let denominator = (self.df * pi).sqrt();
        let term = one + (z * z) / self.df;
        let exponent = -(self.df + one) / two;
        
        // PDF for standardized t-distribution
        let std_pdf = gamma_ratio / denominator * term.powf(exponent);
        
        // Scale the PDF
        std_pdf / self.scale
    }

    /// Calculate the cumulative distribution function (CDF) at a given point
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate the CDF
    ///
    /// # Returns
    ///
    /// * The value of the CDF at the given point
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::t::StudentT;
    ///
    /// let t_dist = StudentT::new(10.0f64, 0.0, 1.0).unwrap();
    /// let cdf_at_zero = t_dist.cdf(0.0);
    /// assert!((cdf_at_zero - 0.5).abs() < 1e-10);
    /// ```
    pub fn cdf(&self, x: F) -> F {
        // Standardize the value
        let z = (x - self.loc) / self.scale;
        
        // Special case at 0 for symmetry
        if z.is_zero() {
            return F::from(0.5).unwrap();
        }
        
        // Calculate CDF for t-distribution
        let half = F::from(0.5).unwrap();
        
        if z > F::zero() {
            // For positive values
            half + half * incomplete_beta(
                half * self.df,
                half,
                self.df / (self.df + z * z),
            )
        } else {
            // For negative values
            half * incomplete_beta(
                half * self.df,
                half,
                self.df / (self.df + z * z),
            )
        }
    }

    /// Inverse of the cumulative distribution function (quantile function)
    ///
    /// # Arguments
    ///
    /// * `p` - Probability value (between 0 and 1)
    ///
    /// # Returns
    ///
    /// * The value x such that CDF(x) = p
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::t::StudentT;
    ///
    /// let t_dist = StudentT::new(10.0f64, 0.0, 1.0).unwrap();
    /// let x = t_dist.ppf(0.975).unwrap();
    /// assert!((x - 2.228).abs() < 1e-3);
    /// ```
    pub fn ppf(&self, p: F) -> StatsResult<F> {
        if p < F::zero() || p > F::one() {
            return Err(StatsError::DomainError(
                "Probability must be between 0 and 1".to_string(),
            ));
        }
        
        // Special cases
        if p == F::zero() {
            return Ok(F::neg_infinity());
        }
        if p == F::one() {
            return Ok(F::infinity());
        }
        if p == F::from(0.5).unwrap() {
            return Ok(self.loc); // Median is at loc for symmetric distributions
        }
        
        // Numerical approximation for quantile
        // Use bisection method for simplicity
        let mut lower = F::from(-100.0).unwrap(); // Reasonable lower bound
        let mut upper = F::from(100.0).unwrap();  // Reasonable upper bound
        let tolerance = F::from(1e-10).unwrap();
        let max_iterations = 100;
        
        for _ in 0..max_iterations {
            let mid = (lower + upper) / F::from(2.0).unwrap();
            let cdf_mid = self.cdf(mid);
            
            if (cdf_mid - p).abs() < tolerance {
                // Found a sufficiently accurate solution
                return Ok(mid);
            }
            
            if cdf_mid < p {
                lower = mid;
            } else {
                upper = mid;
            }
        }
        
        // If we reach here, we didn't converge to the desired accuracy
        // Return the midpoint as an approximation
        Ok(self.loc + self.scale * (lower + upper) / F::from(2.0).unwrap())
    }

    /// Generate random samples from the distribution
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// * Vector of random samples
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_stats::distributions::t::StudentT;
    ///
    /// let t_dist = StudentT::new(10.0f64, 0.0, 1.0).unwrap();
    /// let samples = t_dist.rvs(1000).unwrap();
    /// assert_eq!(samples.len(), 1000);
    /// ```
    pub fn rvs(&self, size: usize) -> StatsResult<Vec<F>> {
        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(size);
        
        for _ in 0..size {
            let sample = self.rand_distr.sample(&mut rng);
            // Apply location and scale transformations
            samples.push(F::from(sample).unwrap() * self.scale + self.loc);
        }
        
        Ok(samples)
    }
}

/// Gamma function approximation
///
/// This is a simple approximation of the gamma function using Stirling's formula.
#[allow(dead_code)]
fn gamma_function<F: Float>(x: F) -> F {
    if x <= F::zero() {
        // Not defined for non-positive values
        return F::nan();
    }

    let one = F::one();
    
    // For integers and half-integers, use exact values
    if x == one {
        return one; // Gamma(1) = 1
    }
    
    if x == F::from(0.5).unwrap() {
        return F::from(std::f64::consts::PI).unwrap().sqrt(); // Gamma(0.5) = sqrt(π)
    }
    
    // For x = n + 0.5 where n is a non-negative integer
    let two = F::from(2.0).unwrap();
    if (x * two).fract() < F::epsilon() && x > F::zero() {
        let n = (x - F::from(0.5).unwrap()).to_f64().unwrap() as i32;
        if n >= 0 {
            let sqrt_pi = F::from(std::f64::consts::PI).unwrap().sqrt();
            let mut result = sqrt_pi;
            let mut factorial = one;
            
            for i in 0..n {
                factorial = factorial * F::from(i as f64 + 0.5).unwrap();
            }
            
            return result * factorial;
        }
    }
    
    // Use Stirling's approximation for other values
    let e = F::from(std::f64::consts::E).unwrap();
    let pi = F::from(std::f64::consts::PI).unwrap();
    
    // Stirling's formula: Gamma(x) ≈ sqrt(2π/x) * (x/e)^x
    let term1 = (two * pi / x).sqrt();
    let term2 = (x / e).powf(x);
    
    term1 * term2
}

/// Incomplete beta function approximation
///
/// This is a simple approximation of the regularized incomplete beta function.
#[allow(dead_code)]
fn incomplete_beta<F: Float>(a: F, b: F, x: F) -> F {
    if x <= F::zero() {
        return F::zero();
    }
    if x >= F::one() {
        return F::one();
    }
    
    // Continued fraction approximation for incomplete beta
    let max_iterations = 100;
    let epsilon = F::from(1e-10).unwrap();
    
    let one = F::one();
    let two = F::from(2.0).unwrap();
    
    // Initialize variables for continued fraction
    let mut h = one;
    let mut am = one;
    let mut bm = one;
    let mut az = one;
    let mut qab = a + b;
    let mut qap = a + one;
    let mut qam = a - one;
    let mut bz = one - qab * x / qap;
    
    // Iterate to convergence
    for m in 1..max_iterations {
        let m_f = F::from(m as f64).unwrap();
        let two_m = F::from(2 * m as f64).unwrap();
        
        // Even m terms
        let tem = m_f * b;
        let d = m_f * (b - m_f) * x / ((qam + two_m) * (a + two_m));
        let ap = az + d * am;
        let bp = bz + d * bm;
        
        // Odd m terms
        let d = -(a + m_f) * (qab + m_f) * x / ((a + two_m) * (qap + two_m));
        let app = ap + d * az;
        let bpp = bp + d * bz;
        
        // Update variables for next iteration
        am = ap;
        bm = bp;
        az = app;
        bz = bpp;
        
        // Check for convergence
        if bz.abs() > F::epsilon() {
            let azbz = az / bz;
            if (azbz - h).abs() / azbz < epsilon {
                break;
            }
            h = azbz;
        }
    }
    
    // Calculate the result
    let beta_term = gamma_function(a) * gamma_function(b) / gamma_function(a + b);
    let x_pow_a = x.powf(a);
    let one_minus_x_pow_b = (one - x).powf(b);
    
    x_pow_a * one_minus_x_pow_b * h / (a * beta_term)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_student_t_creation() {
        // Standard t-distribution
        let t = StudentT::new(10.0, 0.0, 1.0).unwrap();
        assert_eq!(t.df, 10.0);
        assert_eq!(t.loc, 0.0);
        assert_eq!(t.scale, 1.0);
        
        // Custom t-distribution
        let custom = StudentT::new(5.0, 2.0, 3.0).unwrap();
        assert_eq!(custom.df, 5.0);
        assert_eq!(custom.loc, 2.0);
        assert_eq!(custom.scale, 3.0);
        
        // Error cases
        assert!(StudentT::<f64>::new(0.0, 0.0, 1.0).is_err());
        assert!(StudentT::<f64>::new(-1.0, 0.0, 1.0).is_err());
        assert!(StudentT::<f64>::new(1.0, 0.0, 0.0).is_err());
        assert!(StudentT::<f64>::new(1.0, 0.0, -1.0).is_err());
    }
    
    #[test]
    fn test_student_t_pdf() {
        // PDF values for standard t-distribution
        let t = StudentT::new(10.0, 0.0, 1.0).unwrap();
        
        // PDF at x = 0
        let pdf_at_zero = t.pdf(0.0);
        assert_relative_eq!(pdf_at_zero, 0.3940886, epsilon = 1e-6);
        
        // PDF at x = 1
        let pdf_at_one = t.pdf(1.0);
        assert_relative_eq!(pdf_at_one, 0.2484, epsilon = 1e-4);
        
        // PDF at x = -1
        let pdf_at_neg_one = t.pdf(-1.0);
        assert_relative_eq!(pdf_at_neg_one, 0.2484, epsilon = 1e-4);
        
        // Test with location and scale
        let t_loc_scale = StudentT::new(10.0, 1.0, 2.0).unwrap();
        let pdf_at_one_loc_scale = t_loc_scale.pdf(1.0);
        // PDF at the location parameter should equal the PDF at 0 for standard t
        // divided by the scale parameter
        assert_relative_eq!(pdf_at_one_loc_scale, 0.3940886 / 2.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_student_t_cdf() {
        // CDF values for standard t-distribution
        let t = StudentT::new(10.0, 0.0, 1.0).unwrap();
        
        // CDF at x = 0
        let cdf_at_zero = t.cdf(0.0);
        assert_relative_eq!(cdf_at_zero, 0.5, epsilon = 1e-10);
        
        // CDF at x = 1
        let cdf_at_one = t.cdf(1.0);
        assert_relative_eq!(cdf_at_one, 0.8296, epsilon = 1e-4);
        
        // CDF at x = -1
        let cdf_at_neg_one = t.cdf(-1.0);
        assert_relative_eq!(cdf_at_neg_one, 0.1704, epsilon = 1e-4);
        
        // Test with location and scale
        let t_loc_scale = StudentT::new(10.0, 1.0, 2.0).unwrap();
        let cdf_at_one_loc_scale = t_loc_scale.cdf(1.0);
        // CDF at the location parameter should equal 0.5
        assert_relative_eq!(cdf_at_one_loc_scale, 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_student_t_ppf() {
        // Quantile values for standard t-distribution
        let t = StudentT::new(10.0, 0.0, 1.0).unwrap();
        
        // Median (50th percentile)
        let median = t.ppf(0.5).unwrap();
        assert_relative_eq!(median, 0.0, epsilon = 1e-5);
        
        // 97.5th percentile (often used for confidence intervals)
        let p975 = t.ppf(0.975).unwrap();
        assert_relative_eq!(p975, 2.228, epsilon = 1e-3);
        
        // 2.5th percentile
        let p025 = t.ppf(0.025).unwrap();
        assert_relative_eq!(p025, -2.228, epsilon = 1e-3);
        
        // Error cases
        assert!(t.ppf(-0.1).is_err());
        assert!(t.ppf(1.1).is_err());
        
        // Test with location and scale
        let t_loc_scale = StudentT::new(10.0, 1.0, 2.0).unwrap();
        let median_loc_scale = t_loc_scale.ppf(0.5).unwrap();
        assert_relative_eq!(median_loc_scale, 1.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_student_t_rvs() {
        let t = StudentT::new(10.0, 0.0, 1.0).unwrap();
        
        // Generate samples
        let samples = t.rvs(1000).unwrap();
        
        // Check the number of samples
        assert_eq!(samples.len(), 1000);
        
        // Basic statistical checks
        let sum: f64 = samples.iter().sum();
        let mean = sum / 1000.0;
        
        // Mean should be close to 0 (within reason for random samples)
        assert!(mean.abs() < 0.1);
        
        // Test with location and scale
        let t_loc_scale = StudentT::new(10.0, 5.0, 2.0).unwrap();
        let samples_loc_scale = t_loc_scale.rvs(1000).unwrap();
        let sum_loc_scale: f64 = samples_loc_scale.iter().sum();
        let mean_loc_scale = sum_loc_scale / 1000.0;
        
        // Mean should be close to the location parameter (within reason)
        assert!((mean_loc_scale - 5.0).abs() < 0.2);
    }
    
    #[test]
    fn test_gamma_function() {
        // Test known values of the gamma function
        assert_relative_eq!(gamma_function(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_function(2.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma_function(3.0), 2.0, epsilon = 1e-1);
        assert_relative_eq!(gamma_function(4.0), 6.0, epsilon = 1e-1);
        
        // Test gamma(0.5) = sqrt(π)
        let sqrt_pi = std::f64::consts::PI.sqrt();
        assert_relative_eq!(gamma_function(0.5), sqrt_pi, epsilon = 1e-10);
    }
}
