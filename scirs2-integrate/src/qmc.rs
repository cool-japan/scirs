//! Quasi-Monte Carlo integration
//!
//! This module provides Quasi-Monte Carlo integration methods for numerical integration
//! of multidimensional functions.
//!
//! QMC integration uses low-discrepancy sequences to generate evaluation points,
//! offering better convergence rates than traditional Monte Carlo methods for many
//! integration problems, especially in higher dimensions.

use ndarray::{s, Array1, Array2, ArrayView1};
use num_traits::{Float, FromPrimitive};
use rand::random;
use std::fmt;

use crate::error::{IntegrateError, IntegrateResult};

/// Result type for QMC integration
#[derive(Clone, Debug)]
pub struct QMCQuadResult<T> {
    /// The estimate of the integral
    pub integral: T,
    /// The error estimate
    pub standard_error: T,
}

impl<T: fmt::Display> fmt::Display for QMCQuadResult<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QMCQuadResult(integral={}, standard_error={})",
            self.integral, self.standard_error
        )
    }
}

/// Trait for quasi-random number generators
pub trait QRNGEngine {
    /// Generate n points in d dimensions in the unit hypercube [0,1]^d
    fn random(&mut self, n: usize) -> Array2<f64>;

    /// Dimensionality of the generator
    fn dim(&self) -> usize;

    /// Create a new instance from a seed
    fn new_from_seed(&self, seed: u64) -> Box<dyn QRNGEngine>;
}

/// Simple pseudorandom number generator for benchmarking
pub struct RandomGenerator {
    dim: usize,
    // We don't use the seed for random generation, just for new_from_seed
    _seed: u64,
}

impl RandomGenerator {
    /// Create a new random number generator
    pub fn new(dim: usize, seed: Option<u64>) -> Self {
        let _seed = seed.unwrap_or_else(random::<u64>);

        Self { dim, _seed }
    }
}

impl QRNGEngine for RandomGenerator {
    fn random(&mut self, n: usize) -> Array2<f64> {
        let mut result = Array2::zeros((n, self.dim));

        for i in 0..n {
            for j in 0..self.dim {
                result[[i, j]] = random::<f64>();
            }
        }

        result
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn new_from_seed(&self, seed: u64) -> Box<dyn QRNGEngine> {
        Box::new(Self::new(self.dim, Some(seed)))
    }
}

/// Sobol sequence generator
pub struct Sobol {
    dim: usize,
    seed: u64,
    curr_index: usize,
}

impl Sobol {
    /// Create a new Sobol sequence generator
    pub fn new(dim: usize, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(random::<u64>);

        Self {
            dim,
            seed,
            curr_index: 0,
        }
    }

    /// Generate a Sobol sequence point
    ///
    /// This is a simple implementation that doesn't use the sobol crate,
    /// but provides a reasonable approximation of a Sobol sequence for
    /// demonstration purposes.
    fn generate_point(&mut self) -> Vec<f64> {
        // Simple Van der Corput sequence as basis
        let mut result = vec![0.0; self.dim];

        // Basic bit-reversal sequence for each dimension
        for (d, res) in result.iter_mut().enumerate().take(self.dim) {
            let mut i = self.curr_index;
            let mut f = 1.0;

            // Use different base for each dimension
            // Prime number + offset based on dimension and seed
            let base = (d as f64 * 2.0 + 3.0 + (self.seed % 11) as f64) as usize;

            while i > 0 {
                f /= base as f64;
                *res += f * (i % base) as f64;
                i /= base;
            }
        }

        self.curr_index += 1;
        result
    }
}

impl QRNGEngine for Sobol {
    fn random(&mut self, n: usize) -> Array2<f64> {
        let mut result = Array2::zeros((n, self.dim));

        for i in 0..n {
            let point = self.generate_point();
            for j in 0..self.dim {
                result[[i, j]] = point[j];
            }
        }

        result
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn new_from_seed(&self, seed: u64) -> Box<dyn QRNGEngine> {
        Box::new(Self::new(self.dim, Some(seed)))
    }
}

/// Halton sequence generator
pub struct Halton {
    dim: usize,
    seed: u64,
    curr_index: usize,
}

impl Halton {
    /// Create a new Halton sequence generator
    pub fn new(dim: usize, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(random::<u64>);

        Self {
            dim,
            seed,
            curr_index: 0,
        }
    }

    /// Generate a Halton sequence point
    fn generate_point(&mut self) -> Vec<f64> {
        let first_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
        let mut result = vec![0.0; self.dim];

        // Scrambling offset using seed
        let offset = (self.seed % 1000) as usize;

        for d in 0..self.dim {
            let base = if d < first_primes.len() {
                first_primes[d]
            } else {
                // For dimensions beyond stored primes, use a simple formula
                first_primes[d % first_primes.len()] + d
            };

            let mut f = 1.0;
            let mut r = 0.0;
            let mut n = self.curr_index + offset;

            while n > 0 {
                f /= base as f64;
                r += f * (n % base) as f64;
                n /= base;
            }

            result[d] = r;
        }

        self.curr_index += 1;
        result
    }
}

impl QRNGEngine for Halton {
    fn random(&mut self, n: usize) -> Array2<f64> {
        let mut result = Array2::zeros((n, self.dim));

        for i in 0..n {
            let point = self.generate_point();
            for j in 0..self.dim {
                result[[i, j]] = point[j];
            }
        }

        result
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn new_from_seed(&self, seed: u64) -> Box<dyn QRNGEngine> {
        Box::new(Self::new(self.dim, Some(seed)))
    }
}

/// Scale samples from unit hypercube to the integration range [a, b]
pub fn scale<T: Float + FromPrimitive>(
    sample: &Array2<T>,
    a: &Array1<T>,
    b: &Array1<T>,
) -> Array2<T> {
    let mut scaled = Array2::<T>::zeros(sample.raw_dim());
    let dim = sample.shape()[1];

    for i in 0..sample.shape()[0] {
        for j in 0..dim {
            scaled[[i, j]] = a[j] + (b[j] - a[j]) * sample[[i, j]];
        }
    }

    scaled
}

/// Compute an integral in N dimensions using Quasi-Monte Carlo quadrature.
///
/// # Parameters
///
/// * `func` - Function to integrate that takes a point in n-dimensional space and
///   returns a scalar value.
/// * `a` - Lower integration bounds, array of length equal to number of dimensions.
/// * `b` - Upper integration bounds, array of length equal to number of dimensions.
/// * `n_estimates` - Number of statistically independent samples to use (default: 8).
/// * `n_points` - Number of QMC points per sample (default: 1024).
/// * `qrng` - QRNGEngine to use for sampling points. If None, a Halton sequence is used.
/// * `log` - If true, treat func as returning the log of the integrand and return log
///   of the result.
///
/// # Returns
///
/// * `QMCQuadResult` containing the integral estimate and standard error.
///
/// # Examples
///
/// ```
/// use ndarray::{Array1, ArrayView1};
/// use scirs2_integrate::qmc::{qmc_quad, Halton};
///
/// let f = |x: ArrayView1<f64>| x[0].powi(2) * x[1].exp();
/// let a = Array1::from_vec(vec![0.0, 0.0]);
/// let b = Array1::from_vec(vec![1.0, 1.0]);
///
/// let qrng = Halton::new(2, Some(42));
/// let result = qmc_quad(f, &a, &b, None, None, Some(Box::new(qrng)), false).unwrap();
/// println!("Integral: {}, Error: {}", result.integral, result.standard_error);
/// ```
pub fn qmc_quad<F>(
    func: F,
    a: &Array1<f64>,
    b: &Array1<f64>,
    n_estimates: Option<usize>,
    n_points: Option<usize>,
    qrng: Option<Box<dyn QRNGEngine>>,
    log: bool,
) -> IntegrateResult<QMCQuadResult<f64>>
where
    F: Fn(ArrayView1<f64>) -> f64,
{
    let n_estimates = n_estimates.unwrap_or(8);
    let n_points = n_points.unwrap_or(1024);

    // Input validation
    if a.len() != b.len() {
        return Err(IntegrateError::ValueError(
            "Dimension mismatch: 'a' and 'b' must have the same length".to_string(),
        ));
    }

    let dim = a.len();

    // Initialize QRNG if not provided
    let mut qrng = qrng.unwrap_or_else(|| Box::new(Halton::new(dim, None)));

    if qrng.dim() != dim {
        return Err(IntegrateError::ValueError(format!(
            "QRNG dimension ({}) does not match integration dimension ({})",
            qrng.dim(),
            dim
        )));
    }

    // Check if a = b for any dimension, return 0 if so
    for i in 0..dim {
        if (a[i] - b[i]).abs() < f64::EPSILON {
            return Ok(QMCQuadResult {
                integral: if log { f64::NEG_INFINITY } else { 0.0 },
                standard_error: 0.0,
            });
        }
    }

    // Swap limits if a > b and record the sign change
    let mut a_mod = a.clone();
    let mut b_mod = b.clone();
    let mut sign = 1.0;

    for i in 0..dim {
        if a[i] > b[i] {
            a_mod[i] = b[i];
            b_mod[i] = a[i];
            sign *= -1.0;
        }
    }

    // Volume of the hypercube
    let volume = (0..dim).map(|i| b_mod[i] - a_mod[i]).product::<f64>();
    let delta = volume / (n_points as f64);

    // Prepare for multiple estimates
    let mut estimates = Array1::<f64>::zeros(n_estimates);

    // Generate independent samples and compute estimates
    for i in 0..n_estimates {
        // Generate QMC sample
        let sample = qrng.random(n_points);

        // Scale to integration domain
        let x = scale(&sample, &a_mod, &b_mod);

        // Evaluate function at sample points
        let mut sum = 0.0;

        if log {
            let mut max_val = f64::NEG_INFINITY;
            let mut log_values = Vec::with_capacity(n_points);

            for j in 0..n_points {
                let val = func(x.slice(s![j, ..]));
                log_values.push(val);
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute log sum exp
            let mut sum_exp = 0.0;
            for val in log_values {
                sum_exp += (val - max_val).exp();
            }

            estimates[i] = max_val + sum_exp.ln() + delta.ln();
        } else {
            for j in 0..n_points {
                sum += func(x.slice(s![j, ..]));
            }

            estimates[i] = sum * delta;
        }

        // Get a new QRNG for next estimate with different scrambling
        let seed = i as u64 + 1;
        qrng = qrng.new_from_seed(seed);
    }

    // Compute final estimate and error
    let integral = if log {
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..n_estimates {
            if estimates[i] > max_val {
                max_val = estimates[i];
            }
        }

        let mut sum_exp = 0.0;
        for i in 0..n_estimates {
            sum_exp += (estimates[i] - max_val).exp();
        }

        max_val + (sum_exp / (n_estimates as f64)).ln()
    } else {
        estimates.sum() / (n_estimates as f64)
    };

    // Compute standard error
    let standard_error = if n_estimates > 1 {
        if log {
            // For log space, compute standard error differently
            let mean = integral;
            let mut variance = 0.0;

            for i in 0..n_estimates {
                let diff = (estimates[i] - mean).exp();
                variance += (diff - 1.0).powi(2);
            }

            variance /= (n_estimates - 1) as f64;
            (variance / (n_estimates as f64)).sqrt().ln()
        } else {
            let mean = integral;
            let mut variance = 0.0;

            for i in 0..n_estimates {
                variance += (estimates[i] - mean).powi(2);
            }

            variance /= (n_estimates - 1) as f64;
            (variance / (n_estimates as f64)).sqrt()
        }
    } else if log {
        f64::NEG_INFINITY
    } else {
        0.0
    };

    // Apply sign correction for reversed limits
    let final_integral = if log && sign < 0.0 {
        // For negative results in log space, we'd need complex numbers
        // Since we don't support complex results, we'll just negate the result
        // and warn about it in the documentation
        -integral
    } else {
        integral * sign
    };

    Ok(QMCQuadResult {
        integral: final_integral,
        standard_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn test_qmc_integral_1d() {
        // Test integral of x^2 from 0 to 1 (should be 1/3)
        let f = |x: ArrayView1<f64>| x[0].powi(2);
        let a = Array1::from_vec(vec![0.0]);
        let b = Array1::from_vec(vec![1.0]);

        let result = qmc_quad(f, &a, &b, Some(8), Some(1000), None, false).unwrap();

        assert_abs_diff_eq!(result.integral, 1.0 / 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_qmc_integral_2d() {
        // Test integral of x*y from 0 to 1, 0 to 1 (should be 1/4)
        let f = |x: ArrayView1<f64>| x[0] * x[1];
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 1.0]);

        let result = qmc_quad(f, &a, &b, Some(8), Some(1000), None, false).unwrap();

        assert_abs_diff_eq!(result.integral, 0.25, epsilon = 0.01);
    }

    #[test]
    fn test_infinite_limits() {
        // Test integral of e^(-x^2) from -∞ to ∞ (should be sqrt(π) ≈ 1.77)
        let f = |x: ArrayView1<f64>| (-x[0].powi(2)).exp();
        let a = Array1::from_vec(vec![-5.0]); // approximating infinity with a large value
        let b = Array1::from_vec(vec![5.0]);

        let result = qmc_quad(f, &a, &b, Some(8), Some(1000), None, false).unwrap();

        assert_abs_diff_eq!(result.integral, std::f64::consts::PI.sqrt(), epsilon = 0.05);
    }
}
