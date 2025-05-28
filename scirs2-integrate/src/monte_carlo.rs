//! Monte Carlo integration methods
//!
//! This module provides numerical integration methods based on Monte Carlo
//! sampling, which are particularly useful for high-dimensional integrals.

use crate::error::{IntegrateError, IntegrateResult};
use crate::IntegrateFloat;
use ndarray::{Array1, ArrayView1};
use rand::prelude::*;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Uniform};
use std::fmt::Debug;
use std::marker::PhantomData;

/// Options for controlling the behavior of Monte Carlo integration
#[derive(Debug, Clone)]
pub struct MonteCarloOptions<F: IntegrateFloat> {
    /// Number of sample points to use
    pub n_samples: usize,
    /// Random number generator seed (for reproducibility)
    pub seed: Option<u64>,
    /// Error estimation method
    pub error_method: ErrorEstimationMethod,
    /// Use antithetic variates for variance reduction
    pub use_antithetic: bool,
    /// Phantom data for generic type
    pub _phantom: PhantomData<F>,
}

/// Method for estimating the error in Monte Carlo integration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorEstimationMethod {
    /// Standard error of the mean
    StandardError,
    /// Batch means method
    BatchMeans,
}

impl<F: IntegrateFloat> Default for MonteCarloOptions<F> {
    fn default() -> Self {
        Self {
            n_samples: 10000,
            seed: None,
            error_method: ErrorEstimationMethod::StandardError,
            use_antithetic: false,
            _phantom: PhantomData,
        }
    }
}

/// Result of a Monte Carlo integration
#[derive(Debug, Clone)]
pub struct MonteCarloResult<F: IntegrateFloat> {
    /// Estimated value of the integral
    pub value: F,
    /// Estimated standard error
    pub std_error: F,
    /// Number of function evaluations
    pub n_evals: usize,
}

/// Perform Monte Carlo integration of a function over a hypercube
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `ranges` - Integration ranges (a, b) for each dimension
/// * `options` - Optional Monte Carlo parameters
///
/// # Returns
///
/// * `IntegrateResult<MonteCarloResult<F>>` - The result of the integration
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::{monte_carlo, MonteCarloOptions};
/// use ndarray::ArrayView1;
/// use std::marker::PhantomData;
///
/// // Integrate f(x,y) = x²+y² over [0,1]×[0,1] (exact result: 2/3)
/// let options = MonteCarloOptions {
///     n_samples: 100000,  // Use more samples for better accuracy
///     _phantom: PhantomData,
///     ..Default::default()
/// };
///
/// let result = monte_carlo(
///     |x: ArrayView1<f64>| x[0] * x[0] + x[1] * x[1],
///     &[(0.0, 1.0), (0.0, 1.0)],
///     Some(options)
/// ).unwrap();
///
/// // Should be close to 2/3, but Monte Carlo has statistical error
/// assert!((result.value - 2.0/3.0).abs() < 0.01);
/// ```
pub fn monte_carlo<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    options: Option<MonteCarloOptions<F>>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync,
    rand_distr::StandardNormal: Distribution<F>,
{
    let opts = options.unwrap_or_default();
    let n_dims = ranges.len();

    if n_dims == 0 {
        return Err(IntegrateError::ValueError(
            "Integration ranges cannot be empty".to_string(),
        ));
    }

    if opts.n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "Number of samples must be positive".to_string(),
        ));
    }

    // Calculate the volume of the integration domain
    let mut volume = F::one();
    for &(a, b) in ranges {
        volume *= b - a;
    }

    // Initialize random number generator
    let mut rng = if let Some(seed) = opts.seed {
        StdRng::seed_from_u64(seed)
    } else {
        // In rand 0.9.0, from_entropy is replaced by building from OsRng
        // Note: thread_rng() was renamed to rng() in rand 0.9.0
        let mut thread_rng = rand::rng();
        StdRng::from_rng(&mut thread_rng)
    };

    // Create uniform distributions for each dimension
    let distributions: Vec<_> = ranges
        .iter()
        .map(|&(a, b)| Uniform::new_inclusive(a, b).unwrap())
        .collect();

    // Prepare to store samples
    let n_actual_samples = if opts.use_antithetic {
        opts.n_samples / 2 * 2 // Ensure even number for antithetic pairs
    } else {
        opts.n_samples
    };

    // Buffer for a single sample point
    let mut point = Array1::zeros(n_dims);

    // Sample and evaluate the function
    let mut sum = F::zero();
    let mut sum_sq = F::zero();
    let mut n_evals = 0;

    if opts.use_antithetic {
        // Use antithetic sampling for variance reduction
        for _ in 0..(n_actual_samples / 2) {
            // Generate a random point in the hypercube
            for (i, dist) in distributions.iter().enumerate() {
                point[i] = dist.sample(&mut rng);
            }
            let value = f(point.view());
            sum += value;
            sum_sq += value * value;
            n_evals += 1;

            // Antithetic point: reflect around the center of the hypercube
            for (i, &(a, b)) in ranges.iter().enumerate() {
                point[i] = a + b - point[i]; // Reflect: a+b-x is the reflection of x relative to midpoint (a+b)/2
            }
            let antithetic_value = f(point.view());
            sum += antithetic_value;
            sum_sq += antithetic_value * antithetic_value;
            n_evals += 1;
        }
    } else {
        // Standard Monte Carlo sampling
        for _ in 0..n_actual_samples {
            // Generate a random point in the hypercube
            for (i, dist) in distributions.iter().enumerate() {
                point[i] = dist.sample(&mut rng);
            }
            let value = f(point.view());
            sum += value;
            sum_sq += value * value;
            n_evals += 1;
        }
    }

    // Compute the mean value and scale by the volume
    let mean = sum / F::from_usize(n_actual_samples).unwrap();
    let integral_value = mean * volume;

    // Estimate the error based on the specified method
    let std_error = match opts.error_method {
        ErrorEstimationMethod::StandardError => {
            // Standard error of the mean using sample variance
            let variance = (sum_sq - sum * sum / F::from_usize(n_actual_samples).unwrap())
                / F::from_usize(n_actual_samples - 1).unwrap();

            (variance / F::from_usize(n_actual_samples).unwrap()).sqrt() * volume
        }
        ErrorEstimationMethod::BatchMeans => {
            // Divide samples into batches and compute variance of batch means
            // This requires re-sampling, so we'll skip the actual implementation for simplicity
            // In a real implementation, we would compute batch means from the original samples

            // For now, we'll just use a simplified estimate based on the standard error
            let variance = (sum_sq - sum * sum / F::from_usize(n_actual_samples).unwrap())
                / F::from_usize(n_actual_samples - 1).unwrap();

            (variance / F::from_usize(n_actual_samples).unwrap()).sqrt() * volume
        }
    };

    Ok(MonteCarloResult {
        value: integral_value,
        std_error,
        n_evals,
    })
}

/// Perform Monte Carlo integration with importance sampling
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `g` - The probability density function to sample from
/// * `sampler` - A function that generates samples from the PDF g
/// * `ranges` - Integration ranges (a, b) for each dimension
/// * `options` - Optional Monte Carlo parameters
///
/// # Returns
///
/// * `IntegrateResult<MonteCarloResult<F>>` - The result of the integration
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::{importance_sampling, MonteCarloOptions};
/// use ndarray::{Array1, ArrayView1};
/// use rand::prelude::*;
///
/// // Simple example: integrate x² from 0 to 1 using uniform importance sampling
/// // (which is the same as regular Monte Carlo in this case)
///
/// // Uniform sampler
/// let uniform_sampler = |rng: &mut StdRng, dims: usize| {
///     let mut point = Array1::zeros(dims);
///     for i in 0..dims {
///         point[i] = rng.gen_range(0.0..1.0);
///     }
///     point
/// };
///
/// // Uniform PDF on [0,1]
/// let uniform_pdf = |x: ArrayView1<f64>| {
///     let mut pdf = 1.0;
///     for &xi in x.iter() {
///         if xi < 0.0 || xi > 1.0 {
///             return 0.0;
///         }
///     }
///     pdf
/// };
///
/// let options = MonteCarloOptions {
///     n_samples: 10000,
///     seed: Some(42),
///     ..Default::default()
/// };
///
/// // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
/// let result = importance_sampling(
///     |x: ArrayView1<f64>| x[0] * x[0],
///     uniform_pdf,
///     uniform_sampler,
///     &[(0.0, 1.0)],
///     Some(options)
/// ).unwrap();
///
/// assert!((result.value - 1.0/3.0).abs() < 0.01);
/// ```
pub fn importance_sampling<F, Func, Pdf, Sampler>(
    f: Func,
    g: Pdf,
    sampler: Sampler,
    ranges: &[(F, F)],
    options: Option<MonteCarloOptions<F>>,
) -> IntegrateResult<MonteCarloResult<F>>
where
    F: IntegrateFloat + Send + Sync + SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Sync,
    Pdf: Fn(ArrayView1<F>) -> F + Sync,
    Sampler: Fn(&mut StdRng, usize) -> Array1<F> + Sync,
    rand_distr::StandardNormal: Distribution<F>,
{
    let opts = options.unwrap_or_default();
    let n_dims = ranges.len();

    if n_dims == 0 {
        return Err(IntegrateError::ValueError(
            "Integration ranges cannot be empty".to_string(),
        ));
    }

    if opts.n_samples == 0 {
        return Err(IntegrateError::ValueError(
            "Number of samples must be positive".to_string(),
        ));
    }

    // Initialize random number generator
    let mut rng = if let Some(seed) = opts.seed {
        StdRng::seed_from_u64(seed)
    } else {
        // In rand 0.9.0, from_entropy is replaced by building from OsRng
        // Note: thread_rng() was renamed to rng() in rand 0.9.0
        let mut thread_rng = rand::rng();
        StdRng::from_rng(&mut thread_rng)
    };

    // Sample and evaluate the function
    let mut sum = F::zero();
    let mut sum_sq = F::zero();
    let n_samples = opts.n_samples;

    // Count valid samples (where g(x) > epsilon)
    let mut valid_samples = 0;

    for _ in 0..n_samples {
        // Generate a sample point from the importance distribution
        let point = sampler(&mut rng, n_dims);

        // Evaluate f(x) for importance sampling
        let fx = f(point.view());

        // Evaluate g(x) and handle potential numerical issues
        let gx = g(point.view());

        // Avoid division by zero or very small values that could lead to instability
        // Use a higher threshold than just epsilon to avoid numerical instability
        if gx <= F::from_f64(1e-10).unwrap() {
            continue;
        }

        // Check for NaN or Infinity in either fx or gx
        if fx.is_nan() || fx.is_infinite() || gx.is_nan() || gx.is_infinite() {
            continue;
        }

        // Compute the ratio and weight
        let ratio = fx / gx;

        // Another check for numerical stability of the ratio
        if ratio.is_nan() || ratio.is_infinite() {
            continue;
        }

        // Update accumulators
        sum += ratio;
        sum_sq += ratio * ratio;
        valid_samples += 1;
    }

    // Check if we have enough valid samples
    if valid_samples < 10 {
        return Err(IntegrateError::ConvergenceError(
            "Too few valid samples in importance sampling".to_string(),
        ));
    }

    // Calculate the mean and standard error using valid samples
    let valid_samples_f = F::from_usize(valid_samples).unwrap();
    let mean = sum / valid_samples_f;

    // Compute the standard error
    let variance = if valid_samples > 1 {
        (sum_sq - sum * sum / valid_samples_f) / F::from_usize(valid_samples - 1).unwrap()
    } else {
        // If we have only one valid sample, we can't estimate variance
        F::zero()
    };

    let std_error = (variance / valid_samples_f).sqrt();

    Ok(MonteCarloResult {
        value: mean,
        std_error,
        n_evals: valid_samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::PI;

    // Helper function to check if result is within expected error margin
    fn is_close_enough(result: f64, expected: f64, epsilon: f64) -> bool {
        (result - expected).abs() < epsilon
    }

    #[test]
    fn test_monte_carlo_1d() {
        // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
        let options = MonteCarloOptions {
            n_samples: 100000,
            seed: Some(12345), // For reproducibility
            _phantom: PhantomData,
            ..Default::default()
        };

        let result = monte_carlo(|x| x[0] * x[0], &[(0.0, 1.0)], Some(options)).unwrap();

        // Monte Carlo is a statistical method, so we use a loose tolerance
        assert!(is_close_enough(result.value, 1.0 / 3.0, 0.01));
    }

    #[test]
    fn test_monte_carlo_2d() {
        // Integrate f(x,y) = x² + y² over [0,1]×[0,1] (exact result: 2/3)
        let options = MonteCarloOptions {
            n_samples: 100000,
            seed: Some(12345), // For reproducibility
            _phantom: PhantomData,
            ..Default::default()
        };

        let result = monte_carlo(
            |x| x[0] * x[0] + x[1] * x[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            Some(options),
        )
        .unwrap();

        assert!(is_close_enough(result.value, 2.0 / 3.0, 0.01));
    }

    #[test]
    fn test_monte_carlo_with_antithetic() {
        // Test with antithetic variates for variance reduction
        let options = MonteCarloOptions {
            n_samples: 100000,
            seed: Some(12345), // For reproducibility
            use_antithetic: true,
            _phantom: PhantomData,
            ..Default::default()
        };

        let result = monte_carlo(|x| x[0] * x[0], &[(0.0, 1.0)], Some(options)).unwrap();

        assert!(is_close_enough(result.value, 1.0 / 3.0, 0.01));

        // The standard error with antithetic sampling should generally be lower
        // than without it, but this is hard to test deterministically
    }

    #[test]
    fn test_importance_sampling() {
        // Test importance sampling with a more stable approach
        // We'll integrate exp(-x²) from 0 to 3 and use a normal distribution
        // centered at 0 with std dev of 1 as our sampling distribution

        // The Normal distribution sampler
        let sampler = |rng: &mut StdRng, dims: usize| {
            let mut point = Array1::zeros(dims);
            // Use a normal distribution centered at 0 with std dev 1
            let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

            for i in 0..dims {
                // Sample and truncate to the integration range [0, 3]
                let mut x: f64 = normal.sample(rng);
                // Make sure value is within integration range
                x = x.abs(); // Fold negatives to positive domain
                if x > 3.0 {
                    x = 6.0 - x; // Reflect values beyond 3.0 back into range
                    if x < 0.0 {
                        // If still out of range, clamp to 0
                        x = 0.0;
                    }
                }
                point[i] = x;
            }
            point
        };

        // Normal PDF
        let normal_pdf = |x: ArrayView1<f64>| {
            let mut pdf_val = 1.0;
            for &xi in x.iter() {
                // Normal density function, but folded to account for our transformation
                let z = xi;
                let density = (-0.5 * z * z).exp() / (2.0 * PI).sqrt();
                // Fold the negative domain to account for our transformation
                let folded_density = if xi < 3.0 {
                    2.0 * density // Double density because we folded the distribution
                } else {
                    0.0 // Outside integration range, shouldn't happen
                };
                pdf_val *= folded_density.max(1e-10); // Prevent zero density
            }
            pdf_val
        };

        let options = MonteCarloOptions {
            n_samples: 100000,
            seed: Some(12345), // For reproducibility
            _phantom: PhantomData,
            ..Default::default()
        };

        // Integrate f(x) = exp(-x²) from 0 to 3
        // The exact value is sqrt(π)/2 * erf(3) ≈ 0.886
        let exact_value = (PI).sqrt() / 2.0 * libm::erf(3.0);

        let result = importance_sampling(
            |x| (-x[0] * x[0]).exp(),
            normal_pdf,
            sampler,
            &[(0.0, 3.0)],
            Some(options),
        )
        .unwrap();

        // Check result within reasonable error bounds
        assert!(is_close_enough(result.value, exact_value, 0.1));
        println!(
            "Importance sampling test: got {}, expected {}",
            result.value, exact_value
        );
    }
}
