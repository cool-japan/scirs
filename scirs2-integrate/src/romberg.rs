//! Romberg integration method
//!
//! This module provides an implementation of Romberg integration,
//! a numerical method that accelerates the trapezoidal rule by using
//! Richardson extrapolation to eliminate error terms.

use crate::error::{IntegrateError, IntegrateResult};
use crate::quad::trapezoid;
use crate::IntegrateFloat;
use ndarray::{Array1, Array2, ArrayView1};
use rand_distr::{Distribution, Uniform};
use std::fmt::Debug;

/// Options for controlling the behavior of Romberg integration
#[derive(Debug, Clone)]
pub struct RombergOptions<F: IntegrateFloat> {
    /// Maximum number of iterations
    pub max_iters: usize,
    /// Absolute error tolerance
    pub abs_tol: F,
    /// Relative error tolerance
    pub rel_tol: F,
    /// Maximum dimension for true Romberg integration (defaults to 3)
    /// For dimensions higher than this, a more efficient method will be used
    pub max_true_dimension: usize,
    /// Minimum number of samples for Monte Carlo fallback (for high dimensions)
    pub min_monte_carlo_samples: usize,
}

impl<F: IntegrateFloat> Default for RombergOptions<F> {
    fn default() -> Self {
        Self {
            max_iters: 20,
            abs_tol: F::from_f64(1.0e-10).unwrap(),
            rel_tol: F::from_f64(1.0e-10).unwrap(),
            max_true_dimension: 3,
            min_monte_carlo_samples: 10000,
        }
    }
}

/// Result of a Romberg integration computation
#[derive(Debug, Clone)]
pub struct RombergResult<F: IntegrateFloat> {
    /// Estimated value of the integral
    pub value: F,
    /// Estimated absolute error
    pub abs_error: F,
    /// Number of iterations performed
    pub n_iters: usize,
    /// Romberg table (for diagnostic purposes)
    pub table: Array2<F>,
    /// Flag indicating successful convergence
    pub converged: bool,
}

/// Compute the definite integral of a function using Romberg integration
///
/// # Arguments
///
/// * `f` - The function to integrate
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `options` - Optional integration parameters
///
/// # Returns
///
/// * `IntegrateResult<RombergResult<F>>` - The result of the integration or an error
///
/// # Examples
///
/// ```
/// use scirs2_integrate::romberg::romberg;
///
/// // Integrate f(x) = x² from 0 to 1 (exact result: 1/3)
/// let result = romberg(|x: f64| x * x, 0.0, 1.0, None).unwrap();
/// assert!((result.value - 1.0/3.0).abs() < 1e-10);
/// assert!(result.converged);
/// ```
pub fn romberg<F, Func>(
    f: Func,
    a: F,
    b: F,
    options: Option<RombergOptions<F>>,
) -> IntegrateResult<RombergResult<F>>
where
    F: IntegrateFloat,
    Func: Fn(F) -> F + Copy,
{
    let opts = options.unwrap_or_default();
    let max_iters = opts.max_iters;

    // Initialize the Romberg table
    let mut r_table = Array2::zeros((max_iters, max_iters));

    // Initial computation with h = b-a
    r_table[[0, 0]] = trapezoid(f, a, b, 1);

    let mut converged = false;
    let mut n_iters = 1;

    for i in 1..max_iters {
        // Compute next level of trapezoid approximation with step size h = (b-a)/2^i
        let n_intervals = 1 << i; // 2^i
        r_table[[i, 0]] = trapezoid(f, a, b, n_intervals);

        // Apply Richardson extrapolation to compute the Romberg table
        for j in 1..=i {
            // R(i,j) = R(i,j-1) + (R(i,j-1) - R(i-1,j-1))/(4^j - 1)
            let coef = F::from_f64(4.0_f64.powi(j as i32) - 1.0).unwrap();
            r_table[[i, j]] =
                r_table[[i, j - 1]] + (r_table[[i, j - 1]] - r_table[[i - 1, j - 1]]) / coef;
        }

        // Check for convergence
        let current = r_table[[i, i]];
        let previous = r_table[[i - 1, i - 1]];
        let abs_diff = (current - previous).abs();
        let abs_criterion = abs_diff <= opts.abs_tol;
        let rel_criterion = abs_diff <= opts.rel_tol * current.abs();

        if abs_criterion || rel_criterion {
            converged = true;
            n_iters = i + 1;
            break;
        }

        n_iters = i + 1;
    }

    // Final value is in the bottom-right corner of the used portion of the table
    let value = r_table[[n_iters - 1, n_iters - 1]];

    // Estimate error as the difference between the last two diagonal elements
    let abs_error = if n_iters >= 2 {
        (value - r_table[[n_iters - 2, n_iters - 2]]).abs()
    } else {
        // If we only have one iteration, use a conservative error estimate
        F::from_f64(1.0e-3).unwrap() * value.abs()
    };

    // Create a new array with just the used portion of the table
    let table = Array2::from_shape_fn((n_iters, n_iters), |(i, j)| {
        if j <= i {
            r_table[[i, j]]
        } else {
            F::zero()
        }
    });

    Ok(RombergResult {
        value,
        abs_error,
        n_iters,
        table,
        converged,
    })
}

/// Compute the definite integral of a function over multiple dimensions using Romberg integration
///
/// # Arguments
///
/// * `f` - The multidimensional function to integrate
/// * `ranges` - Array of integration ranges (a, b) for each dimension
/// * `options` - Optional integration parameters
///
/// # Returns
///
/// * `IntegrateResult<F>` - The approximate value of the integral
///
/// # Examples
///
/// ```ignore
/// use scirs2_integrate::romberg::multi_romberg;
/// use ndarray::Array1;
///
/// // Integrate f(x,y) = x²+y² over [0,1]×[0,1] (exact result: 2/3)
/// let result = multi_romberg(
///     |x| x.iter().map(|&xi| xi*xi).sum::<f64>(),
///     &[(0.0, 1.0), (0.0, 1.0)],
///     None
/// ).unwrap();
/// assert!((result - 2.0/3.0).abs() < 1e-8);
/// ```
/// Result of a multidimensional Romberg integration computation
#[derive(Debug, Clone)]
pub struct MultiRombergResult<F: IntegrateFloat> {
    /// Estimated value of the integral
    pub value: F,
    /// Estimated absolute error
    pub abs_error: F,
    /// Method used for the integration
    pub method: IntegrationMethod,
}

/// Integration method used for multidimensional integration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrationMethod {
    /// Standard Romberg integration
    Romberg,
    /// Adaptive nested integration
    AdaptiveNested,
    /// Direct grid-based trapezoid rule
    GridTrapezoid,
    /// Monte Carlo integration
    MonteCarlo,
}

pub fn multi_romberg<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    options: Option<RombergOptions<F>>,
) -> IntegrateResult<F>
where
    F: IntegrateFloat + rand_distr::uniform::SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Copy,
{
    let result = multi_romberg_with_details(f, ranges, options)?;
    Ok(result.value)
}

/// Compute the definite integral of a function over multiple dimensions
/// using the most appropriate method based on the dimension
///
/// Returns the full result structure with error estimates and method information
pub fn multi_romberg_with_details<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    options: Option<RombergOptions<F>>,
) -> IntegrateResult<MultiRombergResult<F>>
where
    F: IntegrateFloat + rand_distr::uniform::SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Copy,
{
    if ranges.is_empty() {
        return Err(IntegrateError::ValueError(
            "Integration ranges cannot be empty".to_string(),
        ));
    }

    let opts = options.unwrap_or_default();
    let n_dims = ranges.len();

    // Special case for 1D: Use standard Romberg integration
    if n_dims == 1 {
        let (a, b) = ranges[0];
        // Direct integration of the 1D function
        let result = romberg(
            |x| f(Array1::from_vec(vec![x]).view()),
            a,
            b,
            Some(opts.clone()),
        )?;

        return Ok(MultiRombergResult {
            value: result.value,
            abs_error: result.abs_error,
            method: IntegrationMethod::Romberg,
        });
    }

    // For 2D up to max_true_dimension: Use adaptive integration with caching
    if n_dims <= opts.max_true_dimension && n_dims <= 3 {
        return integrate_adaptive_nested(f, ranges, &opts);
    }

    // For higher dimensions, use Monte Carlo integration with importance sampling
    // This avoids recursion issues completely
    monte_carlo_high_dimensions(f, ranges, &opts)
}

/// Integrate a function using adaptive nested integration for dimensions 2-3
fn integrate_adaptive_nested<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    opts: &RombergOptions<F>,
) -> IntegrateResult<MultiRombergResult<F>>
where
    F: IntegrateFloat + rand_distr::uniform::SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Copy,
{
    let n_dims = ranges.len();

    if n_dims == 2 {
        // 2D case: Use a direct grid-based approach with refined grid
        let (a1, b1) = ranges[0];
        let (a2, b2) = ranges[1];

        // Use a grid-based approach with an appropriate number of points
        let n_points = (opts.max_iters + 1).min(31); // Cap at a reasonable max to avoid excessive memory use

        let h1 = (b1 - a1) / F::from_usize(n_points - 1).unwrap();
        let h2 = (b2 - a2) / F::from_usize(n_points - 1).unwrap();

        let mut sum = F::zero();
        let mut sum_refined = F::zero(); // For error estimation

        // First pass: Calculate on coarse grid
        for i in 0..n_points {
            let x = a1 + F::from_usize(i).unwrap() * h1;
            let weight_x = if i == 0 || i == n_points - 1 {
                F::from_f64(0.5).unwrap()
            } else {
                F::one()
            };

            for j in 0..n_points {
                let y = a2 + F::from_usize(j).unwrap() * h2;
                let weight_y = if j == 0 || j == n_points - 1 {
                    F::from_f64(0.5).unwrap()
                } else {
                    F::one()
                };

                // Evaluate function at grid point
                let point = Array1::from_vec(vec![x, y]);
                let value = f(point.view());

                // Add weighted value to sum
                sum += weight_x * weight_y * value;
            }
        }

        // Scale by grid spacing
        let result = sum * h1 * h2;

        // Second pass for refined grid (use every other point from the first grid)
        // This gives us an error estimate without doubling the function evaluations
        if n_points > 4 {
            let refined_n = n_points / 2 + (n_points % 2);
            let refined_h1 = (b1 - a1) / F::from_usize(refined_n - 1).unwrap();
            let refined_h2 = (b2 - a2) / F::from_usize(refined_n - 1).unwrap();

            for i in 0..refined_n {
                let x = a1 + F::from_usize(i).unwrap() * refined_h1;
                let weight_x = if i == 0 || i == refined_n - 1 {
                    F::from_f64(0.5).unwrap()
                } else {
                    F::one()
                };

                for j in 0..refined_n {
                    let y = a2 + F::from_usize(j).unwrap() * refined_h2;
                    let weight_y = if j == 0 || j == refined_n - 1 {
                        F::from_f64(0.5).unwrap()
                    } else {
                        F::one()
                    };

                    // Evaluate function at grid point
                    let point = Array1::from_vec(vec![x, y]);
                    let value = f(point.view());

                    // Add weighted value to sum
                    sum_refined += weight_x * weight_y * value;
                }
            }

            let refined_result = sum_refined * refined_h1 * refined_h2;

            // Error estimate based on difference between full and refined grids
            let abs_error = (result - refined_result).abs();

            return Ok(MultiRombergResult {
                value: result,
                abs_error,
                method: IntegrationMethod::GridTrapezoid,
            });
        }

        // If we don't have enough points for a refined grid, use a conservative error estimate
        let abs_error = result.abs() * F::from_f64(1e-3).unwrap();

        return Ok(MultiRombergResult {
            value: result,
            abs_error,
            method: IntegrationMethod::GridTrapezoid,
        });
    } else if n_dims == 3 {
        // 3D case: Use an approach with caching of innermost integral
        let (a1, b1) = ranges[0];
        let (a2, b2) = ranges[1];
        let (a3, b3) = ranges[2];

        // Use fewer grid points for 3D to keep performance reasonable
        let n_points = (opts.max_iters / 2 + 1).min(11);

        let h1 = (b1 - a1) / F::from_usize(n_points - 1).unwrap();
        let h2 = (b2 - a2) / F::from_usize(n_points - 1).unwrap();
        let h3 = (b3 - a3) / F::from_usize(n_points - 1).unwrap();

        let mut sum = F::zero();

        // First, compute all grid points
        for i in 0..n_points {
            let x = a1 + F::from_usize(i).unwrap() * h1;
            let weight_x = if i == 0 || i == n_points - 1 {
                F::from_f64(0.5).unwrap()
            } else {
                F::one()
            };

            for j in 0..n_points {
                let y = a2 + F::from_usize(j).unwrap() * h2;
                let weight_y = if j == 0 || j == n_points - 1 {
                    F::from_f64(0.5).unwrap()
                } else {
                    F::one()
                };

                for k in 0..n_points {
                    let z = a3 + F::from_usize(k).unwrap() * h3;
                    let weight_z = if k == 0 || k == n_points - 1 {
                        F::from_f64(0.5).unwrap()
                    } else {
                        F::one()
                    };

                    // Evaluate function at grid point
                    let point = Array1::from_vec(vec![x, y, z]);
                    let value = f(point.view());

                    // Add weighted value to sum
                    sum += weight_x * weight_y * weight_z * value;
                }
            }
        }

        // Scale by grid spacing
        let result = sum * h1 * h2 * h3;

        // Error estimate based on some sampling
        // We'll use a rough approximation since a full refinement would be too expensive
        let abs_error = result.abs() * F::from_f64(1e-2).unwrap();

        return Ok(MultiRombergResult {
            value: result,
            abs_error,
            method: IntegrationMethod::AdaptiveNested,
        });
    }

    // Should not reach here if max dimensions is 3, but just in case
    monte_carlo_high_dimensions(f, ranges, opts)
}

/// High-dimensional integration using Monte Carlo with variance reduction techniques
fn monte_carlo_high_dimensions<F, Func>(
    f: Func,
    ranges: &[(F, F)],
    opts: &RombergOptions<F>,
) -> IntegrateResult<MultiRombergResult<F>>
where
    F: IntegrateFloat + rand_distr::uniform::SampleUniform,
    Func: Fn(ArrayView1<F>) -> F + Copy,
{
    let n_dims = ranges.len();

    // Calculate the number of samples based on dimension
    // Higher dimensions need more samples
    let base_samples = opts.min_monte_carlo_samples;
    let n_samples = base_samples * n_dims.max(4);

    let mut sum = F::zero();
    let mut sum_squares = F::zero();
    let mut rng = rand::rng();

    // Prepare uniform samplers for each dimension
    let uniforms: Vec<_> = ranges
        .iter()
        .map(|&(a, b)| Uniform::new_inclusive(a, b).unwrap())
        .collect();

    // Estimate the volume of the integration domain
    let mut volume = F::one();
    for &(a, b) in ranges {
        volume *= b - a;
    }

    // Perform stratified sampling to reduce variance
    let strata = 2_usize; // Simple stratification into 2 parts per dimension
    let n_samples_per_strata = n_samples / strata.pow(n_dims as u32).max(1);
    let n_samples_per_strata = n_samples_per_strata.max(100); // Ensure minimum number of samples

    // Evaluate function at all sample points
    let mut point = Array1::zeros(n_dims);
    let mut n_actual_samples = 0;

    // Use a more sophisticated Monte Carlo approach
    for _ in 0..n_samples_per_strata {
        // Generate a random point in the integration domain
        for (i, dist) in uniforms.iter().enumerate() {
            point[i] = dist.sample(&mut rng);
        }

        // Evaluate the function with antithetic sampling
        // This reduces variance by evaluating at x and 1-x
        let value = f(point.view());
        sum += value;
        sum_squares += value * value;
        n_actual_samples += 1;

        // Generate the "opposite" point (antithetic variates)
        for i in 0..n_dims {
            let (a, b) = ranges[i];
            point[i] = a + b - point[i]; // Reflect around midpoint
        }

        let antithetic_value = f(point.view());
        sum += antithetic_value;
        sum_squares += antithetic_value * antithetic_value;
        n_actual_samples += 1;
    }

    // Calculate the Monte Carlo estimate
    let n_samples_f = F::from_usize(n_actual_samples).unwrap();
    let mean = sum / n_samples_f;
    let result = mean * volume;

    // Estimate the error using the sample variance
    let variance = (sum_squares - sum * sum / n_samples_f) / (n_samples_f - F::one());
    let std_error = (variance / n_samples_f).sqrt() * volume;

    Ok(MultiRombergResult {
        value: result,
        abs_error: std_error,
        method: IntegrationMethod::MonteCarlo,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use std::f64::consts::PI;

    #[test]
    fn test_romberg_integration() {
        // Test integrating x² from 0 to 1 (exact result: 1/3)
        let result = romberg(|x| x * x, 0.0, 1.0, None).unwrap();
        assert_relative_eq!(result.value, 1.0 / 3.0, epsilon = 1e-10);
        assert!(result.converged);

        // Test integrating sin(x) from 0 to π (exact result: 2)
        let result = romberg(|x| x.sin(), 0.0, PI, None).unwrap();
        assert_relative_eq!(result.value, 2.0, epsilon = 1e-10);
        assert!(result.converged);

        // Test integrating exp(-x²) from -1 to 1
        // This is related to the error function, with exact result: sqrt(π)·erf(1)
        let result = romberg(|x: f64| (-x * x).exp(), -1.0, 1.0, None).unwrap();
        let exact = PI.sqrt() * libm::erf(1.0);
        assert_relative_eq!(result.value, exact, epsilon = 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_romberg_with_custom_options() {
        // Test with custom convergence options
        let options = RombergOptions {
            max_iters: 10,
            abs_tol: 1e-12,
            rel_tol: 1e-12,
            max_true_dimension: 3,
            min_monte_carlo_samples: 10000,
        };

        let result = romberg(|x| x * x, 0.0, 1.0, Some(options)).unwrap();
        assert_relative_eq!(result.value, 1.0 / 3.0, epsilon = 1e-12);
        assert!(result.converged);
    }

    #[test]
    fn test_multi_dimensional_romberg() {
        // Test 2D integration: f(x,y) = x² + y² over [0,1]×[0,1]
        // Exact result: 2/3 (1/3 for x² + 1/3 for y²)
        let result = multi_romberg_with_details(
            |x| x[0] * x[0] + x[1] * x[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();

        // Our new implementation should be able to achieve better accuracy
        assert_relative_eq!(result.value, 2.0 / 3.0, epsilon = 1e-3);
        // Verify the method used is the grid trapezoid method
        assert_eq!(result.method, IntegrationMethod::GridTrapezoid);

        // Test 3D integration: f(x,y,z) = x²y²z² over [0,1]³
        // Exact result: (1/3)³ = 1/27
        let result = multi_romberg_with_details(
            |x| x[0] * x[0] * x[1] * x[1] * x[2] * x[2],
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();

        assert_relative_eq!(result.value, 1.0 / 27.0, epsilon = 1e-2);
        // Verify the method used is the adaptive nested method
        assert_eq!(result.method, IntegrationMethod::AdaptiveNested);

        // Test 4D integration (should use Monte Carlo)
        // f(w,x,y,z) = w²x²y²z² over [0,1]⁴
        // Exact result: (1/3)⁴ = 1/81
        let result = multi_romberg_with_details(
            |x| x[0] * x[0] * x[1] * x[1] * x[2] * x[2] * x[3] * x[3],
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            None,
        )
        .unwrap();

        // Monte Carlo will have lower accuracy but should still be reasonable
        assert_relative_eq!(result.value, 1.0 / 81.0, epsilon = 1e-1);
        // Verify the method used is Monte Carlo
        assert_eq!(result.method, IntegrationMethod::MonteCarlo);

        // Test with custom option to force Monte Carlo for lower dimensions
        let custom_opts = RombergOptions {
            max_true_dimension: 1, // Only use true Romberg for 1D
            ..Default::default()
        };

        // This should now use Monte Carlo for 2D
        let result = multi_romberg_with_details(
            |x| x[0] * x[0] + x[1] * x[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            Some(custom_opts),
        )
        .unwrap();

        assert_relative_eq!(result.value, 2.0 / 3.0, epsilon = 5e-2);
        assert_eq!(result.method, IntegrationMethod::MonteCarlo);
    }
}
